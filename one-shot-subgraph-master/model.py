import torch
import torch.nn as nn
from torch_scatter import scatter

class GlobalCalibrationSEA(torch.nn.Module):
    def __init__(self, hidden_dim, local_hidden_dim, global_hidden_dim, dropout=0.0,
                 global_dropout=0.0, global_eta=0.2, pool_temp=1.0):
        super(GlobalCalibrationSEA, self).__init__()
        self.pool_temp = max(float(pool_temp), 1e-6)
        self.global_eta = float(global_eta)

        self.rel_fc = nn.Linear(hidden_dim * 2, local_hidden_dim)
        self.rel_score = nn.Linear(local_hidden_dim, 1)
        self.trans_fc = nn.Linear(hidden_dim * 3, local_hidden_dim)
        self.trans_score = nn.Linear(local_hidden_dim, 1)

        self.pool_fc = nn.Linear(hidden_dim * 2, global_hidden_dim)
        self.pool_score = nn.Linear(global_hidden_dim, 1)
        self.global_fc = nn.Linear(hidden_dim * 4, global_hidden_dim)
        self.global_score = nn.Linear(global_hidden_dim, 1)

        self.local_dropout = nn.Dropout(dropout)
        self.global_dropout = nn.Dropout(global_dropout)

    def _group_softmax(self, score, index, dim_size):
        score_max = scatter(score, index=index, dim=0, dim_size=dim_size, reduce='max')[index]
        score_exp = torch.exp(score - score_max)
        score_sum = scatter(score_exp, index=index, dim=0, dim_size=dim_size, reduce='sum')[index] + 1e-12
        return score_exp / score_sum

    def forward(self, hs, ho, hr, h_qr, hidden, h_qn, edge_query_idx, node_query_idx, n_query):
        rel_hidden = torch.tanh(self.rel_fc(torch.cat([hr, h_qr], dim=-1)))
        rel_hidden = self.local_dropout(rel_hidden)
        z_rel = self.rel_score(rel_hidden).squeeze(-1)

        trans_hidden = torch.tanh(self.trans_fc(torch.cat([hs, ho, hr], dim=-1)))
        trans_hidden = self.local_dropout(trans_hidden)
        z_trans = self.trans_score(trans_hidden).squeeze(-1)

        pool_hidden = torch.tanh(self.pool_fc(torch.cat([hidden, h_qn], dim=-1)))
        pool_hidden = self.global_dropout(pool_hidden)
        pool_logits = self.pool_score(pool_hidden).squeeze(-1) / self.pool_temp
        pool_alpha = self._group_softmax(pool_logits, node_query_idx, n_query).unsqueeze(-1)
        global_summary = scatter(pool_alpha * hidden, index=node_query_idx, dim=0, dim_size=n_query, reduce='sum')

        edge_global = global_summary[edge_query_idx]
        global_hidden = torch.tanh(self.global_fc(torch.cat([ho, hr, h_qr, edge_global], dim=-1)))
        global_hidden = self.global_dropout(global_hidden)
        z_global = self.global_score(global_hidden).squeeze(-1)

        gate_logit = z_rel + z_trans + self.global_eta * z_global
        return torch.sigmoid(gate_logit).unsqueeze(-1)

class ScoreFCHead(torch.nn.Module):
    def __init__(self, hidden_dim, score_hidden_dim, dropout=0.0):
        super(ScoreFCHead, self).__init__()
        self.fc1 = nn.Linear(hidden_dim * 3, score_hidden_dim)
        self.fc2 = nn.Linear(score_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_node, h_anchor, h_qr):
        pair_feature = torch.cat([h_node * h_anchor, torch.abs(h_node - h_anchor), h_qr], dim=-1)
        hidden = torch.relu(self.fc1(pair_feature))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wo_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def _target_softmax(self, attn_score, obj, n_node):
        score_max = scatter(attn_score, index=obj, dim=0, dim_size=n_node, reduce='max')[obj]
        score_exp = torch.exp(attn_score - score_max)
        score_sum = scatter(score_exp, index=obj, dim=0, dim_size=n_node, reduce='sum')[obj] + 1e-12
        return (score_exp / score_sum).unsqueeze(-1)

    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, shortcut=False,
                node_batch_idxs=None, use_selective_agg=False, sea_gate=None):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        hs = hidden[sub]
        ho = hidden[obj]
        hr = self.rela_embed(rel) # relation embedding of each edge
        h_qr = self.rela_embed(q_rel)[r_idx] # use batch_idx to get the query relation

        if not use_selective_agg:
            # original message aggregation
            message = hs * hr
            alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
            message = alpha * message
            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        else:
            raw_message = hs * hr
            h_qn = self.rela_embed(q_rel)[node_batch_idxs]
            gate = sea_gate(hs, ho, hr, h_qr, hidden, h_qn, r_idx, node_batch_idxs, len(q_rel))
            attn_score = self.w_alpha(
                torch.tanh(self.Ws_attn(hs) + self.Wo_attn(ho) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))
            ).squeeze(-1)
            alpha = self._target_softmax(attn_score, obj, n_node)
            message = alpha * gate * raw_message
            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))

        if shortcut:
            hidden_new = hidden_new + hidden
        
        return hidden_new

class GNN_auto(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNN_auto, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        self.loader = loader
        self.use_selective_agg = getattr(params, 'use_selective_agg', False)
        self.sea_hidden_dim = int(getattr(params, 'sea_hidden_dim', self.hidden_dim))
        if self.sea_hidden_dim <= 0:
            self.sea_hidden_dim = self.hidden_dim
        self.sea_dropout = float(getattr(params, 'sea_dropout', 0.0))
        self.sea_global_hidden_dim = int(getattr(params, 'sea_global_hidden_dim', self.sea_hidden_dim))
        if self.sea_global_hidden_dim <= 0:
            self.sea_global_hidden_dim = self.sea_hidden_dim
        self.sea_global_eta = float(getattr(params, 'sea_global_eta', 0.2))
        self.sea_pool_temp = float(getattr(params, 'sea_pool_temp', 1.0))
        if self.sea_pool_temp <= 0:
            self.sea_pool_temp = 1.0
        self.sea_global_dropout = float(getattr(params, 'sea_global_dropout', self.sea_dropout))
        self.use_score_fc = getattr(params, 'use_score_fc', False)
        self.score_fc_hidden_dim = int(getattr(params, 'score_fc_hidden_dim', 128))
        if self.score_fc_hidden_dim <= 0:
            self.score_fc_hidden_dim = 128
        self.score_fc_dropout = float(getattr(params, 'score_fc_dropout', 0.0))
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        if self.use_selective_agg:
            self.sea_gate = GlobalCalibrationSEA(
                self.hidden_dim,
                self.sea_hidden_dim,
                self.sea_global_hidden_dim,
                dropout=self.sea_dropout,
                global_dropout=self.sea_global_dropout,
                global_eta=self.sea_global_eta,
                pool_temp=self.sea_pool_temp,
            )
            print(
                f'==> SEA: enabled (sea_hidden_dim={self.sea_hidden_dim}, '
                f'global_hidden_dim={self.sea_global_hidden_dim}, dropout={self.sea_dropout}, '
                f'global_dropout={self.sea_global_dropout}, global_eta={self.sea_global_eta}, '
                f'pool_temp={self.sea_pool_temp})'
            )
        else:
            self.sea_gate = None
        if self.use_score_fc:
            self.score_fc_head = ScoreFCHead(self.hidden_dim, self.score_fc_hidden_dim, self.score_fc_dropout)
            self.score_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
            print(f'==> ScoreFC: enabled (hidden_dim={self.score_fc_hidden_dim}, dropout={self.score_fc_dropout})')
        else:
            self.score_fc_head = None
            self.score_rela_embed = None
        
        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        
    def forward(self, q_sub, q_rel, subgraph_data, mode='train'):
        ''' forward with extra propagation '''
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        
        # initialize the hidden
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]
        
        # propagation
        for i in range(self.n_layer):
            # forward
            hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                        shortcut=self.params.shortcut, node_batch_idxs=batch_idxs,
                                        use_selective_agg=self.use_selective_agg, sea_gate=self.sea_gate)
            
            # act_signal is a binary (0/1) tensor 
            # that 1 for non-activated entities and 0 for activated entities
            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
            
            if self.params.concatHidden: hidden_list.append(hidden)

        hidden_last = hidden

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = self.W_final(hidden).squeeze(-1)        
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)

        if self.use_score_fc:
            anchor_hidden = hidden_last[query_sub_idxs][batch_idxs]
            query_hidden = self.score_rela_embed(q_rel)[batch_idxs]
            score_fc = self.score_fc_head(hidden_last, anchor_hidden, query_hidden).squeeze(-1)
            scores = scores + score_fc
        
        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        scores_all[batch_idxs, abs_idxs] = scores

        return scores_all
