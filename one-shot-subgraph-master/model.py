import torch
import torch.nn as nn
from torch_scatter import scatter

class EdgeGateMLP(torch.nn.Module):
    def __init__(self, hidden_dim, gate_hidden_dim, dropout=0.0):
        super(EdgeGateMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim * 4, gate_hidden_dim)
        self.fc2 = nn.Linear(gate_hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hs, ho, hr, h_qr):
        gate_input = torch.cat([hs, ho, hr, h_qr], dim=-1)
        gate_hidden = torch.relu(self.fc1(gate_input))
        gate_hidden = self.dropout(gate_hidden)
        return torch.sigmoid(self.fc2(gate_hidden))

class TargetGate(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(TargetGate, self).__init__()
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, hidden, h_qr):
        gate_input = torch.cat([hidden, h_qr], dim=-1)
        return torch.sigmoid(self.fc(gate_input))

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
                node_batch_idxs=None, use_selective_agg=False, sea_gate=None,
                sea_target_gate=None, sea_use_target_gate=False):
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
            gate = sea_gate(hs, ho, hr, h_qr)
            attn_score = self.w_alpha(
                torch.tanh(self.Ws_attn(hs) + self.Wo_attn(ho) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))
            ).squeeze(-1)
            alpha = self._target_softmax(attn_score, obj, n_node)
            message = alpha * gate * raw_message
            message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))

        if use_selective_agg and sea_use_target_gate:
            h_qn = self.rela_embed(q_rel)[node_batch_idxs]
            tau = sea_target_gate(hidden, h_qn)
            hidden_new = (1 - tau) * hidden + tau * hidden_new
        elif shortcut:
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
        self.sea_use_target_gate = getattr(params, 'sea_use_target_gate', False)
        self.sea_hidden_dim = int(getattr(params, 'sea_hidden_dim', self.hidden_dim))
        if self.sea_hidden_dim <= 0:
            self.sea_hidden_dim = self.hidden_dim
        self.sea_dropout = float(getattr(params, 'sea_dropout', 0.0))
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
            self.sea_gate = EdgeGateMLP(self.hidden_dim, self.sea_hidden_dim, self.sea_dropout)
            self.sea_target_gate = TargetGate(self.hidden_dim) if self.sea_use_target_gate else None
            print(f'==> SEA: enabled (sea_hidden_dim={self.sea_hidden_dim}, dropout={self.sea_dropout}, target_gate={self.sea_use_target_gate})')
        else:
            self.sea_gate = None
            self.sea_target_gate = None
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
                                        use_selective_agg=self.use_selective_agg, sea_gate=self.sea_gate,
                                        sea_target_gate=self.sea_target_gate, sea_use_target_gate=self.sea_use_target_gate)
            
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
