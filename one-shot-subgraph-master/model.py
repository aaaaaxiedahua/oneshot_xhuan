import torch
import torch.nn as nn
from torch_scatter import scatter

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x,
                 use_rule_flow=False, flow_hidden_dim=None, flow_lambda=0.3, flow_rho=0.5):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_rule_flow = use_rule_flow
        self.flow_hidden_dim = flow_hidden_dim if flow_hidden_dim is not None else attn_dim
        self.flow_lambda = flow_lambda
        self.flow_rho = flow_rho
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
        if self.use_rule_flow:
            self.W_flow_rel = nn.Linear(in_dim, in_dim, bias=False)
            self.Wc_path = nn.Linear(in_dim, in_dim, bias=False)
            self.Wr_path = nn.Linear(in_dim, in_dim, bias=False)
            self.Wqr_path = nn.Linear(in_dim, in_dim, bias=False)
            self.flow_hidden = nn.Linear(in_dim, self.flow_hidden_dim)
            self.flow_out = nn.Linear(self.flow_hidden_dim, 1)

    def _compute_flow_prior(self, hr, h_qr):
        flow_state = torch.relu(self.Wr_path(hr) + self.Wqr_path(h_qr))
        flow_state = torch.relu(self.flow_hidden(flow_state))
        return torch.sigmoid(self.flow_out(flow_state)).squeeze(-1)

    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, shortcut=False, edge_flow_prev=None):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        hs = hidden[sub]
        hr = self.rela_embed(rel) # relation embedding of each edge
        h_qr = self.rela_embed(q_rel)[r_idx] # use batch_idx to get the query relation

        # original message aggregation
        message = hs * hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        if self.use_rule_flow:
            alpha_scalar = alpha.squeeze(-1)
            if edge_flow_prev is None:
                edge_flow_prev = self._compute_flow_prior(hr, h_qr)

            incoming_flow = scatter(
                edge_flow_prev.unsqueeze(-1) * self.W_flow_rel(hr),
                index=obj,
                dim=0,
                dim_size=n_node,
                reduce='sum',
            )
            flow_context = incoming_flow[sub]
            path_state = torch.relu(self.Wc_path(flow_context) + self.Wr_path(hr) + self.Wqr_path(h_qr))
            path_state = torch.relu(self.flow_hidden(path_state))
            path_score = torch.sigmoid(self.flow_out(path_state)).squeeze(-1)
            alpha_scalar = (1 - self.flow_lambda) * alpha_scalar + self.flow_lambda * path_score
            edge_flow = self.flow_rho * edge_flow_prev + (1 - self.flow_rho) * alpha_scalar
            alpha = alpha_scalar.unsqueeze(-1)
        else:
            edge_flow = None

        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))

        if shortcut:
            hidden_new = hidden_new + hidden
        
        return hidden_new, edge_flow

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
        self.use_rule_flow = bool(getattr(params, 'use_rule_flow', False))
        self.flow_hidden_dim = getattr(params, 'flow_hidden_dim', self.attn_dim)
        self.flow_lambda = getattr(params, 'flow_lambda', 0.3)
        self.flow_rho = getattr(params, 'flow_rho', 0.5)
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(
                GNNLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.attn_dim,
                    self.n_rel,
                    act=act,
                    use_rule_flow=self.use_rule_flow,
                    flow_hidden_dim=self.flow_hidden_dim,
                    flow_lambda=self.flow_lambda,
                    flow_rho=self.flow_rho,
                )
            )
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        
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
        edge_flow = None
        for i in range(self.n_layer):
            # forward
            hidden, edge_flow = self.gnn_layers[i](
                q_sub,
                q_rel,
                edge_batch_idxs,
                hidden,
                batch_sampled_edges,
                n_node,
                shortcut=self.params.shortcut,
                edge_flow_prev=edge_flow,
            )
            
            # act_signal is a binary (0/1) tensor 
            # that 1 for non-activated entities and 0 for activated entities
            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
            
            if self.params.concatHidden: hidden_list.append(hidden)

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = self.W_final(hidden).squeeze(-1)        
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)
        
        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        scores_all[batch_idxs, abs_idxs] = scores

        return scores_all
