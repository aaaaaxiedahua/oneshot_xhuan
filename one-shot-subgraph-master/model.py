import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix as sp_csr_matrix
from torch_scatter import scatter

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
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, shortcut=False):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        hs = hidden[sub]
        hr = self.rela_embed(rel) # relation embedding of each edge
        h_qr = self.rela_embed(q_rel)[r_idx] # use batch_idx to get the query relation
        
        # message aggregation
        message = hs * hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message        
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') #ori
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))
        
        if shortcut: hidden_new = hidden_new + hidden
        
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
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
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
        for i in range(self.n_layer):
            # forward
            hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                        shortcut=self.params.shortcut)
            
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


class EdgeGNNLayer(torch.nn.Module):
    def __init__(self, hidden_dim, attn_dim, n_rel, act=lambda x:x):
        super(EdgeGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.act = act
        self.W_src = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_dst = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_qr = nn.Linear(hidden_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, edge_hidden, line_src, line_dst, q_rel_emb, n_edge, shortcut=False):
        h_src = edge_hidden[line_src]
        h_dst = edge_hidden[line_dst]
        h_qr = q_rel_emb[line_dst]

        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(
            self.W_src(h_src) + self.W_dst(h_dst) + self.W_qr(h_qr)
        )))
        msg = alpha * self.W_msg(h_src)
        agg = scatter(msg, index=line_dst, dim=0, dim_size=n_edge, reduce='sum')
        hidden_new = self.act(self.W_out(agg))

        if shortcut:
            hidden_new = hidden_new + edge_hidden

        return hidden_new


class EdgeGNN(torch.nn.Module):
    def __init__(self, params, loader):
        super(EdgeGNN, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        # edge initialization
        self.rel_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        self.query_rel_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        self.query_marker = nn.Linear(self.hidden_dim, self.hidden_dim)

        # edge GNN layers
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(EdgeGNNLayer(self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)

    def build_line_graph(self, edges, n_node):
        """Build line graph: edge i -> edge j iff tail(i) == head(j)"""
        h = edges[:, 0].cpu().numpy()
        t = edges[:, 2].cpu().numpy()
        n_edge = len(h)
        ones = np.ones(n_edge)
        edge_ids = np.arange(n_edge)

        tail_mat = sp_csr_matrix((ones, (t, edge_ids)), shape=(n_node, n_edge))
        head_mat = sp_csr_matrix((ones, (h, edge_ids)), shape=(n_node, n_edge))
        line_adj = (tail_mat.T @ head_mat).tocoo()

        src = torch.LongTensor(line_adj.row.copy()).cuda()
        dst = torch.LongTensor(line_adj.col.copy()).cuda()
        return src, dst

    def forward(self, q_sub, q_rel, subgraph_data, mode='train'):
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, edges = subgraph_data
        n_node = len(batch_idxs)
        n_edge = edges.size(0)

        # build line graph
        line_src, line_dst = self.build_line_graph(edges, n_node)

        # initialize edge hidden with relation embedding
        edge_hidden = self.rel_embed(edges[:, 1])

        # mark edges whose head is the query node
        query_node_per_edge = query_sub_idxs[edge_batch_idxs]
        is_query_head = (edges[:, 0] == query_node_per_edge)

        if self.params.initializer == 'binary':
            edge_hidden[is_query_head] = edge_hidden[is_query_head] + 1.0
        elif self.params.initializer == 'relation':
            qr_emb = self.query_rel_embed(q_rel)[edge_batch_idxs]
            edge_hidden[is_query_head] = edge_hidden[is_query_head] + self.query_marker(qr_emb[is_query_head])

        # for concatHidden: aggregate initial edge hidden to nodes
        if self.params.concatHidden:
            node_hidden_list = [scatter(edge_hidden, index=edges[:, 2], dim=0, dim_size=n_node, reduce='sum')]

        # GRU state
        h0 = torch.zeros((1, n_edge, self.hidden_dim)).cuda()
        q_rel_emb = self.query_rel_embed(q_rel)[edge_batch_idxs]

        # edge-edge message passing
        for i in range(self.n_layer):
            edge_hidden = self.gnn_layers[i](
                edge_hidden, line_src, line_dst, q_rel_emb, n_edge,
                shortcut=self.params.shortcut
            )

            act_signal = (edge_hidden.sum(-1) == 0).detach().int()
            edge_hidden = self.dropout(edge_hidden)
            edge_hidden, h0 = self.gate(edge_hidden.unsqueeze(0), h0)
            edge_hidden = edge_hidden.squeeze(0)
            edge_hidden = edge_hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)

            if self.params.concatHidden:
                node_h = scatter(edge_hidden, index=edges[:, 2], dim=0, dim_size=n_node, reduce='sum')
                node_hidden_list.append(node_h)

        # edge -> node aggregation
        node_hidden = scatter(edge_hidden, index=edges[:, 2], dim=0, dim_size=n_node, reduce='sum')

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden: node_hidden = torch.cat(node_hidden_list, dim=-1)
            scores = self.W_final(node_hidden).squeeze(-1)
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: node_hidden = torch.cat(node_hidden_list, dim=-1)
            scores = torch.sum(node_hidden * node_hidden[query_sub_idxs][batch_idxs], dim=-1)

        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        scores_all[batch_idxs, abs_idxs] = scores

        return scores_all