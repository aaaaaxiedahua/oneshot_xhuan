import math
import torch
import torch.nn as nn
from torch_scatter import scatter


class GNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        attn_dim,
        n_rel,
        act=lambda x: x,
        use_composed_path=False,
        path_dim=None,
    ):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_composed_path = use_composed_path
        self.path_dim = path_dim if path_dim is not None else attn_dim
        self.path_scale = math.sqrt(max(1, self.path_dim))

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        if self.use_composed_path:
            self.path_proj = nn.Linear(self.path_dim, self.path_dim, bias=False)
            self.rel_to_path = nn.Linear(in_dim, self.path_dim, bias=False)
            self.path_rel_inter = nn.Linear(self.path_dim, self.path_dim, bias=False)
            self.Wc_attn = nn.Linear(self.path_dim, attn_dim, bias=False)
            self.Wq_path_attn = nn.Linear(in_dim, self.path_dim, bias=False)
            self.path_to_message = nn.Linear(self.path_dim, in_dim, bias=False)

    def forward(
        self,
        q_sub,
        q_rel,
        r_idx,
        hidden,
        edges,
        n_node,
        shortcut=False,
        path_prev=None,
    ):
        # edges: [h, r, t]
        sub = edges[:, 0]
        rel = edges[:, 1]
        obj = edges[:, 2]
        hs = hidden[sub]
        hr = self.rela_embed(rel)
        h_qr = self.rela_embed(q_rel)[r_idx]

        if self.use_composed_path:
            if path_prev is None:
                raise ValueError('path_prev is required when use_composed_path is enabled.')

            path_sub = path_prev[sub]
            rel_path = self.rel_to_path(hr)
            composed_path = torch.tanh(
                self.path_proj(path_sub)
                + rel_path
                + self.path_rel_inter(path_sub * rel_path)
            )
            local_score = self.w_alpha(
                torch.relu(self.Ws_attn(hs) + self.Wc_attn(composed_path))
            )
            query_score = (
                torch.sum(self.Wq_path_attn(h_qr) * composed_path, dim=-1, keepdim=True)
                / self.path_scale
            )
            alpha = torch.sigmoid(local_score + query_score)
            rel_message = torch.tanh(self.path_to_message(composed_path))
            message = hs * rel_message
            path_message = scatter(
                alpha * composed_path,
                index=obj,
                dim=0,
                dim_size=n_node,
                reduce='sum',
            )
        else:
            message = hs * hr
            alpha = torch.sigmoid(
                self.w_alpha(
                    nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))
                )
            )
            path_message = None

        message = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new = self.act(self.W_h(message_agg))

        if shortcut:
            hidden_new = hidden_new + hidden

        return hidden_new, path_message


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
        self.use_composed_path = bool(getattr(params, 'use_composed_path', False))
        self.path_dim = getattr(params, 'path_dim', None) or self.attn_dim
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]

        self.gnn_layers = []
        for _ in range(self.n_layer):
            self.gnn_layers.append(
                GNNLayer(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.attn_dim,
                    self.n_rel,
                    act=act,
                    use_composed_path=self.use_composed_path,
                    path_dim=self.path_dim,
                )
            )
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        if self.use_composed_path:
            self.query_path_init = nn.Embedding(2 * self.n_rel + 1, self.path_dim)
            self.path_gate = nn.GRUCell(self.path_dim, self.path_dim)

        if self.params.initializer == 'relation':
            self.query_rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer + 1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, q_sub, q_rel, subgraph_data, mode='train'):
        """forward with extra propagation"""
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()

        if self.use_composed_path:
            path_memory = torch.zeros(n_node, self.path_dim).cuda()
            path_memory[query_sub_idxs, :] = self.query_path_init(q_rel)
        else:
            path_memory = None

        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)

        if self.params.concatHidden:
            hidden_list = [hidden]

        for i in range(self.n_layer):
            hidden, path_message = self.gnn_layers[i](
                q_sub,
                q_rel,
                edge_batch_idxs,
                hidden,
                batch_sampled_edges,
                n_node,
                shortcut=self.params.shortcut,
                path_prev=path_memory,
            )

            if self.use_composed_path:
                path_next = self.path_gate(path_message, path_memory)
                path_active = (
                    (path_message.abs().sum(-1) > 0)
                    | (path_memory.abs().sum(-1) > 0)
                ).float().unsqueeze(-1)
                path_memory = path_next * path_active

            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1 - act_signal).unsqueeze(-1)
            h0 = h0 * (1 - act_signal).unsqueeze(-1).unsqueeze(0)

            if self.params.concatHidden:
                hidden_list.append(hidden)

        if self.params.readout == 'linear':
            if self.params.concatHidden:
                hidden = torch.cat(hidden_list, dim=-1)
            scores = self.W_final(hidden).squeeze(-1)
        elif self.params.readout == 'multiply':
            if self.params.concatHidden:
                hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)

        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        scores_all[batch_idxs, abs_idxs] = scores
        return scores_all
