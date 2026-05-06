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
        use_edge_reliability=False,
        edge_rel_hidden_dim=None,
        edge_rel_out_dim=None,
        use_rel_smoothing=False,
        rel_smooth_tau=1.0,
        rel_smooth_lambda=0.1,
    ):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_edge_reliability = use_edge_reliability
        self.use_rel_smoothing = use_rel_smoothing
        self.rel_smooth_tau = rel_smooth_tau
        self.rel_smooth_lambda = rel_smooth_lambda

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        if self.use_edge_reliability:
            edge_rel_hidden_dim = edge_rel_hidden_dim if edge_rel_hidden_dim is not None else in_dim
            edge_rel_out_dim = edge_rel_out_dim if edge_rel_out_dim is not None else attn_dim
            self.edge_rel_hidden_dim = edge_rel_hidden_dim
            self.edge_rel_out_dim = edge_rel_out_dim
            self.node_rel_proj = nn.Sequential(
                nn.Linear(in_dim + in_dim, edge_rel_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_rel_hidden_dim, edge_rel_out_dim),
            )
            self.edge_rel_proj = nn.Linear(in_dim + in_dim, edge_rel_out_dim, bias=False)

        if self.use_rel_smoothing:
            self.rel_smooth_proj = nn.Linear(in_dim, in_dim, bias=False)

    def _smoothed_relation_bank(self):
        if not self.use_rel_smoothing:
            return self.rela_embed.weight

        rel_bank = self.rela_embed.weight
        smooth_logits = torch.matmul(
            self.rel_smooth_proj(rel_bank), rel_bank.transpose(0, 1)
        ) / max(self.rel_smooth_tau, 1e-6)
        smooth_attn = torch.softmax(smooth_logits, dim=-1)
        smooth_bank = torch.matmul(smooth_attn, rel_bank)
        return (1.0 - self.rel_smooth_lambda) * rel_bank + self.rel_smooth_lambda * smooth_bank

    def forward(
        self,
        q_sub,
        q_rel,
        batch_idxs,
        r_idx,
        hidden,
        edges,
        n_node,
        shortcut=False,
    ):
        sub = edges[:, 0]
        rel = edges[:, 1]
        obj = edges[:, 2]
        hs = hidden[sub]
        rel_bank = self._smoothed_relation_bank()
        hr = rel_bank[rel]
        q_rel_bank = rel_bank[q_rel]
        h_qr = q_rel_bank[r_idx]

        alpha_input = self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)
        alpha = torch.sigmoid(self.w_alpha(torch.relu(alpha_input)))

        if self.use_edge_reliability:
            # Project node-query states once per layer, then score each edge with
            # a lightweight relation/query offset.
            node_query = q_rel_bank[batch_idxs]
            node_rel_repr = self.node_rel_proj(torch.cat([hidden, node_query], dim=-1))
            rel_offset = self.edge_rel_proj(torch.cat([hr, h_qr], dim=-1))
            rel_score = torch.sum(
                node_rel_repr[sub] * (node_rel_repr[obj] + rel_offset),
                dim=-1,
                keepdim=True,
            ) / (self.edge_rel_out_dim ** 0.5)
            reliability = torch.sigmoid(rel_score)
        else:
            reliability = 1.0

        message = reliability * alpha * (hs * hr)
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
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
        self.use_edge_reliability = bool(getattr(params, 'use_edge_reliability', False))
        self.edge_rel_hidden_dim = getattr(params, 'edge_rel_hidden_dim', None)
        self.edge_rel_out_dim = getattr(params, 'edge_rel_out_dim', None)
        self.use_rel_smoothing = bool(getattr(params, 'use_rel_smoothing', False))
        self.rel_smooth_tau = float(getattr(params, 'rel_smooth_tau', 1.0))
        self.rel_smooth_lambda = float(getattr(params, 'rel_smooth_lambda', 0.1))

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
                    use_edge_reliability=self.use_edge_reliability,
                    edge_rel_hidden_dim=self.edge_rel_hidden_dim,
                    edge_rel_out_dim=self.edge_rel_out_dim,
                    use_rel_smoothing=self.use_rel_smoothing,
                    rel_smooth_tau=self.rel_smooth_tau,
                    rel_smooth_lambda=self.rel_smooth_lambda,
                )
            )
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        if self.params.initializer == 'relation':
            self.query_rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer + 1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, q_sub, q_rel, subgraph_data, mode='train'):
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        device = q_rel.device
        h0 = torch.zeros((1, n_node, self.hidden_dim), device=device)
        hidden = torch.zeros(n_node, self.hidden_dim, device=device)

        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)

        if self.params.concatHidden:
            hidden_list = [hidden]

        for i in range(self.n_layer):
            hidden = self.gnn_layers[i](
                q_sub,
                q_rel,
                batch_idxs,
                edge_batch_idxs,
                hidden,
                batch_sampled_edges,
                n_node,
                shortcut=self.params.shortcut,
            )

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

        scores_all = scores.new_zeros((n, self.loader.n_ent))
        scores_all[batch_idxs, abs_idxs] = scores
        return scores_all
