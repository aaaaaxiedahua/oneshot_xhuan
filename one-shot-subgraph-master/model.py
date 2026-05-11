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
        use_ctx_filter=False,
        ctx_hidden_dim=None,
        ctx_gamma=0.1,
        use_path_history=False,
    ):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_ctx_filter = use_ctx_filter
        self.ctx_gamma = ctx_gamma
        self.use_path_history = use_path_history

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        if self.use_path_history:
            self.Wp_attn = nn.Linear(in_dim, attn_dim, bias=False)
            self.path_gru = nn.GRUCell(in_dim, in_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        if self.use_ctx_filter:
            ctx_hidden_dim = ctx_hidden_dim if ctx_hidden_dim is not None else in_dim
            self.ctx_hidden_dim = ctx_hidden_dim
            self.ctx_edge_interact = nn.Sequential(
                nn.Linear(in_dim * 4, ctx_hidden_dim),
                nn.ReLU(),
                nn.Linear(ctx_hidden_dim, in_dim),
            )
            self.ctx_relation_calibrate = nn.Sequential(
                nn.Linear(in_dim * 3, ctx_hidden_dim),
                nn.ReLU(),
                nn.Linear(ctx_hidden_dim, in_dim),
            )

    def _compute_group_context(self, edge_context, group_index):
        return scatter(edge_context, group_index, dim=0, reduce='mean')

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
        path_state=None,
        group_index=None,
        group_rel=None,
        group_r_idx=None,
    ):
        sub = edges[:, 0]
        rel = edges[:, 1]
        obj = edges[:, 2]

        hs = hidden[sub]
        ho = hidden[obj]
        hr = self.rela_embed(rel)
        h_qr = self.rela_embed(q_rel)[r_idx]

        alpha_input = self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)
        if self.use_path_history:
            alpha_input = alpha_input + self.Wp_attn(path_state[sub])
        alpha = torch.sigmoid(self.w_alpha(torch.relu(alpha_input)))

        if self.use_ctx_filter:
            edge_context = self.ctx_edge_interact(torch.cat([hs, hr, ho, h_qr], dim=-1))
            group_context = self._compute_group_context(edge_context, group_index)
            group_hr = self.rela_embed(group_rel)
            group_h_qr = self.rela_embed(q_rel)[group_r_idx]
            group_delta = torch.tanh(
                self.ctx_relation_calibrate(torch.cat([group_hr, group_context, group_h_qr], dim=-1))
            )
            fused_rel = hr + self.ctx_gamma * group_delta[group_index]
            message = alpha * (hs * fused_rel)
        else:
            message = alpha * (hs * hr)

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new = self.act(self.W_h(message_agg))

        path_state_new = path_state
        if self.use_path_history:
            edge_path = self.path_gru(hr, path_state[sub])
            alpha_sum = scatter(alpha, index=obj, dim=0, dim_size=n_node, reduce='sum')[obj].clamp_min(1e-8)
            path_weight = alpha / alpha_sum
            path_state_new = scatter(path_weight * edge_path, index=obj, dim=0, dim_size=n_node, reduce='sum')

        if shortcut:
            hidden_new = hidden_new + hidden

        return hidden_new, path_state_new


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
        self.use_ctx_filter = bool(getattr(params, 'use_ctx_filter', False))
        self.ctx_hidden_dim = getattr(params, 'ctx_hidden_dim', None)
        self.ctx_gamma = float(getattr(params, 'ctx_gamma', 0.1))
        self.use_path_history = bool(getattr(params, 'use_path_history', False))

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
                    use_ctx_filter=self.use_ctx_filter,
                    ctx_hidden_dim=self.ctx_hidden_dim,
                    ctx_gamma=self.ctx_gamma,
                    use_path_history=self.use_path_history,
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
        path_state = torch.zeros(n_node, self.hidden_dim, device=device) if self.use_path_history else None

        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)

        if self.params.concatHidden:
            hidden_list = [hidden]

        if self.use_ctx_filter:
            # Cache (source, relation) grouping once per sampled subgraph and reuse it in every layer.
            rel_vocab_size = 2 * self.n_rel + 1
            edge_sub = batch_sampled_edges[:, 0]
            edge_rel = batch_sampled_edges[:, 1]
            pair_ids = edge_sub * rel_vocab_size + edge_rel
            _, group_index = torch.unique(pair_ids, return_inverse=True)
            group_rel = scatter(edge_rel, group_index, dim=0, reduce='min').long()
            group_r_idx = scatter(edge_batch_idxs, group_index, dim=0, reduce='min').long()
        else:
            group_index = None
            group_rel = None
            group_r_idx = None

        for i in range(self.n_layer):
            hidden, path_state = self.gnn_layers[i](
                q_sub,
                q_rel,
                batch_idxs,
                edge_batch_idxs,
                hidden,
                batch_sampled_edges,
                n_node,
                shortcut=self.params.shortcut,
                path_state=path_state,
                group_index=group_index,
                group_rel=group_rel,
                group_r_idx=group_r_idx,
            )

            act_signal = (hidden.sum(-1) == 0).detach().int()
            if self.use_path_history:
                path_state = path_state * (1 - act_signal).unsqueeze(-1)
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
