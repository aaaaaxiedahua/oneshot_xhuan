import torch
import torch.nn as nn
from torch_scatter import scatter


def scatter_softmax(src, index, dim_size):
    max_value = scatter(src, index=index, dim=0, dim_size=dim_size, reduce='max')
    exp_src = torch.exp(src - max_value[index])
    normalizer = scatter(exp_src, index=index, dim=0, dim_size=dim_size, reduce='sum')
    return exp_src / normalizer[index].clamp_min(1e-12)


class GNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        attn_dim,
        n_rel,
        act=lambda x: x,
        use_hier_attn=False,
        use_high_order=False,
        high_hidden_dim=32,
        high_topk=8,
        high_dropout=0.0,
        high_lambda=0.7,
    ):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_hier_attn = use_hier_attn
        self.use_high_order = use_high_order
        self.high_topk = high_topk
        self.high_lambda = high_lambda

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        if self.use_hier_attn:
            self.Wo_rel_attn = nn.Linear(in_dim, attn_dim, bias=False)
            self.Wr_rel_attn = nn.Linear(in_dim, attn_dim, bias=False)
            self.Wq_rel_attn = nn.Linear(in_dim, attn_dim)
            self.w_rel_alpha = nn.Linear(attn_dim, 1)
            self.Wx_ent_attn = nn.Linear(in_dim, attn_dim, bias=False)
            self.Wr_ent_attn = nn.Linear(in_dim, attn_dim, bias=False)
            self.Wq_ent_attn = nn.Linear(in_dim, attn_dim)
            self.w_ent_alpha = nn.Linear(attn_dim, 1)

        if self.use_high_order:
            self.high_first = nn.Sequential(
                nn.Linear(in_dim * 3, high_hidden_dim),
                nn.ReLU(),
                nn.Dropout(high_dropout),
                nn.Linear(high_hidden_dim, in_dim, bias=False),
            )
            self.high_second = nn.Sequential(
                nn.Linear(in_dim * 4, high_hidden_dim),
                nn.ReLU(),
                nn.Dropout(high_dropout),
                nn.Linear(high_hidden_dim, in_dim, bias=False),
            )

    @staticmethod
    def grouped_topk_mask(score, group_index, k):
        if k <= 0 or score.numel() == 0:
            return torch.ones_like(score, dtype=torch.bool)
        order_score = torch.argsort(score, descending=True)
        sorted_group_by_score = group_index[order_score]
        order_group = torch.argsort(sorted_group_by_score)
        order = order_score[order_group]
        sorted_group = group_index[order]
        pos = torch.arange(score.numel(), device=score.device)
        is_new = torch.ones_like(sorted_group, dtype=torch.bool)
        is_new[1:] = sorted_group[1:] != sorted_group[:-1]
        start_pos = torch.zeros_like(pos)
        start_pos[is_new] = pos[is_new]
        start_pos = torch.cummax(start_pos, dim=0).values
        rank = pos - start_pos
        keep = torch.zeros_like(score, dtype=torch.bool)
        keep[order[rank < k]] = True
        return keep

    def hierarchical_attention(self, hs, hr, h_qr, hidden_obj, rel, obj, n_node):
        n_relation = 2 * self.n_rel + 1
        group_key = obj * n_relation + rel
        unique_group, inverse = torch.unique(group_key, sorted=False, return_inverse=True)
        n_group = unique_group.shape[0]

        rel_input = self.Wo_rel_attn(hidden_obj) + self.Wr_rel_attn(hr) + self.Wq_rel_attn(h_qr)
        rel_logit_edge = self.w_rel_alpha(torch.tanh(rel_input)).squeeze(-1)
        rel_logit_group = scatter(rel_logit_edge, index=inverse, dim=0, dim_size=n_group, reduce='max')
        group_obj = scatter(obj, index=inverse, dim=0, dim_size=n_group, reduce='max')
        rel_alpha_group = scatter_softmax(rel_logit_group.unsqueeze(-1), group_obj, n_node).squeeze(-1)

        ent_input = self.Wx_ent_attn(hs) + self.Wr_ent_attn(hr) + self.Wq_ent_attn(h_qr)
        ent_logit = self.w_ent_alpha(torch.tanh(ent_input))
        ent_alpha = scatter_softmax(ent_logit, inverse, n_group)
        return rel_alpha_group[inverse].unsqueeze(-1) * ent_alpha

    def high_order_update(self, hidden, hs, hr, h_qr, rel, sub, obj, alpha, pair_hidden, n_node):
        alpha_score = alpha.squeeze(-1).detach()
        keep_first = self.grouped_topk_mask(alpha_score, obj, self.high_topk)
        first_input = torch.cat([hs, hr, h_qr], dim=-1)
        first_msg = self.high_first(first_input) * alpha * keep_first.unsqueeze(-1).float()
        high_mid = scatter(first_msg, index=obj, dim=0, dim_size=n_node, reduce='sum')

        keep_second = self.grouped_topk_mask(alpha_score, obj, self.high_topk)
        valid_second = keep_second & (torch.norm(high_mid[sub], p=2, dim=-1) > 0)
        second_input = torch.cat([high_mid[sub], hidden[sub], hr, h_qr], dim=-1)
        second_msg = self.high_second(second_input) * alpha * valid_second.unsqueeze(-1).float()
        high_agg = scatter(second_msg, index=obj, dim=0, dim_size=n_node, reduce='sum')
        high_hidden = self.act(self.W_h(high_agg))

        has_high = scatter(valid_second.float(), index=obj, dim=0, dim_size=n_node, reduce='sum') > 0
        deviation = torch.norm(high_hidden - pair_hidden, p=2, dim=-1, keepdim=True)
        beta = self.high_lambda / (self.high_lambda + deviation.clamp_min(1e-12))
        beta = beta * has_high.unsqueeze(-1).float()
        return (1 - beta) * pair_hidden + beta * high_hidden

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
        q_rel_embed_override=None,
    ):
        sub = edges[:, 0]
        rel = edges[:, 1]
        obj = edges[:, 2]

        hs = hidden[sub]
        hr = self.rela_embed(rel)
        if q_rel_embed_override is None:
            h_qr = self.rela_embed(q_rel)[r_idx]
        else:
            h_qr = q_rel_embed_override[r_idx]

        raw_message = hs * hr

        if self.use_hier_attn:
            alpha = self.hierarchical_attention(hs, hr, h_qr, hidden[obj], rel, obj, n_node)
        else:
            alpha_input = self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)
            alpha_logit = self.w_alpha(torch.relu(alpha_input))
            alpha = torch.sigmoid(alpha_logit)

        message = alpha * raw_message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new = self.act(self.W_h(message_agg))

        if shortcut:
            hidden_new = hidden_new + hidden

        if self.use_high_order:
            hidden_new = self.high_order_update(hidden, hs, hr, h_qr, rel, sub, obj, alpha, hidden_new, n_node)

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
        self.use_high_order = bool(getattr(params, 'use_high_order', False))
        self.use_hier_attn = bool(getattr(params, 'use_hier_attn', False))
        high_hidden_dim = getattr(params, 'high_hidden_dim', None)
        high_topk = getattr(params, 'high_topk', None)
        high_dropout = getattr(params, 'high_dropout', None)
        high_lambda = getattr(params, 'high_lambda', None)
        self.high_hidden_dim = int(high_hidden_dim) if high_hidden_dim is not None else 32
        self.high_topk = int(high_topk) if high_topk is not None else 8
        self.high_dropout = float(high_dropout) if high_dropout is not None else 0.05
        self.high_lambda = float(high_lambda) if high_lambda is not None else 0.7
        self.readout_dim = self.hidden_dim * (self.n_layer + 1) if params.concatHidden else self.hidden_dim

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
                    use_hier_attn=self.use_hier_attn,
                    use_high_order=self.use_high_order,
                    high_hidden_dim=self.high_hidden_dim,
                    high_topk=self.high_topk,
                    high_dropout=self.high_dropout,
                    high_lambda=self.high_lambda,
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
            hidden_new = self.gnn_layers[i](
                q_sub,
                q_rel,
                batch_idxs,
                edge_batch_idxs,
                hidden,
                batch_sampled_edges,
                n_node,
                shortcut=self.params.shortcut,
            )

            act_signal = (hidden_new.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden_new)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1 - act_signal).unsqueeze(-1)
            h0 = h0 * (1 - act_signal).unsqueeze(-1).unsqueeze(0)

            if self.params.concatHidden:
                hidden_list.append(hidden)

        if self.params.concatHidden:
            hidden = torch.cat(hidden_list, dim=-1)

        if self.params.readout == 'linear':
            scores = self.W_final(hidden).squeeze(-1)
        elif self.params.readout == 'multiply':
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)

        scores_all = scores.new_zeros((n, self.loader.n_ent))
        scores_all[batch_idxs, abs_idxs] = scores
        return scores_all
