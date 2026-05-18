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
        use_exp_attn=False,
        exp_attn_dim=None,
    ):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_exp_attn = use_exp_attn

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        if self.use_exp_attn:
            exp_attn_dim = int(exp_attn_dim) if exp_attn_dim is not None else attn_dim
            self.Wm_exp_attn = nn.Linear(in_dim, exp_attn_dim, bias=False)
            self.Wq_exp_attn = nn.Linear(in_dim, exp_attn_dim)
            self.w_exp_attn = nn.Linear(exp_attn_dim, 1)

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
        hr = self.rela_embed(rel)
        h_qr = self.rela_embed(q_rel)[r_idx]

        raw_message = hs * hr

        if self.use_exp_attn:
            alpha_logit = self.w_exp_attn(torch.relu(self.Wm_exp_attn(raw_message) + self.Wq_exp_attn(h_qr)))
            alpha = scatter_softmax(alpha_logit, obj, n_node)
        else:
            alpha_input = self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)
            alpha_logit = self.w_alpha(torch.relu(alpha_input))
            alpha = torch.sigmoid(alpha_logit)

        message = alpha * raw_message
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
        self.use_exp_attn = bool(getattr(params, 'use_exp_attn', False))
        self.exp_attn_dim = getattr(params, 'exp_attn_dim', None)
        self.use_role_apim = bool(getattr(params, 'use_role_apim', False))
        role_apim_dim = getattr(params, 'role_apim_dim', None)
        role_apim_topk = getattr(params, 'role_apim_topk', None)
        self.role_apim_dim = int(role_apim_dim) if role_apim_dim is not None else 64
        self.role_apim_topk = int(role_apim_topk) if role_apim_topk is not None else 20
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
                    use_exp_attn=self.use_exp_attn,
                    exp_attn_dim=self.exp_attn_dim,
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
        if self.use_role_apim:
            self.role_query_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)
            self.role_src_proj = nn.Linear(self.readout_dim + self.hidden_dim, self.role_apim_dim)
            self.role_ans_proj = nn.Linear(self.readout_dim + self.hidden_dim, self.role_apim_dim)
            self.role_rel_matrix = nn.Embedding(2 * self.n_rel + 1, self.role_apim_dim * self.role_apim_dim)

    def topk_modes(self, mode_prob):
        keep_dim = max(1, min(self.role_apim_dim, self.role_apim_topk))
        if keep_dim >= self.role_apim_dim:
            return mode_prob
        topk_idx = torch.topk(mode_prob, keep_dim, dim=-1).indices
        mask = torch.zeros_like(mode_prob).scatter_(1, topk_idx, 1.0)
        return mode_prob * mask

    def role_apim_readout(self, hidden, q_rel, batch_idxs, query_sub_idxs):
        q_embed = self.role_query_embed(q_rel)[batch_idxs]
        src_hidden = hidden[query_sub_idxs][batch_idxs]

        src_mode = torch.sigmoid(self.role_src_proj(torch.cat([src_hidden, q_embed], dim=-1)))
        ans_mode = torch.sigmoid(self.role_ans_proj(torch.cat([hidden, q_embed], dim=-1)))
        src_mode = self.topk_modes(src_mode)
        ans_mode = self.topk_modes(ans_mode)

        rel_matrix = self.role_rel_matrix(q_rel)[batch_idxs]
        rel_matrix = torch.tanh(rel_matrix.view(-1, self.role_apim_dim, self.role_apim_dim))
        src_trans = torch.bmm(src_mode.unsqueeze(1), rel_matrix).squeeze(1)
        return torch.sum(src_trans * ans_mode, dim=-1)

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

        if self.use_role_apim:
            if self.params.concatHidden:
                hidden = torch.cat(hidden_list, dim=-1)
            scores = self.role_apim_readout(hidden, q_rel, batch_idxs, query_sub_idxs)
        elif self.params.readout == 'linear':
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
