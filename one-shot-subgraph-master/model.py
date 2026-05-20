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
        self.use_rel_context = bool(getattr(params, 'use_rel_context', False))
        self.use_eckge_readout = bool(getattr(params, 'use_eckge_readout', False))
        rel_context_dim = getattr(params, 'rel_context_dim', None)
        eckge_hidden_dim = getattr(params, 'eckge_hidden_dim', None)
        self.rel_context_dim = int(rel_context_dim) if rel_context_dim is not None else 32
        self.eckge_hidden_dim = int(eckge_hidden_dim) if eckge_hidden_dim is not None else 32
        self.eckge_decoder = getattr(params, 'eckge_decoder', None) or 'distmult'
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
        if self.use_rel_context or self.use_eckge_readout:
            self.rel_context_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)

        if self.use_rel_context:
            self.rel_ctx_query_proj = nn.Linear(self.hidden_dim, self.rel_context_dim, bias=False)
            self.rel_ctx_edge_proj = nn.Linear(self.hidden_dim, self.rel_context_dim, bias=False)
            self.rel_ctx_score = nn.Linear(self.rel_context_dim, 1)
            self.rel_ctx_gate = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        if self.use_eckge_readout:
            self.eckge_context = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.eckge_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.eckge_hidden_dim, self.hidden_dim),
                nn.Tanh(),
            )
            self.eckge_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.eckge_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.eckge_hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
            )

    def query_relation_context(self, q_rel, batch_sampled_edges, edge_batch_idxs, n_query):
        q_base = self.rel_context_embed(q_rel)
        if not self.use_rel_context:
            return q_base

        edge_rel = batch_sampled_edges[:, 1]
        edge_rel_embed = self.rel_context_embed(edge_rel)
        edge_query_embed = q_base[edge_batch_idxs]
        attn_input = self.rel_ctx_query_proj(edge_query_embed) + self.rel_ctx_edge_proj(edge_rel_embed)
        attn_logit = self.rel_ctx_score(torch.relu(attn_input))
        attn = scatter_softmax(attn_logit, edge_batch_idxs, n_query)
        rel_context = scatter(attn * edge_rel_embed, index=edge_batch_idxs, dim=0, dim_size=n_query, reduce='sum')

        gate = torch.sigmoid(self.rel_ctx_gate(torch.cat([q_base, rel_context, q_base * rel_context], dim=-1)))
        return (1 - gate) * q_base + gate * rel_context

    def eckge_readout(self, node_hidden, q_rel_context, batch_idxs, query_sub_idxs):
        q_context = q_rel_context[batch_idxs]
        cand_context = self.eckge_context(torch.cat([node_hidden, q_context, node_hidden * q_context], dim=-1))
        gate = self.eckge_gate(torch.cat([q_context, cand_context, q_context * cand_context], dim=-1))
        dynamic_rel = (1 - gate) * q_context + gate * cand_context

        src_hidden = node_hidden[query_sub_idxs][batch_idxs]
        if self.eckge_decoder == 'transe':
            return -torch.norm(src_hidden + dynamic_rel - node_hidden, p=1, dim=-1)
        return torch.sum(src_hidden * dynamic_rel * node_hidden, dim=-1)

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

        q_rel_context = None
        if self.use_rel_context or self.use_eckge_readout:
            q_rel_context = self.query_relation_context(q_rel, batch_sampled_edges, edge_batch_idxs, n)

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
                q_rel_embed_override=q_rel_context if self.use_rel_context else None,
            )

            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1 - act_signal).unsqueeze(-1)
            h0 = h0 * (1 - act_signal).unsqueeze(-1).unsqueeze(0)

            if self.params.concatHidden:
                hidden_list.append(hidden)

        node_hidden = hidden
        if self.params.concatHidden:
            hidden = torch.cat(hidden_list, dim=-1)

        if self.use_eckge_readout:
            scores = self.eckge_readout(node_hidden, q_rel_context, batch_idxs, query_sub_idxs)
        elif self.params.readout == 'linear':
            scores = self.W_final(hidden).squeeze(-1)
        elif self.params.readout == 'multiply':
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)

        scores_all = scores.new_zeros((n, self.loader.n_ent))
        scores_all[batch_idxs, abs_idxs] = scores
        return scores_all
