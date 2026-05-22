import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


class GNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        attn_dim,
        n_rel,
        act=lambda x: x,
    ):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

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
        self.use_qmgf = bool(getattr(params, 'use_qmgf', False))
        self.use_ltsb = bool(getattr(params, 'use_ltsb', False))
        qmgf_hidden_dim = getattr(params, 'qmgf_hidden_dim', None)
        qmgf_temperature = getattr(params, 'qmgf_temperature', None)
        type_bias_weight = getattr(params, 'type_bias_weight', None)
        self.qmgf_hidden_dim = int(qmgf_hidden_dim) if qmgf_hidden_dim is not None else 32
        self.qmgf_temperature = float(qmgf_temperature) if qmgf_temperature is not None else 1.0
        self.type_bias_weight = float(type_bias_weight) if type_bias_weight is not None else 0.1
        self.readout_dim = self.hidden_dim if self.use_qmgf else (
            self.hidden_dim * (self.n_layer + 1) if params.concatHidden else self.hidden_dim
        )

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
                )
            )
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        if self.params.initializer == 'relation':
            self.query_rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden and not self.use_qmgf:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer + 1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)

        if self.use_qmgf:
            self.qmgf_context = nn.Linear(self.hidden_dim * 2, self.qmgf_hidden_dim)
            self.qmgf_hidden = nn.Linear(self.hidden_dim, self.qmgf_hidden_dim, bias=False)
            self.qmgf_query = nn.Linear(self.qmgf_hidden_dim, self.qmgf_hidden_dim, bias=False)
            self.qmgf_score = nn.Linear(self.qmgf_hidden_dim, 1)

        if self.use_ltsb:
            self.type_rel_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.type_prototype = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)

    def qmgf_fusion(self, hidden_list, query_sub_idxs, q_rel_embed, batch_idxs):
        hidden_stack = torch.stack(hidden_list, dim=1)
        query_initial = hidden_list[0][query_sub_idxs]
        query_context = torch.tanh(self.qmgf_context(torch.cat([query_initial, q_rel_embed], dim=-1)))
        layer_score = self.qmgf_score(
            torch.tanh(
                self.qmgf_hidden(hidden_stack)
                + self.qmgf_query(query_context[batch_idxs]).unsqueeze(1)
            )
        ).squeeze(-1)
        layer_weight = torch.softmax(layer_score / max(self.qmgf_temperature, 1e-6), dim=1)
        return torch.sum(layer_weight.unsqueeze(-1) * hidden_stack, dim=1)

    def latent_type_bias(self, q_rel, batch_idxs, batch_sampled_edges, n_node):
        sub = batch_sampled_edges[:, 0]
        rel = batch_sampled_edges[:, 1]
        obj = batch_sampled_edges[:, 2]
        rel_context = self.type_rel_proj(self.gnn_layers[0].rela_embed(rel))
        in_context = scatter(rel_context, index=obj, dim=0, dim_size=n_node, reduce='mean')
        out_context = scatter(rel_context, index=sub, dim=0, dim_size=n_node, reduce='mean')
        type_context = 0.5 * (in_context + out_context)
        type_context = F.normalize(type_context, p=2, dim=-1)
        prototype = F.normalize(self.type_prototype(q_rel)[batch_idxs], p=2, dim=-1)
        return torch.sum(type_context * prototype, dim=-1)

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

        if self.params.concatHidden or self.use_qmgf:
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

            if self.params.concatHidden or self.use_qmgf:
                hidden_list.append(hidden)

        if self.use_qmgf:
            if self.params.initializer == 'relation':
                q_rel_embed = self.query_rela_embed(q_rel)
            else:
                q_rel_embed = self.gnn_layers[0].rela_embed(q_rel)
            hidden = self.qmgf_fusion(hidden_list, query_sub_idxs, q_rel_embed, batch_idxs)
        elif self.params.concatHidden:
            hidden = torch.cat(hidden_list, dim=-1)

        if self.params.readout == 'linear':
            scores = self.W_final(hidden).squeeze(-1)
        elif self.params.readout == 'multiply':
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)

        if self.use_ltsb:
            scores = scores + self.type_bias_weight * self.latent_type_bias(q_rel, batch_idxs, batch_sampled_edges, n_node)

        scores_all = scores.new_zeros((n, self.loader.n_ent))
        scores_all[batch_idxs, abs_idxs] = scores
        return scores_all
