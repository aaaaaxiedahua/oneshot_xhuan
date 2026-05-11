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
        use_evidence_fusion=False,
        fusion_hidden_dim=None,
    ):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_evidence_fusion = use_evidence_fusion

        self.rela_embed = nn.Embedding(2 * n_rel + 1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        if self.use_evidence_fusion:
            self.Wo_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.w_alpha = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

        if self.use_evidence_fusion:
            fusion_hidden_dim = int(fusion_hidden_dim) if fusion_hidden_dim is not None else min(32, in_dim)
            self.evidence_gate = nn.Sequential(
                nn.Linear(in_dim * 3, fusion_hidden_dim),
                nn.ReLU(),
                nn.Linear(fusion_hidden_dim, in_dim),
                nn.Sigmoid(),
            )

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

        alpha_input = self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr)
        if self.use_evidence_fusion:
            ho = hidden[obj]
            alpha_input = alpha_input + self.Wo_attn(ho)
        alpha = torch.sigmoid(self.w_alpha(torch.relu(alpha_input)))

        message = alpha * (hs * hr)
        if self.use_evidence_fusion:
            mean_alpha = scatter(alpha, index=obj, dim=0, dim_size=n_node, reduce='mean')
            strong_mask = (alpha >= mean_alpha[obj]).float()
            strong_agg = scatter(message * strong_mask, index=obj, dim=0, dim_size=n_node, reduce='sum')
            weak_agg = scatter(message * (1 - strong_mask), index=obj, dim=0, dim_size=n_node, reduce='sum')
            node_h_qr = self.rela_embed(q_rel)[batch_idxs]
            gate = self.evidence_gate(torch.cat([strong_agg, weak_agg, node_h_qr], dim=-1))
            message_agg = gate * strong_agg + (1 - gate) * weak_agg
        else:
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
        self.use_evidence_fusion = bool(getattr(params, 'use_evidence_fusion', False))
        self.fusion_hidden_dim = getattr(params, 'fusion_hidden_dim', None)

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
                    use_evidence_fusion=self.use_evidence_fusion,
                    fusion_hidden_dim=self.fusion_hidden_dim,
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
