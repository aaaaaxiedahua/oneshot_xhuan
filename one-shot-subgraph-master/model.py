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
        self.use_layer_expert = bool(getattr(params, 'use_layer_expert', False))
        self.use_evidence_pruning = bool(getattr(params, 'use_evidence_pruning', False))
        layer_expert_dim = getattr(params, 'layer_expert_dim', None)
        pruning_expert_dim = getattr(params, 'pruning_expert_dim', None)
        layer_temperature = getattr(params, 'layer_temperature', None)
        pruning_temperature = getattr(params, 'pruning_temperature', None)
        self.layer_expert_dim = int(layer_expert_dim) if layer_expert_dim is not None else 32
        self.pruning_expert_dim = int(pruning_expert_dim) if pruning_expert_dim is not None else 32
        self.layer_temperature = float(layer_temperature) if layer_temperature is not None else 1.0
        self.pruning_temperature = float(pruning_temperature) if pruning_temperature is not None else 1.0
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
                )
            )
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        if self.params.initializer == 'relation':
            self.query_rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden and not self.use_layer_expert:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer + 1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)

        if self.use_layer_expert or self.use_evidence_pruning:
            self.expert_query_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)

        if self.use_layer_expert:
            self.layer_context = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.layer_expert_dim),
                nn.Tanh(),
            )
            self.layer_experts = nn.Parameter(torch.empty(self.n_layer + 1, self.layer_expert_dim))
            nn.init.xavier_uniform_(self.layer_experts)

        if self.use_evidence_pruning:
            self.pruning_context = nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.pruning_expert_dim),
                nn.Tanh(),
            )
            self.pruning_beta = nn.Sequential(
                nn.Linear(self.pruning_expert_dim + self.hidden_dim, self.pruning_expert_dim),
                nn.ReLU(),
                nn.Linear(self.pruning_expert_dim, 3),
            )
            self.pruning_score = nn.Linear(self.hidden_dim, 1)

    @staticmethod
    def cv_squared(usage):
        usage = usage.float()
        return (usage.std(unbiased=False) / usage.mean().clamp_min(1e-12)).pow(2)

    def build_expert_context(self, query_hidden, q_embed, context_layer):
        return context_layer(torch.cat([query_hidden, q_embed, query_hidden * q_embed], dim=-1))

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

        expert_balance_loss = hidden.new_tensor(0.0)
        query_expert_embed = None
        layer_context = None
        pruning_context = None
        query_initial_hidden = hidden[query_sub_idxs]
        if self.use_layer_expert or self.use_evidence_pruning:
            query_expert_embed = self.expert_query_embed(q_rel)
        if self.use_layer_expert:
            layer_context = self.build_expert_context(query_initial_hidden, query_expert_embed, self.layer_context)
        if self.use_evidence_pruning:
            pruning_context = self.build_expert_context(query_initial_hidden, query_expert_embed, self.pruning_context)

        if self.params.concatHidden or self.use_layer_expert:
            hidden_list = [hidden]

        for i in range(self.n_layer):
            prev_hidden = hidden
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

            if self.use_evidence_pruning:
                q_context = pruning_context[batch_idxs]
                score_evidence = self.pruning_score(prev_hidden).squeeze(-1)
                message_evidence = torch.norm(hidden_new, p=2, dim=-1)
                relation_evidence = torch.nn.functional.cosine_similarity(
                    prev_hidden,
                    query_expert_embed[batch_idxs],
                    dim=-1,
                    eps=1e-8,
                )
                expert_logits = self.pruning_beta(torch.cat([q_context, prev_hidden], dim=-1))
                expert_weights = torch.softmax(expert_logits / max(self.pruning_temperature, 1e-6), dim=-1)
                evidence = torch.stack([score_evidence, message_evidence, relation_evidence], dim=-1)
                pruning_gate = torch.sigmoid(torch.sum(expert_weights * evidence, dim=-1)).unsqueeze(-1)
                hidden_new = pruning_gate * hidden_new + (1 - pruning_gate) * prev_hidden
                expert_usage = expert_weights.sum(dim=0)
                expert_balance_loss = expert_balance_loss + self.cv_squared(expert_usage)

            act_signal = (hidden_new.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden_new)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1 - act_signal).unsqueeze(-1)
            h0 = h0 * (1 - act_signal).unsqueeze(-1).unsqueeze(0)

            if self.params.concatHidden or self.use_layer_expert:
                hidden_list.append(hidden)

        if self.use_layer_expert:
            layer_logits = torch.matmul(layer_context, self.layer_experts.t())
            layer_weights = torch.softmax(layer_logits / max(self.layer_temperature, 1e-6), dim=-1)
            hidden_stack = torch.stack(hidden_list, dim=1)
            node_layer_weights = layer_weights[batch_idxs]
            hidden = torch.sum(node_layer_weights.unsqueeze(-1) * hidden_stack, dim=1)
            expert_balance_loss = expert_balance_loss + self.cv_squared(layer_weights.sum(dim=0))
        elif self.params.concatHidden:
            hidden = torch.cat(hidden_list, dim=-1)

        if self.params.readout == 'linear':
            scores = self.W_final(hidden).squeeze(-1)
        elif self.params.readout == 'multiply':
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)

        scores_all = scores.new_zeros((n, self.loader.n_ent))
        scores_all[batch_idxs, abs_idxs] = scores
        if mode == 'train' and (self.use_layer_expert or self.use_evidence_pruning):
            return scores_all, expert_balance_loss
        return scores_all
