import torch
import torch.nn as nn
from torch_scatter import scatter, scatter_max


# # ========== Module 2: Relation Composition Augmentation (RCA) ==========
# # Learns relation composition patterns from multi-hop paths and adds
# # weighted virtual edges to address graph incompleteness.
# # To disable: set --use_rca to False (default)
# #
# # Hyperparameters:
# #   --compose_dim      : embedding dim for composition scoring
# #   --max_virtual       : max virtual edges per subgraph
# #   --compose_aware     : composition-aware virtual edge embedding
# #   --rca_dropout       : dropout in composition scorer
# #   --rca_mode          : shared (all layers share) / per_layer (each layer independent)
# #   --compose_max_hop   : max hop count for path finding (2 or 3)
# class RelationComposer(nn.Module):
#     """
#     Relation Composition Augmentation (RCA).
#
#     Finds multi-hop paths (2-hop and optionally 3-hop) in the subgraph,
#     scores each composition conditioned on query relation, and adds
#     top-scoring virtual edges.
#
#     Example: (father, father) -> grandfather (2-hop)
#              (born_in, located_in, capital_of) -> geographic (3-hop)
#     """
#     def __init__(self, n_rel, hidden_dim, compose_dim=32, max_virtual=50,
#                  rca_dropout=0.1, compose_aware=False, compose_max_hop=2):
#         super().__init__()
#         self.n_rel = n_rel
#         self.hidden_dim = hidden_dim
#         self.compose_dim = compose_dim
#         self.max_virtual = max_virtual
#         self.virtual_rel_id = 2 * n_rel + 1
#         self.compose_aware = compose_aware
#         self.compose_max_hop = compose_max_hop
#
#         # Relation embeddings for composition scoring
#         self.compose_rel_embed = nn.Embedding(2 * n_rel + 1, compose_dim)
#
#         # 2-hop scorer: (r1, r2, q_rel) -> score
#         # Fixed 2-layer MLP: Linear -> ReLU -> Dropout -> Linear
#         layers_2 = [nn.Linear(3 * compose_dim, compose_dim), nn.ReLU()]
#         if rca_dropout > 0:
#             layers_2.append(nn.Dropout(rca_dropout))
#         layers_2.append(nn.Linear(compose_dim, 1))
#         self.compose_scorer_2hop = nn.Sequential(*layers_2)
#
#         # 3-hop scorer: (r1, r2, r3, q_rel) -> score
#         if compose_max_hop >= 3:
#             layers_3 = [nn.Linear(4 * compose_dim, compose_dim), nn.ReLU()]
#             if rca_dropout > 0:
#                 layers_3.append(nn.Dropout(rca_dropout))
#             layers_3.append(nn.Linear(compose_dim, 1))
#             self.compose_scorer_3hop = nn.Sequential(*layers_3)
#
#         # Composition-aware projections: (r1, r2, ...) -> GNN relation embedding
#         if compose_aware:
#             self.compose_project_2hop = nn.Linear(2 * compose_dim, hidden_dim)
#             if compose_max_hop >= 3:
#                 self.compose_project_3hop = nn.Linear(3 * compose_dim, hidden_dim)
#
#     def _find_2hop_paths(self, edges, edge_batch_idxs, n_node):
#         """
#         Find all 2-hop paths u->w->v in the batched subgraph.
#         Vectorized via sort + expand for GPU efficiency.
#
#         Returns:
#             (src, tgt, r1, r2, batch_idx) or None
#         """
#         sub, rel, obj = edges[:, 0], edges[:, 1], edges[:, 2]
#         device = edges.device
#         n_edges = edges.shape[0]
#
#         if n_edges == 0:
#             return None
#
#         # Sort edges by source node
#         sub_sorted_vals, sub_sort_order = sub.sort()
#         sub_unique, sub_counts = torch.unique_consecutive(sub_sorted_vals, return_counts=True)
#
#         sub_cum = torch.zeros(len(sub_counts) + 1, dtype=torch.long, device=device)
#         sub_cum[1:] = sub_counts.cumsum(0)
#
#         node_to_pos = torch.full((n_node,), -1, dtype=torch.long, device=device)
#         node_to_pos[sub_unique] = torch.arange(len(sub_unique), device=device)
#
#         # For each edge e1 (u->w), check if w has outgoing edges
#         intermediate_pos = node_to_pos[obj]
#         valid_mask = intermediate_pos >= 0
#         valid_in_indices = torch.where(valid_mask)[0]
#
#         if len(valid_in_indices) == 0:
#             return None
#
#         pos_of_valid = intermediate_pos[valid_in_indices]
#         out_counts = sub_counts[pos_of_valid]
#
#         # Cap total paths
#         total_paths = out_counts.sum().item()
#         if total_paths > 200000:
#             n_keep = max(1, int(len(valid_in_indices) * 200000 / total_paths))
#             perm = torch.randperm(len(valid_in_indices), device=device)[:n_keep]
#             valid_in_indices = valid_in_indices[perm]
#             pos_of_valid = intermediate_pos[valid_in_indices]
#             out_counts = sub_counts[pos_of_valid]
#             total_paths = out_counts.sum().item()
#
#         if total_paths == 0:
#             return None
#
#         # Expand
#         in_expanded = valid_in_indices.repeat_interleave(out_counts)
#
#         out_starts = sub_cum[pos_of_valid]
#         group_ids = torch.repeat_interleave(
#             torch.arange(len(out_counts), device=device), out_counts)
#         cum_counts = torch.zeros(len(out_counts) + 1, dtype=torch.long, device=device)
#         cum_counts[1:] = out_counts.cumsum(0)
#         local_offsets = torch.arange(total_paths, device=device, dtype=torch.long) - cum_counts[group_ids]
#         out_sorted_pos = torch.repeat_interleave(out_starts, out_counts) + local_offsets
#         out_expanded = sub_sort_order[out_sorted_pos]
#
#         # Filter: same batch + no self-loops
#         u = sub[in_expanded]
#         v = obj[out_expanded]
#         batch_ok = edge_batch_idxs[in_expanded] == edge_batch_idxs[out_expanded]
#         not_self = u != v
#         keep = batch_ok & not_self
#
#         if not keep.any():
#             return None
#
#         return (u[keep], v[keep],
#                 rel[in_expanded[keep]], rel[out_expanded[keep]],
#                 edge_batch_idxs[in_expanded[keep]])
#
#     def _find_3hop_paths(self, edges, edge_batch_idxs, n_node):
#         """
#         Find 3-hop paths u->w1->w2->v by extending 2-hop paths.
#         u --(r1)--> w1 --(r2)--> w2 --(r3)--> v
#
#         Returns:
#             (src, tgt, r1, r2, r3, batch_idx) or None
#         """
#         # Get 2-hop paths as starting points
#         paths_2 = self._find_2hop_paths(edges, edge_batch_idxs, n_node)
#         if paths_2 is None:
#             return None
#
#         src_2, tgt_2, r1_2, r2_2, batch_2 = paths_2
#         device = edges.device
#
#         # Subsample 2-hop paths if too many (3-hop expansion is multiplicative)
#         if len(src_2) > 50000:
#             perm = torch.randperm(len(src_2), device=device)[:50000]
#             src_2, tgt_2 = src_2[perm], tgt_2[perm]
#             r1_2, r2_2, batch_2 = r1_2[perm], r2_2[perm], batch_2[perm]
#
#         # Build outgoing edge lookup for the 3rd hop
#         sub, rel, obj = edges[:, 0], edges[:, 1], edges[:, 2]
#         sub_sorted_vals, sub_sort_order = sub.sort()
#         sub_unique, sub_counts = torch.unique_consecutive(sub_sorted_vals, return_counts=True)
#         sub_cum = torch.zeros(len(sub_counts) + 1, dtype=torch.long, device=device)
#         sub_cum[1:] = sub_counts.cumsum(0)
#
#         node_to_pos = torch.full((n_node,), -1, dtype=torch.long, device=device)
#         node_to_pos[sub_unique] = torch.arange(len(sub_unique), device=device)
#
#         # For each 2-hop endpoint tgt_2, find outgoing edges
#         ext_pos = node_to_pos[tgt_2]
#         valid_mask = ext_pos >= 0
#         valid_indices = torch.where(valid_mask)[0]
#
#         if len(valid_indices) == 0:
#             return None
#
#         pos_of_valid = ext_pos[valid_indices]
#         out_counts = sub_counts[pos_of_valid]
#
#         # Cap 3-hop paths
#         total = out_counts.sum().item()
#         if total > 100000:
#             n_keep = max(1, int(len(valid_indices) * 100000 / total))
#             perm = torch.randperm(len(valid_indices), device=device)[:n_keep]
#             valid_indices = valid_indices[perm]
#             pos_of_valid = ext_pos[valid_indices]
#             out_counts = sub_counts[pos_of_valid]
#             total = out_counts.sum().item()
#
#         if total == 0:
#             return None
#
#         # Expand 2-hop paths by 3rd hop outgoing count
#         idx_expanded = valid_indices.repeat_interleave(out_counts)
#
#         # Build 3rd hop edge indices
#         out_starts = sub_cum[pos_of_valid]
#         group_ids = torch.repeat_interleave(
#             torch.arange(len(out_counts), device=device), out_counts)
#         cum_counts = torch.zeros(len(out_counts) + 1, dtype=torch.long, device=device)
#         cum_counts[1:] = out_counts.cumsum(0)
#         local_offsets = torch.arange(total, device=device, dtype=torch.long) - cum_counts[group_ids]
#         out_sorted_pos = torch.repeat_interleave(out_starts, out_counts) + local_offsets
#         e3_indices = sub_sort_order[out_sorted_pos]
#
#         # Assemble 3-hop path info
#         u = src_2[idx_expanded]
#         v = obj[e3_indices]
#         hop_r1 = r1_2[idx_expanded]
#         hop_r2 = r2_2[idx_expanded]
#         hop_r3 = rel[e3_indices]
#         hop_batch = batch_2[idx_expanded]
#
#         # Filter: same batch + no self-loops
#         batch_ok = hop_batch == edge_batch_idxs[e3_indices]
#         not_self = u != v
#         keep = batch_ok & not_self
#
#         if not keep.any():
#             return None
#
#         return (u[keep], v[keep],
#                 hop_r1[keep], hop_r2[keep], hop_r3[keep],
#                 hop_batch[keep])
#
#     def forward(self, edges, q_rel, edge_batch_idxs, n_node, batch_size):
#         """
#         Generate virtual edges from multi-hop relation compositions.
#
#         Returns:
#             augmented_edges, edge_weights, augmented_edge_batch_idxs,
#             virtual_rel_embeds (or None), n_virtual
#         """
#         device = edges.device
#         n_original = edges.shape[0]
#
#         # Collect candidates from different hop counts
#         cand_src, cand_tgt, cand_scores, cand_batch = [], [], [], []
#         cand_rels = []   # padded to [n, 3]: [r1, r2, -1] for 2-hop, [r1, r2, r3] for 3-hop
#         cand_hop = []    # hop type: 2 or 3
#
#         # --- 2-hop paths ---
#         paths_2 = self._find_2hop_paths(edges, edge_batch_idxs, n_node)
#         if paths_2 is not None:
#             src, tgt, r1, r2, pbatch = paths_2
#             r1_e = self.compose_rel_embed(r1)
#             r2_e = self.compose_rel_embed(r2)
#             qr_e = self.compose_rel_embed(q_rel[pbatch])
#             scores = torch.sigmoid(self.compose_scorer_2hop(
#                 torch.cat([r1_e, r2_e, qr_e], dim=-1)).squeeze(-1))
#             # Dedup per (u,v): take max score
#             keys = src.long() * n_node + tgt.long()
#             ukeys, inv = torch.unique(keys, return_inverse=True)
#             max_vals, max_idx = scatter_max(scores, inv, dim=0, dim_size=len(ukeys))
#             n_c = len(max_idx)
#             cand_src.append(src[max_idx])
#             cand_tgt.append(tgt[max_idx])
#             cand_scores.append(max_vals)
#             cand_batch.append(pbatch[max_idx])
#             cand_rels.append(torch.stack([
#                 r1[max_idx], r2[max_idx],
#                 torch.full((n_c,), -1, device=device, dtype=torch.long)], dim=1))
#             cand_hop.append(torch.full((n_c,), 2, device=device, dtype=torch.long))
#
#         # --- 3-hop paths (if enabled) ---
#         if self.compose_max_hop >= 3:
#             paths_3 = self._find_3hop_paths(edges, edge_batch_idxs, n_node)
#             if paths_3 is not None:
#                 src, tgt, r1, r2, r3, pbatch = paths_3
#                 r1_e = self.compose_rel_embed(r1)
#                 r2_e = self.compose_rel_embed(r2)
#                 r3_e = self.compose_rel_embed(r3)
#                 qr_e = self.compose_rel_embed(q_rel[pbatch])
#                 scores = torch.sigmoid(self.compose_scorer_3hop(
#                     torch.cat([r1_e, r2_e, r3_e, qr_e], dim=-1)).squeeze(-1))
#                 keys = src.long() * n_node + tgt.long()
#                 ukeys, inv = torch.unique(keys, return_inverse=True)
#                 max_vals, max_idx = scatter_max(scores, inv, dim=0, dim_size=len(ukeys))
#                 n_c = len(max_idx)
#                 cand_src.append(src[max_idx])
#                 cand_tgt.append(tgt[max_idx])
#                 cand_scores.append(max_vals)
#                 cand_batch.append(pbatch[max_idx])
#                 cand_rels.append(torch.stack([
#                     r1[max_idx], r2[max_idx], r3[max_idx]], dim=1))
#                 cand_hop.append(torch.full((n_c,), 3, device=device, dtype=torch.long))
#
#         # --- No candidates ---
#         if len(cand_src) == 0:
#             edge_weights = torch.ones(n_original, device=device)
#             return edges, edge_weights, edge_batch_idxs, None, 0
#
#         # --- Combine and final dedup (2-hop and 3-hop may target same (u,v)) ---
#         all_src = torch.cat(cand_src)
#         all_tgt = torch.cat(cand_tgt)
#         all_scores = torch.cat(cand_scores)
#         all_batch = torch.cat(cand_batch)
#         all_rels = torch.cat(cand_rels, dim=0)   # [N, 3]
#         all_hop = torch.cat(cand_hop)             # [N]
#
#         edge_keys = all_src.long() * n_node + all_tgt.long()
#         unique_keys, inverse = torch.unique(edge_keys, return_inverse=True)
#         final_scores, final_idx = scatter_max(all_scores, inverse, dim=0, dim_size=len(unique_keys))
#         final_src = all_src[final_idx]
#         final_tgt = all_tgt[final_idx]
#         final_batch = all_batch[final_idx]
#         final_rels = all_rels[final_idx]
#         final_hop = all_hop[final_idx]
#
#         # --- Select top-M virtual edges ---
#         n_virtual = min(self.max_virtual * batch_size, len(final_scores))
#         if n_virtual < len(final_scores):
#             topk_vals, topk_idx = torch.topk(final_scores, n_virtual)
#             final_src = final_src[topk_idx]
#             final_tgt = final_tgt[topk_idx]
#             final_batch = final_batch[topk_idx]
#             final_scores = topk_vals
#             final_rels = final_rels[topk_idx]
#             final_hop = final_hop[topk_idx]
#
#         # --- Build virtual edges ---
#         virtual_edges = torch.stack([
#             final_src,
#             torch.full_like(final_src, self.virtual_rel_id),
#             final_tgt
#         ], dim=1)
#         augmented_edges = torch.cat([edges, virtual_edges], dim=0)
#         real_weights = torch.ones(n_original, device=device)
#         edge_weights = torch.cat([real_weights, final_scores], dim=0)
#         augmented_batch = torch.cat([edge_batch_idxs, final_batch], dim=0)
#
#         # --- Composition-aware virtual edge embeddings ---
#         virtual_rel_embeds = None
#         if self.compose_aware and n_virtual > 0:
#             vr_embeds = torch.zeros(n_virtual, self.hidden_dim, device=device)
#             # 2-hop virtual edges: project (r1, r2) -> hidden_dim
#             mask_2 = (final_hop == 2)
#             if mask_2.any():
#                 r1_e = self.compose_rel_embed(final_rels[mask_2, 0])
#                 r2_e = self.compose_rel_embed(final_rels[mask_2, 1])
#                 vr_embeds[mask_2] = self.compose_project_2hop(
#                     torch.cat([r1_e, r2_e], dim=-1))
#             # 3-hop virtual edges: project (r1, r2, r3) -> hidden_dim
#             if self.compose_max_hop >= 3:
#                 mask_3 = (final_hop == 3)
#                 if mask_3.any():
#                     r1_e = self.compose_rel_embed(final_rels[mask_3, 0])
#                     r2_e = self.compose_rel_embed(final_rels[mask_3, 1])
#                     r3_e = self.compose_rel_embed(final_rels[mask_3, 2])
#                     vr_embeds[mask_3] = self.compose_project_3hop(
#                         torch.cat([r1_e, r2_e, r3_e], dim=-1))
#             virtual_rel_embeds = vr_embeds
#
#         return augmented_edges, edge_weights, augmented_batch, virtual_rel_embeds, n_virtual
# # ========== End Module 2: Relation Composition Augmentation (RCA) ==========


class QTARRouter(torch.nn.Module):
    """Query-Target Adaptive Routing (QTAR)."""
    def __init__(self, hidden_dim, n_rel, router_hidden=64, min_edges=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.min_edges = max(1, int(min_edges))
        self.rel_embed = nn.Embedding(2 * n_rel + 1, hidden_dim)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 4, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, 1),
        )

    def _normalize_evidence(self, x):
        maxv = torch.max(x)
        if maxv.item() <= 1e-12:
            return x
        return x / (maxv + 1e-12)

    def forward(self, hidden, edges, q_rel, edge_batch_idxs, evidence, keep_ratio=1.0):
        # edges: [n_edges, 3], hidden: [n_nodes, d], evidence: [n_nodes]
        sub, rel, obj = edges[:, 0], edges[:, 1], edges[:, 2]
        hs = hidden[sub]
        ho = hidden[obj]
        hr = self.rel_embed(rel)
        hq = self.rel_embed(q_rel)[edge_batch_idxs]
        gate = torch.sigmoid(self.router(torch.cat([hs, ho, hr, hq], dim=-1)).squeeze(-1))

        route_scores = gate * evidence[sub]
        route_scores = torch.clamp(route_scores, min=1e-8)

        n_edges = int(edges.shape[0])
        if keep_ratio >= 0.999 or n_edges <= self.min_edges:
            keep_index = torch.arange(n_edges, device=edges.device)
            kept_edges = edges
            kept_batch = edge_batch_idxs
            kept_weights = route_scores
        else:
            target_keep = max(self.min_edges, int(keep_ratio * n_edges))
            target_keep = min(target_keep, n_edges)
            keep_index = torch.topk(route_scores, k=target_keep, largest=True, sorted=False).indices
            kept_edges = edges[keep_index]
            kept_batch = edge_batch_idxs[keep_index]
            kept_weights = route_scores[keep_index]

        # evidence update uses the routed edges
        kept_sub = kept_edges[:, 0]
        kept_obj = kept_edges[:, 2]
        evidence_msg = evidence[kept_sub] * kept_weights
        new_evidence = scatter(evidence_msg, index=kept_obj, dim=0, dim_size=hidden.shape[0], reduce='sum')
        new_evidence = self._normalize_evidence(new_evidence)
        return kept_edges, kept_batch, kept_weights, new_evidence


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x, n_extra_rel=0):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        # # ========== Module 2: +n_extra_rel for virtual relation embedding ==========
        # self.rela_embed = nn.Embedding(2*n_rel+1+n_extra_rel, in_dim)
        # # ========== End Module 2 ==========
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, shortcut=False, edge_weights=None):
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        hs = hidden[sub]
        hr = self.rela_embed(rel)
        h_qr = self.rela_embed(q_rel)[r_idx]

        # # ========== Module 2: Override virtual edge embeddings (compose_aware) ==========
        # if virtual_rel_embeds is not None and n_virtual > 0:
        #     hr = torch.cat([hr[:-n_virtual], virtual_rel_embeds], dim=0)
        # # ========== End Module 2 ==========

        # message aggregation
        message = hs * hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message

        if edge_weights is not None:
            message = message * edge_weights.unsqueeze(-1)

        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
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
        self.current_epoch = 0
        self.latest_aux_loss = torch.tensor(0.0)
        self.qtar_logged_epochs = set()

        # ========== QTAR ==========
        self.use_qtar = hasattr(params, 'use_qtar') and params.use_qtar
        self.qtar_ratio_start = float(getattr(params, 'qtar_ratio_start', 1.0))
        self.qtar_ratio_end = float(getattr(params, 'qtar_ratio_end', 0.8))
        self.qtar_warmup = int(getattr(params, 'qtar_warmup', 5))
        qtar_hidden = int(getattr(params, 'qtar_router_hidden', 64))
        qtar_min_edges = int(getattr(params, 'qtar_min_edges', 64))
        if self.use_qtar:
            self.qtar_router = QTARRouter(self.hidden_dim, self.n_rel, router_hidden=qtar_hidden, min_edges=qtar_min_edges)
            print(
                f'==> QTAR: enabled (ratio_start={self.qtar_ratio_start}, ratio_end={self.qtar_ratio_end}, '
                f'warmup={self.qtar_warmup}, router_hidden={qtar_hidden}, min_edges={qtar_min_edges})'
            )
        # ========== End QTAR ==========

        # # ========== Module 2 ==========
        # self.use_rca = hasattr(params, 'use_rca') and params.use_rca
        # n_extra_rel = 1 if self.use_rca else 0
        # self.rca_mode = getattr(params, 'rca_mode', 'shared') if self.use_rca else 'shared'
        # # ========== End Module 2 ==========

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)

        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        self.readout_dim = self.hidden_dim * (self.n_layer + 1) if self.params.concatHidden else self.hidden_dim
        self.use_readout_refine = hasattr(params, 'use_readout_refine') and params.use_readout_refine
        if self.use_readout_refine:
            refine_hidden = max(1, self.readout_dim // 2)
            self.readout_refine = nn.Sequential(
                nn.Linear(self.readout_dim, refine_hidden),
                nn.ReLU(),
                nn.Dropout(params.dropout),
                nn.Linear(refine_hidden, self.readout_dim),
            )
            print(f'==> ReadoutRefine: enabled (readout_dim={self.readout_dim}, hidden={refine_hidden})')
        if self.params.readout == 'linear':
            self.W_final = nn.Linear(self.readout_dim, 1, bias=False)
        elif self.params.readout == 'pair_mlp':
            self.pair_score_head = nn.Sequential(
                nn.Linear(self.readout_dim * 4, self.readout_dim),
                nn.ReLU(),
                nn.Linear(self.readout_dim, 1),
            )

        # # ========== Module 2: Create Relation Composer(s) ==========
        # if self.use_rca:
        #     composer_kwargs = dict(
        #         n_rel=self.n_rel, hidden_dim=self.hidden_dim,
        #         compose_dim=getattr(params, 'compose_dim', 32),
        #         max_virtual=getattr(params, 'max_virtual', 50),
        #         rca_dropout=getattr(params, 'rca_dropout', 0.1),
        #         compose_aware=getattr(params, 'compose_aware', False),
        #         compose_max_hop=getattr(params, 'compose_max_hop', 2),
        #     )
        #     if self.rca_mode == 'per_layer':
        #         self.relation_composers = nn.ModuleList([
        #             RelationComposer(**composer_kwargs) for _ in range(self.n_layer)])
        #     else:
        #         self.relation_composer = RelationComposer(**composer_kwargs)
        # # ========== End Module 2 ==========

    def set_epoch(self, epoch_idx):
        self.current_epoch = int(epoch_idx)

    def _layer_keep_ratio(self, layer_idx):
        if self.n_layer <= 1:
            return self.qtar_ratio_end
        frac = float(layer_idx) / float(self.n_layer - 1)
        return self.qtar_ratio_start + frac * (self.qtar_ratio_end - self.qtar_ratio_start)

    def forward(self, q_sub, q_rel, subgraph_data, mode='train'):
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        self.latest_aux_loss = torch.zeros(1, device=hidden.device).squeeze(0)
        qtar_log_key = (mode, self.current_epoch)
        should_log_qtar = self.use_qtar and mode == 'train' and qtar_log_key not in self.qtar_logged_epochs
        qtar_log_items = []
        qtar_warmup_skip = False

        # # ========== Module 2: Generate virtual edges ==========
        # edge_weights = None
        # virtual_rel_embeds = None
        # n_virtual = 0
        #
        # if self.use_rca:
        #     if self.rca_mode == 'shared':
        #         result = self.relation_composer(
        #             batch_sampled_edges, q_rel, edge_batch_idxs, n_node, n)
        #         batch_sampled_edges, edge_weights, edge_batch_idxs, virtual_rel_embeds, n_virtual = result
        #     elif self.rca_mode == 'per_layer':
        #         original_edges = batch_sampled_edges
        #         original_edge_batch_idxs = edge_batch_idxs
        # # ========== End Module 2 ==========

        # initialize
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)

        if self.params.concatHidden: hidden_list = [hidden]
        evidence = torch.zeros(n_node, device=hidden.device)
        evidence[query_sub_idxs] = 1.0

        # propagation
        for i in range(self.n_layer):
            # # ========== Module 2: per-layer virtual edges ==========
            # if self.use_rca and self.rca_mode == 'per_layer':
            #     result = self.relation_composers[i](
            #         original_edges, q_rel, original_edge_batch_idxs, n_node, n)
            #     curr_edges, curr_weights, curr_batch, curr_vr, curr_nv = result
            # else:
            #     curr_edges = batch_sampled_edges
            #     curr_weights = edge_weights
            #     curr_batch = edge_batch_idxs
            #     curr_vr = virtual_rel_embeds
            #     curr_nv = n_virtual
            # # ========== End Module 2 ==========

            curr_edges = batch_sampled_edges
            curr_batch = edge_batch_idxs
            curr_edge_weights = None

            if self.use_qtar and not (mode == 'train' and self.current_epoch < self.qtar_warmup):
                keep_ratio = self._layer_keep_ratio(i)
                curr_edges, curr_batch, curr_edge_weights, evidence = self.qtar_router(
                    hidden, batch_sampled_edges, q_rel, edge_batch_idxs, evidence,
                    keep_ratio=keep_ratio
                )
                if should_log_qtar:
                    qtar_log_items.append(
                        f'L{i}:ratio={keep_ratio:.2f},kept={int(curr_edges.shape[0])}/{int(batch_sampled_edges.shape[0])}'
                    )
            elif self.use_qtar and mode == 'train' and self.current_epoch < self.qtar_warmup and should_log_qtar:
                qtar_warmup_skip = True

            hidden = self.gnn_layers[i](q_sub, q_rel, curr_batch, hidden, curr_edges, n_node,
                                        shortcut=self.params.shortcut, edge_weights=curr_edge_weights)

            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)

            if self.params.concatHidden: hidden_list.append(hidden)

        if should_log_qtar:
            if qtar_warmup_skip:
                print(f'==> QTAR[train]: epoch={self.current_epoch}, warmup active, routing skipped')
            else:
                print(f'==> QTAR[train]: epoch={self.current_epoch}, ' + ' | '.join(qtar_log_items))
            self.qtar_logged_epochs.add(qtar_log_key)

        # readout
        if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
        if self.use_readout_refine:
            hidden = hidden + self.readout_refine(hidden)
        if self.params.readout == 'linear':
            scores = self.W_final(hidden).squeeze(-1)
        elif self.params.readout == 'multiply':
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)
        elif self.params.readout == 'pair_mlp':
            query_hidden = hidden[query_sub_idxs][batch_idxs]
            pair_feat = torch.cat(
                [hidden, query_hidden, hidden * query_hidden, torch.abs(hidden - query_hidden)],
                dim=-1,
            )
            scores = self.pair_score_head(pair_feat).squeeze(-1)
        else:
            raise ValueError(f'Unknown readout type: {self.params.readout}')

        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        scores_all[batch_idxs, abs_idxs] = scores
        return scores_all
