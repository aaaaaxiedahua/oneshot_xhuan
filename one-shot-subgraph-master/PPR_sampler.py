import networkx as nx
import pickle as pkl
import time
import numpy as np
import torch
import os
import logging
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict, OrderedDict

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

class pprSampler():
    def __init__(self, n_ent:int, n_rel:int, topk:int, topm:int, homoEdges:list, edge_index:list, data_path:str, split='train', args=None):
        ''' 
            args:
            topk: number of sampled nodes for one head entity 
            edge_index: list of triples [(h,r,t)]
            data_path: path to save the ppr/subgraphs files
        '''
        print('==> initializing ppr sampler...')
        self.args = args
        self.split = split
        self.n_ent = n_ent
        self.n_samp_ent = args.n_samp_ent
        self.n_rel = n_rel
        self.topk = topk
        self.topm = topm
        self.edge_index = edge_index
        self.data_folder = data_path
        self.homoEdges = homoEdges
        self.homoTrainGraph = self.triplesToNxGraph(self.homoEdges)
        self.ppr_savePath = os.path.join(self.data_folder, f'ppr_scores/')
        self.ppr_vector_cache = OrderedDict()
        self.ppr_cache_size = 256
        self.use_lqcd = bool(getattr(args, 'use_lqcd', False))
        self.lqcd_coarse_ratio = float(getattr(args, 'lqcd_coarse_ratio', 1.5))
        self.lqcd_fuse_lambda = float(getattr(args, 'lqcd_fuse_lambda', 0.7))
        self.lqcd_topl = max(1, int(getattr(args, 'lqcd_topl', 2)))
        self.lqcd_rel_gamma = 0.5
        self.lqcd_hub_beta = 1.0
        self.lqcd_self_loop = 1.0
        self.idd_rel = 2 * self.n_rel
        self.lqcd_logged = False
        checkPath(self.ppr_savePath)
        print('==> checking ppr scores for each entity...')
        if self.use_lqcd:
            print(
                f'==> LQCD[{self.split}]: enabled '
                f'(coarse_ratio={self.lqcd_coarse_ratio}, fuse_lambda={self.lqcd_fuse_lambda}, '
                f'topl={self.lqcd_topl}, gamma={self.lqcd_rel_gamma})'
            )
        
        for h in tqdm(range(self.n_ent), ncols=50, leave=False):
            ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
            if os.path.exists(ent_ppr_savePath):
                pass
            else:
                # with default setting to generate ppr scores
                h_ppr_scores = self.generatePPRScoresForOneEntity(h)
                pkl.dump(h_ppr_scores, open(ent_ppr_savePath, 'wb'))
        print('finished.')
        
        # build head to edges with sparse matrix
        heads, edges = [h for (h,r,t) in edge_index], list(range(len(edge_index)))
        print(len(heads), len(edges), max(heads), self.n_ent)
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(edge_index)))

        # change data type
        self.edge_index = torch.LongTensor(self.edge_index)

        # clean cache
        del self.homoEdges
        del self.homoTrainGraph
        
        # build sparse tensor self.PPR_W for matrix-computation PPR
        '''
        tmp_degree, tmp_adj = torch.zeros(self.n_ent, self.n_ent), torch.zeros(self.n_ent, self.n_ent)
        tmp_adj[self.edge_index[:,0], self.edge_index[:,2]] = 1
        tmp_degree = torch.diag(1 / torch.sum(tmp_adj, dim=1))
        self.PPR_W = torch.eye(self.n_ent) + torch.matmul(tmp_degree, tmp_adj)
        self.PPR_W = self.PPR_W.cuda()
        del tmp_adj; del tmp_degree
        '''
        
        # # ========== Module 1: Build relation prior P(v|r) ==========
        # # Makes subgraph extraction query-relation-aware by fusing learned
        # # relation prior with PPR scores. To disable: set --use_rel_prior to False
        # if hasattr(args, 'use_rel_prior') and args.use_rel_prior:
        #     self.mineRelationPatterns(edge_index)
        # # ========== End Module 1 ==========

        print('==> finish sampler initilization.')

    def updateEdges(self, edge_index):
        # co-operate with shuffle_train
        heads, edges = [h for (h,r,t) in edge_index], list(range(len(edge_index)))
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(edge_index)))
        self.edge_index = torch.LongTensor(edge_index)

    def getPPRscores(self, ent):
        ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(ent)}.pkl')
        scores = pkl.load(open(ent_ppr_savePath, 'rb'))
        return scores

    def getPPRarray(self, ent):
        ent = int(ent)
        if ent in self.ppr_vector_cache:
            arr = self.ppr_vector_cache.pop(ent)
            self.ppr_vector_cache[ent] = arr
            return arr

        scores = self.getPPRscores(ent)
        if isinstance(scores, dict):
            # NetworkX pagerank dict is usually ordered by node id in this project,
            # but we index explicitly to avoid relying on insertion order.
            arr = np.array([scores[i] for i in range(self.n_ent)], dtype=np.float32)
        else:
            arr = np.array(scores, dtype=np.float32)
        self.ppr_vector_cache[ent] = arr
        if len(self.ppr_vector_cache) > self.ppr_cache_size:
            self.ppr_vector_cache.popitem(last=False)
        return arr
        
    def generatePPRScoresForOneEntity(self, h, method='nx'):
        if method == 'nx':
            '''
            nx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
            '''
            scores = nx.pagerank(self.homoTrainGraph, personalization={h: 1})
        elif method == 'matrix':
            alpha, iteration = 0.85, 100
            scores = torch.zeros(1, self.n_ent).cuda()
            s = torch.zeros(1, self.n_ent).cuda()
            s[0, h] = 1
            for i in range(iteration):
                scores = alpha * s + (1 - alpha) * torch.matmul(scores, self.PPR_W)            
            scores = scores.cpu().reshape(-1).numpy()
        return scores
    
    def triplesToNxGraph(self, edges):
        ''' edges is the list of [(h,t)] '''
        graph = nx.Graph()
        nodes = list(range(self.n_ent))
        graph.add_nodes_from(nodes)        
        graph.add_edges_from(edges)
        return graph

    def _minmax_normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if x_max - x_min < 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return (x - x_min) / (x_max - x_min + 1e-12)

    def _row_normalize(self, mat):
        mat = np.asarray(mat, dtype=np.float32)
        if mat.size == 0:
            return mat
        denom = np.sum(mat, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-12, None)
        return mat / denom

    def _extract_induced_edges(self, node_ids):
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if node_ids.size == 0:
            return torch.zeros((0, 3), dtype=torch.long)

        selected_edges = self.sparseTrainMatrix[node_ids, :]
        _, edge_ids = selected_edges.nonzero()
        edges = self.edge_index[edge_ids]
        node_tensor = torch.as_tensor(node_ids, dtype=torch.long)
        mask = torch.isin(edges[:, 2], node_tensor)
        return edges[mask, :]

    def _build_lqcd_scores(self, ent, rel, ppr_scores):
        if rel == -1 or self.topk >= self.n_ent:
            return ppr_scores

        coarse_k = int(np.ceil(max(1.0, self.lqcd_coarse_ratio) * self.topk))
        coarse_k = min(self.n_ent, max(self.topk, coarse_k))
        coarse_rank = np.argsort(ppr_scores)[::-1][:coarse_k].tolist()
        coarse_nodes = np.array(list(dict.fromkeys([int(ent)] + coarse_rank)), dtype=np.int64)

        coarse_edges = self._extract_induced_edges(coarse_nodes)
        if coarse_edges.shape[0] == 0:
            if not self.lqcd_logged:
                print(f'==> LQCD[{self.split}]: fallback to raw PPR (query_rel={rel}, reason=no_coarse_edges)')
                self.lqcd_logged = True
            return ppr_scores

        coarse_edges_np = coarse_edges.numpy()
        local_edges = coarse_edges_np[coarse_edges_np[:, 1] != self.idd_rel]
        if local_edges.shape[0] == 0:
            if not self.lqcd_logged:
                print(f'==> LQCD[{self.split}]: fallback to raw PPR (query_rel={rel}, reason=no_local_rel_edges)')
                self.lqcd_logged = True
            return ppr_scores

        node_pos = {int(node): idx for idx, node in enumerate(coarse_nodes.tolist())}
        ppr_local = ppr_scores[coarse_nodes]
        ppr_local_norm = self._minmax_normalize(ppr_local)
        deg_local = np.zeros(len(coarse_nodes), dtype=np.int32)
        rel_sets = defaultdict(set)
        in_edges = defaultdict(list)
        out_edges = defaultdict(list)
        rel_vocab = {int(rel)}

        for h, r, t in local_edges:
            h, r, t = int(h), int(r), int(t)
            rel_vocab.add(r)
            h_idx = node_pos[h]
            t_idx = node_pos[t]
            deg_local[h_idx] += 1
            deg_local[t_idx] += 1
            rel_sets[h].add(r)
            rel_sets[t].add(r)
            out_edges[h].append((h, r, t))
            in_edges[t].append((h, r, t))

        rel_list = sorted(rel_vocab)
        rel_pos = {r_id: idx for idx, r_id in enumerate(rel_list)}
        n_local_rel = len(rel_list)
        cooccur = np.zeros((n_local_rel, n_local_rel), dtype=np.float32)
        transition = np.zeros((n_local_rel, n_local_rel), dtype=np.float32)

        for node, node_rels in rel_sets.items():
            if len(node_rels) < 2:
                continue
            node_idx = node_pos[node]
            hub_weight = 1.0 / (np.log(2.0 + float(deg_local[node_idx])) ** self.lqcd_hub_beta)
            rels = sorted(node_rels)
            pair_norm = hub_weight / max(1.0, (len(rels) * (len(rels) - 1)) / 2.0)
            for i in range(len(rels)):
                for j in range(i + 1, len(rels)):
                    ri = rel_pos[rels[i]]
                    rj = rel_pos[rels[j]]
                    cooccur[ri, rj] += pair_norm
                    cooccur[rj, ri] += pair_norm

        for mid, incoming in in_edges.items():
            outgoing = out_edges.get(mid, [])
            if len(incoming) == 0 or len(outgoing) == 0:
                continue
            mid_idx = node_pos[mid]
            hub_weight = 1.0 / (np.log(2.0 + float(deg_local[mid_idx])) ** self.lqcd_hub_beta)
            trans_weight = ppr_local_norm[mid_idx] * hub_weight
            trans_norm = trans_weight / max(1, len(incoming) * len(outgoing))
            for _, ri, _ in incoming:
                for _, rj, _ in outgoing:
                    transition[rel_pos[int(ri)], rel_pos[int(rj)]] += trans_norm

        relation_graph = self._row_normalize(cooccur) + self._row_normalize(transition)
        relation_graph += self.lqcd_self_loop * np.eye(n_local_rel, dtype=np.float32)
        relation_graph = self._row_normalize(relation_graph)

        query_idx = rel_pos[int(rel)]
        query_onehot = np.zeros(n_local_rel, dtype=np.float32)
        query_onehot[query_idx] = 1.0
        rel_scores = self.lqcd_rel_gamma * query_onehot + (1.0 - self.lqcd_rel_gamma) * (relation_graph.T @ query_onehot)

        node_supports = defaultdict(list)
        for h, r, t in local_edges:
            h, r, t = int(h), int(r), int(t)
            h_idx = node_pos[h]
            t_idx = node_pos[t]
            edge_score = float(rel_scores[rel_pos[r]] * 0.5 * (ppr_local_norm[h_idx] + ppr_local_norm[t_idx]))
            node_supports[h].append(edge_score)
            node_supports[t].append(edge_score)

        node_rel_scores = np.zeros(len(coarse_nodes), dtype=np.float32)
        for node, supports in node_supports.items():
            top_supports = np.sort(np.asarray(supports, dtype=np.float32))[::-1]
            keep_k = min(self.lqcd_topl, top_supports.size)
            node_rel_scores[node_pos[node]] = float(np.mean(top_supports[:keep_k]))

        node_rel_scores = self._minmax_normalize(node_rel_scores)
        fuse_lambda = float(np.clip(self.lqcd_fuse_lambda, 0.0, 1.0))
        fused_local = fuse_lambda * ppr_local_norm + (1.0 - fuse_lambda) * node_rel_scores

        if not self.lqcd_logged:
            rel_score_max = float(np.max(rel_scores)) if rel_scores.size > 0 else 0.0
            rel_score_mean = float(np.mean(rel_scores)) if rel_scores.size > 0 else 0.0
            print(
                f'==> LQCD[{self.split}]: query_rel={rel}, coarse_nodes={len(coarse_nodes)}, '
                f'local_edges={int(local_edges.shape[0])}, local_rels={n_local_rel}, '
                f'topl={self.lqcd_topl}, rel_score_max={rel_score_max:.4f}, rel_score_mean={rel_score_mean:.4f}'
            )
            self.lqcd_logged = True

        fused_scores = np.full(self.n_ent, -np.inf, dtype=np.float32)
        fused_scores[coarse_nodes] = fused_local
        return fused_scores

    # # ========== Module 1: Relation-Path Conditioned Sampling ==========
    # # Mines frequent 2-hop relation path patterns for each relation
    # # and uses path reachability to condition subgraph extraction.
    # #   --rel_path_topk    : top-K patterns per relation
    # #   --path_lambda      : fusion weight for path-based prior
    # #   --fusion_mode      : add / multiply fusion with PPR
    # def mineRelationPatterns(self, edge_index):
    #     """
    #     Mine frequent 2-hop relation path patterns for each relation.
    #
    #     For relation r_q, examines training triples (h, r_q, t) and finds
    #     2-hop paths h --(r1)--> m --(r2)--> t in the graph. Frequent
    #     patterns (r1, r2) indicate common reasoning paths for r_q.
    #
    #     Example: for 'grandfather_of', frequent pattern might be
    #              (father_of, father_of) since grandfather = father + father.
    #
    #     Also builds per-entity adjacency lists for online path scoring.
    #
    #     Args:
    #         edge_index: array of triples [(h, r, t), ...]
    #     """
    #     print('==> mining relation path patterns...')
    #
    #     n_total_rel = 2 * self.n_rel + 1
    #     topk = getattr(self.args, 'rel_path_topk', 10)
    #
    #     # Build adjacency lists: adj_by_rel[entity][rel] = set(neighbors)
    #     self.adj_by_rel = {}
    #     # Build incoming adjacency: tail_in[entity][neighbor] = set(rels)
    #     tail_in = {}
    #
    #     for triple in edge_index:
    #         h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
    #         if h not in self.adj_by_rel:
    #             self.adj_by_rel[h] = {}
    #         if r not in self.adj_by_rel[h]:
    #             self.adj_by_rel[h][r] = set()
    #         self.adj_by_rel[h][r].add(t)
    #         if t not in tail_in:
    #             tail_in[t] = {}
    #         if h not in tail_in[t]:
    #             tail_in[t][h] = set()
    #         tail_in[t][h].add(r)
    #
    #     # Group triples by relation
    #     triples_by_rel = defaultdict(list)
    #     for triple in edge_index:
    #         h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
    #         triples_by_rel[r].append((h, t))
    #
    #     # Mine patterns for each relation
    #     self.relation_patterns = {}
    #
    #     for r_q in tqdm(range(n_total_rel), ncols=50, leave=False, desc='mining'):
    #         if r_q not in triples_by_rel:
    #             self.relation_patterns[r_q] = []
    #             continue
    #
    #         ht_pairs = triples_by_rel[r_q]
    #
    #         # Subsample if too many triples per relation
    #         if len(ht_pairs) > 1000:
    #             indices = np.random.choice(len(ht_pairs), 1000, replace=False)
    #             ht_pairs = [ht_pairs[i] for i in indices]
    #
    #         pattern_counts = defaultdict(int)
    #
    #         for h, t in ht_pairs:
    #             # Find intermediate nodes m: h --(r1)--> m --(r2)--> t
    #             for r1, m_set in self.adj_by_rel[h].items():
    #                 for m in m_set:
    #                     if m == h or m == t:
    #                         continue
    #                     # Check if m connects to t
    #                     if m in tail_in[t]:
    #                         for r2 in tail_in[t][m]:
    #                             pattern_counts[(r1, r2)] += 1
    #
    #         # Sort by frequency and keep top-K
    #         sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])[:topk]
    #
    #         # Normalize weights
    #         if sorted_patterns:
    #             total = sum(c for _, c in sorted_patterns)
    #             self.relation_patterns[r_q] = [((r1, r2), c / total) for (r1, r2), c in sorted_patterns]
    #         else:
    #             self.relation_patterns[r_q] = []
    #
    #     print(f'==> relation patterns mined.')
    #
    # def updateRelationPatterns(self, edge_index):
    #     """Update adjacency lists after shuffle_train. Patterns stay the same."""
    #     self.adj_by_rel = {}
    #     for triple in edge_index:
    #         h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
    #         if h not in self.adj_by_rel:
    #             self.adj_by_rel[h] = {}
    #         if r not in self.adj_by_rel[h]:
    #             self.adj_by_rel[h][r] = set()
    #         self.adj_by_rel[h][r].add(t)
    # # ========== End Module 1 ==========

    def sampleSubgraph(self, ent: int, rel: int = -1):
        # sample subgraph to get the edges
        ppr_scores = self.getPPRarray(ent)
        fused_scores = ppr_scores

        # # ========== Module 1: Fuse path-based prior with PPR scores ==========
        # # When enabled, the node selection combines structural proximity (PPR)
        # # with path reachability scores based on mined relation patterns.
        # # PathScore(v|h, r_q) = sum of pattern weights for patterns that
        # # can reach v from h through 2-hop paths.
        # if hasattr(self.args, 'use_rel_prior') and self.args.use_rel_prior and rel != -1:
        #     path_scores = np.zeros(self.n_ent)
        #     patterns = self.relation_patterns.get(rel, [])
        #
        #     for (r1, r2), weight in patterns:
        #         # Find entities reachable from ent via pattern (r1, r2)
        #         m_set = self.adj_by_rel.get(ent, {}).get(r1, set())
        #         for m in m_set:
        #             v_set = self.adj_by_rel.get(m, {}).get(r2, set())
        #             for v in v_set:
        #                 path_scores[v] += weight
        #
        #     # Normalize to [0, 1]
        #     max_ps = path_scores.max()
        #     if max_ps > 0:
        #         path_scores /= max_ps
        #
        #     lam = self.args.path_lambda
        #     fusion_mode = getattr(self.args, 'fusion_mode', 'add')
        #     if fusion_mode == 'add':
        #         # fused(v) = PPR(v|h) + λ · PathScore(v|h, r_q)
        #         fused_scores = ppr_scores + lam * path_scores
        #     elif fusion_mode == 'multiply':
        #         # fused(v) = PPR(v|h) · PathScore(v|h, r_q)^λ
        #         fused_scores = ppr_scores * np.power(path_scores + 1e-10, lam)
        #     else:
        #         fused_scores = ppr_scores + lam * path_scores
        # else:
        #     fused_scores = ppr_scores
        # # ========== End Module 1 ==========
        # fused_scores = ppr_scores

        if self.use_lqcd and rel != -1:
            fused_scores = self._build_lqcd_scores(ent, rel, ppr_scores)

        # topk sampling
        if self.topk < self.n_ent:
            topk_nodes = sorted(list(set([ent] + np.argsort(fused_scores)[::-1][:self.topk].tolist())))
        else:
            # no sampling
            topk_nodes = list(range(self.n_ent))

        # get candididate edges
        selectd_edges = self.sparseTrainMatrix[topk_nodes, :]	
        _, tmp_edge_index = selectd_edges.nonzero()
        
        # (h,r,t)
        edges = self.edge_index[tmp_edge_index]
        topk_nodes = torch.LongTensor(topk_nodes)
        
        # edge sampling
        mask = torch.isin(edges[:,2], topk_nodes)
        
        # [n_edges, 3]
        sampled_edges = edges[mask, :]
        
        # edge sampling (topm edges for each subgraph)
        edge_num = int(sampled_edges.shape[0])
        # NOTE: if self.topm== 0, then skip edge sampling 
        if self.topm > 0 and edge_num > self.topm:
            # ppr weight
            heads, tails = sampled_edges[:,0], sampled_edges[:,2]
            edge_weights = fused_scores[heads] + fused_scores[tails]
            edge_weights = torch.Tensor(edge_weights)
            index = torch.topk(edge_weights, self.topm).indices
            sampled_edges = sampled_edges[index]
        
        # get node indexing map
        node_index = torch.zeros(self.n_ent).long()
        node_index[topk_nodes] = torch.arange(len(topk_nodes))
              
        # connect head to all tails 
        if self.args.add_manual_edges:
            add_edges_head2tails = torch.zeros((len(topk_nodes), 3)).long()
            add_edges_head2tails[:, 0] = ent
            add_edges_head2tails[:, 1] = 2*self.n_rel + 1
            add_edges_head2tails[:, 2] = topk_nodes
            add_edges_tails2head = torch.zeros((len(topk_nodes), 3)).long()
            add_edges_tails2head[:, 0] = topk_nodes
            add_edges_tails2head[:, 1] = 2*self.n_rel + 2
            add_edges_tails2head[:, 2] = ent
            sampled_edges = torch.cat([sampled_edges, add_edges_head2tails, add_edges_tails2head], dim=0)
        
        return topk_nodes, node_index, sampled_edges

    def getOneSubgraph(self, head: int, rel: int = -1):
        topk_nodes, node_index, sampled_edges = self.sampleSubgraph(head, rel)
        return [head, topk_nodes, node_index, sampled_edges]
        
    def getBatchSubgraph(self, subgraph_list: list):  
        batchsize = len(subgraph_list)
        ent_delta_values = [0]
        batch_sampled_edges = []
        batch_idxs, abs_idxs = [], []
        query_sub_idxs = []
        edge_batch_idxs = []

        for batch_idx in range(batchsize):       
            sub, topk_nodes, node_index, sampled_edges = subgraph_list[batch_idx]
            num_nodes = len(topk_nodes)
            ent_delta = sum(ent_delta_values)

            sampled_edges[:,0] = node_index[sampled_edges[:,0]] + ent_delta
            sampled_edges[:,2] = node_index[sampled_edges[:,2]] + ent_delta
            batch_sampled_edges.append(sampled_edges)
            edge_batch_idxs += [batch_idx] * int(sampled_edges.shape[0])

            ent_delta_values.append(num_nodes)
            batch_idxs += [batch_idx] * num_nodes
            abs_idxs += topk_nodes.tolist()
            query_sub_idxs.append(int(node_index[sub]) + ent_delta)
        
        # [n_batch_ent]
        batch_idxs = torch.LongTensor(batch_idxs)
        # [n_batch_ent]
        abs_idxs = torch.LongTensor(abs_idxs)
        # [n_batch_edges, 3]
        batch_sampled_edges = torch.cat(batch_sampled_edges, dim=0)
        # [n_batch_edges]
        edge_batch_idxs = torch.LongTensor(edge_batch_idxs)
        # [n_batch]
        query_sub_idxs = torch.LongTensor(query_sub_idxs)
        
        return batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges
