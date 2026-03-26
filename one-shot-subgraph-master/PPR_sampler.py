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

RBPPR_META_FIELDS = (
    'enabled',
    'score_delta_mean',
    'changed_topk_ratio',
    'changed_front_ratio',
    'lead_changed',
    'rel_gain_topk',
)

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
        self.use_rbppr = bool(getattr(args, 'use_rbppr', False))
        self.rbppr_lambda = float(np.clip(getattr(args, 'rbppr_lambda', 0.1), 0.0, 1.0))
        self.rbppr_savePath = os.path.join(self.data_folder, 'rbppr_scores/')
        self.rbppr_vector_cache = OrderedDict()
        self.rbppr_cache_size = max(32, 2 * self.n_rel)
        self.rbppr_front_k = min(256, max(1, self.topk))
        self.idd_rel = 2 * self.n_rel
        checkPath(self.ppr_savePath)
        if self.use_rbppr:
            checkPath(self.rbppr_savePath)
        print('==> checking ppr scores for each entity...')
        if self.use_rbppr:
            print(
                f'==> RBPPR[{self.split}]: enabled '
                f'(lambda={self.rbppr_lambda}, bias=tail_freq)'
            )
        
        for h in tqdm(range(self.n_ent), ncols=50, leave=False):
            ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
            if os.path.exists(ent_ppr_savePath):
                pass
            else:
                # with default setting to generate ppr scores
                h_ppr_scores = self.generatePPRScoresForOneEntity(h)
                pkl.dump(h_ppr_scores, open(ent_ppr_savePath, 'wb'))
        if self.use_rbppr:
            self._build_relation_bias_vectors(edge_index)
            print('==> checking relation-aware ppr scores for each relation...')
            for rel_id in tqdm(range(2 * self.n_rel), ncols=50, leave=False):
                rel_ppr_savePath = os.path.join(self.rbppr_savePath, f'{int(rel_id)}.pkl')
                if os.path.exists(rel_ppr_savePath):
                    continue
                rel_ppr_scores = self.generatePPRScoresForPersonalization(self.relation_bias_dict[rel_id])
                pkl.dump(rel_ppr_scores, open(rel_ppr_savePath, 'wb'))
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

    def getRelationPPRscores(self, rel):
        rel_ppr_savePath = os.path.join(self.rbppr_savePath, f'{int(rel)}.pkl')
        scores = pkl.load(open(rel_ppr_savePath, 'rb'))
        return scores

    def getRelationPPRarray(self, rel):
        rel = int(rel)
        if rel in self.rbppr_vector_cache:
            arr = self.rbppr_vector_cache.pop(rel)
            self.rbppr_vector_cache[rel] = arr
            return arr

        scores = self.getRelationPPRscores(rel)
        if isinstance(scores, dict):
            arr = np.array([scores[i] for i in range(self.n_ent)], dtype=np.float32)
        else:
            arr = np.array(scores, dtype=np.float32)
        self.rbppr_vector_cache[rel] = arr
        if len(self.rbppr_vector_cache) > self.rbppr_cache_size:
            self.rbppr_vector_cache.popitem(last=False)
        return arr
        
    def generatePPRScoresForOneEntity(self, h, method='nx'):
        return self.generatePPRScoresForPersonalization({int(h): 1.0}, method=method)

    def generatePPRScoresForPersonalization(self, personalization, method='nx'):
        if method == 'nx':
            '''
            nx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
            '''
            scores = nx.pagerank(self.homoTrainGraph, personalization=personalization)
        elif method == 'matrix':
            alpha, iteration = 0.85, 100
            scores = torch.zeros(1, self.n_ent).cuda()
            s = torch.zeros(1, self.n_ent).cuda()
            if isinstance(personalization, dict):
                for node_id, weight in personalization.items():
                    s[0, int(node_id)] = float(weight)
            else:
                p = np.asarray(personalization, dtype=np.float32).reshape(-1)
                s[0, :] = torch.from_numpy(p).to(s.device)
            for i in range(iteration):
                scores = alpha * s + (1 - alpha) * torch.matmul(scores, self.PPR_W)            
            scores = scores.cpu().reshape(-1).numpy()
        return scores

    def _build_relation_bias_vectors(self, edge_index):
        rel_tail_counts = np.zeros((2 * self.n_rel, self.n_ent), dtype=np.float32)
        for h, r, t in edge_index:
            r = int(r)
            if r == self.idd_rel:
                continue
            rel_tail_counts[r, int(t)] += 1.0

        self.relation_bias_dict = {}
        for rel_id in range(2 * self.n_rel):
            counts = rel_tail_counts[rel_id]
            nz = np.flatnonzero(counts > 0)
            if nz.size == 0:
                self.relation_bias_dict[rel_id] = {0: 1.0}
                continue
            total = float(counts[nz].sum())
            self.relation_bias_dict[rel_id] = {
                int(node_id): float(counts[node_id] / total) for node_id in nz.tolist()
            }

    def _build_rbppr_scores(self, ent, rel, ent_ppr_scores):
        meta = self._empty_rbppr_meta()
        if (not self.use_rbppr) or rel == -1:
            return ent_ppr_scores, meta

        meta['enabled'] = 1.0
        rel_ppr_scores = self.getRelationPPRarray(rel)
        fused_scores = ((1.0 - self.rbppr_lambda) * ent_ppr_scores + self.rbppr_lambda * rel_ppr_scores).astype(np.float32)

        base_norm = self._minmax_normalize(ent_ppr_scores)
        rel_norm = self._minmax_normalize(rel_ppr_scores)
        fused_norm = self._minmax_normalize(fused_scores)

        raw_rank_nodes = [int(node) for node in np.argsort(ent_ppr_scores)[::-1].tolist() if int(node) != int(ent)]
        fused_rank_nodes = [int(node) for node in np.argsort(fused_scores)[::-1].tolist() if int(node) != int(ent)]

        raw_topk_nodes = sorted(list(set([int(ent)] + raw_rank_nodes[:self.topk])))
        fused_topk_nodes = sorted(list(set([int(ent)] + fused_rank_nodes[:self.topk])))
        raw_front_nodes = sorted(list(set([int(ent)] + raw_rank_nodes[:self.rbppr_front_k])))
        fused_front_nodes = sorted(list(set([int(ent)] + fused_rank_nodes[:self.rbppr_front_k])))
        raw_lead = raw_rank_nodes[0] if len(raw_rank_nodes) > 0 else int(ent)
        fused_lead = fused_rank_nodes[0] if len(fused_rank_nodes) > 0 else int(ent)

        score_nodes = sorted(list(set(raw_front_nodes).union(set(fused_front_nodes))))
        if len(score_nodes) > 0:
            meta['score_delta_mean'] = float(np.mean(np.abs(fused_norm[score_nodes] - base_norm[score_nodes])))

        raw_topk_set = set(raw_topk_nodes)
        fused_topk_set = set(fused_topk_nodes)
        raw_front_set = set(raw_front_nodes)
        fused_front_set = set(fused_front_nodes)
        meta['changed_topk_ratio'] = float(
            len(raw_topk_set.symmetric_difference(fused_topk_set)) / max(1, len(raw_topk_set))
        )
        meta['changed_front_ratio'] = float(
            len(raw_front_set.symmetric_difference(fused_front_set)) / max(1, len(raw_front_set))
        )
        meta['lead_changed'] = float(raw_lead != fused_lead)

        fused_top_nodes_wo_head = [node for node in fused_topk_nodes if int(node) != int(ent)]
        if len(fused_top_nodes_wo_head) > 0:
            idx = np.asarray(fused_top_nodes_wo_head, dtype=np.int64)
            meta['rel_gain_topk'] = float(np.mean(rel_norm[idx] - base_norm[idx]))

        return fused_scores, meta
    
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

    def _empty_rbppr_meta(self):
        return {
            'enabled': 0.0,
            'score_delta_mean': 0.0,
            'changed_topk_ratio': 0.0,
            'changed_front_ratio': 0.0,
            'lead_changed': 0.0,
            'rel_gain_topk': 0.0,
        }

    def _finalize_rbppr_meta(self, meta):
        return torch.tensor([meta[key] for key in RBPPR_META_FIELDS], dtype=torch.float32)

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
        base_ppr_scores = self.getPPRarray(ent)
        ppr_scores, rbppr_meta = self._build_rbppr_scores(ent, rel, base_ppr_scores)
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

        # topk sampling
        if self.topk < self.n_ent:
            topk_nodes = sorted(list(set([ent] + np.argsort(fused_scores)[::-1][:self.topk].tolist())))
        else:
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
        
        return (
            topk_nodes,
            node_index,
            sampled_edges,
            self._finalize_rbppr_meta(rbppr_meta),
        )

    def getOneSubgraph(self, head: int, rel: int = -1):
        topk_nodes, node_index, sampled_edges, rbppr_meta = self.sampleSubgraph(head, rel)
        return [head, topk_nodes, node_index, sampled_edges, rbppr_meta]
        
    def getBatchSubgraph(self, subgraph_list: list):  
        batchsize = len(subgraph_list)
        ent_delta_values = [0]
        batch_sampled_edges = []
        batch_idxs, abs_idxs = [], []
        query_sub_idxs = []
        edge_batch_idxs = []
        batch_rbppr_meta = []

        for batch_idx in range(batchsize):       
            sub, topk_nodes, node_index, sampled_edges, rbppr_meta = subgraph_list[batch_idx]
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
            batch_rbppr_meta.append(rbppr_meta)
        
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
        batch_rbppr_meta = torch.stack(batch_rbppr_meta, dim=0)
        
        return batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges, batch_rbppr_meta
