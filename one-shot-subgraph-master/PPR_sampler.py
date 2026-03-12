import networkx as nx
import pickle as pkl
import time
import copy
import numpy as np
import torch
import os
import logging
import copy
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
        self.ppr_cache_size = max(16, int(getattr(args, 'bippr_cache_size', 256)))
        checkPath(self.ppr_savePath)
        print('==> checking ppr scores for each entity...')
        
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

        # ========== R-BiPPR: Bidirectional PPR collision ==========
        self.use_bippr = hasattr(args, 'use_bippr') and args.use_bippr
        self.retriever_type = getattr(args, 'retriever_type', 'distmult').lower()
        self.kge_topk = max(1, int(getattr(args, 'kge_topk', 100)))
        self.collision_lambda = float(getattr(args, 'collision_lambda', 1.0))
        self.train_include_gt_prob = float(getattr(args, 'train_include_gt_prob', 1.0))
        self.train_include_gt_prob = max(0.0, min(1.0, self.train_include_gt_prob))
        self.ret_ent_emb = None
        self.ret_rel_emb = None
        if self.use_bippr:
            self._initBipprRetriever()
        # ========== End R-BiPPR ==========

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

    # ========== R-BiPPR helpers ==========
    def _extractRetrieverWeights(self, state):
        if state is None:
            return None, None

        if isinstance(state, dict):
            direct_ent_keys = ['entity_emb', 'ent_emb']
            direct_rel_keys = ['relation_emb', 'rel_emb']
            for k in direct_ent_keys:
                if k in state and torch.is_tensor(state[k]):
                    ent = state[k]
                    break
            else:
                ent = None
            for k in direct_rel_keys:
                if k in state and torch.is_tensor(state[k]):
                    rel = state[k]
                    break
            else:
                rel = None
            if ent is not None and rel is not None:
                return ent, rel

            if 'model_state_dict' in state and isinstance(state['model_state_dict'], dict):
                msd = state['model_state_dict']
                ent_keys = ['entity_emb.weight', 'ent_emb.weight', 'ent_embedding.weight']
                rel_keys = ['relation_emb.weight', 'rel_emb.weight', 'rel_embedding.weight']
                ent = None
                rel = None
                for k in ent_keys:
                    if k in msd and torch.is_tensor(msd[k]):
                        ent = msd[k]
                        break
                for k in rel_keys:
                    if k in msd and torch.is_tensor(msd[k]):
                        rel = msd[k]
                        break
                if ent is not None and rel is not None:
                    return ent, rel
        return None, None

    def _initBipprRetriever(self):
        ckpt_path = getattr(self.args, 'retriever_ckpt', '')
        if ckpt_path == '' or not os.path.exists(ckpt_path):
            raise FileNotFoundError('R-BiPPR requires --retriever_ckpt with valid checkpoint path.')
        print(f'==> loading retriever checkpoint from: {ckpt_path}')
        state = torch.load(ckpt_path, map_location='cpu')
        ent_emb, rel_emb = self._extractRetrieverWeights(state)
        if ent_emb is None or rel_emb is None:
            raise ValueError('Cannot find entity/relation embeddings in retriever checkpoint.')
        self.ret_ent_emb = ent_emb.float().cpu()
        self.ret_rel_emb = rel_emb.float().cpu()
        if self.ret_ent_emb.shape[0] != self.n_ent:
            raise ValueError(f'Retriever entity size mismatch: {self.ret_ent_emb.shape[0]} vs {self.n_ent}')
        print(f'==> retriever loaded. type={self.retriever_type} ent={tuple(self.ret_ent_emb.shape)} rel={tuple(self.ret_rel_emb.shape)}')

    def _normalizeRelIdx(self, rel):
        rel = int(rel)
        n_ret_rel = int(self.ret_rel_emb.shape[0])
        if 0 <= rel < n_ret_rel:
            return rel
        if n_ret_rel == self.n_rel:
            return rel % self.n_rel
        if n_ret_rel == 2 * self.n_rel:
            return rel % (2 * self.n_rel)
        return rel % n_ret_rel

    def _scoreAllEntities(self, ent, rel):
        rel_idx = self._normalizeRelIdx(rel)
        h = self.ret_ent_emb[int(ent)]  # [d]
        r = self.ret_rel_emb[rel_idx]   # [d]
        if self.retriever_type == 'transe':
            scores = -torch.norm((h + r).unsqueeze(0) - self.ret_ent_emb, p=1, dim=-1)
        else:  # default distmult
            scores = torch.sum(self.ret_ent_emb * (h * r), dim=-1)
        return scores

    def _getRetrieverCandidates(self, ent, rel, gt_cand=None):
        scores = self._scoreAllEntities(ent, rel)
        k = min(self.kge_topk, int(scores.shape[0]))
        top_vals, top_idx = torch.topk(scores, k, largest=True, sorted=True)
        if gt_cand is not None:
            gt_cand = int(gt_cand)
            if gt_cand not in top_idx.tolist():
                gt_score = scores[gt_cand]
                top_idx[-1] = gt_cand
                top_vals[-1] = gt_score
        weights = torch.softmax(top_vals, dim=0).cpu().numpy()
        return top_idx.cpu().numpy().astype(np.int64), weights
    # ========== End R-BiPPR helpers ==========
    
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

    def sampleSubgraph(self, ent: int, rel: int = -1, cand=None):
        # sample subgraph to get the edges
        ppr_scores = self.getPPRarray(ent)
        fused_scores = ppr_scores

        # ========== R-BiPPR: forward + reverse ppr collision ==========
        force_train_cand = None
        if cand is not None and self.split == 'train' and self.use_bippr:
            if np.random.rand() < self.train_include_gt_prob:
                force_train_cand = int(cand)

        if self.use_bippr and rel != -1:
            cand_ids, cand_weights = self._getRetrieverCandidates(ent, rel, gt_cand=force_train_cand)
            rev_scores = np.zeros(self.n_ent, dtype=np.float32)
            for c, w in zip(cand_ids, cand_weights):
                rev_scores += float(w) * self.getPPRarray(int(c))
            fused_scores = ppr_scores * (1.0 + self.collision_lambda * rev_scores)
        # ========== End R-BiPPR ==========

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

        # gurantee the candidates are sampled
        if force_train_cand is not None and self.topk < self.n_ent:
            tmp_scores = copy.deepcopy(fused_scores)
            tmp_scores[force_train_cand] = 1e8
            topk_nodes = sorted(list(set([ent] + np.argsort(tmp_scores)[::-1][:self.topk].tolist())))
        else:
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

    def getOneSubgraph(self, head: int, rel: int = -1, cand=None):
        topk_nodes, node_index, sampled_edges = self.sampleSubgraph(head, rel, cand)
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
