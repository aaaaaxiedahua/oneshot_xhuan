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
from collections import defaultdict

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
        
        # ========== Module 1: Build relation prior P(v|r) ==========
        # Makes subgraph extraction query-relation-aware by fusing learned
        # relation prior with PPR scores. To disable: set --use_rel_prior to False
        if hasattr(args, 'use_rel_prior') and args.use_rel_prior:
            self.mineRelationPatterns(edge_index)
        # ========== End Module 1 ==========

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
    
    # ========== Module 1: Relation-Path Conditioned Sampling ==========
    # Mines frequent 2-hop relation path patterns for each relation
    # and uses path reachability to condition subgraph extraction.
    #   --rel_path_topk    : top-K patterns per relation
    #   --path_lambda      : fusion weight for path-based prior
    #   --fusion_mode      : add / multiply fusion with PPR
    def mineRelationPatterns(self, edge_index):
        """
        Mine frequent 2-hop relation path patterns for each relation.

        For relation r_q, examines training triples (h, r_q, t) and finds
        2-hop paths h --(r1)--> m --(r2)--> t in the graph. Frequent
        patterns (r1, r2) indicate common reasoning paths for r_q.

        Example: for 'grandfather_of', frequent pattern might be
                 (father_of, father_of) since grandfather = father + father.

        Also builds per-entity adjacency lists for online path scoring.

        Args:
            edge_index: array of triples [(h, r, t), ...]
        """
        print('==> mining relation path patterns...')

        n_total_rel = 2 * self.n_rel + 1
        topk = getattr(self.args, 'rel_path_topk', 10)

        # Build adjacency lists: adj_by_rel[entity][rel] = set(neighbors)
        self.adj_by_rel = defaultdict(lambda: defaultdict(set))
        # Build incoming adjacency: tail_in[entity][neighbor] = set(rels)
        tail_in = defaultdict(lambda: defaultdict(set))

        for triple in edge_index:
            h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
            self.adj_by_rel[h][r].add(t)
            tail_in[t][h].add(r)

        # Group triples by relation
        triples_by_rel = defaultdict(list)
        for triple in edge_index:
            h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
            triples_by_rel[r].append((h, t))

        # Mine patterns for each relation
        self.relation_patterns = {}

        for r_q in tqdm(range(n_total_rel), ncols=50, leave=False, desc='mining'):
            if r_q not in triples_by_rel:
                self.relation_patterns[r_q] = []
                continue

            ht_pairs = triples_by_rel[r_q]

            # Subsample if too many triples per relation
            if len(ht_pairs) > 1000:
                indices = np.random.choice(len(ht_pairs), 1000, replace=False)
                ht_pairs = [ht_pairs[i] for i in indices]

            pattern_counts = defaultdict(int)

            for h, t in ht_pairs:
                # Find intermediate nodes m: h --(r1)--> m --(r2)--> t
                for r1, m_set in self.adj_by_rel[h].items():
                    for m in m_set:
                        if m == h or m == t:
                            continue
                        # Check if m connects to t
                        if m in tail_in[t]:
                            for r2 in tail_in[t][m]:
                                pattern_counts[(r1, r2)] += 1

            # Sort by frequency and keep top-K
            sorted_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])[:topk]

            # Normalize weights
            if sorted_patterns:
                total = sum(c for _, c in sorted_patterns)
                self.relation_patterns[r_q] = [((r1, r2), c / total) for (r1, r2), c in sorted_patterns]
            else:
                self.relation_patterns[r_q] = []

        print(f'==> relation patterns mined.')

    def updateRelationPatterns(self, edge_index):
        """Update adjacency lists after shuffle_train. Patterns stay the same."""
        self.adj_by_rel = defaultdict(lambda: defaultdict(set))
        for triple in edge_index:
            h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
            self.adj_by_rel[h][r].add(t)
    # ========== End Module 1 ==========

    def sampleSubgraph(self, ent: int, rel: int = -1, cand=None):
        # sample subgraph to get the edges
        ppr_scores = np.array(list(self.getPPRscores(ent).values()))

        # ========== Module 1: Fuse path-based prior with PPR scores ==========
        # When enabled, the node selection combines structural proximity (PPR)
        # with path reachability scores based on mined relation patterns.
        # PathScore(v|h, r_q) = sum of pattern weights for patterns that
        # can reach v from h through 2-hop paths.
        if hasattr(self.args, 'use_rel_prior') and self.args.use_rel_prior and rel != -1:
            path_scores = np.zeros(self.n_ent)
            patterns = self.relation_patterns.get(rel, [])

            for (r1, r2), weight in patterns:
                # Find entities reachable from ent via pattern (r1, r2)
                m_set = self.adj_by_rel.get(ent, {}).get(r1, set())
                for m in m_set:
                    v_set = self.adj_by_rel.get(m, {}).get(r2, set())
                    for v in v_set:
                        path_scores[v] += weight

            # Normalize to [0, 1]
            max_ps = path_scores.max()
            if max_ps > 0:
                path_scores /= max_ps

            lam = self.args.path_lambda
            fusion_mode = getattr(self.args, 'fusion_mode', 'add')
            if fusion_mode == 'add':
                # fused(v) = PPR(v|h) + λ · PathScore(v|h, r_q)
                fused_scores = ppr_scores + lam * path_scores
            elif fusion_mode == 'multiply':
                # fused(v) = PPR(v|h) · PathScore(v|h, r_q)^λ
                fused_scores = ppr_scores * np.power(path_scores + 1e-10, lam)
            else:
                fused_scores = ppr_scores + lam * path_scores
        else:
            fused_scores = ppr_scores
        # ========== End Module 1 ==========

        # gurantee the candidates are sampled
        if cand != None and self.topk < self.n_ent:
            tmp_scores = copy.deepcopy(fused_scores)
            tmp_scores[cand] = 1e8
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
            edge_weights = ppr_scores[heads] + ppr_scores[tails]
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