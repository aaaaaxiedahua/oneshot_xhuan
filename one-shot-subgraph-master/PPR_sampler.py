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
        self.n_total_rel = 2 * n_rel
        self.topk = topk
        self.topm = topm
        self.edge_index = edge_index
        self.split = split
        self.data_folder = data_path
        self.use_rel_ppr_sampler = bool(getattr(args, 'use_rel_ppr_sampler', False))
        self.rel_fuse_lambda = float(getattr(args, 'rel_fuse_lambda', 0.5))
        self.n_final_samp_ent = int(getattr(args, 'n_final_samp_ent', self.topk))
        if self.use_rel_ppr_sampler and self.n_final_samp_ent >= self.topk:
            raise ValueError('final_topk must sample fewer nodes than topk when --use_rel_ppr_sampler is enabled.')
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
        self.raw_edge_index = np.asarray(edge_index, dtype=np.int64)

        # change data type
        self.edge_index = torch.LongTensor(self.edge_index)
        if self.use_rel_ppr_sampler:
            self.rel_ppr_matrix = self.load_or_build_relation_ppr(self.raw_edge_index)
            print(f'==> relation PPR sampler enabled (final_topk={self.n_final_samp_ent}, lambda={self.rel_fuse_lambda})')
        else:
            self.rel_ppr_matrix = None

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
        
        print('==> finish sampler initilization.')

    def updateEdges(self, edge_index):
        # co-operate with shuffle_train
        heads, edges = [h for (h,r,t) in edge_index], list(range(len(edge_index)))
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(edge_index)))
        self.edge_index = torch.LongTensor(edge_index)
        self.raw_edge_index = np.asarray(edge_index, dtype=np.int64)
    
    def getPPRscores(self, ent):
        ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(ent)}.pkl')
        scores = pkl.load(open(ent_ppr_savePath, 'rb'))
        return scores

    def getPPRscoreArray(self, ent):
        scores = self.getPPRscores(ent)
        if isinstance(scores, dict):
            return np.array([scores[i] for i in range(self.n_ent)], dtype=np.float32)
        return np.asarray(scores, dtype=np.float32)
        
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

    def _inverse_relation(self, rel):
        if rel < self.n_rel:
            return rel + self.n_rel
        if rel < self.n_total_rel:
            return rel - self.n_rel
        return rel

    def _relation_cache_path(self):
        cache_dir = os.path.join(self.data_folder, 'relation_ppr_scores')
        checkPath(cache_dir)
        seed = str(getattr(self.args, 'seed', 'na'))
        fact_ratio = str(getattr(self.args, 'fact_ratio', 'na')).replace('.', 'p')
        remove_1hop = int(bool(getattr(self.args, 'remove_1hop_edges', False)))
        return os.path.join(
            cache_dir,
            f'{self.split}_seed_{seed}_fact_{fact_ratio}_rm1hop_{remove_1hop}_rel_ppr.npy'
        )

    def _row_normalize(self, matrix):
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums

    def build_relation_transition(self, edge_index):
        real_triples = np.asarray(edge_index, dtype=np.int64)
        real_triples = real_triples[real_triples[:, 1] < self.n_total_rel]

        support = np.zeros((self.n_total_rel, self.n_total_rel), dtype=np.float32)
        transition = np.zeros((self.n_total_rel, self.n_total_rel), dtype=np.float32)

        outgoing = defaultdict(list)
        incoming = defaultdict(list)
        pair_to_query_rels = defaultdict(set)

        for h, r, t in real_triples:
            h = int(h); r = int(r); t = int(t)
            outgoing[h].append((r, t))
            incoming[t].append((h, r))
            pair_to_query_rels[(h, t)].add(r)

        shared_entities = set(outgoing.keys()) & set(incoming.keys())
        for middle in tqdm(shared_entities, ncols=50, leave=False, desc=f'rel-graph-{self.split}'):
            in_edges = incoming[middle]
            out_edges = outgoing[middle]
            for h, r_in in in_edges:
                inv_r_in = self._inverse_relation(r_in)
                for r_out, t in out_edges:
                    if t == h and r_out == inv_r_in:
                        continue
                    transition[r_in, r_out] += 1.0
                    query_rels = pair_to_query_rels.get((h, t))
                    if not query_rels:
                        continue
                    for q_rel in query_rels:
                        support[q_rel, r_in] += 1.0
                        support[q_rel, r_out] += 1.0

        relation_graph = support + transition + np.eye(self.n_total_rel, dtype=np.float32)
        return self._row_normalize(relation_graph)

    def build_relation_ppr(self, edge_index):
        alpha = 0.85
        iteration = 100
        transition = self.build_relation_transition(edge_index)
        restart = np.eye(self.n_total_rel, dtype=np.float32)
        scores = restart.copy()
        for _ in range(iteration):
            scores = alpha * restart + (1 - alpha) * np.matmul(scores, transition)
        return scores.astype(np.float32)

    def load_or_build_relation_ppr(self, edge_index):
        cache_path = self._relation_cache_path()
        if os.path.exists(cache_path):
            scores = np.load(cache_path)
            if scores.shape == (self.n_total_rel, self.n_total_rel):
                return scores.astype(np.float32)

        print('==> building relation PPR cache...')
        scores = self.build_relation_ppr(edge_index)
        np.save(cache_path, scores)
        return scores

    def _select_top_nodes(self, ppr_scores, ent, sample_count, cand=None):
        if sample_count >= self.n_ent:
            return list(range(self.n_ent))

        ranking_scores = np.array(ppr_scores, copy=True)
        if cand is not None:
            ranking_scores[cand] = 1e8
        top_nodes = np.argsort(ranking_scores)[::-1][:sample_count].tolist()
        return sorted(list(set([ent] + top_nodes)))

    def _extract_induced_edges(self, top_nodes):
        selected_edges = self.sparseTrainMatrix[top_nodes, :]
        _, tmp_edge_index = selected_edges.nonzero()
        edges = self.edge_index[tmp_edge_index]
        top_nodes_tensor = torch.LongTensor(top_nodes)
        mask = torch.isin(edges[:, 2], top_nodes_tensor)
        return top_nodes_tensor, edges[mask, :]

    def _apply_topm_sampling(self, sampled_edges, ppr_scores):
        edge_num = int(sampled_edges.shape[0])
        if self.topm > 0 and edge_num > self.topm:
            heads, tails = sampled_edges[:, 0], sampled_edges[:, 2]
            edge_weights = ppr_scores[heads.cpu().numpy()] + ppr_scores[tails.cpu().numpy()]
            edge_weights = torch.as_tensor(edge_weights, dtype=torch.float32)
            index = torch.topk(edge_weights, self.topm).indices
            sampled_edges = sampled_edges[index]
        return sampled_edges

    def _relation_rerank_nodes(self, ent, rel, coarse_nodes, coarse_edges, ppr_scores):
        if rel is None or rel < 0 or rel >= self.n_total_rel:
            return coarse_nodes
        if len(coarse_nodes) <= self.n_final_samp_ent:
            return coarse_nodes

        coarse_node_np = coarse_nodes.cpu().numpy()
        base_scores = np.zeros(self.n_ent, dtype=np.float32)
        coarse_ppr = ppr_scores[coarse_node_np]
        coarse_ppr_max = float(np.max(coarse_ppr)) if len(coarse_ppr) > 0 else 0.0
        if coarse_ppr_max <= 0:
            return coarse_nodes
        base_scores[coarse_node_np] = coarse_ppr / coarse_ppr_max

        rel_scores = self.rel_ppr_matrix[int(rel)].astype(np.float32)
        rel_max = float(np.max(rel_scores))
        if rel_max <= 0:
            return coarse_nodes
        rel_scores = rel_scores / rel_max

        support_scores = np.zeros(self.n_ent, dtype=np.float32)
        for head, edge_rel, tail in coarse_edges.tolist():
            if edge_rel >= self.n_total_rel:
                continue
            rel_score = rel_scores[edge_rel]
            support_scores[tail] = max(support_scores[tail], base_scores[head] * rel_score)
            support_scores[head] = max(support_scores[head], base_scores[tail] * rel_score)

        final_scores = base_scores[coarse_node_np] * (1.0 + self.rel_fuse_lambda * support_scores[coarse_node_np])
        head_idx = np.where(coarse_node_np == ent)[0]
        if len(head_idx) > 0:
            final_scores[head_idx[0]] = float(np.max(final_scores)) + 1.0

        final_index = np.argsort(final_scores)[::-1][:self.n_final_samp_ent]
        final_nodes = np.sort(coarse_node_np[final_index])
        return torch.LongTensor(final_nodes)
    
    def sampleSubgraph(self, ent: int, rel: int = None, cand=None):    
        # sample subgraph to get the edges
        ppr_scores = self.getPPRscoreArray(ent)
        coarse_nodes = self._select_top_nodes(ppr_scores, ent, self.topk, cand=cand)
        coarse_nodes, coarse_edges = self._extract_induced_edges(coarse_nodes)

        if self.use_rel_ppr_sampler:
            topk_nodes = self._relation_rerank_nodes(ent, rel, coarse_nodes, coarse_edges, ppr_scores)
            mask = torch.isin(coarse_edges[:, 0], topk_nodes) & torch.isin(coarse_edges[:, 2], topk_nodes)
            sampled_edges = coarse_edges[mask, :]
        else:
            topk_nodes = coarse_nodes
            sampled_edges = coarse_edges

        sampled_edges = self._apply_topm_sampling(sampled_edges, ppr_scores)
        
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

    def getOneSubgraph(self, head: int, rel: int = None, cand=None):
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
