import networkx as nx
import pickle as pkl
import numpy as np
import torch
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix


def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


class pprSampler():
    def __init__(self, n_ent: int, n_rel: int, topk: int, topm: int, homoEdges: list, edge_index: list, data_path: str, split='train', args=None):
        """
        args:
            topk: number of sampled nodes for one head entity
            edge_index: list of triples [(h,r,t)]
            data_path: path to save the ppr/subgraphs files
        """
        print('==> initializing ppr sampler...')
        self.args = args
        self.n_ent = n_ent
        self.n_samp_ent = args.n_samp_ent
        self.n_rel = n_rel
        self.topk = topk
        self.topm = topm
        self.edge_index = edge_index
        self.split = split
        self.data_folder = data_path
        self.homoEdges = homoEdges
        self.homoTrainGraph = self.triplesToNxGraph(self.homoEdges)
        self.ppr_savePath = os.path.join(self.data_folder, 'ppr_scores/')
        checkPath(self.ppr_savePath)
        print('==> checking ppr scores for each entity...')

        for h in tqdm(range(self.n_ent), ncols=50, leave=False):
            ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
            if not os.path.exists(ent_ppr_savePath):
                h_ppr_scores = self.generatePPRScoresForOneEntity(h)
                pkl.dump(h_ppr_scores, open(ent_ppr_savePath, 'wb'))
        print('finished.')

        heads, edges = [h for (h, r, t) in edge_index], list(range(len(edge_index)))
        print(len(heads), len(edges), max(heads), self.n_ent)
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(edge_index)))
        self.raw_edge_index = np.asarray(edge_index, dtype=np.int64)
        self.edge_index = torch.LongTensor(self.edge_index)

        del self.homoEdges
        del self.homoTrainGraph
        print('==> finish sampler initilization.')

    def updateEdges(self, edge_index):
        heads, edges = [h for (h, r, t) in edge_index], list(range(len(edge_index)))
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
        graph = nx.Graph()
        nodes = list(range(self.n_ent))
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return graph

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

    def sampleSubgraph(self, ent: int, rel: int = None, cand=None):
        ppr_scores = self.getPPRscoreArray(ent)
        top_nodes = self._select_top_nodes(ppr_scores, ent, self.topk, cand=cand)
        top_nodes, sampled_edges = self._extract_induced_edges(top_nodes)
        sampled_edges = self._apply_topm_sampling(sampled_edges, ppr_scores)

        node_index = torch.zeros(self.n_ent).long()
        node_index[top_nodes] = torch.arange(len(top_nodes))

        if self.args.add_manual_edges:
            add_edges_head2tails = torch.zeros((len(top_nodes), 3)).long()
            add_edges_head2tails[:, 0] = ent
            add_edges_head2tails[:, 1] = 2 * self.n_rel + 1
            add_edges_head2tails[:, 2] = top_nodes
            add_edges_tails2head = torch.zeros((len(top_nodes), 3)).long()
            add_edges_tails2head[:, 0] = top_nodes
            add_edges_tails2head[:, 1] = 2 * self.n_rel + 2
            add_edges_tails2head[:, 2] = ent
            sampled_edges = torch.cat([sampled_edges, add_edges_head2tails, add_edges_tails2head], dim=0)

        return top_nodes, node_index, sampled_edges

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

            sampled_edges[:, 0] = node_index[sampled_edges[:, 0]] + ent_delta
            sampled_edges[:, 2] = node_index[sampled_edges[:, 2]] + ent_delta
            batch_sampled_edges.append(sampled_edges)
            edge_batch_idxs += [batch_idx] * int(sampled_edges.shape[0])

            ent_delta_values.append(num_nodes)
            batch_idxs += [batch_idx] * num_nodes
            abs_idxs += topk_nodes.tolist()
            query_sub_idxs.append(int(node_index[sub]) + ent_delta)

        batch_idxs = torch.LongTensor(batch_idxs)
        abs_idxs = torch.LongTensor(abs_idxs)
        batch_sampled_edges = torch.cat(batch_sampled_edges, dim=0)
        edge_batch_idxs = torch.LongTensor(edge_batch_idxs)
        query_sub_idxs = torch.LongTensor(query_sub_idxs)

        return batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges
