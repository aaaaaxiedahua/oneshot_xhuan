import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMult(nn.Module):
    def __init__(self, n_ent, n_rel, emb_dim):
        super().__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.emb_dim = emb_dim
        self.entity_emb = nn.Embedding(n_ent, emb_dim)
        self.relation_emb = nn.Embedding(n_rel, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.normalize_entity_embeddings()

    @torch.no_grad()
    def normalize_entity_embeddings(self):
        self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=-1)

    def score_triples(self, head, relation, tail):
        head_vec = self.entity_emb(head)
        relation_vec = self.relation_emb(relation)
        tail_vec = self.entity_emb(tail)
        return torch.sum(head_vec * relation_vec * tail_vec, dim=-1)

    def score_all_tails(self, head, relation):
        head_vec = self.entity_emb(head)
        relation_vec = self.relation_emb(relation)
        query_vec = head_vec * relation_vec
        return torch.matmul(query_vec, self.entity_emb.weight.t())

    def score_all_heads(self, relation, tail):
        relation_vec = self.relation_emb(relation)
        tail_vec = self.entity_emb(tail)
        query_vec = relation_vec * tail_vec
        return torch.matmul(query_vec, self.entity_emb.weight.t())

    def score_negative_tails(self, head, relation, negative_tail):
        head_vec = self.entity_emb(head).unsqueeze(1)
        relation_vec = self.relation_emb(relation).unsqueeze(1)
        tail_vec = self.entity_emb(negative_tail)
        return torch.sum(head_vec * relation_vec * tail_vec, dim=-1)

    def score_negative_heads(self, negative_head, relation, tail):
        head_vec = self.entity_emb(negative_head)
        relation_vec = self.relation_emb(relation).unsqueeze(1)
        tail_vec = self.entity_emb(tail).unsqueeze(1)
        return torch.sum(head_vec * relation_vec * tail_vec, dim=-1)


def load_distmult_embeddings(checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint["entity_embedding"], checkpoint["relation_embedding"]
