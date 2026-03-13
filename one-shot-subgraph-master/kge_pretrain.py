import argparse
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_vocab(path):
    mapping = {}
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            mapping[line.strip()] = idx
    return mapping


def read_triples(path, ent2id, rel2id):
    triples = []
    with open(path, "r") as f:
        for line in f:
            h, r, t = line.strip().split()
            triples.append((ent2id[h], rel2id[r], ent2id[t]))
    return triples


def add_inverse(triples, n_rel):
    out = list(triples)
    for h, r, t in triples:
        out.append((t, r + n_rel, h))
    return out


def build_filters(all_triples):
    filt = defaultdict(set)
    for h, r, t in all_triples:
        filt[(h, r)].add(t)
    return filt


def build_query_answers(triples):
    qa = defaultdict(set)
    for h, r, t in triples:
        qa[(h, r)].add(t)
    return qa


class KGEScorer(nn.Module):
    def __init__(self, n_ent, n_rel, dim, score_fn):
        super().__init__()
        self.score_fn = score_fn
        self.entity_emb = nn.Embedding(n_ent, dim)
        self.relation_emb = nn.Embedding(n_rel, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score(self, h_idx, r_idx, t_idx):
        eh = self.entity_emb(h_idx)
        er = self.relation_emb(r_idx)
        et = self.entity_emb(t_idx)
        if self.score_fn == "transe":
            return -torch.norm(eh + er - et, p=1, dim=-1)
        return torch.sum(eh * er * et, dim=-1)

    def score_all_tails(self, h_idx, r_idx):
        eh = self.entity_emb(h_idx)  # [d]
        er = self.relation_emb(r_idx)  # [d]
        all_t = self.entity_emb.weight  # [n_ent, d]
        if self.score_fn == "transe":
            return -torch.norm((eh + er).unsqueeze(0) - all_t, p=1, dim=-1)
        return torch.sum(all_t * (eh * er), dim=-1)


@torch.no_grad()
def eval_recall_at_k(model, query_answers, all_filters, n_ent, k, device):
    if len(query_answers) == 0:
        return 0.0
    hit = 0
    total = 0
    for (h, r), ans_set in query_answers.items():
        h_t = torch.tensor(h, dtype=torch.long, device=device)
        r_t = torch.tensor(r, dtype=torch.long, device=device)
        scores = model.score_all_tails(h_t, r_t)

        # Filter all known true tails except the current query's target set.
        for t in all_filters[(h, r)]:
            if t not in ans_set:
                scores[t] = -1e30

        topk = torch.topk(scores, k=min(k, n_ent), largest=True).indices.cpu().tolist()
        if any(t in ans_set for t in topk):
            hit += 1
        total += 1
    return float(hit) / float(max(total, 1))


def sample_batch(train_triples, batch_size):
    idx = np.random.randint(0, len(train_triples), size=batch_size)
    batch = [train_triples[i] for i in idx]
    h = torch.tensor([x[0] for x in batch], dtype=torch.long)
    r = torch.tensor([x[1] for x in batch], dtype=torch.long)
    t = torch.tensor([x[2] for x in batch], dtype=torch.long)
    return h, r, t


def sample_corrupted_triples(h, r, t, n_ent, neg_size, device):
    batch_size = h.shape[0]
    rand_ent = torch.randint(low=0, high=n_ent, size=(batch_size, neg_size), device=device)
    corrupt_head = torch.rand((batch_size, neg_size), device=device) < 0.5

    neg_h = h.unsqueeze(1).repeat(1, neg_size)
    neg_r = r.unsqueeze(1).repeat(1, neg_size)
    neg_t = t.unsqueeze(1).repeat(1, neg_size)

    neg_h[corrupt_head] = rand_ent[corrupt_head]
    neg_t[~corrupt_head] = rand_ent[~corrupt_head]
    return neg_h, neg_r, neg_t


def infer_dataset_name(data_path):
    norm = os.path.normpath(data_path)
    name = os.path.basename(norm)
    return name if name != "" else "dataset"


def main():
    parser = argparse.ArgumentParser(description="Train lightweight DistMult/TransE retriever for R-BiPPR")
    parser.add_argument("--data_path", type=str, default="data/WN18RR/")
    parser.add_argument("--score_fn", type=str, default="distmult", choices=["distmult", "transe"])
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--neg_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--entity_norm", action="store_true")
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--recall_k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--save_root", type=str, default="savedModels/retrievers")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ent2id = read_vocab(os.path.join(args.data_path, "entities.txt"))
    rel2id = read_vocab(os.path.join(args.data_path, "relations.txt"))
    n_ent = len(ent2id)
    n_rel = len(rel2id)

    facts = read_triples(os.path.join(args.data_path, "facts.txt"), ent2id, rel2id)
    train = read_triples(os.path.join(args.data_path, "train.txt"), ent2id, rel2id)
    valid = read_triples(os.path.join(args.data_path, "valid.txt"), ent2id, rel2id)
    test = read_triples(os.path.join(args.data_path, "test.txt"), ent2id, rel2id)

    train_pos = add_inverse(facts + train, n_rel)
    valid_eval = add_inverse(valid, n_rel)
    all_known = add_inverse(facts + train + valid + test, n_rel)

    query_answers = build_query_answers(valid_eval)
    all_filters = build_filters(all_known)
    n_rel_total = 2 * n_rel

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"==> device: {device}")
    print(f"==> train triples (with inverse): {len(train_pos)}")
    print(f"==> valid queries (with inverse): {len(query_answers)}")
    print(f"==> training loss: margin ranking (margin={args.margin})")
    use_entity_norm = (args.score_fn == "transe") or args.entity_norm

    model = KGEScorer(n_ent=n_ent, n_rel=n_rel_total, dim=args.emb_dim, score_fn=args.score_fn).to(device)
    if use_entity_norm:
        with torch.no_grad():
            model.entity_emb.weight.data = F.normalize(model.entity_emb.weight.data, p=2, dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_recall = -1.0
    best_state = None
    bad_eval_count = 0
    n_batches = max(1, int(args.steps_per_epoch))

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        epoch_loss = 0.0
        for _ in range(n_batches):
            h, r, t = sample_batch(train_pos, args.batch_size)
            h = h.to(device)
            r = r.to(device)
            t = t.to(device)

            neg_h, neg_r, neg_t = sample_corrupted_triples(
                h=h, r=r, t=t, n_ent=n_ent, neg_size=args.neg_size, device=device
            )

            pos_score = model.score(h, r, t).unsqueeze(1)  # [B, 1]
            neg_score = model.score(
                neg_h.reshape(-1), neg_r.reshape(-1), neg_t.reshape(-1)
            ).reshape(h.shape[0], args.neg_size)  # [B, neg]

            # Margin ranking loss: max(0, margin + s(neg) - s(pos))
            loss = torch.relu(args.margin + neg_score - pos_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_entity_norm:
                with torch.no_grad():
                    model.entity_emb.weight.data = F.normalize(model.entity_emb.weight.data, p=2, dim=1)
            epoch_loss += float(loss.item())

        print(f"[Epoch {epoch:03d}] loss={epoch_loss / n_batches:.6f}")

        if epoch % args.eval_every != 0:
            continue

        model.eval()
        recall = eval_recall_at_k(
            model=model,
            query_answers=query_answers,
            all_filters=all_filters,
            n_ent=n_ent,
            k=args.recall_k,
            device=device,
        )
        print(f"[Eval {epoch:03d}] Recall@{args.recall_k}={recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            best_state = {
                "entity_emb": model.entity_emb.weight.detach().cpu().clone(),
                "relation_emb": model.relation_emb.weight.detach().cpu().clone(),
                "score_fn": args.score_fn,
                "emb_dim": args.emb_dim,
                "best_recall_at_k": best_recall,
                "recall_k": args.recall_k,
                "epoch": epoch,
                "margin": args.margin,
                "entity_norm": use_entity_norm,
            }
            bad_eval_count = 0
            print(f"==> new best Recall@{args.recall_k}: {best_recall:.4f}")
        else:
            bad_eval_count += 1
            print(f"==> no improvement count: {bad_eval_count}/{args.patience}")

        if bad_eval_count >= args.patience:
            print(f"==> early stop at epoch {epoch} (patience={args.patience})")
            break

    if best_state is None:
        model.eval()
        recall = eval_recall_at_k(
            model=model,
            query_answers=query_answers,
            all_filters=all_filters,
            n_ent=n_ent,
            k=args.recall_k,
            device=device,
        )
        best_state = {
            "entity_emb": model.entity_emb.weight.detach().cpu().clone(),
            "relation_emb": model.relation_emb.weight.detach().cpu().clone(),
            "score_fn": args.score_fn,
            "emb_dim": args.emb_dim,
            "best_recall_at_k": recall,
            "recall_k": args.recall_k,
            "epoch": args.max_epoch,
            "margin": args.margin,
            "entity_norm": use_entity_norm,
        }
        print(f"==> fallback save with Recall@{args.recall_k}: {recall:.4f}")

    if args.save_path != "":
        final_save_path = args.save_path
    else:
        dataset = infer_dataset_name(args.data_path)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fname = (
            f"emb{args.emb_dim}"
            f"_rk{args.recall_k}"
            f"_neg{args.neg_size}"
            f"_bs{args.batch_size}"
            f"_spe{args.steps_per_epoch}"
            f"_maxe{args.max_epoch}"
            f"_ev{args.eval_every}"
            f"_{stamp}.ckpt"
        )
        final_save_path = os.path.join(args.save_root, dataset, args.score_fn, fname)

    save_dir = os.path.dirname(final_save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state, final_save_path)
    print(f"==> retriever checkpoint saved: {final_save_path}")


if __name__ == "__main__":
    main()
