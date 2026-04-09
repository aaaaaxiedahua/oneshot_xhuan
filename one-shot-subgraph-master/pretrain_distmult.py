import argparse
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from distmult import DistMult
from utils import cal_performance, cal_ranks


class TripleDataset(Dataset):
    def __init__(self, triples):
        self.triples = torch.LongTensor(triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


class KGData(object):
    def __init__(self, data_path, train_on="facts+train"):
        self.data_path = data_path
        self.entity2id = self._read_vocab("entities.txt")
        self.relation2id = self._read_vocab("relations.txt")
        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)

        self.fact_triples = self._read_triples("facts.txt")
        self.train_triples = self._read_triples("train.txt")
        self.valid_triples = self._read_triples("valid.txt")
        self.test_triples = self._read_triples("test.txt")

        if train_on == "facts+train":
            self.observed_triples = self.fact_triples + self.train_triples
        elif train_on == "train":
            self.observed_triples = list(self.train_triples)
        else:
            raise ValueError(f"Unsupported train_on mode: {train_on}")

        self.all_known_triples = (
            self.fact_triples + self.train_triples + self.valid_triples + self.test_triples
        )
        self.tail_filters, self.head_filters = self._build_filters(self.all_known_triples)

    def _read_vocab(self, filename):
        vocab = {}
        with open(os.path.join(self.data_path, filename), "r", encoding="utf-8") as fin:
            for idx, line in enumerate(fin):
                vocab[line.strip()] = idx
        return vocab

    def _read_triples(self, filename):
        triples = []
        with open(os.path.join(self.data_path, filename), "r", encoding="utf-8") as fin:
            for line in fin:
                head, relation, tail = line.strip().split()
                triples.append(
                    (
                        self.entity2id[head],
                        self.relation2id[relation],
                        self.entity2id[tail],
                    )
                )
        return triples

    def _build_filters(self, triples):
        tail_filters = defaultdict(set)
        head_filters = defaultdict(set)
        for head, relation, tail in triples:
            tail_filters[(head, relation)].add(tail)
            head_filters[(relation, tail)].add(head)
        tail_filters = {k: np.array(sorted(v), dtype=np.int64) for k, v in tail_filters.items()}
        head_filters = {k: np.array(sorted(v), dtype=np.int64) for k, v in head_filters.items()}
        return tail_filters, head_filters


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain DistMult embeddings on the current KG format.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--emb_dim", type=int, default=200)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--eval_batchsize", type=int, default=64)
    parser.add_argument("--neg_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--reg_lambda", type=float, default=1e-6)
    parser.add_argument("--optimizer", type=str, default="adagrad", choices=["adagrad", "adam"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--train_on", type=str, default="facts+train", choices=["facts+train", "train"])
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--disable_entity_norm", action="store_true")
    parser.add_argument("--cpu", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_idx):
    if gpu_idx >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_idx}")
    return torch.device("cpu")


def build_optimizer(model, args):
    if args.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def make_default_save_path(args):
    dataset = os.path.basename(os.path.normpath(args.data_path))
    save_dir = os.path.join(args.data_path, "kge")
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"distmult_dim{args.emb_dim}.pt")


def train_one_epoch(model, loader, optimizer, n_ent, neg_size, reg_lambda, normalize_entities, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in tqdm(loader, ncols=50, leave=False):
        batch = batch.to(device)
        head = batch[:, 0]
        relation = batch[:, 1]
        tail = batch[:, 2]
        batch_size = head.shape[0]

        negative_tail = torch.randint(0, n_ent, (batch_size, neg_size), device=device)
        negative_head = torch.randint(0, n_ent, (batch_size, neg_size), device=device)

        optimizer.zero_grad()
        positive_score = model.score_triples(head, relation, tail)
        negative_tail_score = model.score_negative_tails(head, relation, negative_tail)
        negative_head_score = model.score_negative_heads(negative_head, relation, tail)

        positive_loss = -F.logsigmoid(positive_score).mean()
        negative_loss = -0.5 * (
            F.logsigmoid(-negative_tail_score).mean() + F.logsigmoid(-negative_head_score).mean()
        )
        reg_loss = reg_lambda * (
            model.entity_emb.weight.pow(2).mean() + model.relation_emb.weight.pow(2).mean()
        )
        loss = positive_loss + negative_loss + reg_loss
        loss.backward()
        optimizer.step()

        if normalize_entities:
            model.normalize_entity_embeddings()

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_split(model, kg_data, triples, batch_size, device):
    model.eval()
    all_ranks = []

    for start in tqdm(range(0, len(triples), batch_size), ncols=50, leave=False):
        batch = triples[start:start + batch_size]
        head = torch.LongTensor([triple[0] for triple in batch]).to(device)
        relation = torch.LongTensor([triple[1] for triple in batch]).to(device)
        tail = torch.LongTensor([triple[2] for triple in batch]).to(device)

        tail_scores = model.score_all_tails(head, relation).cpu().numpy()
        tail_labels = np.zeros((len(batch), kg_data.n_ent), dtype=np.float32)
        tail_filters = np.zeros((len(batch), kg_data.n_ent), dtype=np.float32)
        for idx, (h_i, r_i, t_i) in enumerate(batch):
            tail_labels[idx, t_i] = 1.0
            tail_filters[idx, kg_data.tail_filters[(h_i, r_i)]] = 1.0
        all_ranks.extend(cal_ranks(tail_scores, tail_labels, tail_filters))

        head_scores = model.score_all_heads(relation, tail).cpu().numpy()
        head_labels = np.zeros((len(batch), kg_data.n_ent), dtype=np.float32)
        head_filters = np.zeros((len(batch), kg_data.n_ent), dtype=np.float32)
        for idx, (h_i, r_i, t_i) in enumerate(batch):
            head_labels[idx, h_i] = 1.0
            head_filters[idx, kg_data.head_filters[(r_i, t_i)]] = 1.0
        all_ranks.extend(cal_ranks(head_scores, head_labels, head_filters))

    all_ranks = np.array(all_ranks)
    return cal_performance(all_ranks)


def save_checkpoint(model, optimizer, args, kg_data, save_path, best_valid_mrr):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "entity_embedding": model.entity_emb.weight.detach().cpu(),
            "relation_embedding": model.relation_emb.weight.detach().cpu(),
            "entity2id": kg_data.entity2id,
            "relation2id": kg_data.relation2id,
            "config": vars(args),
            "best_valid_mrr": best_valid_mrr,
        },
        save_path,
    )


def load_checkpoint(model, optimizer, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def main():
    args = parse_args()
    if args.only_eval and not args.weight:
        raise ValueError("`--only_eval` requires `--weight`.")
    set_seed(args.seed)
    device = get_device(args.gpu)
    if args.cpu > 0:
        torch.set_num_threads(max(1, args.cpu))

    kg_data = KGData(args.data_path, train_on=args.train_on)
    model = DistMult(kg_data.n_ent, kg_data.n_rel, args.emb_dim).to(device)
    optimizer = build_optimizer(model, args)

    save_path = args.save_path if args.save_path else make_default_save_path(args)

    if args.weight:
        checkpoint = load_checkpoint(model, optimizer if not args.only_eval else None, args.weight, device)
        if args.only_eval:
            valid_mrr, valid_h1, valid_h10 = evaluate_split(
                model, kg_data, kg_data.valid_triples, args.eval_batchsize, device
            )
            test_mrr, test_h1, test_h10 = evaluate_split(
                model, kg_data, kg_data.test_triples, args.eval_batchsize, device
            )
            print(f'Load weight from: {args.weight}')
            print(
                '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t'
                '[TEST] MRR:%.4f H@1:%.4f H@10:%.4f'
                % (valid_mrr, valid_h1, valid_h10, test_mrr, test_h1, test_h10)
            )
            return
        else:
            print(
                'Resume training from: %s (best_valid_mrr=%.4f)'
                % (args.weight, float(checkpoint.get("best_valid_mrr", 0.0)))
            )

    train_dataset = TripleDataset(kg_data.observed_triples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.cpu,
        pin_memory=(device.type == "cuda"),
    )

    best_valid_mrr = -1.0
    patience_count = 0

    print('==> DistMult pretraining')
    print(f'==> device: {device}')
    print(
        '==> observed triples: %d (train_on=%s), valid: %d, test: %d'
        % (
            len(kg_data.observed_triples),
            args.train_on,
            len(kg_data.valid_triples),
            len(kg_data.test_triples),
        )
    )

    start_time = time.time()
    for epoch in range(args.epoch):
        epoch_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            kg_data.n_ent,
            args.neg_size,
            args.reg_lambda,
            normalize_entities=(not args.disable_entity_norm),
            device=device,
        )

        should_eval = ((epoch + 1) % args.eval_every == 0) or epoch == 0 or (epoch + 1 == args.epoch)
        if not should_eval:
            print('Epoch %03d: loss=%.6f' % (epoch, epoch_loss))
            continue

        valid_mrr, valid_h1, valid_h10 = evaluate_split(
            model, kg_data, kg_data.valid_triples, args.eval_batchsize, device
        )
        test_mrr, test_h1, test_h10 = evaluate_split(
            model, kg_data, kg_data.test_triples, args.eval_batchsize, device
        )
        elapsed = time.time() - start_time
        print(
            'Epoch %03d: loss=%.6f [VALID] MRR:%.4f H@1:%.4f H@10:%.4f '
            '[TEST] MRR:%.4f H@1:%.4f H@10:%.4f [TIME] %.2fs'
            % (
                epoch,
                epoch_loss,
                valid_mrr,
                valid_h1,
                valid_h10,
                test_mrr,
                test_h1,
                test_h10,
                elapsed,
            )
        )

        if valid_mrr > best_valid_mrr:
            best_valid_mrr = valid_mrr
            patience_count = 0
            save_checkpoint(model, optimizer, args, kg_data, save_path, best_valid_mrr)
            print(f'Save DistMult checkpoint to: {save_path}')
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f'Early stopping at epoch {epoch + 1}.')
                break


if __name__ == "__main__":
    main()
