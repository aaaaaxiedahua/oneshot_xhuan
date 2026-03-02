# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of the ICLR 2024 paper "Less is More: One-shot Subgraph Reasoning on Large-scale Knowledge Graphs". Performs link prediction on large Knowledge Graphs by: (1) extracting a query-dependent subgraph via Personalized PageRank (PPR), then (2) running a GNN on that subgraph. Supported datasets: WN18RR, NELL995, YAGO3-10.

## Commands

### Evaluate with pretrained checkpoints
```bash
python3 train_auto.py --data_path ./data/WN18RR/ --batchsize 16 --only_eval --gpu 0 \
    --topk 0.1 --topm -1 --weight ./savedModels/WN18RR_topk_0.1_layer_6_ValMRR_0.569.pt
```

### Train from scratch (WN18RR example)
```bash
python3 train_auto.py --data_path ./data/WN18RR/ --batchsize 16 --gpu 0 \
    --topk 0.1 --topm -1 --fact_ratio 0.95
```

### Hyper-parameter search
```bash
python3 search_auto.py --data_path ./data/WN18RR/ --gpu 0 --topk 0.1 \
    --cpu 2 --fact_ratio 0.95 --batchsize 16 --search
```

### View HPO results
```bash
python3 showResults.py --file ./results/WN18RR/search_log.pkl
```

There is no automated test suite. Correctness is validated by comparing MRR/Hit metrics against published results.

## Architecture

### Entry Points
- **`train_auto.py`** — Main entry: loads dataset-specific best configs (hardcoded), builds model, trains or evaluates. Calls `exit()` for unknown datasets.
- **`search_auto.py`** — HPO search entry using Bayesian optimization with Random Forest surrogate.

### Core Pipeline
- **`load_data.py`** (`DataLoader`) — Reads entity/relation/triple files, adds inverse relations (relation IDs offset by `n_rel`, identity at `2*n_rel`). Implements `shuffle_train()` which re-partitions facts+train triples each epoch using `--fact_ratio`.
- **`PPR_sampler.py`** (`pprSampler`) — Computes and caches PPR scores per entity as `data/<dataset>/ppr_scores/<entity_id>.pkl`. Selects top-K nodes per query, extracts induced subgraph. Two separate samplers exist: train (fact-only graph, updated each epoch) and test (full graph, fixed).
- **`model.py`** (`GNN_auto`, `GNNLayer`) — Attention-based message-passing GNN with GRU gating across layers. Uses `torch_scatter.scatter` for aggregation. Configurable activation, initializer (`binary`/`relation`), hidden concat, shortcut, readout (`linear`/`multiply`).
- **`base_model.py`** (`BaseModel`) — Training loop, log-softmax cross-entropy loss, filtered MRR/Hit@1/Hit@10 evaluation, checkpoint save/load, early stopping (patience=20 train, patience=3 HPO).
- **`base_HPO.py`** (`RF_HPO`) — Bayesian HPO with evolutionary candidate generation and Random Forest surrogate.

## Key Development Notes

- **GPU required**: `.cuda()` calls throughout; CPU-only needs code changes.
- **PPR precomputation is expensive**: First run generates per-entity `.pkl` files (hours for WN18RR, days for YAGO). Do not delete `data/<dataset>/ppr_scores/`.
- **torch_scatter**: Must exactly match PyTorch and CUDA versions. Any env change requires reinstalling it.
- **Batch size by dataset**: WN18RR=16, NELL=8, YAGO=4 (GPU memory constraint).
- **No requirements.txt**: Key deps are `torch`, `torch_scatter`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `tqdm`.
- **Dataset-locked configs**: `train_auto.py` hardcodes best hyperparameters per dataset and exits on unrecognized datasets.

## Data Format

Each dataset directory (`data/<name>/`) contains: `entities.txt`, `relations.txt`, `facts.txt`, `train.txt`, `valid.txt`, `test.txt`. Triples are space-separated `<head> <relation> <tail>`.

## Output

- Training logs: `results/<dataset>/<timestamp>.txt`
- HPO logs: `results/<dataset>/search_log.pkl`
- Checkpoints: `<data_path>/saveModel/topk_<topk>_layer_<layer>_ValMRR_<score>.pt`
