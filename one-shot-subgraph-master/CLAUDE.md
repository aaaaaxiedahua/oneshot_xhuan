# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of the ICLR 2024 paper "Less is More: One-shot Subgraph Reasoning on Large-scale Knowledge Graphs". Performs link prediction on large Knowledge Graphs by: (1) extracting a query-dependent subgraph via Personalized PageRank (PPR), then (2) running a GNN on that subgraph. Supported datasets: WN18RR, NELL995, YAGO3-10, family, umls.

## Commands

### Train from scratch
```bash
# WN18RR (batchsize=16)
python3 train_auto.py --data_path ./data/WN18RR/ --batchsize 16 --gpu 0 --topk 0.1 --topm -1 --fact_ratio 0.95

# NELL995 (batchsize=8)
python3 train_auto.py --data_path ./data/nell --batchsize 8 --gpu 0 --topk 0.1 --topm -1 --fact_ratio 0.95

# YAGO3-10 (batchsize=4, fact_ratio=0.995)
python3 train_auto.py --data_path ./data/YAGO --batchsize 4 --gpu 0 --topk 0.1 --topm -1 --fact_ratio 0.995

# family / umls (topk=0.5)
python3 train_auto.py --data_path data/family/ --topk 0.5
python3 train_auto.py --data_path data/umls/ --topk 0.5
```

### Evaluate with pretrained checkpoints
```bash
python3 train_auto.py --data_path ./data/WN18RR/ --batchsize 16 --only_eval --gpu 0 \
    --topk 0.1 --topm -1 --weight ./savedModels/WN18RR_topk_0.1_layer_6_ValMRR_0.569.pt
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

### Experimental modules
```bash
# RBPPR (Relation-Biased PPR): fuses relation tail-frequency bias with entity PPR
python3 train_auto.py --data_path data/WN18RR/ --topk 0.1 --use_rbppr --rbppr_lambda 0.1

# EdgePrune: query-wise cumulative edge pruning across GNN layers
python3 train_auto.py --data_path data/WN18RR/ --topk 0.1 --use_edgeprune \
    --edgeprune_ratio_start 1.0 --edgeprune_ratio_end 0.8

# HPO with experimental modules
python3 search_auto.py --data_path data/WN18RR/ --search --use_rbppr
python3 search_auto.py --data_path data/WN18RR/ --search --use_edgeprune
```

There is no automated test suite. Correctness is validated by comparing MRR/Hit metrics against published results.

## Architecture

### Entry Points
- **`train_auto.py`** -- Main entry: loads dataset-specific best configs (hardcoded `params` dict per dataset name), builds model, trains or evaluates. Calls `exit()` for unknown datasets.
- **`search_auto.py`** -- HPO search entry using Bayesian optimization with Random Forest surrogate (`base_HPO.py:RF_HPO`). Defines search spaces: base `HPO_search_space` + optional `HPO_search_space_RBPPR` / `HPO_search_space_EDGEPRUNE`.

### Core Pipeline (data flow: load_data -> PPR_sampler -> model -> base_model)
- **`load_data.py`** (`DataLoader`, extends `torch.utils.data.Dataset`) -- Reads entity/relation/triple files, adds inverse relations (relation IDs offset by `n_rel`, identity self-loop at `2*n_rel`). `shuffle_train()` re-partitions facts+train triples each epoch using `--fact_ratio`. `collate_fn` calls `getBatchSubgraph` to assemble batched subgraph tensors.
- **`PPR_sampler.py`** (`pprSampler`) -- Computes and caches PPR scores per entity as `data/<dataset>/ppr_scores/<entity_id>.pkl`. Selects top-K nodes per query head, extracts induced subgraph from sparse adjacency. Two separate samplers: train (fact-only graph, updated each epoch via `updateEdges`) and test (full graph, fixed). Optionally computes RBPPR scores cached in `data/<dataset>/rbppr_scores/`.
- **`model.py`** (`GNN_auto`, `GNNLayer`, `EdgePruneLayer`) -- Attention-based message-passing GNN with GRU gating across layers. `GNNLayer` uses `torch_scatter.scatter` for neighbor aggregation. `GNN_auto.forward()` initializes node hidden states (`binary`/`relation`), runs L message-passing layers with optional EdgePrune routing, then applies readout (`linear`/`multiply`/`pair_mlp`). Scores are scattered back to full entity space.
- **`base_model.py`** (`BaseModel`) -- Training loop with log-softmax cross-entropy loss, `ReduceLROnPlateau` scheduler, filtered MRR/Hit@1/Hit@10 evaluation, checkpoint save/load, early stopping (patience=20 train, patience=3 HPO). Tracks RBPPR and EdgePrune diagnostics per epoch.
- **`utils.py`** -- `cal_ranks` (filtered ranking), `cal_performance` (MRR/Hit metrics), `select_gpu`, `checkPath`.
- **`base_HPO.py`** (`RF_HPO`) -- Bayesian HPO with evolutionary candidate generation and Random Forest surrogate.

### Standalone / Unwired Modules
- **`distmult.py` + `pretrain_distmult.py`** -- Standalone DistMult pretraining pipeline with its own CLI (`--data_path`, `--emb_dim`, `--neg_size`, `--reg_lambda`, etc.). Not invoked by `train_auto.py`; used to produce embeddings offline.
- **`optuna_hpo.py`** -- Optional Optuna TPE + Hyperband HPO runner (`OptunaTPEHyperbandHPO`, `TrialReporter`). Gracefully handles missing `optuna` import. Not wired into `search_auto.py` -- intended as a replacement path for the custom `RF_HPO` in `base_HPO.py`.

### Commented-out Modules (Module 1 & Module 2)
Large blocks of commented code exist across `model.py`, `PPR_sampler.py`, `search_auto.py`, `train_auto.py` for two experimental features that were never finalized:
- **Module 1: Relation-Path Conditioned Sampling** -- Mines 2-hop relation path patterns, fuses path reachability with PPR. Flags: `--use_rel_prior`, `--path_lambda`, `--rel_path_topk`, `--fusion_mode`.
- **Module 2: Relation Composition Augmentation (RCA)** -- `RelationComposer` class that finds multi-hop paths in the subgraph, scores compositions, adds virtual edges. Flags: `--use_rca`, `--compose_dim`, `--max_virtual`, `--compose_aware`, `--rca_mode`, `--compose_max_hop`.

These are disabled but preserved for reference. Do not delete without explicit instruction.

## Key Development Notes

- **GPU required**: `.cuda()` calls throughout; CPU-only needs code changes.
- **PPR precomputation is expensive**: First run generates per-entity `.pkl` files (hours for WN18RR, days for YAGO). Do not delete `data/<dataset>/ppr_scores/`. RBPPR adds per-relation `.pkl` files in `rbppr_scores/`.
- **torch_scatter**: Must exactly match PyTorch and CUDA versions. Any env change requires reinstalling it.
- **Batch size by dataset**: WN18RR=16, NELL=8, YAGO=4 (GPU memory constraint).
- **No requirements.txt**: Key deps are `torch`, `torch_scatter`, `numpy`, `scipy`, `scikit-learn`, `networkx`, `tqdm`. Tested versions: torch 2.2.1, torch_scatter 2.1.2+pt22cu121, numpy 1.26.4, scipy 1.12.0, sklearn 1.4.1.post1, networkx 3.2.1.
- **Dataset-locked configs**: `train_auto.py` hardcodes best hyperparameters per dataset (WN18RR, nell, YAGO, family, umls) and exits on unrecognized dataset names.
- **Relation ID convention**: Original relations are `0..n_rel-1`, inverse relations are `n_rel..2*n_rel-1`, identity self-loop is `2*n_rel`.

## Data Format

Each dataset directory (`data/<name>/`) contains: `entities.txt`, `relations.txt`, `facts.txt`, `train.txt`, `valid.txt`, `test.txt`. Triples are space-separated `<head> <relation> <tail>`.

## Output

- Training logs: `results/<dataset>/<timestamp>.txt`
- HPO logs: `results/<dataset>/search_log.pkl`
- Checkpoints: `<data_path>/saveModel/topk_<topk>_layer_<layer>_ValMRR_<score>.pt`
