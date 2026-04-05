# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing SelfGNN (Self-Supervised Graph Neural Networks for Sequential Recommendation) in PyTorch, ported from the original TensorFlow code. The goal is to extend SelfGNN with edge features (star ratings) and node features (user/merchant metadata) on a Yelp merchant recommendation dataset, producing results for an academic paper written in Mongolian.

## Repository Structure

```
gg/
├── selfGNN-Base/        # Baseline SelfGNN (binary adjacency, no features)
├── selfGNN-Feature/     # Extended SelfGNN with edge/node feature flags
├── analysis/            # Scripts that read Results/ and produce figures + LaTeX
├── scripts/             # Shell scripts to run training pipelines
├── Datasets/yelp/       # Preprocessed dataset files (pickled)
├── Results/             # All JSON result files from training runs
└── preprocess_to_yelp_merchant.ipynb  # Dataset preprocessing notebook
```

The two codebases (`selfGNN-Base` and `selfGNN-Feature`) are intentionally separate so the baseline remains untouched and reproducible.

## Running Training

**Baseline (selfGNN-Base):**
```bash
cd selfGNN-Base
python train.py --data yelp-merchant --save_path yelp_merchant_baseline --device cuda
# Or using the script:
bash ../scripts/run_yelp_merchant.sh
```

**Featured configs (selfGNN-Feature):**
```bash
cd selfGNN-Feature
# Config 2: Edge features only
python train.py --data yelp-merchant --use_edge_features --save_path config2_edge

# Config 3: Node features only
python train.py --data yelp-merchant --use_node_features --save_path config3_node

# Config 4: Edge + Node features
python train.py --data yelp-merchant --use_edge_features --use_node_features --save_path config4_both

# Or run C2/C3/C4 sequentially:
bash scripts/run_featured_training.sh
```

**Quick smoke test (1 epoch):**
```bash
python train.py --epoch 1 --data yelp-merchant
```

## Feature Extraction

Before running `selfGNN-Feature` with feature flags, extract features from raw Yelp JSON:
```bash
cd selfGNN-Feature
python feature_extractor.py
```
Outputs to `Datasets/yelp-merchant-features/`: `user_features.npy`, `merchant_features.npy`, `edge_weights.pkl`.

## Analysis and Paper

```bash
python analysis/compare_results.py         # Comparison tables + figures
python analysis/dataset_statistics.py      # Dataset stats JSON
python analysis/generate_paper_tables.py   # LaTeX tables for the paper
```

Results JSON files in `Results/` must exist before running analysis scripts.

## Architecture

**SelfGNN pipeline** (same in both variants):
1. **Graph encode**: Per-graph learnable embeddings → LightGCN message passing (L layers, residual + layer-sum) over T time-interval sub-graphs
2. **Temporal encode**: LSTM over the T sub-graph representations → multi-head self-attention → mean pool → global user/item embeddings
3. **Sequence encode**: Lookup item embeddings for user's interaction sequence + positional embeddings → stack of MHSA layers with LeakyReLU + residual → sum pooling
4. **Prediction**: `dot(user_long, item_long) + dot(leaky_relu(seq_user), item_long)`
5. **Loss**: BPR margin loss + SAL (self-aligned loss) with personalized weight network

**selfGNN-Feature additions:**
- `--use_edge_features`: Replace binary adjacency with star-rating weights (log-sigmoid + symmetric normalization)
- `--use_node_features`: Add MLP projection of user/merchant features, additively fused at layer 0 of each sub-graph

**Dataset files** (pickled, in `Datasets/yelp/` or `Datasets/yelp-merchant/`):
- `trn_mat_time`: `[overall_mat, sub_mats_list, time_mat]`
- `tst_int`: dict of `{user_id: target_item_id}`
- `sequence`: user behavior sequences (0-indexed IDs)
- `test_dict`: negative items per user for evaluation

## Key Hyperparameters (Yelp-Merchant defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| latdim | 64 | Embedding dimension |
| graphNum | 5 | Must match preprocessing GRAPH_NUM |
| gnn_layer | 3 | LightGCN layers |
| att_layer | 2 | Sequence self-attention layers |
| batch | 512 | |
| keepRate | 0.5 | Edge dropout rate |
| ssl_reg | 1e-7 | SAL loss weight |
| patience | 20 | Early stopping in eval epochs |

## Evaluation

- Metric: HR@10, NDCG@10, HR@20, NDCG@20
- Protocol: 1000-item ranking (1 positive + 999 negatives from `test_dict`)
- Expected baseline on Yelp-Merchant: HR@10 in 0.2–0.5 range

## Dependency Setup

```bash
pip install torch numpy scipy matplotlib
```

## Debugging Poor Results

- HR@10 < 0.05: check `test_dict` key indexing (must be 0-indexed), verify `sequence` format, ensure `graphNum` ≤ actual sub-graph count
- Features not helping: check coverage via `feature_extractor.py` output, verify edge weight distribution is not uniform
