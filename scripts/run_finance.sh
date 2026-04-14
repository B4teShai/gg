#!/usr/bin/env bash
# run_finance.sh — Train all model variants on finance-merchant and print results.
#
# Usage:
#   bash run_finance.sh [--device cuda|mps|cpu] [--epoch N] [--seeds "42 100 123"]

set -euo pipefail

DATASET="finance-merchant"
DEVICE="cuda"
EPOCH=150
SEEDS=(42 100 123)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --epoch)  EPOCH="$2";  shift 2 ;;
    --seeds)  read -ra SEEDS <<< "$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/Results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  Dataset : $DATASET"
echo "  Device  : $DEVICE"
echo "  Epochs  : $EPOCH"
echo "  Seeds   : ${SEEDS[*]}"
echo "============================================================"

show_results() {
  local TAG="$1"
  echo ""
  echo "* $TAG"
  python3 - "$RESULTS_DIR" "$TAG" "${SEEDS[@]}" <<'EOF'
import json, sys, math

results_dir = sys.argv[1]
tag         = sys.argv[2]
seeds       = sys.argv[3:]

metrics = {'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
for s in seeds:
    f = f"{results_dir}/{tag}_seed{s}.json"
    try:
        d  = json.load(open(f))
        tr = d.get("test_results", {})
        for m in metrics:
            if m in tr:
                metrics[m].append(float(tr[m]))
    except FileNotFoundError:
        pass

for m, vals in metrics.items():
    if vals:
        mean = sum(vals) / len(vals)
        std  = math.sqrt(sum((v - mean)**2 for v in vals) / max(len(vals)-1, 1)) if len(vals) > 1 else 0.0
        print(f"  {m:<10} {mean:.4f} +/- {std:.4f}  (n={len(vals)})")
    else:
        print(f"  {m}: no results found")
EOF
}

# ── 0. Extract improved 8+8 node features ───────────────────────
echo ""
echo ">>> [0/4] Extracting improved features for $DATASET"
echo "------------------------------------------------------------"
(
  cd "$ROOT_DIR"
  python scripts/extract_features_v2.py --dataset "$DATASET"
)

# ── 1. SelfGNN-Base (no features) ───────────────────────────────
echo ""
echo ">>> [1/4] selfGNN-Base on $DATASET"
echo "------------------------------------------------------------"
for SEED in "${SEEDS[@]}"; do
  echo "  -- seed $SEED"
  (
    cd "$ROOT_DIR/selfGNN-Base"
    python train.py \
      --data "$DATASET" \
      --device "$DEVICE" \
      --epoch "$EPOCH" \
      --seed "$SEED" \
      --save_path "finance_merchant_base_seed${SEED}"
  )
done

# ── 2. SelfGNN-Feature (node features only) ─────────────────────
echo ""
echo ">>> [2/4] selfGNN-Feature (node only) on $DATASET"
echo "------------------------------------------------------------"
for SEED in "${SEEDS[@]}"; do
  echo "  -- seed $SEED"
  (
    cd "$ROOT_DIR/selfGNN-Feature"
    python train.py \
      --data "$DATASET" \
      --device "$DEVICE" \
      --epoch "$EPOCH" \
      --seed "$SEED" \
      --use_node_features \
      --node_mlp_hidden 128 \
      --save_path "finance_merchant_node_seed${SEED}"
  )
done

# ── 3. SelfGNN-Feature (edge features only) ─────────────────────
echo ""
echo ">>> [3/4] selfGNN-Feature (edge only) on $DATASET"
echo "------------------------------------------------------------"
for SEED in "${SEEDS[@]}"; do
  echo "  -- seed $SEED"
  (
    cd "$ROOT_DIR/selfGNN-Feature"
    python train.py \
      --data "$DATASET" \
      --device "$DEVICE" \
      --epoch "$EPOCH" \
      --seed "$SEED" \
      --use_edge_features \
      --save_path "finance_merchant_edge_seed${SEED}"
  )
done

# ── 4. SelfGNN-Feature (node + edge features) ───────────────────
echo ""
echo ">>> [4/4] selfGNN-Feature (node + edge) on $DATASET"
echo "------------------------------------------------------------"
for SEED in "${SEEDS[@]}"; do
  echo "  -- seed $SEED"
  (
    cd "$ROOT_DIR/selfGNN-Feature"
    python train.py \
      --data "$DATASET" \
      --device "$DEVICE" \
      --epoch "$EPOCH" \
      --seed "$SEED" \
      --use_node_features \
      --use_edge_features \
      --node_mlp_hidden 128 \
      --save_path "finance_merchant_node_edge_seed${SEED}"
  )
done

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RESULTS SUMMARY  —  $DATASET  (mean +/- std)"
echo "============================================================"

for TAG in finance_merchant_base finance_merchant_node finance_merchant_edge finance_merchant_node_edge; do
  show_results "$TAG"
done

echo ""
echo "All runs complete for $DATASET."
