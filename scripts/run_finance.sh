#!/usr/bin/env bash
# run_finance.sh — Train all model variants on finance-merchant and print results.
#
# Usage:
#   bash run_finance.sh [--device cuda|mps|cpu] [--epoch N] [--seed N]

set -euo pipefail

DATASET="finance-merchant"
DEVICE="cuda"
EPOCH=150
SEED=100

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --epoch)  EPOCH="$2";  shift 2 ;;
    --seed)   SEED="$2";   shift 2 ;;
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
echo "  Seed    : $SEED"
echo "============================================================"

# ── 1. SelfGNN-Base (no features) ───────────────────────────────
echo ""
echo ">>> [1/4] selfGNN-Base on $DATASET"
echo "------------------------------------------------------------"
(
  cd "$ROOT_DIR/selfGNN-Base"
  python train.py \
    --data "$DATASET" \
    --device "$DEVICE" \
    --epoch "$EPOCH" \
    --seed "$SEED" \
    --save_path "finance_merchant_base"
)
echo "---- done → $RESULTS_DIR/finance_merchant_base.json"

# ── 2. SelfGNN-Feature (node features only) ─────────────────────
echo ""
echo ">>> [2/4] selfGNN-Feature (node only) on $DATASET"
echo "------------------------------------------------------------"
(
  cd "$ROOT_DIR/selfGNN-Feature"
  python train.py \
    --data "$DATASET" \
    --device "$DEVICE" \
    --epoch "$EPOCH" \
    --seed "$SEED" \
    --use_node_features \
    --save_path "finance_merchant_node"
)
echo "---- done → $RESULTS_DIR/finance_merchant_node.json"

# ── 3. SelfGNN-Feature (edge features only) ─────────────────────
echo ""
echo ">>> [3/4] selfGNN-Feature (edge only) on $DATASET"
echo "------------------------------------------------------------"
(
  cd "$ROOT_DIR/selfGNN-Feature"
  python train.py \
    --data "$DATASET" \
    --device "$DEVICE" \
    --epoch "$EPOCH" \
    --seed "$SEED" \
    --use_edge_features \
    --save_path "finance_merchant_edge"
)
echo "---- done → $RESULTS_DIR/finance_merchant_edge.json"

# ── 4. SelfGNN-Feature (node + edge features) ───────────────────
echo ""
echo ">>> [4/4] selfGNN-Feature (node + edge) on $DATASET"
echo "------------------------------------------------------------"
(
  cd "$ROOT_DIR/selfGNN-Feature"
  python train.py \
    --data "$DATASET" \
    --device "$DEVICE" \
    --epoch "$EPOCH" \
    --seed "$SEED" \
    --use_node_features \
    --use_edge_features \
    --save_path "finance_merchant_node_edge"
)
echo "---- done → $RESULTS_DIR/finance_merchant_node_edge.json"

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RESULTS SUMMARY  —  $DATASET"
echo "============================================================"

for TAG in finance_merchant_base finance_merchant_node finance_merchant_edge finance_merchant_node_edge; do
  F="$RESULTS_DIR/$TAG.json"
  if [ -f "$F" ]; then
    echo ""
    echo "▸ $TAG"
    python3 - "$F" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
best = d.get("best_epoch", "?")
tr = d.get("test_results", {})
vr = d.get("val_results", {})
print(f"  Best epoch : {best}")
if tr:
    print(f"  Test  HR@10={tr.get('HR@10',0):.4f}  NDCG@10={tr.get('NDCG@10',0):.4f}"
          f"  HR@20={tr.get('HR@20',0):.4f}  NDCG@20={tr.get('NDCG@20',0):.4f}")
if vr:
    print(f"  Val   HR@10={vr.get('HR@10',0):.4f}  NDCG@10={vr.get('NDCG@10',0):.4f}"
          f"  HR@20={vr.get('HR@20',0):.4f}  NDCG@20={vr.get('NDCG@20',0):.4f}")
EOF
  else
    echo "  $TAG: result file not found ($F)"
  fi
done

echo ""
echo "All runs complete for $DATASET."
