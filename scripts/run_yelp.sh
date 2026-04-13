#!/usr/bin/env bash
# run_yelp.sh — Train all model variants on yelp-merchant and print results.
#
# Usage:
#   bash run_yelp.sh [--device cuda|mps|cpu] [--epoch N] [--seed N]
#

set -euo pipefail

DATASET="yelp-merchant"
DEVICE="cuda"
EPOCH=150
SEED=100

# Parse optional overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --epoch)  EPOCH="$2";  shift 2 ;;
    --seed)   SEED="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/Results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  Dataset : $DATASET"
echo "  Device  : $DEVICE"
echo "  Epochs  : $EPOCH"
echo "  Seed    : $SEED"
echo "============================================================"

# ── 1. SelfGNN-Base ─────────────────────────────────────────────
echo ""
echo ">>> [1/3] selfGNN-Base on $DATASET"
echo "------------------------------------------------------------"
(
  cd "$SCRIPT_DIR/selfGNN-Base"
  python train.py \
    --data "$DATASET" \
    --device "$DEVICE" \
    --epoch "$EPOCH" \
    --seed "$SEED" \
    --save_path "yelp_merchant_base"
)
echo "---- selfGNN-Base done. Result: $RESULTS_DIR/yelp_merchant_base.json"

# ── 2. SelfGNN-Feature  (edge features only) ────────────────────
echo ""
echo ">>> [2/3] selfGNN-Feature (edge features) on $DATASET"
echo "------------------------------------------------------------"
(
  cd "$SCRIPT_DIR/selfGNN-Feature"
  python train.py \
    --data "$DATASET" \
    --device "$DEVICE" \
    --epoch "$EPOCH" \
    --seed "$SEED" \
    --use_edge_features \
    --save_path "yelp_merchant_edge"
)
echo "---- selfGNN-Feature (edge) done. Result: $RESULTS_DIR/yelp_merchant_edge.json"

# ── 3. SelfGNN-Feature  (edge + node features) ──────────────────
echo ""
echo ">>> [3/3] selfGNN-Feature (edge + node features) on $DATASET"
echo "------------------------------------------------------------"
(
  cd "$SCRIPT_DIR/selfGNN-Feature"
  python train.py \
    --data "$DATASET" \
    --device "$DEVICE" \
    --epoch "$EPOCH" \
    --seed "$SEED" \
    --use_edge_features \
    --use_node_features \
    --save_path "yelp_merchant_full"
)
echo "---- selfGNN-Feature (full) done. Result: $RESULTS_DIR/yelp_merchant_full.json"

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  RESULTS SUMMARY  —  $DATASET"
echo "============================================================"

for TAG in yelp_merchant_base yelp_merchant_edge yelp_merchant_full; do
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
