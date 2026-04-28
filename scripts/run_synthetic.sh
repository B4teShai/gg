#!/usr/bin/env bash
# run_synthetic.sh — Full final-submission sweep on synthetic-merchant.
#
# Phase A : 4 SelfGNN variants (base | node | edge | node+edge) → Results2/
# Phase B : 5 baselines (popularity, bprmf, lightgcn, sasrec, bert4rec)
#           → Results_baselines/
#
# Single seed (42) by default, matching the "full sweep, run once" budget.
# GRAPH_NUM is now 8 across all datasets (matches the new 2-year preprocessing).

set -euo pipefail

DATASET="synthetic-merchant"
DEVICE="cuda"
EPOCH=150
SEEDS=(42)
BASELINE_EPOCHS=100
PATIENCE=10
SKIP_BASELINES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)           DEVICE="$2";            shift 2 ;;
    --epoch)            EPOCH="$2";             shift 2 ;;
    --seeds)            read -ra SEEDS <<< "$2"; shift 2 ;;
    --baseline-epochs)  BASELINE_EPOCHS="$2";   shift 2 ;;
    --patience)         PATIENCE="$2";          shift 2 ;;
    --skip-baselines)   SKIP_BASELINES=1;       shift   ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$ROOT_DIR/Results2"
BASELINE_DIR="$ROOT_DIR/Results_baselines"
mkdir -p "$RESULTS_DIR" "$BASELINE_DIR"

echo "============================================================"
echo "  Dataset : $DATASET"
echo "  Device  : $DEVICE"
echo "  SelfGNN epochs : $EPOCH"
echo "  Baseline epochs: $BASELINE_EPOCHS (patience $PATIENCE)"
echo "  Seeds   : ${SEEDS[*]}"
echo "============================================================"

show_results() {
  local DIR="$1"
  local TAG="$2"
  echo ""
  echo "* $TAG"
  python3 - "$DIR" "$TAG" "${SEEDS[@]}" <<'EOF'
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

# ================================================================
# PHASE A — SelfGNN variants
# ================================================================

echo ""
echo ">>> [A1/4] selfGNN-Base on $DATASET"
echo "------------------------------------------------------------"
for SEED in "${SEEDS[@]}"; do
  (
    cd "$ROOT_DIR/selfGNN-Base"
    python train.py \
      --data "$DATASET" --device "$DEVICE" --epoch "$EPOCH" --seed "$SEED" \
      --save_path "synthetic_merchant_base_seed${SEED}"
  )
done

echo ""
echo ">>> [A2/4] selfGNN-Feature (node only) on $DATASET"
echo "------------------------------------------------------------"
for SEED in "${SEEDS[@]}"; do
  (
    cd "$ROOT_DIR/selfGNN-Feature"
    python train.py \
      --data "$DATASET" --device "$DEVICE" --epoch "$EPOCH" --seed "$SEED" \
      --use_node_features --node_mlp_hidden 128 \
      --save_path "synthetic_merchant_node_seed${SEED}"
  )
done

echo ""
echo ">>> [A3/4] selfGNN-Feature (edge only) on $DATASET"
echo "------------------------------------------------------------"
for SEED in "${SEEDS[@]}"; do
  (
    cd "$ROOT_DIR/selfGNN-Feature"
    python train.py \
      --data "$DATASET" --device "$DEVICE" --epoch "$EPOCH" --seed "$SEED" \
      --use_edge_features \
      --save_path "synthetic_merchant_edge_seed${SEED}"
  )
done

echo ""
echo ">>> [A4/4] selfGNN-Feature (node + edge) on $DATASET"
echo "------------------------------------------------------------"
for SEED in "${SEEDS[@]}"; do
  (
    cd "$ROOT_DIR/selfGNN-Feature"
    python train.py \
      --data "$DATASET" --device "$DEVICE" --epoch "$EPOCH" --seed "$SEED" \
      --use_node_features --use_edge_features --node_mlp_hidden 128 \
      --save_path "synthetic_merchant_node_edge_seed${SEED}"
  )
done

# ================================================================
# PHASE B — Baselines
# ================================================================

if [[ "$SKIP_BASELINES" -eq 0 ]]; then
  for MODEL in popularity bprmf lightgcn sasrec bert4rec; do
    echo ""
    echo ">>> [B:$MODEL] baseline on $DATASET"
    echo "------------------------------------------------------------"
    for SEED in "${SEEDS[@]}"; do
      (
        cd "$ROOT_DIR/baselines"
        python train_baseline.py \
          --model "$MODEL" --data "$DATASET" --device "$DEVICE" \
          --seed "$SEED" --epochs "$BASELINE_EPOCHS" --patience "$PATIENCE" \
          --save-path "synthetic_merchant_${MODEL}_seed${SEED}"
      )
    done
  done
fi

# ================================================================
# Summary
# ================================================================

echo ""
echo "============================================================"
echo "  RESULTS SUMMARY  —  $DATASET  (mean +/- std)"
echo "============================================================"

echo ""
echo "--- SelfGNN variants (Results2/) ---"
for TAG in synthetic_merchant_base synthetic_merchant_node synthetic_merchant_edge synthetic_merchant_node_edge; do
  show_results "$RESULTS_DIR" "$TAG"
done

if [[ "$SKIP_BASELINES" -eq 0 ]]; then
  echo ""
  echo "--- Baselines (Results_baselines/) ---"
  for MODEL in popularity bprmf lightgcn sasrec bert4rec; do
    show_results "$BASELINE_DIR" "synthetic_merchant_${MODEL}"
  done
fi

echo ""
echo "All runs complete for $DATASET."
