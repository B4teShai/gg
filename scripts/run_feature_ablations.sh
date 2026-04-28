#!/usr/bin/env bash
set -euo pipefail

DATASET="yelp-merchant"
DEVICE="cuda"
EPOCH=150
SEEDS=(42)
GROUPS=(value time category repeat degree all all_plus_degree)
RUN_NODE_EDGE=0
KEEP_NODE_VALUE_WITH_EDGES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATASET="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --seeds) read -ra SEEDS <<< "$2"; shift 2 ;;
    --groups) read -ra GROUPS <<< "$2"; shift 2 ;;
    --run-node-edge) RUN_NODE_EDGE=1; shift ;;
    --keep-node-value-with-edges) KEEP_NODE_VALUE_WITH_EDGES=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PREFIX="${DATASET//-/_}"

clear_cache() {
  python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
  sleep 2
}

echo "============================================================"
echo "  Feature ablations"
echo "  Dataset : $DATASET"
echo "  Groups  : ${GROUPS[*]}"
echo "  Seeds   : ${SEEDS[*]}"
echo "============================================================"

for GROUP in "${GROUPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> node-only group=$GROUP seed=$SEED"
    (
      cd "$ROOT_DIR/selfGNN-Feature"
      python train.py \
        --data "$DATASET" --device "$DEVICE" --epoch "$EPOCH" --seed "$SEED" \
        --use_node_features --node_feature_groups "$GROUP" \
        --save_path "${PREFIX}_node_${GROUP}_seed${SEED}"
    )
    clear_cache

    if [[ "$RUN_NODE_EDGE" -eq 1 ]]; then
      EXTRA=()
      if [[ "$KEEP_NODE_VALUE_WITH_EDGES" -eq 1 ]]; then
        EXTRA=(--keep_node_value_with_edges)
      fi
      echo ""
      echo ">>> node+edge group=$GROUP seed=$SEED"
      (
        cd "$ROOT_DIR/selfGNN-Feature"
        python train.py \
          --data "$DATASET" --device "$DEVICE" --epoch "$EPOCH" --seed "$SEED" \
          --use_node_features --use_edge_features --node_feature_groups "$GROUP" \
          "${EXTRA[@]}" \
          --save_path "${PREFIX}_node_edge_${GROUP}_seed${SEED}"
      )
      clear_cache
    fi
  done
done

echo ""
echo "Ablation runs complete."
