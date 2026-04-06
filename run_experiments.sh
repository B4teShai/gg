#!/bin/bash
# End-to-end SelfGNN experiment pipeline.
# Usage: bash run_experiments.sh [--device cuda|cpu] [--epoch N]
#
# Steps:
#   1. Preprocess raw Yelp JSON → Datasets/yelp-merchant/
#   2. Extract edge/node features  → Datasets/yelp-merchant/ (same dir)
#   3. Train C1 baseline
#   4. Train C2 (edge features), C3 (node features), C4 (both)
#   5. Run comparison analysis and generate paper tables
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEVICE="cuda"
EPOCH=150
PYTHON="/venv/main/bin/python3"

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --epoch)  EPOCH="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo " SelfGNN Experiment Pipeline"
echo " device=$DEVICE  epochs=$EPOCH"
echo "============================================================"

# ------------------------------------------------------------------ #
# Step 1: Preprocess raw Yelp dataset                                  #
# ------------------------------------------------------------------ #
echo ""
echo "=== [1/5] Preprocessing Yelp dataset ==="
if [ -f "$ROOT/Datasets/yelp-merchant/trn_mat_time" ]; then
    echo "  Preprocessed data already exists — skipping."
else
    cd "$ROOT"
    jupyter nbconvert --to notebook --execute preprocess_to_yelp_merchant.ipynb \
        --output preprocess_to_yelp_merchant.ipynb \
        --ExecutePreprocessor.timeout=3600
    echo "  Preprocessing complete."
fi

# Verify required files
for f in trn_mat_time sequence tst_int test_dict; do
    if [ ! -f "$ROOT/Datasets/yelp-merchant/$f" ]; then
        echo "ERROR: Datasets/yelp-merchant/$f not found after preprocessing."
        exit 1
    fi
done

# ------------------------------------------------------------------ #
# Step 2: Extract features                                             #
# ------------------------------------------------------------------ #
echo ""
echo "=== [2/5] Extracting edge/node features ==="
if [ -f "$ROOT/Datasets/yelp-merchant/user_features.npy" ] && \
   [ -f "$ROOT/Datasets/yelp-merchant/merchant_features.npy" ] && \
   [ -f "$ROOT/Datasets/yelp-merchant/edge_weights.pkl" ]; then
    echo "  Feature files already exist — skipping."
else
    cd "$ROOT"
    $PYTHON selfGNN-Feature/feature_extractor.py
    echo "  Feature extraction complete."
fi

# ------------------------------------------------------------------ #
# Step 3: Train C1 — baseline (binary adjacency, no features)         #
# ------------------------------------------------------------------ #
#echo ""
#echo "=== [3/5] Training C1 (baseline) ==="
#mkdir -p "$ROOT/Results"
#cd "$ROOT/selfGNN-Base"
#$PYTHON train.py \
#    --data yelp-merchant \
#    --save_path yelp_merchant_baseline \
#    --epoch "$EPOCH" --tstEpoch 3 --patience 20 \
#   --device "$DEVICE" \
#   2>&1 | tee "$ROOT/Results/c1_baseline.log"

# ------------------------------------------------------------------ #
# Step 4: Train C2 / C3 / C4                                          #
# ------------------------------------------------------------------ #
echo ""
echo "=== [4/5] Training featured configs ==="
cd "$ROOT/selfGNN-Feature"
COMMON="--data yelp-merchant --epoch $EPOCH --tstEpoch 3 --patience 20 --device $DEVICE"

echo "--- C2: edge features only ---"
$PYTHON train.py $COMMON \
    --use_edge_features \
    --save_path yelp_merchant_edge_feature \
    2>&1 | tee "$ROOT/Results/c2_edge.log"

echo "--- C3: node features only ---"
$PYTHON train.py $COMMON \
    --use_node_features \
    --save_path yelp_merchant_node_feature \
    2>&1 | tee "$ROOT/Results/c3_node.log"

echo "--- C4: edge + node features ---"
$PYTHON train.py $COMMON \
    --use_edge_features --use_node_features \
    --save_path yelp_merchant_edge_node_feature \
    2>&1 | tee "$ROOT/Results/c4_both.log"

# ------------------------------------------------------------------ #
# Step 5: Analysis                                                      #
# ------------------------------------------------------------------ #
echo ""
echo "=== [5/5] Analysis & paper tables ==="
cd "$ROOT"
$PYTHON analysis/compare_results.py
$PYTHON analysis/generate_paper_tables.py

echo ""
echo "============================================================"
echo " Done."
echo " Results:       Results/*.json"
echo " Figures:       paper_figures/*.pdf"
echo " LaTeX tables:  paper_figures/table_*.tex"
echo "============================================================"
