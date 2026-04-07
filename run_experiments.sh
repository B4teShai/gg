#!/bin/bash
# End-to-end SelfGNN experiment pipeline.
# Usage: bash run_experiments.sh [--device cuda|cpu|mps] [--epoch N] [--seeds "48 2048 100"]
#
# Steps:
#   1. Preprocess raw Yelp JSON → Datasets/yelp-merchant/                    (skip if already done)
#   2. Extract edge/node features → Datasets/yelp-merchant/                  (skip if already done)
#   3. Train 5 configurations × N seeds:
#        T1 t1_base    : baseline (binary adjacency, no features)
#        T2 t2_edge    : edge weights only
#        T3 t3_node    : node features only
#        T4 t4_dup     : node + edge features, star kept in nodes (DUPLICATED)
#        T4 t4_nodup   : node + edge features, star zeroed in nodes (NOT DUPLICATED)
#   4. Run multi-seed analysis and generate paper tables / figures
#
# Idempotent: each run is skipped if Results/{tag}.json already exists, so the
# script can resume after interruption. To force a re-run, delete the JSON.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEVICE="cuda"
EPOCH=150
PYTHON="/venv/main/bin/python3"
SEEDS=(48 2048 100)

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --epoch)  EPOCH="$2";  shift 2 ;;
        --seeds)  read -r -a SEEDS <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$ROOT/Results"

echo "============================================================"
echo " SelfGNN Experiment Pipeline"
echo " device=$DEVICE  epochs=$EPOCH  seeds=(${SEEDS[*]})"
echo " configs=5 (t1_base, t2_edge, t3_node, t4_dup, t4_nodup)"
echo " total runs=$(( ${#SEEDS[@]} * 5 ))"
echo "============================================================"

# ------------------------------------------------------------------ #
# Step 1: Preprocess raw Yelp dataset                                  #
# ------------------------------------------------------------------ #
echo ""
echo "=== [1/4] Preprocessing Yelp dataset ==="
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
echo "=== [2/4] Extracting edge/node features ==="
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
# Step 3: Train all configs × all seeds                                #
# ------------------------------------------------------------------ #
echo ""
echo "=== [3/4] Training 5 configs × ${#SEEDS[@]} seeds ==="

COMMON="--data yelp-merchant --epoch $EPOCH --tstEpoch 3 --patience 20 --device $DEVICE"

run_if_missing () {
    # $1 = tag, $2 = python invocation (relative to current cwd)
    local TAG="$1"
    local CMD="$2"
    local JSON="$ROOT/Results/${TAG}.json"
    local LOG="$ROOT/Results/${TAG}.log"
    if [ -f "$JSON" ]; then
        echo "  [skip] $TAG (Results/${TAG}.json already exists)"
    else
        echo "  [run]  $TAG"
        eval "$CMD" 2>&1 | tee "$LOG"
    fi
}

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "###### SEED=$SEED ######"

    # T1 baseline (selfGNN-Base)
    cd "$ROOT/selfGNN-Base"
    run_if_missing "t1_base_s${SEED}" \
        "$PYTHON train.py $COMMON --seed $SEED --save_path t1_base_s${SEED}"

    # T2 / T3 / T4-dup / T4-nodup (selfGNN-Feature)
    cd "$ROOT/selfGNN-Feature"

    run_if_missing "t2_edge_s${SEED}" \
        "$PYTHON train.py $COMMON --seed $SEED --use_edge_features --save_path t2_edge_s${SEED}"

    run_if_missing "t3_node_s${SEED}" \
        "$PYTHON train.py $COMMON --seed $SEED --use_node_features --save_path t3_node_s${SEED}"

    run_if_missing "t4_dup_s${SEED}" \
        "$PYTHON train.py $COMMON --seed $SEED --use_edge_features --use_node_features --keep_duplicate_star --save_path t4_dup_s${SEED}"

    run_if_missing "t4_nodup_s${SEED}" \
        "$PYTHON train.py $COMMON --seed $SEED --use_edge_features --use_node_features --save_path t4_nodup_s${SEED}"
done

# ------------------------------------------------------------------ #
# Step 4: Analysis & paper tables                                      #
# ------------------------------------------------------------------ #
echo ""
echo "=== [4/4] Analysis & paper tables ==="
cd "$ROOT"
$PYTHON analysis/compare_results.py
$PYTHON analysis/generate_paper_tables.py

echo ""
echo "============================================================"
echo " Done."
echo " Per-run results: Results/{tag}_s{seed}.json + .log"
echo " Aggregated:      Results/comparison_summary.json"
echo " Paper figures:   paper_figures/*.pdf"
echo " Convergence fig: paper-tex/fig_convergence.png"
echo " LaTeX tables:    paper_figures/table_*.tex"
echo "============================================================"
