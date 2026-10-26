#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# Clean re-run: Ensemble vs Competitors on BOTH folds
#
# Steps per fold:
#   1. Evaluate competitor baselines (CALT-US etc.)
#   2. Evaluate silver truth reference
#   3. Build ensemble databank (if needed)
#   4. Train ensemble (UnetPlusPlus, 100 epochs)
#   5. Evaluate ensemble
#   6. Print summary
#
# Usage:
#   source .venv/bin/activate.fish  (or .sh for bash)
#   bash scripts/run_fold_comparison.sh
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure MLflow URI
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PROJECT_ROOT}/data/mlflow/mlruns}"

DATASET="BF-C2DL-HSC"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="data/paper_runs/clean_comparison_${TIMESTAMP}"
mkdir -p "$OUT_BASE"

echo "================================================================"
echo "Clean fold comparison — $(date)"
echo "Output: $OUT_BASE"
echo "================================================================"

for FOLD in 1 2; do
    SPLIT="fold-${FOLD}"
    WHOLE_IMG="data/dataframes/${DATASET}/whole_image/${DATASET}_split_${SPLIT}.parquet"
    QA_PARQUET="data/dataframes/${DATASET}/qa_crops/${SPLIT}_sz64_qa_dataset.parquet"
    FOLD_OUT="${OUT_BASE}/${SPLIT}"
    mkdir -p "${FOLD_OUT}"

    echo ""
    echo "================================================================"
    echo "  FOLD ${FOLD} (${SPLIT})"
    echo "================================================================"

    # ── 1. Competitor baselines ────────────────────────────────────────
    echo "[${SPLIT}] Step 1/5: Evaluating competitors..."
    silver-evaluation evaluate-competitor \
        "${WHOLE_IMG}" \
        --output "${FOLD_OUT}/competitors.csv" \
        --mlflow-experiment "clean-compare-${DATASET}-${SPLIT}-${TIMESTAMP}" \
        --mlflow-run-name "competitors"

    # ── 2. Silver truth reference ─────────────────────────────────────
    echo "[${SPLIT}] Step 2/5: Evaluating silver truth..."
    silver-evaluation evaluate-competitor \
        "${WHOLE_IMG}" \
        --competitor SILVER-TRUTH \
        --output "${FOLD_OUT}/silver_truth.csv" \
        --mlflow-experiment "clean-compare-${DATASET}-${SPLIT}-${TIMESTAMP}" \
        --mlflow-run-name "silver_truth"

    # ── 3. Build ensemble databank ────────────────────────────────────
    DB_DIR="${FOLD_OUT}/databank"
    DB_PARQUET="${DB_DIR}/C1_ds1-42-7015_QA--.parquet"
    echo "[${SPLIT}] Step 3/5: Building ensemble databank..."
    silver-ensemble build-databank \
        --dataset-name "${DATASET}" \
        --qa-parquet-path "${QA_PARQUET}" \
        --version C1 \
        --output-dir "${DB_DIR}"

    # ── 4. Train ensemble ─────────────────────────────────────────────
    CKPT_DIR="${FOLD_OUT}/checkpoints"
    echo "[${SPLIT}] Step 4/5: Training ensemble (UnetPlusPlus, 100 epochs)..."
    silver-ensemble ensemble-experiment \
        --name "clean-ensemble-${DATASET}-${SPLIT}-${TIMESTAMP}" \
        --parquet-file "${DB_PARQUET}" \
        --model-type UnetPlusPlus \
        --max-epochs 100 \
        --checkpoints-dir "${CKPT_DIR}"

    # ── 5. Evaluate ensemble ──────────────────────────────────────────
    echo "[${SPLIT}] Step 5/5: Evaluating ensemble..."
    silver-ensemble evaluate-best-checkpoint \
        --checkpoints-dir "${CKPT_DIR}" \
        --databank-path "${DB_PARQUET}" \
        --split-type test \
        --output-dir "${CKPT_DIR}" \
        --mlflow-experiment "clean-compare-${DATASET}-${SPLIT}-${TIMESTAMP}" \
        --mlflow-run-name "ensemble_only" \
        --setup-name "ensemble_only"

    echo "[${SPLIT}] ✓ Done"
done

echo ""
echo "================================================================"
echo "All folds complete. Now printing summary..."
echo "================================================================"

# ── Summary ────────────────────────────────────────────────────────────
.venv/bin/python - "${OUT_BASE}" <<'PYEOF'
import pandas as pd
import os, sys

base = sys.argv[1]
print("\n" + "=" * 70)
print("RESULTS SUMMARY — ENSEMBLE vs COMPETITORS (test split)")
print("=" * 70)

for fold in ["fold-1", "fold-2"]:
    fold_dir = f"{base}/{fold}"
    if not os.path.isdir(fold_dir):
        continue

    print(f"\n--- {fold.upper()} ---")

    # Competitors
    comp = pd.read_csv(f"{fold_dir}/competitors.csv")
    for name, grp in comp.groupby("competitor"):
        test_avg = grp["split_test_average"].iloc[0]
        test_f1 = grp["split_test_f1_average"].iloc[0]
        marker = " ← BEST COMPETITOR" if "CALT" in name else ""
        print(f"  {name:25s}  IoU = {test_avg:.4f}  F1 = {test_f1:.4f}{marker}")

    # Silver truth
    st = pd.read_csv(f"{fold_dir}/silver_truth.csv")
    for name, grp in st.groupby("competitor"):
        test_avg = grp["split_test_average"].iloc[0]
        test_f1 = grp["split_test_f1_average"].iloc[0]
        print(f"  {name:25s}  IoU = {test_avg:.4f}  F1 = {test_f1:.4f}  ← UPPER BOUND")

    # Ensemble
    ckpt_dir = f"{fold_dir}/checkpoints"
    for fn in sorted(os.listdir(ckpt_dir)):
        if fn.endswith(".parquet") and "cell" not in fn:
            df = pd.read_parquet(f"{ckpt_dir}/{fn}")
            if "iou" in df.columns:
                print(f"  {'ENSEMBLE (ours)':25s}  IoU = {df['iou'].mean():.4f}  F1 = {df['f1'].mean():.4f}  ★★★")

print("\n" + "=" * 70)
# Cross-fold average
fold_ious = {}
calt_ious = {}
for fold in ["fold-1", "fold-2"]:
    fold_dir = f"{base}/{fold}"
    if not os.path.isdir(fold_dir):
        continue
    # ensemble
    ckpt_dir = f"{fold_dir}/checkpoints"
    for fn in sorted(os.listdir(ckpt_dir)):
        if fn.endswith(".parquet") and "cell" not in fn:
            df = pd.read_parquet(f"{ckpt_dir}/{fn}")
            if "iou" in df.columns:
                fold_ious[fold] = df["iou"].mean()
    # CALT-US
    comp = pd.read_csv(f"{fold_dir}/competitors.csv")
    calt = comp[comp["competitor"].str.contains("CALT")]
    if len(calt) > 0:
        calt_ious[fold] = calt["split_test_average"].iloc[0]

if len(fold_ious) == 2 and len(calt_ious) == 2:
    ens_avg = sum(fold_ious.values()) / 2
    calt_avg = sum(calt_ious.values()) / 2
    delta = ens_avg - calt_avg
    print(f"CROSS-FOLD AVERAGE:")
    print(f"  CALT-US:         IoU = {calt_avg:.4f}")
    print(f"  ENSEMBLE (ours): IoU = {ens_avg:.4f}  (Δ = {delta:+.4f})")
    if delta > 0:
        print(f"  → ENSEMBLE BEATS CALT-US by {delta:.4f} on average! 🎉")
    else:
        print(f"  → CALT-US still leads by {abs(delta):.4f}")
print("=" * 70)
PYEOF
