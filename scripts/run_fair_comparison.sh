#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# Fair Comparison: CALT-US through the crop-reconstruct evaluation path
#
# Evaluates CALT-US through the SAME crop→reconstruct→full-image pipeline
# used by the ensemble, so IoU numbers are directly comparable.
#
# For each split (fold-1, fold-2, mixed):
#   1. Filter QA parquet to CALT-US rows only
#   2. Run evaluate-fusion-crops with stacked_path (channel 1 = competitor mask)
#   3. Print results
#
# Usage:
#   source .venv/bin/activate.fish  (or .sh for bash)
#   bash scripts/run_fair_comparison.sh
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PROJECT_ROOT}/data/mlflow/mlruns}"

DATASET="BF-C2DL-HSC"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="data/paper_runs/fair_comparison_${TIMESTAMP}"
mkdir -p "$OUT_BASE"

echo "================================================================"
echo "Fair Comparison: CALT-US crop-reconstructed — $(date)"
echo "Output: $OUT_BASE"
echo "================================================================"

# Step 0: Create a Python helper to filter parquet by competitor
cat > /tmp/filter_competitor.py << 'PYEOF'
import pandas as pd
import sys

parquet_in = sys.argv[1]
competitor = sys.argv[2]
parquet_out = sys.argv[3]

df = pd.read_parquet(parquet_in)
filtered = df[df["competitor"] == competitor].copy()
print(f"  Filtered {len(df)} rows → {len(filtered)} rows (competitor={competitor})")
filtered.to_parquet(parquet_out, index=False)
PYEOF

for SPLIT in fold-1 fold-2 mixed; do
    QA_PARQUET="data/dataframes/${DATASET}/qa_crops/${SPLIT}_sz64_qa_dataset.parquet"

    if [ ! -f "$QA_PARQUET" ]; then
        echo "[${SPLIT}] ⚠ QA parquet not found: ${QA_PARQUET} — skipping"
        continue
    fi

    FOLD_OUT="${OUT_BASE}/${SPLIT}"
    mkdir -p "${FOLD_OUT}"

    echo ""
    echo "--- ${SPLIT} ---"

    # Filter to CALT-US
    CALT_PQ="${FOLD_OUT}/calt_us_only.parquet"
    echo "[${SPLIT}] Filtering to CALT-US..."
    .venv/bin/python /tmp/filter_competitor.py "$QA_PARQUET" "CALT-US" "$CALT_PQ"

    # Run crop-reconstruct evaluation
    echo "[${SPLIT}] Running crop-reconstruct evaluation..."
    silver-evaluation evaluate-fusion-crops \
        "$CALT_PQ" \
        --fused-path-column stacked_path \
        --output-dir "${FOLD_OUT}/reconstructed" \
        --output "${FOLD_OUT}/fullimage_eval.csv" \
        --mlflow-experiment "fair-compare-${DATASET}-${TIMESTAMP}" \
        --mlflow-run-name "CALT-US_crop_recon_${SPLIT}" \
        --setup-name "CALT-US_crop_recon" \
        --pipeline-family "competitor" \
        --qa-mode "none"

    echo "[${SPLIT}] ✓ Done"
done

echo ""
echo "================================================================"
echo "All splits complete. Summary:"
echo "================================================================"

.venv/bin/python - "${OUT_BASE}" <<'PYEOF'
import pandas as pd
import os, sys

base = sys.argv[1]
print("\n" + "=" * 70)
print("FAIR COMPARISON: CALT-US via crop-reconstruct (test split)")
print("=" * 70)

results = {}
for split in ["fold-1", "fold-2", "mixed"]:
    csv = f"{base}/{split}/fullimage_eval.csv"
    if not os.path.exists(csv):
        continue
    df = pd.read_csv(csv)
    test = df[df["split"] == "test"]
    if len(test) == 0:
        test = df  # mixed might not have "test" split in the same way
    results[split] = test["iou"].mean()
    print(f"  {split:10s}  CALT-US crop-recon IoU = {test['iou'].mean():.4f}  F1 = {test['f1'].mean():.4f}  n={len(test)}")

# Print comparison with known ensemble numbers
print("\n  Compare with ensemble results from run_fold_comparison.sh:")
print("  (ensemble IoU: fold-1=0.8912, fold-2=0.8787)")
print("  (old whole-image CALT-US: fold-1=0.8526, fold-2=0.8953)")
print("=" * 70)
PYEOF
