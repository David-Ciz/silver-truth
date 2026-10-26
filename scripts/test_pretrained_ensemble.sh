#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# Quick A/B test: pretrained vs scratch ensemble encoder on fold-1
#
# Uses the EXISTING fold-1 databank parquet — no prep needed.
# Results go to MLflow for easy comparison with past runs.
#
# Usage:
#   source .venv/bin/activate
#   export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/mlruns
#   bash scripts/test_pretrained_ensemble.sh
#
# After it finishes, compare in MLflow:
#   mlflow ui --backend-store-uri data/mlflow/mlruns
#   → Look at experiment "pretrained-ensemble-test-BF-C2DL-HSC-fold-1"
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DATABANK="data/paper_runs/ensemble/baseline/fold-1/databank_unfiltered/C1_ds1-42-7015_QA--.parquet"
CKPT_DIR="data/paper_runs/ensemble/pretrained_test/fold-1/checkpoints"
COMPARE_EXP="pretrained-ensemble-test-BF-C2DL-HSC-fold-1"

# Ensure MLflow URI is set
if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
    export MLFLOW_TRACKING_URI="file:$(pwd)/data/mlflow/mlruns"
fi
mkdir -p "$CKPT_DIR"

echo "============================================================"
echo "Training ensemble with PRETRAINED encoder (imagenet weights)"
echo "  databank: $DATABANK"
echo "  checkpoints: $CKPT_DIR"
echo "============================================================"

silver-ensemble ensemble-experiment \
    --name "$COMPARE_EXP" \
    --parquet-file "$DATABANK" \
    --model-type UnetPlusPlus \
    --max-epochs 100 \
    --checkpoints-dir "$CKPT_DIR" \
    --encoder-weights imagenet

echo ""
echo "============================================================"
echo "Evaluating pretrained ensemble checkpoint"
echo "============================================================"

silver-ensemble evaluate-best-checkpoint \
    --checkpoints-dir "$CKPT_DIR" \
    --databank-path "$DATABANK" \
    --split-type test \
    --output-dir "$CKPT_DIR" \
    --mlflow-experiment "$COMPARE_EXP" \
    --mlflow-run-name pretrained_encoder_eval \
    --setup-name pretrained_encoder

echo ""
echo "============================================================"
echo "Re-evaluating OLD scratch checkpoint for comparison"
echo "============================================================"

OLD_CKPT_DIR="data/paper_runs/ensemble/baseline/fold-1/checkpoints_baseline"
silver-ensemble evaluate-best-checkpoint \
    --checkpoints-dir "$OLD_CKPT_DIR" \
    --databank-path "$DATABANK" \
    --split-type test \
    --output-dir "$OLD_CKPT_DIR" \
    --mlflow-experiment "$COMPARE_EXP" \
    --mlflow-run-name scratch_encoder_eval \
    --setup-name scratch_encoder

echo ""
echo "============================================================"
echo "DONE. Compare results at:"
echo "  mlflow ui --backend-store-uri data/mlflow/mlruns"
echo "  → experiment: $COMPARE_EXP"
echo "============================================================"
