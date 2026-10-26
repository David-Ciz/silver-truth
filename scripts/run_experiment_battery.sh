#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# Experiment Battery: run multiple ensemble configurations on all folds
#
# Tests these combinations:
#   1. baseline       — C1, basic augmentation (current default)
#   2. strong_aug     — C1, strong augmentation
#   3. c2_raw         — C2 (with raw image), basic augmentation
#   4. c2_strong      — C2 (with raw image), strong augmentation
#
# For each configuration, trains + evaluates on fold-1 AND fold-2.
# Results are collected into a single timestamped directory and compiled
# into a summary table at the end.
#
# Usage:
#   source .venv/bin/activate.fish
#   bash scripts/run_experiment_battery.sh
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PROJECT_ROOT}/data/mlflow/mlruns}"

DATASET="BF-C2DL-HSC"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="data/paper_runs/experiment_battery_${TIMESTAMP}"
mkdir -p "$OUT_BASE"

# Log file captures all output
LOG="${OUT_BASE}/run.log"

echo "================================================================" | tee "$LOG"
echo "Experiment Battery — $(date)"                                      | tee -a "$LOG"
echo "Output: $OUT_BASE"                                                  | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

# ── Define experiment configurations ─────────────────────────────────
# Format: NAME|DATASET_VERSION|AUGMENTATION|EXTRA_FLAGS
CONFIGS=(
    "baseline|C1|basic|"
    "strong_aug|C1|strong|"
    "c2_raw|C2|basic|"
    "c2_strong|C2|strong|"
)

# ── Run experiments ──────────────────────────────────────────────────
for CONFIG in "${CONFIGS[@]}"; do
    IFS='|' read -r EXP_NAME DS_VERSION AUG EXTRA <<< "$CONFIG"

    for FOLD in 1 2; do
        SPLIT="fold-${FOLD}"
        QA_PARQUET="data/dataframes/${DATASET}/qa_crops/${SPLIT}_sz64_qa_dataset.parquet"
        EXP_DIR="${OUT_BASE}/${EXP_NAME}/${SPLIT}"
        DB_DIR="${EXP_DIR}/databank"
        CKPT_DIR="${EXP_DIR}/checkpoints"

        mkdir -p "$DB_DIR" "$CKPT_DIR"

        echo "" | tee -a "$LOG"
        echo "──────────────────────────────────────────────────────" | tee -a "$LOG"
        echo "  ${EXP_NAME} / ${SPLIT} (version=${DS_VERSION}, aug=${AUG})" | tee -a "$LOG"
        echo "──────────────────────────────────────────────────────" | tee -a "$LOG"

        # Build databank
        echo "[${EXP_NAME}/${SPLIT}] Building databank..." | tee -a "$LOG"
        silver-ensemble build-databank \
            --dataset-name "${DATASET}" \
            --qa-parquet-path "${QA_PARQUET}" \
            --version "${DS_VERSION}" \
            --output-dir "${DB_DIR}" 2>&1 | tee -a "$LOG"

        DB_PARQUET=$(find "${DB_DIR}" -name "*.parquet" | head -1)
        if [ -z "$DB_PARQUET" ]; then
            echo "[${EXP_NAME}/${SPLIT}] ERROR: No databank parquet found!" | tee -a "$LOG"
            continue
        fi

        # Train ensemble
        echo "[${EXP_NAME}/${SPLIT}] Training (${DS_VERSION}, ${AUG})..." | tee -a "$LOG"
        silver-ensemble ensemble-experiment \
            --name "battery-${EXP_NAME}-${DATASET}-${SPLIT}-${TIMESTAMP}" \
            --parquet-file "${DB_PARQUET}" \
            --model-type UnetPlusPlus \
            --max-epochs 100 \
            --checkpoints-dir "${CKPT_DIR}" \
            --dataset-version "${DS_VERSION}" \
            --augmentation "${AUG}" 2>&1 | tee -a "$LOG"

        # Evaluate ensemble
        echo "[${EXP_NAME}/${SPLIT}] Evaluating..." | tee -a "$LOG"
        silver-ensemble evaluate-best-checkpoint \
            --checkpoints-dir "${CKPT_DIR}" \
            --databank-path "${DB_PARQUET}" \
            --split-type test \
            --dataset-version "${DS_VERSION}" \
            --output-dir "${CKPT_DIR}" \
            --mlflow-experiment "battery-compare-${DATASET}-${TIMESTAMP}" \
            --mlflow-run-name "${EXP_NAME}_${SPLIT}" \
            --setup-name "${EXP_NAME}" 2>&1 | tee -a "$LOG"

        echo "[${EXP_NAME}/${SPLIT}] ✓ Done" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
echo "All experiments complete. Compiling results..."                    | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

# ── Compile results ──────────────────────────────────────────────────
.venv/bin/python scripts/compile_results.py "${OUT_BASE}" 2>&1 | tee -a "$LOG"

echo ""
echo "Full log: ${LOG}"
echo "Results:  ${OUT_BASE}/results_summary.md"
