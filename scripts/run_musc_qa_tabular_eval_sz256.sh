#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
VARIANT_PATH="${1:-experiments/variants/musc_qa_tabular_eval_sz256.yaml}"
CANDIDATE_VARIANT="$(basename "${VARIANT_PATH}" .yaml)"
REPORT_DIR="${ROOT_DIR}/data/paper_runs/reports/${CANDIDATE_VARIANT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing virtualenv python: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

for fold in 1 2; do
  "${PYTHON_BIN}" scripts/run_ablation.py --config "${VARIANT_PATH}" --fold "${fold}"
done

"${PYTHON_BIN}" scripts/compare_qa_variants.py \
  --dataset BF-C2DL-MuSC \
  --crop-tag sz256 \
  --baseline-variant musc_crop256 \
  --candidate-variant "${CANDIDATE_VARIANT}" \
  --output-markdown "${REPORT_DIR}/qa_variant_comparison.md" \
  --output-csv "${REPORT_DIR}/qa_variant_comparison.csv"
