#!/bin/bash -l
#SBATCH --job-name=qa_compare
#SBATCH --account=eu-25-40
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
DURABLE_ROOT="/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc"
DATASET="BF-C2DL-HSC"
CROP_TAG="sz64"
BASELINE_VARIANT="baseline"
CANDIDATE_VARIANT="hsc_qa_rank_eval"
THRESHOLDS="0.50,0.60,0.70,0.75"
FOLDS="fold-1,fold-2"

usage() {
    cat <<'EOF'
Usage:
  sbatch scripts/compare_qa_variants_hpc.sh [options]

Options:
  --durable-root PATH        Durable storage root. Default: /mnt/proj1/eu-25-40/innovaite/silver-truth-hpc
  --dataset NAME             Dataset name. Default: BF-C2DL-HSC
  --crop-tag TAG             Crop tag. Default: sz64
  --baseline-variant NAME    Baseline variant. Default: baseline
  --candidate-variant NAME   Candidate variant. Default: hsc_qa_rank_eval
  --thresholds CSV           Threshold list. Default: 0.50,0.60,0.70,0.75
  --folds CSV                Fold list. Default: fold-1,fold-2
  --help                     Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --durable-root)
            DURABLE_ROOT="${2:-}"
            shift 2
            ;;
        --dataset)
            DATASET="${2:-}"
            shift 2
            ;;
        --crop-tag)
            CROP_TAG="${2:-}"
            shift 2
            ;;
        --baseline-variant)
            BASELINE_VARIANT="${2:-}"
            shift 2
            ;;
        --candidate-variant)
            CANDIDATE_VARIANT="${2:-}"
            shift 2
            ;;
        --thresholds)
            THRESHOLDS="${2:-}"
            shift 2
            ;;
        --folds)
            FOLDS="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Missing virtualenv python: ${PYTHON_BIN}" >&2
    exit 1
fi

command -v ml >/dev/null 2>&1 && ml purge >/dev/null 2>&1 || true
ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0
ml libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

source "${PYTHON_BIN%/python}/activate"

REPORT_DIR="${DURABLE_ROOT}/paper_runs/reports/${CANDIDATE_VARIANT}"
mkdir -p "${REPORT_DIR}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/compare_qa_variants.py \
    --paper-runs-root "${DURABLE_ROOT}/paper_runs" \
    --dataset "${DATASET}" \
    --crop-tag "${CROP_TAG}" \
    --baseline-variant "${BASELINE_VARIANT}" \
    --candidate-variant "${CANDIDATE_VARIANT}" \
    --folds "${FOLDS}" \
    --thresholds "${THRESHOLDS}" \
    --output-markdown "${REPORT_DIR}/qa_variant_comparison.md" \
    --output-csv "${REPORT_DIR}/qa_variant_comparison.csv"

echo "Comparison report written to: ${REPORT_DIR}"
