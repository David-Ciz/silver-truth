#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

timestamp="$(date +%Y%m%d_%H%M%S)"
DEFAULT_CAMPAIGN_ROOT="/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc/campaigns/hela_ablation_${timestamp}"
DEFAULT_LOG_ROOT="/home/davidciz/silver-truth/logs"

CAMPAIGN_ROOT="${DEFAULT_CAMPAIGN_ROOT}"
LOG_ROOT="${DEFAULT_LOG_ROOT}"
MLFLOW_BACKEND="file"
PHASE="all"
DRY_RUN=0
CHECK_READINESS=1
READINESS_OUTPUT_DIR="${REPO_ROOT}/data/audits/dataset_readiness"
READINESS_MAX_CROP_FILE_CHECKS=200
FOLDS=(1 2)
CONFIG="experiments/variants/hela_crop256.yaml"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/submit_hela_ablation_hpc.sh [options]

Options:
  --campaign-root PATH    Durable output root for paper_runs and MLflow.
  --log-root PATH         Durable log root for Slurm/job logs.
                          Default: /home/davidciz/silver-truth/logs
  --mlflow-backend MODE   file or sqlite. Default: file
  --phase MODE            A, B, C, or all. Default: all
  --folds "1 2"          Space-separated folds to submit. Default: "1 2"
  --config PATH           Variant config. Default: experiments/variants/hela_crop256.yaml
  --skip-readiness        Skip local dataset/crop/split readiness checks.
  --readiness-output-dir PATH
                          Output directory for readiness reports.
                          Default: data/audits/dataset_readiness
  --readiness-max-crop-file-checks N
                          Number of QA crop TIFFs per fold to inspect. Default: 200
  --dry-run               Print sbatch commands without submitting.
  --help                  Show this message.

Notes:
  - This helper only submits jobs.
  - The repo clone on HPC must already contain the HeLa synchronized data,
    whole-image parquets, and QA crop artifacts expected by run_ablation_hpc.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign-root)
            CAMPAIGN_ROOT="${2:-}"
            shift 2
            ;;
        --log-root)
            LOG_ROOT="${2:-}"
            shift 2
            ;;
        --mlflow-backend)
            MLFLOW_BACKEND="${2:-}"
            shift 2
            ;;
        --phase)
            PHASE="${2:-}"
            shift 2
            ;;
        --folds)
            read -r -a FOLDS <<< "${2:-}"
            shift 2
            ;;
        --config)
            CONFIG="${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-readiness)
            CHECK_READINESS=0
            shift
            ;;
        --readiness-output-dir)
            READINESS_OUTPUT_DIR="${2:-}"
            shift 2
            ;;
        --readiness-max-crop-file-checks)
            READINESS_MAX_CROP_FILE_CHECKS="${2:-}"
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

case "${MLFLOW_BACKEND}" in
    file|sqlite) ;;
    *)
        echo "ERROR: --mlflow-backend must be one of: file, sqlite" >&2
        exit 2
        ;;
esac

case "${PHASE}" in
    A|B|C|all) ;;
    *)
        echo "ERROR: --phase must be one of: A, B, C, all" >&2
        exit 2
        ;;
esac

for fold in "${FOLDS[@]}"; do
    case "${fold}" in
        1|2|mixed) ;;
        *)
            echo "ERROR: folds must be 1, 2, or mixed, got '${fold}'" >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${REPO_ROOT}/${CONFIG}" ]]; then
    echo "ERROR: Missing variant config: ${CONFIG}" >&2
    exit 1
fi

mkdir -p "${LOG_ROOT}/slurm"

echo "Campaign root: ${CAMPAIGN_ROOT}"
echo "Log root: ${LOG_ROOT}"
echo "MLflow backend: ${MLFLOW_BACKEND}"
echo "Phase: ${PHASE}"
echo "Folds: ${FOLDS[*]}"
echo "Config: ${CONFIG}"
echo "Readiness check: ${CHECK_READINESS}"
echo ""

if [[ "${CHECK_READINESS}" -eq 1 ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        PYTHON_BIN="python"
    fi

    readiness_cmd=(
        "${PYTHON_BIN}"
        "${REPO_ROOT}/scripts/check_dataset_readiness.py"
        --config "${CONFIG}"
        --output-dir "${READINESS_OUTPUT_DIR}"
        --max-crop-file-checks "${READINESS_MAX_CROP_FILE_CHECKS}"
    )
    for fold in "${FOLDS[@]}"; do
        readiness_cmd+=(--fold "${fold}")
    done

    echo "Running readiness check before submission..."
    "${readiness_cmd[@]}"
    echo ""
fi

submit_count=0
for fold in "${FOLDS[@]}"; do
    if [[ "${fold}" == "mixed" ]]; then
        job_fold="mixed"
    else
        job_fold="f${fold}"
    fi

    job_name="hela256_${job_fold}_ablation"
    cmd=(
        sbatch
        --job-name "${job_name}"
        --output "${LOG_ROOT}/slurm/%x-%j.out"
        --error "${LOG_ROOT}/slurm/%x-%j.err"
        "${REPO_ROOT}/scripts/run_ablation_hpc.sh"
        --config "${CONFIG}"
        --fold "${fold}"
        --phase "${PHASE}"
        --durable-root "${CAMPAIGN_ROOT}"
        --log-root "${LOG_ROOT}"
        --mlflow-backend "${MLFLOW_BACKEND}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
    submit_count=$((submit_count + 1))
done

echo ""
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry run complete: ${submit_count} job commands generated."
else
    echo "Submission complete: ${submit_count} jobs submitted."
fi
