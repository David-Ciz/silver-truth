#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

timestamp="$(date +%Y%m%d_%H%M%S)"
DEFAULT_CAMPAIGN_ROOT="/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc/campaigns/qa_phase_b_sweep_${timestamp}"
DEFAULT_LOG_ROOT="${HOME}/logs/qa_phase_b_sweep_${timestamp}"

CAMPAIGN_ROOT="${DEFAULT_CAMPAIGN_ROOT}"
LOG_ROOT="${DEFAULT_LOG_ROOT}"
MLFLOW_BACKEND="file"
DRY_RUN=0
FOLDS=(1 2)
DATASETS=(hsc64 musc256)
ARCHITECTURES=(resnet18 resnet50 resnet101 efficientnet_b1 efficientnet_b4 efficientnet_b7)

usage() {
    cat <<'EOF'
Usage:
  bash scripts/submit_qa_sweep_hpc.sh [options]

Options:
  --campaign-root PATH      Durable output root for paper_runs and MLflow.
  --log-root PATH           Durable log root for Slurm/job logs.
  --mlflow-backend MODE     MLflow backend passed through to run_ablation_hpc.sh: file or sqlite.
  --folds "1 2"             Space-separated folds to submit. Default: "1 2"
  --datasets "hsc64 musc256"
                            Dataset/crop presets to submit. Default: both.
  --architectures "..."
                            Space-separated QA architectures.
                            Default: "resnet18 resnet50 resnet101 efficientnet_b1 efficientnet_b4 efficientnet_b7"
  --dry-run                 Print sbatch commands without submitting.
  --help                    Show this message.

Dataset presets:
  hsc64    -> BF-C2DL-HSC, crop_size=64
  musc256  -> BF-C2DL-MuSC, crop_size=256
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
        --folds)
            read -r -a FOLDS <<< "${2:-}"
            shift 2
            ;;
        --datasets)
            read -r -a DATASETS <<< "${2:-}"
            shift 2
            ;;
        --architectures)
            read -r -a ARCHITECTURES <<< "${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
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

for fold in "${FOLDS[@]}"; do
    case "${fold}" in
        1|2) ;;
        *)
            echo "ERROR: folds must be 1 or 2, got '${fold}'" >&2
            exit 2
            ;;
    esac
done

variant_for() {
    local dataset_key="$1"
    local architecture="$2"

    case "${dataset_key}" in
        hsc64)
            printf 'experiments/variants/qa_hsc64_%s.yaml\n' "${architecture}"
            ;;
        musc256)
            printf 'experiments/variants/qa_musc256_%s.yaml\n' "${architecture}"
            ;;
        *)
            echo "ERROR: Unsupported dataset preset '${dataset_key}'" >&2
            exit 2
            ;;
    esac
}

mkdir -p "${LOG_ROOT}/slurm"

echo "Campaign root: ${CAMPAIGN_ROOT}"
echo "Log root: ${LOG_ROOT}"
echo "MLflow backend: ${MLFLOW_BACKEND}"
echo "Folds: ${FOLDS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Architectures: ${ARCHITECTURES[*]}"
echo ""

submit_count=0
for dataset_key in "${DATASETS[@]}"; do
    for architecture in "${ARCHITECTURES[@]}"; do
        variant_path="$(variant_for "${dataset_key}" "${architecture}")"
        if [[ ! -f "${REPO_ROOT}/${variant_path}" ]]; then
            echo "ERROR: Missing variant config: ${variant_path}" >&2
            exit 1
        fi

        for fold in "${FOLDS[@]}"; do
            job_name="qa_${dataset_key}_${architecture}_f${fold}"
            cmd=(
                sbatch
                --job-name "${job_name}"
                --output "${LOG_ROOT}/slurm/%x-%j.out"
                --error "${LOG_ROOT}/slurm/%x-%j.err"
                "${REPO_ROOT}/scripts/run_ablation_hpc.sh"
                --config "${variant_path}"
                --fold "${fold}"
                --phase B
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
    done
done

echo ""
if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Dry run complete: ${submit_count} job commands generated."
else
    echo "Submission complete: ${submit_count} jobs submitted."
fi
