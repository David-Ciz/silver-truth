#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="experiments/variants/hsc_qa_rank_eval.yaml"
DURABLE_ROOT="/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc"
SCRATCH_BASE="/scratch/project/eu-25-40/silver-truth/ablation"
VENV_DIR="${REPO_ROOT}/.venv"
MLFLOW_BACKEND="file"
KEEP_SCRATCH=0
ABLAT_EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  scripts/run_hsc_qa_rank_eval_hpc.sh [options]

This submits:
1. fold-1 HSC QA ranking evaluation
2. fold-2 HSC QA ranking evaluation
3. a dependent comparison job after both folds succeed

Options:
  --config PATH          Variant YAML to submit. Default: experiments/variants/hsc_qa_rank_eval.yaml
  --durable-root PATH    Durable storage root.
  --scratch-base PATH    Scratch base directory.
  --venv-dir PATH        Virtualenv path.
  --mlflow-backend MODE  file or sqlite. Default: file
  --keep-scratch         Preserve scratch directories for the fold jobs.
  --reset                Pass --reset through to run_ablation_hpc.sh.
  --dry-run              Show the sbatch commands without submitting.
  --help                 Show this message.
EOF
}

DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="${2:-}"
            shift 2
            ;;
        --durable-root)
            DURABLE_ROOT="${2:-}"
            shift 2
            ;;
        --scratch-base)
            SCRATCH_BASE="${2:-}"
            shift 2
            ;;
        --venv-dir)
            VENV_DIR="${2:-}"
            shift 2
            ;;
        --mlflow-backend)
            MLFLOW_BACKEND="${2:-}"
            shift 2
            ;;
        --keep-scratch)
            KEEP_SCRATCH=1
            shift
            ;;
        --reset)
            ABLAT_EXTRA_ARGS+=(--reset)
            shift
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

submit_fold_job() {
    local fold="$1"
    local job_name="qa_rank_${fold}"
    local variant_name
    variant_name="$(basename "${CONFIG}" .yaml)"
    local scratch_root="${SCRATCH_BASE}/${variant_name}/${fold}"
    local cmd=(
        sbatch
        --parsable
        --job-name "${job_name}"
        "${REPO_ROOT}/scripts/run_ablation_hpc.sh"
        --config "${CONFIG}"
        --fold "${fold}"
        --durable-root "${DURABLE_ROOT}"
        --scratch-root "${scratch_root}"
        --venv-dir "${VENV_DIR}"
        --mlflow-backend "${MLFLOW_BACKEND}"
    )
    if [[ "${KEEP_SCRATCH}" -eq 1 ]]; then
        cmd+=(--keep-scratch)
    fi
    if [[ "${#ABLAT_EXTRA_ARGS[@]}" -gt 0 ]]; then
        cmd+=("${ABLAT_EXTRA_ARGS[@]}")
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return 0
    fi

    "${cmd[@]}"
}

compare_cmd=(
    sbatch
    --parsable
    --job-name qa_rank_compare
    "${REPO_ROOT}/scripts/compare_qa_variants_hpc.sh"
    --durable-root "${DURABLE_ROOT}"
    --candidate-variant "$(basename "${CONFIG}" .yaml)"
)

if [[ "${DRY_RUN}" -eq 1 ]]; then
    submit_fold_job 1
    submit_fold_job 2
    printf '%q ' "${compare_cmd[@]}"
    printf '\n'
    exit 0
fi

fold1_job_id="$(submit_fold_job 1)"
fold2_job_id="$(submit_fold_job 2)"

compare_with_dependency=(
    sbatch
    --parsable
    --dependency "afterok:${fold1_job_id}:${fold2_job_id}"
    --job-name qa_rank_compare
    "${REPO_ROOT}/scripts/compare_qa_variants_hpc.sh"
    --durable-root "${DURABLE_ROOT}"
    --candidate-variant "$(basename "${CONFIG}" .yaml)"
)
compare_job_id="$("${compare_with_dependency[@]}")"

echo "Submitted fold-1 job: ${fold1_job_id}"
echo "Submitted fold-2 job: ${fold2_job_id}"
echo "Submitted comparison job: ${compare_job_id}"
echo "Reports will be written under: ${DURABLE_ROOT}/paper_runs/reports/$(basename "${CONFIG}" .yaml)"
