#!/bin/bash -l
#SBATCH --job-name=ablation
#SBATCH --account=eu-25-40
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

CONFIG=""
FOLD=""
PHASE="all"
EXTRA_ARGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DURABLE_ROOT="/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc"
SCRATCH_ROOT="/scratch/project/eu-25-40/silver-truth/ablation/${SLURM_JOB_ID:-manual}"
VENV_DIR="${REPO_ROOT}/.venv"

make_abs() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${REPO_ROOT}/${path}"
    fi
}

usage() {
    cat <<'EOF'
Usage:
  sbatch scripts/run_ablation_hpc.sh --config experiments/variants/baseline.yaml --fold 2

Options:
  --config PATH         Variant YAML to run.
  --fold {1|2|mixed}    Fold/split selector.
  --phase {A|B|C|all}   Optional phase filter. Default: all
  --reset               Pass --reset to scripts/run_ablation.py.
  --dry-run             Pass --dry-run to scripts/run_ablation.py.
  --durable-root PATH   Durable storage for paper_runs and MLflow.
  --scratch-root PATH   Scratch runtime root. Default uses SLURM_JOB_ID.
  --venv-dir PATH       Virtualenv to activate. Default: <repo>/.venv
  --help                Show this message.

Override Slurm resources at submit time, for example:
  sbatch --job-name ablation_baseline_f2 --time 12:00:00 scripts/run_ablation_hpc.sh ...
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="${2:-}"
            shift 2
            ;;
        --fold)
            FOLD="${2:-}"
            shift 2
            ;;
        --phase)
            PHASE="${2:-}"
            shift 2
            ;;
        --durable-root)
            DURABLE_ROOT="${2:-}"
            shift 2
            ;;
        --scratch-root)
            SCRATCH_ROOT="${2:-}"
            shift 2
            ;;
        --venv-dir)
            VENV_DIR="${2:-}"
            shift 2
            ;;
        --reset)
            EXTRA_ARGS+=(--reset)
            shift
            ;;
        --dry-run)
            EXTRA_ARGS+=(--dry-run)
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

if [[ -z "${CONFIG}" || -z "${FOLD}" ]]; then
    echo "ERROR: --config and --fold are required." >&2
    exit 2
fi

case "${FOLD}" in
    1|2|mixed) ;;
    *)
        echo "ERROR: --fold must be one of: 1, 2, mixed" >&2
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

if [[ ! -f "${REPO_ROOT}/${CONFIG}" && ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}" >&2
    exit 1
fi

if [[ -f "${REPO_ROOT}/${CONFIG}" ]]; then
    CONFIG="$(cd "${REPO_ROOT}" && pwd)/${CONFIG}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "ERROR: Virtualenv directory not found: ${VENV_DIR}" >&2
    exit 1
fi

DURABLE_ROOT="$(make_abs "${DURABLE_ROOT}")"
SCRATCH_ROOT="$(make_abs "${SCRATCH_ROOT}")"
VENV_DIR="$(make_abs "${VENV_DIR}")"
CONFIG="$(make_abs "${CONFIG}")"

cleanup() {
    if [[ -n "${SCRATCH_ROOT:-}" && -d "${SCRATCH_ROOT}" ]]; then
        rm -rf "${SCRATCH_ROOT}"
    fi
}
trap cleanup EXIT

echo "=== Ablation HPC job ${SLURM_JOB_ID:-manual} started at $(date) ==="
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Repo root: ${REPO_ROOT}"
echo "Config: ${CONFIG}"
echo "Fold: ${FOLD}"
echo "Phase: ${PHASE}"
echo "Scratch root: ${SCRATCH_ROOT}"
echo "Durable root: ${DURABLE_ROOT}"

command -v ml >/dev/null 2>&1 && ml purge >/dev/null 2>&1 || true
ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0
ml libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

source "${VENV_DIR}/bin/activate"

mkdir -p "${SCRATCH_ROOT}/data" "${DURABLE_ROOT}/paper_runs" "${DURABLE_ROOT}/mlflow/mlruns"

echo "Staging data to scratch..."
rsync -a --delete "${REPO_ROOT}/data/synchronized_data/" "${SCRATCH_ROOT}/data/synchronized_data/"
rsync -a --delete "${REPO_ROOT}/data/dataframes/" "${SCRATCH_ROOT}/data/dataframes/"
rsync -a --delete "${REPO_ROOT}/data/qa_crops/" "${SCRATCH_ROOT}/data/qa_crops/"

export MLFLOW_TRACKING_URI="file://${DURABLE_ROOT}/mlflow/mlruns"
export ABLATION_OUTPUT_DIR="${DURABLE_ROOT}/paper_runs"
export ABLATION_DATA_ROOT="${SCRATCH_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "${SCRATCH_ROOT}"

echo "Running ablation pipeline..."
python "${REPO_ROOT}/scripts/run_ablation.py" \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --phase "${PHASE}" \
    "${EXTRA_ARGS[@]}"

echo "=== Ablation HPC job completed at $(date) ==="
echo "Results: ${DURABLE_ROOT}/paper_runs"
echo "MLflow: ${DURABLE_ROOT}/mlflow/mlruns"
