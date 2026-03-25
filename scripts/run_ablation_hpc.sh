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
KEEP_SCRATCH=0
MLFLOW_BACKEND="file"
EXTRA_ARGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
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
  --mlflow-backend MODE MLflow backend: file or sqlite. Default: file
  --keep-scratch        Do not delete the per-job scratch directory on exit.
  --help                Show this message.

Override Slurm resources at submit time, for example:
  sbatch --job-name ablation_baseline_f2 --time 12:00:00 scripts/run_ablation_hpc.sh ...
EOF
}

resolve_dataset() {
    python - <<PY
from pathlib import Path
import sys

sys.path.insert(0, "${REPO_ROOT}")
from scripts.run_ablation import _load_config_with_inheritance

config_path = Path("${CONFIG}")
cfg = _load_config_with_inheritance(config_path)
print(cfg.get("dataset", ""))
PY
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
        --mlflow-backend)
            MLFLOW_BACKEND="${2:-}"
            shift 2
            ;;
        --keep-scratch)
            KEEP_SCRATCH=1
            shift
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

case "${MLFLOW_BACKEND}" in
    file|sqlite) ;;
    *)
        echo "ERROR: --mlflow-backend must be one of: file, sqlite" >&2
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
    if [[ "${KEEP_SCRATCH}" -eq 0 && -n "${SCRATCH_ROOT:-}" && -d "${SCRATCH_ROOT}" ]]; then
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
echo "Keep scratch: ${KEEP_SCRATCH}"
echo "MLflow backend: ${MLFLOW_BACKEND}"

command -v ml >/dev/null 2>&1 && ml purge >/dev/null 2>&1 || true
ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0
ml libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

source "${VENV_DIR}/bin/activate"

mkdir -p "${SCRATCH_ROOT}/data" "${DURABLE_ROOT}/paper_runs" "${DURABLE_ROOT}/mlflow/mlartifacts" "${DURABLE_ROOT}/mlflow/mlruns"

DATASET_NAME="$(resolve_dataset)"
if [[ -z "${DATASET_NAME}" ]]; then
    echo "ERROR: Could not resolve dataset from config: ${CONFIG}" >&2
    exit 1
fi

echo "Staging data to scratch..."
mkdir -p \
    "${SCRATCH_ROOT}/data/synchronized_data" \
    "${SCRATCH_ROOT}/data/dataframes/${DATASET_NAME}" \
    "${SCRATCH_ROOT}/data/qa_crops"
rsync -aL --delete \
    "${REPO_ROOT}/data/synchronized_data/${DATASET_NAME}/" \
    "${SCRATCH_ROOT}/data/synchronized_data/${DATASET_NAME}/"
rsync -aL --delete \
    "${REPO_ROOT}/data/dataframes/${DATASET_NAME}/" \
    "${SCRATCH_ROOT}/data/dataframes/${DATASET_NAME}/"
if [[ -d "${REPO_ROOT}/data/qa_crops/${DATASET_NAME}" ]]; then
    rsync -aL --delete \
        "${REPO_ROOT}/data/qa_crops/${DATASET_NAME}/" \
        "${SCRATCH_ROOT}/data/qa_crops/${DATASET_NAME}/"
fi

if [[ "${MLFLOW_BACKEND}" == "sqlite" ]]; then
    export MLFLOW_TRACKING_URI="sqlite:////${DURABLE_ROOT#/}/mlflow/mlflow.db"
    export SILVER_TRUTH_MLFLOW_ARTIFACT_ROOT="file://${DURABLE_ROOT}/mlflow/mlartifacts"
else
    export MLFLOW_TRACKING_URI="file://${DURABLE_ROOT}/mlflow/mlruns"
    unset SILVER_TRUTH_MLFLOW_ARTIFACT_ROOT || true
fi
export ABLATION_OUTPUT_DIR="${DURABLE_ROOT}/paper_runs"
export ABLATION_DATA_ROOT="${SCRATCH_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "${SCRATCH_ROOT}"

echo "Validating staged inputs..."
python - <<PY
from pathlib import Path
import sys

sys.path.insert(0, "${REPO_ROOT}")
from scripts.run_ablation import load_config

config_path = Path("${CONFIG}")
fold = "${FOLD}"
cfg = load_config(config_path, fold)

required_paths = {
    "whole_image_parquet": Path(cfg["whole_image_parquet_template"]),
    "qa_parquet": Path(cfg["qa_parquet_template"]),
}

missing = [name for name, path in required_paths.items() if not path.exists()]
for name, path in required_paths.items():
    print(f"  {name}: {path}")

if missing:
    print("")
    print("Missing staged inputs:")
    for name in missing:
        print(f"  - {name}: {required_paths[name]}")
    raise SystemExit(1)
PY

echo "Running ablation pipeline..."
python "${REPO_ROOT}/scripts/run_ablation.py" \
    --config "${CONFIG}" \
    --fold "${FOLD}" \
    --phase "${PHASE}" \
    "${EXTRA_ARGS[@]}"

echo "=== Ablation HPC job completed at $(date) ==="
echo "Results: ${DURABLE_ROOT}/paper_runs"
echo "MLflow: ${DURABLE_ROOT}/mlflow/mlruns"
