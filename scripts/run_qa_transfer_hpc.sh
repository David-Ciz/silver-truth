#!/bin/bash -l
#SBATCH --job-name=qa_transfer
#SBATCH --account=eu-25-40
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

CONFIG=""
TARGET_SPLIT=""
SOURCE_MODEL_PATH=""
KEEP_SCRATCH=0
NO_PLOTS=0
MLFLOW_BACKEND="file"
LOG_ROOT="${HOME}/logs/qa_transfer_hpc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
DURABLE_ROOT="/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc"
SCRATCH_ROOT="/scratch/project/eu-25-40/silver-truth/qa_transfer/${SLURM_JOB_ID:-manual}"
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
  sbatch scripts/run_qa_transfer_hpc.sh --config experiments/variants/qa_transfer_muscmixed_to_hsc.yaml --target-split fold-2

Options:
  --config PATH            QA transfer YAML to run.
  --target-split SPLIT     Target split selector: 1, 2, fold-1, fold-2, or mixed.
  --source-model-path PATH Optional explicit source QA checkpoint path.
  --durable-root PATH      Durable storage for paper_runs and MLflow.
  --scratch-root PATH      Scratch runtime root. Default uses SLURM_JOB_ID.
  --log-root PATH          Durable log root. Default: $HOME/logs/qa_transfer_hpc
  --venv-dir PATH          Virtualenv to activate. Default: <repo>/.venv
  --mlflow-backend MODE    MLflow backend: file or sqlite. Default: file
  --keep-scratch           Do not delete the per-job scratch directory on exit.
  --no-plots               Skip QA regression plot generation.
  --help                   Show this message.

Override Slurm resources at submit time, for example:
  sbatch --job-name qa_transfer_f2 --time 04:00:00 scripts/run_qa_transfer_hpc.sh ...
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="${2:-}"
            shift 2
            ;;
        --target-split)
            TARGET_SPLIT="${2:-}"
            shift 2
            ;;
        --source-model-path)
            SOURCE_MODEL_PATH="${2:-}"
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
        --log-root)
            LOG_ROOT="${2:-}"
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
        --no-plots)
            NO_PLOTS=1
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

if [[ -z "${CONFIG}" || -z "${TARGET_SPLIT}" ]]; then
    echo "ERROR: --config and --target-split are required." >&2
    exit 2
fi

case "${TARGET_SPLIT}" in
    1|2|fold-1|fold-2|mixed) ;;
    *)
        echo "ERROR: --target-split must be one of: 1, 2, fold-1, fold-2, mixed" >&2
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
LOG_ROOT="$(make_abs "${LOG_ROOT}")"
if [[ -n "${SOURCE_MODEL_PATH}" ]]; then
    SOURCE_MODEL_PATH="$(make_abs "${SOURCE_MODEL_PATH}")"
fi

CONFIG_STEM="$(basename "${CONFIG}")"
CONFIG_STEM="${CONFIG_STEM%.yaml}"
JOB_NAME_SAFE="${SLURM_JOB_NAME:-qa_transfer}"
JOB_NAME_SAFE="${JOB_NAME_SAFE//[^A-Za-z0-9._-]/_}"
JOB_TAG="${JOB_NAME_SAFE}__${CONFIG_STEM}__target-${TARGET_SPLIT}__job-${SLURM_JOB_ID:-manual}"
JOB_LOG_DIR="${LOG_ROOT}/jobs/${JOB_TAG}"
JOB_RUN_LOG="${JOB_LOG_DIR}/run.log"
JOB_SUMMARY_FILE="${JOB_LOG_DIR}/summary.env"
JOB_INDEX_FILE="${LOG_ROOT}/job_index.tsv"

cleanup() {
    local exit_code=$?
    local final_status="success"
    if [[ "${exit_code}" -ne 0 ]]; then
        final_status="failed"
    fi

    if [[ -n "${JOB_SUMMARY_FILE:-}" ]]; then
        mkdir -p "$(dirname "${JOB_SUMMARY_FILE}")"
        cat > "${JOB_SUMMARY_FILE}" <<EOF
status=${final_status}
exit_code=${exit_code}
job_id=${SLURM_JOB_ID:-manual}
job_name=${SLURM_JOB_NAME:-qa_transfer}
config=${CONFIG}
target_split=${TARGET_SPLIT}
durable_root=${DURABLE_ROOT}
log_root=${LOG_ROOT}
job_log_dir=${JOB_LOG_DIR}
run_log=${JOB_RUN_LOG}
scratch_root=${SCRATCH_ROOT}
completed_at=$(date --iso-8601=seconds 2>/dev/null || date)
EOF
    fi

    if [[ -n "${JOB_INDEX_FILE:-}" ]]; then
        mkdir -p "$(dirname "${JOB_INDEX_FILE}")"
        if [[ ! -f "${JOB_INDEX_FILE}" ]]; then
            printf 'job_id\tjob_name\tconfig\ttarget_split\tstatus\tdurable_root\tjob_log_dir\trun_log\n' > "${JOB_INDEX_FILE}"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${SLURM_JOB_ID:-manual}" \
            "${SLURM_JOB_NAME:-qa_transfer}" \
            "${CONFIG}" \
            "${TARGET_SPLIT}" \
            "${final_status}" \
            "${DURABLE_ROOT}" \
            "${JOB_LOG_DIR}" \
            "${JOB_RUN_LOG}" >> "${JOB_INDEX_FILE}"
    fi

    if [[ "${KEEP_SCRATCH}" -eq 0 && -n "${SCRATCH_ROOT:-}" && -d "${SCRATCH_ROOT}" ]]; then
        rm -rf "${SCRATCH_ROOT}"
    fi
}
trap cleanup EXIT

mkdir -p "${LOG_ROOT}/jobs" "${LOG_ROOT}/slurm" "${JOB_LOG_DIR}"
exec > >(tee -a "${JOB_RUN_LOG}") 2>&1

echo "=== QA transfer HPC job ${SLURM_JOB_ID:-manual} started at $(date) ==="
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Repo root: ${REPO_ROOT}"
echo "Config: ${CONFIG}"
echo "Target split: ${TARGET_SPLIT}"
echo "Scratch root: ${SCRATCH_ROOT}"
echo "Durable root: ${DURABLE_ROOT}"
echo "Log root: ${LOG_ROOT}"
echo "Job log dir: ${JOB_LOG_DIR}"
echo "Combined run log: ${JOB_RUN_LOG}"
echo "Keep scratch: ${KEEP_SCRATCH}"
echo "MLflow backend: ${MLFLOW_BACKEND}"

command -v ml >/dev/null 2>&1 && ml purge >/dev/null 2>&1 || true
ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0
ml libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

source "${VENV_DIR}/bin/activate"

mkdir -p "${SCRATCH_ROOT}/data" "${DURABLE_ROOT}/paper_runs" "${DURABLE_ROOT}/mlflow/mlartifacts" "${DURABLE_ROOT}/mlflow/mlruns"

mapfile -t DATASETS < <(
python - <<PY
from pathlib import Path
import sys

sys.path.insert(0, "${REPO_ROOT}")
from scripts.run_qa_transfer import collect_transfer_datasets, load_transfer_config

cfg = load_transfer_config(Path("${CONFIG}"), "${TARGET_SPLIT}")
for dataset in collect_transfer_datasets(cfg):
    print(dataset)
PY
)

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
    echo "ERROR: Could not resolve datasets from config: ${CONFIG}" >&2
    exit 1
fi

echo "Staging data to scratch..."
mkdir -p "${SCRATCH_ROOT}/data/synchronized_data" "${SCRATCH_ROOT}/data/dataframes" "${SCRATCH_ROOT}/data/qa_crops"
for dataset in "${DATASETS[@]}"; do
    mkdir -p \
        "${SCRATCH_ROOT}/data/synchronized_data/${dataset}" \
        "${SCRATCH_ROOT}/data/dataframes/${dataset}" \
        "${SCRATCH_ROOT}/data/qa_crops/${dataset}"
    rsync -aL --delete \
        "${REPO_ROOT}/data/synchronized_data/${dataset}/" \
        "${SCRATCH_ROOT}/data/synchronized_data/${dataset}/"
    rsync -aL --delete \
        "${REPO_ROOT}/data/dataframes/${dataset}/" \
        "${SCRATCH_ROOT}/data/dataframes/${dataset}/"
    if [[ -d "${REPO_ROOT}/data/qa_crops/${dataset}" ]]; then
        rsync -aL --delete \
            "${REPO_ROOT}/data/qa_crops/${dataset}/" \
            "${SCRATCH_ROOT}/data/qa_crops/${dataset}/"
    fi
done

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
from scripts.run_qa_transfer import load_transfer_config

cfg = load_transfer_config(Path("${CONFIG}"), "${TARGET_SPLIT}")
target_parquet = Path(cfg["target_parquet_template"])
print(f"  target_parquet: {target_parquet}")
if not target_parquet.exists():
    raise SystemExit(f"Missing staged target parquet: {target_parquet}")
PY

echo "Running QA transfer evaluation..."
CMD=(python "${REPO_ROOT}/scripts/run_qa_transfer.py" --config "${CONFIG}" --target-split "${TARGET_SPLIT}")
if [[ -n "${SOURCE_MODEL_PATH}" ]]; then
    CMD+=(--source-model-path "${SOURCE_MODEL_PATH}")
fi
if [[ "${NO_PLOTS}" -eq 1 ]]; then
    CMD+=(--no-plots)
fi
"${CMD[@]}"

echo "=== QA transfer HPC job completed at $(date) ==="
echo "Results: ${DURABLE_ROOT}/paper_runs/qa_transfer"
echo "MLflow: ${DURABLE_ROOT}/mlflow/mlruns"
