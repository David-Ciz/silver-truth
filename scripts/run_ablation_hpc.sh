#!/bin/bash -l
#SBATCH --job-name=ablation
#SBATCH --account=FTA-26-18
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
WORKFLOW="config"
KEEP_SCRATCH=0
MLFLOW_BACKEND="file"
CAMPAIGN_TAG=""
EXTRA_ARGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
DURABLE_ROOT_BASE="/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc/campaigns"
SCRATCH_ROOT_BASE="/scratch/project/eu-25-40/silver-truth/campaigns"
LOG_ROOT_BASE="${HOME}/logs/ablation_hpc/campaigns"
DURABLE_ROOT=""
SCRATCH_ROOT=""
LOG_ROOT=""
DURABLE_ROOT_EXPLICIT=0
SCRATCH_ROOT_EXPLICIT=0
LOG_ROOT_EXPLICIT=0
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
  sbatch scripts/run_ablation_hpc.sh --config experiments/variants/baseline.yaml --fold 2 --workflow reduced

Options:
  --config PATH         Variant YAML to run.
  --fold {1|2|mixed}    Fold/split selector.
  --phase {A|B|C|all}   Optional phase filter. Default: all
  --workflow MODE       config, full, or reduced. Default: config
  --campaign-tag NAME   Optional campaign tag used to derive isolated default
                        durable/scratch/log roots.
  --reset               Pass --reset to scripts/run_ablation.py.
  --dry-run             Pass --dry-run to scripts/run_ablation.py.
  --durable-root PATH   Durable storage for paper_runs and MLflow. Default:
                        /mnt/proj1/.../silver-truth-hpc/campaigns/<campaign-tag>
  --scratch-root PATH   Scratch runtime root. Default:
                        /scratch/project/.../silver-truth/campaigns/<campaign-tag>/<job-id>
  --log-root PATH       Durable log root. Default:
                        $HOME/logs/ablation_hpc/campaigns/<campaign-tag>
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
        --workflow)
            WORKFLOW="${2:-}"
            shift 2
            ;;
        --campaign-tag)
            CAMPAIGN_TAG="${2:-}"
            shift 2
            ;;
        --durable-root)
            DURABLE_ROOT="${2:-}"
            DURABLE_ROOT_EXPLICIT=1
            shift 2
            ;;
        --scratch-root)
            SCRATCH_ROOT="${2:-}"
            SCRATCH_ROOT_EXPLICIT=1
            shift 2
            ;;
        --log-root)
            LOG_ROOT="${2:-}"
            LOG_ROOT_EXPLICIT=1
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

case "${WORKFLOW}" in
    config|full|reduced) ;;
    *)
        echo "ERROR: --workflow must be one of: config, full, reduced" >&2
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

CONFIG_STEM="$(basename "${CONFIG}")"
CONFIG_STEM="${CONFIG_STEM%.yaml}"
CONFIG_STEM_SAFE="${CONFIG_STEM//[^A-Za-z0-9._-]/_}"
WORKFLOW_SAFE="${WORKFLOW//[^A-Za-z0-9._-]/_}"
if [[ -z "${CAMPAIGN_TAG}" ]]; then
    timestamp="$(date +%Y%m%d_%H%M%S)"
    CAMPAIGN_TAG="ablation_${CONFIG_STEM_SAFE}__${WORKFLOW_SAFE}__${timestamp}"
fi
CAMPAIGN_TAG="${CAMPAIGN_TAG//[^A-Za-z0-9._-]/_}"

if [[ "${DURABLE_ROOT_EXPLICIT}" -eq 0 ]]; then
    DURABLE_ROOT="${DURABLE_ROOT_BASE}/${CAMPAIGN_TAG}"
fi
if [[ "${SCRATCH_ROOT_EXPLICIT}" -eq 0 ]]; then
    SCRATCH_ROOT="${SCRATCH_ROOT_BASE}/${CAMPAIGN_TAG}/${SLURM_JOB_ID:-manual}"
fi
if [[ "${LOG_ROOT_EXPLICIT}" -eq 0 ]]; then
    LOG_ROOT="${LOG_ROOT_BASE}/${CAMPAIGN_TAG}"
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

JOB_NAME_SAFE="${SLURM_JOB_NAME:-ablation}"
JOB_NAME_SAFE="${JOB_NAME_SAFE//[^A-Za-z0-9._-]/_}"
JOB_TAG="${JOB_NAME_SAFE}__${CONFIG_STEM}__fold-${FOLD}__${WORKFLOW}__job-${SLURM_JOB_ID:-manual}"
JOB_LOG_DIR="${LOG_ROOT}/jobs/${JOB_TAG}"
JOB_RUN_LOG="${JOB_LOG_DIR}/run.log"
JOB_META_FILE="${JOB_LOG_DIR}/metadata.env"
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
job_name=${SLURM_JOB_NAME:-ablation}
config=${CONFIG}
fold=${FOLD}
phase=${PHASE}
workflow=${WORKFLOW}
campaign_tag=${CAMPAIGN_TAG}
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
            printf 'job_id\tjob_name\tconfig\tfold\tphase\tworkflow\tcampaign_tag\tstatus\tdurable_root\tjob_log_dir\trun_log\n' > "${JOB_INDEX_FILE}"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${SLURM_JOB_ID:-manual}" \
            "${SLURM_JOB_NAME:-ablation}" \
            "${CONFIG}" \
            "${FOLD}" \
            "${PHASE}" \
            "${WORKFLOW}" \
            "${CAMPAIGN_TAG}" \
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
cat > "${JOB_META_FILE}" <<EOF
job_id=${SLURM_JOB_ID:-manual}
job_name=${SLURM_JOB_NAME:-ablation}
config=${CONFIG}
fold=${FOLD}
phase=${PHASE}
workflow=${WORKFLOW}
campaign_tag=${CAMPAIGN_TAG}
durable_root=${DURABLE_ROOT}
log_root=${LOG_ROOT}
job_log_dir=${JOB_LOG_DIR}
run_log=${JOB_RUN_LOG}
scratch_root=${SCRATCH_ROOT}
started_at=$(date --iso-8601=seconds 2>/dev/null || date)
EOF
exec > >(tee -a "${JOB_RUN_LOG}") 2>&1

echo "=== Ablation HPC job ${SLURM_JOB_ID:-manual} started at $(date) ==="
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Repo root: ${REPO_ROOT}"
echo "Config: ${CONFIG}"
echo "Fold: ${FOLD}"
echo "Phase: ${PHASE}"
echo "Workflow: ${WORKFLOW}"
echo "Campaign tag: ${CAMPAIGN_TAG}"
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

DATASET_NAME="$(resolve_dataset)"
if [[ -z "${DATASET_NAME}" ]]; then
    echo "ERROR: Could not resolve dataset from config: ${CONFIG}" >&2
    exit 1
fi

mapfile -t STAGING_VALUES < <(python - <<PY
from pathlib import Path
import sys

sys.path.insert(0, "${REPO_ROOT}")
from scripts.run_ablation import load_config

cfg = load_config(Path("${CONFIG}"), "${FOLD}")
print(cfg["whole_image_parquet_template"])
print(cfg["qa_parquet_template"])
print(cfg["crop_size"])
PY
)
SOURCE_WHOLE_IMAGE_PARQUET="${STAGING_VALUES[0]}"
SOURCE_QA_PARQUET="${STAGING_VALUES[1]}"
CROP_SIZE="${STAGING_VALUES[2]}"

stage_file_to_scratch() {
    local source_path="$1"
    local relative_path=""

    if [[ "${source_path}" = /* ]]; then
        relative_path="${source_path#${REPO_ROOT}/}"
    else
        relative_path="${source_path}"
        source_path="${REPO_ROOT}/${source_path}"
    fi

    if [[ ! -e "${source_path}" ]]; then
        echo "ERROR: Required staging source not found: ${source_path}" >&2
        exit 1
    fi

    mkdir -p "${SCRATCH_ROOT}/$(dirname "${relative_path}")"
    rsync -aL "${source_path}" "${SCRATCH_ROOT}/${relative_path}"
}

echo "Staging data to scratch..."
mkdir -p \
    "${SCRATCH_ROOT}/data/synchronized_data" \
    "${SCRATCH_ROOT}/data/dataframes/${DATASET_NAME}" \
    "${SCRATCH_ROOT}/data/qa_crops/${DATASET_NAME}"
rsync -aL --delete \
    "${REPO_ROOT}/data/synchronized_data/${DATASET_NAME}/" \
    "${SCRATCH_ROOT}/data/synchronized_data/${DATASET_NAME}/"

echo "Staging resolved dataframe inputs only..."
stage_file_to_scratch "${SOURCE_WHOLE_IMAGE_PARQUET}"
stage_file_to_scratch "${SOURCE_QA_PARQUET}"

SOURCE_QA_CROP_DIR="${REPO_ROOT}/data/qa_crops/${DATASET_NAME}/sz${CROP_SIZE}"
if [[ -d "${SOURCE_QA_CROP_DIR}" ]]; then
    echo "Staging QA crop directory: ${SOURCE_QA_CROP_DIR}"
    mkdir -p "${SCRATCH_ROOT}/data/qa_crops/${DATASET_NAME}"
    rsync -aL --delete \
        "${SOURCE_QA_CROP_DIR}/" \
        "${SCRATCH_ROOT}/data/qa_crops/${DATASET_NAME}/sz${CROP_SIZE}/"
else
    echo "WARNING: QA crop directory not found, continuing because not all phases require it: ${SOURCE_QA_CROP_DIR}"
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
    --workflow "${WORKFLOW}" \
    "${EXTRA_ARGS[@]}"

echo "=== Ablation HPC job completed at $(date) ==="
echo "Results: ${DURABLE_ROOT}/paper_runs"
echo "MLflow: ${DURABLE_ROOT}/mlflow/mlruns"
