#!/bin/bash
#SBATCH --job-name resnet50_jaccard
#SBATCH --account eu-25-40
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --ntasks-per-node 64
#SBATCH --time 4:00:00
#SBATCH --output=logs/resnet50_%j.out
#SBATCH --error=logs/resnet50_%j.err

# =============================================================================
# ResNet50 Jaccard Index Training Job
# =============================================================================
#
# This script trains a ResNet50 model to predict Jaccard index from
# microscopy images and segmentation masks.
#
# Data Strategy:
# - Code lives in: /home/davidciz/silver-truth (git repo)
# - Data managed by DVC, cached at: /mnt/proj1/... (project storage - slow)
# - For training: copy data to /scratch/... (fast local SSD)
# - Scratch data persists between jobs, so only copy once
# =============================================================================

set -e  # Exit on any error

# --- Configuration ---
HOME_DIR="/home/davidciz/silver-truth"
PROJECT_DIR="/mnt/proj1/eu-25-40/innovaite"
VENV_DIR="${PROJECT_DIR}/silver-truth-venv"
SCRATCH_DATA="/scratch/project/eu-25-40/silver-truth-data"
PARQUET_FILE="${HOME_DIR}/dataframes/BF-C2DL-HSC_QA_crops_64_split70-15-15_seed42.parquet"
OUTPUT_DIR="${HOME_DIR}/models"
RESULTS_DIR="${HOME_DIR}/results"

# Training hyperparameters
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=50
WEIGHT_DECAY=1e-4
DROPOUT_RATE=0.3
PATIENCE=10
AUGMENT=true  # set to false to disable augmentation
SEED=42
NUM_WORKERS=4
GRAD_CLIP=1.0

# MLflow configuration
MLFLOW_EXPERIMENT="resnet50-jaccard"
MLFLOW_RUN_NAME="job_${SLURM_JOB_ID}"

# --- Setup ---
echo "=== Job started at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"

# Load required modules
ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0
ml libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

# Activate virtual environment (on project storage to avoid home quota)
source ${VENV_DIR}/bin/activate

# Create output directories if they don't exist
mkdir -p ${OUTPUT_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${HOME_DIR}/logs

cd ${HOME_DIR}

# --- Data Setup ---
# Data is managed by DVC with local cache at /mnt/proj1/eu-25-40/innovaite/dvc_store
# We use 'dvc checkout' to create symlinks from the local cache (no network needed)
echo "Checking DVC data in repo..."
DATA_SRC="${HOME_DIR}/data/qa_data/BF-C2DL-HSC_crops_64"

if [ ! -d "${DATA_SRC}" ]; then
    echo "Checking out data from DVC local cache..."
    dvc checkout data/qa_data/BF-C2DL-HSC_crops_64.dvc
fi

# Verify checkout worked
if [ ! -d "${DATA_SRC}" ]; then
    echo "ERROR: Data directory not found after checkout: ${DATA_SRC}"
    echo "Try running 'dvc checkout data/qa_data/BF-C2DL-HSC_crops_64.dvc' manually on login node."
    exit 1
fi

echo "Data source: $(ls -1 ${DATA_SRC} | wc -l) files"

# Step 2: Copy to scratch for fast I/O (only if not already there)
DATA_SCRATCH="${SCRATCH_DATA}/data/qa_data/BF-C2DL-HSC_crops_64"

if [ ! -d "${DATA_SCRATCH}" ]; then
    echo "Copying data to scratch for fast I/O..."
    mkdir -p "${SCRATCH_DATA}/data/qa_data"
    # Use cp -L to follow symlinks and copy actual files
    cp -rL "${DATA_SRC}" "${DATA_SCRATCH}"
    echo "Copy complete: $(ls -1 ${DATA_SCRATCH} | wc -l) files"
else
    echo "Data already on scratch: $(ls -1 ${DATA_SCRATCH} | wc -l) files"
fi

# Verify data is accessible
if [ ! -d "${DATA_SCRATCH}" ]; then
    echo "ERROR: Data directory not found on scratch!"
    exit 1
fi

# --- Training ---
echo ""
echo "=== Starting Training ==="
echo "Parquet file: ${PARQUET_FILE}"
echo "Data root (scratch): ${SCRATCH_DATA}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Weight decay: ${WEIGHT_DECAY}"
echo "Dropout rate: ${DROPOUT_RATE}"
echo "Early stopping patience: ${PATIENCE}"
echo "Data augmentation: ${AUGMENT}"
echo "Random seed: ${SEED}"
echo "Num workers: ${NUM_WORKERS}"
echo "Gradient clipping: ${GRAD_CLIP}"
echo "MLflow experiment: ${MLFLOW_EXPERIMENT}"
echo "MLflow run name: ${MLFLOW_RUN_NAME}"
echo ""

# Build augmentation flag
if [ "${AUGMENT}" = true ]; then
    AUGMENT_FLAG="--augment"
else
    AUGMENT_FLAG="--no-augment"
fi

# Use --data-root to point to scratch where data was copied
python resnet50.py train \
    --parquet-file "${PARQUET_FILE}" \
    --data-root "${SCRATCH_DATA}" \
    --output-model "${OUTPUT_DIR}/resnet50_jaccard_${SLURM_JOB_ID}.pt" \
    --output-excel "${RESULTS_DIR}/results_resnet50_${SLURM_JOB_ID}.xlsx" \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --num-epochs ${NUM_EPOCHS} \
    --weight-decay ${WEIGHT_DECAY} \
    --dropout-rate ${DROPOUT_RATE} \
    --patience ${PATIENCE} \
    ${AUGMENT_FLAG} \
    --seed ${SEED} \
    --num-workers ${NUM_WORKERS} \
    --grad-clip ${GRAD_CLIP} \
    --mlflow-experiment "${MLFLOW_EXPERIMENT}" \
    --mlflow-run-name "${MLFLOW_RUN_NAME}"

echo ""
echo "=== Job completed at $(date) ==="
echo "Model saved to: ${OUTPUT_DIR}/resnet50_jaccard_${SLURM_JOB_ID}.pt"
echo "Results saved to: ${RESULTS_DIR}/results_resnet50_${SLURM_JOB_ID}.xlsx"
