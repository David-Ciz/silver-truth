#!/bin/bash
#SBATCH --job-name resnet50_jaccard
#SBATCH --account EU-25-40
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
# - Data copied to: /scratch/project/<project>/silver-truth-data (shared scratch)
# - The data is copied ONCE and reused across jobs (scratch is persistent)
# - Model outputs saved back to home directory
# =============================================================================

set -e  # Exit on any error

# --- Configuration ---
HOME_DIR="/home/davidciz/silver-truth"
SCRATCH_DATA="/scratch/project/open-25-40/silver-truth-data"
PARQUET_FILE="${HOME_DIR}/dataframes/BF-C2DL-HSC_QA_crops_64_split70-15-15_seed42.parquet"
OUTPUT_DIR="${HOME_DIR}/models"
RESULTS_DIR="${HOME_DIR}/results"

# Training hyperparameters
BATCH_SIZE=32
LEARNING_RATE=1e-4
NUM_EPOCHS=50

# --- Setup ---
echo "=== Job started at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"

# Load required modules
ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0
ml libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

# Activate virtual environment
source ${HOME_DIR}/.venv/bin/activate

# Create output directories if they don't exist
mkdir -p ${OUTPUT_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${HOME_DIR}/logs

# --- Data Setup (One-time copy to shared scratch) ---
# Check if data already exists on scratch
DATA_SOURCE="${HOME_DIR}/data/qa_data/BF-C2DL-HSC_crops_64"
DATA_DEST="${SCRATCH_DATA}/data/qa_data/BF-C2DL-HSC_crops_64"

if [ ! -d "${DATA_DEST}" ]; then
    echo "Copying data to scratch (first time setup)..."
    mkdir -p "${SCRATCH_DATA}/data/qa_data"
    cp -r "${DATA_SOURCE}" "${DATA_DEST}"
    echo "Data copy complete: $(ls -1 ${DATA_DEST} | wc -l) files"
else
    echo "Data already exists on scratch: $(ls -1 ${DATA_DEST} | wc -l) files"
fi

# --- Training ---
echo ""
echo "=== Starting Training ==="
echo "Parquet file: ${PARQUET_FILE}"
echo "Data root: ${SCRATCH_DATA}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo ""

cd ${HOME_DIR}

python resnet50.py train \
    --parquet-file "${PARQUET_FILE}" \
    --data-root "${SCRATCH_DATA}" \
    --output-model "${OUTPUT_DIR}/resnet50_jaccard_${SLURM_JOB_ID}.pt" \
    --output-excel "${RESULTS_DIR}/results_resnet50_${SLURM_JOB_ID}.xlsx" \
    --batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --num-epochs ${NUM_EPOCHS}

echo ""
echo "=== Job completed at $(date) ==="
echo "Model saved to: ${OUTPUT_DIR}/resnet50_jaccard_${SLURM_JOB_ID}.pt"
echo "Results saved to: ${RESULTS_DIR}/results_resnet50_${SLURM_JOB_ID}.xlsx"

