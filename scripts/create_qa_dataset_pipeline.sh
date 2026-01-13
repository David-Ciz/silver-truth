#!/bin/bash
# =============================================================================
# QA Dataset Pipeline Script
# =============================================================================
# This script creates a comprehensive QA dataset with cropped images and
# synchronizes train/val/test splits from an existing parquet file.
#
# Steps:
#   1. Create dataset dataframe from synchronized data
#   2. Create QA dataset with 64x64 crops
#   3. Synchronize train/val/test splits from existing parquet
#
# Usage: ./scripts/create_qa_dataset_pipeline.sh
# =============================================================================

set -e  # Exit on any error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_NAME="BF-C2DL-HSC"
CROP_SIZE=64

# Paths (relative to project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYNCHRONIZED_DATA_DIR="${PROJECT_ROOT}/data/synchronized_data/${DATASET_NAME}"

# Output paths
DATASET_DATAFRAME="${PROJECT_ROOT}/dataframes/${DATASET_NAME}_dataset_dataframe.parquet"
QA_OUTPUT_DIR="${PROJECT_ROOT}/data/qa_data/${DATASET_NAME}_crops_${CROP_SIZE}"
QA_PARQUET="${PROJECT_ROOT}/dataframes/${DATASET_NAME}_QA_crops_${CROP_SIZE}.parquet"

# Source parquet for split synchronization
SOURCE_SPLIT_PARQUET="${PROJECT_ROOT}/qa_BF-C2DL-HSC_split70-15-15_seed42.parquet"

# Final output with synchronized splits
FINAL_OUTPUT_PARQUET="${PROJECT_ROOT}/dataframes/${DATASET_NAME}_QA_crops_${CROP_SIZE}_split70-15-15_seed42.parquet"

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
log_step() {
    echo ""
    echo "============================================================================="
    echo "STEP: $1"
    echo "============================================================================="
    echo ""
}

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[SUCCESS] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
cd "${PROJECT_ROOT}"

echo "============================================================================="
echo "QA Dataset Pipeline"
echo "============================================================================="
echo "Dataset: ${DATASET_NAME}"
echo "Crop Size: ${CROP_SIZE}x${CROP_SIZE}"
echo "Project Root: ${PROJECT_ROOT}"
echo "============================================================================="

# Check if synchronized data exists
if [ ! -d "${SYNCHRONIZED_DATA_DIR}" ]; then
    log_error "Synchronized data directory not found: ${SYNCHRONIZED_DATA_DIR}"
    exit 1
fi

# Check if source split parquet exists
if [ ! -f "${SOURCE_SPLIT_PARQUET}" ]; then
    log_error "Source split parquet not found: ${SOURCE_SPLIT_PARQUET}"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p "$(dirname "${DATASET_DATAFRAME}")"
mkdir -p "${QA_OUTPUT_DIR}"
mkdir -p "$(dirname "${FINAL_OUTPUT_PARQUET}")"

# -----------------------------------------------------------------------------
# Step 1: Create Dataset Dataframe
# -----------------------------------------------------------------------------
log_step "1/4 - Creating Dataset Dataframe"
log_info "Input: ${SYNCHRONIZED_DATA_DIR}"
log_info "Output: ${DATASET_DATAFRAME}"

python cli_preprocessing.py create-dataset-dataframe \
    "${SYNCHRONIZED_DATA_DIR}" \
    --output_path "${DATASET_DATAFRAME}"

if [ -f "${DATASET_DATAFRAME}" ]; then
    log_success "Dataset dataframe created: ${DATASET_DATAFRAME}"
else
    log_error "Failed to create dataset dataframe"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Create QA Dataset with Crops
# -----------------------------------------------------------------------------
log_step "2/4 - Creating QA Dataset with ${CROP_SIZE}x${CROP_SIZE} Crops"
log_info "Input: ${DATASET_DATAFRAME}"
log_info "Output Directory: ${QA_OUTPUT_DIR}"
log_info "Output Parquet: ${QA_PARQUET}"

python cli_qa.py create-dataset \
    "${DATASET_DATAFRAME}" \
    "${QA_OUTPUT_DIR}" \
    "${QA_PARQUET}" \
    --crop \
    --crop-size "${CROP_SIZE}"

if [ -f "${QA_PARQUET}" ]; then
    log_success "QA dataset created: ${QA_PARQUET}"
else
    log_error "Failed to create QA dataset"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 3: Synchronize Train/Val/Test Splits
# -----------------------------------------------------------------------------
log_step "3/4 - Synchronizing Train/Val/Test Splits"
log_info "Source Parquet (with splits): ${SOURCE_SPLIT_PARQUET}"
log_info "Target Parquet: ${QA_PARQUET}"
log_info "Output Parquet: ${FINAL_OUTPUT_PARQUET}"

python scripts/synchronize_train_splits.py \
    --source-parquet "${SOURCE_SPLIT_PARQUET}" \
    --target-parquet "${QA_PARQUET}" \
    --output-parquet "${FINAL_OUTPUT_PARQUET}"

if [ -f "${FINAL_OUTPUT_PARQUET}" ]; then
    log_success "Final QA dataset with splits created: ${FINAL_OUTPUT_PARQUET}"
else
    log_error "Failed to synchronize splits"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 4: Calculate Jaccard/IoU and F1 Scores
# -----------------------------------------------------------------------------
log_step "4/4 - Calculating Jaccard/IoU and F1 Scores"
log_info "Input Parquet: ${FINAL_OUTPUT_PARQUET}"

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from pathlib import Path
from src.evaluation.stacked_jaccard_logic import calculate_evaluation_metrics_cropped
calculate_evaluation_metrics_cropped(Path('${FINAL_OUTPUT_PARQUET}'))
"

if [ $? -eq 0 ]; then
    log_success "Jaccard and F1 scores calculated and added to parquet"
else
    log_error "Failed to calculate evaluation metrics"
    exit 1
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "============================================================================="
echo ""
echo "Output Files:"
echo "  1. Dataset Dataframe:    ${DATASET_DATAFRAME}"
echo "  2. QA Parquet:           ${QA_PARQUET}"
echo "  3. QA with Splits+Scores: ${FINAL_OUTPUT_PARQUET}"
echo "  4. Crop Images:          ${QA_OUTPUT_DIR}/"
echo ""
echo "============================================================================="

