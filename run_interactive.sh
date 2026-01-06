#!/bin/bash
# =============================================================================
# Interactive Training Script for ResNet50
# =============================================================================
#
# Usage:
#   1. SSH to the HPC login node
#   2. Start a tmux session: tmux new -s training
#   3. Run: ./run_interactive.sh
#   4. Detach with: Ctrl+b, then d
#   5. Reconnect later with: tmux attach -t training
#
# Alternative with screen:
#   1. Start screen: screen -S training
#   2. Run: ./run_interactive.sh
#   3. Detach with: Ctrl+a, then d
#   4. Reconnect later with: screen -r training
# =============================================================================

set -e  # Exit on any error

# --- Configuration (same as train_resnet50.sh) ---
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
AUGMENT=true
SEED=42
NUM_WORKERS=4
GRAD_CLIP=1.0

# MLflow configuration
MLFLOW_EXPERIMENT="resnet50-jaccard"
MLFLOW_RUN_NAME="interactive_$(date +%Y%m%d_%H%M%S)"

echo "=== Requesting interactive GPU job ==="
echo "This will allocate a GPU node and run training."
echo "Make sure you're in tmux/screen to survive disconnection!"
echo ""

# Request interactive job with srun
# Using srun instead of salloc to directly run the commands
srun --job-name=resnet50_interactive \
     --account=eu-25-40 \
     --partition=qgpu \
     --nodes=1 \
     --gpus=1 \
     --ntasks-per-node=64 \
     --time=4:00:00 \
     --pty bash -c "
set -e

echo '=== Interactive job started at \$(date) ==='
echo 'Node: \$(hostname)'

# Load required modules
ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0
ml libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

# Activate virtual environment
source ${VENV_DIR}/bin/activate

# Create output directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${RESULTS_DIR}
mkdir -p ${HOME_DIR}/logs

cd ${HOME_DIR}

# --- Data Setup ---
echo 'Checking DVC data in repo...'
DATA_SRC=\"${HOME_DIR}/data/qa_data/BF-C2DL-HSC_crops_64\"

if [ ! -d \"\${DATA_SRC}\" ]; then
    echo 'Checking out data from DVC local cache...'
    dvc checkout data/qa_data/BF-C2DL-HSC_crops_64.dvc
fi

if [ ! -d \"\${DATA_SRC}\" ]; then
    echo 'ERROR: Data directory not found after checkout'
    exit 1
fi

echo \"Data source: \$(ls -1 \${DATA_SRC} | wc -l) files\"

# Copy to scratch if needed
DATA_SCRATCH=\"${SCRATCH_DATA}/data/qa_data/BF-C2DL-HSC_crops_64\"

if [ ! -d \"\${DATA_SCRATCH}\" ]; then
    echo 'Copying data to scratch for fast I/O...'
    mkdir -p \"${SCRATCH_DATA}/data/qa_data\"
    cp -rL \"\${DATA_SRC}\" \"\${DATA_SCRATCH}\"
    echo \"Copy complete: \$(ls -1 \${DATA_SCRATCH} | wc -l) files\"
else
    echo \"Data already on scratch: \$(ls -1 \${DATA_SCRATCH} | wc -l) files\"
fi

# --- Training ---
echo ''
echo '=== Starting Training ==='
echo \"Parquet file: ${PARQUET_FILE}\"
echo \"Data root (scratch): ${SCRATCH_DATA}\"
echo \"Batch size: ${BATCH_SIZE}\"
echo \"Learning rate: ${LEARNING_RATE}\"
echo \"Epochs: ${NUM_EPOCHS}\"
echo ''

python resnet50.py train \\
    --parquet-file \"${PARQUET_FILE}\" \\
    --data-root \"${SCRATCH_DATA}\" \\
    --output-model \"${OUTPUT_DIR}/resnet50_jaccard_interactive.pt\" \\
    --output-excel \"${RESULTS_DIR}/results_resnet50_interactive.xlsx\" \\
    --batch-size ${BATCH_SIZE} \\
    --learning-rate ${LEARNING_RATE} \\
    --num-epochs ${NUM_EPOCHS} \\
    --weight-decay ${WEIGHT_DECAY} \\
    --dropout-rate ${DROPOUT_RATE} \\
    --patience ${PATIENCE} \\
    --augment \\
    --seed ${SEED} \\
    --num-workers ${NUM_WORKERS} \\
    --grad-clip ${GRAD_CLIP} \\
    --mlflow-experiment \"${MLFLOW_EXPERIMENT}\" \\
    --mlflow-run-name \"${MLFLOW_RUN_NAME}\"

echo ''
echo '=== Training completed at \$(date) ==='
"

echo ""
echo "=== Interactive job finished ==="

