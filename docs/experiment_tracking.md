# Experiment Tracking

This project uses a combination of DVC (Data Version Control) and MLflow to ensure that all experiments are reproducible, trackable, and comparable.

## Core Tools

- **MLflow**: Used for logging experiment parameters, metrics, and artifacts. It provides a web-based UI to visualize and compare results across different runs.
- **DVC (Data Version Control)**: Used to version datasets, define reproducible data processing pipelines, and ensure that we can always trace back to the exact version of the data used in any experiment.

## Division of Responsibilities

| **DVC (Deterministic Pipeline)** | **MLflow (Experiments)** |
|----------------------------------|--------------------------|
| Data preprocessing | CNN training runs |
| Dataset splits creation | Hyperparameter tuning |
| QA crops generation | Model comparison |
| Job file generation | Fusion strategy comparison |
| | Evaluation metrics logging |

**Key principle**: DVC handles deterministic, reproducible data transformations. MLflow handles experimental runs where you're comparing different approaches/hyperparameters.

## Workflow

### 1. Data Preparation (DVC)

First, run the DVC pipeline to prepare your data:

```bash
# Run the full pipeline (or specific stages)
dvc repro

# Or run specific stages
dvc repro create_mixed@BF-C2DL-HSC
dvc repro generate_job_files
```

This will:
- Create dataset dataframes with proper train/val/test splits
- Generate QA crops for CNN training
- Generate job files for fusion experiments

### 2. Fusion Experiments (MLflow)

Compare different fusion strategies using the experiment script:

```bash
# Run all flat models (no weights required)
python scripts/run_fusion_experiment.py \
    --dataset BF-C2DL-HSC \
    --campaign 02 \
    --time-points "46,56,287,448,799,879,1585,1748" \
    --flat-models-only

# Run specific models
python scripts/run_fusion_experiment.py \
    --dataset BF-C2DL-HSC \
    --campaign 02 \
    --time-points "0-1000" \
    --models THRESHOLD_FLAT --models MAJORITY_FLAT --models BIC_FLAT_VOTING

# Run all models (requires weight files for some)
python scripts/run_fusion_experiment.py \
    --dataset BF-C2DL-HSC \
    --campaign 02 \
    --time-points "0-1000" \
    --all-models
```

### 3. CNN Training (MLflow)

Train QA models with MLflow tracking:

```bash
python cnn.py train \
    --parquet-file data/qa_crops/BF-C2DL-HSC/mixed_sz64/qa_dataset.parquet \
    --model-type efficientnet_b4 \
    --dropout-rate 0.5 \
    --batch-size 16 \
    --num-epochs 50 \
    --mlflow-experiment "cnn-jaccard"
```

### 4. Viewing Results

To launch the MLflow UI and view the results of your experiments:

```bash
mlflow ui
```

This will start a local web server. Navigate to `http://127.0.0.1:5000` in your browser to see the tracking dashboard.

## Data Versioning

The `data/` directory is tracked by DVC. When changes are made to the dataset:

```bash
dvc add data
git commit data.dvc -m "Updated dataset with new preprocessing"
```

This ensures each Git commit is linked to a specific, versioned state of the data.
