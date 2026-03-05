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

## MLflow Storage Strategy

Use a **single shared local MLflow tracking store** for the whole project:

- Canonical path: `data/mlflow/mlruns`
- All experiment families share this store
- Separation happens by **MLflow experiment name**, not by different storage folders

Why:
- one UI endpoint for all runs
- easier cross-comparison across QA/fusion/ablation
- less confusion for future reruns

### Naming Convention

Use experiment names with clear prefixes:

- `qa-cnn-<dataset>-<fold>`
- `qa-filtering-<dataset>-<fold>`
- `fusion-<dataset>-<fold>`
- `paper-ablation-<dataset>`
- `paper-threshold-sweep-<dataset>`

### Current Defaults

Primary CLIs now default to the shared local store (`data/mlflow/mlruns`).
You can still override with CLI flags when needed (for isolated runs or remote MLflow servers).

### One-Time Setup Per Shell

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/mlruns
mkdir -p data/mlflow/mlruns
```

### MLflow UI

```bash
mlflow ui --backend-store-uri data/mlflow/mlruns
```

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
# Run crop-based fusion experiment with shared MLflow store
silver-fusion run-fusion-crops \
    --qa-parquet data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet \
    --flat-models-only \
    --mlflow-experiment fusion-BF-C2DL-HSC-fold1 \
    --mlflow-tracking-path data/mlflow/mlruns
```

### 3. CNN Training (MLflow)

Train QA models with MLflow tracking:

```bash
silver-qa cnn train \
    --parquet-file data/dataframes/BF-C2DL-HSC/qa_crops/mixed_sz64_qa_dataset.parquet \
    --input-channels "0,1" \
    --model-type efficientnet_b4 \
    --dropout-rate 0.5 \
    --batch-size 16 \
    --num-epochs 50 \
    --mlflow-experiment "qa-cnn-BF-C2DL-HSC-mixed" \
    --mlflow-tracking-uri data/mlflow/mlruns
```

### 4. Viewing Results

To launch the MLflow UI and view the results of your experiments:

```bash
mlflow ui --backend-store-uri data/mlflow/mlruns
```

This will start a local web server. Navigate to `http://127.0.0.1:5000` in your browser to see the tracking dashboard.

## Data Versioning

The `data/` directory is tracked by DVC. When changes are made to the dataset:

```bash
dvc add data
git commit data.dvc -m "Updated dataset with new preprocessing"
```

This ensures each Git commit is linked to a specific, versioned state of the data.
