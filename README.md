# Silver Truth

Tools for processing, synchronizing, and evaluating cell tracking data, with a
full ablation pipeline for the paper experiments.

## Project Structure

- `src/silver_truth` — main package (CLI groups, evaluation, fusion, ensemble, QA)
- `scripts/` — orchestration scripts, notably **`run_ablation.py`** (see below)
- `experiments/` — YAML experiment configs (`base.yaml` + `variants/`)
- `data/` — raw data, parquets, models, MLflow store (DVC tracked)
- `docs/` — design documents; see [`docs/pipeline_design.md`](docs/pipeline_design.md)
  and [`docs/ablation_experiment_plan.md`](docs/ablation_experiment_plan.md)

---

## Installation

1. **Clone:**
   ```bash
   git clone https://github.com/David-Ciz/silver-truth
   cd silver-truth
   ```

2. **Install in editable mode:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

---

## 🚀 Ablation Pipeline Runner

`scripts/run_ablation.py` is the single entry point for running all paper experiments.
It reads a YAML variant config, sequences the full pipeline (Phase A → B → C), and
checkpoints each step so re-running safely resumes from where it stopped.

### One-liner to run a fold

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/mlruns

# Run the baseline experiment, fold 1 (all three phases)
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1

# Same for fold 2 — run in a second terminal for parallelism
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 2
```

### Inspect before running

```bash
# List all steps that will execute (with IDs) without running anything
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1 --list-steps

# Dry-run: print every resolved shell command without executing
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1 --dry-run
```

### Resume after training

Training steps (QA model, ensemble) exit cleanly after they finish.  The checkpoint
state in `.state/<run_id>.json` records completion, so **just re-run the same command**
to continue with the next step:

```bash
# After `silver-qa cnn train` finishes, pick up from the QA evaluation steps:
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1
```

### Experiment variants — changing one thing at a time

```bash
# Swap QA model (ResNet18 instead of ResNet50)
python scripts/run_ablation.py --config experiments/variants/resnet18_qa.yaml --fold 1

# Swap ensemble architecture (plain U-Net instead of U-Net++)
python scripts/run_ablation.py --config experiments/variants/unet_ensemble.yaml --fold 1

# Smaller crops (32×32)
python scripts/run_ablation.py --config experiments/variants/crop32.yaml --fold 1

# Different dataset
python scripts/run_ablation.py --config experiments/variants/fluo_dataset.yaml --fold 1
```

Each variant file in `experiments/variants/` only overrides what changes — everything
else is inherited from `experiments/base.yaml`.

### Run a single phase

```bash
# Only Phase A (unfiltered baselines — no training required)
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1 --phase A

# Only Phase B (QA training + validity)
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1 --phase B

# Only Phase C (QA-filtered ablation — requires Phase B complete)
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1 --phase C
```

### Reset and re-run from scratch

```bash
python scripts/run_ablation.py --config experiments/variants/baseline.yaml --fold 1 --reset
```

### View results in MLflow

```bash
mlflow ui --backend-store-uri data/mlflow/mlruns
# open http://127.0.0.1:5000
```

MLflow experiments are named `{phase}-{dataset}-{variant}-fold{N}` and every run is
tagged with `dataset`, `fold`, `variant`, `ablation_mode`, `qa_threshold` for easy
filtering in the Compare Runs view.

---

## DVC (Data Version Control)

This project uses DVC to manage large data files and datasets.

### Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -e .[dev]
   ```

2. **Configure DVC remote** (one-time setup):
   ```bash
   dvc remote add hpc_storage ssh://karolina.it4i.cz/mnt/proj1/eu-25-40/innovaite/dvc_store
   dvc remote default hpc_storage
   dvc config cache.shared group
   ```

3. **Pull data**:
   ```bash
   # Pull all data
   dvc pull
   
   # Or pull specific QA dataset
   dvc pull data/qa_crops/BF-C2DL-HSC/mixed_sz64/
   ```

📚 **For detailed DVC setup, SSH configuration, and working with QA datasets, see the [DVC Guide](docs/dvc_guide.md) in the wiki.**

## MLflow Tracking Store

Use one shared tracking folder for the project:

- `data/mlflow/mlruns`

Set it once per shell:

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=file:$(pwd)/data/mlflow/mlruns
mkdir -p data/mlflow/mlruns
```

Open UI:

```bash
mlflow ui --backend-store-uri data/mlflow/mlruns
```

See [docs/experiment_tracking.md](docs/experiment_tracking.md) for command-specific overrides and naming conventions.

---

## Detailed Workflows

### 1. Preprocessing (`silver-preprocessing`)

Synchronize datasets and create dataframes.

```bash
# Synchronize
silver-preprocessing synchronize-datasets <datasets_folder> <output_directory>

# Create DataFrame
silver-preprocessing create-dataset-dataframe <synchronized_dataset_dir> --output_path <output.parquet>

# Measure per-cell segmentation sizes and bbox fit rate for crop decisions
silver-preprocessing segmentation-size-stats \
  data/synchronized_data/BF-C2DL-MuSC/01_GT/SEG \
  data/synchronized_data/BF-C2DL-MuSC/02_GT/SEG \
  --crop-size 64 --crop-size 96 --crop-size 128
```

### 2. Fusion (`silver-fusion`)

Fuse segmentations from multiple competitors.

```bash
# Generate Jobs
silver-fusion generate-jobfiles --parquet-file <file> --campaign-number 01 --output-dir job_files

# Run Fusion
silver-fusion run-fusion --job-file job_files/job_01.txt ...
```

### 3. Evaluation (`silver-evaluation`)

Evaluate results against ground truth.

```bash
# Evaluate competitors (full-image IoU/F1)
silver-evaluation evaluate-competitor <dataset_dataframe> --output results.csv

# Evaluate Java fusion crops (reconstruct full images, then score)
silver-evaluation evaluate-fusion-crops <fused_parquet> \
  --fused-path-column bic_flat_voting \
  --output-dir results/fullimage/ \
  --output results/fullimage_eval.csv

# Filter a QA-enriched parquet before fusion (requires merge-qa-predictions first)
silver-evaluation filter-parquet <paper_ready_parquet> \
  --mode full_pipeline --threshold 0.75 --output filtered.parquet

silver-evaluation filter-parquet <paper_ready_parquet> \
  --mode qa_only --output filtered_top1.parquet

# QA model evaluation
silver-evaluation evaluate-qa-model <predictions.xlsx> --output-dir results/qa/
silver-evaluation evaluate-qa-filtering <predictions.xlsx> --output-dir results/qa_filter/
silver-evaluation merge-qa-predictions <parquet> <predictions.xlsx> --output enriched.parquet
```
