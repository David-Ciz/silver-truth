# Silver Truth

This repository contains tools for processing, synchronizing, and evaluating cell tracking data.

## Project Structure

- `src/silver_truth`: Main source code and package logic.
- `scripts`: Orchestration scripts and research notebooks.
- `data`: Raw data, synchronized datasets, and models (DVC tracked).
- `results`: Evaluation outputs and experiment logs (Git ignored).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/David-Ciz/silver-truth
    cd silver-truth
    ```

2.  **Install in editable mode:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    pip install -e .[dev]
    ```
    *Note: Installing with `-e` ensures that changes you make to the code are immediately reflected without reinstalling.*

## Quick Start

We provide a master pipeline script to run the entire workflow:

```bash
python scripts/run_pipeline.py run-all
```

Or run individual steps using the installed system commands:

- **Preprocessing**: `silver-preprocessing ...`
- **Fusion**: `silver-fusion ...`
- **Evaluation**: `silver-evaluation ...`
- **Ensemble**: `silver-ensemble ...`
- **QA**: `silver-qa ...`

## DVC (Data Version Control)

This project uses DVC to manage large data files.
- **Pull data**: `dvc pull`
- **Add new data**:
  ```bash
  dvc add data/new_dataset.parquet
  git add data/new_dataset.parquet.dvc .gitignore
  git commit -m "Add new dataset"
  ```

---

## Detailed Workflows

### 1. Preprocessing (`silver-preprocessing`)

Synchronize datasets and create dataframes.

```bash
# Synchronize
silver-preprocessing synchronize-datasets <datasets_folder> <output_directory>

# Create DataFrame
silver-preprocessing create-dataset-dataframe <synchronized_dataset_dir> --output_path <output.parquet>
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
silver-evaluation evaluate-competitor <dataset_dataframe> --output results.csv
```
