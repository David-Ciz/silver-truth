# Silver Truth

This repository contains tools for processing, synchronizing, and evaluating cell tracking data.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/silver-truth.git
    cd silver-truth
    ```

2.  Create a virtual environment and install the required dependencies:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .[dev]
    ```

## Usage

This project provides command-line tools for data processing and evaluation, available through `preprocessing.py` and `evaluation.py`.

### Preprocessing (`preprocessing.py`)

#### Synchronize Datasets

Synchronizes all segmentations with tracking markers in all the datasets.

**Usage:**

```bash
python preprocessing.py synchronize-datasets <path_to_datasets_folder> <path_to_output_directory>
```

**Example:**

```bash
python preprocessing.py synchronize-datasets data/inputs-2020-07 data/synchronized_data
```

#### Create Dataset DataFrame

Creates a pandas dataframe with dataset information from synchronized datasets.

**Usage:**

```bash
python preprocessing.py create-dataset-dataframe <path_to_synchronized_dataset_dir> --output_path <path_to_output_parquet_file>
```

**Example:**

```bash
python preprocessing.py create-dataset-dataframe data/synchronized_data/BF-C2DL-HSC --output_path BF-C2DL-HSC_dataset_dataframe.parquet
```

#### Compress TIF Files

Compresses all TIF files in a directory and its subdirectories using LZW compression.

**Usage:**

```bash
python preprocessing.py compress-tifs <path_to_directory>
```

**Example:**

```bash
python preprocessing.py compress-tifs data/synchronized_data
```

### Evaluation (`evaluation.py`)

#### Evaluate Competitor

Evaluates competitor segmentation results against ground truth using the Jaccard index.

**Usage:**

```bash
python evaluation.py evaluate-competitor <path_to_dataset_dataframe> [OPTIONS]
```

**Example:**

```bash
python evaluation.py evaluate-competitor BF_C2DL-HSC_dataset_dataframe.parquet --competitor MyCompetitor --output results.csv
```

To evaluate all competitors, omit the `--competitor` flag.
