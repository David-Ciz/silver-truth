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
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .[dev]
    
   Create requirements
   pip freeze > requirements.txt
    ```

## Usage

This project provides command-line tools for data processing and evaluation, available through `preprocessing.py` and `evaluation.py`.

### Workflow

The typical workflow involves these steps, executed in order:

1.  **Synchronize Datasets (`preprocessing.py`)**
    Synchronizes all segmentations with tracking markers in all the datasets.

    **Usage:**

    ```bash
    python preprocessing.py synchronize-datasets <path_to_datasets_folder> <path_to_output_directory>
    ```

    **Example:**

    ```bash
    python preprocessing.py synchronize-datasets data/inputs-2020-07 data/synchronized_data
    ```

2.  **Create Dataset DataFrame (`preprocessing.py`)**
    Creates a pandas dataframe with dataset information from synchronized datasets. This `.parquet` file is a crucial input for subsequent steps.

    **Usage:**

    ```bash
    python preprocessing.py create-dataset-dataframe <path_to_synchronized_dataset_dir> --output_path <path_to_output_parquet_file>
    ```

    **Example:**

    ```bash
    python preprocessing.py create-dataset-dataframe data/synchronized_data/BF-C2DL-HSC --output_path BF-C2DL-HSC_dataset_dataframe.parquet
    ```

3.  **Generate Job Files (`run_fusion.py`)**
    Generates job files required by the `run-fusion` command. It uses the `.parquet` file created in the previous step.

    **Usage:**

    ```bash
    python run_fusion.py generate-jobfiles --parquet-file <path_to_parquet_file> --campaign-number <campaign_number> --output-dir <output_directory>
    ```

    **Example:**

    ```bash
    python run_fusion.py generate-jobfiles --parquet-file BF-C2DL-HSC_dataset_dataframe.parquet --campaign-number 01 --output-dir job_files
    ```

4.  **Run Fusion (`run_fusion.py`)**
    Executes the fusion process using the generated job files.

    **Usage:**

    ```bash
    python run_fusion.py run-fusion --jar-path <path_to_jar> --job-file <path_to_job_file> --output-pattern <output_pattern> --time-points <time_points> --num-threads <num_threads> --model <model> [OPTIONS]
    ```

    **Example:**

    ```bash
    python run_fusion.py run-fusion --jar-path src/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar --job-file job_files/BF-C2DL-HSC_01_job_file.txt --output-pattern data/fused/BF-C2DL-HSC_fused_TTT.tif --time-points "1-10" --num-threads 4 --model "weighted_average"
    ```

5.  **Evaluate Competitor (`evaluation.py`)**
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

### Utility Commands

#### Compress TIF Files (`preprocessing.py`)

Compresses all TIF files in a directory and its subdirectories using LZW compression. This can be run at any point to save disk space.

**Usage:**

```bash
python preprocessing.py compress-tifs <path_to_directory>
```

**Example:**

```bash
python preprocessing.py compress-tifs data/synchronized_data
```
