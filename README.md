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
    python preprocessing.py synchronize-datasets "C:\Users\wei0068\Desktop\IT4I\inputs-2020-07" "C:\Users\wei0068\Desktop\IT4I\synchronized_data"
    ```

2.  **Create Dataset DataFrame (`preprocessing.py`)**
    Creates a pandas dataframe with dataset information from synchronized datasets. This `.parquet` file is a crucial input for subsequent steps.

    **Usage:**

    ```bash
    python preprocessing.py create-dataset-dataframe <path_to_synchronized_dataset_dir> --output_path <path_to_output_parquet_file>
    ```

    **Examples:**

    ```bash
    # Jednotlivý dataset
    python preprocessing.py create-dataset-dataframe "C:\Users\wei0068\Desktop\IT4I\synchronized_data\BF-C2DL-MuSC" --output_path "BF-C2DL-MuSC_dataset_dataframe.parquet"
    
    # Další příklady
    python preprocessing.py create-dataset-dataframe "C:\Users\wei0068\Desktop\IT4I\synchronized_data\DIC-C2DH-HeLa" --output_path "DIC-C2DH-HeLa_dataset_dataframe.parquet"
    
    # Automatické vytvoření všech dataframes
    python create_all_dataframes.py
    ```

3.  **Generate Job Files (`run_fusion.py`)**
    Generates job files required by the `run-fusion` command. It uses the `.parquet` file created in the previous step.

    **Usage:**

    ```bash
    python run_fusion.py generate-jobfiles --parquet-file <path_to_parquet_file> --campaign-number <campaign_number> --output-dir <output_directory>
    ```

    **Examples:**

    ```bash
    # Jednotlivý dataset a kampaň
    python run_fusion.py generate-jobfiles --parquet-file "BF-C2DL-MuSC_dataset_dataframe.parquet" --campaign-number "01" --output-dir "job_files"
    
    # Další příklady
    python run_fusion.py generate-jobfiles --parquet-file "DIC-C2DH-HeLa_dataset_dataframe.parquet" --campaign-number "02" --output-dir "job_files"
    
    # Automatické vytvoření všech job files
    python generate_all_jobfiles.py
    ```

4.  **Run Fusion (`run_fusion.py`)**
    Executes the fusion process using the generated job files.

    **Usage:**

    ```bash
    python run_fusion.py run-fusion --jar-path <path_to_jar> --job-file <path_to_job_file> --output-pattern <output_pattern> --time-points <time_points> --num-threads <num_threads> --model <model> [OPTIONS]
    ```

    **Example:**

    ```bash
    # Run fusion for a single dataset
    python run_fusion.py run-fusion --jar-path "src/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar" --job-file "job_files/BF-C2DL-MuSC_01_job_file.txt" --output-pattern "fused_results/BF-C2DL-MuSC_01_fused_TTTT.tif" --time-points "0-61" --num-threads 2 --model "threshold_use"

   python run_fusion.py run-fusion --jar-path "src/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar" --job-file "job_files/DIC-C2DH-HeLa_01_job_file.txt" --output-pattern "fused_results/DIC-C2DH-HeLa_01_fused_TTT.tif" --time-points "0-10" --num-threads 2 --model "majority_flat"

   python run_fusion.py run-fusion --jar-path "src/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar" --job-file "job_files/Fluo-C2DL-MSC_01_job_file.txt" --output-pattern "fused_results/Fluo-C2DL-MSC_01_fused_TTT.tif" --time-points "0-10" --num-threads 2 --model "threshold_use"

    
    # Run fusion for all datasets automatically
    python run_all_fusion.py
    ```

    **Available Models:**
    - `majority_flat`: Majority voting with flat weights
    - `threshold_flat`: Threshold-based fusion with flat weights
    - `bic_flat_voting`: BIC algorithm with flat voting
    - `bic_weighted_voting`: BIC algorithm with weighted voting
    - `simple`: Simple fusion (may have compatibility issues)
    - `threshold_user`: Threshold-based with user-defined parameters


44. parquet soubor fusion
python fussion_parquet --dataset "BF-C2DL-MuSC"

python fussion_parquet --dataset "DIC-C2DH-HeLa"

python fussion_parquet --dataset "Fluo-C2DL-MSC"


5.  **Evaluate Competitor (`evaluation.py`)**
    Evaluates competitor segmentation results against ground truth using the Jaccard index.

    **Usage:**

    ```bash
    python evaluation.py evaluate-competitor <path_to_dataset_dataframe> [OPTIONS]
    ```

    **Examples:**

    ```bash
    # Evaluate all competitors in a dataset
    python evaluation.py evaluate-competitor "fused_results_parquet/BF-C2DL-MuSC_dataset_dataframe_with_fused.parquet" --output "evaluation_results_BF-C2DL-MuSC.csv"

    python evaluation.py evaluate-competitor "fused_results_parquet/DIC-C2DH-HeLa_dataset_dataframe_with_fused.parquet" --output "evaluation_results_DIC-C2DH-HeLa.csv"
    
    # Evaluate specific competitor
    python evaluation.py evaluate-competitor "dataframes/BF-C2DL-MuSC_dataset_dataframe.parquet" --competitor "MU-Lux-CZ" --output "evaluation_results_BF-C2DL-MuSC_MU-Lux-CZ.csv"
    
    # Run evaluation for all datasets automatically
    python run_all_evaluations.py
    
    # Analyze and summarize all evaluation results
    python analyze_evaluation_results.py
    ```

    **Available Options:**
    - `--competitor`: Specify a single competitor to evaluate (optional)
    - `--output, -o`: Path to save detailed results as CSV
    - `--visualize, -v`: Generate visualization of results (placeholder)
    - `--campaign-col`: Column name that identifies the campaign (default: 'campaign_number')

### Utility Commands

#### Compress TIF Files (`preprocessing.py`)

Compresses all TIF files in a directory and its subdirectories using LZW compression. This can be run at any point to save disk space.

**Usage:**

```bash
python preprocessing.py compress-tifs <path_to_directory>
```

**Example:**

```bash
python preprocessing.py compress-tifs "C:\Users\wei0068\Desktop\IT4I\synchronized_data"

python visualize_tif.py "fused_results/BF-C2DL-MuSC_01_fused_0001.tif"
python show_objects.py "fused_results/BF-C2DL-MuSC_01_fused_0001.tif"
```

