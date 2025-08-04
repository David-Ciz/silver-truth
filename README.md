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
    # Single dataset
    python preprocessing.py create-dataset-dataframe "C:\Users\wei0068\Desktop\IT4I\synchronized_data\BF-C2DL-MuSC" --output_path "BF-C2DL-MuSC_dataset_dataframe.parquet"
    
    # All datasets
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
    # Single dataset
    python run_fusion.py generate-jobfiles --parquet-file "BF-C2DL-MuSC_dataset_dataframe.parquet" --campaign-number "01" --output-dir "job_files"
    
    # All jobfiles
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
    # Single dataset
    python run_fusion.py run-fusion --jar-path "src/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar" --job-file "job_files/BF-C2DL-MuSC_01_job_file.txt" --output-pattern "fused_results/BF-C2DL-MuSC_01_fused_TTTT.tif" --time-points "0-61" --num-threads 2 --model "majority_flat"
 
    # All datasets
    Change default fusion settings - DEFAULT_TIME_POINTS and DEFAULT_MODEL

    python run_all_fusion.py
    ```

    **Available Models:**
    - `majority_flat`: Majority voting with flat weights
    - `threshold_flat`: Threshold-based fusion with flat weights
    - `bic_flat_voting`: BIC algorithm with flat voting
    - `bic_weighted_voting`: BIC algorithm with weighted voting
    - `simple`: Simple fusion (may have compatibility issues)
    - `threshold_user`: Threshold-based with user-defined parameters

5.  **Run Fusion (`fusion_parquet.py`)**
Create a parquet files with information about fused images

**Example:**
python fussion_parquet --dataset "BF-C2DL-MuSC"

6.  **Evaluate Competitor (`evaluation.py`)**
    Evaluates competitor segmentation results against ground truth using the Jaccard index.

    **Usage:**

    ```bash
    python evaluation.py evaluate-competitor <path_to_dataset_dataframe> [OPTIONS]
    ```

    **Examples:**

    ```bash
    # Evaluate all competitors in a dataset
    python evaluation.py evaluate-competitor "fused_results_parquet/BF-C2DL-MuSC_dataset_dataframe_with_fused.parquet" --output "evaluation_results_BF-C2DL-MuSC.csv"
        
    # Run evaluation for all datasets automatically
    python run_all_evaluations.py
    
    # Analyze and summarize all evaluation results
    python analyze_evaluation_results.py
    ```

    **Available Options:**
    - `--competitor`: Specify a single competitor to evaluate (optional)
    - `--output, -o`: Path to save detailed results as CSV
    - `--detailed`: Create detailed per-cell evaluation results in parquet format
    - `--visualize, -v`: Generate visualization of results (placeholder)
    - `--campaign-col`: Column name that identifies the campaign (default: 'campaign_number')

7.  **Detailed Cell-Level Evaluation (`detailed_evaluation.py`)**
    Provides detailed Jaccard score evaluation for individual cells, storing results in parquet format for efficient analysis.

    **Usage:**

    ```bash
    python detailed_evaluation.py <path_to_dataset_dataframe> [OPTIONS]
    ```

    **Examples:**

    ```bash
    # Detailed evaluation for all competitors in a dataset
    python detailed_evaluation.py "dataframes/BF-C2DL-MuSC_dataset_dataframe.parquet" --output "detailed_BF-C2DL-MuSC.parquet"
    
    # Detailed evaluation for specific competitor
    python detailed_evaluation.py "dataframes/BF-C2DL-MuSC_dataset_dataframe.parquet" --competitor "MU-Lux-CZ" --output "detailed_BF-C2DL-MuSC_MU-Lux-CZ.parquet"
    
    # Run detailed evaluation for all datasets automatically
    python run_all_detailed_evaluations.py
    ```

    **Available Options:**
    - `--competitor`: Specify a single competitor to evaluate (if not specified, evaluates all)
    - `--output, -o`: Output parquet file path
    - `--campaign-col`: Column name for campaign identification (default: 'campaign_number')

8.  **Analyze Detailed Results (`analyze_detailed_results.py`)**
    Analyzes detailed cell evaluation results and generates comprehensive reports and visualizations.

    **Usage:**

    ```bash
    python analyze_detailed_results.py <path_to_detailed_results_parquet> [OPTIONS]
    ```

    **Examples:**

    ```bash
    # Analyze detailed results with default settings
    python analyze_detailed_results.py "detailed_BF-C2DL-MuSC.parquet"
    
    # Analyze with custom output directory and save summary
    python analyze_detailed_results.py "detailed_BF-C2DL-MuSC.parquet" --output-dir "analysis_BF-C2DL-MuSC" --save-summary
    ```

    **Available Options:**
    - `--output-dir`: Directory for analysis outputs (default: 'detailed_analysis')
    - `--save-summary`: Save summary statistics to CSV file

    **Generated Outputs:**
    - Comprehensive statistical summary report
    - Jaccard score distribution histograms
    - Competitor comparison visualizations (boxplots, violin plots)
    - Cell size vs. performance analysis
    - Performance heatmaps by dataset and competitor
    - Summary statistics CSV files

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

------------------------------------------------------------------------
python preprocessing.py create-qa-dataset-cli "dataframes/BF-C2DL-MuSC_dataset_dataframe.parquet" "qa_crops_BF-C2DL-MuSC" "qa_BF-C2DL-MuSC_dataset.parquet" --crop --crop-size 64

python preprocessing.py create-qa-dataset-cli "dataframes/DIC-C2DH-HeLa_dataset_dataframe.parquet" "qa_crops_DIC-C2DH-HeLa" "qa_DIC-C2DH-HeLa_dataset.parquet" --crop --crop-size 64
---------
python qa_evaluation.py "qa_BF-C2DL-MuSC_dataset.parquet" "dataframes/BF-C2DL-MuSC_dataset_dataframe.parquet" --output "qa_jaccard_results_BF-C2DL-MuSC.csv"

python qa_evaluation.py "qa_DIC-C2DH-HeLa_dataset.parquet" "dataframes/DIC-C2DH-HeLa_dataset_dataframe.parquet" --output "qa_jaccard_results_DIC-C2DH-HeLa.csv"

# QA evaluation with automatic parquet conversion for detailed analysis
python qa_evaluation.py "qa_BF-C2DL-MuSC_dataset.parquet" "dataframes/BF-C2DL-MuSC_dataset_dataframe.parquet" --output "qa_jaccard_results_BF-C2DL-MuSC.csv" --parquet-output "qa_BF-C2DL-MuSC_detailed_cells.parquet"



8.  **Convert QA Results to Detailed Parquet (`convert_qa_to_parquet.py`)**
    Converts existing QA evaluation CSV results to detailed parquet format for enhanced analysis.

    **Usage:**

    ```bash
    # Convert single QA result file
    python convert_qa_to_parquet.py convert-csv <qa_csv_results> <qa_dataframe_parquet> [--output OUTPUT]
    
    # Batch convert all QA results
    python convert_qa_to_parquet.py batch-convert [--qa-results-dir DIR] [--qa-dataframes-dir DIR] [--output-dir DIR]
    ```

    **Examples:**

    ```bash
    # Convert single file
    python convert_qa_to_parquet.py convert-csv "qa_jaccard_results_BF-C2DL-MuSC.csv" "qa_BF-C2DL-MuSC_dataset.parquet" --output "qa_BF-C2DL-MuSC_detailed_cells.parquet"
    
    # Batch convert all QA results in current directory
    python convert_qa_to_parquet.py batch-convert
    
    # Use analyze_detailed_results.py on converted QA parquet files
    python analyze_detailed_results.py "qa_BF-C2DL-MuSC_detailed_cells.parquet"
    ```

    **Generated Parquet Structure:**
    - All standard detailed evaluation columns
    - QA-specific metadata (crop coordinates, stacked paths)
    - `evaluation_type: 'qa_cropped'` to distinguish from full image evaluation

```