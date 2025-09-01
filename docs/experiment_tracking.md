# Experiment Tracking Strategy

This document outlines a comprehensive strategy for using MLflow to track experiments in the `silver-truth` project. The goal is to ensure full reproducibility, from raw data to final results, and to provide clear insights into the performance of different models and configurations.

## 1. Guiding Principles

-   **Centralized Entry Point**: A new, top-level `run_experiments.py` script will be the single, unified entry point for launching any type of experiment. This ensures consistency in how runs are logged and makes the main workflow entry point highly visible.
-   **Complete Reproducibility**: For any given result in MLflow, you will be able to trace back to the exact code version (Git commit), data version (DVC hash), and parameters used to produce it.
-   **Log Inputs and Outputs**: Every run will log its important inputs (like data files and configurations) and outputs (like models, metrics, and result files) as MLflow artifacts.

## 2. MLflow Experiment Structure

We will use separate MLflow "Experiments" to keep different types of work organized:

-   **`data-processing`**: For runs that only involve preprocessing or data characterization. This is perfect for tracking your baseline "state-of-the-art" evaluations that don't involve new modeling.
-   **`fusion-experiments`**: For all runs related to the fusion algorithms.
-   **`qa-model-experiments`**: For training and evaluating your PyTorch Quality Assurance models.
-   **`ensembling-experiments`**: For the ensembling work.

## 3. How to Track a Full Workflow (Example: Fusion)

Here’s how we would track a fusion experiment from start to finish using the new `run_experiments.py` script:

1.  **Start the Run**: You would execute a command from the root of the repository:
    ```bash
    python run_experiments.py run-fusion --model BIC_FLAT_VOTING --job-file ...
    ```

2.  **Code Versioning**: The script will automatically record the current **Git commit hash** as a tag in MLflow (`git_commit`).

3.  **Data Versioning**:
    -   The script will log the DVC hash of your `data/` directory, linking the run to the precise version of your raw data.
    -   The input `.parquet` dataframe file used for the run will be logged as an **artifact**. This gives you a snapshot of the exact data that went into the experiment.

4.  **Parameter & Configuration Tracking**:
    -   All command-line parameters (`--model`, `--threshold`, etc.) are logged.
    -   The **`job_file.txt`** will be logged as an artifact. This provides an explicit record of which competitors were included in the fusion process.

5.  **Execution**: The script calls the `fuse_segmentations` and `run_evaluation` Python functions directly.

6.  **Log Metrics & Results**:
    -   Key metrics from the evaluation (e.g., overall Jaccard, per-campaign averages) are logged.
    -   The detailed evaluation CSV and the final, updated `.parquet` file (with the new fused image paths) are logged as **artifacts**.
    -   A few sample output images can also be logged for quick visual checks directly in the MLflow UI.

## 4. How This Strategy Solves Key Challenges

-   **Data Evolution**: By logging the input and output parquet files for each run, you get a clear, step-by-step history of how your data evolves.
-   **Tracking Baselines**: Your "state-of-the-art" runs can be executed and logged under the `data-processing` experiment, creating a clear benchmark against which all other experiments can be compared.
-   **Reproducibility**: To reproduce any result, a user can simply:
    1.  Find the run in the MLflow UI.
    2.  `git checkout` the logged commit.
    3.  `dvc checkout` using the logged data hash.
    4.  Re-run the command with the exact parameters logged in the run.
-   **Tracking Competitor Selection**: Logging the `job_file` as an artifact provides an unambiguous record of which competitors were used for each fusion run.

## 5. Viewing Results

To launch the MLflow UI and view the results of your experiments, run:

```bash
mlflow ui
```

This will start a local web server. You can navigate to the displayed URL (usually `http://127.0.0.1:5000`) in your browser to see the tracking dashboard.