# Experiment Tracking

This project uses a combination of DVC (Data Version Control) and MLflow to ensure that all experiments are reproducible, trackable, and comparable.

## Core Tools

- **MLflow**: Used for logging experiment parameters, metrics, and artifacts. It provides a web-based UI to visualize and compare results across different runs.
- **DVC (Data Version Control)**: Used to version datasets and large models without bloating the Git repository. It ensures that we can always trace back to the exact version of the data used in any experiment.

## Workflow

The primary entry point for conducting experiments is the `run_experiment.py` script.

### Running an Experiment

To run an experiment, use the following command:

```bash
python run_experiment.py [OPTIONS]
```

This script will:

1.  **Start an MLflow Run**: A new run is created under a specified experiment name (default: `silver-truth-qa`).
2.  **Log Parameters**: All command-line options and configuration parameters are logged to MLflow.
3.  **Execute the Pipeline**: The script orchestrates the necessary steps, such as running the fusion process (`cli_fusion.py`) and evaluating the results (`cli_evaluation.py`).
4.  **Log Results**: Key metrics (e.g., Jaccard scores) and output artifacts (e.g., evaluation CSVs, sample images) are logged to MLflow for analysis.

### Viewing Results

To launch the MLflow UI and view the results of your experiments, run:

```bash
mlflow ui
```

This will start a local web server. You can navigate to the displayed URL (usually `http://127.0.0.1:5000`) in your browser to see the tracking dashboard.

### Data Versioning

The `data/` directory is tracked by DVC. When changes are made to the dataset (e.g., new preprocessing steps), you can version the new data using:

```bash
dvc add data
git commit data.dvc -m "Updated dataset with new preprocessing"
```

This ensures that each Git commit is linked to a specific, versioned state of the data.
