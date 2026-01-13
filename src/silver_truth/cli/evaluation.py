import click
from pathlib import Path
from typing import Optional
import logging

import mlflow

from silver_truth.evaluation.evaluation_logic import evaluate_competitor_logic
from silver_truth.evaluation.stacked_jaccard_logic import calculate_evaluation_metrics
from silver_truth.metrics.qa_model_evaluation import (
    evaluate_qa_model_from_excel,
    merge_predictions_to_parquet,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@click.command()
@click.argument("dataset_dataframe_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--competitor", help="Competitor name to evaluate. If None, evaluate all."
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Path to save results as CSV",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Generate visualization of results (Placeholder)",
)
@click.option(
    "--campaign-col",
    default="campaign_number",
    help="Column name that identifies the campaign",
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Create detailed per-cell evaluation results in parquet format",
)
def evaluate_competitor(
    dataset_dataframe_path: Path,
    competitor: Optional[str] = None,
    output: Optional[Path] = None,
    visualize: bool = False,
    campaign_col: str = "campaign_number",
    detailed: bool = False,
):
    """
    Evaluates competitor segmentation results against ground truth using Jaccard index.

    This script is a wrapper around the core evaluation logic in `run_evaluation`.
    With --detailed flag, also creates detailed per-cell evaluation results.
    """
    evaluate_competitor_logic(
        dataset_dataframe_path=dataset_dataframe_path,
        competitor=competitor,
        output=output,
        visualize=visualize,
        campaign_col=campaign_col,
        detailed=detailed,
    )


@click.command()
@click.argument("parquet_path", type=click.Path(exists=True, path_type=Path))
def calculate_evaluation_metrics_cli(parquet_path: Path):
    """
    For each row in the given parquet file, reads the stacked_path image,
    calculates the Jaccard and F1 scores between its [0] and [1] layers,
    and saves the results as new columns 'jaccard_score' and 'f1_score' in the same parquet file.
    """
    calculate_evaluation_metrics(parquet_path)


@click.command()
@click.argument("excel_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save evaluation results and plots",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Disable plot generation",
)
@click.option(
    "--mlflow-run-id",
    type=str,
    default=None,
    help="MLflow run ID to log metrics to an existing run",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default=None,
    help="MLflow experiment name (creates new run if --mlflow-run-id not provided)",
)
@click.option(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="MLflow run name for new runs",
)
def evaluate_qa_model(
    excel_path: Path,
    output_dir: Optional[Path] = None,
    no_plots: bool = False,
    mlflow_run_id: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
):
    """
    Evaluate QA model predictions from an Excel file.

    The Excel file should contain sheets for train, validation, and test splits,
    with columns: cell_id, Jaccard index, Predicted Jaccard index.

    Calculates R², MAE, RMSE, tolerance-based accuracy, and generates plots.

    Optionally logs metrics to MLflow (either to an existing run or a new one).

    NOTE: This evaluates the QA MODEL predictions, not the final ensemble results.
    """
    results = evaluate_qa_model_from_excel(
        excel_path=excel_path,
        output_dir=output_dir,
        generate_plots=not no_plots,
    )

    # Log to MLflow if requested
    if mlflow_run_id or mlflow_experiment:
        _log_qa_metrics_to_mlflow(
            results=results,
            excel_path=excel_path,
            output_dir=output_dir,
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment=mlflow_experiment,
            mlflow_run_name=mlflow_run_name,
        )


def _log_qa_metrics_to_mlflow(
    results: dict,
    excel_path: Path,
    output_dir: Optional[Path],
    mlflow_run_id: Optional[str],
    mlflow_experiment: Optional[str],
    mlflow_run_name: Optional[str],
):
    """Log QA evaluation metrics to MLflow."""

    def log_metrics_for_split(split_results: dict, split_name: str):
        """Flatten and log metrics for a single split."""
        metrics = {}
        for key, value in split_results.items():
            if isinstance(value, (int, float)) and key != "split":
                # Prefix with split name for clarity
                metric_name = f"{split_name}_{key}"
                metrics[metric_name] = value
        if metrics:
            mlflow.log_metrics(metrics)

    if mlflow_run_id:
        # Log to existing run
        with mlflow.start_run(run_id=mlflow_run_id):
            logging.info(f"Logging QA metrics to existing MLflow run: {mlflow_run_id}")
            for split_name, split_results in results.items():
                log_metrics_for_split(split_results, split_name)

            # Log artifacts
            mlflow.log_artifact(str(excel_path))
            if output_dir and output_dir.exists():
                mlflow.log_artifacts(str(output_dir), artifact_path="evaluation")
    else:
        # Create new run
        if mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name=mlflow_run_name):
            logging.info(f"Created new MLflow run: {mlflow.active_run().info.run_id}")

            # Log the excel path as a parameter
            mlflow.log_param("excel_path", str(excel_path))

            for split_name, split_results in results.items():
                log_metrics_for_split(split_results, split_name)

            # Log artifacts
            mlflow.log_artifact(str(excel_path))
            if output_dir and output_dir.exists():
                mlflow.log_artifacts(str(output_dir), artifact_path="evaluation")

            logging.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")


@click.command()
@click.argument("parquet_path", type=click.Path(exists=True, path_type=Path))
@click.argument("excel_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for merged parquet file",
)
def merge_qa_predictions(
    parquet_path: Path,
    excel_path: Path,
    output: Optional[Path] = None,
):
    """
    Merge QA model predictions from Excel into an existing parquet file.

    Adds 'predicted_jaccard_index' column to the parquet file by matching cell_id.
    This is useful for analyzing model performance across train/val/test splits.
    """
    merge_predictions_to_parquet(
        parquet_path=parquet_path,
        excel_path=excel_path,
        output_path=output,
    )


@click.group()
def cli():
    """Main entry point for command-line tools."""
    pass


cli.add_command(evaluate_competitor)
cli.add_command(calculate_evaluation_metrics_cli)
cli.add_command(evaluate_qa_model)
cli.add_command(merge_qa_predictions)


if __name__ == "__main__":
    cli()
