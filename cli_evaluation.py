import click
from pathlib import Path
from typing import Optional
import logging

from src.evaluation.evaluation_logic import evaluate_competitor_logic
from src.evaluation.stacked_jaccard_logic import calculate_evaluation_metrics
from src.metrics.qa_model_evaluation import (
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
def evaluate_qa_model(
    excel_path: Path,
    output_dir: Optional[Path] = None,
    no_plots: bool = False,
):
    """
    Evaluate QA model predictions from an Excel file.

    The Excel file should contain sheets for train, validation, and test splits,
    with columns: cell_id, Jaccard index, Predicted Jaccard index.

    Calculates R², MAE, RMSE, tolerance-based accuracy, and generates plots.

    NOTE: This evaluates the QA MODEL predictions, not the final ensemble results.
    """
    evaluate_qa_model_from_excel(
        excel_path=excel_path,
        output_dir=output_dir,
        generate_plots=not no_plots,
    )


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
