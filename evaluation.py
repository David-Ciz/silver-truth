import click
from pathlib import Path
from typing import Optional

from src.metrics.evaluation_logic import run_evaluation


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
def evaluate_competitor(
    dataset_dataframe_path: Path,
    competitor: Optional[str] = None,
    output: Optional[Path] = None,
    visualize: bool = False,
    campaign_col: str = "campaign_number",
):
    """
    Evaluates competitor segmentation results against ground truth using Jaccard index.

    This script is a wrapper around the core evaluation logic in `run_evaluation`.
    """
    run_evaluation(
        dataset_dataframe_path=dataset_dataframe_path,
        competitor=competitor,
        output=output,
        visualize=visualize,
        campaign_col=campaign_col,
    )


@click.group()
def cli():
    """Main entry point for command-line tools."""
    pass


cli.add_command(evaluate_competitor)


if __name__ == "__main__":
    cli()
