import click
from pathlib import Path
from typing import Optional

from src.metrics.qa_evaluation_logic import run_qa_evaluation


@click.command()
@click.argument("qa_dataframe_path", type=click.Path(exists=True, path_type=Path))
@click.argument("ground_truth_dataframe_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--competitor", 
    help="Competitor name to evaluate. If None, evaluate all."
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
def evaluate_qa_dataset(
    qa_dataframe_path: Path,
    ground_truth_dataframe_path: Path,
    competitor: Optional[str] = None,
    output: Optional[Path] = None,
    visualize: bool = False,
):
    """
    Evaluates QA cropped images against ground truth using Jaccard index.
    
    This script evaluates the stacked TIFF files created by the QA data preprocessor
    against the original ground truth segmentation images.
    
    Args:
        qa_dataframe_path: Path to the QA dataframe (parquet file with cropped images metadata)
        ground_truth_dataframe_path: Path to the original dataset dataframe with GT information
    """
    run_qa_evaluation(
        qa_dataframe_path=qa_dataframe_path,
        ground_truth_dataframe_path=ground_truth_dataframe_path,
        competitor=competitor,
        output=output,
        visualize=visualize,
    )


if __name__ == "__main__":
    evaluate_qa_dataset()
