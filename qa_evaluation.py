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
    "--parquet-output",
    type=click.Path(path_type=Path),
    help="Path to save detailed results as parquet",
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
    parquet_output: Optional[Path] = None,
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
    # Run standard QA evaluation
    result = run_qa_evaluation(
        qa_dataframe_path=qa_dataframe_path,
        ground_truth_dataframe_path=ground_truth_dataframe_path,
        competitor=competitor,
        output=output,
        visualize=visualize,
    )
    
    # If CSV output was created and parquet output is requested, convert to parquet
    if output and output.exists() and parquet_output:
        try:
            from convert_qa_to_parquet import convert_qa_csv_to_detailed_parquet
            
            print(f"Converting results to detailed parquet format...")
            convert_qa_csv_to_detailed_parquet(
                str(output), 
                str(qa_dataframe_path), 
                str(parquet_output)
            )
            print(f"✅ Detailed parquet results saved to: {parquet_output}")
            
        except ImportError:
            print("❌ convert_qa_to_parquet module not found. Parquet conversion skipped.")
        except Exception as e:
            print(f"❌ Error during parquet conversion: {e}")
    
    return result


if __name__ == "__main__":
    evaluate_qa_dataset()
