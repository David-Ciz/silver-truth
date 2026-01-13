import click
from pathlib import Path
from typing import Optional

from silver_truth.qa.preprocessing import create_qa_dataset
from silver_truth.qa.evaluation import run_qa_evaluation
from silver_truth.qa.result_conversion import (
    convert_qa_csv_to_detailed_parquet,
    create_parquet_from_qa_results,
)


@click.group()
def cli():
    """A CLI tool for the Quality Assurance (QA) workflow."""
    pass


@cli.command()
@click.argument("dataset_dataframe_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("output_parquet_path", type=click.Path(file_okay=True, dir_okay=False))
@click.option(
    "--crop", default=False, is_flag=True, help="Create crops for the QA dataset"
)
@click.option("--crop-size", default=64, help="Size of the crops for the QA dataset")
def create_dataset(
    dataset_dataframe_path: str,
    output_dir: str,
    output_parquet_path: str,
    crop: bool = False,
    crop_size: int = 64,
) -> None:
    """Creates a QA dataset for cell-level analysis."""
    create_qa_dataset(
        dataset_dataframe_path, output_dir, output_parquet_path, crop, crop_size
    )


@cli.command()
@click.argument("qa_dataframe_path", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "ground_truth_dataframe_path", type=click.Path(exists=True, path_type=Path)
)
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
def evaluate(
    qa_dataframe_path: Path,
    ground_truth_dataframe_path: Path,
    competitor: Optional[str] = None,
    output: Optional[Path] = None,
    visualize: bool = False,
):
    """Evaluates a QA dataset."""
    run_qa_evaluation(
        qa_dataframe_path,
        ground_truth_dataframe_path,
        competitor,
        output,
        visualize,
    )


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("qa_dataframe_path", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output parquet file path")
def convert_results(
    csv_path: str, qa_dataframe_path: str, output: Optional[str] = None
):
    """
    Convert QA evaluation CSV results to detailed parquet format.
    """
    result = convert_qa_csv_to_detailed_parquet(csv_path, qa_dataframe_path, output)
    if result:
        print(f"✅ Conversion successful: {result}")
    else:
        print("❌ Conversion failed")


@cli.command()
@click.option(
    "--qa-results-dir",
    default=".",
    help="Directory containing QA CSV result files",
)
@click.option(
    "--qa-dataframes-dir",
    default=".",
    help="Directory containing QA dataframe parquet files",
)
@click.option(
    "--output-dir",
    default="detailed_qa_results",
    help="Directory to save converted parquet files",
)
def batch_convert_results(qa_results_dir: str, qa_dataframes_dir: str, output_dir: str):
    """
    Batch convert all QA CSV results to detailed parquet format.
    """
    created_files = create_parquet_from_qa_results(
        qa_results_dir, qa_dataframes_dir, output_dir
    )

    print("\n✅ Batch conversion completed!")
    print(f"📁 Created {len(created_files)} detailed parquet files in: {output_dir}")

    if created_files:
        print("\n📄 Created files:")
        for file in created_files:
            print(f"   - {Path(file).name}")


if __name__ == "__main__":
    cli()
