import click
from pathlib import Path
from typing import Optional

from silver_truth.qa.preprocessing import (
    attach_split_to_qa_dataset,
    create_qa_dataset,
)
from silver_truth.qa.evaluation import run_qa_evaluation
from silver_truth.qa.result_conversion import (
    convert_qa_csv_to_detailed_parquet,
    create_parquet_from_qa_results,
)
from silver_truth.experiment_tracking import DEFAULT_MLFLOW_TRACKING_URI


@click.group()
def cli():
    """A CLI tool for the Quality Assurance (QA) workflow."""
    pass


@cli.command()
@click.option(
    "--dataset-dataframe-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the dataset dataframe",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory to write dataset outputs",
)
@click.option(
    "--output-parquet-path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to write output parquet file",
)
@click.option(
    "--crop", default=False, is_flag=True, help="Create crops for the QA dataset"
)
@click.option("--crop-size", default=64, help="Size of the crops for the QA dataset")
@click.option(
    "--centering",
    type=click.Choice(
        [
            "competitor",
            "competitor_consensus",
            "competitor_individual",
            "gt_mask",
            "tracking_marker",
        ]
    ),
    default="competitor",
    help="Strategy for centering crops.",
)
@click.option(
    "--exclude-competitors",
    multiple=True,
    help="Competitors to exclude from the dataset.",
)
def create_dataset(
    dataset_dataframe_path: str,
    output_dir: str,
    output_parquet_path: str,
    crop: bool = False,
    crop_size: int = 64,
    centering: str = "competitor",
    exclude_competitors: tuple = (),
) -> None:
    """Creates a QA dataset for cell-level analysis.

    CENTERING STRATEGIES:
    - competitor (default): Alias of competitor_consensus.
    - competitor_consensus: Center by agreement of all competitors for each cell label.
    - competitor_individual: Center each crop independently per competitor.
    - gt_mask: Center on the Ground Truth mask centroid. Best for alignment but potential data leakage.
    - tracking_marker: Center on the Tracking Marker (TRA) centroid. Good compromise for alignment.
    """
    create_qa_dataset(
        dataset_dataframe_path,
        output_dir,
        output_parquet_path,
        crop,
        crop_size,
        centering,
        list(exclude_competitors),
    )


@cli.command()
@click.option(
    "--qa-base-parquet",
    required=True,
    type=click.Path(exists=True),
    help="Path to base QA parquet file (generated once per dataset/crop-size).",
)
@click.option(
    "--dataset-dataframe-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to split-specific whole-image dataset parquet.",
)
@click.option(
    "--output-parquet-path",
    required=True,
    type=click.Path(file_okay=True, dir_okay=False),
    help="Path to write split-annotated QA parquet file.",
)
def attach_split(
    qa_base_parquet: str,
    dataset_dataframe_path: str,
    output_parquet_path: str,
) -> None:
    """Attach split labels to an existing QA parquet.
    DEPRECATED"""
    attach_split_to_qa_dataset(
        qa_base_parquet,
        dataset_dataframe_path,
        output_parquet_path,
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


@cli.group()
def cnn() -> None:
    """CNN training and evaluation for QA Jaccard prediction."""
    pass


@cnn.command("train")
@click.option(
    "--parquet-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to input parquet file with split column.",
)
@click.option(
    "--data-root",
    type=click.Path(exists=True),
    default=None,
    help="Root directory prepended to stacked_path.",
)
@click.option(
    "--target-column",
    type=str,
    default=None,
    help="Target column with ground-truth jaccard values (auto-detected if omitted).",
)
@click.option(
    "--input-channels",
    type=str,
    default="0,1",
    show_default=True,
    help="Comma-separated channel indices from stacked TIFF used as model input.",
)
@click.option(
    "--output-model",
    type=click.Path(),
    default="cnn_jaccard.pt",
    help="Path to save the trained model checkpoint.",
)
@click.option(
    "--output-excel",
    type=click.Path(),
    default="results_cnn.xlsx",
    help="Path to save evaluation results.",
)
@click.option("--batch-size", type=int, default=16, help="Batch size for training.")
@click.option("--learning-rate", type=float, default=1e-4, help="Learning rate.")
@click.option("--num-epochs", type=int, default=50, help="Number of training epochs.")
@click.option(
    "--weight-decay",
    type=float,
    default=1e-4,
    help="Weight decay (L2 regularization).",
)
@click.option("--dropout-rate", type=float, default=0.3, help="Dropout rate.")
@click.option(
    "--model-type",
    type=click.Choice(
        [
            "resnet18",
            "resnet50",
            "resnet101",
            "efficientnet_b1",
            "efficientnet_b4",
            "efficientnet_b7",
        ]
    ),
    default="resnet50",
    help="Model architecture to use.",
)
@click.option(
    "--patience", type=int, default=10, help="Early stopping patience (0 disables)."
)
@click.option(
    "--augment/--no-augment", default=True, help="Enable or disable data augmentation."
)
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option(
    "--num-workers", type=int, default=4, help="Number of DataLoader workers."
)
@click.option(
    "--grad-clip",
    type=float,
    default=1.0,
    help="Gradient clipping max norm (0 disables).",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=DEFAULT_MLFLOW_TRACKING_URI,
    show_default=True,
    help="MLflow tracking URI.",
)
@click.option(
    "--mlflow-experiment", type=str, default="cnn-jaccard", help="MLflow experiment."
)
@click.option(
    "--mlflow-run-name", type=str, default=None, help="Optional MLflow run name."
)
def cnn_train(
    parquet_file: str,
    data_root: Optional[str],
    target_column: Optional[str],
    input_channels: str,
    output_model: str,
    output_excel: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    weight_decay: float,
    dropout_rate: float,
    model_type: str,
    patience: int,
    augment: bool,
    seed: int,
    num_workers: int,
    grad_clip: float,
    mlflow_tracking_uri: Optional[str],
    mlflow_experiment: str,
    mlflow_run_name: Optional[str],
) -> None:
    """Train the QA CNN model."""
    from silver_truth.qa import cnn as qa_cnn

    qa_cnn.train(
        parquet_file=parquet_file,
        data_root=data_root,
        target_column=target_column,
        input_channels=input_channels,
        output_model=output_model,
        output_excel=output_excel,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        patience=patience,
        augment=augment,
        seed=seed,
        num_workers=num_workers,
        grad_clip=grad_clip,
        model_type=model_type,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
    )


@cnn.command("evaluate")
@click.option(
    "--parquet-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to input parquet file with split column.",
)
@click.option(
    "--data-root",
    type=click.Path(exists=True),
    default=None,
    help="Root directory prepended to stacked_path.",
)
@click.option(
    "--target-column",
    type=str,
    default=None,
    help="Target column with ground-truth jaccard values (auto-detected if omitted).",
)
@click.option(
    "--input-channels",
    type=str,
    default="0,1",
    show_default=True,
    help="Comma-separated channel indices from stacked TIFF used as model input.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model checkpoint.",
)
@click.option(
    "--output-excel",
    type=click.Path(),
    default="results_cnn.xlsx",
    help="Path to save evaluation results.",
)
@click.option("--batch-size", type=int, default=16, help="Batch size for evaluation.")
def cnn_evaluate(
    parquet_file: str,
    data_root: Optional[str],
    target_column: Optional[str],
    input_channels: str,
    model_path: str,
    output_excel: str,
    batch_size: int,
) -> None:
    """Evaluate a trained QA CNN model."""
    from silver_truth.qa import cnn as qa_cnn

    qa_cnn.evaluate(
        parquet_file=parquet_file,
        data_root=data_root,
        target_column=target_column,
        input_channels=input_channels,
        model_path=model_path,
        output_excel=output_excel,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    cli()
