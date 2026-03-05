import logging
from typing import Optional
from pathlib import Path

import click

from silver_truth.fusion.fusion import (
    FusionModel,
    fuse_segmentations,
    add_fused_images_to_dataframe_logic,
    process_all_datasets_logic,
)
from silver_truth.fusion.crops_experiment import (
    ALL_MODELS as CROP_EXPERIMENT_MODELS,
    MLFLOW_TRACKING_PATH as DEFAULT_MLFLOW_TRACKING_PATH,
    run_crops_fusion_experiment,
)
from silver_truth.job_file_generator import generate_job_file

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.group()
def cli():
    """A CLI tool for cell tracking fusion and job file generation."""
    pass


@cli.command()
@click.option(
    "--jar-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the executable Java JAR file (e.g., fusers-all-dependencies.jar).",
    default="src/silver_truth/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar",
)
@click.option(
    "--job-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the job specification file listing input image patterns.",
)
@click.option(
    "--output-pattern",
    required=True,
    help="Output filename pattern, including 'TTT' or 'TTTT' (e.g., '/path/to/fused_TTT.tif').",
)
@click.option(
    "--time-points",
    required=True,
    default="0-90000",
    help='Timepoints to process as a string (e.g., "1-9,23,25").',
)
@click.option(
    "--num-threads", required=True, type=int, help="Number of processing threads."
)
@click.option(
    "--model",
    required=True,
    type=click.Choice([e.name for e in FusionModel], case_sensitive=False),
    help="The fusion model to use.",
)
@click.option(
    "--threshold",
    default=1.0,
    type=float,
    show_default=True,
    help="Voting threshold for merging.",
)
@click.option(
    "--cmv-mode",
    default=None,
    help='Enable Combinatorial Model Validation mode (e.g., "CMV", "CMV2_8").',
)
@click.option(
    "--seg-folder",
    default=None,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Optional path to ground truth folder for scoring.",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable debug logging and show Java process output.",
)
def run_fusion(
    jar_path: str,
    job_file: str,
    output_pattern: str,
    time_points: str,
    num_threads: int,
    model: str,
    threshold: float,
    cmv_mode: Optional[str],
    seg_folder: Optional[str],
    debug: bool,
):
    """
    Runs the Fusers Java segmentation fusion tool via a command-line interface.
    """
    try:
        # Convert the string model name from the CLI back to the Enum member
        fusion_model_enum = FusionModel[model.upper()]

        click.echo(
            click.style(
                f"Starting fusion process with model: {fusion_model_enum.value}",
                fg="green",
            )
        )

        fuse_segmentations(
            jar_path=jar_path,
            job_file_path=job_file,
            output_path_pattern=output_pattern,
            time_points=time_points,
            num_threads=num_threads,
            fusion_model=fusion_model_enum,
            threshold=threshold,
            cmv_mode=cmv_mode,
            seg_eval_folder=seg_folder,
            debug=debug,
        )

        click.echo(
            click.style("Fusion process completed successfully!", fg="green", bold=True)
        )

    except RuntimeError as e:
        click.echo(click.style(f"Error: {e}", fg="red", bold=True))
        # Exit with a non-zero status code to indicate failure
        exit(1)
    except Exception as e:
        click.echo(
            click.style(f"An unexpected error occurred: {e}", fg="red", bold=True)
        )
        exit(1)


@cli.command()
@click.option(
    "--parquet-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the Parquet dataset file (e.g., BF-C2DL-HSC_dataset_dataframe.parquet).",
)
@click.option(
    "--campaign-number",
    required=True,
    help="The campaign number (e.g., '01').",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, resolve_path=True),
    help="Directory where the job file will be saved.",
)
@click.option(
    "--tracking-marker-column",
    default="tracking_markers",
    show_default=True,
    help="The column name in the parquet file that contains the tracking marker paths.",
)
@click.option(
    "--competitor-columns",
    multiple=True,
    help="Column names in the parquet file that contain competitor result paths. Can be specified multiple times. If not provided, will use competitor_config.json or all columns except known non-competitor columns.",
)
@click.option(
    "--config-path",
    default=None,
    help="Path to the competitor configuration JSON file. If not provided, will auto-determine based on campaign number (e.g., competitor_config_campaign01.json).",
)
def generate_jobfiles(
    parquet_file: str,
    campaign_number: str,
    output_dir: str,
    tracking_marker_column: str,
    competitor_columns: tuple[str],
    config_path: str,
):
    """
    Generates a job file for a specific dataset.
    """
    try:
        # Convert tuple to list for generate_job_file function
        competitor_cols_list = list(competitor_columns) if competitor_columns else None

        generate_job_file(
            parquet_file_path=parquet_file,
            campaign_number=campaign_number,
            output_dir=output_dir,
            tracking_marker_column=tracking_marker_column,
            competitor_columns=competitor_cols_list,
            config_path=config_path,
        )
        click.echo(
            click.style(
                "Job file generation completed successfully!", fg="green", bold=True
            )
        )
    except Exception as e:
        click.echo(click.style(f"Error generating job file: {e}", fg="red", bold=True))
        exit(1)


@cli.command()
@click.option("--dataset", type=str, help="Dataset name to help infer defaults")
@click.option(
    "--input-parquet",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Explicit path to the dataset parquet file",
)
@click.option(
    "--fused-results-dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Directory containing fused TIFF outputs",
)
@click.option(
    "--output-parquet",
    type=click.Path(dir_okay=False, resolve_path=True),
    help="Destination parquet path; defaults alongside input",
)
@click.option(
    "--base-dir",
    type=click.Path(file_okay=False, resolve_path=True),
    default=None,
    help="Legacy base directory containing dataframes/ and fused_results/",
)
@click.option(
    "--model-name",
    type=str,
    default=None,
    help="Name of the fusion model (for column naming). If not provided, will be inferred from directory structure.",
)
@click.option(
    "--process-all", is_flag=True, help="Process every dataset under base-dir"
)
def add_fused_images(
    dataset,
    input_parquet,
    fused_results_dir,
    output_parquet,
    base_dir,
    model_name,
    process_all,
):
    """Add fused image paths to dataset dataframes."""
    if process_all:
        if not base_dir:
            click.echo("--process-all requires --base-dir to locate datasets", err=True)
            exit(1)
        process_all_datasets_logic(base_dir)
        return

    if not dataset and not input_parquet:
        click.echo(
            "Provide at least --dataset or --input-parquet so the dataframe can be located",
            err=True,
        )
        exit(1)

    success = add_fused_images_to_dataframe_logic(
        dataset_name=dataset,
        input_parquet_path=input_parquet,
        fused_results_dir=fused_results_dir,
        output_parquet_path=output_parquet,
        base_dir=base_dir,
        model_name=model_name,
    )
    if not success:
        exit(1)


@cli.command()
@click.option(
    "--qa-parquet",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to QA crops parquet.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="fusion_results_crops",
    show_default=True,
    help="Directory for fusion outputs and summary CSVs.",
)
@click.option(
    "--models",
    multiple=True,
    type=click.Choice(CROP_EXPERIMENT_MODELS, case_sensitive=False),
    help="Fusion models to run. Can be specified multiple times.",
)
@click.option("--all-models", is_flag=True, help="Run all available fusion models.")
@click.option(
    "--flat-models-only",
    is_flag=True,
    help="Run only flat (non-weighted) fusion models.",
)
@click.option(
    "--weights-column",
    default=None,
    help=(
        "Optional QA parquet column containing competitor weights. "
        "If not provided/found, weighted fusion models are skipped."
    ),
)
@click.option(
    "--num-threads",
    default=4,
    show_default=True,
    type=int,
    help="Fusion thread count.",
)
@click.option(
    "--chunk-size",
    default=500,
    show_default=True,
    type=int,
    help="Number of synthetic timepoints per Java invocation.",
)
@click.option(
    "--mlflow-experiment",
    default="fusion-crops-baseline",
    show_default=True,
    help="MLflow experiment name.",
)
@click.option(
    "--mlflow-tracking-path",
    default=DEFAULT_MLFLOW_TRACKING_PATH,
    show_default=True,
    help="MLflow tracking directory.",
)
@click.option(
    "--skip-fusion",
    is_flag=True,
    help="Skip fusion and only evaluate existing outputs.",
)
@click.option(
    "--keep-job-dir", is_flag=True, help="Keep generated fusion job directory."
)
@click.option(
    "--job-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Explicit job directory. If provided, it is reused/updated and not removed.",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable debug output from the Java fusion process.",
)
def run_fusion_crops(
    qa_parquet: Path,
    output_dir: Path,
    models: tuple[str, ...],
    all_models: bool,
    flat_models_only: bool,
    weights_column: Optional[str],
    num_threads: int,
    chunk_size: int,
    mlflow_experiment: str,
    mlflow_tracking_path: str,
    skip_fusion: bool,
    keep_job_dir: bool,
    job_dir: Optional[Path],
    debug: bool,
):
    """Run fusion on QA crops with MLflow tracking."""
    try:
        result = run_crops_fusion_experiment(
            qa_parquet=qa_parquet,
            output_dir=output_dir,
            models=models,
            all_models=all_models,
            flat_models_only=flat_models_only,
            weights_column=weights_column,
            num_threads=num_threads,
            chunk_size=chunk_size,
            mlflow_experiment=mlflow_experiment,
            mlflow_tracking_path=mlflow_tracking_path,
            skip_fusion=skip_fusion,
            keep_job_dir=keep_job_dir,
            job_dir=job_dir,
            debug=debug,
        )
    except Exception as e:
        click.echo(
            click.style(
                f"Error while running fusion crops experiment: {e}",
                fg="red",
                bold=True,
            )
        )
        exit(1)

    click.echo(click.style("Fusion crops experiment completed.", fg="green", bold=True))
    click.echo(f"MLflow run ID: {result['mlflow_run_id']}")
    click.echo(f"Summary CSV: {result['summary_path']}")
    if result.get("leaderboard_path"):
        click.echo(f"Leaderboard CSV: {result['leaderboard_path']}")
    click.echo(f"Output dir: {result['output_dir']}")
    if result.get("job_dir"):
        click.echo(f"Job dir: {result['job_dir']}")


if __name__ == "__main__":
    cli()
