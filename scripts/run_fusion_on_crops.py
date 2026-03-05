#!/usr/bin/env python3
"""Run fusion experiments on QA crops with MLflow tracking."""

import logging
from pathlib import Path
from typing import Optional

import click

from silver_truth.fusion.crops_experiment import (
    ALL_MODELS,
    run_crops_fusion_experiment,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.command()
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
    type=click.Choice(ALL_MODELS, case_sensitive=False),
    help="Fusion models to run. Can be specified multiple times.",
)
@click.option("--all-models", is_flag=True, help="Run all available fusion models.")
@click.option(
    "--weights-column",
    default=None,
    help=(
        "Optional QA parquet column containing competitor weights. "
        "If not provided/found, weighted fusion models are skipped."
    ),
)
@click.option(
    "--num-threads", default=4, show_default=True, help="Fusion thread count."
)
@click.option(
    "--chunk-size",
    default=0,
    show_default=True,
    help="Number of synthetic timepoints per Java invocation (<=0 disables chunking).",
)
@click.option(
    "--mlflow-experiment",
    default="fusion-crops-baseline",
    show_default=True,
    help="MLflow experiment name.",
)
@click.option(
    "--mlflow-tracking-path",
    default="data/fusion_experiments/mlruns",
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
def main(
    qa_parquet: Path,
    output_dir: Path,
    models: tuple[str, ...],
    all_models: bool,
    weights_column: Optional[str],
    num_threads: int,
    chunk_size: int,
    mlflow_experiment: str,
    mlflow_tracking_path: str,
    skip_fusion: bool,
    keep_job_dir: bool,
    job_dir: Optional[Path],
    debug: bool,
) -> None:
    """Run fusion on QA crops for multiple models."""
    try:
        result = run_crops_fusion_experiment(
            qa_parquet=qa_parquet,
            output_dir=output_dir,
            models=models,
            all_models=all_models,
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
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("\nFusion crops experiment finished.")
    click.echo(f"MLflow run ID: {result['mlflow_run_id']}")
    click.echo(f"Summary CSV: {result['summary_path']}")
    if result.get("leaderboard_path"):
        click.echo(f"Leaderboard CSV: {result['leaderboard_path']}")
    click.echo(f"Output dir: {result['output_dir']}")
    if result.get("job_dir"):
        click.echo(f"Job dir: {result['job_dir']}")


if __name__ == "__main__":
    main()
