import click
import logging
from pathlib import Path
from typing import Optional
import silver_truth.ensemble.ensemble as ensemble
import silver_truth.ensemble.utils as utils
from silver_truth.ensemble.datasets import Version
from silver_truth.ensemble.models import ModelType


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _parse_split_sets(split_sets: str) -> list[float]:
    values = [float(v.strip()) for v in split_sets.split(",")]
    if len(values) != 3:
        raise click.BadParameter(
            "split-sets must have exactly 3 values: train,val,test"
        )
    if abs(sum(values) - 1.0) > 1e-8:
        raise click.BadParameter("split-sets must sum to 1.0")
    return values


@click.command("ensemble-experiment")
@click.option("--name", required=True, help="The name of the Ensemble experiment.")
@click.option(
    "--parquet-file",
    "--parquet_file",
    "parquet_file",
    required=True,
    help="The path of the Ensemble databank parquet file.",
)
@click.option(
    "--model-type",
    type=click.Choice([m.name for m in ModelType], case_sensitive=False),
    default="UnetPlusPlus",
    show_default=True,
    help="Model architecture to train.",
)
@click.option(
    "--max-epochs",
    type=int,
    default=100,
    show_default=True,
    help="Maximum training epochs.",
)
def ensemble_experiment(
    name: str,
    parquet_file: str,
    model_type: str,
    max_epochs: int,
):
    """Runs an Ensemble experiment via command-line interface."""
    try:
        databank_name = Path(parquet_file).stem
        ensemble.run_experiment(
            name,
            databank_name,
            parquet_file,
            [{"model_type": ModelType[model_type], "max_epochs": max_epochs}],
        )
    except Exception as e:
        click.echo(
            click.style(f"An unexpected error occurred: {e}", fg="red", bold=True)
        )
        exit(1)


@click.command("build-databank")
@click.option(
    "--dataset-name",
    required=True,
    type=click.Choice(sorted(utils.ORIGINAL_DATASETS.keys()), case_sensitive=False),
    help="Dataset name used in the QA parquet.",
)
@click.option(
    "--qa-parquet-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Split-aware QA parquet path.",
)
@click.option(
    "--version",
    default="C1",
    type=click.Choice([v.name for v in Version], case_sensitive=False),
    show_default=True,
    help="Ensemble dataset version.",
)
@click.option(
    "--crop-size",
    type=int,
    default=64,
    show_default=True,
    help="Expected crop size metadata.",
)
@click.option(
    "--split-seed",
    type=int,
    default=42,
    show_default=True,
    help="Split random seed.",
)
@click.option(
    "--split-sets",
    default="0.7,0.15,0.15",
    show_default=True,
    help="Comma-separated train,val,test split ratios that sum to 1.0.",
)
@click.option(
    "--qa-column",
    default=None,
    help="Optional QA score column name used for gating (e.g., QA-eb7-1).",
)
@click.option(
    "--qa-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="QA threshold used when --qa-column is provided.",
)
def build_databank(
    dataset_name: str,
    qa_parquet_path: str,
    version: str,
    crop_size: int,
    split_seed: int,
    split_sets: str,
    qa_column: Optional[str],
    qa_threshold: float,
) -> None:
    """Build an Ensemble databank parquet and image folder from a QA parquet."""
    build_opt = {
        "name": dataset_name,
        "version": Version[version],
        "crop_size": crop_size,
        "split_seed": split_seed,
        "split_sets": _parse_split_sets(split_sets),
        "qa": qa_column if qa_column else None,
        "qa_threshold": qa_threshold if qa_column else None,
    }
    output_parquet = ensemble.build_databank(build_opt, qa_parquet_path)
    click.echo(output_parquet)


@click.command("evaluate-checkpoint")
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a trained .ckpt checkpoint.",
)
@click.option(
    "--databank-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an Ensemble databank parquet.",
)
@click.option(
    "--split-type",
    type=click.Choice(["train", "validation", "test", "all"], case_sensitive=False),
    default="test",
    show_default=True,
    help="Which split to evaluate.",
)
def evaluate_checkpoint(model_path: str, databank_path: str, split_type: str) -> None:
    """Run inference from a checkpoint and print mean IoU/F1."""
    summary = ensemble.evaluate_checkpoint(model_path, databank_path, split_type)
    click.echo(f"output_parquet: {summary['output_parquet_path']}")
    click.echo(
        f"split={summary['split']} count={summary['count']} "
        f"iou_mean={summary['iou_mean']:.6f} f1_mean={summary['f1_mean']:.6f}"
    )


@click.command("evaluate-best-checkpoint")
@click.option(
    "--checkpoints-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing one or more .ckpt files.",
)
@click.option(
    "--databank-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to an Ensemble databank parquet.",
)
@click.option(
    "--split-type",
    type=click.Choice(["train", "validation", "test", "all"], case_sensitive=False),
    default="test",
    show_default=True,
    help="Which split to evaluate.",
)
@click.option(
    "--pattern",
    default="*.ckpt",
    show_default=True,
    help="Glob pattern used to locate checkpoints under checkpoints-dir.",
)
def evaluate_best_checkpoint(
    checkpoints_dir: str, databank_path: str, split_type: str, pattern: str
) -> None:
    """Pick the newest checkpoint in a folder and evaluate it."""
    ckpt_candidates = sorted(
        Path(checkpoints_dir).glob(pattern), key=lambda p: p.stat().st_mtime
    )
    if not ckpt_candidates:
        raise click.ClickException(
            f"No checkpoints found in '{checkpoints_dir}' with pattern '{pattern}'."
        )
    best_ckpt = str(ckpt_candidates[-1])
    click.echo(f"selected_checkpoint: {best_ckpt}")
    summary = ensemble.evaluate_checkpoint(best_ckpt, databank_path, split_type)
    click.echo(f"output_parquet: {summary['output_parquet_path']}")
    click.echo(
        f"split={summary['split']} count={summary['count']} "
        f"iou_mean={summary['iou_mean']:.6f} f1_mean={summary['f1_mean']:.6f}"
    )


@click.group()
def cli():
    """Main entry point for command-line tools."""
    pass


cli.add_command(ensemble_experiment)
cli.add_command(build_databank)
cli.add_command(evaluate_checkpoint)
cli.add_command(evaluate_best_checkpoint)

if __name__ == "__main__":
    cli()
