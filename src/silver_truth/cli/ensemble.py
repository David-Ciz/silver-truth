import click
import logging
import mlflow
from pathlib import Path
from typing import Optional
import silver_truth.ensemble.ensemble as ensemble
import silver_truth.ensemble.utils as utils
from silver_truth.ensemble.datasets import Version
from silver_truth.ensemble.models import ModelType
from silver_truth.experiment_tracking import (
    DEFAULT_MLFLOW_TRACKING_URI,
    infer_dataset_name_from_text,
    log_standardized_single_split_metrics,
    start_managed_mlflow_run,
    set_common_mlflow_tags,
    set_evaluation_tags,
)


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _log_ensemble_evaluation_to_mlflow(
    *,
    summary: dict,
    databank_path: str,
    model_path: str,
    mlflow_tracking_uri: str,
    mlflow_experiment: Optional[str],
    mlflow_run_name: Optional[str],
    setup_name: Optional[str],
    qa_mode: Optional[str],
    qa_threshold: Optional[float],
) -> None:
    if not mlflow_experiment:
        return

    dataset_tag = infer_dataset_name_from_text([databank_path, model_path])
    with start_managed_mlflow_run(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        run_name=mlflow_run_name or setup_name or Path(model_path).stem,
    ):
        set_common_mlflow_tags(
            dataset=dataset_tag,
            split=str(summary.get("evaluation_level", "unknown")),
        )
        set_evaluation_tags(
            pipeline_family="ensemble",
            evaluation_level=str(summary.get("evaluation_level", "unknown")),
            setup_name=setup_name or Path(model_path).stem,
            qa_mode=qa_mode,
            qa_threshold=qa_threshold,
        )
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("databank_path", databank_path)
        mlflow.log_param("output_parquet_path", str(summary["output_parquet_path"]))
        log_standardized_single_split_metrics(
            split=str(summary["split"]),
            iou=float(summary["iou_mean"]),
            f1=float(summary["f1_mean"]),
            count=int(summary["count"]),
        )
        if "cell_iou_mean" in summary:
            mlflow.log_metric("cell_test_iou", float(summary["cell_iou_mean"]))
        if "cell_f1_mean" in summary:
            mlflow.log_metric("cell_test_f1", float(summary["cell_f1_mean"]))
        if "cell_count" in summary:
            mlflow.log_metric("cell_eval_count", float(summary["cell_count"]))
        mlflow.log_artifact(str(summary["output_parquet_path"]))


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
@click.option(
    "--checkpoints-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Directory to write model checkpoints into.  "
        "Defaults to the package default (data/ensemble_data/results/checkpoints/{databank_name}).  "
        "Pass an explicit path to keep multiple experiments' checkpoints separate."
    ),
)
@click.option(
    "--encoder-name",
    type=str,
    default="resnet34",
    show_default=True,
    help="Encoder backbone name (any smp-supported encoder, e.g. resnet18, resnet34, resnet50, efficientnet-b0).",
)
@click.option(
    "--encoder-weights",
    type=str,
    default=None,
    help="Pretrained weights for the encoder. Use 'imagenet' for ImageNet pretrained weights, or omit for random init.",
)
@click.option(
    "--init-from-checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional ensemble checkpoint to use for weight initialization before training.",
)
@click.option(
    "--dataset-version",
    type=click.Choice(["C1", "C2"], case_sensitive=False),
    default="C1",
    show_default=True,
    help=(
        "Dataset version to use. "
        "C1: normalized competitor overlap only (1 channel). "
        "C2: overlap + raw microscopy image (2 channels)."
    ),
)
@click.option(
    "--augmentation",
    type=click.Choice(
        [
            "basic",
            "strong",
            "basic_vflip",
            "basic_brightness",
            "basic_noise",
            "basic_vflip_brightness",
        ],
        case_sensitive=False,
    ),
    default="basic",
    show_default=True,
    help=(
        "Augmentation preset. "
        "basic: HorizontalFlip + RandomRotate90. "
        "strong: adds elastic transform, brightness/contrast jitter, Gaussian noise, and blur. "
        "basic_vflip/basic_brightness/basic_noise/basic_vflip_brightness: targeted lighter sweeps."
    ),
)
@click.option(
    "--batch-size",
    type=int,
    default=7,
    show_default=True,
    help="Training batch size for the ensemble dataloader.",
)
def ensemble_experiment(
    name: str,
    parquet_file: str,
    model_type: str,
    max_epochs: int,
    checkpoints_dir: Optional[Path],
    encoder_name: str,
    encoder_weights: Optional[str],
    init_from_checkpoint: Optional[str],
    dataset_version: str,
    augmentation: str,
    batch_size: int,
):
    """Runs an Ensemble experiment via command-line interface."""
    try:
        databank_name = Path(parquet_file).stem
        run_params = {
            "model_type": ModelType[model_type],
            "max_epochs": max_epochs,
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "init_from_checkpoint": init_from_checkpoint,
            "dataset_version": dataset_version.upper(),
            "augmentation": augmentation.lower(),
            "batch_size": batch_size,
        }
        ensemble.run_experiment(
            name,
            databank_name,
            parquet_file,
            [run_params],
            checkpoints_dir=str(checkpoints_dir)
            if checkpoints_dir is not None
            else None,
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
@click.option(
    "--aggregation-level",
    type=click.Choice(["cell", "image"], case_sensitive=False),
    default="cell",
    show_default=True,
    help=(
        "Build per-cell crops ('cell', existing behavior) or one full-image sample per GT image "
        "('image', requires QA parquet generated with crop=False)."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Directory to write the databank parquet and image folder into.  "
        f"Defaults to the package default ({utils.DATABANKS_DIR})."
    ),
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
    aggregation_level: str,
    output_dir: Optional[Path],
) -> None:
    """Build an Ensemble databank parquet and image folder from a QA parquet."""
    build_opt = {
        "name": dataset_name,
        "databank": Version[version],
        "crop_size": crop_size,
        "split_seed": split_seed,
        "split_sets": _parse_split_sets(split_sets),
        "qa": qa_column if qa_column else None,
        "qa_threshold": qa_threshold if qa_column else None,
        "aggregation_level": aggregation_level.lower(),
    }
    output_parquet = ensemble.build_databank(
        build_opt,
        qa_parquet_path,
        output_dir=str(output_dir) if output_dir is not None else None,
    )
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
@click.option(
    "--dataset-version",
    type=click.Choice(["C1", "C2"], case_sensitive=False),
    default=None,
    help="Optional ensemble dataset version override. Defaults to inferring from the checkpoint.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write evaluation parquets into. Defaults to the checkpoint directory.",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=DEFAULT_MLFLOW_TRACKING_URI,
    show_default=True,
    help="MLflow tracking URI.",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default=None,
    help="Optional MLflow experiment for final ensemble evaluation logging.",
)
@click.option(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="Optional MLflow run name.",
)
@click.option(
    "--setup-name",
    type=str,
    default=None,
    help="Logical setup name used in MLflow tags.",
)
@click.option(
    "--qa-mode",
    type=str,
    default=None,
    help="Optional QA mode tag for MLflow.",
)
@click.option(
    "--qa-threshold",
    type=float,
    default=None,
    help="Optional QA threshold tag for MLflow.",
)
def evaluate_checkpoint(
    model_path: str,
    databank_path: str,
    split_type: str,
    dataset_version: Optional[str],
    output_dir: Optional[Path],
    mlflow_tracking_uri: str,
    mlflow_experiment: Optional[str],
    mlflow_run_name: Optional[str],
    setup_name: Optional[str],
    qa_mode: Optional[str],
    qa_threshold: Optional[float],
) -> None:
    """Run inference from a checkpoint and print mean IoU/F1."""
    summary = ensemble.evaluate_checkpoint(
        model_path,
        databank_path,
        split_type,
        output_dir=str(output_dir) if output_dir is not None else None,
        dataset_version=dataset_version,
    )
    _log_ensemble_evaluation_to_mlflow(
        summary=summary,
        databank_path=databank_path,
        model_path=model_path,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        setup_name=setup_name,
        qa_mode=qa_mode,
        qa_threshold=qa_threshold,
    )
    click.echo(f"output_parquet: {summary['output_parquet_path']}")
    click.echo(
        f"split={summary['split']} evaluation_level={summary['evaluation_level']} "
        f"count={summary['count']} iou_mean={summary['iou_mean']:.6f} "
        f"f1_mean={summary['f1_mean']:.6f}"
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
    "--dataset-version",
    type=click.Choice(["C1", "C2"], case_sensitive=False),
    default=None,
    help="Optional ensemble dataset version override. Defaults to inferring from the checkpoint.",
)
@click.option(
    "--pattern",
    default="*.ckpt",
    show_default=True,
    help="Glob pattern used to locate checkpoints under checkpoints-dir.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to write evaluation parquets into. Defaults to the checkpoint directory.",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=DEFAULT_MLFLOW_TRACKING_URI,
    show_default=True,
    help="MLflow tracking URI.",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default=None,
    help="Optional MLflow experiment for final ensemble evaluation logging.",
)
@click.option(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="Optional MLflow run name.",
)
@click.option(
    "--setup-name",
    type=str,
    default=None,
    help="Logical setup name used in MLflow tags.",
)
@click.option(
    "--qa-mode",
    type=str,
    default=None,
    help="Optional QA mode tag for MLflow.",
)
@click.option(
    "--qa-threshold",
    type=float,
    default=None,
    help="Optional QA threshold tag for MLflow.",
)
def evaluate_best_checkpoint(
    checkpoints_dir: str,
    databank_path: str,
    split_type: str,
    dataset_version: Optional[str],
    pattern: str,
    output_dir: Optional[Path],
    mlflow_tracking_uri: str,
    mlflow_experiment: Optional[str],
    mlflow_run_name: Optional[str],
    setup_name: Optional[str],
    qa_mode: Optional[str],
    qa_threshold: Optional[float],
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
    summary = ensemble.evaluate_checkpoint(
        best_ckpt,
        databank_path,
        split_type,
        output_dir=str(output_dir) if output_dir is not None else None,
        dataset_version=dataset_version,
    )
    _log_ensemble_evaluation_to_mlflow(
        summary=summary,
        databank_path=databank_path,
        model_path=best_ckpt,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        setup_name=setup_name,
        qa_mode=qa_mode,
        qa_threshold=qa_threshold,
    )
    click.echo(f"output_parquet: {summary['output_parquet_path']}")
    click.echo(
        f"split={summary['split']} evaluation_level={summary['evaluation_level']} "
        f"count={summary['count']} iou_mean={summary['iou_mean']:.6f} "
        f"f1_mean={summary['f1_mean']:.6f}"
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
