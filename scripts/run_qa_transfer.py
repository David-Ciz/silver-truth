#!/usr/bin/env python3
"""
Run zero-shot QA transfer evaluation across datasets.

This script is intentionally separate from ``scripts/run_ablation.py``.
The ablation runner assumes a single split-aware parquet that contains the
train/validation/test regime for both training and evaluation. Cross-dataset QA
transfer needs different semantics:

- source model checkpoint trained elsewhere
- target parquet from a different dataset
- evaluation-only workflow on the target split

The script resolves a small YAML config, evaluates the source QA model on the
target dataset, writes the same workbook/metric artifacts as Phase B, and logs a
transfer-specific MLflow run.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import click
import mlflow
import yaml  # type: ignore[import-untyped]

from silver_truth.experiment_tracking import (
    DEFAULT_MLFLOW_TRACKING_URI,
    metric_split_alias,
    resolve_mlflow_tracking_uri,
    set_common_mlflow_tags,
    set_evaluation_tags,
    start_managed_mlflow_run,
)
from silver_truth.metrics.qa_model_evaluation import evaluate_qa_model_from_excel
from silver_truth.qa import cnn as qa_cnn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
VARIANTS_DIR = EXPERIMENTS_DIR / "variants"
ABLATION_DATA_ROOT_ENV = "ABLATION_DATA_ROOT"
ABLATION_OUTPUT_DIR_ENV = "ABLATION_OUTPUT_DIR"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk and return an empty mapping for blank files."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_extends_path(current_path: Path, extends: str) -> Path:
    """Resolve a config inheritance target relative to the current file or variants dir."""
    candidate = Path(extends)
    if candidate.is_absolute():
        return candidate

    for resolved in (
        (current_path.parent / candidate).resolve(),
        (VARIANTS_DIR / candidate).resolve(),
    ):
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Could not resolve extends target '{extends}' from {current_path}"
    )


def _load_config_with_inheritance(path: Path) -> dict[str, Any]:
    """Load a YAML config and recursively merge any ``extends`` chain."""
    raw = _load_yaml(path)
    extends = raw.pop("extends", None)
    if not extends:
        return raw

    parent_path = _resolve_extends_path(path, str(extends))
    parent = _load_config_with_inheritance(parent_path)
    return {**parent, **raw}


def _resolve_templates(obj: Any, ctx: dict[str, Any]) -> Any:
    """Recursively resolve ``{field}`` placeholders using the provided context."""
    if isinstance(obj, str):
        try:
            return obj.format_map(ctx)
        except (KeyError, ValueError):
            return obj
    if isinstance(obj, dict):
        return {key: _resolve_templates(value, ctx) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_resolve_templates(value, ctx) for value in obj]
    return obj


def normalize_target_split(split: str) -> str:
    """Normalize supported target split names to the canonical parquet naming."""
    split_str = str(split).strip().lower()
    if split_str in {"1", "fold-1"}:
        return "fold-1"
    if split_str in {"2", "fold-2"}:
        return "fold-2"
    if split_str == "mixed":
        return "mixed"
    raise ValueError(f"Unsupported target split: {split}")


def load_transfer_config(config_path: Path, target_split: str) -> dict[str, Any]:
    """Load, normalize, and resolve a QA transfer config for one target split."""
    config = _load_config_with_inheritance(config_path)
    normalized_target_split = normalize_target_split(target_split)

    runtime_root = Path(
        os.getenv(ABLATION_DATA_ROOT_ENV, str(PROJECT_ROOT))
    ).expanduser()
    paper_runs_root = Path(
        os.getenv(ABLATION_OUTPUT_DIR_ENV, str(PROJECT_ROOT / "data/paper_runs"))
    ).expanduser()

    config["project_root"] = str(PROJECT_ROOT)
    config["runtime_root"] = str(runtime_root.resolve())
    config["paper_runs_root"] = str(paper_runs_root.resolve())
    config["target_split"] = normalized_target_split
    config["source_crop_tag"] = f"sz{config['source_crop_size']}"
    config["target_crop_tag"] = f"sz{config['target_crop_size']}"
    config.setdefault("mlflow_tracking_uri", DEFAULT_MLFLOW_TRACKING_URI)
    config.setdefault(
        "mlflow_experiment",
        f"qa-transfer-{config['source_dataset']}-to-{config['target_dataset']}",
    )
    config.setdefault("batch_size", 16)
    config.setdefault("generate_plots", True)
    config.setdefault("input_channels", "0,1")

    allowed_target_splits = config.get("target_splits")
    if allowed_target_splits:
        normalized_allowed = {
            normalize_target_split(value) for value in allowed_target_splits
        }
        if normalized_target_split not in normalized_allowed:
            raise ValueError(
                f"Target split '{normalized_target_split}' is not allowed by config "
                f"{sorted(normalized_allowed)}"
            )

    for _ in range(3):
        config = _resolve_templates(config, config)

    return config


def collect_transfer_datasets(config: dict[str, Any]) -> list[str]:
    """Return the unique datasets touched by a transfer run in stable order."""
    datasets = [str(config["source_dataset"]), str(config["target_dataset"])]
    unique: list[str] = []
    for dataset in datasets:
        if dataset not in unique:
            unique.append(dataset)
    return unique


def _resolve_source_model_path(
    config: dict[str, Any], source_model_override: str | None
) -> Path:
    """Resolve the source checkpoint path, honoring an explicit CLI override."""
    if source_model_override:
        return Path(source_model_override).expanduser()
    return Path(str(config["source_model_template"])).expanduser()


def _build_transfer_paths(
    config: dict[str, Any], source_model_override: str | None
) -> dict[str, Path]:
    """Build the resolved filesystem paths used by one QA transfer evaluation."""
    source_model = _resolve_source_model_path(config, source_model_override)
    target_parquet = Path(str(config["target_parquet_template"])).expanduser()
    transfer_root = Path(str(config["transfer_output_root_template"])).expanduser()

    run_label = f"{config['source_dataset']}_to_{config['target_dataset']}_{config['target_split']}"
    predictions_excel = transfer_root / f"{run_label}_predictions.xlsx"
    metrics_dir = transfer_root / "metrics"
    metadata_path = transfer_root / "run_metadata.json"
    resolved_config_path = transfer_root / "resolved_config.yaml"

    return {
        "source_model": source_model,
        "target_parquet": target_parquet,
        "transfer_root": transfer_root,
        "predictions_excel": predictions_excel,
        "metrics_dir": metrics_dir,
        "metadata_path": metadata_path,
        "resolved_config_path": resolved_config_path,
    }


def _flatten_results_for_mlflow(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Flatten per-split QA metrics into MLflow-friendly metric keys."""
    flat_metrics: dict[str, float] = {}
    for split_name, split_results in results.items():
        split_alias = metric_split_alias(split_name)
        for metric_name, value in split_results.items():
            if metric_name == "split":
                continue
            if isinstance(value, (int, float)):
                numeric = float(value)
                if numeric == numeric and numeric not in {float("inf"), float("-inf")}:
                    flat_metrics[f"{split_alias}_{metric_name}"] = numeric
    return flat_metrics


def _write_metadata(
    metadata_path: Path,
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    source_model_override: str | None,
    dry_run: bool,
) -> None:
    """Persist the resolved runtime inputs so a transfer run can be audited later."""
    payload = {
        "source_dataset": config["source_dataset"],
        "source_split": config["source_split"],
        "source_crop_size": int(config["source_crop_size"]),
        "source_variant": config["source_variant"],
        "source_model_type": config["source_model_type"],
        "target_dataset": config["target_dataset"],
        "target_split": config["target_split"],
        "target_crop_size": int(config["target_crop_size"]),
        "input_channels": str(config["input_channels"]),
        "batch_size": int(config["batch_size"]),
        "generate_plots": bool(config["generate_plots"]),
        "mlflow_tracking_uri": str(config["mlflow_tracking_uri"]),
        "mlflow_experiment": str(config["mlflow_experiment"]),
        "source_model_override": source_model_override,
        "dry_run": dry_run,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_resolved_config(path: Path, config: dict[str, Any]) -> None:
    """Write the fully resolved config used by the transfer run."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)


def _log_transfer_run(
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    results: dict[str, dict[str, Any]],
) -> None:
    """Create one MLflow run containing the transfer metrics and artifacts."""
    tracking_uri = resolve_mlflow_tracking_uri(str(config["mlflow_tracking_uri"]))
    mlflow_run_name = (
        f"{config['source_variant']}__{config['source_split']}__to__"
        f"{config['target_dataset']}__{config['target_split']}"
    )

    with start_managed_mlflow_run(
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment=str(config["mlflow_experiment"]),
        run_name=mlflow_run_name,
    ):
        set_common_mlflow_tags(
            dataset=str(config["target_dataset"]),
            split=str(config["target_split"]),
            repo_root=PROJECT_ROOT,
            extra_tags={
                "source_dataset": config["source_dataset"],
                "source_split": config["source_split"],
                "source_variant": config["source_variant"],
                "source_model_type": config["source_model_type"],
                "target_dataset": config["target_dataset"],
                "target_split": config["target_split"],
                "target_crop_size": config["target_crop_size"],
                "transfer_mode": "zero_shot",
            },
        )
        set_evaluation_tags(
            pipeline_family="qa_model",
            evaluation_level="qa_transfer",
            setup_name=paths["transfer_root"].name,
            variant=str(config["source_variant"]),
            fold=str(config["target_split"]),
            extra_tags={
                "source_dataset": config["source_dataset"],
                "source_split": config["source_split"],
                "target_dataset": config["target_dataset"],
                "target_split": config["target_split"],
                "transfer_mode": "zero_shot",
            },
        )

        mlflow.log_params(
            {
                "source_dataset": str(config["source_dataset"]),
                "source_split": str(config["source_split"]),
                "source_crop_size": int(config["source_crop_size"]),
                "source_variant": str(config["source_variant"]),
                "source_model_type": str(config["source_model_type"]),
                "target_dataset": str(config["target_dataset"]),
                "target_split": str(config["target_split"]),
                "target_crop_size": int(config["target_crop_size"]),
                "batch_size": int(config["batch_size"]),
                "input_channels": str(config["input_channels"]),
                "source_model_path": str(paths["source_model"]),
                "target_parquet_path": str(paths["target_parquet"]),
                "predictions_excel_path": str(paths["predictions_excel"]),
            }
        )
        mlflow.log_metrics(_flatten_results_for_mlflow(results))
        mlflow.log_artifact(str(paths["predictions_excel"]))
        mlflow.log_artifact(str(paths["metadata_path"]))
        mlflow.log_artifact(str(paths["resolved_config_path"]))
        if paths["metrics_dir"].exists():
            mlflow.log_artifacts(str(paths["metrics_dir"]), artifact_path="evaluation")


def run_transfer(
    config_path: Path,
    target_split: str,
    source_model_override: str | None = None,
    dry_run: bool = False,
    no_plots: bool = False,
) -> dict[str, Any]:
    """Execute one zero-shot QA transfer evaluation and return the resolved plan."""
    config = load_transfer_config(config_path, target_split)
    paths = _build_transfer_paths(config, source_model_override)
    transfer_plan = {
        "config": config,
        "paths": {name: str(path) for name, path in paths.items()},
    }

    logger.info("Source dataset: %s", config["source_dataset"])
    logger.info("Target dataset: %s", config["target_dataset"])
    logger.info("Target split: %s", config["target_split"])
    logger.info("Source model: %s", paths["source_model"])
    logger.info("Target parquet: %s", paths["target_parquet"])
    logger.info("Transfer root: %s", paths["transfer_root"])

    if dry_run:
        click.echo(yaml.safe_dump(transfer_plan, sort_keys=False))
        return transfer_plan

    if not paths["source_model"].exists():
        raise FileNotFoundError(
            f"Source model checkpoint not found: {paths['source_model']}"
        )
    if not paths["target_parquet"].exists():
        raise FileNotFoundError(f"Target parquet not found: {paths['target_parquet']}")

    paths["transfer_root"].mkdir(parents=True, exist_ok=True)
    paths["metrics_dir"].mkdir(parents=True, exist_ok=True)
    _write_metadata(
        paths["metadata_path"],
        config=config,
        paths=paths,
        source_model_override=source_model_override,
        dry_run=dry_run,
    )
    _write_resolved_config(paths["resolved_config_path"], config)

    qa_cnn.evaluate(
        parquet_file=str(paths["target_parquet"]),
        data_root=str(config["runtime_root"]),
        input_channels=str(config["input_channels"]),
        model_path=str(paths["source_model"]),
        output_excel=str(paths["predictions_excel"]),
        batch_size=int(config["batch_size"]),
    )

    results = evaluate_qa_model_from_excel(
        excel_path=paths["predictions_excel"],
        output_dir=paths["metrics_dir"],
        generate_plots=not no_plots and bool(config["generate_plots"]),
    )
    _log_transfer_run(config=config, paths=paths, results=results)

    test_results = results.get("test", {})
    logger.info(
        "Completed transfer run. Test Pearson=%s, Test Spearman=%s",
        test_results.get("pearson_correlation"),
        test_results.get("spearman_correlation"),
    )
    return transfer_plan


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="QA transfer YAML config.",
)
@click.option(
    "--target-split",
    type=click.Choice(["1", "2", "fold-1", "fold-2", "mixed"], case_sensitive=False),
    required=True,
    help="Target split to evaluate on.",
)
@click.option(
    "--source-model-path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional explicit source QA checkpoint path.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the resolved transfer plan without running evaluation.",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Skip generating regression plots for the target evaluation.",
)
def main(
    config_path: Path,
    target_split: str,
    source_model_path: str | None,
    dry_run: bool,
    no_plots: bool,
) -> None:
    """CLI entrypoint for zero-shot QA transfer evaluation."""
    run_transfer(
        config_path=config_path,
        target_split=target_split,
        source_model_override=source_model_path,
        dry_run=dry_run,
        no_plots=no_plots,
    )


if __name__ == "__main__":
    main()
