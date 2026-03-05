#!/usr/bin/env python
"""
Run fusion experiments with MLflow tracking.

This script runs multiple fusion strategies on a dataset and logs all results
to MLflow for comparison. It:
1. Auto-detects timepoints that have GT (no need to manually specify)
2. Runs fusion for both campaigns (01 and 02)
3. Adds fused images to parquet using existing add_fused_images logic
4. Evaluates using existing evaluation functions (no duplicate logic)
5. Logs everything to MLflow with train/val/test breakdown

Usage:
    # Run all flat models on both campaigns:
    python scripts/run_fusion_experiment.py \
        --dataset BF-C2DL-HSC \
        --parquet-file data/dataframes/BF-C2DL-HSC_split_mixed.parquet \
        --flat-models-only

    # Run specific models:
    python scripts/run_fusion_experiment.py \
        --dataset BF-C2DL-HSC \
        --parquet-file data/dataframes/BF-C2DL-HSC_split_mixed.parquet \
        --models THRESHOLD_FLAT --models MAJORITY_FLAT
"""

import click
import mlflow
import sys
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import logging
from datetime import datetime


from silver_truth.fusion.fusion import (
    add_fused_images_to_dataframe_logic,
    fuse_segmentations,
    FusionModel,
)
from silver_truth.experiment_tracking import (
    set_common_mlflow_tags,
)
from silver_truth.metrics.evaluation_logic import evaluate_by_split


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Models that DON'T require weights (flat voting)
FLAT_MODELS = [
    "THRESHOLD_FLAT",
    "MAJORITY_FLAT",
    "SIMPLE",
]

# Models that DO require weights
# NOTE: BIC_FLAT_VOTING still needs weights in the job file even though it's "flat" voting
WEIGHTED_MODELS = [
    "THRESHOLD_USER",
    "BIC_FLAT_VOTING",
    "BIC_WEIGHTED_VOTING",
]

ALL_MODELS = FLAT_MODELS + WEIGHTED_MODELS

# MLflow tracking path (same as ensemble)
MLFLOW_TRACKING_PATH = "data/fusion_experiments/mlruns"


def get_gt_timepoints(df: pd.DataFrame, campaign: str) -> List[int]:
    """
    Get list of timepoints that have ground truth images for a given campaign.
    This avoids fusing images we can't evaluate.
    """
    campaign_df = df[df["campaign_number"] == campaign].copy()

    # Filter to rows with GT
    gt_df = campaign_df[campaign_df["gt_image"].notna()]

    # Check if GT files actually exist
    def gt_exists(path):
        if pd.isna(path):
            return False
        return Path(path).exists()

    gt_df = gt_df[gt_df["gt_image"].apply(gt_exists)]

    # Extract time_id
    if "time_id" in gt_df.columns:
        timepoints = sorted(gt_df["time_id"].dropna().astype(int).unique().tolist())
    else:
        # Try to extract from composite_key (e.g., "01_1707.tif" -> 1707)
        timepoints = []
        for key in gt_df["composite_key"].dropna():
            # Remove file extension and extract the number after the underscore
            key_str = str(key).replace(".tif", "").replace(".TIF", "")
            match = re.search(r"_t?(\d+)$", key_str)
            if match:
                timepoints.append(int(match.group(1)))
        timepoints = sorted(set(timepoints))

    return timepoints


def timepoints_to_string(timepoints: List[int]) -> str:
    """Convert list of timepoints to the string format expected by fusion CLI."""
    if not timepoints:
        return ""
    return ",".join(str(t) for t in timepoints)


def add_weights_to_job_file(input_job_file: Path, output_job_file: Path) -> None:
    """
    Convert a job file to one with weights by adding " 1" to each competitor line.
    The last line (ground truth/tracking markers) should not have a weight.
    """
    with open(input_job_file, "r") as f:
        lines = f.readlines()

    output_lines = []
    for i, line in enumerate(lines):
        line = line.rstrip("\n")
        if not line.strip():  # Skip empty lines
            continue

        # Last non-empty line is the GT/tracking markers - no weight
        is_last = (i == len(lines) - 1) or all(
            not line_item.strip() for line_item in lines[i + 1 :]
        )

        if is_last:
            output_lines.append(line + "\n")
        else:
            # Add weight " 1" to competitor lines
            output_lines.append(line + " 1\n")

    output_job_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_job_file, "w") as f:
        f.writelines(output_lines)

    logger.info(f"Created job file with weights: {output_job_file}")


def get_job_file_path(
    dataset: str, campaign: str, needs_weights: bool, split: str = "mixed"
) -> Path:
    """Get the appropriate job file path based on whether weights are needed."""
    base_dir = PROJECT_ROOT / "data" / "job_files" / dataset / split
    if needs_weights:
        return base_dir / f"{dataset}_{campaign}_job_file_with_weights.txt"
    else:
        return base_dir / f"{dataset}_{campaign}_job_file.txt"


def run_single_fusion(
    dataset: str,
    campaign: str,
    model: str,
    timepoints: List[int],
    num_threads: int,
    threshold: float,
    output_base_dir: Path,
    split: str = "mixed",
) -> Dict[str, Any]:
    """
    Run a single fusion model for one campaign and return results.

    Returns:
        dict with keys: success, output_dir, model, campaign, error (if failed)
    """
    needs_weights = model in WEIGHTED_MODELS
    job_file = get_job_file_path(dataset, campaign, needs_weights, split)

    if not job_file.exists():
        if needs_weights:
            # Try to auto-create job file with weights from base job file
            base_job_file = get_job_file_path(dataset, campaign, False, split)
            if base_job_file.exists():
                logger.info(f"Auto-creating job file with weights from {base_job_file}")
                try:
                    add_weights_to_job_file(base_job_file, job_file)
                except Exception as e:
                    return {
                        "success": False,
                        "model": model,
                        "campaign": campaign,
                        "error": f"Failed to create job file with weights: {e}",
                    }
            else:
                return {
                    "success": False,
                    "model": model,
                    "campaign": campaign,
                    "error": f"Job file not found: {job_file} (and base file {base_job_file} doesn't exist)",
                }
        else:
            return {
                "success": False,
                "model": model,
                "campaign": campaign,
                "error": f"Job file not found: {job_file}",
            }

    # Create model-specific output directory with split organization
    model_lower = model.lower()
    output_dir = output_base_dir / dataset / split / campaign / model_lower
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / f"{dataset}_{campaign}_fused_TTTT.tif")
    time_points_str = timepoints_to_string(timepoints)

    # JAR path
    jar_path = (
        PROJECT_ROOT
        / "src/silver_truth/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar"
    )

    logger.info(f"Running fusion: {model} on campaign {campaign} ({split} split)")
    logger.info(f"  Job file: {job_file}")
    logger.info(f"  Timepoints: {len(timepoints)} ({time_points_str[:50]}...)")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  JAR: {jar_path}")

    try:
        # Convert model string to FusionModel enum
        fusion_model_enum = FusionModel[model.upper()]

        # Call fusion directly instead of subprocess
        fuse_segmentations(
            jar_path=str(jar_path),
            job_file_path=str(job_file),
            output_path_pattern=output_pattern,
            time_points=time_points_str,
            num_threads=num_threads,
            fusion_model=fusion_model_enum,
            threshold=threshold,
            cmv_mode=None,
            seg_eval_folder=None,
            debug=True,  # Enable debug to see actual output
        )

        logger.info("  ✓ Fusion completed successfully")
        return {
            "success": True,
            "model": model,
            "campaign": campaign,
            "output_dir": output_dir,
            "output_pattern": output_pattern,
            "num_timepoints": len(timepoints),
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"  ✗ Fusion failed: {error_msg}")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "model": model,
            "campaign": campaign,
            "error": error_msg,
        }


@click.command()
@click.option(
    "--dataset",
    required=True,
    help="Dataset name (e.g., BF-C2DL-HSC)",
)
@click.option(
    "--parquet-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the dataset parquet file with split column",
)
@click.option(
    "--models",
    multiple=True,
    type=click.Choice(ALL_MODELS, case_sensitive=False),
    help="Fusion models to run. Can be specified multiple times.",
)
@click.option(
    "--all-models",
    is_flag=True,
    help="Run all available fusion models",
)
@click.option(
    "--flat-models-only",
    is_flag=True,
    help="Run only flat (non-weighted) models",
)
@click.option(
    "--campaigns",
    multiple=True,
    default=["01", "02"],
    help="Campaigns to run (default: both 01 and 02)",
)
@click.option(
    "--num-threads",
    default=4,
    help="Number of threads for fusion",
)
@click.option(
    "--threshold",
    default=1.0,
    help="Voting threshold for merging",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="fusion_results",
    help="Base directory for fusion outputs",
)
@click.option(
    "--mlflow-experiment",
    default="fusion-baseline",
    help="MLflow experiment name",
)
@click.option(
    "--skip-fusion",
    is_flag=True,
    help="Skip fusion step (only evaluate existing results)",
)
def main(
    dataset: str,
    parquet_file: Path,
    models: tuple,
    all_models: bool,
    flat_models_only: bool,
    campaigns: tuple,
    num_threads: int,
    threshold: float,
    output_dir: Path,
    mlflow_experiment: str,
    skip_fusion: bool,
):
    """
    Run fusion experiments with multiple models and track results in MLflow.

    This script:
    1. Auto-detects which timepoints have ground truth
    2. Runs fusion for all specified models on both campaigns
    3. Adds fused images to the parquet
    4. Evaluates and logs metrics by train/val/test split

    Examples:

        # Run all flat models (recommended for baseline):
        python scripts/run_fusion_experiment.py \\
            --dataset BF-C2DL-HSC \\
            --parquet-file data/dataframes/BF-C2DL-HSC_split_mixed.parquet \\
            --flat-models-only

        # Run specific models:
        python scripts/run_fusion_experiment.py \\
            --dataset BF-C2DL-HSC \\
            --parquet-file data/dataframes/BF-C2DL-HSC_split_mixed.parquet \\
            --models THRESHOLD_FLAT --models MAJORITY_FLAT

        # Only evaluate (skip fusion if already run):
        python scripts/run_fusion_experiment.py \\
            --dataset BF-C2DL-HSC \\
            --parquet-file data/dataframes/BF-C2DL-HSC_split_mixed.parquet \\
            --flat-models-only --skip-fusion
    """
    # Determine which models to run
    if all_models:
        models_to_run = ALL_MODELS
    elif flat_models_only:
        models_to_run = FLAT_MODELS
    elif models:
        models_to_run = [m.upper() for m in models]
    else:
        click.echo("Error: Specify --models, --all-models, or --flat-models-only")
        raise SystemExit(1)

    campaigns_list = list(campaigns)

    # Extract split type from parquet filename
    parquet_name = parquet_file.name
    if "split_mixed" in parquet_name:
        split = "mixed"
    elif "split_fold-1" in parquet_name:
        split = "fold-1"
    elif "split_fold-2" in parquet_name:
        split = "fold-2"
    else:
        split = "mixed"  # default

    logger.info("=" * 70)
    logger.info(f"Fusion Baseline Experiment: {dataset}")
    logger.info(f"  Split type: {split}")
    logger.info(f"  Parquet: {parquet_file}")
    logger.info(f"  Models: {models_to_run}")
    logger.info(f"  Campaigns: {campaigns_list}")
    logger.info("=" * 70)

    # Load parquet to detect GT timepoints
    df = pd.read_parquet(parquet_file)
    logger.info(f"Loaded {len(df)} rows from parquet")

    # Setup MLflow
    mlflow_path = PROJECT_ROOT / MLFLOW_TRACKING_PATH
    mlflow_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(mlflow_path))
    mlflow.set_experiment(mlflow_experiment)

    output_base = PROJECT_ROOT / output_dir

    # Start parent MLflow run
    run_name = f"{dataset}_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as parent_run:
        set_common_mlflow_tags(dataset=dataset, split=split, repo_root=PROJECT_ROOT)
        mlflow.set_tag("run_kind", "experiment_parent")
        mlflow.set_tag("parent_scope", "dataset_split")
        mlflow.log_params(
            {
                "dataset": dataset,
                "parquet_file": str(parquet_file),
                "campaigns": ",".join(campaigns_list),
                "models": ",".join(models_to_run),
                "num_threads": num_threads,
                "threshold": threshold,
            }
        )

        all_results = []

        # Run each model
        for model in models_to_run:
            model_lower = model.lower()
            logger.info(f"\n{'='*60}")
            logger.info(f"Model: {model}")
            logger.info("=" * 60)

            with mlflow.start_run(run_name=model, nested=True) as model_run:
                set_common_mlflow_tags(
                    dataset=dataset, split=split, repo_root=PROJECT_ROOT
                )
                mlflow.set_tag("run_kind", "model_run")
                mlflow.log_params(
                    {
                        "model": model,
                        "needs_weights": model in WEIGHTED_MODELS,
                    }
                )

                fusion_results = []

                # Run fusion for each campaign
                for campaign in campaigns_list:
                    # Auto-detect GT timepoints
                    timepoints = get_gt_timepoints(df, campaign)

                    if not timepoints:
                        logger.warning(
                            f"No GT timepoints found for campaign {campaign}"
                        )
                        continue

                    logger.info(f"Campaign {campaign}: {len(timepoints)} GT timepoints")
                    mlflow.log_param(f"campaign_{campaign}_timepoints", len(timepoints))

                    if not skip_fusion:
                        result = run_single_fusion(
                            dataset=dataset,
                            campaign=campaign,
                            model=model,
                            timepoints=timepoints,
                            num_threads=num_threads,
                            threshold=threshold,
                            output_base_dir=output_base,
                            split=split,
                        )
                        fusion_results.append(result)
                    else:
                        # Just verify output exists
                        expected_dir = output_base / dataset / campaign / model_lower
                        fusion_results.append(
                            {
                                "success": expected_dir.exists(),
                                "model": model,
                                "campaign": campaign,
                                "output_dir": expected_dir,
                                "skipped": True,
                            }
                        )

                # Check if all campaigns succeeded
                all_succeeded = len(fusion_results) > 0 and all(
                    r.get("success", False) for r in fusion_results
                )

                if all_succeeded:
                    mlflow.set_tag("fusion_status", "success")

                    # Add fused images to parquet
                    logger.info(f"Adding fused images to parquet for {model}...")

                    fused_parquet = (
                        output_base
                        / dataset
                        / split
                        / f"{dataset}_{split}_{model_lower}_with_fused.parquet"
                    )
                    fused_results_dir = output_base / dataset / split

                    try:
                        add_fused_images_to_dataframe_logic(
                            dataset_name=dataset,
                            input_parquet_path=str(parquet_file),
                            fused_results_dir=str(fused_results_dir),
                            output_parquet_path=str(fused_parquet),
                            model_name=model_lower,
                        )

                        # The column name is based on the directory structure
                        # We need to find what column was added
                        fused_df = pd.read_parquet(fused_parquet)
                        new_cols = [c for c in fused_df.columns if c not in df.columns]

                        if new_cols:
                            fused_column = new_cols[
                                0
                            ]  # Should be the model's fused column
                            logger.info(f"Fused column added: {fused_column}")

                            # Compute metrics by split using centralized evaluation function
                            logger.info("Computing evaluation metrics...")
                            metrics = evaluate_by_split(fused_parquet, fused_column)

                            if metrics:
                                # Log metrics to MLflow
                                for metric_split, split_metrics in metrics.items():
                                    for metric_name, value in split_metrics.items():
                                        if isinstance(value, (int, float)):
                                            mlflow.log_metric(
                                                f"{metric_split}_{metric_name}", value
                                            )

                                # Print summary
                                logger.info(f"\n  Results for {model}:")
                                for metric_split in [
                                    "train",
                                    "validation",
                                    "test",
                                    "overall",
                                ]:
                                    if metric_split in metrics:
                                        m = metrics[metric_split]
                                        logger.info(
                                            f"    {metric_split:12}: Jaccard={m['mean_jaccard']:.4f}±{m['std_jaccard']:.4f}, "
                                            f"F1={m['mean_f1']:.4f}±{m['std_f1']:.4f} (n={m['count']})"
                                        )

                                all_results.append(
                                    {
                                        "model": model,
                                        "success": True,
                                        "metrics": metrics,
                                        "mlflow_run_id": model_run.info.run_id,
                                    }
                                )
                            else:
                                logger.warning("No metrics computed")
                                all_results.append(
                                    {
                                        "model": model,
                                        "success": True,
                                        "metrics": {},
                                        "mlflow_run_id": model_run.info.run_id,
                                    }
                                )
                        else:
                            logger.warning(
                                "No new columns added - fused images may not have been mapped"
                            )
                            all_results.append(
                                {
                                    "model": model,
                                    "success": False,
                                    "error": "No fused column added",
                                }
                            )
                    except Exception as e:
                        logger.error(f"Failed to process fused results: {e}")
                        import traceback

                        traceback.print_exc()
                        mlflow.set_tag("evaluation_status", "failed")
                        all_results.append(
                            {
                                "model": model,
                                "success": False,
                                "error": str(e),
                            }
                        )
                else:
                    mlflow.set_tag("fusion_status", "failed")
                    errors = [
                        r.get("error", "Unknown")
                        for r in fusion_results
                        if not r.get("success")
                    ]
                    mlflow.set_tag("errors", "; ".join(errors[:3]))
                    all_results.append(
                        {
                            "model": model,
                            "success": False,
                            "error": "; ".join(errors),
                        }
                    )

        # Final summary
        successful = sum(1 for r in all_results if r.get("success"))

        mlflow.log_metrics(
            {
                "models_successful": successful,
                "models_failed": len(all_results) - successful,
            }
        )

        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 70)
        logger.info(f"MLflow Run ID: {parent_run.info.run_id}")
        logger.info(f"View results: mlflow ui --backend-store-uri {mlflow_path}")

        # Print summary table
        click.echo("\n" + "=" * 90)
        click.echo(
            f"{'Model':<20} {'Status':<10} {'Train Jaccard':<15} {'Val Jaccard':<15} {'Test Jaccard':<15}"
        )
        click.echo("=" * 90)

        for r in all_results:
            model = r["model"]
            if r.get("success") and r.get("metrics"):
                m = r["metrics"]
                train_j = (
                    f"{m.get('train', {}).get('mean_jaccard', 0):.4f}"
                    if "train" in m
                    else "N/A"
                )
                val_j = (
                    f"{m.get('validation', {}).get('mean_jaccard', 0):.4f}"
                    if "validation" in m
                    else "N/A"
                )
                test_j = (
                    f"{m.get('test', {}).get('mean_jaccard', 0):.4f}"
                    if "test" in m
                    else "N/A"
                )
                click.echo(
                    f"  ✓ {model:<18} {'OK':<10} {train_j:<15} {val_j:<15} {test_j:<15}"
                )
            else:
                error = r.get("error", "Failed")[:30]
                click.echo(f"  ✗ {model:<18} {'FAILED':<10} {error}")

        click.echo("=" * 90)


if __name__ == "__main__":
    main()
