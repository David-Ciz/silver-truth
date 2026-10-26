#!/usr/bin/env python
"""
Debug script to test a single fusion model directly without MLflow overhead.
This is useful for debugging why a model like BIC_FLAT_VOTING is failing.

Usage:
    python scripts/debug_fusion.py \
        --model BIC_FLAT_VOTING \
        --dataset BF-C2DL-HSC \
        --campaign 01 \
        --timepoints 1707,1708,1709
"""

import click
import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from silver_truth.fusion.fusion import fuse_segmentations, FusionModel
from silver_truth.job_file_generator import generate_job_file

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SPLIT_TO_PARQUET_SUFFIX = {
    "mixed": "split_mixed",
    "fold-1": "split_fold-1",
    "fold-2": "split_fold-2",
}


def regenerate_job_files(dataset: str, campaign: str, split: str) -> tuple[Path, Path]:
    parquet_suffix = SPLIT_TO_PARQUET_SUFFIX.get(split)
    if parquet_suffix is None:
        raise ValueError(f"Unsupported split: {split}")

    parquet_file = (
        PROJECT_ROOT
        / "data"
        / "dataframes"
        / dataset
        / "whole_image"
        / f"{dataset}_{parquet_suffix}.parquet"
    )
    output_dir = PROJECT_ROOT / "data" / "job_files" / dataset / split
    base_job_file = Path(
        generate_job_file(
            parquet_file_path=str(parquet_file),
            campaign_number=campaign,
            output_dir=str(output_dir),
        )
    )
    weighted_job_file = output_dir / f"{dataset}_{campaign}_job_file_with_weights.txt"
    return base_job_file, weighted_job_file


@click.command()
@click.option("--model", required=True, help="Fusion model to test (e.g., BIC_FLAT_VOTING)")
@click.option("--dataset", default="BF-C2DL-HSC", help="Dataset name")
@click.option("--campaign", default="01", help="Campaign number (01 or 02)")
@click.option("--timepoints", default="1707", help="Comma-separated timepoints to process")
@click.option("--split", default="mixed", help="Split type (mixed, fold-1, fold-2)")
@click.option("--num-threads", default=4, help="Number of threads")
@click.option("--threshold", default=1.0, help="Voting threshold")
def main(model, dataset, campaign, timepoints, split, num_threads, threshold):
    """Debug a single fusion model run."""

    logger.info("=" * 70)
    logger.info("FUSION DEBUG MODE")
    logger.info("=" * 70)
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Campaign: {campaign}")
    logger.info(f"Timepoints: {timepoints}")
    logger.info(f"Split: {split}")
    logger.info("=" * 70)

    # Determine if we need weights
    # NOTE: BIC_FLAT_VOTING still needs weights in the job file even though it's "flat" voting
    weighted_models = ["THRESHOLD_USER", "BIC_FLAT_VOTING", "BIC_WEIGHTED_VOTING"]
    needs_weights = model in weighted_models

    try:
        base_job_file, weighted_job_file = regenerate_job_files(
            dataset=dataset,
            campaign=campaign,
            split=split,
        )
    except Exception as e:
        logger.error(f"Failed to regenerate job files: {e}")
        sys.exit(1)

    job_file = weighted_job_file if needs_weights else base_job_file

    logger.info(f"Job file: {job_file}")

    # Output directory
    output_dir = PROJECT_ROOT / "fusion_results_debug" / dataset / split / campaign / model.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / f"{dataset}_{campaign}_fused_TTTT.tif")

    # JAR path
    jar_path = PROJECT_ROOT / "src/silver_truth/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar"

    if not jar_path.exists():
        logger.error(f"JAR file not found: {jar_path}")
        sys.exit(1)

    logger.info(f"JAR: {jar_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output pattern: {output_pattern}")
    logger.info("")
    logger.info("Starting fusion...")
    logger.info("=" * 70)

    try:
        # Convert model string to enum
        fusion_model_enum = FusionModel[model.upper()]

        # Run fusion with debug enabled
        fuse_segmentations(
            jar_path=str(jar_path),
            job_file_path=str(job_file),
            output_path_pattern=output_pattern,
            time_points=timepoints,
            num_threads=num_threads,
            fusion_model=fusion_model_enum,
            threshold=threshold,
            cmv_mode=None,
            seg_eval_folder=None,
            debug=True,
        )

        logger.info("=" * 70)
        logger.info("✓ FUSION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)

        # Check if any output files were created
        output_files = list(output_dir.glob("*.tif"))
        logger.info(f"Output files created: {len(output_files)}")
        for f in output_files[:5]:  # Show first 5
            logger.info(f"  - {f.name}")
        if len(output_files) > 5:
            logger.info(f"  ... and {len(output_files) - 5} more")

    except Exception as e:
        logger.error("=" * 70)
        logger.error("✗ FUSION FAILED!")
        logger.error("=" * 70)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
