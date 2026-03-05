import os
import subprocess
import logging
import re
from enum import Enum
from typing import Dict, Optional, Tuple
import pandas as pd
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FusionModel(Enum):
    """Enumeration for the fusion models available in Fusers.java."""

    THRESHOLD_FLAT = "Threshold - flat weights"
    THRESHOLD_USER = "Threshold - user weights"
    MAJORITY_FLAT = "Majority - flat weights"
    SIMPLE = "SIMPLE"
    BIC_FLAT_VOTING = "BICv2 with FlatVoting, SingleMaskFailSafe and CollisionResolver"
    BIC_WEIGHTED_VOTING = (
        "BICv2 with WeightedVoting, SingleMaskFailSafe and CollisionResolver"
    )


def fuse_segmentations(
    jar_path: str,
    job_file_path: str,
    output_path_pattern: str,
    time_points: str,
    num_threads: int,
    fusion_model: FusionModel,
    threshold: float = 1.0,
    cmv_mode: Optional[str] = None,
    seg_eval_folder: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Runs the Fusers Java tool by calling the flexible RunFusersCli wrapper."""
    # (The body of this function remains exactly the same as before)
    jar_path = os.path.abspath(jar_path)
    job_file_path = os.path.abspath(job_file_path)
    output_dir = os.path.dirname(os.path.abspath(output_path_pattern))
    if not os.path.isdir(output_dir):
        logging.info(f"Output directory '{output_dir}' does not exist. Creating it...")
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Successfully created output directory: {output_dir}")
    if seg_eval_folder:
        seg_eval_folder = os.path.abspath(seg_eval_folder)

    logging.info(f"Preparing to run Fusers with model: '{fusion_model.value}'")
    logging.info(f"Job file: {job_file_path}")

    command = [
        "java",
        "-cp",
        jar_path,
        "de.mpicbg.ulman.fusion.RunFusersCli",
        fusion_model.value,
        job_file_path,
        str(threshold),
        output_path_pattern,
        time_points,
        str(num_threads),
    ]
    if cmv_mode:
        command.append(cmv_mode)
    if seg_eval_folder:
        command.append(seg_eval_folder)

    logging.info(f"Executing command: {' '.join(command)}")

    # Always capture output for debugging, but also print to console in debug mode
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )

    # Always log the output for debugging
    if debug or result.returncode != 0:
        logging.info(f"Return Code: {result.returncode}")
        if result.stdout:
            logging.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logging.info(f"STDERR:\n{result.stderr}")

    if result.returncode != 0:
        logging.error("The Fusers Java process failed.")
        logging.error(f"Return Code: {result.returncode}")
        if result.stdout:
            logging.error(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logging.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"Execution of Fusers (RunFusersCli) Java tool failed with return code {result.returncode}"
        )
    else:
        logging.info("Fusers Java process completed successfully.")


def add_fused_images_to_dataframe_logic(
    dataset_name: Optional[str] = None,
    input_parquet_path: Optional[str] = None,
    fused_results_dir: Optional[str] = None,
    output_parquet_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """Process a single dataset to add fused image paths.

    Args:
        dataset_name: Name of the dataset
        input_parquet_path: Path to input parquet file
        fused_results_dir: Directory containing fused results
        output_parquet_path: Path to output parquet file
        base_dir: Base directory containing dataframes/ and fused_results/
        model_name: Name of the fusion model (for column naming). If not provided, will be inferred from directory structure.
    """

    def _resolve_input_path() -> Path:
        if input_parquet_path:
            return Path(input_parquet_path).expanduser().resolve()
        if base_dir and dataset_name:
            return (
                Path(base_dir)
                / "dataframes"
                / f"{dataset_name}_dataset_dataframe.parquet"
            ).resolve()
        raise ValueError("input_parquet_path or base_dir+dataset_name must be provided")

    def _resolve_output_path(resolved_input: Path) -> Path:
        if output_parquet_path:
            return Path(output_parquet_path).expanduser().resolve()
        if base_dir and dataset_name:
            return (
                Path(base_dir)
                / "fused_results_parquet"
                / f"{dataset_name}_dataset_dataframe_with_fused.parquet"
            ).resolve()
        return resolved_input.with_name(
            resolved_input.stem.replace("_dataset_dataframe", "_with_fused")
            + resolved_input.suffix
        )

    def _resolve_fused_dir() -> Path:
        if fused_results_dir:
            return Path(fused_results_dir).expanduser().resolve()
        if base_dir:
            return (Path(base_dir) / "fused_results").resolve()
        raise ValueError("fused_results_dir or base_dir must be provided")

    def _infer_dataset_name(resolved_input: Path) -> str:
        inferred = resolved_input.stem
        suffix = "_dataset_dataframe"
        if inferred.endswith(suffix):
            inferred = inferred[: -len(suffix)]
        return inferred

    def _extract_campaign_time(
        composite_key: str,
    ) -> Tuple[Optional[str], Optional[int]]:
        if not composite_key:
            return None, None
        stem = Path(composite_key).stem
        parts = stem.split("_")
        if len(parts) < 2:
            return None, None
        campaign = parts[0]
        time_part = parts[-1]
        if not time_part.isdigit():
            return None, None
        return campaign, int(time_part)

    def _build_fused_lookup(
        fused_dir: Path, dataset: str, model: Optional[str] = None
    ) -> Dict[Tuple[str, int], Path]:
        pattern = re.compile(
            rf"^{re.escape(dataset)}_(?P<campaign>[^_]+)_fused_(?P<time>\d+)\.tif$"
        )
        lookup: Dict[Tuple[str, int], Path] = {}

        # If a specific model is provided, only search in that model's subdirectories
        if model:
            # Search in: fused_dir/01/model/, fused_dir/02/model/, etc.
            for campaign_dir in fused_dir.iterdir():
                if campaign_dir.is_dir() and campaign_dir.name in ["01", "02"]:
                    model_dir = campaign_dir / model
                    if model_dir.exists():
                        for path in model_dir.glob(f"{dataset}_*_fused_*.tif"):
                            match = pattern.match(path.name)
                            if not match:
                                continue
                            key = (match.group("campaign"), int(match.group("time")))
                            if key in lookup:
                                logging.warning(
                                    "Duplicate fused image detected for %s campaign %s time %s; keeping first",
                                    dataset,
                                    key[0],
                                    key[1],
                                )
                                continue
                            lookup[key] = path
        else:
            # Search recursively in all subdirectories for fused images (old behavior)
            for path in fused_dir.rglob(f"{dataset}_*_fused_*.tif"):
                match = pattern.match(path.name)
                if not match:
                    continue
                key = (match.group("campaign"), int(match.group("time")))
                if key in lookup:
                    logging.warning(
                        "Duplicate fused image detected for %s campaign %s time %s; keeping first",
                        dataset,
                        key[0],
                        key[1],
                    )
                    continue
                lookup[key] = path
        return lookup

    def _infer_model_name(fused_dir: Path, dataset: str) -> str:
        """
        Try to infer the model name from the directory structure by looking for
        model-specific subdirectories in the fused results.
        Pattern: dataset/split/campaign/model_name/ or dataset/campaign/model_name/
        """
        # If fused_dir itself is a model directory, use that name
        if fused_dir.name not in [
            dataset,
            "fusion_results",
            "fused_results",
            "mixed",
            "fold-1",
            "fold-2",
        ]:
            return fused_dir.name

        # Look for model subdirectories under campaign directories
        # e.g., fused_dir/01/threshold_flat, fused_dir/02/majority_flat
        model_names = set()
        for campaign_dir in fused_dir.iterdir():
            if campaign_dir.is_dir() and campaign_dir.name in ["01", "02"]:
                # Found a campaign directory, look for model subdirectories
                for model_dir in campaign_dir.iterdir():
                    if model_dir.is_dir():
                        model_names.add(model_dir.name)

        if model_names:
            # Return the most common model name (or first if only one)
            return sorted(model_names)[0]

        # Final fallback to dataset name
        return dataset

    input_path = _resolve_input_path()
    if not dataset_name:
        dataset_name = _infer_dataset_name(input_path)
    fused_dir = _resolve_fused_dir()
    output_path = _resolve_output_path(input_path)

    # Use explicit model_name if provided, otherwise try to infer it from directory structure
    if model_name:
        # Use the explicitly provided model name
        fused_col_name = model_name
    else:
        # Try to infer from directory structure as fallback
        fused_col_name = _infer_model_name(fused_dir, dataset_name)

    if not input_path.exists():
        print(f"Warning: Parquet file not found: {input_path}")
        return False
    if not fused_dir.exists():
        print(f"Warning: Fused results directory not found: {fused_dir}")
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing dataset: {dataset_name}")
    df = pd.read_parquet(input_path).copy()
    fused_lookup = _build_fused_lookup(fused_dir, dataset_name, fused_col_name)
    if not fused_lookup:
        print(
            f"No fused images found for dataset '{dataset_name}' in {fused_dir} with model '{fused_col_name}'"
        )

    def _lookup_path(composite_key: str) -> Optional[str]:
        campaign, time_idx = _extract_campaign_time(composite_key)
        if campaign is None or time_idx is None:
            return None
        path = fused_lookup.get((campaign, time_idx))
        return str(path) if path else None

    # Add the fused column with the properly determined model name
    df[fused_col_name] = df["composite_key"].astype(str).map(_lookup_path)

    campaign_numbers = df.get("campaign_number")
    if campaign_numbers is not None:
        unique_campaigns = sorted(df["campaign_number"].dropna().unique())
    else:
        unique_campaigns = []

    for campaign in unique_campaigns:
        mask = df["campaign_number"] == campaign
        mapped_count = df.loc[mask, fused_col_name].notna().sum()
        total_count = mask.sum()
        print(
            f"Campaign {campaign}: {mapped_count}/{total_count} images have fused counterparts"
        )
        if mapped_count:
            print(
                df.loc[
                    mask & df[fused_col_name].notna(),
                    ["composite_key", fused_col_name],
                ].head()
            )

    df.to_parquet(output_path, index=False)
    print(f"Modified dataframe saved to: {output_path}")
    return True


def process_all_datasets_logic(base_dir):
    """Process all datasets in the dataframes directory."""
    dataframes_dir = Path(base_dir) / "dataframes"

    if not dataframes_dir.exists():
        print(f"Error: Dataframes directory not found: {dataframes_dir}")
        return

    # Find all parquet files
    parquet_files = list(dataframes_dir.glob("*_dataset_dataframe.parquet"))

    if not parquet_files:
        print("No dataset parquet files found!")
        return

    print(f"Found {len(parquet_files)} datasets to process:")

    success_count = 0
    for parquet_file in parquet_files:
        # Extract dataset name from filename
        dataset_name = parquet_file.name.replace("_dataset_dataframe.parquet", "")
        print(f"\n{'='*50}")

        output_parquet = (
            Path(base_dir)
            / "fused_results_parquet"
            / f"{dataset_name}_dataset_dataframe_with_fused.parquet"
        )
        fused_dir = Path(base_dir) / "fused_results"
        if add_fused_images_to_dataframe_logic(
            dataset_name=dataset_name,
            input_parquet_path=str(parquet_file),
            fused_results_dir=str(fused_dir),
            output_parquet_path=str(output_parquet),
        ):
            success_count += 1
        else:
            print(f"Failed to process: {dataset_name}")

    print(f"\n{'='*50}")
    print(
        f"Processing complete: {success_count}/{len(parquet_files)} datasets processed successfully"
    )
