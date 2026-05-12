import ast
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Set, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
import numpy as np

# Constants
RAW_DATA_FOLDERS = {"01", "02"}
GT_FOLDER_FIRST = "01_GT"
GT_FOLDER_SECOND = "02_GT"
ST_FOLDER_FIRST = "01_ST"
ST_FOLDER_SECOND = "02_ST"
SEG_FOLDER = "SEG"
TRA_FOLDER = "TRA"
RES_FOLDER_FIRST = "01_RES"
RES_FOLDER_SECOND = "02_RES"
SILVER_TRUTH_COLUMN = "SILVER-TRUTH"
REFERENCE_COLUMNS_ATTR = "reference_columns"
SPLIT_AUDIT_ATTR = "split_audit_json"


def is_valid_competitor_folder(folder):
    """Check if the folder has the expected structure for a competitor."""
    return (folder / RES_FOLDER_FIRST).is_dir()


def process_dataset_directory(
    directory: Path,
) -> tuple[Dict[str, dict], Set[str]]:
    """
    Process the dataset directory structure and extract file information.

    Args:
        directory: Path to the synchronized dataset directory

    Returns:
        Dictionary containing organized dataset information
    """
    dataset_info: defaultdict[str, dict] = defaultdict(dict)
    competitor_columns = set()
    for dataset_subfolder in directory.iterdir():
        if not dataset_subfolder.is_dir():
            continue

        # Process source data (campaigns 01 and 02)
        if dataset_subfolder.name in ["01", "02"]:
            process_source_images(dataset_subfolder, dataset_info)

        # Process ground truth and tracking markers
        elif dataset_subfolder.name == GT_FOLDER_FIRST:
            process_gt_data(dataset_subfolder, "01", dataset_info)
        elif dataset_subfolder.name == GT_FOLDER_SECOND:
            process_gt_data(dataset_subfolder, "02", dataset_info)
        elif dataset_subfolder.name == ST_FOLDER_FIRST:
            process_silver_truth_data(dataset_subfolder, "01", dataset_info)
        elif dataset_subfolder.name == ST_FOLDER_SECOND:
            process_silver_truth_data(dataset_subfolder, "02", dataset_info)

        # Process competitor results
        elif is_valid_competitor_folder(dataset_subfolder):
            competitor_key = dataset_subfolder.name
            competitor_columns.add(competitor_key)
            process_competitor_data(dataset_subfolder, dataset_info)

    return dataset_info, competitor_columns


def process_source_images(
    folder: Path, dataset_info: Dict[str, Dict[str, Any]]
) -> None:
    """Process source image files from a campaign folder."""
    campaign_number = folder.name

    for image in folder.iterdir():
        if image.suffix == ".tif":
            image_number = extract_image_number(image.name)
            composite_key = f"{campaign_number}_{image_number}"
            dataset_info[composite_key]["source_image"] = str(image)
            dataset_info[composite_key]["campaign_number"] = campaign_number
            dataset_info[composite_key]["time_frame"] = int(image_number.split(".")[0])


def process_gt_data(
    folder: Path, campaign_number: str, dataset_info: Dict[str, Dict[str, Any]]
) -> None:
    """Process ground truth and tracking marker files."""
    gt_subfolder = folder / SEG_FOLDER
    tra_subfolder = folder / TRA_FOLDER

    # Process ground truth images
    for image in gt_subfolder.iterdir():
        if image.suffix == ".tif":
            image_number = extract_image_number(image.name)
            composite_key = f"{campaign_number}_{image_number}"
            dataset_info[composite_key]["gt_image"] = str(image)
            dataset_info[composite_key]["campaign_number"] = campaign_number
            dataset_info[composite_key]["time_frame"] = int(image_number.split(".")[0])

    # Process tracking marker images
    for image in tra_subfolder.iterdir():
        if image.suffix == ".tif":
            image_number = extract_image_number(image.name)
            composite_key = f"{campaign_number}_{image_number}"
            dataset_info[composite_key]["tracking_markers"] = str(image)
            dataset_info[composite_key]["campaign_number"] = campaign_number
            dataset_info[composite_key]["time_frame"] = int(image_number.split(".")[0])


def process_silver_truth_data(
    folder: Path, campaign_number: str, dataset_info: Dict[str, Dict[str, Any]]
) -> None:
    """Process silver-truth segmentation files stored under 01_ST/02_ST."""
    seg_subfolder = folder / SEG_FOLDER
    if not seg_subfolder.is_dir():
        return

    for image in seg_subfolder.iterdir():
        if image.suffix == ".tif":
            image_number = extract_image_number(image.name)
            composite_key = f"{campaign_number}_{image_number}"
            dataset_info[composite_key][SILVER_TRUTH_COLUMN] = str(image)
            dataset_info[composite_key]["campaign_number"] = campaign_number
            dataset_info[composite_key]["time_frame"] = int(image_number.split(".")[0])


def process_competitor_data(
    folder: Path, dataset_info: Dict[str, Dict[str, Any]]
) -> None:
    """Process competitor result files."""
    res1_subfolder = folder / RES_FOLDER_FIRST
    res2_subfolder = folder / RES_FOLDER_SECOND

    # Process campaign 01 results
    process_competitor_campaign(res1_subfolder, "01", folder, dataset_info)

    # Process campaign 02 results
    process_competitor_campaign(res2_subfolder, "02", folder, dataset_info)


def process_competitor_campaign(
    res_folder: Path,
    campaign_number: str,
    competitor_folder: Path,
    dataset_info: Dict[str, Dict[str, Any]],
) -> None:
    """Process competitor result files for a specific campaign."""
    for image in res_folder.iterdir():
        if image.suffix == ".tif":
            image_number = extract_image_number(image.name)
            composite_key = f"{campaign_number}_{image_number}"
            dataset_info[composite_key][str(competitor_folder.name)] = str(image)
            dataset_info[composite_key]["campaign_number"] = campaign_number
            dataset_info[composite_key]["time_frame"] = int(image_number.split(".")[0])


def extract_image_number(filename: str) -> str:
    """Extract the image number from a filename."""
    import re

    # Try to extract number from different patterns
    # Pattern 1: man_seg followed by digits (GT files)
    match = re.search(r"man_seg(\d+)\.tif$", filename)
    if match:
        number = match.group(1)
        return f"{number.zfill(4)}.tif"  # Normalize to 4 digits with .tif extension

    # Pattern 2: mask followed by digits (competitor files)
    match = re.search(r"mask(\d+)\.tif$", filename)
    if match:
        number = match.group(1)
        return f"{number.zfill(4)}.tif"  # Normalize to 4 digits with .tif extension

    # Pattern 3: man_track followed by digits (tracking files)
    match = re.search(r"man_track(\d+)\.tif$", filename)
    if match:
        number = match.group(1)
        return f"{number.zfill(4)}.tif"  # Normalize to 4 digits with .tif extension

    # Pattern 4: t followed by digits (source images)
    match = re.search(r"t(\d+)\.tif$", filename)
    if match:
        number = match.group(1)
        return f"{number.zfill(4)}.tif"  # Normalize to 4 digits with .tif extension

    # Fallback to original logic if no pattern matches
    return filename[-8:]


def convert_to_dataframe(dataset_info: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert the dataset information dictionary to a pandas DataFrame.

    Args:
        dataset_info: Dictionary containing dataset information

    Returns:
        Pandas DataFrame with organized dataset information
    """
    # If dataset is empty, return empty dataframe
    if not dataset_info:
        return pd.DataFrame()

    # Initialize DataFrame with composite_key as the first column
    df = pd.DataFrame({"composite_key": list(dataset_info.keys())})

    # Collect all possible column names
    all_columns: Set[str] = set()
    for image_data in dataset_info.values():
        all_columns.update(image_data.keys())

    # Add each column to the DataFrame
    for column in all_columns:
        df[column] = [data.get(column, None) for data in dataset_info.values()]

    return df


def get_competitor_columns(df: pd.DataFrame) -> List[str]:
    """
    Get the list of competitor columns from the DataFrame.

    Args:
        df: DataFrame potentially containing competitor columns in attrs

    Returns:
        List of competitor column names
    """
    competitor_columns = list(df.attrs.get("competitor_columns", []))
    reference_columns = set(df.attrs.get(REFERENCE_COLUMNS_ATTR, []))
    return [
        column
        for column in competitor_columns
        if column not in reference_columns and column != SILVER_TRUTH_COLUMN
    ]


def save_dataframe_to_parquet_with_metadata(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves a DataFrame to a Parquet file along with its metadata stored in df.attrs.

    The metadata is added to the Parquet schema. Since Parquet metadata must be bytes,
    both keys and values are converted to strings and then encoded.

    Parameters:
        df: The pandas DataFrame to save.
        output_path: The file path to store the Parquet file.
    """

    if "creation_time" in df.attrs:
        del df.attrs["creation_time"]
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    # Convert DataFrame to a PyArrow table without preserving the index.
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Convert DataFrame.attrs to a bytes-based dictionary.
    # This stores all metadata as string values.
    metadata = {
        str(key).encode(): str(value).encode() for key, value in df.attrs.items()
    }

    # Replace the table's schema metadata with our custom metadata.
    table = table.replace_schema_metadata(metadata)

    # Write the table to a Parquet file.
    pq.write_table(table, output_path_obj)


def load_dataframe_from_parquet_with_metadata(input_path: str) -> pd.DataFrame:
    """
    Loads a DataFrame from a Parquet file and restores its metadata into df.attrs.

    Parameters:
        input_path: The Parquet file path to load.

    Returns:
        A pandas DataFrame with .attrs populated from the Parquet schema metadata.
    """
    # Read the table using pyarrow.
    table = pq.read_table(input_path)
    df = table.to_pandas()

    # Retrieve custom metadata.
    meta = table.schema.metadata
    if meta:
        # Decode the metadata and update the DataFrame's attrs.
        df.attrs = {key.decode(): meta[key].decode() for key in meta}
        # Convert string representations of lists back to actual lists
        for key, value in df.attrs.items():
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                try:
                    df.attrs[key] = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    # Keep as string if conversion fails
                    pass
    return df


def _make_paths_relative(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Convert absolute paths in the dataframe to relative paths starting from 'data/'.
    This makes the parquet files portable between team members.

    Args:
        df: DataFrame with path columns
        base_dir: The base directory (synchronized_dataset_dir) used to find the 'data' folder

    Returns:
        DataFrame with relative paths
    """
    # Find the 'data' directory in the path
    data_dir = None
    for parent in base_dir.parents:
        if (parent / "data").exists():
            data_dir = parent
            break

    if data_dir is None:
        # Fallback: just use paths as-is
        return df

    # Convert all path columns to relative paths
    path_columns = ["source_image", "gt_image", "tracking_markers"] + [
        col
        for col in df.columns
        if col
        not in [
            "composite_key",
            "campaign_number",
            "source_image",
            "gt_image",
            "tracking_markers",
            "time_frame",
            "split",
            "has_gt",
            "gt_cell_count",
        ]
    ]

    for col in path_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(Path(x).relative_to(data_dir))
                if isinstance(x, str) and x and Path(x).is_absolute()
                else x
            )

    return df


def count_cells_in_image(image_path: str) -> int:
    """
    Count the number of cells in an image.
    Assumes instance segmentation where each cell has a unique integer ID > 0.
    """
    if not image_path or not isinstance(image_path, (str, Path)):
        return 0

    try:
        if not Path(image_path).exists():
            return 0

        img = tifffile.imread(image_path)
        # Assuming background is 0
        unique_labels = np.unique(img)
        # Subtract 1 for background if 0 is present
        count = len(unique_labels) - 1 if 0 in unique_labels else len(unique_labels)
        return count
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return 0


def _has_valid_gt_path(path_value: Any) -> bool:
    return (
        path_value is not None
        and isinstance(path_value, (str, Path))
        and Path(path_value).exists()
    )


def _ensure_supervised_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "gt_image" not in df.columns:
        df["has_gt"] = False
        df["gt_cell_count"] = 0
        return df

    if "has_gt" not in df.columns:
        df["has_gt"] = df["gt_image"].apply(_has_valid_gt_path)
    else:
        df["has_gt"] = df["has_gt"].fillna(False).astype(bool)

    if "gt_cell_count" not in df.columns:
        df["gt_cell_count"] = 0
        gt_mask = df["has_gt"] & df["gt_image"].notna()
        if gt_mask.any():
            df.loc[gt_mask, "gt_cell_count"] = (
                df.loc[gt_mask, "gt_image"].apply(count_cells_in_image).astype(int)
            )
    else:
        df["gt_cell_count"] = (
            pd.to_numeric(df["gt_cell_count"], errors="coerce").fillna(0).astype(int)
        )

    df.loc[~df["has_gt"], "gt_cell_count"] = 0
    return df


def _parse_split_ratios(
    split_ratios: str, expected: int, mode_name: str
) -> list[float]:
    ratios = [float(r) for r in split_ratios.split(",")]
    if len(ratios) != expected:
        raise ValueError(
            f"{mode_name} requires exactly {expected} ratios. Got {len(ratios)}: {split_ratios}"
        )
    total = sum(ratios)
    if total <= 0:
        raise ValueError(
            f"{mode_name} ratios must sum to a positive value: {split_ratios}"
        )
    return [ratio / total for ratio in ratios]


def _summarize_split_counts(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split_name in ["train", "validation", "test"]:
        split_df = df[df["split"] == split_name].copy()
        gt_df = split_df[split_df["has_gt"]].copy()
        summary[split_name] = {
            "raw_frames": int(len(split_df)),
            "gt_images": int(len(gt_df)),
            "gt_cells": int(gt_df["gt_cell_count"].sum()) if not gt_df.empty else 0,
            "campaigns": sorted(
                split_df["campaign_number"].dropna().astype(str).unique().tolist()
            )
            if "campaign_number" in split_df.columns
            else [],
            "time_frame_min": int(split_df["time_frame"].min())
            if not split_df.empty and "time_frame" in split_df.columns
            else None,
            "time_frame_max": int(split_df["time_frame"].max())
            if not split_df.empty and "time_frame" in split_df.columns
            else None,
            "gt_time_frames": sorted(gt_df["time_frame"].dropna().astype(int).tolist())
            if not gt_df.empty and "time_frame" in gt_df.columns
            else [],
            "gt_composite_keys": gt_df["composite_key"].dropna().astype(str).tolist()
            if "composite_key" in gt_df.columns
            else [],
        }
    return summary


def _evaluate_fold_constraints(
    split_summary: dict[str, dict[str, Any]],
    *,
    held_in_gt_cells: int,
    train_seq: str,
    test_seq: str,
) -> dict[str, Any]:
    train_summary = split_summary["train"]
    validation_summary = split_summary["validation"]
    test_summary = split_summary["test"]

    actual_val_fraction = (
        validation_summary["gt_cells"] / held_in_gt_cells if held_in_gt_cells else 0.0
    )
    train_val_campaigns = set(train_summary["campaigns"]) | set(
        validation_summary["campaigns"]
    )
    test_campaigns = set(test_summary["campaigns"])

    checks = {
        "test_sequence_is_disjoint": train_val_campaigns == {train_seq}
        and test_campaigns == {test_seq}
        and train_val_campaigns.isdisjoint(test_campaigns),
        "train_supervised_cells_gt_validation": train_summary["gt_cells"]
        > validation_summary["gt_cells"],
        "validation_supervised_cells_min_fraction": actual_val_fraction >= 0.10,
        "validation_supervised_cells_max_fraction": actual_val_fraction <= 0.40,
        "validation_gt_images_min": validation_summary["gt_images"] >= 1,
        "train_gt_images_min": train_summary["gt_images"] >= 2,
        "test_gt_images_min": test_summary["gt_images"] >= 1,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "actual_val_fraction_by_gt_cells": float(actual_val_fraction),
    }


def _apply_fold_candidate(
    df: pd.DataFrame,
    *,
    train_seq: str,
    test_seq: str,
    validation_gt_keys: set[str],
    selection_mode: str,
    held_in_gt_cells: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    result["split"] = "train"
    result.loc[result["campaign_number"] == test_seq, "split"] = "test"

    train_mask = result["campaign_number"] == train_seq
    held_in_gt = result[train_mask & result["has_gt"]].copy()
    validation_gt = held_in_gt[
        held_in_gt["composite_key"].isin(validation_gt_keys)
    ].copy()

    if selection_mode == "contiguous_gt_block" and not validation_gt.empty:
        min_time = int(validation_gt["time_frame"].min())
        max_time = int(validation_gt["time_frame"].max())
        validation_mask = train_mask & result["time_frame"].between(min_time, max_time)
        result.loc[validation_mask, "split"] = "validation"
    else:
        result.loc[result["composite_key"].isin(validation_gt_keys), "split"] = (
            "validation"
        )

    split_summary = _summarize_split_counts(result)
    constraint_eval = _evaluate_fold_constraints(
        split_summary,
        held_in_gt_cells=held_in_gt_cells,
        train_seq=train_seq,
        test_seq=test_seq,
    )
    validation_frames = split_summary["validation"]["raw_frames"]
    validation_gt_images = split_summary["validation"]["gt_images"]
    target_gt_images = max(
        1, round(len(held_in_gt) * constraint_eval["actual_val_fraction_by_gt_cells"])
    )

    audit = {
        "selection_mode": selection_mode,
        "selected_validation_gt_keys": sorted(validation_gt_keys),
        "selected_validation_gt_images": validation_gt["gt_image"]
        .dropna()
        .astype(str)
        .tolist()
        if "gt_image" in validation_gt.columns
        else [],
        "selected_validation_time_frames": sorted(
            validation_gt["time_frame"].dropna().astype(int).tolist()
        )
        if "time_frame" in validation_gt.columns
        else [],
        "split_summary": split_summary,
        "constraints": constraint_eval,
        "score": (
            0 if constraint_eval["passed"] else 1,
            abs(constraint_eval["actual_val_fraction_by_gt_cells"]),
            abs(validation_gt_images - target_gt_images),
            validation_frames,
        ),
    }
    return result, audit


def _score_candidate(
    audit: dict[str, Any],
    *,
    target_val_fraction: float,
    fallback_preference: int,
) -> tuple[Any, ...]:
    summary = audit["split_summary"]["validation"]
    actual_val_fraction = audit["constraints"]["actual_val_fraction_by_gt_cells"]
    return (
        0 if audit["constraints"]["passed"] else 1,
        abs(actual_val_fraction - target_val_fraction),
        fallback_preference,
        summary["gt_images"],
        summary["raw_frames"],
    )


def _select_validation_gt_key_candidates(
    held_in_gt: pd.DataFrame, target_val_fraction: float
) -> tuple[dict[str, set[str]], str]:
    held_in_gt = held_in_gt.sort_values("time_frame").reset_index(drop=True)
    if held_in_gt.empty:
        return {
            "contiguous_gt_block": set(),
            "subset_sum_fallback": set(),
        }, "contiguous_gt_block"

    total_cells = int(held_in_gt["gt_cell_count"].sum())
    target_cells = total_cells * target_val_fraction
    target_gt_images = max(1, round(len(held_in_gt) * target_val_fraction))

    best_contiguous: tuple[tuple[Any, ...], set[str]] | None = None
    for start_idx in range(len(held_in_gt)):
        running_cells = 0
        for end_idx in range(start_idx, len(held_in_gt)):
            running_cells += int(held_in_gt.iloc[end_idx]["gt_cell_count"])
            candidate_keys = set(
                held_in_gt.iloc[start_idx : end_idx + 1]["composite_key"].astype(str)
            )
            score = (
                abs(running_cells - target_cells),
                abs(len(candidate_keys) - target_gt_images),
                len(candidate_keys),
            )
            if best_contiguous is None or score < best_contiguous[0]:
                best_contiguous = (score, candidate_keys)

    counts = held_in_gt["gt_cell_count"].astype(int).tolist()
    keys = held_in_gt["composite_key"].astype(str).tolist()
    states: dict[int, list[int]] = {0: []}
    for idx, count in enumerate(counts):
        updated = dict(states)
        for running_sum, subset in states.items():
            new_sum = running_sum + count
            new_subset = subset + [idx]
            existing_subset = updated.get(new_sum)
            if existing_subset is None or abs(len(new_subset) - target_gt_images) < abs(
                len(existing_subset) - target_gt_images
            ):
                updated[new_sum] = new_subset
        states = updated

    best_subset: tuple[tuple[Any, ...], set[str]] | None = None
    for running_sum, subset in states.items():
        if not subset:
            continue
        candidate_keys = {keys[idx] for idx in subset}
        score = (
            abs(running_sum - target_cells),
            abs(len(subset) - target_gt_images),
            len(subset),
        )
        if best_subset is None or score < best_subset[0]:
            best_subset = (score, candidate_keys)

    candidates = {
        "contiguous_gt_block": best_contiguous[1]
        if best_contiguous is not None
        else set(keys[:1]),
        "subset_sum_fallback": best_subset[1]
        if best_subset is not None
        else set(keys[:1]),
    }
    preferred_mode = (
        "subset_sum_fallback"
        if best_subset is not None
        and best_contiguous is not None
        and best_subset[0] < best_contiguous[0]
        else "contiguous_gt_block"
    )
    return candidates, preferred_mode


def _safe_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _build_split_audit_markdown(audit: dict[str, Any]) -> str:
    counts = audit["split_counts"]
    lines = [
        f"# Split Audit — {audit['dataset']} {audit['fold']}",
        "",
        f"- Created at: {audit['created_at']}",
        f"- Split strategy: {audit['split_strategy']}",
        f"- Validation selection mode: {audit['validation_selection_mode']}",
        f"- Target validation fraction by GT cells: {audit['target_val_fraction']:.4f}",
        f"- Actual validation fraction by GT cells: {audit['actual_val_fraction_by_gt_cells']:.4f}",
        f"- Whole-image hard constraints passed: {audit['whole_image_hard_constraints_passed']}",
        "",
        "| Split | Raw frames | GT images | GT cells | Campaigns |",
        "|---|---:|---:|---:|---|",
    ]
    for split_name in ["train", "validation", "test"]:
        split_counts = counts[split_name]
        lines.append(
            f"| {split_name} | {split_counts['raw_frames']} | {split_counts['gt_images']} | {split_counts['gt_cells']} | {', '.join(split_counts['campaigns']) or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Validation GT frames",
            "",
            ", ".join(audit["selected_validation_gt_keys"]) or "(none)",
            "",
            "## Whole-image hard constraints",
            "",
        ]
    )
    for check_name, passed in audit["whole_image_constraints"].items():
        lines.append(f"- {check_name}: {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines) + "\n"


def _write_split_audit_files(audit: dict[str, Any], output_path: Path) -> None:
    audit_dir = output_path.parent.parent / "split_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_prefix = audit_dir / f"{audit['dataset']}_{audit['fold']}_split_audit"
    audit_prefix.with_suffix(".json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    audit_prefix.with_suffix(".md").write_text(
        _build_split_audit_markdown(audit), encoding="utf-8"
    )


def add_stratified_split(df: pd.DataFrame, split_ratios: str) -> pd.DataFrame:
    """
    Splits data into train/val/test based on cell counts for images WITH Ground Truth.
    Images without Ground Truth are marked as 'unlabeled'.
    """
    # 1. Parse Ratios
    try:
        ratios = _parse_split_ratios(split_ratios, expected=3, mode_name="Mixed mode")
    except ValueError as e:
        print(f"Invalid split ratios: {e}. Returning unsplit df.")
        return df

    df = _ensure_supervised_columns(df)

    # 3. Split the DataFrame
    df_labeled = df[df["has_gt"]].copy()
    df_unlabeled = df[~df["has_gt"]].copy()
    print(f"Labeled images (for training): {len(df_labeled)}")
    print(f"Unlabeled images (for inference/silver truth): {len(df_unlabeled)}")

    # 4. Stratify ONLY the Labeled Data
    if not df_labeled.empty:
        # Sort desc (Greedy Bin Packing)
        df_sorted = df_labeled.sort_values(by="gt_cell_count", ascending=False)

        total_cells = df_sorted["gt_cell_count"].sum()
        target_train = total_cells * ratios[0]
        target_val = total_cells * ratios[1]

        current_train = 0
        current_val = 0
        splits = []

        for count in df_sorted["gt_cell_count"]:
            if current_train + count <= target_train:
                splits.append("train")
                current_train += count
            elif current_val + count <= target_val:
                splits.append("validation")
                current_val += count
            else:
                splits.append("test")

        df_sorted["split"] = splits
        df_labeled = df_sorted

    # 5. Handle Unlabeled Data
    df_unlabeled["split"] = "unlabeled"

    # 6. Recombine
    df_final = pd.concat([df_labeled, df_unlabeled])

    return df_final


def add_fold_split(df: pd.DataFrame, mode: str, split_ratios: str) -> pd.DataFrame:
    """
    Add 'split' column based on Fold strategy (Leave-One-Sequence-Out).

    Args:
        df: DataFrame with 'campaign_number' and 'time_frame'.
        mode: 'fold-1' (Train 01, Test 02) or 'fold-2' (Train 02, Test 01).
        split_ratios: String "train,val" (e.g. "80,20") defining split of the training sequence.
    """
    if "campaign_number" not in df.columns:
        raise ValueError("Dataframe must have 'campaign_number' column.")
    if "time_frame" not in df.columns:
        # Fallback if time_frame missing? Maybe try to parse from composite_key?
        # But we added it to extraction, so should be there if re-generated.
        # If loading old parquet, might need to extract.
        pass  # Assume it's there for now as we updated extraction.

    train_seq = "01" if mode == "fold-1" else "02"
    test_seq = "02" if mode == "fold-1" else "01"

    try:
        ratios = _parse_split_ratios(split_ratios, expected=2, mode_name="Fold mode")
        val_ratio = ratios[1]
        print(
            f"Fold Split: Using Validation Ratio {val_ratio:.2f} (from {split_ratios})"
        )
    except ValueError as e:
        print(f"Error parsing split ratios for fold mode: {e}")
        raise

    print(f"Fold Config: Train on {train_seq}, Test on {test_seq}")

    result = _ensure_supervised_columns(df)
    if "time_frame" not in result.columns:
        result["time_frame"] = result["composite_key"].apply(
            lambda x: int(str(x).split("_")[1].split(".")[0])
        )

    train_seq_gt = result[
        (result["campaign_number"] == train_seq) & result["has_gt"]
    ].copy()
    train_seq_gt = train_seq_gt.sort_values("time_frame")
    held_in_gt_cells = int(train_seq_gt["gt_cell_count"].sum())

    if train_seq_gt.empty:
        raise ValueError(
            f"No GT-labeled frames found in held-in training sequence {train_seq}."
        )

    candidate_keys_by_mode, preferred_mode = _select_validation_gt_key_candidates(
        train_seq_gt, val_ratio
    )
    best_result: pd.DataFrame | None = None
    best_audit: dict[str, Any] | None = None

    for selection_mode in [
        preferred_mode,
        "subset_sum_fallback",
        "contiguous_gt_block",
    ]:
        candidate_result, candidate_audit = _apply_fold_candidate(
            result,
            train_seq=train_seq,
            test_seq=test_seq,
            validation_gt_keys=candidate_keys_by_mode[selection_mode],
            selection_mode=selection_mode,
            held_in_gt_cells=held_in_gt_cells,
        )
        candidate_score = _score_candidate(
            candidate_audit,
            target_val_fraction=val_ratio,
            fallback_preference=0 if selection_mode == preferred_mode else 1,
        )
        if best_audit is None or candidate_score < _score_candidate(
            best_audit,
            target_val_fraction=val_ratio,
            fallback_preference=0
            if best_audit["selection_mode"] == preferred_mode
            else 1,
        ):
            best_result = candidate_result
            best_audit = candidate_audit

    assert best_result is not None and best_audit is not None
    split_counts = best_audit["split_summary"]
    print(
        "Split Stats: Train=%d, Val=%d, Test=%d | Val GT cells frac=%.3f"
        % (
            split_counts["train"]["raw_frames"],
            split_counts["validation"]["raw_frames"],
            split_counts["test"]["raw_frames"],
            best_audit["constraints"]["actual_val_fraction_by_gt_cells"],
        )
    )

    audit = {
        "dataset": None,
        "fold": mode,
        "train_sequence": train_seq,
        "test_sequence": test_seq,
        "split_strategy": "supervised_gt_frame_cells",
        "split_unit": "gt_cells",
        "target_val_fraction": float(val_ratio),
        "actual_val_fraction_by_gt_cells": float(
            best_audit["constraints"]["actual_val_fraction_by_gt_cells"]
        ),
        "actual_val_fraction_by_qa_rows": None,
        "train_gt_images": split_counts["train"]["gt_images"],
        "validation_gt_images": split_counts["validation"]["gt_images"],
        "test_gt_images": split_counts["test"]["gt_images"],
        "train_gt_cells": split_counts["train"]["gt_cells"],
        "validation_gt_cells": split_counts["validation"]["gt_cells"],
        "test_gt_cells": split_counts["test"]["gt_cells"],
        "train_qa_rows": None,
        "validation_qa_rows": None,
        "test_qa_rows": None,
        "selected_validation_gt_images": best_audit["selected_validation_gt_images"],
        "selected_validation_gt_keys": best_audit["selected_validation_gt_keys"],
        "validation_selection_mode": best_audit["selection_mode"],
        "whole_image_hard_constraints_passed": bool(
            best_audit["constraints"]["passed"]
        ),
        "whole_image_constraints": best_audit["constraints"]["checks"],
        "split_counts": split_counts,
        "git_sha": _safe_git_sha(),
        "dvc_stage": f"create_{mode.replace('-', '')}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    best_result.attrs = dict(df.attrs)
    best_result.attrs[SPLIT_AUDIT_ATTR] = json.dumps(audit)
    return best_result


def create_dataset_dataframe_logic(
    synchronized_dataset_dir: Path | str,
    output_path: Path | str,
    split_mode: str = "mixed",
    split_ratios: Optional[str] = None,
    seed: int = 42,
) -> None:
    synchronized_dataset_dir = Path(
        synchronized_dataset_dir
    ).resolve()  # Use absolute path for processing
    if output_path is None:
        output_path = f"{synchronized_dataset_dir.name}_dataset_dataframe.parquet"

    output_path = Path(output_path)

    dataset_info, competitor_columns = process_dataset_directory(
        synchronized_dataset_dir
    )
    dataset_dataframe = convert_to_dataframe(dataset_info)
    dataset_dataframe = _ensure_supervised_columns(dataset_dataframe)

    # Apply split based on mode
    print(f"Applying split mode: {split_mode}")
    if split_mode == "mixed":
        # Default for mixed if not provided
        ratios = split_ratios if split_ratios else "70,15,15"
        dataset_dataframe = add_stratified_split(dataset_dataframe, ratios)
    elif split_mode.startswith("fold-"):
        # Default for fold if not provided (80,20 to match previous 0.2 val_ratio)
        ratios = split_ratios if split_ratios else "80,20"
        dataset_dataframe = add_fold_split(dataset_dataframe, split_mode, ratios)
    else:
        print(f"Unknown split mode '{split_mode}', applying default mixed split.")
        ratios = split_ratios if split_ratios else "70,15,15"
        dataset_dataframe = add_stratified_split(dataset_dataframe, ratios)

    # Convert paths to relative (starting with data/)
    dataset_dataframe = _make_paths_relative(
        dataset_dataframe, synchronized_dataset_dir
    )

    # Store metadata
    dataset_dataframe.attrs["base_directory"] = str(synchronized_dataset_dir)
    dataset_dataframe.attrs["competitor_columns"] = list(competitor_columns)
    reference_columns = []
    if SILVER_TRUTH_COLUMN in dataset_dataframe.columns:
        reference_columns.append(SILVER_TRUTH_COLUMN)
    dataset_dataframe.attrs[REFERENCE_COLUMNS_ATTR] = reference_columns
    dataset_dataframe.attrs["created_by"] = "David-Ciz"  #
    dataset_dataframe.attrs["creation_time"] = pd.Timestamp.now()

    split_audit_json = dataset_dataframe.attrs.pop(SPLIT_AUDIT_ATTR, None)
    if split_audit_json:
        split_audit = json.loads(split_audit_json)
        split_audit["dataset"] = synchronized_dataset_dir.name
        dataset_dataframe.attrs.update(
            {
                "split_strategy": split_audit["split_strategy"],
                "split_unit": split_audit["split_unit"],
                "target_val_fraction": split_audit["target_val_fraction"],
                "actual_val_fraction_by_gt_cells": split_audit[
                    "actual_val_fraction_by_gt_cells"
                ],
                "validation_selection_mode": split_audit["validation_selection_mode"],
                "selected_validation_gt_keys": split_audit[
                    "selected_validation_gt_keys"
                ],
                "whole_image_hard_constraints_passed": split_audit[
                    "whole_image_hard_constraints_passed"
                ],
                "git_sha": split_audit["git_sha"],
                "dvc_stage": split_audit["dvc_stage"],
                "created_at": split_audit["created_at"],
            }
        )

    save_dataframe_to_parquet_with_metadata(dataset_dataframe, str(output_path))

    if split_audit_json:
        split_audit = json.loads(split_audit_json)
        split_audit["dataset"] = synchronized_dataset_dir.name
        _write_split_audit_files(split_audit, output_path)
