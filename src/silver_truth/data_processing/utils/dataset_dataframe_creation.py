import ast
import os

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
SEG_FOLDER = "SEG"
TRA_FOLDER = "TRA"
RES_FOLDER_FIRST = "01_RES"
RES_FOLDER_SECOND = "02_RES"


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
    if "competitor_columns" in df.attrs:
        return df.attrs["competitor_columns"]
    else:
        # If attrs not available, return empty list or implement fallback logic
        return []


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
    pq.write_table(table, output_path)


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


def add_stratified_split(df: pd.DataFrame, split_ratios: str) -> pd.DataFrame:
    """
    Splits data into train/val/test based on cell counts for images WITH Ground Truth.
    Images without Ground Truth are marked as 'unlabeled'.
    """
    # 1. Parse Ratios
    try:
        ratios = [float(r) for r in split_ratios.split(",")]
        if len(ratios) != 3:
            raise ValueError("Split ratios must be a list of 3 numbers.")
        total_ratio = sum(ratios)
        ratios = [r / total_ratio for r in ratios]
    except ValueError as e:
        print(f"Invalid split ratios: {e}. Returning unsplit df.")
        return df

    # 2. Identify Rows with Valid GT
    # We define a helper to check if GT exists on disk
    def has_valid_gt(path):
        return path is not None and isinstance(path, str) and os.path.exists(path)

    print("Verifying Ground Truth existence...")
    # Create a mask for valid data
    df["has_gt"] = df["gt_image"].apply(has_valid_gt)
    print(df)
    # 3. Split the DataFrame
    df_labeled = df[df["has_gt"]].copy()
    df_unlabeled = df[~df["has_gt"]].copy()
    print(df_unlabeled)
    print(f"Labeled images (for training): {len(df_labeled)}")
    print(f"Unlabeled images (for inference/silver truth): {len(df_unlabeled)}")

    # 4. Stratify ONLY the Labeled Data
    if not df_labeled.empty:
        # Calculate cell counts (Assuming count_cells_in_image is defined)
        df_labeled["_cell_count"] = df_labeled["gt_image"].apply(count_cells_in_image)

        # Sort desc (Greedy Bin Packing)
        df_sorted = df_labeled.sort_values(by="_cell_count", ascending=False)

        total_cells = df_sorted["_cell_count"].sum()
        target_train = total_cells * ratios[0]
        target_val = total_cells * ratios[1]

        current_train = 0
        current_val = 0
        splits = []

        for count in df_sorted["_cell_count"]:
            if current_train + count <= target_train:
                splits.append("train")
                current_train += count
            elif current_val + count <= target_val:
                splits.append("validation")
                current_val += count
            else:
                splits.append("test")

        df_sorted["split"] = splits
        df_labeled = df_sorted.drop(columns=["_cell_count"])

    # 5. Handle Unlabeled Data
    df_unlabeled["split"] = "unlabeled"

    # 6. Recombine
    df_final = pd.concat([df_labeled, df_unlabeled])

    # Drop the helper column
    df_final = df_final.drop(columns=["has_gt"])

    return df_final

    # Restore original order if needed? The user didn't specify, but let's just drop the temp col
    # and return the df with 'split'. Since we sorted, the index is shuffled.
    # If we want to preserve order we could sort back by index, but it probably doesn't matter for parquet.

    # Print actual splits
    actual_train = df_sorted[df_sorted["split"] == "train"]["_temp_cell_count"].sum()
    actual_val = df_sorted[df_sorted["split"] == "validation"]["_temp_cell_count"].sum()
    actual_test = df_sorted[df_sorted["split"] == "test"]["_temp_cell_count"].sum()

    print(
        f"Actual Split Counts (Cells): Train={actual_train}, Val={actual_val}, Test={actual_test}"
    )
    print(
        f"Actual Ratios: Train={actual_train/total_cells:.2f}, Val={actual_val/total_cells:.2f}, Test={actual_test/total_cells:.2f}"
    )

    df_sorted = df_sorted.drop(columns=["_temp_cell_count"])
    return df_sorted


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

    # Parse ratios
    try:
        ratios = [float(r) for r in split_ratios.split(",")]
        if len(ratios) != 2:
            raise ValueError(
                f"Fold mode requires exactly 2 ratios (Train, Val). Got {len(ratios)}: {split_ratios}"
            )
        # Normalize
        total = sum(ratios)
        val_ratio = ratios[1] / total
        print(
            f"Fold Split: Using Validation Ratio {val_ratio:.2f} (from {split_ratios})"
        )
    except ValueError as e:
        print(f"Error parsing split ratios for fold mode: {e}")
        # Fallback or strict fail? Strict fail is better for clarity.
        raise

    print(f"Fold Config: Train on {train_seq}, Test on {test_seq}")

    # 1. Assign Test Set
    df.loc[df["campaign_number"] == test_seq, "split"] = "test"

    # 2. Assign Train/Validation (Temporal split of Train Sequence)
    # We grab the training sequence data
    train_seq_mask = df["campaign_number"] == train_seq

    # We need to sort by time_frame to split temporally
    # Let's get the indices of the train sequence rows
    train_seq_indices = df[train_seq_mask].index

    # Create a temporary view/copy of the train sequence part
    train_seq_df = df.loc[train_seq_indices].copy()

    # Ensure time_frame is int
    if "time_frame" not in train_seq_df.columns:
        # attempt to parse from key "XX_YYYY.tif" -> YYYY
        # composite_key: "01_0000.tif"
        train_seq_df["time_frame"] = train_seq_df["composite_key"].apply(
            lambda x: int(x.split("_")[1].split(".")[0])
        )

    train_seq_df = train_seq_df.sort_values("time_frame")

    n_train_seq = len(train_seq_df)
    # Calculate split point. We want val_ratio to be validation (end of sequence)
    # Train is (1 - val_ratio)
    split_idx = int(n_train_seq * (1 - val_ratio))

    print(
        f"Splitting Train Sequence {train_seq}: Total {n_train_seq}, Split Index {split_idx}"
    )

    # Get the composite_keys for train and val
    train_keys = set(train_seq_df.iloc[:split_idx]["composite_key"])
    val_keys = set(train_seq_df.iloc[split_idx:]["composite_key"])

    # Update main dataframe
    df.loc[df["composite_key"].isin(train_keys), "split"] = "train"
    df.loc[df["composite_key"].isin(val_keys), "split"] = "validation"

    # Stats
    n_train = len(train_keys)
    n_val = len(val_keys)
    n_test = len(df[df["split"] == "test"])
    print(f"Split Stats: Train={n_train}, Val={n_val}, Test={n_test}")

    return df


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

    dataset_info, competitor_columns = process_dataset_directory(
        synchronized_dataset_dir
    )
    dataset_dataframe = convert_to_dataframe(dataset_info)

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
    dataset_dataframe.attrs["created_by"] = "David-Ciz"  #
    dataset_dataframe.attrs["creation_time"] = pd.Timestamp.now()
    save_dataframe_to_parquet_with_metadata(dataset_dataframe, str(output_path))
