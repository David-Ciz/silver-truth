import ast

import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Set
import pyarrow as pa
import pyarrow.parquet as pq

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
) -> tuple[Dict[str, Dict[str, Any]], Set[str]]:
    """
    Process the dataset directory structure and extract file information.

    Args:
        directory: Path to the synchronized dataset directory

    Returns:
        Dictionary containing organized dataset information
    """
    dataset_info = defaultdict(dict)
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
            image_number = image.name[-8:]
            composite_key = f"{campaign_number}_{image_number}"
            dataset_info[composite_key]["source_image"] = str(image)
            dataset_info[composite_key]["campaign_number"] = campaign_number


def process_gt_data(
    folder: Path, campaign_number: str, dataset_info: Dict[str, Dict[str, Any]]
) -> None:
    """Process ground truth and tracking marker files."""
    gt_subfolder = folder / SEG_FOLDER
    tra_subfolder = folder / TRA_FOLDER

    # Process ground truth images
    for image in gt_subfolder.iterdir():
        if image.suffix == ".tif":
            image_number = image.name[-8:]
            composite_key = f"{campaign_number}_{image_number}"
            dataset_info[composite_key]["gt_image"] = str(image)
            dataset_info[composite_key]["campaign_number"] = campaign_number

    # Process tracking marker images
    for image in tra_subfolder.iterdir():
        if image.suffix == ".tif":
            image_number = image.name[-8:]
            composite_key = f"{campaign_number}_{image_number}"
            dataset_info[composite_key]["tracking_markers"] = str(image)
            dataset_info[composite_key]["campaign_number"] = campaign_number


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
            image_number = image.name[-8:]
            composite_key = f"{campaign_number}_{image_number}"
            dataset_info[composite_key][str(competitor_folder.name)] = str(image)
            dataset_info[composite_key]["campaign_number"] = campaign_number


def extract_image_number(filename: str) -> str:
    """Extract the image number from a filename."""
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
    all_columns = set()
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
