import os
import json
from src.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)


def load_competitor_config(config_path: str = "competitor_config.json") -> dict:
    """
    Load competitor configuration from JSON file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Dictionary mapping dataset names to lists of competitor names
    """
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"Warning: Configuration file {config_path} not found. Using all available competitors.")
        return {}


def generate_job_file(
    parquet_file_path: str,
    campaign_number: str,
    output_dir: str,
    tracking_marker_column: str = "tracking_markers",
    competitor_columns: list[str] | None = None,
    config_path: str = "competitor_config.json",
):
    """
    Generates a job file for a specific campaign, listing competitor results and tracking markers.

    Args:
        parquet_file_path (str): Path to the Parquet dataset file.
        campaign_number (str): The campaign number (e.g., "01").
        output_dir (str): Directory where the job file will be saved.
        tracking_marker_column (str): The column name in the parquet file that contains the tracking marker paths.
        competitor_columns (list[str]): A list of column names in the parquet file that contain competitor result paths.
                                        If None, will use configuration file or all available competitors.
        config_path (str): Path to the competitor configuration JSON file.
    """
    df = load_dataframe_from_parquet_with_metadata(parquet_file_path)
    
    # Extract dataset name from parquet file path
    parquet_path = os.path.basename(parquet_file_path)
    dataset_name = parquet_path.replace("_dataset_dataframe.parquet", "")

    # Load competitor configuration
    competitor_config = load_competitor_config(config_path)

    # Filter for the specific campaign number
    filtered_df = df[df["campaign_number"] == campaign_number]

    if filtered_df.empty:
        print(f"No data found for campaign number: {campaign_number}")
        return

    # Determine competitor columns if not provided
    if competitor_columns is None:
        # First try to get from configuration
        if dataset_name in competitor_config:
            competitor_columns = competitor_config[dataset_name]
            print(f"Using configured competitors for {dataset_name}: {competitor_columns}")
        else:
            # Fallback to all available competitors
            all_columns = df.columns.tolist()
            exclude_columns = [
                "composite_key",
                "gt_image",
                "source_image",
                tracking_marker_column,
                "campaign_number",
            ]
            competitor_columns = [col for col in all_columns if col not in exclude_columns]
            print(f"No configuration found for {dataset_name}, using all available competitors: {competitor_columns}")

    # Validate that configured competitors exist in the dataframe
    available_competitors = [col for col in competitor_columns if col in df.columns]
    missing_competitors = [col for col in competitor_columns if col not in df.columns]
    
    if missing_competitors:
        print(f"Warning: The following configured competitors are not available in {dataset_name}: {missing_competitors}")
    
    if not available_competitors:
        print(f"Error: No valid competitors found for dataset {dataset_name}")
        return
        
    print(f"Will use competitors: {available_competitors}")

    job_file_content = []

    # Extract unique base paths for competitors and add to job file
    competitor_paths = set()
    for col in available_competitors:
        if col in filtered_df.columns:
            # Take the first non-null path and auto-detect the numeric format
            sample_path = filtered_df[col].dropna().iloc[0]
            base_path = os.path.dirname(sample_path)
            filename = os.path.basename(sample_path)
            
            # Auto-detect the numeric format in the filename
            if "mask" in filename:
                # Find the number of digits used for numbering
                import re
                match = re.search(r'mask(\d+)\.tif', filename)
                if match:
                    num_digits = len(match.group(1))
                    if num_digits == 3:
                        formatted_path = os.path.join(base_path, "maskTTT.tif")
                    else:  # Default to 4 digits
                        formatted_path = os.path.join(base_path, "maskTTTT.tif")
                else:
                    # Fallback to TTTT if pattern doesn't match
                    formatted_path = os.path.join(base_path, "maskTTTT.tif")
            else:
                # Fallback to TTTT if no mask pattern found
                formatted_path = os.path.join(base_path, "maskTTTT.tif")
            competitor_paths.add(formatted_path)

    for path in sorted(list(competitor_paths)):
        job_file_content.append(path)

    # Add tracking markers as the last line
    if not filtered_df[tracking_marker_column].empty:
        sample_tracking_path = filtered_df[tracking_marker_column].dropna().iloc[0]
        base_tracking_path = os.path.dirname(sample_tracking_path)
        tracking_filename = os.path.basename(sample_tracking_path)
        
        # Auto-detect the numeric format in the tracking filename
        if "man_track" in tracking_filename:
            import re
            match = re.search(r'man_track(\d+)\.tif', tracking_filename)
            if match:
                num_digits = len(match.group(1))
                if num_digits == 3:
                    formatted_tracking_path = os.path.join(base_tracking_path, "man_trackTTT.tif")
                else:  # Default to 4 digits
                    formatted_tracking_path = os.path.join(base_tracking_path, "man_trackTTTT.tif")
            else:
                # Fallback to TTTT if pattern doesn't match
                formatted_tracking_path = os.path.join(base_tracking_path, "man_trackTTTT.tif")
        else:
            # Fallback to TTTT if no man_track pattern found
            formatted_tracking_path = os.path.join(base_tracking_path, "man_trackTTTT.tif")
        job_file_content.append(formatted_tracking_path)
    else:
        print(f"Warning: No tracking markers found for campaign: {campaign_number}")

    output_file_name = f"{dataset_name}_{campaign_number}_job_file.txt"
    output_file_path = os.path.join(output_dir, output_file_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file_path, "w") as f:
        for line in job_file_content:
            f.write(f"{line}\n")

    print(f"Job file generated at: {output_file_path}")
