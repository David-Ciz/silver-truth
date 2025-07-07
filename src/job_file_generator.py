import os
from src.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)


def generate_job_file(
    parquet_file_path: str,
    campaign_number: str,
    output_dir: str,
    tracking_marker_column: str = "tracking_markers",
    competitor_columns: list[str] | None = None,
):
    """
    Generates a job file for a specific campaign, listing competitor results and tracking markers.

    Args:
        parquet_file_path (str): Path to the Parquet dataset file.
        campaign_number (str): The campaign number (e.g., "01").
        output_dir (str): Directory where the job file will be saved.
        tracking_marker_column (str): The column name in the parquet file that contains the tracking marker paths.
        competitor_columns (list[str]): A list of column names in the parquet file that contain competitor result paths.
                                        If None, all columns except 'composite_key', 'gt_image', 'source_image',
                                        and 'tracking_markers' will be considered competitor columns.
    """
    df = load_dataframe_from_parquet_with_metadata(parquet_file_path)

    # Filter for the specific campaign number
    filtered_df = df[df["campaign_number"] == campaign_number]

    if filtered_df.empty:
        print(f"No data found for campaign number: {campaign_number}")
        return

    # Determine competitor columns if not provided
    if competitor_columns is None:
        # Exclude known non-competitor columns
        all_columns = df.columns.tolist()
        exclude_columns = [
            "composite_key",
            "gt_image",
            "source_image",
            tracking_marker_column,
            "campaign_number",
        ]
        competitor_columns = [col for col in all_columns if col not in exclude_columns]

    job_file_content = []

    # Extract unique base paths for competitors and add to job file
    competitor_paths = set()
    for col in competitor_columns:
        if col in filtered_df.columns:
            # Take the first non-null path and replace the numeric part with TTTT
            sample_path = filtered_df[col].dropna().iloc[0]
            base_path = os.path.dirname(sample_path)
            # Assuming the filename is maskXXXX.tif, replace XXXX with TTTT
            formatted_path = os.path.join(base_path, "maskTTTT.tif")
            competitor_paths.add(formatted_path)

    for path in sorted(list(competitor_paths)):
        job_file_content.append(path)

    # Add tracking markers as the last line
    if not filtered_df[tracking_marker_column].empty:
        sample_tracking_path = filtered_df[tracking_marker_column].dropna().iloc[0]
        base_tracking_path = os.path.dirname(sample_tracking_path)
        # Assuming the filename is man_trackXXXX.tif, replace XXXX with TTTT
        formatted_tracking_path = os.path.join(base_tracking_path, "man_trackTTTT.tif")
        job_file_content.append(formatted_tracking_path)
    else:
        print(f"Warning: No tracking markers found for campaign: {campaign_number}")

    output_file_name = f"campaign_{campaign_number}_job_file.txt"
    output_file_path = os.path.join(output_dir, output_file_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file_path, "w") as f:
        for line in job_file_content:
            f.write(f"{line}\n")

    print(f"Job file generated at: {output_file_path}")
