import click
import pandas as pd
from pathlib import Path
import data_processing.utils.dataset_dataframe_creation as ddc


def normalize_gt_image_path(path: str) -> str:
    """
    Normalize gt_image path to enable matching between relative and absolute paths.
    Extracts the relevant portion: campaign_GT/SEG/man_segXXXX.tif
    """
    if pd.isna(path):
        return path
    path_obj = Path(path)
    parts = path_obj.parts
    # Find the index of the campaign folder (e.g., "01_GT" or "02_GT")
    for i, part in enumerate(parts):
        if part.endswith("_GT"):
            # Return the path from the campaign_GT folder onwards
            return str(Path(*parts[i:]))
    # Fallback: return just the filename
    return path_obj.name


def synchronize_splits(
    source_df: pd.DataFrame, target_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Synchronizes the 'split' column from a source DataFrame to a target DataFrame.

    The synchronization is based on the 'gt_image' and 'label' columns. Any row
    in the target DataFrame that has the same combination of 'gt_image' and 'label'
    as a row in the source DataFrame will receive the 'split' value from that source row.

    Handles both relative and absolute paths by normalizing them before matching.

    Args:
        source_df: DataFrame with the correct 'split' information.
        target_df: DataFrame to be updated.

    Returns:
        The target DataFrame with the 'split' column updated.
    """
    # Create a mapping from (gt_image, label) to split value from the source dataframe.
    # Drop duplicates to ensure one mapping per key.
    # We only need the 'gt_image', 'label', and 'split' columns.
    if "split" not in source_df.columns:
        raise ValueError("Source dataframe must contain a 'split' column.")

    # Create copies to avoid modifying originals
    source_copy = source_df[["gt_image", "label", "split"]].dropna().copy()
    target_df_copy = target_df.copy()

    # Normalize gt_image paths for matching
    source_copy["gt_image_normalized"] = source_copy["gt_image"].apply(
        normalize_gt_image_path
    )
    target_df_copy["gt_image_normalized"] = target_df_copy["gt_image"].apply(
        normalize_gt_image_path
    )

    # Create the mapping dataframe with normalized paths
    split_mapping_df = source_copy[
        ["gt_image_normalized", "label", "split"]
    ].drop_duplicates()

    # Merge the target dataframe with the split mapping using normalized paths
    merged_df = pd.merge(
        target_df_copy.drop(columns=["split"], errors="ignore"),
        split_mapping_df,
        on=["gt_image_normalized", "label"],
        how="left",
    )

    # Remove the temporary normalized column
    merged_df = merged_df.drop(columns=["gt_image_normalized"])

    return merged_df


@click.command()
@click.option(
    "--source-parquet",
    required=True,
    type=click.Path(exists=True),
    help="Path to the source Parquet file with split information.",
)
@click.option(
    "--target-parquet",
    required=True,
    type=click.Path(exists=True),
    help="Path to the target Parquet file to be updated.",
)
@click.option(
    "--output-parquet",
    required=True,
    help="Path to save the updated target Parquet file.",
)
def main(source_parquet, target_parquet, output_parquet):
    """
    Synchronize 'split' column between two Parquet files.
    """
    print("Loading dataframes...")
    source_df = ddc.load_dataframe_from_parquet_with_metadata(source_parquet)
    target_df = ddc.load_dataframe_from_parquet_with_metadata(target_parquet)

    print("Synchronizing splits...")
    updated_target_df = synchronize_splits(source_df, target_df)

    # Preserve metadata from the original target dataframe
    if hasattr(target_df, "attrs"):
        updated_target_df.attrs = target_df.attrs

    print(f"Saving updated dataframe to {output_parquet}...")
    ddc.save_dataframe_to_parquet_with_metadata(updated_target_df, output_parquet)

    print("Synchronization complete.")


if __name__ == "__main__":
    main()
