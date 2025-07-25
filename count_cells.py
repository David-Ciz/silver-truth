import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_dataframe_with_competitors(
    dataset_dataframe_path: Path,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Loads a dataframe and identifies competitor columns."""
    try:
        df = pd.read_parquet(dataset_dataframe_path)
        competitor_columns = df.attrs.get("competitor_columns", [])

        if not competitor_columns:
            logging.warning(
                "No 'competitor_columns' found in dataframe attributes. Inferring from columns."
            )
            potential_cols = [
                col
                for col in df.columns
                if col
                not in [
                    "composite_key",
                    "raw_image",
                    "gt_image",
                    "campaign_number",
                    "sequence_id",
                    "time_id",
                ]
                and isinstance(df[col].iloc[0], str)
                and Path(df[col].iloc[0]).suffix in [".tif", ".tiff"]
            ]
            if potential_cols:
                competitor_columns = potential_cols
                logging.info(f"Inferred competitor columns: {competitor_columns}")
            else:
                logging.error("Could not infer competitor columns.")
                return None, []
        return df, competitor_columns
    except Exception as e:
        logging.error(f"Failed to load or process dataframe: {e}")
        return None, []


@click.command()
@click.argument("dataset_dataframe_path", type=click.Path(exists=True, path_type=Path))
def count_total_cells(dataset_dataframe_path: Path):
    """
    Counts the total number of unique cells and processed files for each competitor
    for images that have a corresponding ground truth.

    DATASET_DATAFRAME_PATH: Path to the dataset dataframe Parquet file.
    """
    df, competitor_columns = load_dataframe_with_competitors(dataset_dataframe_path)
    if df is None:
        return

    # --- Data Filtering ---
    initial_rows = len(df)
    df_gt = df[df["gt_image"].notna()].copy()
    df_gt = df_gt[df_gt["gt_image"].apply(lambda p: Path(p).exists())]

    if len(df_gt) < initial_rows:
        logging.info(
            f"Filtered out {initial_rows - len(df_gt)} rows with missing or non-existent ground truth paths."
        )

    if df_gt.empty:
        logging.error("No rows remain after filtering for valid ground truth images.")
        return

    total_cell_count = 0
    per_competitor_counts = {
        comp: {"cells": 0, "files": 0} for comp in competitor_columns
    }

    logging.info("Starting cell count...")
    for comp in tqdm(competitor_columns, desc="Competitors"):
        comp_cell_count = 0
        files_processed = 0
        for seg_path_str in tqdm(
            df_gt[comp].dropna(), desc=f"Images for {comp}", leave=False
        ):
            seg_path = Path(seg_path_str)
            if seg_path.exists():
                files_processed += 1
                try:
                    seg_image = tifffile.imread(seg_path)
                    unique_labels = np.unique(seg_image)
                    num_cells = len(unique_labels) - (1 if 0 in unique_labels else 0)
                    comp_cell_count += num_cells
                except Exception as e:
                    logging.warning(f"Could not process file {seg_path}: {e}")
            else:
                logging.warning(f"Path not found: {seg_path}")

        per_competitor_counts[comp]["cells"] = comp_cell_count
        per_competitor_counts[comp]["files"] = files_processed
        total_cell_count += comp_cell_count

    # --- Print Results ---
    print("\n--- Cell Count Summary ---")
    for competitor, counts in per_competitor_counts.items():
        print(
            f"- Competitor '{competitor}': {counts['cells']:,} cells in {counts['files']:,} files"
        )
    print("--------------------------")
    print(f"Total potential images to be generated: {total_cell_count:,}")


if __name__ == "__main__":
    count_total_cells()
