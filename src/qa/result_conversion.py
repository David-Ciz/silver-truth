"""
Convert QA Jaccard evaluation CSV results to detailed parquet format

This script takes existing CSV results from qa_evaluation.py and converts them
to a rich parquet format suitable for detailed analysis and visualization.
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Optional
import os
import src.data_processing.utils.parquet_utils as p_utils

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def convert_qa_csv_to_detailed_parquet(
    csv_path: str, qa_dataframe_path: str, output_path: Optional[str] = None
) -> str:
    """
    Convert QA evaluation CSV to detailed parquet format.

    Args:
        csv_path: Path to CSV file with QA evaluation results
        qa_dataframe_path: Path to original QA dataframe for additional metadata
        output_path: Optional output path for parquet file

    Returns:
        Path to created parquet file
    """
    try:
        # Load CSV results
        logging.info(f"Loading CSV results from: {csv_path}")
        csv_df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(csv_df)} evaluation records")

        # Load QA dataframe for additional metadata
        logging.info(f"Loading QA metadata from: {qa_dataframe_path}")
        qa_df = pd.read_parquet(qa_dataframe_path)

        # Create detailed records
        detailed_records = []

        for _, row in csv_df.iterrows():
            # Get additional metadata from QA dataframe
            qa_row = qa_df[qa_df["cell_id"] == row["cell_id"]]

            if qa_row.empty:
                logging.warning(f"No QA metadata found for cell_id: {row['cell_id']}")
                continue

            qa_info = qa_row.iloc[0]

            # Extract dataset name from various sources
            dataset_name = ""
            if "dataset_name" in qa_info:
                dataset_name = qa_info["dataset_name"]
            elif "stacked_path" in qa_info:
                # Try to extract from path pattern
                path_parts = str(qa_info["stacked_path"]).split("/")
                for part in path_parts:
                    if any(x in part for x in ["BF-C2DL", "DIC-C2DH", "Fluo-C2DL"]):
                        dataset_name = part
                        break

            # Extract sequence name
            sequence_name = qa_info.get("sequence_name", "")
            if not sequence_name:
                sequence_name = qa_info.get("campaign_number", "")

            # Extract time point from original_image_key (e.g., "t002" -> 2)
            time_point = 0
            original_key = row["original_image_key"]
            if isinstance(original_key, str) and original_key.startswith("t"):
                try:
                    time_point = int(original_key[1:].lstrip("0") or "0")
                except ValueError:
                    time_point = 0

            # Create detailed record
            detailed_record = {
                "cell_id": row["label"],  # Use label as the actual cell ID
                "jaccard_score": row["jaccard_score"],
                "cell_area_gt": qa_info.get(
                    "cell_area", 0
                ),  # From QA metadata if available
                "cell_area_pred": 0,  # Not available in QA workflow
                "intersection_area": 0,  # Not directly available
                "matched_pred_label": row[
                    "label"
                ],  # In QA, we're evaluating specific labels
                "dataset_name": dataset_name,
                "sequence_name": sequence_name,
                "time_point": time_point,
                "competitor": row["competitor"],
                "campaign_number": row["campaign"],
                "original_image_key": row["original_image_key"],
                "qa_cell_id": row["cell_id"],  # QA-specific cell identifier
                "stacked_path": row["stacked_path"],
                "crop_coordinates": qa_info.get("crop_coordinates", ""),
                "crop_size": qa_info.get("crop_size", 0),
                "evaluation_type": "qa_cropped",  # Distinguish from full image evaluation
            }

            # Add crop coordinate details if available
            for coord_col in [
                "crop_x_start",
                "crop_x_end",
                "crop_y_start",
                "crop_y_end",
            ]:
                if coord_col in qa_info:
                    detailed_record[coord_col] = qa_info[coord_col]

            detailed_records.append(detailed_record)

        # Create DataFrame
        detailed_df = pd.DataFrame(detailed_records)

        if detailed_df.empty:
            logging.error("No detailed records created")
            raise ValueError("No detailed records created")

        # Generate output filename if not provided
        if not output_path:
            csv_name = Path(csv_path).stem
            output_path = f"{csv_name}_detailed_cells.parquet"

        # Save to parquet
        detailed_df.to_parquet(output_path)

        # Print summary
        logging.info("=" * 60)
        logging.info("QA TO DETAILED PARQUET CONVERSION SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Results saved to: {output_path}")
        logging.info(f"Total cells converted: {len(detailed_df):,}")
        logging.info(f"Competitors: {detailed_df['competitor'].nunique()}")
        logging.info(f"Datasets: {detailed_df['dataset_name'].nunique()}")
        logging.info(
            f"Average Jaccard score: {detailed_df['jaccard_score'].mean():.4f}"
        )
        logging.info(
            f"Median Jaccard score: {detailed_df['jaccard_score'].median():.4f}"
        )

        # Per-competitor summary
        competitor_summary = (
            detailed_df.groupby("competitor")["jaccard_score"]
            .agg(["count", "mean", "median", "std"])
            .round(4)
        )
        logging.info("\nPer-Competitor Summary:")
        logging.info(competitor_summary.to_string())

        return output_path

    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        raise


def create_parquet_from_qa_results(
    qa_results_dir: str = ".",
    qa_dataframes_dir: str = ".",
    output_dir: str = "detailed_qa_results",
) -> list:
    """
    Batch convert all QA CSV results to parquet format.

    Args:
        qa_results_dir: Directory containing QA CSV result files
        qa_dataframes_dir: Directory containing QA dataframe parquet files
        output_dir: Directory to save converted parquet files

    Returns:
        List of created parquet files
    """
    import glob

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Find QA result CSV files
    csv_pattern = str(Path(qa_results_dir) / "qa_jaccard_results_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        logging.warning(f"No QA result CSV files found matching pattern: {csv_pattern}")
        return []

    logging.info(f"Found {len(csv_files)} QA result CSV files")

    created_files = []

    for csv_file in csv_files:
        try:
            csv_path = Path(csv_file)

            # Try to find corresponding QA dataframe
            # Pattern: qa_jaccard_results_BF-C2DL-MuSC.csv -> qa_BF-C2DL-MuSC_dataset.parquet
            dataset_part = csv_path.stem.replace("qa_jaccard_results_", "")
            qa_parquet_pattern = f"qa_{dataset_part}_dataset.parquet"
            qa_parquet_path = Path(qa_dataframes_dir) / qa_parquet_pattern

            if not qa_parquet_path.exists():
                # Try alternative patterns
                alternative_patterns = [
                    f"qa_{dataset_part}.parquet",
                    f"{dataset_part}_qa_dataset.parquet",
                ]

                for pattern in alternative_patterns:
                    alt_path = Path(qa_dataframes_dir) / pattern
                    if alt_path.exists():
                        qa_parquet_path = alt_path
                        break

                if not qa_parquet_path.exists():
                    logging.warning(f"No QA dataframe found for {csv_file}, skipping")
                    continue

            # Generate output filename
            output_filename = f"{dataset_part}_detailed_qa_cells.parquet"
            output_path = Path(output_dir) / output_filename

            # Convert
            logging.info(f"Converting: {csv_path.name} -> {output_filename}")
            result_path = convert_qa_csv_to_detailed_parquet(
                str(csv_path), str(qa_parquet_path), str(output_path)
            )

            if result_path:
                created_files.append(result_path)

        except Exception as e:
            logging.error(f"Error processing {csv_file}: {e}")
            continue

    logging.info(
        f"Successfully converted {len(created_files)} files to detailed parquet format"
    )
    return created_files


def excel2csv(filepath):
    """
    Receives an excel with train, validation and test sheets and converts to an ordered csv. 
    """
    # load excel
    df_dict = pd.read_excel(filepath, sheet_name=None)
    df_sheets = []
    for col_name in ["train", "validation", "test"]:
        df = df_dict[col_name]
        df[p_utils.SPLITS_COLUMN] = [col_name] * len(df)
        df_sheets.append(df)

    # Concatenate all sheets
    df = pd.concat(df_sheets)
    # sort by cell_id
    df.sort_values(by="cell_id", inplace=True)
    new_filepath = os.path.splitext(filepath)[0] + ".csv"
    # save csv
    df.to_csv(new_filepath, index=None, header=True)
    return new_filepath