#!/usr/bin/env python3
"""
Automatic script to create dataset dataframes for all synchronized datasets.

This script scans the synchronized_data directory and creates dataframe files
for each dataset using the preprocessing.py create-dataset-dataframe command.
"""

import subprocess
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
SYNCHRONIZED_DATA_DIR = Path(
    r"C:\Users\wei0068\Desktop\Cell_Tracking\synchronized_data"
)
OUTPUT_DIR = Path("dataframes")  # Dataframes directory


def run_command(command):
    """Run a command and return True if successful, False otherwise."""
    try:
        logging.info(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Success: {result.returncode}")
        if result.stdout:
            logging.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False


def create_dataframe_for_dataset(dataset_path):
    """Create a dataframe for a single dataset."""
    dataset_name = dataset_path.name

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / f"{dataset_name}_dataset_dataframe.parquet"

    logging.info(f"Processing dataset: {dataset_name}")
    logging.info(f"Output file: {output_file}")

    command = [
        sys.executable,
        "preprocessing.py",
        "create-dataset-dataframe",
        str(dataset_path),
        "--output_path",
        str(output_file),
    ]

    return run_command(command)


def main():
    """Main function to process all datasets."""
    if not SYNCHRONIZED_DATA_DIR.exists():
        logging.error(f"Synchronized data directory not found: {SYNCHRONIZED_DATA_DIR}")
        sys.exit(1)

    # Get all dataset directories
    dataset_dirs = [d for d in SYNCHRONIZED_DATA_DIR.iterdir() if d.is_dir()]

    if not dataset_dirs:
        logging.warning(f"No dataset directories found in {SYNCHRONIZED_DATA_DIR}")
        return

    logging.info(f"Found {len(dataset_dirs)} datasets to process")

    successful = 0
    failed = 0
    failed_datasets = []

    for dataset_dir in sorted(dataset_dirs):
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing dataset: {dataset_dir.name}")
        logging.info(f"{'='*50}")

        if create_dataframe_for_dataset(dataset_dir):
            successful += 1
            logging.info(f"Successfully processed: {dataset_dir.name}")
        else:
            failed += 1
            failed_datasets.append(dataset_dir.name)
            logging.error(f"Failed to process: {dataset_dir.name}")

    # Summary
    logging.info(f"\n{'='*50}")
    logging.info("PROCESSING SUMMARY")
    logging.info(f"{'='*50}")
    logging.info(f"Total datasets: {len(dataset_dirs)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")

    if failed_datasets:
        logging.error(f"Failed datasets: {', '.join(failed_datasets)}")

    if failed > 0:
        sys.exit(1)
    else:
        logging.info("All datasets processed successfully!")


if __name__ == "__main__":
    main()
