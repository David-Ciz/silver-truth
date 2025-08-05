"""
Automatic script to generate job files for all dataset dataframes.

This script finds all dataset dataframe files (.parquet) and generates
job files for each dataset using the run_fusion.py generate-jobfiles command.
"""

import subprocess
import sys
from pathlib import Path
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
DATAFRAMES_PATTERNS = [
    "*_dataset_dataframe.parquet",  # Legacy: root directory
    "dataframes/*_dataset_dataframe.parquet",  # New: dataframes folder
]
OUTPUT_DIR = "job_files"
CAMPAIGNS = ["01", "02"]  # Generate job files for both campaigns


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


def extract_dataset_name(parquet_file):
    """Extract dataset name from parquet filename."""
    filename = Path(parquet_file).stem
    # Remove the "_dataset_dataframe" suffix
    return filename.replace("_dataset_dataframe", "")


def generate_jobfile_for_dataframe(parquet_file, campaign_number):
    """Generate a job file for a single dataframe."""
    dataset_name = extract_dataset_name(parquet_file)
    logging.info(f"Processing dataframe: {parquet_file}")
    logging.info(f"Dataset: {dataset_name}, Campaign: {campaign_number}")

    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    command = [
        sys.executable,
        "run_fusion.py",
        "generate-jobfiles",
        "--parquet-file",
        str(parquet_file),
        "--campaign-number",
        campaign_number,
        "--output-dir",
        OUTPUT_DIR,
    ]

    return run_command(command)


def main():
    """Main function to process all dataframes."""
    # Find all dataframe files (check both root and dataframes folder)
    parquet_files = []
    for pattern in DATAFRAMES_PATTERNS:
        parquet_files.extend(glob.glob(pattern))

    if not parquet_files:
        logging.warning(
            f"No dataframe files found matching patterns: {DATAFRAMES_PATTERNS}"
        )
        logging.info("Make sure you have run 'python create_all_dataframes.py' first.")
        return

    logging.info(f"Found {len(parquet_files)} dataframe files to process")
    logging.info(f"Generating job files for campaigns: {', '.join(CAMPAIGNS)}")

    successful = 0
    failed = 0
    failed_files = []

    # Process each dataframe for each campaign
    for campaign in CAMPAIGNS:
        logging.info(f"\n{'='*60}")
        logging.info(f"PROCESSING CAMPAIGN {campaign}")
        logging.info(f"{'='*60}")

        for parquet_file in sorted(parquet_files):
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing: {parquet_file} (Campaign {campaign})")
            logging.info(f"{'='*50}")

            if generate_jobfile_for_dataframe(parquet_file, campaign):
                successful += 1
                logging.info(
                    f"Successfully processed: {parquet_file} (Campaign {campaign})"
                )
            else:
                failed += 1
                failed_files.append(f"{parquet_file} (Campaign {campaign})")
                logging.error(
                    f"Failed to process: {parquet_file} (Campaign {campaign})"
                )

    # Summary
    logging.info(f"\n{'='*50}")
    logging.info("PROCESSING SUMMARY")
    logging.info(f"{'='*50}")
    logging.info(f"Total dataframes: {len(parquet_files)}")
    logging.info(f"Campaigns processed: {', '.join(CAMPAIGNS)}")
    logging.info(f"Total combinations processed: {len(parquet_files) * len(CAMPAIGNS)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Job files saved to: {OUTPUT_DIR}")

    if failed_files:
        logging.error(f"Failed files: {', '.join(failed_files)}")

    if failed > 0:
        sys.exit(1)
    else:
        logging.info("All job files generated successfully!")


if __name__ == "__main__":
    main()
