#!/usr/bin/env python3
"""
Automatic script to run fusion for all generated job files.

This script finds all job files and runs the fusion process for each one
using the run_fusion.py run-fusion command with predefined settings.
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
JOB_FILES_DIR = "job_files"
JOB_FILES_PATTERN = "*_job_file.txt"
OUTPUT_DIR = "fused_results"
JAR_PATH = "src/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar"

# Default fusion settings
DEFAULT_TIME_POINTS = "0-61"
DEFAULT_NUM_THREADS = 2
DEFAULT_MODEL = "threshold_flat"


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


def extract_job_info(job_file_path):
    """Extract dataset name and campaign from job file name."""
    filename = Path(job_file_path).stem
    # Expected format: DATASET_CAMPAIGN_job_file
    # Example: BF-C2DL-MuSC_01_job_file.txt
    parts = filename.replace("_job_file", "").split("_")
    if len(parts) >= 2:
        campaign = parts[-1]
        dataset = "_".join(parts[:-1])
        return dataset, campaign
    else:
        # Fallback: use full filename without extension
        return filename, "01"


def run_fusion_for_job(job_file_path):
    """Run fusion for a single job file."""
    job_file = Path(job_file_path)
    dataset_name, campaign = extract_job_info(job_file)

    logging.info(f"Processing job file: {job_file.name}")
    logging.info(f"Dataset: {dataset_name}, Campaign: {campaign}")

    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Generate output pattern
    output_pattern = f"{OUTPUT_DIR}/{dataset_name}_{campaign}_fused_TTTT.tif"

    # Check if JAR file exists
    if not Path(JAR_PATH).exists():
        logging.error(f"JAR file not found: {JAR_PATH}")
        return False

    command = [
        sys.executable,
        "run_fusion.py",
        "run-fusion",
        "--jar-path",
        JAR_PATH,
        "--job-file",
        str(job_file),
        "--output-pattern",
        output_pattern,
        "--time-points",
        DEFAULT_TIME_POINTS,
        "--num-threads",
        str(DEFAULT_NUM_THREADS),
        "--model",
        DEFAULT_MODEL,
    ]

    return run_command(command)


def main():
    """Main function to process all job files."""
    # Find all job files
    job_files_pattern = f"{JOB_FILES_DIR}/{JOB_FILES_PATTERN}"
    job_files = glob.glob(job_files_pattern)

    if not job_files:
        logging.warning(f"No job files found matching pattern: {job_files_pattern}")
        logging.info("Make sure you have run 'python generate_all_jobfiles.py' first.")
        return

    logging.info(f"Found {len(job_files)} job files to process")

    # Check if JAR file exists before starting
    if not Path(JAR_PATH).exists():
        logging.error(f"JAR file not found: {JAR_PATH}")
        logging.error(
            "Please make sure the Java JAR file is available at the specified path."
        )
        sys.exit(1)

    successful = 0
    failed = 0
    failed_jobs = []

    for job_file in sorted(job_files):
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing: {job_file}")
        logging.info(f"{'='*50}")

        if run_fusion_for_job(job_file):
            successful += 1
            logging.info(f"Successfully processed: {job_file}")
        else:
            failed += 1
            failed_jobs.append(job_file)
            logging.error(f"Failed to process: {job_file}")

    # Summary
    logging.info(f"\n{'='*50}")
    logging.info("PROCESSING SUMMARY")
    logging.info(f"{'='*50}")
    logging.info(f"Total job files: {len(job_files)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Results saved to: {OUTPUT_DIR}")
    logging.info(f"Using model: {DEFAULT_MODEL}")
    logging.info(f"Time points: {DEFAULT_TIME_POINTS}")
    logging.info(f"Threads: {DEFAULT_NUM_THREADS}")

    if failed_jobs:
        logging.error(f"Failed jobs: {', '.join([Path(j).name for j in failed_jobs])}")

    if failed > 0:
        sys.exit(1)
    else:
        logging.info("All fusion processes completed successfully!")


if __name__ == "__main__":
    main()
