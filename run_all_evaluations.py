#!/usr/bin/env python3
"""
Automatic script to run evaluations for all dataset dataframes.

This script finds all dataset dataframe files (.parquet) and runs
evaluation for each one using the evaluation.py evaluate-competitor command.
"""

import subprocess
import sys
from pathlib import Path
import logging
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
DATAFRAMES_PATTERN = "dataframes/*_dataset_dataframe.parquet"
RESULTS_DIR = "evaluation_results"

def run_command(command):
    """Run a command and return True if successful, False otherwise."""
    try:
        logging.info(f"Running: {' '.join(command)}")
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True
        )
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

def run_evaluation_for_dataframe(parquet_file):
    """Run evaluation for a single dataframe."""
    dataset_name = extract_dataset_name(parquet_file)
    logging.info(f"Processing dataframe: {parquet_file}")
    logging.info(f"Dataset: {dataset_name}")
    
    # Create results directory if it doesn't exist
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Generate output filename
    output_file = Path(RESULTS_DIR) / f"evaluation_results_{dataset_name}.csv"
    
    command = [
        sys.executable,
        "evaluation.py", 
        "evaluate-competitor",
        str(parquet_file),
        "--output", str(output_file)
    ]
    
    return run_command(command)

def main():
    """Main function to process all dataframes."""
    # Find all dataframe files
    parquet_files = glob.glob(DATAFRAMES_PATTERN)
    
    if not parquet_files:
        logging.warning(f"No dataframe files found matching pattern: {DATAFRAMES_PATTERN}")
        logging.info("Make sure you have run 'python create_all_dataframes.py' first.")
        return
    
    logging.info(f"Found {len(parquet_files)} dataframe files to evaluate")
    
    successful = 0
    failed = 0
    no_gt = 0
    failed_files = []
    no_gt_files = []
    
    for parquet_file in sorted(parquet_files):
        logging.info(f"\n{'='*50}")
        logging.info(f"Evaluating: {parquet_file}")
        logging.info(f"{'='*50}")
        
        result = run_evaluation_for_dataframe(parquet_file)
        
        if result:
            successful += 1
            logging.info(f"✅ Successfully evaluated: {parquet_file}")
        else:
            # Check if it's a "no GT" issue vs other failure
            dataset_name = extract_dataset_name(parquet_file)
            try:
                import pandas as pd
                df = pd.read_parquet(parquet_file)
                if "gt_image" not in df.columns:
                    no_gt += 1
                    no_gt_files.append(parquet_file)
                    logging.warning(f"⚠️  No GT available for: {parquet_file} (tracking-only dataset)")
                else:
                    failed += 1
                    failed_files.append(parquet_file)
                    logging.error(f"❌ Failed to evaluate: {parquet_file}")
            except Exception:
                failed += 1
                failed_files.append(parquet_file)
                logging.error(f"❌ Failed to evaluate: {parquet_file}")
    
    # Summary
    logging.info(f"\n{'='*50}")
    logging.info("EVALUATION SUMMARY")
    logging.info(f"{'='*50}")
    logging.info(f"Total dataframes: {len(parquet_files)}")
    logging.info(f"✅ Successful evaluations: {successful}")
    logging.info(f"⚠️  No GT available (tracking-only): {no_gt}")
    logging.info(f"❌ Failed evaluations: {failed}")
    logging.info(f"Results saved to: {RESULTS_DIR}")
    
    if no_gt_files:
        logging.info(f"\nDatasets without segmentation GT (normal for tracking-only datasets):")
        for file in no_gt_files:
            logging.info(f"  - {extract_dataset_name(file)}")
    
    if failed_files:
        logging.error(f"\nFailed evaluations:")
        for file in failed_files:
            logging.error(f"  - {extract_dataset_name(file)}")
    
    if failed > 0:
        sys.exit(1)
    else:
        if successful > 0:
            logging.info("🎉 All available evaluations completed successfully!")
        else:
            logging.warning("⚠️  No evaluations could be performed (no datasets with segmentation GT)")
            sys.exit(1)

if __name__ == "__main__":
    main()
