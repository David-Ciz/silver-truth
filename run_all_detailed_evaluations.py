"""
Batch script for running detailed evaluation on all datasets

This script automates the process of running detailed cell-level evaluation
for all available datasets in the project.
"""

import subprocess
import logging
from pathlib import Path
import glob
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def find_dataset_parquet_files(base_dir: str = ".") -> list:
    """
    Find all dataset dataframe parquet files
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of paths to dataset parquet files
    """
    patterns = [
        "*_dataset_dataframe.parquet",
        "dataframes/*_dataset_dataframe.parquet", 
        "dataframes/*.parquet"
    ]
    
    parquet_files = []
    for pattern in patterns:
        found_files = glob.glob(str(Path(base_dir) / pattern))
        parquet_files.extend(found_files)
    
    # Remove duplicates and filter for dataset files
    unique_files = list(set(parquet_files))
    dataset_files = [f for f in unique_files if 'dataset_dataframe' in f]
    
    return sorted(dataset_files)

def run_detailed_evaluation(parquet_file: str, output_dir: str = "detailed_results"):
    """
    Run detailed evaluation for a single parquet file
    
    Args:
        parquet_file: Path to the dataset parquet file
        output_dir: Directory to save results
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Extract dataset name for output file
        dataset_name = Path(parquet_file).stem.replace('_dataset_dataframe', '')
        output_file = Path(output_dir) / f"{dataset_name}_detailed_evaluation.parquet"
        
        # Run detailed evaluation
        cmd = [
            sys.executable, 
            "detailed_evaluation.py",
            parquet_file,
            "--output", str(output_file)
        ]
        
        logging.info(f"Running detailed evaluation for: {dataset_name}")
        logging.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"✅ Successfully completed evaluation for {dataset_name}")
            return str(output_file)
        else:
            logging.error(f"❌ Failed evaluation for {dataset_name}")
            logging.error(f"Error: {result.stderr}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Exception during evaluation of {parquet_file}: {e}")
        return None

def run_detailed_analysis(detailed_results_files: list, base_output_dir: str = "detailed_analysis"):
    """
    Run analysis on all detailed results files
    
    Args:
        detailed_results_files: List of detailed evaluation parquet files
        base_output_dir: Base directory for analysis outputs
    """
    for results_file in detailed_results_files:
        if not Path(results_file).exists():
            logging.warning(f"Results file not found: {results_file}")
            continue
            
        try:
            # Create dataset-specific analysis directory
            dataset_name = Path(results_file).stem.replace('_detailed_evaluation', '')
            analysis_dir = Path(base_output_dir) / dataset_name
            
            cmd = [
                sys.executable,
                "analyze_detailed_results.py", 
                results_file,
                "--output-dir", str(analysis_dir),
                "--save-summary"
            ]
            
            logging.info(f"Running analysis for: {dataset_name}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info(f"✅ Successfully completed analysis for {dataset_name}")
            else:
                logging.error(f"❌ Failed analysis for {dataset_name}")
                logging.error(f"Error: {result.stderr}")
                
        except Exception as e:
            logging.error(f"❌ Exception during analysis of {results_file}: {e}")

def main():
    """Main execution function"""
    logging.info("🚀 Starting batch detailed evaluation process")
    
    # Find all dataset parquet files
    parquet_files = find_dataset_parquet_files()
    
    if not parquet_files:
        logging.error("❌ No dataset parquet files found!")
        logging.info("Expected files matching patterns:")
        logging.info("  - *_dataset_dataframe.parquet")
        logging.info("  - dataframes/*_dataset_dataframe.parquet")
        return
    
    logging.info(f"📂 Found {len(parquet_files)} dataset files:")
    for f in parquet_files:
        logging.info(f"   - {f}")
    
    # Run detailed evaluation for each dataset
    output_dir = "detailed_results"
    detailed_results_files = []
    
    for parquet_file in parquet_files:
        result_file = run_detailed_evaluation(parquet_file, output_dir)
        if result_file:
            detailed_results_files.append(result_file)
    
    if not detailed_results_files:
        logging.error("❌ No successful evaluations completed!")
        return
    
    logging.info(f"✅ Completed {len(detailed_results_files)} detailed evaluations")
    
    # Run analysis on all results
    logging.info("📊 Starting detailed analysis phase")
    run_detailed_analysis(detailed_results_files, "detailed_analysis")
    
    logging.info("🎉 Batch detailed evaluation and analysis completed!")
    logging.info(f"📁 Results saved in:")
    logging.info(f"   - Detailed evaluations: {output_dir}/")
    logging.info(f"   - Analysis reports: detailed_analysis/")

if __name__ == "__main__":
    main()
