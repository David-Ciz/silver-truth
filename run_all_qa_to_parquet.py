"""
Batch script for converting all QA evaluation results to detailed parquet format

This script automates the conversion of QA CSV results to detailed parquet files
that can be used with the analyze_detailed_results.py tool.
"""

import subprocess
import logging
from pathlib import Path
import glob
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    """Main execution function"""
    logging.info("🚀 Starting batch QA to parquet conversion")
    
    # Run the batch conversion
    try:
        cmd = [
            sys.executable,
            "convert_qa_to_parquet.py",
            "batch-convert"
        ]
        
        logging.info("Running batch conversion...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("✅ Batch conversion completed successfully")
            print(result.stdout)
        else:
            logging.error("❌ Batch conversion failed")
            print(result.stderr)
            return
        
        # Find created parquet files and run analysis
        detailed_qa_dir = Path("detailed_qa_results")
        if detailed_qa_dir.exists():
            parquet_files = list(detailed_qa_dir.glob("*_detailed_qa_cells.parquet"))
            
            if parquet_files:
                logging.info(f"📊 Running analysis on {len(parquet_files)} parquet files")
                
                for parquet_file in parquet_files:
                    try:
                        dataset_name = parquet_file.stem.replace('_detailed_qa_cells', '')
                        analysis_dir = f"qa_analysis_{dataset_name}"
                        
                        cmd = [
                            sys.executable,
                            "analyze_detailed_results.py",
                            str(parquet_file),
                            "--output-dir", analysis_dir,
                            "--save-summary"
                        ]
                        
                        logging.info(f"Analyzing: {parquet_file.name}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            logging.info(f"✅ Analysis completed for {dataset_name}")
                        else:
                            logging.warning(f"⚠️ Analysis failed for {dataset_name}: {result.stderr}")
                            
                    except Exception as e:
                        logging.error(f"❌ Error analyzing {parquet_file}: {e}")
            else:
                logging.warning("No detailed parquet files found for analysis")
        
        logging.info("🎉 Batch QA conversion and analysis completed!")
        logging.info("📁 Check these directories for results:")
        logging.info("   - detailed_qa_results/ (parquet files)")
        logging.info("   - qa_analysis_*/ (analysis reports)")
        
    except Exception as e:
        logging.error(f"❌ Error during batch processing: {e}")

if __name__ == "__main__":
    main()
