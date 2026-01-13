
import click
import logging
from silver_truth.cli import preprocessing, fusion, ensemble, evaluation, qa

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@click.group()
def cli():
    """
    Silver Truth Research Pipeline
    ==============================

    This script acts as the main orchestration tool for your research workflow.
    Instead of remembering 8 different python scripts, you can use this tool to 
    run steps in order or individually.

    Available Commands:
    - check:         Verify that the environment and imports are working.
    - step1-prepare: Synchronize datasets and prepare dataframes.
    - step2-fusion:  Run the fusion algorithm (requires Java).
    - step3-ensemble: Train the ensemble model.
    - run-all:       Run the full sequence (experimental).
    """
    pass

@cli.command()
def check():
    """Verify that the environment is set up and Silver Truth is importable."""
    logging.info("Checking environment...")
    try:
        import silver_truth
        logging.info(f"Silver Truth package found at: {silver_truth.__file__}")
        logging.info("✅ Environment looks good!")
    except ImportError as e:
        logging.error(f"❌ Failed to import silver_truth: {e}")
        exit(1)

@cli.command()
@click.option("--dataset-dir", required=True, type=click.Path(exists=True), help="Path to raw dataset folder (e.g. inputs-2020-07)")
@click.option("--output-dir", required=True, type=click.Path(), help="Where to save synchronized data")
def step1_prepare(dataset_dir, output_dir):
    """
    Step 1: Synchronize and prepare data.
    
    This wraps 'silver-preprocessing synchronize-datasets' logic.
    """
    logging.info("Step 1: Synchronizing datasets...")
    preprocessing.synchronize_datasets_logic(dataset_dir, output_dir, debug=False)
    logging.info("✅ Step 1 Complete.")

@cli.command()
@click.option("--job-file", required=True, type=click.Path(exists=True), help="Path to the generated job file")
def step2_fusion(job_file):
    """
    Step 2: Run fusion (Example Wrapper).
    
    This command helps you remember how to run the Java fusion tool.
    In the future, we can automate the Java call here.
    """
    logging.info("Step 2: Fusion is currently run via Java/CLI wrapper.")
    logging.info(f"To run this manually, execute:\n  silver-fusion run-fusion --job-file {job_file} ...")
    # You could add subprocess.run(...) here if you know the exact args.

@cli.command()
def step3_ensemble():
    """Step 3: Train Ensemble (Placeholder)."""
    logging.info("Step 3: Starting Ensemble Training...")
    # ensemble.run_experiment(...)
    pass

@cli.command()
def run_all():
    """
    Run the Full Pipeline.
    
    Use this to chain commands together for a 'one-click' reproduction of results.
    """
    logging.info("Starting Full Research Pipeline...")
    logging.warning("This is an example pipeline. Please configure it in scripts/run_pipeline.py")
    pass

if __name__ == "__main__":
    cli()
