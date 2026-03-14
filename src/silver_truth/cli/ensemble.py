import click
import logging
from silver_truth.ensemble.datasets import Version
from silver_truth.ensemble.databanks_builds import Databank_type
from silver_truth.ensemble.models import ModelType
import silver_truth.ensemble.ensemble as ensemble

# from pathlib import Path


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

@click.command()
@click.option("--qa_parquet_path", required=True, help="Path to the QA's parquet.")
@click.option("--name", required=True, help="Name of the original dataset.")
@click.option("--databank", type=int, default=2, required=False, help="Databank type: 1 (Single) or 2 (Normalized). Default: 2")
@click.option("--crop_size", type=int, default=64, required=False,  help="Size of each cropped image. Default: 64.")
@click.option("--split_seed", type=int, default=42, required=False, help="Seed used in split operation. Default: 42.")
@click.option("--split_sets", type=str, default="0.7 0.15 0.15", required=False, help="Split proportions. Default: \"0.7 0.15 0.15\"")
@click.option("--qa", type=str, default=None, required=False, help="QA reference. Default: None.")
@click.option("--qa_threshold", type=float, default=None, required=False, help="QA threshold. Default: None.")
def build_databank(qa_parquet_path, name, databank, crop_size, split_seed, split_sets, qa, qa_threshold):
    build_opt = {
        "name": name,
        "databank": Databank_type(databank),
        "crop_size": crop_size,
        "split_seed": split_seed,
        "split_sets": list(map(float, split_sets.split())),
        "qa": qa,
        "qa_threshold": qa_threshold,
    }
    ensemble.build_databank(build_opt, qa_parquet_path)


@click.command()
@click.option("--exp_name", required=True, help="The name of the Ensemble experiment.")
@click.option("--name", required=True, help="Name of the original dataset.")
@click.option("--dataset", type=int, required=True, help="Dataset Version: 1 (A1), 2 (B1), 3 (B2), 4 (B3), 5 (C1), 6 (C2).")
@click.option("--databank", type=int, default=2, required=False, help="Databank type: 1 (Single) or 2 (Normalized). Default: 2")
@click.option("--split_seed", type=int, default=42, required=False, help="Seed used in split operation. Default: 42.")
@click.option("--split_sets", type=str, default="0.7 0.15 0.15", required=False, help="Split proportions. Default: \"0.7 0.15 0.15\"")
@click.option("--qa", type=str, default=None, required=False, help="QA reference. Default: None.")
@click.option("--qa_threshold", type=float, default=None, required=False, help="QA threshold. Default: None.")
@click.option("--model", type=int, default=1, required=False, help="Model type: 1 (Unet), 2 (Unet++), .... Default: 1")
@click.option("--max_epochs", type=int, default=100, required=False, help="Maximum number of training epochs. Default: 100")
def ensemble_experiment(exp_name, name, dataset, databank, split_seed, split_sets, qa, qa_threshold, model, max_epochs):
    """Runs an Ensemble experiment via command-line interface."""
    build_opt = {
        "name": name,
        "databank": Databank_type(databank),
        "split_seed": split_seed,
        "split_sets": list(map(float, split_sets.split())),
        "qa": qa,
        "qa_threshold": qa_threshold,
        "dataset": Version(dataset),
    }
    run_sequence = [{"model_type": ModelType(model), "max_epochs": max_epochs, "databank_opt": build_opt}]

    try:
        ensemble.run_experiment(exp_name, run_sequence)
    except Exception as e:
        click.echo(
            click.style(f"An unexpected error occurred: {e}", fg="red", bold=True)
        )
        exit(1)


@click.group()
def cli():
    """Main entry point for command-line tools."""
    pass

cli.add_command(build_databank)
cli.add_command(ensemble_experiment)

if __name__ == "__main__":
    cli()
