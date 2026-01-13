import click
import logging
import silver_truth.ensemble.ensemble as ensemble
from silver_truth.ensemble.models import ModelType

# from pathlib import Path


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@click.command()
@click.option("--name", required=True, help="The name of the Ensemble experiment.")
# @click.option(
#    "--model", required=True, help="The A.I. model that replaces fusion."
# )
@click.option(
    "--parquet_file",
    required=True,
    help="The path of the Ensemble databank parquet file.",
)
def ensemble_experiment(
    name: str,
    parquet_file: str,
):
    """Runs an Ensemble experiment via command-line interface."""
    try:
        ensemble.run_experiment(
            name,
            parquet_file,
            [{"model_type": ModelType.UnetPlusPlus, "max_epochs": 100}],
        )
    except Exception as e:
        click.echo(
            click.style(f"An unexpected error occurred: {e}", fg="red", bold=True)
        )
        exit(1)


@click.group()
def cli():
    """Main entry point for command-line tools."""
    pass


cli.add_command(ensemble_experiment)

if __name__ == "__main__":
    cli()
