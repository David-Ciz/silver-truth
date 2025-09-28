import click
import logging
from src.ensemble.ensemble import run_ensemble_experiment

# from pathlib import Path


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@click.command()
@click.option(
    "--model", required=True, help="The A.I. model that replaces fusion."
)
def ensemble_experiment(model: str,):
    """Runs an Ensemble experiment via command-line interface."""
    try:
        run_ensemble_experiment(model, {})
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
