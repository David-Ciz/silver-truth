import click
import logging
import src.ensemble.ensemble as ensemble

# from pathlib import Path


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@click.command()
@click.option(
    "--name", required=True, help="The name of the Ensemble experiment."
)
@click.option(
    "--model", required=True, help="The A.I. model that replaces fusion."
)
def ensemble_experiment(name: str, model: str,):
    """Runs an Ensemble experiment via command-line interface."""
    try:
        ensemble.run_experiment(name, model, {})
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
