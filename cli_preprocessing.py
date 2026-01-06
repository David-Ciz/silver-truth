import logging
from pathlib import Path

import click

from src.data_processing.label_synchronizer import (
    synchronize_labels_with_tracking_markers,
    verify_folder_synchronization_logic,
    verify_dataset_synchronization_logic,
    synchronize_datasets_logic,
)
from src.data_processing.compression import compress_tifs_logic
from src.data_processing.utils.dataset_dataframe_creation import (
    create_dataset_dataframe_logic,
)
from src.data_processing.utils.parquet_utils import add_split_type

# Configure logging globally
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@click.command()
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.argument("tracking_dir_01", type=click.Path(exists=True))
@click.argument("tracking_dir_02", type=click.Path(exists=True))
def verify_dataset_synchronization(
    dataset_dir: Path | str, tracking_dir_01: Path | str, tracking_dir_02: Path | str
):
    """
    This script verifies the synchronization between segmentations and tracking markers in a dataset.

    DATASET_DIR: Path to the dataset directory.
    TRACKING_DIR_01: Path to the 01 tracking directory.
    TRACKING_DIR_02: Path to the 02 tracking directory.
    """
    desynchronized_subfolders = verify_dataset_synchronization_logic(
        dataset_dir, tracking_dir_01, tracking_dir_02
    )
    if desynchronized_subfolders:
        click.echo(f"Desynchronized subfolders: {', '.join(desynchronized_subfolders)}")
    else:
        click.echo("All images are synchronized.")


@click.command()
@click.argument("label_folder", type=click.Path(exists=True))
@click.argument("tracking_folder", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug mode for verbose logging.")
def verify_folder_synchronization(label_folder, tracking_folder, debug):
    """
    This script verifies the synchronization between labels and tracking markers in a folder.

    LABEL_FOLDER: Path to the folder containing label images.
    TRACKING_FOLDER: Path to the folder containing tracking marker images.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    desynchronized_images = verify_folder_synchronization_logic(
        label_folder, tracking_folder
    )

    if desynchronized_images:
        click.echo("The following images are not synchronized:")
        click.echo(", ".join(map(str, desynchronized_images)))
    else:
        click.echo("All images are synchronized.")


@click.command()
@click.argument("input_segmentation_folder", type=click.Path(exists=True))
@click.argument("tra_markers_folder", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
def synchronize_labels(input_segmentation_folder, tra_markers_folder, output_directory):
    """
    This script synchronizes labels with tracking markers.

    INPUT_SEGMENTATION_FOLDER: Path to the folder containing segmentation results.
    TRA_MARKERS_FOLDER: Path to the folder containing TRA marker files.
    OUTPUT_DIRECTORY: Path to the folder where synchronized data will be saved.
    """
    if not Path(output_directory).is_dir():
        logging.info(f"Creating output directory: {output_directory}")
        Path(output_directory).mkdir(parents=True, exist_ok=True)
    synchronize_labels_with_tracking_markers(
        input_segmentation_folder, tra_markers_folder, output_directory
    )


@click.command()
@click.argument("datasets_folder", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
@click.option(
    "--debug", is_flag=True, help="Enable debug mode with verbose Fiji output"
)
def synchronize_datasets(datasets_folder, output_directory, debug):
    """Synchronizes all segmentations with tracking markers in all the datasets

    DATASETS_FOLDER: Path to the folder containing datasets.
    OUTPUT_DIRECTORY: Path to the folder where synchronized data will be saved.
    DEBUG: Enable debug mode with verbose output
    """
    synchronize_datasets_logic(datasets_folder, output_directory, debug)


@click.command()
@click.argument("synchronized_dataset_dir", type=click.Path(exists=True))
@click.option(
    "--output_path", type=click.Path(), help="Path to save the dataset dataframe"
)
def create_dataset_dataframe(
    synchronized_dataset_dir: Path | str, output_path: Path | str
) -> None:
    """
    Creates a pandas dataframe with dataset information from synchronized datasets.

    Args:
        SYNCHRONIZED_DATASET_DIR: Path to the synchronized dataset directory.
        OUTPUT_PATH: Path to save the parquet dataset dataframe.
    """
    create_dataset_dataframe_logic(synchronized_dataset_dir, output_path)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--non-recursive",
    "-n",
    is_flag=True,
    help="Process only the specified directory, not subdirectories",
)
@click.option(
    "--dry-run",
    "-d",
    is_flag=True,
    help="Show what would be done without modifying files",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Display detailed information for each file"
)
def compress_tifs(directory, non_recursive, dry_run, verbose):
    """Compress TIFF files using LZW compression in place.

    This tool finds all TIFF files in the specified DIRECTORY (and subdirectories)
    and compresses them using lossless LZW compression, overwriting the original files.

    Examples:

    \b
    # Compress all TIFFs in current directory and subdirectories
    python compress_tifs.py .

    \b
    # Only show what would be done without making changes
    python compress_tifs.py /path/to/images --dry-run

    \b
    # Process only the current directory, not subdirectories
    python compress_tifs.py . --non-recursive

    \b
    # Show detailed information for each file
    python compress_tifs.py /path/to/images --verbose
    """
    click.echo("TIFF Compression Tool")
    click.echo(f"{'=' * 30}")
    click.echo(f"Target directory: {directory}")
    click.echo(f"Mode: {'Non-recursive' if non_recursive else 'Recursive'}")
    click.echo(f"Dry run: {'Yes' if dry_run else 'No'}")
    click.echo(f"{'=' * 30}")

    compress_tifs_logic(directory, not non_recursive, dry_run, verbose)

    if dry_run:
        click.echo(
            click.style(
                "\nThis was a dry run. No files were modified. Run without --dry-run to apply changes.",
                fg="yellow",
            )
        )


@click.command()
@click.argument("parquet_path", type=click.Path(exists=True, dir_okay=False))
def add_split_column(parquet_path: str):
    """
    Adds a 'split' column to a given parquet file.

    The split is done with a 70-15-15 ratio for train-validation-test sets,
    using a fixed seed of 42 for reproducibility.

    PARQUET_PATH: Path to the input parquet file.
    """
    seed = 42
    splits = [0.7, 0.15, 0.15]
    logging.info(
        f"Adding split column to {parquet_path} with seed {seed} and splits {splits}"
    )
    output_path = add_split_type(parquet_path, seed, splits)
    click.echo(f"Successfully added split column. New file saved at: {output_path}")


@click.group()
def cli():
    pass


cli.add_command(synchronize_labels)
cli.add_command(verify_folder_synchronization)
cli.add_command(synchronize_datasets)
cli.add_command(verify_dataset_synchronization)
cli.add_command(create_dataset_dataframe)
cli.add_command(compress_tifs)
cli.add_command(add_split_column)


if __name__ == "__main__":
    cli()
