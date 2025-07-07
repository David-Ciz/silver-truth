import logging
import shutil
from pathlib import Path

import click
import pandas as pd
import tqdm
from src.data_processing.label_synchronizer import (
    synchronize_labels_with_tracking_markers,
    process_segmentation_folders,
    verify_folder_synchronization_logic,
    process_directory,
)

from src.data_processing.utils.dataset_dataframe_creation import (
    convert_to_dataframe,
    is_valid_competitor_folder,
    process_dataset_directory,
    save_dataframe_to_parquet_with_metadata,
)

# Constants
RAW_DATA_FOLDERS = {"01", "02"}
GT_FOLDER_FIRST = "01_GT"
GT_FOLDER_SECOND = "02_GT"
SEG_FOLDER = "SEG"
TRA_FOLDER = "TRA"
RES_FOLDER_FIRST = "01_RES"
RES_FOLDER_SECOND = "02_RES"

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
    dataset_dir = Path(dataset_dir)
    tracking_dir_01 = Path(tracking_dir_01)
    tracking_dir_02 = Path(tracking_dir_02)

    dataset_data = {
        "01": {"segmentations": [], "tracking_folder": tracking_dir_01},
        "02": {"segmentations": [], "tracking_folder": tracking_dir_02},
    }
    for dataset_subfolder in dataset_dir.iterdir():
        if not dataset_subfolder.is_dir():
            continue
        # Skip raw data folders just in case
        if dataset_subfolder.name in RAW_DATA_FOLDERS:
            continue
        # Process GT folders
        elif dataset_subfolder.name == GT_FOLDER_FIRST:
            seg_folder = dataset_subfolder / SEG_FOLDER
            if seg_folder.exists():
                dataset_data["01"]["segmentations"].append(seg_folder)
        elif dataset_subfolder.name == GT_FOLDER_SECOND:
            seg_folder = dataset_subfolder / SEG_FOLDER
            if seg_folder.exists():
                dataset_data["02"]["segmentations"].append(seg_folder)

        # Process competitor folders
        elif is_valid_competitor_folder(dataset_subfolder):
            seg_folder_01 = dataset_subfolder / RES_FOLDER_FIRST
            seg_folder_02 = dataset_subfolder / RES_FOLDER_SECOND
            if seg_folder_01.exists():
                dataset_data["01"]["segmentations"].append(seg_folder_01)
            if seg_folder_02.exists():
                dataset_data["02"]["segmentations"].append(seg_folder_02)

    # Verify the synchronization for each dataset type
    desynchronized_subfolders = set()
    for dataset_type, data in tqdm.tqdm(dataset_data.items()):
        if data["segmentations"] is None:
            logging.error(
                f"⚠️ Segmentation folder not found for dataset, type {dataset_type}"
            )
        else:
            click.echo(
                f"Verifying synchronization for subfolder {data['segmentations']}"
            )
            # Choose the corresponding tracking folder based on dataset type
            tracking_folder = data["tracking_folder"]
            desynchronized_images = verify_folder_synchronization_logic(
                str(data["segmentations"]), str(tracking_folder)
            )
            if desynchronized_images:
                desynchronized_subfolders.add(dataset_type)
                click.echo("The following images are not synchronized:")
                click.echo(", ".join(desynchronized_images))
            else:
                click.echo(f"All images for {data['segmentations']} are synchronized.")
    if desynchronized_subfolders:
        click.echo(f"Desynchronized subfolders: {', '.join(desynchronized_subfolders)}")
    else:
        click.echo("All images are synchronized.")


@click.command()
@click.argument("label_folder", type=click.Path(exists=True))
@click.argument("tracking_folder", type=click.Path(exists=True))
def verify_folder_synchronization(label_folder, tracking_folder):
    desynchronized_images = verify_folder_synchronization_logic(
        label_folder, tracking_folder
    )

    if desynchronized_images:
        click.echo("The following images are not synchronized:")
        click.echo(", ".join(desynchronized_images))
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

    datasets_folder = Path(datasets_folder)
    output_directory = Path(output_directory)

    # Configure logging level based on debug flag
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("Debug mode enabled - showing all Fiji output")
    failed_datasets = []
    processed_datasets = 0
    skipped_datasets = 0
    dataset_dirs = [d for d in datasets_folder.iterdir() if d.is_dir()]
    for dataset in tqdm.tqdm(dataset_dirs, desc="Processing datasets"):
        if not dataset.is_dir():
            continue
        logging.info(f"Processing dataset: {dataset.name}")
        dataset_data = {
            "01": {"segmentations": [], "tracking_folder": None},
            "02": {"segmentations": [], "tracking_folder": None},
        }
        dataset_output_directory = output_directory / dataset.name
        for dataset_subfolder in dataset.iterdir():
            if not dataset_subfolder.is_dir():
                logging.info(
                    f"Skipping {dataset_subfolder.name} as it is not a dataset folder"
                )
                continue
            # Copy raw data folders
            elif dataset_subfolder.name in RAW_DATA_FOLDERS:
                target_folder = dataset_output_directory / dataset_subfolder.name
                if not target_folder.exists():
                    logging.info(f"Copying raw data folder: {dataset_subfolder.name}")
                    shutil.copytree(dataset_subfolder, target_folder)
            # Process first GT folder
            elif dataset_subfolder.name == GT_FOLDER_FIRST:
                dataset_data["01"]["segmentations"].append(
                    dataset_subfolder / SEG_FOLDER
                )
                dataset_data["01"]["tracking_folder"] = dataset_subfolder / TRA_FOLDER
            elif dataset_subfolder.name == GT_FOLDER_SECOND:
                dataset_data["02"]["segmentations"].append(
                    dataset_subfolder / SEG_FOLDER
                )
                dataset_data["02"]["tracking_folder"] = dataset_subfolder / TRA_FOLDER
            elif is_valid_competitor_folder(dataset_subfolder):
                dataset_data["01"]["segmentations"].append(
                    dataset_subfolder / RES_FOLDER_FIRST
                )
                dataset_data["02"]["segmentations"].append(
                    dataset_subfolder / RES_FOLDER_SECOND
                )

        # Check if tracking folders were found
        missing_tracking = False
        for dataset_type, data in dataset_data.items():
            if data["tracking_folder"] is None:
                missing_tracking = True
                logging.error(
                    f"⚠️ Tracking folder not found for dataset {dataset.name}, type {dataset_type}"
                )

        if missing_tracking:
            logging.error(
                f"⛔ Skipping dataset {dataset.name} due to missing tracking data"
            )
            failed_datasets.append(dataset.name)
            skipped_datasets += 1
            continue
            # Process each type of dataset
        for dataset_type, data in dataset_data.items():
            process_segmentation_folders(
                data["segmentations"],
                data["tracking_folder"],
                dataset_output_directory,
                dataset_type,
                debug,
            )
        processed_datasets += 1

    # Summary at the end of processing
    logging.info("\n--- Processing Summary ---")
    logging.info(f"✅ Successfully processed datasets: {processed_datasets}")

    if skipped_datasets > 0:
        logging.info(
            f"⚠️ Skipped datasets due to missing tracking data: {skipped_datasets}"
        )
        logging.info(f"Failed datasets: {', '.join(failed_datasets)}")


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
    synchronized_dataset_dir = Path(synchronized_dataset_dir)
    if output_path is None:
        output_path = f"{synchronized_dataset_dir.name}_dataset_dataframe.parquet"

    dataset_info, competitor_columns = process_dataset_directory(
        synchronized_dataset_dir
    )
    dataset_dataframe = convert_to_dataframe(dataset_info)
    # Store metadata
    dataset_dataframe.attrs["base_directory"] = str(synchronized_dataset_dir)
    dataset_dataframe.attrs["competitor_columns"] = list(competitor_columns)
    dataset_dataframe.attrs["created_by"] = (
        "David-Ciz"  # Current user from your message
    )
    dataset_dataframe.attrs["creation_time"] = pd.Timestamp.now()
    save_dataframe_to_parquet_with_metadata(dataset_dataframe, str(output_path))


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
    click.echo(f"Verbose: {'Yes' if verbose else 'No'}")
    click.echo(f"{'=' * 30}")

    process_directory(
        directory, recursive=not non_recursive, dryrun=dry_run, verbose=verbose
    )

    if dry_run:
        click.echo(
            click.style(
                "\nThis was a dry run. No files were modified. Run without --dry-run to apply changes.",
                fg="yellow",
            )
        )


@click.group()
def cli():
    pass


cli.add_command(synchronize_labels)
cli.add_command(verify_folder_synchronization)
cli.add_command(synchronize_datasets)
cli.add_command(verify_dataset_synchronization)
cli.add_command(create_dataset_dataframe)
cli.add_command(compress_tifs)
if __name__ == "__main__":
    cli()

# @click.command()
# @click.argument('label_img_path', type=click.Path(exists=True))
# @click.argument('tracking_img_path', type=click.Path(exists=True))
# def main(label_img_path, tracking_img_path):
#     """
#     This script validates the synchronization between two images.
#
#     LABEL_IMG_PATH: Path to the label image.
#     TRACKING_IMG_PATH: Path to the tracking image.
#     """
#     label_img = tifffile.imread(label_img_path)
#     tracking_img = tifffile.imread(tracking_img_path)
#
#     is_synchronized = verify_synchronization(label_img, tracking_img)
#
#     if is_synchronized:
#         click.echo("The images are synchronized.")
#     else:
#         click.echo("The images are not synchronized.")
#
# if __name__ == '__main__':
#     main()
