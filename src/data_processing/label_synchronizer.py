"""
This module contains the Label synchronization logic,
which is used to synchronize the labels and tracking data and in turn,
synchronize the labels between competitors, silver-truth, and the ground truth.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jaccard

import logging

import tqdm
from tifffile import tifffile

from PIL import Image
import click


def process_segmentation_folders(
    segmentation_folders, tracking_folder, dataset_output_directory, dataset_type, debug
):
    """Process segmentation folders and synchronize with tracking markers."""
    if not segmentation_folders:
        logging.warning(f"No segmentation folders found for {dataset_type}")
        return

    # Extract the last two parts of the path
    last_two_parts = tracking_folder.parts[-2:]
    # Join the last two parts back into a path
    tracking_subfolder = Path(*last_two_parts)
    output_dataset_tracking_folder = dataset_output_directory / tracking_subfolder
    if not output_dataset_tracking_folder.exists():
        shutil.copytree(tracking_folder, output_dataset_tracking_folder)

    for segmentation_folder in tqdm.tqdm(
        segmentation_folders, desc=f"Synchronizing {dataset_type} segmentations"
    ):
        # Extract the last two parts of the path
        last_two_parts = segmentation_folder.parts[-2:]
        # Join the last two parts back into a path
        competitor_subfolder = Path(*last_two_parts)
        output_dataset_folder = dataset_output_directory / competitor_subfolder
        if output_dataset_folder.is_dir():
            logging.info(
                f"Output directory already exists, skipping: {output_dataset_folder}"
            )
            continue
        if not output_dataset_folder.is_dir():
            logging.info(f"Creating output directory: {output_dataset_folder}")
            output_dataset_folder.mkdir(parents=True, exist_ok=True)
        synchronize_labels_with_tracking_markers(
            segmentation_folder, tracking_folder, output_dataset_folder, debug
        )


def synchronize_dataset():
    pass


def synchronize_labels_with_tracking_markers(
    input_segmentation_folder: str,
    tra_markers_folder: str,
    output_directory: str,
    debug: bool = False,
) -> None:
    """
    Synchronize segmentation labels with tracking markers using the standalone Java jar.
    The command invoked is:
        java -cp src/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar RunLabelSyncer2
             <input_segmentation_folder> <tra_markers_folder> <output_directory>

    Args:
        input_segmentation_folder: Path to the input segmentation folder (must be absolute).
        tra_markers_folder: Path to the tracking markers folder (must be absolute).
        output_directory: Path to the output directory (must be absolute).
        debug: If True, print debugging output; otherwise, suppress output.
    """
    # Ensure the provided folder paths are absolute
    input_segmentation_folder = os.path.abspath(input_segmentation_folder)
    tra_markers_folder = os.path.abspath(tra_markers_folder)
    output_directory = os.path.abspath(output_directory)

    # Get the directory of the current script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the jar file relative to the current script directory
    jar_path = os.path.join(
        current_script_dir,
        "cell_tracking_java_helpers",
        "label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar",
    )

    logging.info(f"Running LabelSyncer2 using jar at '{jar_path}'")
    logging.info(f"Input Segmentation Folder: {input_segmentation_folder}")
    logging.info(f"TRA Markers Folder: {tra_markers_folder}")
    logging.info(f"Output Directory: {output_directory}")

    # Prepare the command to run the Java application
    command = [
        "java",
        "-cp",
        jar_path,
        "de.mpicbg.ulman.fusion.RunLabelSyncer2",
        # If the class is in a package, use the fully-qualified name (e.g., de.mpicbg.ulman.fusion.RunLabelSyncer2)
        input_segmentation_folder,
        tra_markers_folder,
        output_directory,
    ]

    if debug:
        subprocess.run(command)
    else:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# Clean up
# os.remove(macro_file)


def get_numeric_part(filename):
    match = re.search(r"(\d+)\.tif$", filename)
    return match.group(1) if match else None


def verify_folder_synchronization_logic(label_folder, tracking_folder):
    label_files = {
        get_numeric_part(f): f for f in os.listdir(label_folder) if f.endswith(".tif")
    }
    tracking_files = {
        get_numeric_part(f): f
        for f in os.listdir(tracking_folder)
        if f.endswith(".tif")
    }

    common_keys = set(label_files.keys()).intersection(tracking_files.keys())
    desynchronized_images = []

    for key in tqdm.tqdm(common_keys):
        label_img_path = os.path.join(label_folder, label_files[key])
        tracking_img_path = os.path.join(tracking_folder, tracking_files[key])

        label_img = tifffile.imread(label_img_path)
        tracking_img = tifffile.imread(tracking_img_path)

        is_synchronized = verify_synchronization(label_img, tracking_img)

        if not is_synchronized:
            desynchronized_images.append(key)

    return desynchronized_images


def verify_synchronization(label_img, tracking_img):
    """
    This function verifies the synchronization of the labels between two images.
    """
    # sanity checks
    if label_img is None or tracking_img is None:
        logging.warning("One of the images is missing.")
        return False
    try:
        label_img.shape == tracking_img.shape
    except AttributeError:
        logging.error("The images are not of the same shape.")
        return False

    tracking_uniques = np.unique(tracking_img)
    label_uniques = np.unique(label_img)

    if len(tracking_uniques) == 1:
        logging.error("The tracking image is empty.")
        return False

    if len(label_uniques) == 1:
        logging.warning("The label image is empty.")

    # for label in label_uniques:
    #     if label not in tracking_uniques:
    #         logging.error(
    #             "The label image contains labels that are not contained in tracking image."
    #         )
    #         return False

    for label in label_uniques:
        if label in tracking_uniques and label in label_uniques:
            label_layer = (label_img == label).astype(int)
            tracking_layer = (tracking_img == label).astype(int)
            j_value = jaccard(label_layer.flatten(), tracking_layer.flatten())
            if j_value == 0:
                logging.error(f"Jaccard index for label {label} is {j_value}.")
                return False

    return True


def compress_tif_file(file_path, dryrun=False):
    """Compress a TIF file using LZW compression and overwrite the original file."""
    try:
        # Check if file is a TIFF
        if not file_path.lower().endswith((".tif", ".tiff")):
            return False, f"Skipped: {file_path} (not a TIFF file)"

        # Get file size before compression
        original_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

        if dryrun:
            return True, f"Would compress: {file_path} ({original_size:.2f}MB)"

        # Try to read with tifffile first
        try:
            img = tifffile.imread(file_path)

            # Create a temporary file
            temp_file = file_path + ".temp"

            # Save with LZW compression
            tifffile.imwrite(temp_file, img, compression="lzw")

            # Check if operation was successful
            if os.path.exists(temp_file):
                # Replace the original file
                os.replace(temp_file, file_path)
            else:
                return False, f"Failed: {file_path} (temporary file not created)"

        except Exception as e1:
            # Fall back to PIL if tifffile fails
            try:
                img = Image.open(file_path)

                # Create a temporary file
                temp_file = file_path + ".temp"

                # Save with LZW compression
                img.save(temp_file, compression="tiff_lzw")

                # Check if operation was successful
                if os.path.exists(temp_file):
                    # Replace the original file
                    os.replace(temp_file, file_path)
                else:
                    return False, f"Failed: {file_path} (temporary file not created)"

            except Exception as e2:
                return False, f"Failed: {file_path} (errors: {str(e1)} and {str(e2)})"

        # Get file size after compression
        new_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

        return (
            True,
            f"Compressed: {file_path} ({original_size:.2f}MB → {new_size:.2f}MB, saved {original_size - new_size:.2f}MB)",
        )

    except Exception as e:
        return False, f"Error: {file_path} ({str(e)})"


def process_directory(directory, recursive=True, dryrun=False, verbose=False):
    """Process all TIF files in the given directory and its subdirectories if recursive is True."""
    success_count = 0
    total_count = 0
    total_saved_mb = 0
    errors = []

    # Get all TIFF files in the given directory
    tiff_files = []
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".tif", ".tiff")):
                    tiff_files.append(os.path.join(root, file))
    else:
        tiff_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith((".tif", ".tiff"))
        ]

    if not tiff_files:
        click.echo(f"No TIFF files found in {directory}")
        return

    click.echo(f"Found {len(tiff_files)} TIFF files to process")

    if dryrun:
        click.echo(
            click.style(
                "DRY RUN MODE: No files will be modified", fg="yellow", bold=True
            )
        )

    # Process each TIFF file with a progress bar
    with click.progressbar(tiff_files, label="Compressing TIF files") as bar:
        for file_path in bar:
            total_count += 1

            # Get file size before compression
            original_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

            success, message = compress_tif_file(file_path, dryrun)

            if success:
                success_count += 1
                # Calculate saved space if not in dry run
                if not dryrun:
                    new_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    saved_mb = original_size - new_size
                    total_saved_mb += saved_mb
                if verbose:
                    click.echo(message)
            else:
                errors.append(message)
                if verbose:
                    click.echo(click.style(message, fg="red"))

    # Summary
    click.echo("\nCompression Summary:")
    click.echo(f"Total files processed: {total_count}")
    click.echo(
        f"Successfully {('would be ' if dryrun else '')}compressed: {success_count}"
    )

    if not dryrun:
        click.echo(
            click.style(f"Total space saved: {total_saved_mb:.2f} MB", fg="green")
        )

    if total_count - success_count > 0:
        click.echo(click.style(f"Failed: {total_count - success_count}", fg="red"))

    if errors and verbose:
        click.echo("\nErrors:")
        for error in errors:
            click.echo(f"  {error}")
