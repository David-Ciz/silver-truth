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


from src.data_processing.utils.dataset_dataframe_creation import (
    is_valid_competitor_folder,
)

# Constants
RAW_DATA_FOLDERS = {"01", "02"}
GT_FOLDER_FIRST = "01_GT"
GT_FOLDER_SECOND = "02_GT"
SEG_FOLDER = "SEG"
TRA_FOLDER = "TRA"
RES_FOLDER_FIRST = "01_RES"
RES_FOLDER_SECOND = "02_RES"


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
    if label_img.shape != tracking_img.shape:
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
            if j_value > 0:
                logging.error(f"Jaccard index for label {label} is {j_value}.")
                return False

    return True


def verify_dataset_synchronization_logic(
    dataset_dir: Path | str, tracking_dir_01: Path | str, tracking_dir_02: Path | str
):
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
            # Choose the corresponding tracking folder based on dataset type
            tracking_folder = data["tracking_folder"]
            desynchronized_images = verify_folder_synchronization_logic(
                str(data["segmentations"]), str(tracking_folder)
            )
            if desynchronized_images:
                desynchronized_subfolders.add(dataset_type)
    return desynchronized_subfolders


def synchronize_datasets_logic(datasets_folder, output_directory, debug):
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
    logging.info("--- Processing Summary ---")
    logging.info(f"✅ Successfully processed datasets: {processed_datasets}")

    if skipped_datasets > 0:
        logging.info(
            f"⚠️ Skipped datasets due to missing tracking data: {skipped_datasets}"
        )
        logging.info(f"Failed datasets: {', '.join(failed_datasets)}")
