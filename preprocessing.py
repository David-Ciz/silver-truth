import tifffile
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import pathlib
from pathlib import Path
import pandas as pd
import click
from sklearn.metrics import jaccard_score
import logging

@click.group()
def cli():
    pass


def preprocess_dataset(dataset_df_path, merged_images_out_dir):
    merged_images_paths = []
    for row in tqdm.tqdm(dataset_df.itertuples(),total=len(dataset_df)):
        merged_gt_image = merge_source_with_mask(row.Mask_file, row.Gt_source_file, row.Gt_mask_file, row.Label, row.J_value)
        merged_image_path = modify_file_path(row.Mask_file, row.Label, merged_images_out_dir)
        pathlib.Path(merged_image_path).parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(merged_image_path, merged_gt_image)
        merged_images_paths.append(merged_image_path)
    dataset_df['merged_image_path'] = merged_images_paths
    dataset_df.to_csv('preprocessed_dataset.csv')


def modify_file_path(mask_file_path, label, out_dir):
    mask_file_path = Path(mask_file_path)
    new_path_name = f"{mask_file_path.stem}_{label}{mask_file_path.suffix}"
    # Create the new path with out_dir as the highest folder
    relative_path = mask_file_path.relative_to(mask_file_path.anchor)  # Get path relative to root
    new_path = Path(out_dir) / relative_path.parent / new_path_name

    return new_path

def merge_source_with_mask(mask_file, gt_source_file, gt_mask_file, label, j_value):
    gt_source = tifffile.imread(gt_source_file)
    gt_masks = tifffile.imread(gt_mask_file)
    gt_mask = gt_masks == label
    pred_masks = tifffile.imread(mask_file)
    pred_mask = get_label_mask(pred_masks, gt_mask, label, j_value)
    if pred_mask is None:
        raise Exception(f"no label mapping found for {mask_file}, with {label}, found")
    else:
        return np.stack([gt_source, pred_mask], axis=0)


def get_label_mask(pred_masks, gt_mask, label, j_value):
    # try if the masks are sorted correctly
    unique_mask_labels = np.unique(pred_masks)
    if label in unique_mask_labels:
        exp_pred_mask = pred_masks == label
        calculated_j_value = np.round(jaccard_score(gt_mask, exp_pred_mask, average="micro"),6)
        if calculated_j_value == j_value:
            return exp_pred_mask
    # if the masks are not sorted correctly, we try every mask
    for mask_label in unique_mask_labels[1:]:
        exp_pred_mask = pred_masks == mask_label
        calculated_j_value = np.round(jaccard_score(gt_mask, exp_pred_mask, average="micro"),6)
        if calculated_j_value == j_value:
            return exp_pred_mask
        else:
            continue
    return None


@cli.command()
@click.argument('dataset-dir', type=click.Path(file_okay=False, dir_okay=True, writable=False, path_type=Path))
@click.argument('gt-dir',type=click.Path(file_okay=False, dir_okay=True, writable=False, path_type=Path))
def create_dataset_csv(dataset_dir, gt_dir):
    pathlib.Path(dataset_dir, gt_dir)
    dataset_dfs = []
    competitor_folders = [x for x in dataset_dir.iterdir() if x.is_dir() and x.name not in ('01_GT', '02_GT', '01', '02')]
    for competitor_folder in competitor_folders:
        segmentation_logs = list(competitor_folder.glob("**/*.txt"))
        for segmentation_log in segmentation_logs:
            dataset_df = parse_segmentation_data(segmentation_log, gt_dir)
            dataset_dfs.append(dataset_df)
    dataset_df = pd.concat(dataset_dfs)
    return dataset_df


def find_gt_mask_tif(gt_number: int, gt_dir_path: Path, seg_log_path: Path) -> str:
    subfolder = seg_log_path.parents[0].name.split('_')[0]
    subfolder = f"{subfolder}_GT/SEG"
    gt_search_dir = gt_dir_path / subfolder
    all_gt_files = list(gt_search_dir.glob("man*.tif"))
    number_pattern = re.compile(rf'man_seg0*{gt_number}\D')
    matching_files = [f for f in all_gt_files if number_pattern.search(f.name)]
    if not matching_files:
        raise FileNotFoundError(f"No gt mask file found for number {gt_number} in {gt_search_dir}")

    if len(matching_files) > 1:
        print(f"Warning: Multiple gt mask file found for number {gt_number}.: \n {matching_files} Using the first one.")

    return str(matching_files[0])


def find_gt_source_tif(gt_number: int, gt_dir_path: Path, seg_log_path: Path) -> str:
    subfolder = seg_log_path.parents[0].name.split('_')[0]
    gt_search_dir = gt_dir_path / subfolder
    gt_pattern = f"t*{gt_number:d}.tif"
    all_gt_files = list(gt_search_dir.glob("t*.tif"))

    # Use regex to match the exact number
    number_pattern = re.compile(rf't0*{gt_number}\D')
    matching_files = [f for f in all_gt_files if number_pattern.search(f.name)]
    if not matching_files:
        raise FileNotFoundError(f"No gt source file found for number {gt_number} in {gt_search_dir}")

    if len(matching_files) > 1:
        print(
            f"Warning: Multiple gt source files found for number {gt_number}.: \n {matching_files} Using the first one.")

    return str(matching_files[0])


def find_mask_tif(mask_number: int, seg_log_path: Path) -> str:
    seg_log_dir = seg_log_path.parent
    mask_pattern = f"mask*{mask_number:d}.tif"
    all_mask_files = list(seg_log_dir.glob("mask*.tif"))

    # Use regex to match the exact number
    number_pattern = re.compile(rf'mask0*{mask_number}\D')
    matching_files = [f for f in all_mask_files if number_pattern.search(f.name)]
    if not matching_files:
        raise FileNotFoundError(f"No mask file found for number {mask_number} in {seg_log_dir}")

    if len(matching_files) > 1:
        print(f"Warning: Multiple mask files found for number {mask_number}. Using the first one.")

    return str(matching_files[0])

def parse_segmentation_data(seg_log_path, gt_dir_path):
    with open(seg_log_path, 'r') as f:
        file_content = f.readlines()
    # Lists to store parsed data
    t_values = []
    mask_files = []
    gt_mask_files = []
    gt_source_files = []
    labels = []
    j_values = []

    # Regular expressions to match T value and GT_label/J pairs
    t_pattern = re.compile(r'----------T=(\d+) Z=\d+----------')
    data_pattern = re.compile(r'GT_label=(\d+) J=([\d.]+)')

    current_t = None
    current_mask = None
    current_mask_gt = None

    for line in file_content:
        t_match = t_pattern.match(line)
        if t_match:
            current_t = int(t_match.group(1))
            current_mask = find_mask_tif(current_t, seg_log_path)
            current_source_gt = find_gt_source_tif(current_t, gt_dir_path, seg_log_path)
            current_mask_gt = find_gt_mask_tif(current_t, gt_dir_path, seg_log_path)
        else:
            data_match = data_pattern.match(line)
            if data_match and current_t is not None:
                label = int(data_match.group(1))
                j_value = float(data_match.group(2))
                mask_files.append(current_mask)
                gt_source_files.append(current_source_gt)
                gt_mask_files.append(current_mask_gt)
                labels.append(label)
                j_values.append(j_value)

    return pd.DataFrame({
        'Mask_file': mask_files,
        'Gt_source_file': gt_source_files,
        'Gt_mask_file': gt_mask_files,
        'Label': labels,
        'J_value': j_values
    })

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()