import re

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Set

import tqdm
from tifffile import tifffile

from preprocessing.file_utils import find_mask_tif, find_gt_source_tif, find_gt_mask_tif, modify_file_path
from preprocessing.image_preprocessing import merge_source_with_mask


def parse_segmentation_data(seg_log_path: Path, gt_dir_path: Path) -> pd.DataFrame:
    """
    Parse segmentation data from a log file and create a DataFrame.

    Args:
        seg_log_path (Path): Path to the segmentation log file.
        gt_dir_path (Path): Path to the ground truth directory.

    Returns:
        pd.DataFrame: DataFrame containing parsed segmentation data.
    """
    with open(seg_log_path, 'r') as f:
        file_content = f.readlines()

    parsed_data: List[Tuple[str, str, str, int, float]] = []
    t_pattern = re.compile(r'----------T=(\d+) Z=\d+----------')
    data_pattern = re.compile(r'GT_label=(\d+) J=([\d.]+)')

    current_t = None
    current_mask = None
    current_source_gt = None
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
                parsed_data.append((current_mask, current_source_gt, current_mask_gt, label, j_value))

    return pd.DataFrame(parsed_data, columns=['Mask_file', 'Gt_source_file', 'Gt_mask_file', 'Label', 'J_value'])


def preprocess_dataset(dataset_df_path: Path, merged_images_out_dir: Path) -> None:
    """
    Preprocess the dataset by merging source images with masks.

    Args:
        dataset_df_path (Path): Path to the input dataset CSV file.
        merged_images_out_dir (Path): Path to the output directory for merged images.
    """
    dataset_df = pd.read_csv(dataset_df_path)
    merged_images_paths: List[Path] = []
    current_mask_file: Optional[str] = None
    current_gt_source_file: Optional[str] = None
    current_gt_mask_file: Optional[str] = None
    gt_masks: Optional[np.ndarray] = None
    pred_masks: Optional[np.ndarray] = None
    gt_source: Optional[np.ndarray] = None
    used_labels: Set[int] = set()

    for row in tqdm.tqdm(dataset_df.itertuples(), total=len(dataset_df)):
        mask_file, gt_source_file, gt_mask_file = row.Mask_file, row.Gt_source_file, row.Gt_mask_file
        label, j_value = row.Label, row.J_value

        if current_mask_file != mask_file:
            pred_masks = tifffile.imread(mask_file)
            current_mask_file = mask_file
            used_labels = set()

        if current_gt_source_file != gt_source_file:
            gt_source = tifffile.imread(gt_source_file)
            current_gt_source_file = gt_source_file

        if current_gt_mask_file != gt_mask_file:
            gt_masks = tifffile.imread(gt_mask_file)
            current_gt_mask_file = gt_mask_file

        gt_mask = gt_masks == label
        merged_gt_image, used_label = merge_source_with_mask(pred_masks, gt_source, gt_mask, label, j_value, used_labels)
        used_labels.add(used_label)

        merged_image_path = modify_file_path(mask_file, label, merged_images_out_dir)
        merged_image_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(merged_image_path), merged_gt_image)
        merged_images_paths.append(merged_image_path)

    dataset_df['merged_image_path'] = merged_images_paths
    dataset_df.to_csv(merged_images_out_dir / 'preprocessed_dataset.csv', index=False)