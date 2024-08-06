import re
from pathlib import Path


def find_gt_mask_tif(gt_number: int, gt_dir_path: Path, seg_log_path: Path) -> str:
    """
    Find the ground truth mask TIF file.

    Args:
        gt_number (int): Ground truth number.
        gt_dir_path (Path): Path to the ground truth directory.
        seg_log_path (Path): Path to the segmentation log file.

    Returns:
        str: Path to the ground truth mask TIF file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
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
    """
    Find the ground truth source TIF file.

    Args:
        gt_number (int): Ground truth number.
        gt_dir_path (Path): Path to the ground truth directory.
        seg_log_path (Path): Path to the segmentation log file.

    Returns:
        str: Path to the ground truth source TIF file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    subfolder = seg_log_path.parents[0].name.split('_')[0]
    gt_search_dir = gt_dir_path / subfolder
    all_gt_files = list(gt_search_dir.glob("t*.tif"))
    number_pattern = re.compile(rf't0*{gt_number}\D')
    matching_files = [f for f in all_gt_files if number_pattern.search(f.name)]

    if not matching_files:
        raise FileNotFoundError(f"No gt source file found for number {gt_number} in {gt_search_dir}")

    if len(matching_files) > 1:
        print(
            f"Warning: Multiple gt source files found for number {gt_number}.: \n {matching_files} Using the first one.")

    return str(matching_files[0])


def find_mask_tif(mask_number: int, seg_log_path: Path) -> str:
    """
    Find the mask TIF file.

    Args:
        mask_number (int): Mask number.
        seg_log_path (Path): Path to the segmentation log file.

    Returns:
        str: Path to the mask TIF file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    seg_log_dir = seg_log_path.parent
    all_mask_files = list(seg_log_dir.glob("mask*.tif"))
    number_pattern = re.compile(rf'mask0*{mask_number}\D')
    matching_files = [f for f in all_mask_files if number_pattern.search(f.name)]

    if not matching_files:
        raise FileNotFoundError(f"No mask file found for number {mask_number} in {seg_log_dir}")

    if len(matching_files) > 1:
        print(f"Warning: Multiple mask files found for number {mask_number}. Using the first one.")

    return str(matching_files[0])


def modify_file_path(mask_file_path: str, label: int, out_dir: Path) -> Path:
    """
    Modify the file path for merged images.

    Args:
        mask_file_path (str): Path to the original mask file.
        label (int): Label used for the mask.
        out_dir (Path): Output directory for merged images.

    Returns:
        Path: Modified file path for the merged image.
    """
    mask_file_path = Path(mask_file_path)
    new_path_name = f"{mask_file_path.stem}_{label}{mask_file_path.suffix}"
    relative_path = mask_file_path.relative_to(mask_file_path.anchor)  # Get path relative to root
    new_path = out_dir / relative_path.parent / new_path_name
    return new_path
