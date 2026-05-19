from tqdm import tqdm
import tifffile
import torch
import numpy as np
from scipy.ndimage import find_objects
import silver_truth.ensemble.external as ext


ORIGINAL_DATASETS = {
    "BF-C2DL-HSC": "ds1",
    "BF-C2DL-MuSC": "ds2",
}

DATABANKS_DIR = "data/ensemble_data/databanks"


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_cell_size_hist(dataset_dataframe_path: str, num_bins: int = 20) -> dict:
    """
    Returns the cell size histogram of a dataset.
    """
    height_arr = []
    width_arr = []

    # loads the QA dataset
    df = ext.load_parquet(dataset_dataframe_path)
    checked_gt_images = []

    for gt_image_path in tqdm(
        df["gt_image"], total=len(df), desc="calculating cell size histogram"
    ):
        # check if the gt image was already processed
        if gt_image_path in checked_gt_images:
            continue

        checked_gt_images.append(gt_image_path)

        # load gt image
        gt_image = tifffile.imread(gt_image_path).astype(int)

        # calculate cell size
        for obj in find_objects(gt_image):
            if obj is not None:
                slice_y, slice_x = obj
                height_arr.append(slice_y.stop - slice_y.start)
                width_arr.append(slice_x.stop - slice_x.start)

    return {"height_hist":np.histogram(height_arr, num_bins), "height_raw": np.array(height_arr), 
            "width_hist":np.histogram(width_arr, num_bins), "width_raw": np.array(width_arr)}



def find_largest_gt_cell_size(dataset_dataframe_path: str) -> tuple[int, str]:
    """
    Finds the largest ground truth segmentation.
    """
    largest_size = 0
    largest_size_img = ""

    # loads the QA dataset
    df = ext.load_parquet(dataset_dataframe_path)
    checked_gt_images = []

    for gt_image_path in tqdm(
        df["gt_image"], total=len(df), desc="checking GT cell segmentation size"
    ):
        # check if the gt image was already processed
        if gt_image_path in checked_gt_images:
            continue

        checked_gt_images.append(gt_image_path)

        # load gt image
        gt_image = tifffile.imread(gt_image_path).astype(int)

        # check for largest size
        for obj in find_objects(gt_image):
            if obj is not None:
                slice_y, slice_x = obj
                max_seg_size = max(
                    slice_y.stop - slice_y.start, slice_x.stop - slice_x.start
                )
                if largest_size < max_seg_size:
                    largest_size = max_seg_size
                    largest_size_img = gt_image_path
    return largest_size, largest_size_img


def get_databank_name(build_opt: dict) -> str:
    """
    Returns the databank name according to a build options dictionary.
    """
    qa_name = (f'{build_opt["qa"]}_t{int(build_opt["qa_threshold"]*100)}' if build_opt["qa"] else "QA--")
    f1vn = get_fold1_version_name(build_opt)
    return f'{build_opt["databank"].name}_{ORIGINAL_DATASETS[build_opt["name"]]}{f1vn}-{get_splits_name(build_opt)}_{qa_name}'


def get_fold1_version_name(build_opt: dict) -> str:
    if "fold1_version" in build_opt:
        f1v = build_opt["fold1_version"]
        assert(f1v == 0 or f1v == 1 or f1v == 2)
        return f"-f1v{f1v}"
    else:
        return ""


def get_splits_name(build_opt: dict) -> str:
    """
    Constructs the folder name part of the split databank build definition.
    """
    return f'{build_opt["split_seed"]}-{int(build_opt["split_sets"][0]*100)}{int(build_opt["split_sets"][1]*100)}'
