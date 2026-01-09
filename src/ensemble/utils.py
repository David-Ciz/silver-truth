from tqdm import tqdm
import tifffile
from scipy.ndimage import find_objects
import src.ensemble.external as ext
import torch

ORIGINAL_DATASETS = {
    "BF-C2DL-HSC": "ds1",
    "BF-C2DL-MuSC": "ds2",
 }

DATABANKS_DIR = "data/ensemble_data/databanks"


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def find_largest_gt_cell_size(dataset_dataframe_path: str) -> tuple[int,str]:
    """
    Finds the largest ground truth segmentation.
    """
    largest_size = 0
    largest_size_img = ""

    # loads the QA dataset
    df = ext.load_parquet(dataset_dataframe_path)
    checked_gt_images = []

    for gt_image_path in tqdm(
        df['gt_image'], total=len(df), desc="checking GT cell segmentation size"
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
                max_seg_size = max(slice_y.stop - slice_y.start, 
                                   slice_x.stop - slice_x.start)
                if largest_size < max_seg_size:
                    largest_size = max_seg_size
                    largest_size_img = gt_image_path
    return largest_size, largest_size_img


def get_databank_name(build_opt: dict) -> str:
    """
    Returns the databank name according to a build options dictionary.
    """
    qa_name = f"{build_opt["qa"]}_t{int(build_opt["qa_threshold"]*100)}" if build_opt["qa"] else "QA--"
    return f"{build_opt["version"].name}_{ORIGINAL_DATASETS[build_opt["name"]]}-{get_splits_name(build_opt)}_{qa_name}"


def get_splits_name(build_opt: dict) -> str:
    """
    Constructs the folder name part of the split databank build definition.
    """
    return f"{build_opt["split_seed"]}-{int(build_opt["split_sets"][0]*100)}{int(build_opt["split_sets"][1]*100)}"