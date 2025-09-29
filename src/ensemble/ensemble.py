#import torch
#import lightning as pl
from tqdm import tqdm
import tifffile
from scipy.ndimage import find_objects
import logging

from src.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)

# Basic Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#print("Device:", device)

#seed_value = 42
#pl.seed_everything(seed_value)


def _find_largest_gt_cell_size(dataset_dataframe_path: str) -> int:
    """
    Finds the largest ground truth segmentation.
    """
    largest_size = 0

    # loads the QA dataset
    df = load_dataframe_from_parquet_with_metadata(dataset_dataframe_path)
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
    return largest_size



def run_ensemble_experiment(model: str, parameters: dict):
    pass
    


