#import torch
#import lightning as pl
import pandas as pd
import os
from pathlib import Path
from os.path import join
import numpy as np
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



def create_analysis_dataset(qa_dataset_dataframe_path: str, output_path: str) -> None:
    """
    Creates an image dataset for helping visualizing the differences between GT and proposed segmentations.
    Should use cropped output of create_qa_dataset (in order to get the center of the cell).
    """
    # create output path if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # load the dataframe
    df = load_dataframe_from_parquet_with_metadata(qa_dataset_dataframe_path)
    
    for row in tqdm(
        df.itertuples(), total=len(df), desc="Processing images"
    ): 
        # load gt image 
        gt_image = tifffile.imread(row.gt_image) # type: ignore
        combined_qa_image = tifffile.imread(row.stacked_path).astype(np.uint8) # type: ignore
        #segmented_qa_image = combined_qa_image[1]
        seg_crop = combined_qa_image[1]

        gt_crop = gt_image[row.crop_y_start:row.crop_y_end, 
                           row.crop_x_start:row.crop_x_end]

        # contains the rest of the gt segmentations, shown in blue
        blue_layer = np.logical_and(gt_crop > 0, gt_crop != row.label).astype(np.uint8) * 255
        gt_crop = (gt_crop == row.label).astype(np.uint8) * 255
        stacked_crop = np.stack([seg_crop, gt_crop, blue_layer], axis=0)

        # campaign - image id - cell id - competitor
        campaign, img_id, competitor, suffix = Path(row.stacked_path).name.split("_") # type: ignore
        cell_id, __ = suffix.split(".")
        new_name = f"{campaign}_{img_id}_{cell_id}_{competitor}.tif"
        # Save the image
        new_img_path = join(output_path, new_name)
        # output folder
        tifffile.imwrite(new_img_path, stacked_crop)



def build_initial_dataset_v001(
        qa_dataset_dataframe_path: str, 
        output_path: str,
        crop_size: int = 64, 
        apply_blue_layer: bool = True
        ) -> None: 
    """
    Generate the initial dataset for the ensemble.
    
    For each gt_image, for each label, all the competitors cropped segmentations (from qa) "votes" (each ON pixel equals 1 vote) /
    are summed into a single image layer, and then normalized by the number of competitors.
    The union of this sum image and the gt image is used to find the minimum appropriate crop size.
    For each gt image, a new image (crop size) is created with the cropped sum image as the 1st layer /
    and the cropped gt image as 2nd layer.

    """
    # dataset id
    dataset_id = "v001"
    composed_output_path = os.path.join(output_path, dataset_id)
    # destination path of the created images
    images_output_path = os.path.join(composed_output_path, "images")
    # create images path if it doesn't exist
    Path(images_output_path).mkdir(parents=True, exist_ok=True)

    # output parquet support file
    data_list = []

    # loads the QA dataset
    df = load_dataframe_from_parquet_with_metadata(qa_dataset_dataframe_path)
    # get the gt images
    unique_gt_images = df["gt_image"].unique()
    # get crop size
    qa_crop_size = df.iloc[0]["crop_size"]
    qa_crop_half_size = qa_crop_size // 2
    # sets the size of the array that will contain the summed images
    canvas_size = qa_crop_size * 4
    canvas_half_size = canvas_size // 2

    # go through the gt images
    for gt_image_path in tqdm(
        unique_gt_images, total=len(unique_gt_images), desc="Iterating over gt"
    ): 
        # get the competitors segmentation for the given gt image
        df_shared_gt = df[df["gt_image"] == gt_image_path]
        # load gt image
        gt_image = tifffile.imread(gt_image_path)
        # find the different labels in a gt
        labels = np.unique(gt_image)[1:]

        # go through each label
        for label in labels:
            # dataframe with competitors segmentation for the same cell
            df_same_cell = df_shared_gt[df_shared_gt["label"] == label]
            # create array for the summation
            canvas = np.zeros((canvas_size, canvas_size), dtype=np.int32)
            # select first row
            first_row = df_same_cell.iloc[0]
            # gets the original center that is used to center and overlap the competitors segmentations
            qa_crop_original_center_y, qa_crop_original_center_x = first_row["original_center_y"], first_row["original_center_x"]

            # go through the competitors segmentations and add them to canvas
            for row in df_same_cell.itertuples():
                # load competitor segmentation from qa
                competitor_qa_image = tifffile.imread(row.stacked_path)[1]
                # finds the starting point for the crop square
                start_y = canvas_half_size - qa_crop_half_size + row.original_center_y - qa_crop_original_center_y
                start_x = canvas_half_size - qa_crop_half_size + row.original_center_x - qa_crop_original_center_x
                # create an image that is the sum of all segmentations of a label
                canvas[start_y : start_y + qa_crop_size, start_x : start_x + qa_crop_size] += competitor_qa_image

            # normalized competitors sumation
            canvas = (canvas // len(df_same_cell)).astype(np.uint8)
            # find the center of the image summation
            canvas_mask = (canvas > 0).astype(np.uint8)
            obj_slice_y, obj_slice_x = find_objects(canvas_mask)[0]
            obj_center_y = obj_slice_y.start + ((obj_slice_y.stop - obj_slice_y.start) // 2)
            obj_center_x = obj_slice_x.start + ((obj_slice_x.stop - obj_slice_x.start) // 2)

            # crop canvas
            crop_half_size = crop_size // 2
            obj_min_y, obj_min_x = obj_center_y - crop_half_size, obj_center_x - crop_half_size
            canvas_crop = canvas[obj_min_y : obj_min_y + crop_size, obj_min_x : obj_min_x + crop_size]
            # crop full gt image according to canvas and nem center of segmentation summation
            gt_crop_min_y = qa_crop_original_center_y - canvas_half_size + obj_min_y
            gt_crop_min_x = qa_crop_original_center_x - canvas_half_size + obj_min_x
            gt_crop = gt_image[gt_crop_min_y : gt_crop_min_y + crop_size, 
                               gt_crop_min_x : gt_crop_min_x + crop_size]
            gt_crop = (gt_crop == label).astype(np.uint8) * 255

            # stack layers
            if apply_blue_layer:
                empty_blue_layer = np.zeros((crop_size, crop_size), dtype=np.uint8)
                stacked_crop = np.stack([canvas_crop, gt_crop, empty_blue_layer], axis=0)
            else:
                stacked_crop = np.stack([canvas_crop, gt_crop], axis=0)

            # set new dataset image path
            campaign, img_id, __, __ = first_row.cell_id.split("_")
            new_image_name = f"{campaign}_{img_id}_{label}.tif"
            new_image_path = os.path.join(images_output_path, new_image_name)
            # save image
            tifffile.imwrite(new_image_path, stacked_crop)

            # save details
            data_list.append(
                {
                    "campaign": campaign,
                    "image_id": img_id,
                    "label": label,
                    "crop_size": crop_size,
                    "image_path": new_image_path,
                }
            )

    # convert list to dataframe
    output_df = pd.DataFrame(data_list)
    # build output parquet path
    parquet_output_path = os.path.join(composed_output_path, f"ensemble_dataset_{dataset_id}.parquet")
    # save to parquet file
    output_df.to_parquet(parquet_output_path)




def run_ensemble_experiment(model: str, parameters: dict):
    pass
    


