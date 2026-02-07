import os
from pathlib import Path
import tifffile
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.ndimage import find_objects
from silver_truth.ensemble import utils
import silver_truth.ensemble.external as ext
import silver_truth.data_processing.utils.parquet_utils as p_utils
from silver_truth.ensemble.datasets import Version

SPLIT_COL = p_utils.SPLITS_COLUMN


def _filter_by_qa(df_same_cell: pd.DataFrame, build_opt: dict) -> pd.DataFrame:
    """
    Optionally gates competitors by QA score for one GT label.
    Falls back to the highest-QA competitor when none pass the threshold.
    """
    qa_col = build_opt.get("qa")
    qa_threshold = build_opt.get("qa_threshold")
    if qa_col is None or qa_threshold is None:
        return df_same_cell

    df_same_cell_thresh = df_same_cell[df_same_cell[qa_col] > qa_threshold]
    if len(df_same_cell_thresh) > 0:
        return df_same_cell_thresh

    max_ite = df_same_cell[qa_col].values.argmax()
    return df_same_cell.iloc[max_ite : max_ite + 1]


def _read_segmentation_mask(stacked_path: str) -> np.ndarray:
    """
    Reads a QA stacked image and returns the competitor segmentation mask layer.
    """
    image = tifffile.imread(stacked_path)
    if image.ndim == 3:
        # QA stacks are expected to be channel-first.
        return image[1].astype(np.uint8)
    return image.astype(np.uint8)


def _build_databank_image_level(
    build_opt: dict,
    qa_dataset_path: str,
    output_path: str,
    databank_foldername: str,
    images_output_path: str,
) -> str:
    """
    Build one ensemble sample per full image.

    Input layer is a normalized vote map over all GT labels (all cells) for that image.
    Target layer is a full-image binary GT mask (all cells).
    """
    data_list = []
    df = ext.load_parquet(qa_dataset_path)

    if "crop_size" in df.columns and df["crop_size"].notna().any():
        raise ValueError(
            "Image-level databank requires a full-image QA parquet (crop_size must be None)."
        )

    unique_gt_images = df["gt_image"].unique()
    print(f"\n{build_opt}")
    num_removed_segs = 0

    for gt_image_path in tqdm(
        unique_gt_images, total=len(unique_gt_images), desc="Iterating over gt"
    ):
        df_shared_gt = df[df["gt_image"] == gt_image_path]
        if len(df_shared_gt) == 0:
            continue

        first_row = df_shared_gt.iloc[0]
        assert (df_shared_gt[SPLIT_COL].values == first_row[SPLIT_COL]).all()

        gt_image = tifffile.imread(gt_image_path)
        gt_binary = (gt_image > 0).astype(np.uint8) * 255

        vote_canvas = np.zeros(gt_binary.shape, dtype=np.int32)
        selected_qa_scores = []
        labels = np.unique(gt_image)[1:]

        for label in labels:
            df_same_cell = df_shared_gt[df_shared_gt["label"] == label]
            if len(df_same_cell) == 0:
                continue

            num_segs = len(df_same_cell)
            df_same_cell = _filter_by_qa(df_same_cell, build_opt)
            num_removed_segs += num_segs - len(df_same_cell)
            if build_opt.get("qa"):
                selected_qa_scores.extend(df_same_cell[build_opt["qa"]].tolist())

            label_canvas = np.zeros(gt_binary.shape, dtype=np.int32)
            for row in df_same_cell.itertuples():
                competitor_mask = _read_segmentation_mask(row.stacked_path)
                if competitor_mask.shape != gt_binary.shape:
                    raise ValueError(
                        f"Shape mismatch for {row.stacked_path}: "
                        f"mask={competitor_mask.shape}, gt={gt_binary.shape}."
                    )
                label_canvas += competitor_mask.astype(np.int32)

            label_canvas = (label_canvas // len(df_same_cell)).astype(np.uint8)
            vote_canvas += label_canvas

        vote_canvas = np.clip(vote_canvas, 0, 255).astype(np.uint8)
        empty_blue_layer = np.zeros_like(gt_binary, dtype=np.uint8)
        stacked_image = np.stack([vote_canvas, gt_binary, empty_blue_layer], axis=0)

        campaign = str(first_row["campaign_number"])
        image_id = str(first_row["original_image_key"])
        new_image_name = f"{campaign}_{image_id}.tif"
        new_image_path = os.path.join(images_output_path, new_image_name)
        tifffile.imwrite(new_image_path, stacked_image)

        if selected_qa_scores:
            qa_jaccard_avg = float(np.mean(selected_qa_scores))
            qa_jaccard_min = float(np.min(selected_qa_scores))
        else:
            qa_jaccard_avg = 0
            qa_jaccard_min = 0

        data_list.append(
            {
                "campaign": campaign,
                "image_id": image_id,
                "crop_size": None,
                "image_path": new_image_path,
                "gt_image": gt_image_path,
                SPLIT_COL: first_row[SPLIT_COL],
                "qa_jaccard_avg": qa_jaccard_avg,
                "qa_jaccard_min": qa_jaccard_min,
            }
        )

    print(
        f"Number of removed segmentations: {num_removed_segs} ({((num_removed_segs/len(df))*100):.2f}%)"
    )
    output_df = pd.DataFrame(data_list)
    parquet_output_path = os.path.join(output_path, f"{databank_foldername}.parquet")
    output_df.to_parquet(parquet_output_path)
    ext.compress_images(images_output_path)
    return parquet_output_path


def build_analysis_databank_full(qa_dataset_path: str, output_path: str) -> None:
    """
    Creates an image dataset for helping visualizing the differences between GT and proposed segmentations.
    """
    # create output path if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # load the dataframe
    df = ext.load_parquet(qa_dataset_path)

    done_imgs = []

    for row in tqdm(df.itertuples(), total=len(df), desc="Processing images"):
        id = f"{row.campaign_number}_{row.original_image_key}_{row.competitor}"
        if id in done_imgs:
            continue
        done_imgs.append(id)

        # load gt image
        seg_img = tifffile.imread(row.segmentation_path)  # type: ignore
        gt_image = tifffile.imread(row.gt_image)  # type: ignore
        ori_image = tifffile.imread(row.original_image_path)  # type: ignore

        seg_img = (seg_img > 0).astype(np.uint8) * 255
        gt_image = (gt_image > 0).astype(np.uint8) * 255
        stacked_crop = np.stack([seg_img, gt_image, ori_image], axis=0)

        # campaign - image id - cell id - competitor
        campaign, img_id, competitor, _ = Path(row.stacked_path).name.split("_")  # type: ignore
        new_name = f"{campaign}_{img_id}_{competitor}.tif"
        # Save the image
        new_img_path = os.path.join(output_path, new_name)
        # output folder
        tifffile.imwrite(new_img_path, stacked_crop)

    # compress images
    ext.compress_images(output_path)


def build_analysis_databank(qa_dataset_path: str, output_path: str) -> None:
    """
    Creates an image dataset for helping visualizing the differences between GT and proposed segmentations.
    Should use cropped output of create_qa_dataset (in order to get the center of the cell).
    """
    # create output path if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # load the dataframe
    df = ext.load_parquet(qa_dataset_path)

    for row in tqdm(df.itertuples(), total=len(df), desc="Processing images"):
        # campaign - image id - cell id - competitor
        campaign, img_id, competitor, suffix = Path(row.stacked_path).name.split("_")  # type: ignore
        cell_id, __ = suffix.split(".")
        new_name = f"{campaign}_{img_id}_{cell_id}_{competitor}.tif"
        new_img_path = os.path.join(output_path, new_name)

        # load gt image
        gt_image = tifffile.imread(row.gt_image)  # type: ignore
        combined_qa_image = tifffile.imread(row.stacked_path).astype(np.uint8)  # type: ignore
        seg_crop = combined_qa_image[1]

        crop_y_start, crop_y_end = row.crop_y_start, row.crop_y_end
        crop_x_start, crop_x_end = row.crop_x_start, row.crop_x_end

        if crop_y_start < 0:  # type: ignore
            y_inc = -crop_y_start  # type: ignore
            gt_image = np.vstack((np.zeros((y_inc, gt_image.shape[1])), gt_image))  # type: ignore
            crop_y_end += y_inc
            crop_y_start = 0
        if crop_x_start < 0:  # type: ignore
            x_inc = -crop_x_start  # type: ignore
            gt_image = np.hstack((np.zeros((gt_image.shape[0], x_inc)), gt_image))
            crop_x_end += x_inc
            crop_x_start = 0
        if gt_image.shape[0] < crop_y_end:
            gt_image = np.vstack(
                (
                    gt_image,
                    np.zeros((crop_y_end - gt_image.shape[0], gt_image.shape[1])),
                )
            )
        if gt_image.shape[1] < crop_x_end:  # type: ignore
            gt_image = np.hstack(
                (
                    gt_image,
                    np.zeros((gt_image.shape[0], crop_x_end - gt_image.shape[1])),
                )
            )

        gt_crop = gt_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # contains the rest of the gt segmentations, shown in blue
        blue_layer = (
            np.logical_and(gt_crop > 0, gt_crop != row.label).astype(np.uint8) * 255
        )
        gt_crop = (gt_crop == row.label).astype(np.uint8) * 255
        stacked_crop = np.stack([seg_crop, gt_crop, blue_layer], axis=0)

        # save to output folder
        tifffile.imwrite(new_img_path, stacked_crop)

    # compress images
    ext.compress_images(output_path)


def build_databank(build_opt: dict, qa_dataset_path: str, output_path: str) -> str:
    if build_opt["version"] == Version.B1:
        return build_databank_B1(build_opt, qa_dataset_path, output_path)
    elif build_opt["version"] == Version.C1:
        return build_databank_C1(build_opt, qa_dataset_path, output_path)

    raise Exception("Error: Dataset version not yet supported.")


def build_databank_B1(build_opt: dict, qa_dataset_path: str, output_path: str) -> str:
    """
    Generate the databank for dataset B1 versions.

    Each image has a competitor's segmentation on the first layer, the corresponding gt image on the second layer,
    and an empty third layer.
    """
    # destination path of the created images
    databank_foldername = utils.get_databank_name(build_opt)
    images_output_path = os.path.join(output_path, databank_foldername)
    # create images path if it doesn't exist
    Path(images_output_path).mkdir(parents=True, exist_ok=True)

    # load the dataframe
    df = ext.load_parquet(qa_dataset_path)

    # output parquet support file
    data_list = []

    for row in tqdm(df.itertuples(), total=len(df), desc="Processing images"):
        qa_jaccard = 0
        if build_opt["qa"]:
            qa_jaccard = getattr(row, row.qa)  # type: ignore
            if row.qa_threshold < qa_jaccard:
                continue

        # campaign - image id - cell id - competitor
        campaign, img_id, competitor, suffix = Path(row.stacked_path).name.split("_")  # type: ignore
        cell_id, __ = suffix.split(".")
        new_name = f"{campaign}_{img_id}_{cell_id}_{competitor}.tif"
        new_img_path = os.path.join(images_output_path, new_name)

        # load gt image
        gt_image = tifffile.imread(row.gt_image)  # type: ignore
        combined_qa_image = tifffile.imread(row.stacked_path).astype(np.uint8)  # type: ignore
        seg_crop = combined_qa_image[1]

        crop_y_start, crop_y_end = row.crop_y_start, row.crop_y_end
        crop_x_start, crop_x_end = row.crop_x_start, row.crop_x_end

        if crop_y_start < 0:  # type: ignore
            y_inc = -crop_y_start  # type: ignore
            gt_image = np.vstack((np.zeros((y_inc, gt_image.shape[1])), gt_image))  # type: ignore
            crop_y_end += y_inc
            crop_y_start = 0
        if crop_x_start < 0:  # type: ignore
            x_inc = -crop_x_start  # type: ignore
            gt_image = np.hstack((np.zeros((gt_image.shape[0], x_inc)), gt_image))
            crop_x_end += x_inc
            crop_x_start = 0
        if gt_image.shape[0] < crop_y_end:
            gt_image = np.vstack(
                (
                    gt_image,
                    np.zeros((crop_y_end - gt_image.shape[0], gt_image.shape[1])),
                )
            )
        if gt_image.shape[1] < crop_x_end:  # type: ignore
            gt_image = np.hstack(
                (
                    gt_image,
                    np.zeros((gt_image.shape[0], crop_x_end - gt_image.shape[1])),
                )
            )

        gt_crop = gt_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

        # contains the rest of the gt segmentations, shown in blue
        empty_blue_layer = np.zeros(
            (build_opt["crop_size"], build_opt["crop_size"]), dtype=np.uint8
        )
        gt_crop = (gt_crop == row.label).astype(np.uint8) * 255
        stacked_crop = np.stack([seg_crop, gt_crop, empty_blue_layer], axis=0)

        # save to output folder
        tifffile.imwrite(new_img_path, stacked_crop)

        # save details
        data_list.append(
            {
                "campaign": campaign,
                "image_id": img_id,
                "label": row.label,
                "crop_size": row.crop_size,
                "image_path": new_img_path,
                "gt_image": row.gt_image,
                SPLIT_COL: getattr(row, SPLIT_COL),
                # TODO: add jaccard?
                # "qa_jaccard": qa_jaccard,
            }
        )

    # convert list to dataframe
    output_df = pd.DataFrame(data_list)
    parquet_output_path = os.path.join(output_path, f"{databank_foldername}.parquet")
    # save to parquet file
    output_df.to_parquet(parquet_output_path)

    # compress images
    ext.compress_images(images_output_path)

    return parquet_output_path


def build_databank_C1(build_opt: dict, qa_dataset_path: str, output_path: str) -> str:
    """
    Generate the databank for dataset C1 versions.

    For each gt_image, for each label, all the competitors cropped segmentations (from qa) "votes" (each ON pixel equals 1 vote) /
    are summed into a single image layer, and then normalized by the number of competitors.
    The union of this sum image and the gt image is used to find the minimum appropriate crop size.
    For each gt image, a new image (crop size) is created with the cropped sum image as the 1st layer /
    and the cropped gt image as 2nd layer.
    """
    # TODO: update description.

    aggregation_level = build_opt.get("aggregation_level", "cell")
    if aggregation_level not in {"cell", "image"}:
        raise ValueError(
            f"Unsupported aggregation_level='{aggregation_level}'. Use 'cell' or 'image'."
        )

    # destination path of the created images
    databank_foldername = utils.get_databank_name(build_opt)
    images_output_path = os.path.join(output_path, databank_foldername)
    # create images path if it doesn't exist
    Path(images_output_path).mkdir(parents=True, exist_ok=True)

    if aggregation_level == "image":
        return _build_databank_image_level(
            build_opt=build_opt,
            qa_dataset_path=qa_dataset_path,
            output_path=output_path,
            databank_foldername=databank_foldername,
            images_output_path=images_output_path,
        )

    # output parquet support file
    data_list = []

    # loads the QA dataset
    df = ext.load_parquet(qa_dataset_path)

    # get the gt images
    unique_gt_images = df["gt_image"].unique()
    # get crop size
    crop_size = df.iloc[0]["crop_size"]
    if pd.isna(crop_size):
        raise ValueError(
            "Cell-level databank requires cropped QA inputs (crop_size must be set). "
            "Use aggregation_level='image' for full-image QA inputs."
        )
    crop_size = int(crop_size)
    qa_crop_half_size = crop_size // 2
    # sets the size of the array that will contain the summed images
    canvas_size = crop_size * 4 if crop_size <= 64 else crop_size * 2
    canvas_half_size = canvas_size // 2

    print(f"\n{build_opt}")
    num_removed_segs = 0

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
            # confirm that all cells have the same split type
            assert (df_same_cell.split.values == first_row.split).all()

            # gets the original center that is used to center and overlap the competitors segmentations
            qa_crop_original_center_y, qa_crop_original_center_x = (
                first_row["original_center_y"],
                first_row["original_center_x"],
            )

            num_segs = len(df_same_cell)
            # filter segmentations according to QA
            df_same_cell = _filter_by_qa(df_same_cell, build_opt)

            num_removed_segs += num_segs - len(df_same_cell)
            # go through the competitors segmentations and add them to canvas
            for row in df_same_cell.itertuples():
                # load competitor segmentation from qa
                competitor_qa_image = _read_segmentation_mask(row.stacked_path)
                # finds the starting point for the crop square
                start_y = (
                    canvas_half_size
                    - qa_crop_half_size
                    + row.original_center_y
                    - qa_crop_original_center_y
                )
                start_x = (
                    canvas_half_size
                    - qa_crop_half_size
                    + row.original_center_x
                    - qa_crop_original_center_x
                )
                # create an image that is the sum of all segmentations of a label
                canvas[
                    start_y : start_y + crop_size, start_x : start_x + crop_size
                ] += competitor_qa_image

            # normalized competitors sumation
            canvas = (canvas // len(df_same_cell)).astype(np.uint8)
            # find the center of the image summation
            canvas_mask = (canvas > 0).astype(np.uint8)
            obj_slice_y, obj_slice_x = find_objects(canvas_mask)[0]
            obj_center_y = obj_slice_y.start + (
                (obj_slice_y.stop - obj_slice_y.start) // 2
            )
            obj_center_x = obj_slice_x.start + (
                (obj_slice_x.stop - obj_slice_x.start) // 2
            )

            # crop canvas
            crop_half_size = crop_size // 2
            obj_min_y, obj_min_x = (
                obj_center_y - crop_half_size,
                obj_center_x - crop_half_size,
            )
            canvas_crop = canvas[
                obj_min_y : obj_min_y + crop_size, obj_min_x : obj_min_x + crop_size
            ]
            # crop full gt image according to canvas and nem center of segmentation summation
            gt_crop_min_y_raw = qa_crop_original_center_y - canvas_half_size + obj_min_y
            gt_crop_max_y_raw = gt_crop_min_y_raw + crop_size
            gt_crop_min_x_raw = qa_crop_original_center_x - canvas_half_size + obj_min_x
            gt_crop_max_x_raw = gt_crop_min_x_raw + crop_size

            # Mutable coordinates used only for extracting GT crop (with padding).
            gt_crop_min_y = gt_crop_min_y_raw
            gt_crop_max_y = gt_crop_max_y_raw
            gt_crop_min_x = gt_crop_min_x_raw
            gt_crop_max_x = gt_crop_max_x_raw

            gt_temp = gt_image.copy()

            if gt_crop_min_y < 0:
                y_inc = -gt_crop_min_y
                gt_temp = np.vstack((np.zeros((y_inc, gt_temp.shape[1])), gt_temp))
                gt_crop_max_y += y_inc
                gt_crop_min_y = 0
            if gt_crop_min_x < 0:
                x_inc = -gt_crop_min_x
                gt_temp = np.hstack((np.zeros((gt_temp.shape[0], x_inc)), gt_temp))
                gt_crop_max_x += x_inc
                gt_crop_min_x = 0
            if gt_image.shape[0] < gt_crop_max_y:
                gt_temp = np.vstack(
                    (
                        gt_temp,
                        np.zeros((gt_crop_max_y - gt_temp.shape[0], gt_temp.shape[1])),
                    )
                )
            if gt_image.shape[1] < gt_crop_max_x:
                gt_temp = np.hstack(
                    (
                        gt_temp,
                        np.zeros((gt_temp.shape[0], gt_crop_max_x - gt_temp.shape[1])),
                    )
                )

            gt_crop = gt_temp[gt_crop_min_y:gt_crop_max_y, gt_crop_min_x:gt_crop_max_x]
            gt_crop = (gt_crop == label).astype(np.uint8) * 255

            # stack layers
            # apply_blue_layer:
            empty_blue_layer = np.zeros((crop_size, crop_size), dtype=np.uint8)
            stacked_crop = np.stack([canvas_crop, gt_crop, empty_blue_layer], axis=0)

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
                    "campaign_number": first_row["campaign_number"],
                    "image_id": img_id,
                    "original_image_key": first_row["original_image_key"],
                    "label": label,
                    "crop_size": crop_size,
                    "image_path": new_image_path,
                    "gt_image": gt_image_path,
                    "recon_crop_y_start": int(gt_crop_min_y_raw),
                    "recon_crop_y_end": int(gt_crop_max_y_raw),
                    "recon_crop_x_start": int(gt_crop_min_x_raw),
                    "recon_crop_x_end": int(gt_crop_max_x_raw),
                    "qa_crop_y_start": int(first_row["crop_y_start"]),
                    "qa_crop_y_end": int(first_row["crop_y_end"]),
                    "qa_crop_x_start": int(first_row["crop_x_start"]),
                    "qa_crop_x_end": int(first_row["crop_x_end"]),
                    "original_center_y": int(qa_crop_original_center_y),
                    "original_center_x": int(qa_crop_original_center_x),
                    "qa_centering": first_row.get("centering", None),
                    "qa_center_agreement_count": first_row.get(
                        "center_agreement_count", None
                    ),
                    SPLIT_COL: first_row[SPLIT_COL],
                    "qa_jaccard_avg": df_same_cell[build_opt["qa"]].mean()
                    if build_opt["qa"]
                    else 0,
                    "qa_jaccard_min": df_same_cell[build_opt["qa"]].min()
                    if build_opt["qa"]
                    else 0,
                }
            )

    print(
        f"Number of removed segmentations: {num_removed_segs} ({((num_removed_segs/len(df))*100):.2f}%)"
    )

    # convert list to dataframe
    output_df = pd.DataFrame(data_list)
    # build output parquet path
    parquet_output_path = os.path.join(output_path, f"{databank_foldername}.parquet")
    # save to parquet file
    output_df.to_parquet(parquet_output_path)

    # compress images
    ext.compress_images(images_output_path)

    return parquet_output_path
