from typing import Optional, Any

import pandas as pd
import tifffile
import numpy as np
from pathlib import Path
from scipy.ndimage import find_objects, center_of_mass
from tqdm import tqdm
import logging

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)

# Basic Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_qa_dataset(
    dataset_dataframe_path: str,
    output_dir: str,
    output_parquet_path: str,
    crop: bool = True,
    crop_size: int = 64,
    centering: str = "competitor",
    exclude_competitors: Optional[list[Any]] = None,
):
    """
    Generates a QA dataset from images that have corresponding Ground Truth annotations.
    If crop=True, it creates stacked cell crops.
    If crop=False, it creates stacked full images with individual cell masks.

    Only processes images that have valid Ground Truth files (gt_image column is not null
    and the GT file exists on disk).

    Args:
        dataset_dataframe_path: Path to the main Parquet file that tracks all the
            images and competitors.
        output_dir: The directory where the stacked images will be saved.
        output_parquet_path: The path to save the resulting dataframe.parquet file.
        crop: Whether to crop individual cells or use the full image.
        crop_size: If cropping, the size of the bounding box for the cell crops.
        centering: Strategy for centering crops. Supported values:
            - competitor: alias of competitor_consensus (default).
            - competitor_consensus: center using agreement across all competitors.
            - competitor_individual: center independently per competitor crop.
            - gt_mask: center from GT mask.
            - tracking_marker: center from tracking marker mask (TRA), fallback to GT.
        exclude_competitors: List of competitor names to exclude from the dataset.
    """
    logging.info("Starting QA dataset creation.")
    logging.info(f"  - Dataset dataframe path: {dataset_dataframe_path}")
    logging.info(f"  - Output directory: {output_dir}")
    logging.info(f"  - Output parquet path: {output_parquet_path}")
    logging.info(f"  - Crop: {crop}")
    centering_alias = {"competitor": "competitor_consensus"}
    centering_mode = centering_alias.get(centering, centering)
    supported_centering = {
        "competitor_consensus",
        "competitor_individual",
        "gt_mask",
        "tracking_marker",
    }
    if centering_mode not in supported_centering:
        raise ValueError(
            f"Unsupported centering='{centering}'. "
            f"Use one of: {sorted(list(supported_centering | set(centering_alias.keys())))}"
        )

    if crop:
        logging.info(f"  - Crop size: {crop_size}")
        logging.info(f"  - Centering: {centering_mode}")

    df = load_dataframe_from_parquet_with_metadata(dataset_dataframe_path)
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_path}")

    # Find project root (directory containing 'data' folder) for relative path conversion
    project_root = None
    for parent in output_path.parents:
        if (parent / "data").exists():
            project_root = parent
            break

    def to_relative_path(abs_path):
        """Convert absolute path to relative path starting from project root."""
        if abs_path is None:
            return None
        p = Path(abs_path)
        if project_root and p.is_absolute():
            try:
                return str(p.relative_to(project_root))
            except ValueError:
                return str(p)
        return str(p)

    data_list = []
    competitor_columns = df.attrs.get("competitor_columns", [])

    # If no competitor_columns in attrs, infer them from columns
    if not competitor_columns:
        # Exclude standard columns to get competitor columns
        excluded_columns = [
            "composite_key",
            "campaign_number",
            "gt_image",
            "tracking_markers",
            "source_image",
            "tracking_marker",  # Typo in some datasets
        ]
        competitor_columns = [col for col in df.columns if col not in excluded_columns]
        logging.info(f"Inferred competitor columns: {competitor_columns}")

    if exclude_competitors:
        competitor_columns = [
            c for c in competitor_columns if c not in exclude_competitors
        ]
        logging.info(f"Columns after exclusion: {competitor_columns}")

    logging.info(f"Using competitor columns: {competitor_columns}")

    # Filter for rows that have ground truth images
    initial_count = len(df)

    # First filter: check if gt_image column has non-null values
    df_with_gt = df[df["gt_image"].notna()].copy()

    # Second filter: check if GT files actually exist on disk
    gt_exists_mask = df_with_gt["gt_image"].apply(
        lambda gt_path: Path(gt_path).exists() if gt_path else False
    )
    df_filtered = df_with_gt[gt_exists_mask].copy()

    filtered_count = len(df_filtered)
    logging.info(
        f"Filtered dataset: {initial_count} -> {filtered_count} images (only those with existing Ground Truth)"
    )

    if filtered_count == 0:
        logging.warning("No images with valid Ground Truth found. Exiting.")
        return

    def _center_from_binary_mask(mask: np.ndarray) -> Optional[tuple[int, int]]:
        if mask.size == 0 or not np.any(mask):
            return None

        try:
            cy, cx = center_of_mass(mask)
            if np.isnan(cy) or np.isnan(cx):
                raise ValueError("center_of_mass returned NaN")
            return int(round(float(cy))), int(round(float(cx)))
        except Exception:
            objects = find_objects(mask.astype(np.uint8))
            if not objects or objects[0] is None:
                return None
            slice_y, slice_x = objects[0]
            return (slice_y.start + slice_y.stop) // 2, (
                slice_x.start + slice_x.stop
            ) // 2

    def _build_raw_image_path(
        gt_path: Path, campaign: str, frame: str
    ) -> Optional[Path]:
        try:
            gt_parts = gt_path.parts
            synchronized_data_idx = gt_parts.index("synchronized_data")
            base_path = Path(*gt_parts[: synchronized_data_idx + 2])
        except ValueError:
            logging.error(
                "Path %s does not contain 'synchronized_data'. Cannot infer raw image path.",
                gt_path,
            )
            return None

        frame_int = int(frame)
        candidates = [
            base_path / campaign / f"t{frame_int:03d}.tif",
            base_path / campaign / f"t{frame_int:04d}.tif",
            base_path / campaign / f"t{frame}.tif",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_tracking_image(
        row_data: pd.Series, gt_path: Path, gt_image: np.ndarray
    ) -> np.ndarray:
        if "tracking_markers" in row_data:
            marker_path = row_data["tracking_markers"]
            if marker_path and not pd.isna(marker_path) and Path(marker_path).exists():
                return tifffile.imread(marker_path)

        marker_inferred = (
            str(gt_path).replace("SEG", "TRA").replace("man_seg", "man_track")
        )
        if Path(marker_inferred).exists():
            return tifffile.imread(marker_inferred)

        logging.warning(
            "Tracking marker missing for %s. Using empty TRA mask.",
            row_data.get("composite_key"),
        )
        return np.zeros_like(gt_image)

    # Use tqdm to create a progress bar for the main loop
    for __, row in tqdm(
        df_filtered.iterrows(), total=df_filtered.shape[0], desc="Processing images"
    ):
        # Construct raw image path from composite_key
        composite_key = row["composite_key"]  # e.g., "01_0061.tif"
        campaign_number = row["campaign_number"]  # e.g., "01"

        # Extract time frame number from composite_key (e.g., "0061" from "01_0061.tif")
        time_frame = composite_key.split("_")[1].split(".")[0]  # "0061"

        gt_image_path = Path(row["gt_image"])
        raw_image_path = _build_raw_image_path(
            gt_image_path, campaign_number, time_frame
        )

        if raw_image_path is None:
            logger.error(
                "Raw image file not found for time frame %s (campaign=%s, gt=%s)",
                time_frame,
                campaign_number,
                gt_image_path,
            )
            continue

        try:
            raw_image = tifffile.imread(raw_image_path)
            gt_image = tifffile.imread(gt_image_path)
            tra_image = _load_tracking_image(row, gt_image_path, gt_image)
        except Exception as e:
            logging.error("Error reading raw/gt/tra for %s: %s", raw_image_path.stem, e)
            continue

        gt_labels = np.unique(gt_image)[1:]
        gt_label_set = set(int(value) for value in gt_labels.tolist())

        segmentations_by_competitor: dict[str, np.ndarray] = {}
        labels_by_competitor: dict[str, set[int]] = {}
        for competitor in competitor_columns:
            segmentation_path_str = row[competitor]
            if not segmentation_path_str or not Path(segmentation_path_str).exists():
                logging.warning(
                    "Segmentation file not found for %s and competitor %s. Skipping.",
                    raw_image_path.stem,
                    competitor,
                )
                continue

            try:
                segmentation = tifffile.imread(segmentation_path_str)
            except Exception as e:
                logging.error(
                    "Error reading segmentation for %s and competitor %s: %s",
                    raw_image_path.stem,
                    competitor,
                    e,
                )
                continue

            segmentation_labels = np.unique(segmentation)[1:]
            segmentation_label_set = set(
                int(value) for value in segmentation_labels.tolist()
            )
            extra_labels = sorted(segmentation_label_set - gt_label_set)
            if extra_labels:
                logging.warning(
                    "Segmentation %s contains labels not in GT: %s. These are ignored.",
                    Path(segmentation_path_str).name,
                    extra_labels,
                )

            labels_by_competitor[competitor] = segmentation_label_set & gt_label_set
            segmentations_by_competitor[competitor] = segmentation

        if not segmentations_by_competitor:
            continue

        shared_centers: dict[int, Optional[tuple[int, int]]] = {}
        center_agreement_count: dict[int, int] = {}
        if crop and centering_mode in {
            "competitor_consensus",
            "gt_mask",
            "tracking_marker",
        }:
            labels_with_any_seg = set()
            for label_set in labels_by_competitor.values():
                labels_with_any_seg.update(label_set)

            for label in sorted(labels_with_any_seg):
                center_value: Optional[tuple[int, int]] = None
                center_count = 0

                if centering_mode == "gt_mask":
                    center_value = _center_from_binary_mask(
                        (gt_image == label).astype(np.uint8)
                    )
                    center_count = 1 if center_value is not None else 0
                elif centering_mode == "tracking_marker":
                    center_value = _center_from_binary_mask(
                        (tra_image == label).astype(np.uint8)
                    )
                    if center_value is None:
                        center_value = _center_from_binary_mask(
                            (gt_image == label).astype(np.uint8)
                        )
                    center_count = 1 if center_value is not None else 0
                else:  # competitor_consensus
                    centers = []
                    for (
                        competitor_name,
                        segmentation,
                    ) in segmentations_by_competitor.items():
                        if label not in labels_by_competitor.get(
                            competitor_name, set()
                        ):
                            continue
                        center = _center_from_binary_mask(
                            (segmentation == label).astype(np.uint8)
                        )
                        if center is not None:
                            centers.append(center)

                    if centers:
                        ys = [value[0] for value in centers]
                        xs = [value[1] for value in centers]
                        center_value = (
                            int(round(float(np.median(ys)))),
                            int(round(float(np.median(xs)))),
                        )
                        center_count = len(centers)
                    else:
                        center_value = _center_from_binary_mask(
                            (gt_image == label).astype(np.uint8)
                        )
                        center_count = 1 if center_value is not None else 0

                shared_centers[label] = center_value
                center_agreement_count[label] = center_count

        for competitor, segmentation in segmentations_by_competitor.items():
            labels_to_process = sorted(labels_by_competitor.get(competitor, set()))
            if not labels_to_process:
                continue

            for label in labels_to_process:
                if crop:
                    if centering_mode == "competitor_individual":
                        center_coords = _center_from_binary_mask(
                            (segmentation == label).astype(np.uint8)
                        )
                        agreement_count = 1 if center_coords is not None else 0
                    else:
                        center_coords = shared_centers.get(label)
                        agreement_count = center_agreement_count.get(label, 0)

                    if center_coords is None:
                        logging.warning(
                            "Could not determine center for label %s in %s with strategy %s.",
                            label,
                            raw_image_path.stem,
                            centering_mode,
                        )
                        continue

                    center_y, center_x = center_coords
                    half_size = crop_size // 2
                    y_start, y_end = center_y - half_size, center_y + half_size
                    x_start, x_end = center_x - half_size, center_x + half_size

                    pad_top = max(0, -y_start)
                    pad_bottom = max(0, y_end - raw_image.shape[0])
                    pad_left = max(0, -x_start)
                    pad_right = max(0, x_end - raw_image.shape[1])

                    img_y_start = max(0, y_start)
                    img_y_end = min(raw_image.shape[0], y_end)
                    img_x_start = max(0, x_start)
                    img_x_end = min(raw_image.shape[1], x_end)

                    raw_crop = raw_image[img_y_start:img_y_end, img_x_start:img_x_end]
                    seg_crop = (
                        segmentation[img_y_start:img_y_end, img_x_start:img_x_end]
                        == label
                    ).astype(np.uint8) * 255
                    gt_crop = (
                        gt_image[img_y_start:img_y_end, img_x_start:img_x_end] == label
                    ).astype(np.uint8) * 255
                    tra_crop = (
                        tra_image[img_y_start:img_y_end, img_x_start:img_x_end] == label
                    ).astype(np.uint8) * 255

                    raw_crop = np.pad(
                        raw_crop,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="constant",
                    )
                    seg_crop = np.pad(
                        seg_crop,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="constant",
                    )
                    gt_crop = np.pad(
                        gt_crop,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="constant",
                    )
                    tra_crop = np.pad(
                        tra_crop,
                        ((pad_top, pad_bottom), (pad_left, pad_right)),
                        mode="constant",
                    )

                    stacked_crop = np.stack(
                        [raw_crop, seg_crop, gt_crop, tra_crop], axis=0
                    )
                    assert stacked_crop.shape == (
                        4,
                        crop_size,
                        crop_size,
                    ), f"Shape mismatch: {stacked_crop.shape}"

                    cell_id = (
                        f"c{campaign_number}_{raw_image_path.stem}_{competitor}_{label}"
                    )
                    stacked_path = output_path / f"{cell_id}.tif"
                    tifffile.imwrite(stacked_path, stacked_crop)

                    data_list.append(
                        {
                            "cell_id": cell_id,
                            "stacked_path": to_relative_path(stacked_path),
                            "original_image_key": raw_image_path.stem,
                            "campaign_number": campaign_number,
                            "competitor": competitor,
                            "label": label,
                            "crop_y_start": y_start,
                            "crop_y_end": y_end,
                            "crop_x_start": x_start,
                            "crop_x_end": x_end,
                            "original_center_y": center_y,
                            "original_center_x": center_x,
                            "center_agreement_count": agreement_count,
                            "crop_size": crop_size,
                            "centering": centering_mode,
                            "original_image_path": to_relative_path(raw_image_path),
                            "gt_image": to_relative_path(row["gt_image"]),
                        }
                    )
                else:
                    cell_mask = (segmentation == label).astype(np.uint8) * 255
                    stacked_image = np.stack([raw_image, cell_mask], axis=0)

                    cell_id = (
                        f"c{campaign_number}_{raw_image_path.stem}_{competitor}_{label}"
                    )
                    stacked_path = output_path / f"{cell_id}.tif"
                    tifffile.imwrite(stacked_path, stacked_image)

                    data_list.append(
                        {
                            "cell_id": cell_id,
                            "stacked_path": to_relative_path(stacked_path),
                            "original_image_key": raw_image_path.stem,
                            "campaign_number": campaign_number,
                            "competitor": competitor,
                            "label": label,
                            "crop_y_start": 0,
                            "crop_y_end": raw_image.shape[0],
                            "crop_x_start": 0,
                            "crop_x_end": raw_image.shape[1],
                            "original_center_y": raw_image.shape[0] // 2,
                            "original_center_x": raw_image.shape[1] // 2,
                            "center_agreement_count": len(segmentations_by_competitor),
                            "crop_size": None,
                            "centering": centering_mode,
                            "original_image_path": to_relative_path(raw_image_path),
                            "gt_image": to_relative_path(row["gt_image"]),
                        }
                    )

    output_df = pd.DataFrame(data_list)
    output_df.to_parquet(output_parquet_path)

    logging.info("QA dataset creation complete.")
    if crop:
        logging.info(
            f"  - {len(output_df)} cropped cell images created in {output_path}"
        )
    else:
        logging.info(
            f"  - {len(output_df)} full-size images with cell masks created in {output_path}"
        )
    logging.info(f"  - QA dataframe saved to {output_parquet_path}")


def attach_split_to_qa_dataset(
    qa_base_parquet_path: str,
    dataset_dataframe_path: str,
    output_parquet_path: str,
) -> None:
    """
    Attach split labels (train/validation/test/...) to an existing QA dataset parquet.

    This enables generating expensive QA crops once, then materializing split-specific
    QA parquet files by joining against different dataset dataframe split variants.

    Join keys are:
    1) campaign_number + gt_image (preferred)
    2) campaign_number + original_image_path/source_image (fallback)
    """
    logging.info("Starting QA split attachment.")
    logging.info(f"  - Base QA parquet: {qa_base_parquet_path}")
    logging.info(f"  - Split dataframe: {dataset_dataframe_path}")
    logging.info(f"  - Output parquet: {output_parquet_path}")

    qa_df = pd.read_parquet(qa_base_parquet_path)
    dataset_df = load_dataframe_from_parquet_with_metadata(dataset_dataframe_path)

    if "split" not in dataset_df.columns:
        raise ValueError(
            f"Dataset dataframe has no 'split' column: {dataset_dataframe_path}"
        )
    if (
        "campaign_number" not in qa_df.columns
        or "campaign_number" not in dataset_df.columns
    ):
        raise ValueError(
            "Both QA and dataset dataframes must contain 'campaign_number'."
        )

    if "gt_image" in qa_df.columns and "gt_image" in dataset_df.columns:
        qa_join_cols = ["campaign_number", "gt_image"]
        dataset_join_cols = ["campaign_number", "gt_image"]
    elif (
        "original_image_path" in qa_df.columns and "source_image" in dataset_df.columns
    ):
        qa_join_cols = ["campaign_number", "original_image_path"]
        dataset_join_cols = ["campaign_number", "source_image"]
    else:
        raise ValueError(
            "Could not determine join keys. Expected either "
            "(campaign_number + gt_image) or "
            "(campaign_number + original_image_path/source_image)."
        )

    split_lookup = (
        dataset_df[dataset_join_cols + ["split"]]
        .dropna(subset=dataset_join_cols + ["split"])
        .copy()
    )

    # Validate split lookup uniqueness to avoid ambiguous assignments.
    split_variants = split_lookup.groupby(dataset_join_cols)["split"].nunique()
    ambiguous = split_variants[split_variants > 1]
    if not ambiguous.empty:
        raise ValueError(
            "Ambiguous split assignment detected for join keys in dataset dataframe."
        )

    split_lookup = split_lookup.drop_duplicates(subset=dataset_join_cols, keep="first")
    qa_without_split = qa_df.drop(columns=["split"], errors="ignore")

    merged = qa_without_split.merge(
        split_lookup,
        left_on=qa_join_cols,
        right_on=dataset_join_cols,
        how="left",
        validate="many_to_one",
    )

    # Drop right-hand join columns when names differ.
    extra_join_cols = [col for col in dataset_join_cols if col not in qa_join_cols]
    if extra_join_cols:
        merged = merged.drop(columns=extra_join_cols)

    missing_split = merged["split"].isna().sum()
    if missing_split:
        sample_rows = merged.loc[merged["split"].isna(), qa_join_cols].head(5)
        raise ValueError(
            f"Failed to assign split for {missing_split} QA row(s). "
            f"Sample unmatched keys: {sample_rows.to_dict(orient='records')}"
        )

    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path)
    logging.info(
        "QA split attachment complete. Rows: %d, output: %s",
        len(merged),
        output_path,
    )
