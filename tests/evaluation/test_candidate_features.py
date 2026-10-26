from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from silver_truth.evaluation.candidate_features import (
    build_candidate_features,
    canonicalize_candidate_source,
    write_candidate_feature_table,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_id": "c02_t001_A_1",
                "original_image_key": "t001",
                "campaign_number": "02",
                "competitor": "A",
                "label": 1,
                "crop_y_start": 10,
                "crop_y_end": 20,
                "crop_x_start": 30,
                "crop_x_end": 40,
                "original_center_y": 15,
                "original_center_x": 35,
                "center_agreement_count": 3,
                "split": "train",
                "time_frame": 1,
                "jaccard_score": 0.8,
                "f1_score": 0.9,
                "predicted_jaccard_index": 0.7,
            },
            {
                "cell_id": "c02_t001_B_1",
                "original_image_key": "t001",
                "campaign_number": "02",
                "competitor": "B",
                "label": 1,
                "crop_y_start": 12,
                "crop_y_end": 22,
                "crop_x_start": 32,
                "crop_x_end": 42,
                "original_center_y": 17,
                "original_center_x": 37,
                "center_agreement_count": 5,
                "split": "train",
                "time_frame": 1,
                "jaccard_score": 0.6,
                "f1_score": 0.75,
                "predicted_jaccard_index": 0.9,
            },
            {
                "cell_id": "c02_t002_A_1",
                "original_image_key": "t002",
                "campaign_number": "02",
                "competitor": "A",
                "label": 1,
                "crop_y_start": 11,
                "crop_y_end": 21,
                "crop_x_start": 31,
                "crop_x_end": 41,
                "original_center_y": 16,
                "original_center_x": 36,
                "center_agreement_count": 4,
                "split": "test",
                "time_frame": 2,
                "jaccard_score": 0.85,
                "f1_score": 0.92,
                "predicted_jaccard_index": 0.8,
            },
        ]
    )


def test_build_candidate_features_adds_ranks_priors_and_temporal_distances() -> None:
    features = build_candidate_features(_candidate_rows())

    assert set(features["candidate_id"]) == {
        "c02_t001_A_1",
        "c02_t001_B_1",
        "c02_t002_A_1",
    }
    assert features.loc[
        features["candidate_id"] == "c02_t001_B_1", "qa_rank_in_image"
    ].iloc[0] == 1.0
    assert features.loc[
        features["candidate_id"] == "c02_t001_B_1",
        "fusion_support_rank_in_image",
    ].iloc[0] == 1.0
    assert features.loc[
        features["candidate_id"] == "c02_t002_A_1", "source_train_mean_iou"
    ].iloc[0] == 0.8
    assert features.loc[
        features["candidate_id"] == "c02_t002_A_1",
        "temporal_prev_min_center_dist",
    ].iloc[0] > 0
    assert features.loc[
        features["candidate_id"] == "c02_t002_A_1",
        "source_temporal_prev_min_center_dist",
    ].iloc[0] > 0
    assert features.loc[
        features["candidate_id"] == "c02_t002_A_1",
        "source_temporal_prev_support_score",
    ].iloc[0] > 0
    assert features.loc[
        features["candidate_id"] == "c02_t002_A_1",
        "temporal_support_score",
    ].iloc[0] > 0


def test_build_candidate_features_canonicalizes_known_source_aliases() -> None:
    rows = _candidate_rows()
    rows.loc[0, "competitor"] = "BGU-IL (5)"
    rows.loc[1, "competitor"] = "BGU-IL-5"

    features = build_candidate_features(rows)

    assert canonicalize_candidate_source("BGU-IL (5)") == "BGU-IL-5"
    assert features["candidate_source_raw"].tolist()[:2] == ["BGU-IL (5)", "BGU-IL-5"]
    assert features["candidate_source"].tolist()[:2] == ["BGU-IL-5", "BGU-IL-5"]


def test_write_candidate_feature_table_parses_paper_input_metadata(
    tmp_path: Path,
) -> None:
    input_path = (
        tmp_path
        / "paper_inputs"
        / "Dataset"
        / "sz64"
        / "variant"
        / "fold-1_paper_ready.parquet"
    )
    input_path.parent.mkdir(parents=True)
    _candidate_rows().to_parquet(input_path, index=False)

    output_path = tmp_path / "features.csv"
    features = write_candidate_feature_table([input_path], output_path)
    persisted = pd.read_csv(output_path)

    assert output_path.exists()
    assert features["dataset"].unique().tolist() == ["Dataset"]
    assert features["crop_tag"].unique().tolist() == ["sz64"]
    assert features["variant"].unique().tolist() == ["variant"]
    assert features["fold"].unique().tolist() == ["fold-1"]
    assert len(persisted) == 3


def test_build_candidate_features_can_add_image_mask_features(tmp_path: Path) -> None:
    rows = _candidate_rows()
    crop = np.zeros((4, 8, 8), dtype=np.uint16)
    crop[0, 2:6, 2:6] = 100
    crop[0, 1:7, 1:7] += 10
    crop[1, 3:5, 3:6] = 255
    crop_path = tmp_path / "crop.tif"
    tifffile.imwrite(crop_path, crop, photometric="minisblack")
    rows["stacked_path"] = str(crop_path)

    features = build_candidate_features(rows, include_image_features=True)

    assert features["mask_area_pixels"].iloc[0] == 6.0
    assert features["mask_component_count"].iloc[0] == 1.0
    assert features["mask_extent"].iloc[0] == 1.0
    assert features["raw_contrast"].iloc[0] > 0


def test_build_candidate_features_can_add_rect_image_mask_features(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    crop = np.zeros((4, 80, 80), dtype=np.uint16)
    crop[0, 22:58, 22:58] = 10
    crop[0, 35:38, 36:40] = 100
    crop[1, 35:38, 36:40] = 255
    crop_path = tmp_path / "crop.tif"
    tifffile.imwrite(crop_path, crop, photometric="minisblack")
    rows["stacked_path"] = str(crop_path)

    features = build_candidate_features(rows, include_rect_image_features=True)

    assert features["mask_area_pixels"].isna().all()
    assert features["rect_mask_area_pixels"].iloc[0] == 12.0
    assert features["rect_mask_component_count"].iloc[0] == 1.0
    assert features["rect_mask_bbox_aspect_ratio"].iloc[0] == 4.0 / 3.0
    assert features["rect_mask_area_fraction"].iloc[0] > 12.0 / (80.0 * 80.0)
    assert features["rect_raw_contrast"].iloc[0] > 0


def test_build_candidate_features_can_add_multiscale_rect_features(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    crop = np.zeros((4, 80, 80), dtype=np.uint16)
    crop[0, 20:60, 20:60] = 10
    crop[0, 35:38, 36:40] = 100
    crop[1, 35:38, 36:40] = 255
    crop_path = tmp_path / "crop.tif"
    tifffile.imwrite(crop_path, crop, photometric="minisblack")
    rows["stacked_path"] = str(crop_path)

    features = build_candidate_features(
        rows, include_multiscale_rect_image_features=True
    )

    assert features["tight_mask_area_pixels"].iloc[0] == 12.0
    assert features["tight_mask_area_fraction"].iloc[0] == 1.0
    assert features["far_mask_area_pixels"].iloc[0] == 12.0
    assert features["far_mask_area_fraction"].iloc[0] < 1.0
    assert features["far_raw_contrast"].iloc[0] > 0


def test_build_candidate_features_can_add_crop_geometry_features(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, np.zeros((20, 30), dtype=np.uint16))
    crop = np.zeros((4, 10, 10), dtype=np.uint16)
    crop[1, 0:3, 0:4] = 255
    crop_path = tmp_path / "crop.tif"
    tifffile.imwrite(crop_path, crop, photometric="minisblack")
    rows["original_image_path"] = str(raw_path)
    rows["stacked_path"] = str(crop_path)
    rows.loc[:, "crop_y_start"] = -2
    rows.loc[:, "crop_y_end"] = 8
    rows.loc[:, "crop_x_start"] = -1
    rows.loc[:, "crop_x_end"] = 9

    features = build_candidate_features(rows, include_crop_geometry_features=True)

    assert features["crop_overflow_top"].iloc[0] == 2.0
    assert features["crop_overflow_left"].iloc[0] == 1.0
    assert features["crop_visible_height"].iloc[0] == 8.0
    assert features["crop_visible_width"].iloc[0] == 9.0
    assert features["crop_visible_fraction"].iloc[0] == 0.72
    assert features["crop_has_padding"].iloc[0] == 1.0
    assert features["mask_touching_square_crop_border"].iloc[0] == 1.0
    assert features["mask_touching_image_border"].iloc[0] == 1.0


def test_build_candidate_features_can_add_full_image_features(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    raw = np.zeros((20, 20), dtype=np.uint16)
    raw[1:8, 1:8] = 10
    raw[2:5, 2:6] = 100
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, raw)
    crop = np.zeros((4, 10, 10), dtype=np.uint16)
    crop[1, 4:7, 4:8] = 255
    crop_path = tmp_path / "crop.tif"
    tifffile.imwrite(crop_path, crop, photometric="minisblack")
    rows["original_image_path"] = str(raw_path)
    rows["stacked_path"] = str(crop_path)
    rows.loc[:, "crop_y_start"] = -2
    rows.loc[:, "crop_y_end"] = 8
    rows.loc[:, "crop_x_start"] = -2
    rows.loc[:, "crop_x_end"] = 8

    features = build_candidate_features(rows, include_full_image_features=True)

    assert features["full_mask_area_pixels"].iloc[0] == 12.0
    assert features["full_mask_retained_fraction"].iloc[0] == 1.0
    assert features["full_mask_bbox_aspect_ratio"].iloc[0] == 4.0 / 3.0
    assert features["full_raw_contrast"].iloc[0] > 0
    assert features["full_raw_background_std"].isna().all()


def test_build_candidate_features_can_add_full_image_boundary_noise_features(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    raw = np.zeros((20, 20), dtype=np.uint16)
    raw[1:8, 1:8] = 10
    raw[2:5, 2:6] = 100
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, raw)
    crop = np.zeros((4, 10, 10), dtype=np.uint16)
    crop[1, 4:7, 4:8] = 255
    crop_path = tmp_path / "crop.tif"
    tifffile.imwrite(crop_path, crop, photometric="minisblack")
    rows["original_image_path"] = str(raw_path)
    rows["stacked_path"] = str(crop_path)
    rows.loc[:, "crop_y_start"] = -2
    rows.loc[:, "crop_y_end"] = 8
    rows.loc[:, "crop_x_start"] = -2
    rows.loc[:, "crop_x_end"] = 8

    features = build_candidate_features(
        rows, include_full_image_boundary_noise_features=True
    )

    assert features["full_raw_background_std"].iloc[0] > 0
    assert features["full_raw_contrast_to_noise"].iloc[0] > 0
    assert features["full_raw_inside_p25"].iloc[0] == 100.0
    assert features["full_raw_inside_p75"].iloc[0] == 100.0
    assert features["full_raw_boundary_gradient_mean"].iloc[0] > 0
    assert features["full_mask_hole_fraction"].iloc[0] == 0.0
    assert np.isfinite(features["full_mask_eccentricity"].iloc[0])


def test_build_candidate_features_adds_full_image_hole_fraction(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    raw = np.full((30, 30), 10, dtype=np.uint16)
    raw[10:20, 10:20] = 100
    raw_path = tmp_path / "raw.tif"
    tifffile.imwrite(raw_path, raw)
    crop = np.zeros((4, 20, 20), dtype=np.uint16)
    crop[1, 5:15, 5:15] = 255
    crop[1, 8:12, 8:12] = 0
    crop_path = tmp_path / "crop.tif"
    tifffile.imwrite(crop_path, crop, photometric="minisblack")
    rows["original_image_path"] = str(raw_path)
    rows["stacked_path"] = str(crop_path)
    rows.loc[:, "crop_y_start"] = 5
    rows.loc[:, "crop_y_end"] = 25
    rows.loc[:, "crop_x_start"] = 5
    rows.loc[:, "crop_x_end"] = 25

    features = build_candidate_features(
        rows, include_full_image_boundary_noise_features=True
    )

    assert features["full_mask_area_pixels"].iloc[0] == 84.0
    assert features["full_mask_hole_fraction"].iloc[0] == 16.0 / 84.0
