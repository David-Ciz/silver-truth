from pathlib import Path

import pandas as pd
import pytest

from scripts.build_source_aware_boundary_floor_features import add_source_aware_floor


def test_source_aware_boundary_floor_assigns_by_calibrated_source_quality(
    tmp_path: Path,
) -> None:
    feature_root = tmp_path / "features"
    source_path = feature_root / "dataset" / "fold-1_paper_ready_pairwise_ranked.parquet"
    source_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_source": "weak",
                "true_iou": 0.20,
                "split": "train",
                "candidate_id": "weak-train",
            },
            {
                "candidate_source": "weak",
                "true_iou": 0.40,
                "split": "validation",
                "candidate_id": "weak-val",
            },
            {
                "candidate_source": "strong",
                "true_iou": 0.80,
                "split": "train",
                "candidate_id": "strong-train",
            },
            {
                "candidate_source": "strong",
                "true_iou": 1.00,
                "split": "validation",
                "candidate_id": "strong-val",
            },
            {
                "candidate_source": "weak",
                "true_iou": 1.00,
                "split": "test",
                "candidate_id": "weak-test-not-calibration",
            },
        ]
    ).to_parquet(source_path, index=False)

    output_path = add_source_aware_floor(
        source_path,
        output_root=tmp_path / "augmented",
        relative_root=feature_root,
        source_column="candidate_source",
        target_column="true_iou",
        split_column="split",
        calibration_splits=("train", "validation"),
        floor_column="object_vote_boundary_floor",
        low_quality_floor=0.25,
        high_quality_floor=0.90,
        quality_quantile=0.50,
        invert=False,
    )

    augmented = pd.read_parquet(output_path)
    floors = augmented.groupby("candidate_source")["object_vote_boundary_floor"].first()
    qualities = augmented.groupby("candidate_source")[
        "object_vote_boundary_floor_source_quality"
    ].first()

    assert output_path == (
        tmp_path / "augmented" / "dataset" / "fold-1_paper_ready_pairwise_ranked.parquet"
    )
    assert floors.to_dict() == {"strong": 0.90, "weak": 0.25}
    assert qualities.to_dict() == pytest.approx({"strong": 0.90, "weak": 0.30})
    assert float(
        augmented["object_vote_boundary_floor_quality_threshold"].iloc[0]
    ) == pytest.approx(0.60)


def test_source_aware_boundary_floor_invert_flips_floor_assignment(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "features.parquet"
    pd.DataFrame(
        [
            {"candidate_source": "weak", "true_iou": 0.30},
            {"candidate_source": "strong", "true_iou": 0.90},
        ]
    ).to_parquet(source_path, index=False)

    output_path = add_source_aware_floor(
        source_path,
        output_root=tmp_path / "augmented",
        relative_root=None,
        source_column="candidate_source",
        target_column="true_iou",
        split_column="split",
        calibration_splits=("train", "validation"),
        floor_column="object_vote_boundary_floor",
        low_quality_floor=0.25,
        high_quality_floor=0.90,
        quality_quantile=0.50,
        invert=True,
    )

    augmented = pd.read_parquet(output_path)
    floors = augmented.groupby("candidate_source")["object_vote_boundary_floor"].first()

    assert output_path == tmp_path / "augmented" / "features.parquet"
    assert floors.to_dict() == {"strong": 0.25, "weak": 0.90}
