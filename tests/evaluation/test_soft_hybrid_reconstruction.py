from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from silver_truth.evaluation.soft_hybrid_reconstruction import (
    reconstruct_selected_soft_hybrid_candidates,
    selected_candidates_for_reconstruction,
)


def test_selected_candidates_for_reconstruction_joins_scores() -> None:
    candidates = pd.DataFrame(
        [
            {"cell_id": "a", "stacked_path": "a.tif", "label": 1},
            {"cell_id": "b", "stacked_path": "b.tif", "label": 2},
        ]
    )
    selected = pd.DataFrame(
        [
            {"candidate_id": "b", "soft_hybrid_score": 0.9},
        ]
    )

    rows = selected_candidates_for_reconstruction(candidates, selected)

    assert rows["cell_id"].tolist() == ["b"]
    assert rows["soft_hybrid_score"].tolist() == [0.9]


def test_reconstruct_selected_soft_hybrid_candidates_writes_eval_csv(
    tmp_path: Path,
) -> None:
    gt = np.zeros((12, 12), dtype=np.uint16)
    gt[2:7, 3:8] = 1
    gt_path = tmp_path / "gt.tif"
    tifffile.imwrite(gt_path, gt)

    stacked = np.zeros((2, 6, 6), dtype=np.uint8)
    stacked[1, :5, :5] = 255
    crop_path = tmp_path / "crop.tif"
    tifffile.imwrite(crop_path, stacked)

    candidates = pd.DataFrame(
        [
            {
                "cell_id": "candidate-1",
                "stacked_path": str(crop_path),
                "original_image_key": "t001",
                "campaign_number": "02",
                "gt_image": str(gt_path),
                "split": "test",
                "label": 1,
                "crop_y_start": 2,
                "crop_y_end": 8,
                "crop_x_start": 3,
                "crop_x_end": 9,
            }
        ]
    )
    candidate_path = tmp_path / "paper_ready.parquet"
    candidates.to_parquet(candidate_path, index=False)

    selected = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-1",
                "soft_hybrid_score": 0.95,
            }
        ]
    )
    selected_path = tmp_path / "selected.csv"
    selected.to_csv(selected_path, index=False)

    output_dir = tmp_path / "soft_hybrid"
    eval_df = reconstruct_selected_soft_hybrid_candidates(
        candidate_parquet=candidate_path,
        selected_candidates_path=selected_path,
        output_dir=output_dir,
    )

    assert (output_dir / "fullimage_eval.csv").exists()
    assert (output_dir / "fullimage" / "02_t001_reconstructed.tif").exists()
    assert eval_df["iou"].iloc[0] == 1.0
    assert eval_df["f1"].iloc[0] == 1.0
    assert eval_df["cells_placed"].iloc[0] == 1
