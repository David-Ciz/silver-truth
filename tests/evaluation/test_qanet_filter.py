from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from silver_truth.evaluation.qanet_filter import (
    QANetFilterConfig,
    score_with_qanet_filter,
    train_qanet_filter,
)
from silver_truth.evaluation.ranker_pruned_fusion import (
    RankerPruningConfig,
    build_ranker_pruned_candidate_table,
)


def _candidate_rows(tmp_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(10):
        split = "train" if idx < 6 else "validation" if idx < 8 else "test"
        source = "A" if idx % 2 == 0 else "B"
        quality = 0.2 + (0.7 * (idx % 2))
        stack_path = tmp_path / f"candidate_{idx}.tif"
        _write_stacked_crop(stack_path, high_quality=idx % 2 == 1)
        rows.append(
            {
                "cell_id": f"c{idx}",
                "original_image_key": f"img_{idx // 2}",
                "campaign_number": 1,
                "competitor": source,
                "label": idx // 2,
                "crop_y_start": 0,
                "crop_y_end": 16,
                "crop_x_start": 0,
                "crop_x_end": 16,
                "original_center_y": float(idx // 2),
                "original_center_x": float(idx // 2),
                "center_agreement_count": 2,
                "split": split,
                "stacked_path": str(stack_path),
                "jaccard_score": quality,
                "f1_score": quality,
                "predicted_jaccard_index": quality,
            }
        )
    return rows


def _write_stacked_crop(path: Path, *, high_quality: bool) -> None:
    raw = np.zeros((16, 16), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.uint8)
    if high_quality:
        raw[4:12, 4:12] = 1.0
        mask[4:12, 4:12] = 1
    else:
        raw[2:8, 2:8] = 0.5
        mask[1:15, 1:15] = 1
    tifffile.imwrite(path, np.stack([raw, mask]).astype(np.float32))


def test_qanet_filter_trains_scores_and_writes_payload(tmp_path: Path) -> None:
    candidates = pd.DataFrame(_candidate_rows(tmp_path))
    model_path = tmp_path / "qanet.joblib"

    payload = train_qanet_filter(
        candidates,
        output_model=model_path,
        config=QANetFilterConfig(
            crop_size=16,
            batch_size=2,
            max_epochs=2,
            use_source_identity=True,
            device="cpu",
        ),
    )
    scored = score_with_qanet_filter(candidates, model_payload=payload)

    assert model_path.exists()
    assert payload["train_rows"] == 6
    assert payload["source_vocab"]
    assert scored["qanet_score"].between(0.0, 1.0).all()
    assert "qanet_rank_in_image" in scored.columns
    assert "qanet_rank_in_cluster" in scored.columns
    assert "ranker_cluster_id" in scored.columns


def test_qanet_score_can_drive_ranker_pruning(tmp_path: Path) -> None:
    candidates = pd.DataFrame(_candidate_rows(tmp_path))
    payload = train_qanet_filter(
        candidates,
        config=QANetFilterConfig(
            crop_size=16,
            batch_size=2,
            max_epochs=1,
            device="cpu",
        ),
    )
    scored = score_with_qanet_filter(candidates, model_payload=payload)

    filtered, stats = build_ranker_pruned_candidate_table(
        candidates,
        scored,
        config=RankerPruningConfig(
            policy="source_balanced_topk",
            score_column="qanet_score",
            top_k=1,
            source_top_k=1,
        ),
    )

    assert not filtered.empty
    assert stats.loc[0, "score_column"] == "qanet_score"
    assert filtered["_ranker_pruning_score"].between(0.0, 1.0).all()
