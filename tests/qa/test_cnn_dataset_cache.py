from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from silver_truth.qa import cnn as qa_cnn


tifffile = pytest.importorskip("tifffile")


def _write_stacked_tiff(path: Path, value: int) -> None:
    stacked = np.zeros((4, 8, 8), dtype=np.uint8)
    stacked[0] = value
    stacked[1] = value + 1
    tifffile.imwrite(path, stacked)


def test_jaccard_dataset_reuses_shared_image_cache(tmp_path: Path, monkeypatch) -> None:
    image_a = tmp_path / "a.tif"
    image_b = tmp_path / "b.tif"
    _write_stacked_tiff(image_a, 10)
    _write_stacked_tiff(image_b, 20)

    parquet_path = tmp_path / "qa.parquet"
    pd.DataFrame(
        [
            {
                "stacked_path": str(image_a),
                "jaccard_score": 0.1,
                "split": "train",
                "cell_id": "cell-a",
            },
            {
                "stacked_path": str(image_b),
                "jaccard_score": 0.2,
                "split": "validation",
                "cell_id": "cell-b",
            },
        ]
    ).to_parquet(parquet_path)

    original_imread = qa_cnn.tifffile.imread
    read_calls: list[str] = []

    def counting_imread(path):
        read_calls.append(str(path))
        return original_imread(path)

    monkeypatch.setattr(qa_cnn.tifffile, "imread", counting_imread)

    shared_cache: dict[str, np.ndarray] = {}
    train_dataset = qa_cnn.JaccardDataset(
        parquet_path,
        image_cache=shared_cache,
        preload_images=True,
    )
    eval_dataset = qa_cnn.JaccardDataset(
        parquet_path,
        image_cache=shared_cache,
    )

    assert len(read_calls) == 2
    assert set(read_calls) == {str(image_a), str(image_b)}
    assert len(shared_cache) == 2

    first_image, first_target = train_dataset[0]
    second_image, second_target = eval_dataset[1]

    assert len(read_calls) == 2
    assert tuple(first_image.shape) == (2, 8, 8)
    assert tuple(second_image.shape) == (2, 8, 8)
    assert first_target.item() == pytest.approx(0.1)
    assert second_target.item() == pytest.approx(0.2)


def test_prepare_images_for_model_normalizes_uint8_batches() -> None:
    images = torch.tensor(
        [[[[0, 255], [128, 64]], [[255, 0], [64, 128]]]], dtype=torch.uint8
    )

    normalized = qa_cnn.prepare_images_for_model(images, augment=False)

    assert normalized.dtype == torch.float32
    assert normalized.shape == images.shape
    assert normalized[0, 0, 0, 0].item() == pytest.approx(-1.0)
    assert normalized[0, 0, 0, 1].item() == pytest.approx(1.0)


def test_jaccard_dataset_supports_variable_input_channels(tmp_path: Path) -> None:
    image = tmp_path / "sample.tif"
    _write_stacked_tiff(image, 7)

    parquet_path = tmp_path / "qa.parquet"
    pd.DataFrame(
        [
            {
                "stacked_path": str(image),
                "jaccard_score": 0.4,
                "split": "train",
                "cell_id": "cell-a",
            }
        ]
    ).to_parquet(parquet_path)

    dataset = qa_cnn.JaccardDataset(parquet_path, input_channels="0,1,3")
    image_tensor, target = dataset[0]

    assert tuple(image_tensor.shape) == (3, 8, 8)
    assert target.item() == pytest.approx(0.4)


def test_grouped_jaccard_dataset_groups_candidates_by_logical_cell(
    tmp_path: Path,
) -> None:
    image_a = tmp_path / "a.tif"
    image_b = tmp_path / "b.tif"
    image_c = tmp_path / "c.tif"
    _write_stacked_tiff(image_a, 1)
    _write_stacked_tiff(image_b, 2)
    _write_stacked_tiff(image_c, 3)

    parquet_path = tmp_path / "qa.parquet"
    pd.DataFrame(
        [
            {
                "stacked_path": str(image_a),
                "jaccard_score": 0.9,
                "split": "train",
                "cell_id": "cell-a1",
                "campaign_number": "01",
                "original_image_key": "img-1",
                "label": 1,
            },
            {
                "stacked_path": str(image_b),
                "jaccard_score": 0.4,
                "split": "train",
                "cell_id": "cell-a2",
                "campaign_number": "01",
                "original_image_key": "img-1",
                "label": 1,
            },
            {
                "stacked_path": str(image_c),
                "jaccard_score": 0.7,
                "split": "train",
                "cell_id": "cell-b1",
                "campaign_number": "01",
                "original_image_key": "img-2",
                "label": 2,
            },
        ]
    ).to_parquet(parquet_path)

    dataset = qa_cnn.JaccardDataset(parquet_path)
    grouped_dataset = qa_cnn.GroupedJaccardDataset(dataset, [0, 1, 2])

    assert len(grouped_dataset) == 2
    group_sizes = sorted(grouped_dataset[idx][0].shape[0] for idx in range(2))
    assert group_sizes == [1, 2]


def test_jaccard_dataset_emits_metadata_features(tmp_path: Path) -> None:
    image = tmp_path / "sample.tif"
    _write_stacked_tiff(image, 32)

    parquet_path = tmp_path / "qa.parquet"
    pd.DataFrame(
        [
            {
                "stacked_path": str(image),
                "jaccard_score": 0.4,
                "split": "train",
                "cell_id": "cell-a",
                "center_agreement_count": 3,
                "crop_size": 8,
            }
        ]
    ).to_parquet(parquet_path)

    dataset = qa_cnn.JaccardDataset(
        parquet_path,
        metadata_features="center_agreement_count,crop_size,mask_area_ratio",
    )
    image_tensor, metadata_tensor, target = dataset[0]

    assert tuple(image_tensor.shape) == (2, 8, 8)
    assert tuple(metadata_tensor.shape) == (3,)
    assert metadata_tensor[0].item() == pytest.approx(1.0)
    assert metadata_tensor[1].item() == pytest.approx(1.0)
    assert metadata_tensor[2].item() == pytest.approx(1.0)
    assert target.item() == pytest.approx(0.4)


def test_pairwise_ranking_loss_penalizes_wrong_ordering() -> None:
    module = object.__new__(qa_cnn.JaccardLightningModule)

    targets = torch.tensor([0.9, 0.2], dtype=torch.float32)
    correct_preds = torch.tensor([0.8, 0.1], dtype=torch.float32)
    wrong_preds = torch.tensor([0.1, 0.8], dtype=torch.float32)
    group_sizes = torch.tensor([2], dtype=torch.long)

    correct_loss = module._pairwise_ranking_loss(correct_preds, targets, group_sizes)
    wrong_loss = module._pairwise_ranking_loss(wrong_preds, targets, group_sizes)

    assert correct_loss.item() < wrong_loss.item()
