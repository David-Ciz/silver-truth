from pathlib import Path

import pytest
import torch

from silver_truth.qa import cnn as qa_cnn


def test_initialize_model_from_checkpoint_loads_weights(tmp_path: Path) -> None:
    source_model = qa_cnn.Jaccard(model_type="resnet18")
    for parameter in source_model.parameters():
        torch.nn.init.constant_(parameter, 0.25)

    checkpoint_path = tmp_path / "source.pt"
    qa_cnn.save_model(
        source_model,
        checkpoint_path,
        metadata={"model_type": "resnet18", "input_channels": "0,1"},
    )

    target_model = qa_cnn.JaccardLightningModule(model_type="resnet18")
    metadata = qa_cnn.initialize_model_from_checkpoint(
        target_model,
        checkpoint_path,
        expected_model_type="resnet18",
        expected_input_channels="0,1",
    )

    loaded_state = target_model.model.state_dict()
    source_state = source_model.state_dict()
    for key, tensor in source_state.items():
        assert torch.allclose(loaded_state[key], tensor)
    assert metadata["model_type"] == "resnet18"


def test_initialize_model_from_checkpoint_rejects_model_type_mismatch(
    tmp_path: Path,
) -> None:
    source_model = qa_cnn.Jaccard(model_type="resnet18")
    checkpoint_path = tmp_path / "source.pt"
    qa_cnn.save_model(
        source_model,
        checkpoint_path,
        metadata={"model_type": "resnet18", "input_channels": "0,1"},
    )

    target_model = qa_cnn.JaccardLightningModule(model_type="resnet18")
    with pytest.raises(ValueError, match="model_type mismatch"):
        qa_cnn.initialize_model_from_checkpoint(
            target_model,
            checkpoint_path,
            expected_model_type="resnet50",
        )


def test_initialize_model_from_checkpoint_rejects_input_channel_mismatch(
    tmp_path: Path,
) -> None:
    source_model = qa_cnn.Jaccard(model_type="resnet18")
    checkpoint_path = tmp_path / "source.pt"
    qa_cnn.save_model(
        source_model,
        checkpoint_path,
        metadata={"model_type": "resnet18", "input_channels": "0,1"},
    )

    target_model = qa_cnn.JaccardLightningModule(model_type="resnet18")
    with pytest.raises(ValueError, match="input_channels mismatch"):
        qa_cnn.initialize_model_from_checkpoint(
            target_model,
            checkpoint_path,
            expected_model_type="resnet18",
            expected_input_channels="0,2",
        )
