import torch

from silver_truth.qa import cnn as qa_cnn


class _FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((x.shape[0], 3), dtype=x.dtype, device=x.device)


def test_jaccard_sigmoid_output_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(qa_cnn, "resnet18", lambda weights: _FakeBackbone())

    model = qa_cnn.Jaccard(model_type="resnet18", output_activation="sigmoid")
    torch.nn.init.constant_(model.fc.weight, 10.0)
    torch.nn.init.constant_(model.fc.bias, 10.0)

    predictions = model(torch.zeros((4, 2, 8, 8)))

    assert torch.all(predictions >= 0.0)
    assert torch.all(predictions <= 1.0)


def test_jaccard_linear_output_can_remain_unbounded(monkeypatch) -> None:
    monkeypatch.setattr(qa_cnn, "resnet18", lambda weights: _FakeBackbone())

    model = qa_cnn.Jaccard(model_type="resnet18", output_activation="linear")
    torch.nn.init.constant_(model.fc.weight, 10.0)
    torch.nn.init.constant_(model.fc.bias, 10.0)

    predictions = model(torch.zeros((4, 2, 8, 8)))

    assert torch.all(predictions > 1.0)
