from types import SimpleNamespace

import torch

from silver_truth.ensemble import ensemble, training
from silver_truth.ensemble.datasets import Version


class _Dataset:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return self._samples[index]


def test_training_eval_sets_stack_c2_samples_into_batch_dimension() -> None:
    dataset = _Dataset(
        [
            (torch.zeros((2, 4, 4)), torch.zeros((1, 4, 4))),
            (torch.ones((2, 4, 4)), torch.ones((1, 4, 4))),
            (torch.full((2, 4, 4), 2.0), torch.full((1, 4, 4), 2.0)),
        ]
    )

    imgs, gts = training._get_eval_sets(dataset, is_single_input=False)

    assert imgs.shape == (3, 2, 4, 4)
    assert gts.shape == (3, 1, 4, 4)


def test_training_eval_sets_preserve_singleton_leading_axis_for_multi_image_samples() -> None:
    dataset = _Dataset(
        [
            (torch.zeros((1, 8, 4, 4)), torch.zeros((1, 1, 4, 4))),
            (torch.ones((1, 8, 4, 4)), torch.ones((1, 1, 4, 4))),
        ]
    )

    imgs, gts = training._get_eval_sets(dataset, is_single_input=False)

    assert imgs.shape == (2, 8, 4, 4)
    assert gts.shape == (2, 1, 4, 4)


def test_ensemble_eval_sets_stack_c2_samples_into_batch_dimension() -> None:
    dataset = _Dataset(
        [
            (torch.zeros((2, 4, 4)), torch.zeros((1, 4, 4))),
            (torch.ones((2, 4, 4)), torch.ones((1, 4, 4))),
        ]
    )

    imgs, gts = ensemble._get_eval_sets(dataset)

    assert imgs.shape == (2, 2, 4, 4)
    assert gts.shape == (2, 1, 4, 4)


def test_resolve_dataset_version_prefers_explicit_override_and_falls_back_to_checkpoint_inputs() -> None:
    one_input_model = SimpleNamespace(hparams=SimpleNamespace(num_inputs=1))
    two_input_model = SimpleNamespace(hparams=SimpleNamespace(num_inputs=2))
    three_input_model = SimpleNamespace(hparams=SimpleNamespace(num_inputs=3))

    assert ensemble._resolve_dataset_version(one_input_model) == Version.C1
    assert ensemble._resolve_dataset_version(two_input_model) == Version.C2
    assert ensemble._resolve_dataset_version(three_input_model) == Version.C3
    assert ensemble._resolve_dataset_version(two_input_model, "c1") == Version.C1
