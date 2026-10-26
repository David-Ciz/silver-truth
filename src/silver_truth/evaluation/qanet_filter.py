from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tifffile
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from silver_truth.evaluation.candidate_features import (
    build_candidate_features,
    canonicalize_candidate_source,
)
from silver_truth.evaluation.candidate_ranker import add_cluster_ranker_features


@dataclass(frozen=True)
class QANetFilterConfig:
    target_column: str = "jaccard_score"
    train_splits: tuple[str, ...] = ("train",)
    calibration_splits: tuple[str, ...] = ("validation",)
    stacked_path_column: str = "stacked_path"
    split_column: str = "split"
    source_column: str = "competitor"
    crop_size: int = 64
    batch_size: int = 16
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_state: int = 13
    num_workers: int = 0
    use_source_identity: bool = False
    source_embedding_dim: int = 8
    min_training_rows: int = 3
    min_calibration_rows: int = 3
    cluster_distance: float = 4.0
    device: str = "auto"
    # Training augmentation
    augment: bool = False
    augment_brightness_std: float = 0.15
    augment_contrast_std: float = 0.15


def train_qanet_filter(
    candidates: pd.DataFrame,
    *,
    output_model: Path | None = None,
    config: QANetFilterConfig = QANetFilterConfig(),
) -> dict[str, Any]:
    _validate_candidate_table(candidates, config=config, require_target=True)
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)

    target = pd.to_numeric(candidates[config.target_column], errors="coerce")
    train_mask = _split_mask(candidates, config.split_column, config.train_splits)
    train_mask &= target.notna() & _existing_path_mask(
        candidates, config.stacked_path_column
    )
    if int(train_mask.sum()) < config.min_training_rows:
        raise ValueError(
            f"Need at least {config.min_training_rows} QANet training rows, "
            f"found {int(train_mask.sum())}."
        )

    source_vocab = _source_vocab(candidates.loc[train_mask], config=config)
    train_dataset = QANetCandidateDataset(
        candidates.loc[train_mask].reset_index(drop=True),
        config=config,
        source_vocab=source_vocab,
        require_target=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    device = _resolve_device(config.device)
    model = QANetRegressor(
        use_source_identity=config.use_source_identity,
        source_vocab_size=len(source_vocab) + 1,
        source_embedding_dim=config.source_embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    model.train()
    losses: list[float] = []
    for _ in range(config.max_epochs):
        epoch_losses = []
        for batch in loader:
            image = batch["image"].to(device)
            source_id = batch["source_id"].to(device)
            y = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(image, source_id)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch_losses:
            losses.append(float(np.mean(epoch_losses)))

    payload = {
        "state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "config": asdict(config),
        "source_vocab": source_vocab,
        "train_rows": int(train_mask.sum()),
        "calibration_rows": int(
            (
                _split_mask(candidates, config.split_column, config.calibration_splits)
                & target.notna()
                & _existing_path_mask(candidates, config.stacked_path_column)
            ).sum()
        ),
        "train_losses": losses,
    }
    if output_model is not None:
        output_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, output_model)
    return payload


def score_with_qanet_filter(
    candidates: pd.DataFrame,
    *,
    model_payload: dict[str, Any] | Path,
) -> pd.DataFrame:
    payload = _load_payload(model_payload)
    config = QANetFilterConfig(**payload["config"])
    _validate_candidate_table(candidates, config=config, require_target=False)

    source_vocab = payload.get("source_vocab", {})
    device = _resolve_device(config.device)
    model = QANetRegressor(
        use_source_identity=config.use_source_identity,
        source_vocab_size=len(source_vocab) + 1,
        source_embedding_dim=config.source_embedding_dim,
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    dataset = QANetCandidateDataset(
        candidates.reset_index(drop=True),
        config=config,
        source_vocab=source_vocab,
        require_target=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    scores: list[np.ndarray] = []
    raw_scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            source_id = batch["source_id"].to(device)
            raw = model(image, source_id)
            raw_np = raw.detach().cpu().numpy()
            raw_scores.append(raw_np)
            scores.append(np.clip(raw_np, 0.0, 1.0))

    scored = build_candidate_features(candidates).copy()
    scored = add_cluster_ranker_features(
        scored, cluster_distance=config.cluster_distance
    )
    scored["qanet_raw_score"] = np.concatenate(raw_scores) if raw_scores else []
    scored["qanet_score"] = np.concatenate(scores) if scores else []
    scored["qanet_rank_in_image"] = scored.groupby("image_id", dropna=False)[
        "qanet_score"
    ].rank(method="average", pct=True)
    if "ranker_cluster_id" in scored.columns:
        scored["qanet_rank_in_cluster"] = scored.groupby(
            "ranker_cluster_id", dropna=False
        )["qanet_score"].rank(method="average", pct=True)
    else:
        scored["qanet_rank_in_cluster"] = scored["qanet_rank_in_image"]
    return scored


def write_qanet_scored_candidates(
    candidate_path: Path,
    model_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    candidates = _read_table(candidate_path)
    scored = score_with_qanet_filter(candidates, model_payload=model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        scored.to_parquet(output_path, index=False)
    else:
        scored.to_csv(output_path, index=False)
    return scored


class QANetCandidateDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        candidates: pd.DataFrame,
        *,
        config: QANetFilterConfig,
        source_vocab: dict[str, int],
        require_target: bool,
    ) -> None:
        self.candidates = candidates.reset_index(drop=True)
        self.config = config
        self.source_vocab = source_vocab
        self.require_target = require_target

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.candidates.iloc[index]
        image = _candidate_tensor(
            Path(str(row[self.config.stacked_path_column])),
            crop_size=self.config.crop_size,
        )
        if self.require_target and self.config.augment:
            image = _augment_candidate_tensor(
                image,
                brightness_std=self.config.augment_brightness_std,
                contrast_std=self.config.augment_contrast_std,
            )
        source = canonicalize_candidate_source(row.get(self.config.source_column, ""))
        source_id = self.source_vocab.get(source, 0)
        item = {
            "image": image,
            "source_id": torch.tensor(source_id, dtype=torch.long),
        }
        if self.require_target:
            target = float(row[self.config.target_column])
            item["target"] = torch.tensor(np.clip(target, 0.0, 1.0), dtype=torch.float32)
        return item


class QANetRegressor(nn.Module):
    def __init__(
        self,
        *,
        use_source_identity: bool,
        source_vocab_size: int,
        source_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.use_source_identity = use_source_identity
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        if use_source_identity:
            self.source_embedding = nn.Embedding(
                max(1, source_vocab_size), source_embedding_dim
            )
            head_input = 64 + source_embedding_dim
        else:
            self.source_embedding = None
            head_input = 64
        self.head = nn.Sequential(
            nn.Linear(head_input, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, source_id: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(image).flatten(1)
        if self.source_embedding is not None:
            encoded = torch.cat([encoded, self.source_embedding(source_id)], dim=1)
        return self.head(encoded).squeeze(1)


def _augment_candidate_tensor(
    tensor: torch.Tensor,
    *,
    brightness_std: float,
    contrast_std: float,
) -> torch.Tensor:
    """Apply random brightness/contrast jitter and spatial flips to a (3,H,W) tensor.

    Only the raw image channel (index 0) is jittered; mask and boundary channels
    are flipped spatially but not intensity-augmented.
    """
    # Random horizontal and vertical flips (apply to all channels consistently)
    if torch.rand(1).item() > 0.5:
        tensor = tensor.flip(-1)
    if torch.rand(1).item() > 0.5:
        tensor = tensor.flip(-2)
    # Intensity augmentation on raw channel only
    if brightness_std > 0 or contrast_std > 0:
        raw = tensor[0].clone()
        if brightness_std > 0:
            raw = raw + float(torch.randn(1).item()) * brightness_std
        if contrast_std > 0:
            mean_val = raw.mean()
            scale = 1.0 + float(torch.randn(1).item()) * contrast_std
            raw = (raw - mean_val) * scale + mean_val
        raw = raw.clamp(-5.0, 5.0)
        tensor = torch.cat([raw.unsqueeze(0), tensor[1:]], dim=0)
    return tensor


def _candidate_tensor(path: Path, *, crop_size: int) -> torch.Tensor:
    image = tifffile.imread(path)
    if image.ndim == 3:
        raw = image[0].astype(np.float32, copy=False)
        mask = image[1] > 0
    elif image.ndim == 2:
        raw = image.astype(np.float32, copy=False)
        mask = image > 0
    else:
        raise ValueError(f"Unsupported stacked candidate image shape: {image.shape}")
    raw = _normalize_raw(raw)
    mask_float = mask.astype(np.float32, copy=False)
    boundary = _mask_boundary(mask).astype(np.float32, copy=False)
    array = np.stack([raw, mask_float, boundary]).astype(np.float32, copy=False)
    tensor = torch.from_numpy(array)
    if tensor.shape[-2:] != (crop_size, crop_size):
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(crop_size, crop_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor


def _normalize_raw(raw: np.ndarray) -> np.ndarray:
    raw = raw.astype(np.float32, copy=False)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return np.zeros_like(raw, dtype=np.float32)
    mean = float(finite.mean())
    std = float(finite.std())
    if std <= 1e-6:
        return np.zeros_like(raw, dtype=np.float32)
    return np.clip((raw - mean) / std, -5.0, 5.0).astype(np.float32, copy=False)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not bool(mask.any()):
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask)
    dilated = ndimage.binary_dilation(mask)
    return np.logical_xor(dilated, eroded)


def _source_vocab(candidates: pd.DataFrame, *, config: QANetFilterConfig) -> dict[str, int]:
    if not config.use_source_identity or config.source_column not in candidates.columns:
        return {}
    sources = sorted(
        {
            canonicalize_candidate_source(value)
            for value in candidates[config.source_column].dropna().tolist()
        }
    )
    return {source: index + 1 for index, source in enumerate(sources)}


def _validate_candidate_table(
    candidates: pd.DataFrame,
    *,
    config: QANetFilterConfig,
    require_target: bool,
) -> None:
    required = {config.stacked_path_column, config.split_column, "cell_id"}
    if require_target:
        required.add(config.target_column)
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")


def _existing_path_mask(candidates: pd.DataFrame, column: str) -> pd.Series:
    if column not in candidates.columns:
        return pd.Series(False, index=candidates.index)
    return candidates[column].map(lambda value: Path(str(value)).exists())


def _split_mask(
    candidates: pd.DataFrame, split_column: str, splits: tuple[str, ...]
) -> pd.Series:
    if split_column not in candidates.columns:
        return pd.Series(True, index=candidates.index)
    return candidates[split_column].astype(str).isin(splits)


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_payload(model_payload: dict[str, Any] | Path) -> dict[str, Any]:
    if isinstance(model_payload, (str, Path)):
        return joblib.load(model_payload)
    return model_payload


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or apply a small QANet-style candidate quality filter."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("candidates", type=Path)
    train_parser.add_argument("--output-model", required=True, type=Path)
    train_parser.add_argument("--train-split", action="append", default=None)
    train_parser.add_argument("--calibration-split", action="append", default=None)
    train_parser.add_argument("--target-column", default="jaccard_score")
    train_parser.add_argument("--crop-size", type=int, default=64)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--max-epochs", type=int, default=20)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--use-source-identity", action="store_true")
    train_parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable brightness/contrast jitter and flip augmentation during training.",
    )
    train_parser.add_argument(
        "--augment-brightness-std",
        type=float,
        default=0.15,
        help="Std of additive brightness noise applied to the raw channel.",
    )
    train_parser.add_argument(
        "--augment-contrast-std",
        type=float,
        default=0.15,
        help="Std of multiplicative contrast scale factor applied to the raw channel.",
    )

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("candidates", type=Path)
    score_parser.add_argument("--model", required=True, type=Path)
    score_parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "train":
        candidates = _read_table(args.candidates)
        config = QANetFilterConfig(
            target_column=args.target_column,
            train_splits=tuple(args.train_split or ("train",)),
            calibration_splits=tuple(args.calibration_split or ("validation",)),
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
            use_source_identity=args.use_source_identity,
            augment=args.augment,
            augment_brightness_std=args.augment_brightness_std,
            augment_contrast_std=args.augment_contrast_std,
        )
        payload = train_qanet_filter(
            candidates, output_model=args.output_model, config=config
        )
        print(
            "Wrote QANet filter to "
            f"{args.output_model} with {payload['train_rows']} train rows and "
            f"{payload['calibration_rows']} calibration rows"
        )
    elif args.command == "score":
        scored = write_qanet_scored_candidates(args.candidates, args.model, args.output)
        print(f"Wrote {len(scored)} QANet-scored rows to {args.output}")


if __name__ == "__main__":
    main()
