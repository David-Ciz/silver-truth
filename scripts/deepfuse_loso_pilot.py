#!/usr/bin/env python3
"""Leave-sequence-out DeepFuse-style pilot on synchronized CTC masks.

This is not the released TensorFlow script. It ports the published DeepFuse
branch/fusion architecture to PyTorch so we can run the strict question:
train on one sequence, infer on the other without reading test SEG GT, and
only then score against test SEG.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Fluo-N2DH-GOWT1")
    parser.add_argument("--data-root", type=Path, default=Path("data/synchronized_data"))
    parser.add_argument("--train-seq", default="01")
    parser.add_argument("--test-seq", default="02")
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpc_results/deepfuse_loso_pilot"),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class ObjectSample:
    time_id: str
    label: int
    y0: int
    x0: int


@lru_cache(maxsize=512)
def read_mask(path: str) -> np.ndarray:
    return tifffile.imread(path)


def crop_with_padding(image: np.ndarray, y0: int, x0: int, size: int) -> np.ndarray:
    crop = np.zeros((size, size), dtype=image.dtype)
    y1 = y0 + size
    x1 = x0 + size
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(image.shape[0], y1)
    src_x1 = min(image.shape[1], x1)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return crop
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    crop[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = image[src_y0:src_y1, src_x0:src_x1]
    return crop


def paste_prob(
    label_image: np.ndarray,
    best_prob: np.ndarray,
    prob_crop: np.ndarray,
    label: int,
    y0: int,
    x0: int,
    threshold: float,
) -> None:
    size = prob_crop.shape[0]
    y1 = y0 + size
    x1 = x0 + size
    dst_y0 = max(0, y0)
    dst_x0 = max(0, x0)
    dst_y1 = min(label_image.shape[0], y1)
    dst_x1 = min(label_image.shape[1], x1)
    if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
        return
    src_y0 = dst_y0 - y0
    src_x0 = dst_x0 - x0
    prob_view = prob_crop[
        src_y0 : src_y0 + (dst_y1 - dst_y0),
        src_x0 : src_x0 + (dst_x1 - dst_x0),
    ]
    update = (prob_view >= threshold) & (
        prob_view > best_prob[dst_y0:dst_y1, dst_x0:dst_x1]
    )
    best_prob[dst_y0:dst_y1, dst_x0:dst_x1][update] = prob_view[update]
    label_image[dst_y0:dst_y1, dst_x0:dst_x1][update] = label


def crop_origin_from_mask(mask: np.ndarray, crop_size: int) -> tuple[int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    center_y = int(round((int(ys.min()) + int(ys.max())) / 2))
    center_x = int(round((int(xs.min()) + int(xs.max())) / 2))
    return center_y - crop_size // 2, center_x - crop_size // 2


def discover_sources(dataset_dir: Path) -> list[str]:
    sources = [
        path.name
        for path in dataset_dir.iterdir()
        if (path / "01_RES").is_dir() and (path / "02_RES").is_dir()
    ]
    return sorted(sources)


def gt_seg_path(dataset_dir: Path, seq: str, time_id: str) -> Path:
    return dataset_dir / f"{seq}_GT" / "SEG" / f"man_seg{time_id}.tif"


def source_mask_path(dataset_dir: Path, source: str, seq: str, time_id: str) -> Path:
    return dataset_dir / source / f"{seq}_RES" / f"mask{time_id}.tif"


def collect_train_samples(
    dataset_dir: Path, train_seq: str, crop_size: int
) -> list[ObjectSample]:
    samples: list[ObjectSample] = []
    for gt_path in sorted((dataset_dir / f"{train_seq}_GT" / "SEG").glob("man_seg*.tif")):
        time_id = gt_path.stem[-3:]
        gt = read_mask(str(gt_path))
        for label in [int(v) for v in np.unique(gt) if v != 0]:
            origin = crop_origin_from_mask(gt == label, crop_size)
            if origin is None:
                continue
            samples.append(ObjectSample(time_id, label, origin[0], origin[1]))
    return samples


class DeepFuseObjectDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        seq: str,
        sources: list[str],
        samples: list[ObjectSample],
        crop_size: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.seq = seq
        self.sources = sources
        self.samples = samples
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        channels = []
        for source in self.sources:
            path = source_mask_path(self.dataset_dir, source, self.seq, sample.time_id)
            mask = read_mask(str(path))
            binary = (mask == sample.label).astype(np.float32)
            channels.append(crop_with_padding(binary, sample.y0, sample.x0, self.crop_size))
        gt = read_mask(str(gt_seg_path(self.dataset_dir, self.seq, sample.time_id)))
        target = (gt == sample.label).astype(np.float32)
        target_crop = crop_with_padding(target, sample.y0, sample.x0, self.crop_size)
        x = torch.from_numpy(np.stack(channels, axis=0)).float()
        y = torch.from_numpy(target_crop[None, :, :]).float()
        return x, y


class DeepFuseTorch(nn.Module):
    def __init__(self, n_sources: int, filters: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, filters, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filters, filters, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filters, filters, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                )
                for _ in range(n_sources)
            ]
        )
        self.head = nn.Conv2d(filters * n_sources, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x[:, idx : idx + 1]) for idx, branch in enumerate(self.branches)]
        return self.head(torch.cat(outputs, dim=1))


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def train_model(
    model: nn.Module,
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[float]]:
    val_count = max(1, int(round(len(dataset) * 0.2)))
    train_count = len(dataset) - val_count
    train_ds, val_ds = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = dice_loss(model(x), y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                val_losses.append(float(dice_loss(model(x), y).detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def object_stack_for_label(
    dataset_dir: Path,
    sources: list[str],
    seq: str,
    time_id: str,
    label: int,
    y0: int,
    x0: int,
    crop_size: int,
) -> torch.Tensor:
    channels = []
    for source in sources:
        mask = read_mask(str(source_mask_path(dataset_dir, source, seq, time_id)))
        binary = (mask == label).astype(np.float32)
        channels.append(crop_with_padding(binary, y0, x0, crop_size))
    return torch.from_numpy(np.stack(channels, axis=0)[None, :, :, :]).float()


def majority_reconstruct_frame(
    dataset_dir: Path,
    sources: list[str],
    seq: str,
    time_id: str,
    shape: tuple[int, int],
) -> np.ndarray:
    source_masks = [
        read_mask(str(source_mask_path(dataset_dir, source, seq, time_id)))
        for source in sources
    ]
    labels = sorted({int(v) for mask in source_masks for v in np.unique(mask) if v != 0})
    votes = np.zeros(shape, dtype=np.uint16)
    counts = np.zeros(shape, dtype=np.uint16)
    threshold = max(1, len(sources) // 2 + 1)
    for label in labels:
        count = np.zeros(shape, dtype=np.uint16)
        for mask in source_masks:
            count += mask == label
        update = count >= threshold
        votes[update] = label
        counts[update] = count[update]
    return votes


def predict_frame(
    model: nn.Module,
    dataset_dir: Path,
    sources: list[str],
    seq: str,
    time_id: str,
    crop_size: int,
    threshold: float,
    device: torch.device,
) -> np.ndarray:
    source_masks = [
        read_mask(str(source_mask_path(dataset_dir, source, seq, time_id)))
        for source in sources
    ]
    shape = source_masks[0].shape
    labels = sorted({int(v) for mask in source_masks for v in np.unique(mask) if v != 0})
    pred = np.zeros(shape, dtype=np.uint16)
    best_prob = np.zeros(shape, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for label in labels:
            union = np.zeros(shape, dtype=bool)
            for mask in source_masks:
                union |= mask == label
            origin = crop_origin_from_mask(union, crop_size)
            if origin is None:
                continue
            y0, x0 = origin
            x = object_stack_for_label(
                dataset_dir, sources, seq, time_id, label, y0, x0, crop_size
            ).to(device)
            prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
            paste_prob(pred, best_prob, prob, label, y0, x0, threshold)
    return pred


def labelwise_iou(gt: np.ndarray, pred: np.ndarray) -> list[float]:
    scores = []
    for label in [int(v) for v in np.unique(gt) if v != 0]:
        gt_mask = gt == label
        pred_mask = pred == label
        union = np.logical_or(gt_mask, pred_mask).sum()
        if union == 0:
            continue
        scores.append(float(np.logical_and(gt_mask, pred_mask).sum() / union))
    return scores


def binary_scores(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    gt_fg = gt > 0
    pred_fg = pred > 0
    inter = np.logical_and(gt_fg, pred_fg).sum()
    union = np.logical_or(gt_fg, pred_fg).sum()
    iou = float(inter / union) if union else 1.0
    denom = gt_fg.sum() + pred_fg.sum()
    dice = float(2 * inter / denom) if denom else 1.0
    return iou, dice


def score_predictions(
    gt_paths: list[Path], pred_dir: Path, majority_dir: Path
) -> dict[str, object]:
    rows = []
    deepfuse_label_scores: list[float] = []
    majority_label_scores: list[float] = []
    for gt_path in gt_paths:
        time_id = gt_path.stem[-3:]
        gt = read_mask(str(gt_path))
        pred = read_mask(str(pred_dir / f"mask{time_id}.tif"))
        majority = read_mask(str(majority_dir / f"mask{time_id}.tif"))
        pred_label = labelwise_iou(gt, pred)
        maj_label = labelwise_iou(gt, majority)
        pred_binary_iou, pred_binary_dice = binary_scores(gt, pred)
        maj_binary_iou, maj_binary_dice = binary_scores(gt, majority)
        deepfuse_label_scores.extend(pred_label)
        majority_label_scores.extend(maj_label)
        rows.append(
            {
                "time_id": time_id,
                "deepfuse_label_iou": float(np.mean(pred_label)) if pred_label else None,
                "majority_label_iou": float(np.mean(maj_label)) if maj_label else None,
                "deepfuse_binary_iou": pred_binary_iou,
                "majority_binary_iou": maj_binary_iou,
                "deepfuse_binary_dice": pred_binary_dice,
                "majority_binary_dice": maj_binary_dice,
            }
        )
    return {
        "frames": rows,
        "deepfuse_mean_label_iou": float(np.mean(deepfuse_label_scores)),
        "majority_mean_label_iou": float(np.mean(majority_label_scores)),
        "deepfuse_mean_binary_iou": float(np.mean([r["deepfuse_binary_iou"] for r in rows])),
        "majority_mean_binary_iou": float(np.mean([r["majority_binary_iou"] for r in rows])),
        "deepfuse_mean_binary_dice": float(
            np.mean([r["deepfuse_binary_dice"] for r in rows])
        ),
        "majority_mean_binary_dice": float(np.mean([r["majority_binary_dice"] for r in rows])),
        "n_frames": len(rows),
        "n_label_scores": len(deepfuse_label_scores),
    }


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dir = args.data_root / args.dataset
    sources = discover_sources(dataset_dir)
    if not sources:
        raise RuntimeError(f"No synchronized source folders found under {dataset_dir}")

    samples = collect_train_samples(dataset_dir, args.train_seq, args.crop_size)
    if len(samples) < 2:
        raise RuntimeError("Not enough train samples")

    run_dir = (
        args.output_dir
        / args.dataset
        / f"train{args.train_seq}_test{args.test_seq}_crop{args.crop_size}"
    )
    pred_dir = run_dir / "pred_deepfuse"
    majority_dir = run_dir / "pred_majority"
    pred_dir.mkdir(parents=True, exist_ok=True)
    majority_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    print(f"dataset={args.dataset} sources={len(sources)} samples={len(samples)} device={device}")
    dataset = DeepFuseObjectDataset(
        dataset_dir, args.train_seq, sources, samples, args.crop_size
    )
    model = DeepFuseTorch(len(sources), args.filters).to(device)
    history = train_model(model, dataset, args, device)

    gt_paths = sorted((dataset_dir / f"{args.test_seq}_GT" / "SEG").glob("man_seg*.tif"))
    for gt_path in gt_paths:
        time_id = gt_path.stem[-3:]
        gt = read_mask(str(gt_path))
        pred = predict_frame(
            model,
            dataset_dir,
            sources,
            args.test_seq,
            time_id,
            args.crop_size,
            args.threshold,
            device,
        )
        majority = majority_reconstruct_frame(
            dataset_dir, sources, args.test_seq, time_id, gt.shape
        )
        tifffile.imwrite(pred_dir / f"mask{time_id}.tif", pred)
        tifffile.imwrite(majority_dir / f"mask{time_id}.tif", majority)
        print(f"predicted time={time_id}")

    metrics = score_predictions(gt_paths, pred_dir, majority_dir)
    summary = {
        "args": vars(args) | {"data_root": str(args.data_root), "output_dir": str(args.output_dir)},
        "sources": sources,
        "history": history,
        "metrics": metrics,
        "note": (
            "PyTorch DeepFuse-style LOSO pilot. Test SEG is read only after "
            "predictions are written. Source labels come from synchronized result masks."
        ),
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
