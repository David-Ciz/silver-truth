#!/usr/bin/env python3
"""
train_qanet_all_folds.py
------------------------
Train and score QANet for every registered dataset/fold combination.

For each fold:
  1. Train on that fold's train split.
  2. Score the full qa_crops parquet (all splits) to produce qanet_scored_candidates.csv.

Outputs land in:
  {output_root}/{dataset}/{sz}/{variant}/fold-{n}/
    qanet.joblib
    qanet_scored_candidates.csv

Usage:
    # Run all folds (skips already-done), 20 epochs, 64px crops
    python scripts/train_qanet_all_folds.py

    # Run with augmentation, 40 epochs, native per-dataset crop sizes
    python scripts/train_qanet_all_folds.py --augment --max-epochs 40 --native-crop-size \
        --output-root hpc_results/soft_hybrid_probe/qanet_augmented_probe

    # Run only specific datasets
    python scripts/train_qanet_all_folds.py --datasets Fluo-C2DL-MSC

    # Re-run even if output already exists
    python scripts/train_qanet_all_folds.py --force

    # Dry run
    python scripts/train_qanet_all_folds.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# Registry: one entry per (dataset, crop_sz, variant, fold)
# qa_crop_parquet: input for both training and scoring
# native_crop_size: largest useful crop size for this dataset (capped at 256
#   for practical CPU training; MSC native=1024 is capped to 256).
# -------------------------------------------------------------------------
FOLDS = [
    dict(
        dataset="BF-C2DL-HSC",
        crop_sz="sz64",
        variant="baseline",
        fold="fold-1",
        qa_crop_parquet="data/dataframes/BF-C2DL-HSC/qa_crops/fold-1_sz64_qa_dataset.parquet",
        native_crop_size=64,
    ),
    dict(
        dataset="BF-C2DL-HSC",
        crop_sz="sz64",
        variant="baseline",
        fold="fold-2",
        qa_crop_parquet="data/dataframes/BF-C2DL-HSC/qa_crops/fold-2_sz64_qa_dataset.parquet",
        native_crop_size=64,
    ),
    dict(
        dataset="BF-C2DL-MuSC",
        crop_sz="sz256",
        variant="musc_crop256",
        fold="fold-1",
        qa_crop_parquet="data/dataframes/BF-C2DL-MuSC/qa_crops/fold-1_sz256_qa_dataset.parquet",
        native_crop_size=256,
    ),
    dict(
        dataset="BF-C2DL-MuSC",
        crop_sz="sz256",
        variant="musc_crop256",
        fold="fold-2",
        qa_crop_parquet="data/dataframes/BF-C2DL-MuSC/qa_crops/fold-2_sz256_qa_dataset.parquet",
        native_crop_size=256,
    ),
    dict(
        dataset="DIC-C2DH-HeLa",
        crop_sz="sz256",
        variant="hela_crop256",
        fold="fold-1",
        qa_crop_parquet="data/dataframes/DIC-C2DH-HeLa/qa_crops/fold-1_sz256_qa_dataset.parquet",
        native_crop_size=256,
    ),
    dict(
        dataset="DIC-C2DH-HeLa",
        crop_sz="sz256",
        variant="hela_crop256",
        fold="fold-2",
        qa_crop_parquet="data/dataframes/DIC-C2DH-HeLa/qa_crops/fold-2_sz256_qa_dataset.parquet",
        native_crop_size=256,
    ),
    dict(
        dataset="Fluo-C2DL-MSC",
        crop_sz="sz1024",
        variant="msc_crop1024",
        fold="fold-1",
        qa_crop_parquet="data/dataframes/Fluo-C2DL-MSC/qa_crops/fold-1_sz1024_qa_dataset.parquet",
        native_crop_size=256,  # capped from 1024 for practical CPU training
    ),
    dict(
        dataset="Fluo-C2DL-MSC",
        crop_sz="sz1024",
        variant="msc_crop1024",
        fold="fold-2",
        qa_crop_parquet="data/dataframes/Fluo-C2DL-MSC/qa_crops/fold-2_sz1024_qa_dataset.parquet",
        native_crop_size=256,  # capped from 1024 for practical CPU training
    ),
    dict(
        dataset="Fluo-N2DH-GOWT1",
        crop_sz="sz160",
        variant="gowt1_crop160",
        fold="fold-1",
        qa_crop_parquet="data/dataframes/Fluo-N2DH-GOWT1/qa_crops/fold-1_sz160_qa_dataset.parquet",
        native_crop_size=160,
    ),
    dict(
        dataset="Fluo-N2DH-GOWT1",
        crop_sz="sz160",
        variant="gowt1_crop160",
        fold="fold-2",
        qa_crop_parquet="data/dataframes/Fluo-N2DH-GOWT1/qa_crops/fold-2_sz160_qa_dataset.parquet",
        native_crop_size=160,
    ),
]

DEFAULT_OUTPUT_ROOT = Path("hpc_results/soft_hybrid_probe/qanet_20ep_probe")
RUNNER = Path("scripts/run_qanet_filter.py")


def output_dir(root: Path, entry: dict) -> Path:
    return (
        root
        / entry["dataset"]
        / entry["crop_sz"]
        / entry["variant"]
        / entry["fold"]
    )


def is_done(root: Path, entry: dict) -> bool:
    odir = output_dir(root, entry)
    return (odir / "qanet_scored_candidates.csv").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QANet for all folds")
    parser.add_argument(
        "--datasets",
        nargs="+",
        metavar="DS",
        help="Only run these dataset names (default: all)",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        metavar="FOLD",
        help="Only run these fold names e.g. fold-1 (default: all)",
    )
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument(
        "--crop-size",
        type=int,
        default=64,
        help="Crop size for all datasets (ignored if --native-crop-size is set).",
    )
    parser.add_argument(
        "--native-crop-size",
        action="store_true",
        help=(
            "Use each dataset's native crop size instead of a fixed crop size. "
            "MSC native is capped at 256 for practical CPU training."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable brightness/contrast jitter and flip augmentation during training.",
    )
    parser.add_argument(
        "--augment-brightness-std",
        type=float,
        default=0.15,
        help="Brightness jitter std (used with --augment).",
    )
    parser.add_argument(
        "--augment-contrast-std",
        type=float,
        default=0.15,
        help="Contrast jitter std (used with --augment).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            f"Output root directory (default: {DEFAULT_OUTPUT_ROOT}). "
            "Set to a different path (e.g. qanet_augmented_probe) for augmented runs."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    runner = repo / RUNNER
    output_root = repo / args.output_root

    folds = FOLDS
    if args.datasets:
        folds = [f for f in folds if f["dataset"] in args.datasets]
    if args.folds:
        folds = [f for f in folds if f["fold"] in args.folds]

    total = ran = skipped = failed = 0

    for entry in folds:
        total += 1
        label = f"{entry['dataset']}/{entry['fold']}"

        odir = output_dir(output_root, entry)
        model_path = odir / "qanet.joblib"
        scored_path = odir / "qanet_scored_candidates.csv"
        qa_parquet = repo / entry["qa_crop_parquet"]

        if not args.force and is_done(output_root, entry):
            print(f"[DONE]  {label}  — scored CSV already exists, skipping")
            skipped += 1
            continue

        if not qa_parquet.exists():
            print(f"[WARN]  {label}  — qa_crop_parquet not found: {qa_parquet}")
            skipped += 1
            continue

        odir.mkdir(parents=True, exist_ok=True)

        # Determine crop size for this fold
        if args.native_crop_size:
            crop_size = entry["native_crop_size"]
        else:
            crop_size = args.crop_size

        # --- Step 1: train ---
        train_cmd = [
            sys.executable, str(runner), "train",
            str(qa_parquet),
            "--output-model", str(model_path),
            "--max-epochs", str(args.max_epochs),
            "--crop-size", str(crop_size),
            "--device", args.device,
            "--use-source-identity",
        ]
        if args.augment:
            train_cmd += [
                "--augment",
                "--augment-brightness-std", str(args.augment_brightness_std),
                "--augment-contrast-std", str(args.augment_contrast_std),
            ]
        print(f"\n[TRAIN] {label}  (crop_size={crop_size}, augment={args.augment})")
        print("  " + " ".join(train_cmd))

        if not args.dry_run:
            result = subprocess.run(train_cmd, cwd=repo)
            if result.returncode != 0:
                print(f"[FAIL]  {label}  — train exit code {result.returncode}", file=sys.stderr)
                failed += 1
                continue

        # --- Step 2: score ---
        score_cmd = [
            sys.executable, str(runner), "score",
            str(qa_parquet),
            "--model", str(model_path),
            "--output", str(scored_path),
        ]
        print(f"[SCORE] {label}")
        print("  " + " ".join(score_cmd))

        if not args.dry_run:
            result = subprocess.run(score_cmd, cwd=repo)
            if result.returncode != 0:
                print(f"[FAIL]  {label}  — score exit code {result.returncode}", file=sys.stderr)
                failed += 1
                continue
            print(f"[OK]    {label}")
            ran += 1
        else:
            ran += 1

    prefix = "[DRY-RUN] " if args.dry_run else ""
    print(f"\n{prefix}Summary: {total} total, {ran} ran, {skipped} skipped, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
