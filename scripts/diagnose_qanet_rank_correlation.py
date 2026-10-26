#!/usr/bin/env python3
"""Compute Spearman rank correlation between QANet scores and true IoU.

Used to assess whether the QANet smoke-pass model has learned useful ranking
information even if the overall fusion IoU was not improved.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


FOLD_SPECS = [
    {
        "dataset": "BF-C2DL-MuSC",
        "fold": "fold-1",
        "scored_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "BF-C2DL-MuSC/sz256/musc_crop256/fold-1/qanet_scored_candidates.csv"
        ),
    },
    {
        "dataset": "BF-C2DL-MuSC",
        "fold": "fold-2",
        "scored_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "BF-C2DL-MuSC/sz256/musc_crop256/fold-2/qanet_scored_candidates.csv"
        ),
    },
    {
        "dataset": "Fluo-C2DL-MSC",
        "fold": "fold-1",
        "scored_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-1/qanet_scored_candidates.csv"
        ),
    },
    {
        "dataset": "Fluo-C2DL-MSC",
        "fold": "fold-2",
        "scored_csv": (
            "hpc_results/soft_hybrid_probe/qanet_filter_probe/"
            "Fluo-C2DL-MSC/sz1024/msc_crop1024/fold-2/qanet_scored_candidates.csv"
        ),
    },
]


def main() -> None:
    header = (
        f"{'Dataset':20s} {'Fold':6s} {'Split':12s} "
        f"{'Spearman r':12s} {'p-value':12s} {'N':5s}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for spec in FOLD_SPECS:
        p = Path(spec["scored_csv"])
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        df = pd.read_csv(p)
        for split in ("test", "validation", "train"):
            sub = df[df["split"] == split].dropna(
                subset=["qanet_score", "true_iou"]
            )
            if len(sub) < 5:
                continue
            r, pval = spearmanr(sub["qanet_score"], sub["true_iou"])
            print(
                f"{spec['dataset']:20s} {spec['fold']:6s} {split:12s} "
                f"{r:12.4f} {pval:12.4g} {len(sub):5d}"
            )
            records.append(
                {
                    "dataset": spec["dataset"],
                    "fold": spec["fold"],
                    "split": split,
                    "spearman_r": r,
                    "p_value": pval,
                    "n": len(sub),
                }
            )

    out_dir = Path("hpc_results/soft_hybrid_probe/qanet_filter_probe")
    out_path = out_dir / "qanet_rank_correlation.csv"
    if records:
        pd.DataFrame(records).to_csv(out_path, index=False)
        print(f"\nWrote rank correlation table: {out_path}")


if __name__ == "__main__":
    main()
