#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"qa_(?P<dataset>hsc64|musc256)_(?P<model>.+?)_qa_predictions_"
    r"(?P<fold>fold-[12]|mixed)_evaluation_metrics\.csv$"
)

DATASET_TITLES = {
    "hsc64": "HSC `sz64`",
    "musc256": "MuSC `sz256`",
}


def load_summary_table(sweep_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((sweep_root / "mlflow" / "mlruns").rglob("*_evaluation_metrics.csv")):
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            continue

        meta = match.groupdict()
        df = pd.read_csv(path)
        if df.columns[0].startswith("Unnamed"):
            df = df.rename(columns={df.columns[0]: "row_name"})

        row: dict[str, object] = {
            "dataset": meta["dataset"],
            "model": meta["model"],
            "fold": meta["fold"],
            "source": str(path),
        }
        for split in ("train", "validation", "test"):
            split_df = df[df["split"] == split]
            if split_df.empty:
                continue
            split_row = split_df.iloc[0]
            prefix = "val" if split == "validation" else split
            row[f"{prefix}_pearson"] = float(split_row["pearson_correlation"])
            row[f"{prefix}_spearman"] = float(split_row["spearman_correlation"])
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No QA evaluation CSVs found under {(sweep_root / 'mlflow' / 'mlruns')}"
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["dataset", "fold", "test_spearman", "test_pearson"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def _format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    metric_columns = [
        "train_pearson",
        "train_spearman",
        "val_pearson",
        "val_spearman",
        "test_pearson",
        "test_spearman",
    ]
    for column in metric_columns:
        if column not in formatted.columns:
            continue
        formatted[column] = formatted[column].map(lambda value: round(float(value), 4))
    return formatted


def _format_root(root: Path) -> str:
    return str(root).rstrip("/")


def _command_block(
    *,
    sweep_root: Path,
    output_path: str,
    mixed_root: Path | None,
) -> list[str]:
    lines = [
        "```bash",
        ".venv/bin/python scripts/summarize_qa_sweep.py \\",
        f"  --sweep-root {_format_root(sweep_root)} \\",
    ]
    if mixed_root is not None:
        lines.append(f"  --mixed-root {_format_root(mixed_root)} \\")
    lines.append(f"  --output {output_path}")
    lines.append("```")
    return lines


def build_markdown(
    summary: pd.DataFrame,
    sweep_root: Path,
    mixed_summary: pd.DataFrame | None = None,
    mixed_root: Path | None = None,
) -> str:
    lines: list[str] = [
        "# QA Sweep Results",
        "",
        "This page summarizes the QA-only HPC result bundles that were archived"
        " locally for reuse.",
        "",
        "Source artifacts:",
        "",
        f"- Architecture sweep MLflow root: `{_format_root(sweep_root)}/mlflow/mlruns`",
        f"- Architecture sweep QA regression summaries: `{_format_root(sweep_root)}/mlflow/mlruns/*/artifacts/evaluation/*_evaluation_metrics.csv`",
        "",
        "The tables below report Pearson and Spearman correlations for the `train`, `validation`, and `test` splits.",
        "",
        "## Recreate These Results",
        "",
        "Regenerate this page from the saved QA sweep artifacts with:",
        "",
        *_command_block(
            sweep_root=sweep_root,
            output_path="docs/qa_sweep_results.md",
            mixed_root=mixed_root,
        ),
        "",
        "If you want to inspect the generated markdown before overwriting the docs page, write to a temporary path first:",
        "",
        *_command_block(
            sweep_root=sweep_root,
            output_path="/tmp/qa_sweep_results.md",
            mixed_root=mixed_root,
        ),
        "",
        "## Stored Campaigns",
        "",
        f"- `architecture_sweep`: `{_format_root(sweep_root)}`",
        "",
    ]
    if mixed_root is not None:
        lines.insert(len(lines) - 1, f"- `mixed_resnet18`: `{_format_root(mixed_root)}`")

    display_columns = [
        "fold",
        "model",
        "train_pearson",
        "train_spearman",
        "val_pearson",
        "val_spearman",
        "test_pearson",
        "test_spearman",
    ]
    rename_map = {
        "fold": "fold",
        "model": "model",
        "train_pearson": "train Pearson",
        "train_spearman": "train Spearman",
        "val_pearson": "val Pearson",
        "val_spearman": "val Spearman",
        "test_pearson": "test Pearson",
        "test_spearman": "test Spearman",
    }

    for dataset in ("hsc64", "musc256"):
        lines.append(f"## {DATASET_TITLES[dataset]}")
        lines.append("")
        dataset_df = summary[summary["dataset"] == dataset][display_columns]
        dataset_df = _format_metric_table(dataset_df).rename(columns=rename_map)
        lines.append(dataset_df.to_markdown(index=False))
        lines.append("")

    avg = (
        summary.groupby(["dataset", "model"], as_index=False)
        .agg(
            test_pearson=("test_pearson", "mean"),
            test_spearman=("test_spearman", "mean"),
        )
        .sort_values(
            ["dataset", "test_spearman", "test_pearson"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )
    avg = _format_metric_table(
        avg.rename(
            columns={
                "test_pearson": "train_pearson",
                "test_spearman": "train_spearman",
                "dataset": "dataset",
                "model": "model",
            }
        )
    ).rename(
        columns={
            "train_pearson": "mean test Pearson",
            "train_spearman": "mean test Spearman",
        }
    )[["dataset", "model", "mean test Pearson", "mean test Spearman"]]
    lines.extend(
        [
            "## Fold-Averaged Test Ranking",
            "",
            avg.to_markdown(index=False),
            "",
            "## Short Readout",
            "",
            "- `resnet18` is the most stable choice across both datasets.",
            "- `resnet101` is the best `HSC sz64` model on fold-averaged test correlations, but only by a modest margin over `resnet18`.",
            "- `MuSC sz256` favors `resnet18`, with `efficientnet_b1` close behind.",
            "- `efficientnet_b4` is consistently poor and does not look worth keeping in the next round.",
            "",
        ]
    )

    if mixed_summary is not None and not mixed_summary.empty:
        mixed_display = mixed_summary[
            [
                "dataset",
                "model",
                "train_pearson",
                "train_spearman",
                "val_pearson",
                "val_spearman",
                "test_pearson",
                "test_spearman",
            ]
        ].copy()
        mixed_display = _format_metric_table(mixed_display).rename(
            columns={
                "dataset": "dataset",
                "model": "model",
                "train_pearson": "train Pearson",
                "train_spearman": "train Spearman",
                "val_pearson": "val Pearson",
                "val_spearman": "val Spearman",
                "test_pearson": "test Pearson",
                "test_spearman": "test Spearman",
            }
        )
        lines.extend(
            [
                "## Mixed Split Results",
                "",
                "These are separate from the fold-locked sweep above. They use the `mixed` split",
                "with `resnet18` only and are kept as auxiliary QA behavior checks.",
                "",
                mixed_display.to_markdown(index=False),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize saved QA sweep evaluation CSVs into a markdown report."
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("hpc_run_results/qa_campaigns/qa_sweep"),
        help="Root directory containing the exported architecture sweep MLflow store.",
    )
    parser.add_argument(
        "--mixed-root",
        type=Path,
        default=Path("hpc_run_results/qa_campaigns/qa_mixed"),
        help="Root directory containing the exported mixed-split MLflow store.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the markdown report.",
    )
    args = parser.parse_args()

    summary = load_summary_table(args.sweep_root)
    mixed_summary = None
    if args.mixed_root.exists():
        mixed_summary = load_summary_table(args.mixed_root)
    markdown = build_markdown(
        summary,
        args.sweep_root,
        mixed_summary=mixed_summary,
        mixed_root=args.mixed_root if args.mixed_root.exists() else None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote QA sweep report to {args.output}")


if __name__ == "__main__":
    main()
