from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PRESET_PRIORITY = {
    "balanced": 0,
    "support_qa": 1,
    "fusion_heavy": 2,
    "qa_heavy": 3,
    "temporal_heavy": 4,
    "no_qa": 5,
}


def load_strategy_summaries(
    inputs: Iterable[tuple[str, Path]],
) -> pd.DataFrame:
    frames = []
    for strategy, path in inputs:
        df = pd.read_csv(path)
        df = df.copy()
        df["strategy"] = strategy
        df["dataset"] = df["input_path"].astype(str).str.extract(
            r"paper_inputs/([^/]+)/"
        )
        df["fold"] = df["input_path"].astype(str).str.extract(
            r"(fold-\d+)_paper_ready"
        )
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def validation_select_strategies(
    summary: pd.DataFrame,
    *,
    validation_epsilon: float = 0.01,
) -> pd.DataFrame:
    if validation_epsilon < 0:
        raise ValueError("validation_epsilon must be non-negative.")

    required = {"dataset", "fold", "strategy", "preset", "split", "mean_iou"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Strategy summary is missing columns: {missing}")

    validation = _split_rows(summary, "validation").rename(
        columns={"mean_iou": "validation_mean_iou", "mean_f1": "validation_mean_f1"}
    )
    test = _split_rows(summary, "test").rename(
        columns={"mean_iou": "test_mean_iou", "mean_f1": "test_mean_f1"}
    )
    merged = validation.merge(
        test,
        on=["dataset", "fold", "strategy", "preset", "input_path", "output_dir"],
        how="inner",
        suffixes=("", "_test"),
    )
    if merged.empty:
        return pd.DataFrame()

    merged["_best_validation_mean_iou"] = merged.groupby(
        ["dataset", "fold"]
    )["validation_mean_iou"].transform("max")
    merged["_validation_gap"] = (
        merged["_best_validation_mean_iou"] - merged["validation_mean_iou"]
    )
    merged["_preset_priority"] = (
        merged["preset"].map(PRESET_PRIORITY).fillna(len(PRESET_PRIORITY)).astype(int)
    )
    eligible = merged[merged["_validation_gap"] <= validation_epsilon].copy()

    selected = (
        eligible.sort_values(
            [
                "dataset",
                "fold",
                "_preset_priority",
                "strategy",
                "validation_mean_iou",
                "test_mean_iou",
            ],
            ascending=[True, True, True, True, False, False],
        )
        .groupby(["dataset", "fold"], as_index=False)
        .first()
    )
    selected["selection_rule"] = (
        "validation_epsilon_conservative_preset:"
        + f"{validation_epsilon:g}"
    )
    return selected[
        [
            "dataset",
            "fold",
            "strategy",
            "preset",
            "selection_rule",
            "validation_mean_iou",
            "validation_mean_f1",
            "test_mean_iou",
            "test_mean_f1",
            "input_path",
            "output_dir",
        ]
    ]


def summarize_selected_by_dataset(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    return (
        selected.groupby("dataset", as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_validation_iou=("validation_mean_iou", "mean"),
            mean_test_iou=("test_mean_iou", "mean"),
            mean_test_f1=("test_mean_f1", "mean"),
            selected_strategies=("strategy", lambda s: ";".join(map(str, s))),
            selected_presets=("preset", lambda s: ";".join(map(str, s))),
        )
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def write_validation_selection(
    *,
    inputs: Iterable[tuple[str, Path]],
    output_dir: Path,
    validation_epsilon: float = 0.005,
) -> dict[str, Path]:
    summary = load_strategy_summaries(inputs)
    selected = validation_select_strategies(
        summary, validation_epsilon=validation_epsilon
    )
    dataset_summary = summarize_selected_by_dataset(selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_candidates": output_dir / "strategy_candidates.csv",
        "selected": output_dir / "validation_selected_strategy.csv",
        "dataset_summary": output_dir / "validation_selected_dataset_summary.csv",
    }
    summary.to_csv(paths["all_candidates"], index=False)
    selected.to_csv(paths["selected"], index=False)
    dataset_summary.to_csv(paths["dataset_summary"], index=False)
    return paths


def _split_rows(summary: pd.DataFrame, split: str) -> pd.DataFrame:
    keep_columns = [
        "dataset",
        "fold",
        "strategy",
        "preset",
        "split",
        "mean_iou",
        "mean_f1",
        "input_path",
        "output_dir",
    ]
    present = [column for column in keep_columns if column in summary.columns]
    return summary[summary["split"].astype(str).eq(split)][present].copy()


def _parse_strategy_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        strategy = re.sub(r"[^A-Za-z0-9_]+", "_", path.parent.name).strip("_")
        return strategy or path.stem, path
    strategy, path = value.split("=", 1)
    return strategy, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select soft-hybrid strategy/preset by validation IoU."
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Either path/to/summary.csv or strategy_name=path/to/summary.csv.",
    )
    parser.add_argument(
        "--validation-epsilon",
        type=float,
        default=0.01,
        help=(
            "Validation IoU tolerance for conservative preset tie-breaking. "
            "Defaults to 0.01."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    paths = write_validation_selection(
        inputs=[_parse_strategy_arg(value) for value in args.summary],
        output_dir=args.output_dir,
        validation_epsilon=args.validation_epsilon,
    )
    for key, path in paths.items():
        print(f"Wrote {key}: {path}")


if __name__ == "__main__":
    main()
