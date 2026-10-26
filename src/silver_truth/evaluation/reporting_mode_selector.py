from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DISPLAY_COLUMNS = [
    ("dataset_label", "Dataset"),
    ("fold_label", "Fold"),
    ("selected_mode", "Selected mode"),
    ("selected_method", "Selected method"),
    ("selection_validation_iou", "Validation IoU"),
    ("selected_test_iou", "Test IoU"),
    ("strict_qa_ensemble_iou", "Strict val-selected"),
    ("full_deployable_iou", "Full val-selected"),
    ("best_fusion_iou", "Best fusion-only"),
    ("silver_truth_iou", "Silver truth"),
]

FOLD_SORT_ORDER = {"fold-1": 0, "fold-2": 1, "mean": 2}


def build_reporting_mode_candidates(
    *,
    full_combined: pd.DataFrame,
    strict_combined: pd.DataFrame,
    soft_strategy_summaries: Iterable[tuple[str, pd.DataFrame]] = (),
) -> pd.DataFrame:
    metadata = _metadata_from_combined(full_combined)
    candidates = [
        _combined_mode_candidates(
            full_combined,
            mode="full_deployable",
            mode_label="Full deployable",
            strict_eligible=False,
            taxonomy_note=(
                "Deployable table mode; may include full_pipeline, but never "
                "counts fusion_only as ours."
            ),
        ),
        _combined_mode_candidates(
            strict_combined,
            mode="strict_qa_ensemble",
            mode_label="Strict QA/ensemble",
            strict_eligible=True,
            taxonomy_note=(
                "Strict table mode; only QA/ensemble families are eligible."
            ),
        ),
    ]
    for mode, summary in soft_strategy_summaries:
        candidates.append(_soft_strategy_candidates(summary, metadata, mode=mode))

    out = pd.concat(candidates, ignore_index=True)
    out = out.dropna(subset=["selected_test_iou"], how="all")
    return _sort_rows(out)


def validation_select_reporting_modes(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "crop_tag",
        "variant",
        "fold",
        "mode",
        "selection_validation_iou",
        "selected_test_iou",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"Reporting mode candidates are missing columns: {missing}")

    fold_rows = candidates[candidates["row_type"].eq("fold")].copy()
    fold_rows = fold_rows.dropna(
        subset=["selection_validation_iou", "selected_test_iou"]
    )
    if fold_rows.empty:
        return pd.DataFrame()

    selected = (
        fold_rows.sort_values(
            [
                "dataset",
                "crop_tag",
                "variant",
                "fold",
                "selection_validation_iou",
                "selected_test_iou",
                "mode",
            ],
            ascending=[True, True, True, True, False, False, True],
        )
        .groupby(["dataset", "crop_tag", "variant", "fold"], as_index=False)
        .first()
    )
    selected["selection_rule"] = "max_validation_iou_across_reporting_modes"
    selected["selected_mode"] = selected["mode_label"]
    selected = _add_reference_mode_iou(selected, candidates)

    mean_rows = _mean_rows(selected)
    return _sort_rows(pd.concat([selected, mean_rows], ignore_index=True))


def write_reporting_mode_selection(
    *,
    full_combined_csv: Path,
    strict_combined_csv: Path,
    soft_strategy_paths: Iterable[tuple[str, Path]],
    output_dir: Path,
) -> dict[str, Path]:
    full_combined = pd.read_csv(full_combined_csv)
    strict_combined = pd.read_csv(strict_combined_csv)
    soft_summaries = [
        (mode, pd.read_csv(path)) for mode, path in soft_strategy_paths
    ]
    candidates = build_reporting_mode_candidates(
        full_combined=full_combined,
        strict_combined=strict_combined,
        soft_strategy_summaries=soft_summaries,
    )
    selected = validation_select_reporting_modes(candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidates": output_dir / "reporting_mode_candidates.csv",
        "selected": output_dir / "validation_selected_reporting_mode.csv",
        "selected_markdown": output_dir / "validation_selected_reporting_mode.md",
    }
    candidates.to_csv(paths["candidates"], index=False)
    selected.to_csv(paths["selected"], index=False)
    paths["selected_markdown"].write_text(_render_selected_table(selected))
    return paths


def _metadata_from_combined(combined: pd.DataFrame) -> pd.DataFrame:
    fold_rows = combined[combined["row_type"].astype(str).eq("fold")].copy()
    columns = [
        "dataset",
        "crop_tag",
        "variant",
        "fold",
        "best_fusion_iou",
        "best_competitor_iou",
        "median_competitor_iou",
        "worst_competitor_iou",
        "silver_truth_iou",
    ]
    present = [column for column in columns if column in fold_rows.columns]
    return fold_rows[present].drop_duplicates(
        subset=["dataset", "fold"], keep="last"
    )


def _combined_mode_candidates(
    combined: pd.DataFrame,
    *,
    mode: str,
    mode_label: str,
    strict_eligible: bool,
    taxonomy_note: str,
) -> pd.DataFrame:
    fold_rows = combined[combined["row_type"].astype(str).eq("fold")].copy()
    out = _base_candidate_columns(fold_rows)
    out["mode"] = mode
    out["mode_label"] = mode_label
    out["strict_table_eligible"] = strict_eligible
    out["taxonomy_note"] = taxonomy_note
    out["selected_method"] = fold_rows.get("validation_selected_ours_method")
    out["selected_test_iou"] = pd.to_numeric(
        fold_rows.get("validation_selected_ours_iou"), errors="coerce"
    ).fillna(pd.to_numeric(fold_rows.get("best_ours_iou"), errors="coerce"))
    out["selection_validation_iou"] = pd.to_numeric(
        fold_rows.get("validation_selected_validation_iou"), errors="coerce"
    )
    return out


def _soft_strategy_candidates(
    summary: pd.DataFrame, metadata: pd.DataFrame, *, mode: str
) -> pd.DataFrame:
    rows = summary.copy()
    rows = rows.rename(
        columns={
            "strategy": "soft_strategy",
            "preset": "soft_preset",
            "validation_mean_iou": "selection_validation_iou",
            "test_mean_iou": "selected_test_iou",
            "test_mean_f1": "selected_test_f1",
        }
    )
    rows = rows.merge(metadata, on=["dataset", "fold"], how="left")
    if "crop_tag" not in rows.columns:
        rows["crop_tag"] = ""
    if "variant" not in rows.columns:
        rows["variant"] = ""

    out = _base_candidate_columns(rows)
    out["mode"] = mode
    out["mode_label"] = mode.replace("_", " ")
    out["strict_table_eligible"] = False
    out["taxonomy_note"] = (
        "Soft-hybrid deployable candidate mode; not fusion_only, but not part "
        "of the strict QA/ensemble-only table."
    )
    out["selected_method"] = (
        rows.get("soft_strategy", mode).astype(str)
        + ":"
        + rows.get("soft_preset", "").astype(str)
    )
    out["selected_test_iou"] = pd.to_numeric(
        rows.get("selected_test_iou"), errors="coerce"
    )
    out["selection_validation_iou"] = pd.to_numeric(
        rows.get("selection_validation_iou"), errors="coerce"
    )
    return out


def _base_candidate_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    for column in (
        "dataset",
        "crop_tag",
        "variant",
        "fold",
        "best_fusion_iou",
        "best_competitor_iou",
        "median_competitor_iou",
        "worst_competitor_iou",
        "silver_truth_iou",
    ):
        out[column] = rows[column] if column in rows.columns else pd.NA
    out["row_type"] = "fold"
    return out


def _mean_rows(selected: pd.DataFrame) -> pd.DataFrame:
    records = []
    numeric_columns = [
        "best_fusion_iou",
        "best_competitor_iou",
        "median_competitor_iou",
        "worst_competitor_iou",
        "silver_truth_iou",
        "selection_validation_iou",
        "selected_test_iou",
    ]
    for (dataset, crop_tag, variant), group in selected.groupby(
        ["dataset", "crop_tag", "variant"], dropna=False
    ):
        record: dict[str, object] = {
            "dataset": dataset,
            "crop_tag": crop_tag,
            "variant": variant,
            "fold": "mean",
            "row_type": "mean",
            "mode": "validation_selected",
            "mode_label": "Validation selected",
            "selected_mode": "Validation selected",
            "selected_method": "validation-selected per fold",
            "strict_table_eligible": False,
            "taxonomy_note": "Mean of validation-selected reporting-mode rows.",
            "selection_rule": "mean_of_selected_folds",
        }
        for column in numeric_columns:
            record[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
        for column in ("strict_qa_ensemble_iou", "full_deployable_iou"):
            record[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
        records.append(record)
    return pd.DataFrame.from_records(records)


def _add_reference_mode_iou(
    selected: pd.DataFrame, candidates: pd.DataFrame
) -> pd.DataFrame:
    out = selected.copy()
    key_columns = ["dataset", "crop_tag", "variant", "fold"]
    for mode, column in (
        ("strict_qa_ensemble", "strict_qa_ensemble_iou"),
        ("full_deployable", "full_deployable_iou"),
    ):
        reference = candidates[candidates["mode"].eq(mode)][
            key_columns + ["selected_test_iou"]
        ].rename(columns={"selected_test_iou": column})
        out = out.merge(reference, on=key_columns, how="left")
    return out


def _sort_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["_fold_sort"] = out["fold"].map(FOLD_SORT_ORDER).fillna(99)
    return (
        out.sort_values(["dataset", "crop_tag", "variant", "_fold_sort", "mode"])
        .drop(columns=["_fold_sort"])
        .reset_index(drop=True)
    )


def _format_score(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    return f"{float(number):.4f}"


def _render_selected_table(selected: pd.DataFrame) -> str:
    if selected.empty:
        return ""
    rendered = selected.copy()
    rendered["dataset_label"] = rendered.apply(
        lambda row: f"{row['dataset']} ({row['crop_tag']})", axis=1
    )
    rendered["fold_label"] = rendered["fold"].map(
        lambda value: str(value).removeprefix("fold-").replace("mean", "Mean")
    )
    display = rendered[[source for source, _ in DISPLAY_COLUMNS]].copy()
    for column, _ in DISPLAY_COLUMNS:
        if column.endswith("_iou"):
            display[column] = display[column].map(_format_score)
    return display.rename(columns=dict(DISPLAY_COLUMNS)).to_markdown(
        index=False, disable_numparse=True
    )


def _parse_mode_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    mode, path = value.split("=", 1)
    return mode, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build validation-selected reporting-mode tables."
    )
    parser.add_argument("--full-combined-csv", required=True, type=Path)
    parser.add_argument("--strict-combined-csv", required=True, type=Path)
    parser.add_argument(
        "--soft-strategy",
        action="append",
        default=[],
        help=(
            "Soft strategy selection CSV as mode_name=path. Can be repeated. "
            "Expected input is validation_selected_strategy.csv."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    paths = write_reporting_mode_selection(
        full_combined_csv=args.full_combined_csv,
        strict_combined_csv=args.strict_combined_csv,
        soft_strategy_paths=[_parse_mode_path(value) for value in args.soft_strategy],
        output_dir=args.output_dir,
    )
    for key, path in paths.items():
        print(f"Wrote {key}: {path}")


if __name__ == "__main__":
    main()
