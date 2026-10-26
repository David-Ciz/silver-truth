#!/usr/bin/env python3
"""Build combined paper result tables from available run summaries.

This script refreshes the combined paper-result summary from the available HPC
run summaries plus the existing combined CSV, then writes:

- a full combined CSV/Markdown table
- a QA/Ensemble-only CSV/Markdown table where best_ours excludes fusion methods

The QA/Ensemble-only table only allows methods from these families:
- qa_only
- ensemble_only
- ensemble_qa
- ensemble_qa_retrained
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DISPLAY_COLUMNS = [
    ("dataset_label", "Dataset"),
    ("fold_label", "Fold"),
    ("best_fusion_iou", "Best fusion-only"),
    ("best_competitor_iou", "Best competitor"),
    ("median_competitor_iou", "Median competitor"),
    ("worst_competitor_iou", "Worst competitor"),
    ("silver_truth_iou", "Silver truth"),
    ("best_ours_iou", "Best ours"),
    ("best_ours_method", "Ours method"),
]

SUMMARY_KEY_COLUMNS = ["dataset", "crop_tag", "variant", "fold"]
FOLD_SORT_ORDER = {"fold-1": 0, "fold-2": 1, "mean": 2}
ALLOWED_QA_ENSEMBLE_FAMILIES = {
    "qa_only",
    "ensemble_only",
    "ensemble_qa",
    "ensemble_qa_retrained",
}
OURS_DEPLOYABLE_FAMILIES = ALLOWED_QA_ENSEMBLE_FAMILIES | {
    "full_pipeline",
    "soft_hybrid",
}


def _format_score(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return ""
    return f"{float(number):.4f}"


def _render_table(summary: pd.DataFrame) -> str:
    rendered = summary.copy()
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
    display = display.rename(columns=dict(DISPLAY_COLUMNS))
    return display.to_markdown(index=False, disable_numparse=True)


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.copy()
    ordered["_fold_sort"] = ordered["fold"].map(FOLD_SORT_ORDER).fillna(99)
    ordered = ordered.sort_values(["dataset", "crop_tag", "variant", "_fold_sort"])
    return ordered.drop(columns=["_fold_sort"]).reset_index(drop=True)


def _load_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["_source_summary"] = str(path)
    frame["_paper_runs_root"] = str(path.parent.parent)
    return frame


def _build_full_combined(existing_combined_csv: Path, repo_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if existing_combined_csv.exists():
        frames.append(_load_summary(existing_combined_csv))

    run_summaries = sorted(repo_root.glob("hpc_results/**/paper_result_summary.csv"))
    for summary_path in run_summaries:
        frames.append(_load_summary(summary_path))

    if not frames:
        raise FileNotFoundError("No paper_result_summary.csv inputs found.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=SUMMARY_KEY_COLUMNS, keep="last")
    combined = combined.drop(columns=["_source_summary"], errors="ignore")
    combined = _recompute_ours_from_ablation(
        combined,
        repo_root,
        allowed_families=OURS_DEPLOYABLE_FAMILIES,
        mean_method_label="foldwise best",
        validation_mean_method_label="validation-selected per fold",
    )
    return _sort_summary(combined)


def _resolve_ablation_summary_path(path_str: object, repo_root: Path) -> Path | None:
    if not isinstance(path_str, str) or not path_str.strip():
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _map_to_local_paper_runs(path: Path, paper_runs_root: Path | None) -> Path | None:
    if paper_runs_root is None:
        return None
    parts = list(path.parts)
    if "paper_runs" not in parts:
        return None
    paper_runs_index = parts.index("paper_runs")
    relative_parts = parts[paper_runs_index + 1 :]
    return paper_runs_root.joinpath(*relative_parts)


def _resolve_ablation_summary_path_with_fallback(
    path_str: object,
    repo_root: Path,
    paper_runs_root: object,
) -> Path | None:
    resolved = _resolve_ablation_summary_path(path_str, repo_root)
    if resolved is None:
        return None
    if resolved.exists():
        return resolved

    local_root = None
    if isinstance(paper_runs_root, str) and paper_runs_root.strip():
        local_root = Path(paper_runs_root)
    candidate = _map_to_local_paper_runs(resolved, local_root)
    if candidate is not None and candidate.exists():
        return candidate
    return resolved


def _select_best_qa_ensemble_method(
    ablation_summary_path: Path | None,
) -> tuple[float | None, str | None]:
    if ablation_summary_path is None or not ablation_summary_path.exists():
        return None, None

    summary = pd.read_csv(ablation_summary_path)
    candidates = summary[
        summary["family"].isin(ALLOWED_QA_ENSEMBLE_FAMILIES)
    ].copy()
    if candidates.empty:
        return None, None

    test_iou = pd.to_numeric(candidates.get("test_mean_iou"), errors="coerce")
    mean_iou = pd.to_numeric(candidates.get("mean_iou"), errors="coerce")
    candidates["selection_iou"] = test_iou.fillna(mean_iou)
    candidates = candidates.sort_values(
        ["selection_iou", "method_key"], ascending=[False, True]
    )
    best = candidates.iloc[0]
    value = pd.to_numeric(best["selection_iou"], errors="coerce")
    if pd.isna(value):
        return None, None
    return float(value), str(best["method_key"])


def _select_best_method(
    ablation_summary_path: Path | None,
    *,
    allowed_families: set[str],
) -> tuple[float | None, str | None]:
    if ablation_summary_path is None or not ablation_summary_path.exists():
        return None, None

    summary = pd.read_csv(ablation_summary_path)
    candidates = summary[summary["family"].isin(allowed_families)].copy()
    if candidates.empty:
        return None, None

    test_iou = pd.to_numeric(candidates.get("test_mean_iou"), errors="coerce")
    mean_iou = pd.to_numeric(candidates.get("mean_iou"), errors="coerce")
    candidates["selection_iou"] = test_iou.fillna(mean_iou)
    candidates = candidates.dropna(subset=["selection_iou"])
    if candidates.empty:
        return None, None

    candidates = candidates.sort_values(
        ["selection_iou", "method_key"], ascending=[False, True]
    )
    best = candidates.iloc[0]
    return float(best["selection_iou"]), str(best["method_key"])


def _select_validation_method(
    ablation_summary_path: Path | None,
    *,
    allowed_families: set[str],
) -> tuple[float | None, str | None, float | None]:
    if ablation_summary_path is None or not ablation_summary_path.exists():
        return None, None, None

    summary = pd.read_csv(ablation_summary_path)
    candidates = summary[summary["family"].isin(allowed_families)].copy()
    if candidates.empty:
        return None, None, None

    candidates["selection_iou"] = pd.to_numeric(
        candidates.get("test_mean_iou"), errors="coerce"
    ).fillna(pd.to_numeric(candidates.get("mean_iou"), errors="coerce"))
    candidates["validation_selection_iou"] = pd.to_numeric(
        candidates.get("validation_mean_iou"), errors="coerce"
    )
    candidates = candidates.dropna(
        subset=["selection_iou", "validation_selection_iou"]
    )
    if candidates.empty:
        return None, None, None

    candidates = candidates.sort_values(
        ["validation_selection_iou", "method_key"], ascending=[False, True]
    )
    best = candidates.iloc[0]
    return (
        float(best["selection_iou"]),
        str(best["method_key"]),
        float(best["validation_selection_iou"]),
    )


def _select_validation_qa_ensemble_method(
    ablation_summary_path: Path | None,
) -> tuple[float | None, str | None, float | None]:
    if ablation_summary_path is None or not ablation_summary_path.exists():
        return None, None, None

    summary = pd.read_csv(ablation_summary_path)
    candidates = summary[
        summary["family"].isin(ALLOWED_QA_ENSEMBLE_FAMILIES)
    ].copy()
    if candidates.empty:
        return None, None, None

    candidates["selection_iou"] = pd.to_numeric(
        candidates.get("test_mean_iou"), errors="coerce"
    ).fillna(pd.to_numeric(candidates.get("mean_iou"), errors="coerce"))
    candidates["validation_selection_iou"] = pd.to_numeric(
        candidates.get("validation_mean_iou"), errors="coerce"
    )
    candidates = candidates.dropna(
        subset=["selection_iou", "validation_selection_iou"]
    )
    if candidates.empty:
        return None, None, None

    candidates = candidates.sort_values(
        ["validation_selection_iou", "method_key"], ascending=[False, True]
    )
    best = candidates.iloc[0]
    test_value = pd.to_numeric(best["selection_iou"], errors="coerce")
    validation_value = pd.to_numeric(best["validation_selection_iou"], errors="coerce")
    if pd.isna(test_value) or pd.isna(validation_value):
        return None, None, None
    return float(test_value), str(best["method_key"]), float(validation_value)


def _recompute_ours_from_ablation(
    summary: pd.DataFrame,
    repo_root: Path,
    *,
    allowed_families: set[str],
    mean_method_label: str,
    validation_mean_method_label: str,
) -> pd.DataFrame:
    updated = summary.copy()
    fold_mask = updated["row_type"].eq("fold")

    for idx, row in updated[fold_mask].iterrows():
        ablation_summary_path = _resolve_ablation_summary_path_with_fallback(
            row.get("ablation_method_summary_path"),
            repo_root,
            row.get("_paper_runs_root"),
        )
        best_iou, best_method = _select_best_method(
            ablation_summary_path, allowed_families=allowed_families
        )
        if best_iou is not None and best_method is not None:
            updated.at[idx, "best_ours_iou"] = best_iou
            updated.at[idx, "best_ours_method"] = best_method

        (
            selected_iou,
            selected_method,
            selected_validation_iou,
        ) = _select_validation_method(
            ablation_summary_path, allowed_families=allowed_families
        )
        if (
            selected_iou is not None
            and selected_method is not None
            and selected_validation_iou is not None
        ):
            updated.at[idx, "validation_selected_ours_iou"] = selected_iou
            updated.at[idx, "validation_selected_ours_method"] = selected_method
            updated.at[
                idx, "validation_selected_validation_iou"
            ] = selected_validation_iou

    fold_rows = updated[fold_mask].copy()
    mean_mask = updated["row_type"].eq("mean")
    for idx, mean_row in updated[mean_mask].iterrows():
        selector = (
            (fold_rows["dataset"] == mean_row["dataset"])
            & (fold_rows["crop_tag"] == mean_row["crop_tag"])
            & (fold_rows["variant"] == mean_row["variant"])
        )
        grouped_rows = fold_rows[selector]
        if grouped_rows.empty:
            continue
        updated.at[idx, "best_ours_iou"] = float(
            pd.to_numeric(grouped_rows["best_ours_iou"], errors="coerce").mean()
        )
        updated.at[idx, "best_ours_method"] = mean_method_label
        updated.at[idx, "validation_selected_ours_iou"] = float(
            pd.to_numeric(
                grouped_rows["validation_selected_ours_iou"], errors="coerce"
            ).mean()
        )
        updated.at[idx, "validation_selected_ours_method"] = (
            validation_mean_method_label
        )
        updated.at[idx, "validation_selected_validation_iou"] = float(
            pd.to_numeric(
                grouped_rows["validation_selected_validation_iou"], errors="coerce"
            ).mean()
        )

    return updated


def _build_qa_ensemble_only(summary: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    filtered = summary.copy()
    fold_mask = filtered["row_type"].eq("fold")

    best_replacements: list[tuple[int, float, str]] = []
    validation_replacements: list[tuple[int, float, str, float]] = []
    for idx, row in filtered[fold_mask].iterrows():
        ablation_summary_path = _resolve_ablation_summary_path_with_fallback(
            row.get("ablation_method_summary_path"),
            repo_root,
            row.get("_paper_runs_root"),
        )
        best_iou, best_method = _select_best_qa_ensemble_method(ablation_summary_path)
        if best_iou is not None and best_method is not None:
            best_replacements.append((idx, best_iou, best_method))

        (
            selected_iou,
            selected_method,
            selected_validation_iou,
        ) = _select_validation_qa_ensemble_method(ablation_summary_path)
        if (
            selected_iou is not None
            and selected_method is not None
            and selected_validation_iou is not None
        ):
            validation_replacements.append(
                (idx, selected_iou, selected_method, selected_validation_iou)
            )

    for idx, best_iou, best_method in best_replacements:
        filtered.at[idx, "best_ours_iou"] = best_iou
        filtered.at[idx, "best_ours_method"] = best_method

    for (
        idx,
        selected_iou,
        selected_method,
        selected_validation_iou,
    ) in validation_replacements:
        filtered.at[idx, "validation_selected_ours_iou"] = selected_iou
        filtered.at[idx, "validation_selected_ours_method"] = selected_method
        filtered.at[idx, "validation_selected_validation_iou"] = selected_validation_iou

    mean_mask = filtered["row_type"].eq("mean")
    fold_rows = filtered[fold_mask].copy()
    for idx, mean_row in filtered[mean_mask].iterrows():
        selector = (
            (fold_rows["dataset"] == mean_row["dataset"])
            & (fold_rows["crop_tag"] == mean_row["crop_tag"])
            & (fold_rows["variant"] == mean_row["variant"])
        )
        grouped_rows = fold_rows[selector]
        if grouped_rows.empty:
            continue
        mean_best_ours = pd.to_numeric(
            grouped_rows["best_ours_iou"], errors="coerce"
        ).mean()
        mean_validation_selected = pd.to_numeric(
            grouped_rows["validation_selected_ours_iou"], errors="coerce"
        ).mean()
        mean_validation_selection = pd.to_numeric(
            grouped_rows["validation_selected_validation_iou"], errors="coerce"
        ).mean()
        filtered.at[idx, "best_ours_iou"] = float(mean_best_ours)
        filtered.at[idx, "best_ours_method"] = "foldwise best qa/ensemble-only"
        filtered.at[idx, "validation_selected_ours_iou"] = float(
            mean_validation_selected
        )
        filtered.at[idx, "validation_selected_ours_method"] = (
            "validation-selected qa/ensemble-only per fold"
        )
        filtered.at[idx, "validation_selected_validation_iou"] = float(
            mean_validation_selection
        )

    return _sort_summary(filtered)


def _write_outputs(summary: pd.DataFrame, csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    persisted = summary.drop(columns=["_source_summary", "_paper_runs_root"], errors="ignore")
    persisted.to_csv(csv_path, index=False)
    md_path.write_text(_render_table(persisted) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build combined paper result tables from available summaries."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=Path("hpc_results/combined/reports/paper_result_summary_all_available.csv"),
        help="Existing combined CSV used as a base for older completed runs.",
    )
    parser.add_argument(
        "--qa-ensemble-only-csv",
        type=Path,
        default=Path(
            "hpc_results/combined/reports/paper_result_summary_all_available_qa_ensemble_only.csv"
        ),
        help="Output CSV path for the QA/Ensemble-only best-ours summary.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    combined_csv = (
        args.combined_csv
        if args.combined_csv.is_absolute()
        else (repo_root / args.combined_csv).resolve()
    )
    qa_only_csv = (
        args.qa_ensemble_only_csv
        if args.qa_ensemble_only_csv.is_absolute()
        else (repo_root / args.qa_ensemble_only_csv).resolve()
    )

    full_summary = _build_full_combined(combined_csv, repo_root)
    full_md = combined_csv.with_suffix(".md")
    _write_outputs(full_summary, combined_csv, full_md)

    qa_ensemble_summary = _build_qa_ensemble_only(full_summary, repo_root)
    qa_only_md = qa_only_csv.with_suffix(".md")
    _write_outputs(qa_ensemble_summary, qa_only_csv, qa_only_md)

    print("Full combined table:")
    print(_render_table(full_summary))
    print(f"\nWrote full CSV: {combined_csv}")
    print(f"Wrote full Markdown: {full_md}")
    print("\nQA/Ensemble-only table:")
    print(_render_table(qa_ensemble_summary))
    print(f"\nWrote QA/Ensemble-only CSV: {qa_only_csv}")
    print(f"Wrote QA/Ensemble-only Markdown: {qa_only_md}")


if __name__ == "__main__":
    main()
