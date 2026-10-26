from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

DEFAULT_PHASE1_ROOT = Path("hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe")
DEFAULT_BEST_BY_DATASET = (
    DEFAULT_PHASE1_ROOT / "all_available_simple_policy_best_by_dataset.csv"
)
DEFAULT_OUTPUT_ROOT = Path("hpc_results/soft_hybrid_probe/qa_pruned_ensemble")
DEFAULT_FEATURE_ROOT = Path(
    "hpc_results/soft_hybrid_probe/"
    "all_available_square_rect_full_image_local_support_pairwise_source_trainonly/"
    "features"
)

STRICT_ENSEMBLE_ROW_FAMILIES = {
    "qa_only",
    "ensemble_only",
    "ensemble_qa",
    "ensemble_qa_retrained",
}


def add_repo_paths() -> None:
    for path in (REPO_ROOT, SRC_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def discover_best_policy_jobs(
    phase1_root: Path = DEFAULT_PHASE1_ROOT,
    best_by_dataset: Path = DEFAULT_BEST_BY_DATASET,
) -> list[dict[str, Any]]:
    add_repo_paths()
    from scripts.run_ranker_pruned_non_simple_sweep import discover_jobs

    return discover_jobs(phase1_root, best_by_dataset)


def discover_policy_jobs(
    phase1_root: Path = DEFAULT_PHASE1_ROOT,
    *,
    policies: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected = set(policies or ())
    jobs: list[dict[str, Any]] = []
    for fold_dir in sorted(phase1_root.glob("*/*/*/fold-*")):
        dataset = fold_dir.parents[2].name
        crop_tag = fold_dir.parents[1].name
        variant = fold_dir.parents[0].name
        for run_dir in sorted(fold_dir.glob("*_simple_escalated")):
            policy = _policy_from_run_dir(run_dir)
            if selected and policy not in selected:
                continue
            filtered = _find_filtered_parquet(run_dir, policy)
            if filtered is None:
                continue
            jobs.append(
                {
                    "dataset": dataset,
                    "crop_tag": crop_tag,
                    "variant": variant,
                    "fold": fold_dir.name,
                    "policy": policy,
                    "filtered_parquet": filtered,
                    "previous_comparison": run_dir / "ranker_pruned_fusion_comparison.csv",
                    "pruning_summary": run_dir / "ranker_pruning_summary.csv",
                }
            )
    return jobs


def feature_path_for_job(job: dict[str, Any], feature_root: Path) -> Path:
    return (
        feature_root
        / str(job["dataset"])
        / str(job["crop_tag"])
        / str(job["variant"])
        / f"{job['fold']}_paper_ready_pairwise_ranked.parquet"
    )


def output_dir_for_job(
    *,
    output_root: Path,
    job: dict[str, Any],
    ensemble_variant: str,
    threshold: float,
) -> Path:
    threshold_token = f"thr{int(round(threshold * 100)):03d}"
    return (
        output_root
        / str(job["dataset"])
        / str(job["crop_tag"])
        / str(job["variant"])
        / str(job["fold"])
        / str(job["policy"])
        / f"{ensemble_variant}_{threshold_token}"
    )


def normalize_weighted_vote_outputs(
    *,
    output_dir: Path,
    job: dict[str, Any],
    ensemble_variant: str,
    method_key: str,
    method_label: str,
    result_family: str,
    row_family: str = "ensemble_qa",
    soft_gate: str = "",
) -> Path:
    for filename in (
        "weighted_object_vote_summary.csv",
        "weighted_object_vote_comparison.csv",
    ):
        path = output_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = normalize_result_rows(
            frame,
            job=job,
            ensemble_variant=ensemble_variant,
            method_key=method_key,
            method_label=method_label,
            result_family=result_family,
            row_family=row_family,
            soft_gate=soft_gate,
        )
        frame.to_csv(path, index=False)
    return output_dir / "weighted_object_vote_comparison.csv"


def normalize_result_rows(
    frame: pd.DataFrame,
    *,
    job: dict[str, Any],
    ensemble_variant: str,
    method_key: str,
    method_label: str,
    result_family: str,
    row_family: str = "ensemble_qa",
    soft_gate: str = "",
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["row_family"] = row_family
    normalized["method_key"] = method_key
    normalized["method_label"] = method_label
    normalized["result_family"] = result_family
    normalized["ensemble_variant"] = ensemble_variant
    normalized["qa_pruned_pool"] = True
    normalized["qa_pruning_policy"] = str(job.get("policy", ""))
    normalized["soft_gate"] = soft_gate
    normalized["source_candidate_parquet"] = str(job.get("filtered_parquet", ""))
    for column in ("dataset", "crop_tag", "variant", "fold"):
        if column in job:
            normalized[column] = str(job[column])
    if "filtered_parquet" in job:
        normalized["input_path"] = str(job["filtered_parquet"])
    return normalized


def combine_comparison_paths(
    comparison_paths: Iterable[Path],
    *,
    output_root: Path,
    prefix: str,
    context_path: Path | None = DEFAULT_BEST_BY_DATASET,
) -> dict[str, Path]:
    paths = [path for path in comparison_paths if path.exists()]
    if not paths:
        raise SystemExit("No comparison rows were produced.")

    output_root.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    dataset_summary = summarize_dataset(rows)
    overall_summary = summarize_overall(dataset_summary)

    combined_path = output_root / f"{prefix}_comparison_rows.csv"
    dataset_path = output_root / f"{prefix}_dataset_summary.csv"
    overall_path = output_root / f"{prefix}_overall_summary.csv"
    report_path = output_root / f"{prefix}_summary.md"

    rows.to_csv(combined_path, index=False)
    dataset_summary.to_csv(dataset_path, index=False)
    overall_summary.to_csv(overall_path, index=False)
    write_markdown_report(
        output_path=report_path,
        title=prefix.replace("_", " ").title(),
        dataset_summary=dataset_summary,
        overall_summary=overall_summary,
        context=load_simple_context(context_path) if context_path else pd.DataFrame(),
    )
    return {
        "combined": combined_path,
        "dataset_summary": dataset_path,
        "overall_summary": overall_path,
        "report": report_path,
    }


def summarize_dataset(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "result_family",
        "row_family",
        "method_key",
        "method_label",
        "ensemble_variant",
        "qa_pruning_policy",
        "soft_gate",
        "dataset",
        "crop_tag",
        "variant",
        "weights_column",
        "boundary_support_column",
        "boundary_floor",
        "boundary_floor_column",
        "boundary_iterations",
        "top_k_per_group",
        "threshold",
    ]
    present = [column for column in group_cols if column in rows.columns]
    return (
        rows.groupby(present, dropna=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            n_folds=("fold", "nunique"),
            n_cells=("n_cells", "sum") if "n_cells" in rows.columns else ("mean_iou", "size"),
            mean_candidates_used=(
                "mean_candidates_used",
                "mean",
            )
            if "mean_candidates_used" in rows.columns
            else ("mean_iou", "count"),
        )
        .reset_index()
        .sort_values(["dataset", "mean_iou"], ascending=[True, False])
    )


def summarize_overall(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "result_family",
        "row_family",
        "method_key",
        "method_label",
        "ensemble_variant",
        "qa_pruning_policy",
        "soft_gate",
        "weights_column",
        "boundary_support_column",
        "boundary_floor",
        "boundary_floor_column",
        "boundary_iterations",
        "top_k_per_group",
        "threshold",
    ]
    present = [column for column in group_cols if column in dataset_summary.columns]
    fixed_policy = (
        dataset_summary.groupby(present, dropna=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            n_datasets=("dataset", "nunique"),
            n_folds=("n_folds", "sum"),
            n_cells=("n_cells", "sum"),
            mean_candidates_used=("mean_candidates_used", "mean"),
        )
        .reset_index()
    )
    fixed_policy["selection_scope"] = "fixed_policy"

    collapsed_cols = [column for column in present if column != "qa_pruning_policy"]
    dataset_policy = (
        dataset_summary.groupby(collapsed_cols, dropna=False)
        .agg(
            mean_iou=("mean_iou", "mean"),
            mean_f1=("mean_f1", "mean"),
            n_datasets=("dataset", "nunique"),
            n_folds=("n_folds", "sum"),
            n_cells=("n_cells", "sum"),
            mean_candidates_used=("mean_candidates_used", "mean"),
        )
        .reset_index()
    )
    if "qa_pruning_policy" in present:
        dataset_policy["qa_pruning_policy"] = "dataset_policy"
    dataset_policy["selection_scope"] = "dataset_policy"

    return (
        pd.concat([dataset_policy, fixed_policy], ignore_index=True, sort=False)
        .sort_values(["selection_scope", "mean_iou"], ascending=[True, False])
        .reset_index(drop=True)
    )


def load_simple_context(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    context = pd.read_csv(path)
    if context.empty:
        return context
    context = context.copy()
    context["result_family"] = "full_pipeline_context"
    context["row_family"] = "full_pipeline"
    context["method_key"] = "full_pipeline__qa_pruned_simple__dataset_policy"
    context["method_label"] = "QA-pruned SIMPLE context"
    context["ensemble_variant"] = ""
    context["qa_pruning_policy"] = context["policy"]
    context["soft_gate"] = ""
    return context


def write_markdown_report(
    *,
    output_path: Path,
    title: str,
    dataset_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    context: pd.DataFrame | None = None,
) -> None:
    lines = [
        f"# {title}",
        "",
        "Strict table scope: only our QA and ENSEMBLE methods.",
        "QA-pruned SIMPLE is shown only as deployable full-pipeline context.",
        "",
        "## Overall Strict QA/ENSEMBLE Rows",
        "",
    ]
    lines.extend(_markdown_table(overall_summary))
    lines.extend(["", "## Dataset Strict QA/ENSEMBLE Rows", ""])
    lines.extend(_markdown_table(dataset_summary))
    if context is not None and not context.empty:
        context_cols = [
            "dataset",
            "method_label",
            "qa_pruning_policy",
            "mean_iou",
            "mean_f1",
            "folds",
        ]
        present = [column for column in context_cols if column in context.columns]
        lines.extend(["", "## Deployable Full-Pipeline Context", ""])
        lines.extend(_markdown_table(context[present]))
        if "mean_iou" in context.columns:
            lines.extend(
                [
                    "",
                    f"QA-pruned SIMPLE context mean IoU: {context['mean_iou'].mean():.6f}",
                ]
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def filter_strict_ensemble_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if "row_family" not in rows.columns:
        return rows.iloc[0:0].copy()
    return rows[rows["row_family"].isin(STRICT_ENSEMBLE_ROW_FAMILIES)].copy()


def _markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows."]
    shown = frame.copy().fillna("")
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    return shown.to_markdown(index=False).splitlines()


def _policy_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name.removesuffix("_simple_escalated")
    if name.startswith("source_balanced_topk"):
        return "source_balanced_topk"
    return name


def _find_filtered_parquet(run_dir: Path, policy: str) -> Path | None:
    matches = sorted(run_dir.glob(f"*_paper_ready_{policy}.parquet"))
    if matches:
        return matches[0]
    matches = sorted(run_dir.glob("*.parquet"))
    return matches[0] if matches else None
