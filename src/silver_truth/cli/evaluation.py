import click
from pathlib import Path
from typing import Optional
import logging

import mlflow
import pandas as pd
import json

from silver_truth.ensemble.reconstruction import (
    reconstruct_labeled_full_images_from_paths,
)
from silver_truth.evaluation.evaluation_logic import evaluate_competitor_logic
from silver_truth.evaluation.stacked_jaccard_logic import (
    calculate_evaluation_metrics,
    calculate_evaluation_metrics_cropped,
)
from silver_truth.evaluation.reporting import (
    analyze_overflow_impact,
    generate_hsc_main_comparison_bundle,
    generate_hsc_reporting_bundle,
    write_hsc_main_comparison_bundle,
    write_overflow_impact_bundle,
    write_hsc_reporting_bundle,
)
from silver_truth.evaluation.qa_reporting import (
    _summarize_ablation_outputs,
    _validation_selected_full_pipeline,
    generate_ablation_diagnostics_report,
    write_ablation_diagnostics_report,
)
from silver_truth.evaluation.preflight import (
    build_split_sanity_audit,
    write_split_sanity_bundle,
)
from silver_truth.metrics.qa_model_evaluation import (
    evaluate_qa_model_from_excel,
    merge_predictions_to_parquet,
)
from silver_truth.qa.filtering_evaluation import run_qa_filtering_evaluation
from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    SILVER_TRUTH_COLUMN,
)
from silver_truth.experiment_tracking import (
    DEFAULT_MLFLOW_TRACKING_URI,
    infer_dataset_name_from_text,
    log_standardized_split_metrics,
    start_managed_mlflow_run,
    set_common_mlflow_tags,
    set_evaluation_tags,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@click.command()
@click.argument("dataset_dataframe_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--competitor", help="Competitor name to evaluate. If None, evaluate all."
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Path to save results as CSV",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Generate visualization of results (Placeholder)",
)
@click.option(
    "--campaign-col",
    default="campaign_number",
    help="Column name that identifies the campaign",
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Create detailed per-cell evaluation results in parquet format",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=DEFAULT_MLFLOW_TRACKING_URI,
    show_default=True,
    help="MLflow tracking URI.",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default=None,
    help="MLflow experiment name. If set, logs per-split metrics for each competitor.",
)
@click.option(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="MLflow run name prefix (competitor name is appended automatically).",
)
def evaluate_competitor(
    dataset_dataframe_path: Path,
    competitor: Optional[str] = None,
    output: Optional[Path] = None,
    visualize: bool = False,
    campaign_col: str = "campaign_number",
    detailed: bool = False,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_experiment: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
):
    """
    Evaluates competitor segmentation results against ground truth using Jaccard index.

    This script is a wrapper around the core evaluation logic in `run_evaluation`.
    With --detailed flag, also creates detailed per-cell evaluation results.
    """
    evaluate_competitor_logic(
        dataset_dataframe_path=dataset_dataframe_path,
        competitor=competitor,
        output=output,
        visualize=visualize,
        campaign_col=campaign_col,
        detailed=detailed,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
    )


@click.command()
@click.argument("parquet_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--mode",
    type=click.Choice(["auto", "full", "cropped"]),
    default="auto",
    show_default=True,
    help="How to compute GT-vs-seg metrics for stacked_path images.",
)
def calculate_evaluation_metrics_cli(parquet_path: Path, mode: str):
    """
    Calculate and persist `jaccard_score` and `f1_score` into a QA parquet file.

    Modes:
    - full: for full-size stacks
    - cropped: for QA crop stacks
    - auto: infer from parquet columns (uses cropped when crop coords are present)
    """
    selected_mode = mode
    if mode == "auto":
        df = pd.read_parquet(parquet_path)
        crop_cols = {"crop_y_start", "crop_y_end", "crop_x_start", "crop_x_end"}
        has_crop_cols = crop_cols.issubset(set(df.columns))
        selected_mode = "cropped" if has_crop_cols else "full"
        logging.info("Auto mode selected '%s' for %s", selected_mode, parquet_path)

    if selected_mode == "cropped":
        calculate_evaluation_metrics_cropped(parquet_path)
        return

    calculate_evaluation_metrics(parquet_path)


@click.command()
@click.argument("excel_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save evaluation results and plots",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Disable plot generation",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=DEFAULT_MLFLOW_TRACKING_URI,
    show_default=True,
    help="MLflow tracking URI.",
)
@click.option(
    "--mlflow-run-id",
    type=str,
    default=None,
    help="MLflow run ID to log metrics to an existing run",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default=None,
    help="MLflow experiment name (creates new run if --mlflow-run-id not provided)",
)
@click.option(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="MLflow run name for new runs",
)
def evaluate_qa_model(
    excel_path: Path,
    output_dir: Optional[Path] = None,
    no_plots: bool = False,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_run_id: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
):
    """
    Evaluate QA model predictions from an Excel file.

    The Excel file should contain sheets for train, validation, and test splits,
    with columns: cell_id, Jaccard index, Predicted Jaccard index.

    Calculates R², MAE, RMSE, tolerance-based accuracy, and generates plots.

    Optionally logs metrics to MLflow (either to an existing run or a new one).

    NOTE: This evaluates the QA MODEL predictions, not the final ensemble results.
    """
    results = evaluate_qa_model_from_excel(
        excel_path=excel_path,
        output_dir=output_dir,
        generate_plots=not no_plots,
    )

    # Log to MLflow if requested
    if mlflow_run_id or mlflow_experiment:
        _log_qa_metrics_to_mlflow(
            results=results,
            excel_path=excel_path,
            output_dir=output_dir,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment=mlflow_experiment,
            mlflow_run_name=mlflow_run_name,
        )


def _log_qa_metrics_to_mlflow(
    results: dict,
    excel_path: Path,
    output_dir: Optional[Path],
    mlflow_tracking_uri: str,
    mlflow_run_id: Optional[str],
    mlflow_experiment: Optional[str],
    mlflow_run_name: Optional[str],
):
    """Log QA evaluation metrics to MLflow."""

    def log_metrics_for_split(split_results: dict, split_name: str):
        """Flatten and log metrics for a single split."""
        metrics = {}
        for key, value in split_results.items():
            if isinstance(value, (int, float)) and key != "split":
                # Prefix with split name for clarity
                metric_name = f"{split_name}_{key}"
                metrics[metric_name] = value
        if metrics:
            mlflow.log_metrics(metrics)

    if mlflow_run_id:
        # Log to existing run
        with start_managed_mlflow_run(
            run_id=mlflow_run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
        ):
            logging.info(f"Logging QA metrics to existing MLflow run: {mlflow_run_id}")
            for split_name, split_results in results.items():
                log_metrics_for_split(split_results, split_name)

            # Log artifacts
            mlflow.log_artifact(str(excel_path))
            if output_dir and output_dir.exists():
                mlflow.log_artifacts(str(output_dir), artifact_path="evaluation")
    else:
        # Create new run
        with start_managed_mlflow_run(
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
            run_name=mlflow_run_name,
        ):
            active = mlflow.active_run()
            if active is not None:
                logging.info(f"Created new MLflow run: {active.info.run_id}")
            else:
                logging.info("Created new MLflow run.")

            set_evaluation_tags(
                pipeline_family="qa_model",
                evaluation_level="qa_regression",
                setup_name=excel_path.stem,
            )

            # Log the excel path as a parameter
            mlflow.log_param("excel_path", str(excel_path))

            for split_name, split_results in results.items():
                log_metrics_for_split(split_results, split_name)

            # Log artifacts
            mlflow.log_artifact(str(excel_path))
            if output_dir and output_dir.exists():
                mlflow.log_artifacts(str(output_dir), artifact_path="evaluation")

            active = mlflow.active_run()
            if active is not None:
                logging.info(f"MLflow run ID: {active.info.run_id}")


@click.command("evaluate-qa-filtering")
@click.argument("excel_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--thresholds",
    type=str,
    default="0.50,0.60,0.70,0.75,0.80,0.85,0.90",
    show_default=True,
    help="Comma-separated filtering thresholds in [0,1].",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save filtering metrics/plots.",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Disable plot generation.",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=DEFAULT_MLFLOW_TRACKING_URI,
    show_default=True,
    help="MLflow tracking URI.",
)
@click.option(
    "--mlflow-run-id",
    type=str,
    default=None,
    help="MLflow run ID to log metrics to an existing run.",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default=None,
    help="MLflow experiment name (creates new run if --mlflow-run-id not provided).",
)
@click.option(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="MLflow run name for new runs.",
)
def evaluate_qa_filtering(
    excel_path: Path,
    thresholds: str,
    output_dir: Optional[Path] = None,
    no_plots: bool = False,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_run_id: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
):
    """
    Evaluate QA predictions as a thresholded filter ("keep" vs "filter out").

    Computes confusion-matrix-based metrics for each split and threshold.
    """
    result = run_qa_filtering_evaluation(
        excel_path=excel_path,
        thresholds_csv=thresholds,
        output_dir=output_dir,
        generate_plots=not no_plots,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_run_id=mlflow_run_id,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
    )

    metrics_csv_path = result["metrics_csv_path"]
    summary_csv_path = result["summary_csv_path"]
    click.echo(f"Saved filtering metrics: {metrics_csv_path}")
    click.echo(f"Saved best-threshold summary: {summary_csv_path}")


@click.command()
@click.argument("parquet_path", type=click.Path(exists=True, path_type=Path))
@click.argument("excel_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for merged parquet file",
)
def merge_qa_predictions(
    parquet_path: Path,
    excel_path: Path,
    output: Optional[Path] = None,
):
    """
    Merge QA model predictions from Excel into an existing parquet file.

    Adds 'predicted_jaccard_index' column to the parquet file by matching cell_id.
    This is useful for analyzing model performance across train/val/test splits.
    """
    merge_predictions_to_parquet(
        parquet_path=parquet_path,
        excel_path=excel_path,
        output_path=output,
    )


@click.command("audit-experiment-inputs")
@click.option("--dataset", required=True, type=str, help="Dataset name.")
@click.option("--crop-size", required=True, type=int, help="QA crop size.")
@click.option(
    "--split-name",
    required=True,
    type=str,
    help="Split name, e.g. fold-1 or fold-2.",
)
@click.option(
    "--whole-image-parquet",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Whole-image split parquet.",
)
@click.option(
    "--qa-parquet",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Split-attached QA parquet.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for split_sanity outputs.",
)
def audit_experiment_inputs(
    dataset: str,
    crop_size: int,
    split_name: str,
    whole_image_parquet: Path,
    qa_parquet: Path,
    output_dir: Path,
) -> None:
    """Audit split sanity before QA or ensemble training."""
    audit, sample_manifest = build_split_sanity_audit(
        dataset=dataset,
        crop_size=crop_size,
        split_name=split_name,
        whole_image_parquet=whole_image_parquet,
        qa_parquet=qa_parquet,
    )
    written = write_split_sanity_bundle(audit, sample_manifest, output_dir)

    click.echo(json.dumps(audit, indent=2))
    click.echo(f"Wrote JSON: {written['json']}")
    click.echo(f"Wrote Markdown: {written['markdown']}")
    click.echo(f"Wrote sample manifest: {written['sample_manifest']}")

    if not audit["scientifically_valid"]:
        failed = [name for name, passed in audit["hard_checks"].items() if not passed]
        raise click.ClickException("Split sanity gate failed: " + ", ".join(failed))


@click.group()
def cli():
    """Main entry point for command-line tools."""
    pass


# ---------------------------------------------------------------------------
# evaluate-fusion-crops
# ---------------------------------------------------------------------------


@click.command("evaluate-fusion-crops")
@click.argument(
    "parquet_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--fused-path-column",
    required=True,
    help=(
        "Column in the parquet that contains the on-disk path to each fused crop TIF "
        "(e.g. 'bic_flat_voting').  This is the lower-cased model name written by "
        "silver-fusion run-crops-experiment."
    ),
)
@click.option(
    "--output-dir",
    "-d",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory where reconstructed full-image TIFs will be written.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to write per-image IoU/F1 CSV.  Defaults to <output-dir>/fullimage_eval.csv.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Binarisation threshold applied to each fused mask before reconstruction.",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=DEFAULT_MLFLOW_TRACKING_URI,
    show_default=True,
    help="MLflow tracking URI.",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default=None,
    help="Optional MLflow experiment used for reconstructed-image evaluation logging.",
)
@click.option(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="Optional MLflow run name.",
)
@click.option(
    "--setup-name",
    type=str,
    default=None,
    help="Logical setup name used in MLflow tags. Defaults to fused-path-column.",
)
@click.option(
    "--pipeline-family",
    type=str,
    default="fusion",
    show_default=True,
    help="Pipeline family tag written to MLflow.",
)
@click.option(
    "--qa-mode",
    type=str,
    default=None,
    help="Optional QA mode tag, e.g. fusion_only or full_pipeline.",
)
@click.option(
    "--qa-threshold",
    type=float,
    default=None,
    help="Optional QA threshold tag for MLflow.",
)
@click.option(
    "--priority-column",
    "priority_columns",
    multiple=True,
    help=(
        "Optional column(s) used to resolve overlapping reconstructed labels. "
        "Higher values win; columns are tried in the order provided."
    ),
)
def evaluate_fusion_crops(
    parquet_path: Path,
    fused_path_column: str,
    output_dir: Path,
    output: Optional[Path],
    threshold: float,
    mlflow_tracking_uri: str,
    mlflow_experiment: Optional[str],
    mlflow_run_name: Optional[str],
    setup_name: Optional[str],
    pipeline_family: str,
    qa_mode: Optional[str],
    qa_threshold: Optional[float],
    priority_columns: tuple[str, ...],
) -> None:
    """
    Reconstruct labeled full-image segmentations from per-cell crops and evaluate
    them with the canonical full-image, per-label IoU/F1 metric.

    Reads the parquet produced by ``silver-fusion run-crops-experiment``, places each
    fused crop back into the full image at its recorded coordinates using its row's
    ``label`` value, and then scores the reconstructed labeled image exactly like the
    competitor / silver-truth baseline evaluator does.

    The parquet must contain ``gt_image``, ``crop_y_start/end/x_start/x_end`` (or
    their ``recon_crop_*`` equivalents), ``label``, and the fused-path column
    specified by ``--fused-path-column``.
    """
    df = pd.read_parquet(parquet_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = reconstruct_labeled_full_images_from_paths(
        databank_df=df,
        fused_path_column=fused_path_column,
        output_dir=output_dir,
        threshold=threshold,
        priority_columns=priority_columns,
    )

    if results_df.empty:
        logging.warning(
            "No images could be reconstructed — check paths, labels, and GT columns."
        )
    else:
        mean_iou = results_df["iou"].mean()
        mean_f1 = results_df["f1"].mean()
        logging.info(
            "Reconstructed %d labeled images — mean IoU=%.4f  mean F1=%.4f",
            len(results_df),
            mean_iou,
            mean_f1,
        )

    csv_path = output if output is not None else output_dir / "fullimage_eval.csv"
    results_df.to_csv(csv_path, index=False)

    if mlflow_experiment:
        dataset_tag = infer_dataset_name_from_text(
            [parquet_path, *df.get("gt_image", pd.Series(dtype=str)).dropna().head(10)]
        )
        split_metrics: dict[str, float] = {}
        if not results_df.empty:
            if "split" in results_df.columns:
                for split_name, split_df in results_df.groupby("split"):
                    split_key = str(split_name)
                    split_metrics[f"{split_key}_mean_jaccard"] = float(
                        split_df["iou"].mean()
                    )
                    split_metrics[f"{split_key}_mean_f1"] = float(split_df["f1"].mean())
                    split_metrics[f"{split_key}_count"] = float(len(split_df))
            split_metrics["overall_mean_jaccard"] = float(results_df["iou"].mean())
            split_metrics["overall_mean_f1"] = float(results_df["f1"].mean())
            split_metrics["overall_count"] = float(len(results_df))

        with start_managed_mlflow_run(
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
            run_name=mlflow_run_name or fused_path_column,
        ):
            set_common_mlflow_tags(dataset=dataset_tag, split="full_image_label")
            set_evaluation_tags(
                pipeline_family=pipeline_family,
                evaluation_level="full_image_label",
                setup_name=setup_name or fused_path_column,
                qa_mode=qa_mode,
                qa_threshold=qa_threshold,
                extra_tags={
                    "fused_path_column": fused_path_column,
                    "reconstruction_source": "cell_crops",
                },
            )
            mlflow.log_param("parquet_path", str(parquet_path))
            mlflow.log_param("fused_path_column", fused_path_column)
            mlflow.log_param("output_csv", str(csv_path))
            mlflow.log_param("threshold", threshold)
            if priority_columns:
                mlflow.log_param("priority_columns", ",".join(priority_columns))
            log_standardized_split_metrics(split_metrics)
            mlflow.log_artifact(str(csv_path))

    click.echo(f"Full-image evaluation written to: {csv_path}")


# ---------------------------------------------------------------------------
# filter-parquet
# ---------------------------------------------------------------------------


@click.command("filter-parquet")
@click.argument(
    "parquet_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--mode",
    type=click.Choice(["qa_only", "full_pipeline"]),
    required=True,
    help=(
        "Filtering mode.  "
        "'qa_only': keep only the single highest-predicted-quality crop per cell "
        "(top-1 by predicted_jaccard_index) — no fusion needed downstream.  "
        "'full_pipeline': keep all crops whose predicted_jaccard_index >= threshold; "
        "for any cell with no passing crop, fall back to its top-1 crop so no cell "
        "is silently dropped."
    ),
)
@click.option(
    "--threshold",
    type=float,
    default=0.75,
    show_default=True,
    help="QA score threshold used by 'full_pipeline' mode (ignored for 'qa_only').",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Path for the filtered output parquet.",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=DEFAULT_MLFLOW_TRACKING_URI,
    show_default=True,
    help="MLflow tracking URI.",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default=None,
    help="Optional MLflow experiment to log filtering statistics to.",
)
@click.option(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="Optional MLflow run name.",
)
def filter_parquet(
    parquet_path: Path,
    mode: str,
    threshold: float,
    output: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: Optional[str],
    mlflow_run_name: Optional[str],
) -> None:
    """
    Filter a QA-enriched parquet by predicted_jaccard_index.

    Requires that the parquet already has a ``predicted_jaccard_index`` column
    (added by ``silver-evaluation merge-qa-predictions``).

    Use this as a preprocessing step before passing to
    ``silver-fusion run-crops-experiment`` or
    ``silver-evaluation evaluate-fusion-crops``.
    """
    _KEY_COLS = ["campaign_number", "original_image_key", "label"]

    df = pd.read_parquet(parquet_path)
    if "predicted_jaccard_index" not in df.columns:
        raise click.ClickException(
            "Column 'predicted_jaccard_index' not found.  "
            "Run 'silver-evaluation merge-qa-predictions' first."
        )

    if "competitor" in df.columns:
        reference_mask = df["competitor"].astype(str) == SILVER_TRUTH_COLUMN
        reference_count = int(reference_mask.sum())
        if reference_count:
            df = df.loc[~reference_mask].copy()
            logging.warning(
                "Dropped %d '%s' reference rows before QA filtering.",
                reference_count,
                SILVER_TRUTH_COLUMN,
            )

    n_rows_in = len(df)
    n_cells_total = df[_KEY_COLS].drop_duplicates().shape[0]
    filter_stats: dict = {
        "mode": mode,
        "threshold": threshold,
        "total_rows_in": n_rows_in,
        "total_cells": n_cells_total,
    }

    if mode == "qa_only":
        filtered = (
            df.sort_values("predicted_jaccard_index", ascending=False)
            .groupby(_KEY_COLS, as_index=False)
            .first()
        )
        n_out = filtered[_KEY_COLS].drop_duplicates().shape[0]
        filter_stats["total_rows_out"] = len(filtered)
        filter_stats["cells_passing_threshold"] = n_out
        filter_stats["cells_fallback_top1"] = 0
        filter_stats["pct_cells_filtered"] = 0.0
        logging.info(
            "qa_only: %d cells in → %d cells out (top-1 per cell, %d rows → %d rows)",
            n_cells_total,
            n_out,
            n_rows_in,
            len(filtered),
        )
    else:  # full_pipeline
        passing = df[df["predicted_jaccard_index"] >= threshold].copy()
        covered_keys = set(map(tuple, passing[_KEY_COLS].values.tolist()))
        not_covered = df[~df[_KEY_COLS].apply(tuple, axis=1).isin(covered_keys)]
        fallback = (
            not_covered.sort_values("predicted_jaccard_index", ascending=False)
            .groupby(_KEY_COLS, as_index=False)
            .first()
        )
        filtered = pd.concat([passing, fallback], ignore_index=True)
        n_passing_cells = len(covered_keys)
        n_fallback_cells = fallback[_KEY_COLS].drop_duplicates().shape[0]
        pct_filtered = 100.0 * (n_cells_total - n_passing_cells) / max(n_cells_total, 1)

        filter_stats["total_rows_out"] = len(filtered)
        filter_stats["cells_passing_threshold"] = n_passing_cells
        filter_stats["cells_fallback_top1"] = n_fallback_cells
        filter_stats["pct_cells_filtered"] = round(pct_filtered, 2)

        logging.info(
            "full_pipeline t=%.2f: %d rows in → %d rows out | "
            "%d/%d cells pass threshold (%.1f%% fell back to top-1)",
            threshold,
            n_rows_in,
            len(filtered),
            n_passing_cells,
            n_cells_total,
            pct_filtered,
        )

    # ── Prominent summary ────────────────────────────────────────────────
    click.echo("")
    click.echo("╔══════════════════════════════════════════════════════╗")
    click.echo(f"║  QA FILTER: mode={mode}  threshold={threshold:.2f}             ║")
    click.echo("╠══════════════════════════════════════════════════════╣")
    click.echo(
        f"║  Rows:  {n_rows_in:>6d} → {filter_stats['total_rows_out']:>6d}                          ║"
    )
    click.echo(f"║  Cells: {n_cells_total:>6d} total                              ║")
    click.echo(
        f"║         {filter_stats['cells_passing_threshold']:>6d} pass threshold                    ║"
    )
    click.echo(
        f"║         {filter_stats['cells_fallback_top1']:>6d} fallback to top-1                 ║"
    )
    click.echo(
        f"║         {filter_stats['pct_cells_filtered']:>5.1f}% of cells needed fallback        ║"
    )
    click.echo("╚══════════════════════════════════════════════════════╝")
    click.echo("")

    # ── MLflow logging ───────────────────────────────────────────────────
    if mlflow_experiment:
        with start_managed_mlflow_run(
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
            run_name=mlflow_run_name or f"filter_{mode}_t{threshold}",
        ):
            dataset_tag = infer_dataset_name_from_text([str(parquet_path)])
            set_common_mlflow_tags(dataset=dataset_tag, split="filter")
            set_evaluation_tags(
                pipeline_family="qa_filter",
                evaluation_level="filter",
                setup_name=f"{mode}_t{threshold}",
                qa_mode=mode,
                qa_threshold=threshold,
            )
            mlflow.log_param("parquet_path", str(parquet_path))
            mlflow.log_param("output_path", str(output))
            for key, value in filter_stats.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(key, value)

    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output, index=False)
    click.echo(f"Filtered parquet written to: {output}  ({len(filtered)} rows)")


@click.command("score-parquet")
@click.argument(
    "parquet_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--mode",
    type=click.Choice(["oracle", "competitor_prior"]),
    required=True,
    help=(
        "'oracle': copy true per-candidate quality into the score column. "
        "'competitor_prior': score candidates by train-split mean quality per competitor."
    ),
)
@click.option(
    "--source-column",
    default="jaccard_score",
    show_default=True,
    help="Column containing true per-candidate quality scores.",
)
@click.option(
    "--score-column",
    default="predicted_jaccard_index",
    show_default=True,
    help="Output score column used by downstream QA filtering.",
)
@click.option(
    "--split-column",
    default="split",
    show_default=True,
    help="Split column used by competitor_prior mode.",
)
@click.option(
    "--train-split",
    default="train",
    show_default=True,
    help="Split value used to estimate competitor priors.",
)
@click.option(
    "--competitor-column",
    default="competitor",
    show_default=True,
    help="Competitor/rater identifier column used by competitor_prior mode.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Path for the scored output parquet.",
)
def score_parquet(
    parquet_path: Path,
    mode: str,
    source_column: str,
    score_column: str,
    split_column: str,
    train_split: str,
    competitor_column: str,
    output: Path,
) -> None:
    """Write diagnostic QA scores into a paper-ready parquet."""
    df = pd.read_parquet(parquet_path)
    if source_column not in df.columns:
        raise click.ClickException(
            f"Column '{source_column}' not found in {parquet_path}."
        )

    scored = df.copy()
    if mode == "oracle":
        scored[score_column] = pd.to_numeric(scored[source_column], errors="coerce")
    else:
        missing_columns = [
            column
            for column in (split_column, competitor_column)
            if column not in scored.columns
        ]
        if missing_columns:
            raise click.ClickException(
                "Missing required column(s) for competitor_prior mode: "
                + ", ".join(missing_columns)
            )

        train_rows = scored[scored[split_column].astype(str) == str(train_split)]
        if train_rows.empty:
            raise click.ClickException(
                f"No rows found where {split_column} == '{train_split}'."
            )

        train_scores = pd.to_numeric(train_rows[source_column], errors="coerce")
        priors = (
            train_rows.assign(_score_for_prior=train_scores)
            .groupby(competitor_column)["_score_for_prior"]
            .mean()
            .dropna()
        )
        if priors.empty:
            raise click.ClickException(
                "Could not estimate any competitor priors from the training split."
            )

        fallback = float(train_scores.mean())
        scored[score_column] = (
            scored[competitor_column].map(priors).fillna(fallback).astype(float)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output, index=False)
    click.echo(
        f"Scored parquet written to: {output} "
        f"({len(scored)} rows, mode={mode}, score_column={score_column})"
    )


@click.command("report-hsc-results")
@click.option(
    "--paper-runs-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/paper_runs"),
    show_default=True,
    help="Root directory containing paper run outputs.",
)
@click.option(
    "--variant",
    type=str,
    default="baseline",
    show_default=True,
    help="Ablation variant name under data/paper_runs/ablation/.",
)
@click.option(
    "--qa-threshold",
    type=float,
    default=0.75,
    show_default=True,
    help="QA threshold used for the default full-pipeline and ensemble_qa report rows.",
)
@click.option(
    "--fusion-model",
    type=str,
    default="simple",
    show_default=True,
    help="Fusion model subdirectory to use for fusion_only reporting.",
)
@click.option(
    "--full-pipeline-model",
    type=str,
    default="simple",
    show_default=True,
    help="Fusion model subdirectory to use for full_pipeline reporting.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/paper_runs/reports/hsc_baseline"),
    show_default=True,
    help="Directory where inventory, per-image metrics, summary tables, and markdown are written.",
)
@click.option(
    "--bootstrap-samples",
    type=int,
    default=10000,
    show_default=True,
    help="Number of bootstrap resamples for confidence intervals.",
)
@click.option(
    "--bootstrap-seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for bootstrap confidence intervals.",
)
def report_hsc_results(
    paper_runs_root: Path,
    variant: str,
    qa_threshold: float,
    fusion_model: str,
    full_pipeline_model: str,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
):
    """
    Consolidate HSC fold-safe baseline outputs into one reporting bundle.

    The command auto-discovers the current HSC baseline artifacts under
    data/paper_runs/, normalizes them to one per-image table, computes bootstrap
    confidence intervals, runs default paired comparisons, and writes CSV/Markdown
    outputs for advisor or manuscript use.
    """
    bundle = generate_hsc_reporting_bundle(
        paper_runs_root=paper_runs_root,
        variant=variant,
        qa_threshold=qa_threshold,
        fusion_model=fusion_model,
        full_pipeline_model=full_pipeline_model,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    written = write_hsc_reporting_bundle(output_dir, bundle)

    inventory = bundle["inventory"]
    found = int(inventory["exists"].sum())
    missing = int((~inventory["exists"]).sum())
    click.echo(f"Report bundle written to: {output_dir}")
    click.echo(f"Artifacts found: {found} | missing: {missing}")
    click.echo(f"Core summary: {written['core_summary']}")
    click.echo(f"Paired comparisons: {written['comparisons']}")
    click.echo(f"Markdown summary: {written['markdown']}")


@click.command("report-hsc-main-comparison")
@click.option(
    "--comparison-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/paper_runs/clean_comparison_20260312_231619"),
    show_default=True,
    help="Root directory containing fold-1/ and fold-2/ clean comparison exports.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/paper_runs/reports/hsc_main_comparison"),
    show_default=True,
    help="Directory where the main-comparison statistical note is written.",
)
@click.option(
    "--bootstrap-samples",
    type=int,
    default=10000,
    show_default=True,
    help="Number of stratified bootstrap resamples for confidence intervals.",
)
@click.option(
    "--bootstrap-seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for bootstrap confidence intervals.",
)
def report_hsc_main_comparison(
    comparison_root: Path,
    output_dir: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
):
    """
    Write the manuscript-safe HSC main comparison statistical note.

    This report focuses on the clean cross-fold HSC comparison between the
    learned ensemble, SILVER-TRUTH, and the competitor baselines. Confidence
    intervals are bootstrapped over test images within each fold and then
    averaged across folds to stay aligned with fold-level reporting.
    """
    bundle = generate_hsc_main_comparison_bundle(
        comparison_root=comparison_root,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    written = write_hsc_main_comparison_bundle(output_dir, bundle)

    click.echo(f"Main comparison note written to: {output_dir}")
    click.echo(f"Core summary: {written['core_summary']}")
    click.echo(f"Paired comparisons: {written['comparisons']}")
    click.echo(f"Markdown summary: {written['markdown']}")


@click.command("report-overflow-impact")
@click.argument("results_path", type=click.Path(exists=True, path_type=Path))
@click.argument("dataset_dataframe_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--crop-size",
    required=True,
    type=int,
    help="Square crop size used to define overflow vs fit.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory where enriched and summary tables will be written.",
)
@click.option(
    "--metric-column",
    "metric_columns",
    multiple=True,
    help=(
        "Optional metric column(s) to summarize. "
        "Defaults to jaccard/f1-style columns when present."
    ),
)
def report_overflow_impact(
    results_path: Path,
    dataset_dataframe_path: Path,
    crop_size: int,
    output_dir: Path,
    metric_columns: tuple[str, ...],
) -> None:
    """
    Summarize per-cell metrics split by whether the GT bbox overflows a chosen crop size.

    Typical input is a detailed per-cell result parquet/CSV, for example the output of
    `silver-evaluation evaluate-competitor --detailed`.
    """
    bundle = analyze_overflow_impact(
        results_path=results_path,
        dataset_dataframe_path=dataset_dataframe_path,
        crop_size=crop_size,
        metric_columns=list(metric_columns) if metric_columns else None,
    )
    written = write_overflow_impact_bundle(output_dir, bundle)

    overall = bundle.get("overall_summary", pd.DataFrame())
    if not overall.empty:
        click.echo(overall.to_string(index=False))
        click.echo("")

    for name, path in written.items():
        click.echo(f"{name}: {path}")


@click.command("report-ablation-diagnostics")
@click.option(
    "--qa-metrics-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV emitted by `evaluate-qa-model`.",
)
@click.option(
    "--qa-filtering-csv",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV emitted by `evaluate-qa-filtering`.",
)
@click.option(
    "--ablation-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Ablation output directory for one dataset/crop/variant/fold.",
)
@click.option("--dataset", required=True, type=str, help="Dataset name.")
@click.option("--crop-tag", required=True, type=str, help="Crop tag, e.g. sz64.")
@click.option("--variant", required=True, type=str, help="Variant name.")
@click.option("--split-name", required=True, type=str, help="Split name, e.g. fold-1.")
@click.option(
    "--default-threshold",
    type=float,
    default=0.75,
    show_default=True,
    help="Default QA threshold to highlight in the diagnostics.",
)
@click.option(
    "--validation-selection-fusion-model",
    default=None,
    help=(
        "Restrict validation-selected full-pipeline reporting to this fusion model. "
        "Use the same value as threshold selection, e.g. simple."
    ),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory where the QA ablation diagnostics bundle is written.",
)
def report_ablation(
    qa_metrics_csv: Path,
    qa_filtering_csv: Path,
    ablation_dir: Path,
    dataset: str,
    crop_tag: str,
    variant: str,
    split_name: str,
    default_threshold: float,
    validation_selection_fusion_model: Optional[str],
    output_dir: Path,
) -> None:
    """
    Consolidate ablation outputs into one diagnostics bundle.

    This report is meant to catch cases where apparent gains are brittle,
    threshold-sensitive, tail-risky, or otherwise not supported by the
    underlying diagnostics.
    """
    bundle = generate_ablation_diagnostics_report(
        qa_metrics_csv=qa_metrics_csv,
        qa_filtering_csv=qa_filtering_csv,
        ablation_dir=ablation_dir,
        dataset=dataset,
        crop_tag=crop_tag,
        variant=variant,
        split_name=split_name,
        default_threshold=default_threshold,
        validation_selection_fusion_model=validation_selection_fusion_model,
    )
    written = write_ablation_diagnostics_report(output_dir, bundle)

    overview = bundle["overview"]
    flags = bundle["diagnostic_flags"]
    if not overview.empty:
        click.echo(overview.to_string(index=False))
        click.echo("")

    triggered_flags = (
        flags[flags["triggered"].fillna(False)] if not flags.empty else flags
    )
    click.echo(
        f"Triggered flags: {0 if triggered_flags is None else len(triggered_flags)}"
    )
    click.echo(f"Markdown report: {written['report_markdown']}")
    click.echo(f"Method summary: {written['ablation_method_summary']}")


@click.command("select-ablation-threshold")
@click.option(
    "--ablation-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Ablation output directory for one dataset/crop/variant/fold.",
)
@click.option(
    "--fusion-model",
    default="simple",
    show_default=True,
    help="Full-pipeline fusion model used for validation threshold selection.",
)
@click.option(
    "--thresholds",
    default=None,
    help=(
        "Optional comma-separated threshold whitelist. When set, selection only "
        "considers full-pipeline rows whose threshold is in this grid."
    ),
)
@click.option(
    "--output-json",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to write the selected threshold record as JSON.",
)
@click.option(
    "--output-threshold",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to write only the selected numeric threshold.",
)
@click.option(
    "--output-label",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to write the selected threshold label, e.g. 0.40.",
)
def select_ablation_threshold(
    ablation_dir: Path,
    fusion_model: str,
    thresholds: Optional[str],
    output_json: Path,
    output_threshold: Path,
    output_label: Path,
) -> None:
    """Select a QA threshold from full-pipeline validation performance only."""
    summary = _summarize_ablation_outputs(ablation_dir)
    if summary.empty:
        raise click.ClickException(f"No ablation outputs found in {ablation_dir}.")

    model_key = str(fusion_model).lower()
    candidates = summary[
        (summary["family"] == "full_pipeline")
        & (summary["fusion_model"].astype(str) == model_key)
    ].copy()
    if candidates.empty:
        raise click.ClickException(
            f"No full_pipeline rows found for fusion model '{model_key}'."
        )

    if thresholds:
        allowed_thresholds = {
            round(float(value.strip()), 6)
            for value in thresholds.split(",")
            if value.strip()
        }
        candidate_thresholds = pd.to_numeric(
            candidates["threshold"], errors="coerce"
        ).round(6)
        candidates = candidates[candidate_thresholds.isin(allowed_thresholds)].copy()
        if candidates.empty:
            raise click.ClickException(
                "No full_pipeline rows remained after applying threshold whitelist "
                f"'{thresholds}'."
            )

    selected = _validation_selected_full_pipeline(candidates)
    if selected is None or pd.isna(selected.get("threshold")):
        raise click.ClickException(
            "Could not select a threshold from validation metrics."
        )

    threshold = float(selected["threshold"])
    label = f"{threshold:.2f}"
    record = {
        "method_key": str(selected["method_key"]),
        "threshold": threshold,
        "threshold_label": label,
        "fusion_model": model_key,
        "selection_metric": "validation_mean_iou",
        "validation_mean_iou": float(selected["validation_mean_iou"]),
        "validation_mean_f1": float(selected["validation_mean_f1"]),
        "test_mean_iou": float(selected["test_mean_iou"]),
        "test_mean_f1": float(selected["test_mean_f1"]),
        "source_path": str(selected["source_path"]),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_threshold.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(record, indent=2) + "\n")
    output_threshold.write_text(f"{threshold:.6f}\n")
    output_label.write_text(f"{label}\n")
    click.echo(json.dumps(record, indent=2))


@click.command("summarize-paper-results")
@click.option(
    "--paper-runs-root",
    default=Path("data/paper_runs"),
    show_default=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root paper-runs directory containing reports/ and baselines/.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Output CSV path. Defaults to "
        "<paper-runs-root>/reports/paper_result_summary.csv."
    ),
)
def summarize_paper_results(paper_runs_root: Path, output: Optional[Path]) -> None:
    """Write the final paper table summary from generated report artifacts."""
    output_path = (
        output
        if output is not None
        else paper_runs_root / "reports" / "paper_result_summary.csv"
    )
    rows = _build_paper_result_summary(paper_runs_root)
    if rows.empty:
        raise click.ClickException(
            f"No ablation method summaries found under {paper_runs_root / 'reports'}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    click.echo(f"Paper result summary written to: {output_path} ({len(rows)} rows)")


def _build_paper_result_summary(paper_runs_root: Path) -> pd.DataFrame:
    report_root = paper_runs_root / "reports"
    records: list[dict[str, object]] = []
    for summary_path in sorted(report_root.rglob("ablation_method_summary.csv")):
        metadata = _parse_ablation_summary_path(report_root, summary_path)
        if metadata is None:
            continue

        method_summary = pd.read_csv(summary_path)
        if method_summary.empty:
            continue

        baseline_stats = _load_baseline_stats(paper_runs_root, metadata)
        record = {
            **metadata,
            "row_type": "fold",
            **baseline_stats,
            **_summarize_methods_for_paper_table(method_summary),
            "ablation_method_summary_path": str(summary_path),
        }
        records.append(record)

    if not records:
        return pd.DataFrame()

    fold_df = pd.DataFrame.from_records(records)
    mean_rows = [
        _build_mean_summary_row(group)
        for _, group in fold_df.groupby(["dataset", "crop_tag", "variant"], sort=True)
    ]
    result = pd.concat([fold_df, pd.DataFrame(mean_rows)], ignore_index=True)
    return result[_paper_result_columns()].sort_values(
        ["dataset", "crop_tag", "variant", "row_type", "fold"],
        ascending=[True, True, True, True, True],
    )


def _parse_ablation_summary_path(
    report_root: Path, summary_path: Path
) -> dict[str, str] | None:
    try:
        rel = summary_path.relative_to(report_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 5 or parts[-2:] != (
        "ablation_diagnostics",
        "ablation_method_summary.csv",
    ):
        return None
    return {
        "dataset": parts[0],
        "crop_tag": parts[1],
        "variant": parts[2],
        "fold": parts[3],
    }


def _load_baseline_stats(
    paper_runs_root: Path, metadata: dict[str, str]
) -> dict[str, object]:
    stem = (
        f"{metadata['dataset']}_{metadata['crop_tag']}_"
        f"{metadata['variant']}_{metadata['fold']}"
    )
    competitor_path = paper_runs_root / "baselines" / f"{stem}_competitors.csv"
    silver_path = paper_runs_root / "baselines" / f"{stem}_silver_truth.csv"

    stats: dict[str, object] = {
        "best_competitor_iou": float("nan"),
        "best_competitor_name": "",
        "median_competitor_iou": float("nan"),
        "median_competitor_name": "",
        "worst_competitor_iou": float("nan"),
        "worst_competitor_name": "",
        "silver_truth_iou": float("nan"),
    }

    if competitor_path.exists():
        competitors = pd.read_csv(competitor_path)
        if {"competitor", "split_test_average"}.issubset(competitors.columns):
            per_competitor = (
                competitors.groupby("competitor")["split_test_average"]
                .first()
                .dropna()
                .astype(float)
                .sort_values(ascending=False)
            )
            if not per_competitor.empty:
                stats["best_competitor_iou"] = float(per_competitor.iloc[0])
                stats["best_competitor_name"] = str(per_competitor.index[0])
                stats["worst_competitor_iou"] = float(per_competitor.iloc[-1])
                stats["worst_competitor_name"] = str(per_competitor.index[-1])
                stats["median_competitor_iou"] = float(per_competitor.median())
                stats["median_competitor_name"] = _median_competitor_name(
                    per_competitor
                )

    if silver_path.exists():
        silver = pd.read_csv(silver_path)
        if "split_test_average" in silver.columns and not silver.empty:
            stats["silver_truth_iou"] = float(
                pd.to_numeric(silver["split_test_average"], errors="coerce")
                .dropna()
                .iloc[0]
            )

    return stats


def _median_competitor_name(per_competitor: pd.Series) -> str:
    ordered = per_competitor.sort_values(ascending=True)
    n = len(ordered)
    if n == 0:
        return ""
    if n % 2 == 1:
        return str(ordered.index[n // 2])
    return f"{ordered.index[n // 2 - 1]} / {ordered.index[n // 2]}"


def _summarize_methods_for_paper_table(summary: pd.DataFrame) -> dict[str, object]:
    diagnostics = {"oracle_qa_only", "competitor_prior_qa_only"}
    deployable = summary[~summary["family"].isin(diagnostics)].copy()
    deployable = deployable.dropna(subset=["test_mean_iou"])

    best_fusion = _best_row(
        summary[summary["family"] == "fusion_only"], "test_mean_iou"
    )
    best_ours = _best_row(deployable, "test_mean_iou")
    validation_selected = _best_row(
        deployable.dropna(subset=["validation_mean_iou", "test_mean_iou"]),
        "validation_mean_iou",
    )
    oracle = _best_row(summary[summary["family"] == "oracle_qa_only"], "test_mean_iou")
    competitor_prior = _best_row(
        summary[summary["family"] == "competitor_prior_qa_only"], "test_mean_iou"
    )

    return {
        "best_fusion_iou": _row_float(best_fusion, "test_mean_iou"),
        "best_fusion_method": _row_str(best_fusion, "method_key"),
        "best_ours_iou": _row_float(best_ours, "test_mean_iou"),
        "best_ours_method": _row_str(best_ours, "method_key"),
        "validation_selected_ours_iou": _row_float(
            validation_selected, "test_mean_iou"
        ),
        "validation_selected_ours_method": _row_str(validation_selected, "method_key"),
        "validation_selected_validation_iou": _row_float(
            validation_selected, "validation_mean_iou"
        ),
        "oracle_bound_iou": _row_float(oracle, "test_mean_iou"),
        "competitor_prior_iou": _row_float(competitor_prior, "test_mean_iou"),
    }


def _best_row(df: pd.DataFrame, metric: str) -> Optional[pd.Series]:
    if df.empty or metric not in df.columns:
        return None
    candidates = df.dropna(subset=[metric]).copy()
    if candidates.empty:
        return None
    sort_columns = [metric]
    f1_metric = metric.replace("_iou", "_f1")
    if f1_metric in candidates.columns:
        sort_columns.append(f1_metric)
    return candidates.sort_values(sort_columns, ascending=False).iloc[0]


def _row_float(row: Optional[pd.Series], column: str) -> float:
    if row is None or column not in row or pd.isna(row[column]):
        return float("nan")
    return float(row[column])


def _row_str(row: Optional[pd.Series], column: str) -> str:
    if row is None or column not in row or pd.isna(row[column]):
        return ""
    return str(row[column])


def _build_mean_summary_row(group: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": str(group["dataset"].iloc[0]),
        "crop_tag": str(group["crop_tag"].iloc[0]),
        "variant": str(group["variant"].iloc[0]),
        "fold": "mean",
        "row_type": "mean",
        "best_fusion_method": "foldwise best",
        "best_competitor_name": _mean_name(group["best_competitor_name"]),
        "median_competitor_name": "",
        "worst_competitor_name": _mean_name(group["worst_competitor_name"]),
        "best_ours_method": "foldwise best",
        "validation_selected_ours_method": "validation-selected per fold",
        "ablation_method_summary_path": "",
    }
    for column in _paper_result_numeric_columns():
        row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
    return row


def _mean_name(values: pd.Series) -> str:
    unique = [str(value) for value in values.dropna().unique() if str(value)]
    if len(unique) == 1:
        return unique[0]
    return "foldwise"


def _paper_result_numeric_columns() -> list[str]:
    return [
        "best_fusion_iou",
        "best_competitor_iou",
        "median_competitor_iou",
        "worst_competitor_iou",
        "silver_truth_iou",
        "best_ours_iou",
        "validation_selected_ours_iou",
        "validation_selected_validation_iou",
        "oracle_bound_iou",
        "competitor_prior_iou",
    ]


def _paper_result_columns() -> list[str]:
    return [
        "dataset",
        "crop_tag",
        "variant",
        "fold",
        "row_type",
        "best_fusion_iou",
        "best_fusion_method",
        "best_competitor_iou",
        "best_competitor_name",
        "median_competitor_iou",
        "median_competitor_name",
        "worst_competitor_iou",
        "worst_competitor_name",
        "silver_truth_iou",
        "best_ours_iou",
        "best_ours_method",
        "validation_selected_ours_iou",
        "validation_selected_ours_method",
        "validation_selected_validation_iou",
        "oracle_bound_iou",
        "competitor_prior_iou",
        "ablation_method_summary_path",
    ]


cli.add_command(evaluate_competitor)
cli.add_command(calculate_evaluation_metrics_cli)
cli.add_command(evaluate_qa_model)
cli.add_command(evaluate_qa_filtering)
cli.add_command(merge_qa_predictions)
cli.add_command(audit_experiment_inputs)
cli.add_command(evaluate_fusion_crops)
cli.add_command(filter_parquet)
cli.add_command(score_parquet)
cli.add_command(report_hsc_results)
cli.add_command(report_hsc_main_comparison)
cli.add_command(report_overflow_impact)
cli.add_command(report_ablation)
cli.add_command(select_ablation_threshold)
cli.add_command(summarize_paper_results)


if __name__ == "__main__":
    cli()
