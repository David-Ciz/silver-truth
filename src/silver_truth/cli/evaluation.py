import click
from pathlib import Path
from typing import Optional
import logging

import mlflow
import pandas as pd

from silver_truth.ensemble.reconstruction import reconstruct_full_images_from_paths
from silver_truth.evaluation.evaluation_logic import evaluate_competitor_logic
from silver_truth.evaluation.stacked_jaccard_logic import (
    calculate_evaluation_metrics,
    calculate_evaluation_metrics_cropped,
)
from silver_truth.evaluation.reporting import (
    generate_hsc_reporting_bundle,
    write_hsc_reporting_bundle,
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
) -> None:
    """
    Reconstruct full-image segmentations from per-cell fused crops and evaluate IoU/F1.

    Reads the parquet produced by ``silver-fusion run-crops-experiment``, places each
    fused crop back into the full image at its recorded coordinates, and scores the
    result against the full GT mask.

    The parquet must contain ``gt_image``, ``crop_y_start/end/x_start/x_end`` (or
    their ``recon_crop_*`` equivalents), and the fused-path column specified by
    ``--fused-path-column``.
    """
    df = pd.read_parquet(parquet_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = reconstruct_full_images_from_paths(
        databank_df=df,
        fused_path_column=fused_path_column,
        output_dir=output_dir,
        threshold=threshold,
    )

    if results_df.empty:
        logging.warning(
            "No images could be reconstructed — check paths and GT columns."
        )
    else:
        mean_iou = results_df["iou"].mean()
        mean_f1 = results_df["f1"].mean()
        logging.info(
            "Reconstructed %d images — mean IoU=%.4f  mean F1=%.4f",
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
            set_common_mlflow_tags(dataset=dataset_tag, split="image_reconstructed")
            set_evaluation_tags(
                pipeline_family=pipeline_family,
                evaluation_level="image_reconstructed",
                setup_name=setup_name or fused_path_column,
                qa_mode=qa_mode,
                qa_threshold=qa_threshold,
                extra_tags={
                    "fused_path_column": fused_path_column,
                },
            )
            mlflow.log_param("parquet_path", str(parquet_path))
            mlflow.log_param("fused_path_column", fused_path_column)
            mlflow.log_param("output_csv", str(csv_path))
            mlflow.log_param("threshold", threshold)
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


cli.add_command(evaluate_competitor)
cli.add_command(calculate_evaluation_metrics_cli)
cli.add_command(evaluate_qa_model)
cli.add_command(evaluate_qa_filtering)
cli.add_command(merge_qa_predictions)
cli.add_command(evaluate_fusion_crops)
cli.add_command(filter_parquet)
cli.add_command(report_hsc_results)


if __name__ == "__main__":
    cli()
