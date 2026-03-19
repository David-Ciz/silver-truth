import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    REFERENCE_COLUMNS_ATTR,
    SILVER_TRUTH_COLUMN,
)
from silver_truth.metrics.evaluation_logic import run_evaluation
from silver_truth.experiment_tracking import (
    DEFAULT_MLFLOW_TRACKING_URI,
    log_standardized_single_split_metrics,
    set_common_mlflow_tags,
    set_evaluation_tags,
    start_managed_mlflow_run,
)


def evaluate_competitor_logic(
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
    With --mlflow-experiment, logs per-split/per-campaign Jaccard metrics to MLflow
    (one run per competitor).
    """
    # Run standard evaluation
    results = run_evaluation(
        dataset_dataframe_path=dataset_dataframe_path,
        competitor=competitor,
        output=output,
        visualize=visualize,
        campaign_col=campaign_col,
    )

    # Log to MLflow if requested
    if mlflow_experiment and results:
        _log_competitor_metrics_to_mlflow(
            results=results,
            dataset_dataframe_path=dataset_dataframe_path,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
            mlflow_run_name=mlflow_run_name,
        )

    # Run detailed evaluation if requested
    if detailed:
        try:
            from detailed_evaluation import DetailedCellEvaluator

            logging.info("Starting detailed per-cell evaluation...")

            # Load dataset
            df = pd.read_parquet(dataset_dataframe_path)
            evaluator = DetailedCellEvaluator(df)

            # Determine output path for detailed results
            if output:
                detailed_output = output.with_suffix(".parquet").with_name(
                    output.stem + "_detailed.parquet"
                )
            else:
                dataset_name = dataset_dataframe_path.stem
                detailed_output = Path(f"{dataset_name}_detailed_evaluation.parquet")

            # Run detailed evaluation
            if competitor:
                detailed_results = evaluator.evaluate_competitor_detailed(
                    competitor, campaign_col
                )
            else:
                reference_columns = set(df.attrs.get(REFERENCE_COLUMNS_ATTR, []))
                # Auto-detect competitors and evaluate all
                potential_competitors = [
                    col
                    for col in df.columns
                    if col
                    not in [
                        "composite_key",
                        "raw_image",
                        "gt_image",
                        campaign_col,
                        "sequence_id",
                        "time_id",
                        "tracking_markers",
                    ]
                    and col not in reference_columns
                    and col != SILVER_TRUTH_COLUMN
                    and df[col].dtype == "object"
                    and df[col].notna().any()
                ]

                competitors = []
                for col in potential_competitors:
                    sample_val = (
                        df[col].dropna().iloc[0] if df[col].notna().any() else ""
                    )
                    if isinstance(sample_val, str) and sample_val.endswith(
                        (".tif", ".tiff")
                    ):
                        competitors.append(col)

                all_detailed_results = []
                for comp in competitors:
                    logging.info(f"Detailed evaluation for competitor: {comp}")
                    comp_results = evaluator.evaluate_competitor_detailed(
                        comp, campaign_col
                    )
                    if not comp_results.empty:
                        all_detailed_results.append(comp_results)

                if all_detailed_results:
                    detailed_results = pd.concat(
                        all_detailed_results, ignore_index=True
                    )
                else:
                    detailed_results = pd.DataFrame()

            # Save detailed results
            if not detailed_results.empty:
                detailed_results.to_parquet(detailed_output)
                logging.info(f"Detailed results saved to: {detailed_output}")
                logging.info(f"Total cells evaluated: {len(detailed_results):,}")
                logging.info(
                    f"Average Jaccard score: {detailed_results['jaccard_score'].mean():.4f}"
                )
            else:
                logging.warning("No detailed evaluation results generated")

        except ImportError:
            logging.error(
                "detailed_evaluation module not found. Please ensure detailed_evaluation.py is available."
            )
        except Exception as e:
            logging.error(f"Error during detailed evaluation: {e}")


def _log_competitor_metrics_to_mlflow(
    results: dict,
    dataset_dataframe_path: Path,
    mlflow_tracking_uri: str,
    mlflow_experiment: str,
    mlflow_run_name: Optional[str],
) -> None:
    """
    Log per-split, per-campaign and overall Jaccard averages to MLflow.
    One run is created per competitor so they appear as separate entries in the UI.
    """
    try:
        import mlflow
    except ImportError:
        logging.error("mlflow not installed — cannot log competitor metrics.")
        return

    overall_averages = results.get("overall_averages", {})
    overall_f1_averages = results.get("overall_f1_averages", {})
    per_campaign_averages = results.get("per_campaign_averages", {})
    per_split_averages = results.get("per_split_averages", {})
    per_split_f1_averages = results.get("per_split_f1_averages", {})
    competitors = list(overall_averages.keys())

    dataset_name = dataset_dataframe_path.stem

    for comp in competitors:
        run_name = f"{mlflow_run_name}__{comp}" if mlflow_run_name else comp
        with start_managed_mlflow_run(
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
            run_name=run_name,
        ):
            run = mlflow.active_run()
            if run:
                logging.info(
                    f"MLflow: logging '{comp}' → experiment '{mlflow_experiment}' run {run.info.run_id}"
                )

            is_reference = comp == SILVER_TRUTH_COLUMN
            set_common_mlflow_tags(dataset=dataset_name, split="full_image_label")
            extra_tags = {
                "competitor": comp,
                "f1_available": "true",
            }
            if is_reference:
                extra_tags["reference_kind"] = "silver_truth"

            set_evaluation_tags(
                pipeline_family="reference" if is_reference else "competitor",
                evaluation_level="full_image_label",
                setup_name="silver_truth" if is_reference else comp,
                extra_tags=extra_tags,
            )
            mlflow.log_param("competitor", comp)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("parquet", str(dataset_dataframe_path))

            # Overall
            overall = overall_averages.get(comp, float("nan"))
            if not pd.isna(overall):
                mlflow.log_metric("jaccard_overall", overall)
            overall_f1 = overall_f1_averages.get(comp, float("nan"))
            if not pd.isna(overall_f1):
                mlflow.log_metric("f1_overall", overall_f1)

            # Per-split  (test = the paper number you care about)
            for split_name, metric_key in [
                ("train", "jaccard_train"),
                ("validation", "jaccard_val"),
                ("test", "jaccard_test"),
            ]:
                val = per_split_averages.get(comp, {}).get(split_name, float("nan"))
                if not pd.isna(val):
                    mlflow.log_metric(metric_key, val)
            for split_name, metric_key in [
                ("train", "f1_train"),
                ("validation", "f1_val"),
                ("test", "f1_test"),
            ]:
                val = per_split_f1_averages.get(comp, {}).get(split_name, float("nan"))
                if not pd.isna(val):
                    mlflow.log_metric(metric_key, val)

            log_standardized_single_split_metrics(
                split="test",
                iou=per_split_averages.get(comp, {}).get("test"),
                f1=per_split_f1_averages.get(comp, {}).get("test"),
            )
            log_standardized_single_split_metrics(
                split="validation",
                iou=per_split_averages.get(comp, {}).get("validation"),
                f1=per_split_f1_averages.get(comp, {}).get("validation"),
                count=None,
            )
            log_standardized_single_split_metrics(
                split="train",
                iou=per_split_averages.get(comp, {}).get("train"),
                f1=per_split_f1_averages.get(comp, {}).get("train"),
                count=None,
            )
            overall_iou = overall_averages.get(comp, float("nan"))
            if not pd.isna(overall_iou):
                mlflow.log_metric("overall_iou", float(overall_iou))
            if not pd.isna(overall_f1):
                mlflow.log_metric("overall_f1", float(overall_f1))

            # Per-campaign
            for campaign, camp_avg in per_campaign_averages.get(comp, {}).items():
                if not pd.isna(camp_avg):
                    mlflow.log_metric(f"jaccard_campaign_{campaign}", camp_avg)
