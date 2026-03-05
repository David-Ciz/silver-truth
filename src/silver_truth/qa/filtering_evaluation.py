"""QA filtering evaluation orchestration and metrics."""

from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from silver_truth.experiment_tracking import DEFAULT_MLFLOW_TRACKING_URI

TRUE_COLUMN_CANDIDATES = ("Jaccard index", "jaccard_index")
PRED_COLUMN_CANDIDATES = ("Predicted Jaccard index", "predicted_jaccard_index")


def parse_thresholds(thresholds_csv: str) -> List[float]:
    """Parse and validate a comma-separated list of thresholds in [0, 1]."""
    values: List[float] = []
    for raw in thresholds_csv.split(","):
        value = float(raw.strip())
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Threshold must be in [0,1], got {value}")
        values.append(value)
    return sorted(set(values))


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _select_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing required column. Candidates: {candidates}")


def calculate_filtering_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Calculate thresholded QA filtering metrics for one split."""
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty arrays provided.")

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n_samples = int(len(y_true))
    if n_samples == 0:
        raise ValueError("No valid samples after NaN filtering.")

    actual_good = y_true >= threshold
    predicted_good = y_pred >= threshold
    predicted_bad = ~predicted_good
    actual_bad = ~actual_good

    tp = int(np.sum(predicted_good & actual_good))
    fp = int(np.sum(predicted_good & actual_bad))
    fn = int(np.sum(predicted_bad & actual_good))
    tn = int(np.sum(predicted_bad & actual_bad))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    balanced_accuracy = 0.5 * (recall + specificity)

    filtered_count = int(np.sum(predicted_bad))
    kept_count = int(np.sum(predicted_good))

    return {
        "threshold": float(threshold),
        "n_samples": float(n_samples),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "actual_good_count": float(np.sum(actual_good)),
        "actual_bad_count": float(np.sum(actual_bad)),
        "kept_count": float(kept_count),
        "filtered_count": float(filtered_count),
        "kept_pct": 100.0 * _safe_div(kept_count, n_samples),
        "filtered_pct": 100.0 * _safe_div(filtered_count, n_samples),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "balanced_accuracy": balanced_accuracy,
    }


def _plot_filter_curves(results_df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    for split in sorted(results_df["split"].unique()):
        split_df = results_df[results_df["split"] == split].sort_values("threshold")
        thresholds = split_df["threshold"].values

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(thresholds, split_df["precision"], marker="o", label="precision")
        axes[0].plot(thresholds, split_df["recall"], marker="o", label="recall")
        axes[0].plot(thresholds, split_df["f1"], marker="o", label="f1")
        axes[0].set_title(f"{split}: Quality-classification metrics")
        axes[0].set_xlabel("Threshold")
        axes[0].set_ylabel("Score")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(
            thresholds, split_df["filtered_pct"], marker="o", label="% filtered"
        )
        axes[1].plot(thresholds, split_df["fpr"] * 100.0, marker="o", label="fpr (%)")
        axes[1].plot(thresholds, split_df["fnr"] * 100.0, marker="o", label="fnr (%)")
        axes[1].set_title(f"{split}: Filtering behavior")
        axes[1].set_xlabel("Threshold")
        axes[1].set_ylabel("Percent")
        axes[1].set_ylim(0.0, 100.0)
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        out_path = output_dir / f"{stem}_{split}_filtering_curves.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def evaluate_qa_filtering_from_excel(
    excel_path: Path,
    thresholds: List[float],
    output_dir: Optional[Path] = None,
    generate_plots: bool = True,
) -> Dict[str, object]:
    """Evaluate QA filtering performance across all sheets in an Excel file."""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    if not thresholds:
        raise ValueError("At least one threshold is required.")

    if output_dir is None:
        output_dir = excel_path.parent / f"{excel_path.stem}_filtering_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(excel_path)
    rows: List[Dict[str, Any]] = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        if df.empty:
            continue

        true_col = _select_column(df, TRUE_COLUMN_CANDIDATES)
        pred_col = _select_column(df, PRED_COLUMN_CANDIDATES)
        y_true = df[true_col].to_numpy(dtype=float)
        y_pred = df[pred_col].to_numpy(dtype=float)

        for threshold in thresholds:
            metrics = calculate_filtering_metrics(y_true, y_pred, threshold)
            row: Dict[str, Any] = {**metrics, "split": str(sheet)}
            rows.append(row)

    if not rows:
        raise ValueError("No valid split rows found in the input Excel file.")

    results_df = (
        pd.DataFrame(rows).sort_values(["split", "threshold"]).reset_index(drop=True)
    )
    metrics_csv_path = output_dir / f"{excel_path.stem}_filtering_metrics.csv"
    results_df.to_csv(metrics_csv_path, index=False)

    summary_rows = []
    for split in sorted(results_df["split"].unique()):
        split_df = results_df[results_df["split"] == split]
        best_row = split_df.sort_values(
            by=["f1", "fnr", "threshold"],
            ascending=[False, True, True],
        ).iloc[0]
        summary_rows.append(best_row.to_dict())

    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)
    summary_csv_path = output_dir / f"{excel_path.stem}_filtering_best_thresholds.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    if generate_plots:
        _plot_filter_curves(results_df, output_dir, excel_path.stem)

    return {
        "results_df": results_df,
        "summary_df": summary_df,
        "output_dir": output_dir,
        "metrics_csv_path": metrics_csv_path,
        "summary_csv_path": summary_csv_path,
    }


def _log_qa_filtering_metrics_to_mlflow(
    result: Dict[str, object],
    excel_path: Path,
    mlflow_tracking_uri: str,
    mlflow_run_id: Optional[str],
    mlflow_experiment: Optional[str],
    mlflow_run_name: Optional[str],
) -> None:
    metrics_df = result["results_df"]
    output_dir = result["output_dir"]

    def log_threshold_metrics() -> None:
        tracked_metrics = [
            "precision",
            "recall",
            "f1",
            "specificity",
            "fpr",
            "fnr",
            "balanced_accuracy",
            "filtered_pct",
        ]
        for row in metrics_df.itertuples(index=False):
            threshold_tag = f"t{int(round(float(row.threshold) * 100)):02d}"
            split_name = str(row.split)
            for metric_name in tracked_metrics:
                mlflow.log_metric(
                    f"{split_name}_{threshold_tag}_{metric_name}",
                    float(getattr(row, metric_name)),
                )

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            log_threshold_metrics()
            mlflow.log_artifact(str(excel_path))
            mlflow.log_artifacts(str(output_dir), artifact_path="qa_filtering")
        return

    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=mlflow_run_name):
        mlflow.log_param("excel_path", str(excel_path))
        log_threshold_metrics()
        mlflow.log_artifact(str(excel_path))
        mlflow.log_artifacts(str(output_dir), artifact_path="qa_filtering")


def run_qa_filtering_evaluation(
    excel_path: Path,
    thresholds_csv: str,
    output_dir: Optional[Path] = None,
    generate_plots: bool = True,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_run_id: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    mlflow_run_name: Optional[str] = None,
) -> Dict[str, object]:
    """Main entry point used by CLI and DVC stages."""
    thresholds = parse_thresholds(thresholds_csv)
    result = evaluate_qa_filtering_from_excel(
        excel_path=excel_path,
        thresholds=thresholds,
        output_dir=output_dir,
        generate_plots=generate_plots,
    )

    if mlflow_run_id or mlflow_experiment:
        _log_qa_filtering_metrics_to_mlflow(
            result=result,
            excel_path=excel_path,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_run_id=mlflow_run_id,
            mlflow_experiment=mlflow_experiment,
            mlflow_run_name=mlflow_run_name,
        )

    return result
