"""
QA Model Evaluation Module.

This module provides functions to evaluate the performance of trained QA models
that predict Jaccard index values. It includes:
- Regression metrics (R², MAE, RMSE, MSE)
- Tolerance-based accuracy at various thresholds
- Correlation analysis
- Visualization (scatter plots, residual plots)
- Split-based evaluation (train/val/test)

NOTE: This evaluates the QA MODEL predictions, not the final ensemble results.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics for QA model evaluation.

    Args:
        y_true: Ground truth Jaccard index values
        y_pred: Predicted Jaccard index values

    Returns:
        Dictionary with R², MAE, RMSE, MSE, and correlation metrics
    """
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return {}

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    metrics = {
        "r2_score": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mse": mean_squared_error(y_true, y_pred),
        "pearson_correlation": stats.pearsonr(y_true, y_pred)[0],
        "pearson_pvalue": stats.pearsonr(y_true, y_pred)[1],
        "spearman_correlation": stats.spearmanr(y_true, y_pred)[0],
        "spearman_pvalue": stats.spearmanr(y_true, y_pred)[1],
        "mean_residual": np.mean(y_pred - y_true),
        "std_residual": np.std(y_pred - y_true),
        "n_samples": len(y_true),
    }

    return metrics


def calculate_tolerance_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, tolerances: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Calculate tolerance-based accuracy at various thresholds.

    Tolerance accuracy measures the percentage of predictions that fall
    within a specified tolerance of the true value.

    Args:
        y_true: Ground truth Jaccard index values
        y_pred: Predicted Jaccard index values
        tolerances: List of tolerance thresholds (e.g., 0.05 = 5%)

    Returns:
        Dictionary with accuracy at each tolerance level
    """
    if tolerances is None:
        tolerances = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    abs_errors = np.abs(y_pred - y_true)

    tolerance_metrics = {}
    for tol in tolerances:
        accuracy = np.mean(abs_errors <= tol) * 100
        tolerance_metrics[f"accuracy_within_{int(tol*100):02d}pct"] = accuracy

    return tolerance_metrics


def evaluate_qa_model_from_excel(
    excel_path: Path, output_dir: Optional[Path] = None, generate_plots: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate QA model predictions from an Excel file with train/val/test sheets.

    Args:
        excel_path: Path to Excel file with sheets: 'train', 'validation', 'test'
        output_dir: Directory to save plots and results (optional)
        generate_plots: Whether to generate visualization plots

    Returns:
        Dictionary with evaluation results per split
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load all sheets
    xl = pd.ExcelFile(excel_path)
    sheet_names = xl.sheet_names

    logging.info(f"Loading QA model results from: {excel_path}")
    logging.info(f"Found sheets: {sheet_names}")

    all_results = {}
    all_data = {}

    for sheet in sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        logging.info(f"\n=== Evaluating {sheet.upper()} set ({len(df)} samples) ===")

        # Standardize column names
        if "Jaccard index" in df.columns:
            y_true = df["Jaccard index"].values
        elif "jaccard_index" in df.columns:
            y_true = df["jaccard_index"].values
        else:
            logging.error(f"No Jaccard index column found in {sheet}")
            continue

        if "Predicted Jaccard index" in df.columns:
            y_pred = df["Predicted Jaccard index"].values
        elif "predicted_jaccard_index" in df.columns:
            y_pred = df["predicted_jaccard_index"].values
        else:
            logging.error(f"No Predicted Jaccard index column found in {sheet}")
            continue

        # Calculate metrics
        regression_metrics = calculate_regression_metrics(y_true, y_pred)
        tolerance_metrics = calculate_tolerance_accuracy(y_true, y_pred)

        # Combine metrics
        split_results = {"split": sheet, **regression_metrics, **tolerance_metrics}

        all_results[sheet] = split_results
        all_data[sheet] = df

        # Print results
        print_evaluation_results(split_results, sheet)

    # Generate combined analysis
    if len(all_data) > 1:
        combined_df = pd.concat(
            [df.assign(split=sheet) for sheet, df in all_data.items()],
            ignore_index=True,
        )

        y_true_all = (
            combined_df["Jaccard index"].values
            if "Jaccard index" in combined_df.columns
            else combined_df["jaccard_index"].values
        )
        y_pred_all = (
            combined_df["Predicted Jaccard index"].values
            if "Predicted Jaccard index" in combined_df.columns
            else combined_df["predicted_jaccard_index"].values
        )

        logging.info(
            f"\n=== COMBINED EVALUATION (all splits, {len(combined_df)} samples) ==="
        )
        combined_metrics = {
            **calculate_regression_metrics(y_true_all, y_pred_all),
            **calculate_tolerance_accuracy(y_true_all, y_pred_all),
        }
        all_results["combined"] = combined_metrics
        print_evaluation_results(combined_metrics, "combined")

    # Generate plots
    if generate_plots:
        if output_dir:
            plot_dir = output_dir
        else:
            plot_dir = excel_path.parent / f"{excel_path.stem}_evaluation"
            plot_dir.mkdir(exist_ok=True)

        generate_evaluation_plots(all_data, plot_dir, excel_path.stem)
        logging.info(f"\nPlots saved to: {plot_dir}")

    # Save results to CSV
    if output_dir:
        results_df = pd.DataFrame(all_results).T
        results_path = output_dir / f"{excel_path.stem}_evaluation_metrics.csv"
        results_df.to_csv(results_path)
        logging.info(f"Metrics saved to: {results_path}")

    return all_results


def print_evaluation_results(results: Dict[str, Any], split_name: str):
    """Print formatted evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SET EVALUATION")
    print(f"{'='*60}")

    if "n_samples" in results:
        print(f"  Samples: {results['n_samples']}")

    print("\n  Regression Metrics:")
    print("  -------------------")
    if "r2_score" in results:
        print(f"    R² Score:           {results['r2_score']:.4f}")
    if "mae" in results:
        print(f"    MAE:                {results['mae']:.4f}")
    if "rmse" in results:
        print(f"    RMSE:               {results['rmse']:.4f}")
    if "pearson_correlation" in results:
        print(
            f"    Pearson Corr:       {results['pearson_correlation']:.4f} (p={results.get('pearson_pvalue', 'N/A'):.2e})"
        )
    if "spearman_correlation" in results:
        print(f"    Spearman Corr:      {results['spearman_correlation']:.4f}")

    print("\n  Tolerance-Based Accuracy:")
    print("  -------------------------")
    tolerance_keys = [k for k in results.keys() if k.startswith("accuracy_within_")]
    for key in sorted(tolerance_keys):
        tol_pct = key.replace("accuracy_within_", "").replace("pct", "")
        print(f"    Within ±{tol_pct}%:        {results[key]:.1f}%")

    print(f"{'='*60}\n")


def generate_evaluation_plots(
    all_data: Dict[str, pd.DataFrame], output_dir: Path, model_name: str
):
    """
    Generate comprehensive visualization plots for QA model evaluation.

    Creates:
    - Scatter plot: Predicted vs Actual (per split)
    - Combined scatter plot with all splits
    - Residual distribution plot
    - Error distribution by Jaccard range
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"train": "#2ecc71", "validation": "#3498db", "test": "#e74c3c"}

    # 1. Individual scatter plots per split
    fig, axes = plt.subplots(1, len(all_data), figsize=(6 * len(all_data), 5))
    if len(all_data) == 1:
        axes = [axes]

    for ax, (split, df) in zip(axes, all_data.items()):
        y_true = (
            df["Jaccard index"].values
            if "Jaccard index" in df.columns
            else df["jaccard_index"].values
        )
        y_pred = (
            df["Predicted Jaccard index"].values
            if "Predicted Jaccard index" in df.columns
            else df["predicted_jaccard_index"].values
        )

        color = colors.get(split, "#95a5a6")
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, c=color, edgecolors="none")

        # Perfect prediction line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect prediction")

        # Add regression line
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        z = np.polyfit(y_true[mask], y_pred[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(y_true[mask].min(), y_true[mask].max(), 100)
        ax.plot(
            x_line,
            p(x_line),
            color="red",
            alpha=0.7,
            linewidth=2,
            label=f"Fit: y={z[0]:.2f}x+{z[1]:.2f}",
        )

        # Calculate metrics for title
        r2 = r2_score(y_true[mask], y_pred[mask])
        mae = mean_absolute_error(y_true[mask], y_pred[mask])

        ax.set_xlabel("Actual Jaccard Index", fontsize=11)
        ax.set_ylabel("Predicted Jaccard Index", fontsize=11)
        ax.set_title(
            f"{split.upper()} (n={len(df)}, R²={r2:.3f}, MAE={mae:.3f})", fontsize=12
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name}_scatter_per_split.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # 2. Combined scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))

    for split, df in all_data.items():
        y_true = (
            df["Jaccard index"].values
            if "Jaccard index" in df.columns
            else df["jaccard_index"].values
        )
        y_pred = (
            df["Predicted Jaccard index"].values
            if "Predicted Jaccard index" in df.columns
            else df["predicted_jaccard_index"].values
        )

        color = colors.get(split, "#95a5a6")
        ax.scatter(
            y_true,
            y_pred,
            alpha=0.5,
            s=20,
            c=color,
            edgecolors="none",
            label=f"{split} (n={len(df)})",
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Jaccard Index", fontsize=12)
    ax.set_ylabel("Predicted Jaccard Index", fontsize=12)
    ax.set_title(f"{model_name} - Predicted vs Actual Jaccard Index", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name}_scatter_combined.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # 3. Residual distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    all_residuals = []
    all_y_true = []
    split_labels = []

    for split, df in all_data.items():
        y_true = (
            df["Jaccard index"].values
            if "Jaccard index" in df.columns
            else df["jaccard_index"].values
        )
        y_pred = (
            df["Predicted Jaccard index"].values
            if "Predicted Jaccard index" in df.columns
            else df["predicted_jaccard_index"].values
        )
        residuals = y_pred - y_true

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        all_residuals.extend(residuals[mask])
        all_y_true.extend(y_true[mask])
        split_labels.extend([split] * mask.sum())

    all_residuals = np.array(all_residuals)
    all_y_true = np.array(all_y_true)

    # Histogram of residuals
    axes[0].hist(all_residuals, bins=50, edgecolor="black", alpha=0.7, color="#3498db")
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=2)
    axes[0].axvline(
        x=np.mean(all_residuals),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {np.mean(all_residuals):.4f}",
    )
    axes[0].set_xlabel("Residual (Predicted - Actual)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Distribution of Residuals", fontsize=12)
    axes[0].legend()

    # Residuals vs Actual
    axes[1].scatter(all_y_true, all_residuals, alpha=0.3, s=15, c="#3498db")
    axes[1].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Actual Jaccard Index", fontsize=11)
    axes[1].set_ylabel("Residual (Predicted - Actual)", fontsize=11)
    axes[1].set_title("Residuals vs Actual Values", fontsize=12)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name}_residuals.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # 4. Error analysis by Jaccard range
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bins for Jaccard ranges
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    mae_by_bin = []
    counts_by_bin = []

    for i in range(len(bins) - 1):
        mask = (all_y_true >= bins[i]) & (all_y_true < bins[i + 1])
        if mask.sum() > 0:
            mae_by_bin.append(np.mean(np.abs(all_residuals[mask])))
            counts_by_bin.append(mask.sum())
        else:
            mae_by_bin.append(0)
            counts_by_bin.append(0)

    bars = ax.bar(bin_labels, mae_by_bin, color="#3498db", edgecolor="black", alpha=0.7)

    # Add count labels on bars
    for bar, count in zip(bars, counts_by_bin):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Actual Jaccard Index Range", fontsize=12)
    ax.set_ylabel("Mean Absolute Error", fontsize=12)
    ax.set_title("MAE by Jaccard Index Range", fontsize=14)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name}_error_by_range.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    logging.info(
        "Generated plots: scatter_per_split, scatter_combined, residuals, error_by_range"
    )


def merge_predictions_to_parquet(
    parquet_path: Path, excel_path: Path, output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Merge predicted Jaccard index values from Excel to the parquet file.

    Args:
        parquet_path: Path to the QA parquet file (with split column)
        excel_path: Path to Excel file with predictions
        output_path: Path to save merged parquet (optional, defaults to adding '_with_predictions' suffix)

    Returns:
        DataFrame with added 'predicted_jaccard_index' column
    """
    # Load parquet
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded parquet with {len(df)} rows")

    # Load Excel sheets
    xl = pd.ExcelFile(excel_path)
    predictions_dfs = []

    for sheet in xl.sheet_names:
        sheet_df = pd.read_excel(xl, sheet_name=sheet)

        # Standardize column names
        if "Predicted Jaccard index" in sheet_df.columns:
            sheet_df = sheet_df.rename(
                columns={"Predicted Jaccard index": "predicted_jaccard_index"}
            )

        predictions_dfs.append(sheet_df[["cell_id", "predicted_jaccard_index"]])

    predictions_df = pd.concat(predictions_dfs, ignore_index=True)
    logging.info(f"Loaded {len(predictions_df)} predictions from Excel")

    # Merge
    df = df.merge(predictions_df, on="cell_id", how="left")

    # Count merged
    merged_count = df["predicted_jaccard_index"].notna().sum()
    logging.info(f"Successfully merged {merged_count}/{len(df)} predictions")

    # Save
    if output_path is None:
        output_path = parquet_path.with_name(
            parquet_path.stem + "_with_predictions.parquet"
        )

    df.to_parquet(output_path, index=False)
    logging.info(f"Saved merged parquet to: {output_path}")

    return df


def evaluate_predictions_by_split_from_parquet(
    parquet_path: Path, generate_plots: bool = True, output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate QA model predictions from a parquet file with split and predicted_jaccard_index columns.

    This is useful after merging predictions into the parquet file.

    Args:
        parquet_path: Path to parquet file with 'split', 'jaccard_score', and 'predicted_jaccard_index'
        generate_plots: Whether to generate visualization plots
        output_dir: Directory to save results

    Returns:
        Dictionary with evaluation results per split
    """
    df = pd.read_parquet(parquet_path)

    required_cols = ["split", "jaccard_score", "predicted_jaccard_index"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_results = {}
    all_data = {}

    for split in df["split"].unique():
        split_df = df[df["split"] == split].copy()

        # Rename for consistency with Excel evaluation
        split_df = split_df.rename(
            columns={
                "jaccard_score": "Jaccard index",
                "predicted_jaccard_index": "Predicted Jaccard index",
            }
        )

        y_true = split_df["Jaccard index"].values
        y_pred = split_df["Predicted Jaccard index"].values

        logging.info(
            f"\n=== Evaluating {split.upper()} set ({len(split_df)} samples) ==="
        )

        regression_metrics = calculate_regression_metrics(y_true, y_pred)
        tolerance_metrics = calculate_tolerance_accuracy(y_true, y_pred)

        split_results = {"split": split, **regression_metrics, **tolerance_metrics}

        all_results[split] = split_results
        all_data[split] = split_df

        print_evaluation_results(split_results, split)

    if generate_plots and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        generate_evaluation_plots(all_data, output_dir, parquet_path.stem)

    return all_results
