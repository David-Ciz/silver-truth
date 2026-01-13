import pandas as pd
import numpy as np
from scipy import stats


def analyze_model(excel_file, model_name):
    """Analyze model performance from results Excel file."""

    print("=" * 70)
    print(f"MODEL ANALYSIS: {model_name}")
    print(f"File: {excel_file}")
    print("=" * 70)
    print()

    # Load all sheets
    train_df = pd.read_excel(excel_file, sheet_name="train")
    val_df = pd.read_excel(excel_file, sheet_name="validation")
    test_df = pd.read_excel(excel_file, sheet_name="test")

    # Standardize column names
    for df in [train_df, val_df, test_df]:
        if "Jaccard index" in df.columns:
            df.rename(
                columns={
                    "Jaccard index": "jaccard_score",
                    "Predicted Jaccard index": "predicted_jaccard",
                },
                inplace=True,
            )

    # Combine all data
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print(
        f"Total samples: {len(all_df)} (train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)})"
    )
    print()

    # === Basic Metrics ===
    print("=== Basic Metrics by Split ===")
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        y_true = df["jaccard_score"].values
        y_pred = df["predicted_jaccard"].values

        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        pearson_r, _ = stats.pearsonr(y_true, y_pred)
        spearman_r, _ = stats.spearmanr(y_true, y_pred)

        print(
            f"{name:12} R2={r2:7.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  Pearson={pearson_r:.4f}  Spearman={spearman_r:.4f}"
        )

    print()

    # === Prediction Range ===
    print("=== Prediction Statistics ===")
    print(f'Predicted min: {all_df["predicted_jaccard"].min():.4f}')
    print(f'Predicted max: {all_df["predicted_jaccard"].max():.4f}')
    print(f'Predicted mean: {all_df["predicted_jaccard"].mean():.4f}')
    print(
        f'Predictions > 1.0: {(all_df["predicted_jaccard"] > 1.0).sum()} ({(all_df["predicted_jaccard"] > 1.0).sum()/len(all_df)*100:.1f}%)'
    )
    print(f'Predictions < 0.5: {(all_df["predicted_jaccard"] < 0.5).sum()}')
    print()

    # === Model Predictions by Actual Quality ===
    print("=== Model Predictions by Actual Quality ===")
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ["<0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    all_df["quality_bin"] = pd.cut(all_df["jaccard_score"], bins=bins, labels=labels)

    for label in labels:
        subset = all_df[all_df["quality_bin"] == label]
        if len(subset) > 0:
            print(
                f'  Actual {label}: n={len(subset):4d}, predicted mean={subset["predicted_jaccard"].mean():.3f}, '
                f'min={subset["predicted_jaccard"].min():.3f}, max={subset["predicted_jaccard"].max():.3f}'
            )

    print()

    # === Threshold Analysis ===
    print("=== Using PREDICTED threshold to filter (for best ensemble) ===")
    print()
    print(
        f'{"Pred Thresh":<12} {"Kept":<8} {"% Data":<8} {"Mean Actual":<12} {"% Excellent":<12} {"% Good+":<10}'
    )
    print("-" * 70)

    pred_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    for pt in pred_thresholds:
        kept_mask = all_df["predicted_jaccard"] >= pt
        kept = all_df[kept_mask]
        n_kept = len(kept)
        pct_kept = n_kept / len(all_df) * 100
        mean_actual = kept["jaccard_score"].mean() if n_kept > 0 else 0

        pct_excellent = (
            (kept["jaccard_score"] >= 0.9).sum() / n_kept * 100 if n_kept > 0 else 0
        )
        pct_good_plus = (
            (kept["jaccard_score"] >= 0.8).sum() / n_kept * 100 if n_kept > 0 else 0
        )

        print(
            f">= {pt:<9} {n_kept:<8} {pct_kept:<8.1f} {mean_actual:<12.4f} {pct_excellent:<12.1f} {pct_good_plus:<10.1f}"
        )

    print()

    # === Best threshold recommendation ===
    print("=== Recommendation for AGGRESSIVE filtering (>= 0.85) ===")
    pt = 0.85
    kept = all_df[all_df["predicted_jaccard"] >= pt]
    if len(kept) > 0:
        print(f"    - Keeps {len(kept)} samples ({len(kept)/len(all_df)*100:.1f}%)")
        print(f'    - Mean actual Jaccard: {kept["jaccard_score"].mean():.4f}')
        print(
            f'    - {(kept["jaccard_score"] >= 0.8).sum()/len(kept)*100:.1f}% are good or better'
        )
    else:
        print("    - No samples pass this threshold!")

    print()
    return all_df


# Analyze both models
print("\n" + "#" * 70)
print("#" + " " * 20 + "OLD MODEL ANALYSIS" + " " * 28 + "#")
print("#" * 70 + "\n")
old_df = analyze_model(
    "results_resnet50_fixed.xlsx", "OLD MODEL (results_resnet50_fixed)"
)

print("\n" + "#" * 70)
print("#" + " " * 18 + "NEW MODEL (no sigmoid)" + " " * 26 + "#")
print("#" * 70 + "\n")
new_df = analyze_model(
    "results/results_resnet50_local.xlsx",
    "NEW MODEL - No Sigmoid (results_resnet50_local)",
)

print("\n" + "#" * 70)
print("#" + " " * 18 + "NEW MODEL (with sigmoid)" + " " * 24 + "#")
print("#" * 70 + "\n")
sigmoid_df = analyze_model(
    "results/results_resnet50_sigmoid.xlsx",
    "NEW MODEL - With Sigmoid (results_resnet50_sigmoid)",
)

print("\n" + "#" * 70)
print("#" + " " * 14 + "NEW MODEL (sigmoid + weighted sampling)" + " " * 13 + "#")
print("#" * 70 + "\n")
weighted_df = analyze_model(
    "results/results_resnet50_sigmoid_weighted.xlsx",
    "NEW MODEL - Sigmoid + Weighted (results_resnet50_sigmoid_weighted)",
)

# === Comparison ===
print("\n" + "=" * 80)
print("COMPARISON: All Models on TEST Set")
print("=" * 80)
print()

# Compare test set performance
old_test = pd.read_excel("results_resnet50_fixed.xlsx", sheet_name="test")
new_test = pd.read_excel("results/results_resnet50_local.xlsx", sheet_name="test")
sigmoid_test = pd.read_excel("results/results_resnet50_sigmoid.xlsx", sheet_name="test")
weighted_test = pd.read_excel(
    "results/results_resnet50_sigmoid_weighted.xlsx", sheet_name="test"
)

# Standardize names
for df in [old_test, new_test, sigmoid_test, weighted_test]:
    df.rename(
        columns={
            "Jaccard index": "jaccard_score",
            "Predicted Jaccard index": "predicted_jaccard",
        },
        inplace=True,
    )

print(
    f'{"Model":<30} {"MAE":<8} {"Spearman":<10} {"Pred Min":<10} {"Pred Max":<10} {"%Good+ @0.85":<12}'
)
print("-" * 80)

for name, df in [
    ("OLD", old_test),
    ("NEW (no sigmoid)", new_test),
    ("NEW (sigmoid)", sigmoid_test),
    ("NEW (sigmoid+weighted)", weighted_test),
]:
    y_true = df["jaccard_score"].values
    y_pred = df["predicted_jaccard"].values
    mae = np.mean(np.abs(y_pred - y_true))
    spearman_r, _ = stats.spearmanr(y_true, y_pred)
    pred_min = y_pred.min()
    pred_max = y_pred.max()
    filtered = df[df["predicted_jaccard"] >= 0.85]
    pct_good = (
        (filtered["jaccard_score"] >= 0.8).sum() / len(filtered) * 100
        if len(filtered) > 0
        else 0
    )
    print(
        f"{name:<30} {mae:<8.4f} {spearman_r:<10.4f} {pred_min:<10.4f} {pred_max:<10.4f} {pct_good:<12.1f}"
    )
