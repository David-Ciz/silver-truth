#!/usr/bin/env python3
"""
Compile experiment results into a summary markdown table.

Usage:
    python scripts/compile_results.py <results_dir>
    python scripts/compile_results.py data/paper_runs/experiment_battery_20260312_235500

Reads all evaluation parquets and CSVs from the experiment output directory
and produces a results_summary.md with comparison tables.
"""
import os
import sys
from pathlib import Path

import pandas as pd


def find_ensemble_results(base_dir: Path) -> list[dict]:
    """Walk the experiment directory and extract IoU/F1 from evaluation parquets."""
    results = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".parquet") and "cell" not in f and "databank" not in root:
                path = Path(root) / f
                try:
                    df = pd.read_parquet(path)
                    if "iou" not in df.columns:
                        continue

                    # Extract experiment name and fold from path
                    rel = path.relative_to(base_dir)
                    parts = list(rel.parts)

                    exp_name = parts[0] if len(parts) > 0 else "unknown"
                    fold = parts[1] if len(parts) > 1 else "unknown"

                    results.append({
                        "experiment": exp_name,
                        "fold": fold,
                        "iou": df["iou"].mean(),
                        "f1": df["f1"].mean(),
                        "n_images": len(df),
                        "path": str(path),
                    })
                except Exception as e:
                    print(f"  Warning: could not read {path}: {e}", file=sys.stderr)

    return results


def find_fullimage_csvs(base_dir: Path) -> list[dict]:
    """Find fullimage_eval.csv files from fusion/ablation experiments."""
    results = []
    for csv_path in base_dir.rglob("fullimage_eval.csv"):
        try:
            df = pd.read_csv(csv_path)
            if "iou" not in df.columns:
                continue

            test = df[df["split"] == "test"] if "split" in df.columns else df
            if len(test) == 0:
                continue

            rel = csv_path.relative_to(base_dir)
            parts = list(rel.parts)
            exp_name = parts[0] if len(parts) > 0 else "unknown"
            fold = parts[1] if len(parts) > 1 else "unknown"

            results.append({
                "experiment": exp_name,
                "fold": fold,
                "iou": test["iou"].mean(),
                "f1": test["f1"].mean(),
                "n_images": len(test),
                "path": str(csv_path),
            })
        except Exception as e:
            print(f"  Warning: could not read {csv_path}: {e}", file=sys.stderr)

    return results


def load_competitor_baselines() -> dict[str, dict[str, float]]:
    """Load CALT-US competitor baselines from existing results if available."""
    baselines = {}
    base = Path("data/paper_runs/baselines")
    if not base.exists():
        return baselines

    for csv_path in base.glob("*_competitors.csv"):
        try:
            df = pd.read_csv(csv_path)
            # Extract fold from filename
            name = csv_path.stem
            if "fold-1" in name:
                fold = "fold-1"
            elif "fold-2" in name:
                fold = "fold-2"
            elif "mixed" in name:
                fold = "mixed"
            else:
                continue

            calt = df[df["competitor"].str.contains("CALT", na=False)]
            if len(calt) > 0:
                baselines[fold] = {
                    "iou": calt["split_test_average"].iloc[0],
                    "f1": calt["split_test_f1_average"].iloc[0],
                }
        except Exception:
            pass

    return baselines


def load_fair_comparison() -> dict[str, dict[str, float]]:
    """Load CALT-US crop-reconstructed results from fair comparison runs."""
    fair = {}
    base = Path("data/paper_runs")
    for d in sorted(base.glob("fair_comparison_*")):
        for fold_dir in d.iterdir():
            if not fold_dir.is_dir():
                continue
            csv = fold_dir / "fullimage_eval.csv"
            if csv.exists():
                try:
                    df = pd.read_csv(csv)
                    test = df[df["split"] == "test"] if "split" in df.columns else df
                    if len(test) > 0:
                        fair[fold_dir.name] = {
                            "iou": test["iou"].mean(),
                            "f1": test["f1"].mean(),
                        }
                except Exception:
                    pass
    return fair


def main():
    if len(sys.argv) < 2:
        # If no argument, find the latest experiment battery
        base = Path("data/paper_runs")
        candidates = sorted(base.glob("experiment_battery_*"))
        if not candidates:
            # Try clean_comparison
            candidates = sorted(base.glob("clean_comparison_*"))
        if not candidates:
            print("Usage: python scripts/compile_results.py <results_dir>")
            sys.exit(1)
        results_dir = candidates[-1]
        print(f"Using latest results directory: {results_dir}")
    else:
        results_dir = Path(sys.argv[1])

    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    # Collect results
    ensemble_results = find_ensemble_results(results_dir)
    csv_results = find_fullimage_csvs(results_dir)
    all_results = ensemble_results + csv_results
    competitor_baselines = load_competitor_baselines()
    fair_baselines = load_fair_comparison()

    if not all_results and not competitor_baselines:
        print("No results found!")
        sys.exit(1)

    # Build summary
    lines = []
    lines.append(f"# Experiment Results Summary")
    lines.append(f"")
    lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Source**: `{results_dir}`")
    lines.append("")

    # Main comparison table
    lines.append("## Ensemble Configuration Comparison (test split IoU)")
    lines.append("")

    # Get all unique experiments and folds
    experiments = sorted(set(r["experiment"] for r in all_results))
    folds = sorted(set(r["fold"] for r in all_results))

    if experiments and folds:
        # Header
        header = "| Configuration |"
        sep = "|---|"
        for fold in folds:
            header += f" {fold} IoU | {fold} F1 |"
            sep += "---:|---:|"
        lines.append(header)
        lines.append(sep)

        # CALT-US baselines
        calt_row = "| **CALT-US** (whole-image) |"
        calt_fair_row = "| **CALT-US** (crop-recon) |"
        has_fair = False
        for fold in folds:
            if fold in competitor_baselines:
                b = competitor_baselines[fold]
                calt_row += f" {b['iou']:.4f} | {b['f1']:.4f} |"
            else:
                calt_row += " — | — |"
            if fold in fair_baselines:
                b = fair_baselines[fold]
                calt_fair_row += f" {b['iou']:.4f} | {b['f1']:.4f} |"
                has_fair = True
            else:
                calt_fair_row += " — | — |"
        lines.append(calt_row)
        if has_fair:
            lines.append(calt_fair_row)

        # Experiment results
        for exp in experiments:
            row = f"| **{exp}** |"
            for fold in folds:
                matches = [r for r in all_results if r["experiment"] == exp and r["fold"] == fold]
                if matches:
                    r = matches[0]
                    # Bold if beats CALT-US fair comparison
                    calt_iou = fair_baselines.get(fold, competitor_baselines.get(fold, {})).get("iou", 999)
                    iou_str = f"**{r['iou']:.4f}**" if r["iou"] > calt_iou else f"{r['iou']:.4f}"
                    row += f" {iou_str} | {r['f1']:.4f} |"
                else:
                    row += " — | — |"
            lines.append(row)

        lines.append("")

    # Cross-fold averages
    if len(folds) >= 2 and experiments:
        lines.append("## Cross-Fold Averages")
        lines.append("")
        lines.append("| Configuration | Avg IoU | Avg F1 | vs CALT-US |")
        lines.append("|---|---:|---:|---:|")

        calt_avg_iou = sum(b.get("iou", 0) for b in fair_baselines.values()) / max(len(fair_baselines), 1)
        if not fair_baselines:
            calt_avg_iou = sum(b.get("iou", 0) for b in competitor_baselines.values()) / max(len(competitor_baselines), 1)

        for exp in experiments:
            exp_results = [r for r in all_results if r["experiment"] == exp]
            if exp_results:
                avg_iou = sum(r["iou"] for r in exp_results) / len(exp_results)
                avg_f1 = sum(r["f1"] for r in exp_results) / len(exp_results)
                delta = avg_iou - calt_avg_iou
                delta_str = f"**+{delta:.4f}**" if delta > 0 else f"{delta:.4f}"
                lines.append(f"| {exp} | {avg_iou:.4f} | {avg_f1:.4f} | {delta_str} |")

        lines.append("")

    # Details
    if all_results:
        lines.append("## Detailed Results")
        lines.append("")
        lines.append("| Experiment | Fold | IoU | F1 | N |")
        lines.append("|---|---|---:|---:|---:|")
        for r in sorted(all_results, key=lambda x: (x["experiment"], x["fold"])):
            lines.append(f"| {r['experiment']} | {r['fold']} | {r['iou']:.4f} | {r['f1']:.4f} | {r['n_images']} |")
        lines.append("")

    # Write output
    summary_path = results_dir / "results_summary.md"
    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
