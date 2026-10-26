#!/usr/bin/env python3
"""Build context tables for QA/ranker-pruned fusion experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from silver_truth.evaluation.ranker_pruned_fusion import (
    write_ranker_pruned_fusion_context_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine ranker-pruned fusion comparison rows with fusion-only, "
            "direct soft-hybrid, best competitor, and silver-truth context."
        )
    )
    parser.add_argument(
        "--comparison",
        action="append",
        type=Path,
        default=[],
        help="ranker_pruned_fusion_comparison.csv path. Can be repeated.",
    )
    parser.add_argument(
        "--comparison-glob",
        action="append",
        default=[],
        help="Glob pattern for comparison CSVs. Can be repeated.",
    )
    parser.add_argument(
        "--paper-summary",
        type=Path,
        default=Path("hpc_results/combined/reports/paper_result_summary_all_available.csv"),
    )
    parser.add_argument(
        "--soft-hybrid-selection",
        type=Path,
        default=Path(
            "hpc_results/soft_hybrid_probe/"
            "reporting_mode_selection_full_image_vs_local_support_modes_combined/"
            "validation_selected_reporting_mode.csv"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    comparison_paths = list(args.comparison)
    for pattern in args.comparison_glob:
        comparison_paths.extend(Path().glob(pattern))
    comparison_paths = sorted(set(comparison_paths))
    if not comparison_paths:
        raise SystemExit("No comparison CSVs matched.")

    context = write_ranker_pruned_fusion_context_table(
        comparison_paths=comparison_paths,
        paper_summary_path=args.paper_summary,
        soft_hybrid_selection_path=args.soft_hybrid_selection,
        output_path=args.output,
    )
    print(f"Wrote context table: {args.output}")
    print(context[context["row_type"].eq("mean")].to_string(index=False))


if __name__ == "__main__":
    main()
