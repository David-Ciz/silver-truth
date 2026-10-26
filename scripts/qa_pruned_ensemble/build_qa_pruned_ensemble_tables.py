#!/usr/bin/env python3
"""Build strict QA-pruned ENSEMBLE tables plus separate SIMPLE context."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_BEST_BY_DATASET,
    DEFAULT_OUTPUT_ROOT,
    filter_strict_ensemble_rows,
    load_simple_context,
    summarize_dataset,
    summarize_overall,
    write_markdown_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-glob",
        action="append",
        default=None,
        help=(
            "Glob for weighted_object_vote_comparison.csv files. Defaults to the "
            "hard- and soft-gated QA-pruned ENSEMBLE output roots."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "tables",
    )
    parser.add_argument("--simple-context", type=Path, default=DEFAULT_BEST_BY_DATASET)
    args = parser.parse_args()

    patterns = args.comparison_glob or [
        str(DEFAULT_OUTPUT_ROOT / "hard_gated_object_vote/**/weighted_object_vote_comparison.csv"),
        str(DEFAULT_OUTPUT_ROOT / "soft_gated_object_vote/**/weighted_object_vote_comparison.csv"),
        str(DEFAULT_OUTPUT_ROOT / "soft_gate_sweep/**/weighted_object_vote_comparison.csv"),
        str(DEFAULT_OUTPUT_ROOT / "learned_candidate_set_weight/**/weighted_object_vote_comparison.csv"),
    ]
    comparison_paths: list[Path] = []
    for pattern in patterns:
        comparison_paths.extend(Path().glob(pattern))
    comparison_paths = sorted(set(comparison_paths))
    if not comparison_paths:
        raise SystemExit("No QA-pruned ENSEMBLE comparison rows matched.")

    rows = pd.concat([pd.read_csv(path) for path in comparison_paths], ignore_index=True)
    strict_rows = filter_strict_ensemble_rows(rows)
    if strict_rows.empty:
        raise SystemExit("Matched rows did not contain strict QA/ENSEMBLE row families.")

    dataset_summary = summarize_dataset(strict_rows)
    overall_summary = summarize_overall(dataset_summary)
    context = load_simple_context(args.simple_context)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output_dir / "qa_pruned_ensemble_all_rows.csv", index=False)
    strict_rows.to_csv(args.output_dir / "qa_pruned_ensemble_strict_rows.csv", index=False)
    dataset_summary.to_csv(
        args.output_dir / "qa_pruned_ensemble_dataset_summary.csv",
        index=False,
    )
    overall_summary.to_csv(
        args.output_dir / "qa_pruned_ensemble_overall_summary.csv",
        index=False,
    )
    if not context.empty:
        context.to_csv(args.output_dir / "qa_pruned_simple_context_rows.csv", index=False)
    write_markdown_report(
        output_path=args.output_dir / "qa_pruned_ensemble_summary.md",
        title="QA-Pruned ENSEMBLE",
        dataset_summary=dataset_summary,
        overall_summary=overall_summary,
        context=context,
    )
    print(f"Wrote QA-pruned ENSEMBLE tables to {args.output_dir}")
    print(overall_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
