from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from silver_truth.ensemble.reconstruction import reconstruct_labeled_full_images_from_paths


DEFAULT_PRIORITY_COLUMNS = [
    "soft_hybrid_score",
    "candidate_ranker_score",
    "soft_fusion_support",
    "soft_qa",
    "fusion_support",
    "qa_score",
]


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def reconstruct_selected_soft_hybrid_candidates(
    *,
    candidate_parquet: Path,
    selected_candidates_path: Path,
    output_dir: Path,
    eval_csv_path: Path | None = None,
    crop_path_column: str = "stacked_path",
    threshold: float = 0.5,
    priority_columns: list[str] | None = None,
) -> pd.DataFrame:
    candidates = pd.read_parquet(candidate_parquet)
    selected = load_table(selected_candidates_path)
    selected_rows = selected_candidates_for_reconstruction(candidates, selected)

    fullimage_dir = output_dir / "fullimage"
    eval_df = reconstruct_labeled_full_images_from_paths(
        selected_rows,
        fused_path_column=crop_path_column,
        output_dir=fullimage_dir,
        threshold=threshold,
        priority_columns=priority_columns or DEFAULT_PRIORITY_COLUMNS,
    )

    csv_path = eval_csv_path or output_dir / "fullimage_eval.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(csv_path, index=False)
    return eval_df


def selected_candidates_for_reconstruction(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    if "cell_id" not in candidates.columns:
        raise ValueError("Candidate parquet must include a 'cell_id' column.")
    if "candidate_id" not in selected.columns:
        raise ValueError("Selected candidates must include a 'candidate_id' column.")

    selected_features = selected.drop_duplicates(subset=["candidate_id"]).copy()
    feature_columns = [
        column
        for column in selected_features.columns
        if column not in candidates.columns or column == "candidate_id"
    ]
    selected_features = selected_features[feature_columns].rename(
        columns={"candidate_id": "cell_id"}
    )
    merged = candidates.merge(selected_features, on="cell_id", how="inner")

    if merged.empty:
        raise ValueError("No selected candidate IDs matched the candidate parquet.")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct and evaluate full images from a selected soft-hybrid "
            "candidate set."
        )
    )
    parser.add_argument("--candidate-parquet", required=True, type=Path)
    parser.add_argument("--selected-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--crop-path-column", default="stacked_path")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    eval_df = reconstruct_selected_soft_hybrid_candidates(
        candidate_parquet=args.candidate_parquet,
        selected_candidates_path=args.selected_candidates,
        output_dir=args.output_dir,
        eval_csv_path=args.eval_csv,
        crop_path_column=args.crop_path_column,
        threshold=args.threshold,
    )
    print(
        f"Wrote {len(eval_df)} full-image evaluation rows to "
        f"{args.eval_csv or args.output_dir / 'fullimage_eval.csv'}"
    )


if __name__ == "__main__":
    main()
