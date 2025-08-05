import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.metrics.evaluation_logic import run_evaluation


def evaluate_competitor_logic(
    dataset_dataframe_path: Path,
    competitor: Optional[str] = None,
    output: Optional[Path] = None,
    visualize: bool = False,
    campaign_col: str = "campaign_number",
    detailed: bool = False,
):
    """
    Evaluates competitor segmentation results against ground truth using Jaccard index.

    This script is a wrapper around the core evaluation logic in `run_evaluation`.
    With --detailed flag, also creates detailed per-cell evaluation results.
    """
    # Run standard evaluation
    run_evaluation(
        dataset_dataframe_path=dataset_dataframe_path,
        competitor=competitor,
        output=output,
        visualize=visualize,
        campaign_col=campaign_col,
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
