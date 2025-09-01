import logging
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import tifffile

from src.data_processing.utils.dataset_dataframe_creation import (
    load_dataframe_from_parquet_with_metadata,
)
from src.metrics.metrics import calculate_jaccard_scores
from src.metrics.utils import print_results

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_evaluation(
    dataset_dataframe_path: Path,
    competitor: Optional[str] = None,
    output: Optional[Path] = None,
    visualize: bool = False,
    campaign_col: str = "campaign_number",
) -> Dict[str, Any]:
    """
    Evaluates competitor segmentation results against ground truth using Jaccard index.

    Calculates per-label scores, per-image averages, per-campaign averages (based on
    all individual label scores within the campaign), and overall averages.

    Returns a dictionary with all the results.
    """
    try:
        df = load_dataframe_from_parquet_with_metadata(str(dataset_dataframe_path))
        # === DATASET DIAGNOSTICS ===
        dataset_name = dataset_dataframe_path.stem.replace("_dataset_dataframe", "")
        logging.info(f"=== EVALUATING DATASET: {dataset_name} ===")
        logging.info(f"Total rows in dataset: {len(df)}")

        # Check gt_image availability
        gt_available = df["gt_image"].notna().sum()
        gt_percentage = round(gt_available / len(df) * 100, 1) if len(df) > 0 else 0

        if gt_available == 0:
            logging.error(
                f"Dataset '{dataset_name}' has NO ground truth segmentation files!"
            )
            logging.error("   All gt_image values are null. Cannot perform evaluation.")
            raise ValueError(
                f"Dataset '{dataset_name}' has no ground truth segmentation files."
            )
        elif gt_available < len(df) * 0.05:  # Less than 5%
            logging.warning(
                f"Dataset '{dataset_name}' has very few GT files: {gt_available}/{len(df)} ({gt_percentage}%)"
            )
            logging.warning(
                "   This is normal for some datasets, but evaluation will be limited."
            )
        else:
            logging.info(
                f"Dataset '{dataset_name}' has {gt_available}/{len(df)} GT files ({gt_percentage}%)"
            )

        competitor_columns = df.attrs.get("competitor_columns", [])
        if len(competitor_columns) == 0:
            logging.warning(
                "No competitor columns specified in dataframe attributes. Attempting to infer."
            )
            # Infer competitor columns by excluding known non-competitor columns
            potential_cols = [
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
                and isinstance(df[col].iloc[0], str)
                and Path(df[col].iloc[0]).suffix in [".tif", ".tiff"]
                and not col.startswith("Unnamed")
            ]
            if potential_cols:
                competitor_columns = potential_cols
                logging.info(f"Inferred competitor columns: {competitor_columns}")
            else:
                logging.error("Could not infer competitor columns from dataframe columns.")
                return {}
                # Check competitor data availability
            logging.info(
                f"Found {len(competitor_columns)} competitors: {competitor_columns}"
            )
        for comp in competitor_columns:
            comp_available = df[comp].notna().sum()
            comp_percentage = (
                round(comp_available / len(df) * 100, 1) if len(df) > 0 else 0
            )
            if comp_available == 0:
                logging.warning(
                    f"Competitor '{comp}': NO files available (0/{len(df)})"
                )
            elif comp_available < len(df) * 0.5:  # Less than 50%
                logging.warning(
                    f"Competitor '{comp}': {comp_available}/{len(df)} files ({comp_percentage}%)"
                )
            else:
                logging.info(
                    f"Competitor '{comp}': {comp_available}/{len(df)} files ({comp_percentage}%)"
                )

    except FileNotFoundError:
        logging.error(f"Dataset dataframe file not found at: {dataset_dataframe_path}")
        return {}
    except Exception as e:
        logging.error(
            f"Error loading dataframe or reading attributes from {dataset_dataframe_path}: {e}"
        )
        return {}
    # --- Competitor Selection ---
    if competitor:
        if competitor not in competitor_columns:
            logging.error(
                f"Competitor '{competitor}' not found in available competitor columns: {competitor_columns}"
            )
            return {}
        competitor_columns = [competitor]
        logging.info(f"Evaluating specific competitor: {competitor}")
    else:
        logging.info(f"Evaluating all competitors found/inferred: {competitor_columns}")

    # --- Data Filtering ---
    initial_rows = len(df)
    filtered_df = df[df["gt_image"].notna()].copy()

    # Further filter: Check if GT path actually exists
    # Convert to Path safely, handling potential non-string types gracefully
    def check_path(p):
        if pd.isna(p):
            return False
        try:
            return Path(p).exists()
        except TypeError:
            return False

    filtered_df = filtered_df[filtered_df["gt_image"].apply(check_path)]
    rows_after_gt_exist = len(filtered_df)
    gt_missing_files = initial_rows - rows_after_gt_exist

    logging.info(
        f"Final dataset for evaluation: {rows_after_gt_exist} rows with valid GT files"
    )

    if gt_missing_files > 0:
        logging.warning(
            f"Filtered out {gt_missing_files} rows with missing or non-existent ground truth files."
        )

    logging.info(
        f"Filtered out {initial_rows - rows_after_gt_exist} rows with missing or non-existent ground truth paths."
    )

    if filtered_df.empty:
        logging.error(
            "No rows remaining after filtering for valid ground truth images."
        )
        logging.error("   Cannot perform evaluation without ground truth data.")
        return {}

    # --- Campaign Column Handling ---
    if campaign_col not in filtered_df.columns:
        logging.warning(
            f"Campaign column '{campaign_col}' not found. Attempting to extract from 'composite_key'."
        )
        try:
            filtered_df[campaign_col] = filtered_df["composite_key"].apply(
                lambda x: str(x).split("/")[0]
                if isinstance(x, str) and "/" in x
                else (str(x) if pd.notna(x) else "unknown_campaign")
            )
            logging.info(
                f"Successfully extracted campaign names into '{campaign_col}' column."
            )
        except Exception as e:
            logging.error(
                f"Failed to extract campaign names from 'composite_key': {e}. Cannot proceed without campaign information."
            )
            return {}

    # --- Get Unique Campaigns ---
    campaigns = sorted(filtered_df[campaign_col].unique())
    if not campaigns:
        logging.error(f"No campaigns found using column '{campaign_col}'.")
        return {}
    logging.info(f"Found {len(campaigns)} campaigns: {campaigns}")

    all_results = {
        comp: {camp: {} for camp in campaigns} for comp in competitor_columns
    }
    per_image_averages = {
        comp: {camp: {} for camp in campaigns} for comp in competitor_columns
    }
    per_campaign_averages = {comp: {} for comp in competitor_columns}
    overall_averages = {}
    all_labels = set()

    # --- Processing Loop ---
    for comp in competitor_columns:
        logging.info(f"--- Processing Competitor: {comp} ---")
        competitor_all_label_scores = []

        if comp not in filtered_df.columns:
            logging.warning(
                f"Competitor column '{comp}' not found in the filtered dataframe. Skipping."
            )
            for campaign in campaigns:
                per_campaign_averages[comp][campaign] = float("nan")
            overall_averages[comp] = float("nan")
            continue

        for campaign in campaigns:
            logging.info(f"  Processing Campaign: {campaign}")
            campaign_df = filtered_df[filtered_df[campaign_col] == campaign]
            campaign_all_label_scores = []
            image_count = 0
            processed_count = 0
            skipped_count = 0
            for index, row in campaign_df.iterrows():
                image_count += 1
                gt_path_str = row.get("gt_image")
                seg_path_str = row.get(comp)
                composite_key = row.get("composite_key", f"Row_{index}")

                if not isinstance(gt_path_str, str) or not isinstance(
                    seg_path_str, str
                ):
                    logging.warning(
                        f"    {composite_key}: Skipping due to invalid GT ('{gt_path_str}') or SEG ('{seg_path_str}') path type."
                    )
                    all_results[comp][campaign][composite_key] = {}
                    per_image_averages[comp][campaign][composite_key] = float("nan")
                    skipped_count += 1
                    continue

                gt_path = Path(gt_path_str)
                seg_path = Path(seg_path_str)

                if not gt_path.exists():
                    logging.warning(
                        f"    {composite_key}: Skipping because GT file disappeared: {gt_path}"
                    )
                    all_results[comp][campaign][composite_key] = {}
                    per_image_averages[comp][campaign][composite_key] = float("nan")
                    skipped_count += 1
                    continue
                if not seg_path.exists():
                    logging.warning(
                        f"    {composite_key}: Skipping because competitor '{comp}' file not found: {seg_path}"
                    )
                    all_results[comp][campaign][composite_key] = {}
                    per_image_averages[comp][campaign][composite_key] = float("nan")
                    skipped_count += 1
                    continue

                try:
                    gt_img = tifffile.imread(gt_path)
                    seg_img = tifffile.imread(seg_path)

                    jaccard_scores = calculate_jaccard_scores(gt_img, seg_img)
                    all_results[comp][campaign][composite_key] = jaccard_scores
                    all_labels.update(jaccard_scores.keys())

                    if jaccard_scores:
                        image_avg = sum(jaccard_scores.values()) / len(jaccard_scores)
                        per_image_averages[comp][campaign][composite_key] = image_avg
                        campaign_all_label_scores.extend(list(jaccard_scores.values()))
                    else:
                        per_image_averages[comp][campaign][composite_key] = 0.0

                    processed_count += 1

                except Exception as e:
                    logging.error(
                        f"    {composite_key}: Error processing images ({gt_path}, {seg_path}): {e}. Skipping image."
                    )
                    all_results[comp][campaign][composite_key] = {}
                    per_image_averages[comp][campaign][composite_key] = float("nan")
                    skipped_count += 1

            logging.info(
                f"    Campaign '{campaign}': Processed {processed_count}, Skipped {skipped_count} (Total: {image_count}) images."
            )
            # --- Campaign Average Calculation (using raw scores) ---
            if campaign_all_label_scores:
                campaign_avg = sum(campaign_all_label_scores) / len(
                    campaign_all_label_scores
                )
                per_campaign_averages[comp][campaign] = campaign_avg
                competitor_all_label_scores.extend(campaign_all_label_scores)
                logging.info(
                    f"    Campaign '{campaign}' avg Jaccard for '{comp}': {campaign_avg:.4f} (from {len(campaign_all_label_scores)} label scores)"
                )
            else:
                per_campaign_averages[comp][campaign] = float("nan")
                logging.warning(
                    f"    No valid Jaccard scores found for '{comp}' in campaign '{campaign}'. Average set to NaN."
                )

        if competitor_all_label_scores:
            overall_avg = sum(competitor_all_label_scores) / len(
                competitor_all_label_scores
            )
            overall_averages[comp] = overall_avg
            logging.info(
                f"  Overall avg Jaccard for '{comp}': {overall_avg:.4f} (from {len(competitor_all_label_scores)} label scores across all campaigns)"
            )
        else:
            overall_averages[comp] = float("nan")
            logging.warning(
                f"  No valid Jaccard scores found for '{comp}' across all campaigns. Overall average set to NaN."
            )

    logging.info("--- Evaluation Summary ---")

    # Calculate and display evaluation statistics
    valid_scores_count = 0
    nan_scores_count = 0
    total_evaluations = 0

    for comp in competitor_columns:
        if comp in overall_averages and not pd.isna(overall_averages[comp]):
            valid_scores_count += 1
        else:
            nan_scores_count += 1
        total_evaluations += 1

    logging.info(f"Dataset '{dataset_name}' evaluation completed:")
    logging.info(
        f"  • Competitors with valid scores: {valid_scores_count}/{total_evaluations}"
    )
    logging.info(
        f"  • Competitors with no scores (NaN): {nan_scores_count}/{total_evaluations}"
    )

    if valid_scores_count == 0:
        logging.warning("No valid Jaccard scores calculated for any competitor!")
        logging.warning("   This might indicate:")
        logging.warning("   - Missing competitor segmentation files")
        logging.warning(
            "   - Mismatched label numbers between GT and competitor segmentations"
        )
        logging.warning("   - File format issues")
    elif nan_scores_count > 0:
        logging.warning(f"{nan_scores_count} competitors have no valid scores")
    else:
        logging.info("All competitors have valid evaluation scores")

    print_results(
        all_results,
        per_image_averages,
        per_campaign_averages,
        overall_averages,
        campaigns,
    )

    # --- Optional CSV Output ---
    if output:
        logging.info(f"Preparing detailed results for CSV output to {output}")
        output_data = []
        for comp, campaigns_data in all_results.items():
            for campaign, images_data in campaigns_data.items():
                for image_key, labels_data in images_data.items():
                    img_avg = (
                        per_image_averages.get(comp, {})
                        .get(campaign, {})
                        .get(image_key, float("nan"))
                    )
                    camp_avg = per_campaign_averages.get(comp, {}).get(
                        campaign, float("nan")
                    )
                    overall_avg = overall_averages.get(comp, float("nan"))

                    if labels_data:
                        for label, score in labels_data.items():
                            output_data.append(
                                {
                                    "competitor": comp,
                                    "campaign": campaign,
                                    "image_key": image_key,
                                    "label": label,
                                    "jaccard_score": score,
                                    "image_average": img_avg,
                                    "campaign_average": camp_avg,
                                    "overall_competitor_average": overall_avg,
                                }
                            )
                    else:
                        output_data.append(
                            {
                                "competitor": comp,
                                "campaign": campaign,
                                "image_key": image_key,
                                "label": None,
                                "jaccard_score": float("nan"),
                                "image_average": img_avg,
                                "campaign_average": camp_avg,
                                "overall_competitor_average": overall_avg,
                            }
                        )

        if output_data:
            results_df = pd.DataFrame(output_data)
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(output, index=False, float_format="%.6f")
                logging.info(f"Successfully saved results to {output}")
            except Exception as e:
                logging.error(f"Failed to save results to CSV '{output}': {e}")
        else:
            logging.warning("No results data generated to save to CSV.")

    if visualize:
        logging.warning(
            "Visualization flag is set, but visualization logic is not implemented."
        )

    logging.info("Evaluation script finished.")

    return {
        "all_results": all_results,
        "per_image_averages": per_image_averages,
        "per_campaign_averages": per_campaign_averages,
        "overall_averages": overall_averages,
    }
