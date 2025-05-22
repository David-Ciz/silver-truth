from datetime import datetime
import logging # Added for potential warnings

from tabulate import tabulate
import pandas as pd # Added for checking NaN


def print_results(all_results, per_image_averages, per_campaign_averages, overall_averages, campaigns):
    """Format and print the results, including specific details for DREX-US campaign 02."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nResults generated on: {timestamp}")

    # --- Standard Output (Keep as is) ---

    # Print overall competitor averages
    print("\n=== OVERALL COMPETITOR AVERAGES ===")
    # Use .items() for potentially safer iteration if overall_averages could be modified elsewhere
    overall_table_data = [(comp, f"{avg:.4f}" if pd.notna(avg) else "NaN") for comp, avg in overall_averages.items()]
    if overall_table_data:
        print(tabulate(overall_table_data, headers=["Competitor", "Average Jaccard"], tablefmt="grid"))
    else:
        print("No overall average data available.")

    # Print per-campaign averages
    print("\n=== PER-CAMPAIGN AVERAGES ===")
    competitor_list = list(overall_averages.keys()) # Get fixed list of competitors
    if competitor_list:
        campaign_header = ["Campaign"] + competitor_list
        campaign_table_data = []
        for campaign in sorted(campaigns):
            row = [campaign]
            for comp in competitor_list:
                # Safely get campaign average, default to NaN if missing
                avg = per_campaign_averages.get(comp, {}).get(campaign, float('nan'))
                row.append(f"{avg:.4f}" if pd.notna(avg) else "NaN")
            campaign_table_data.append(row)

        if campaign_table_data:
            print(tabulate(campaign_table_data, headers=campaign_header, tablefmt="grid"))
        else:
            print("No per-campaign average data available.")
    else:
        print("No competitors found for per-campaign averages.")


    # Print per-image averages by campaign
    for campaign in sorted(campaigns):
        print(f"\n=== CAMPAIGN: {campaign} (Per-Image Average Jaccard) ===")

        if not competitor_list:
             print("No competitors found.")
             continue

        image_header = ["Image Key"] + competitor_list
        image_table_data = []

        # Collect all image keys across all competitors for this campaign
        all_image_keys_for_campaign = set()
        for comp in competitor_list:
            # Safely access nested dictionary
            campaign_data = per_image_averages.get(comp, {}).get(campaign, {})
            all_image_keys_for_campaign.update(campaign_data.keys())

        if not all_image_keys_for_campaign:
            print("No images processed or found for this campaign.")
            continue

        for image_key in sorted(list(all_image_keys_for_campaign)): # Sort keys for consistent order
            row = [image_key]
            for comp in competitor_list:
                 # Safely get image average, default to NaN if missing
                avg = per_image_averages.get(comp, {}).get(campaign, {}).get(image_key, float('nan'))
                row.append(f"{avg:.4f}" if pd.notna(avg) else "NaN")
            image_table_data.append(row)

        if image_table_data:
            print(tabulate(image_table_data, headers=image_header, tablefmt="grid"))
        # No explicit "else" needed here, as the loop or key collection handles empty cases


    # # --- Specific Debug Output for DREX-US Campaign 02 ---
    # print("\n=== DETAILED SCORES FOR DREX-US / CAMPAIGN 02 ===")
    # target_comp = 'DREX-US'
    # target_campaign = '02' # Assuming campaign name is '02' as a string
    #
    # if target_comp in all_results and target_campaign in all_results.get(target_comp, {}):
    #     campaign_data = all_results[target_comp][target_campaign]
    #     if not campaign_data:
    #         print(f"No images found or processed for competitor '{target_comp}' in campaign '{target_campaign}'.")
    #     else:
    #         # Sort images for consistent output
    #         sorted_image_keys = sorted(campaign_data.keys())
    #         for image_key in sorted_image_keys:
    #             scores = campaign_data[image_key] # This is {label: score}
    #             print(f"\n  Image: {image_key}")
    #             if scores:
    #                 label_table = []
    #                 # Sort labels for consistent output
    #                 for label, score in sorted(scores.items()):
    #                      # Handle potential NaN scores in output
    #                     score_str = f"{score:.6f}" if pd.notna(score) else "NaN"
    #                     label_table.append([label, score_str])
    #                 print(tabulate(label_table, headers=["Label", "Jaccard"], tablefmt="simple", stralign="left", numalign="right"))
    #             else:
    #                 print("    No scores calculated for this image (e.g., file missing, no overlap, error).")
    # else:
    #     # More specific message about why data isn't available
    #     if target_comp not in all_results:
    #         print(f"Competitor '{target_comp}' not found in the results.")
    #     elif target_campaign not in all_results.get(target_comp, {}):
    #          print(f"Campaign '{target_campaign}' not found for competitor '{target_comp}'.")
    #     else:
    #          # Should not happen if previous conditions are met, but as a fallback
    #          print(f"Could not retrieve data for Competitor '{target_comp}', Campaign '{target_campaign}'.")
    #
    # # --- End of Specific Debug Output ---
    #
    # # Commented out the generic per-label score printing to avoid excessive output
    # # unless specifically requested. The section above provides the targeted detail.
    # #
    # # # Print per-label scores for each image by campaign
    # # print("\n=== PER-LABEL SCORES BY CAMPAIGN ===")
    # # for campaign in sorted(campaigns):
    # #     print(f"\n-- CAMPAIGN: {campaign} --")
    # #     # ... (rest of the original commented code) ...