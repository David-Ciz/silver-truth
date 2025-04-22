from datetime import datetime

from tabulate import tabulate


def print_results(all_results, per_image_averages, per_campaign_averages, overall_averages, campaigns):
    """Format and print the results in a clear, tabular format."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nResults generated on: {timestamp}")

    # Print overall competitor averages
    print("\n=== OVERALL COMPETITOR AVERAGES ===")
    overall_table = []
    for comp, avg in overall_averages.items():
        overall_table.append([comp, f"{avg:.4f}"])
    print(tabulate(overall_table, headers=["Competitor", "Average Jaccard"], tablefmt="grid"))

    # Print per-campaign averages
    print("\n=== PER-CAMPAIGN AVERAGES ===")
    campaign_table = [["Campaign"] + list(overall_averages.keys())]
    for campaign in sorted(campaigns):
        row = [campaign]
        for comp in overall_averages.keys():
            row.append(f"{per_campaign_averages[comp].get(campaign, 0):.4f}")
        campaign_table.append(row)
    print(tabulate(campaign_table[1:], headers=campaign_table[0], tablefmt="grid"))

    # Print per-image averages by campaign
    for campaign in sorted(campaigns):
        print(f"\n=== CAMPAIGN: {campaign} ===")
        image_table = [["Image Key"] + list(overall_averages.keys())]

        # Collect all image keys across all competitors for this campaign
        all_image_keys = set()
        for comp in overall_averages.keys():
            if campaign in per_image_averages[comp]:
                all_image_keys.update(per_image_averages[comp][campaign].keys())

        for image_key in sorted(all_image_keys):
            row = [image_key]
            for comp in overall_averages.keys():
                if campaign in per_image_averages[comp]:
                    row.append(f"{per_image_averages[comp][campaign].get(image_key, 0):.4f}")
                else:
                    row.append("N/A")
            image_table.append(row)

        print(tabulate(image_table[1:], headers=image_table[0], tablefmt="grid"))
    #
    # # Print per-label scores for each image by campaign
    # print("\n=== PER-LABEL SCORES BY CAMPAIGN ===")
    # for campaign in sorted(campaigns):
    #     print(f"\n-- CAMPAIGN: {campaign} --")
    #     for comp in all_results:
    #         if campaign in all_results[comp]:
    #             print(f"\n  Competitor: {comp}")
    #             for image_key, scores in all_results[comp][campaign].items():
    #                 if scores:
    #                     print(f"    Image: {image_key}")
    #                     label_table = []
    #                     for label, score in scores.items():
    #                         label_table.append([label, f"{score:.4f}"])
    #                     print(tabulate(label_table, headers=["Label", "Jaccard"], tablefmt="simple", indent=6))
