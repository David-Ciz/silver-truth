import pandas as pd
from pathlib import Path

parquet_path = "/Users/davidciz/Work/silver-truth/BF-C2DL-HSC_dataset_dataframe.parquet"
fused_images_dir = "/Users/davidciz/Work/silver-truth/data/fused/BF-C2DL-HSC_01"
output_parquet_path = (
    "/Users/davidciz/Work/silver-truth/BF-C2DL-HSC_dataset_dataframe_with_fused.parquet"
)

df = pd.read_parquet(parquet_path)

# Create a new column for fused images, initialized to None
df["fused_images"] = None
# Add fusion_model column and set to the selected model (update as needed)
selected_fusion_model = "majority_flat"  # Change this to the actual model used
df["fusion_model"] = None

# Get a set of existing fused image filenames for efficient lookup
existing_fused_images = {p.name for p in Path(fused_images_dir).glob("fused_*.tif")}

# Filter for Campaign 01 rows
df_campaign_01 = df[df["campaign_number"] == "01"].copy()


# Function to extract the numeric part from composite_key
def extract_numeric_id(composite_key):
    return composite_key[3:-4]


# Map fused images based on composite_key
fused_image_mapping = {}
for _, row in df_campaign_01.iterrows():
    numeric_id = extract_numeric_id(row["composite_key"])
    if numeric_id is not None:
        fused_filename = f"fused_{numeric_id}.tif"  # Use 03d for consistent formatting

        if fused_filename in existing_fused_images:
            fused_image_mapping[row["composite_key"]] = str(
                Path(fused_images_dir) / fused_filename
            )
            # Set fusion_model for this row
            df.loc[df["composite_key"] == row["composite_key"], "fusion_model"] = selected_fusion_model

# Apply the mapping to the DataFrame
df.loc[df["campaign_number"] == "01", "fused_images"] = df_campaign_01[
    "composite_key"
].map(fused_image_mapping)

# Save the modified DataFrame
df.to_parquet(output_parquet_path, index=False)

print(f"Modified dataframe saved to: {output_parquet_path}")
print(df[df["campaign_number"] == "01"][["composite_key", "fused_images"]].head())
