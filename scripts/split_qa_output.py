import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Config
QA_PARQUET = "qa_BF-C2DL-MuSC_dataset.parquet"  # Change as needed
OUT_DIR = "qa_output_split"  # Output directory for splits
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Load QA dataset
qa_df = pd.read_parquet(QA_PARQUET)

# Shuffle and split
train_df, temp_df = train_test_split(qa_df, test_size=(1-TRAIN_RATIO), random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=42)

# Create output directory if needed
import os
os.makedirs(OUT_DIR, exist_ok=True)

# Save splits
train_df.to_parquet(f"{OUT_DIR}/train.parquet", index=False)
val_df.to_parquet(f"{OUT_DIR}/val.parquet", index=False)
test_df.to_parquet(f"{OUT_DIR}/test.parquet", index=False)

print(f"Train: {len(train_df)}\nVal: {len(val_df)}\nTest: {len(test_df)}")
