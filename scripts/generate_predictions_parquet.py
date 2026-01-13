import pandas as pd
import torch
import numpy as np
import tifffile
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn


# Load model architecture (same as in resnet50.py)
class JaccardResNet(nn.Module):
    def __init__(self, dropout_rate=0.3, use_sigmoid=False):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace the final fc layer with dropout + linear
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove original fc
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, 1)  # Regression output
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x


# Load the trained model (sigmoid version)
device = torch.device("cpu")
checkpoint = torch.load("models/resnet50_jaccard_sigmoid.pt", map_location=device)
metadata = checkpoint.get("metadata", {})
dropout_rate = metadata.get("dropout_rate", 0.3)
use_sigmoid = metadata.get("use_sigmoid", True)
model = JaccardResNet(dropout_rate=dropout_rate, use_sigmoid=use_sigmoid)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded (use_sigmoid={use_sigmoid})")

# Load original dataframe
df = pd.read_parquet("dataframes/BF-C2DL-HSC_QA_crops_64_split70-15-15_seed42.parquet")
print(f"Loaded {len(df)} samples")


# Normalization function (same as in resnet50.py)
def tensor_normalize(tensor, mean, std):
    """Normalize tensor with given mean and std."""
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def load_and_preprocess(image_path):
    """Load and preprocess a single image."""
    img_np = tifffile.imread(image_path)

    if img_np.ndim == 2:
        img_np = np.stack([img_np, img_np], axis=0)
    elif img_np.ndim == 3:
        if img_np.shape[0] == 2:
            pass  # already (2, H, W)
        elif img_np.shape[-1] == 2:
            img_np = np.transpose(img_np, (2, 0, 1))
        else:
            raise ValueError(f"Image does not have 2 channels. Shape: {img_np.shape}")

    # Normalize to [0, 1] then to [-1, 1]
    img_np = img_np.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np)
    tensor = tensor_normalize(tensor, mean=[0.5, 0.5], std=[0.5, 0.5])
    return tensor


# Generate predictions for all samples
predictions = []
batch_size = 32

print("Generating predictions...")
with torch.no_grad():
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        images = []
        for _, row in batch_df.iterrows():
            img = load_and_preprocess(row["stacked_path"])
            images.append(img)

        batch = torch.stack(images)
        outputs = model(batch).squeeze().numpy()
        if outputs.ndim == 0:
            outputs = [outputs.item()]
        predictions.extend(outputs)

        if (i // batch_size) % 20 == 0:
            print(f"  Processed {i+len(batch_df)}/{len(df)}")

print(f"Generated {len(predictions)} predictions")

# Add predictions to dataframe
df["predicted_jaccard"] = predictions

# Save new parquet
output_path = "dataframes/BF-C2DL-HSC_QA_crops_64_split70-15-15_seed42_with_predictions_sigmoid_2026-01-07.parquet"
df.to_parquet(output_path)
print(f"Saved to {output_path}")

# Show summary
print()
print("=== Prediction Summary ===")
print(f'Min: {df["predicted_jaccard"].min():.4f}')
print(f'Max: {df["predicted_jaccard"].max():.4f}')
print(f'Mean: {df["predicted_jaccard"].mean():.4f}')
print()
print("Columns:", list(df.columns))
