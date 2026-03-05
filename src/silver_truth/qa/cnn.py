import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import (
    resnet18,
    resnet50,
    resnet101,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    efficientnet_b1,
    efficientnet_b4,
    efficientnet_b7,
    EfficientNet_B1_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B7_Weights,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tifffile
import numpy as np
from pathlib import Path
import random
import mlflow
from typing import Optional, Sequence

from silver_truth.metrics.qa_model_evaluation import (
    calculate_regression_metrics,
    calculate_tolerance_accuracy,
)
from silver_truth.experiment_tracking import DEFAULT_MLFLOW_TRACKING_URI


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JaccardDataset(Dataset):
    DEFAULT_TARGET_CANDIDATES = (
        "jaccard_score",
        "qa_jaccard",
        "Jaccard index",
    )
    DEFAULT_INPUT_CHANNELS = (0, 1)

    def __init__(
        self,
        parquet_file,
        data_root=None,
        transform=None,
        augment=False,
        target_column=None,
        input_channels: Optional[Sequence[int] | str] = None,
    ):
        self.data = pd.read_parquet(parquet_file)
        # If data_root is provided, convert to Path, else None
        self.data_root = Path(data_root) if data_root else None
        self.transform = transform
        self.augment = augment
        self.target_column = self._resolve_target_column(target_column)
        self.input_channels = self._resolve_input_channels(input_channels)

    def _resolve_target_column(self, target_column):
        if target_column:
            if target_column not in self.data.columns:
                raise ValueError(
                    f"Requested target column '{target_column}' not found in parquet. "
                    f"Available columns: {sorted(self.data.columns.tolist())}"
                )
            return target_column

        for candidate in self.DEFAULT_TARGET_CANDIDATES:
            if candidate in self.data.columns:
                return candidate

        raise ValueError(
            "Could not infer target column for training/evaluation. "
            f"Tried {list(self.DEFAULT_TARGET_CANDIDATES)}. "
            f"Available columns: {sorted(self.data.columns.tolist())}. "
            "Pass a column explicitly via --target-column."
        )

    def _resolve_input_channels(
        self, input_channels: Optional[Sequence[int] | str]
    ) -> tuple[int, int]:
        if input_channels is None:
            return self.DEFAULT_INPUT_CHANNELS

        if isinstance(input_channels, str):
            parts = [part.strip() for part in input_channels.split(",") if part.strip()]
            parsed = tuple(int(part) for part in parts)
        else:
            parsed = tuple(int(value) for value in input_channels)

        if len(parsed) != 2:
            raise ValueError(
                f"input_channels must contain exactly 2 channels, got {parsed}."
            )
        if len(set(parsed)) != 2:
            raise ValueError(
                f"input_channels must contain two distinct channel indices, got {parsed}."
            )
        if any(channel < 0 for channel in parsed):
            raise ValueError(
                f"input_channels must be non-negative indices, got {parsed}."
            )
        return parsed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # This is the relative path from the parquet, e.g., "data/qa_data/BF-C2DL-HSC_crops_64/xxx.tif"
        rel_path = row["stacked_path"]

        # If data_root is provided (e.g., on HPC scratch), prepend it to the relative path
        if self.data_root:
            image_path = self.data_root / rel_path
        else:
            image_path = rel_path

        jaccard = row[self.target_column]

        # Read the stacked TIFF with tifffile (PIL doesn't handle multi-frame TIFFs correctly).
        # QA crops can be 2-channel (raw, seg) or 4-channel (raw, seg, gt, tra).
        img_np = tifffile.imread(image_path)

        if img_np.ndim == 2:
            # Single-channel fallback (mostly for synthetic testing).
            img_np = np.stack([img_np, img_np], axis=0)
        elif img_np.ndim == 3:
            # Accept both CHW and HWC layouts.
            if img_np.shape[0] <= 8:
                pass  # already CHW
            elif img_np.shape[-1] <= 8:
                img_np = np.transpose(img_np, (2, 0, 1))
            else:
                raise ValueError(
                    f"Could not infer channel axis for image at {image_path}. Shape: {img_np.shape}"
                )
        else:
            raise ValueError(
                f"Image at {image_path} has unsupported shape {img_np.shape}."
            )

        available_channels = img_np.shape[0]
        max_requested_channel = max(self.input_channels)
        if max_requested_channel >= available_channels:
            raise ValueError(
                f"Image at {image_path} has {available_channels} channels, "
                f"but requested input_channels={self.input_channels}."
            )

        img_np = img_np[list(self.input_channels), :, :]

        # Apply augmentation if enabled (for training)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img_np = np.flip(img_np, axis=2).copy()  # flip along W axis
            # Random vertical flip
            if random.random() > 0.5:
                img_np = np.flip(img_np, axis=1).copy()  # flip along H axis
            # Random 90-degree rotations
            k = random.randint(0, 3)  # 0, 90, 180, or 270 degrees
            if k > 0:
                img_np = np.rot90(img_np, k, axes=(1, 2)).copy()

        # Normalize to [0, 1] range
        # Images are uint8 (0-255)
        img_np = img_np.astype(np.float32) / 255.0

        image = torch.from_numpy(img_np)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(jaccard, dtype=torch.float32)


# Define the model
class Jaccard(nn.Module):
    def __init__(self, dropout_rate=0.3, model_type="resnet50"):
        super(Jaccard, self).__init__()

        if model_type == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_type == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_type == "resnet101":
            self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_type == "efficientnet_b1":
            self.model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
            self.model.features[0][0] = nn.Conv2d(
                2, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
        elif model_type == "efficientnet_b4":
            self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
            self.model.features[0][0] = nn.Conv2d(
                2, 48, kernel_size=3, stride=2, padding=1, bias=False
            )
        elif model_type == "efficientnet_b7":
            self.model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
            self.model.features[0][0] = nn.Conv2d(
                2, 64, kernel_size=3, stride=2, padding=1, bias=False
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Adjust first conv layer for 2 channels
        if "resnet" in model_type:
            self.model.conv1 = nn.Conv2d(
                2, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, 1)  # Regression output

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def tensor_normalize(tensor, mean, std):
    """Normalize tensor with given mean and std."""
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


class NormalizeTransform:
    """Transform to normalize tensor to [-1, 1] from [0, 1]."""

    def __call__(self, x):
        return tensor_normalize(x, mean=[0.5, 0.5], std=[0.5, 0.5])


def get_transform():
    """
    Get the default transform for the dataset.

    Assumes input is already normalized to [0, 1] range.
    Maps [0, 1] -> [-1, 1] which is standard for pretrained models.
    """
    return NormalizeTransform()


def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=1.0):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(dim=1), targets)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion, device):
    """Evaluate the model on validation set."""
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), targets)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate_model_with_ids(model, dataset, indices, batch_size, device):
    """
    Evaluate model and return predictions, actuals, and cell_ids.

    This function creates a non-shuffled dataloader to ensure correct
    alignment between predictions and cell_ids.

    Args:
        model: The trained model
        dataset: The full JaccardDataset
        indices: List of indices for this split
        batch_size: Batch size for evaluation
        device: torch device

    Returns:
        Tuple of (predictions, actuals, cell_ids)
    """
    model.eval()
    predictions = []
    actuals = []
    cell_ids = []

    eval_subset = Subset(dataset, indices)
    eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(eval_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            predictions.extend(outputs.squeeze(dim=1).cpu().numpy())
            actuals.extend(targets.cpu().numpy())

            # Get cell_ids for this batch - correctly aligned since shuffle=False
            current_batch_size = images.shape[0]
            start_idx = batch_idx * batch_size
            end_idx = start_idx + current_batch_size
            batch_indices = indices[start_idx:end_idx]
            batch_cell_ids = dataset.data.iloc[batch_indices]["cell_id"].tolist()
            cell_ids.extend(batch_cell_ids)

    return predictions, actuals, cell_ids


def save_model(model, path, metadata=None):
    """
    Save the model checkpoint with optional metadata.

    Args:
        model: The model to save
        path: Path to save the checkpoint
        metadata: Optional dict with training metadata (epochs, lr, etc.)
    """
    checkpoint = {"model_state_dict": model.state_dict(), "metadata": metadata or {}}
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path, device):
    """
    Load a model from checkpoint.

    Args:
        path: Path to the checkpoint
        device: torch device

    Returns:
        Tuple of (model, metadata)
    """
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint.get("metadata", {})

    # Get model parameters from metadata (with defaults for backward compatibility)
    dropout_rate = metadata.get("dropout_rate", 0.3)
    model_type = metadata.get("model_type", "resnet50")

    model = Jaccard(dropout_rate=dropout_rate, model_type=model_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {path}")
    return model, metadata


def save_results_to_excel(train_results, val_results, test_results, output_path):
    """Save evaluation results to Excel with separate sheets per split."""
    with pd.ExcelWriter(output_path) as writer:
        any_written = False
        if train_results is not None and not train_results.empty:
            train_results.to_excel(writer, sheet_name="train", index=False)
            any_written = True
        if val_results is not None and not val_results.empty:
            val_results.to_excel(writer, sheet_name="validation", index=False)
            any_written = True
        if test_results is not None and not test_results.empty:
            test_results.to_excel(writer, sheet_name="test", index=False)
            any_written = True
        if not any_written:
            pd.DataFrame({"info": ["No data"]}).to_excel(
                writer, sheet_name="info", index=False
            )
    print(f"Results saved to {output_path}")


def get_split_indices(dataset):
    """Get train/val/test indices from dataset based on 'split' column."""
    train_indices = dataset.data[dataset.data["split"] == "train"].index.tolist()
    val_indices = dataset.data[dataset.data["split"] == "validation"].index.tolist()
    test_indices = dataset.data[dataset.data["split"] == "test"].index.tolist()
    return train_indices, val_indices, test_indices


def train(
    parquet_file,
    data_root=None,
    target_column=None,
    input_channels: Optional[Sequence[int] | str] = None,
    output_model="cnn_jaccard.pt",
    output_excel="results_cnn.xlsx",
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=50,
    weight_decay=1e-4,
    dropout_rate=0.3,
    patience=10,
    augment=True,
    seed=42,
    num_workers=4,
    grad_clip=1.0,
    model_type="resnet50",
    mlflow_tracking_uri=DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_experiment="cnn-jaccard",
    mlflow_run_name=None,
):
    """Train the CNN model for Jaccard index prediction."""
    # Set seed for reproducibility
    set_seed(seed)
    print(f"Random seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    # Training dataset with augmentation
    train_dataset = JaccardDataset(
        parquet_file,
        data_root=data_root,
        transform=get_transform(),
        augment=augment,
        target_column=target_column,
        input_channels=input_channels,
    )
    # Validation/test dataset without augmentation
    eval_dataset = JaccardDataset(
        parquet_file,
        data_root=data_root,
        transform=get_transform(),
        augment=False,
        target_column=target_column,
        input_channels=input_channels,
    )

    print(f"Using target column: {train_dataset.target_column}")
    print(f"Using input channels: {train_dataset.input_channels}")

    train_indices, val_indices, test_indices = get_split_indices(train_dataset)

    print(
        f"Dataset splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )
    print(f"Data augmentation: {'enabled' if augment else 'disabled'}")

    # DataLoaders
    train_data = Subset(train_dataset, train_indices)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_data = Subset(eval_dataset, val_indices)
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Model, criterion, optimizer
    model = Jaccard(dropout_rate=dropout_rate, model_type=model_type).to(device)
    print(f"Using model: {model_type}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Early stopping setup
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    last_lr = learning_rate
    final_epoch = 0

    # Start MLflow run
    with mlflow.start_run(run_name=mlflow_run_name):
        # Log hyperparameters
        mlflow.log_params(
            {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "weight_decay": weight_decay,
                "dropout_rate": dropout_rate,
                "patience": patience,
                "augment": augment,
                "seed": seed,
                "num_workers": num_workers,
                "grad_clip": grad_clip,
                "model_type": model_type,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
                "parquet_file": str(parquet_file),
                "target_column": str(train_dataset.target_column),
                "input_channels": ",".join(
                    str(channel) for channel in train_dataset.input_channels
                ),
                "device": str(device),
            }
        )

        # Training loop
        print(f"\nStarting training for {num_epochs} epochs...")
        print(
            f"Weight decay: {weight_decay}, Dropout: {dropout_rate}, Early stopping patience: {patience}"
        )

        for epoch in range(num_epochs):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, grad_clip
            )
            val_loss = validate_epoch(model, val_loader, criterion, device)

            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}"
            )

            # Log if LR changed
            if current_lr != last_lr:
                print(f"  -> Learning rate reduced to {current_lr:.2e}")
                last_lr = current_lr

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
                print("  -> New best validation loss!")
            else:
                epochs_without_improvement += 1

            final_epoch = epoch + 1

            if patience > 0 and epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Restored best model with validation loss: {best_val_loss:.4f}")

        # Log final metrics
        mlflow.log_metrics(
            {
                "best_val_loss": best_val_loss,
                "final_epoch": final_epoch,
            }
        )

        # Save model
        metadata = {
            "parquet_file": str(parquet_file),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "model_type": model_type,
            "augmentation": augment,
            "best_val_loss": best_val_loss,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
            "target_column": str(train_dataset.target_column),
            "input_channels": ",".join(
                str(channel) for channel in train_dataset.input_channels
            ),
        }
        save_model(model, output_model, metadata)

        # Log model artifact to MLflow
        mlflow.log_artifact(output_model)

        # Run evaluation and save results
        print("\nRunning evaluation on all splits...")
        eval_metrics = _run_evaluation(
            model,
            eval_dataset,
            train_indices,
            val_indices,
            test_indices,
            batch_size,
            device,
            output_excel,
        )

        # Log evaluation metrics
        if eval_metrics:
            mlflow.log_metrics(eval_metrics)

        # Log results excel as artifact
        mlflow.log_artifact(output_excel)

        active = mlflow.active_run()
        if active is not None:
            print(f"\nMLflow run ID: {active.info.run_id}")


def evaluate(
    parquet_file,
    data_root=None,
    target_column=None,
    input_channels: Optional[Sequence[int] | str] = None,
    model_path=None,
    output_excel="results_cnn.xlsx",
    batch_size=16,
):
    """Evaluate a trained model on the dataset and save results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if model_path is None:
        raise ValueError("model_path is required for evaluation.")

    model, metadata = load_model(model_path, device)
    print(f"Model metadata: {metadata}")

    effective_target_column = (
        target_column if target_column is not None else metadata.get("target_column")
    )
    effective_input_channels = (
        input_channels if input_channels is not None else metadata.get("input_channels")
    )

    # Dataset
    dataset = JaccardDataset(
        parquet_file,
        data_root=data_root,
        transform=get_transform(),
        target_column=effective_target_column,
        input_channels=effective_input_channels,
    )
    print(f"Using target column: {dataset.target_column}")
    print(f"Using input channels: {dataset.input_channels}")
    train_indices, val_indices, test_indices = get_split_indices(dataset)

    print(
        f"Dataset splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    # Run evaluation
    _run_evaluation(
        model,
        dataset,
        train_indices,
        val_indices,
        test_indices,
        batch_size,
        device,
        output_excel,
    )


def _run_evaluation(
    model,
    dataset,
    train_indices,
    val_indices,
    test_indices,
    batch_size,
    device,
    output_excel,
):
    """Internal function to run evaluation on all splits and save results."""

    # Evaluate all splits
    train_predictions, train_actuals, train_cell_ids = evaluate_model_with_ids(
        model, dataset, train_indices, batch_size, device
    )
    val_predictions, val_actuals, val_cell_ids = evaluate_model_with_ids(
        model, dataset, val_indices, batch_size, device
    )
    test_predictions, test_actuals, test_cell_ids = evaluate_model_with_ids(
        model, dataset, test_indices, batch_size, device
    )

    # Create DataFrames
    train_results = pd.DataFrame(
        {
            "cell_id": train_cell_ids,
            "Jaccard index": train_actuals,
            "Predicted Jaccard index": train_predictions,
        }
    )
    val_results = pd.DataFrame(
        {
            "cell_id": val_cell_ids,
            "Jaccard index": val_actuals,
            "Predicted Jaccard index": val_predictions,
        }
    )
    test_results = pd.DataFrame(
        {
            "cell_id": test_cell_ids,
            "Jaccard index": test_actuals,
            "Predicted Jaccard index": test_predictions,
        }
    )

    # Save results
    save_results_to_excel(train_results, val_results, test_results, output_excel)

    # Calculate comprehensive metrics for MLflow
    metrics = {}
    print("\n=== Evaluation Summary ===")
    for name, df in [
        ("train", train_results),
        ("val", val_results),
        ("test", test_results),
    ]:
        if not df.empty:
            y_true = df["Jaccard index"].values
            y_pred = df["Predicted Jaccard index"].values

            # Get comprehensive metrics
            regression_metrics = calculate_regression_metrics(y_true, y_pred)
            tolerance_metrics = calculate_tolerance_accuracy(y_true, y_pred)

            # Add to metrics dict with split prefix
            for key, value in regression_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"{name}_{key}"] = value
            for key, value in tolerance_metrics.items():
                metrics[f"{name}_{key}"] = value

            # Print summary
            print(f"{name.capitalize()}: {len(df)} samples")
            print(
                f"  R²: {regression_metrics.get('r2_score', 0):.4f}, "
                f"MAE: {regression_metrics.get('mae', 0):.4f}, "
                f"RMSE: {regression_metrics.get('rmse', 0):.4f}"
            )
            print(
                f"  Pearson: {regression_metrics.get('pearson_correlation', 0):.4f}, "
                f"Spearman: {regression_metrics.get('spearman_correlation', 0):.4f}"
            )

    return metrics
