import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import click
import tifffile
import numpy as np
from pathlib import Path
import random
import mlflow
import mlflow.pytorch

from src.metrics.qa_model_evaluation import (
    calculate_regression_metrics,
    calculate_tolerance_accuracy,
)


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
    def __init__(self, parquet_file, data_root=None, transform=None, augment=False):
        self.data = pd.read_parquet(parquet_file)
        # If data_root is provided, convert to Path, else None
        self.data_root = Path(data_root) if data_root else None
        self.transform = transform
        self.augment = augment

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

        jaccard = row["jaccard_score"]

        # Read the stacked TIFF with tifffile (PIL doesn't handle multi-frame TIFFs correctly)
        # Shape: (2, H, W) - channel-first
        # Dtype: uint8 (0-255)
        # Channel 0: Raw microscope image (grayscale)
        # Channel 1: Segmentation mask (binary: 0 or 255)
        img_np = tifffile.imread(image_path)

        if img_np.ndim == 2:
            # If there is only one channel, expand to 2 channels (for testing)
            img_np = np.stack([img_np, img_np], axis=0)
        elif img_np.ndim == 3:
            # Handle different channel arrangements
            if img_np.shape[0] == 2:
                pass  # already (2, H, W) - correct format
            elif img_np.shape[-1] == 2:
                # (H, W, 2) -> transpose to (2, H, W)
                img_np = np.transpose(img_np, (2, 0, 1))
            else:
                raise ValueError(
                    f"Image at {image_path} does not have 2 channels. Shape: {img_np.shape}"
                )
        else:
            raise ValueError(
                f"Image at {image_path} has unsupported shape {img_np.shape}."
            )

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
class JaccardResNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(JaccardResNet, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # Adjust for 2 channels

        # Replace the final fc layer with dropout + linear
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove original fc
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, 1)  # Regression output

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def tensor_normalize(tensor, mean, std):
    """Normalize tensor with given mean and std."""
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def get_transform():
    """
    Get the default transform for the dataset.

    Assumes input is already normalized to [0, 1] range.
    Maps [0, 1] -> [-1, 1] which is standard for pretrained models.
    """
    return lambda x: tensor_normalize(x, mean=[0.5, 0.5], std=[0.5, 0.5])


def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm=1.0):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), targets)
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
            predictions.extend(outputs.squeeze().cpu().numpy())
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
    model = JaccardResNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = checkpoint.get("metadata", {})
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


# ============================================================================
# CLI Commands
# ============================================================================


@click.group()
def cli():
    """ResNet50 Jaccard Index Prediction - Training and Evaluation CLI."""
    pass


@cli.command()
@click.option(
    "--parquet-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input Parquet file with split column.",
)
@click.option(
    "--data-root",
    type=click.Path(exists=True),
    default=None,
    help="Root directory for data (prepended to stacked_path). Use for HPC scratch.",
)
@click.option(
    "--output-model",
    type=click.Path(),
    default="resnet50_jaccard.pt",
    help="Path to save the trained model.",
)
@click.option(
    "--output-excel",
    type=click.Path(),
    default="results_resnet50.xlsx",
    help="Path to save the evaluation results.",
)
@click.option("--batch-size", type=int, default=16, help="Batch size for training.")
@click.option("--learning-rate", type=float, default=1e-4, help="Learning rate.")
@click.option("--num-epochs", type=int, default=50, help="Number of training epochs.")
@click.option(
    "--weight-decay", type=float, default=1e-4, help="Weight decay (L2 regularization)."
)
@click.option(
    "--dropout-rate", type=float, default=0.3, help="Dropout rate before final layer."
)
@click.option(
    "--patience", type=int, default=10, help="Early stopping patience (0 to disable)."
)
@click.option(
    "--augment/--no-augment", default=True, help="Enable/disable data augmentation."
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
@click.option(
    "--num-workers", type=int, default=4, help="Number of DataLoader workers."
)
@click.option(
    "--grad-clip",
    type=float,
    default=1.0,
    help="Gradient clipping max norm (0 to disable).",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=None,
    help="MLflow tracking URI (default: ./mlruns).",
)
@click.option(
    "--mlflow-experiment",
    type=str,
    default="resnet50-jaccard",
    help="MLflow experiment name.",
)
@click.option(
    "--mlflow-run-name",
    type=str,
    default=None,
    help="MLflow run name (default: auto-generated).",
)
def train(
    parquet_file,
    data_root,
    output_model,
    output_excel,
    batch_size,
    learning_rate,
    num_epochs,
    weight_decay,
    dropout_rate,
    patience,
    augment,
    seed,
    num_workers,
    grad_clip,
    mlflow_tracking_uri,
    mlflow_experiment,
    mlflow_run_name,
):
    """Train the ResNet50 model for Jaccard index prediction."""
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
        parquet_file, data_root=data_root, transform=get_transform(), augment=augment
    )
    # Validation/test dataset without augmentation
    eval_dataset = JaccardDataset(
        parquet_file, data_root=data_root, transform=get_transform(), augment=False
    )

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
    model = JaccardResNet(dropout_rate=dropout_rate).to(device)
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
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
                "parquet_file": str(parquet_file),
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
            "augmentation": augment,
            "best_val_loss": best_val_loss,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
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

        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")


@cli.command()
@click.option(
    "--parquet-file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input Parquet file with split column.",
)
@click.option(
    "--data-root",
    type=click.Path(exists=True),
    default=None,
    help="Root directory for data (prepended to stacked_path). Use for HPC scratch.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trained model checkpoint.",
)
@click.option(
    "--output-excel",
    type=click.Path(),
    default="results_resnet50.xlsx",
    help="Path to save the evaluation results.",
)
@click.option("--batch-size", type=int, default=16, help="Batch size for evaluation.")
def evaluate(parquet_file, data_root, model_path, output_excel, batch_size):
    """Evaluate a trained model on the dataset and save results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, metadata = load_model(model_path, device)
    print(f"Model metadata: {metadata}")

    # Dataset
    dataset = JaccardDataset(
        parquet_file, data_root=data_root, transform=get_transform()
    )
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


if __name__ == "__main__":
    cli()
