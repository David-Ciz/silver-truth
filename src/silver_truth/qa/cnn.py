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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import MLFlowLogger
import torchmetrics
from tqdm import tqdm
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
        rel_path = row["stacked_path"]

        if self.data_root:
            image_path = self.data_root / rel_path
        else:
            image_path = rel_path

        jaccard = row[self.target_column]

        img_np = tifffile.imread(image_path)

        if img_np.ndim == 2:
            img_np = np.stack([img_np, img_np], axis=0)
        elif img_np.ndim == 3:
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

        if self.augment:
            if random.random() > 0.5:
                img_np = np.flip(img_np, axis=2).copy()
            if random.random() > 0.5:
                img_np = np.flip(img_np, axis=1).copy()
            k = random.randint(0, 3)
            if k > 0:
                img_np = np.rot90(img_np, k, axes=(1, 2)).copy()

        img_np = img_np.astype(np.float32) / 255.0

        image = torch.from_numpy(img_np)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(jaccard, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Backbone / head
# ---------------------------------------------------------------------------

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

        if "resnet" in model_type:
            self.model.conv1 = nn.Conv2d(
                2, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# PyTorch Lightning module
# ---------------------------------------------------------------------------

class JaccardLightningModule(pl.LightningModule):
    """LightningModule wrapping the Jaccard backbone for regression training."""

    def __init__(
        self,
        model_type: str = "resnet50",
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Jaccard(dropout_rate=dropout_rate, model_type=model_type)
        self.criterion = nn.MSELoss()

        # torchmetrics — reset per epoch automatically
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_r2 = torchmetrics.R2Score()

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    def _shared_step(self, batch):
        images, targets = batch
        preds = self(images).squeeze(dim=1)
        loss = self.criterion(preds, targets)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        self.val_mae.update(preds, targets)
        self.val_r2.update(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_mae", self.val_mae.compute(), prog_bar=True)
        self.log("val_r2", self.val_r2.compute(), prog_bar=True)
        self.val_mae.reset()
        self.val_r2.reset()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers kept for evaluate() / _run_evaluation()
# ---------------------------------------------------------------------------

def evaluate_model_with_ids(model, dataset, indices, batch_size, device):
    """
    Evaluate model and return predictions, actuals, and cell_ids.

    Args:
        model: The trained model (nn.Module or LightningModule)
        dataset: The full JaccardDataset
        indices: List of indices for this split
        batch_size: Batch size for evaluation
        device: torch device

    Returns:
        Tuple of (predictions, actuals, cell_ids)
    """
    # Unwrap LightningModule if needed
    nn_model = model.model if isinstance(model, JaccardLightningModule) else model
    nn_model.eval()
    predictions = []
    actuals = []
    cell_ids = []

    eval_subset = Subset(dataset, indices)
    eval_loader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(eval_loader):
            images, targets = images.to(device), targets.to(device)
            outputs = nn_model(images)
            predictions.extend(outputs.squeeze(dim=1).cpu().numpy())
            actuals.extend(targets.cpu().numpy())

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
        model: The model to save (nn.Module or LightningModule)
        path: Path to save the checkpoint
        metadata: Optional dict with training metadata
    """
    nn_model = model.model if isinstance(model, JaccardLightningModule) else model
    checkpoint = {"model_state_dict": nn_model.state_dict(), "metadata": metadata or {}}
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Train the CNN model for Jaccard index prediction using PyTorch Lightning."""
    set_seed(seed)
    pl.seed_everything(seed, workers=True)
    print(f"Random seed: {seed}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_dataset = JaccardDataset(
        parquet_file,
        data_root=data_root,
        transform=get_transform(),
        augment=augment,
        target_column=target_column,
        input_channels=input_channels,
    )
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

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # LightningModule
    # ------------------------------------------------------------------
    lightning_model = JaccardLightningModule(
        model_type=model_type,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    print(f"Using model: {model_type}")

    # ------------------------------------------------------------------
    # Logger + callbacks
    # ------------------------------------------------------------------
    mlf_logger = MLFlowLogger(
        experiment_name=mlflow_experiment,
        run_name=mlflow_run_name,
        tracking_uri=mlflow_tracking_uri,
        log_model=False,  # we handle model saving ourselves
    )

    # Log all hyper-parameters up front
    mlf_logger.log_hyperparams(
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
                str(ch) for ch in train_dataset.input_channels
            ),
        }
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ------------------------------------------------------------------
    # Trainer — device selected automatically (MPS / CUDA / CPU)
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=grad_clip,
        callbacks=callbacks,
        logger=mlf_logger,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    print(f"\nStarting training for {num_epochs} epochs...")
    print(
        f"Weight decay: {weight_decay}, Dropout: {dropout_rate}, "
        f"Early stopping patience: {patience}"
    )

    trainer.fit(lightning_model, train_loader, val_loader)

    # ------------------------------------------------------------------
    # Retrieve the best checkpoint and save in legacy format
    # ------------------------------------------------------------------
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path:
        print(f"Loading best checkpoint: {best_ckpt_path}")
        lightning_model = JaccardLightningModule.load_from_checkpoint(best_ckpt_path)

    best_val_loss = float(trainer.checkpoint_callback.best_model_score or 0.0)
    final_epoch = trainer.current_epoch

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
            str(ch) for ch in train_dataset.input_channels
        ),
    }
    save_model(lightning_model, output_model, metadata)

    # Log remaining summary metrics and artifacts via the active MLflow run
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    run_id = mlf_logger.run_id
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {"best_val_loss": best_val_loss, "final_epoch": final_epoch}
        )
        mlflow.log_artifact(output_model)

        # Infer device used by the trainer for evaluation
        device = lightning_model.device if hasattr(lightning_model, "device") else torch.device("cpu")

        eval_metrics = _run_evaluation(
            lightning_model,
            eval_dataset,
            train_indices,
            val_indices,
            test_indices,
            batch_size,
            device,
            output_excel,
        )
        if eval_metrics:
            mlflow.log_metrics(eval_metrics)
        mlflow.log_artifact(output_excel)

    print(f"\nMLflow run ID: {run_id}")


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
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

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
    train_predictions, train_actuals, train_cell_ids = evaluate_model_with_ids(
        model, dataset, train_indices, batch_size, device
    )
    val_predictions, val_actuals, val_cell_ids = evaluate_model_with_ids(
        model, dataset, val_indices, batch_size, device
    )
    test_predictions, test_actuals, test_cell_ids = evaluate_model_with_ids(
        model, dataset, test_indices, batch_size, device
    )

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

    save_results_to_excel(train_results, val_results, test_results, output_excel)

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

            regression_metrics = calculate_regression_metrics(y_true, y_pred)
            tolerance_metrics = calculate_tolerance_accuracy(y_true, y_pred)

            for key, value in regression_metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"{name}_{key}"] = value
            for key, value in tolerance_metrics.items():
                metrics[f"{name}_{key}"] = value

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
