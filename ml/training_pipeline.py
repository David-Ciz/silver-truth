import copy
import logging
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ml.data.datasets import CustomDataset
from ml.metrics.calculate_metrics import calculate_regression_metrics
from ml.models.models import SimpleNet, EnhancedSimpleNet, ModifiedResNet18, IntermidiateNet, SimpleFCNet


def train_run(train_df, train_transform, val_df, val_transform, run_name, config):
    # Access hyperparameters
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    momentum = config['momentum']
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # choose model to train. TODO: switch between models using config file
    # model = SimpleNet()
    # model = IntermidiateNet()
    # model = EnhancedSimpleNet()
    model = ModifiedResNet18()
    # model = SimpleFCNet()
    # choose your favourite optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_dataset = CustomDataset(train_df, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = CustomDataset(val_df, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # temporary tensorflow model architecture vizualization, commented out for now.

    # image, label =train_dataset.__getitem__(1)
    # print(image)
    # writer = SummaryWriter("runs/SimpleNet")
    # writer.add_graph(model, image.unsqueeze(0))
    # writer.close()
    # return
    # Set up TensorBoard
    writer = SummaryWriter(f"runs/{run_name}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters in the model: {total_params}")
    best_val_loss = float('inf')
    best_mse = 0
    best_mae = 0
    best_model_wrt_loss = None
    best_model_wrt_mse = None

    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        model, train_loss, train_predictions, train_labels = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_metrics = calculate_regression_metrics('train', train_labels, train_predictions)
        train_metrics['train_loss'] = train_loss

        val_loss, val_predictions, val_labels = validate_epoch(model, val_loader, loss_fn, device)
        val_metrics = calculate_regression_metrics('val', val_labels, val_predictions)
        val_metrics['val_loss'] = val_loss


        # Save the model with the lowest validation loss
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_model_wrt_loss = copy.deepcopy(model.state_dict())

        # # Save the model with the highest F1 score on validation
        # if train_metrics['train_f1'] > best_f1:
        #     best_f1 = train_metrics['train_f1']
        #     best_model_wrt_f1 = copy.deepcopy(model.state_dict())

        # Save the model with the lowest MSE
        if val_metrics['val_mse'] < best_mse:
            best_mse = val_metrics['val_mse']
            best_model_wrt_mse = copy.deepcopy(model.state_dict())

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/train', train_metrics['train_mse'], epoch)
        writer.add_scalar('MSE/val', val_metrics['val_mse'], epoch)
        writer.add_scalar('MAE/train', train_metrics['train_mae'], epoch)
        writer.add_scalar('MAE/val', val_metrics['val_mae'], epoch)

        logging.info(
            f"Epoch {epoch + 1}, "
            f"Train Loss: {train_loss:.5f}, Train MSE: {train_metrics['train_mse']:.5f}, "
            f"Train MAE: {train_metrics['train_mae']:.5f}, Train RMSE: {train_metrics['train_rmse']:.5f}, "
            f"Val Loss: {val_loss:.5f}, Val MSE: {val_metrics['val_mse']:.5f}, "
            f"Val MAE: {val_metrics['val_mae']:.5f}, Val RMSE: {val_metrics['val_rmse']:.5f}"
        )


    # At the end of training, after finding the best models...
    # Create a new model instance to load the state dicts
    best_loss_model = ModifiedResNet18()
    best_loss_model.load_state_dict(best_model_wrt_loss)
    # Save the best model to disk
    best_model_path = os.path.join("models", f'best_model_loss.pth')
    torch.save(best_loss_model, best_model_path)
    logging.info(f"Saved best model (MSE: {best_mse:.5f}) to {best_model_path}")

    best_mse_model = ModifiedResNet18()
    best_mse_model.load_state_dict(best_model_wrt_mse)
    # Save the best model to disk
    best_model_path = os.path.join("models", f'best_model_mse.pth')
    torch.save(best_mse_model, best_model_path)
    logging.info(f"Saved best model (MSE: {best_mse:.5f}) to {best_model_path}")
    writer.close()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save the training and validation DataFrames to temporary CSV files
        train_csv_path = os.path.join(tmp_dir, "train_data.csv")
        val_csv_path = os.path.join(tmp_dir, "val_data.csv")
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        # # Log the temporary CSV files as artifacts to MinIO
        # mlflow.log_artifact(train_csv_path, "train_data")
        # mlflow.log_artifact(val_csv_path, "val_data")
    return best_mse, best_mae  # Return appropriate metrics for regression


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss_avg = 0.0
    predictions = []
    total_labels = []
    # torch.nn

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        epoch_loss_avg += loss.item()
        loss.backward()
        optimizer.step()

        predictions.extend(outputs.cpu().detach().numpy())
        total_labels.extend(labels.cpu().numpy())
    epoch_loss_avg /= len(predictions)
    return model, epoch_loss_avg, predictions, total_labels


def validate_epoch(model, val_loader, loss_fn, device):
    model.eval()
    epoch_loss_avg = 0.0
    predictions = []
    total_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            epoch_loss_avg += loss.item()

            predictions.extend(outputs.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
    epoch_loss_avg /= len(predictions)
    return epoch_loss_avg, predictions, total_labels
