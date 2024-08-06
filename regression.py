import logging
import os
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, KFold

from ml.data.datasets import CustomDataset, setup_transformations
import click

from ml.metrics import calculate_metrics
from ml.training_pipeline import train_run, validate_epoch
from PIL import Image


@click.group
def cli():
    pass


@cli.command('train')
@click.argument('dataset-csv', type=click.Path(exists=True))
@click.argument('config-path', type=click.Path(exists=True))
@click.option('--run_name', type=str, default=None, help='Name for MLFLOW tracking')
@click.option('--cross-validate', type=int, default=None, help='Number of splits for cross validation, None means no '
                                                               'cross validation')
@click.option('--augment', type=bool, default=False, help='Augment during training')
def train(dataset_csv: str, config_path, run_name, cross_validate, augment):
    """
    Trains a neural network for regression of a jaccard value.
    Expects data in directory defined in the 'table.csv'.
    :param dataset_csv: csv containing paths to dataset images and GT jaccard values.
    :param config-path: path to basic model configurations, containing hyperparameters.
    """
    # load all tables into a single dataframe
    df = pd.read_csv(dataset_csv)
    # load the configuration file
    with open(config_path, 'rb') as config_file:
        config = tomllib.load(config_file)

    # Compute mean and std of the dataset for transformations
    #dataset = CustomDataset(df, transform=transforms.ToTensor())
    #loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)
    #mean, std = compute_dataset_mean_std(loader)
    # TODO: track mean and std
    # TODO: scale between 0-1 and after that, do a mean and std normalization
    # Define your transformations
    train_transform, val_transform = setup_transformations(False)
    # if cross_validate:
    #     kf = KFold(n_splits=cross_validate)  # or any other number of splits
    #     mlflow.set_experiment("k-fold")
    #     all_f1_metrics = []
    #     all_accuracy_metrics = []
    #     for fold, (train_index, val_index) in enumerate(kf.split(final_df)):
    #         train_df = final_df.iloc[train_index]
    #         val_df = final_df.iloc[val_index]
    #         logging.info(f"beginning fold number: {fold}")
    #         best_f1, best_accuracy = train_run(train_df, train_transform, val_df, val_transform, run_name, config)
    #         all_f1_metrics.append(best_f1)
    #         all_accuracy_metrics.append(best_accuracy)
    #     # Initialize sums
    #     sum_accuracy = 0
    #     sum_f1 = 0
    #
    #     # Sum up metrics
    #     for f1 in all_f1_metrics:
    #         sum_f1 += f1
    #     for accuracy in all_accuracy_metrics:
    #         sum_accuracy += accuracy
    #
    #     # Calculate averages
    #     average_accuracy = sum_accuracy / len(all_accuracy_metrics)
    #     average_f1 = sum_f1 / len(all_f1_metrics)
    #
    #     logging.info(f"Average fold Accuracy: {average_accuracy}")
    #     logging.info(f"Average fold F1 Score: {average_f1}")
    #     mlflow.log_metric("average_fold_accuracy", average_accuracy)
    #     mlflow.log_metric("average_fold_f1", average_f1)
    #
    # else:
    # Split the data
    train_df, val_df = train_test_split(df, test_size=0.2)
    logging.info(f"Beginning a new train run")
    train_run(train_df, train_transform, val_df, val_transform, run_name, config)

# @cli.command()
# @click.argument('data-dir', type=click.Path(exists=True))
# @click.argument('model-path', type=click.Path(exists=True))
# @click.argument('batch-size', type=int)
# @click.option('--infer-train', type=bool, default=False)
# def infer(data_dir: str, model_path: str, batch_size: int, infer_train: bool):
#     """infers dataset using the specified model."""
#     mlflow.set_tracking_uri("http://localhost:5000")
#     run_id = "c3fad7e45e46423ab7a50e8be4533d5d"
#     model = mlflow.pytorch.load_model(f"runs:/{run_id}/best_model_val_loss")
#     train_data_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="train_data/train_data.csv")
#     val_data_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="val_data/val_data.csv")
#     train_data = pd.read_csv(train_data_path)
#     val_data = pd.read_csv(val_data_path)
#     if infer_train:
#         val_data = pd.concat([train_data, val_data])
#
#     val_transform = transforms.Compose([
#         transforms.Resize((48, 48)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=33241.7852, std=117.7661)
#     ])
#     val_dataset = CustomDataset(val_data, transform=val_transform)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     val_loss, val_predictions, val_labels = validate_epoch(model, val_loader)
#     val_metrics = calculate_metrics('val', val_labels, val_predictions)
#     logging.info(
#         f"Val Loss: {val_loss:.5f}, "
#         f"Val Acc: {val_metrics['val_accuracy']:.2f}, Val F1: {val_metrics['val_f1']:.2f}, Val TP: {val_metrics['val_TP']}, "
#         f"Val TN: {val_metrics['val_TN']}, Val FP: {val_metrics['val_FP']}, Val FN: {val_metrics['val_FN']}"
#     )
#     val_data["predictions"] = val_predictions
#     val_data.to_csv("inference_all_results.csv")
#
#
# @cli.command()
# @click.argument('data-dir')
# @click.argument('results-csv-path')
# @click.argument('out-dir')
# @click.option("--visualize-label", default=True)
# @click.option("--overlay", default=True)
# def visualize_inference(data_dir: str | os.PathLike, results_csv_path: str | os.PathLike, out_dir: str, visualize_label, overlay: bool):
#     df = pd.read_csv(results_csv_path)
#     df = df.sort_values(by="path")
#     current_frame = None
#     base_image = None
#     labeled_base_image = None
#     overlay_base_image = None
#     for index, row in df.iterrows():
#         current_image_path = Path(row["path"])
#         frame = current_image_path.parent.name
#         if frame != current_frame:
#             if base_image is not None:
#                 inferred_frame_path = Path(out_dir).joinpath(current_frame)
#                 inferred_frame_path.mkdir(parents=True, exist_ok=True)
#                 cv2.imwrite(str(inferred_frame_path.joinpath("inferred_base.jpg")), base_image)
#                 cv2.imwrite(str(inferred_frame_path.joinpath("labeled_base.jpg")), labeled_base_image)
#                 cv2.imwrite(str(inferred_frame_path.joinpath("overlay_base.jpg")), overlay_base_image)
#             current_frame = frame
#             base_image_path = Path(data_dir).joinpath(frame).joinpath("base.jpg")
#             base_image = cv2.imread(str(base_image_path))
#             if visualize_label:
#                 labeled_base_image = base_image.copy()
#             if overlay:
#                 overlay_base_image = base_image.copy()
#         x1, y1, x2, y2 = int(row['bbox_x1'])+1, int(row['_y1'])+1, int(row['_x2'])-1, int(row['_y2']-1)
#         if row["predictions"] == 1:
#             cv2.rectangle(base_image,(x1,y1), (x2,y2), (0,255,0), 2)
#         elif row["predictions"] == 0:
#             cv2.rectangle(base_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         if visualize_label:
#             if row['label']:
#                 cv2.rectangle(labeled_base_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             else:
#                 cv2.rectangle(labeled_base_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         if overlay:
#             if row["predictions"] == 1 and row['label'] == 1:
#                 cv2.rectangle(overlay_base_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             if row["predictions"] == 0 and row['label'] == 0:
#                 cv2.rectangle(overlay_base_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
#             if row["predictions"] == 1 and row['label'] == 0:
#                 cv2.rectangle(overlay_base_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             if row["predictions"] == 0 and row['label'] == 1:
#                 cv2.rectangle(overlay_base_image, (x1, y1), (x2, y2), (0, 0, 255), 2)


# def hyperparameter_search():
#     param_grid = {
#         'input_channels': [1],  # Since your images are grayscale
#         'num_filters': [16, 32, 64],  # Number of filters in the conv layer
#         'num_layers': [1, 2, 3],  # Number of conv layers
#         'kernel_size': [3, 5],  # Size of the kernel in the conv layer
#         'padding': [1, 2],  # Padding in the conv layer
#         'pool_size': [2],  # Size of the max-pooling window
#         'pool_stride': [2],  # Stride of the max-pooling window
#         'fc_size': [1]  # Output size of the fully connected layer
#     }
#


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()