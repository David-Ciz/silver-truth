import logging
from typing import List

import tifffile
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import pathlib
from pathlib import Path
import pandas as pd
import click

from preprocessing.data_preprocessing import parse_segmentation_data


@click.group()
def cli():
    """CLI group for dataset operations."""
    pass


@cli.command()
@click.argument('dataset_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument('gt_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
def create_dataset_csv(dataset_dir: Path, gt_dir: Path) -> None:
    """
    Create a CSV dataset aggregating information from competitors segmentation logs.

    Args:
        dataset_dir (Path): Path to the dataset directory.
        gt_dir (Path): Path to the ground truth directory.
    """
    dataset_dfs: List[pd.DataFrame] = []
    competitor_folders = [x for x in dataset_dir.iterdir() if x.is_dir() and x.name not in ('01_GT', '02_GT', '01')]

    for competitor_folder in competitor_folders:
        segmentation_logs = list(competitor_folder.glob("**/*.txt"))
        for segmentation_log in segmentation_logs:
            dataset_df = parse_segmentation_data(segmentation_log, gt_dir)
            dataset_dfs.append(dataset_df)

    dataset_df = pd.concat(dataset_dfs)
    output_file = dataset_dir / "dataset.csv"
    dataset_df.to_csv(output_file, index=False)
    click.echo(f"Dataset CSV created: {output_file}")


@cli.command()
@click.argument('dataset_csv', type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path))
@click.argument('merged_images_out_dir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
def preprocess_dataset_cli(dataset_csv: Path, merged_images_out_dir: Path) -> None:
    """
    Preprocess the dataset by merging source images with masks.

    Args:
        dataset_csv (Path): Path to the input dataset CSV file.
        merged_images_out_dir (Path): Path to the output directory for merged images.
    """
    preprocess_dataset(dataset_csv, merged_images_out_dir)
    click.echo(f"Preprocessed dataset saved: {merged_images_out_dir}/preprocessed_dataset.csv")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()