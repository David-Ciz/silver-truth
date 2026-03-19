import logging
import json
from pathlib import Path

import click

from silver_truth.data_processing.label_synchronizer import (
    synchronize_labels_with_tracking_markers,
    verify_folder_synchronization_logic,
    verify_dataset_synchronization_logic,
    synchronize_datasets_logic,
)
from silver_truth.data_processing.compression import compress_tifs_logic
from silver_truth.data_processing.utils.dataset_dataframe_creation import (
    create_dataset_dataframe_logic,
)
from silver_truth.data_processing.segmentation_stats import (
    build_oversized_cell_audit,
    collect_segmentation_object_stats,
    collect_segmentation_object_stats_from_dataframes,
    save_oversized_cell_visualizations,
    save_segmentation_object_stats,
    save_segmentation_summary,
    summarize_segmentation_object_stats,
)

# Configure logging globally
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _parse_rect_size(value: str) -> tuple[int, int]:
    """Parse a rectangular size option formatted as HEIGHTxWIDTH."""
    normalized = value.lower().replace(" ", "")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise click.BadParameter(
            f"Invalid rectangular size '{value}'. Use HEIGHTxWIDTH, for example 128x256."
        )

    try:
        height, width = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise click.BadParameter(
            f"Invalid rectangular size '{value}'. Use HEIGHTxWIDTH, for example 128x256."
        ) from exc

    if height <= 0 or width <= 0:
        raise click.BadParameter(
            f"Invalid rectangular size '{value}'. Height and width must be positive."
        )

    return height, width


@click.command()
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.argument("tracking_dir_01", type=click.Path(exists=True))
@click.argument("tracking_dir_02", type=click.Path(exists=True))
def verify_dataset_synchronization(
    dataset_dir: Path | str, tracking_dir_01: Path | str, tracking_dir_02: Path | str
):
    """
    This script verifies the synchronization between segmentations and tracking markers in a dataset.

    DATASET_DIR: Path to the dataset directory.
    TRACKING_DIR_01: Path to the 01 tracking directory.
    TRACKING_DIR_02: Path to the 02 tracking directory.
    """
    desynchronized_subfolders = verify_dataset_synchronization_logic(
        dataset_dir, tracking_dir_01, tracking_dir_02
    )
    if desynchronized_subfolders:
        click.echo(f"Desynchronized subfolders: {', '.join(desynchronized_subfolders)}")
    else:
        click.echo("All images are synchronized.")


@click.command()
@click.argument("label_folder", type=click.Path(exists=True))
@click.argument("tracking_folder", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug mode for verbose logging.")
def verify_folder_synchronization(label_folder, tracking_folder, debug):
    """
    This script verifies the synchronization between labels and tracking markers in a folder.

    LABEL_FOLDER: Path to the folder containing label images.
    TRACKING_FOLDER: Path to the folder containing tracking marker images.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    desynchronized_images = verify_folder_synchronization_logic(
        label_folder, tracking_folder
    )

    if desynchronized_images:
        click.echo("The following images are not synchronized:")
        click.echo(", ".join(map(str, desynchronized_images)))
    else:
        click.echo("All images are synchronized.")


@click.command()
@click.argument("input_segmentation_folder", type=click.Path(exists=True))
@click.argument("tra_markers_folder", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
def synchronize_labels(input_segmentation_folder, tra_markers_folder, output_directory):
    """
    This script synchronizes labels with tracking markers.

    INPUT_SEGMENTATION_FOLDER: Path to the folder containing segmentation results.
    TRA_MARKERS_FOLDER: Path to the folder containing TRA marker files.
    OUTPUT_DIRECTORY: Path to the folder where synchronized data will be saved.
    """
    if not Path(output_directory).is_dir():
        logging.info(f"Creating output directory: {output_directory}")
        Path(output_directory).mkdir(parents=True, exist_ok=True)
    synchronize_labels_with_tracking_markers(
        input_segmentation_folder, tra_markers_folder, output_directory
    )


@click.command()
@click.argument("datasets_folder", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
@click.option(
    "--debug", is_flag=True, help="Enable debug mode with verbose Fiji output"
)
def synchronize_datasets(datasets_folder, output_directory, debug):
    """Synchronizes all segmentations with tracking markers in all the datasets

    DATASETS_FOLDER: Path to the folder containing datasets.
    OUTPUT_DIRECTORY: Path to the folder where synchronized data will be saved.
    DEBUG: Enable debug mode with verbose output
    """
    synchronize_datasets_logic(datasets_folder, output_directory, debug)


@click.command()
@click.argument("synchronized_dataset_dir", type=click.Path(exists=True))
@click.option(
    "--output_path", type=click.Path(), help="Path to save the dataset dataframe"
)
@click.option(
    "--split-mode",
    type=click.Choice(["mixed", "fold-1", "fold-2"], case_sensitive=False),
    default="mixed",
    help="Split strategy: 'mixed' (stratified) or 'fold-X' (leave-one-sequence-out).",
)
@click.option(
    "--split-ratios",
    type=str,
    help="Comma-separated split ratios. Mixed: three values (Train,Val,Test). Fold: two values (Train,Val) for the training sequence.",
    default=None,
)
@click.option("--seed", default=42, help="Random seed for reproducibility.")
def create_dataset_dataframe(
    synchronized_dataset_dir: Path | str,
    output_path: Path | str,
    split_mode: str,
    split_ratios: str,
    seed: int,
) -> None:
    """
    Creates a pandas dataframe with dataset information from synchronized datasets.

    Args:
        SYNCHRONIZED_DATASET_DIR: Path to the synchronized dataset directory.
        OUTPUT_PATH: Path to save the parquet dataset dataframe.
    """
    create_dataset_dataframe_logic(
        synchronized_dataset_dir, output_path, split_mode, split_ratios, seed
    )


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "--non-recursive",
    "-n",
    is_flag=True,
    help="Process only the specified directory, not subdirectories",
)
@click.option(
    "--dry-run",
    "-d",
    is_flag=True,
    help="Show what would be done without modifying files",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Display detailed information for each file"
)
def compress_tifs(directory, non_recursive, dry_run, verbose):
    """Compress TIFF files using LZW compression in place.

    This tool finds all TIFF files in the specified DIRECTORY (and subdirectories)
    and compresses them using lossless LZW compression, overwriting the original files.

    Examples:

    \b
    # Compress all TIFFs in current directory and subdirectories
    python compress_tifs.py .

    \b
    # Only show what would be done without making changes
    python compress_tifs.py /path/to/images --dry-run

    \b
    # Process only the current directory, not subdirectories
    python compress_tifs.py . --non-recursive

    \b
    # Show detailed information for each file
    python compress_tifs.py /path/to/images --verbose
    """
    click.echo("TIFF Compression Tool")
    click.echo(f"{'=' * 30}")
    click.echo(f"Target directory: {directory}")
    click.echo(f"Mode: {'Non-recursive' if non_recursive else 'Recursive'}")
    click.echo(f"Dry run: {'Yes' if dry_run else 'No'}")
    click.echo(f"{'=' * 30}")

    compress_tifs_logic(directory, not non_recursive, dry_run, verbose)

    if dry_run:
        click.echo(
            click.style(
                "\nThis was a dry run. No files were modified. Run without --dry-run to apply changes.",
                fg="yellow",
            )
        )


@click.command("segmentation-size-stats")
@click.argument(
    "inputs",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--crop-size",
    "crop_sizes",
    multiple=True,
    type=int,
    default=(64,),
    show_default=True,
    help="Crop sizes to evaluate using bounding-box fit rate. Pass multiple times.",
)
@click.option(
    "--rect-size",
    "rect_sizes",
    multiple=True,
    callback=lambda _ctx, _param, values: tuple(
        _parse_rect_size(value) for value in values
    ),
    help=(
        "Rectangular crop sizes to evaluate, formatted as HEIGHTxWIDTH. "
        "Reports both fixed-orientation and swappable-orientation fit rates."
    ),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Optional path to save the JSON summary.",
)
@click.option(
    "--per-object-output",
    type=click.Path(path_type=Path),
    help="Optional path to save per-object stats (.parquet or .csv).",
)
@click.option(
    "--show-outliers-for",
    type=int,
    help=(
        "Optional crop size to audit for clipped cells. "
        "Writes an oversized-cell CSV/JSON and optional PNG context views."
    ),
)
@click.option(
    "--outlier-output-dir",
    type=click.Path(path_type=Path),
    help=(
        "Optional directory for oversized-cell audit outputs. "
        "Defaults to ./segmentation_outliers_sz{N} when --show-outliers-for is used."
    ),
)
@click.option(
    "--max-visualizations",
    type=int,
    default=25,
    show_default=True,
    help="Maximum number of oversized-cell PNGs to generate.",
)
@click.option(
    "--context-pad",
    type=int,
    default=32,
    show_default=True,
    help="Extra pixels of context around the audited cell in visualization PNGs.",
)
def segmentation_size_stats(
    inputs: tuple[Path, ...],
    crop_sizes: tuple[int, ...],
    rect_sizes: tuple[tuple[int, int], ...],
    output: Path | None,
    per_object_output: Path | None,
    show_outliers_for: int | None,
    outlier_output_dir: Path | None,
    max_visualizations: int,
    context_pad: int,
):
    """Summarize cell size statistics from GT folders or dataset parquets."""
    if not inputs:
        raise click.UsageError(
            "Provide at least one segmentation directory or dataset parquet."
        )

    input_paths = tuple(Path(path) for path in inputs)
    parquet_inputs = tuple(path for path in input_paths if path.suffix == ".parquet")
    directory_inputs = tuple(path for path in input_paths if path.is_dir())

    if parquet_inputs and directory_inputs:
        raise click.UsageError(
            "Do not mix directories and parquet files in one invocation."
        )

    if parquet_inputs:
        stats_df = collect_segmentation_object_stats_from_dataframes(parquet_inputs)
    elif directory_inputs:
        stats_df = collect_segmentation_object_stats(directory_inputs)
    else:
        raise click.UsageError(
            "Inputs must be segmentation directories or .parquet dataset files."
        )

    summary = summarize_segmentation_object_stats(
        stats_df, crop_sizes=crop_sizes, rect_sizes=rect_sizes
    )

    click.echo(json.dumps(summary, indent=2))

    if output is not None:
        save_segmentation_summary(summary, output)
        click.echo(f"Saved summary JSON to {output}")

    if per_object_output is not None:
        save_segmentation_object_stats(stats_df, per_object_output)
        click.echo(f"Saved per-object stats to {per_object_output}")

    if show_outliers_for is not None:
        audit_df = build_oversized_cell_audit(stats_df, show_outliers_for)
        audit_output_dir = (
            outlier_output_dir
            if outlier_output_dir is not None
            else Path.cwd() / f"segmentation_outliers_sz{show_outliers_for}"
        )
        audit_output_dir.mkdir(parents=True, exist_ok=True)
        audit_csv_path = audit_output_dir / f"oversized_cells_sz{show_outliers_for}.csv"
        summary_json_path = (
            audit_output_dir / f"oversized_cells_sz{show_outliers_for}_summary.json"
        )
        visualizations_dir = (
            audit_output_dir / f"oversized_cells_sz{show_outliers_for}_png"
        )

        audit_df.to_csv(audit_csv_path, index=False)
        created_files = save_oversized_cell_visualizations(
            audit_df,
            visualizations_dir,
            crop_size=show_outliers_for,
            max_visualizations=max_visualizations,
            context_pad=context_pad,
        )

        audit_summary = {
            "crop_size": show_outliers_for,
            "n_cells_total": int(len(stats_df)),
            "n_oversized_cells": int(len(audit_df)),
            "n_gt_images_with_oversized_cells": int(audit_df["gt_image"].nunique())
            if not audit_df.empty
            else 0,
            "n_visualizations_written": len(created_files),
            "audit_csv": str(audit_csv_path),
            "visualizations_dir": str(visualizations_dir),
        }
        summary_json_path.write_text(
            json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8"
        )

        click.echo(json.dumps({"outlier_audit": audit_summary}, indent=2))
        click.echo(f"Saved audit CSV to {audit_csv_path}")
        click.echo(f"Saved summary JSON to {summary_json_path}")
        if created_files:
            click.echo(
                f"Saved {len(created_files)} visualization PNGs to {visualizations_dir}"
            )


@click.group()
def cli():
    pass


cli.add_command(synchronize_labels)
cli.add_command(verify_folder_synchronization)
cli.add_command(synchronize_datasets)
cli.add_command(verify_dataset_synchronization)
cli.add_command(create_dataset_dataframe)
cli.add_command(compress_tifs)
cli.add_command(segmentation_size_stats)


if __name__ == "__main__":
    cli()
