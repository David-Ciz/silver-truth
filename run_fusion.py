import logging
from typing import Optional

import click

from src.fusion.fusion import FusionModel, fuse_segmentations

# --- 1. Core Logic (unchanged) ---
# This part contains your function and Enum, which can be kept in the same file
# or imported from another module (e.g., from src.fusion.fusion import ...)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- 2. Click CLI Definition ---
# This part replaces your `if __name__ == '__main__':` block.


@click.command()
@click.option(
    "--jar-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the executable Java JAR file (e.g., fusers-all-dependencies.jar).",
    default="src/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar",
)
@click.option(
    "--job-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the job specification file listing input image patterns.",
)
@click.option(
    "--output-pattern",
    required=True,
    help="Output filename pattern, including 'TTT' or 'TTTT' (e.g., '/path/to/fused_TTT.tif').",
)
@click.option(
    "--time-points",
    required=True,
    help='Timepoints to process as a string (e.g., "1-9,23,25").',
)
@click.option(
    "--num-threads", required=True, type=int, help="Number of processing threads."
)
@click.option(
    "--model",
    required=True,
    type=click.Choice([e.name for e in FusionModel], case_sensitive=False),
    help="The fusion model to use.",
)
@click.option(
    "--threshold",
    default=1.0,
    type=float,
    show_default=True,
    help="Voting threshold for merging.",
)
@click.option(
    "--cmv-mode",
    default=None,
    help='Enable Combinatorial Model Validation mode (e.g., "CMV", "CMV2_8").',
)
@click.option(
    "--seg-folder",
    default=None,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Optional path to ground truth folder for scoring.",
)
@click.option(
    "--debug/--no-debug",
    default=False,
    show_default=True,
    help="Enable debug logging and show Java process output.",
)
def run_fusion(
    jar_path: str,
    job_file: str,
    output_pattern: str,
    time_points: str,
    num_threads: int,
    model: str,
    threshold: float,
    cmv_mode: Optional[str],
    seg_folder: Optional[str],
    debug: bool,
):
    """
    Runs the Fusers Java segmentation fusion tool via a command-line interface.

    Args:
        jar_path (str): Path to the executable Java JAR file (e.g., `fusers-all-dependencies.jar`).
        job_file (str): Path to the job specification file listing input image patterns.
        output_pattern (str): Output filename pattern, including 'TTT' or 'TTTT' (e.g., `/path/to/fused_TTT.tif`).
        time_points (str): Timepoints to process as a string (e.g., `"1-9,23,25"`).
        num_threads (int): Number of processing threads.
        model (str): The fusion model to use, corresponding to `FusionModel` enum values.
        threshold (float): Voting threshold for merging.
        cmv_mode (Optional[str]): Combinatorial Model Validation mode (e.g., `"CMV"`, `"CMV2_8"`).
        seg_folder (Optional[str]): Optional path to ground truth folder for scoring.
        debug (bool): Enable debug logging and show Java process output.

    Raises:
        RuntimeError: If the Fusers Java process fails.
        Exception: For any unexpected errors.

    Returns:
        None
    """
    try:
        # Convert the string model name from the CLI back to the Enum member
        fusion_model_enum = FusionModel[model.upper()]

        click.echo(
            click.style(
                f"Starting fusion process with model: {fusion_model_enum.value}",
                fg="green",
            )
        )

        fuse_segmentations(
            jar_path=jar_path,
            job_file_path=job_file,
            output_path_pattern=output_pattern,
            time_points=time_points,
            num_threads=num_threads,
            fusion_model=fusion_model_enum,
            threshold=threshold,
            cmv_mode=cmv_mode,
            seg_eval_folder=seg_folder,
            debug=debug,
        )

        click.echo(
            click.style("Fusion process completed successfully!", fg="green", bold=True)
        )

    except RuntimeError as e:
        click.echo(click.style(f"Error: {e}", fg="red", bold=True))
        # Exit with a non-zero status code to indicate failure
        exit(1)
    except Exception as e:
        click.echo(
            click.style(f"An unexpected error occurred: {e}", fg="red", bold=True)
        )
        exit(1)


if __name__ == "__main__":
    run_fusion()
