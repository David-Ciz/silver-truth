import os
import subprocess
import logging
from enum import Enum
from typing import Optional

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FusionModel(Enum):
    """Enumeration for the fusion models available in Fusers.java."""

    THRESHOLD_FLAT = "Threshold - flat weights"
    THRESHOLD_USER = "Threshold - user weights"
    MAJORITY_FLAT = "Majority - flat weights"
    SIMPLE = "SIMPLE"
    BIC_FLAT_VOTING = "BICv2 with FlatVoting, SingleMaskFailSafe and CollisionResolver"
    BIC_WEIGHTED_VOTING = (
        "BICv2 with WeightedVoting, SingleMaskFailSafe and CollisionResolver"
    )


def fuse_segmentations(
    jar_path: str,
    job_file_path: str,
    output_path_pattern: str,
    time_points: str,
    num_threads: int,
    fusion_model: FusionModel,
    threshold: float = 1.0,
    cmv_mode: Optional[str] = None,
    seg_eval_folder: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Runs the Fusers Java tool by calling the flexible RunFusersCli wrapper."""
    # (The body of this function remains exactly the same as before)
    jar_path = os.path.abspath(jar_path)
    job_file_path = os.path.abspath(job_file_path)
    output_dir = os.path.dirname(os.path.abspath(output_path_pattern))
    if not os.path.isdir(output_dir):
        logging.warning(
            f"Output directory '{output_dir}' does not exist. Creating the directory"
        )
        os.makedirs(output_dir, exist_ok=True)

    if seg_eval_folder:
        seg_eval_folder = os.path.abspath(seg_eval_folder)

    logging.info(f"Preparing to run Fusers with model: '{fusion_model.value}'")
    logging.info(f"Job file: {job_file_path}")

    command = [
        "java",
        "-cp",
        jar_path,
        "de.mpicbg.ulman.fusion.RunFusersCli",
        fusion_model.value,
        job_file_path,
        str(threshold),
        output_path_pattern,
        time_points,
        str(num_threads),
    ]
    if cmv_mode:
        command.append(cmv_mode)
    if seg_eval_folder:
        command.append(seg_eval_folder)

    logging.info(f"Executing command: {' '.join(command)}")
    stdout_pipe = None if debug else subprocess.PIPE
    stderr_pipe = None if debug else subprocess.PIPE
    result = subprocess.run(
        command, stdout=stdout_pipe, stderr=stderr_pipe, text=True, check=False
    )

    if result.returncode != 0:
        logging.error("The Fusers Java process failed.")
        if not debug:
            logging.error(f"Return Code: {result.returncode}")
            logging.error(f"STDOUT:\n{result.stdout}")
            logging.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError("Execution of Fusers (RunFusersCli) Java tool failed.")
    else:
        logging.info("Fusers Java process completed successfully.")
