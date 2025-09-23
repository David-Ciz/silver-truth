import json
import logging
import re
from typing import Optional

import click
import mlflow
import git
import yaml
import pandas as pd
from pathlib import Path

from src.fusion.fusion import (
    fuse_segmentations,
    FusionModel,
    add_fused_images_to_dataframe_logic,
)
from src.metrics.evaluation_logic import run_evaluation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_git_commit_hash() -> Optional[str]:
    """Gets the current git commit hash."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        return None


def get_dvc_hash(dvc_file_path: Path) -> Optional[str]:
    """Gets the DVC hash from a .dvc file."""
    if not dvc_file_path.exists():
        return None
    try:
        with open(dvc_file_path, "r") as f:
            dvc_data = yaml.safe_load(f)
            if "outs" in dvc_data and dvc_data["outs"] and "md5" in dvc_data["outs"][0]:
                return dvc_data["outs"][0]["md5"]
    except Exception as e:
        logging.warning(f"Could not read or parse DVC file {dvc_file_path}: {e}")
    return None


def sanitize_metric_name(name: str) -> str:
    """Sanitizes a string to be a valid MLflow metric name."""
    return re.sub(r"[^a-zA-Z0-9_./-s:]", "_", name)


def format_time_points(time_points: list[int]) -> str:
    """Formats a list of time points into a compact string (e.g., '1-3,5,8-10')."""
    if not time_points:
        return ""
    time_points = sorted(list(set(time_points)))
    ranges = []
    start = time_points[0]
    end = time_points[0]
    for i in range(1, len(time_points)):
        if time_points[i] == end + 1:
            end = time_points[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = time_points[i]
            end = time_points[i]
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ",".join(ranges)


@click.group()
def cli():
    """A centralized CLI for running and tracking experiments with MLflow."""
    pass


@cli.command()
@click.option(
    "--experiment-name",
    default="fusion-experiments",
    help="Name for the MLflow experiment.",
)
@click.option(
    "--jar-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default="src/data_processing/cell_tracking_java_helpers/label-fusion-ng-2.2.0-SNAPSHOT-jar-with-dependencies.jar",
    help="Path to the executable Java JAR file.",
)
@click.option(
    "--job-file",
    "job_file_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the job specification file.",
)
@click.option(
    "--output-pattern",
    "output_path_pattern",
    required=True,
    help="Output filename pattern (e.g., '/path/to/fused_TTT.tif').",
)
@click.option(
    "--model",
    "fusion_model_name",
    required=True,
    type=click.Choice([e.name for e in FusionModel], case_sensitive=False),
    help="The fusion model to use.",
)
@click.option(
    "--threshold", default=1.0, type=float, help="Voting threshold for merging."
)
@click.option(
    "--num-threads", default=8, type=int, help="Number of processing threads."
)
@click.option(
    "--time-points",
    default=None,
    help="Timepoints to process (e.g., '1-9,23,25'). If not provided, inferred from GT.",
)
@click.option(
    "--dataset-dataframe-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the dataset dataframe for evaluation.",
)
@click.option(
    "--evaluation-output",
    type=click.Path(path_type=Path),
    help="Path to save evaluation results as CSV.",
)
def run_fusion_experiment(
    experiment_name: str,
    jar_path: str,
    job_file_path: str,
    output_path_pattern: str,
    fusion_model_name: str,
    threshold: float,
    num_threads: int,
    time_points: Optional[str],
    dataset_dataframe_path: Path,
    evaluation_output: Optional[Path],
):
    """Runs a fusion experiment and logs it to MLflow."""
    mlflow.set_experiment(experiment_name)

    # --- Infer Time Points if not provided ---
    if time_points is None:
        logging.info("Inferring time points from ground truth images...")
        df = pd.read_parquet(dataset_dataframe_path)
        gt_df = df[df["gt_image"].notna()]
        if gt_df.empty:
            raise ValueError("No ground truth images found to infer time points from.")
        inferred_points = sorted(
            [
                int(key.split("_")[1].replace(".tif", ""))
                for key in gt_df["composite_key"]
                if "_" in key
            ]
        )
        time_points = format_time_points(inferred_points)
        logging.info(f"Inferred time points: {time_points}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"Starting MLflow run: {run_id}")

        # --- Log Git Commit ---
        commit_hash = get_git_commit_hash()
        if commit_hash:
            mlflow.set_tag("git_commit", commit_hash)
            logging.info(f"Logged Git commit: {commit_hash}")

        # --- Log DVC Hash ---
        dvc_hash = get_dvc_hash(Path("data.dvc"))
        if dvc_hash:
            mlflow.set_tag("dvc_data_hash", dvc_hash)
            logging.info(f"Logged DVC data hash: {dvc_hash}")

        # --- Log Parameters ---
        params = {
            "jar_path": jar_path,
            "job_file": job_file_path,
            "output_pattern": output_path_pattern,
            "model": fusion_model_name,
            "threshold": threshold,
            "num_threads": num_threads,
            "time_points": time_points,
            "dataset_dataframe_path": str(dataset_dataframe_path),
        }
        mlflow.log_params(params)
        logging.info(f"Logged parameters: {params}")

        # --- Log Job File as Artifact ---
        mlflow.log_artifact(job_file_path, "input_configs")
        logging.info(f"Logged job file as artifact: {job_file_path}")

        # --- Run Fusion ---
        try:
            fusion_model_enum = FusionModel[fusion_model_name.upper()]
            logging.info("Starting fusion process...")
            fuse_segmentations(
                jar_path=jar_path,
                job_file_path=job_file_path,
                output_path_pattern=output_path_pattern,
                time_points=time_points,
                num_threads=num_threads,
                fusion_model=fusion_model_enum,
                threshold=threshold,
            )
            logging.info("Fusion process completed successfully.")
        except Exception as e:
            logging.error(f"Fusion process failed: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", str(e))
            raise

        # --- Add Fused Images to DataFrame ---
        try:
            logging.info("Adding fused images to dataframe...")
            dataset_name = dataset_dataframe_path.stem.replace("_dataset_dataframe", "")
            updated_dataframe_path = dataset_dataframe_path.with_name(
                f"{dataset_name}_dataset_dataframe_with_fused.parquet"
            )
            fused_images_dir = Path(output_path_pattern).parent

            add_fused_images_to_dataframe_logic(
                input_parquet_path=dataset_dataframe_path,
                output_parquet_path=updated_dataframe_path,
                fused_images_dir=fused_images_dir,
                dataset_name=dataset_name,
            )
            logging.info(f"Saved updated dataframe to: {updated_dataframe_path}")
            mlflow.log_artifact(str(updated_dataframe_path), "dataframes")

        except Exception as e:
            logging.error(f"Adding fused images to dataframe failed: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", str(e))
            raise

        # --- Run Evaluation ---
        try:
            logging.info("Starting evaluation...")
            eval_results = run_evaluation(
                dataset_dataframe_path=updated_dataframe_path,  # Use the new dataframe
                competitor="fused_images",  # Evaluate the new column
                output=evaluation_output,
            )
            logging.info("Evaluation completed successfully.")

            # --- Log Metrics ---
            if eval_results and eval_results.get("overall_averages"):
                overall_averages = eval_results["overall_averages"]
                mlflow.log_metrics(
                    {
                        sanitize_metric_name(f"overall_avg_{k}"): v
                        for k, v in overall_averages.items()
                    }
                )
                logging.info(f"Logged overall average metrics: {overall_averages}")

            if eval_results and eval_results.get("per_campaign_averages"):
                per_campaign_averages = eval_results["per_campaign_averages"]
                for comp, camp_avgs in per_campaign_averages.items():
                    for camp, avg in camp_avgs.items():
                        mlflow.log_metric(
                            sanitize_metric_name(f"campaign_avg_{comp}_{camp}"), avg
                        )
                logging.info("Logged per-campaign average metrics.")

            # --- Log Artifacts ---
            if eval_results and eval_results.get("all_results"):
                results_path = Path(mlflow.get_artifact_uri()) / "detailed_results.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(eval_results["all_results"], f, indent=4)
                mlflow.log_artifact(str(results_path))
                logging.info("Logged detailed results as JSON artifact.")

            if evaluation_output and evaluation_output.exists():
                mlflow.log_artifact(str(evaluation_output), "evaluation_outputs")
                logging.info(f"Logged evaluation output CSV: {evaluation_output}")

        except Exception as e:
            logging.error(f"Evaluation process failed: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", str(e))
            raise

        logging.info(f"MLflow run {run_id} finished successfully.")


@cli.command()
@click.option(
    "--experiment-name",
    default="baseline-evaluations",
    help="Name for the MLflow experiment.",
)
@click.option(
    "--dataset-dataframe-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the dataset dataframe for evaluation.",
)
@click.option(
    "--evaluation-output",
    type=click.Path(path_type=Path),
    help="Path to save evaluation results as CSV.",
)
def run_baseline_evaluation(
    experiment_name: str,
    dataset_dataframe_path: Path,
    evaluation_output: Optional[Path],
):
    """Runs a baseline evaluation for all competitors and logs it to MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"Starting MLflow run: {run_id}")

        # --- Log Git Commit ---
        commit_hash = get_git_commit_hash()
        if commit_hash:
            mlflow.set_tag("git_commit", commit_hash)
            logging.info(f"Logged Git commit: {commit_hash}")

        # --- Log DVC Hash ---
        dvc_hash = get_dvc_hash(Path("data.dvc"))
        if dvc_hash:
            mlflow.set_tag("dvc_data_hash", dvc_hash)
            logging.info(f"Logged DVC data hash: {dvc_hash}")

        # --- Log Parameters ---
        params = {
            "dataset_dataframe_path": str(dataset_dataframe_path),
        }
        mlflow.log_params(params)
        logging.info(f"Logged parameters: {params}")

        # --- Log DataFrame as Artifact ---
        mlflow.log_artifact(str(dataset_dataframe_path), "dataframes")

        # --- Run Evaluation ---
        try:
            logging.info("Starting evaluation for all competitors...")
            eval_results = run_evaluation(
                dataset_dataframe_path=dataset_dataframe_path,
                output=evaluation_output,
            )
            logging.info("Evaluation completed successfully.")

            # --- Log Metrics ---
            if eval_results and eval_results.get("overall_averages"):
                overall_averages = eval_results["overall_averages"]
                mlflow.log_metrics(
                    {
                        sanitize_metric_name(f"overall_avg_{k}"): v
                        for k, v in overall_averages.items()
                    }
                )
                logging.info(f"Logged overall average metrics: {overall_averages}")

            if eval_results and eval_results.get("per_campaign_averages"):
                per_campaign_averages = eval_results["per_campaign_averages"]
                for comp, camp_avgs in per_campaign_averages.items():
                    for camp, avg in camp_avgs.items():
                        mlflow.log_metric(
                            sanitize_metric_name(f"campaign_avg_{comp}_{camp}"), avg
                        )
                logging.info("Logged per-campaign average metrics.")

            # --- Log Artifacts ---
            if eval_results and eval_results.get("all_results"):
                results_path = Path(mlflow.get_artifact_uri()) / "detailed_results.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(eval_results["all_results"], f, indent=4)
                mlflow.log_artifact(str(results_path))
                logging.info("Logged detailed results as JSON artifact.")

            if evaluation_output and evaluation_output.exists():
                mlflow.log_artifact(str(evaluation_output), "evaluation_outputs")
                logging.info(f"Logged evaluation output CSV: {evaluation_output}")

        except Exception as e:
            logging.error(f"Evaluation process failed: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", str(e))
            raise

        logging.info(f"MLflow run {run_id} finished successfully.")


if __name__ == "__main__":
    cli()
