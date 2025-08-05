import click
import mlflow
from pathlib import Path

from src.fusion.fusion import FusionModel, fuse_segmentations
from src.metrics.evaluation_logic import run_evaluation


@click.command()
@click.option(
    "--experiment-name",
    default="silver-truth-qa",
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
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the job specification file.",
)
@click.option(
    "--output-pattern",
    required=True,
    help="Output filename pattern (e.g., '/path/to/fused_TTTT.tif').",
)
@click.option(
    "--model", required=True, help="The fusion model to use (e.g., 'LOG_AND')."
)
@click.option(
    "--threshold", default=1.0, type=float, help="Voting threshold for merging."
)
@click.option(
    "--num-threads", default=4, type=int, help="Number of processing threads."
)
@click.option(
    "--dataset-dataframe-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the dataset dataframe Parquet file for evaluation.",
)
@click.option(
    "--evaluation-output",
    type=click.Path(path_type=Path),
    help="Path to save evaluation results as CSV",
)
def run_experiment_and_evaluate(
    experiment_name,
    jar_path,
    job_file,
    output_pattern,
    model,
    threshold,
    num_threads,
    dataset_dataframe_path,
    evaluation_output,
):
    """
    Main entry point for running a silver-truth QA experiment and evaluating the results.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run started with ID: {run_id}")

        # --- 1. Log Parameters ---
        params = {
            "jar_path": jar_path,
            "job_file": job_file,
            "output_pattern": output_pattern,
            "model": model,
            "threshold": threshold,
            "num_threads": num_threads,
            "dataset_dataframe_path": str(dataset_dataframe_path),
        }
        mlflow.log_params(params)
        print(f"Logged parameters: {params}")

        # --- 2. Execute Fusion Pipeline ---
        print("Starting fusion process...")
        try:
            fusion_model_enum = FusionModel[model.upper()]
            fuse_segmentations(
                jar_path=jar_path,
                job_file_path=job_file,
                output_path_pattern=output_pattern,
                time_points="0-90000",  # Assuming a default, this should also be a parameter
                num_threads=num_threads,
                fusion_model=fusion_model_enum,
                threshold=threshold,
                debug=True,
            )
            print("Fusion process completed successfully.")
            mlflow.set_tag("fusion_status", "success")
        except Exception as e:
            print(f"Error during fusion process: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("fusion_status", "failed")
            return

        # --- 3. Run Evaluation ---
        print("Starting evaluation...")
        try:
            results = run_evaluation(
                dataset_dataframe_path=dataset_dataframe_path,
                output=evaluation_output,
            )
            if results:
                print("Evaluation completed successfully.")
                mlflow.set_tag("evaluation_status", "success")

                # --- 4. Log Metrics ---
                print("Logging evaluation metrics to MLflow...")
                for competitor, avg_score in results.get(
                    "overall_averages", {}
                ).items():
                    mlflow.log_metric(f"{competitor}_overall_jaccard", avg_score)

                for competitor, campaign_scores in results.get(
                    "per_campaign_averages", {}
                ).items():
                    for campaign, score in campaign_scores.items():
                        mlflow.log_metric(f"{competitor}_{campaign}_jaccard", score)
                print("Metrics logged successfully.")
            else:
                print("Evaluation produced no results.")
                mlflow.set_tag("evaluation_status", "no_results")

        except Exception as e:
            print(f"Error during evaluation: {e}")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("evaluation_status", "failed")
            return

        print("Experiment run finished successfully.")


if __name__ == "__main__":
    run_experiment_and_evaluate()
