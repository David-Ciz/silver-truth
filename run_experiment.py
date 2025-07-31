import click
import mlflow
import subprocess


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
    help="Output filename pattern (e.g., '/path/to/fused_TTT.tif').",
)
@click.option(
    "--model", required=True, help="The fusion model to use (e.g., 'LOG_AND')."
)
@click.option(
    "--threshold", default=1.0, type=float, help="Voting threshold for merging."
)
@click.option(
    "--num-threads", default=8, type=int, help="Number of processing threads."
)
def run_experiment(
    experiment_name, jar_path, job_file, output_pattern, model, threshold, num_threads
):
    """
    Main entry point for running a silver-truth QA experiment.
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
        }
        mlflow.log_params(params)
        print(f"Logged parameters: {params}")

        # --- 2. Execute Pipeline ---
        print("Constructing and executing the fusion command...")

        # Note: This assumes run_fusion.py is executable and in the path.
        # A more robust solution would be to call the Python function directly.
        cmd = [
            "python",
            "run_fusion.py",
            "run-fusion",
            "--jar-path",
            jar_path,
            "--job-file",
            job_file,
            "--output-pattern",
            output_pattern,
            "--model",
            model,
            "--threshold",
            str(threshold),
            "--num-threads",
            str(num_threads),
            "--time-points",
            "0-90000",  # Assuming a default, this should also be a parameter
        ]

        try:
            # Using subprocess.run to execute the command
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Fusion process stdout:", result.stdout)
            print("Fusion process stderr:", result.stderr)
            print("Fusion process completed successfully.")

            # TODO: Add call to evaluation.py here

            # --- 3. Log Results ---
            # Placeholder for actual evaluation metrics
            # In a real scenario, you would parse the output of evaluation.py
            mlflow.log_metric("overall_jaccard", 0.99)  # Placeholder value
            print("Logged placeholder metric.")

        except subprocess.CalledProcessError as e:
            print(f"Error during fusion process: {e}")
            print("Stderr:", e.stderr)
            mlflow.set_tag("status", "failed")
            return

        print("Experiment run finished successfully.")


if __name__ == "__main__":
    run_experiment()
