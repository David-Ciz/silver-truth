import mlflow
import src.ensemble.external as ext

# TODO: add to env
_mlflow_mlruns_path = "data/ensemble_data/results/mlruns"


# TODO: it's almost copy&paste from ensemble/ensemble.py (except the return)
def _set_mlflow_experiment(name: str) -> str:
    mlflow.set_tracking_uri(
        _mlflow_mlruns_path
    )  # needs to be set before mlflow.get_experiment_by_name()
    # find mlflow experiment
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(name, _mlflow_mlruns_path)
    else:
        experiment_id = mlflow.set_experiment(name).experiment_id
    return experiment_id


def _evaluate_strategy(result):
    # load parquet file
    df = ext.load_parquet(result["file"])
    df.sort_values(by=["image_path"])
    for index, row in enumerate(df.itertuples()):
        mlflow.log_metric("f1", value=row.f1, step=index)  # type: ignore
        mlflow.log_metric("iou", value=row.iou, step=index)  # type: ignore
    mlflow.log_metric("avg_f1", value=df["f1"].mean())
    mlflow.log_metric("avg_iou", value=df["iou"].mean())


def evaluate_strategies(experiment_name, results):
    experiment_id = _set_mlflow_experiment(experiment_name)
    for result in results:
        try:
            # start mlflow run
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=result["strategy"]
            ) as _mlflow_run:
                # run_id = mlflow_run.info.run_id
                _evaluate_strategy(result)
        except Exception as ex:
            print(f"Error during Ensemble experiment: {ex}")
            mlflow.set_tag("status", "failed")
