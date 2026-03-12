from pathlib import Path
from silver_truth.experiment_tracking import DEFAULT_MLFLOW_TRACKING_URI

# Resolve to absolute path so MLflow never falls back to experiment ID 0
# regardless of the working directory the process was launched from.
_raw = DEFAULT_MLFLOW_TRACKING_URI
if _raw.startswith("file:"):
    mlflow_mlruns_path = _raw
else:
    mlflow_mlruns_path = Path(_raw).resolve().as_uri()
