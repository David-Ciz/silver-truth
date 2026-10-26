# Cell manuscript: pretrained Deep Sets weighting models

Ten models used for the 0.867563 mean cell-crop IoU result in *Quality Aware Candidate Selection for Silver Truth in Cell Segmentation*.

Each filename identifies the dataset and evaluation fold. Fold 1 evaluates video 02; fold 2 evaluates video 01. Each model was fitted on pooled train rows across five datasets, excluding every row from the evaluated dataset-video pair. Model settings: 80 epochs, hidden dimension 64, seed 13, target temperature 0.08, cluster-softmax temperature 1, boundary support floor 0.75, vote threshold 0.48.

The joblib payload contains the network state, feature definitions, fitted numerical preprocessing, categorical vocabulary and training configuration. These are Deep Sets weighting models; ResNet18 quality estimators and pairwise-ranker checkpoints are not included. Competitor masks and private training/test data are not included.

## Score candidate features

After installing the repository dependencies, load a prepared candidate-feature table with the same feature schema and run:

```python
from pathlib import Path
import pandas as pd
from silver_truth.evaluation.candidate_set_weight import score_with_candidate_set_weight_predictor

features = pd.read_parquet("prepared_soft_gate_candidates.parquet")
scored = score_with_candidate_set_weight_predictor(
    features,
    model_payload=Path("pretrained/cell_deep_sets/BF-C2DL-MuSC_fold-1_leave_target_video_out.joblib"),
)
```

Candidate preparation, training and voting entry point: `scripts/qa_pruned_ensemble/run_pruned_candidate_set_weight.py`. Its `--help` lists the feature-table and model settings. Manuscript crop/full-image evaluation is preserved in `scripts/paper/`.

`reference_fold_scores.csv` contains the manuscript crop and reconstruction results. The ten saved models were re-scored on 26 October 2026 and reproduce their archived raw predictions; all ten crop IoU results were reproduced using the manuscript evaluator.
