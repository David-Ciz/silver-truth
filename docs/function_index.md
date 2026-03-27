# Function Index

This page is a quick map to core functions and CLI commands used for paper experiments.

For a broader "what can I run?" view, including maintained scripts and agent-oriented task lookup, see [Executable Index](executable_index.md).

## Protocol First

- [Paper Protocol (Fold-Locked)](paper_protocol.md)
  - purpose: authoritative order for fold-specific DVC prep and runner-driven Phase A/B/C execution.
- [Ablation Experiment Plan](ablation_experiment_plan.md)
  - purpose: per-experiment breakdown of CLI commands, inputs/outputs, and the maintained manual equivalents behind the runner.

## Paper Experiments

The maintained entry point is `scripts/run_ablation.py`, backed by `experiments/base.yaml` and `experiments/variants/*.yaml`.
The runner calls the underlying CLI commands, so each step is still independently runnable when needed.

Key steps in order:
1. DVC prep (whole-image + QA crop parquets + job files)
2. `python scripts/run_ablation.py --phase A` — competitor, fusion, and ensemble baselines
3. `python scripts/run_ablation.py --phase B` — QA training, QA evaluation, and prediction merge
4. `python scripts/run_ablation.py --phase C` — QA-filtered ablation and threshold sweep
5. Optional manual postprocessing scripts/tables after the fold runs complete

## QA Training + Evaluation

- `silver-qa cnn train`
  - source: `silver_truth.qa.cnn`
  - purpose: train QA ResNet50 on fold crop parquet; outputs `.pt` model + Excel predictions.
- `silver-evaluation evaluate-qa-model`
  - source: `silver_truth.metrics.qa_model_evaluation.evaluate_qa_model_from_excel`
  - purpose: regression metrics (MAE, RMSE, R², Pearson r, Spearman rho) from QA Excel.
- `silver-evaluation evaluate-qa-filtering`
  - source: `silver_truth.qa.filtering_evaluation.run_qa_filtering_evaluation`
  - purpose: thresholded QA confusion metrics (precision/recall/F1 at each threshold).
- `silver-evaluation merge-qa-predictions`
  - source: `silver_truth.metrics.qa_model_evaluation.merge_predictions_to_parquet`
  - purpose: add `predicted_jaccard_index` column to a crop parquet from a QA Excel file.

## Preprocessing Helpers

- `silver-preprocessing verify-folder-synchronization`
  - source: `silver_truth.data_processing.label_synchronizer.verify_folder_synchronization_logic`
  - purpose: verify that a segmentation folder and TRA folder are label-synchronized.
- `silver-preprocessing verify-dataset-synchronization`
  - source: `silver_truth.data_processing.label_synchronizer.verify_dataset_synchronization_logic`
  - purpose: run synchronization checks across a synchronized dataset layout.
- `silver-preprocessing segmentation-size-stats`
  - source: `silver_truth.data_processing.segmentation_stats`
  - purpose: report per-cell area, bounding-box size, square/rectangular crop-fit rates, aspect-ratio summaries, and optional clipped-cell audits for candidate crop sizes such as `64`, `96`, `128`, `256`, or `512`.

## Crop-Size Decision Path

- Task: find GT cell sizes across the full MuSC dataset and decide which crop sizes are actually safe.
- Preferred command: `silver-preprocessing segmentation-size-stats data/dataframes/BF-C2DL-MuSC/whole_image/BF-C2DL-MuSC_split_mixed.parquet --crop-size 64 --crop-size 96 --crop-size 128 --crop-size 256 --crop-size 512 --rect-size 128x256`
- Reusable functions:
  - `silver_truth.data_processing.segmentation_stats.collect_segmentation_object_stats`
  - `silver_truth.data_processing.segmentation_stats.collect_segmentation_object_stats_from_dataframes`
  - `silver_truth.data_processing.segmentation_stats.summarize_segmentation_object_stats`
- Rectangle interpretation:
  - `fit_bbox_rate_rect128x256` is fixed orientation
  - `fit_bbox_rate_rect128x256_swappable` allows either `128x256` or `256x128`
- Optional outlier inspection:
  - add `--show-outliers-for 256 --outlier-output-dir <dir>` to generate parquet-driven CSV/PNG audits for clipped cells
- Legacy alternative to avoid for this task:
  - `silver_truth.ensemble.utils.find_largest_gt_cell_size` from `ztest_commands.py` only reports the single largest cell and is not the maintained dataset-wide analysis path.

## Competitor Baselines

- `silver-evaluation evaluate-competitor`
  - source: `silver_truth.evaluation.evaluation_logic.evaluate_competitor_logic` → `silver_truth.metrics.evaluation_logic.run_evaluation`
  - purpose: full-image, label-by-label, sparse-GT-correct IoU/F1 evaluation on whole-image parquets.
- `silver-evaluation report-overflow-impact`
  - source: `silver_truth.evaluation.reporting.analyze_overflow_impact`
  - purpose: join per-cell result tables with GT bbox overflow flags for a chosen crop size and report whether clipped cells materially change scores.
- `silver_truth.metrics.metrics.calculate_jaccard_scores`
  - purpose: per-label IoU from full GT vs full segmentation image; the inner loop of `run_evaluation`.

## Fusion (JAR)

- `silver-fusion run-fusion-crops`
  - source: `silver_truth.fusion.crops_experiment.run_crops_fusion_experiment`
  - purpose: run Java fusion tool on QA crop parquet for one or more models; evaluates at **crop level** (not paper-ready).
- `silver_truth.fusion.fusion.fuse_segmentations`
  - purpose: direct call to the Java fusion JAR for a given job file and model.

## Full-Image Fusion Evaluation

- `silver-evaluation evaluate-fusion-crops`
  - source: `silver_truth.cli.evaluation.evaluate_fusion_crops`
  - purpose: read fusion output parquet from `run-fusion-crops`, reconstruct per-image label arrays from fused crops at their crop coordinates, and evaluate with sparse-GT full-image scoring.
- `silver-evaluation filter-parquet`
  - source: `silver_truth.cli.evaluation.filter_parquet`
  - purpose: create `qa_only` or thresholded `full_pipeline` parquets before fusion/evaluation.

## Ensemble (Learned Fusion)

- `silver-ensemble build-databank`
  - source: `silver_truth.ensemble.ensemble.build_databank`
  - purpose: prepare ensemble training/eval parquet from QA data.
- `silver-ensemble ensemble-experiment`
  - source: `silver_truth.ensemble.ensemble.run_experiment`
  - purpose: train ensemble models with MLflow logging.
- `silver-ensemble evaluate-checkpoint`
  - source: `silver_truth.ensemble.ensemble.evaluate_checkpoint`
  - purpose: evaluate checkpoints, including full-image reconstruction when metadata exists.

## Documentation Pattern

1. Keep pages like this one handwritten (task-oriented index).
2. Add docstrings on source functions/classes.
3. Use `PYTHONPATH=src .venv/bin/python scripts/generate_api_docs.py` to refresh `docs/api_reference_generated.md`.
