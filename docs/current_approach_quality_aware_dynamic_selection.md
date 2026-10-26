# Current Approach: Quality-Aware Dynamic Candidate Selection

## Purpose

This document summarizes the current approach in enough detail for another
agent to rewrite it into a dissertation or book-style chapter. It focuses on
the conceptual method, how it differs from the original QA/ENSEMBLE idea, what
the current implementation does, and which results should be treated as the
current state of evidence.

## Short Summary

The current method is best described as:

```text
quality-aware dynamic candidate selection for silver-truth segmentation
```

It uses multiple candidate segmentation masks for the same object, synchronizes
them into local object clusters, scores each candidate using QA-like and
contextual evidence, selects the most reliable candidates, and reconstructs a
final segmentation mask.

The method evolved from the original QA-gated ENSEMBLE idea. The original idea
was to predict segmentation quality, filter low-quality candidates, and feed
the remaining candidates into an ENSEMBLE model. The current evidence suggests
that the most robust decision is not a hard QA threshold or a direct neural
mask-generating ENSEMBLE. It is a local candidate-ranking problem.

## Dissertation Framing

The dissertation proposal is about improving segmentation using ensemble
methods with quality assurance. It highlights a limitation of Cell Tracking
Challenge silver-truth generation: candidate methods are filtered mostly at the
model or dataset level, but individual object instances can vary strongly in
quality.

The current method addresses exactly that limitation:

- it operates at object-instance level;
- it uses quality-related evidence to select or prune candidates;
- it works with multiple segmentation sources;
- it can be used before or instead of fusion;
- it targets improved silver-truth masks.

The safest dissertation wording is:

> The original QA-gated ENSEMBLE architecture was reformulated as
> quality-aware dynamic candidate selection. Instead of relying on one global QA
> threshold or a fixed-input neural mask generator, the system ranks candidate
> masks inside synchronized object clusters using QA, source reliability,
> temporal context, geometry, and image-support features.

This is still aligned with QA-driven dynamic ensemble pruning. It should not be
presented as unrelated feature engineering.

## Reporting Taxonomy

Use this terminology consistently:

- **FUSION is not ENSEMBLE.**
- **FUSION is not our method.**
- **ENSEMBLE is our method.**
- `fusion_only_*` rows are external fusion baselines.
- `ensemble_only` and `ensemble_qa_*` rows are our ENSEMBLE methods.
- `qa_only_*` rows are QA-selection ablations.
- `full_pipeline_*` rows are deployable pipeline rows that apply our
  QA/filtering logic to a fusion proposal set.
- The current soft-hybrid / ranker rows are deployable quality-aware candidate
  selection rows, not strict QA/ENSEMBLE rows.

Use separate tables for:

1. strict QA/ENSEMBLE methods;
2. deployable quality-aware candidate selection;
3. external fusion baselines;
4. external competitors and silver truth.

## Method Data Flow

The current pipeline can be described as:

```text
paper-ready candidate table
-> candidate feature extraction
-> optional candidate/pairwise ranker
-> soft-hybrid scoring
-> local conflict resolution and candidate selection
-> reconstruction of selected candidates into segmentation masks
-> validation-based strategy selection
-> reporting against fusion, competitors, and silver truth
```

The important unit is the synchronized object cluster: several candidate masks
from different sources may correspond to the same real cell. The task is to
decide which candidate or subset of candidates should represent that cell.

## Implementation Map

Main scripts:

| File | Role |
| --- | --- |
| `scripts/tune_soft_hybrid_grid.py` | Builds feature tables, trains rankers, runs preset grid, reconstructs masks, writes `soft_hybrid_grid_summary.csv`. |
| `scripts/select_soft_hybrid_strategy.py` | Selects strategies by validation IoU with conservative preset tie-breaking. |
| `scripts/run_soft_hybrid_batch.py` | Batch runner for soft-hybrid experiments. |
| `scripts/build_reporting_mode_tables.py` | Builds reporting-mode tables combining strict QA/ENSEMBLE, deployable rows, and soft-hybrid rows. |

Main modules:

| File | Role |
| --- | --- |
| `src/silver_truth/evaluation/candidate_features.py` | Builds candidate-level feature tables from `paper_ready.parquet` and image data. |
| `src/silver_truth/evaluation/candidate_ranker.py` | Candidate ranker and ranker feature column definitions. |
| `src/silver_truth/evaluation/cluster_pairwise_ranker.py` | Pairwise ranker inside local candidate clusters. |
| `src/silver_truth/evaluation/soft_hybrid.py` | Soft-hybrid scoring and candidate selection. |
| `src/silver_truth/evaluation/soft_hybrid_reconstruction.py` | Reconstructs selected candidates into evaluated masks. |
| `src/silver_truth/evaluation/soft_hybrid_strategy_selection.py` | Validation-based selection across strategies and presets. |
| `src/silver_truth/evaluation/reporting_mode_selector.py` | Combines strict, deployable, and soft-hybrid modes for reporting. |

Important tests:

| Test file | Coverage |
| --- | --- |
| `tests/evaluation/test_candidate_features.py` | Feature extraction. |
| `tests/evaluation/test_candidate_ranker.py` | Candidate ranker behavior. |
| `tests/evaluation/test_cluster_pairwise_ranker.py` | Pairwise ranker behavior. |
| `tests/evaluation/test_soft_hybrid.py` | Soft-hybrid scoring and selection. |
| `tests/evaluation/test_soft_hybrid_reconstruction.py` | Reconstruction from selected candidates. |
| `tests/evaluation/test_soft_hybrid_strategy_selection.py` | Validation-based strategy selection. |
| `tests/evaluation/test_reporting_mode_selector.py` | Reporting mode selection and taxonomy. |

## Feature Families

The current approach combines several feature families.

| Feature family | Purpose |
| --- | --- |
| QA score | Estimate candidate mask reliability. |
| Fusion/support evidence | Capture local agreement among candidate sources. |
| Source identity / source prior | Model source reliability differences. |
| Temporal support | Capture frame-to-frame plausibility. |
| Morphology / geometry | Capture shape, size, position, border contact, and crop geometry. |
| Crop image features | Use raw image evidence inside stacked square crops. |
| Rectangular image features | Use tighter object-local image/mask statistics. |
| Full-image features | Measure object evidence in true image coordinates. |
| Boundary/noise features | Capture contrast-to-noise, boundary gradients, holes, eccentricity. |
| Local image support score | Cluster-relative image evidence used as a soft-hybrid support term. |

Important `tune_soft_hybrid_grid.py` flags:

```text
--use-pairwise-cluster-ranker
--use-source-identity
--ranker-train-split train
--include-image-features
--include-rect-image-features
--include-full-image-features
--include-full-image-boundary-noise-features
--include-crop-geometry-features
--use-local-image-support-score
--local-image-support-mode average
--local-image-support-mode boundary_mean
--local-image-support-mode boundary_min
--local-image-support-mode full_boundary
--local-image-support-mode rect_boundary
```

The strongest recent selected setup uses square, rectangular, and full-image
features with pairwise source-aware ranking. The latest support-mode selector
adds a tiny improvement by selecting a full-boundary support variant for one
MSC fold.

## Current Result Paths

Strict and full paper summaries:

```text
hpc_results/combined/reports/paper_result_summary_all_available.csv
hpc_results/combined/reports/paper_result_summary_all_available_qa_ensemble_only.csv
```

Current selected full-image / geometry soft-hybrid result:

```text
hpc_results/soft_hybrid_probe/all_available_full_image_vs_geometry_combined_validation_selection/
```

Important files inside it:

```text
validation_selected_strategy.csv
validation_selected_dataset_summary.csv
validation_vs_fusion_simple_summary.csv
```

Current support-mode reporting selection:

```text
hpc_results/soft_hybrid_probe/reporting_mode_selection_full_image_vs_local_support_modes_combined/
```

Important files inside it:

```text
validation_selected_reporting_mode.csv
validation_selected_reporting_mode.md
reporting_mode_candidates.csv
```

## Current Numbers

Against strict original QA/ENSEMBLE methods:

| Dataset | Strict original QA/ENSEMBLE foldwise best | Strict original QA/ENSEMBLE validation-selected | Current soft-hybrid selected | Change vs foldwise best | Change vs validation-selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF-C2DL-HSC | 0.8658 | 0.8387 | 0.8689 | +0.0030 | +0.0302 |
| BF-C2DL-MuSC | 0.7896 | 0.7638 | 0.7871 | -0.0024 | +0.0234 |
| DIC-C2DH-HeLa | 0.8701 | 0.8701 | 0.9598 | +0.0896 | +0.0896 |
| Fluo-C2DL-MSC | 0.6154 | 0.6154 | 0.7242 | +0.1088 | +0.1088 |
| Fluo-N2DH-GOWT1 | 0.9279 | 0.9279 | 0.9496 | +0.0217 | +0.0217 |
| **Mean** | **0.8138** | **0.8032** | **0.8579** | **+0.0441** | **+0.0547** |

Against best `fusion_only_*` baselines:

| Dataset | Best `fusion_only_*` IoU | Current soft-hybrid IoU | Delta |
| --- | ---: | ---: | ---: |
| BF-C2DL-HSC | 0.8726 | 0.8689 | -0.0037 |
| BF-C2DL-MuSC | 0.7307 | 0.7871 | +0.0564 |
| DIC-C2DH-HeLa | 0.9307 | 0.9598 | +0.0291 |
| Fluo-C2DL-MSC | 0.7463 | 0.7242 | -0.0221 |
| Fluo-N2DH-GOWT1 | 0.9350 | 0.9496 | +0.0146 |
| **Mean** | **0.8431** | **0.8579** | **+0.0149** |

Against broader context:

| Comparison | Mean IoU |
| --- | ---: |
| Current soft-hybrid | 0.8579 |
| Current support-mode selected variant | 0.8581 |
| Best `fusion_only_*` baseline | 0.8431 |
| Best competitor | 0.8712 |
| Silver truth | 0.8819 |

Interpretation:

- The current method beats fusion-only on average.
- It wins against fusion-only on three of five datasets.
- It loses to fusion-only on HSC and MSC.
- It is still below the best external competitor average.
- The support-mode improvement is real but very small: about `+0.000165` mean
  IoU over the current selected full-image baseline.

## Known Failure Modes

The current method is not yet a clean replacement for fusion.

Main regressions:

| Dataset | Issue |
| --- | --- |
| BF-C2DL-HSC | Small regression versus fusion-only; likely strategy selection or conservative tie-breaking issue. |
| Fluo-C2DL-MSC | Large regression versus fusion-only and best competitor; highest-priority failure case. |

MSC is the main blocker because:

- current method loses to fusion-only by about `-0.0221` IoU;
- current method loses to the best competitor by about `-0.0505` IoU;
- image features helped MSC substantially in earlier probes;
- full-boundary/local-support features helped one MSC fold slightly;
- the remaining gap may be reconstruction/fusion-related rather than purely
  candidate-ranking-related.

Important diagnostic question:

> Is fusion better on MSC because it selects better object candidates, or
> because averaging/voting boundaries produces a better final mask than direct
> candidate selection?

That question motivates the next major experiment: run fusion after QA/ranker
pruning.

## Why This Is Not Just Old-School Feature Engineering

The current approach uses engineered features, but those features are serving a
scientific purpose. They expose which signals matter before building a higher
capacity neural system.

The earlier neural ENSEMBLE approach may have struggled because:

- the input crops were too large and background-heavy;
- the true problem is a variable-size set problem;
- a fixed stacked-mask U-Net is not naturally source-aware or
  permutation-invariant;
- the objective should often be ranking/pruning, not full mask generation;
- there may be too little cross-dataset data for a high-capacity model;
- temporal context was not central enough in the neural input;
- source identity and source reliability were not represented explicitly enough.

The current feature/ranker system should therefore be treated as a blueprint
for the next neural model.

## Neural Direction To Preserve

The likely next neural direction is:

```text
object-local candidate encoder
+ source identity / source embedding
+ temporal candidate context
+ permutation-aware set aggregation
+ pairwise or listwise ranking objective
+ optional fusion after neural QA pruning
```

This is more promising than a generic stacked-mask U-Net because it matches the
actual decision problem:

- candidates are variable in number;
- source order should not matter;
- local candidate comparison is more stable than global QA regression;
- the final output can still be produced by classical fusion after neural
  pruning.

This neural model should come after the QA-pruned fusion and ablation work,
because those experiments will reveal which signals are worth learning.

## Chapter-Level Narrative

A dissertation/book chapter can use this structure:

1. **Problem.** CTC silver-truth generation relies on model-level filtering and
   fusion, but object-level candidate quality varies.
2. **Original hypothesis.** QA can identify unreliable masks and improve
   ENSEMBLE/fusion.
3. **Observed limitation.** Hard QA thresholds and first neural ENSEMBLE
   variants were brittle under dataset/fold transfer.
4. **Reformulation.** Treat the task as quality-aware dynamic candidate
   selection inside synchronized object clusters.
5. **Method.** Extract QA, source, temporal, geometry, and image-support
   features; train a pairwise/source-aware ranker; select candidates; reconstruct
   masks.
6. **Results.** Strong improvement over strict QA/ENSEMBLE baselines and mean
   improvement over fusion-only, but still dataset-specific failures.
7. **Analysis.** MSC and HSC regressions show why direct candidate selection is
   not enough; QA-pruned fusion is the next decisive test.
8. **Future architecture.** Use the engineered ranker as a guide for a neural
   candidate-ranking model rather than returning to a generic mask generator.

## Related Handoff Documents

Continue from these documents:

```text
docs/next_steps_quality_aware_dynamic_selection.md
docs/dissertation_alignment_and_architecture_evolution.md
docs/next_agent_rectangular_image_feature_plan.md
docs/fusion_qa_soft_hybrid_and_ranker_plan.md
```
