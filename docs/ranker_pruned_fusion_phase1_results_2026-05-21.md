# Ranker-Pruned Fusion Phase 1 Results

Date: 2026-05-21

## Scope

This note summarizes the first all-available run of QA/ranker-pruned fusion.
The tested deployable pipeline rows use:

```text
full_pipeline__qa_pruned_simple__{policy}
```

These are not ENSEMBLE rows. They are full-pipeline rows that apply QA/ranker
pruning to the fusion proposal set and then run SIMPLE fusion.

## Inputs

Ranked candidate tables:

```text
hpc_results/soft_hybrid_probe/all_available_square_rect_full_image_local_support_pairwise_source_trainonly/features/
```

Paper-ready candidate inputs were resolved from each feature table's
`source_path`, which points to the original fusion-ready parquet.

## Outputs

Main aggregate artifacts:

```text
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_simple_policy_comparison_rows.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_simple_policy_dataset_summary.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_simple_policy_overall_summary.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_simple_policy_best_by_dataset.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_simple_policy_context_table.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_simple_policy_context_dataset_selected.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_simple_policy_overall_context_summary.csv
```

Top2-only aggregate artifacts from the first pass:

```text
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_top2_simple_comparison_rows.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_top2_simple_dataset_summary.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_top2_simple_pruning_summary.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_top2_simple_pruning_dataset_summary.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_top2_simple_context_table.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_top2_simple_context_dataset_selected.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/all_available_top2_simple_overall_context_summary.csv
```

## Fixed-Policy Means

Mean across dataset means:

| Policy | Backend | Mean IoU | Mean F1 | Mean retained candidates |
| --- | --- | ---: | ---: | ---: |
| `source_balanced_topk` | SIMPLE | 0.8654 | 0.9232 | 42.05% |
| `top1_per_cluster` | SIMPLE | 0.8582 | 0.9160 | 14.02% |
| `margin_pruned` | SIMPLE | 0.8561 | 0.9148 | 17.17% |
| `fusion_safe_pruning` | SIMPLE | 0.8556 | 0.9152 | 24.29% |
| `top2_per_cluster` | SIMPLE | 0.8542 | 0.9159 | 28.04% |

## Best Policy By Dataset

| Dataset | Best policy | IoU | F1 | Retained candidates |
| --- | --- | ---: | ---: | ---: |
| BF-C2DL-HSC | `source_balanced_topk` | 0.8712 | 0.9301 | 61.44% |
| BF-C2DL-MuSC | `source_balanced_topk` | 0.8066 | 0.8891 | 75.56% |
| DIC-C2DH-HeLa | `top1_per_cluster` | 0.9598 | 0.9793 | 9.16% |
| Fluo-C2DL-MSC | `top2_per_cluster` | 0.7503 | 0.8480 | 17.57% |
| Fluo-N2DH-GOWT1 | `top1_per_cluster` | 0.9498 | 0.9731 | 6.46% |

## Top2 Dataset Means

| Dataset | QA-pruned SIMPLE IoU | QA-pruned SIMPLE F1 | Retained candidates |
| --- | ---: | ---: | ---: |
| BF-C2DL-HSC | 0.8460 | 0.9151 | 40.96% |
| BF-C2DL-MuSC | 0.7813 | 0.8720 | 50.44% |
| DIC-C2DH-HeLa | 0.9511 | 0.9746 | 18.32% |
| Fluo-C2DL-MSC | 0.7503 | 0.8480 | 17.57% |
| Fluo-N2DH-GOWT1 | 0.9425 | 0.9696 | 12.93% |

## Context Mean

Mean across dataset means using the best QA-pruned SIMPLE policy per dataset:

| Row family | Label | Mean IoU |
| --- | --- | ---: |
| `silver_truth` | Silver truth | 0.8819 |
| `competitor` | Best competitor | 0.8712 |
| `full_pipeline_best_policy` | Best QA-pruned SIMPLE policy | 0.8675 |
| `reporting_mode_selected` | Validation-selected reporting mode | 0.8496 |
| `fusion_only_best` | Best fusion-only baseline | 0.8434 |

For a single fixed pruning policy, `source_balanced_topk` is the strongest
setting with mean IoU 0.8654.

## Interpretation

The first full pass supports the Phase 1 hypothesis: object-level QA/ranker
pruning before SIMPLE fusion improves over the best `fusion_only_*` baseline on
average. A single fixed policy, `source_balanced_topk`, improves the mean IoU
from 0.8434 to 0.8654. Selecting the best QA-pruned SIMPLE policy per dataset
reaches 0.8675, close to but still below the best external competitor average
of 0.8712 and official silver truth average of 0.8819.

Dataset-level behavior is policy-dependent. `source_balanced_topk` is best for
HSC and MuSC, `top1_per_cluster` is best for HeLa and GOWT1, and
`top2_per_cluster` remains best for MSC.

## Implementation Notes

Java/ImageJ fusion must be run outside the sandbox in this environment. The job
file format and rank-slot input layout are supported by `label-fusion-ng`; the
in-sandbox failure occurs while the Java TIFF reader opens staged images.

## Phase 2B Non-SIMPLE Backend Sweep

Date: 2026-05-22

The best SIMPLE pruning policy for each dataset was re-run with the currently
supported non-SIMPLE backends:

```text
MAJORITY_FLAT
THRESHOLD_FLAT
BIC_FLAT_VOTING
THRESHOLD_USER
BIC_WEIGHTED_VOTING
```

Weighted backends used `pairwise_cluster_score` as the candidate weight column.

New artifacts:

```text
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/non_simple_best_policy_backend_sweep/non_simple_best_policy_comparison_rows.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/non_simple_best_policy_backend_sweep/non_simple_best_policy_dataset_summary.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/non_simple_best_policy_backend_sweep/non_simple_best_policy_best_by_dataset.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/non_simple_best_policy_backend_sweep/non_simple_best_policy_overall_summary.csv
hpc_results/soft_hybrid_probe/ranker_pruned_fusion_probe/non_simple_best_policy_backend_sweep/non_simple_best_policy_context_table.csv
```

Mean across dataset means:

| Selection | Mean IoU | Mean F1 |
| --- | ---: | ---: |
| Best non-SIMPLE backend by dataset | 0.8668 | 0.9237 |
| Fixed `BIC_WEIGHTED_VOTING` | 0.8668 | 0.9237 |
| Fixed `MAJORITY_FLAT` | 0.8668 | 0.9237 |
| Fixed `BIC_FLAT_VOTING` | 0.8423 | 0.9062 |
| Fixed `THRESHOLD_FLAT` | 0.8423 | 0.9062 |
| Fixed `THRESHOLD_USER` | 0.4849 | 0.5332 |
| Best QA-pruned SIMPLE policy | 0.8675 | 0.9239 |

Best non-SIMPLE backend by dataset:

| Dataset | Policy | Backend | IoU | Delta vs best SIMPLE policy |
| --- | --- | --- | ---: | ---: |
| BF-C2DL-HSC | `source_balanced_topk` | `BIC_WEIGHTED_VOTING` | 0.8712 | +0.0000 |
| BF-C2DL-MuSC | `source_balanced_topk` | `BIC_WEIGHTED_VOTING` | 0.8095 | +0.0029 |
| DIC-C2DH-HeLa | `top1_per_cluster` | `BIC_FLAT_VOTING` | 0.9598 | +0.0000 |
| Fluo-C2DL-MSC | `top2_per_cluster` | `BIC_WEIGHTED_VOTING` | 0.7438 | -0.0065 |
| Fluo-N2DH-GOWT1 | `top1_per_cluster` | `BIC_FLAT_VOTING` | 0.9498 | +0.0000 |

Interpretation:

- Non-SIMPLE backends do not improve the best QA-pruned SIMPLE result overall.
- `BIC_WEIGHTED_VOTING` and `MAJORITY_FLAT` tie in this sweep and are slightly
  below best-policy SIMPLE on average.
- MuSC gains slightly from weighted/BIC or majority voting, but MSC regresses.
- `THRESHOLD_USER` produced zero IoU on the single-slot `top1_per_cluster`
  datasets, so do not use it as evidence against QA pruning until that backend
  behavior is diagnosed.
