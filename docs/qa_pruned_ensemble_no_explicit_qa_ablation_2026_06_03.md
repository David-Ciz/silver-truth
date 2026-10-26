# No-Explicit-QA Deep Sets ENSEMBLE Ablation

Date: 2026-06-03

## Purpose

This ablation tests whether the newest QA-pruned Deep Sets ENSEMBLE benefits from
explicit QA features inside the final candidate-set weight model.

It keeps the same QA-pruned candidate pool and the same object-vote settings as
the current best strict run, but removes the explicit QA feature columns from the
Deep Sets weight model:

- `qa_score`
- `qa_rank_in_image`
- `ranker_cluster_qa_rank`

Because relative candidate-set features are generated from the configured base
features, removing those three columns also removes their candidate-set
rank/centered/gap derivatives. The no-explicit-QA model has 212 numeric features
plus 26 one-hot categorical features, for input dimension 238. The full model has
224 numeric features plus 26 one-hot categorical features, for input dimension
250.

Important limitation: this is not a fully no-QA pipeline. The candidate pool is
still QA-pruned, and `pairwise_cluster_score` may still encode upstream QA-related
information. This isolates explicit QA evidence inside the final Deep Sets
ENSEMBLE.

## Command

```bash
.venv/bin/python scripts/qa_pruned_ensemble/run_pruned_candidate_set_weight.py \
  --model-type deep_sets \
  --candidate-run clipped_score_boundary_floor075_thr050 \
  --run-label-suffix trainonly_ltv_boundary_floor075_e080_no_explicit_qa \
  --train-split train \
  --deep-sets-epochs 80 \
  --weight-transform raw_cluster_softmax \
  --weight-temperature 1.0 \
  --vote-threshold 0.48 \
  --boundary-support-column local_image_support_score \
  --boundary-floor 0.75 \
  --boundary-iterations 1 \
  --exclude-explicit-qa-features \
  --overwrite
```

## Overall Result

| Method | Mean IoU | Mean F1 | Cells |
|---|---:|---:|---:|
| Full explicit-QA Deep Sets ENSEMBLE | 0.867563 | 0.924697 | 1767 |
| No-explicit-QA Deep Sets ENSEMBLE | 0.866866 | 0.924363 | 1767 |
| Delta, no-QA minus full | -0.000697 | -0.000334 | 1767 |

Explicit QA features add only a very small overall gain in this setup.

## Dataset Breakdown

| Dataset | Full IoU | No-explicit-QA IoU | Delta | Full F1 | No-explicit-QA F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| BF-C2DL-HSC | 0.871801 | 0.870762 | -0.001039 | 0.930447 | 0.929831 | -0.000616 |
| BF-C2DL-MuSC | 0.809146 | 0.808597 | -0.000549 | 0.890908 | 0.890382 | -0.000526 |
| DIC-C2DH-HeLa | 0.953121 | 0.952632 | -0.000488 | 0.975707 | 0.975446 | -0.000260 |
| Fluo-C2DL-MSC | 0.756011 | 0.754545 | -0.001466 | 0.853907 | 0.853602 | -0.000305 |
| Fluo-N2DH-GOWT1 | 0.947737 | 0.947792 | +0.000054 | 0.972517 | 0.972555 | +0.000037 |

## Fold Breakdown

| Dataset | Fold | Full IoU | No-explicit-QA IoU | Delta |
|---|---|---:|---:|---:|
| BF-C2DL-HSC | fold-1 | 0.848530 | 0.847399 | -0.001131 |
| BF-C2DL-HSC | fold-2 | 0.895072 | 0.894126 | -0.000947 |
| BF-C2DL-MuSC | fold-1 | 0.783587 | 0.784713 | +0.001126 |
| BF-C2DL-MuSC | fold-2 | 0.834704 | 0.832481 | -0.002223 |
| DIC-C2DH-HeLa | fold-1 | 0.954826 | 0.954826 | +0.000000 |
| DIC-C2DH-HeLa | fold-2 | 0.951416 | 0.950439 | -0.000977 |
| Fluo-C2DL-MSC | fold-1 | 0.773948 | 0.766583 | -0.007364 |
| Fluo-C2DL-MSC | fold-2 | 0.738074 | 0.742507 | +0.004433 |
| Fluo-N2DH-GOWT1 | fold-1 | 0.966460 | 0.966089 | -0.000371 |
| Fluo-N2DH-GOWT1 | fold-2 | 0.929015 | 0.929494 | +0.000479 |

## Interpretation

The explicit QA features are not carrying the final ENSEMBLE. Removing them loses
only 0.000697 mean IoU overall. This supports the interpretation from the real
case study: QA is a noisy, sometimes useful signal, but the stronger parts of the
newest ENSEMBLE are the candidate-set context, source/geometry/support evidence,
and Deep Sets weighting.

This does not mean QA is useless in the whole pipeline. The pool is still
QA-pruned, and some upstream ranking features may still contain QA influence.
What this ablation says is narrower and cleaner: explicit QA columns inside the
final Deep Sets weight model provide only a small marginal gain.

## Artifacts

- Result root:
  `hpc_results/soft_hybrid_probe/qa_pruned_ensemble/learned_candidate_set_weight/deep_sets_ltv_clipped_score_boundary_floor075_thr050_trainonly_ltv_boundary_floor075_e080_no_explicit_qa/`
- Overall summary:
  `qa_pruned_ensemble_candidate_set_weight_overall_summary.csv`
- Dataset summary:
  `qa_pruned_ensemble_candidate_set_weight_dataset_summary.csv`
- Fold rows:
  `qa_pruned_ensemble_candidate_set_weight_comparison_rows.csv`
