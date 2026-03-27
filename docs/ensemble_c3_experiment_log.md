# Ensemble C3 Experiment Log
_Date: 2026-03-27_

## Summary

This log records the `C3` ensemble-input experiment introduced on the
`david/ensemble-improvements` branch.

The result is negative:

- `C3` did **not** improve the ensemble on `BF-C2DL-HSC`.
- `C3` did **not** improve the ensemble on `BF-C2DL-MuSC`.
- The experiment should be treated as a failed input-formulation branch, not as a
  new default.

## Idea

The baseline ensemble input (`C1`) uses a single channel:

- normalized overlap of all competitor masks

The `C3` variant was intended to preserve more structure from the set of
competitor masks without returning to raw-image inputs.

`C3` input channels:

1. normalized overlap / consensus map
2. union mask of all competitor segmentations
3. disagreement map derived from the consensus level

Motivation:

- `C1` may collapse too much information into one channel
- `C2` (overlap + raw image) had already performed poorly
- a mask-only richer representation was a cleaner next test than more QA
  thresholding

## Configs Used

HSC:

- `experiments/variants/ensemble_consensus3_sz64_hsc.yaml`

MuSC:

- `experiments/variants/ensemble_consensus3_sz256_musc.yaml`

Execution mode:

- Phase A ensemble-only runs on HPC via `scripts/run_ablation_hpc.sh`
- MLflow file backend on HPC, then exported locally for leaderboard comparison

## Results

### HSC (`BF-C2DL-HSC`, `sz64`)

New `C3` results:

| Fold | IoU |
|---|---:|
| fold-1 | 0.8588 |
| fold-2 | 0.8493 |
| mean | 0.8541 |

Earlier strong `C1` reference:

| Fold | IoU |
|---|---:|
| fold-1 | 0.8912 |
| fold-2 | 0.8787 |
| mean | 0.8805 |

Delta vs earlier `C1` reference:

| Fold | Delta IoU |
|---|---:|
| fold-1 | -0.0324 |
| fold-2 | -0.0294 |
| mean | -0.0264 |

Interpretation:

- clear regression on both folds
- no evidence that the extra summary channels help

### MuSC (`BF-C2DL-MuSC`, `sz256`)

New `C3` results:

| Fold | IoU |
|---|---:|
| fold-1 | 0.7874 |
| fold-2 | 0.7722 |
| mean | 0.7798 |

Earlier `sz256` reference:

| Fold | IoU |
|---|---:|
| fold-1 | 0.7852 |
| fold-2 | 0.8049 |
| mean | 0.7951 |

Delta vs earlier `sz256` reference:

| Fold | Delta IoU |
|---|---:|
| fold-1 | +0.0022 |
| fold-2 | -0.0327 |
| mean | -0.0153 |

Interpretation:

- fold-1 is effectively unchanged
- fold-2 regresses substantially
- overall result is worse than the prior baseline

## Readout

The practical conclusion is:

> richer hand-crafted consensus summary channels did not improve the learned
> ensemble over the simpler overlap-only baseline.

Possible explanations:

- the extra channels are mostly redundant with the overlap map
- the disagreement channel may add noisy structure without giving the model a
  better decision rule
- the current model/training setup may not be strong enough to exploit this kind
  of engineered mask summary

## Decision

Recommended status:

- mark `C3` as a negative result
- do not promote it into the default experiment path
- do not spend more time tuning this exact representation

## What To Try Next

Priority order:

1. Keep `C1 + basic` as the ensemble reference.
2. Focus on data regime / transfer learning rather than more hand-crafted mask
   summary channels.
3. If QA is revisited for ensemble work, pass it as a soft signal rather than a
   hard threshold gate.
4. Prefer experiments that preserve competitor identity or confidence more
   directly, instead of collapsing them into fixed handcrafted summaries.

More specific candidate directions:

- transfer / pretraining from related datasets
- target-only fine-tuning after pretraining
- per-competitor-channel inputs or a small fixed competitor bank
- soft QA-weighted overlap instead of thresholded pruning
- architectures that can use variable sets of candidate masks more directly

## Notes

- The leaderboard output for these runs only showed `ensemble_baseline` rows
  because Phase A was run in isolation; competitor, fusion, and silver-truth rows
  were not part of these exported databases.
- These results are still useful: they cleanly test whether the `C3` input
  formulation alone improves the ensemble. It does not.
