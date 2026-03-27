# Advisor Update And Next Steps
_Prepared: 2026-03-13_

## Executive Position

The repository is no longer missing the basic ablation scaffolding. Most of the advisor-requested machinery already exists in runnable form:

- QA regression evaluation exists.
- QA filtering validity evaluation exists.
- Parquet filtering for `qa_only` and `full_pipeline` exists.
- Full-image reconstruction/evaluation for crop-based outputs exists.
- The main experiment orchestrator is `scripts/run_ablation.py`.

The current issue is different: the present fold-locked evidence does **not** yet support the strongest intended paper claim that the QA referee is the mechanism driving the gain.

At the moment, the strongest supported claim is closer to:

> the learned ensemble is promising, but the current QA gating stage adds little or can hurt under thesis-fold evaluation.

## Immediate Reframe

The project narrative should be reframed immediately:

- QA is not the main supported mechanism at the moment
- the ensemble is the stronger supported contribution
- QA should be treated as secondary unless the final fold-safe HSC statistics rescue it
- MuSC should be run next as a controlled generalization/behavior study, not as proof that QA is already validated

## What Has Already Been Implemented

### QA validity and filtering

Already present in the live code:

- `silver-evaluation evaluate-qa-model`
- `silver-evaluation evaluate-qa-filtering`
- `silver-evaluation merge-qa-predictions`
- `silver-evaluation filter-parquet`

This means the following advisor requests are substantially covered already:

- correlation/statistical QA analysis
- filtering precision/recall-style logging
- restricted ablation preprocessing for `qa_only` and thresholded `full_pipeline`

### Ablation execution

The maintained orchestration path is:

- `scripts/run_ablation.py`
- `experiments/base.yaml`
- `experiments/variants/*.yaml`

This runner already supports:

- `fusion_only`
- `qa_only`
- `full_pipeline`
- threshold sweeps
- `ensemble_only`
- `ensemble_qa`
- `ensemble_qa_retrained`

### Transfer support

Transfer configuration is already prepared:

- source pretraining: `experiments/variants/ensemble_ref_musc.yaml`
- HSC fine-tuning: `experiments/variants/ensemble_transfer_musc_to_hsc.yaml`

## What We Have Tried And What We Learned

### 1. Fold-locked HSC results are not yet a QA success story

The fold-based results summarized in the repo show:

- best competitor (`CALT-US`) average IoU: `0.8739`
- `fusion_only / SIMPLE`: `0.8659`
- `qa_only / top-1`: `0.8236`
- `full_pipeline / SIMPLE` at `t=0.75`: `0.8144`
- `ensemble_only`: `0.8451`

Interpretation:

- the current fold-based QA-gated pipeline does not beat the best competitor
- QA filtering is currently hurting more than helping
- the thesis-fold setting is especially unstable because fold-2 has very little training data

### 2. Mixed HSC results suggest the ensemble itself is the stronger idea

On mixed training:

- best competitor (`CALT-US`): `0.8830`
- `SILVER-TRUTH`: `0.8876`
- `fusion_only__simple`: `0.8919`
- `full_pipeline_t0.70__simple`: `0.9018`
- `ensemble_only`: `0.9054`

Interpretation:

- the learned ensemble becomes the strongest row once data is less scarce
- QA has at best marginal effect on the learned ensemble in this regime
- the data regime, not the QA mechanism, appears to be the dominant lever

### 3. Several local optimization ideas were tested and did not rescue the fold problem

Already explored:

- `C2` input variant (overlap + raw image): worse than baseline
- stronger augmentation: worse than baseline
- ImageNet encoder initialization: roughly neutral
- alternative small sweep variants: no meaningful gain over baseline

Operational conclusion:

- stop spending time on local augmentation/C2 tuning
- move effort to data regime and generalization questions

### 4. MuSC-to-HSC transfer is conceptually ready but practically blocked

The repo documents a transfer path from `BF-C2DL-MuSC` to `BF-C2DL-HSC`, but the size analysis on 2026-03-13 found that `64x64` crops are not valid for MuSC:

- only `53.6%` of MuSC GT boxes fit within `64`
- MuSC seq02 is especially problematic (`41.6%` fit at `64`)
- HSC fits `64x64` cleanly, but MuSC does not

Current implication:

- do **not** run MuSC transfer as the main next experiment on `sz64`
- first open a crop-size branch for MuSC (`96` and likely `128`)

## What We Should Tell The Advisor

Suggested message:

1. The required ablation and QA-analysis infrastructure is mostly already implemented in the repository.
2. We have already run the core HSC experiments and the current fold-locked evidence does **not** validate the intended claim that QA filtering is the main source of improvement.
3. What currently looks strongest is the learned ensemble itself, especially when training data is less constrained.
4. We therefore should not oversell the QA referee as a proven mechanism until we finish the missing statistical package and rerun the key fold-safe analyses.
5. The most credible next step is not more local tuning, but either:
   - finishing the fold-safe evidence package on HSC, or
   - expanding/generalizing the ensemble story with an additional dataset
6. MuSC transfer remains a good idea in principle, but only after fixing MuSC crop size because `64x64` is demonstrably invalid there.

## Recommended Next-Step Plan

### Priority 1: Finish the paper-safe evidence package on HSC

Do this before new narrative claims:

1. Freeze the HSC default reference as `C1 + basic` ensemble baseline.
2. Re-run or consolidate the fold-safe ablation outputs in a single report:
   - `fusion_only`
   - `qa_only`
   - `full_pipeline`
   - `ensemble_only`
3. Export QA validity outputs cleanly for both folds:
   - Pearson/Spearman
   - scatter plot
   - thresholded filtering metrics
4. Implement the still-missing statistical reporting layer:
   - bootstrap 95% CI over test images
   - paired comparison vs baseline rows
5. Convert all of that into paper tables/figures with one locked reporting script.

Reason:

- without this package, the work is still vulnerable to the exact reviewer criticism the advisor raised

### Priority 2: Reframe QA as secondary unless the new fold-safe statistics rescue it

Current evidence does not justify leading with:

> QA filtering improves the full pipeline.

Safer current framing:

> learned fusion is the main effective component, and QA filtering is an exploratory quality-control mechanism whose benefit is currently inconsistent under strict fold evaluation.

Only promote QA back to the central contribution if the final fold-safe evidence actually supports it.

### Priority 3: Open the MuSC crop-size branch before any transfer run

Required sequence:

1. regenerate MuSC QA crops at a larger size
2. validate size fit quantitatively again
3. manually inspect clipping
4. run MuSC fold baselines
5. only then run MuSC -> HSC transfer

Preferred first size:

- `128`

Reason:

- `96` still leaves too many MuSC objects oversized

### Priority 4: Add one more dataset for generalization

This is still worthwhile for publication strength, but after Priority 1.

Best role for the second dataset:

- support the ensemble/generalization claim
- not yet another large QA-tuning surface

This can be done either by:

- the existing `fluo_dataset` variant path, if data prep is available for that dataset
- or a more controlled second CTC dataset with the same reporting structure as HSC

## Practical Internal Work Order

### Immediate

1. Audit and lock the exact fold-safe result files we will use in the manuscript.
2. Implement the missing CI/significance reporting.
3. Produce one consolidated advisor-facing result table from existing HSC runs.
4. Reframe manuscript/internal summaries to ensemble-first, QA-secondary language.

### After that

1. Start MuSC crop-size remediation.
2. Run MuSC fold baselines at the validated larger crop size.
3. Decide whether transfer or second-dataset replication gives the stronger publication return.

### Stop doing

- more C2 experimentation
- more aggressive augmentation sweeps
- claiming QA is already validated as the core mechanism

## Important Repo Note

The maintained experiment entry point is the config-driven `scripts/run_ablation.py` workflow.
