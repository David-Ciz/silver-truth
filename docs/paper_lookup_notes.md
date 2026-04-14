# Paper Lookup Notes

Prepared on 2026-03-26 from the local MLflow store at `data/mlflow/mlruns` and the saved QA Excel/filtering artifacts.

Current status:
- Use `docs/manuscript_artifact_registry.md` as the frozen source-of-truth for manuscript tables and figures.
- Treat this file as lookup/history notes rather than the final citation registry.

## 1. Implementation details

### QA network (HSC)

Reference run:
- Experiment: `phaseB-qa-train-BF-C2DL-HSC-baseline-fold-2`
- Run ID: `498f8d9dadfd4c3c9a1f1fc45cf86645`

Recovered from MLflow params:

| Item | Value |
| --- | --- |
| Backbone | `resnet18` |
| Learning rate | `0.0001` |
| Batch size | `16` |
| Epochs | `50` |
| Weight decay | `0.0001` |
| Patience | `10` |
| Input channels | `0,1` |
| Augmentation flag | `True` |
| Training time | `561.31 s` (`9m 21s`) |

Augmentations in code:
- `prepare_images_for_model()` in `src/silver_truth/qa/cnn.py` applies random horizontal flip, random vertical flip, and random 90-degree rotations when `augment=True`.

PyTorch version:
- `requirements.txt` does **not** pin `torch`.
- The active project environment is `torch==2.9.1`.
- The repo does pin `pytorch_lightning==2.5.5` and `segmentation-models-pytorch==0.5.0`.

GPU model:
- No HPC or Slurm log for this QA run is present in the workspace.
- The local training logs that are available show Apple Metal (`GPU available: True (mps)`), so the exact HPC GPU model is not recoverable from the current repository state.

Timing caveat:
- The local HSC MLflow runs vary a lot by backbone and device:
  - `498f8d9dadfd4c3c9a1f1fc45cf86645` (`resnet18`, fold-2): `561.31 s`
  - `3175fd27b606403b8de2d5dacde48970` (`resnet50`, fold-1): `3228.13 s`
  - `ffe76b09668145af96526a707ebb63f2` (`resnet50`, fold-2): `4664.64 s`
- The Karolina log you provided on 2026-03-27 shows a CUDA run on `NVIDIA A100-SXM4-40GB` finishing a MuSC `resnet18` QA training step in about `35 s`, so training time should be reported together with hardware/context rather than as a single universal number.

### Fusion engine / ensemble (HSC)

Reference training run:
- Experiment: `phaseA-ensemble-BF-C2DL-HSC-baseline-fold-2`
- Run ID: `79ac3b45da17436698d0ba954babd98f`

Recovered from MLflow params:

| Item | Value |
| --- | --- |
| Model type | `ModelType.UnetPlusPlus` |
| Batch size | `7` |
| Loss | `LossType.MSE` |
| Augmentation preset | `basic` |
| Dataset transform | `HorizontalFlip + RandomRotate90 + ToTensorV2` |
| Random seed | `42` |
| Training time | `50.23 s` |

Recovered from config/code:

| Item | Value |
| --- | --- |
| Encoder backbone | `resnet34` |
| Encoder weights | `null` |
| Max epochs | `100` |
| Learning rate | `1e-3` (Adam in `SMP_Model.configure_optimizers()`) |

Notes:
- `ensemble_encoder_name: resnet34` is defined in `experiments/base.yaml`.
- `silver-ensemble ensemble-experiment` defaults to `--max-epochs 100`, `--encoder-name resnet34`, `--batch-size 7`, and `--augmentation basic`.
- The local battery log at `data/paper_runs/experiment_battery_20260313_105152/run.log` shows an early-stopped run ending at `Epoch 33/99` for the same baseline setup.
- No HPC log with a concrete GPU model is present in the repo for the ensemble runs either.

## 2. HSC ablation values requested for Table 3

Primary source:
- `paper-compare-BF-C2DL-HSC-baseline-fold-1`
- `paper-compare-BF-C2DL-HSC-baseline-fold-2`

| Run name | Fold-1 IoU | Fold-2 IoU | Notes |
| --- | --- | --- | --- |
| `ensemble_baseline` | `0.8911853362` | `0.8388671789` | Fold-1 is stored as `ensemble_only` in the fold-1 compare experiment. Fold-2 has two `ensemble_baseline` runs; the later run `a5a6aa8fd0e8423fad1feb0ab8c0a498` is the newest value. |
| `qa_only__top1` | `0.8652157014` | `0.8197688256` | Directly from `paper-compare-*` |
| `full_pipeline_t0.75__simple` | `0.8776689259` | `0.8204002306` | Directly from `paper-compare-*` |
| `ensemble_qa_retrained_t0.75` | `0.8866293938` | `0.8133766311` | Directly from `paper-compare-*` |

Important discrepancy:
- The draft values you referenced do not match the current fold-specific compare runs.
- `0.8873181280` exists, but it belongs to `phaseC-full-pipeline-crop-eval-BF-C2DL-HSC-baseline-mixed` run `4b91bda653284f7cac2dd50575813289`, not to fold-2.
- I could not find a fold-specific HSC `ensemble_qa_retrained_t0.75` value equal to `0.8700` anywhere in the local MLflow store.

## 3. MuSC sz256 ablation values requested for Table 3

Primary source:
- `paper-compare-BF-C2DL-MuSC-musc_crop256-fold-2`

Available locally:

| Run name | Fold-1 IoU | Fold-2 IoU |
| --- | --- | --- |
| `ensemble_baseline` | not present locally | `0.7424886045` |
| `qa_only__top1` | not present locally | `0.7790467436` |
| `full_pipeline_t0.75__simple` | not present locally | `0.7834170785` |

Blocking issue:
- The local MLflow store only contains MuSC experiments for `*-fold-2`.
- There are no `paper-compare-BF-C2DL-MuSC-*-fold-1` experiments in the workspace, so the requested fold-1 MuSC values cannot be recovered from the current repository state.

## 4. Per-competitor IoU table for Section 4.2 (HSC)

Source:
- `paper-compare-BF-C2DL-HSC-baseline-fold-1`
- `paper-compare-BF-C2DL-HSC-baseline-fold-2`

| Competitor | Fold-1 IoU | Fold-2 IoU |
| --- | --- | --- |
| `CALT-US` | `0.8525728720` | `0.8952722662` |
| `DREX-US` | `0.6407895795` | `0.6665891735` |
| `KIT-Sch-GE` | `0.8165821918` | `0.8344695282` |
| `KTH-SE` | `0.7536075289` | `0.6801403872` |
| `MU-Lux-CZ` | `0.7966509390` | `0.8789615922` |

MLflow naming note:
- The HSC KTH row is logged as `competitor__KTH-SE (5)`.

## 5. QA validity analysis for Section 5.4 (HSC test sheets)

Sources:
- Authoritative fold 1 QA workbook from the March 27, 2026 threshold sweep:
  `/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc/campaigns/ablation_threshold_sweep_2026-03-27/paper_runs/qa_results/BF-C2DL-HSC/sz64/hsc_crop64_qa_predictions_fold-1.xlsx`
  (`qa_regression` run `0537f033bcb641d2afbb6024a1ecbd3d`)
- Authoritative fold 2 QA workbook from the March 27, 2026 threshold sweep:
  `/mnt/proj1/eu-25-40/innovaite/silver-truth-hpc/campaigns/ablation_threshold_sweep_2026-03-27/paper_runs/qa_results/BF-C2DL-HSC/sz64/hsc_crop64_qa_predictions_fold-2.xlsx`
  (`qa_regression` run `a705d23af00642c6b6e4362a1812afea`)

Important note:
- The local cached `data/mlflow/.../baseline_qa_predictions_fold-*.xlsx` files do exist, but they do **not** match the March 27, 2026 threshold-sweep runs and should not be used for the paper figure.
- The stale local files have test counts `1809` and `1509`, while the authoritative export DB reports `1500` and `1254` test samples for the latest HSC64 QA runs.

Authoritative summary metrics from `hpc_run_results/hsc-qafix-export_export.db`:

| Fold | Pearson r | Spearman rho | Filtered count at `t=0.75` | Filtered pct at `t=0.75` |
| --- | --- | --- | --- | --- |
| Fold 1 | `0.3273443779` | `0.1817055109` | unavailable locally | unavailable locally |
| Fold 2 | `0.0825282679` | `0.0719059907` | unavailable locally | unavailable locally |

### QA calibration comparison: actual vs predicted Jaccard

These numbers are from the authoritative `qa_regression` metrics logged in the March 27, 2026 export DB:

| Fold | Actual mean | Predicted mean | Bias (`pred - actual`) | MAE | RMSE | Pearson r | Spearman rho |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fold 1 | `0.7952` | `0.8164` | `+0.0212` | `0.0580` | `0.0786` | `0.3273` | `0.1817` |
| Fold 2 | `0.8043` | `0.7054` | `-0.0989` | `0.1427` | `0.1686` | `0.0825` | `0.0719` |

Interpretation:
- Fold 1 still shows only weak association on the held-out test set.
- Fold 2 is substantially worse on the held-out test set, with near-zero correlation and clear underestimation.
- The row-level March 27 workbooks are not present in this workspace, so the filtering counts and the final scatter panel cannot be recomputed locally until those files are synced.

Split note:
- These authoritative numbers are from the `test` split metrics in the export DB for the March 27, 2026 threshold-sweep run.
- The earlier local cached workbooks produced more optimistic values, but those are stale and should not be cited.

### Paper-ready wording for Section 5.4

Suggested paragraph:

> The QA regressor showed limited generalization on the held-out HSC64 test folds. In the March 27, 2026 threshold-sweep run, the test correlation between predicted and actual Jaccard reached only `r=0.3273` on fold 1 and `r=0.0825` on fold 2, with Spearman correlations `rho=0.1817` and `rho=0.0719`, respectively. Fold 2 also showed underestimation, with mean predicted Jaccard `0.7054` versus mean actual Jaccard `0.8043`. These results indicate that the current HSC64 QA model provides at best weak ranking signal on fold 1 and very poor ranking signal on fold 2, so threshold-based QA gating should be treated cautiously.

Suggested short follow-up sentence if needed:

> In the current HSC64 setting, the learned QA model appears weak on fold 1 and near-uninformative on fold 2, which makes hard thresholding unstable.

Suggested figure caption:

> Predicted IoU versus actual IoU on the held-out HSC64 test cells for fold-1 and fold-2. In the March 27, 2026 threshold-sweep run, the QA regressor shows weak association with the ground-truth IoU on fold-1 (Pearson `r=0.3273`, Spearman `rho=0.1817`) and very weak association on fold-2 (Pearson `r=0.0825`, Spearman `rho=0.0719`). The row-level workbooks needed to regenerate this exact scatter are not yet synced into the local workspace.

## 6. Generated figures

Figure script:
- `scripts/generate_paper_figures.py`

Outputs:
- `data/paper_runs/figures/hsc_qa_scatter.png` currently reflects stale local cached workbooks and should be regenerated after syncing the March 27, 2026 HSC64 QA workbooks.
- `data/paper_runs/figures/musc_context_curve.png`
