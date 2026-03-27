# Paper Lookup Notes

Prepared on 2026-03-26 from the local MLflow store at `data/mlflow/mlruns` and the saved QA Excel/filtering artifacts.

## 1. Implementation details

### QA network (HSC)

Reference run:
- Experiment: `phaseB-qa-train-BF-C2DL-HSC-baseline-fold-1`
- Run ID: `3175fd27b606403b8de2d5dacde48970`

Recovered from MLflow params:

| Item | Value |
| --- | --- |
| Backbone | `resnet50` |
| Learning rate | `0.0001` |
| Batch size | `16` |
| Epochs | `50` (`final_epoch=46`) |
| Weight decay | `0.0001` |
| Patience | `10` |
| Input channels | `0,1` |
| Augmentation flag | `True` |
| Training time | `3228.13 s` (`53m 48s`) |

Augmentations in code:
- `prepare_images_for_model()` in `src/silver_truth/qa/cnn.py` applies random horizontal flip, random vertical flip, and random 90-degree rotations when `augment=True`.

PyTorch version:
- `requirements.txt` does **not** pin `torch`.
- The active project environment is `torch==2.9.1`.
- The repo does pin `pytorch_lightning==2.5.5` and `segmentation-models-pytorch==0.5.0`.

GPU model:
- No HPC or Slurm log for this QA run is present in the workspace.
- The local training logs that are available show Apple Metal (`GPU available: True (mps)`), so the exact HPC GPU model is not recoverable from the current repository state.

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
- Fold 1 Excel: `data/mlflow/mlruns/621395107482066401/e384717d5ef54524bfc47a59dc45af3e/artifacts/baseline_qa_predictions_fold-1.xlsx`
- Fold 2 Excel: `data/mlflow/mlruns/695533842535544002/58061dff8ea34d69b5bfaefb08325e4b/artifacts/baseline_qa_predictions_fold-2.xlsx`
- Fold 1 filtering CSV: `data/mlflow/mlruns/358557697386534952/f59c9f6a1b014fff809da13b359bd8d2/artifacts/qa_filtering/baseline_qa_predictions_fold-1_filtering_metrics.csv`
- Fold 2 filtering CSV: `data/mlflow/mlruns/503523312460336893/cb41bad08d124d798cddbdaf5685e3e6/artifacts/qa_filtering/baseline_qa_predictions_fold-2_filtering_metrics.csv`

Computed from the `test` sheets with `scipy.stats.pearsonr()`:

| Fold | Pearson r | Filtered count at `t=0.75` | Filtered pct at `t=0.75` |
| --- | --- | --- | --- |
| Fold 1 | `0.3627442551` | `597` | `33.0017%` |
| Fold 2 | `0.2285150920` | `1502` | `99.5361%` |

## 6. Generated figures

Figure script:
- `scripts/generate_paper_figures.py`

Outputs:
- `data/paper_runs/figures/hsc_qa_scatter.png`
- `data/paper_runs/figures/musc_context_curve.png`
