# Training implementation and recipes

The default inference command loads the fitted selected models. It does not train. This repository also includes the training implementations, frozen hyperparameters, fit scopes and training histories used to produce those models.

## Source encoder

The primary source model was fitted on 625 Charades Training annotation records, representing 623 unique media files and 211 people. Each video is divided into 8-second windows with stride 4 seconds. For each of 157 actions, the target is its occupied fraction of sampled frames; invalid annotation classes are masked. Sampling balances people and unique clips.

| Setting | Primary source training |
|---|---|
| Initialization | IG65M → Kinetics R(2+1)D-34 |
| Updated parameters | `layer3`, `layer4`, new `FC157`; other parameters fixed |
| Batch normalization | Running statistics fixed; selected affine parameters learned |
| Optimizer | AdamW; weight decay 0.01 |
| Learning rates | Backbone 1e-6; source head 1e-3 |
| Epochs / batch | 3 / 16 windows |
| Precision | BF16 forward; FP32 master parameters |
| Seed | 20267914 |

The relocatable source recipe is `training/charades_rgb625.json`. After obtaining and preparing the authorized external data, the source trainer is:

```bash
PYTHONPATH=core:core/src python core/scripts/train_charades_r34_source.py \
  --inputs /absolute/path/to/prepared_charades_windows \
  --config training/charades_rgb625.json \
  --arm rgb --output outputs/source_training
```

The input folder is produced by `core/scripts/produce_charades_training_inputs.py`; it contains `rgb_report.json`, `depth_report.json`, `metadata/<video>.json`, and per-video NPZ window arrays. The original source study prepared paired RGB/depth inputs and the retained source trainer reads both reports even for the RGB arm. `core/scripts/prepare_charades_source_pack.py` contains the source-only selection algorithm; `produce_charades_training_inputs.py` contains frame sampling and occupancy-label construction. `training/charades_input_original.json` records the producer configuration. `pack_charades_source_windows.py` can create the lossless cache consumed by `--packed-cache`.

These archived preparation drivers retain original storage defaults. Adapt their data locations to your authorized data copy before running them. This release supplies the fitted source model, so source-video access and depth generation are not needed to reproduce the selected predictions.

## Target HAU presence

The source backbone is fixed while fitting the target head. The target head uses two views and eight windows per recording. Its layer sizes are 512 → 128 → 40, with GELU and dropout 0.1. It learns from partial action labels, uses normalized temporal LSE at temperature 0.5, and adds an auxiliary HARn action loss with weight 0.25. Training uses AdamW at learning rate 0.001, weight decay 0.01, 40 epochs and batch size 32; a cosine schedule decays the learning rate.

`core/scripts/extract_r34_source_hau.py` implements target window-feature extraction. `core/scripts/prepare_r34_external_warmup.py` produces the auxiliary HARn source features. The nested head fitting driver is:

```bash
PYTHONPATH=core:core/src python core/scripts/run_action_r34_source_nested.py \
  --config /absolute/path/to/your_bound_hau_recipe.json \
  --output-dir /absolute/path/to/new_hau_training_run
```

Use `training/hau_presence_original.json` as the exact archived recipe. Bind `train_csv`, `window_bank`, `auxiliary_bank`, `option_control`, the declared input entries, and implementation entries to the newly prepared files. The archived input checks identify the original experiment; when you regenerate inputs, their records must describe those actual files. `training/hau_pool_extraction_original.json` records the feature producer. The original driver includes generic and synthetic-depth control arms; those are study controls, not additional inputs to the primary inference profile.

The nested study uses held-out people and inner-person selection. The selected final head is fitted on the corresponding complete Training scope. Records are in `training_configs/primary_hau_frozen_config.json`, `primary_hau_protocol.json`, `primary_hau_training_scope.json`, and `primary_hau_training_history.json`.

## Remaining trained components

| Component | Training / export source | Fitted settings and scope |
|---|---|---|
| HAU action evidence | `run_hau_kuno_presence.py`, `run_hau_repeated_option_frequency.py`, `extract_hau_skeleton_action_scores.py`, `hau_action_checkpoint.py` | Bound action descriptors in `core/conditioned_package.json` |
| Pairwise sequence model | `run_hau_pairwise_temporal.py`, `temporal_checkpoint.py` | Bound sequence descriptors and fitted forest states |
| D10 action-set prior | `core/src/cuhkx/server_temporal_d10_backoff.py` | `core/configs/conditioned_d10.json`, fitted counts in `artifacts/d10_model/model.json` |
| Emotion context | `run_emotion_nested.py`, `run_emotion_recipe_nested.py` | Bound emotion descriptors |
| Local media–text emotion model | `run_local_emotion_distillation.py`, `core/src/cuhkx/local_emotion_student.py` | `core/package_manifest.json` and bound local model descriptor |
| HARn heads | `run_harn_tail_nested.py`, `run_harn_seed_ensemble.py`, `run_harn_three_family_nested.py` | Bound HARn descriptors |
| Secondary paired-source model | `train_charades_crossmodal_source.py`, `fit_r34_external_target_fast.py` | `training_configs/secondary_paired_source_*.json` |
| Secondary temporal residual | `core/src/cuhkx/temporal_consumer_residual.py` | `training_configs/secondary_temporal_residual_receipt.json` |

Paths in the table are under `core/scripts/` unless given in full. The K-KUNO ensemble is an upstream public trained model; it was not trained by this team. Its model card and attribution are retained in `core/delivery/licenses/`.

## Reproduction target

Loading the released fitted states reproduces the selected competition system. Refitting from authorized training media is a separate experiment: the source preparation and archived research drivers are supplied for inspection and adaptation, but this public export does not claim a newly validated single-command retraining of every historical nested study. No retraining is needed for the end-to-end raw-Test reproduction described in the README.
