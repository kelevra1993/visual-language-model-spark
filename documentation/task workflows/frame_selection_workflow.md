# Frame Selection Workflow

This document follows one frame selection run (SEL: in every whole sequence, the frames to choose for every wanted class) from `main()` to the evaluation of a checkpoint: which function runs when, what it receives and what it returns. The **Call Tree Overview** gives the whole run on one page. Each stage then has one step per function, with a short explanation and a **real** output. The frame selection trainer is the sequence labelling trainer fed with whole sequences. Steps shared with it are explained in `sequence_labelling_workflow.md`, those shared with the sequence classification (maximum sequence length, padding, frame encoder) in `sequence_classification_workflow.md`, and those of every task in `classification_workflow.md`. The steps specific to the selection (unselected class, whole sequences checked against the wanted classes, selection loss, Hungarian and greedy decodings, selection and ranking metrics, selections file, selection images) are explained in full.

The examples come from real runs on CPU, on a small generated dataset (`create_synthetic_dataset(task="frame_selection")`): 500 training, 60 validation and 60 test sequences of 4 to 12 frames of 128 pixels, each holding one frame of each wanted class (`circle`, `triangle`, `star`), the other frames `unselected`.
- **Stages 1 and 2**: `configurations/frame_selection/configuration_synthetic_sequences.yaml` (maximum sequence length 50, 256 pixel frames, batch size 16, bidirectional GRU, Hungarian decoding, one frame per class, selection loss weight 1).
- **Stages 3, 4 and 5**: the same configuration shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 (an intermediate one every 3), a summary every 4, a validation batch every 10. After 12 iterations every frame prefers `unselected`: the metrics are poor on purpose.

## Read This First
The frame selector chooses, in every whole sequence, exactly `Selection.frames_per_class` frames for every wanted class. Every class of the label mapping except `Selection.unselected_category` is wanted. Its network is the sequence labeller (`sequence_labelling_workflow.md`): one probability vector per frame over the wanted classes and the unselected class. It is trained with the per-frame cross entropy plus a selection loss in which the frames of a sequence compete for every wanted class. The frames are then chosen by the Hungarian assignment or a greedy rule. Both are scored at every evaluation; the one of `Selection.decoding` draws the images, gives the confidence thresholds and runs in the deployed decoder.

Choose it rather than M2M when every sequence must give a fixed number of frames of every class and the right frame depends on the other frames (the sharpest view of each side of an object, the example of the [tasks reference](../reference/tasks_reference.md#frame-selection-sel)). Choose M2M when every frame has a class of its own.

### The Task At A Glance
So that the stages below can be followed on one page, this block takes a sequence of the synthetic configuration from its manifest to the decodings, the evaluation files and the exported decoder.

```text
<split>/data/<sequence>/*.png   <split>/labels.json: {"frame_category_ids": [1, 1, 4, 1, 1, 1, 1, 2, 1, 1, 1, 3]}
        │ SequenceDataset(label_type="selection"): an unmapped category gives the unselected class, and a sequence
        │ without exactly frames_per_class (1) frames of every wanted class stops the training
        ▼
batch   images [16, T, 3, 256, 256] float32   sequence_mask [16, T]   labels [16, T] (-100 on the padding frames)
        │ encode_frames(): Backbone on the real frames only [N, 64, 16, 16], average pooling -> [16, T, 64]
        ▼
GRU 1x128 bidirectional ─> [16, T, 256] ─> ClassificationHead [64] ─> class_logits [16, T, 4]
        │ classification_loss: cross entropy of every real frame (class_weights, label_smoothing)
        │ selection_loss x selection_loss_weight 1.0: softmax of every wanted class over the frames of its sequence
        ▼
test    probabilities [T, 4] of every sequence ─> hungarian and greedy decodings ─> chosen frames [W=3, N=1]
        │ selection, ranking and frame metrics, thresholds at target_precision 0.99, bootstrap intervals
        ▼
Evaluation/Iteration_<n>/   class_thresholds.json   class_metrics.csv   confusion_matrix.csv .png (decoded frames)
                            selections.csv   Samples/<sequence>.png   Errors/<CLASS>/<sequence>.png
export  frame_selector_<format>: images [B, T, 3, 256, 256] ─> class_probabilities [B, T, 4], with preprocessing.json
        (and the ONNX metadata): the input contract, keep_ratio 256 linear, padding 0 centred, BGR, 0-255, NTCHW
        frame_selector_decoder (Python backend, CPU) ─> selected_frames, selected_probabilities, low_confidence,
        candidate_frames, candidate_probabilities; frame_selector_<format>_ensemble chains both
```

### Files To Know
Whenever a part of the task has to change, this table gives the file holding it and the functions to read first.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/frame_selection_trainer.py`, on `sequence_labelling_trainer.py` | `FrameSelectionTrainer`: `build_dataloaders`, `build_model`, `decode_sequence`, `compute_evaluation_results`, `build_selection_table`, `write_selections_file` |
| Dataset and loaders | `vision_training/data/sequence/sequence_dataset.py`, `vision_training/data/loaders.py` | `SequenceDataset` (`label_type="selection"`), `find_unselected_class_index`, `describe_selection_problem`, `get_sequence_dataloaders` |
| Model | `vision_training/models/sequence/frame_selector.py`, on `sequence_labeller.py` | `FrameSelector`: `wanted_class_mask` buffer, `selection_loss_weight` |
| Losses | `vision_training/losses/sequence_losses.py` | `compute_sequence_labelling_loss`, `compute_frame_selection_loss` |
| Decoding | `vision_training/models/sequence/frame_selection_decoding.py` (numpy and scipy only) | `decode_frame_selection`, `decode_hungarian_selection`, `decode_greedy_selection`, `compute_frame_ranks`, `flag_low_confidence`, `run_frame_selection` |
| Metrics and images | `vision_training/evaluation/sequence/frame_selection_metrics.py`, `vision_training/utilities/visualization.py`, `vision_training/evaluation/recall_intervals.py` | `compute_selection_metrics`, `compute_ranking_metrics`, `find_selection_thresholds`, `build_frame_selection_image`, `compute_bootstrap_recall_intervals` |
| Configuration checks | `vision_training/configuration_processing/sequence_checks.py` | `check_frame_selection_configuration`, `check_selection_configuration` |
| Synthetic data | `vision_training/data/synthetic/create_synthetic_dataset.py` | `generate_selection_sequence`, `SELECTION_CATEGORIES` |
| Export and deployment | `vision_training/deployment/export_model.py`, `frame_selection_deployment.py`, `frame_selection_decoder_backend.py`, `deployment_utilities.py` | `export_experiment`, `install_staged_models`, `build_decoding_settings`, `find_class_thresholds_path`, `export_frame_selection_decoding`, `TritonPythonModel` of the decoder |

### Run It
In order to generate the dataset, then train, evaluate and export the task, set the configuration, then run these commands from the repository root.

```bash
# Generate 2,000 / 200 / 200 sequences of 4 to 12 frames of 256 pixels, one shape per frame (other sizes with --split_sizes)
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=frame_selection --output_folder="$HOME/Data/sequence_data/synthetic_selected_shapes"

# Train, after setting experiment_folder, the three data folders and Data.label_mapping.mapping_file_path
uv run python -m vision_training.main --configuration_path="vision_training/configurations/frame_selection/configuration_synthetic_sequences.yaml"

# Evaluate every checkpoint into <run>/Checkpoint_Evaluation/; --target_precisions takes a single number for this task
uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder <run folder> --iteration all

# Export the networks, the decoder and the ensembles (run it without arguments for its help)
uv run python -m vision_training.deployment.export_model --experiment_folder <run folder> --iteration <iteration>
```

### Configuration Keys To Know First

If one wishes to change the synthetic configuration, these keys change the selection, its loss and its evaluation the most.

| Key | Synthetic value | Effect |
|---|---|---|
| `Selection.unselected_category` | `"unselected"` | category of the frames not chosen; every other class is wanted; unmapped categories become unselected |
| `Selection.frames_per_class` | `1` | frames chosen per wanted class; every sequence must hold exactly this many (`FPC1`) |
| `Selection.selection_loss_weight` | `1.0` | weight of the selection loss; 0 trains with the per-frame loss alone (`SLW1`, no token at 0) |
| `Selection.decoding` | `"hungarian"` | decoding of the images, thresholds, best checkpoint and deployed decoder; both are scored |
| `TemporalModel.type`, `bidirectional` | `"gru"`, `True` | every frame compared with the whole sequence (`GRU1x128-BI`) |
| `ExperimentConfiguration.evaluation.top_k` | `2` | top-k selection accuracy and candidate frames of the decoder; a setting of this task only |
| `ExperimentConfiguration.evaluation.target_precision` | `0.99` | class confidence thresholds, behind the `low_confidence` flags |
| `ExperimentConfiguration.evaluation.save_predictions` | `True` | `selections.csv` |

### What A Run Writes

To find your way in a run folder, this table gives what the training and every evaluated checkpoint write.

| Output | Contents |
|---|---|
| `Results.txt` | one block per checkpoint: the selection metrics of each decoding, the ranking metrics, the frame metrics, and with `per_class_report` the `Per Class Selection` and `Per Class Recall` tables (Step 4.15) |
| `Evaluation/Iteration_<n>/` | `class_thresholds.json`, `class_metrics.csv`, `confusion_matrix.csv` (and `.png`) of the decoded frames, `selections.csv` with `save_predictions` (Steps 4.13 and 4.14) |
| `Samples/`, `Errors/<CLASS>/` | the selection image of a sequence (Step 4.6) |
| TensorBoard | Train and Validation: `Total Loss`, `Classification Loss`, `Selection Loss`, `Accuracy`; Test: the 13 global metrics, the per-class scalars, the images |
| `evaluation_curves.png`, `Evaluation/evaluation_summary.csv` | the selection accuracy of `Selection.decoding` per iteration; both sent to MLflow |
| `Input_Samples/`, `data.json` | the first 10 sequences of every split (Step 2.14); the frames of every class |

### Pitfalls

In case the chosen frames or the scores look wrong, these behaviours of the task are the first to check.

- **Every sequence must hold exactly `frames_per_class` frames of every wanted class**, or the training stops (Step 2.8). A wanted class left out of the label mapping turns its frames into unselected ones.
- **The Hungarian decoding maximises the probability of the whole labelling, not the number of right frames.** On the 20 sequences of Stage 5, the checkpoint of iteration 3 gets 3 true triangle frames with the greedy decoding and 1 with the Hungarian one (Step 5.6). The best checkpoint, thresholds and deployment follow `Selection.decoding` only.
- **Tied probabilities give every tied frame the best rank**: equal probabilities everywhere score a Top-k Selection Accuracy of 100 % and a Mean True Frame Rank of 1.0 (`compute_frame_ranks()`).
- **The export needs the thresholds of an evaluation.** It reads `class_thresholds.json` of the exported iteration (`find_class_thresholds_path()`) after installing the networks, so a failure there leaves new networks next to an older decoder. A checkpoint never evaluated gets thresholds of 0: run `evaluate_checkpoint.py` first. Averaged weights use the thresholds of the trained weights.
- **A threshold flags the choices below it, right or wrong, and lets confident wrong choices through.** In the run of the frame selection tutorials (`~/Data/tutorial_runs`, 200 test sequences), at iteration 1000 (95.83 % Hungarian selection accuracy) the thresholds flag 21 of the 25 wrong choices and 148 of the 575 right ones.
- **The exported network has no sequence mask**: the sequences of one request must have the same number of frames, from 1 to `maximum_sequence_length`. The decoding runs in the Python backend on the CPU.

## Table of Contents
- [Read This First](#read-this-first)
- [Call Tree Overview](#call-tree-overview)
- [Stage 1 — Entry Point and Configuration Loading](#stage-1--entry-point-and-configuration-loading)
- [Stage 2 — Trainer Initialization](#stage-2--trainer-initialization)
- [Stage 3 — Training Loop](#stage-3--training-loop)
- [Stage 4 — Test Evaluation of a Checkpoint](#stage-4--test-evaluation-of-a-checkpoint)
- [Stage 5 — Standalone Checkpoint Evaluation (Optional)](#stage-5--standalone-checkpoint-evaluation-optional)

---

## Call Tree Overview

Before reading the steps, this tree gives the functions of the run in the order they run.

```text
main() -> load_experiment_configuration() -> check_frame_selection_configuration()
       -> FrameSelectionTrainer  [STAGE 2]: get_sequence_dataloaders(label_type="selection") -> SequenceDataset
          -> describe_selection_problem(); build_model() -> FrameSelector (SequenceLabeller + wanted_class_mask)
       -> run_training_iterations()  [STAGE 3]: FrameSelector.forward() -> SequenceLabeller.forward()
          + compute_frame_selection_loss(); save_model(); run_test_evaluation()
run_test_evaluation()  [STAGE 4]: get_sequence_predictions(), dump_errors() / save_sample_images()
   -> compute_evaluation_results(): decode_frame_selection() (hungarian and greedy), compute_selection_metrics(),
      compute_ranking_metrics(), frame metrics, find_selection_thresholds(), compute_bootstrap_recall_intervals(),
      write_class_metrics_file(), write_selections_file()  -> report_evaluation() + update_evaluation_curves()
scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]: same as every task
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs of the loader on `configurations/frame_selection/configuration_synthetic_sequences.yaml`, its data folders and mapping file redirected to the generated dataset.

### Step 1.1 — `main()`
The entry point of every task (`classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The sections of the sequence labelling (`sequence_labelling_workflow.md`, Step 1.2), plus `Selection`. This table lists its keys.

| Key | Meaning |
|---|---|
| `unselected_category` | the manifest category of the frames not chosen; every other class is wanted |
| `decoding` | `hungarian` or `greedy` (Step 4.9); both are scored, this one draws the images, gives the thresholds and runs in the deployed decoder |
| `frames_per_class` | the frames chosen for every wanted class in every sequence |
| `selection_loss_weight` | the weight of the selection loss (Step 3.10); 0 trains with the per-frame loss alone |

**Example: the sections, the temporal module and the selection settings**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, ClassificationHead, Selection
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'top_k': 2, 'target_precision': 0.99,
                           'save_predictions': True, 'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
TemporalModel:            {'type': 'gru', 'hidden_size': 128, 'number_layers': 1, ..., 'bidirectional': True, 'causal': False, ...}
Selection:                {'unselected_category': 'unselected', 'decoding': 'hungarian', 'frames_per_class': 1, 'selection_loss_weight': 1.0}
```

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'TemporalModel', 'ClassificationHead', 'Selection']
number_iterations: '1e10'      <- still a string
```

### Step 1.4 — `build_label_dictionary()`
The categories of the manifest are the classes of the frames, the unselected category included.

**Example: the label dictionary of the selected shapes**
```text
dict: {1: 0, 2: 1, 3: 2, 4: 3}       <- categories unselected, circle, triangle, star
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_frame_selection_configuration()`:
- the sequence labelling checks: a causal module (`causal: True`, or a GRU or LSTM not bidirectional) is refused with `Backbone.normalization.feature_map_normalization: "sequence"`, whose statistics come from the whole sequence;
- a single `evaluation.target_precision` (a list is refused: the decoder keeps one threshold per class) and a positive integer `evaluation.top_k`;
- `check_selection_configuration()` on the `Selection` section.

The dataset checks later that the unselected category is in the manifests (Step 2.8).

**Example: output with `Selection.decoding: "beam"` and `Selection.selection_loss_weight: -1.0` in the YAML**
```text
Invalid Configuration File : configuration_wrong.yaml
  - 'Selection.decoding' must be one of ['hungarian', 'greedy'], got 'beam'.
  - 'Selection.selection_loss_weight' must be a number >= 0, got -1.0.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="frame_selection") -> <class 'vision_training.trainers.frame_selection_trainer.FrameSelectionTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs of building the trainer on the redirected `configuration_synthetic_sequences.yaml` (256 pixel frames, maximum sequence length 50, batch size 16), run folder in a temporary folder. The construction took 0.3 s on CPU.

### Step 2.1 — `FrameSelectionTrainer.__init__()`
It reads `Selection.decoding` and `Selection.frames_per_class`, then runs the sequence labelling constructor. `LABEL_TYPE = "selection"` makes the loaders read whole sequences with one class per frame. `CLASS_SUPPORT_COLUMN = "True Frames"` names the column counting the true frames of every wanted class. The trainer overrides the loaders (to find the unselected class), the model (to mark it), the tracked metrics, the evaluation accumulator, the error and sample images and the whole evaluation results.

**Example: the attributes after the construction**
```text
self.decoding                   = 'hungarian'
self.frames_per_class           = 1
self.unselected_class_index     = 0
self.wanted_class_indices       = [1, 2, 3]
self.top_k                      = 2
SAMPLE_NAME = 'Sequences'   COUNTS_FRAMES = True   TASK_NAME = 'frame_selection'
```

### Step 2.2 — `BaseTrainer.__init__()`
The constructor of every task (`classification_workflow.md`, Step 2.2); `input_image_size` is 256, the frame size.

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The sections of the sequence labelling (`T50`, `GRU1x128-BI`), then `FPC<frames_per_class>` and `SLW<selection_loss_weight>` when the weight is above 0. The decoding gives no token, since both are scored.

**Example output**
```text
SEL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4_FPC1-SLW1
```

### Step 2.5 — `DataAugmenter()`

The augmenter of the sequence tasks. The configuration activates 6 of the 28 types (`random_crop`, `random_left_right_flip`, `random_rotation`, `brightness_contrast_saturation`, `random_noise`, `random_haze`), each at probability 0.25 (the flip at 0.5), a sequence being augmented with probability 0.4, with `independent_frames` and `shuffle_frames` off.

### Step 2.6 — `setup_training_paths()`
As for every task (`classification_workflow.md`, Step 2.6): a resumed run keeps the classes of its `label_dictionary.json`, then the run folder gets the configuration copy and `label_dictionary.json`. `data.json` follows the loaders (Step 2.7).

**Example: the run folder after this step** (the configuration file held 19,732 bytes on 2026-11-20)
```text
SEL_IN-KAR-256_..._MC4_FPC1-SLW1/
├── Tensorboard/
├── Weights/
├── label_dictionary.json          50 bytes
└── training_configuration.yaml    19,287 bytes
```

### Step 2.7 — `build_dataloaders()` → `get_sequence_dataloaders(label_type="selection")` + `save_data_record()`
The loaders of the sequence labelling, every dataset receiving the `Selection` section; the validation loader only when the run reads it. The trainer reads the unselected class of the built datasets, which must agree, and lists the wanted classes (`get_wanted_class_indices()`). `check_loaders_give_batches()` refuses a split without batches. `save_data_record()` writes `data.json`: the frames of every class per split (unmapped categories counted as unselected, `get_unmapped_class_index()`), and per split its folder, the SHA-256 of its `labels.json` and its frames.

**Example output**
```text
class_names          = ['unselected', 'circle', 'triangle', 'star']
unselected_class_index = 0     wanted_class_indices = [1, 2, 3]
train        500 sequences -> 32 batches of 16 (one epoch = 32 iterations)
validation    60 sequences ->  4 batches of 16
test          60 sequences ->  4 batches of 16
```

### Step 2.8 — `SequenceDataset.__init__()` → `describe_selection_problem()`
With the `selection` label type, the dataset finds the class of the unselected category and gives it to the frames of unmapped categories. Every sequence is kept whole, each frame with its class. Sequences longer than `maximum_sequence_length` are left out with red warnings. `describe_selection_problem()` then checks that every kept sequence holds exactly `frames_per_class` frames of every wanted class. The invalid sequences are reported in one error, the first 20 listed (`MAXIMUM_LISTED_INVALID_SEQUENCES`), and the training stops. A sequence whose `frame_category_ids` does not have one category per frame is refused before, by an error of the same form.

**Example: the sample the loader gets from the first shuffled sequence**
```text
dataset[0]:   images (11, 3, 256, 256)   sequence_mask [True x 11]
              labels [0, 0, 0, 2, 0, 3, 0, 0, 0, 1, 0]                        <- one class per frame, no padding yet
get_class_counts() = [2450, 500, 500, 500]                                   <- frames of every class (3950 in all)
```

**Example: the error with `Selection.frames_per_class: 2`**
```text
ValueError: 500 sequences cannot be used for the frame selection, which needs every wanted class exactly 2 times ('Selection.frames_per_class') : labels.json
 - train_000000 : circle 1 frames, triangle 1 frames, star 1 frames
 ...
 - ... and 480 more
```
With `maximum_sequence_length: 8`, the warnings name every left-out sequence and the run goes on with the 280 training sequences of at most 8 frames (18 iterations per epoch).

### Step 2.9 — `build_native_loader()`

The PyTorch `DataLoader` of the classification, with `collate_padded_sequences`: 32 training batches of 16. It was recorded with 8 workers; every loader now has 1 worker and pins its batches only on a discrete GPU.

### Step 2.10 — `get_test_sample_classes()` + `choose_sample_identifiers()`

`get_sample_class_indices()` lists the wanted classes of the frames of every test sequence. Every sequence counts for its most frequent class, the lowest index on a tie. Here every sequence holds each wanted class once, so all fall under `circle`, and the 8 samples are spread evenly: identifiers 0, 8, 17, 25, 34, 42, 51 and 59.

### Step 2.11 — `build_model()` → `FrameSelector.__init__()`
The frame selector is the sequence labeller (`sequence_labelling_workflow.md`, Step 2.11). It adds `selection_loss_weight` and the `wanted_class_mask` buffer; the trainer sets 0 on the unselected class. The mask limits the selection loss to the wanted classes and, saved with the weights, tells exported models which class is unselected (`get_unselected_class_index()`).

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.temporal_module          RecurrentTemporalModule    148,992 parameters      GRU 64 -> 128, bidirectional, output_features 256
model.classification_head      ClassificationHead          16,708 parameters      Dropout -> Linear(256, 64) -> ReLU -> Dropout -> Linear(64, 4)
model.wanted_class_mask        buffer [4] torch.float32 [0.0, 1.0, 1.0, 1.0]      <- unselected class set to 0 by the trainer
model.selection_loss_weight    1.0
total                                                     349,348 parameters
```

### Step 2.12 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
The selection loss is tracked next to the per-frame loss; the total loss is their sum. The accuracy of the real frames is the only rate, printed as a percentage (Step 3.14).

**Example output**
```text
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
{'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'selection_loss': 'Selection Loss', 'accuracy': 'Accuracy'}
```

### Step 2.13 — `print_experiment_information()` (→ `describe_split()` → `count_split_sequences()`)
**Example output** (shortened)
```text
- Train Data Folder       : train (500 sequences, 3950 frames, 500 sequences of at most 50 frames)
- Validation Data Folder  : validation (60 sequences, 481 frames, 60 sequences of at most 50 frames)
- Test Data Folder        : test (60 sequences, 463 frames, 60 sequences of at most 50 frames)
- Iterations Per Epoch    : 32
```

### Step 2.14 — `print_summary()` / `view_input_batches()` / `restore_last_model()`
The panels of the sequence labelling (temporal module `Output Features : 256`, head `Number Of Classes : 4`). The Export Sizes panel estimates the network alone, the GRU as one weighted operator (`sequence_classification_workflow.md`, Step 2.21); the decoder and ensembles are not estimated. With `view_input_images`, frames of wanted classes are captioned with their number and class, unselected frames with their number only (`get_frame_captions()`), and the panel starts with the frame of every wanted class (`get_sample_caption()`).

**Example: the model size and the captions of the first training sequence**
```text
Model Size: Backbone 183,648, Temporal Module 148,992, Classification Head 16,708, Total Model 349,348 parameters (1.40 MB)
get_sample_caption = 'Selected : circle 9, triangle 3, star 5'
get_frame_captions = ['', '', '', 'triangle', '', 'star', '', '', '', 'circle', '']
```
A new run prints `We Generate A New Model That Will Be Trained From Scratch.` and starts at iteration 1.

---

## Stage 3 — Training Loop

> The examples below are real outputs of the shortened CPU run (64 pixel frames, batch size 4, 12 iterations; an epoch is 125 iterations). Its run name is `SEL_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4_FPC1-SLW1`. It predates the input samples (2026-11-18, `classification_workflow.md`, Step 3.1).

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (`classification_workflow.md`, Step 3.1). This table shows what happens at each iteration of the run.

| Event | Iterations |
|---|---|
| Checkpoint saved | 3, 6, 9, 12 (nothing more after the loop) |
| Validation batch | 10 |
| Console summary | 4, 8, 12 |
| Test evaluation | 6, 12, last in the iteration |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()` / `start_experiment_tracking()`
**Example output** (the MLflow line came out during the first test evaluation)
```text
Tensorboard/Train/        Total Loss, Classification Loss, Selection Loss, Accuracy, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Selection Loss, Accuracy                   (iteration 10)
MLflow Run Created In The Experiment 'frame_selection' : 618a35766a1b4c60a9f2703b93255530 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.3 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12 (`classification_workflow.md`, Step 3.6), with the label dictionary. The checkpoint holds the `wanted_class_mask`, so the unselected class travels with the weights. Current checkpoints also hold `pretrained_backbone_blocks`.

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   4,289,933 bytes      Weights/Iteration_12/model_000012.pt  4,289,997 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
model_state: 95 tensors, the first 'class_weights', 'wanted_class_mask', then 'backbone.convolution_blocks.0. ...'
mlflow_run_id: None (iterations 3 and 6), '618a35766a1b4c60a9f2703b93255530' (iterations 9 and 12)
```
The 64 more bytes of iterations 9 and 12 are the MLflow run identifier.

### Step 3.4 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration (`classification_workflow.md`, Step 3.7). See Stage 4.

### Step 3.5 — `get_next_batch()` → `SequenceDataset.__getitem__()` + `collate_padded_sequences()`
The sample of the sequence labelling (`sequence_labelling_workflow.md`, Step 3.5): the whole sequence with the `[T]` class of every frame. `collate_padded_sequences()` pads to the longest sequence with black frames, a False mask and the label -100.

**Example: the first batch of the run** (sequences of 7, 12, 4 and 10 frames, padded to 12)
```text
images               [4, 12, 3, 64, 64] torch.uint8
labels               [4, 12] torch.int64
                     [[2, 0, 0, 3, 0, 0, 1, -100, -100, -100, -100, -100], [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
                      [3, 2, 1, 0, -100, -100, -100, -100, -100, -100, -100, -100], [0, 0, 3, 2, 0, 0, 0, 0, 1, 0, -100, -100]]
```

### Step 3.6 — `MetricTracker.count_frames()`

The tracker counts the real frames of every interval: 130 for iterations 1 to 4, 129 for iterations 5 to 8.

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
The images go to float32; the labels stay `[4, 12]` int64 and the mask `[4, 12]` bool.

### Step 3.8 — `augment_batch()`
The augmentation of the sequence labelling (`sequence_labelling_workflow.md`, Step 3.8): with `shuffle_frames`, the frame labels move with their frames.

**Example: which sequences of the first batch were changed** (`probability: 0.4`)
```text
images: changed per sample [True, False, False, False]      labels: changed per sample [False, False, False, False]
```

### Step 3.9 — `forward_pass()` → `FrameSelector.forward(sequence_mask=..., labels=...)`
The sequence labeller forward pass (`sequence_labelling_workflow.md`, Step 3.9) gives `class_logits` `[B, T, K]` and the per-frame `classification_loss`. The frame selector adds `selection_loss`, the selection loss times its weight (zero when the weight is 0).

**Example: the first batch through the model**
```text
encode_frames       -> (4, 12, 64)    zero vectors per sequence [5, 0, 8, 2]      <- the padding frames
temporal_module     -> (4, 12, 256)   one vector per frame (128 per direction)
classification_head -> (4, 12, 4)
classification_loss   1.3489 (scalar)            <- close to ln 4 = 1.3863
selection_loss        2.0299 (scalar)            <- the mean of ln 7, ln 12, ln 4 and ln 10 is 2.0299
```

### Step 3.10 — `compute_frame_selection_loss()`
The per-frame loss answers "which class is this frame"; the selection loss answers "which frames of this sequence are the best for this class". For every wanted class of every sequence, the softmax of the class logit is taken over the real frames of the sequence. Its cross entropy with the target `1/N` on each of the `N` true frames is computed. The losses are averaged over the wanted classes (`wanted_class_mask`) that have true frames. The frames thus compete for every class, as in the decodings. Equal logits give `ln T` for `T` real frames, hence the 2.0299 of Step 3.9.

### Step 3.11 — `compute_batch_metrics()` → `compute_frame_accuracy()`
The share of real frames whose most probable class is their class, unselected frames included.

**Example: the first batch**
```text
accuracy = 0.6364          (21 of the 33 real frames: the share of the unselected frames, 3 frames per sequence being selected)
```

### Step 3.12 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
Over the 12 iterations the classification loss goes down from 1.3489 to 0.9288, while the selection loss stays between 1.8282 and 2.2086. The total loss of iteration 1 is 1.3489 + 2.0299 = 3.3788.

**Example: the Train curves** (shortened)
```text
Classification Loss  1: 1.3489, 2: 1.3133, 3: 1.3075, 4: 1.2753, ..., 10: 1.1266, 11: 1.0786, 12: 0.9288
Selection Loss       1: 2.0299, 2: 2.0844, 3: 2.0355, 4: 2.0710, ..., 10: 1.8623, 11: 1.9130, 12: 2.1741
Accuracy             1: 0.6364, 2: 0.6364, 3: 0.6250, 4: 0.6250, ..., 10: 0.5714, 11: 0.6000, 12: 0.6923
```

### Step 3.13 — `training_loss.backward()` + `step_optimizer()` + validation pass
**Example: the validation batch**
```text
Tensorboard/Validation at iteration 10: Total Loss 3.3946, Classification Loss 1.2079, Selection Loss 2.1867, Accuracy 0.6667
```

### Step 3.14 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The interval averages go to TensorBoard, MLflow (with `train/frame_throughput`) and the console.

**Example output: iteration 12** (shortened)
```text
|   Iteration: 12    |     Total Sequences: 48      |    Interval Sequences: 16   |   Throughput: 9.7 Sequences Per Second   |    81.8 Frames Per Second   |
--------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     3.0707    -0.1035 Best
Moving Average of Valid Total Loss                     :     3.3946
Moving Average of Train Classification Loss            :     1.0391    -0.1232 Best
Moving Average of Train Selection Loss                 :     2.0316    +0.0197
Moving Average of Train Accuracy                       :    63.70 %    +1.92 % Best
Moving Average of Valid Accuracy                       :    66.67 %
...
```

### Step 3.15 — end of the loop
Iteration 12 is already saved and evaluated, so the loop does nothing more (`classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 60 Test Sequences, 463 Frames · 2026-10-10 07:20 ───────────────      <- the evaluation of iteration 12
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```

---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of iterations 6 and 12 of the Stage 3 run on its 60 test sequences (463 frames), with `per_class_report: True`, `top_k: 2`, `save_predictions: True` and error images on. Both checkpoints give `unselected` as the most probable class of every frame. The outputs predate the recall intervals and the evaluation curves; the background and swapped picks are counted from the captured confusion matrix.

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (`classification_workflow.md`, Step 4.1).

**Example output: the progress bar** (60 sequences, batch size 4)
```text
Test Evaluation Iteration 12: 100%|██████████| 15/15 [00:01<00:00,  8.48it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The selection hook keeps, for every test sequence, the probabilities and true classes of its real frames and its identifier. The decodings need whole sequences, so they run once the split is evaluated (Step 4.7). It also holds the error images folder and counts.


### Step 4.3 — `prepare_batch()` + `predict_batch()`
The model gives `class_logits` and `class_probabilities` of shape `[4, 12, 4]` for a test batch.

### Step 4.4 — `accumulate_evaluation_batch()` → `get_sequence_predictions()`
`get_sequence_predictions()` reads the `[T, K]` probabilities (float64) and `[T]` true classes of the real frames of a sequence. Then `dump_errors()` runs while the frames are still in the batch.

### Step 4.5 — `dump_errors()` → `decode_sequence()` + `build_selection_image()`
Every test sequence is decoded with the decoding of the run. A wanted class is wrong when its chosen frames are not exactly its true frames. The image (that of Step 4.6) is saved as `Errors/<CLASS>/<sequence>.png` for every class picked wrong, at most `maximum_per_class_pair` per class.

**Example: the error images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Errors/
├── circle/      test_000001.png   test_000053.png
├── star/        test_000001.png   test_000053.png
└── triangle/    test_000001.png   test_000053.png      <- the same image in the 3 folders
report line: "Error Images : 6 in 3 classes, at most 2 per class"
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()` → `build_frame_selection_image()`
The selection hook decodes the sequence (`decode_sequence()`) and draws it with `build_selection_image()`. This table lists the parts of the image.

| Part | Contents |
|---|---|
| frame grid | numbered 160 pixel thumbnails, 8 per row; solid green border for a frame chosen for its true class, solid red for a wrong choice, dashed yellow for a true frame left unchosen, each labelled with class and probability; rightly unselected frames dimmed |
| heat map | the probability of every wanted class over the frames, chosen frames marked with a dot, true frames outlined in white, then a legend bar |
| caption bar | `<right>/<selections> selections right`, green when all are right, red otherwise |

The sample images are `Samples/test_000000.png` to `test_000059.png` (8 images): 480 x 1316 pixels for 9 to 16 frames (two rows), 316 x (164 x T + 4) for T frames up to 8.

**Example: the sample images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Samples/   test_000000.png  test_000008.png  ...  test_000059.png   (8 images)
build_sample_image -> (480, 1316, 3) uint8      a sequence of 9 to 16 frames: two rows of 8 thumbnails
                      (316, 164 x T + 4, 3)     a sequence of T frames, at most 8: one row
```

### Step 4.7 — `compute_evaluation_results()`
The frame selection trainer replaces the whole computation. This table lists its parts.

| Part | Step |
|---|---|
| both decodings on the same probabilities, each scored by `compute_selection_metrics()` | Steps 4.9 and 4.10 |
| `compute_ranking_metrics()` on the probabilities before decoding | Step 4.11 |
| the classification frame metrics on the most probable class of every frame | Step 4.12 |
| thresholds, intervals, per-class file, decoded confusion matrix (decoding of the run) | Step 4.13 |
| the selections file | Step 4.14 |

The headline metric is the selection accuracy of the decoding of the run. `percent_metrics` lists every headline metric but `Mean True Frame Rank`; `metrics` and the scalars keep 0 to 1.

**Example: the results of iteration 12** (shortened)
```text
metrics           = {'Hungarian Selection Accuracy': 0.1222, 'Hungarian Fully Right Sequences': 0.0, ...,
                     'Mean True Frame Rank': 4.1444, 'Frame Accuracy': 0.6112, 'Frame Balanced Accuracy': 0.25, 'Frame Macro F1': 0.1897}
headline_metric   = 'Hungarian Selection Accuracy'
```

### Step 4.8 — `compute_frame_ranks()`
The rank of every frame for every wanted class among the frames of its sequence, ordered by their probability for that class (1 the most probable; tied frames take the best rank). It feeds the ranking metrics (Step 4.11) and the `true_ranks` column of the selections file.

**Example: the test sequence test_000001 at iteration 12** (10 frames; probabilities in the order unselected, circle, triangle, star)
```text
true classes [2, 0, 0, 1, 0, 0, 0, 3, 0, 0]       <- triangle at frame 0, circle at frame 3, star at frame 7
frame 0 [0.3982, 0.1816, 0.1945, 0.2256]     frame 8 [0.4113, 0.1793, 0.2006, 0.2088]
frame 3 [0.4385, 0.1641, 0.1894, 0.208]      frame 9 [0.3876, 0.1917, 0.2078, 0.2129]     ...
ranks [T, W] (circle, triangle, star):
   [[2, 4, 1], [4, 6, 2], [6, 8, 4], [8, 10, 6], [10, 9, 8], [9, 7, 10], [7, 5, 9], [5, 3, 7], [3, 2, 5], [1, 1, 3]]
```
The true circle frame (3) has rank 8, the true triangle frame (0) rank 4, the true star frame (7) rank 7.

### Step 4.9 — `decode_frame_selection()` → `decode_hungarian_selection()` / `decode_greedy_selection()`
A decoding turns the `[T, K]` probabilities into the `[W, N]` frames chosen for the `W` wanted classes, `N = frames_per_class` each, by decreasing probability (`NO_FRAME` where the sequence is too short). The same function runs in the evaluation, the local inference and the Triton decoder.
- **Hungarian**: the gain of frame `f` for class `c` is `log p(f, c) − log p(f, unselected)`, every class column repeated `N` times. `linear_sum_assignment()` finds the assignment of largest total gain, every frame chosen at most once: the most probable labelling in which every wanted class holds `N` frames.
- **Greedy**: every frame takes its most probable class. A class taken by too many frames keeps its most probable ones. Then every class still short, in class order, takes the unselected frames most probable for it. It is cheaper, but can give a frame to a class that only slightly prefers it.

`convert_selection_to_frame_labels()` then gives every chosen frame its class and the others the unselected class, for the decoded confusion matrix.

**Example: the two decodings of test_000001 at iteration 12**
```text
hungarian: selected [W, N] [[9], [8], [0]]   decoded frame classes [3, 0, 0, 0, 0, 0, 0, 0, 2, 1]
greedy:    selected [W, N] [[9], [8], [0]]   decoded frame classes [3, 0, 0, 0, 0, 0, 0, 0, 2, 1]
```
Every frame prefers `unselected`, so the greedy decoding completes the classes in order: circle takes frame 9 (0.1917), triangle frame 8 (0.2006, frame 9 being taken), star frame 0 (0.2256). None is a true frame. The two decodings never differ on this test split; on the reduced split of Stage 5 they differ at iteration 3.

### Step 4.10 — `compute_selection_metrics()`
The scores of one decoding over all test sequences. This table defines them.

| Metric | Definition |
|---|---|
| Selection Accuracy | share of the true frames of the wanted classes chosen for their class, per class (its recall) and overall; the right and true frames of every sequence and class are kept for the interval of Step 4.13 |
| Fully Right Sequences | share of sequences where every wanted class gets exactly its true frames |
| Background Picks | share of picks that took a truly unselected frame |
| Swapped Picks | share of picks that took the true frame of another wanted class |

Right picks, background picks and swapped picks make up every pick. A chosen frame of any other class stops the evaluation with a `ValueError`.

**Example: `Evaluation/Iteration_12/confusion_matrix.csv`** (Hungarian decoded classes; the same at iteration 6)
```text
true \ predicted,unselected,circle,triangle,star
unselected,185,32,33,33
circle,28,7,8,17
triangle,36,13,8,3
star,34,8,11,7
```

**Example: how the numbers of iteration 12 are obtained**
```text
selection accuracy   = (7 + 8 + 7) / (60 + 60 + 60)      = 22 / 180  = 0.1222
background picks     = (32 + 33 + 33) / 180              = 98 / 180  = 0.5444   (unselected row)
swapped picks        = (8 + 17 + 13 + 3 + 8 + 11) / 180  = 60 / 180  = 0.3333   (wanted rows, other wanted columns)
fully right sequences = 0.0
```

### Step 4.11 — `compute_ranking_metrics()`
The scores of the probabilities before decoding, from the ranks of the true frames (Step 4.8). **Top-k Selection Accuracy** is the share of true frames among the `top_k` frames most probable for their class. **Mean True Frame Rank** is the mean rank of the true frames. They tell how far a wrong selection was from the right frame.

**Example: iteration 12**
```text
Top-2 Selection Accuracy 0.3333     Mean True Frame Rank 4.1444
   circle   top-2 0.4000   mean rank 3.9500
   triangle top-2 0.3500   mean rank 3.9500
   star     top-2 0.2500   mean rank 4.5333
```

### Step 4.12 — the frame metrics: `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()`
The classification metrics (`classification_workflow.md`, Steps 4.9 to 4.11) on the most probable class of every real frame, before decoding.

**Example: iteration 12**
```text
Frame Accuracy          = 283 / 463                        = 0.6112   (every frame predicted unselected: only the 283 unselected frames are right)
Frame Balanced Accuracy = (1.0 + 3 x 0.0) / 4              = 0.2500
Frame Macro F1          = (2 x 0.6112 x 1.0 / 1.6112) / 4  = 0.1897
```

### Step 4.13 — `find_selection_thresholds()` + `write_class_metrics_file()` + `save_confusion_matrix_table()` / `save_confusion_matrix_image()`
For every wanted class, `find_selection_thresholds()` takes the probabilities of the frames the run's decoding chose and whether each is a true frame. It finds the probability from which these choices reach `target_precision` with the most true frames. In deployment, a chosen frame below its class threshold, or any frame of a class without threshold, is flagged low confidence for a human to check. The thresholds go to `class_thresholds.json` with the decoding and the unselected class.

The frames of a sequence are chosen together, so `compute_bootstrap_recall_intervals()` gives a 95 % Korn-Graubard interval of every class selection accuracy, from 1000 bootstrap resamples of the test sequences (fixed seed).

`class_metrics.csv` holds every wanted class whatever `per_class_report`, with `hungarian_accuracy_lower` and `hungarian_accuracy_upper` here. With `per_class_report`, `build_recall_interval_table()` adds the `Per Class Recall` table (`True Frames`, `Hungarian Accuracy`, `95 % Lower`, `95 % Upper`, worst first), and the confusion matrix image is drawn (here titled `Decoded Frames · Iteration 12 · Hungarian Selection Accuracy 12.22 %`).

**Example: `Evaluation/Iteration_6/class_thresholds.json`** (at iteration 12 every class is `null`)
```text
{
    "target_precision": 0.99,
    "decoding": "hungarian",
    "unselected_class": "unselected",
    "classes": {
        "circle": null,
        "triangle": {
            "threshold": 0.22613857686519623,
            "recall": 0.016666666666666666
        },
        "star": null
    }
}
```
At iteration 6, the triangle frames chosen at 0.2261 or more are all true, but they are 1 of the 60 true triangle frames.

**Example: `Evaluation/Iteration_12/class_metrics.csv`** (captured before the interval columns)
```text
class_index,class,true_frames,hungarian_accuracy,greedy_accuracy,top_k_accuracy,mean_rank,threshold_at_target_precision,recall_at_target_precision
1,circle,60,0.116667,0.116667,0.400000,3.950000,,
2,triangle,60,0.133333,0.133333,0.350000,3.950000,,
3,star,60,0.116667,0.116667,0.250000,4.533333,,
```

### Step 4.14 — `write_selections_file()`   (only with `save_predictions`)
One row per test sequence and wanted class, so both decodings can be studied without the model: the true frames with probabilities and ranks, the frames of each decoding with probabilities and right count, and the low-confidence flags of the run's decoding (`flag_low_confidence()`). Frames are named by file, separated by `;`. Rows follow the evaluation order.

**Example: `Evaluation/Iteration_12/selections.csv`** (181 lines; first rows)
```text
sequence,class,true_frames,true_probabilities,true_ranks,hungarian_frames,hungarian_probabilities,hungarian_right_selections,greedy_frames,greedy_probabilities,greedy_right_selections,hungarian_low_confidence
test_000001,circle,test_000001/000003.png,0.164078,8,test_000001/000009.png,0.191714,0,test_000001/000009.png,0.191714,0,1
test_000001,triangle,test_000001/000000.png,0.194501,4,test_000001/000008.png,0.200584,0,test_000001/000008.png,0.200584,0,1
...
test_000019,circle,test_000019/000004.png,0.194757,1,test_000019/000004.png,0.194757,1,test_000019/000004.png,0.194757,1,1
```
No class has a threshold at iteration 12, so every chosen frame is flagged, the right circle of test_000019 included.

### Step 4.15 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
Each time a checkpoint is evaluated, its block goes to the console and `Results.txt`, its scalars and images to TensorBoard, its metrics and files to MLflow. Then `update_evaluation_curves()` writes its row into `Evaluation/evaluation_summary.csv` and draws `evaluation_curves.png` again.

**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 60 Test Sequences, 463 Frames · 2026-10-10 07:20 ────────────────────────────────────────────────────────
  Hungarian Selection Accuracy   12.22 %   Hungarian Fully Right Sequences   0.00 %
  Hungarian Background Picks     54.44 %   Hungarian Swapped Picks           33.33 %
  Greedy Selection Accuracy      12.22 %   ...
  Top-2 Selection Accuracy       33.33 %   Mean True Frame Rank              4.1444
  Frame Accuracy                 61.12 %   ...
  Images, Thresholds And Confusion Matrix : hungarian decoding ('Selection.decoding')
  Error Images : 6 in 3 classes, at most 2 per class
  Sample Images : 8 fixed test sequences

  Per Class Selection
  ...  (the columns of class_metrics.csv in Step 4.13, worst Hungarian accuracy first)
```
After the two evaluations, `Tensorboard/Test` holds 22 scalars (13 global + 9 per class) and 9 images, and MLflow the `test/` metrics, `test/best_Hungarian Selection Accuracy`, 9 `test_class/` metrics and the files. A run started after 2026-11-18 also sends `Input_Samples/`, `evaluation_summary.csv` and `evaluation_curves.png`.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of `scripts/evaluation/evaluate_checkpoint.py` on the Stage 3 run (checkpoints 3, 6, 9 and 12), with `--iteration all`, `--batch_size 16` and the first 20 test sequences (`labels.json` reduced, `data/` linked). Like Stage 4, they predate the recall intervals and the background and swapped picks, whose values are counted from the captured choices. They also predate the folder per split (2026-11-20): given as `--test_data_folder`, this reduced split now goes to `Checkpoint_Evaluation_split_<parent folder>_test_reduced/`, not `Checkpoint_Evaluation/`. Each of the 20 sequences holds 3 true frames, 60 of their 155 frames, so each decoding makes 60 picks and leaves 95 frames truly unselected.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The script of every task (`classification_workflow.md`, Stage 5). The test sequences are checked with the `Selection` section of the run, and both decodings are scored on every checkpoint.

### Step 5.2 — trainer construction
The trainer is built from the run's `training_configuration.yaml`, with the classes of its `label_dictionary.json` (`classification_workflow.md`, Step 5.2).

**Example: the split line of the launch information**
```text
- Test Data Folder        : test_reduced (20 sequences, 155 frames, 20 sequences of at most 50 frames)
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'frame_selection' : 5cfdfa9ad0e84bd0a36cd3e352b41020 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
Should several checkpoints compete for the deployment, this step tells which one to keep. Every evaluation adds its row to `Checkpoint_Evaluation/evaluation_summary.csv` (increasing iteration order) and draws `evaluation_curves.png` again. The best checkpoint of the Hungarian selection accuracy is printed.

**Example output** (the four checkpoints tie, so they are all listed)
```text
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
  - Evaluation Curves : evaluation_curves.png
  - Best Checkpoint : iterations 12, 9, 6, 3 tie (Hungarian Selection Accuracy 15.00 %)

iteration,Hungarian Selection Accuracy,Hungarian Fully Right Sequences,Hungarian Background Picks,Hungarian Swapped Picks, ...
3,0.15,0.0,0.5666666666666667,0.2833333333333333,0.18333333333333332, ...
...
12,0.15,0.0,0.5,0.35,0.15, ...
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 3 on the 20 sequences**, where the two decodings differ (shortened)
```text
Iteration 3 · 20 Test Sequences, 155 Frames · 2026-10-10 07:20 ─────────────────────────────────────────────────────────
  Hungarian Selection Accuracy   15.00 %   Hungarian Fully Right Sequences   0.00 %
  Hungarian Background Picks     56.67 %   Hungarian Swapped Picks           28.33 %
  Greedy Selection Accuracy      18.33 %   Greedy Fully Right Sequences      0.00 %
  ...
                            Hungarian     Greedy      Top-2     Mean   Threshold    Recall
   Class      True Frames    Accuracy   Accuracy   Accuracy     Rank     @ P99 %   @ P99 %
  ─────────────────────────────────────────────────────────────────────────────────────────
   triangle            20      5.00 %    15.00 %    35.00 %   3.8500           —         —
   circle              20     20.00 %    20.00 %    50.00 %   3.9500           —         —
   star                20     20.00 %    20.00 %    35.00 %   4.2500      0.2495   10.00 %
```
The greedy decoding finds 3 true triangle frames and the Hungarian one 1 (15.00 % against 5.00 % of 20): the Hungarian assignment maximises the total gain of the three classes, not the number of right frames. The best checkpoint is still chosen on the Hungarian selection accuracy, the decoding of the run.
