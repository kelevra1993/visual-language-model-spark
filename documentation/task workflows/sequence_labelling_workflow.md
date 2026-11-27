# Sequence Labelling Workflow

This document follows one sequence labelling run (many to many, M2M: one class per frame) from `main()` to the evaluation of a checkpoint. The **Call Tree Overview** gives the whole run on one page; each stage then has one step per function, with a **real** output. Steps shared with the sequence classification are explained in `sequence_classification_workflow.md`, steps shared by every task in `classification_workflow.md`; here they keep one line and the real output.

The examples come from CPU runs on a small dataset of `create_synthetic_dataset(task="sequence_labelling", image_size=128, maximum_objects=10, minimum_frames=2, maximum_frames=12)`: 500 / 60 / 60 sequences of 2 to 12 frames. Every frame is `no_change` (always the first frame), `appearing_<shape>` or `disappearing_<shape>`: 71 classes, about half of the frames `no_change`. The quoted `synthetic_changing_shapes` dataset (250,000 / 50,000 / 50,000 sequences) was generated in October 2026, before the generator moved shapes between frames (2026-11-18). The tutorials keep the run `tutorials/M2M_IN-KAR-256_SE-RES-BN-B1x16.2x16.2x16.4x32.5x32.5x32-PT_T50_LSTM2x96-BI_HFLAT.96.96-HLN-CWB_MC71` (`~/Data/tutorial_runs`, 2026-11-15, October dataset); its retrain on the new dataset was stopped at about 1000 iterations (accuracy 12.43 % against 54.89 %).

The table below gives the configuration behind each stage.

| Stages | Configuration |
|---|---|
| 1 and 2 | `configurations/sequence_labelling/configuration_synthetic_sequences.yaml`: at most 50 frames fed whole, 512 pixel frames, batch size 16, a GRU |
| 3, 4 and 5 | the same with 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 (intermediate every 3), a summary every 4, a validation batch every 10, 2 workers, error images and MLflow on |

After 12 iterations the network predicts mostly `no_change` (first loss near ln 71 = 4.263), so the metrics are poor on purpose. The runs were recorded with `top_k: 2` (`Top-2 Accuracy`); the current code fixes k at 5 (`Top-5 Accuracy`). Console rates are shown as the current code writes them (`27.87 %`, `+14.70 %`); files, TensorBoard and MLflow keep values between 0 and 1.

## Read This First
Before choosing this task, know that every frame goes through the backbone and the pooling of `ClassificationHead`, a temporal module keeping the time axis, and the head. The loss is the cross entropy of the labelled frames (`frame_category_ids` of `labels.json`). The table below says which sequence task fits which answer.

| Answer wanted | Task |
|---|---|
| a class per frame (where something happens) | this one (M2M) |
| one class per sequence | sequence classification (M2O, `sequence_classification_workflow.md`) |
| a before and an after sequence | sequence comparison (CMP) |
| exactly `Selection.frames_per_class` frames of every wanted class per sequence | frame selection (SEL, `frame_selection_workflow.md`) |

### The Task At A Glance
The diagram below follows one batch from the files to the evaluation folder.
```text
<split>/data/<sequence>/000000.png ...   <split>/labels.json: {"frames": [...], "frame_category_ids": [1, 1, 48]}
        │ SequenceDataset(label_type="frame"): one sample per whole sequence, an unmapped category gives -100
        ▼
sample  images [T, 3, 512, 512] uint8   sequence_mask [T] bool   labels [T] int64   (T at most 50 frames)
        │ collate_padded_sequences(): black frames, mask False and label -100 up to the longest sequence
        ▼
batch   images [16, T, 3, 512, 512] (float32 after prepare_batch())   sequence_mask [16, T]   labels [16, T]
        │ encode_frames(): Backbone on the real frames only [N, 64, 32, 32], average pooling, zeros on padding
        ▼
frames  [16, T, 64] ─> GRU 1x128, packed ─> [16, T, 128] ─> ClassificationHead [64] ─> class_logits [16, T, 71]
        │ compute_sequence_labelling_loss(): cross entropy of the labelled frames only (ignore_index -100),
        │ class_weights (class_weighting "none") and label_smoothing 0.0; batch metric compute_frame_accuracy()
        ▼
test    frame metrics: Accuracy, Balanced Accuracy, Macro F1, Top-5 Accuracy, Macro ROC AUC, Macro Average Precision
        sequence metrics: Sequence Exact Match, Edit Score, Segment F1@10, Segment F1@25, Segment F1@50
        ▼
Evaluation/Iteration_<n>/   class_metrics.csv   class_thresholds.json   confusion_matrix.csv   predictions.csv
                            confusion_matrix.png   roc_curves.png   precision_recall_curves.png   (per_class_report)
                            Samples/<sequence>.png   Errors/<TRUE>-TO-<PREDICTED>/<sequence>_frame<i>_<score>.png
```

### Files To Know
To find your way in the code, the table below lists the file and main functions of every role.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/sequence_labelling_trainer.py`, on `sequence_classification_trainer.py` and `classification_trainer.py` | `SequenceLabellingTrainer`: `compute_batch_metrics`, `accumulate_evaluation_batch`, `dump_errors`, `build_sample_image`, `build_prediction_rows`, `compute_extra_headline_metrics`, `compute_segment_results`, `compute_recall_intervals` |
| Dataset and loaders | `vision_training/data/sequence/sequence_dataset.py`, `vision_training/data/loaders.py` | `SequenceDataset` (`label_type="frame"`), `collate_padded_sequences`, `get_sequence_dataloaders` |
| Model | `vision_training/models/sequence/sequence_labeller.py`, `sequence_model.py`, `frame_encoder.py`, `temporal_modules.py` | `SequenceLabeller`, `SequenceFrameModel`, `encode_frames`, `build_temporal_module`, `RecurrentTemporalModule` |
| Head | `vision_training/models/classification/classification_head.py` | `ClassificationHead`, `build_global_pooling`, `compute_pooled_features` |
| Loss | `vision_training/losses/sequence_losses.py`, `vision_training/losses/classification_losses.py` | `compute_sequence_labelling_loss`, `compute_balanced_class_weights` |
| Metrics and evaluation | `vision_training/evaluation/sequence/sequence_metrics.py`, `segment_metrics.py`, `vision_training/evaluation/classification/`, `vision_training/evaluation/recall_intervals.py` | `compute_frame_accuracy`, `compute_sequence_exact_matches`, `group_sequence_predictions`, `compute_segment_metrics`, `compute_class_score_curves`, `find_class_thresholds`, `sum_counts_by_group`, `compute_bootstrap_recall_intervals` |
| Configuration checks | `vision_training/configuration_processing/sequence_checks.py` | `check_sequence_labelling_configuration`, `check_sequence_settings`, `check_temporal_model_configuration` |
| Run name | `vision_training/configuration_processing/run_name.py` | `build_sequence_sections`, `build_temporal_tokens` |
| Synthetic data | `vision_training/data/synthetic/create_synthetic_dataset.py` | `generate_shape_change_sequence`, `CHANGE_CATEGORIES` |
| Export | `vision_training/deployment/export_specifications.py`, `vision_training/models/sequence/inference_sequence_model.py`, `vision_training/deployment/input_contract.py` | `InferenceSequenceModel`: `sequence_labeller_<format>`, `images` [B, T, 3, S, S] to `class_probabilities` [B, T, K], no sequence mask; `build_input_contract`: `preprocessing.json` (`BGR`, `keep_ratio`, `NTCHW`, `0-255`) |
| Export sizes at launch | `vision_training/deployment/export_size_estimates.py` | `estimate_export_sizes`, `probe_inference_graph`, `print_export_size_summary` |

### Run It
If one wishes to run the whole task, generate the data, train, then evaluate the checkpoints.
```bash
# Generate 512 pixel frames, 250,000 / 50,000 / 50,000 sequences of 2 to 12 frames
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=sequence_labelling --output_folder="$HOME/Data/sequence_data/synthetic_changing_shapes"

# Train, after setting experiment_folder, the three data folders and Data.label_mapping.mapping_file_path
uv run python -m vision_training.main --configuration_path="vision_training/configurations/sequence_labelling/configuration_synthetic_sequences.yaml"

# Evaluate every checkpoint into <run>/Checkpoint_Evaluation/ (split, batch size and the other settings are arguments too)
uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder <run folder> --iteration all
```
`scripts/data/create_sequence_split_subset.py` writes `test_10000` (10,000 of the 50,000 test sequences, seed 0), the test split of `documentation/architecture_benchmarking/sequence_labelling_experiment_plan.md`.

```bash
# Write the subset, its frames linked to those of the test split (--number_sequences and --seed change the draw)
uv run python -m scripts.data.create_sequence_split_subset --source_split_folder="$HOME/Data/sequence_data/synthetic_changing_shapes/test" --output_split_folder="$HOME/Data/sequence_data/synthetic_changing_shapes/test_10000"
```

Should you start on sequences of your own, `configuration_example.yaml` holds the settings kept by that plan: learning rate 1e-3, `class_weighting: "balanced"`, a bidirectional LSTM of 2 layers of 256, `flatten` frame pooling, batch normalization. The plan kept 2 layers and `flatten` with a pretrained backbone (1 layer and `max` from scratch); the example leaves it off (`Backbone.pretrained.experiment_folder: null`). Its 384 pixel frames end at 64 x 6 x 6, 2,304 values per frame, for `M2M_IN-KAR-384_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64.4x64-LMP_T50_LSTM2x256-BI_HFLAT.256-CWB_MC71` and 7,486,375 parameters (backbone 512,864, LSTM 6,823,936, head 149,575). The plan trained 256 pixel frames.

### Configuration Keys To Know First
Before editing a configuration, the table below gives the keys that change this task most.

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `ExperimentConfiguration.task` | `"sequence_labelling"` | `SequenceLabellingTrainer`, the M2M checks and the `M2M` run name code |
| `Data.sequence_settings.maximum_sequence_length` | `50` | longest sequence fed whole (`T50` token); a longer one is left out |
| `TemporalModel.type` | `"gru"` | `gru`, `lstm`, `transformer` or `temporal_convolution`; poolings are refused |
| `TemporalModel.bidirectional`, `TemporalModel.causal` | `False`, `False` | bidirectional (`-BI`) also sees later frames; `causal` (`CAU`) only earlier ones; both together are refused, and so is a causal module (or a one-way GRU or LSTM) with `feature_map_normalization: "sequence"` |
| `ClassificationHead.global_pooling` | `"average"` | 64 features per frame; `flatten` keeps the positions (65,536 features at 512 pixels) |
| `ClassificationHead.class_weighting` | `"none"` | `"balanced"`: weight N / (K x n_c) over the labelled training frames (`CWB`) |
| `ClassificationHead.label_smoothing` | `0.0` | smoothing of the per-frame cross entropy (`LS<rate>` token) |
| `ExperimentConfiguration.evaluation.per_class_report` | `True` | per-class tables and scalars, confusion matrix image, curves |
| `ExperimentConfiguration.evaluation.save_predictions` | `True` | `predictions.csv`, one row per labelled frame |
| `ExperimentConfiguration.evaluation.error_dumps` | `activated: False`, `maximum_per_class_pair: 20` | one image per wrong frame in `Errors/<TRUE>-TO-<PREDICTED>/` |

### What A Run Writes
Whenever a checkpoint of the labeller is evaluated, the run folder, TensorBoard and MLflow receive the following.

| Output | Content |
|---|---|
| `Results.txt` | one block per checkpoint (Step 4.13): frame metrics, sequence metrics, the segment note; with `per_class_report` the `Per Class Metrics` and `Per Class Recall` tables (column `Frames`, recall with its 95 % interval, worst first) |
| `Evaluation/Iteration_<n>/` | `class_metrics.csv` (`samples` are labelled frames), `class_thresholds.json`, `confusion_matrix.csv` (above 1,000 classes `confusion_cells.csv`); with `per_class_report` `confusion_matrix.png`, `roc_curves.png`, `precision_recall_curves.png`; with `save_predictions` `predictions.csv`, one row per labelled frame |
| `Evaluation/evaluation_summary.csv`, `evaluation_curves.png` | one row per evaluated iteration, increasing, every scalar; the curves of `Accuracy`, redrawn after each evaluation (Step 4.13) |
| `Samples/<sequence>.png` | frame grid of 8 fixed test sequences, 4 frames per row, every frame captioned `true/predicted`, `X` on the wrong ones |
| `Errors/<TRUE>-TO-<PREDICTED>/<sequence>_frame<index>_<score>.png` | one image per wrong frame, with `error_dumps.activated` |
| TensorBoard | `Train/` and `Validation/` losses and `Accuracy` (labelled frames); `Test/` the global and per-class metrics, the samples and the per-class report images |
| `data.json` | the `frames` of every class in every split |

### Pitfalls
In case a run looks good or bad for the wrong reason, check these.
- **`no_change` everywhere keeps a fair frame accuracy.** The former README run (`readme_screenshots`, this configuration, 2000 iterations from scratch, 2026-10-23, October dataset, deleted since) gave Accuracy 46.97 % (the share of `no_change` frames), Balanced Accuracy 1.41 % (1/71), Macro F1 0.90 % and Sequence Exact Match 5.89 % (the 589 sequences without change). Read the balanced accuracy, the macro F1 and the segment metrics. That run has no class weights; in the plan, a launch without them stayed at the macro F1 of an all-`no_change` model for 6000 iterations, so the plan kept `class_weighting: "balanced"`.
- **Unmapped categories get the label -100.** They are left out of the loss and the metrics, and end a segment. A sequence without one category per frame is refused (Step 2.8). A batch without labelled frame gives a loss of 0; a test sequence without one is left out of `Sequence Exact Match` and of `<S> Test Sequences`.
- **`label_smoothing` above 0 with `class_weighting: "balanced"`** pulls every frame towards the rarest classes: on the October training split, `no_change` would weigh 0.030 against 1.82 to 1.89 for a change class (819,173 of 1,749,690 frames).
- **`test_10000` copies the labels and links the frames** of `synthetic_changing_shapes/test`: generating the dataset again makes its labels name other frames without error. The full 50,000 test sequences took 26 minutes to evaluate at 256 pixels.
- **No windows.** A sequence over `maximum_sequence_length` is left out, never cut (Step 2.8). A step runs the backbone on up to `batch_size` x `maximum_sequence_length` frames.
- **The TensorRT FP32 engine misses the tolerance of the export check.** TensorRT runs it with TF32, which moves the probabilities over 50 frames far beyond 1e-4. The check accepts it within 4 times the difference TF32 makes in the eager model ([TensorRT check](../reference/tools_reference.md#tensorrt-check)). The FP16 engine of the tutorial run fails its check (2.53e-02 against 1e-02): export it in FP32.

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
So that the whole run fits on one page, the tree below lists the functions specific to this task in call order, with the stage that details them.

```text
main()  --configuration_path to a sequence_labelling configuration
├── load_experiment_configuration()  [STAGE 1]
│   ├── build_label_dictionary()                      {category id: class index}, one category per frame
│   └── check_sequence_labelling_configuration()      check_sequence_settings(), check_temporal_model_configuration()
├── SequenceLabellingTrainer(...)  [STAGE 2]          LABEL_TYPE "frame", CLASS_SUPPORT_COLUMN "Frames"
│   ├── get_sequence_dataloaders(label_type="frame") -> SequenceDataset   -100 for an unmapped category
│   ├── get_test_sample_classes() + choose_sample_identifiers()          a sequence counts for its most frequent class
│   └── SequenceLabeller.__init__()   build_temporal_module(keep_time_axis=True), ClassificationHead on every frame
└── run_training_loop()  [STAGE 3]
    ├── get_next_batch() -> collate_padded_sequences()                   labels [B, T], -100 on padding
    ├── augment_batch()                                                  shuffle_frames() moves the labels too
    ├── SequenceLabeller.forward()   encode_frames() [B, T, F] -> temporal_module [B, T, F'] -> head [B, T, K]
    │   └── compute_sequence_labelling_loss()                            labelled frames only
    ├── compute_batch_metrics() -> compute_frame_accuracy()
    └── run_test_evaluation()  [STAGE 4]
        ├── accumulate_evaluation_batch() -> compute_sequence_exact_matches(), dump_errors()
        ├── save_sample_images() -> build_sample_image()
        ├── compute_extra_headline_metrics() -> compute_segment_results()   Edit Score, Segment F1@10/25/50
        ├── compute_recall_intervals()       sum_counts_by_group() + compute_bootstrap_recall_intervals()
        ├── write_predictions_file() -> build_prediction_rows()           one row per labelled frame
        └── report_evaluation()
scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]    same as every task
```

---

## Stage 1 — Entry Point and Configuration Loading

> Real outputs of the loader on `configuration_synthetic_sequences.yaml`, its data folders and mapping file redirected to the generated dataset.

### Step 1.1 — `main()`
The entry point of every task (`classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The sections are those of the sequence classification (`sequence_classification_workflow.md`, Step 1.2).

**Example: the sections, the sequence settings and the temporal module**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, ClassificationHead
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'target_precision': 0.99,
                           'save_predictions': True, 'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
sequence_settings:        {'maximum_sequence_length': 50}
TemporalModel:            {'type': 'gru', 'hidden_size': 128, 'number_layers': 1, 'number_heads': 4, 'kernel_size': 3, 'dropout_rate': 0.0,
                           'bidirectional': False, 'causal': False, 'positional_embedding': True, 'pretrained': {...}}
```

### Step 1.3 — `load_configuration()`
```text
number_iterations: '1e10'      <- still a string
```

### Step 1.4 — `build_label_dictionary()`
The categories of the manifest are the frame classes.
```text
dict: {1: 0, 2: 1, 3: 2, 4: 3, ..., 70: 69, 71: 70}      71 entries, 71 distinct classes
```

### Step 1.5 — `check_configuration()`
After the common checks, `check_sequence_labelling_configuration()` runs those of the sequence classification. The temporal module must keep one vector per frame, so the poolings are refused. A causal module classifies a frame as soon as it arrives; it is refused with `feature_map_normalization: "sequence"`, whose statistics see every frame.

**Example: `TemporalModel.type: "mean_pooling"` and `maximum_sequence_length: 0`**
```text
Invalid Configuration File : configuration_wrong.yaml
  - 'Data.sequence_settings.maximum_sequence_length' must be an integer >= 1, got 0.
  - 'TemporalModel.type' must be one of ['gru', 'lstm', 'transformer', 'temporal_convolution'] for this task, got 'mean_pooling'.
SystemExit code 1
```
A `sequence_settings` section without `maximum_sequence_length` gives `Missing configuration key 'Data.sequence_settings.maximum_sequence_length'.`

### Step 1.6 — `get_trainer_class()`
```text
get_trainer_class(task="sequence_labelling") -> <class 'vision_training.trainers.sequence_labelling_trainer.SequenceLabellingTrainer'>
```

---

## Stage 2 — Trainer Initialization

> Real outputs of building the trainer on the redirected `configuration_synthetic_sequences.yaml` (512 pixel frames, at most 50 frames, batch size 16). The construction took 0.1 s on CPU.

### Step 2.1 — `SequenceLabellingTrainer.__init__()`
No constructor of its own: the sequence classification trainer with `LABEL_TYPE = "frame"` and `CLASS_SUPPORT_COLUMN = "Frames"`. It overrides the batch metric, the evaluation accumulator, the error and sample images, the predictions file, the extra headline metrics and the recall intervals (`compute_recall_intervals()`, `RECALL_INTERVAL_DESCRIPTION`).
```text
self.top_k = 5   <- TOP_K_ACCURACY_CLASSES     SAMPLE_NAME = 'Sequences'   COUNTS_FRAMES = True   TASK_NAME = 'sequence_labelling'
```

### Step 2.2 — `BaseTrainer.__init__()`
```text
self.device = device(type='cpu')   self.per_class_report = True   self.number_sample_images = 8   self.input_image_size = 512
self.batch_size = 16   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
Forced on the CPU here; a normal run on this machine prints `cuda`.
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The sections of the sequence classification (`sequence_classification_workflow.md`, Step 2.4): `T50` (maximum length), `GRU1x128` (one layer of 128, not bidirectional), `CAU` after a causal transformer or temporal convolution, `MC71` for the 71 frame classes.
```text
M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71
```

### Step 2.5 — `DataAugmenter()`
```text
activated True   probability 0.4   independent_frames False   shuffle_frames {'activated': False, 'probability': 0.25}
6 of 28 types activated: random_crop, random_left_right_flip, random_rotation, brightness_contrast_saturation, random_noise, random_haze
```

### Step 2.6 — `setup_training_paths()`
As for every task (`classification_workflow.md`, Step 2.6): a resumed run keeps the classes of its `label_dictionary.json` (`restore_run_label_dictionary()`), then the run folder gets the configuration copy and `label_dictionary.json`.
```text
<run name>/   Tensorboard/   Weights/   label_dictionary.json (977 bytes)   training_configuration.yaml (30,689 bytes)
```

### Step 2.7 — `build_dataloaders()` → `get_sequence_dataloaders(label_type="frame")` + `save_data_record()`
Every sequence is one sample. Loader rules are those of `classification_workflow.md`, Step 2.7 and `sequence_classification_workflow.md`, Step 2.8. `save_data_record()` writes `data.json`: the frames of every class per split, and per split the data folder, the SHA-256 of `labels.json` and the images.
```text
train        500 sequences -> 32 batches of 16 (one epoch = 32 iterations)
validation    60 sequences ->  4 batches of 16       test   60 sequences ->  4 batches of 16
collate      functools.partial(<function collate_padded_sequences>, members_per_sample=1)
data.json    classes[0] = {'class_index': 0, 'class_name': 'no_change', 'category_ids': [1],
                           'frames': {'train': 1609, 'validation': 193, 'test': 231}}
```
This record comes from a dataset generated again with the same settings (3,401 training frames instead of 3,388).

### Step 2.8 — `SequenceDataset.__init__(split_name=...)`
Sequences are read as in `sequence_classification_workflow.md`, Step 2.9, but every sequence gives `frame_category_ids`. The record keeps the class of each frame, -100 if unmapped. A list of the wrong length raises a `ValueError` listing the first 20 such sequences (`MAXIMUM_LISTED_INVALID_SEQUENCES`, `list_invalid_sequences()`). A sequence over `maximum_sequence_length` is left out with a red warning; `split_name` names the split.
```text
labels.json:  {'id': 1, 'name': 'train_000000', 'frames': [... 10 frames], 'frame_category_ids': [1, 1, 50, 16, 10, 36, 6, 8, 1, 1]}
dataset[0]:   images (9, 3, 512, 512) torch.uint8        <- real frames only, padded by the collate
              sequence_mask (9,) torch.bool, all True    labels (9,) torch.int64 [0, 0, 46, 20, 0, 0, 0, 41, 46]
get_class_counts() = [1600, 23, 25, 22, 29, ..., 27, 16, 20]      <- labelled frames of every class, 3,388 in all
with maximum_sequence_length 10:
15 Sequences Of The Test Split Are Longer Than The Maximum Sequence Length (10) And Are Left Out Of The Test.
count_split_sequences() = (60, 427, 45)          <- sequences and frames of the manifest, sequences kept
```

### Step 2.9 — `build_native_loader()`
Recorded with 8 workers and pinned memory; every loader now has 1 worker and pins its batches only on a discrete GPU.
```text
len(loader) = 32   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.10 — `get_test_sample_classes()` + `choose_sample_identifiers()`
Every test sequence counts for its most frequent labelled class (lowest index on a tie). Here all 60 fall in the `no_change` group, so the 8 samples are spread over it: `{0, 8, 17, 25, 34, 42, 51, 59}`.

### Step 2.11 — `build_model()` → `SequenceLabeller.__init__()`
The parts of the sequence classifier (`sequence_classification_workflow.md`, Steps 2.13 to 2.16), but the temporal module is built with `keep_time_axis=True`: one vector per frame, classified by the head. With `class_weighting: "balanced"`, the weights come from the labelled frames.
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          RecurrentTemporalModule     74,496 parameters      GRU(64, 128, batch_first=True), output_features 128
model.classification_head      ClassificationHead          12,871 parameters      Dropout -> Linear(128, 64) -> ReLU -> Dropout -> Linear(64, 71)
model.class_weights            buffer [71] torch.float32 [1.0, 1.0, 1.0, ...]
total                                                     271,015 parameters
```
The GRU reads the frames in order only, so a frame vector depends on the earlier frames. A bidirectional GRU would give 256 features per frame. The two dropout layers are built even at rate 0, so the weight names do not depend on it.

### Step 2.12 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
```text
pretrained_checkpoint_paths = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
MetricTracker: batch_size 16, validation_iteration_ratio 25, sample_name 'Sequences', counts_frames True
```

### Step 2.13 — `print_experiment_information()` (→ `describe_split()` → `count_split_sequences()`)
Every split line gives its sequences and frames, then the sequences kept and the maximum length.
```text
- Train Data Folder       : train (500 sequences, 3388 frames, 500 sequences of at most 50 frames)
- Test Data Folder        : test (60 sequences, 427 frames, 60 sequences of at most 50 frames)
- Iterations Per Epoch    : 32
```

### Step 2.14 — `print_summary()` / `view_input_batches()` / `restore_last_model()`
The panels of `sequence_classification_workflow.md`, Step 2.21, with 128 temporal features and 71 classes. Blocks 2 and 4 change the channel count, so they have no residual shortcut. With `view_input_images`, every frame is captioned with its class (`get_frame_captions()`, `none` if unmapped).
```text
Model Size: Backbone 183,648, Temporal Module 74,496, Classification Head 12,871, Total Model 271,015 parameters (1.09 MB)
get_sample_caption() -> 'Classes : no_change > disappearing_parallelogram > appearing_l_shape > no_change > ...'
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```

---

## Stage 3 — Training Loop

> Real outputs of a CPU run of the shortened configuration (64 pixel frames, batch size 4, 12 iterations; see the top of this document). Run name `M2M_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71`; an epoch is 125 iterations.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (`classification_workflow.md`, Step 3.1). The table shows what each iteration does after its training step.

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | ✔ | nothing: iteration 12 is saved |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |
| Test evaluation | | | | | | ✔ | | | | | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()` / `start_experiment_tracking()`
```text
Tensorboard/Train/        Total Loss, Classification Loss, Accuracy, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Accuracy                   (iteration 10)
```

### Step 3.3 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12, so checkpoint N holds N steps (`classification_workflow.md`, Step 3.6).
```text
Weights/Iteration_3/model_000003.pt   3,345,255 bytes   (the same for 6, 9 and 12)
model_state: 90 tensors, the first 'class_weights'      format_version 2
```
The sizes predate the label dictionary in checkpoints; with it, a checkpoint of the deleted README run held 3,348,821 bytes.

### Step 3.4 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration (`classification_workflow.md`, Step 3.7); see Stage 4.

### Step 3.5 — `get_next_batch()` → `SequenceDataset.__getitem__()` + `collate_padded_sequences()`
As in `sequence_classification_workflow.md`, Step 3.7, the label being the `[T]` class of every frame. `collate_padded_sequences()` pads to the longest sequence: black frames, `False` in the mask, -100 in the labels.
```text
images               [4, 12, 3, 64, 64] torch.uint8
sequence_mask        [4, 12] torch.bool     real frames per sequence [7, 12, 11, 3]
labels               [4, 12] torch.int64
                     [[0, 35, 0, 43, 0, 0, 0, -100, -100, -100, -100, -100],
                      ...
                      [0, 0, 61, -100, -100, -100, -100, -100, -100, -100, -100, -100]]
```

### Step 3.6 — `MetricTracker.count_frames()`
The real frames of the three intervals; the validation batch does not add to the count.
```text
interval_frames = 109 (iterations 1 to 4: 33, 59, 83, 109), 116 (iterations 5 to 8), 93 (iterations 9 to 12)
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
```text
images [4, 12, 3, 64, 64] torch.uint8 -> torch.float32     labels [4, 12] torch.int64     sequence_mask [4, 12] torch.bool
```

### Step 3.8 — `augment_batch()`
The augmentation of the sequence classification (`sequence_classification_workflow.md`, Step 3.12). When `shuffle_frames` permutes the real frames, the frame labels are gathered in the same order, so every frame keeps its class.
```text
iteration 1   images: changed per sample [False, False, False, True]   labels: changed per sample [False, False, False, False]
iteration 2   images: changed per sample [False, False, True, False]   labels: unchanged
```

### Step 3.9 — `forward_pass()` → `SequenceLabeller.forward(sequence_mask=..., labels=...)`
`encode_frames()` gives `[B, T, F]` (`sequence_classification_workflow.md`, Step 3.14). The GRU reads the packed real frames (`pack_padded_sequence`) and returns its output at every frame, padded back to `[B, T, F']` with zeros. The head classifies every frame vector.
```text
encode_frames       -> (4, 12, 64)    zero vectors per sequence [5, 0, 1, 9]       <- the padding frames
temporal_module     -> (4, 12, 128)   zero vectors per sequence [5, 0, 1, 9]
classification_head -> (4, 12, 71)
returned: class_logits [4, 12, 71], class_probabilities [4, 12, 71], classification_loss 4.3000   <- close to ln 71 = 4.263
```

### Step 3.10 — `compute_sequence_labelling_loss()`
The softmax cross entropy of every labelled frame (`ignore_index` -100), with `class_weights` and `label_smoothing`. The sum is divided by the class weights of the labelled frames, clamped above 0, so a batch without labelled frame gives 0, not NaN.

### Step 3.11 — `compute_batch_metrics()` → `compute_frame_accuracy()`
The share of labelled frames whose argmax is right.
```text
accuracy = 0.0          (0 of the 7 + 12 + 11 + 3 = 33 labelled frames right)
```

### Step 3.12 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
The accuracy rises as the network learns to predict `no_change`, the most frequent class.
```text
Total Loss   1: 4.3000, 2: 4.2443, 3: 4.2126, 4: 4.1890, ..., 11: 4.0403, 12: 3.8194
Accuracy     1: 0.0000, 2: 0.0000, 3: 0.0000, 4: 0.0385, 5: 0.1304, 6: 0.2424, 7: 0.3333, 8: 0.5000, ..., 12: 0.4545
```

### Step 3.13 — `training_loss.backward()` + `step_optimizer()` + validation pass
The gradients are applied as they are; their norm was measured for this example.
```text
gradient norm: iteration 1: 0.9540   iteration 2: 0.9521   iteration 3: 0.7110        learning rate 0.001 (constant)
Tensorboard/Validation at iteration 10: Total Loss 4.1790, Classification Loss 4.1790, Accuracy 0.0 (0 of 18 labelled frames)
```

### Step 3.14 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The batch accuracies are averaged over the batches of the interval (4.2365 and 0.0096 at iteration 4; MLflow also gets `train/frame_throughput` 12.2612). The summary of iteration 12:
```text
|   Iteration: 12    |     Total Sequences: 48      |    Interval Sequences: 16   |   Throughput: 2.6 Sequences Per Second   |    15.3 Frames Per Second   |
Moving Average of Train Total Loss                     :     3.9174    -0.1656 Best
Moving Average of Train Accuracy                       :    44.86 %   +14.70 % Best
Moving Average of Valid Accuracy                       :     0.00 %
```
The interval of iteration 8 took 00:01:06, since it includes the test evaluation of iteration 6.

### Step 3.15 — end of the loop
Iteration 12 is already saved and evaluated, so the loop saves and evaluates nothing more (`classification_workflow.md`, Step 3.25).
```text
Last Checkpoint Saved At Iteration 12. Exiting Training.
M2M_IN-KAR-64_..._MC71/
├── Evaluation/         Iteration_6/, Iteration_12/
├── Tensorboard/        Train/, Validation/, Test/
├── Weights/            Iteration_3/, Iteration_6/, Iteration_9/, Iteration_12/, full-checkpoint
├── data.json, label_dictionary.json, mlflow_run.json, Results.txt, training_configuration.yaml
```
The evaluation files sent to MLflow are listed in `TRACKED_EVALUATION_FILES` of `base_trainer.py`.

---

## Stage 4 — Test Evaluation of a Checkpoint

> Real outputs of the evaluation of iterations 6 and 12 of the Stage 3 run on its test split (60 sequences, 427 labelled frames), with `error_dumps` on and `maximum_per_class_pair: 2`. Iteration 6 gives `disappearing_hollow_ellipse` to every frame; iteration 12 gives `no_change` to 307 frames and `disappearing_hollow_ellipse` to 120.
>
> They predate the ROC and precision-recall curves (2026-10-20) and the recall intervals (2026-11-01). With `per_class_report`, the current code adds per class the curves (`compute_class_score_curves()`, `classification_workflow.md`, Steps 4.14 and 4.15), `Macro ROC AUC` and `Macro Average Precision`, the `ROC AUC` and `Average Precision` columns and scalars, and a `Per Class Recall` table; `recall_lower` and `recall_upper` are written whatever the setting.

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (`classification_workflow.md`, Step 4.1).
```text
Test Evaluation Iteration 12: 100%|██████████| 15/15 [00:08<00:00,  1.86it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
It keeps only the labelled frames: probabilities, labels, sequence and position (for the predictions file and the segments), and per sequence whether all its labelled frames are right. It also holds the error image folder and the counts per class pair.
```text
'probabilities':  15 tensors, the first [20, 71] torch.float64     'exact_matches': 15 tensors, the first [4] torch.bool
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
The first test batch, whose longest sequence has 6 frames:
```text
class_logits [4, 6, 71] torch.float32     class_probabilities [4, 6, 71] torch.float32
```

### Step 4.4 — `accumulate_evaluation_batch()` → `compute_sequence_exact_matches()`
It keeps the labelled frames (`labels != -100`) with their sequence and position. `compute_sequence_exact_matches()` tells, for every sequence holding a labelled frame, whether all are right. `dump_errors()` then runs while the frames are in the batch.

### Step 4.5 — `dump_errors()` → `build_frame_prediction_captions()` + `build_frame_grid()`
Every wrong frame gives one image: the grid of its sequence, every frame captioned with its number and `true/predicted` (`X ` before a wrong frame, `none` for an unlabelled one), under a bar `frame <index> : <true> -> <predicted> <score>`. It is saved as `Errors/<TRUE>-TO-<PREDICTED>/<sequence>_frame<index>_<score>.png`, at most `maximum_per_class_pair` per pair.
```text
Evaluation/Iteration_12/Errors/
├── appearing_circle-TO-no_change/                 test_000041_frame3_0.019.png   test_000041_frame9_0.020.png
├── no_change-TO-disappearing_hollow_ellipse/      test_000032_frame0_0.017.png   test_000041_frame0_0.017.png
...                                                96 pairs, 158 images
report line: "Error Images : 158 in 96 class pairs, at most 2 per pair"
```
Iteration 6 gives 133 images in 68 class pairs.

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
The frame grid of the sequence (`FRAME_GRID_COLUMNS = 4`), every frame captioned `true/predicted` with `X ` on wrong ones, under a bar giving the right labelled frames, green when all are right, red otherwise.
```text
build_sample_image -> (96, 276, 3) uint8 for 2 to 4 frames, (164, 276, 3) for 5 to 8, (232, 276, 3) for 9 to 12
caption: "<right frames>/<labelled frames> labelled frames right (true/predicted)"
```

### Step 4.7 — `compute_evaluation_results()`
The frame metrics of the classification on the labelled frames (Steps 4.8 and 4.9), with `per_class_report` their curves. `compute_extra_headline_metrics()` adds the sequence metrics (Steps 4.10 and 4.11), `compute_recall_intervals()` the recall intervals (Step 4.9), `describe_extra_notes()` the segment counts. Every test frame is evaluated once. The headline metric is the frame accuracy.

**Example: the results of iteration 12** (recorded before the curves)
```text
evaluated_samples = '60 Test Sequences, 427 Labelled Frames'
metrics           = {'Accuracy': 0.2787, 'Balanced Accuracy': 0.0084, 'Macro F1': 0.0067, 'Top-2 Accuracy': 0.3466,
                     'Sequence Exact Match': 0.0, 'Edit Score': 0.1424, 'Segment F1@10': 0.2047, 'Segment F1@25': 0.1151,
                     'Segment F1@50': 0.0384}
class_scalars     = 213 (71 classes x 3): Precision/no_change, Recall/no_change, F1/no_change, ...
```
With the curves, a class needs test frames for its `ROC AUC` and `Average Precision`, and frames of other classes for the ROC AUC. The deleted README run gave `'Macro ROC AUC': 0.5826, 'Macro Average Precision': 0.0194` and 366 Test scalars (11 global, 5 for each of the 71 classes).

### Step 4.8 — the frame metrics: `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()` + `compute_top_k_accuracy()`
The metrics of the classification (`classification_workflow.md`, Steps 4.9 to 4.11), every labelled frame being one sample. Above 1,000 classes (`MAXIMUM_DENSE_CONFUSION_CLASSES`), only the non-zero cells are kept, as `confusion_cells.csv`.

**Example: how the numbers of iteration 12 follow from its confusion matrix** (307 frames predicted `no_change`, 120 `disappearing_hollow_ellipse`)
```text
accuracy          = 119 / 427                                       = 0.2787   (only no_change frames are right)
no_change         precision = 119 / 307 = 0.3876   recall = 119 / 206 = 0.5777   F1 = 0.4639
balanced accuracy = 0.5777 / 69                                     = 0.0084   (69 classes have test frames)
macro F1          = 0.4639 / 69                                     = 0.0067
```
`appearing_star` and `appearing_semicircle` have no test frame, so they are left out of the averages.

### Step 4.9 — `find_class_thresholds()` + `write_class_metrics_file()` + `save_confusion_matrix_image()`
The frames of a sequence share its content, so an exact interval over frames would be far too narrow. The overridden `compute_recall_intervals()` works per test sequence instead:

| Part | What it does |
|---|---|
| `sum_counts_by_group()` (`vision_training/evaluation/recall_intervals.py`) | adds up the true frames of every class and the right ones per test sequence, checked against the frames of the metrics |
| `compute_bootstrap_recall_intervals()` | 1000 resamples of the test sequences with replacement (fixed seed), sequences kept whole, for the variance of every class recall; a resample without frames of a class does not count for it |
| effective sample size | `p (1 - p) / variance` capped at the class frames, reduced by `(z / t)²`, `t` the Student quantile at the sequences holding the class minus one; the bounds are the exact (Clopper-Pearson) interval at that size |
| zero variance, recall strictly between 0 and 1 | the size is the class frames, reduced the same way |
| recall 0 or 1, or a single sequence | the exact interval over the sequences holding the class, each one sample |

The bounds are written as `recall_lower` and `recall_upper` after `recall` (empty without test frames) and shown in `Per Class Recall`.

In `class_metrics.csv` of iteration 12 (72 lines), `no_change` reads `0,no_change,206,0.387622,0.577670,0.463938,0.020103,0.009709`. In `class_thresholds.json` (`target_precision` 0.99), most classes are `null`. The current header adds `recall_lower`, `recall_upper`, `roc_auc` and `average_precision` (`build_class_metrics_rows()`).

### Step 4.10 — Sequence Exact Match
The share of test sequences whose labelled frames are all right (the `exact_matches` of Step 4.4): the strictest metric of the task. A sequence of unmapped frames only is left out of this share and of `<S> Test Sequences`.
```text
Sequence Exact Match = 0.0      (no test sequence has all its labelled frames right)
```

### Step 4.11 — `compute_segment_results()` → `group_sequence_predictions()` + `compute_segment_metrics()`
Frame metrics miss fragmentation: 10 frames of one class with one wrong frame in the middle keep accuracy 0.9 but give three segments for one event. The segment metrics score the events.

| Function or metric | What it does |
|---|---|
| `group_sequence_predictions()` | sorts the labelled frames by sequence then frame (`np.lexsort`), takes the most probable class, splits per sequence; a frame given twice raises a `ValueError` |
| `extract_segments()` | cuts every sequence into segments of consecutive frames of one class, a gap of unlabelled frames ending a segment |
| **Edit Score** | per sequence, 1 minus the edit distance between predicted and true segment class orders (`compute_edit_distance()`: insert, remove, replace) over the larger number of segments, averaged |
| **Segment F1@10/25/50** | `count_segment_matches()` matches every predicted segment, in order, to the true segment of its class it overlaps most; right when the IoU reaches the overlap and no earlier prediction took that segment; F1 from counts summed over sequences |

**Example: grouped sequences of iteration 12** (0 `no_change`, 68 `disappearing_hollow_ellipse`; a segment is `(class, first frame, last frame)`)
```text
group_sequence_predictions: 427 labelled frames -> 60 sequences
frames [0, ..., 11]  true [0, 0, 42, 0, 0, 0, 0, 23, 32, 0, 3, 0]  predicted [68, 68, 0 x 10]
   true segments [(0, 0, 1), (42, 2, 2), (0, 3, 6), (23, 7, 7), (32, 8, 8), (0, 9, 9), (3, 10, 10), (0, 11, 11)]
   predicted segments [(68, 0, 1), (0, 2, 11)]   edit distance 7   edit score 0.1250
Segments : 355 true and 114 predicted in 60 test sequences
Edit Score 0.1424   Segment F1@10 0.2047   Segment F1@25 0.1151   Segment F1@50 0.0384
```
In the last sequence, 2 predicted segments must become 8 true ones: 1 replaced and 6 inserted, 1 - 7/8 = 0.125.

### Step 4.12 — `write_predictions_file()` → `build_prediction_rows()`   (only with `save_predictions`)
One row per labelled frame, named by its sequence and frame file (427 rows, 76 columns, 305,323 bytes).
```text
sequence,frame,true_class,predicted_class,correct,probability_no_change,probability_appearing_square, ...
test_000032,test_000032/000000.png,no_change,disappearing_hollow_ellipse,0,0.015759,0.013611, ...
test_000032,test_000032/000002.png,no_change,no_change,1,0.017983, ...
```

### Step 4.13 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The block of iteration 12, also appended to `Results.txt`:
```text
Iteration 12 · 60 Test Sequences, 427 Labelled Frames · 2026-10-18 08:48 ───────────────────────────────────────────────
  Accuracy               27.87 %   Balanced Accuracy   0.84 %
  Macro F1               0.67 %    Top-2 Accuracy      34.66 %
  Sequence Exact Match   0.00 %    Edit Score          14.24 %
  Segment F1@10          20.47 %   Segment F1@25       11.51 %
  Segment F1@50          3.84 %
  Segments : 355 true and 114 predicted in 60 test sequences, an unlabelled frame ending a segment
  Error Images : 158 in 96 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences

  Per Class Metrics
   Class                              Frames   Precision    Recall        F1     @ P99 %   @ P99 %
   ...
   no_change                             206     38.76 %   57.77 %   46.39 %      0.0201    0.97 %
```

The deleted README run (iteration 2000, October `test_10000`, 69,918 labelled frames) read Accuracy 46.97 %, Macro ROC AUC 58.26 %, Edit Score 25.81 % and `Segments : 58133 true and 10000 predicted`: it gave `no_change` to every frame. With its wider columns, `build_table_renderable()` narrows the class column from 32 to 30 characters (`disappearing_hollow_parallelo…`).

TensorBoard `Test/` received 222 scalars (9 global + 213 per class) and 9 images; MLflow 233 metrics. `sanitise_metric_name()` writes `Segment F1@10` as `Segment F1_at_10` for MLflow (originals in `metric_names.json`). Then `update_evaluation_curves()` runs as for every task (`classification_workflow.md`, Step 4.18): it adds the row to `Evaluation/evaluation_summary.csv` and redraws `evaluation_curves.png` with the `Accuracy` of every iteration.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> Real outputs of `scripts/evaluation/evaluate_checkpoint.py` on the Stage 3 run (checkpoints 3, 6, 9 and 12, the first three captured before their training step), with `--iteration all`, `--batch_size 16` and a `--test_data_folder` of the first 20 test sequences. Captured before the recall intervals.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (`classification_workflow.md`, Stage 5). Test sequences are read with the `sequence_settings` of the run, so a longer sequence is left out with the warnings of Step 2.8.

### Step 5.2 — trainer construction
Built from the `training_configuration.yaml` of the run, with the classes of its `label_dictionary.json` (`classification_workflow.md`, Step 5.2).
```text
- Test Data Folder        : test_reduced (20 sequences, 161 frames, 20 sequences of at most 50 frames)
```

### Step 5.3 — `get_checkpoint_iterations()`
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
```text
Evaluating : model_000012.pt
MLflow Run Created In The Experiment 'sequence_labelling' : 0a6b17420e9b4be1a981fe2d96db3cfd
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
Iterations 9, 6 and 3 predict one class everywhere and give the same row.
```text
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
  - Evaluation Curves : evaluation_curves.png
  - Best Checkpoint : iteration 12 (Accuracy 29.81 %)

iteration,Accuracy,Balanced Accuracy,Macro F1,Top-2 Accuracy,Sequence Exact Match,Edit Score,Segment F1@10,Segment F1@25,Segment F1@50, ...
3,0.0,0.0,0.0,0.012422360248447204,0.0,0.0,0.0,0.0,0.0, ...
12,0.2981366459627329,0.01309328968903437,0.010264086389393778,0.36024844720496896,0.0,0.12883116883116885, ...
```
The file has 223 columns: the iteration, 9 global metrics, then precision, recall and F1 of the 71 classes (the current code adds the curve columns, `build_summary_columns()`). It is written in `<run name>/Checkpoint_Evaluation/` (`Checkpoint_Evaluation_split_<parent folder>_<split folder>/` for another split, such as this one), rows in increasing iteration order (`classification_workflow.md` Step 5.5).

### Step 5.6 — `run_test_evaluation()` for each checkpoint
The block has the layout of Step 4.13. On the 20 sequences, iteration 12 reads Accuracy 29.81 %, Edit Score 12.88 % and `Segments : 128 true and 38 predicted in 20 test sequences`; header and segment note count the same 20 sequences.
