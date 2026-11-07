# Sequence Labelling Workflow

This document follows one sequence labelling run (many to many, M2M: one class per frame of the sequences) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The sequence labelling trainer is the sequence classification trainer with one class per frame: the steps it shares with the sequence classification (whole sequences, padding of the batches, frame encoder) are explained in `sequence_classification_workflow.md`, and the steps shared by every task in `classification_workflow.md`; here they keep a short explanation and the real output of the labelling run, and the steps specific to the frame labels (labels per frame, temporal module keeping the time axis, frame metrics, segment metrics of the sequences, frame error images) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training. The synthetic dataset of the configuration, `synthetic_changing_shapes` (250,000 training, 50,000 validation and 50,000 test sequences), is too large for short CPU runs, so a small one was generated in a temporary folder with `create_synthetic_dataset(task="sequence_labelling", image_size=128, maximum_objects=10, minimum_frames=2, maximum_frames=12)`: 500 training, 60 validation and 60 test sequences of 2 to 12 frames of 128 pixels. Every frame is labelled `no_change` (always the first frame), `appearing_<shape>` or `disappearing_<shape>`: 71 classes for the 35 shape classes, about half of the frames being `no_change`. The data folders and the mapping file of the configuration were redirected to it.
- **Stages 1 and 2**: `configurations/sequence_labelling/configuration_synthetic_sequences.yaml` (sequences of at most 50 frames fed whole, 512 pixel frames, batch size 16, a GRU), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the error images and the MLflow tracking turned on. After 12 iterations the network predicts mostly `no_change` (71 classes: a first loss near ln 71 = 4.263), so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps. These runs were recorded when the k of the top-k accuracy was a setting (`top_k: 2`), so their reports show `Top-2 Accuracy` where the current code, whose k is fixed at 5, writes `Top-5 Accuracy`.

## Read This First
The sequence labeller gives one class to every frame of a sequence (many to many): every frame goes through the shared backbone and the global pooling of `ClassificationHead`, a temporal module keeping the time axis relates every frame vector to the frames it may see, and the classification head classifies every frame vector, trained with the cross entropy of the labelled frames. The labels are the `frame_category_ids` of `labels.json`, one category per frame, and every sequence is fed whole. Choose it when the answer is a class per frame (where something happens in the sequence); choose the sequence classification (M2O, `sequence_classification_workflow.md`) for one class per sequence, the sequence comparison (CMP) for a before and an after sequence, and the frame selection (SEL, `frame_selection_workflow.md`) when every sequence must give exactly `Selection.frames_per_class` frames of every wanted class, chosen by a decoding over the whole sequence.

### The Task At A Glance
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
| Export | `vision_training/deployment/export_specifications.py`, `vision_training/models/sequence/inference_sequence_model.py` | `InferenceSequenceModel`: `sequence_labeller_<format>`, `images` [B, T, 3, S, S] to `class_probabilities` [B, T, K], no sequence mask |

### Run It
```bash
# Dataset: set task = "sequence_labelling" in the main block of vision_training/data/synthetic/create_synthetic_dataset.py,
# whose output_folders, image_sizes and split_sizes give synthetic_changing_shapes, 512 pixel frames, 250,000 / 50,000 /
# 50,000 sequences of 2 to 12 frames
uv run python -m vision_training.data.synthetic.create_synthetic_dataset

# Training, after setting experiment_folder, the three data folders and Data.label_mapping.mapping_file_path
uv run python -m vision_training.main --configuration_path="vision_training/configurations/sequence_labelling/configuration_synthetic_sequences.yaml"

# Evaluation of the checkpoints of a run: set run_name, experiment_folder, iteration (None, a number or "all"),
# test_data_folder, batch_size, track_evaluation, maximum_error_images_per_class_pair and output_folder in its main block
uv run python -m scripts.evaluation.evaluate_checkpoint
```
`scripts/data/create_sequence_split_subset.py` writes a fixed random subset of a split (`test_10000`, 10,000 of the 50,000 test sequences, seed 0), the test split of the `sequence_labelling/run-*.yaml` launches (`documentation/architecture_benchmarking/sequence_labelling_experiment_plan.md`).

### Configuration Keys To Know First
| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `ExperimentConfiguration.task` | `"sequence_labelling"` | `SequenceLabellingTrainer`, the M2M checks and the `M2M` run name code |
| `Data.sequence_settings.maximum_sequence_length` | `50` | longest sequence fed whole, a longer one being left out with a red warning; `T50` token; positions of the transformer embedding |
| `TemporalModel.type` | `"gru"` | `gru`, `lstm`, `transformer` or `temporal_convolution`; the poolings, which give one vector per sequence, are refused |
| `TemporalModel.bidirectional`, `TemporalModel.causal` | `False`, `False` | a bidirectional GRU or LSTM also sees the frames after every frame (`-BI`, 256 features); `causal` (transformer, temporal convolution, `CAU`) only the frames before it; both together are refused |
| `ClassificationHead.global_pooling` | `"average"` | frame vector of 64 features at any image size; `flatten` keeps the positions (64 x 32 x 32 = 65,536 features at 512 pixels) |
| `ClassificationHead.class_weighting` | `"none"` | `"balanced"`: weight N / (K x n_c) of every class over the labelled training frames (`CWB` token) |
| `ClassificationHead.label_smoothing` | `0.0` | smoothing of the per-frame cross entropy (`LS<rate>` token) |
| `ExperimentConfiguration.evaluation.per_class_report` | `True` | per-class table, per-class recall table and scalars, confusion matrix image, ROC and precision-recall curves |
| `ExperimentConfiguration.evaluation.save_predictions` | `True` | `predictions.csv`, one row per labelled frame |
| `ExperimentConfiguration.evaluation.error_dumps` | `activated: False`, `maximum_per_class_pair: 20` | one image per wrong frame in `Errors/<TRUE>-TO-<PREDICTED>/` |

### What A Run Writes
- `Results.txt`, one block per evaluated checkpoint: `Iteration <n> · <S> Test Sequences, <F> Labelled Frames · <date>`, the frame metrics, `Sequence Exact Match`, `Edit Score`, `Segment F1@10`, `@25` and `@50`, the note `Segments : <true> true and <predicted> predicted in <S> test sequences, an unlabelled frame ending a segment`, then the `Per Class Metrics` and `Per Class Recall` tables (column `Frames`) with `per_class_report`, the second giving the frame recall of every class with its 95% confidence interval, a Korn-Graubard interval from a bootstrap over the test sequences, worst recall first.
- `Evaluation/Iteration_<n>/`: `class_metrics.csv` (its `samples` are labelled frames, `recall_lower` and `recall_upper` follow `recall`), `class_thresholds.json`, `confusion_matrix.csv`, and with `per_class_report` `confusion_matrix.png`, `roc_curves.png` and `precision_recall_curves.png`; `predictions.csv` with `save_predictions`, one row per labelled frame (`sequence`, `frame`, `true_class`, `predicted_class`, `correct`, `probability_<class>`).
- `Samples/<sequence>.png`: the frame grid of 8 fixed test sequences, 4 frames per row, every frame captioned `true/predicted` and `X` on the wrong ones; `Errors/<TRUE>-TO-<PREDICTED>/<sequence>_frame<index>_<score>.png`, one image per wrong frame, with `error_dumps.activated`.
- TensorBoard: `Train/` and `Validation/` `Total Loss`, `Classification Loss` and `Accuracy` (of the labelled frames), `Train/` also `Learning Rate`; `Test/` the global metrics, the `Precision`, `Recall`, `F1`, `ROC AUC` and `Average Precision` of every class, `Samples/<sequence>`, `Confusion Matrix`, `ROC Curves` and `Precision Recall Curves`.
- `data.json` counts the `frames` of every class in every split.

### Pitfalls
- A run predicting `no_change` everywhere keeps a fair frame accuracy: the synthetic configuration trained 2000 iterations (README screenshots, `suggestions.md` K29) gives Accuracy 0.4697 (the share of `no_change` frames of `test_10000`), Balanced Accuracy 0.0141 (1/71), Macro F1 0.0090 and Sequence Exact Match 0.0589 (the 589 sequences without change). Its `predictions.csv` gives `no_change` a probability of 0.9999 on the 10,000 first frames, always `no_change`, and about 0.38 on the 59,918 others, their share of `no_change`: read the balanced accuracy, the macro F1 and the segment metrics.
- The frames of unmapped categories get the label -100: they are left out of the loss and the metrics, and end a segment. Nothing checks that `frame_category_ids` holds one label per frame (`suggestions.md` L10), and a batch without labelled frame gives a NaN loss while the sequences without labelled frame count as exact matches (L11, `sequence_losses.py`, `compute_sequence_exact_matches()`).
- A causal module (`causal: True`, or a GRU or LSTM that is not bidirectional) with `Backbone.normalization.feature_map_normalization: "sequence"` still uses the future frames: the statistics of every backbone layer are those of the whole sequence (`bug_standards.md` B8).
- `label_smoothing` above 0 with `class_weighting: "balanced"` pulls every frame towards the rarest classes (`suggestions.md` C2); on the synthetic training split the balanced weight of `no_change` would be 0.030 against 1.82 to 1.89 for a change class (819,173 of the 1,749,690 frames, `data.json`).
- `test_10000` holds a copy of the labels and a link to the frames of `synthetic_changing_shapes/test`: generating the dataset again makes its labels name other frames without any error (`bug_standards.md` B2, dropped). The test split of the synthetic configuration holds all 50,000 test sequences, whose evaluation took 26 minutes at 256 pixels in the experiment plan.
- `Data.sequence_settings` only takes `maximum_sequence_length`: a sequence is never cut into windows, a longer one is left out with a red warning (Step 2.8), and a step runs the backbone on up to `batch_size` x `maximum_sequence_length` frames (comment of `batch_size` in the configuration).

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


```text
main()
│   Entry point of every task. Give a sequence_labelling configuration with --configuration_path to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   ├── load_configuration(configuration_path)
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category id: class index}; one category per frame, the frames of unmapped
│   │       categories being ignored by the loss and the metrics.
│   └── check_configuration()
│       ├── common checks
│       └── check_sequence_labelling_configuration()
│               ├── the data, head and evaluation checks of the classification
│               ├── check_sequence_settings()             maximum_sequence_length a positive integer, multi_class head
│               └── check_temporal_model_configuration()  an ordered module only (GRU, LSTM, transformer, temporal
│                                                         convolution): the poolings give one vector per sequence;
│                                                         causal allowed
│
├── get_trainer_class(task)  -> SequenceLabellingTrainer
│
├── SequenceLabellingTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   The SequenceClassificationTrainer with LABEL_TYPE = "frame" and CLASS_SUPPORT_COLUMN = "Frames".
│   │
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter()
│       ├── restore_run_label_dictionary() / setup_training_paths()
│       │       The classes of a resumed run kept from its label_dictionary.json, then the run folder with the
│       │       configuration and label_dictionary.json (classification_workflow.md, Step 2.6).
│       ├── build_dataloaders() -> get_sequence_dataloaders(label_type="frame")
│       │   └── SequenceDataset.__init__(split_name=...)
│       │           One sample per whole sequence, each frame keeping the class of its category
│       │           ('frame_category_ids'), -100 for an unmapped category; a sequence longer than
│       │           maximum_sequence_length is left out with a warning.
│       ├── check_loaders_give_batches()             refuses a split whose loader gives no batch
│       ├── save_data_record()                       data.json: classes, data folder, labels SHA-256, images
│       ├── get_test_sample_classes() + choose_sample_identifiers()
│       │       The classes of the labelled frames of every test sequence: a sequence counts for its most frequent
│       │       class.
│       ├── build_model()
│       │   └── SequenceLabeller.__init__()
│       │       ├── build_backbone() + build_global_pooling()
│       │       ├── build_temporal_module(keep_time_axis=True)   one vector per frame, in its temporal context
│       │       ├── ClassificationHead()                         applied to every frame vector
│       │       └── class_weights buffer                         ones, or balanced on the labelled frames
│       ├── initialise_pretrained_components() / build_optimizer() / MetricTracker(counts_frames=True)
│       ├── print_experiment_information() -> describe_split() -> count_split_sequences()
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       ├── view_input_batches() -> show_input_samples()   every frame captioned with its class
│       └── restore_last_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch() -> SequenceDataset.__getitem__() + collate_padded_sequences()
        │   │       labels [B, T]: the class of every frame, -100 on the padding frames, T the longest sequence
        │   │       of the batch; a batch failing to load is dropped and the next one taken
        │   ├── MetricTracker.count_frames()
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch() / augment_batch()   shuffle_frames() moves the frame labels with their frames
        │   │   ├── forward_pass() -> SequenceLabeller.forward(sequence_mask=..., labels=...)
        │   │   │   ├── encode_frames()                  [B, T, F]
        │   │   │   ├── temporal_module                  [B, T, F'] (zeros on the padding frames)
        │   │   │   ├── classification_head              class_logits [B, T, K]
        │   │   │   └── compute_sequence_labelling_loss()   cross entropy over the labelled frames only
        │   │   ├── compute_batch_metrics() -> compute_frame_accuracy()   accuracy over the labelled frames
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()
        │   ├── save_model() on the multiples of weight_saving_iterations and intermediate_weight_saving_iterations
        │   ├── validation pass
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary() + report_dropped_batches()
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations, last    -> [STAGE 4]
        └── save_model() of the steps taken since the last checkpoint, if any; then run_training_loop() calls
                shutdown_loader_workers() and finish_experiment_tracking(), whatever ended the loop

run_test_evaluation(iteration)  [STAGE 4]
├── create_evaluation_accumulator()
│       The probabilities and labels of the labelled frames, the sequence and position of every labelled frame,
│       whether every labelled frame of each sequence is right, and the folder and counts of the error images.
├── for each test batch:
│   ├── prepare_batch() / predict_batch()
│   ├── accumulate_evaluation_batch() -> compute_sequence_exact_matches()
│   │   └── dump_errors() -> build_frame_prediction_captions() + build_frame_grid()   one image per wrong frame
│   └── save_sample_images() -> build_sample_image()   frame grid with true/predicted classes of every frame
├── compute_evaluation_results()
│   ├── the frame metrics of the classification     confusion matrix, accuracy, balanced accuracy, macro F1, top-k,
│   │                                               thresholds; with per_class_report the ROC and precision-recall
│   │                                               curves of every class (compute_class_score_curves()), with
│   │                                               Macro ROC AUC and Macro Average Precision
│   ├── compute_extra_headline_metrics()
│   │   ├── Sequence Exact Match                    share of the sequences whose labelled frames are all right
│   │   └── compute_segment_results()
│   │       ├── group_sequence_predictions()        the labelled frames grouped sequence by sequence, in frame order
│   │       └── compute_segment_metrics()           extract_segments(), compute_edit_distance(),
│   │                                               count_segment_matches(): Edit Score, Segment F1@10/25/50
│   ├── compute_recall_intervals()                  overridden: sum_counts_by_group() adds up the true and right
│   │                                               frames of every class per test sequence, then
│   │                                               compute_bootstrap_recall_intervals() gives the 95% interval of
│   │                                               every frame recall over 1000 resamples of the sequences
│   ├── class_metrics.csv, confusion_matrix.png and save_score_curve_images() (the images with per_class_report)
│   ├── write_predictions_file() -> build_prediction_rows()   one row per labelled frame
│   ├── describe_extra_notes()                      the number of true and predicted segments
│   └── build_recall_interval_table()               the Per Class Recall table (column Frames), worst recall first
└── report_evaluation() -> report_evaluation_results() + track_evaluation()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/sequence_labelling/configuration_synthetic_sequences.yaml`, its data folders and mapping file redirected to the generated dataset.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

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
**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'TemporalModel', 'ClassificationHead']
number_iterations: '1e10'      <- still a string
```

### Step 1.4 — `build_label_dictionary()`
The categories of the manifest are the classes of the frames.

**Example: the label dictionary of the changing shapes**
```text
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '<datasets>/synthetic_changing_shapes/label_mapping.json'}
dict: {1: 0, 2: 1, 3: 2, 4: 3, ..., 70: 69, 71: 70}
71 entries, 71 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_sequence_labelling_configuration()`: the checks of the sequence classification (`maximum_sequence_length` a positive integer, a `multi_class` head), except that the temporal module must model the frame order and keep one vector per frame (the poolings summarise a whole sequence into one vector, so they are refused), and that it may be causal: a causal module only looks at the past frames, so the class of a frame can be given as soon as the frame arrives.

**Example: output with `TemporalModel.type: "mean_pooling"` and `maximum_sequence_length: 0` in the YAML**
```text
------------------------------------------------------
Invalid Configuration File : configuration_wrong.yaml
------------------------------------------------------
  - 'Data.sequence_settings.maximum_sequence_length' must be an integer >= 1, got 0.
  - 'TemporalModel.type' must be one of ['gru', 'lstm', 'transformer', 'temporal_convolution'] for this task, got 'mean_pooling'.
SystemExit code 1
```

**Example: output with a `sequence_settings` section written for an older version (without `maximum_sequence_length`)**
```text
----------------------------------------------------
Invalid Configuration File : configuration_old.yaml
----------------------------------------------------
  - Missing configuration key 'Data.sequence_settings.maximum_sequence_length'.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="sequence_labelling") -> <class 'vision_training.trainers.sequence_labelling_trainer.SequenceLabellingTrainer'>
```


---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on the redirected `configuration_synthetic_sequences.yaml` (512 pixel frames, sequences of at most 50 frames, batch size 16), with the run folder redirected to a temporary folder. The whole construction took 0.1 s on CPU.

### Step 2.1 — `SequenceLabellingTrainer.__init__()`
The labelling trainer has no constructor of its own: it is the sequence classification trainer with `LABEL_TYPE = "frame"` (the loaders read one class per frame) and `CLASS_SUPPORT_COLUMN = "Frames"` (the per-class tables count frames). It overrides the batch metric, the evaluation accumulator, the error images, the sample images, the predictions file and the extra headline metrics, since every frame has its own class, and the recall intervals (`compute_recall_intervals()`, `RECALL_INTERVAL_DESCRIPTION`), the frames of a sequence being correlated.

**Example: the attributes after the construction**
```text
self.sequence_settings          = {'maximum_sequence_length': 50}
self.LABEL_TYPE                 = 'frame'
self.CLASS_SUPPORT_COLUMN       = 'Frames'
self.top_k                      = 5                     <- TOP_K_ACCURACY_CLASSES
self.save_predictions           = True
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
SAMPLE_NAME = 'Sequences'   COUNTS_FRAMES = True   TASK_NAME = 'sequence_labelling'
```

### Step 2.2 — `BaseTrainer.__init__()`
**Example: some attributes set by the constructor**
```text
self.device = device(type='cpu')   self.per_class_report = True   self.number_sample_images = 8   self.input_image_size = 512
self.batch_size = 16   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The sections of the sequence classification (`sequence_classification_workflow.md`, Step 2.4): `T50` (the maximum sequence length), then `GRU1x128` (one layer of 128, not bidirectional), and `CAU` would follow a causal transformer or temporal convolution. `MC71` gives the 71 frame classes.

**Example output**
```text
M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71
```

### Step 2.5 — `DataAugmenter()`
**Example: the settings of the configuration**
```text
activated True   probability 0.4   independent_frames False   shuffle_frames {'activated': False, 'probability': 0.25}
6 of 28 types activated: random_crop, random_left_right_flip, random_rotation, brightness_contrast_saturation, random_noise,
                         random_haze
```

### Step 2.6 — `setup_training_paths()`
As for every task (see `classification_workflow.md`, Step 2.6): a resumed run first keeps the classes of its `label_dictionary.json` (`restore_run_label_dictionary()`), then the run folder is created with the copy of the configuration and `label_dictionary.json`, the class index of every category of the run.

**Example: the run folder after this step**
```text
M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71/
├── Tensorboard/
├── Weights/
├── label_dictionary.json          977 bytes      the class index of every category of the run
└── training_configuration.yaml    30,689 bytes   the copy of the launched configuration file, comments included
```

### Step 2.7 — `build_dataloaders()` → `get_sequence_dataloaders(label_type="frame")` + `save_data_record()`
Every sequence is one sample, so a split gives as many samples as it has sequences (minus those longer than `maximum_sequence_length`, Step 2.8). The training loader leaves out its last incomplete batch only with a batch normalized classification head (`ClassificationHead.fully_connected_normalization: "batch"`, see `sequence_classification_workflow.md`, Step 2.8), which is not the case here. Right after the loaders, `check_loaders_give_batches()` refuses a split whose loader gives no batch, then `save_data_record()` writes `data.json` next to the configuration: the classes with their frames in every split and, for every split, its data folder, the SHA-256 of its `labels.json` and its images, so that the data a model was trained on stays known. `analyse_training_data()` gives no file for this task.

**Example output**
```text
class_names = ['no_change', 'appearing_square', 'appearing_circle', ..., 'disappearing_hollow_semicircle', 'disappearing_hollow_capsule']   (71)
train        500 sequences -> 32 batches of 16 (one epoch = 32 iterations)
validation    60 sequences ->  4 batches of 16
test          60 sequences ->  4 batches of 16
collate      functools.partial(<function collate_padded_sequences>, members_per_sample=1)
data.json    167,356 bytes   keys ['created', 'classes', 'splits']
             classes[0] = {'class_index': 0, 'class_name': 'no_change', 'category_ids': [1],
                           'frames': {'train': 1609, 'validation': 193, 'test': 231}}
             splits['train'] = {'data_folder': '<generated dataset>/train', 'labels_sha256': '40ec1bf2...',
                                'number_images': 3401, 'images': ['train_000000/000000.png', 'train_000000/000001.png', ...]}
```
The record and the run folder of Step 2.6 were taken again with the current code on a dataset generated with the same settings (3,401 training frames instead of the 3,388 of Step 2.8): the first capture held the former `class_names` list instead of `classes`.

### Step 2.8 — `SequenceDataset.__init__(split_name=...)`
The sequences are read as for the sequence classification (`sequence_classification_workflow.md`, Step 2.9), but every sequence gives `frame_category_ids`, one category per frame, and its record keeps the class of each of its frames, -100 for a frame of an unmapped category (ignored by the loss and the metrics). Every sequence is fed whole, so every frame is in exactly one sample. A sequence longer than `maximum_sequence_length` is left out of its split, with one red warning per sequence and a total; `split_name` names the split in these warnings.

**Example: a manifest sequence, and the record of a sequence**
```text
labels.json:  {'id': 1, 'name': 'train_000000', 'frames': ['train_000000/000000.png', ... 10 frames], 'frame_category_ids': [1, 1, 50, 16, 10, 36, 6, 8, 1, 1]}
              (500 sequences of 2 to 12 frames, 3388 frames; 71 categories {'id': 1, 'name': 'no_change'}, {'id': 2, 'name': 'appearing_square'}, ...)
record:       {'identifier': 419, 'sequence_identifier': 420, 'sequence_name': 'train_000419',
               'frames': ['train_000419/000000.png', ... 9 frames], 'frame_labels': [0, 0, 46, 20, 0, 0, 0, 41, 46]}
```

**Example: the first sequence after shuffling, and the sample the loader gets from it** (a sequence of 9 frames)
```text
dataset[0]: images (9, 3, 512, 512) torch.uint8                                     <- the real frames only, padded by the collate
            sequence_mask (9,) torch.bool [True, True, True, True, True, True, True, True, True]
            labels (9,) torch.int64 [0, 0, 46, 20, 0, 0, 0, 41, 46]                 <- one class per frame
            sample_identifiers 419
get_class_counts() = [1600, 23, 25, 22, 29, ..., 27, 16, 20]                        <- labelled frames of every class
```
The class counts add up to the 3,388 frames of the split: no frame is counted twice. `no_change` holds 1,600 of them.

**Example: the warnings of the test split with `maximum_sequence_length: 10`** (shortened)
```text
Sequence 'test_000002' Of The Test Split Has 12 Frames, More Than The Maximum Sequence Length (10): It Is Left Out Of The Test.
... (15 lines: test_000002, 04, 07, 08, 09, 10, 12, 15, 16, 23, 31, 50, 52, 54, 56)
15 Sequences Of The Test Split Are Longer Than The Maximum Sequence Length (10) And Are Left Out Of The Test.
count_split_sequences() = (60, 427, 45)          <- sequences and frames of the manifest, sequences kept
```

### Step 2.9 — `build_native_loader()`
**Example output** (recorded with 8 workers, when the number of workers was a setting; every loader now has 1)
```text
len(loader) = 32   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.10 — `get_test_sample_classes()` + `choose_sample_identifiers()`
Every test sequence gives the classes of its labelled frames, and `choose_sample_identifiers()` groups it under its most frequent class (the lowest class index on a tie), the groups receiving one sample each in turn, in class order. Every sequence starts with a `no_change` frame and seldom holds the same change twice, so all 60 test sequences of this dataset fall in the `no_change` group, and the 8 sample sequences are spread evenly over that group. A sequence holding one change more often than `no_change` forms a group of its own and gets a sample: in the dataset generated again for Step 2.7, `test_000030` (classes `[0, 32, 55, 43, 64, 11, 32, 34, 45, 46]`, `appearing_hollow_nonagon` twice) makes the samples `{0, 10, 19, 29, 30, 40, 49, 59}`.

**Example output**
```text
get_test_sample_classes() -> 60 entries, e.g. {28: [0, 19, 0, 0, 49, 52, 16, 45, 0], 23: [0, 0, 23, 24, 28, 55, 0, 0, 37, 58, 0, 0], 37: [0, 24, 0, 0], 33: [0, 44, 0, 11]}
sample_image_identifiers  -> {0, 8, 17, 25, 34, 42, 51, 59}
names                     -> test_000000, test_000008, test_000017, test_000025, test_000034, test_000042, test_000051, test_000059
```

### Step 2.11 — `build_model()` → `SequenceLabeller.__init__()`
The labeller has the parts of the sequence classifier (`sequence_classification_workflow.md`, Steps 2.13 to 2.16), but its temporal module is built with `keep_time_axis=True`: it returns one vector per frame, the frame seen in its temporal context, and the head classifies every one of them. With `class_weighting: "balanced"`, the class weights come from the labelled frames of every class.

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          RecurrentTemporalModule     74,496 parameters      GRU(64, 128, batch_first=True), output_features 128
model.classification_head      ClassificationHead          12,871 parameters      Dropout -> Linear(128, 64) -> ReLU -> Dropout -> Linear(64, 71)
model.class_weights            buffer [71] torch.float32 [1.0, 1.0, 1.0, ...]
total                                                     271,015 parameters
```
The GRU reads the frames in order only (`bidirectional: False`): the vector of a frame then depends on the frames before it. A bidirectional GRU would give 256 features per frame, seeing the frames after it too. The two dropout layers of the head (`dropout_rate: 0.2`) are built even at a rate of 0, so that the names of the weights do not depend on it.

### Step 2.12 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
**Example output**
```text
pretrained_checkpoint_paths = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
{'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'accuracy': 'Accuracy'}
MetricTracker: batch_size 16, validation_iteration_ratio 25, sample_name 'Sequences', counts_frames True
```

### Step 2.13 — `print_experiment_information()` (→ `describe_split()` → `count_split_sequences()`)
Every split line gives its sequences and frames, then the sequences kept and the maximum sequence length.

**Example output**
```text
- Run Name                : M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71
...
- Data Record             : data.json
- Train Data Folder       : train (500 sequences, 3388 frames, 500 sequences of at most 50 frames)
- Validation Data Folder  : validation (60 sequences, 390 frames, 60 sequences of at most 50 frames)
- Test Data Folder        : test (60 sequences, 427 frames, 60 sequences of at most 50 frames)
- Iterations Per Epoch    : 32
- Batch Size              : 16
...
- MLflow Tracking         : Off
```

### Step 2.14 — `print_summary()` / `view_input_batches()` / `restore_last_model()`
The summary panels are those of the sequence classification (`sequence_classification_workflow.md`, Step 2.21), the temporal module panel giving `Output Features : 128` and the head panel `Input Features : 128` and `Number Of Classes : 71`. In the backbone panel, blocks 2 and 4 change the channel count with 2 layers, so they add no residual shortcut and list `Squeeze-and-Excitation` alone (`ConvolutionBlock.count_residual_additions()`, `bug_standards.md` H4, dropped). With `view_input_images`, every frame of the grids is captioned with its number and its class (`get_frame_captions()`, `none` for an unmapped category), and the information panel of the sequence starts with the order of its classes (`get_sample_caption()`).

**Example: the modules and shapes of the backbone panel, and the model size panel** (the model rebuilt on the CPU)
```text
- Convolution Block 1   Modules : Squeeze-and-Excitation                       [1, 3, 512, 512]  -> [1, 16, 256, 256]
- Convolution Block 2   Modules : Squeeze-and-Excitation                       [1, 16, 256, 256] -> [1, 32, 128, 128]
- Convolution Block 3   Modules : Residual Shortcuts, Squeeze-and-Excitation   [1, 32, 128, 128] -> [1, 32, 64, 64]
- Convolution Block 4   Modules : Squeeze-and-Excitation                       [1, 32, 64, 64]   -> [1, 64, 32, 32]
- Convolution Block 5   Modules : Residual Shortcuts, Squeeze-and-Excitation   [1, 64, 32, 32]   -> [1, 64, 32, 32]
Final Backbone output shape: [1, 64, 32, 32]
Model Size: Backbone 183,648, Temporal Module 74,496, Classification Head 12,871, Total Model 271,015 parameters (1.09 MB)
```

**Example: the captions of the sequence of Step 2.8**
```text
get_sample_caption() -> 'Classes : no_change > disappearing_parallelogram > appearing_l_shape > no_change > disappearing_hexagon > disappearing_parallelogram'
get_frame_captions() -> ['no_change', 'no_change', 'disappearing_parallelogram', 'appearing_l_shape', 'no_change', 'no_change', 'no_change',
                         'disappearing_hexagon', 'disappearing_parallelogram']
```

**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of the redirected `configuration_synthetic_sequences.yaml` shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the error images and the MLflow tracking on. Its run name is `M2M_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71`; with batches of 4 sequences, an epoch is 125 iterations.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1): from `start_iteration` to `number_iterations`, the total of the run, each iteration takes its training step, then saves its checkpoint, runs its validation batch and prints its summary on the iterations set for each, and evaluates its checkpoint last.

**Example: what happens at each iteration of the run**, in the order of the rows within an iteration, after its training step

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | ✔ | nothing: iteration 12 is saved |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |
| Test evaluation | | | | | | ✔ | | | | | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()` / `start_experiment_tracking()`
**Example output**
```text
Tensorboard/Train/        Total Loss, Classification Loss, Accuracy, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Accuracy                   (iteration 10)
MLflow Run Created In The Experiment 'sequence_labelling' : 924c26859f0f4a3dbea80d87b4eb67a6 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.3 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12, so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6), with the label dictionary of the run.

**Example: the checkpoints and the keys of the last one**
```text
Checkpoint Folder :
 - Iteration_3
 - Model Saved In : model_000003.pt
Weights/Iteration_3/model_000003.pt   3,345,255 bytes      Weights/Iteration_6/model_000006.pt   3,345,255 bytes
Weights/Iteration_9/model_000009.pt   3,345,255 bytes      Weights/Iteration_12/model_000012.pt  3,345,255 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
model_state: 90 tensors, the first 'class_weights', then 'backbone.convolution_blocks.0. ...'      format_version 2
```
The file sizes were measured before the checkpoints held the label dictionary of the run (2026-10-21); the checkpoint of iteration 2000 of the README screenshot run of this configuration (`suggestions.md` K), the same 90 tensors with its label dictionary, its MLflow run identifier and the run name of 512 pixel frames, holds 3,348,821 bytes.

### Step 3.4 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration, after its checkpoint and its summary (see `classification_workflow.md`, Step 3.7), see Stage 4.

### Step 3.5 — `get_next_batch()` → `SequenceDataset.__getitem__()` + `collate_padded_sequences()`
The sequence of the sequence classification (`sequence_classification_workflow.md`, Step 3.7: every frame read with `read_image()`, a batch failing to load dropped and the next one taken), its label being the `[T]` class of every frame. `collate_padded_sequences()` pads every sequence of the batch to the longest one: black frames, `False` in the mask and -100 in the labels, so that the padding frames are ignored by the loss and the metrics.

**Example: the first batch of the run** (the longest sequence of the batch has 12 frames)
```text
images               [4, 12, 3, 64, 64] torch.uint8
sequence_mask        [4, 12] torch.bool     real frames per sequence [7, 12, 11, 3]
labels               [4, 12] torch.int64
                     [[0, 35, 0, 43, 0, 0, 0, -100, -100, -100, -100, -100],
                      [0, 0, 52, 36, 35, 0, 46, 0, 0, 0, 0, 0],
                      [0, 0, 55, 0, 0, 36, 48, 58, 0, 39, 32, -100],
                      [0, 0, 61, -100, -100, -100, -100, -100, -100, -100, -100, -100]]
sample_identifiers   [4] torch.int64     [457, 480, 168, 34]
second batch         images [4, 10, 3, 64, 64], labels[0] = [0, 8, 0, 0, 11, -100, -100, -100, -100, -100], sample_identifiers [163, 278, 272, 443]
```

### Step 3.6 — `MetricTracker.count_frames()`
**Example: the real frames of the three intervals**
```text
interval_frames = 109 (iterations 1 to 4: 33, 59, 83, 109), 116 (iterations 5 to 8), 93 (iterations 9 to 12)
```
The validation batch of iteration 10 does not add to the count.

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
**Example: before and after**
```text
images [4, 12, 3, 64, 64] torch.uint8 -> torch.float32     labels [4, 12] torch.int64     sequence_mask [4, 12] torch.bool
```

### Step 3.8 — `augment_batch()`
The augmentation of the sequence classification (`sequence_classification_workflow.md`, Step 3.12). When `shuffle_frames` permutes the real frames of a sequence, it returns the frame order of every sequence, and the frame labels are gathered in the same order, so that every frame keeps its class.

**Example: which sequences of the first three batches were changed** (`probability: 0.4`)
```text
iteration 1   images: changed per sample [False, False, False, True]   labels: changed per sample [False, False, False, False]
iteration 2   images: changed per sample [False, False, True, False]   labels: unchanged
iteration 3   images: unchanged                                         labels: unchanged
```

### Step 3.9 — `forward_pass()` → `SequenceLabeller.forward(sequence_mask=..., labels=...)`
`encode_frames()` gives the `[B, T, F]` frame vectors (`sequence_classification_workflow.md`, Step 3.14). The temporal module keeps the time axis: the GRU reads the packed real frames of every sequence (`pack_padded_sequence`) and returns its output at every frame, padded back to `[B, T, F']` with zeros on the padding frames. The head is applied to every frame vector.

**Example: the first batch through the model**
```text
encode_frames       -> (4, 12, 64)    zero vectors per sequence [5, 0, 1, 9]       <- the padding frames
temporal_module     -> (4, 12, 128)   one vector per frame, zero vectors per sequence [5, 0, 1, 9]
classification_head -> (4, 12, 71)
```

**Example: the dictionary returned in training mode**
```text
class_logits          [4, 12, 71] torch.float32
class_probabilities   [4, 12, 71] torch.float32
classification_loss   4.3000 (scalar)            <- close to ln 71 = 4.263
```

### Step 3.10 — `compute_sequence_labelling_loss()`
The cross entropy of the softmax of every frame, averaged over the labelled frames only: the padding frames and the frames of unmapped categories (-100) are ignored (`ignore_index`). It uses `class_weights` and `label_smoothing` like the loss of the classification.

### Step 3.11 — `compute_batch_metrics()` → `compute_frame_accuracy()`
The share of the labelled frames whose predicted class (the argmax of their logits) is right.

**Example: the first batch**
```text
accuracy = 0.0          (0 of the 7 + 12 + 11 + 3 = 33 labelled frames right)
```

### Step 3.12 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
**Example: the Train curves**
```text
Total Loss   1: 4.3000, 2: 4.2443, 3: 4.2126, 4: 4.1890, 5: 4.1762, 6: 4.0613, 7: 4.0971, 8: 3.9974, 9: 3.8736, 10: 3.9365, 11: 4.0403, 12: 3.8194
Accuracy     1: 0.0000, 2: 0.0000, 3: 0.0000, 4: 0.0385, 5: 0.1304, 6: 0.2424, 7: 0.3333, 8: 0.5000, 9: 0.5556, 10: 0.4643, 11: 0.3200, 12: 0.4545
```
The accuracy rises as the network learns to predict `no_change`, the most frequent class.

### Step 3.13 — `training_loss.backward()` + `step_optimizer()` + validation pass
**Example output**
```text
gradient norm: iteration 1: 0.9540   iteration 2: 0.9521   iteration 3: 0.7110        learning rate 0.001 (constant)
Tensorboard/Validation at iteration 10: Total Loss 4.1790, Classification Loss 4.1790, Accuracy 0.0 (0 of 18 labelled frames)
```

### Step 3.14 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The batch accuracies are averaged over the batches of the interval, each batch having its own number of labelled frames.

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 16.9459, 'classification_loss': 16.9459, 'accuracy': 0.0385, 'interval_frames': 109}
after:  {'interval_batches': 4, 'total_loss': 4.2365, 'classification_loss': 4.2365, 'accuracy': 0.0096, 'interval_frames': 109}
MLflow, step 4: {'train/Total Loss': 4.2365, 'train/Classification Loss': 4.2365, 'train/Accuracy': 0.0096,
                 'train/throughput': 1.7998, 'train/interval_seconds': 8.8898, 'learning_rate': 0.001,
                 'train/frame_throughput': 12.2612}
```

**Example output: iterations 4 and 12**
```text
===========================================================================================================================================================
|     Iteration: 4    |     Total Sequences: 16     |    Interval Sequences: 16    |   Throughput: 1.8 Sequences Per Second  |   12.3 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   4.2365
Moving Average of Train Classification Loss            :   4.2365
Moving Average of Train Accuracy                       :   0.0096
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:08.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
...
===========================================================================================================================================================
|    Iteration: 12    |     Total Sequences: 48     |    Interval Sequences: 16    |   Throughput: 2.6 Sequences Per Second  |   15.3 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   3.9174  -0.1656 Best
Moving Average of Valid Total Loss                     :   4.1790
Moving Average of Train Classification Loss            :   3.9174  -0.1656 Best
Moving Average of Valid Classification Loss            :   4.1790
Moving Average of Train Accuracy                       :   0.4486  +0.1470 Best
Moving Average of Valid Accuracy                       :   0.0000
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:06.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
```
The interval of iteration 8 (loss 4.0830, accuracy 0.3015) took 00:01:06, since it includes the test evaluation of iteration 6, so its throughput is 0.2 Sequences Per Second.

### Step 3.15 — end of the loop
The checkpoint of iteration 12 is already saved and evaluated inside its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 60 Test Sequences, 427 Labelled Frames · 2026-10-18 08:48 ───────────────      <- the evaluation of iteration 12, last in the iteration
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```

**Example: the run folder after the training**
```text
M2M_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71/
├── Evaluation/         Iteration_6/, Iteration_12/
├── Tensorboard/        Train/, Validation/, Test/
├── Weights/            Iteration_3/, Iteration_6/, Iteration_9/, Iteration_12/, full-checkpoint
├── data.json
├── label_dictionary.json
├── mlflow_run.json
├── Results.txt
└── training_configuration.yaml
MLflow artifacts: metric_names.json, Results.txt, data/data.json, configuration/training_configuration.yaml,
                  evaluation/iteration_6/ and evaluation/iteration_12/ (class_metrics.csv, class_thresholds.json, confusion_matrix.png)
```
This run was recorded before the ROC and precision-recall curves of the per-class report (2026-10-20): with `per_class_report`, every evaluation folder now also holds `roc_curves.png` and `precision_recall_curves.png`, sent to MLflow with the confusion matrix (`TRACKED_EVALUATION_FILES` of `base_trainer.py`; see Stage 4).


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (60 sequences, 427 labelled frames), with `per_class_report: True`, `number_sample_images: 8`, `top_k: 2` (then a setting, now fixed at 5), `save_predictions: True`, `error_dumps` activated with `maximum_per_class_pair: 2`, and the MLflow tracking on. The checkpoint of iteration 6 gives `disappearing_hollow_ellipse` to every frame; the one of iteration 12 gives `no_change` to 307 frames and `disappearing_hollow_ellipse` to the 120 others. The outputs of iteration 6 were captured when its checkpoint was still saved before its training step, after 5 steps; it now holds 6 steps.
>
> They were also captured before the ROC and precision-recall curves of the per-class report were added (2026-10-20). With `per_class_report`, the current code computes for every class, one against the others, its ROC and precision-recall curves on the labelled frames (`compute_class_score_curves()`, see `classification_workflow.md`, Steps 4.14 and 4.15): `Macro ROC AUC` and `Macro Average Precision` follow `Top-2 Accuracy` in the headline metrics, the per-class table, `class_metrics.csv` and the Test scalars gain the `ROC AUC` and `Average Precision` of every class, and `roc_curves.png` and `precision_recall_curves.png` are written and sent to TensorBoard and MLflow. Steps 4.7, 4.9, 4.13 and 5.5 give the differences next to the recorded outputs.
>
> All the examples of this stage, those of the README screenshot run included, were also captured before the per-class recall intervals were added (2026-11-01): `class_metrics.csv` now holds `recall_lower` and `recall_upper` after `recall`, and with `per_class_report` the report shows a `Per Class Recall` table (column `Frames`) after `Per Class Metrics`, with the frame recall of every class and the bounds of its interval (Step 4.9).

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1).

**Example output: the progress bar** (60 sequences, batch size 4)
```text
Test Evaluation Iteration 12: 100%|██████████| 15/15 [00:08<00:00,  1.86it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The labelling hook keeps only the labelled frames: their probabilities and labels, the sequence and the position in it of every one of them (to name the frame in the predictions file and to place it in its sequence for the segment metrics), and, for every sequence, whether all its labelled frames are right. It also holds the folder of the error images and their counts per class pair.

**Example: the accumulator at the end of the loop of iteration 6**
```text
'probabilities':              15 tensors, the first [20, 71] torch.float64     <- the 20 labelled frames of the first batch, 427 in all
'labels':                     15 tensors, the first [20] torch.int64
'frame_sequence_identifiers': 15 tensors, the first [20] torch.int64
'frame_positions':            15 tensors, the first [20] torch.int64
'exact_matches':              15 tensors, the first [4] torch.bool             <- 60 in all, one per sequence
'errors_folder':              .../Evaluation/Iteration_6/Errors
'error_counts':               {}  at the start, the images written per class pair
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
**Example: the predictions of the first test batch** (its longest sequence has 6 frames)
```text
class_logits          [4, 6, 71] torch.float32
class_probabilities   [4, 6, 71] torch.float32
```

### Step 4.4 — `accumulate_evaluation_batch()` → `compute_sequence_exact_matches()`
It keeps the probabilities and the labels of the labelled frames (`labels != -100`), with their sequence and position, and `compute_sequence_exact_matches()` tells for every sequence whether all its labelled frames are right. Then `dump_errors()` runs while the frames are still in the batch.

### Step 4.5 — `dump_errors()` → `build_frame_prediction_captions()` + `build_frame_grid()`
Every wrong frame gives one image: the grid of its sequence, every real frame captioned with its number and `true/predicted` (with `X ` before a wrong frame, e.g. `4 X disappearing_kite/no_change` in `README/sequence_labelling_sample.png`, and `none` as the true class of an unlabelled frame), under a caption bar `frame <index> : <true> -> <predicted> <score>` naming the wrong frame, its classes and the probability of its predicted class. It is saved as `Errors/<TRUE>-TO-<PREDICTED>/<sequence>_frame<index>_<score>.png`, at most `maximum_per_class_pair` per pair.

**Example: the error images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Errors/
├── appearing_circle-TO-no_change/                 test_000041_frame3_0.019.png   test_000041_frame9_0.020.png
│                                                  <- frames 3 and 9 of test_000041
├── disappearing_square-TO-no_change/              test_000032_frame4_0.019.png   test_000035_frame4_0.019.png
├── no_change-TO-disappearing_hollow_ellipse/      test_000032_frame0_0.017.png   test_000041_frame0_0.017.png
...                                                96 pairs, 158 images
report line: "Error Images : 158 in 96 class pairs, at most 2 per pair"
```
The checkpoint of iteration 6 gives 133 images in 68 class pairs.

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
The labelling hook draws the frame grid of the sequence (`FRAME_GRID_COLUMNS = 4` frames per row), every real frame captioned with its number and `true/predicted`, with `X ` before a wrong frame, under a caption bar giving the number of labelled frames that are right, in green when all are and in red otherwise.

**Example: the sample images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Samples/
test_000000.png  test_000008.png  test_000017.png  test_000025.png
test_000034.png  test_000042.png  test_000051.png  test_000059.png
build_sample_image -> (96, 276, 3) uint8      a sequence of 2 to 4 frames: one row of 64 pixel frames under the caption bar
                      (164, 276, 3) uint8     a sequence of 5 to 8 frames: two rows
                      (232, 276, 3) uint8     a sequence of 9 to 12 frames: three rows
caption: "<right frames>/<labelled frames> labelled frames right (true/predicted)"
TensorBoard: Samples/test_000000, ..., Samples/test_000059
```

### Step 4.7 — `compute_evaluation_results()`
The frame metrics of the classification are computed on the labelled frames (Steps 4.8 and 4.9), with `per_class_report` their ROC and precision-recall curves, then `compute_extra_headline_metrics()` adds the sequence metrics (Steps 4.10 and 4.11), `compute_recall_intervals()` the 95% confidence interval of the frame recall of every class, a Korn-Graubard interval from a bootstrap over the test sequences (Step 4.9), and `describe_extra_notes()` the number of segments. `describe_evaluated_samples()` gives the sequences and their labelled frames: every frame of the test split is evaluated once. The headline metric is the frame accuracy.

**Example: the results of iteration 12** (recorded before the curves)
```text
evaluated_samples = '60 Test Sequences, 427 Labelled Frames'
metrics           = {'Accuracy': 0.2787, 'Balanced Accuracy': 0.0084, 'Macro F1': 0.0067, 'Top-2 Accuracy': 0.3466,
                     'Sequence Exact Match': 0.0, 'Edit Score': 0.1424, 'Segment F1@10': 0.2047, 'Segment F1@25': 0.1151,
                     'Segment F1@50': 0.0384}
notes             = ['Segments : 355 true and 114 predicted in 60 test sequences, an unlabelled frame ending a segment',
                     'Error Images : 158 in 96 class pairs, at most 2 per pair', 'Sample Images : 8 fixed test sequences']
headline_metric   = 'Accuracy'
class_scalars     = 213 (71 classes x 3): Precision/no_change, Recall/no_change, F1/no_change, Precision/appearing_square, ...
tables            = Per Class Metrics: Class, Frames, Precision, Recall, F1, Threshold @ P0.99, Recall @ P0.99
files             = confusion_matrix.csv, class_thresholds.json, class_metrics.csv, confusion_matrix.png, predictions.csv,
                    Errors, Samples          (Results.txt is printed as a link before them, by report_evaluation_results())
```
With the current code, `metrics` holds `'Macro ROC AUC'` and `'Macro Average Precision'` right after `'Top-2 Accuracy'`, `class_scalars` the `ROC AUC/<class>` and `Average Precision/<class>` of every class having them (a class needs test frames for both, and frames of other classes for the ROC AUC), the table the `ROC AUC` and `Average Precision` columns after `F1`, and `files` `roc_curves.png` and `precision_recall_curves.png` after `confusion_matrix.png`. The README screenshot run of this configuration (2000 iterations, `test_10000`) gives `'Macro ROC AUC': 0.5826, 'Macro Average Precision': 0.0194` at iteration 2000, and 366 Test scalars (11 global, 5 for each of the 71 classes).

### Step 4.8 — the frame metrics: `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()` + `compute_top_k_accuracy()`
The metrics of the classification (see `classification_workflow.md`, Steps 4.9 to 4.11), every labelled frame of every sequence being one sample.

**Example: `Evaluation/Iteration_12/confusion_matrix.csv`** (71 x 71, shortened)
```text
true \ predicted,no_change,appearing_square,appearing_circle, ... ,disappearing_hollow_ellipse,disappearing_hollow_semicircle,disappearing_hollow_capsule
no_change,119,0,0, ... ,87,0,0
appearing_square,1,0,0, ... ,1,0,0
...
disappearing_hollow_ellipse,2,0,0, ... ,0,0,0
...
columns: 307 frames predicted no_change, 120 predicted disappearing_hollow_ellipse, 0 for the 69 other classes
```

**Example: how the numbers of iteration 12 are obtained**
```text
accuracy          = 119 / 427                                       = 0.2787   (only no_change frames are right)
no_change         precision = 119 / 307 = 0.3876   recall = 119 / 206 = 0.5777   F1 = 0.4639
balanced accuracy = 0.5777 / 69                                     = 0.0084   (69 classes have test frames)
macro F1          = 0.4639 / 69                                     = 0.0067
```
`appearing_star` and `appearing_semicircle` have no test frame, so they are left out of the averages.

### Step 4.9 — `find_class_thresholds()` + `write_class_metrics_file()` + `save_confusion_matrix_image()`
The frames of a sequence share its content and are predicted together, so an exact interval over the frames would be far too narrow. The labelling trainer overrides `compute_recall_intervals()`: the true frames of every class and those whose most probable class is the true one are added up per test sequence (`sum_counts_by_group()` of `vision_training/evaluation/recall_intervals.py`, checked against the frames of the metrics), then `compute_bootstrap_recall_intervals()` computes the Korn-Graubard interval of every class. It draws 1000 resamples of the test sequences with replacement (fixed seed), every sequence kept whole, for the variance of the recall of every class, a resample without frames of a class not counting for it. The effective sample size of the class, `p (1 - p) / variance` capped at its frames, is reduced by `(z / t)²`, `t` being the Student quantile at the sequences holding the class minus one, so that the interval widens when few sequences hold it, and the bounds are the exact (Clopper-Pearson) interval of the recall at that size. With a recall of 0 or 1 in every sequence, every resample gives the same recall and the bounds are the exact interval over the sequences holding the class, each counted as one sample, instead of an interval of zero width. The bounds are written as `recall_lower` and `recall_upper` after `recall`, empty for a class without test frames, and shown in the `Per Class Recall` table with `per_class_report`.

**Example: `Evaluation/Iteration_12/class_metrics.csv`** (72 lines; the `samples` column counts labelled frames)
```text
class_index,class,samples,precision,recall,f1,threshold_at_target_precision,recall_at_target_precision
0,no_change,206,0.387622,0.577670,0.463938,0.020103,0.009709
1,appearing_square,2,0.000000,0.000000,0.000000,,
...
69,disappearing_hollow_semicircle,4,0.000000,0.000000,0.000000,0.014210,0.250000
...
```
In `class_thresholds.json` (`target_precision` 0.99), `no_change` reaches a precision of 0.99 from a probability of 0.0201 (recall 0.0097) and `disappearing_hollow_semicircle` from 0.0142 (recall 0.25); most of the other classes are `null`.

With `per_class_report`, the current header has two more columns, filled from the curves of every class:
```text
class_index,class,samples,precision,recall,f1,roc_auc,average_precision,threshold_at_target_precision,recall_at_target_precision
0,no_change,32843,0.469736,1.000000,0.639211,0.657150,0.712039,0.407468,0.307585        <- README screenshot run, iteration 2000
```

### Step 4.10 — Sequence Exact Match
The share of the test sequences whose labelled frames are all right, from the `exact_matches` of Step 4.4: the strictest metric of the task.

**Example output**
```text
Sequence Exact Match = 0.0      (no test sequence has all its labelled frames right)
```

### Step 4.11 — `compute_segment_results()` → `group_sequence_predictions()` + `compute_segment_metrics()`
The frame metrics do not see whether the prediction of a sequence is fragmented: a sequence of 10 frames of one class predicted with one wrong frame in the middle keeps a frame accuracy of 0.9, but it gives three segments for one event. The segment metrics score the events of the test sequences:
- `group_sequence_predictions()` sorts the labelled frames by sequence then by frame (`np.lexsort`), takes the most probable class of every frame and splits them sequence by sequence; every frame is evaluated once, and a frame given twice raises a `ValueError`;
- `extract_segments()` cuts every sequence into segments of consecutive frames of one class, a gap of unlabelled frames ending a segment, as true segments and predicted segments;
- **Edit Score**: for every sequence, 1 minus the edit distance between the orders of the predicted and true segment classes (`compute_edit_distance()`: the segments to insert, remove or replace) over the larger number of segments, averaged over the sequences;
- **Segment F1@10/25/50**: `count_segment_matches()` takes every predicted segment in order and finds the true segment of its class it overlaps most; it is right when their intersection over union reaches the overlap (10, 25 or 50 %) and no earlier predicted segment matched that true segment. The F1 is computed from the counts summed over all the sequences.

**Example: the grouped sequences of the evaluation of iteration 12** (class indices: 0 `no_change`, 68 `disappearing_hollow_ellipse`)
```text
group_sequence_predictions: 427 labelled frames -> 60 sequences
frames [0, 1]  true [0, 0]  predicted [68, 68]
   true segments [(0, 0, 1)]   predicted segments [(68, 0, 1)]   edit distance 1   edit score 0.0000
frames [0, 1]  true [0, 22]  predicted [68, 68]
   true segments [(0, 0, 0), (22, 1, 1)]   predicted segments [(68, 0, 1)]   edit distance 2   edit score 0.0000
frames [0, ..., 11]  true [0, 0, 42, 0, 0, 0, 0, 23, 32, 0, 3, 0]  predicted [68, 68, 0 x 10]
   true segments [(0, 0, 1), (42, 2, 2), (0, 3, 6), (23, 7, 7), (32, 8, 8), (0, 9, 9), (3, 10, 10), (0, 11, 11)]
   predicted segments [(68, 0, 1), (0, 2, 11)]   edit distance 7   edit score 0.1250
Segments : 355 true and 114 predicted in 60 test sequences
Edit Score 0.1424   Segment F1@10 0.2047   Segment F1@25 0.1151   Segment F1@50 0.0384
```
A segment is `(class, first frame, last frame)`. In the last sequence the 2 predicted segments must become the 8 true ones: the first is replaced and 6 are inserted, an edit distance of 7 over 8 segments, an edit score of 1 - 7/8 = 0.125.

### Step 4.12 — `write_predictions_file()` → `build_prediction_rows()`   (only with `save_predictions`)
One row per labelled frame of every sequence, named by its sequence and its frame file.

**Example: `Evaluation/Iteration_12/predictions.csv`** (427 rows, 76 columns, 305,323 bytes; first rows, columns shortened)
```text
sequence,frame,true_class,predicted_class,correct,probability_no_change,probability_appearing_square, ... ,probability_disappearing_hollow_capsule
test_000032,test_000032/000000.png,no_change,disappearing_hollow_ellipse,0,0.015759,0.013611, ...
test_000032,test_000032/000001.png,appearing_square,disappearing_hollow_ellipse,0,0.017056, ...
test_000032,test_000032/000002.png,no_change,no_change,1,0.017983, ...
```

### Step 4.13 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 60 Test Sequences, 427 Labelled Frames · 2026-10-18 08:48 ───────────────────────────────────────────────
  Accuracy               0.2787   Balanced Accuracy   0.0084
  Macro F1               0.0067   Top-2 Accuracy      0.3466
  Sequence Exact Match   0.0000   Edit Score          0.1424
  Segment F1@10          0.2047   Segment F1@25       0.1151
  Segment F1@50          0.0384
  Segments : 355 true and 114 predicted in 60 test sequences, an unlabelled frame ending a segment
  Error Images : 158 in 96 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                                             Threshold    Recall
   Class                              Frames   Precision   Recall       F1     @ P0.99   @ P0.99
  ───────────────────────────────────────────────────────────────────────────────────────────────
   appearing_square                        2      0.0000   0.0000   0.0000           —         —
   appearing_circle                        6      0.0000   0.0000   0.0000           —         —
   ...
   disappearing_hollow_semicircle          4      0.0000   0.0000   0.0000      0.0142    0.2500
   disappearing_hollow_capsule             4      0.0000   0.0000   0.0000           —         —
   no_change                             206      0.3876   0.5777   0.4639      0.0201    0.0097
   ...
   appearing_semicircle                    0           —        —        —           —         —
  Worst F1 first, ties in class order. — : no frames of the class, or the precision 0.99 is never reached.
  - Results : Results.txt
  - ...
```

**Example: the same block with the current code** (the README screenshot run of this configuration, 2000 iterations at 512 pixels, on `test_10000`; shortened)
```text
Iteration 2000 · 10000 Test Sequences, 69918 Labelled Frames · 2026-10-23 05:38 ────────────────────────────────────────
  Accuracy               0.4697   Balanced Accuracy         0.0141
  Macro F1               0.0090   Top-2 Accuracy            0.4774
  Macro ROC AUC          0.5826   Macro Average Precision   0.0194
  Sequence Exact Match   0.0589   Edit Score                0.2581
  Segment F1@10          0.2822   Segment F1@25             0.1833
  Segment F1@50          0.0711
  Segments : 58133 true and 10000 predicted in 10000 test sequences, an unlabelled frame ending a segment
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                                                         Average   Threshold    Recall
   Class                              Frames   Precision   Recall       F1   ROC AUC   Precision     @ P0.99   @ P0.99
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   appearing_square                      559      0.0000   0.0000   0.0000    0.5835      0.0099           —         —
   ...
   no_change                           32843      0.4697   1.0000   0.6392    0.6571      0.7120      0.4075    0.3076
  Worst F1 first, ties in class order. — : no frames of the class, no negative for the ROC AUC, or the precision 0.99 is
  never reached.
```
That run gives `no_change` to every frame (`suggestions.md` K29): its 10,000 predicted segments are one per sequence.

**Example: TensorBoard and MLflow after the two evaluations** (recorded before the curves)
```text
Tensorboard/Test: 222 scalars (9 global + 213 per class), 9 images (8 samples + Confusion Matrix)
    scalar Accuracy        6: 0.0047, 12: 0.2787
    scalar Edit Score      6: 0.0038, 12: 0.1424
    scalar Segment F1@50   6: 0.0,    12: 0.0384
MLflow: 233 metrics: test/Accuracy, ..., test/Edit Score, test/Segment F1_at_10, test/Segment F1_at_25, test/Segment F1_at_50,
        test/Sequence Exact Match, test/best_Accuracy (steps 6, 12), 213 test_class/<metric>/<class>, train/*, validation/*, learning_rate
```
MLflow only accepts letters, digits, `_`, `-`, `.`, `/` and spaces in a metric name, so `sanitise_metric_name()` writes `Segment F1@10` as `Segment F1_at_10` (the original names are kept in the `metric_names.json` artifact); TensorBoard and the console keep the `@`. With the current code, `Tensorboard/Test` also holds `Macro ROC AUC`, `Macro Average Precision`, `ROC AUC/<class>`, `Average Precision/<class>` and the `ROC Curves` and `Precision Recall Curves` images (366 scalars and 11 images in the README screenshot run), all sent to MLflow too.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12, captured when those of iterations 3, 6 and 9 were still saved before their training step) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 20 sequences of the test split (its `labels.json` reduced, its `data/` folder linked). They were also captured before the per-class recall intervals (2026-11-01, Stage 4).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5). The test sequences are read with the `sequence_settings` of the run, so a test sequence longer than its `maximum_sequence_length` is left out with the warnings of Step 2.8.

### Step 5.2 — trainer construction
The trainer is built from the `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it rather than those of the mapping file (`classification_workflow.md`, Step 5.2).

**Example: the split line of the launch information**
```text
- Test Data Folder        : test_reduced (20 sequences, 161 frames, 20 sequences of at most 50 frames)
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
-----------------------------
Evaluating : model_000012.pt
-----------------------------
MLflow Run Created In The Experiment 'sequence_labelling' : 0a6b17420e9b4be1a981fe2d96db3cfd (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output** (iterations 9, 6 and 3 predict the same class everywhere and give the same row)
```text
-------------------------------------------------------------
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
-------------------------------------------------------------
  - Best Checkpoint : iteration 12 (Accuracy 0.2981)

iteration,Accuracy,Balanced Accuracy,Macro F1,Top-2 Accuracy,Sequence Exact Match,Edit Score,Segment F1@10,Segment F1@25,Segment F1@50, ...
12,0.2981366459627329,0.01309328968903437,0.010264086389393778,0.36024844720496896,0.0,0.12883116883116885,0.1927710843373494,0.10843373493975904,0.012048192771084338, ...
9,0.0,0.0,0.0,0.012422360248447204,0.0,0.0,0.0,0.0,0.0, ...
6,0.0,0.0,0.0,0.012422360248447204,0.0,0.0,0.0,0.0,0.0, ...
3,0.0,0.0,0.0,0.012422360248447204,0.0,0.0,0.0,0.0,0.0, ...
```
The file has 223 columns: the iteration, the 9 global metrics, then the precision, recall and F1 of the 71 classes. It is written in `evaluations/<run name>/Evaluation/`, next to the `Iteration_12`, `Iteration_9`, `Iteration_6` and `Iteration_3` folders of the evaluation. With the current code (Stage 4), `Macro ROC AUC` and `Macro Average Precision` follow `Top-2 Accuracy`, and the `ROC AUC` and `Average Precision` columns of every class having them follow its F1, the columns being all the scalars of the evaluation (`build_summary_columns()`).

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 12 on the 20 sequences** (shortened, recorded before the curves of Stage 4)
```text
Iteration 12 · 20 Test Sequences, 161 Labelled Frames · 2026-10-18 08:49 ───────────────────────────────────────────────
  Accuracy               0.2981   Balanced Accuracy   0.0131
  Macro F1               0.0103   Top-2 Accuracy      0.3602
  Sequence Exact Match   0.0000   Edit Score          0.1288
  Segment F1@10          0.1928   Segment F1@25       0.1084
  Segment F1@50          0.0120
  Segments : 128 true and 38 predicted in 20 test sequences, an unlabelled frame ending a segment
  Error Images : 75 in 54 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences
  ...
   no_change                              78      0.3967   0.6154   0.4824      0.0201    0.0256
   ...
```
The header and the segment note count the same 20 sequences, every sequence being evaluated whole.
