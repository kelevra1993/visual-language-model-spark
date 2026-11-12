# Frame Selection Workflow

This document follows one frame selection run (SEL: in every whole sequence, the frames to choose for every wanted class) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The frame selection trainer is the sequence labelling trainer fed with whole sequences: the steps it shares with the sequence labelling (one class per frame, temporal module keeping the time axis) are explained in `sequence_labelling_workflow.md`, those shared with the sequence classification (maximum sequence length, padding, frame encoder) in `sequence_classification_workflow.md`, and the steps shared by every task in `classification_workflow.md`; here they keep a short explanation and the real output of the selection run, and the steps specific to the selection (unselected class, whole sequences checked against the wanted classes, selection loss, Hungarian and greedy decodings, selection and ranking metrics, selections file, selection images) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training. The synthetic dataset of the configuration, `synthetic_selected_shapes`, was not on this machine when they were recorded (it was generated on 2026-10-21 for the README screenshot run, with the settings of the main block of the generator: 2,000 training, 200 validation and 200 test sequences of 4 to 12 frames of 256 pixels), so a small one was generated in a temporary folder with `create_synthetic_dataset(task="frame_selection")`: 500 training, 60 validation and 60 test sequences of 4 to 12 frames of 128 pixels, every sequence holding one frame of each of the 3 wanted classes (`circle`, `triangle`, `star`), its other frames being `unselected`. The data folders and the mapping file of the configuration were redirected to it.
- **Stages 1 and 2**: `configurations/frame_selection/configuration_synthetic_sequences.yaml` (a maximum sequence length of 50 frames, 256 pixel frames, batch size 16, a bidirectional GRU, Hungarian decoding, one frame per class, selection loss weight 1), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers (then a setting; every loader now has 1), with the error images and the MLflow tracking turned on. After 12 iterations the network gives the most probability to `unselected` on every frame and almost the same probabilities to all the frames of a sequence, so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

## Read This First
The frame selector chooses, in every whole sequence, exactly `Selection.frames_per_class` frames for every wanted class, every class of the label mapping except `Selection.unselected_category` being wanted. Its network is the sequence labeller (`sequence_labelling_workflow.md`): one probability vector per frame over the wanted classes and the unselected class, trained with the per-frame cross entropy plus a selection loss in which the frames of a sequence compete for every wanted class. The frames are then chosen from these probabilities by the Hungarian assignment or a greedy rule, both scored at every evaluation, the one of `Selection.decoding` being drawn, giving the confidence thresholds and run by the deployed decoder. Choose it rather than the sequence labelling (M2M) when every sequence must give a fixed number of frames of every class and the right frame depends on the other frames of its sequence (the sharpest view of each side of an object, the example of the README); choose M2M when every frame has a class of its own, any number of frames possibly holding it.

### The Task At A Glance
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
        │ compute_selection_metrics() per decoding, compute_ranking_metrics() (Top-2, Mean True Frame Rank),
        │ frame metrics of the most probable class, find_selection_thresholds() at target_precision 0.99,
        │ Korn-Graubard interval (bootstrap over the test sequences) of the selection accuracy of every wanted class
        ▼
Evaluation/Iteration_<n>/   class_thresholds.json   class_metrics.csv   confusion_matrix.csv .png (decoded frames)
                            selections.csv   Samples/<sequence>.png   Errors/<CLASS>/<sequence>.png
export  frame_selector_<format>: images [B, T, 3, 256, 256] ─> class_probabilities [B, T, 4], with preprocessing.json
        (and the ONNX metadata): the input contract, keep_ratio 256 linear, padding 0 centred, BGR, 0-255, NTCHW
        frame_selector_decoder (Python backend, CPU) ─> selected_frames, selected_probabilities, low_confidence,
        candidate_frames, candidate_probabilities; frame_selector_<format>_ensemble chains both
```

### Files To Know
| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/frame_selection_trainer.py`, on `sequence_labelling_trainer.py` | `FrameSelectionTrainer`: `build_dataloaders`, `build_model`, `decode_sequence`, `dump_errors`, `build_sample_image`, `compute_evaluation_results`, `build_selection_table`, `write_selections_file` |
| Dataset and loaders | `vision_training/data/sequence/sequence_dataset.py`, `vision_training/data/loaders.py` | `SequenceDataset` (`label_type="selection"`), `find_unselected_class_index`, `describe_selection_problem`, `collate_padded_sequences`, `get_sequence_dataloaders` |
| Model | `vision_training/models/sequence/frame_selector.py`, on `sequence_labeller.py` | `FrameSelector`: `wanted_class_mask` buffer, `selection_loss_weight`, `get_unselected_class_index` |
| Head | `vision_training/models/classification/classification_head.py` | `ClassificationHead` |
| Losses | `vision_training/losses/sequence_losses.py` | `compute_sequence_labelling_loss`, `compute_frame_selection_loss` |
| Decoding | `vision_training/models/sequence/frame_selection_decoding.py` (numpy and scipy only) | `decode_frame_selection`, `decode_hungarian_selection`, `decode_greedy_selection`, `compute_frame_ranks`, `flag_low_confidence`, `run_frame_selection` |
| Metrics and images | `vision_training/evaluation/sequence/frame_selection_metrics.py`, `vision_training/utilities/visualization.py` | `compute_selection_metrics`, `compute_ranking_metrics`, `find_selection_thresholds`, `build_frame_selection_image`; `compute_bootstrap_recall_intervals` and `build_recall_interval_table` of `vision_training/evaluation/recall_intervals.py` |
| Configuration checks | `vision_training/configuration_processing/sequence_checks.py` | `check_frame_selection_configuration`, `check_selection_configuration` |
| Synthetic data | `vision_training/data/synthetic/create_synthetic_dataset.py` | `generate_selection_sequence`, `SELECTION_CATEGORIES` |
| Export and deployment | `vision_training/deployment/export_model.py`, `frame_selection_deployment.py`, `frame_selection_decoder_backend.py`, `deployment_utilities.py`, `input_contract.py` | `build_decoding_settings`, `find_class_thresholds_path`, `export_frame_selection_decoding`, `TritonPythonModel` of the decoder, `build_input_contract`, `write_preprocessing_file` |

### Run It
```bash
# Dataset: set task = "frame_selection" in the main block of vision_training/data/synthetic/create_synthetic_dataset.py,
# whose output_folders, image_sizes and split_sizes give synthetic_selected_shapes, 256 pixel frames, 2,000 / 200 / 200
# sequences of 4 to 12 frames, one shape per frame
uv run python -m vision_training.data.synthetic.create_synthetic_dataset

# Training, after setting experiment_folder, the three data folders and Data.label_mapping.mapping_file_path
uv run python -m vision_training.main --configuration_path="vision_training/configurations/frame_selection/configuration_synthetic_sequences.yaml"

# Evaluation of the checkpoints of a run: set run_name, experiment_folder, iteration (None, a number or "all"),
# test_data_folder, batch_size, track_evaluation, maximum_error_images_per_class_pair and target_precisions (None
# or a single number, a list being refused for this task) in its main block; it writes in
# <run>/Checkpoint_Evaluation/
uv run python -m scripts.evaluation.evaluate_checkpoint

# Export with the decoder and the ensembles: set experiment_folder, iteration, use_averaged_weights, formats,
# maximum_batch_size and tensorrt_precision in its main block, use_saved_temperature staying False (classification only)
uv run python -m vision_training.deployment.export_model
```

### Configuration Keys To Know First
| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `Selection.unselected_category` | `"unselected"` | manifest category of the frames not chosen, which the label mapping must map; every other class is wanted, and the frames of unmapped categories become unselected |
| `Selection.frames_per_class` | `1` | frames chosen for every wanted class; every sequence must hold exactly this many true frames of each (`FPC1` token) |
| `Selection.selection_loss_weight` | `1.0` | weight of the selection loss added to the per-frame loss, 0 training with the per-frame loss alone (`SLW1` token, none at 0) |
| `Selection.decoding` | `"hungarian"` | decoding of the images, the thresholds, the best checkpoint and the deployed decoder, both decodings being scored; no token |
| `ExperimentConfiguration.task` | `"frame_selection"` | `FrameSelectionTrainer`, the SEL checks and the `SEL` run name code |
| `TemporalModel.type`, `TemporalModel.bidirectional` | `"gru"`, `True` | every frame compared with the whole sequence (`GRU1x128-BI`, 256 features per frame) |
| `Data.sequence_settings.maximum_sequence_length` | `50` | longest sequence fed whole, a longer one being left out with a red warning (`T50` token) |
| `ExperimentConfiguration.evaluation.top_k` | `2` | top-k selection accuracy, and candidate frames of every class returned by the deployed decoder; a setting of this task only, the other classification-like tasks counting a fixed 5 classes |
| `ExperimentConfiguration.evaluation.target_precision` | `0.99` | confidence threshold of the chosen frames of every class (`class_thresholds.json`), behind the `low_confidence` flags |
| `ExperimentConfiguration.evaluation.save_predictions` | `True` | `selections.csv`, one row per test sequence and wanted class |

### What A Run Writes
- `Results.txt`, one block per evaluated checkpoint: `Iteration <n> · <S> Test Sequences, <F> Frames · <date>`, the `Selection Accuracy`, `Fully Right Sequences`, `Unselected Precision` and `Unselected Recall` of each decoding, `Top-2 Selection Accuracy`, `Mean True Frame Rank`, `Frame Accuracy`, `Frame Balanced Accuracy`, `Frame Macro F1`, the note naming the decoding of the images and thresholds, then the `Per Class Selection` and `Per Class Recall` tables with `per_class_report`, the second giving the selection accuracy of every wanted class with the decoding of the run and its 95% confidence interval, a Korn-Graubard interval from a bootstrap over the test sequences, worst first. Unlike the sequence labelling it is built on, the frame selection writes every rate between 0 and 1 with four decimals (`0.9950`), in the report, the console summaries and the best checkpoint line (`get_percent_tracked_metrics()`, `percent_columns` and `percent_metrics` left empty).
- `Evaluation/Iteration_<n>/`: `class_thresholds.json` (the target precision, the decoding, the unselected class, and the threshold and recall of every wanted class, `null` when never reached), `class_metrics.csv` (one row per wanted class, `<decoding>_accuracy_lower` and `<decoding>_accuracy_upper` following the accuracy of the decoding of the run), `confusion_matrix.csv` of the decoded frames and, with `per_class_report`, `confusion_matrix.png`; `selections.csv` with `save_predictions`.
- `Samples/<sequence>.png` (8 fixed test sequences, evenly spread) and `Errors/<CLASS>/<sequence>.png` (with `error_dumps.activated`): the thumbnail grid of the whole sequence with the borders of the selection, the probability heat map and the number of right selections.
- TensorBoard: `Train/` and `Validation/` `Total Loss`, `Classification Loss`, `Selection Loss` and `Accuracy` (of the real frames), `Train/` also `Learning Rate`; `Test/` the 13 global metrics, `Hungarian Selection Accuracy/<class>`, `Greedy Selection Accuracy/<class>`, `Mean True Frame Rank/<class>`, `Samples/<sequence>` and `Confusion Matrix`.
- `data.json` counts the frames of every class, the frames of unmapped categories with the unselected class.

### Pitfalls
- Every sequence must hold exactly `frames_per_class` frames of every wanted class, otherwise the training stops at startup with the list of the first 20 such sequences (Step 2.8, `describe_selection_problem()`); a wanted class left out of the label mapping turns its frames into unselected ones (`SequenceDataset`).
- The Hungarian decoding maximises the probability of the whole labelling, not the number of right frames: on the 20 sequences of Stage 5, the checkpoint of iteration 3 gets 3 true triangle frames with the greedy decoding and 1 with the Hungarian one (Step 5.6). The best checkpoint, the thresholds and the deployment follow `Selection.decoding` only.
- Tied probabilities give every tied frame the best rank, so a model giving the same probabilities to every frame scores a Top-k Selection Accuracy and a Mean True Frame Rank of 1.0 (`compute_frame_ranks()`, kept as it is).
- A class never reaching `target_precision` gets a `null` threshold, every frame chosen for it being flagged low confidence (Step 4.14). The export reads the `class_thresholds.json` of the exported iteration in `Evaluation/Iteration_<n>/` (training) or `Checkpoint_Evaluation/Iteration_<n>/` (`evaluate_checkpoint.py`), and stops when both hold one (`build_decoding_settings()`, `find_class_thresholds_path()`). A checkpoint exported without any evaluation gets thresholds of 0, so no chosen frame is flagged, and `decoding_settings.json` names its wanted classes by their indices: evaluate an intermediate checkpoint with `evaluate_checkpoint.py` before exporting it. An export of the averaged weights reads the thresholds of the trained weights (`Iteration_<n>/class_thresholds.json`, not those of its `EMA` subfolder), a limit kept as it is.
- The synthetic configuration is solved quickly: the README screenshot run reaches a Hungarian selection accuracy of 0.9950 at iteration 1000 and 1.0000 at iteration 2000, so its sample images show no wrong choice (`suggestions.md` K34). Its 3 wrong choices of iteration 1000 are triangles (`selections.csv`: `test_000004`, `test_000134`, `test_000052`), the last one chosen at 0.4446, above the triangle threshold of 0.1830, so not flagged.
- The exported network has no sequence mask: the sequences of one request must have the same number of frames, from 1 to `maximum_sequence_length` (`InferenceSequenceModel`, README Deployment), and the decoding runs in the Python backend on the CPU, after the network.

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
│   Entry point of every task. Give a frame_selection configuration with --configuration_path to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   ├── load_configuration(configuration_path)
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category id: class index}; the unselected category is one of the classes.
│   └── check_configuration()
│       ├── common checks
│       └── check_frame_selection_configuration()
│               ├── the data, head and evaluation checks of the classification, a single target_precision
│               ├── evaluation.top_k                      a positive integer
│               ├── check_sequence_settings()             maximum_sequence_length: a positive integer, the
│               │                                         longest sequence fed whole
│               ├── check_temporal_model_configuration()  an ordered module only (GRU, LSTM, transformer, temporal
│               │                                         convolution); causal allowed
│               └── check_selection_configuration()       unselected_category, decoding, frames_per_class,
│                                                         selection_loss_weight
│
├── get_trainer_class(task)  -> FrameSelectionTrainer
│
├── FrameSelectionTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   The SequenceLabellingTrainer with LABEL_TYPE = "selection" and CLASS_SUPPORT_COLUMN = "True Frames", reading
│   │   Selection.decoding and Selection.frames_per_class first.
│   │
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter()
│       ├── restore_run_label_dictionary() / setup_training_paths()
│       │       The classes of a resumed run kept from its label_dictionary.json, then the run folder with the
│       │       configuration and label_dictionary.json (classification_workflow.md, Step 2.6).
│       ├── build_dataloaders() -> get_sequence_dataloaders(label_type="selection")
│       │   ├── SequenceDataset.__init__() -> describe_selection_problem()
│       │   │       Every sequence whole, every frame keeping its class (the frames of unmapped categories being
│       │   │       unselected); a sequence longer than maximum_sequence_length is left out with a red warning, and a
│       │   │       sequence without exactly frames_per_class frames of every wanted class stops the training with
│       │   │       the list of such sequences.
│       │   └── unselected_class_index + get_wanted_class_indices()   the same unselected class in the 3 splits
│       ├── check_loaders_give_batches()                             refuses a split whose loader gives no batch
│       ├── save_data_record()                                       data.json, the frames of unmapped categories
│       │                                                            counted as unselected (get_unmapped_class_index())
│       ├── get_test_sample_classes() + choose_sample_identifiers()
│       │       The wanted classes of the frames of every test sequence, the unselected frames left out.
│       ├── build_model()
│       │   └── FrameSelector.__init__()
│       │       ├── SequenceLabeller.__init__()                 backbone, temporal module (keep_time_axis=True), head
│       │       └── selection_loss_weight + wanted_class_mask buffer   the trainer sets 0 on the unselected class
│       ├── initialise_pretrained_components() / build_optimizer() / MetricTracker(counts_frames=True)
│       ├── print_experiment_information() -> describe_split() -> count_split_sequences()
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       ├── view_input_batches() -> show_input_samples()   the frames of the wanted classes captioned
│       └── restore_last_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch() -> SequenceDataset.__getitem__() + collate_padded_sequences()
        │   │       labels [B, T]: the class of every frame, -100 on the padding frames added by the collate; a
        │   │       batch failing to load is dropped and the next one taken
        │   ├── MetricTracker.count_frames()
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch() / augment_batch()
        │   │   ├── forward_pass() -> FrameSelector.forward(sequence_mask=..., labels=...)
        │   │   │   ├── SequenceLabeller.forward()          class_logits [B, T, K], classification_loss (per frame)
        │   │   │   └── compute_frame_selection_loss()      x selection_loss_weight -> selection_loss (per class,
        │   │   │                                           over the frames of each sequence)
        │   │   ├── compute_batch_metrics() -> compute_frame_accuracy()
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()
        │   ├── save_model() on the multiples of weight_saving_iterations and intermediate_weight_saving_iterations
        │   ├── validation pass
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary() + report_dropped_batches()
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations, last    -> [STAGE 4]
        └── save_model() of the steps taken since the last checkpoint, if any; then run_training_loop() calls
                shutdown_loader_workers() and finish_experiment_tracking(), whatever ended the loop

run_test_evaluation(iteration)  [STAGE 4]
├── create_evaluation_accumulator()        the probabilities and true classes of the real frames of every sequence
├── for each test batch:
│   ├── prepare_batch() / predict_batch()
│   ├── accumulate_evaluation_batch() -> get_sequence_predictions()
│   │   └── dump_errors() -> decode_sequence() + build_selection_image()     Errors/<CLASS>/<sequence>.png
│   └── save_sample_images() -> build_sample_image() -> decode_sequence() + build_selection_image()
│                                                         -> build_frame_selection_image()
├── compute_evaluation_results()
│   ├── for the hungarian and greedy decodings:
│   │   ├── decode_frame_selection() -> decode_hungarian_selection() / decode_greedy_selection()   [W, N] frames
│   │   ├── convert_selection_to_frame_labels()                   the decoded class of every frame
│   │   └── compute_selection_metrics()                           selection accuracy, fully right sequences,
│   │                                                             unselected precision and recall, right and
│   │                                                             true frames of every sequence and wanted class
│   ├── compute_ranking_metrics() -> compute_frame_ranks()        top-k selection accuracy, mean true frame rank
│   ├── the frame metrics of the most probable class              compute_confusion_matrix() +
│   │                                                             compute_metrics_from_confusion_matrix()
│   ├── find_selection_thresholds()                               class_thresholds.json
│   ├── compute_bootstrap_recall_intervals()                      95% interval of the selection accuracy of every
│   │                                                             wanted class with the decoding of the run
│   ├── write_class_metrics_file() / save_confusion_matrix_table() / save_confusion_matrix_image()
│   ├── write_selections_file()                                   selections.csv (only with save_predictions)
│   └── build_selection_table() + build_recall_interval_table()   Per Class Selection and Per Class Recall tables
└── report_evaluation() -> report_evaluation_results() + track_evaluation()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/frame_selection/configuration_synthetic_sequences.yaml`, its data folders and mapping file redirected to the generated dataset.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The sections are those of the sequence labelling (`sequence_labelling_workflow.md`, Step 1.2), plus `Selection`:
- `unselected_category`: the name of the manifest category of the frames that are not chosen; every other class of the label mapping is a wanted class;
- `decoding`: how the frames of every wanted class are chosen from the per-frame probabilities, `hungarian` or `greedy` (Step 4.9). Both are scored at every evaluation; this one draws the images, gives the confidence thresholds and runs in the deployed decoder;
- `frames_per_class`: the number of frames chosen for every wanted class in every sequence;
- `selection_loss_weight`: the weight of the selection loss added to the per-frame loss (Step 3.10), 0 training with the per-frame loss alone.

**Example: the sections, the temporal module and the selection settings**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, ClassificationHead, Selection
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'top_k': 2, 'target_precision': 0.99,
                           'save_predictions': True, 'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
TemporalModel:            {'type': 'gru', 'hidden_size': 128, 'number_layers': 1, 'number_heads': 4, 'kernel_size': 3, 'dropout_rate': 0.0,
                           'bidirectional': True, 'causal': False, 'positional_embedding': True, 'pretrained': {...}}
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
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '<datasets>/synthetic_selected_shapes/label_mapping.json'}
dict: {1: 0, 2: 1, 3: 2, 4: 3}       <- categories unselected, circle, triangle, star
4 entries, 4 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_frame_selection_configuration()`: the checks of the sequence labelling, `Data.sequence_settings` taking only `maximum_sequence_length` as for every sequence task, with a single `evaluation.target_precision` (a list is refused, the deployed decoder keeping one threshold per class) and a positive integer `evaluation.top_k`, then `check_selection_configuration()` checks the `Selection` section. Whether the unselected category is in the manifests is checked later, by the dataset (Step 2.8).

**Example: output with `Selection.decoding: "beam"` and `Selection.selection_loss_weight: -1.0` in the YAML**
```text
------------------------------------------------------
Invalid Configuration File : configuration_wrong.yaml
------------------------------------------------------
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

> The examples below are real outputs from building the trainer on the redirected `configuration_synthetic_sequences.yaml` (256 pixel frames, a maximum sequence length of 50 frames, batch size 16), with the run folder redirected to a temporary folder. The whole construction took 0.3 s on CPU.

### Step 2.1 — `FrameSelectionTrainer.__init__()`
The constructor reads `Selection.decoding` and `Selection.frames_per_class`, then runs the constructor of the sequence labelling trainer. `LABEL_TYPE = "selection"` makes the loaders read whole sequences with one class per frame, and `CLASS_SUPPORT_COLUMN = "True Frames"` names the column of the per-class tables counting the true frames of every wanted class. The trainer overrides the loaders (to find the unselected class), the model (to mark it), the tracked metrics, the evaluation accumulator, the error images, the sample images and the whole computation of the evaluation results.

**Example: the attributes after the construction**
```text
self.sequence_settings          = {'maximum_sequence_length': 50}
self.LABEL_TYPE                 = 'selection'
self.CLASS_SUPPORT_COLUMN       = 'True Frames'
self.decoding                   = 'hungarian'
self.frames_per_class           = 1
self.unselected_class_index     = 0
self.wanted_class_indices       = [1, 2, 3]
self.top_k                      = 2
self.save_predictions           = True
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
SAMPLE_NAME = 'Sequences'   COUNTS_FRAMES = True   TASK_NAME = 'frame_selection'
```

### Step 2.2 — `BaseTrainer.__init__()`
**Example: some attributes set by the constructor**
```text
self.device = device(type='cpu')   self.per_class_report = True   self.number_sample_images = 8   self.input_image_size = 256
self.batch_size = 16   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The sections of the sequence labelling (`T50` for the maximum sequence length, `GRU1x128-BI` for one bidirectional layer of 128), followed by the selection section: `FPC<frames_per_class>`, then `SLW<selection_loss_weight>` when the selection loss is used (a weight above 0). The decoding gives no token, since both decodings are scored at every evaluation.

**Example output**
```text
SEL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4_FPC1-SLW1
```

### Step 2.5 — `DataAugmenter()`
**Example: the settings of the configuration**
```text
activated True   probability 0.4   independent_frames False   shuffle_frames {'activated': False, 'probability': 0.25}
6 of 28 types activated: random_crop, random_left_right_flip, random_rotation, brightness_contrast_saturation, random_noise,
                         random_haze
```

### Step 2.6 — `setup_training_paths()`
As for every task (see `classification_workflow.md`, Step 2.6): a resumed run first keeps the classes of its `label_dictionary.json` (`restore_run_label_dictionary()`), then the run folder is created with the copy of the configuration and `label_dictionary.json`. The record of the data, `data.json`, is written after the loaders (Step 2.7).

**Example: the run folder after this step**
```text
SEL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4_FPC1-SLW1/
├── Tensorboard/
├── Weights/
├── label_dictionary.json          50 bytes       the class index of every category of the run
└── training_configuration.yaml    19,287 bytes   the copy of the launched configuration file, comments included
```
The size of `training_configuration.yaml` is the one of the configuration file on 2026-11-12, its comments shortened and its fixed settings removed since the record.

### Step 2.7 — `build_dataloaders()` → `get_sequence_dataloaders(label_type="selection")` + `save_data_record()`
The loaders of the sequence labelling, every dataset receiving the `Selection` section. The trainer then reads the unselected class of the three datasets, which must be the same, and lists the wanted classes (`get_wanted_class_indices()`: every class but the unselected one, in class order), the rows of every selection of the evaluation. Right after the loaders, `check_loaders_give_batches()` refuses a split whose loader gives no batch; the training loader leaves out its last incomplete batch only with a batch normalized classification head (`sequence_classification_workflow.md`, Step 2.8), which is not the case here. Then `save_data_record()` writes `data.json` next to the configuration: the classes with their frames in every split, the frames of the categories the label mapping leaves out being counted in the unselected class, as the loaders give them (`get_unmapped_class_index()`), and for every split its data folder, the SHA-256 of its `labels.json` and its frames.

**Example output**
```text
class_names          = ['unselected', 'circle', 'triangle', 'star']
unselected_class_index = 0     wanted_class_indices = [1, 2, 3]
train        500 sequences -> 32 batches of 16 (one epoch = 32 iterations)
validation    60 sequences ->  4 batches of 16
test          60 sequences ->  4 batches of 16
collate      functools.partial(<function collate_padded_sequences>, members_per_sample=1)
data.json    179,097 bytes   keys ['created', 'classes', 'splits']
             classes[0] = {'class_index': 0, 'class_name': 'unselected', 'category_ids': [1],
                           'frames': {'train': 2544, 'validation': 308, 'test': 301}}
             splits['train'] = {'data_folder': '<generated dataset>/train', 'labels_sha256': '00beab16...',
                                'number_images': 4044, 'images': ['train_000000/000000.png', 'train_000000/000001.png', ...]}
```
The record and the run folder of Step 2.6 were taken again with the current code on a dataset generated with the same settings (4,044 training frames instead of the 3,950 of Step 2.8).

### Step 2.8 — `SequenceDataset.__init__()` → `describe_selection_problem()`
With the `selection` label type, the dataset finds the class of the unselected category, gives it to the frames of unmapped categories, and keeps every sequence whole, each frame keeping its class. A sequence longer than `maximum_sequence_length` frames is left out with a red warning, and a total per split, as for every sequence task. `describe_selection_problem()` then checks every kept sequence: it must hold exactly `frames_per_class` frames of every wanted class, since the decodings always choose that many frames per class. All the invalid sequences are listed at once and the training stops.

**Example: a manifest sequence, the first sequence after shuffling, and the sample the loader gets from it**
```text
labels.json:  {'id': 1, 'name': 'train_000000', 'frames': ['train_000000/000000.png', ... 11 frames],
               'frame_category_ids': [1, 1, 1, 1, 3, 4, 2, 1, 1, 1, 1]}
              (500 sequences of 4 to 12 frames, 3950 frames; 4 categories {'id': 1, 'name': 'unselected'}, ...)
sequence:     {'identifier': 460, 'sequence_identifier': 461, 'sequence_name': 'train_000460',
               'frames': ['train_000460/000000.png', ... 11 frames], 'frame_labels': [0, 0, 0, 2, 0, 3, 0, 0, 0, 1, 0]}
dataset[0]:   images (11, 3, 256, 256)
              sequence_mask [True x 11]
              labels [0, 0, 0, 2, 0, 3, 0, 0, 0, 1, 0]                        <- one class per frame, no padding yet
get_class_counts() = [2450, 500, 500, 500]                                   <- frames of every class
```
Every training sequence holds one frame of each wanted class, so the 3950 frames give 500 frames of each wanted class and 2450 unselected frames.

**Example: the warnings with `maximum_sequence_length: 8`, then the error with `Selection.frames_per_class: 2`** (on a dataset generated again with the same settings, 3957 training frames)
```text
Sequence 'train_000001' Of The Training Split Has 11 Frames, More Than The Maximum Sequence Length (8): It Is Left Out Of The Training.
Sequence 'train_000003' Of The Training Split Has 9 Frames, More Than The Maximum Sequence Length (8): It Is Left Out Of The Training.
...
220 Sequences Of The Training Split Are Longer Than The Maximum Sequence Length (8) And Are Left Out Of The Training.
25 Sequences Of The Validation Split Are Longer Than The Maximum Sequence Length (8) And Are Left Out Of The Validation.
28 Sequences Of The Test Split Are Longer Than The Maximum Sequence Length (8) And Are Left Out Of The Test.
- Train Data Folder       : train (500 sequences, 3957 frames, 280 sequences of at most 8 frames)
- Iterations Per Epoch    : 18

ValueError: 500 sequences cannot be used for the frame selection, which needs every wanted class exactly 2 times ('Selection.frames_per_class') : labels.json
 - train_000000 : circle 1 frames, triangle 1 frames, star 1 frames
 - train_000001 : circle 1 frames, triangle 1 frames, star 1 frames
 ...
 - train_000019 : circle 1 frames, triangle 1 frames, star 1 frames
 - ... and 480 more
```
The warnings name every left-out sequence, in red; the run goes on with the 280 sequences of at most 8 frames. The error lists the first 20 invalid sequences and stops the training.

### Step 2.9 — `build_native_loader()`
**Example output** (recorded with 8 workers, when the number of workers was a setting; every loader now has 1)
```text
len(loader) = 32   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.10 — `get_test_sample_classes()` + `choose_sample_identifiers()`
`get_sample_class_indices()` lists the wanted classes of the frames of every test sequence, the unselected frames left out. Every sequence counts for its most frequent class, the lowest class index on a tie: here every sequence holds the three wanted classes once, so they all fall in the group of `circle`, and the 8 sample sequences are spread evenly over that group.

**Example output**
```text
get_test_sample_classes() -> 60 entries, e.g. {31: [1, 3, 2], 38: [2, 3, 1], 16: [3, 2, 1], 37: [3, 2, 1]}
sample_image_identifiers  -> {0, 8, 17, 25, 34, 42, 51, 59}
names                     -> test_000000, test_000008, test_000017, test_000025, test_000034, test_000042, test_000051, test_000059
```

### Step 2.11 — `build_model()` → `FrameSelector.__init__()`
The frame selector is the sequence labeller (`sequence_labelling_workflow.md`, Step 2.11): one vector per frame from the temporal module, classified by the head into the wanted classes and the unselected class. It adds `selection_loss_weight` and the `wanted_class_mask` buffer, ones at construction; the trainer then sets 0 on the unselected class. The mask limits the selection loss to the wanted classes, and, saved with the weights, it gives the unselected class to the exported and locally run models, which have no manifest (`get_unselected_class_index()`).

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          RecurrentTemporalModule    148,992 parameters      GRU 64 -> 128, bidirectional, output_features 256
model.classification_head      ClassificationHead          16,708 parameters      Dropout -> Linear(256, 64) -> ReLU -> Dropout -> Linear(64, 4)
model.class_weights            buffer [4] torch.float32 [1.0, 1.0, 1.0, 1.0]
model.wanted_class_mask        buffer [4] torch.float32 [0.0, 1.0, 1.0, 1.0]      <- unselected class set to 0 by the trainer
model.selection_loss_weight    1.0
total                                                     349,348 parameters
```

### Step 2.12 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
The selection loss is tracked next to the per-frame loss; the total loss is their sum.

**Example output**
```text
pretrained_checkpoint_paths = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
{'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'selection_loss': 'Selection Loss', 'accuracy': 'Accuracy'}
MetricTracker: batch_size 16, validation_iteration_ratio 25, sample_name 'Sequences', counts_frames True
```

### Step 2.13 — `print_experiment_information()` (→ `describe_split()` → `count_split_sequences()`)
**Example output**
```text
- Run Name                : SEL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4_FPC1-SLW1
...
- Train Data Folder       : train (500 sequences, 3950 frames, 500 sequences of at most 50 frames)
- Validation Data Folder  : validation (60 sequences, 481 frames, 60 sequences of at most 50 frames)
- Test Data Folder        : test (60 sequences, 463 frames, 60 sequences of at most 50 frames)
- Iterations Per Epoch    : 32
- Batch Size              : 16
...
- MLflow Tracking         : Off
```

### Step 2.14 — `print_summary()` / `view_input_batches()` / `restore_last_model()`
The summary panels are those of the sequence labelling, the temporal module panel giving `Output Features : 256` and the head panel `Input Features : 256` and `Number Of Classes : 4`. In the backbone panel, blocks 2 and 4 change the channel count with 2 layers, so they add no residual shortcut and list `Squeeze-and-Excitation` alone (`ConvolutionBlock.count_residual_additions()`, `bug_standards.md` H4, dropped). With `view_input_images`, the frames of the wanted classes are captioned with their number and their class and the unselected frames with their number only, their caption being empty (`get_frame_captions()`), and the information panel of the sequence starts with the frame of every wanted class (`get_sample_caption()`).

**Example: the modules and shapes of the backbone panel, and the model size panel** (the model rebuilt on the CPU)
```text
- Convolution Block 1   Modules : Squeeze-and-Excitation                       [1, 3, 256, 256]  -> [1, 16, 128, 128]
- Convolution Block 2   Modules : Squeeze-and-Excitation                       [1, 16, 128, 128] -> [1, 32, 64, 64]
- Convolution Block 3   Modules : Residual Shortcuts, Squeeze-and-Excitation   [1, 32, 64, 64]   -> [1, 32, 32, 32]
- Convolution Block 4   Modules : Squeeze-and-Excitation                       [1, 32, 32, 32]   -> [1, 64, 16, 16]
- Convolution Block 5   Modules : Residual Shortcuts, Squeeze-and-Excitation   [1, 64, 16, 16]   -> [1, 64, 16, 16]
Final Backbone output shape: [1, 64, 16, 16]
Model Size: Backbone 183,648, Temporal Module 148,992, Classification Head 16,708, Total Model 349,348 parameters (1.40 MB)
```

**Example: the captions of the first training sequence**
```text
get_sample_caption = 'Selected : circle 9, triangle 3, star 5'
get_frame_captions = ['', '', '', 'triangle', '', 'star', '', '', '', 'circle', '']
```

**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of the redirected `configuration_synthetic_sequences.yaml` shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the error images and the MLflow tracking on. Its run name is `SEL_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4_FPC1-SLW1`; with batches of 4 sequences, an epoch is 125 iterations.

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
Tensorboard/Train/        Total Loss, Classification Loss, Selection Loss, Accuracy, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Selection Loss, Accuracy                   (iteration 10)
MLflow Run Created In The Experiment 'frame_selection' : 618a35766a1b4c60a9f2703b93255530 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```
In the captured console this line came out during the progress bar of the first test evaluation (iteration 6).

### Step 3.3 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12, so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6), with the label dictionary of the run. The checkpoint holds the `wanted_class_mask`, so the unselected class travels with the weights.

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   4,289,933 bytes      Weights/Iteration_6/model_000006.pt   4,289,933 bytes
Weights/Iteration_9/model_000009.pt   4,289,997 bytes      Weights/Iteration_12/model_000012.pt  4,289,997 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
model_state: 95 tensors, the first 'class_weights', 'wanted_class_mask', then 'backbone.convolution_blocks.0. ...'
mlflow_run_id: None (iterations 3 and 6), '618a35766a1b4c60a9f2703b93255530' (iterations 9 and 12)
wanted_class_mask in every checkpoint: [0.0, 1.0, 1.0, 1.0]
```
The file sizes were measured before the checkpoints held the label dictionary of the run (2026-10-21): the 64 more bytes of iterations 9 and 12 come with their MLflow run identifier. The checkpoint of iteration 2000 of the README screenshot run of this configuration (`suggestions.md` K), the same 95 tensors at 256 pixels with its label dictionary and MLflow run identifier, holds 4,292,067 bytes.

### Step 3.4 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration, after its checkpoint and its summary (see `classification_workflow.md`, Step 3.7), see Stage 4.

### Step 3.5 — `get_next_batch()` → `SequenceDataset.__getitem__()` + `collate_padded_sequences()`
The sample of the sequence labelling (`sequence_labelling_workflow.md`, Step 3.5: every frame read with `read_image()`, a batch failing to load dropped and the next one taken): the whole sequence with the `[T]` class of every frame. `collate_padded_sequences()` pads every sequence of the batch to its longest sequence with black frames, a False mask and the label -100.

**Example: the first batch of the run** (sequences of 7, 12, 4 and 10 frames: the longest has 12 frames, so every sequence is padded to 12)
```text
images               [4, 12, 3, 64, 64] torch.uint8
sequence_mask        [4, 12] torch.bool
labels               [4, 12] torch.int64
                     [[2, 0, 0, 3, 0, 0, 1, -100, -100, -100, -100, -100], [0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1],
                      [3, 2, 1, 0, -100, -100, -100, -100, -100, -100, -100, -100], [0, 0, 3, 2, 0, 0, 0, 0, 1, 0, -100, -100]]
sample_identifiers   [4] torch.int64     [102, 300, 207, 32]
```

### Step 3.6 — `MetricTracker.count_frames()`
**Example: the frames of the first two intervals**
```text
interval_frames = 130 (iterations 1 to 4), 129 (iterations 5 to 8)
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
**Example: before and after**
```text
images [4, 12, 3, 64, 64] torch.uint8 -> torch.float32     labels [4, 12] torch.int64     sequence_mask [4, 12] torch.bool
```

### Step 3.8 — `augment_batch()`
The augmentation of the sequence labelling (`sequence_labelling_workflow.md`, Step 3.8): with `shuffle_frames`, the frame labels move with their frames.

**Example: which sequences of the first batch were changed** (`probability: 0.4`)
```text
images: changed per sample [True, False, False, False]      labels: changed per sample [False, False, False, False]
```

### Step 3.9 — `forward_pass()` → `FrameSelector.forward(sequence_mask=..., labels=...)`
The forward pass of the sequence labeller (`sequence_labelling_workflow.md`, Step 3.9) gives `class_logits` `[B, T, K]` and the per-frame `classification_loss`; the frame selector then adds the `selection_loss`, the selection loss times its weight (a zero tensor when the weight is 0).

**Example: the first batch through the model**
```text
encode_frames       -> (4, 12, 64)    zero vectors per sequence [5, 0, 8, 2]      <- the padding frames
temporal_module     -> (4, 12, 256)   one vector per frame (128 per direction)
classification_head -> (4, 12, 4)
```

**Example: the dictionary returned in training mode**
```text
class_logits          [4, 12, 4] torch.float32
class_probabilities   [4, 12, 4] torch.float32
classification_loss   1.3489 (scalar)            <- close to ln 4 = 1.3863
selection_loss        2.0299 (scalar)            <- the mean of ln 7, ln 12, ln 4 and ln 10 is 2.0299
```

### Step 3.10 — `compute_frame_selection_loss()`
The per-frame loss answers "which class is this frame"; the selection loss answers "which frames of this sequence are the best for this class". For every wanted class of every sequence, the softmax of the logit of the class is taken over the real frames of the sequence (the padding frames get no probability), and its cross entropy with the target `1/N` on each of the `N` true frames of the class is computed; the losses are averaged over the wanted classes (from `wanted_class_mask`) that have true frames in their sequence. The frames of a sequence thus compete for every class, as they do in the decodings.

A model giving the same logit to all the frames of a sequence gets `ln T` for a sequence of `T` real frames, whatever the class: the sequences of the first batch have 7, 12, 4 and 10 real frames, and the mean of their `ln T` is 2.0299, the selection loss of Step 3.9.

### Step 3.11 — `compute_batch_metrics()` → `compute_frame_accuracy()`
The share of the real frames whose most probable class is their class, the unselected frames included.

**Example: the first batch**
```text
accuracy = 0.6364          (21 of the 7 + 12 + 4 + 10 = 33 real frames, i.e. the share of the unselected frames: 3 frames per
                            sequence are selected)
```

### Step 3.12 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
**Example: the Train curves**
```text
Total Loss           1: 3.3788, 2: 3.3977, 3: 3.3431, 4: 3.3463, 5: 3.0884, 6: 3.0870, 7: 3.2481, 8: 3.2734, 9: 3.1994, 10: 2.9889, 11: 2.9916, 12: 3.1028
Classification Loss  1: 1.3489, 2: 1.3133, 3: 1.3075, 4: 1.2753, 5: 1.2602, 6: 1.2139, 7: 1.1103, 8: 1.0648, 9: 1.0223, 10: 1.1266, 11: 1.0786, 12: 0.9288
Selection Loss       1: 2.0299, 2: 2.0844, 3: 2.0355, 4: 2.0710, 5: 1.8282, 6: 1.8731, 7: 2.1378, 8: 2.2086, 9: 2.1771, 10: 1.8623, 11: 1.9130, 12: 2.1741
Accuracy             1: 0.6364, 2: 0.6364, 3: 0.6250, 4: 0.6250, 5: 0.5556, 6: 0.5556, 7: 0.6757, 8: 0.6842, 9: 0.6842, 10: 0.5714, 11: 0.6000, 12: 0.6923
```
The total loss of iteration 1 is 1.3489 + 2.0299 = 3.3788. Over the 12 iterations the classification loss goes down from 1.3489 to 0.9288, while the selection loss stays between 1.8282 and 2.2086.

### Step 3.13 — `training_loss.backward()` + `step_optimizer()` + validation pass
**Example: the norm of the gradients of the first steps, measured for this document and applied as they are, the learning rate and the validation batch**
```text
gradient norm: iteration 1: 1.0351   iteration 2: 0.9668   iteration 3: 0.839        learning rate 0.001 (constant)
Tensorboard/Validation at iteration 10: Total Loss 3.3946, Classification Loss 1.2079, Selection Loss 2.1867, Accuracy 0.6667
```

### Step 3.14 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 13.4659, 'classification_loss': 5.2449, 'selection_loss': 8.2209, 'accuracy': 2.5227, 'interval_frames': 130}
after:  {'interval_batches': 4, 'total_loss': 3.3665, 'classification_loss': 1.3112, 'selection_loss': 2.0552, 'accuracy': 0.6307, 'interval_frames': 130}
MLflow, step 4: {'train/Total Loss': 3.3665, 'train/Classification Loss': 1.3112, 'train/Selection Loss': 2.0552, 'train/Accuracy': 0.6307,
                 'train/throughput': 4.5455, 'train/interval_seconds': 3.52, 'learning_rate': 0.001, 'train/frame_throughput': 36.9323}
```

**Example output: iterations 4 and 12**
```text
===========================================================================================================================================================
|     Iteration: 4    |     Total Sequences: 16     |    Interval Sequences: 16    |   Throughput: 4.5 Sequences Per Second  |   36.9 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   3.3665
Moving Average of Train Classification Loss            :   1.3112
Moving Average of Train Selection Loss                 :   2.0552
Moving Average of Train Accuracy                       :   0.6307
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:03.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
...
===========================================================================================================================================================
|    Iteration: 12    |     Total Sequences: 48     |    Interval Sequences: 16    |   Throughput: 9.7 Sequences Per Second  |   81.8 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   3.0707  -0.1035 Best
Moving Average of Valid Total Loss                     :   3.3946
Moving Average of Train Classification Loss            :   1.0391  -0.1232 Best
Moving Average of Valid Classification Loss            :   1.2079
Moving Average of Train Selection Loss                 :   2.0316  +0.0197
Moving Average of Valid Selection Loss                 :   2.1867
Moving Average of Train Accuracy                       :   0.6370  +0.0192 Best
Moving Average of Valid Accuracy                       :   0.6667
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:01.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
```

### Step 3.15 — end of the loop
The checkpoint of iteration 12 is already saved and evaluated inside its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 60 Test Sequences, 463 Frames · 2026-10-10 07:20 ───────────────      <- the evaluation of iteration 12, last in the iteration
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (60 sequences, 463 frames), with `per_class_report: True`, `number_sample_images: 8`, `top_k: 2`, `save_predictions: True`, `error_dumps` activated with `maximum_per_class_pair: 2`, and the MLflow tracking on. Both checkpoints give `unselected` as the most probable class of every frame, with almost the same probabilities on all the frames of a sequence. The outputs of iteration 6 were captured when its checkpoint was still saved before its training step, after 5 steps; it now holds 6 steps. They were also captured before the per-class recall intervals were added (2026-11-01): `class_metrics.csv` now holds `hungarian_accuracy_lower` and `hungarian_accuracy_upper` after `hungarian_accuracy` (the decoding of this run), and with `per_class_report` the report shows a `Per Class Recall` table after `Per Class Selection` (Step 4.13).

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1).

**Example output: the progress bar** (60 sequences, batch size 4)
```text
Test Evaluation Iteration 12: 100%|██████████| 15/15 [00:01<00:00,  8.48it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The selection hook keeps, for every test sequence, the class probabilities and the true classes of its real frames, and its identifier: the decodings need the whole sequence, so they run once the whole split is evaluated (Step 4.7). It also holds the folder and the counts of the error images.

**Example: the accumulator at the end of the loop of iteration 12**
```text
'probabilities':         60 arrays, the first [10, 4] float64        <- one per sequence, its real frames only
'frame_labels':          60 arrays, the first [2, 0, 0, 1, 0, 0, 0, 3, 0, 0]
'sequence_identifiers':  60 integers
'errors_folder':         <run>/Evaluation/Iteration_12/Errors
'error_counts':          2 images for each of circle, triangle and star     <- keyed by the folder name, the wanted class
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
**Example: the predictions of a test batch**
```text
class_logits          [4, 12, 4] torch.float32
class_probabilities   [4, 12, 4] torch.float32
```

### Step 4.4 — `accumulate_evaluation_batch()` → `get_sequence_predictions()`
`get_sequence_predictions()` reads the `[T, K]` probabilities (in float64) and the `[T]` true classes of the real frames of a sequence, the input of the decodings, the metrics and the images. Then `dump_errors()` runs while the frames are still in the batch.

### Step 4.5 — `dump_errors()` → `decode_sequence()` + `build_selection_image()`
Every test sequence is decoded with the decoding of the run; a wanted class is wrong when its chosen frames are not exactly its true frames. The image of the sequence (the one of the sample images, Step 4.6) is built once and saved as `Errors/<CLASS>/<sequence>.png` in the folder of every class picked wrong, at most `maximum_per_class_pair` per class.

**Example: the error images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Errors/
├── circle/      test_000001.png   test_000053.png
├── star/        test_000001.png   test_000053.png
└── triangle/    test_000001.png   test_000053.png      <- the same image in the 3 folders (69,762 and 83,068 bytes)
report line: "Error Images : 6 in 3 classes, at most 2 per class"
```
The first two sequences evaluated have all three classes wrong, so they fill the 2 images of every class.

### Step 4.6 — `save_sample_images()` → `build_sample_image()` → `build_frame_selection_image()`
The selection hook decodes the sequence with the decoding of the run (`decode_sequence()`) and draws it with `build_selection_image()`:
- the frames as numbered thumbnails of 160 pixels, 8 per row, with the borders of the selection: solid green for a frame chosen for its true class, solid red for a frame chosen for a class it does not hold, dashed yellow for a true frame left unchosen, each border labelled with its class and probability, the rightly unselected frames dimmed;
- under the grid, the probability heat map of every wanted class over the frames (the chosen frames marked with a dot, the true frames outlined in white), then a legend bar;
- above, a caption bar `<right>/<selections> selections right`, green when all the selections are right and red otherwise.

**Example: the sample images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Samples/
test_000000.png  test_000008.png  test_000017.png  test_000025.png
test_000034.png  test_000042.png  test_000051.png  test_000059.png
build_sample_image -> (480, 1316, 3) uint8      a sequence of 9 to 16 frames: two rows of 8 thumbnails
                      (316, 164 x T + 4, 3)     a sequence of T frames, at most 8: one row
```

### Step 4.7 — `compute_evaluation_results()`
The frame selection trainer replaces the whole computation:
- both decodings choose the frames of every sequence on the same probabilities (Step 4.9), and each is scored with `compute_selection_metrics()` (Step 4.10);
- `compute_ranking_metrics()` scores the probabilities before any decoding (Step 4.11);
- the frame metrics of the classification are computed on the most probable class of every frame (Step 4.12);
- the decoding of the run gives the confidence thresholds, the interval of the selection accuracy of every wanted class, the per-class file and the decoded confusion matrix (Step 4.13), and the selections file (Step 4.14).

The headline metric is the selection accuracy of the decoding of the run.

**Example: the results of iteration 12**
```text
evaluated_samples = '60 Test Sequences, 463 Frames'
metrics           = {'Hungarian Selection Accuracy': 0.1222, 'Hungarian Fully Right Sequences': 0.0, 'Hungarian Unselected Precision': 0.6537,
                     'Hungarian Unselected Recall': 0.6537, 'Greedy Selection Accuracy': 0.1222, 'Greedy Fully Right Sequences': 0.0,
                     'Greedy Unselected Precision': 0.6537, 'Greedy Unselected Recall': 0.6537, 'Top-2 Selection Accuracy': 0.3333,
                     'Mean True Frame Rank': 4.1444, 'Frame Accuracy': 0.6112, 'Frame Balanced Accuracy': 0.25, 'Frame Macro F1': 0.1897}
notes             = ["Images, Thresholds And Confusion Matrix : hungarian decoding ('Selection.decoding')",
                     'Error Images : 6 in 3 classes, at most 2 per class', 'Sample Images : 8 fixed test sequences']
headline_metric   = 'Hungarian Selection Accuracy'
files             = ['Class Thresholds', 'Class Metrics', 'Confusion Matrix Table', 'Confusion Matrix Image', 'Selections', 'Error Images', 'Sample Images']
class_scalars     = 9: Hungarian Selection Accuracy/circle, Greedy Selection Accuracy/circle, Mean True Frame Rank/circle, ...
tables            = Per Class Selection: Class, True Frames, Hungarian Accuracy, Greedy Accuracy, Top-2 Accuracy, Mean Rank,
                    Threshold @ P0.99, Recall @ P0.99
```

### Step 4.8 — `compute_frame_ranks()`
The rank of every frame for every wanted class among the frames of its sequence ordered by their probability for that class, 1 for the most probable (tied frames take the best of their ranks). It is the input of the ranking metrics (Step 4.11) and of the `true_ranks` column of the selections file.

**Example: the test sequence test_000001 at iteration 12** (10 frames; probabilities in the class order unselected, circle, triangle, star)
```text
true classes [2, 0, 0, 1, 0, 0, 0, 3, 0, 0]       <- triangle at frame 0, circle at frame 3, star at frame 7
frame 0 [0.3982, 0.1816, 0.1945, 0.2256]     frame 5 [0.4403, 0.164, 0.1905, 0.2053]
frame 1 [0.4189, 0.1721, 0.1918, 0.2171]     frame 6 [0.4356, 0.1666, 0.1924, 0.2054]
frame 2 [0.4315, 0.1668, 0.1902, 0.2115]     frame 7 [0.4264, 0.1714, 0.1956, 0.2065]
frame 3 [0.4385, 0.1641, 0.1894, 0.208]      frame 8 [0.4113, 0.1793, 0.2006, 0.2088]
frame 4 [0.4412, 0.1632, 0.1895, 0.2061]     frame 9 [0.3876, 0.1917, 0.2078, 0.2129]
ranks [T, W] (circle, triangle, star):
   [[2, 4, 1], [4, 6, 2], [6, 8, 4], [8, 10, 6], [10, 9, 8], [9, 7, 10], [7, 5, 9], [5, 3, 7], [3, 2, 5], [1, 1, 3]]
```
The true circle frame (3) has the rank 8 for circle, the true triangle frame (0) the rank 4 for triangle, the true star frame (7) the rank 7 for star.

### Step 4.9 — `decode_frame_selection()` → `decode_hungarian_selection()` / `decode_greedy_selection()`
A decoding turns the `[T, K]` probabilities of a sequence into the `[W, N]` frames chosen for the `W` wanted classes, `N = frames_per_class` each, ordered by decreasing probability for their class (`NO_FRAME` where the sequence has fewer frames than selections). The same function runs in the evaluation, the local inference and the Triton decoder.
- **Hungarian**: the gain of choosing frame `f` for class `c` is `log p(f, c) − log p(f, unselected)`, every class column repeated `N` times, and `linear_sum_assignment()` finds the assignment with the largest total gain, every frame chosen at most once: the most probable labelling of the whole sequence in which every wanted class holds `N` frames. A frame wanted by two classes goes to the class that loses least without it.
- **Greedy**: every frame takes its most probable class; a wanted class taken by too many frames keeps its most probable ones (the others become unselected); then every class still short of frames, in class order, takes the unselected frames most probable for it. It is cheaper, but it can give a frame to a class that only slightly prefers it over another class that has no better frame.

`convert_selection_to_frame_labels()` then gives every chosen frame the class of its row and every other frame the unselected class, for the unselected precision and recall and the decoded confusion matrix.

**Example: the two decodings of test_000001 at iteration 12** (probabilities of Step 4.8)
```text
hungarian: selected [W, N] [[9], [8], [0]]   decoded frame classes [3, 0, 0, 0, 0, 0, 0, 0, 2, 1]
greedy:    selected [W, N] [[9], [8], [0]]   decoded frame classes [3, 0, 0, 0, 0, 0, 0, 0, 2, 1]
```
Every frame has `unselected` as its most probable class, so the greedy decoding completes the classes in order: circle takes frame 9 (0.1917, its most probable frame), triangle frame 8 (0.2006, frame 9 being taken), star frame 0 (0.2256). The Hungarian assignment gives the same frames here, and none of the three is a true frame. On the whole test split of this run the two decodings never differ; on the reduced split of Stage 5 they differ at iteration 3.

### Step 4.10 — `compute_selection_metrics()`
The scores of one decoding over all the test sequences:
- **Selection Accuracy**: the share of the true frames of the wanted classes chosen for their class, per class (the recall of the class) and over all the classes, with the right and true frames of every sequence and wanted class (`sequence_class_right_frames`, `sequence_class_true_frames`) for the interval of Step 4.13;
- **Fully Right Sequences**: the share of the sequences in which every wanted class gets exactly its true frames;
- **Unselected Precision** and **Unselected Recall**: the frames left unchosen that are truly unselected, among the frames left unchosen and among the truly unselected frames.

**Example: `Evaluation/Iteration_12/confusion_matrix.csv`** (the decoded classes of the Hungarian decoding; the same at iteration 6)
```text
true \ predicted,unselected,circle,triangle,star
unselected,185,32,33,33
circle,28,7,8,17
triangle,36,13,8,3
star,34,8,11,7
```

**Example: how the numbers of iteration 12 are obtained**
```text
selection accuracy   = (7 + 8 + 7) / (60 + 60 + 60)    = 22 / 180  = 0.1222
   circle 7 / 60 = 0.1167   triangle 8 / 60 = 0.1333   star 7 / 60 = 0.1167
unselected precision = 185 / (185 + 28 + 36 + 34)      = 185 / 283 = 0.6537
unselected recall    = 185 / (185 + 32 + 33 + 33)      = 185 / 283 = 0.6537
fully right sequences = 0.0
```
Every sequence has 3 true selected frames and the decoding chooses 3 frames, so the frames left unchosen are as many as the truly unselected frames (283), and the unselected precision equals the unselected recall.

### Step 4.11 — `compute_ranking_metrics()`
The scores of the probabilities before any decoding, from the ranks of the true frames (Step 4.8): the **Top-k Selection Accuracy** is the share of the true frames among the `top_k` frames most probable for their class, and the **Mean True Frame Rank** the mean rank of the true frames, per class and over all of them. They tell how far a wrong selection was from the right frame.

**Example: iteration 12**
```text
Top-2 Selection Accuracy 0.3333     Mean True Frame Rank 4.1444
   circle   top-2 0.4000   mean rank 3.9500
   triangle top-2 0.3500   mean rank 3.9500
   star     top-2 0.2500   mean rank 4.5333
```
The mean of the three is 4.1444, all the classes having 60 true frames.

### Step 4.12 — the frame metrics: `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()`
The metrics of the classification (see `classification_workflow.md`, Steps 4.9 to 4.11) on the most probable class of every real frame, before any decoding.

**Example: iteration 12**
```text
Frame Accuracy          = 283 / 463                        = 0.6112   (every frame predicted unselected: only the 283 unselected frames are right)
Frame Balanced Accuracy = (1.0 + 3 x 0.0) / 4              = 0.2500
Frame Macro F1          = (2 x 0.6112 x 1.0 / 1.6112) / 4  = 0.1897
```

### Step 4.13 — `find_selection_thresholds()` + `write_class_metrics_file()` + `save_confusion_matrix_table()` / `save_confusion_matrix_image()`
`find_selection_thresholds()` takes, for every wanted class, the probabilities of the frames the decoding of the run chose for it and whether each is a true frame, and finds the lowest probability from which these choices reach `target_precision`, with the share of all the true frames of the class kept. In deployment, a chosen frame below the threshold of its class, or every frame of a class without threshold, is flagged as low confidence for a human to check. The thresholds are saved in `class_thresholds.json`, with the decoding and the unselected class. The selection accuracy of every wanted class with the decoding of the run is its recall: the frames of a sequence being chosen together, `compute_bootstrap_recall_intervals()` of `vision_training/evaluation/recall_intervals.py` gives its 95% Korn-Graubard confidence interval from the right and true frames of every sequence of Step 4.10: the variance of the accuracy over 1000 resamples of the test sequences with replacement (fixed seed) gives an effective sample size, reduced for the degrees of freedom of the sequences holding the class, at which the exact (Clopper-Pearson) interval is taken, or the exact interval over those sequences when they all have the same accuracy, so that it never has zero width. The metrics of every wanted class go to `class_metrics.csv` whatever the per-class report, the bounds as `<decoding>_accuracy_lower` and `<decoding>_accuracy_upper` after the accuracy of the decoding of the run (`hungarian_accuracy_lower` and `hungarian_accuracy_upper` here), and with the per-class report `build_recall_interval_table()` shows them in the `Per Class Recall` table (`True Frames`, `Hungarian Accuracy`, `95% Lower`, `95% Upper`, worst accuracy first) after the `Per Class Selection` table. The confusion matrix of the decoded classes is saved as a table, and as an image (titled with the decoding and its selection accuracy) only with the per-class report.

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
At iteration 6, the triangle frames chosen with a probability of at least 0.2261 are all true frames, but they are 1 of the 60 true triangle frames (recall 0.0167).

**Example: `Evaluation/Iteration_12/class_metrics.csv`**
```text
class_index,class,true_frames,hungarian_accuracy,greedy_accuracy,top_k_accuracy,mean_rank,threshold_at_target_precision,recall_at_target_precision
1,circle,60,0.116667,0.116667,0.400000,3.950000,,
2,triangle,60,0.133333,0.133333,0.350000,3.950000,,
3,star,60,0.116667,0.116667,0.250000,4.533333,,
```

### Step 4.14 — `write_selections_file()`   (only with `save_predictions`)
One row per test sequence and wanted class, so that the choices of both decodings can be studied without running the model: the true frames with their probabilities and ranks, the frames chosen by each decoding with their probabilities and the number of right ones, and the low-confidence flags of the decoding of the run (`flag_low_confidence()`). The frames are named by their file, several frames being separated by `;`. The rows follow the order in which the sequences were evaluated.

**Example: `Evaluation/Iteration_12/selections.csv`** (181 lines, 22,538 bytes; first rows)
```text
sequence,class,true_frames,true_probabilities,true_ranks,hungarian_frames,hungarian_probabilities,hungarian_right_selections,greedy_frames,greedy_probabilities,greedy_right_selections,hungarian_low_confidence
test_000001,circle,test_000001/000003.png,0.164078,8,test_000001/000009.png,0.191714,0,test_000001/000009.png,0.191714,0,1
test_000001,triangle,test_000001/000000.png,0.194501,4,test_000001/000008.png,0.200584,0,test_000001/000008.png,0.200584,0,1
test_000001,star,test_000001/000007.png,0.206539,7,test_000001/000000.png,0.225646,0,test_000001/000000.png,0.225646,0,1
...
test_000019,circle,test_000019/000004.png,0.194757,1,test_000019/000004.png,0.194757,1,test_000019/000004.png,0.194757,1,1
```
No class has a threshold at iteration 12, so every chosen frame is flagged (`1`), the right choice of circle in test_000019 included.

### Step 4.15 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 60 Test Sequences, 463 Frames · 2026-10-10 07:20 ────────────────────────────────────────────────────────
  Hungarian Selection Accuracy     0.1222   Hungarian Fully Right Sequences   0.0000
  Hungarian Unselected Precision   0.6537   Hungarian Unselected Recall       0.6537
  Greedy Selection Accuracy        0.1222   Greedy Fully Right Sequences      0.0000
  Greedy Unselected Precision      0.6537   Greedy Unselected Recall          0.6537
  Top-2 Selection Accuracy         0.3333   Mean True Frame Rank              4.1444
  Frame Accuracy                   0.6112   Frame Balanced Accuracy           0.2500
  Frame Macro F1                   0.1897
  Images, Thresholds And Confusion Matrix : hungarian decoding ('Selection.decoding')
  Error Images : 6 in 3 classes, at most 2 per class
  Sample Images : 8 fixed test sequences

  Per Class Selection
                            Hungarian     Greedy      Top-2     Mean   Threshold    Recall
   Class      True Frames    Accuracy   Accuracy   Accuracy     Rank     @ P0.99   @ P0.99
  ─────────────────────────────────────────────────────────────────────────────────────────
   circle              60      0.1167     0.1167     0.4000   3.9500           —         —
   star                60      0.1167     0.1167     0.2500   4.5333           —         —
   triangle            60      0.1333     0.1333     0.3500   3.9500           —         —
  Worst hungarian accuracy first, ties in class order. Accuracy : share of the true frames of the class chosen for it.
  Rank : mean rank of the true frames among the frames ordered by their probability for the class, 1 being the best.
  Threshold : probability from which the hungarian choices reach the precision 0.99, with the share of the true frames
  kept. — : never reached.
  - Results : Results.txt
  - Class Thresholds : class_thresholds.json
  - ...
```

**Example: TensorBoard and MLflow after the two evaluations**
```text
Tensorboard/Test: 22 scalars (13 global + 9 per class), 9 images (8 samples + Confusion Matrix)
    scalar Hungarian Selection Accuracy    6: 0.1222, 12: 0.1222
    scalar Mean True Frame Rank            6: 4.1000, 12: 4.1444
    scalar Mean True Frame Rank/star       6: 4.4500, 12: 4.5333
MLflow: test/Hungarian Selection Accuracy, ..., test/Mean True Frame Rank, test/Top-2 Selection Accuracy,
        test/best_Hungarian Selection Accuracy (steps 6, 12), 9 test_class/<metric>/<class>
artifacts: Results.txt, configuration/training_configuration.yaml, metric_names.json,
           evaluation/iteration_<6|12>/class_metrics.csv, class_thresholds.json, confusion_matrix.png
```
This run was recorded before the record of the data existed (2026-10-15): a new run now also sends `data/data.json` (`start_experiment_tracking()`), as the README screenshot run of this configuration does.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12, captured when those of iterations 3, 6 and 9 were still saved before their training step) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 20 sequences of the test split (its `labels.json` reduced, its `data/` folder linked). They were also captured before the per-class recall intervals (2026-11-01, Stage 4).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5). The test sequences are checked with the `Selection` section of the run, and both decodings are scored on every checkpoint.

### Step 5.2 — trainer construction
The trainer is built from the `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it rather than those of the mapping file (`classification_workflow.md`, Step 5.2).

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
**Example output** (the four checkpoints tie on the Hungarian selection accuracy, so the four iterations are listed in their evaluation order)
```text
-------------------------------------------------------------
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
-------------------------------------------------------------
  - Best Checkpoint : iterations 12, 9, 6, 3 tie (Hungarian Selection Accuracy 0.1500)

iteration,Hungarian Selection Accuracy,Hungarian Fully Right Sequences,Hungarian Unselected Precision,Hungarian Unselected Recall,
Greedy Selection Accuracy,Greedy Fully Right Sequences,Greedy Unselected Precision,Greedy Unselected Recall,Top-2 Selection Accuracy,
Mean True Frame Rank,Frame Accuracy,Frame Balanced Accuracy,Frame Macro F1,Hungarian Selection Accuracy/circle, ... ,Mean True Frame Rank/star
12,0.15,0.0,0.6842105263157895,0.6842105263157895,0.15,0.0,0.6842105263157895,0.6842105263157895,0.38333333333333336,4.05,0.6129032258064516,0.25,0.19, ...
9,0.15,0.0,0.6842105263157895,0.6842105263157895,0.15,0.0,0.6842105263157895,0.6842105263157895,0.38333333333333336,3.9833333333333334,0.6129032258064516,0.25,0.19, ...
6,0.15,0.0,0.6842105263157895,0.6842105263157895,0.15,0.0,0.6842105263157895,0.6842105263157895,0.38333333333333336,4.016666666666667,0.6129032258064516,0.25,0.19, ...
3,0.15,0.0,0.6421052631578947,0.6421052631578947,0.18333333333333332,0.0,0.6631578947368421,0.6631578947368421,0.4,4.016666666666667,0.6129032258064516,0.25,0.19, ...
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 3 on the 20 sequences**, where the two decodings differ (shortened)
```text
-----------------------------
Evaluating : model_000003.pt
-----------------------------
Iteration 3 · 20 Test Sequences, 155 Frames · 2026-10-10 07:20 ─────────────────────────────────────────────────────────
  Hungarian Selection Accuracy     0.1500   Hungarian Fully Right Sequences   0.0000
  Hungarian Unselected Precision   0.6421   Hungarian Unselected Recall       0.6421
  Greedy Selection Accuracy        0.1833   Greedy Fully Right Sequences      0.0000
  Greedy Unselected Precision      0.6632   Greedy Unselected Recall          0.6632
  Top-2 Selection Accuracy         0.4000   Mean True Frame Rank              4.0167
  Frame Accuracy                   0.6129   Frame Balanced Accuracy           0.2500
  Frame Macro F1                   0.1900
  ...
                            Hungarian     Greedy      Top-2     Mean   Threshold    Recall
   Class      True Frames    Accuracy   Accuracy   Accuracy     Rank     @ P0.99   @ P0.99
  ─────────────────────────────────────────────────────────────────────────────────────────
   triangle            20      0.0500     0.1500     0.3500   3.8500           —         —
   circle              20      0.2000     0.2000     0.5000   3.9500           —         —
   star                20      0.2000     0.2000     0.3500   4.2500      0.2495    0.1000
```
The greedy decoding finds 2 more true triangle frames than the Hungarian one (0.1500 against 0.0500): the Hungarian assignment maximises the total gain of the three classes, not the number of right frames. The best checkpoint is still chosen on the Hungarian selection accuracy, the decoding of the run.
