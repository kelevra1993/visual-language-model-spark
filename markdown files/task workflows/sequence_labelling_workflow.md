# Sequence Labelling Workflow

This document follows one sequence labelling run (many to many, M2M: one class per frame of the windows) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The sequence labelling trainer is the sequence classification trainer with one class per frame: the steps it shares with the sequence classification (windows, padding, frame encoder) are explained in `sequence_classification_workflow.md`, and the steps shared by every task in `classification_workflow.md`; here they keep a short explanation and the real output of the labelling run, and the steps specific to the frame labels (labels per frame, temporal module keeping the time axis, frame metrics, segment metrics of the whole sequences, frame error images) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training. The synthetic dataset of the configuration, `synthetic_changing_shapes`, is not on this machine, so a small one was generated in a temporary folder with `create_synthetic_dataset(task="sequence_labelling")`: 500 training, 60 validation and 60 test sequences of 2 to 12 frames of 128 pixels, a shape changing class during the sequence, 9 shape classes. The data folders and the mapping file of the configuration were redirected to it.
- **Stages 1 and 2**: `configurations/sequence_labelling/configuration_synthetic_sequences.yaml` (windows of 8 frames every 8 frames, 512 pixel frames, batch size 16, a GRU), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the error images and the MLflow tracking turned on. After 12 iterations the network gives one class to every frame (9 classes: a loss near ln 9 = 2.197), so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

## Table of Contents
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
│   Entry point of every task. Point the path to a sequence_labelling configuration to run this workflow.
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
│               ├── check_sequence_settings()             window settings, batch padding, multi_class head
│               └── check_temporal_model_configuration()  an ordered module only (GRU, LSTM, transformer, temporal
│                                                         convolution): the poolings give one vector per window; causal
│                                                         allowed
│
├── get_trainer_class(task)  -> SequenceLabellingTrainer
│
├── SequenceLabellingTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   The SequenceClassificationTrainer with LABEL_TYPE = "frame" and CLASS_SUPPORT_COLUMN = "Frames".
│   │
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter() / setup_training_paths()
│       ├── build_dataloaders() -> get_sequence_dataloaders(label_type="frame")
│       │   └── SequenceDataset.__init__() -> compute_window_starts()
│       │           The windows of the sequence classification, each frame keeping the class of its category
│       │           ('frame_category_ids'), -100 for an unmapped category.
│       ├── get_test_sample_classes() + choose_sample_identifiers()
│       │       The classes of the labelled frames of every test window: a window counts for its most frequent class.
│       ├── build_model()
│       │   └── SequenceLabeller.__init__()
│       │       ├── build_backbone() + build_global_pooling()
│       │       ├── build_temporal_module(keep_time_axis=True)   one vector per frame, in its temporal context
│       │       ├── ClassificationHead()                         applied to every frame vector
│       │       └── class_weights buffer                         ones, or balanced on the labelled frames
│       ├── initialise_pretrained_components() / build_optimizer() / MetricTracker(counts_frames=True)
│       ├── print_experiment_information() -> describe_split() -> count_split_sequences()
│       ├── view_input_batches() -> show_input_samples()   every frame captioned with its class
│       └── restore_last_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── for each training_iteration:
        │   ├── save_model() + run_test_evaluation() when due                    -> [STAGE 4]
        │   ├── get_next_batch() -> SequenceDataset.__getitem__() + collate_longest_windows()
        │   │       labels [B, T]: the class of every frame, -100 on the padding frames
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
        │   ├── training_loss.backward() / step_optimizer() / validation pass
        │   └── average_trackers() + log_interval_metrics() + print_interval_summary()
        └── final save_model() + run_test_evaluation(), finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
├── create_evaluation_accumulator()
│       The probabilities and labels of the labelled frames, the window and position of every labelled frame, and
│       whether every labelled frame of each window is right.
├── for each test batch:
│   ├── prepare_batch() / predict_batch()
│   ├── accumulate_evaluation_batch() -> compute_sequence_exact_matches()
│   ├── dump_errors() -> build_frame_prediction_captions() + build_frame_grid()   one image per wrong frame
│   └── save_sample_images() -> build_sample_image()   frame grid with true/predicted classes of every frame
├── compute_evaluation_results()
│   ├── the frame metrics of the classification     confusion matrix, accuracy, balanced accuracy, macro F1, top-k,
│   │                                               thresholds, class_metrics.csv
│   ├── compute_extra_headline_metrics()
│   │   ├── Sequence Exact Match                    share of the windows whose labelled frames are all right
│   │   └── compute_segment_results()
│   │       ├── merge_window_predictions()          whole test sequences rebuilt from their windows, the
│   │       │                                       probabilities of a frame held by two windows averaged
│   │       └── compute_segment_metrics()           extract_segments(), compute_edit_distance(),
│   │                                               count_segment_matches(): Edit Score, Segment F1@10/25/50
│   ├── describe_extra_notes()                      the number of true and predicted segments
│   └── write_predictions_file() -> build_prediction_rows()   one row per labelled frame
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

**Example: the sections and the temporal module**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, ClassificationHead
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'top_k': 2, 'target_precision': 0.99,
                           'save_predictions': True, 'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
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
dict: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
9 entries, 9 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_sequence_labelling_configuration()`: the checks of the sequence classification, except that the temporal module must model the frame order and keep one vector per frame (the poolings summarise a whole window into one vector, so they are refused), and that it may be causal: a causal module only looks at the past frames, so the class of a frame can be given as soon as the frame arrives.

**Example: output with `TemporalModel.type: "mean_pooling"` and `short_sequences: "drop"` in the YAML**
```text
-----------------------------------------------------------------------------------------------
Invalid Configuration File : file:///.../configuration_wrong.yaml
-----------------------------------------------------------------------------------------------
  - 'Data.sequence_settings.short_sequences' must be one of ['pad', 'skip'], got 'drop'.
  - 'TemporalModel.type' must be one of ['gru', 'lstm', 'transformer', 'temporal_convolution'] for this task, got 'mean_pooling'.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="sequence_labelling") -> <class 'vision_training.trainers.sequence_labelling_trainer.SequenceLabellingTrainer'>
```


---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on the redirected `configuration_synthetic_sequences.yaml` (512 pixel frames, windows of 8 frames, batch size 16), with the run folder redirected to a temporary folder. The whole construction took 0.3 s on CPU.

### Step 2.1 — `SequenceLabellingTrainer.__init__()`
The labelling trainer has no constructor of its own: it is the sequence classification trainer with `LABEL_TYPE = "frame"` (the loaders read one class per frame) and `CLASS_SUPPORT_COLUMN = "Frames"` (the per-class table counts frames). It overrides the batch metric, the evaluation accumulator, the error images, the sample images, the predictions file and the extra headline metrics, since every frame has its own class.

**Example: the attributes after the construction**
```text
self.sequence_settings          = {'sequence_length': 8, 'window_stride': 8, 'short_sequences': 'pad', 'batch_padding': 'longest'}
self.LABEL_TYPE                 = 'frame'
self.CLASS_SUPPORT_COLUMN       = 'Frames'
self.top_k                      = 2
self.save_predictions           = True
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
SAMPLE_NAME = 'Sequences'   COUNTS_FRAMES = True   TASK_NAME = 'sequence_labelling'
```

### Step 2.2 — `BaseTrainer.__init__()`
**Example: some attributes set by the constructor**
```text
self.device = device(type='cpu')   self.per_class_report = True   self.number_sample_images = 8   self.input_image_size = 512
self.batch_size = 16   self.mixed_precision = False   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The sections of the sequence classification (`sequence_classification_workflow.md`, Step 2.4): `T8`, then `GRU1x128` (one layer of 128, not bidirectional), and `CAU` would follow a causal transformer or temporal convolution.

**Example output**
```text
M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128_HAVG.64_MC9
```

### Step 2.5 — `DataAugmenter()`
**Example: the settings of the configuration**
```text
activated True   probability 0.4   independent_frames False   shuffle_frames {'activated': False, 'probability': 0.5}
8 of 16 types activated: random_crop, random_left_right_flip, random_up_down_flip, random_rotation, random_90_rotation,
                         brightness_contrast_saturation, random_noise, random_colour_augmentation
```

### Step 2.6 — `setup_training_paths()`
**Example: the run folder after this step**
```text
M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128_HAVG.64_MC9/
├── Tensorboard/
├── Weights/
└── training_configuration.yaml    4,675 bytes
```

### Step 2.7 — `build_dataloaders()` → `get_sequence_dataloaders(label_type="frame")`
**Example output**
```text
class_names = ['square', 'circle', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'plus_sign', 'star', 'octagon']
train        676 windows -> 43 batches of 16 (one epoch = 43 iterations)
validation    86 windows ->  6 batches of 16
test          80 windows ->  5 batches of 16
collate      functools.partial(<function collate_longest_windows>, members_per_sample=1)
```

### Step 2.8 — `SequenceDataset.__init__()` → `compute_window_starts()`
The windows are cut as for the sequence classification (`sequence_classification_workflow.md`, Step 2.9), but every sequence gives `frame_category_ids`, one category per frame, and every window keeps the class of each of its frames, -100 for a frame of an unmapped category (ignored by the loss and the metrics). A sequence longer than a window gives overlapping windows, so some frames are in two windows.

**Example: a manifest sequence, and the two windows of a sequence of 10 frames**
```text
labels.json:  {'id': 1, 'name': 'train_000000', 'frames': ['train_000000/000000.png', ... 7 frames], 'frame_category_ids': [6, 6, 6, 6, 6, 3, 3]}
              (500 sequences of 2 to 12 frames, 3446 frames; 9 categories {'id': 1, 'name': 'square'}, ...)
train_000228 (10 frames, frame_category_ids [5, 8, 8, 8, 8, 8, 8, 8, 2, 9]):
  {'identifier': 303, 'sequence_identifier': 229, 'sequence_name': 'train_000228', 'window_start': 0,
   'frames': [000000.png ... 000007.png], 'frame_labels': [4, 7, 7, 7, 7, 7, 7, 7]}
  {'identifier': 304, 'sequence_identifier': 229, 'sequence_name': 'train_000228', 'window_start': 2,
   'frames': [000002.png ... 000009.png], 'frame_labels': [7, 7, 7, 7, 7, 7, 1, 8]}      <- frames 2 to 7 in both windows
```

**Example: the first window after shuffling, and the sample the loader gets from it** (a sequence of 2 frames)
```text
window:     {'identifier': 118, 'sequence_identifier': 89, 'sequence_name': 'train_000088', 'window_start': 0,
             'frames': ['train_000088/000000.png', 'train_000088/000001.png'], 'frame_labels': [7, 3]}
dataset[0]: images (8, 3, 512, 512) torch.uint8
            sequence_mask (8,) torch.bool [True, True, False, False, False, False, False, False]
            labels (8,) torch.int64 [7, 3, -100, -100, -100, -100, -100, -100]      <- one class per frame, -100 on the padding
            sample_identifiers 118
get_class_counts() = [523, 511, 498, 410, 559, 399, 488, 609, 414]                  <- labelled frames of every class
```
The class counts add up to 4,411 frames for 3,446 frames in the split: the frames held by two windows are counted twice.

### Step 2.9 — `build_native_loader()`
**Example output**
```text
len(loader) = 43   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.10 — `get_test_sample_classes()` + `choose_sample_identifiers()`
Every test window gives the classes of its labelled frames, and `choose_sample_identifiers()` groups it under its most frequent class (the lowest class index on a tie). With 9 classes and 8 sample windows, the first 8 classes get one window each, the first of their group.

**Example output**
```text
get_test_sample_classes() -> 80 entries, e.g. {63: [6, 6, 1, 1, 4, 4, 4, 4], 8: [6, 6, 0, 0, 0, 0, 3, 3], 43: [2, 2, 0, 0, 0, 0], 5: [7, 7, 7, 7, 7, 5, 5, 7]}
sample_image_identifiers  -> [0, 1, 2, 4, 5, 6, 10, 13]
names                     -> test_000000_000000, test_000001_000000, test_000002_000000, test_000003_000000,
                             test_000004_000000, test_000005_000000, test_000007_000000, test_000010_000000
```

### Step 2.11 — `build_model()` → `SequenceLabeller.__init__()`
The labeller has the parts of the sequence classifier (`sequence_classification_workflow.md`, Steps 2.13 to 2.16), but its temporal module is built with `keep_time_axis=True`: it returns one vector per frame, the frame seen in its temporal context, and the head classifies every one of them. With `class_weighting: "balanced"`, the class weights come from the labelled frames of every class.

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          RecurrentTemporalModule     74,496 parameters      GRU(64, 128, batch_first=True), output_features 128
model.classification_head      ClassificationHead           8,841 parameters      Linear(128, 64) -> ReLU -> Linear(64, 9)
model.class_weights            buffer [9] torch.float32 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```
The GRU reads the frames in order only (`bidirectional: False`): the vector of a frame then depends on the frames before it. A bidirectional GRU would give 256 features per frame, seeing the frames after it too.

### Step 2.12 — `initialise_pretrained_components()` + `build_optimizer()` + `get_tracked_metrics()`
**Example output**
```text
pretrained_checkpoint_paths = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
{'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'accuracy': 'Accuracy'}
MetricTracker: batch_size 16, validation_iteration_ratio 25, sample_name 'Sequences', counts_frames True
```

### Step 2.13 — `print_experiment_information()` (→ `describe_split()` → `count_split_sequences()`)
**Example output**
```text
- Run Name                : M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128_HAVG.64_MC9
...
- Train Data Folder       : file://<datasets>/synthetic_changing_shapes/train (500 sequences, 3446 frames, 676 windows of at most 8 frames)
- Validation Data Folder  : file://<datasets>/synthetic_changing_shapes/validation (60 sequences, 449 frames, 86 windows of at most 8 frames)
- Test Data Folder        : file://<datasets>/synthetic_changing_shapes/test (60 sequences, 434 frames, 80 windows of at most 8 frames)
- Iterations Per Epoch    : 43
- Batch Size              : 16
...
- MLflow Tracking         : Off
```

### Step 2.14 — `print_summary()` / `view_input_batches()` / `restore_last_model()`
The summary panels are those of the sequence classification, the temporal module panel giving `Output Features : 128`. With `view_input_images`, every frame of the grids is captioned with its class (`get_frame_captions()`, `none` for an unmapped category), and the window with the order of its classes (`get_window_caption()`, e.g. `Classes : hexagon > triangle`).

**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of the redirected `configuration_synthetic_sequences.yaml` shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the error images and the MLflow tracking on. Its run name is `M2M_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128_HAVG.64_MC9`; with batches of 4 windows, an epoch is 169 iterations.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
**Example: what happens at each iteration of the run**

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | | ✔ (iteration 12) |
| Test evaluation | | | | | | ✔ | | | | | | | ✔ (iteration 12) |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()` / `start_experiment_tracking()`
**Example output**
```text
Tensorboard/Train/        Total Loss, Classification Loss, Accuracy, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Accuracy                   (iteration 10)
MLflow Run Created In The Experiment 'sequence_labelling' : 39f5fb6183164c37a921cee2a6ae2f86 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.3 — `save_model()` → `save_checkpoint()`
**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   3,296,551 bytes      Weights/Iteration_6/model_000006.pt   3,296,551 bytes
Weights/Iteration_9/model_000009.pt   3,296,615 bytes      Weights/Iteration_12/model_000012.pt  3,296,615 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'mlflow_run_id',
 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
model_state: 90 tensors, the first 'class_weights', then 'backbone.convolution_blocks.0. ...'
```

### Step 3.4 — `run_test_evaluation()`
At iterations 6 and 12 (the final checkpoint), see Stage 4.

### Step 3.5 — `get_next_batch()` → `SequenceDataset.__getitem__()` + `collate_longest_windows()`
The window of the sequence classification (`sequence_classification_workflow.md`, Step 3.7), its label being the `[T]` class of every frame, -100 on the padding frames. `collate_longest_windows()` cuts the labels with the frames, since they have a frame axis.

**Example: the first batch of the run** (the longest window of the batch has 8 frames, so nothing is cut)
```text
images               [4, 8, 3, 64, 64] torch.uint8
sequence_mask        [4, 8] torch.bool
                     [[True x 8], [True, True, True, False, False, False, False, False], [True x 8], [True, True, True, True, False, False, False, False]]
labels               [4, 8] torch.int64
                     [[3, 3, 3, 3, 5, 8, 8, 8], [2, 1, 1, -100, -100, -100, -100, -100], [1, 1, 1, 6, 6, 6, 6, 6], [8, 7, 5, 6, -100, -100, -100, -100]]
sample_identifiers   [4] torch.int64     [273, 499, 386, 34]
```

### Step 3.6 — `MetricTracker.count_frames()`
**Example: the frames of the first two intervals**
```text
interval_frames = 89 (iterations 1 to 4), 112 (iterations 5 to 8)
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
**Example: before and after**
```text
images [4, 8, 3, 64, 64] torch.uint8 -> torch.float32     labels [4, 8] torch.int64     sequence_mask [4, 8] torch.bool
```

### Step 3.8 — `augment_batch()`
The augmentation of the sequence classification (`sequence_classification_workflow.md`, Step 3.12). When `shuffle_frames` permutes the real frames of a window, the frame labels are gathered in the same order (`torch.gather()` with the frame order of the window), so that every frame keeps its class.

**Example: which windows of the first batch were changed** (`probability: 0.4`)
```text
images: changed per sample [False, False, True, True]      labels: changed per sample [False, False, False, False]
```

### Step 3.9 — `forward_pass()` → `SequenceLabeller.forward(sequence_mask=..., labels=...)`
`encode_frames()` gives the `[B, T, F]` frame vectors (`sequence_classification_workflow.md`, Step 3.14). The temporal module keeps the time axis: the GRU reads the packed real frames of every window and returns its output at every frame, padded back to `[B, T, F']` with zeros on the padding frames. The head is applied to every frame vector.

**Example: the first batch through the trained model**
```text
encode_frames   -> (4, 8, 64)    zero vectors per window [0, 5, 0, 4]       <- the padding frames
temporal_module -> (4, 8, 128)   one vector per frame
classification_head -> (4, 8, 9)
```

**Example: the dictionary returned in training mode**
```text
class_logits          [4, 8, 9] torch.float32
class_probabilities   [4, 8, 9] torch.float32
classification_loss   2.1962 (scalar)            <- close to ln 9 = 2.197
```

### Step 3.10 — `compute_sequence_labelling_loss()`
The cross entropy of the softmax of every frame, averaged over the labelled frames only: the padding frames and the frames of unmapped categories (-100) are ignored (`ignore_index`). It uses `class_weights` and `label_smoothing` like the loss of the classification.

### Step 3.11 — `compute_batch_metrics()` → `compute_frame_accuracy()`
The share of the labelled frames whose predicted class (the argmax of their logits) is right.

**Example: the first batch**
```text
accuracy = 0.0870          (2 of the 8 + 3 + 8 + 4 = 23 labelled frames right)
```

### Step 3.12 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
**Example: the Train curves**
```text
Total Loss   1: 2.1962, 2: 2.1879, 3: 2.2749, 4: 2.2360, 5: 2.2211, 6: 2.1614, 7: 2.1669, 8: 2.1590, 9: 2.1973, 10: 2.2007, 11: 2.1900, 12: 2.2046
Accuracy     1: 0.0870, 2: 0.0714, 3: 0.0000, 4: 0.0769, 5: 0.1071, 6: 0.0345, 7: 0.2609, 8: 0.2500, 9: 0.1304, 10: 0.0312, 11: 0.2500, 12: 0.1250
```

### Step 3.13 — `training_loss.backward()` + `step_optimizer()` + validation pass
**Example output**
```text
gradient norm: iteration 1: 0.5323   iteration 2: 0.5271   iteration 3: 0.7529        learning rate 0.001 (constant)
Tensorboard/Validation at iteration 10: Total Loss 2.1684, Classification Loss 2.1684, Accuracy 0.2333
```

### Step 3.14 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The batch accuracies are averaged over the batches of the interval, each batch having its own number of labelled frames.

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 8.895, 'classification_loss': 8.895, 'accuracy': 0.2353, 'interval_frames': 89}
after:  {'interval_batches': 4, 'total_loss': 2.2237, 'classification_loss': 2.2237, 'accuracy': 0.0588, 'interval_frames': 89}
MLflow, step 4: {'train/Total Loss': 2.2237, 'train/Classification Loss': 2.2237, 'train/Accuracy': 0.0588,
                 'train/throughput': 5.7679, 'train/interval_seconds': 2.774, 'learning_rate': 0.001,
                 'train/frame_throughput': 32.0837}
```

**Example output: iterations 4 and 12**
```text
===========================================================================================================================================================
|     Iteration: 4    |     Total Sequences: 16     |    Interval Sequences: 16    |   Throughput: 5.8 Sequences Per Second  |   32.1 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   2.2237
Moving Average of Train Classification Loss            :   2.2237
Moving Average of Train Accuracy                       :   0.0588
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:02.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
...
===========================================================================================================================================================
|    Iteration: 12    |     Total Sequences: 48     |    Interval Sequences: 16    |  Throughput: 11.7 Sequences Per Second  |   81.0 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   2.1982  +0.0210
Moving Average of Valid Total Loss                     :   2.1684
Moving Average of Train Classification Loss            :   2.1982  +0.0210
Moving Average of Valid Classification Loss            :   2.1684
Moving Average of Train Accuracy                       :   0.1342  -0.0290
Moving Average of Valid Accuracy                       :   0.2333
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:01.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
```

### Step 3.15 — end of the loop
**Example output**
```text
Model successfully saved at iteration 12. Exiting Training.
Iteration 12 · 80 Test Sequences, 541 Labelled Frames · 2026-10-10 07:15 ───────────────      <- the evaluation of the final checkpoint
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (60 sequences cut into 80 windows), with `per_class_report: True`, `number_sample_images: 8`, `top_k: 2`, `save_predictions: True`, `error_dumps` activated with `maximum_per_class_pair: 2`, and the MLflow tracking on. The checkpoint of iteration 6 gives `circle` to every frame, the one of iteration 12 gives `rectangle` to every frame.

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1).

**Example output: the progress bar** (80 windows, batch size 4)
```text
Test Evaluation Iteration 12: 100%|██████████| 20/20 [00:01<00:00, 10.05it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The labelling hook keeps only the labelled frames: their probabilities and labels, the window and the position in it of every one of them (to name the frame in the predictions file and to place it in its sequence for the segment metrics), and, for every window, whether all its labelled frames are right.

**Example: the accumulator at the end of the loop of iteration 6**
```text
'probabilities':            20 tensors, the first [28, 9] torch.float64     <- the 28 labelled frames of the first batch
'labels':                   20 tensors, the first [28] torch.int64
'frame_window_identifiers': 20 tensors, the first [28] torch.int64
'frame_positions':          20 tensors, the first [28] torch.int64
'exact_matches':            20 tensors, the first [4] torch.bool
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
**Example: the predictions of a test batch**
```text
class_logits          [4, 8, 9] torch.float32
class_probabilities   [4, 8, 9] torch.float32
```

### Step 4.4 — `accumulate_evaluation_batch()` → `compute_sequence_exact_matches()`
It keeps the probabilities and the labels of the labelled frames (`labels != -100`), with their window and position, and `compute_sequence_exact_matches()` tells for every window whether all its labelled frames are right. Then `dump_errors()` runs while the frames are still in the batch.

### Step 4.5 — `dump_errors()` → `build_frame_prediction_captions()` + `build_frame_grid()`
Every wrong frame gives one image: the grid of its window, every real frame captioned `true/predicted` (with `X` before a wrong frame), under a caption bar naming the wrong frame, its classes and the probability of its predicted class. It is saved as `Errors/<TRUE>-TO-<PREDICTED>/<window>_frame<index>_<score>.png`, at most `maximum_per_class_pair` per pair, the window being named after its sequence and its first frame.

**Example: the error images of the checkpoint of iteration 12** (every frame predicted as `rectangle`)
```text
Evaluation/Iteration_12/Errors/
├── circle-TO-rectangle/      test_000025_000000_frame7_0.122.png   test_000032_000000_frame1_0.122.png
├── hexagon-TO-rectangle/     test_000005_000003_frame1_0.122.png   test_000005_000003_frame2_0.122.png
│                             <- frames 1 and 2 of the window of test_000005 starting at its frame 3
├── octagon-TO-rectangle/     test_000016_000000_frame7_0.122.png   test_000016_000003_frame4_0.122.png
...                           8 pairs x 2 images
report line: "Error Images : 16 in 8 class pairs, at most 2 per pair"
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
The labelling hook draws the frame grid of the window, every real frame captioned `true/predicted` with `X` before a wrong frame, under a caption bar giving the number of labelled frames that are right, in green when all are and in red otherwise.

**Example: the sample images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Samples/
test_000000_000000.png  test_000001_000000.png  test_000002_000000.png  test_000003_000000.png
test_000004_000000.png  test_000005_000000.png  test_000007_000000.png  test_000010_000000.png
build_sample_image -> [96, 276, 3] torch.uint8      one row of up to 4 frames of 64 pixels under the caption bar
caption: "<right frames>/<labelled frames> labelled frames right (true/predicted)"
```

### Step 4.7 — `compute_evaluation_results()`
The frame metrics of the classification are computed on the labelled frames (Steps 4.8 and 4.9), then `compute_extra_headline_metrics()` adds the window and sequence metrics (Steps 4.10 and 4.11), and `describe_extra_notes()` the number of segments. `describe_evaluated_samples()` gives the windows and their labelled frames: the 434 frames of the test split are held 541 times by the 80 windows, since the overlapping windows hold some frames twice. The headline metric is the frame accuracy.

**Example: the results of iteration 12**
```text
evaluated_samples = '80 Test Sequences, 541 Labelled Frames'
metrics           = {'Accuracy': 0.1183, 'Balanced Accuracy': 0.1111, 'Macro F1': 0.0235, 'Top-2 Accuracy': 0.1941,
                     'Sequence Exact Match': 0.0, 'Edit Score': 0.1111, 'Segment F1@10': 0.1688, 'Segment F1@25': 0.1266,
                     'Segment F1@50': 0.0759}
notes             = ['Segments : 177 true and 60 predicted in 60 whole test sequences, their windows being merged and an
                      unlabelled frame ending a segment', 'Error Images : 16 in 8 class pairs, at most 2 per pair',
                     'Sample Images : 8 fixed test sequences']
headline_metric   = 'Accuracy'
class_scalars     = 27: Precision/square, Recall/square, F1/square, Precision/circle, ...
tables            = Per Class Metrics: Class, Frames, Precision, Recall, F1, Threshold @ P0.99, Recall @ P0.99
```

### Step 4.8 — the frame metrics: `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()` + `compute_top_k_accuracy()`
The metrics of the classification (see `classification_workflow.md`, Steps 4.9 to 4.11), every labelled frame of every window being one sample.

**Example: `Evaluation/Iteration_12/confusion_matrix.csv`**
```text
true \ predicted,square,circle,triangle,rectangle,pentagon,hexagon,plus_sign,star,octagon
square,0,0,0,68,0,0,0,0,0
circle,0,0,0,41,0,0,0,0,0
triangle,0,0,0,64,0,0,0,0,0
rectangle,0,0,0,64,0,0,0,0,0
...
octagon,0,0,0,48,0,0,0,0,0
```

**Example: how the numbers of iteration 12 are obtained**
```text
accuracy          = 64 / 541                                   = 0.1183   (only the rectangle frames are right)
rectangle         precision = 64 / 541 = 0.1183   recall = 1.0   F1 = 2 x 0.1183 x 1.0 / (0.1183 + 1.0) = 0.2116
balanced accuracy = (1.0 + 8 x 0.0) / 9                        = 0.1111
macro F1          = (0.2116 + 8 x 0.0) / 9                     = 0.0235
```

### Step 4.9 — `find_class_thresholds()` + `write_class_metrics_file()` + `save_confusion_matrix_image()`
**Example: `Evaluation/Iteration_12/class_metrics.csv`** (the `samples` column counts labelled frames)
```text
class_index,class,samples,precision,recall,f1,threshold_at_target_precision,recall_at_target_precision
0,square,68,0.000000,0.000000,0.000000,,
1,circle,41,0.000000,0.000000,0.000000,,
2,triangle,64,0.000000,0.000000,0.000000,,
3,rectangle,64,0.118299,1.000000,0.211570,,
...
```
No class reaches a precision of 0.99 here, so every class of `class_thresholds.json` is `null`.

### Step 4.10 — Sequence Exact Match
The share of the test windows whose labelled frames are all right, from the `exact_matches` of Step 4.4: the strictest metric of the task.

**Example output**
```text
Sequence Exact Match = 0.0      (no test window has all its labelled frames of the one class predicted everywhere)
```

### Step 4.11 — `compute_segment_results()` → `merge_window_predictions()` + `compute_segment_metrics()`
The frame metrics do not see whether the prediction of a sequence is fragmented: a sequence of 10 frames of one class predicted with one wrong frame in the middle keeps a frame accuracy of 0.9, but it gives three segments for one event. The segment metrics score the events of the whole test sequences:
- `merge_window_predictions()` places every labelled frame in its sequence (the start of its window plus its position), merges the windows of every sequence, and averages the probabilities of a frame held by two windows before taking its class;
- `extract_segments()` cuts every sequence into segments of consecutive frames of one class, a gap of unlabelled frames ending a segment, as true segments and predicted segments;
- **Edit Score**: for every sequence, 1 minus the edit distance between the orders of the predicted and true segment classes (`compute_edit_distance()`: the segments to insert, remove or replace) over the larger number of segments, averaged over the sequences;
- **Segment F1@10/25/50**: `count_segment_matches()` takes every predicted segment in order and finds the true segment of its class it overlaps most; it is right when their intersection over union reaches the overlap (10, 25 or 50 %) and no earlier predicted segment matched that true segment. The F1 is computed from the counts summed over all the sequences.

**Example: the merged sequences of the evaluation of iteration 12**
```text
merge_window_predictions: 541 labelled frames of the windows -> 60 sequences, 434 distinct frames
frames [0, 1]  true [3, 6]  predicted [3, 3]
   true segments [(3, 0, 0), (6, 1, 1)]   predicted segments [(3, 0, 1)]   edit distance 1   edit score 0.5000
frames [0, 1]  true [4, 8]  predicted [3, 3]
   true segments [(4, 0, 0), (8, 1, 1)]   predicted segments [(3, 0, 1)]   edit distance 2   edit score 0.0000
frames [0, ..., 10]  true [2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 5]  predicted [3 x 11]
   true segments [(2, 0, 2), (5, 3, 3), (2, 4, 9), (5, 10, 10)]   predicted segments [(3, 0, 10)]   edit distance 4   edit score 0.0000
Segments : 177 true and 60 predicted in 60 whole test sequences      <- one predicted segment per sequence
Edit Score 0.1111   Segment F1@10 0.1688   Segment F1@25 0.1266   Segment F1@50 0.0759
```
A segment is `(class, first frame, last frame)`. In the first sequence the predicted segment has the class `3` of the first true segment, so only the true segment `6` is missing: an edit distance of 1 over 2 segments, an edit score of 0.5.

### Step 4.12 — `write_predictions_file()` → `build_prediction_rows()`   (only with `save_predictions`)
One row per labelled frame of every window, named by its sequence and its frame file: a frame held by two windows has two rows.

**Example: `Evaluation/Iteration_12/predictions.csv`** (541 rows, 74,373 bytes; first rows, columns shortened)
```text
sequence,frame,true_class,predicted_class,correct,probability_square,probability_circle, ... ,probability_octagon
test_000059,test_000059/000000.png,triangle,rectangle,0,0.113589,0.120350, ... ,0.117674
test_000059,test_000059/000001.png,triangle,rectangle,0,0.114056,0.120155, ... ,0.117584
test_000059,test_000059/000002.png,pentagon,rectangle,0,0.114399,0.119891, ... ,0.117497
```

### Step 4.13 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 80 Test Sequences, 541 Labelled Frames · 2026-10-10 07:15 ───────────────────────────────────────────────
  Accuracy               0.1183   Balanced Accuracy   0.1111
  Macro F1               0.0235   Top-2 Accuracy      0.1941
  Sequence Exact Match   0.0000   Edit Score          0.1111
  Segment F1@10          0.1688   Segment F1@25       0.1266
  Segment F1@50          0.0759
  Segments : 177 true and 60 predicted in 60 whole test sequences, their windows being merged and an unlabelled frame
  ending a segment
  Error Images : 16 in 8 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                      Threshold    Recall
   Class       Frames   Precision   Recall       F1     @ P0.99   @ P0.99
  ────────────────────────────────────────────────────────────────────────
   square          68      0.0000   0.0000   0.0000           —         —
   circle          41      0.0000   0.0000   0.0000           —         —
   ...
   octagon         48      0.0000   0.0000   0.0000           —         —
   rectangle       64      0.1183   1.0000   0.2116           —         —
  Worst F1 first, ties in class order. — : no frames of the class, or the precision 0.99 is never reached.
  - Results : file:///.../Results.txt
  - ...
```

**Example: TensorBoard and MLflow after the two evaluations**
```text
Tensorboard/Test: 36 scalars (9 global + 27 per class), 9 images (8 samples + Confusion Matrix)
    scalar Accuracy        6: 0.0758, 12: 0.1183
    scalar Edit Score      6: 0.0861, 12: 0.1111
    scalar Segment F1@50   6: 0.0253, 12: 0.0759
MLflow: test/Accuracy, ..., test/Edit Score, test/Segment F1_at_10, test/Segment F1_at_25, test/Segment F1_at_50,
        test/Sequence Exact Match, test/best_Accuracy (steps 6, 12), 27 test_class/<metric>/<class>
```
MLflow only accepts letters, digits, `_`, `-`, `.`, `/` and spaces in a metric name, so `sanitise_metric_name()` writes `Segment F1@10` as `Segment F1_at_10` (the original names are kept in the `metric_names.json` artifact); TensorBoard and the console keep the `@`.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 20 sequences of the test split (its `labels.json` reduced, its `data/` folder linked).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5). The windows of the test split are cut with the `sequence_settings` of the run, and the segment metrics merge them back into the sequences of that split.

### Step 5.2 — trainer construction
**Example: the split line of the launch information**
```text
- Test Data Folder        : file:///.../test_reduced (20 sequences, 143 frames, 28 windows of at most 8 frames)
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'sequence_labelling' : f6ccac8284bf428aa5a9bea81d43ce8a (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output** (iterations 12 and 9 tie on the accuracy, so the first one evaluated, 12, is kept)
```text
Evaluation Summary Of 4 Checkpoints : file:///.../runs/evaluations/M2M_IN-KAR-64_..._MC9/Evaluation/evaluation_summary.csv
  - Best Checkpoint : iteration 12 (Accuracy 0.1514)

iteration,Accuracy,Balanced Accuracy,Macro F1,Top-2 Accuracy,Sequence Exact Match,Edit Score,Segment F1@10,Segment F1@25,Segment F1@50, ...
12,0.15135135135135136,0.1111111111111111,0.02921231090245175,0.17297297297297298,0.0,0.10833333333333335,0.16470588235294117,0.1411764705882353,0.09411764705882354, ...
9,0.15135135135135136,0.1111111111111111,0.02921231090245175,0.17297297297297298,0.0,0.10833333333333335,0.16470588235294117,0.1411764705882353,0.09411764705882354, ...
6,0.021621621621621623,0.1111111111111111,0.004703115814226926,0.17297297297297298,0.0,0.04583333333333334,0.07058823529411765,0.04705882352941177,0.023529411764705885, ...
3,0.021621621621621623,0.1111111111111111,0.004703115814226926,0.17297297297297298,0.0,0.04583333333333334,0.07058823529411765,0.04705882352941177,0.023529411764705885, ...
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 12 on the 20 sequences** (shortened)
```text
Evaluating : file:///.../runs/M2M_IN-KAR-64_..._MC9/Weights/Iteration_12/model_000012.pt
Iteration 12 · 28 Test Sequences, 185 Labelled Frames · 2026-10-10 07:15 ───────────────────────────────────────────────
  Accuracy               0.1514   Balanced Accuracy   0.1111
  Macro F1               0.0292   Top-2 Accuracy      0.1730
  Sequence Exact Match   0.0000   Edit Score          0.1083
  Segment F1@10          0.1647   Segment F1@25       0.1412
  Segment F1@50          0.0941
  Segments : 65 true and 20 predicted in 20 whole test sequences, their windows being merged and an unlabelled frame
  ending a segment
  Error Images : 16 in 8 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences
  ...
   hexagon         41      0.0000   0.0000   0.0000      0.0985    0.0488
   ...
   rectangle       28      0.1514   1.0000   0.2629           —         —
```
The header counts the 28 windows of the 20 sequences ("28 Test Sequences"), and the segment note the 20 whole sequences.
