# Sequence Classification Workflow

This document follows one sequence classification run (many to one, M2O: one class per window of frames) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`, and the evaluation of the sequence classification is the one of the classification: here they keep a short explanation and the real output of the sequence run, and the steps specific to the sequences (windows, padding, frame encoder, temporal module, frame grids) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training:
- **Stages 1 and 2**: `configurations/sequence_classification/configuration_synthetic_sequences.yaml` (synthetic moving shapes, 4 classes `moving_left`, `moving_right`, `moving_up`, `moving_down`, sequences of 8 frames, 512 pixel frames, batch size 16), with the run folder redirected to a temporary folder. The dataset of `configuration_example.yaml` is not on this machine.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the error images and the MLflow tracking turned on. After 12 iterations the network predicts `moving_left` for every window (4 classes: 1/4 = 25% accuracy, a loss near ln 4 = 1.386), so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

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
│   Entry point of every task: loads the YAML file whose path it hard-codes, looks up the trainer of its task, builds
│   it and starts the training loop. Point the path to a sequence_classification configuration to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, TemporalModel, ClassificationHead), converts the types, adds label_mode and label_dictionary to
│   │   'Data' and runs the checks.
│   │
│   ├── load_configuration(configuration_path)
│   │       Plain YAML reader, no conversion.
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category id: class index}; one category per sequence, no background class.
│   └── check_configuration()
│       ├── common checks                         experiment, data, backbone, optimization (see classification)
│       └── check_sequence_classification_configuration()
│               ├── the data, head and evaluation checks of the classification
│               ├── check_sequence_settings()             sequence_length, window_stride, short_sequences ("pad" or
│               │                                         "skip"), batch_padding ("sequence_length" or "longest"),
│               │                                         a multi_class head, no "flatten" pooling
│               └── check_temporal_model_configuration()  any pooling or ordered type, sizes, dropout, not causal,
│                                                         not bidirectional when causal, pretrained settings
│
├── get_trainer_class(task)  -> SequenceClassificationTrainer
│
├── SequenceClassificationTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   Keeps Data.sequence_settings, warns when the frames are shuffled for an ordered temporal module, then runs the
│   │   ClassificationTrainer constructor (top_k, target_precision, save_predictions, error_dumps, output_mode).
│   │
│   └── BaseTrainer.__init__()
│       │   Same order as for every task: device, run name, settings, augmenter, paths, loaders, sample windows, model,
│       │   pretrained components, optimizer, metric tracker, printouts, restoration.
│       │
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter() / setup_training_paths()
│       │       The run name adds the window (T<frames>) and the temporal module (e.g. GRU1x128-BI) before the head
│       │       and the classes.
│       ├── build_dataloaders()
│       │   │   Builds the three loaders and keeps class_names, taken from the categories of the training manifest.
│       │   │
│       │   └── get_sequence_dataloaders(label_type="sequence")
│       │       │   One SequenceDataset per split from <folder>/labels.json and the frames of <folder>/data/, and
│       │       │   with batch_padding "longest" the collate collate_longest_windows (members_per_sample=1).
│       │       │
│       │       ├── SequenceDataset.__init__() -> compute_window_starts()
│       │       │       Cuts every sequence into windows of sequence_length frames, one every window_stride frames plus
│       │       │       one ending on the last frame; a sequence shorter than a window gives one padded window ("pad")
│       │       │       or none ("skip"). The windows of unmapped categories are left out. Shuffled once.
│       │       └── build_native_loader()
│       ├── get_test_sample_classes() + choose_sample_identifiers()
│       │       The class of every test window, so that the sample windows are spread over the classes.
│       ├── build_model()
│       │   │   Creates the SequenceClassifier from the registry, then copies the class weights of the loss.
│       │   │
│       │   └── SequenceClassifier.__init__()
│       │       ├── build_backbone()            the shared backbone, run on every frame
│       │       ├── build_global_pooling()      one vector per frame, as wide as the last backbone block
│       │       ├── build_temporal_module()     a pooling (mean, max, attention) or an ordered module (GRU, LSTM,
│       │       │                               transformer, temporal convolution), summarising the frame vectors of a
│       │       │                               window into one vector (keep_time_axis=False)
│       │       ├── ClassificationHead()        fully connected block and classifier, as for the images
│       │       └── class_weights buffer        ones, or balanced on the windows of every class
│       ├── initialise_pretrained_components()  Backbone, TemporalModel and ClassificationHead can come from other runs
│       ├── build_optimizer()
│       ├── get_tracked_metrics() + MetricTracker.__init__(counts_frames=True)
│       ├── print_experiment_information() -> describe_split() -> count_split_sequences()
│       │       "N sequences, F frames, W windows of at most L frames"; an epoch is W windows.
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       │       The summary adds the Temporal Module panel; the size panel lists the temporal module on its own line.
│       ├── view_input_batches() -> show_input_samples()  (only with view_input_images, then exits)
│       └── restore_last_model() -> get_last_checkpoint_iteration() / load_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── setup_tensorboard_writers() / initialize_training_loop_state() / start_experiment_tracking()
        ├── for each training_iteration:
        │   ├── save_model() + run_test_evaluation() when due                    -> [STAGE 4]
        │   ├── get_next_batch()
        │   │   ├── SequenceDataset.__getitem__() -> read_window_frames()
        │   │   │       Reads and resizes the real frames of the window, then black padding frames up to
        │   │   │       sequence_length, with the sequence mask (True for the real frames) and the class of the window.
        │   │   └── collate_longest_windows()   (batch_padding "longest")
        │   │           Cuts every window of the batch to its longest window before stacking.
        │   ├── MetricTracker.count_frames()     real frames of the batch, for the frame throughput
        │   ├── set_training_mode()
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()             frames to float32, labels and sequence mask on the device
        │   │   ├── augment_batch()             the same random augmentation for every frame of a window (photometric
        │   │   │                               types frame by frame with independent_frames), then shuffle_frames()
        │   │   ├── forward_pass() -> SequenceClassifier.forward(sequence_mask=..., labels=...)
        │   │   │   ├── encode_frames()          backbone + pooling on the real frames only, zeros for the padding
        │   │   │   ├── temporal_module          [B, T, F] frame vectors -> [B, F'] window vectors
        │   │   │   ├── classification_head      class_logits [B, K]
        │   │   │   └── compute_classification_loss()
        │   │   ├── compute_batch_metrics()      accuracy of the batch
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()
        │   ├── validation pass every validation_iteration_ratio iterations
        │   └── average_trackers() + log_interval_metrics() + print_interval_summary()   (sequences and Frames Per Second)
        ├── final save_model() + run_test_evaluation()
        └── close the writers, finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
└── the evaluation of the classification (see classification_workflow.md, Stage 4), on windows:
    ├── create_evaluation_accumulator() / prepare_batch() / predict_batch() / accumulate_evaluation_batch()
    ├── dump_errors() -> build_input_image()           the real frames of the window laid out on a grid
    ├── save_sample_images() -> build_sample_image()   the frame grid captioned with the true and predicted classes
    ├── compute_evaluation_results()                   confusion matrix, accuracy, balanced accuracy, macro F1, top-k,
    │                                                  thresholds, class_metrics.csv, predictions.csv (sequence, first
    │                                                  and last frame of every window)
    └── report_evaluation() -> report_evaluation_results() + track_evaluation()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task: loads training_configuration.yaml, builds the SequenceClassificationTrainer in the output
    folder, then evaluate_checkpoints() -> start_evaluation_tracking() + summarise_checkpoint_evaluations()
    -> run_test_evaluation()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/sequence_classification/configuration_synthetic_sequences.yaml`.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1): it loads the YAML file whose path it hard-codes, looks up the trainer of `ExperimentConfiguration.task` and starts its training loop. Nothing in it is specific to the sequences.

### Step 1.2 — `load_experiment_configuration()`
It splits the YAML into the two dictionaries read by every later stage, converts the values YAML reads with the wrong type, adds `label_mode` and `label_dictionary` to `Data` (Step 1.4) and runs the checks (Step 1.5). The sequence classification has the `TemporalModel` section on top of the sections of the classification, its `Data` section has a `sequence_settings` subsection (the windows), and its evaluation settings are those of the classification.

**Example: the sections and the settings of the sequences**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, ClassificationHead
Data keys after loading:  DALI, loading, image_settings, sequence_settings, label_mapping, data_augmentation, label_mode, label_dictionary
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'top_k': 2, 'target_precision': 0.99,
                           'save_predictions': True, 'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
TemporalModel:            {'type': 'gru', 'hidden_size': 128, 'number_layers': 1, 'number_heads': 4, 'kernel_size': 3, 'dropout_rate': 0.0,
                           'bidirectional': True, 'causal': False, 'positional_embedding': True, 'pretrained': {...}}
ClassificationHead:       {'global_pooling': 'average', 'fully_connected': [64], 'dropout_rate': 0.2, 'output_mode': 'multi_class',
                           'label_smoothing': 0.0, 'class_weighting': 'none', 'pretrained': {...}}
```
Every setting of `TemporalModel` is required, even those its type does not read (`number_heads` and `positional_embedding` for the transformer, `kernel_size` for the temporal convolution), so that the type can be changed without adding keys.

### Step 1.3 — `load_configuration()`
Plain YAML reader, as for every task: no conversion and no check.

**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'TemporalModel', 'ClassificationHead']
number_iterations: '1e10'      <- still a string
learning_rate:     0.001
```

### Step 1.4 — `build_label_dictionary()`
The mapping file gives the class index of every category of the manifest, a category being the class of a whole sequence. Several categories can be merged into one class by giving them the same index, and the sequences of a category left out of the mapping are left out of the windows (Step 2.9).

**Example: the label dictionary of the moving shapes**
```text
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '/home/.../sequence_data/synthetic_moving_shapes/label_mapping.json'}
dict: {1: 0, 2: 1, 3: 2, 4: 3}
4 entries, 4 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_sequence_classification_configuration()`: the data, head and evaluation checks of the classification, `check_sequence_settings()` (the window settings, the batch padding, a `multi_class` head since the manifests give one category per sequence, and no `flatten` pooling since the temporal module reads vectors as wide as the last backbone block), and `check_temporal_model_configuration()` (the type, the sizes, the dropout, and a module that is not causal: a causal module, which only sees the past frames, is for the sequence labelling). Every problem is reported together, then the program exits.

**Example: output with `causal: True` and `batch_padding: "shortest"` in the YAML**
```text
-----------------------------------------------------------------------------------------------
Invalid Configuration File : file:///.../configuration_wrong.yaml
-----------------------------------------------------------------------------------------------
  - 'Data.sequence_settings.batch_padding' must be one of ['sequence_length', 'longest'], got 'shortest'.
  - 'TemporalModel.causal' must be False for this task, which summarises whole sequences (causal modules are for the sequence labelling task).
  - 'TemporalModel.bidirectional' must be False when 'TemporalModel.causal' is True: reading the sequence backwards looks at the future frames.
SystemExit code 1
```
The single `causal: True` gives two errors, since the configuration also reads the windows backwards (`bidirectional: True`).

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="sequence_classification") -> <class 'vision_training.trainers.sequence_classification_trainer.SequenceClassificationTrainer'>
```


---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on `configurations/sequence_classification/configuration_synthetic_sequences.yaml` (512 pixel frames, windows of 8 frames, batch size 16), with the run folder redirected to a temporary folder. The whole construction took 0.5 s on CPU.

### Step 2.1 — `SequenceClassificationTrainer.__init__()`
Before calling the constructor of the classification trainer, it keeps `Data.sequence_settings`, read by its hooks (the window length of the batches, the split descriptions), and prints a warning when the frames of the training windows are shuffled (`data_augmentation.shuffle_frames`) while the temporal module models their order (GRU, LSTM, transformer, temporal convolution): shuffling is meant for the windows whose frame order carries no information, with a pooling. The class attributes describe the task to the shared code: `LABEL_TYPE = "sequence"` (one class per window), `MEMBERS_PER_SAMPLE = 1` (one window per sample, 2 for the pairs of the comparison), `SAMPLE_NAME = "Sequences"` and `COUNTS_FRAMES = True` (the frame throughput, Step 3.8).

**Example: the attributes after the construction**
```text
self.sequence_settings          = {'sequence_length': 8, 'window_stride': 8, 'short_sequences': 'pad', 'batch_padding': 'longest'}
self.LABEL_TYPE                 = 'sequence'
self.MEMBERS_PER_SAMPLE         = 1
self.top_k                      = 2                     <- the settings of the classification trainer
self.save_predictions           = True
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
SAMPLE_NAME = 'Sequences'   COUNTS_FRAMES = True   TASK_NAME = 'sequence_classification'
```

### Step 2.2 — `BaseTrainer.__init__()`
The shared constructor (see `classification_workflow.md`, Step 2.2), reached through `ClassificationTrainer.__init__()`, runs the steps below in the same order as for every task. `input_image_size` is the size of every frame.

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
The input and backbone tokens are those of every image task, the backbone being applied to every frame. `build_sequence_sections()` then adds the window, `T<sequence_length>`, and the temporal module (`build_temporal_tokens()`): `MEAN`, `MAX`, `ATT<hidden size>`, `GRU` or `LSTM<layers>x<hidden size>` with `BI` when bidirectional, `TRF<layers>x<hidden size>.<heads>` (with `NPE` without positional embedding) or `TCN<layers>x<hidden size>.<kernel size>`, plus the transfer tokens. The head (`HAVG.64`: average pooling, one hidden layer of 64) and the classes (`MC4`) end the name. The window stride, the padding settings and the temporal dropout give no token, so that they can change between two resumptions of a run.

**Example output**
```text
M2O_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128-BI_HAVG.64_MC4
```

### Step 2.5 — `DataAugmenter()`
The same augmenter as the classification. Here every sample is a window `[T, 3, H, W]`, augmented as a whole (Step 3.12), and two settings are read for the sequences: `independent_frames` (the photometric types drawn frame by frame) and `shuffle_frames`.

**Example: the settings of the configuration**
```text
activated True   probability 0.4   independent_frames False   shuffle_frames {'activated': False, 'probability': 0.5}
5 of 14 types activated: random_crop, random_rotation, brightness_contrast_saturation, random_noise, random_colour_augmentation
```

### Step 2.6 — `setup_training_paths()`
**Example: the run folder after this step**
```text
M2O_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128-BI_HAVG.64_MC4/
├── Tensorboard/
├── Weights/
└── training_configuration.yaml    4,311 bytes
```

### Step 2.7 — `build_dataloaders()`
This hook builds the loaders of the three splits with `label_type="sequence"` (Step 2.8), then keeps `class_names`, the names of the categories of the training manifest in class index order.

**Example output**
```text
class_names = ['moving_left', 'moving_right', 'moving_up', 'moving_down']
```

### Step 2.8 — `get_sequence_dataloaders()`
For each split it builds a `SequenceDataset` (Step 2.9) from `<folder>/labels.json` and the frames of `<folder>/data/`, then wraps it in the Native loader (Step 2.10): the sequence tasks always read their frames with OpenCV on the CPU. Every window is padded to `sequence_length` frames by the dataset, so the default collate could stack them; with `batch_padding: "longest"`, the collate is `collate_longest_windows()` instead, which cuts the windows of every batch to its longest window (Step 3.7), so that fewer padding frames are moved and run through the temporal module when the windows are short.

**Example: size of the three splits** (one window per sequence here, every sequence having 8 frames)
```text
train       2000 windows -> 125 batches of 16 (one epoch = 125 iterations)
validation   200 windows ->  13 batches of 16
test         200 windows ->  13 batches of 16
collate     functools.partial(<function collate_longest_windows>, members_per_sample=1)
```

### Step 2.9 — `SequenceDataset.__init__()` → `compute_window_starts()`
It reads the manifest of the split: its `categories` and its `sequences`, each giving its `id`, its `name`, its ordered `frames` (paths relative to `data/`) and its `category_id`. The windows of a sequence of an unmapped category are left out. `compute_window_starts()` cuts every sequence into windows of `sequence_length` frames, one starting every `window_stride` frames, plus a last one ending on the last frame when the stride does not reach it, so that every frame belongs to a window; a sequence shorter than one window gives one window padded with black frames (`short_sequences: "pad"`) or none (`"skip"`). Every window keeps its sequence, its first frame in the sequence, its frames and the class of each frame (the class of the sequence), and the windows are shuffled once and kept by identifier for the evaluation outputs. No frame is read here.

**Example: a manifest sequence, and the window starts for sequence_length 8 and window_stride 8**
```text
labels.json:  {'id': 1, 'name': 'train_000000', 'frames': ['train_000000/000000.png', 'train_000000/000001.png', ...], 'category_id': 2}
              (2000 sequences, all of 8 frames; 4 categories {'id': 1, 'name': 'moving_left'}, ...)
compute_window_starts( 3 frames) = [0]           <- one window padded with 5 black frames
compute_window_starts( 8 frames) = [0]
compute_window_starts(12 frames) = [0, 4]        <- the last window ends on the last frame, frames 4 to 7 are in both
compute_window_starts(20 frames) = [0, 8, 12]
```

**Example: the first window after shuffling, and the sample the loader gets from it**
```text
window:     {'identifier': 1920, 'sequence_identifier': 1921, 'sequence_name': 'train_001920', 'window_start': 0,
             'frames': ['train_001920/000000.png', ..., 'train_001920/000007.png'], 'frame_labels': [3, 3, 3, 3, 3, 3, 3, 3]}
dataset[0]: images (8, 3, 512, 512) torch.uint8, sequence_mask (8,) torch.bool [True x 8], labels () torch.int64 3,
            sample_identifiers 1920
get_sample_name(1920)  = 'train_001920_000000'          (the sequence and the first frame of the window)
describe_sample(1920)  = {'sequence': 'train_001920', 'first_frame': 'train_001920/000000.png', 'last_frame': 'train_001920/000007.png'}
get_class_counts()     = [495, 503, 488, 514]           (windows of every class, for the balanced class weights)
```

### Step 2.10 — `build_native_loader()`
The same PyTorch `DataLoader` as for the classification (see `classification_workflow.md`, Step 2.10), with the collate of Step 2.8.

**Example output**
```text
len(loader) = 125   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
`get_sample_class_indices()` of the test dataset gives the class of every test window, so `choose_sample_identifiers()` groups the windows by class and gives the 8 sample windows to the 4 classes in turn (2 each), each taken at the ends of its sorted identifiers.

**Example output**
```text
get_test_sample_classes() -> 200 entries, e.g. {159: [1], 177: [3], 57: [3], 93: [0]}
sample_image_identifiers  -> [0, 1, 3, 12, 194, 195, 198, 199]
names                     -> test_000000_000000, test_000001_000000, test_000003_000000, test_000012_000000,
                             test_000194_000000, test_000195_000000, test_000198_000000, test_000199_000000
```

### Step 2.12 — `build_model()`
The classification hook: it creates the `SequenceClassifier` from the registry `TASK_MODEL_CLASSES["sequence_classification"]` (Step 2.13), then, with `class_weighting: "balanced"`, sets `model.class_weights` from `get_class_counts()` of the training dataset (the windows of every class). `BaseTrainer` then moves the model to the device and the dtype.

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          RecurrentTemporalModule    148,992 parameters
model.classification_head      ClassificationHead          16,708 parameters
model.class_weights            buffer [4] torch.float32 [1.0, 1.0, 1.0, 1.0]
model.number_classes           4
```

### Step 2.13 — `SequenceClassifier.__init__()`
The model turns every frame into one vector with the shared backbone and a global pooling (Step 2.14), summarises the vectors of a window into one vector with the temporal module (Step 2.15), and gives the class logits with the head of the classification (Step 2.16). The backbone is the one of the image tasks, so a backbone trained on images can be transferred to it.

### Step 2.14 — `build_backbone()` + `build_global_pooling()`
The backbone shared by every task (see `classification_workflow.md`, Step 2.14), built for a frame of `image_settings.size`. The global pooling (`average` here, or `max`) reduces its final feature map to a vector as wide as the last backbone block: 64 features per frame here.

**Example: the backbone panel of `print_summary()` for a 512×512 frame** (shortened)
```text
│  - Convolution Block 1                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 3, 512, 512]                                         │
│    - Output Shape : [1, 16, 256, 256]                                        │
│  ...                                                                         │
│  - Convolution Block 5                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 64, 32, 32]                                          │
│    - Output Shape : [1, 64, 32, 32]                                          │
Final Backbone output shape: [1, 64, 32, 32]
```

### Step 2.15 — `build_temporal_module()` → `RecurrentTemporalModule`
It builds the module of `TemporalModel.type` with `keep_time_axis=False`, so that it returns one vector per window:
- **poolings**, which ignore the frame order: `mean_pooling` and `max_pooling` (the mean or the maximum of the real frames, no weights) and `attention_pooling` (a weighted mean, the weight of every frame learned from its vector);
- **ordered modules**: `gru` and `lstm` (the final state of the last layer after the last real frame, both directions concatenated when bidirectional), `transformer` (with a learned positional embedding unless `positional_embedding` is False) and `temporal_convolution`.

Every module gives the size of its output in `output_features`, read by the head.

**Example: the module of the configuration** (64 features per frame, hidden size 128, bidirectional)
```text
RecurrentTemporalModule(
  (recurrent_network): GRU(64, 128, batch_first=True, bidirectional=True)
)
output_features = 256          (2 directions x 128)
```

### Step 2.16 — `ClassificationHead.__init__()`
The head of the classification (see `classification_workflow.md`, Step 2.17), fed with the window vector.

**Example output**
```text
ClassificationHead(
  (fully_connected_block): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=256, out_features=64, bias=True)
    (2): ReLU(inplace=True)
  )
  (dropout): Dropout(p=0.2, inplace=False)
  (classifier): Linear(in_features=64, out_features=4, bias=True)
)
```

### Step 2.17 — `initialise_pretrained_components()`
The sequence classification can transfer the backbone, the temporal module and the head of other runs (the poolings without weights cannot be transferred). The head may have more classes than its source run, appended after them as in the classification.

**Example: a run from scratch**
```text
pretrained_configurations['Backbone']           = {'experiment_folder': None, 'iteration': None, 'freeze': False, ...}
pretrained_configurations['TemporalModel']      = {'experiment_folder': None, 'iteration': None, 'freeze': False, 'learning_rate_multiplier': 1.0}
pretrained_configurations['ClassificationHead'] = {'experiment_folder': None, 'iteration': None, 'freeze': False, 'learning_rate_multiplier': 1.0}
pretrained_checkpoint_paths                     = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
```

### Step 2.18 — `build_optimizer()`
**Example output**
```text
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
```

### Step 2.19 — `get_tracked_metrics()` + `MetricTracker.__init__()`
The metrics of the classification: the total loss, the classification loss and the accuracy of every batch. The tracker is created with `counts_frames=True`, so that it also counts the real frames of every interval (Step 3.8).

**Example output**
```text
{'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'accuracy': 'Accuracy'}
MetricTracker: batch_size 16, validation_iteration_ratio 25, sample_name 'Sequences', counts_frames True
```

### Step 2.20 — `print_experiment_information()` (→ `describe_split()` → `count_split_sequences()`)
`describe_split()` counts the sequences, the frames and the windows of every split from its manifest (`count_split_sequences()`, cutting the windows as the dataset does), so that the number of iterations per epoch is the number of training windows over the batch size. With `batch_padding: "longest"` the windows are described as having "at most" `sequence_length` frames.

**Example output**
```text
- Run Name                : M2O_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128-BI_HAVG.64_MC4
- Experiment Folder       : file:///.../example_runs/M2O_IN-KAR-512_..._MC4
...
- Train Data Folder       : file:///home/.../sequence_data/synthetic_moving_shapes/train (2000 sequences, 16000 frames, 2000 windows of at most 8 frames)
- Validation Data Folder  : file:///home/.../sequence_data/synthetic_moving_shapes/validation (200 sequences, 1600 frames, 200 windows of at most 8 frames)
- Test Data Folder        : file:///home/.../sequence_data/synthetic_moving_shapes/test (200 sequences, 1600 frames, 200 windows of at most 8 frames)
- Iterations Per Epoch    : 125
- Batch Size              : 16
- Total Iterations        : 10000000000
- Save And Evaluate Every : 1000 iterations
- Save Only Every         : 500 iterations
...
- Validation Iter. Ratio  : 25
- MLflow Tracking         : Off
```

### Step 2.21 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The backbone panel is shown in Step 2.14 and the receptive field panel is the one of the backbone. The sequence classifier adds the Temporal Module panel, which lists every setting of the section, then the head panel; the size panel lists the temporal module on its own line.

**Example output: the temporal module, head and size panels**
```text
╭────────────────────────────── Temporal Module ───────────────────────────────╮
│ Type               : gru                                                     │
│ Hidden Size        : 128                                                     │
│ Number Layers      : 1                                                       │
│ Number Heads       : 4                                                       │
│ Kernel Size        : 3                                                       │
│ Dropout Rate       : 0.0                                                     │
│ Bidirectional      : True                                                    │
│ Causal             : False                                                   │
│ Positional Embedding : True                                                  │
│ Output Features    : 256                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────── Classification Head ─────────────────────────────╮
│ Global Pooling     : average                                                 │
│ Input Features     : 256                                                     │
│ Hidden Layers      : [64]                                                    │
│ Dropout Rate       : 0.2                                                     │
│ Output Mode        : multi_class                                             │
│ Number Of Classes  : 4                                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 183,648                       │
│ Temporal Module                                          │
│   - Parameters           : 148,992                       │
│ Classification Head                                      │
│   - Parameters           : 16,708                        │
│ Total Model                                              │
│   - Parameters           : 349,348                       │
│   - Estimated Size       : 1.40 MB                       │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.22 — `view_input_batches()` → `show_input_samples()`
With `diagnosis_modes.view_input_images`, the real frames of an augmented training window and of a validation window are laid out on grids of 4 columns (`build_sample_frame_grid()` → `build_frame_grid()`), every frame numbered, and captioned with the class of the window (`Class : moving_up`). It needs a display, so it has no console output to show here.

### Step 2.23 — `restore_last_model()` → `get_last_checkpoint_iteration()` / `load_model()`
The same restoration as every task (see `classification_workflow.md`, Steps 2.26 to 2.28).

**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of `configuration_synthetic_sequences.yaml` shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the error images and the MLflow tracking on. Its run name is `M2O_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_GRU1x128-BI_HAVG.64_MC4`; with batches of 4 windows, an epoch is 500 iterations.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1).

**Example: what happens at each iteration of the run**

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | | ✔ (iteration 12) |
| Test evaluation | | | | | | ✔ | | | | | | | ✔ (iteration 12) |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |

### Step 3.2 — `setup_tensorboard_writers()`
**Example: the curves after the run**
```text
Tensorboard/Train/        Total Loss, Classification Loss, Accuracy, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Accuracy                   (iteration 10)
```

### Step 3.3 — `initialize_training_loop_state()` + `MetricTracker.create_trackers()`
The trackers of the sequence tasks hold one more counter, `interval_frames`, the real frames of the interval (Step 3.8).

**Example: the trackers of an interval**
```text
{'start_time': ..., 'interval_batches': 0, 'total_loss': 0.0, 'classification_loss': 0.0, 'accuracy': 0.0, 'interval_frames': 0}
```

### Step 3.4 — `start_experiment_tracking()`
The MLflow run is created in the experiment `sequence_classification` (the task name), in the local database of the experiments root folder. The run is opened by a background thread, so the line can appear in the middle of the first evaluation.

**Example output**
```text
MLflow Run Created In The Experiment 'sequence_classification' : 7a866950d51f4561ab95dff7eb3ddad9 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.5 — `save_model()` → `save_checkpoint()`
The checkpoint stores the source runs of the three transferable components.

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   4,289,671 bytes      Weights/Iteration_6/model_000006.pt   4,289,671 bytes
Weights/Iteration_9/model_000009.pt   4,289,735 bytes      Weights/Iteration_12/model_000012.pt  4,289,735 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'mlflow_run_id',
 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
model_state: 94 tensors, the first 'class_weights', then 'backbone.convolution_blocks.0. ...'
```
The checkpoints of iterations 3 and 6 have `mlflow_run_id: None`, since the MLflow run was not open yet when they were saved; those of iterations 9 and 12 hold `'7a866950d51f4561ab95dff7eb3ddad9'`.

### Step 3.6 — `run_test_evaluation()`
At iterations 6 and 12 (the final checkpoint), see Stage 4.

### Step 3.7 — `get_next_batch()` → `SequenceDataset.__getitem__()` → `read_window_frames()` + `collate_longest_windows()`
`__getitem__()` reads the window: `read_window_frames()` reads every real frame with OpenCV (BGR, `uint8`) and resizes it like an image of the other tasks (`preprocess_image()`), then leaves black padding frames up to `sequence_length`, with the sequence mask, True for the real frames, which come first. The label is the class of the window (an `int64` scalar). With `batch_padding: "longest"`, `collate_longest_windows()` then cuts the frames and the mask of every window of the batch to the number of real frames of its longest window before stacking them, so that a batch of short windows holds few padding frames. It runs in the loader workers.

**Example: the first batch of the run** (every window has 8 real frames here, so nothing is cut)
```text
images               [4, 8, 3, 64, 64] torch.uint8      [B, T, C, H, W]
sequence_mask        [4, 8] torch.bool                  all True
labels               [4] torch.int64                    [2, 3, 3, 0]
sample_identifiers   [4] torch.int64                    [322, 324, 1436, 1870]
```

### Step 3.8 — `MetricTracker.count_frames()`
Before the batch goes to the device, the tracker adds its real frames (the sum of the sequence mask) to the `interval_frames` of the interval, from which the console summary and MLflow get the frame throughput. The padding frames are not counted, so the throughput is the same whether the batches are padded to the sequence length or to their longest window.

**Example: the frames of the first interval**
```text
interval_frames = 128          (4 batches x 4 windows x 8 frames)
```

### Step 3.9 — `set_training_mode()`
`model.train()`: dropout on, batch normalization statistics updated.

### Step 3.10 — `run_model_iteration()`
The same function for the training and the validation batches (see `classification_workflow.md`, Step 3.11): prepare, augment, forward, sum of the `_loss` outputs, batch metrics, tracking.

**Example: the first training batch**
```text
classification_loss = 1.4035   total_loss = 1.4035   accuracy = 0.25
```

### Step 3.11 — `prepare_batch()`
The classification hook moves the frames (`float32`, still in [0, 255] BGR) and the labels to the device; the sequence hook adds the sequence mask, as booleans.

**Example: before and after**
```text
images         [4, 8, 3, 64, 64] torch.uint8   -> [4, 8, 3, 64, 64] torch.float32
sequence_mask  [4, 8] torch.bool               -> [4, 8] torch.bool
labels         [4] torch.int64                 -> [4] torch.int64
```

### Step 3.12 — `augment_batch()` → `DataAugmenter.augment_batch()` + `shuffle_frames()`
The augmenter decides for every window whether it is augmented (`probability: 0.4`), then applies every activated type to the whole window `[T, 3, H, W]`, so that all its frames share the same crop, rotation and colour change and the motion stays coherent; with `independent_frames`, the photometric types are drawn again for every frame. Then `shuffle_frames()` permutes the real frames of every window with the `shuffle_frames` probability when it is activated (off here), the padding frames staying last; for the sequence labelling the frame labels follow their frames. The window length given to it is that of the batch (`get_window_length()`), which is the longest window with `batch_padding: "longest"`.

**Example: which windows of the first batch were changed**
```text
images:        changed per sample [False, True, False, False]
sequence_mask: changed per sample [False, False, False, False]
labels:        changed per sample [False, False, False, False]
```

### Step 3.13 — `forward_pass()` → `SequenceClassifier.forward(sequence_mask=..., labels=...)`
The trainer passes the frames, the mask and the labels, so that the model returns its loss with its logits.

**Example: the dictionary returned in training mode**
```text
class_logits          [4, 4] torch.float32
class_probabilities   [4, 4] torch.float32
classification_loss   1.4035 (scalar)                   <- close to ln 4 = 1.386, the loss of a model that does not know yet
```

### Step 3.14 — `encode_frames()`
The real frames of all the windows of the batch are gathered into one batch of images (`input_tensor[sequence_mask]`) and run through the backbone and the global pooling, so that the padding frames cost no computation and stay out of the batch normalization statistics. The vectors are then put back at the position of their frame, the padding frames getting zero vectors. The window of every frame is also given to the backbone, for the `sequence` normalization of the feature maps, which computes its statistics over the frames of each window. Without mask (deployment windows, all real) the frames run as one fixed-shape batch, which the exports need.

**Example: the first batch through the trained model**
```text
images [4, 8, 3, 64, 64] -> backbone on the 32 real frames -> encode_frames -> [4, 8, 64]
zero vectors per window [0, 0, 0, 0]          (no padding frame in these windows)
```

### Step 3.15 — temporal module: `RecurrentTemporalModule.forward()`
The GRU reads the frame vectors of every window in order. With a mask, the windows are packed (`pack_padded_sequence()` with the number of real frames of each), so that the GRU stops at the last real frame of every window and never reads its padding frames; the vector of the window is the final state of the last layer in both directions, concatenated. The other modules use the mask the same way: the poolings and the transformer leave the padding frames out.

**Example: the first batch**
```text
[4, 8, 64] frame vectors -> temporal_module -> [4, 256] window vectors
```

### Step 3.16 — `ClassificationHead.forward()` + `compute_classification_loss()`
The head turns every window vector into the class logits, then the loss of the classification is computed (see `classification_workflow.md`, Steps 3.16 to 3.18): the cross entropy of the softmax, weighted by `class_weights`, with `label_smoothing`.

**Example: the first batch**
```text
[4, 256] -> classification_head -> class_logits [4, 4]      classification_loss = 1.4035
```

### Step 3.17 — `compute_batch_metrics()`
The accuracy of the batch, from the argmax of the logits, without a second forward pass.

**Example: the first batch**
```text
accuracy = 0.25          (1 window of 4 right)
```

### Step 3.18 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
**Example: the Train curves**
```text
Total Loss   1: 1.4035, 2: 1.3691, 3: 1.3889, 4: 1.4091, 5: 1.3453, 6: 1.2685, 7: 1.5344, 8: 1.3843, 9: 1.3543, 10: 1.4542, 11: 1.2087, 12: 1.6073
Accuracy     1: 0.2500, 2: 0.5000, 3: 0.2500, 4: 0.2500, 5: 0.2500, 6: 0.5000, 7: 0.2500, 8: 0.2500, 9: 0.5000, 10: 0.2500, 11: 0.5000, 12: 0.0000
```

### Step 3.19 — `training_loss.backward()` + `step_optimizer()`
**Example: the gradient norm of the first steps and the learning rate**
```text
iteration 1: 1.0781    iteration 2: 1.2063    iteration 3: 1.235        learning rate 0.001 (constant)
```

### Step 3.20 — validation pass
**Example: iteration 10**
```text
Tensorboard/Validation at iteration 10: Total Loss 1.3571, Classification Loss 1.3571, Accuracy 0.5000
```

### Step 3.21 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The sums of the interval are divided by the number of batches it ran, the frame counter is kept as it is. The header gives the windows per second (`Sequences Per Second`) and the real frames per second, and MLflow receives the second one as `train/frame_throughput`.

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 5.5706, 'classification_loss': 5.5706, 'accuracy': 1.25, 'interval_frames': 128}
after:  {'interval_batches': 4, 'total_loss': 1.3927, 'classification_loss': 1.3927, 'accuracy': 0.3125, 'interval_frames': 128}
MLflow, step 4: {'train/Total Loss': 1.3927, 'train/Classification Loss': 1.3927, 'train/Accuracy': 0.3125,
                 'train/throughput': 5.5671, 'train/interval_seconds': 2.874, 'learning_rate': 0.001,
                 'train/frame_throughput': 44.5371}
```

**Example output: iterations 4 and 12**
```text
===========================================================================================================================================================
|     Iteration: 4    |     Total Sequences: 16     |    Interval Sequences: 16    |   Throughput: 5.6 Sequences Per Second  |   44.5 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   1.3927
Moving Average of Train Classification Loss            :   1.3927
Moving Average of Train Accuracy                       :   0.3125
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:02.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
...
===========================================================================================================================================================
|    Iteration: 12    |     Total Sequences: 48     |    Interval Sequences: 16    |  Throughput: 10.7 Sequences Per Second  |   85.6 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   1.4061  +0.0230               <- red: higher than at iteration 8
Moving Average of Valid Total Loss                     :   1.3571
Moving Average of Train Classification Loss            :   1.4061  +0.0230
Moving Average of Valid Classification Loss            :   1.3571
Moving Average of Train Accuracy                       :   0.3125  +0.0000
Moving Average of Valid Accuracy                       :   0.5000
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:01.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
```
The interval of iteration 8 shows 2.3 Sequences Per Second, since it holds the evaluation of iteration 6.

### Step 3.22 — end of the loop
**Example output**
```text
Model successfully saved at iteration 12. Exiting Training.
Iteration 12 · 200 Test Sequences · 2026-10-09 10:12 ───────────────      <- the evaluation of the final checkpoint
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (200 windows of 8 frames of 64×64 pixels), with `per_class_report: True`, `number_sample_images: 8`, `top_k: 2`, `save_predictions: True`, `error_dumps` activated with `maximum_per_class_pair: 2`, and the MLflow tracking on. Both checkpoints predict `moving_left` for every window.

### Step 4.1 — `run_test_evaluation()`
The evaluation of the classification (see `classification_workflow.md`, Stage 4), every sample being a window. The test windows are never augmented nor shuffled.

**Example output: the progress bar** (200 windows, batch size 4)
```text
Test Evaluation Iteration 6: 100%|██████████| 50/50 [00:05<00:00,  9.67it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The lists of the classification: the probabilities, labels and sample identifiers of every batch, the `Errors` folder and the counter of error images per pair of classes.

**Example: the accumulator at the end of the loop**
```text
'probabilities':      50 tensors, the first [4, 4] torch.float64
'labels':             50 tensors, the first [4] torch.int64
'sample_identifiers': 50 tensors, the first [4] torch.int64
'errors_folder':      PosixPath('.../Evaluation/Iteration_6/Errors')
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
The same preparation as in training (Step 3.11); `predict_batch()` runs the model with the sequence mask and without labels, so no loss is computed.

**Example: the predictions of a test batch**
```text
class_logits          [4, 4] torch.float32
class_probabilities   [4, 4] torch.float32      (no 'classification_loss' key)
```

### Step 4.4 — `accumulate_evaluation_batch()`
The classification hook: the probabilities (as `float64`), the labels and the window identifiers are kept on the CPU, and `dump_errors()` runs while the frames are still in the batch.

### Step 4.5 — `dump_errors()` → `reserve_error_image()` + `build_input_image()`
As for the classification, every wrong prediction is written in `Errors/<TRUE>-TO-<PREDICTED>/`, at most `maximum_per_class_pair` per pair. The image is the one of `build_input_image()`: the real frames of the window laid out on a grid of 4 columns with a 4 pixel gray gap, every frame numbered, captioned with both classes and the probability of the predicted class. The file is named after the window (its sequence and its first frame) and that probability.

**Example: the error images of the checkpoint of iteration 12**
```text
build_input_image(sample_index=0) -> (140, 276, 3) uint8      2 rows x 4 columns of 64 x 64 frames
Evaluation/Iteration_12/Errors/
├── moving_down-TO-moving_left/     test_000046_000000_0.277.png   test_000106_000000_0.277.png
├── moving_right-TO-moving_left/    test_000039_000000_0.277.png   test_000108_000000_0.277.png
└── moving_up-TO-moving_left/       test_000074_000000_0.277.png   test_000168_000000_0.277.png
report line: "Error Images : 6 in 3 class pairs, at most 2 per pair"
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
For every fixed sample window of Step 2.11, the classification hook draws the frame grid of `build_input_image()` with a caption bar giving its true class, its predicted class and the probability of the predicted class, in green when right and in red when wrong. It is saved as `Samples/<window name>.png` and sent to TensorBoard under `Samples/<window name>`.

**Example: the sample images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Samples/
test_000000_000000.png  test_000001_000000.png  test_000003_000000.png  test_000012_000000.png
test_000194_000000.png  test_000195_000000.png  test_000198_000000.png  test_000199_000000.png
build_sample_image -> [164, 276, 3] torch.uint8      the 140 x 276 frame grid under the caption bar
note of the report: "Sample Images : 8 fixed test sequences"
```

### Step 4.7 — `compute_evaluation_results()`
The results of the classification (see `classification_workflow.md`, Step 4.8). The headline metric, used by Stage 5 and MLflow to choose the best checkpoint, is the accuracy.

**Example: the results of iteration 6**
```text
evaluated_samples = '200 Test Sequences'
metrics           = {'Accuracy': 0.245, 'Balanced Accuracy': 0.25, 'Macro F1': 0.0984, 'Top-2 Accuracy': 0.475}
notes             = ['Error Images : 6 in 3 class pairs, at most 2 per pair', 'Sample Images : 8 fixed test sequences']
headline_metric   = 'Accuracy'
files             = Confusion Matrix Table, Class Thresholds, Class Metrics, Confusion Matrix Image, Predictions, Error Images, Sample Images
images            = Confusion Matrix        sample_images = Samples/test_000000_000000, Samples/test_000199_000000, ... (8)
class_scalars     = 12: Precision/moving_left, Recall/moving_left, F1/moving_left, Precision/moving_right, ...
```

### Step 4.8 — `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()`
The confusion matrix counts the windows of every pair (true class, predicted class); the precision, recall, F1, accuracy, balanced accuracy and macro F1 come from it (see `classification_workflow.md`, Steps 4.9 and 4.10).

**Example: `Evaluation/Iteration_6/confusion_matrix.csv`**
```text
true \ predicted,moving_left,moving_right,moving_up,moving_down
moving_left,49,0,0,0
moving_right,52,0,0,0
moving_up,53,0,0,0
moving_down,46,0,0,0
```

**Example: how the numbers of iteration 6 are obtained**
```text
accuracy          = 49 / 200                                  = 0.245    (only the moving_left windows are right)
moving_left       precision = 49 / 200 = 0.245    recall = 49 / 49 = 1.0    F1 = 2 x 0.245 x 1.0 / (0.245 + 1.0) = 0.3936
the 3 other classes   precision = 0 (never predicted)    recall = 0.0    F1 = 0.0
balanced accuracy = (1.0 + 3 x 0.0) / 4                       = 0.25
macro F1          = (0.3936 + 3 x 0.0) / 4                    = 0.0984
```

### Step 4.9 — `compute_top_k_accuracy()`
The share of windows whose true class is among the `top_k` most probable classes.

**Example output** (`top_k: 2`)
```text
Top-2 Accuracy = 0.4750 at iteration 6, 0.5050 at iteration 12      (about 2 / 4 = 0.5: chance level for 4 classes)
```

### Step 4.10 — `find_class_thresholds()`
For every class, the lowest probability threshold keeping a precision of at least `target_precision`, with the recall kept, saved in `class_thresholds.json`.

**Example: `Evaluation/Iteration_6/class_thresholds.json`**
```text
{
    "target_precision": 0.99,
    "classes": {
        "moving_left": {
            "threshold": 0.27333855628967285,
            "recall": 0.02040816326530612            <- 1 of the 49 moving_left windows
        },
        "moving_right": {
            "threshold": 0.24627721309661865,
            "recall": 0.019230769230769232           <- 1 of the 52 moving_right windows
        },
        "moving_up": null,
        "moving_down": null
    }
}
```

### Step 4.11 — `build_class_metrics_rows()` + `write_class_metrics_file()` + `save_confusion_matrix_image()`
`class_metrics.csv` is written at every evaluation; `confusion_matrix.png` only with `per_class_report` (see `classification_workflow.md`, Steps 4.14 and 4.15).

**Example: `Evaluation/Iteration_6/class_metrics.csv`**
```text
class_index,class,samples,precision,recall,f1,threshold_at_target_precision,recall_at_target_precision
0,moving_left,49,0.245000,1.000000,0.393574,0.273339,0.020408
1,moving_right,52,0.000000,0.000000,0.000000,0.246277,0.019231
2,moving_up,53,0.000000,0.000000,0.000000,,
3,moving_down,46,0.000000,0.000000,0.000000,,
```

### Step 4.12 — `write_predictions_file()`   (only with `save_predictions`)
One row per test window, in the order of the test loader. `describe_sample()` of the sequence dataset names a window by its sequence and its first and last real frames, since a long sequence is cut into several windows.

**Example: `Evaluation/Iteration_6/predictions.csv`** (200 rows, 24,309 bytes; first rows)
```text
sequence,first_frame,last_frame,true_class,predicted_class,correct,probability_moving_left,probability_moving_right,probability_moving_up,probability_moving_down
test_000089,test_000089/000000.png,test_000089/000007.png,moving_up,moving_left,0,0.273332,0.246274,0.232063,0.248331
test_000027,test_000027/000000.png,test_000027/000007.png,moving_left,moving_left,1,0.273332,0.246274,0.232063,0.248331
test_000123,test_000123/000000.png,test_000123/000007.png,moving_up,moving_left,0,0.273332,0.246274,0.232063,0.248331
```
The probabilities are almost the same for every window: the checkpoint of iteration 6 was trained for 5 steps only.

### Step 4.13 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 200 Test Sequences · 2026-10-09 10:12 ───────────────────────────────────────────────────────────────────
  Accuracy   0.2450   Balanced Accuracy   0.2500
  Macro F1   0.0984   Top-2 Accuracy      0.5050
  Error Images : 6 in 3 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                          Threshold    Recall
   Class          Samples   Precision   Recall       F1     @ P0.99   @ P0.99
  ────────────────────────────────────────────────────────────────────────────
   moving_right        52      0.0000   0.0000   0.0000      0.2440    0.0192
   moving_up           53      0.0000   0.0000   0.0000           —         —
   moving_down         46      0.0000   0.0000   0.0000           —         —
   moving_left         49      0.2450   1.0000   0.3936      0.2771    0.0612
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 0.99 is never reached.
  - Results : file:///.../Results.txt
  - Confusion Matrix Table : file:///.../Evaluation/Iteration_12/confusion_matrix.csv
  - Class Thresholds : file:///.../Evaluation/Iteration_12/class_thresholds.json
  - Class Metrics : file:///.../Evaluation/Iteration_12/class_metrics.csv
  - Confusion Matrix Image : file:///.../Evaluation/Iteration_12/confusion_matrix.png
  - Predictions : file:///.../Evaluation/Iteration_12/predictions.csv
  - Error Images : file:///.../Evaluation/Iteration_12/Errors
  - Sample Images : file:///.../Evaluation/Iteration_12/Samples
```

**Example: the run folder, TensorBoard and MLflow after the two evaluations**
```text
Evaluation/Iteration_6/ and Iteration_12/   Errors/ (3 pairs x 2)  Samples/ (8)  class_metrics.csv  class_thresholds.json
                                            confusion_matrix.csv  confusion_matrix.png  predictions.csv
Tensorboard/Test: 16 scalars (4 global + 12 per class), 9 images (8 samples + Confusion Matrix)
    scalar Accuracy          6: 0.2450, 12: 0.2450
    scalar Top-2 Accuracy    6: 0.4750, 12: 0.5050
MLflow: test/Accuracy, test/Balanced Accuracy, test/Macro F1, test/Top-2 Accuracy, test/best_Accuracy (steps 6, 12),
        12 test_class/<metric>/<class>, train/frame_throughput (steps 4, 8, 12),
        artifacts evaluation/iteration_6/ and iteration_12/: class_metrics.csv, class_thresholds.json, confusion_matrix.png
```


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 40 sequences of the test split (its `labels.json` reduced, its `data/` folder linked).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5): set `run_name` to the sequence run, and `test_data_folder` to another labelled split (holding `labels.json` and `data/`) if needed. The windows are cut with the `sequence_settings` of the run.

### Step 5.2 — trainer construction
The script loads `training_configuration.yaml` of the run, points it to the output folder, the reduced split and the batch size, and builds a `SequenceClassificationTrainer`.

**Example: the split line of the launch information**
```text
- Test Data Folder        : file:///.../test_reduced (40 sequences, 320 frames, 40 windows of at most 8 frames)
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'sequence_classification' : 841723d3fc1a4c6191371f888d19af87 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
The best checkpoint is the one with the highest headline metric; on a tie the first one evaluated is kept, the last checkpoint here, since the checkpoints are evaluated from the last one.

**Example output** (every checkpoint predicts `moving_left` for the 40 windows, so they all have the same accuracy)
```text
Evaluation Summary Of 4 Checkpoints : file:///.../runs/evaluations/M2O_IN-KAR-64_..._MC4/Evaluation/evaluation_summary.csv
  - Best Checkpoint : iteration 12 (Accuracy 0.2750)

iteration,Accuracy,Balanced Accuracy,Macro F1,Top-2 Accuracy,Precision/moving_left,Recall/moving_left,F1/moving_left, ...
12,0.275,0.25,0.10784313725490198,0.55,0.275,1.0,0.4313725490196079, ...
9,0.275,0.25,0.10784313725490198,0.55,0.275,1.0,0.4313725490196079, ...
6,0.275,0.25,0.10784313725490198,0.45,0.275,1.0,0.4313725490196079, ...
3,0.275,0.25,0.10784313725490198,0.45,0.275,1.0,0.4313725490196079, ...
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 12 on the 40 sequences**
```text
Evaluating : file:///.../runs/M2O_IN-KAR-64_..._MC4/Weights/Iteration_12/model_000012.pt
Iteration 12 · 40 Test Sequences · 2026-10-09 10:12 ────────────────────────────────────────────────────────────────────
  Accuracy   0.2750   Balanced Accuracy   0.2500
  Macro F1   0.1078   Top-2 Accuracy      0.5500
  Error Images : 6 in 3 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                          Threshold    Recall
   Class          Samples   Precision   Recall       F1     @ P0.99   @ P0.99
  ────────────────────────────────────────────────────────────────────────────
   moving_right        11      0.0000   0.0000   0.0000      0.2439    0.0909
   moving_up           11      0.0000   0.0000   0.0000           —         —
   moving_down          7      0.0000   0.0000   0.0000           —         —
   moving_left         11      0.2750   1.0000   0.4314      0.2771    0.1818
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 0.99 is never reached.
```
The sample windows were chosen from the 40 windows of this split, so they are not the same windows as in Stage 4.
