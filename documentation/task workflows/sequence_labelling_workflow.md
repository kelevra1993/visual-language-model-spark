# Sequence Labelling Workflow

This document follows one sequence labelling run (many to many, M2M: one class per frame of the sequences) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The sequence labelling trainer is the sequence classification trainer with one class per frame: the steps it shares with the sequence classification (whole sequences, padding of the batches, frame encoder) are explained in `sequence_classification_workflow.md`, and the steps shared by every task in `classification_workflow.md`; here they keep a short explanation and the real output of the labelling run, and the steps specific to the frame labels (labels per frame, temporal module keeping the time axis, frame metrics, segment metrics of the sequences, frame error images) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training. The synthetic dataset of the configuration, `synthetic_changing_shapes` (250,000 training, 50,000 validation and 50,000 test sequences), is too large for short CPU runs, so a small one was generated in a temporary folder with `create_synthetic_dataset(task="sequence_labelling", image_size=128, maximum_objects=10, minimum_frames=2, maximum_frames=12)`: 500 training, 60 validation and 60 test sequences of 2 to 12 frames of 128 pixels. Every frame is labelled `no_change` (always the first frame), `appearing_<shape>` or `disappearing_<shape>`: 71 classes for the 35 shape classes, about half of the frames being `no_change`. The data folders and the mapping file of the configuration were redirected to it.
- **Stages 1 and 2**: `configurations/sequence_labelling/configuration_synthetic_sequences.yaml` (sequences of at most 50 frames fed whole, 512 pixel frames, batch size 16, a GRU), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the error images and the MLflow tracking turned on. After 12 iterations the network predicts mostly `no_change` (71 classes: a first loss near ln 71 = 4.263), so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

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
│       ├── restore_run_label_dictionary() / move_previous_run_folder() / setup_training_paths()
│       │       The classes of a resumed run kept from its label_dictionary.json, a previous training of the same
│       │       settings moved to v<n>_<run name> when the run starts from scratch, then the run folder with the
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
        └── save_model() of the steps taken since the last checkpoint, if any, finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
├── create_evaluation_accumulator()
│       The probabilities and labels of the labelled frames, the sequence and position of every labelled frame,
│       whether every labelled frame of each sequence is right, and the folder and counts of the error images.
├── for each test batch:
│   ├── prepare_batch() / predict_batch()
│   ├── accumulate_evaluation_batch() -> compute_sequence_exact_matches()
│   ├── dump_errors() -> build_frame_prediction_captions() + build_frame_grid()   one image per wrong frame
│   └── save_sample_images() -> build_sample_image()   frame grid with true/predicted classes of every frame
├── compute_evaluation_results()
│   ├── the frame metrics of the classification     confusion matrix, accuracy, balanced accuracy, macro F1, top-k,
│   │                                               thresholds, class_metrics.csv
│   ├── compute_extra_headline_metrics()
│   │   ├── Sequence Exact Match                    share of the sequences whose labelled frames are all right
│   │   └── compute_segment_results()
│   │       ├── group_sequence_predictions()        the labelled frames grouped sequence by sequence, in frame order
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

**Example: the sections, the sequence settings and the temporal module**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, ClassificationHead
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'top_k': 2, 'target_precision': 0.99,
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
The labelling trainer has no constructor of its own: it is the sequence classification trainer with `LABEL_TYPE = "frame"` (the loaders read one class per frame) and `CLASS_SUPPORT_COLUMN = "Frames"` (the per-class table counts frames). It overrides the batch metric, the evaluation accumulator, the error images, the sample images, the predictions file and the extra headline metrics, since every frame has its own class.

**Example: the attributes after the construction**
```text
self.sequence_settings          = {'maximum_sequence_length': 50}
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
The sections of the sequence classification (`sequence_classification_workflow.md`, Step 2.4): `T50` (the maximum sequence length), then `GRU1x128` (one layer of 128, not bidirectional), and `CAU` would follow a causal transformer or temporal convolution. `MC71` gives the 71 frame classes.

**Example output**
```text
M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71
```

### Step 2.5 — `DataAugmenter()`
**Example: the settings of the configuration**
```text
activated True   probability 0.4   independent_frames False   shuffle_frames {'activated': False, 'probability': 0.5}
6 of 20 types activated: random_crop, random_left_right_flip, random_rotation, brightness_contrast_saturation, random_noise,
                         random_haze
```

### Step 2.6 — `setup_training_paths()`
As for every task (see `classification_workflow.md`, Step 2.6): a resumed run first keeps the classes of its `label_dictionary.json` (`restore_run_label_dictionary()`), a training from scratch renames the folder of a previous training of the same settings `v<n>_<run name>` (`move_previous_run_folder()`), then the run folder is created with the copy of the configuration and `label_dictionary.json`, the class index of every category of the run.

**Example: the run folder after this step**
```text
M2M_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128_HAVG.64_MC71/
├── Tensorboard/
├── Weights/
├── label_dictionary.json          the class index of every category of the run
└── training_configuration.yaml    4,881 bytes
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
data.json    151,989 bytes   keys ['created', 'class_names', 'splits']
             splits['train'] = {'data_folder': '<datasets>/synthetic_changing_shapes/train', 'labels_sha256': 'e5172364...',
                                'number_images': 3388, 'images': ['train_000000/000000.png', 'train_000000/000001.png', ...]}
```

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
**Example output**
```text
len(loader) = 32   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.10 — `get_test_sample_classes()` + `choose_sample_identifiers()`
Every test sequence gives the classes of its labelled frames, and `choose_sample_identifiers()` groups it under its most frequent class (the lowest class index on a tie). Since about half of the frames are `no_change`, all 60 test sequences fall in the `no_change` group, so the 8 sample sequences are spread evenly over that group.

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
model.classification_head      ClassificationHead          12,871 parameters      Linear(128, 64) -> ReLU -> Linear(64, 71)
model.class_weights            buffer [71] torch.float32 [1.0, 1.0, 1.0, ...]
total                                                     271,015 parameters
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
The summary panels are those of the sequence classification, the temporal module panel giving `Output Features : 128` and the head panel `Number Of Classes : 71`. With `view_input_images`, every frame of the grids is captioned with its class (`get_frame_captions()`, `none` for an unmapped category), and the information panel of the sequence starts with the order of its classes (`get_sample_caption()`).

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


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (60 sequences, 427 labelled frames), with `per_class_report: True`, `number_sample_images: 8`, `top_k: 2`, `save_predictions: True`, `error_dumps` activated with `maximum_per_class_pair: 2`, and the MLflow tracking on. The checkpoint of iteration 6 gives `disappearing_hollow_ellipse` to every frame; the one of iteration 12 gives `no_change` to 307 frames and `disappearing_hollow_ellipse` to the 120 others. The outputs of iteration 6 were captured when its checkpoint was still saved before its training step, after 5 steps; it now holds 6 steps.

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
Every wrong frame gives one image: the grid of its sequence, every real frame captioned `true/predicted` (with `X ` before a wrong frame, and `none` as the true class of an unlabelled frame), under a caption bar `frame <index> : <true> -> <predicted> <score>` naming the wrong frame, its classes and the probability of its predicted class. It is saved as `Errors/<TRUE>-TO-<PREDICTED>/<sequence>_frame<index>_<score>.png`, at most `maximum_per_class_pair` per pair.

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
The labelling hook draws the frame grid of the sequence (`FRAME_GRID_COLUMNS = 4` frames per row), every real frame captioned `true/predicted` with `X ` before a wrong frame, under a caption bar giving the number of labelled frames that are right, in green when all are and in red otherwise.

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
The frame metrics of the classification are computed on the labelled frames (Steps 4.8 and 4.9), then `compute_extra_headline_metrics()` adds the sequence metrics (Steps 4.10 and 4.11), and `describe_extra_notes()` the number of segments. `describe_evaluated_samples()` gives the sequences and their labelled frames: every frame of the test split is evaluated once. The headline metric is the frame accuracy.

**Example: the results of iteration 12**
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
files             = Results.txt, confusion_matrix.csv, class_thresholds.json, class_metrics.csv, confusion_matrix.png,
                    predictions.csv, Errors, Samples
```

### Step 4.8 — the frame metrics: `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()` + `compute_top_k_accuracy()`
The metrics of the classification (see `classification_workflow.md`, Steps 4.8 to 4.11), every labelled frame of every sequence being one sample.

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

**Example: TensorBoard and MLflow after the two evaluations**
```text
Tensorboard/Test: 222 scalars (9 global + 213 per class), 9 images (8 samples + Confusion Matrix)
    scalar Accuracy        6: 0.0047, 12: 0.2787
    scalar Edit Score      6: 0.0038, 12: 0.1424
    scalar Segment F1@50   6: 0.0,    12: 0.0384
MLflow: 233 metrics: test/Accuracy, ..., test/Edit Score, test/Segment F1_at_10, test/Segment F1_at_25, test/Segment F1_at_50,
        test/Sequence Exact Match, test/best_Accuracy (steps 6, 12), 213 test_class/<metric>/<class>, train/*, validation/*, learning_rate
```
MLflow only accepts letters, digits, `_`, `-`, `.`, `/` and spaces in a metric name, so `sanitise_metric_name()` writes `Segment F1@10` as `Segment F1_at_10` (the original names are kept in the `metric_names.json` artifact); TensorBoard and the console keep the `@`.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12, captured when those of iterations 3, 6 and 9 were still saved before their training step) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 20 sequences of the test split (its `labels.json` reduced, its `data/` folder linked).

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
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
  - Best Checkpoint : iteration 12 (Accuracy 0.2981)

iteration,Accuracy,Balanced Accuracy,Macro F1,Top-2 Accuracy,Sequence Exact Match,Edit Score,Segment F1@10,Segment F1@25,Segment F1@50, ...
12,0.2981366459627329,0.01309328968903437,0.010264086389393778,0.36024844720496896,0.0,0.12883116883116885,0.1927710843373494,0.10843373493975904,0.012048192771084338, ...
9,0.0,0.0,0.0,0.012422360248447204,0.0,0.0,0.0,0.0,0.0, ...
6,0.0,0.0,0.0,0.012422360248447204,0.0,0.0,0.0,0.0,0.0, ...
3,0.0,0.0,0.0,0.012422360248447204,0.0,0.0,0.0,0.0,0.0, ...
```
The file has 223 columns: the iteration, the 9 global metrics, then the precision, recall and F1 of the 71 classes. It is written in `evaluations/<run name>/Evaluation/`, next to the `Iteration_12`, `Iteration_9`, `Iteration_6` and `Iteration_3` folders of the evaluation.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 12 on the 20 sequences** (shortened)
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
