# Sequence Comparison Workflow

This document follows one sequence comparison run (CMP: one class per pair of sequences, a before member and an after member) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The sequence comparison trainer is the sequence classification trainer with pairs instead of windows: the steps it shares with the sequence classification are explained in `sequence_classification_workflow.md`, and the steps shared by every task in `classification_workflow.md`; here they keep a short explanation and the real output of the comparison run, and the steps specific to the pairs (member windows, pair batches, member split, member fusion) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training:
- **Stages 1 and 2**: `configurations/sequence_comparison/configuration_synthetic_pairs.yaml` (synthetic object loss pairs, 2 classes `loss_of_object` and `same_number_of_objects`, members of 2 to 6 frames, member windows of 8 frames, 512 pixel frames, batch size 16, a mean pooling and a difference fusion), with the run folder redirected to a temporary folder. The dataset of `configuration_example.yaml` is not on this machine.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the error images and the MLflow tracking turned on. After 12 iterations the network predicts `same_number_of_objects` for every test pair (2 classes: a loss near ln 2 = 0.693), so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

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
│   Entry point of every task: reads the path of the YAML file from --configuration_path, loads it, looks up the
│   trainer of its task, builds it and starts the training loop. Give a sequence_comparison configuration to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, TemporalModel, Fusion, ClassificationHead), converts the types, adds label_mode and
│   │   label_dictionary to 'Data' and runs the checks.
│   │
│   ├── load_configuration(configuration_path)
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category id: class index}; one category per pair.
│   └── check_configuration()
│       ├── common checks                         experiment, data, backbone, optimization (see classification)
│       └── check_sequence_comparison_configuration()
│               ├── the data, head and evaluation checks of the classification
│               ├── check_sequence_settings()             sequence_length, short_sequences, batch_padding (no
│               │                                         window_stride: a member is one window)
│               ├── check_temporal_model_configuration()  any pooling or ordered type, not causal
│               └── check_fusion_configuration()          type, heads and dropout of the cross attention, the
│                                                         frames sharing the sequence normalization ("member", "pair")
│
├── get_trainer_class(task)  -> SequenceComparisonTrainer
│
├── SequenceComparisonTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   The SequenceClassificationTrainer with LABEL_TYPE = "pair", MEMBERS_PER_SAMPLE = 2 and SAMPLE_NAME = "Pairs".
│   │
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter() / setup_training_paths()
│       │       The run name adds the member window (T<frames>), the temporal module and the fusion (FCAT, FDIF,
│       │       FXA<heads>, SNP when both members share the sequence normalization) before the head and the classes.
│       ├── build_dataloaders()
│       │   └── get_sequence_dataloaders(label_type="pair")
│       │       │   One SequencePairDataset per split, and with batch_padding "longest" the collate
│       │       │   collate_longest_windows (members_per_sample=2).
│       │       │
│       │       ├── SequencePairDataset.__init__() -> compute_member_frame_indices()
│       │       │       Each member becomes one window of at most sequence_length frames: all its frames, or the
│       │       │       middle frame of each of sequence_length equal parts of a longer member. The pairs of unmapped
│       │       │       categories, with an empty member, or with a short member when short_sequences is "skip", are
│       │       │       left out. Shuffled once.
│       │       └── build_native_loader()
│       ├── get_test_sample_classes() + choose_sample_identifiers()   the class of every test pair
│       ├── build_model()
│       │   └── SequenceComparator.__init__()
│       │       ├── build_backbone() + build_global_pooling()   one vector per frame of both members
│       │       ├── build_temporal_module()                     shared by both members, one vector per member
│       │       ├── MemberFusion()                              [before, after] or [before, after, after - before],
│       │       │                                               with a cross attention between the frames for
│       │       │                                               "cross_attention"
│       │       ├── ClassificationHead()
│       │       └── class_weights buffer
│       ├── initialise_pretrained_components()  Backbone, TemporalModel and ClassificationHead can come from other runs
│       ├── build_optimizer() / get_tracked_metrics() + MetricTracker.__init__(counts_frames=True)
│       ├── print_experiment_information() -> describe_split() -> count_split_pairs()
│       │       "N pairs, F frames, P kept pairs of at most 2 x L frames"; an epoch is P pairs.
│       ├── print_summary()                     adds the Member Fusion panel
│       ├── view_input_batches() -> show_input_samples()   the before grid above the after grid
│       └── restore_last_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── for each training_iteration:
        │   ├── save_model() + run_test_evaluation() when due                    -> [STAGE 4]
        │   ├── get_next_batch()
        │   │   ├── SequencePairDataset.__getitem__() -> read_window_frames() for each member
        │   │   │       The before window then the after window along the frame axis: [2 x sequence_length] frames.
        │   │   └── collate_longest_windows(members_per_sample=2)
        │   │           Cuts both windows of every pair to the longest member of the batch: [B, 2T, ...].
        │   ├── MetricTracker.count_frames()     real frames of both members
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch() / augment_batch()   one augmentation draw per pair, for its 2T frames
        │   │   ├── forward_pass() -> split_pair_batch() -> SequenceComparator.forward()
        │   │   │   ├── encode_members()                 both members as one batch of windows -> encode_frames()
        │   │   │   ├── member_fusion.attend_across_members()   cross attention only
        │   │   │   ├── summarise_members()              the shared temporal module on both members as one batch
        │   │   │   ├── member_fusion.combine_member_vectors()
        │   │   │   ├── classification_head              class_logits [B, K]
        │   │   │   └── compute_classification_loss()
        │   │   └── compute_batch_metrics() / MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer() / validation pass
        │   └── average_trackers() + log_interval_metrics() + print_interval_summary()   (pairs and Frames Per Second)
        └── final save_model() + run_test_evaluation(), finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
└── the evaluation of the classification on pairs: the error and sample images show the before grid above the after
    grid, and predictions.csv names every pair.

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/sequence_comparison/configuration_synthetic_pairs.yaml`.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The comparison has the `TemporalModel` and `Fusion` sections on top of the sections of the classification. Its `sequence_settings` have no `window_stride`, since every member is one window.

**Example: the sections and the settings of the pairs**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, Fusion, ClassificationHead
Data keys after loading:  DALI, loading, image_settings, sequence_settings, label_mapping, data_augmentation, label_mode, label_dictionary
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'top_k': 1, 'target_precision': 0.99,
                           'save_predictions': True, 'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
TemporalModel:            {'type': 'mean_pooling', 'hidden_size': 128, 'number_layers': 1, 'number_heads': 4, 'kernel_size': 3, ...}
Fusion:                   {'type': 'difference', 'number_heads': 4, 'dropout_rate': 0.0, 'sequence_normalization_scope': 'member'}
ClassificationHead:       {'global_pooling': 'average', 'fully_connected': [64], 'dropout_rate': 0.2, 'output_mode': 'multi_class', ...}
```
Every setting of `Fusion` is required, even those its type does not read (`number_heads` and `dropout_rate` are read by the cross attention only, `sequence_normalization_scope` with the `sequence` normalization of the backbone only).

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'TemporalModel', 'Fusion', 'ClassificationHead']
number_iterations: '1e10'      <- still a string
```

### Step 1.4 — `build_label_dictionary()`
**Example: the label dictionary of the object loss pairs**
```text
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '/home/.../sequence_data/synthetic_object_loss_pairs/label_mapping.json'}
dict: {1: 0, 2: 1}
2 entries, 2 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_sequence_comparison_configuration()`: the checks of the sequence classification without the window stride (Step 1.5 of `sequence_classification_workflow.md`), then `check_fusion_configuration()`: the fusion type, the heads and the dropout of the cross attention, the normalization scope, and, for the cross attention, frame vectors (as wide as the last backbone block) that can be split between its heads.

**Example: output with `Fusion.type: "cross_attention"`, `number_heads: 3` and `sequence_normalization_scope: "frame"` in the YAML**
```text
-----------------------------------------------------------------------------------------------
Invalid Configuration File : file:///.../configuration_wrong.yaml
-----------------------------------------------------------------------------------------------
  - 'Fusion.sequence_normalization_scope' must be one of ['member', 'pair'], got 'frame'.
  - The channels of the last backbone block (64) must be a multiple of 'Fusion.number_heads' (3) for the cross attention.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="sequence_comparison") -> <class 'vision_training.trainers.sequence_comparison_trainer.SequenceComparisonTrainer'>
```


---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on `configurations/sequence_comparison/configuration_synthetic_pairs.yaml` (512 pixel frames, member windows of 8 frames, batch size 16), with the run folder redirected to a temporary folder. The whole construction took 0.2 s on CPU.

### Step 2.1 — `SequenceComparisonTrainer.__init__()`
The comparison trainer has no constructor of its own: it is the sequence classification trainer (`sequence_classification_workflow.md`, Step 2.1) with other class attributes. `LABEL_TYPE = "pair"` makes the loaders read pairs, and `MEMBERS_PER_SAMPLE = 2` tells the shared code that every sample holds two windows one after the other along its frame axis, the before window then the after window: the collate cuts both to the longest member of the batch (Step 3.5), the augmentation shuffles each on its own (Step 3.8) and the input viewer and the evaluation images draw both (Step 4.5). The losses, the metrics and the evaluation are those of the classification, a pair being classified as a whole.

**Example: the attributes after the construction**
```text
self.sequence_settings          = {'sequence_length': 8, 'short_sequences': 'pad', 'batch_padding': 'longest'}
self.LABEL_TYPE                 = 'pair'
self.MEMBERS_PER_SAMPLE         = 2
self.top_k                      = 1
self.save_predictions           = True
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
SAMPLE_NAME = 'Pairs'   COUNTS_FRAMES = True   TASK_NAME = 'sequence_comparison'
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
`build_sequence_comparison_sections()` gives the window (`T8`: frames per member window), the temporal module (`MEAN`, see `sequence_classification_workflow.md`, Step 2.4), then the fusion: `FCAT` (concatenation), `FDIF` (difference) or `FXA<heads>` (cross attention), followed by `SNP` when the backbone uses the `sequence` normalization over both members of a pair. The head and the classes end the name.

**Example output**
```text
CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_MEAN_FDIF_HAVG.64_MC2
```

### Step 2.5 — `DataAugmenter()`
**Example: the settings of the configuration**
```text
activated True   probability 0.4   independent_frames False   shuffle_frames {'activated': False, 'probability': 0.5}
6 of 16 types activated: random_left_right_flip, random_up_down_flip, random_90_rotation, brightness_contrast_saturation,
                         random_noise, random_colour_augmentation
```

### Step 2.6 — `setup_training_paths()`
**Example: the run folder after this step**
```text
CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_MEAN_FDIF_HAVG.64_MC2/
├── Tensorboard/
├── Weights/
└── training_configuration.yaml    4,409 bytes
```

### Step 2.7 — `build_dataloaders()`
The hook of the sequence classification, with `label_type="pair"`; the class names are the categories of the training manifest.

**Example output**
```text
class_names = ['loss_of_object', 'same_number_of_objects']
```

### Step 2.8 — `get_sequence_dataloaders()`
With `label_type="pair"` every split gets a `SequencePairDataset` (Step 2.9), and with `batch_padding: "longest"` the collate is `collate_longest_windows()` with `members_per_sample=2`, which cuts both windows of every pair to the longest member of the batch.

**Example: size of the three splits**
```text
train       2000 pairs -> 125 batches of 16 (one epoch = 125 iterations)
validation   200 pairs ->  13 batches of 16
test         200 pairs ->  13 batches of 16
collate     functools.partial(<function collate_longest_windows>, members_per_sample=2)
```

### Step 2.9 — `SequencePairDataset.__init__()` → `compute_member_frame_indices()`
It reads the manifest of the split: its `categories` and its `pairs`, each giving its `id`, its `name`, the ordered frames of its `before` and `after` members (paths relative to `data/`) and its `category_id`. A member is never cut into several windows: `compute_member_frame_indices()` keeps all the frames of a member that fits in `sequence_length` frames, and the middle frame of each of `sequence_length` equal parts of a longer member, so that the window covers the whole member in order. A pair is left out when its category is unmapped, when a member is empty, or, with `short_sequences: "skip"`, when a member is shorter than the window. The pairs are shuffled once and kept by identifier for the evaluation outputs.

**Example: a manifest pair, and the frames kept for a member window of 8 frames**
```text
labels.json:  {'id': 1, 'name': 'train_000000', 'before': ['train_000000/before/000000.png', ... 5 frames],
               'after': ['train_000000/after/000000.png', ...], 'category_id': 2}
              (2000 pairs, members of 2 to 6 frames; 2 categories {'id': 1, 'name': 'loss_of_object'}, ...)
compute_member_frame_indices( 3 frames) = [0, 1, 2]                    <- padded with 5 black frames
compute_member_frame_indices( 8 frames) = [0, 1, 2, 3, 4, 5, 6, 7]
compute_member_frame_indices(12 frames) = [0, 2, 3, 5, 6, 8, 9, 11]    <- the middle frame of 8 parts of 1.5 frames
compute_member_frame_indices(20 frames) = [1, 3, 6, 8, 11, 13, 16, 18]
```

**Example: the first pair after shuffling, and the sample the loader gets from it**
```text
pair:       {'identifier': 1983, 'pair_identifier': 1984, 'name': 'train_001983',
             'before': ['train_001983/before/000000.png', ... 4 frames], 'after': ['train_001983/after/000000.png', ... 3 frames], 'label': 1}
dataset[0]: images (16, 3, 512, 512) torch.uint8       the before window (8 frames) then the after window (8 frames)
            sequence_mask (16,) torch.bool [True, True, True, True, False, False, False, False,       <- 4 real before frames
                                            True, True, True, False, False, False, False, False]      <- 3 real after frames
            labels () torch.int64 1, sample_identifiers 1983
get_sample_name(1983) = 'train_001983'
describe_sample(1983) = {'pair': 'train_001983'}
get_class_counts()    = [992, 1008]
```

### Step 2.10 — `build_native_loader()`
**Example output**
```text
len(loader) = 125   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
The class of every test pair, so that the 8 sample pairs are shared by the 2 classes (4 each), each evenly spread over its sorted identifiers.

**Example output**
```text
get_test_sample_classes() -> 200 entries, e.g. {160: [0], 56: [0], 42: [0], 113: [0]}
sample_image_identifiers  -> [0, 3, 65, 68, 123, 137, 198, 199]
names                     -> test_000000, test_000003, test_000065, test_000068, test_000123, test_000137, test_000198, test_000199
```

### Step 2.12 — `build_model()`
The classification hook creates the `SequenceComparator` from the registry `TASK_MODEL_CLASSES["sequence_comparison"]` (Step 2.13) and sets its class weights (`get_class_counts()`: the pairs of every class, with `class_weighting: "balanced"`).

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          MaskedMeanPooling                0 parameters
model.member_fusion            MemberFusion                     0 parameters
model.classification_head      ClassificationHead          12,482 parameters
model.class_weights            buffer [2] torch.float32 [1.0, 1.0]
model.number_classes           2
```
With a mean pooling and a difference fusion, only the backbone and the head have weights.

### Step 2.13 — `SequenceComparator.__init__()`
The comparator classifies two sequences together (here, whether an object was lost between them). The shared backbone and global pooling turn every frame of both members into a vector, the member fusion lets the frames of each member attend to the other member (cross attention only), the shared temporal module summarises every member into one vector, the member fusion combines the two member vectors, and the classification head gives the logits of the pair. Both members go through the same weights, so that the difference of their vectors measures a change. It also reads the normalization of the backbone: with `sequence` normalization, the statistics are computed over the frames of each member (`sequence_normalization_scope: "member"`) or of both members of a pair (`"pair"`).

### Step 2.14 — `build_backbone()` + `build_global_pooling()`
The backbone and pooling of the sequence classification (`sequence_classification_workflow.md`, Step 2.14): 64 features per frame.

### Step 2.15 — `build_temporal_module()` → `MaskedMeanPooling`
The temporal module of the sequence classification (`sequence_classification_workflow.md`, Step 2.15), shared by both members. The mean pooling averages the vectors of the real frames of a member, ignoring their order.

**Example output**
```text
temporal module: MaskedMeanPooling()
output_features = 64
```

### Step 2.16 — `MemberFusion.__init__()`
It combines the two member vectors into the input of the head: `[before, after]` for `"concatenation"` (2 × 64 = 128 features), `[before, after, after - before]` for `"difference"` and `"cross_attention"` (3 × 64 = 192 features), the difference making the change between the members explicit. With `"cross_attention"` it also builds one multi-head attention layer (`Fusion.number_heads`, `Fusion.dropout_rate`) with a layer normalization, through which the frames of each member attend to the frames of the other member before the temporal module.

**Example output** (difference fusion, no weights)
```text
MemberFusion()   type difference   output_features 192
sequence_normalization = False   normalization_scope = member      (the backbone uses the batch normalization)
```

### Step 2.17 — `ClassificationHead.__init__()`
**Example output**
```text
ClassificationHead(
  (fully_connected_block): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=192, out_features=64, bias=True)
    (2): ReLU(inplace=True)
  )
  (dropout): Dropout(p=0.2, inplace=False)
  (classifier): Linear(in_features=64, out_features=2, bias=True)
)
```

### Step 2.18 — `initialise_pretrained_components()`
The backbone, the temporal module and the head can come from other runs. The fusion has no `pretrained` settings.

**Example: a run from scratch**
```text
pretrained_checkpoint_paths = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
```

### Step 2.19 — `build_optimizer()` + `get_tracked_metrics()` + `MetricTracker.__init__()`
**Example output**
```text
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
{'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'accuracy': 'Accuracy'}
MetricTracker: batch_size 16, validation_iteration_ratio 25, sample_name 'Pairs', counts_frames True
```

### Step 2.20 — `print_experiment_information()` (→ `describe_split()` → `count_split_pairs()`)
`count_split_pairs()` counts the pairs, the frames of both members and the kept pairs of every split, choosing the member frames as the dataset does; the kept pairs are the samples of an epoch.

**Example output**
```text
- Run Name                : CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_MEAN_FDIF_HAVG.64_MC2
...
- Train Data Folder       : file:///home/.../sequence_data/synthetic_object_loss_pairs/train (2000 pairs, 16096 frames, 2000 kept pairs of at most 2 x 8 frames)
- Validation Data Folder  : file:///home/.../sequence_data/synthetic_object_loss_pairs/validation (200 pairs, 1586 frames, 200 kept pairs of at most 2 x 8 frames)
- Test Data Folder        : file:///home/.../sequence_data/synthetic_object_loss_pairs/test (200 pairs, 1590 frames, 200 kept pairs of at most 2 x 8 frames)
- Iterations Per Epoch    : 125
- Batch Size              : 16
- Save And Evaluate Every : 500 iterations
- Save Only Every         : 250 iterations
...
- MLflow Tracking         : Off
```

### Step 2.21 — `print_summary()` / `print_model_size_summary()`
The comparator adds the Member Fusion panel between the temporal module and the head panels; the size panel lists the fusion on its own line.

**Example output** (shortened)
```text
╭────────────────────────────── Temporal Module ───────────────────────────────╮
│ Type               : mean_pooling                                            │
│ ...                                                                          │
│ Output Features    : 64                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────── Member Fusion ────────────────────────────────╮
│ Type               : difference                                              │
│ Number Heads       : 4                                                       │
│ Dropout Rate       : 0.0                                                     │
│ Sequence Normalization Scope : member                                        │
│ Output Features    : 192                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 183,648                       │
│ Temporal Module                                          │
│   - Parameters           : 0                             │
│ Member Fusion                                            │
│   - Parameters           : 0                             │
│ Classification Head                                      │
│   - Parameters           : 12,482                        │
│ Total Model                                              │
│   - Parameters           : 196,130                       │
│   - Estimated Size       : 0.79 MB                       │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.22 — `view_input_batches()` → `show_input_samples()`
With `diagnosis_modes.view_input_images`, the overridden `build_sample_frame_grid()` lays out the real frames of the before member above those of the after member, every frame captioned with its member (`0 before`, `0 after`), and the information panel gives the class of the pair, the real frames of both members (`Real frames : 3 + 2`) and, for the training pair, the augmentations and the order of the frames of each member (`Frame order : shuffled, showing frames 2, 0, 1 | 5, 4`). It needs a display, so it has no console output to show here.

### Step 2.23 — `restore_last_model()`
**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of `configuration_synthetic_pairs.yaml` shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the error images and the MLflow tracking on. Its run name is `CMP_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_MEAN_FDIF_HAVG.64_MC2`; with batches of 4 pairs, an epoch is 500 iterations.

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
trackers: {'start_time': ..., 'interval_batches': 0, 'total_loss': 0.0, 'classification_loss': 0.0, 'accuracy': 0.0, 'interval_frames': 0}
MLflow Run Created In The Experiment 'sequence_comparison' : 037372ca4aac463685e61176d80e825e (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.3 — `save_model()` → `save_checkpoint()`
**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   2,441,735 bytes      Weights/Iteration_6/model_000006.pt   2,441,735 bytes
Weights/Iteration_9/model_000009.pt   2,441,735 bytes      Weights/Iteration_12/model_000012.pt  2,441,735 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'mlflow_run_id',
 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
model_state: 86 tensors, the first 'class_weights', then 'backbone.convolution_blocks.0. ...'
mlflow_run_id: None in the checkpoints of iterations 3 and 6 (the MLflow run was not open yet), '037372ca4aac463685e61176d80e825e' in those of 9 and 12
```

### Step 3.4 — `run_test_evaluation()`
At iterations 6 and 12 (the final checkpoint), see Stage 4.

### Step 3.5 — `get_next_batch()` → `SequencePairDataset.__getitem__()` + `collate_longest_windows()`
`__getitem__()` reads the window of each member with `read_window_frames()` (Step 3.7 of `sequence_classification_workflow.md`) and concatenates them along the frame axis: the before window, then the after window, each of `sequence_length` frames with its real frames first. `collate_longest_windows(members_per_sample=2)` then cuts both windows of every pair to the number of real frames of the longest member of the batch, so that both members keep the same number of frames and can be split in two halves before the model.

**Example: the first batch of the run** (the longest member of the batch has 5 frames, so every window is cut from 8 to 5 frames)
```text
images               [4, 10, 3, 64, 64] torch.uint8      [B, 2 x 5, C, H, W]
sequence_mask        [4, 10] torch.bool
                     [[1, 1, 1, 0, 0,  1, 1, 1, 1, 0],        <- 3 before frames, 4 after frames
                      [1, 1, 0, 0, 0,  1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0,  1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1,  1, 1, 0, 0, 0]]        <- the longest member: 5 before frames
labels               [4] torch.int64     [1, 1, 1, 0]
sample_identifiers   [4] torch.int64     [1002, 878, 629, 915]
```

### Step 3.6 — `MetricTracker.count_frames()`
The real frames of both members of every pair are counted.

**Example: the frames of the first two intervals**
```text
interval_frames = 122 (iterations 1 to 4), 121 (iterations 5 to 8)      16 pairs per interval, about 7.6 frames per pair
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
The same function as for every task. `prepare_batch()` of the sequence classification moves the frames, labels and the `[B, 2T]` sequence mask to the device.

**Example: the first training batch**
```text
images [4, 10, 3, 64, 64] torch.uint8 -> torch.float32      first batch: classification_loss = 0.7293   accuracy = 0.25
```

### Step 3.8 — `augment_batch()`
The augmentation of the sequence classification (`sequence_classification_workflow.md`, Step 3.12): one draw per sample, so both members of a pair get the same crop, flip and colour change. `shuffle_frames()` shuffles each member window on its own (`get_window_length()` gives the frames of one member: `2T / MEMBERS_PER_SAMPLE`).

**Example: which pairs of the first batch were changed** (`probability: 0.4`)
```text
images: changed per sample [False, False, False, False]      <- no pair drawn for this batch
```

### Step 3.9 — `forward_pass()` → `split_pair_batch()` → `SequenceComparator.forward()`
`split_pair_batch()` splits the frames and the mask of the batch in two halves along the frame axis (`split_pair_members()`), the arguments of the comparator.

**Example: the first batch**
```text
before_images (4, 5, 3, 64, 64)   after_images (4, 5, 3, 64, 64)   before_mask (4, 5)   after_mask (4, 5)
before_mask [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
after_mask  [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]]
```

**Example: the dictionary returned in training mode**
```text
class_logits          [4, 2] torch.float32
class_probabilities   [4, 2] torch.float32   [[0.4927, 0.5073], [0.5466, 0.4534], [0.5240, 0.4760], [0.4938, 0.5062]]
classification_loss   0.7293 (scalar)                    <- close to ln 2 = 0.693
```

### Step 3.10 — `encode_members()` → `encode_frames()`
With masks (training and evaluation), both members have the same number of frames, so they are stacked as one batch of `2B` windows and encoded together by `encode_frames()` (`sequence_classification_workflow.md`, Step 3.14): the batch normalization statistics are computed over the frames of both members, and the padding frames get zero vectors. With the `sequence` normalization and the `"pair"` scope, the before and after members of a pair form one normalization group. Without masks (deployment, every frame real), each member is encoded on its own, so the two members may have different numbers of frames.

**Example: the first batch through the trained model**
```text
encode_members -> before (4, 5, 64), after (4, 5, 64)
zero vectors per member: before [2, 3, 2, 0], after [1, 3, 1, 3]      <- the padding frames of every member
```

### Step 3.11 — `member_fusion.attend_across_members()`
Only for `"cross_attention"`: the frames of each member attend to the real frames of the other member (both directions computed from the vectors before any update, with a residual connection and a layer normalization), then the padding frames are set back to zero vectors. With the difference fusion of this run, the vectors are returned unchanged.

### Step 3.12 — `summarise_members()`
The shared temporal module summarises the two members as one batch of `2B` windows with their masks, then the result is split back into the before and after vectors.

**Example: the first batch** (masked mean of the real frames of every member)
```text
summarise_members -> before (4, 64), after (4, 64)
```

### Step 3.13 — `member_fusion.combine_member_vectors()` + `ClassificationHead.forward()` + `compute_classification_loss()`
**Example: the first batch**
```text
combine_member_vectors -> (4, 192)      [before, after, after - before]
classification_head    -> (4, 2)        classification_loss = 0.7293
```

### Step 3.14 — `compute_batch_metrics()` + `MetricTracker.accumulate()` + `log_to_tensorboard()`
**Example: the Train curves**
```text
Total Loss   1: 0.7293, 2: 0.6944, 3: 0.6955, 4: 0.6334, 5: 0.6722, 6: 0.7452, 7: 0.6591, 8: 0.6701, 9: 0.6278, 10: 0.6093, 11: 0.5966, 12: 0.5344
Accuracy     1: 0.2500, 2: 0.5000, 3: 0.5000, 4: 1.0000, 5: 0.7500, 6: 0.0000, 7: 0.7500, 8: 0.7500, 9: 0.7500, 10: 0.7500, 11: 1.0000, 12: 1.0000
```

### Step 3.15 — `training_loss.backward()` + `step_optimizer()` + validation pass
**Example output**
```text
gradient norm: iteration 1: 1.7714   iteration 2: 1.1179   iteration 3: 1.4277        learning rate 0.001 (constant)
Tensorboard/Validation at iteration 10: Total Loss 0.6939, Classification Loss 0.6939, Accuracy 0.5000
```

### Step 3.16 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The header gives the pairs per second and the real frames of both members per second.

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 2.7526, 'classification_loss': 2.7526, 'accuracy': 2.25, 'interval_frames': 122}
after:  {'interval_batches': 4, 'total_loss': 0.6882, 'classification_loss': 0.6882, 'accuracy': 0.5625, 'interval_frames': 122}
MLflow, step 4: {'train/Total Loss': 0.6882, 'train/Classification Loss': 0.6882, 'train/Accuracy': 0.5625,
                 'train/throughput': 6.0327, 'train/interval_seconds': 2.6522, 'learning_rate': 0.001,
                 'train/frame_throughput': 45.9996}
```

**Example output: iterations 4 and 12**
```text
===============================================================================================================================================
|     Iteration: 4    |     Total Pairs: 16     |    Interval Pairs: 16    |   Throughput: 6.0 Pairs Per Second  |   46.0 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   0.6882
Moving Average of Train Classification Loss            :   0.6882
Moving Average of Train Accuracy                       :   0.5625
-----------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:02.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
===============================================================================================================================================
...
===============================================================================================================================================
|    Iteration: 12    |     Total Pairs: 48     |    Interval Pairs: 16    |  Throughput: 12.2 Pairs Per Second  |   96.8 Frames Per Second   |
-----------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   0.5920  -0.0947 Best
Moving Average of Valid Total Loss                     :   0.6939
Moving Average of Train Classification Loss            :   0.5920  -0.0947 Best
Moving Average of Valid Classification Loss            :   0.6939
Moving Average of Train Accuracy                       :   0.8750  +0.3125 Best
Moving Average of Valid Accuracy                       :   0.5000
-----------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:01.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
===============================================================================================================================================
```

### Step 3.17 — end of the loop
**Example output**
```text
Model successfully saved at iteration 12. Exiting Training.
Iteration 12 · 200 Test Pairs · 2026-10-10 07:10 ───────────────      <- the evaluation of the final checkpoint
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (200 pairs), with `per_class_report: True`, `number_sample_images: 8`, `top_k: 1`, `save_predictions: True`, `error_dumps` activated with `maximum_per_class_pair: 2`, and the MLflow tracking on. Both checkpoints predict `same_number_of_objects` for every test pair.

### Step 4.1 — `run_test_evaluation()`
The evaluation of the classification (see `classification_workflow.md`, Stage 4), every sample being a pair.

**Example output: the progress bar** (200 pairs, batch size 4)
```text
Test Evaluation Iteration 6: 100%|██████████| 50/50 [00:04<00:00, 10.45it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
**Example: the accumulator at the end of the loop**
```text
'probabilities': 50 tensors, the first [4, 2] torch.float64    'labels': 50 tensors, the first [4] torch.int64
'sample_identifiers': 50 tensors, the first [4] torch.int64    'errors_folder': PosixPath('.../Evaluation/Iteration_6/Errors')
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
`predict_batch()` runs the comparator on the split members with their masks and without labels. The checkpoint of iteration 6 was trained for 5 steps only, and every pair gets almost the same probabilities.

**Example: the predictions of the first test batch** (members of 4 frames at most)
```text
images [4, 8, 3, 64, 64]   sequence_mask [4, 8]      (2 x 4 frames)
class_logits          [4, 2] torch.float32
class_probabilities   [4, 2] torch.float32   [[0.4922, 0.5078], [0.4922, 0.5078], [0.4922, 0.5078], [0.4922, 0.5078]]
```

### Step 4.4 — `accumulate_evaluation_batch()`
The classification hook: the probabilities, labels and pair identifiers are kept on the CPU, and `dump_errors()` runs while the frames are still in the batch.

### Step 4.5 — `dump_errors()` → `reserve_error_image()` + `build_input_image()`
`build_input_image()` of the sequence classification draws the overridden `build_sample_frame_grid()`: the grid of the real frames of the before member (4 columns) above the grid of the after member, every frame captioned with its member. The file is named after the pair and the probability of the predicted class.

**Example: the error images of the checkpoint of iteration 12**
```text
build_input_image(sample_index=0) -> (144, 276, 3) uint8      a before row and an after row of 64 x 64 frames
Evaluation/Iteration_12/Errors/
└── loss_of_object-TO-same_number_of_objects/     test_000039_0.524.png   test_000073_0.524.png
report line: "Error Images : 2 in 1 class pairs, at most 2 per pair"
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
The frame grids of every fixed sample pair, under a caption bar giving the true class, the predicted class and its probability, in green when right and in red when wrong. The caption is wider than the grids here, so the image is padded on the right.

**Example: the sample images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Samples/
test_000000.png  test_000003.png  test_000065.png  test_000068.png  test_000123.png  test_000137.png  test_000198.png  test_000199.png
build_sample_image -> [168, 302, 3] torch.uint8
note of the report: "Sample Images : 8 fixed test pairs"
```

### Step 4.7 — `compute_evaluation_results()`
The results of the classification. With `top_k: 1`, the top-k accuracy is the accuracy.

**Example: the results of iteration 6**
```text
evaluated_samples = '200 Test Pairs'
metrics           = {'Accuracy': 0.46, 'Balanced Accuracy': 0.5, 'Macro F1': 0.3151, 'Top-1 Accuracy': 0.46}
notes             = ['Error Images : 2 in 1 class pairs, at most 2 per pair', 'Sample Images : 8 fixed test pairs']
headline_metric   = 'Accuracy'
files             = Confusion Matrix Table, Class Thresholds, Class Metrics, Confusion Matrix Image, Predictions, Error Images, Sample Images
class_scalars     = 6: Precision/loss_of_object, Recall/loss_of_object, F1/loss_of_object, Precision/same_number_of_objects, ...
```

### Step 4.8 — `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()`
**Example: `Evaluation/Iteration_6/confusion_matrix.csv`**
```text
true \ predicted,loss_of_object,same_number_of_objects
loss_of_object,0,108
same_number_of_objects,0,92
```

**Example: how the numbers of iteration 6 are obtained**
```text
accuracy                = 92 / 200                                  = 0.46
same_number_of_objects  precision = 92 / 200 = 0.46    recall = 1.0    F1 = 2 x 0.46 x 1.0 / (0.46 + 1.0) = 0.6301
loss_of_object          precision = 0 (never predicted)   recall = 0.0   F1 = 0.0
balanced accuracy       = (0.0 + 1.0) / 2                           = 0.5
macro F1                = (0.0 + 0.6301) / 2                        = 0.3151
```

### Step 4.9 — `find_class_thresholds()` + `write_class_metrics_file()`
**Example: `Evaluation/Iteration_6/class_metrics.csv` and `class_thresholds.json`**
```text
class_index,class,samples,precision,recall,f1,threshold_at_target_precision,recall_at_target_precision
0,loss_of_object,108,0.000000,0.000000,0.000000,0.492252,0.027778
1,same_number_of_objects,92,0.460000,1.000000,0.630137,,

{"target_precision": 0.99, "classes": {"loss_of_object": {"threshold": 0.4922524094581604, "recall": 0.027777777777777776},
                                       "same_number_of_objects": null}}
```

### Step 4.10 — `write_predictions_file()`   (only with `save_predictions`)
`describe_sample()` of the pair dataset names a pair by its name in the manifest.

**Example: `Evaluation/Iteration_12/predictions.csv`** (200 rows, 15,039 bytes; first rows)
```text
pair,true_class,predicted_class,correct,probability_loss_of_object,probability_same_number_of_objects
test_000073,loss_of_object,same_number_of_objects,0,0.476044,0.523956
test_000148,same_number_of_objects,same_number_of_objects,1,0.476041,0.523959
test_000023,same_number_of_objects,same_number_of_objects,1,0.476021,0.523979
```

### Step 4.11 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 200 Test Pairs · 2026-10-10 07:10 ───────────────────────────────────────────────────────────────────────
  Accuracy   0.4600   Balanced Accuracy   0.5000
  Macro F1   0.3151   Top-1 Accuracy      0.4600
  Error Images : 2 in 1 class pairs, at most 2 per pair
  Sample Images : 8 fixed test pairs

  Per Class Metrics
                                                                    Threshold    Recall
   Class                    Samples   Precision   Recall       F1     @ P0.99   @ P0.99
  ──────────────────────────────────────────────────────────────────────────────────────
   loss_of_object               108      0.0000   0.0000   0.0000      0.4760    0.1944
   same_number_of_objects        92      0.4600   1.0000   0.6301           —         —
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 0.99 is never reached.
  - Results : file:///.../Results.txt
  - Confusion Matrix Table : file:///.../Evaluation/Iteration_12/confusion_matrix.csv
  - ...
  - Sample Images : file:///.../Evaluation/Iteration_12/Samples
```

**Example: TensorBoard and MLflow after the two evaluations**
```text
Tensorboard/Test: 10 scalars (4 global + 6 per class), 9 images (8 samples + Confusion Matrix)
    scalar Accuracy          6: 0.4600, 12: 0.4600
MLflow: test/Accuracy, test/Balanced Accuracy, test/Macro F1, test/Top-1 Accuracy, test/best_Accuracy (steps 6, 12),
        6 test_class/<metric>/<class>, train/frame_throughput (steps 4, 8, 12)
```


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 40 pairs of the test split (its `labels.json` reduced, its `data/` folder linked).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5): set `run_name` to the comparison run, and `test_data_folder` to another labelled split (holding a pairs `labels.json` and `data/`) if needed.

### Step 5.2 — trainer construction
**Example: the split line of the launch information**
```text
- Test Data Folder        : file:///.../test_reduced (40 pairs, 339 frames, 40 kept pairs of at most 2 x 8 frames)
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'sequence_comparison' : 7f5220908af54c13aaf92eb8ec7f147b (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output** (every checkpoint predicts `same_number_of_objects` for the 40 pairs, so the four iterations tie and are listed in their evaluation order)
```text
Evaluation Summary Of 4 Checkpoints : file:///.../runs/evaluations/CMP_IN-KAR-64_..._MC2/Evaluation/evaluation_summary.csv
  - Best Checkpoint : iterations 12, 9, 6, 3 tie (Accuracy 0.5250)

iteration,Accuracy,Balanced Accuracy,Macro F1,Top-1 Accuracy,Precision/loss_of_object,Recall/loss_of_object,F1/loss_of_object, ...
12,0.525,0.5,0.34426229508196726,0.525,0.0,0.0,0.0,0.525,1.0,0.6885245901639345
9,0.525,0.5,0.34426229508196726,0.525,0.0,0.0,0.0,0.525,1.0,0.6885245901639345
6,0.525,0.5,0.34426229508196726,0.525,0.0,0.0,0.0,0.525,1.0,0.6885245901639345
3,0.525,0.5,0.34426229508196726,0.525,0.0,0.0,0.0,0.525,1.0,0.6885245901639345
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 9 on the 40 pairs**
```text
Evaluating : file:///.../runs/CMP_IN-KAR-64_..._MC2/Weights/Iteration_9/model_000009.pt
Iteration 9 · 40 Test Pairs · 2026-10-10 07:10 ─────────────────────────────────────────────────────────────────────────
  Accuracy   0.5250   Balanced Accuracy   0.5000
  Macro F1   0.3443   Top-1 Accuracy      0.5250
  Error Images : 2 in 1 class pairs, at most 2 per pair
  Sample Images : 8 fixed test pairs

  Per Class Metrics
                                                                    Threshold    Recall
   Class                    Samples   Precision   Recall       F1     @ P0.99   @ P0.99
  ──────────────────────────────────────────────────────────────────────────────────────
   loss_of_object                19      0.0000   0.0000   0.0000      0.4871    0.1053
   same_number_of_objects        21      0.5250   1.0000   0.6885      0.5129    0.1905
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 0.99 is never reached.
```
The sample pairs were chosen from the 40 pairs of this split, so they are not the same pairs as in Stage 4.
