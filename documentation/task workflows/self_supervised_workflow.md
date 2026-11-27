# Self-Supervised Workflow

This document follows one self-supervised run from `main()` to the evaluation of a checkpoint: which function runs when, what it receives and what it returns. The **Call Tree Overview** gives the whole run on one page. Each stage then has one step per function, with a short explanation and a **real** output. The steps shared by every task are explained in `classification_workflow.md`. Here they keep a short note; the steps specific to this task (the two views, the projection head, the InfoNCE and Barlow Twins losses, the k-NN evaluation, the effective rank) are explained in full.

The examples come from real runs on CPU:
- **Stages 1 and 2**: `configurations/self_supervised/configuration_synthetic_shapes.yaml` (9 shape classes, 256 pixel images, batch size 128, SimCLR), run folder in a temporary folder, 2 loading workers.
- **Stages 3, 4 and 5**: the same configuration shortened to 64 pixel images, batch size 8, 12 iterations, a checkpoint every 6 (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, the first 1000 test images, MLflow on, seeded with 0. A second run of 4 iterations with `ProjectionHead.method: barlow_twins` (no MLflow) gives the Barlow Twins outputs. Stage 5 evaluates on the first 200 test images. After 12 iterations the representations are almost random, so the metrics are poor on purpose.

Stages 1 and 2 date from 2026-10-24 (Steps 1.2 and 2.1 from 2026-11-12), Stages 3 to 5 from 2026-10-23. The grouped batches example of Step 2.10 was checked against the code only.

## Read This First

The self-supervised task (`task: "self_supervised"`, run name code `SSL`) trains the backbone without labels. Every image is drawn twice by `Data.data_augmentation` (with the hard-coded `SELF_SUPERVISED_AUGMENTATION_STRENGTHS`). Both views go through the backbone, a global pooling giving the **representation**, and a projection head giving the **embedding**. The loss of `ProjectionHead.method` pulls the two views of an image together: InfoNCE over the 2N views of the batch (`simclr`) or the Barlow Twins cross-correlation pushed to the identity (`barlow_twins`). The labels are read only by the test evaluation: a k-nearest neighbour classification of the test images among a fixed sample of training images, next to the effective rank that watches the collapse.

Choose it to pre-train the backbone on images too many to label, then load it into any task with `Backbone.pretrained` (the projection head is dropped). Choose CLS when the images are labelled. The task reuses the dataset, loaders and metrics code of CLS. The exported encoder (`image_encoder`) returns L2-normalized representations for similarity searches.

### The Task At A Glance

To find your way in the stages below, this block follows one image from its manifest to its two views, the loss, the k-NN evaluation and the export.

```text
<split>/labels.json  {"id": 1, "file_name": "train_000000.png", "category_id": 6}    label_mapping.json {"1": 0, ..}
<split>/data/train_000000.png (512 px) ─► ClassificationDataset(keep_unlabelled_samples=True): read_image, resize 256
   sample {images [3, 256, 256] uint8, labels: class index or -100 (read by the evaluation only), sample_identifiers}
DataLoader (batch 128, drop_last for train and validation) ─► prepare_batch: images [128, 3, 256, 256] float32
forward_pass ─► DataAugmenter.create_views (float32, autocast off): first and second views [128, 3, 256, 256]
   every activated type drawn per image and per view; random_solarize and random_black_out on the second view only
SelfSupervisedModel.encode() of the 2N = 256 views in one pass (the batch normalization sees both views):
   Backbone: InputNormalizer x / 255 - 0.5, 6 ConvolutionBlocks (SE, shortcuts in blocks 3 and 5) ─► [256, 128, 8, 8]
   ─► global_pooling average ─► representations [256, 128] ─► ProjectionHead: Linear 512, BatchNorm, ReLU, Linear 128
   ─► embeddings [256, 128], split into the first [128, 128] and the second [128, 128] views
loss: simclr: info_nce_loss = cross entropy of every view against the 2N - 1 others, cosine / 0.2, partner at i ± N
      barlow_twins: barlow_twins_loss = Σ (C_ii - 1)² + 0.005 · Σ_i≠j C_ij², C the [128, 128] cross-correlation
batch metrics: positive_pair_accuracy (or diagonal and off-diagonal correlations), embedding_standard_deviation
   ─► TensorBoard Train every iteration, Validation every 5 (validation_iteration_ratio), console every 10
every weight_saving_iterations: test images encoded without views ─► representations [50000, 128] kept on the CPU
   ─► reference training images (50 per class at most, fixed) encoded with the same weights
   ─► k-NN of the labelled test images among them (20 neighbours, weights exp(cosine / 0.07)): k-NN Accuracy,
      Balanced Accuracy, Macro F1, per class precision, recall with its 95 % interval, F1; Effective Rank,
      Representation Standard Deviation
   ─► Results.txt block, Tensorboard/Test, Evaluation/Iteration_<n>/ (class_metrics.csv, Samples/)
export: InferenceEncoder (image_encoder), images [B, 3, 256, 256] ─► representations [B, 128], L2-normalized
transfer: Backbone.pretrained of any task loads the backbone; the projection head is dropped
```

### Files To Know

Should you need to change one part of the task, this table gives the file holding it and the functions to read first.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/self_supervised_trainer.py` | `SelfSupervisedTrainer`: `build_reference_dataloader`, `forward_pass`, `predict_batch`, `compute_nearest_neighbour_results`, `compute_evaluation_results`, `build_sample_image` |
| Shared training loop | `vision_training/trainers/base_trainer.py` | `BaseTrainer`: `run_training_iterations`, `run_model_iteration`, `run_test_evaluation` |
| Dataset and loaders | `vision_training/data/classification/classification_dataset.py`, `vision_training/data/loaders.py`, `vision_training/data/grouped_batch_sampler.py` | `ClassificationDataset(keep_unlabelled_samples=True)`, `get_classification_dataloaders(self_supervised=True)`, `GroupedBatchSampler` |
| Views | `vision_training/data/augmentations.py` | `DataAugmenter.create_views`, `SELF_SUPERVISED_AUGMENTATION_STRENGTHS`, `SECOND_VIEW_ONLY_TYPES` |
| Model and head | `vision_training/models/self_supervised/self_supervised_model.py`, `vision_training/models/self_supervised/projection_head.py` | `SelfSupervisedModel` (`represent`, `encode`, `compute_method_loss`, `forward`), `ProjectionHead` |
| Losses | `vision_training/losses/self_supervised_losses.py` | `compute_info_nce_loss`, `compute_barlow_twins_loss`, `compute_embedding_standard_deviation` |
| Metrics | `vision_training/evaluation/self_supervised/representation_metrics.py`, `vision_training/evaluation/recall_intervals.py` | `select_reference_samples`, `compute_nearest_neighbour_predictions`, `compute_effective_rank`, `compute_clopper_pearson_intervals` |
| Configuration checks, run name | `vision_training/configuration_processing/self_supervised_checks.py`, `vision_training/configuration_processing/run_name.py` | `check_self_supervised_configuration`, `check_projection_head_configuration`, `build_self_supervised_sections` |
| Synthetic data | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset`, `build_classification_labels`, `build_shape_count_labels` |
| Export | `vision_training/models/self_supervised/inference_encoder.py`, `vision_training/deployment/export_model.py`, `vision_training/deployment/export_size_estimates.py` | `InferenceEncoder` (model `image_encoder`, input `images`, output `representations`), `export_experiment`, `estimate_export_sizes` |
| Standalone evaluation | `scripts/evaluation/evaluate_checkpoint.py`, `vision_training/evaluation/checkpoint_evaluation.py` | `evaluate_checkpoints`, `summarise_checkpoint_evaluations` |

### Run It

In order to train and evaluate the task on the synthetic shapes from scratch, set the values of the main blocks and of the configuration, then run these commands from the repository root.

```bash
# Create the synthetic dataset of configuration_synthetic_shapes.yaml (500,000 / 50,000 / 50,000 images of 512 pixels);
# --task=self_supervised creates the counted shapes of the configuration_counted_shapes_*.yaml files instead
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=classification --output_folder="$HOME/Data/classifier_data/synthetic_shapes"

# Train: set experiment_folder, the three data folders and Data.label_mapping.mapping_file_path first
uv run python -m vision_training.main --configuration_path="vision_training/configurations/self_supervised/configuration_synthetic_shapes.yaml"

# Evaluate checkpoints: set run_name, experiment_folder, iteration, test_data_folder, batch_size and track_evaluation
# in the main block first; the evaluation is written in <run>/Checkpoint_Evaluation/ (Step 5.1)
uv run python -m scripts.evaluation.evaluate_checkpoint
```

With `diagnosis_modes.view_input_images: True`, the training command shows the two views of the training images instead (needs a display). The dataset on disk, generated on 2026-11-18, holds 35 classes (a k-NN chance level of 1/35). The examples of this document have the 9 classes of the older datasets.

### Configuration Keys To Know First

Before changing the synthetic configuration, this table lists the keys that change the views, the loss or the evaluation the most.

| Key | Synthetic value | Effect |
|---|---|---|
| `ProjectionHead.method` | `"simclr"` | InfoNCE (`SIMCLR-T<temperature>`) or `"barlow_twins"` (`BT-OD<weight>`); selects the batch metrics |
| `ProjectionHead.simclr.temperature`, `ProjectionHead.barlow_twins.off_diagonal_weight` | `0.2`, `0.005` | divides the InfoNCE cosines; weighs the Barlow Twins off-diagonal terms; both checked > 0 |
| `ProjectionHead.global_pooling` | `"average"` | the 128-value representation of the k-NN and the export; `"flatten"` gives 8,192 values tied to the image size |
| `ProjectionHead.fully_connected` | `[512, 128]` | hidden layers, then the embedding size (`PAVG.512.128`) |
| `ProjectionHead.fully_connected_normalization` | `"batch"` | `PBN`; `"layer"` gives `PLN`; `"none"` when the list holds the embedding layer only |
| `Data.data_augmentation.activated` | `True` | must be True; its global `probability` is ignored |
| `Data.data_augmentation.types.random_crop` | `activated: True, probability: 1.0` | crop of 20 % to 100 % of the area, aspect ratio x1/2 to x2; probability must be 1.0 |
| `ExperimentConfiguration.batch_size` | `128` | N: every view is compared with the 2N - 1 = 255 others; at least 2 |
| `ExperimentConfiguration.evaluation.number_neighbours` | `20` | k of the k-NN |
| `ExperimentConfiguration.evaluation.number_reference_images_per_class` | `50` | reference training images per class, the same at every evaluation |
| `Data.grouped_batches.activated` | `False` | True builds batches from blocks of `images_per_block` (32) images sharing `groups[block_key]` (`"camera"`), the hard negatives |

### What A Run Writes

If you are looking for an output of a run, this table gives where it is written.

| Output | Contents |
|---|---|
| `Results.txt` | one block per checkpoint: `k-NN Accuracy` (headline), `k-NN Balanced Accuracy`, `k-NN Macro F1`, `Effective Rank`, `Representation Standard Deviation`, the k-NN note, and with `per_class_report` the tables `Per Class k-NN Metrics` and `Per Class k-NN Recall` (Step 4.12). Without labelled test image, only the two collapse metrics, `Effective Rank` becoming the headline. |
| `Evaluation/Iteration_<n>/` | `class_metrics.csv` (`class_index,class,samples,precision,recall,recall_lower,recall_upper,f1`) and `Samples/<sample>.png` (Step 4.6); `EMA/` for the moving average. No confusion matrix, threshold, prediction file, curve or error image. |
| `evaluation_curves.png`, `Evaluation/evaluation_summary.csv` | the curve of the headline metric and one row per evaluated iteration |
| `Input_Samples/` | the two views of the first 10 training and validation images, the first 10 test images whole (Step 2.21) |
| `Tensorboard/Train`, `Validation`, `Test` | the batch metrics of the method (Step 2.18), `Learning Rate`; in Test the 5 global metrics, the per-class k-NN scalars and the sample images |
| `Weights/Iteration_<n>/model_<n>.pt` | the whole model, projection head included, to resume; only the backbone initialises other runs |
| MLflow | `test/<metric>`, `test_class/k-NN <Metric>/<class>`, `test/best_k-NN Accuracy` with the tag `best_iteration`, and the files above |

### Pitfalls

Suppose a self-supervised run behaves unexpectedly: these behaviours are the first to check.

- **Early checkpoints look collapsed.** The evaluation uses batch normalization running statistics (momentum 0.1, `BATCH_NORMALIZATION_MOMENTUM`), far from the data after a few iterations. Effective rank near 1, views similarity 1.000, validation InfoNCE of exactly ln(2N - 1) (Step 4.9).
- **A cue shared by both views.** The colour of a shape (colour types off) or its position (crops too large) lets the loss fall without learning. A `Positive Pair Accuracy` near 100 % long before the k-NN accuracy rises shows it ([tasks reference, Self-Supervised Learning](../reference/tasks_reference.md#self-supervised-learning-ssl)).
- **A crop can miss a small object.** Crops go down to a fifth of the area. Both views of `test_000006.png` (a circle near the left border) are crops of 22 % and 24 % to the right of the circle: the loss pulls two backgrounds together.
- **Views, batch size, learning rate and grouped batches give no run name token.** Changing only these resumes the previous folder unless `run_name_suffix` is set.
- **Blocks 2, 4 and 6 add no residual shortcut**, and a transferred backbone keeps this layout (Step 2.14).
- **The k-NN only predicts the classes of its reference images.** A test class without labelled training image is never right (yellow note). The reference images are encoded again at every evaluation: 1,750 for the 35 classes of the synthetic configuration, about 218,000 for the ImageNet-22k example (`configuration_example.yaml`).

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

To see the whole run on one page before reading the steps, this tree gives every function in the order it runs.

```text
main()                                          reads --configuration_path, builds the trainer, trains
├── load_experiment_configuration()             load_configuration(), check_configuration_keys(),
│                                               build_label_dictionary(), check_configuration()
│                                               -> check_self_supervised_configuration()
├── get_trainer_class(task)  -> SelfSupervisedTrainer
├── SelfSupervisedTrainer(...)  [STAGE 2]       then build_reference_dataloader() (k-NN reference images)
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() / DataAugmenter(self_supervised=True) / setup_training_paths()
│       ├── build_dataloaders() -> get_classification_dataloaders(self_supervised=True)
│       │       ClassificationDataset(keep_unlabelled_samples=True), build_grouped_batch_sampler(), build_native_loader()
│       ├── choose_sample_identifiers()          8 fixed test images
│       ├── build_model() -> SelfSupervisedModel: build_backbone(), global_pooling, ProjectionHead()
│       ├── initialise_pretrained_components() / build_optimizer() / MetricTracker.__init__()
│       ├── print_experiment_information() / print_summary() / print_export_size_summary()
│       ├── view_input_batches()                only with view_input_images, then exits
│       └── restore_last_model()
└── trainer.run_training_loop() -> run_training_iterations()  [STAGE 3]
    └── for each training_iteration up to number_iterations:
        ├── get_next_batch() -> ClassificationDataset.__getitem__()
        ├── run_model_iteration(): prepare_batch(), augment_batch() (unchanged), forward_pass()
        │       DataAugmenter.create_views() -> SelfSupervisedModel.forward(): encode(), compute_method_loss(),
        │       compute_embedding_standard_deviation()
        ├── training_loss.backward() / step_optimizer()
        ├── save_model() / validation pass / print_interval_summary()   on their iterations
        └── run_test_evaluation()                on multiples of weight_saving_iterations  -> [STAGE 4]

run_test_evaluation(iteration)  [STAGE 4]
├── for each test batch: predict_batch() (no second view), accumulate_evaluation_batch(), save_sample_images()
├── compute_evaluation_results(): compute_effective_rank(), compute_nearest_neighbour_results()
│       encode_reference_images(), compute_nearest_neighbour_predictions(), confusion matrix, recall intervals
└── report_evaluation() -> report_evaluation_results() + track_evaluation() + update_evaluation_curves()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── SelfSupervisedTrainer writing in <run>/Checkpoint_Evaluation/, evaluate_checkpoints() -> run_test_evaluation()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs of the loader on `configurations/self_supervised/configuration_synthetic_shapes.yaml`.

### Step 1.1 — `main()`
The entry point of every task (`classification_workflow.md`, Step 1.1). `parse_configuration_path()` reads `--configuration_path`. `main()` loads the file, looks up the trainer of `ExperimentConfiguration.task` and starts its loop. Loading and construction run inside `read_manifests_once()` (`vision_training/data/manifest_cache.py`), so every `labels.json` is parsed once per launch.

```bash
# Launch the self-supervised run
python -m vision_training.main --configuration_path vision_training/configurations/self_supervised/configuration_synthetic_shapes.yaml
```

### Step 1.2 — `load_experiment_configuration()`
It splits the YAML into the experiment and model dictionaries and converts the types (`1e10` is read as a string). It adds `tracking_root`, `run_name` (None, built in Step 2.4) and `standalone_evaluation_folder` (None for a training, Step 5.2). It adds `label_mode` and `label_dictionary` to `Data` (Step 1.4) and runs the checks (Step 1.5). The task has one section of its own, `ProjectionHead`.

**Example: the sections and the converted values**
```text
raw keys:               ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'ProjectionHead']
raw number_iterations:  '1e10'                       -> after loading: 10000000000
model keys:             ['Optimization', 'Data', 'Backbone', 'ProjectionHead']
task:                   'self_supervised'
learning_rate: 0.0005   batch_size: 128   information_dump: 10   weight_saving_iterations: 250
ProjectionHead:         {'global_pooling': 'average', 'fully_connected': [512, 128], 'fully_connected_normalization': 'batch',
                         'method': 'simclr', 'simclr': {'temperature': 0.2},
                         'barlow_twins': {'off_diagonal_weight': 0.005}}
```

### Step 1.3 — `load_configuration()`
Plain YAML reader, as for every task: no conversion and no check.

**Example output**
```text
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '.../classifier_data/synthetic_shapes/label_mapping.json'}
```

### Step 1.4 — `build_label_dictionary()`
It reads the mapping file of `Data.label_mapping` (`vision_training/data/label_mapping.py`). There is no background class. The training never reads the labels: they only name the classes of the k-NN evaluation. A configuration read from a run folder takes the classes of its `label_dictionary.json` instead (`read_run_label_dictionary()`, Step 5.2).

**Example output**
```text
label_mode:       multi_class
label_dictionary: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_self_supervised_configuration()` (`vision_training/configuration_processing/self_supervised_checks.py`). This table lists what it checks.

| Part | Checks |
|---|---|
| data | `multi_class` label mode, classes 0 to K-1; `Data.data_augmentation.activated` True (its types draw the views); `random_crop` at probability 1.0 when activated; no sequence settings (`check_image_task_sequence_settings()`) |
| grouped batches | `check_grouped_batches_configuration()`: `activated` a boolean, `block_key` a non-empty string, `batch_key` null or a non-empty string, `images_per_block` at least 2 and dividing the batch size |
| head | `check_projection_head_configuration()`: pooling, layers (positive integers, embedding size last), normalization (`"none"` with the embedding layer only), method (`simclr` or `barlow_twins`), `simclr.temperature` and `barlow_twins.off_diagonal_weight` above 0 whatever the method |
| batch and evaluation | batch size of at least 2; `number_neighbours` and `number_reference_images_per_class` positive integers (`check_self_supervised_evaluation_configuration()`) |

Every error is listed before the program stops.

**Example output: a configuration with errors** (shortened)
```text
Invalid Configuration File : configuration_wrong.yaml
  - 'Data.data_augmentation.types.random_crop.probability' must be 1.0 for the self-supervised task when 'Data.data_augmentation.types.random_crop.activated' is True, got 0.5: the random resized crop is drawn for every view, or turned off to keep the whole images.
  - 'ExperimentConfiguration.batch_size' must be at least 2 for the self-supervised task: the losses compare the views of the images of a batch with each other.
```

Before these checks, `check_removed_settings()` refuses removed sections such as `Data.view_augmentation` (its probabilities go to the types of `Data.data_augmentation`) and the fixed settings (`resume_training`, `mixed_precision`, `Data.loading`...). `check_unknown_keys()` then refuses keys the task does not read, such as a leftover `ClassificationHead` section.


### Step 1.6 — `get_trainer_class()`
**Example output**
```text
<class 'vision_training.trainers.self_supervised_trainer.SelfSupervisedTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs of building the trainer on `configurations/self_supervised/configuration_synthetic_shapes.yaml` (256 pixels, batch size 128, 2 loading workers), run folder in a temporary folder. The construction took 14.3 s on CPU.

### Step 2.1 — `SelfSupervisedTrainer.__init__()`
Before `BaseTrainer.__init__()`, it stores `self.method` (`ProjectionHead.method`), `self.number_neighbours` and `self.number_reference_images_per_class` (from `evaluation`). After it, `build_reference_dataloader()` chooses the reference images of the k-NN with `select_reference_samples()`: in every class, the labelled training images of smallest file name MD5 hash, so the same images whatever the shuffling. Their loader reads them in order, without augmentation. A labelled test split next to a training split without labels stops with a `ValueError`. A test class without labelled training image is announced in yellow. The class attributes `DROPS_LAST_TRAINING_BATCH` (Step 2.19) and `SELF_SUPERVISED_AUGMENTATION` (Step 2.5) inform the shared code.

**Example output**
```text
method: simclr   number_neighbours: 20   number_reference_images_per_class: 50
TASK_NAME: self_supervised   SAMPLE_NAME: Images   COUNTS_FRAMES: False   DROPS_LAST_TRAINING_BATCH: True
SELF_SUPERVISED_AUGMENTATION: True
reference images: 450 in 4 batches of 128
reference_notes: ['k-NN : 20 neighbours among 450 labelled training images of 9 classes (50 per class at most)']
```

### Step 2.2 — `BaseTrainer.__init__()`
The constructor of every task (`classification_workflow.md`, Step 2.2).

**Example: the settings it stores**
```text
device: cpu   per_class_report: True   number_sample_images: 8   input_image_size: 256   batch_size: 128
start_iteration: 1   dtype: torch.float32
```

### Step 2.3 — `get_device()`
**Example output**
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The run name gives the task (`SSL`), the input, the backbone, then the head of `build_self_supervised_sections()`: `PAVG` (average pooling), the head layers, `PBN` (batch normalization) and the method, `SIMCLR-T<temperature>` or `BT-OD<off-diagonal weight>`. There is no class count. A `run_name_suffix` such as `_v1` ends the name; it is needed to start a new folder when only the views, the batch size or the learning rate change.

**Example output**
```text
SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2
```

### Step 2.5 — `DataAugmenter()`
The `DataAugmenter` of every task (`vision_training/data/augmentations.py`), here with `self_supervised=True`. The types keep the flags and probabilities of the configuration and take the hard-coded `SELF_SUPERVISED_AUGMENTATION_STRENGTHS` (a crop of 20 % to 100 % of the area, a colour jitter of x0.6 to x1.2, a hue turned by up to 36 degrees...; table in the [training reference, Data Augmentation](../reference/training_reference.md#data-augmentation)). Its `create_views(images)` draws the two views of a batch (Step 3.12):
- every activated type is drawn for each view of each image with its own probability; the global `probability` is ignored;
- `random_solarize` and `random_black_out` (`SECOND_VIEW_ONLY_TYPES`) are drawn for the second view only, as in BYOL and Barlow Twins;
- the views are computed in float32 with the autocast off, and returned with the augmentation report of every view.

**Example: the settings of the augmenter**
```text
activated: True   probability: 1.0 (ignored by the views)   self_supervised: True
18 of 28 types activated: {'random_crop': 1.0, 'random_left_right_flip': 0.5, 'random_up_down_flip': 0.5,
    'random_rotation': 0.5, 'random_perspective': 0.5, 'brightness_contrast_saturation': 0.8, 'random_hue': 0.8,
    'random_noise': 0.2, 'gray_scale_augment': 0.2, 'random_gamma': 0.3, 'random_linear_illumination': 0.3,
    'random_plasma_shadow': 0.2, 'random_blur': 0.5, 'random_motion_blur': 0.1, 'random_haze': 0.3,
    'random_solarize': 0.2, 'random_black_out': 0.3, 'random_erasing': 0.25}
second view only: ['random_solarize', 'random_black_out']
crop_area_range: (0.2, 1.0)   brightness_factor_range: (0.6, 1.2)   maximum_hue_shift: 0.1   ...
```

### Step 2.6 — `setup_training_paths()`
It creates the run folder, copies the YAML file as written and writes `label_dictionary.json`; the data record follows once the loaders exist. Before it, `restore_run_label_dictionary()` and `check_resumed_data()` handle a resumed run, as for every task (`classification_workflow.md`, Step 2.6).

**Example: the run folder after the construction** (the configuration file was 28,255 bytes on 2026-10-24 and 16,041 bytes on 2026-11-20)
```text
Tensorboard/                    folder
Weights/                        folder
data.json                       17,002,746 bytes
label_dictionary.json           110 bytes
training_configuration.yaml     28,255 bytes
```

### Step 2.7 — `build_dataloaders()`
It calls `get_classification_dataloaders(..., self_supervised=True)` (Step 2.8), merges the categories of the three manifests by identifier and stores `class_names` in class index order. The validation loader is built only when the run reads it (`classification_workflow.md`, Step 2.7); otherwise the validation categories come from its `labels.json`.

**Example output**
```text
class_names: ['square', 'circle', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'plus_sign', 'star', 'octagon']
```

### Step 2.8 — `get_classification_dataloaders(self_supervised=True)`

The classification loaders (`vision_training/data/loaders.py`) with three differences: the mode is always `multi_class`, the datasets keep unlabelled images (Step 2.9), and the training and validation loaders drop their last incomplete batch. The unlabelled images of every split are printed in one yellow block, the grouped batches in one blue block.

With `Data.grouped_batches` on, `build_grouped_batch_sampler()` builds a `GroupedBatchSampler` for the training and validation splits. At every pass, images sharing a (batch value, block value) are cut into blocks of `images_per_block`, joined into full batches; leftovers form random batches. A pass still gives `floor(N / batch_size)` batches. A split without the keys keeps random batches with a yellow warning; a split where only some samples hold them stops the launch.

This table shows the three splits of the synthetic dataset.

| Split | Images | Batches | drop_last |
|---|---|---|---|
| train | 500000 | 3906 | True |
| validation | 50000 | 390 | True |
| test | 50000 | 391 | False |

### Step 2.9 — `ClassificationDataset.__init__()`
The classification dataset (`vision_training/data/classification/classification_dataset.py`) with `keep_unlabelled_samples=True`: an image without a mapped category is kept with the label -100 (`IGNORED_LABEL_INDEX`) and counted in `number_unlabelled_samples`. Every record keeps the optional `groups` of its sample. The kept samples are shuffled once, then packed into `SerializedRecords` (`classification_workflow.md`, Step 2.9). `samples.find_record()` finds a record by its identifier (used by the input viewer, Step 2.21).

**Example: the manifest, the first kept sample and `dataset[0]`**
```text
manifest keys: ['info', 'categories', 'samples']   500000 samples, 9 categories
manifest sample:  {'id': 1, 'file_name': 'train_000000.png', 'category_id': 6}
first kept:       {'identifier': 145458, 'file_name': 'train_145457.png', 'class_indices': [5], 'groups': {}}
dataset[0]:       images [3, 256, 256] torch.uint8 in [14, 231]   labels () torch.int64 = 5   sample_identifiers 145458
```

### Step 2.10 — `build_native_loader()`

The torch DataLoader of the classification: 3906 training batches of 128, `drop_last` True. With grouped batches, it receives the `GroupedBatchSampler` as `batch_sampler`, so it reads `batch_size None` and `drop_last False`, its length being the number of batches of the sampler (2 for 10 images in two cameras, batches of 4, blocks of 2). The examples were recorded with 2 workers; every loader now has 1 worker and pins only on a discrete GPU.

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
`get_test_sample_classes()` gives the class of every test image (unlabelled images form their own group), so `choose_sample_identifiers()` spreads the 8 sample images over the classes.

**Example output**
```text
sample identifiers:  {1, 2, 6, 7, 9, 11, 13, 15}
                     test_000000, test_000001, test_000005, test_000006, test_000008, test_000010, test_000012, test_000014
                     classes [7], [6], [5], [1], [0], [3], [2], [4]                       (no octagon among the 8)
```

### Step 2.12 — `build_model()`
Creates the `SelfSupervisedModel` from the model registry (`TASK_MODEL_CLASSES["self_supervised"]`).

### Step 2.13 — `SelfSupervisedModel.__init__()`
The model (`vision_training/models/self_supervised/self_supervised_model.py`) has three parts: the `backbone` (Step 2.14), the `global_pooling` giving the **representation** kept after training, and the `projection_head` giving the **embedding** of the loss, thrown away after training. Its methods:
- `represent()`: backbone then pooling, `[N, 128]`.
- `encode()`: returns `representations` and `embeddings`.
- `forward(input_tensor, second_view_tensor=None)`: without second view it only encodes (Stage 4). With it, it encodes both views as one batch of 2N, splits the embeddings, calls `compute_method_loss()` and adds the `embedding_standard_deviation` of the first views (Step 3.15).

This table lists the components and their parameters.

| Component | Contents | Parameters |
|---|---|---|
| backbone | Backbone | 438,368 |
| global_pooling | Sequential(AdaptiveAvgPool2d(1), Flatten) | 0 |
| projection_head | FullyConnectedBlock(Linear 128→512, FeatureBatchNormalization(512, momentum 0.1), ReLU) + embedding_layer Linear 512→128 | 132,736 |
| Total | representation size 128, embedding size 128 | 571,104 |

### Step 2.14 — `build_backbone()`
The backbone of every task (`classification_workflow.md`, Step 2.14). `Backbone.print_summary()` traces a zero image and lists the modules of each block. Blocks 2, 4 and 6 change the channel count with 2 layers, so they add no residual shortcut (`count_residual_additions()` gives 0, 0, 1, 0, 1, 0).

**Example: the block output shapes** (256 pixels, from the Backbone Architecture panel)
```text
Block 1 [1, 16, 128, 128]   Block 2 [1, 32, 64, 64]   Block 3 [1, 32, 32, 32] (Residual Shortcuts)
Block 4 [1, 64, 16, 16]     Block 5 [1, 64, 8, 8] (Residual Shortcuts)          Block 6 [1, 128, 8, 8]
every block: Squeeze-and-Excitation   Final Backbone output shape: [1, 128, 8, 8]
```

### Step 2.15 — `ProjectionHead.__init__()` and its panel
The head (`vision_training/models/self_supervised/projection_head.py`) is a `FullyConnectedBlock` (linear, normalization, ReLU for every size of `fully_connected` but the last), then a plain linear `embedding_layer` to the last size, with no normalization nor activation. `print_projection_head_summary()` adds its panel.

**Example output**
```text
╭────────────────────────────── Projection Head ───────────────────────────────╮
│ Global Pooling       : average                                               │
│ Representation Size  : 128                                                   │
│ Layers               : [512, 128]                                            │
│ Normalization        : batch (momentum 0.1)                                  │
│ Embedding Size       : 128                                                   │
│ Method               : simclr (temperature 0.2)                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 2.16 — `initialise_pretrained_components()`
Only the `Backbone` can come from another run; the projection head is always trained from scratch.

**Example output**
```text
pretrained settings: {'Backbone': None}
```

### Step 2.17 — `build_optimizer()`
**Example output**
```text
Adam, 1 parameter group, learning rate 0.0005, weight decay 0.0   exponential moving average: None
```

### Step 2.18 — `get_tracked_metrics()` + `MetricTracker.__init__()`
The tracked metrics depend on the method (`METHOD_TRACKED_METRICS`): its loss, its batch metrics and the embedding standard deviation. `get_percent_tracked_metrics()` (`METHOD_PERCENT_TRACKED_METRICS`) gives the rates printed as percentages: the positive pair accuracy of SimCLR, none for Barlow Twins.

**Example: the tracked metrics of both methods**
```text
simclr:       {'total_loss': 'Total Loss', 'info_nce_loss': 'InfoNCE Loss', 'positive_pair_accuracy': 'Positive Pair Accuracy',
               'embedding_standard_deviation': 'Embedding Standard Deviation'}
barlow_twins: {'total_loss': 'Total Loss', 'barlow_twins_loss': 'Barlow Twins Loss', 'cross_correlation_diagonal': 'Diagonal Correlation',
               'cross_correlation_off_diagonal': 'Off-Diagonal Correlation', 'embedding_standard_deviation': 'Embedding Standard Deviation'}
get_percent_tracked_metrics(): simclr ['positive_pair_accuracy'], barlow_twins []
```

### Step 2.19 — `print_experiment_information()` (→ `describe_split()` → `count_split_samples()`)
`describe_split()` counts the images and categories of every manifest. The 3906 iterations per epoch are 500000 / 128 rounded down, since `DROPS_LAST_TRAINING_BATCH` is set (`classification_workflow.md`, Step 2.10).

**Example output** (shortened)
```text
=== Experiment Launch Information ===
- Run Name                : SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2
- Train Data Folder       : train (500000 images, 9 categories)
- Validation Data Folder  : validation (50000 images, 9 categories)
- Test Data Folder        : test (50000 images, 9 categories)
- Iterations Per Epoch    : 3906
- Batch Size              : 128
- Save And Evaluate Every : 250 iterations
- Save Only Every         : 125 iterations
- Learning Rate           : 0.0005
- Validation Iter. Ratio  : 5
...
```

### Step 2.20 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The backbone panel (Step 2.14), the Projection Head panel (Step 2.15), the receptive field (with `diagnosis_modes.compute_receptive_field`) and the size of every component. Then the Export Sizes panel (`estimate_export_sizes()`, `print_export_size_summary()`, as in `classification_workflow.md`, Step 2.24). The exported encoder never runs the projection head, so the exported parameters, ONNX files and TensorRT engines leave out its 132,736 parameters. The PyTorch weights and TorchScript file still hold them, and the checkpoint keeps them to resume.

The Model Size panel gives the parameters of Step 2.13 (backbone 1.76 MB, projection head 0.54 MB, total 2.29 MB).

**Example output: the Export Sizes panel** (estimated on CPU on 2026-11-25, shortened)
```text
│ Exported Parameters : 438,368 of 571,104                 │
│    PyTorch FP32 Weights         2.33 MB  ± 1 %           │
│    ONNX                         1.78 MB  ± 3 %           │
│    TensorRT FP16 Engine  1.16 - 2.12 MB  range           │
│    Training Checkpoint          6.96 MB  ± 1 %           │
```

### Step 2.21 — `view_input_batches()`
Only with `diagnosis_modes.view_input_images`, then the program exits. `augment_input_batch()` draws the two views with `create_views()`. `build_input_windows()` shows them as "View 1" and "View 2", each next to a panel with the class and, under "View augmentations", the drawn augmentations (`describe_report()`). The input samples written at every launch (`classification_workflow.md`, Step 3.1) draw the training and validation images as two views (`AUGMENTED_INPUT_SPLITS`) and the test images whole.

With `Data.grouped_batches` on, `show_input_batch()` first shows every training batch whole: `build_batch_mosaic()` (`vision_training/utilities/visualization.py`) lays out its thumbnails cut into blocks, up to 16 per row. `describe_input_batch()` writes its header (position, grouped or random, batch and block values). SPACE goes to the views of the first image, ENTER stops. It needs a display and was not run for this document.

### Step 2.22 — `restore_last_model()` → `get_last_checkpoint_iteration()` / `load_model()`
**Example output** (a new run folder)
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.
```

---

## Stage 3 — Training Loop

> The examples below are real outputs of a CPU run of `configuration_synthetic_shapes.yaml` shortened as said at the top (64 pixels, batch size 8, 12 iterations, seeded with 0). Its run name is `SSL_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2`. The Barlow Twins examples come from the first 4 iterations with `ProjectionHead.method: barlow_twins` (run name ending `_BT-OD0.005`).

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (`classification_workflow.md`, Step 3.1). Each iteration takes its training step, then saves, validates and prints on the iterations set for each, and evaluates last. This table shows what happens at each iteration of the run.

| Event | Iterations |
|---|---|
| Checkpoint saved | 3, 6, 9, 12 (nothing more after the loop) |
| Validation batch | 10 |
| Console summary | 4, 8, 12 |
| Test evaluation | 6, 12, last in the iteration |

### Step 3.2 — `setup_tensorboard_writers()`
**Example: the curves after the run**
```text
Tensorboard/Train/        Total Loss, InfoNCE Loss, Positive Pair Accuracy, Embedding Standard Deviation, Learning Rate   (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, InfoNCE Loss, Positive Pair Accuracy, Embedding Standard Deviation                  (iteration 10)
```

### Step 3.3 — `initialize_training_loop_state()` + `MetricTracker.create_trackers()`

The trackers of an interval start at 0: `interval_batches`, `total_loss`, `info_nce_loss`, `positive_pair_accuracy` and `embedding_standard_deviation`.

### Step 3.4 — `start_experiment_tracking()`
A background thread creates the MLflow run in the experiment `self_supervised`, so a slow database never delays the training. Here the message came after the summary of iteration 4: the checkpoint of iteration 3 holds `mlflow_run_id: None` (Step 3.5).

**Example output**
```text
MLflow Run Created In The Experiment 'self_supervised' : a42373adfbf94d839fa79699e0103b2b (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.5 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12 (`classification_workflow.md`, Step 3.6). The checkpoint holds the whole model, projection head included, to resume. A checkpoint written now also holds `pretrained_backbone_blocks` (`LOADED_BACKBONE_BLOCKS_METADATA_KEY`), `None` for this run.

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   6,967,879 bytes      Weights/Iteration_6/model_000006.pt   6,967,943 bytes
Weights/Iteration_9/model_000009.pt   6,967,943 bytes      Weights/Iteration_12/model_000012.pt  6,967,943 bytes
Weights/full-checkpoint               207 bytes: model_checkpoint_path: "Iteration_12", all_model_checkpoint_paths 3, 6, 9, 12
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone']
model_state: 108 tensors   format_version: 2   task: 'self_supervised'   pretrained_backbone: None
mlflow_run_id: None at iteration 3, 'a42373adfbf94d839fa79699e0103b2b' at 6, 9 and 12
```

### Step 3.6 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration (`classification_workflow.md`, Step 3.7). See Stage 4.

### Step 3.7 — `get_next_batch()` → `ClassificationDataset.__getitem__()`
Every image is read with `read_image()` and resized (`classification_workflow.md`, Steps 3.8 and 3.9). A batch that fails to load is dropped and the next one taken. The labels come with the batch but the training never reads them.

**Example: the first batch of the run**
```text
images               [8, 3, 64, 64] torch.uint8 in [0, 253]
labels               [8] torch.int64                  [5, 8, 2, 7, 8, 2, 6, 0]
sample_identifiers   [8] torch.int64                  [261621, 86285, 422277, 300679, 496458, 349760, 109902, 396162]
```

### Step 3.8 — `set_training_mode()`
`model.train()`: the batch normalization uses the statistics of the batch (Step 4.9 explains what changes at evaluation).

### Step 3.9 — `run_model_iteration()`
The same function for training and validation batches (`classification_workflow.md`, Step 3.11). With a single loss, `total_loss` is the loss of the method.

**Example: the first training batch**
```text
info_nce_loss = 3.0699   total_loss = 3.0699   positive_pair_accuracy = 0.0
```

### Step 3.10 — `prepare_batch()`
**Example: before and after**
```text
images  [8, 3, 64, 64] torch.uint8   -> [8, 3, 64, 64] torch.float32
labels  [8] torch.int64              -> [8] torch.int64
```

### Step 3.11 — `augment_batch()`
It returns the batch unchanged. The views are drawn in `forward_pass()` (Step 3.12), so the validation batches, which skip `augment_batch()`, get random views too.

### Step 3.12 — `forward_pass()` → `DataAugmenter.create_views()`
`forward_pass()` draws two views of every image, then calls `SelfSupervisedModel.forward(input_tensor=first_views, second_view_tensor=second_views)`. Every activated type is drawn for each image and each view. Only the second view can be solarized or blacked out.

**Example: the augmentations drawn for the two views of one image** (drawn outside the run with seed 0, `describe_report()` giving the lines)
```text
image 0, view 1:
  Geometric :   random_crop : 57x63 pixels (88 % of the area), at top 1 and left 7 pixels; random_left_right_flip; ...
  Photometric : random_hue : hue +16.5 degrees; random_linear_illumination : -36 levels growing towards the left
image 0, view 2:
  Geometric :   random_crop : 59x45 pixels (65 % of the area), at top 5 and left 0 pixels; random_left_right_flip; ...
  Photometric : brightness_contrast_saturation : brightness x0.99, contrast x0.88, saturation x0.86; ...
```

### Step 3.13 — `SelfSupervisedModel.forward()` → `encode()`
Both views go through the model as one batch of 2N = 16, so the batch normalization sees both. The embeddings are then split back. Under the bfloat16 autocast the outputs are bfloat16.

**Example: the outputs of the components on the 16 views**
```text
backbone           [16, 128, 2, 2] torch.bfloat16
global_pooling     [16, 128] torch.bfloat16 in [0, 0.96]      the representations
projection_head    [16, 128] torch.bfloat16                   the embeddings, split into [8, 128] + [8, 128]
```

### Step 3.14 — `compute_method_loss()` → `compute_info_nce_loss()` (SimCLR)
In `vision_training/losses/self_supervised_losses.py`, in float32 outside the autocast. The 2N embeddings are L2-normalized; their cosine similarities are divided by the temperature. Every view is a classification among the 2N - 1 other views, whose right answer is its partner, at index i + N for a first view and i - N for a second one. The loss is the cross entropy of these 2N classifications: about ln(2N - 1) for random embeddings, here ln 15 = 2.7081. The positive pair accuracy is the share of views whose most similar view is their partner.

**Example: the first iteration**
```text
view 0: cosine with its partner 0.13, mean cosine with the 14 other views 0.39, highest cosine 0.82 (view 13)
        -> its partner is not its most similar view, and its own row of the loss is about 4.72
info_nce_loss = 3.0699   positive_pair_accuracy = 0.0   (ln 15 = 2.7081 for random embeddings)
```

This table gives some of the 12 training iterations (learning rate 0.0005; the gradient norm was measured for this document, the trainer neither prints nor clips it).

| Iteration | InfoNCE loss | Positive pair accuracy | Gradient norm |
|---|---|---|---|
| 1 | 3.0699 | 0.0 | 68.36 |
| 4 | 2.3775 | 0.1875 | 37.75 |
| … | | | |
| 10 | 2.6380 | 0.125 | 19.07 |
| 12 | 2.7278 | 0.0 | 20.53 |

### Step 3.15 — `compute_embedding_standard_deviation()`
The standard deviation over the batch of every feature of the L2-normalized first-view embeddings, averaged over the features and multiplied by the square root of D. Spread embeddings give about 1, collapsed ones 0. It is computed without gradient and only watches the training.

**Example: the training batches**
```text
embedding_standard_deviation between 0.50 and 0.73 over the 12 iterations: 0.73 at iteration 1, 0.50 at iteration 12
```

### Step 3.16 — `compute_method_loss()` → `compute_barlow_twins_loss()` (Barlow Twins)
Every embedding feature is normalized over the batch (`BARLOW_TWINS_NORMALIZATION_EPSILON` added to the variance). Then the `[D, D]` cross-correlation C of the first views with the second views is computed. The loss pushes the diagonal to 1 (same feature value on both views) and the rest to 0 (features carry different information):

```text
barlow_twins_loss = Σ_i (C_ii - 1)² + off_diagonal_weight · Σ_i≠j C_ij²
```

The diagonal correlation is the mean of the diagonal; the off-diagonal correlation is the mean absolute value of the other terms.

**Example: the first iterations of the Barlow Twins run**
```text
cross-correlation C: [128, 128]
iteration 1: barlow_twins_loss = 165.0846   cross_correlation_diagonal = -0.0430   cross_correlation_off_diagonal = 0.3012
             embedding_standard_deviation = 0.7320   gradient norm = 6000.94
iterations 2 to 4: barlow_twins_loss = 95.06, 172.97, 93.75
```
With a diagonal near 0, the invariance term alone is close to D = 128. This explains the size of the loss and its gradients compared with InfoNCE.

### Step 3.17 — `compute_batch_metrics()`
It returns the batch metrics given by the model with its loss, without extra computation.

### Step 3.18 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
Adds the batch metrics to the interval trackers and writes them to `Tensorboard/Train` at every iteration.

### Step 3.19 — `training_loss.backward()` + `step_optimizer()`
As for every task (`classification_workflow.md`).

### Step 3.20 — validation pass
Every 10 iterations, one validation batch runs through `run_model_iteration()` without gradient, with new random views, in evaluation mode.

**Example: the validation batch of iteration 10**
```text
info_nce_loss = 2.7081 (= ln 15)   positive_pair_accuracy = 0.0   embedding_standard_deviation = 0.0005
```
In evaluation mode the batch normalization uses running statistics, not yet meaningful after 10 updates. All embeddings are almost the same, so the loss is exactly the one of random embeddings (Step 4.9).

### Step 3.21 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()` + `report_dropped_batches()`
The interval sums are divided by the number of batches. Batches dropped because their loading failed are counted by split, only when there are some (`classification_workflow.md`, Step 3.24).

**Example: the sums and the averages of the first interval**
```text
sums:  {'interval_batches': 4, 'total_loss': 10.8112, ...}
after:  {'interval_batches': 4, 'total_loss': 2.7028, 'info_nce_loss': 2.7028, 'positive_pair_accuracy': 0.1406, 'embedding_standard_deviation': 0.7179}
```

The summary prints the positive pair accuracy as a percentage with its change in points, the rest with 4 decimals. At iteration 12 the loss beats the 2.7028 of iteration 4, hence "Best".

**Example output: iteration 12**
```text
========================================================================================================================
|    Iteration: 12    |      Total Images: 96      |    Interval Images: 32    |   Throughput: 11.3 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     2.6904    -0.1385 Best
Moving Average of Valid Total Loss                     :     2.7081
...
Moving Average of Train Positive Pair Accuracy         :     7.81 %    -1.56 %
Moving Average of Valid Positive Pair Accuracy         :     0.00 %
...
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:02.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```
The interval of iteration 8 holds the test evaluation of iteration 6, hence its throughput of 2.8 images per second.

The Barlow Twins summary has no rate: every value keeps 4 decimals (first interval: total loss 131.7183, diagonal correlation 0.1465, off-diagonal correlation 0.3666).

### Step 3.22 — end of the loop
Iteration 12 is already saved and evaluated, so the loop saves and evaluates nothing more (`classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 1000 Test Images · 2026-10-23 02:19 ───────────────      <- the evaluation of iteration 12
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```

---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of iterations 6 and 12 of the Stage 3 run (1000 test images, batch size 8, `per_class_report: True`). They predate three changes: the recall intervals (2026-11-01, the `recall_lower` and `recall_upper` columns and the `Per Class k-NN Recall` table), the reference training images (2026-11-11: the k-NN numbers below come from a leave-one-out vote among test images, and the note now names the training images) and the evaluation summary and curves (2026-11-18).

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (`classification_workflow.md`, Step 4.1). The test images are encoded as they are, without views, as in deployment. Then the k-NN encodes the reference images with the same weights, under its own progress bar `k-NN Reference Images` (`(EMA Weights)` added for the averaged weights; Step 4.8).

**Example output: the progress bar** (1000 images, batch size 8)
```text
Test Evaluation Iteration 6: 100%|██████████| 125/125
```

### Step 4.2 — `create_evaluation_accumulator()`
The effective rank and the k-NN need the whole split, so the representations are kept on the CPU until the end of the loop.

**Example: the accumulator at the start and at the end of the loop**
```text
start: {'representations': [], 'labels': [], 'sample_identifiers': []}
end:   125 entries in each list, every representation entry [8, 128] torch.float32
```

### Step 4.3 — `prepare_batch()`
Same function as in training (Step 3.10).

### Step 4.4 — `predict_batch()`
`SelfSupervisedModel.forward(input_tensor=images)` without second view: `encode()` only.

**Example: the first test batch of iteration 12**
```text
images            [8, 3, 64, 64] torch.float32 in [0, 245]
representations   [8, 128] torch.float32 in [0, 0.084]
embeddings        [8, 128] torch.float32 in [-0.24, 0.22]
```

### Step 4.5 — `accumulate_evaluation_batch()`
Keeps the representations (`.detach().float().cpu()`), labels and identifiers. The embeddings are not kept: the representation is what is evaluated.

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
For each fixed test image of Step 2.11: the image next to two random views, captioned with its class (or `unlabelled`) and the cosine similarity of the two view representations. The views are drawn from the seed `SAMPLE_VIEWS_SEED` (20000) plus the sample identifier, inside `torch.random.fork_rng()`, so every evaluation compares the same views and the training random state is unchanged. `add_caption_bar()` writes the caption in a 24 pixel black bar (`CAPTION_BAR_HEIGHT`).

**Example: the sample images of iteration 12**
```text
image and views: (64, 192, 3)  ->  with the caption bar: (88, 192, 3), wider for a long caption (88, 195, 3), (88, 196, 3)
captions:        'pentagon | Views Similarity 1.000', 'star | Views Similarity 1.000', 'plus_sign | Views Similarity 1.000'
files:           Evaluation/Iteration_12/Samples/test_000000.png, test_000001.png, test_000005.png ... (8 images)
```
A similarity of 1.000 here does not mean invariance: every image gets almost the same representation at this point (Step 4.9).

### Step 4.7 — `compute_evaluation_results()`
It concatenates the representations and labels of the split, then computes the **effective rank** (Step 4.10) and the **representation standard deviation** (Step 3.15 on the representations) over every test image. With at least one labelled test image it adds the **k-NN metrics** (Step 4.8), the k-NN accuracy being the headline; without, the headline is the effective rank (Step 5.6). The three k-NN metrics are listed in `percent_metrics`, so the report writes them as percentages.

This table gives the global metrics of both checkpoints (between 0 and 1; with 9 classes, chance is about 0.11).

| Iteration | k-NN Accuracy | k-NN Balanced Accuracy | k-NN Macro F1 | Effective Rank | Representation Standard Deviation |
|---|---|---|---|---|---|
| 6 | 0.1080 | 0.1053 | 0.1030 | 1.0141 | 0.0002 |
| 12 | 0.1210 | 0.1165 | 0.1113 | 1.0458 | 0.0008 |

### Step 4.8 — `compute_nearest_neighbour_results()` → `compute_nearest_neighbour_predictions()`

It classifies every labelled test image by its neighbours among the reference images (`vision_training/evaluation/self_supervised/representation_metrics.py`):
- `encode_reference_images()` encodes the reference images of Step 2.1 with the evaluated weights. Test images are never compared with each other, so a split of one image per class (ImageNet-22k) works too.
- The L2-normalized representations are compared by cosine, in chunks of `SIMILARITY_CHUNK_SIZE` (1024) test images.
- The k = min(20, M) nearest reference images vote with weight exp(similarity / 0.07) (`NEAREST_NEIGHBOUR_TEMPERATURE`), summed per class. The class with most votes wins.
- `compute_confusion_matrix()` and `compute_metrics_from_confusion_matrix()` give the metrics (`classification_workflow.md`, Step 4.9), sparse above 1,000 classes (`MAXIMUM_DENSE_CONFUSION_CLASSES`; the dense ImageNet-22k matrix would take 3.8 GB).
- `compute_clopper_pearson_intervals()` gives the exact 95 % interval of every recall (`classification_workflow.md`, Step 4.14); `write_class_metrics_file()` saves `class_metrics.csv`.

**Example: image 0 of the test split at iteration 12** (label 6, plus_sign; old leave-one-out vote)
```text
top 20 similarities: all 1.0
neighbour labels:    [2, 7, 4, 5, 6, 8, 2, 5, 5, 7, 2, 2, 0, 0, 6, 3, 3, 4, 4, 5]
votes per class:     [3.20e6, 0, 6.40e6, 3.20e6, 4.80e6, 6.40e6, 3.20e6, 3.20e6, 1.60e6]   (exp(1 / 0.07) = 1.6e6 per neighbour)
prediction:          2 (triangle), wrong
```
With all similarities at 1.0 the vote is a plain count: triangle and hexagon tie, and `argmax` keeps the first.

### Step 4.9 — why the representations look collapsed
The test images are encoded in evaluation mode, so the batch normalization uses running statistics. After 12 updates at momentum 0.1 (`num_batches_tracked` 12) they are far from the data (running variance of the last backbone layer 0.30 on average). The normalized features of all images end up nearly identical.

**Example: one test batch at iteration 12, in both modes**
```text
evaluation mode (running statistics):  mean pairwise cosine 1.0     feature standard deviation 2.2e-5
training mode (batch statistics):      mean pairwise cosine 0.774   feature standard deviation 0.114
```
This explains the effective rank near 1, the views similarity of 1.000, the validation loss of exactly ln 15 and the k-NN near chance. The representations are positive after the ReLU and pooling, which also brings their directions closer. A longer training lets the running statistics settle.

### Step 4.10 — `compute_effective_rank()`
The RankMe effective rank of the `[N, F]` representations: the singular values are normalized into a distribution p, and the rank is the exponential of its entropy (`EFFECTIVE_RANK_EPSILON` 1e-12, NaN when not finite). It goes from 1 (one direction) to F (every direction used equally).

**Example: iteration 12**
```text
representations:  [1000, 128] in [0, 0.084], mean 0.0171
singular values:  9.2822, 0.0077, 0.0033, 0.0026, 0.0019, ...   (sum 9.3237)
p_1 = 9.2822 / 9.3237 = 0.99555   entropy = 0.0448   effective rank = exp(0.0448) = 1.0458
```

### Step 4.11 — the per-class table and scalars
"Per Class k-NN Metrics" lists every class with its labelled test images, precision, recall and F1, worst F1 first. "Per Class k-NN Recall" (`build_recall_interval_table()`) gives the recall with `95 % Lower` and `95 % Upper`, worst recall first. Each class also gives the scalars `k-NN Precision/<class>`, `k-NN Recall/<class>` and `k-NN F1/<class>`. `class_metrics.csv` keeps the class order of the label dictionary.

**Example: `Evaluation/Iteration_12/class_metrics.csv`** (captured before the recall bounds)
```text
class_index,class,samples,precision,recall,f1
0,square,91,0.100000,0.065934,0.079470
1,circle,118,0.153153,0.144068,0.148472
2,triangle,125,0.150685,0.264000,0.191860
...
8,octagon,114,0.087379,0.078947,0.082949
```

### Step 4.12 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
Each time a checkpoint is evaluated, its block goes to the console and `Results.txt`, its scalars and samples to TensorBoard, its metrics and `class_metrics.csv` to MLflow. Then `update_evaluation_curves()` writes its row into `Evaluation/evaluation_summary.csv` and draws `evaluation_curves.png` again.

**Example output: the block of iteration 12** (also appended to `Results.txt`; the file names at the end are clickable links)
```text
Iteration 12 · 1000 Test Images · 2026-10-23 02:19 ─────────────────────────────────────────────────────────────────────
  k-NN Accuracy                       12.10 %   k-NN Balanced Accuracy   11.65 %
  k-NN Macro F1                       11.13 %   Effective Rank           1.0458
  Representation Standard Deviation   0.0008
  Labelled Test Images : 1000 of 1000
  k-NN : 20 neighbours among the other labelled test images
  Sample Images : 8 fixed test images

  Per Class k-NN Metrics
   Class       Samples   Precision    Recall        F1
  ─────────────────────────────────────────────────────
   pentagon         99      4.76 %    2.02 %    2.84 %
   ...
   triangle        125     15.07 %   26.40 %   19.19 %
  k-nearest neighbour classification (20 neighbours) of the labelled test images by the other ones. Worst F1 first, ties
  in class order. — : no labelled test image of the class.
  - Results : Results.txt
  - Class Metrics : class_metrics.csv
  - Sample Images : Samples
```

After the two evaluations, `Tensorboard/Test` holds 32 scalars (5 global + 27 per class) at steps 6 and 12, and MLflow 44 metric keys. A run started after 2026-11-18 also sends `Input_Samples/`, `evaluation_curves.png` and `Evaluation/evaluation_summary.csv` to MLflow. With `Optimization.exponential_moving_average` on, the averaged weights are evaluated the same way, in `Evaluation/Iteration_<n>/EMA/` and `Tensorboard/Test_EMA`.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of `scripts/evaluation/evaluate_checkpoint.py` on the Stage 3 run (checkpoints 3, 6, 9 and 12), with `iteration = "all"`, `batch_size = 16` and the first 200 test images (`test_200`). A second evaluation of iteration 12 uses the same images without categories (`test_unlabelled_200`). As in Stage 4, the k-NN numbers come from the old leave-one-out vote and the recall table is missing. The folder names are those of the current `<run>/Checkpoint_Evaluation/`.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The script of every task (`classification_workflow.md`, Stage 5), set in its main block: `run_name`, `experiment_folder` (the run folder), `iteration` (`None` for the last, an integer or `"all"`) and `test_data_folder`. The task has no `evaluation.error_dumps` section, so `maximum_error_images_per_class_pair` is skipped with a yellow warning. `save_predictions`, `target_precisions`, `nms_criterion` and `nms_distance_exponent` are skipped the same way. A run saved before the reference training images lacks `evaluation.number_reference_images_per_class`, which the checks refuse: add the key to its `training_configuration.yaml` first. The split needs no labels (Step 5.6).

### Step 5.2 — trainer construction
The script loads the run's `training_configuration.yaml` with the classes of its `label_dictionary.json`. `point_to_checkpoint_evaluation()` sends the files to `<run name>/Checkpoint_Evaluation/`, or `<run name>/Checkpoint_Evaluation_split_<parent folder>_<split folder>/` for another split such as `test_200`. Then `disable_pretrained_settings()` runs and a `SelfSupervisedTrainer` is built inside `read_manifests_once()`. `tracking_root` keeps the experiments root folder, so the evaluation is tracked in the backend of the run.

**Example output** (shortened)
```text
- Test Data Folder        : test_200 (200 images, 9 categories)
- Batch Size              : 16
- MLflow Tracking         : sqlite:///.../runs/mlflow.db
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
The MLflow run `<run name>-evaluation` is created in the backend of the evaluated run and tagged with the training run.

**Example output**
```text
MLflow Run Created In The Experiment 'self_supervised' : fea7baf32e174d2eb2c5f3583ad3cd74 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
tags: evaluation_of a42373adfbf94d839fa79699e0103b2b, evaluated_iterations "12, 9, 6, 3", test_data_folder, best_iteration 3
metrics: 33 keys, 132 rows
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
So that the checkpoint to keep can be chosen, every evaluation adds its row to `Checkpoint_Evaluation/evaluation_summary.csv` (increasing iteration order, rows of earlier launches kept) and draws `evaluation_curves.png` again. The best checkpoint of the k-NN accuracy is printed after the last one. The file has 33 columns: the iteration, the 5 global metrics and the 27 per-class metrics.

**Example output**
```text
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
  - Evaluation Curves : evaluation_curves.png
  - Best Checkpoint : iteration 3 (k-NN Accuracy 12.50 %)

iteration,k-NN Accuracy,k-NN Balanced Accuracy,k-NN Macro F1,Effective Rank,Representation Standard Deviation, ...
3,0.125,0.1118...,0.1045...,1.0072...,0.000103..., ...
...
12,0.11,0.09638447971781304,0.08247118900298986,1.0432647447459944,0.0008116378448903561, ...
```


### Step 5.6 — `run_test_evaluation()` for each checkpoint
Each checkpoint prints a block like Step 4.12. This table gives the global metrics of the four checkpoints on the 200 images (as in `evaluation_summary.csv`).

| Iteration | k-NN Accuracy | k-NN Balanced Accuracy | k-NN Macro F1 | Effective Rank | Representation Standard Deviation |
|---|---|---|---|---|---|
| 12 | 0.1100 | 0.0964 | 0.0825 | 1.0433 | 0.0008 |
| 9 | 0.0750 | 0.0632 | 0.0523 | 1.0233 | 0.0004 |
| 6 | 0.0950 | 0.0848 | 0.0840 | 1.0132 | 0.0002 |
| 3 | 0.1250 | 0.1118 | 0.1045 | 1.0073 | 0.0001 |

The effective rank and the spread grow as the batch normalization statistics settle (Step 4.9). The k-NN accuracy only moves around chance (0.11) on 200 images, which makes iteration 3 the best checkpoint here.

On the same 200 images without labels (`test_unlabelled_200`), a yellow block counts the 200 samples kept without class, the k-NN is skipped (`Labelled Test Images : 0 of 200`) and the best checkpoint is chosen by `Effective Rank` (1.0433 at iteration 12). Without labels there is no class table nor `class_metrics.csv`, the samples are captioned `unlabelled`, and the MLflow run has 3 metric keys. In the same `Checkpoint_Evaluation/` folder as the labelled evaluation, the row of iteration 12 replaces the labelled one with empty k-NN cells; the rows of 3, 6 and 9 stay.
