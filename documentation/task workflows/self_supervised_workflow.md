# Self-Supervised Workflow

This document follows one self-supervised run from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`; here they keep a short explanation and the real output of the self-supervised run, and the steps specific to the self-supervised task (the two random views, the projection head, the InfoNCE and Barlow Twins losses, the k-nearest neighbour evaluation and the effective rank) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for other trainings:
- **Stages 1 and 2**: `configurations/self_supervised/configuration_synthetic_shapes.yaml` (synthetic shapes, 9 shape classes, 256 pixel images, batch size 128, SimCLR), with the run folder redirected to a temporary folder and 2 loading workers.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel images, batch size 8, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, a test split of 1000 images and the MLflow tracking turned on. A second run, identical except for `ProjectionHead.method: barlow_twins`, gives the Barlow Twins outputs of Stage 3. After 12 iterations the representations are almost random, so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

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
│   trainer of its task, builds it and starts the training loop. Give a self_supervised configuration to run this
│   workflow.
│
├── parse_configuration_path()
│       argparse reader of --configuration_path, which stops with "No configuration file at ..." when the path is not
│       a file.
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, ProjectionHead), converts the types, adds label_mode and label_dictionary to 'Data' and runs
│   │   the checks.
│   │
│   ├── load_configuration(configuration_path)
│   │       Plain YAML reader, no conversion.
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category identifier: class index}, without background. The classes only serve
│   │       the k-nearest neighbour evaluation: the training never reads a label.
│   └── check_configuration()
│       ├── common checks                         experiment, data, backbone, optimization (see classification)
│       └── check_self_supervised_configuration()
│               ├── data:       multi_class label mode, classes 0..K-1, Data.data_augmentation off (the views replace it)
│               ├── no sequence settings (check_image_task_sequence_settings)
│               ├── views:      random_resized_crop scale in (0, 1] and aspect ratio, probabilities in [0, 1] (or one
│               │               per view), rotation degrees, perspective distortion, jigsaw grid, colour jitter
│               │               strengths, hue shift, blur kernels (the Gaussian one below twice the input size),
│               │               sigma and angle, noise, crops of at least 2 pixels on each side, and the ranges of the
│               │               gamma, haze, illumination, plasma shadow, blur direction, solarization (addition
│               │               strictly within (-0.5, 0.5)) and erasing
│               ├── grouped:    Data.grouped_batches: activated flag, block_key a string, batch_key null or a string,
│               │               images_per_block >= 2, dividing the batch size when the section is on
│               ├── head:       check_projection_head_configuration(): pooling, layers, normalization, method
│               │               (simclr or barlow_twins) and its temperature or off-diagonal weight
│               └── batch:      batch size of at least 2, number_neighbours > 0
│
├── get_trainer_class(task)  -> SelfSupervisedTrainer
│
├── SelfSupervisedTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   Stores the method, the number of neighbours of the k-NN evaluation and the ViewAugmenter that draws the two
│   │   random views of every image.
│   │
│   └── BaseTrainer.__init__()
│       │   Same order as for every task: device, run name, settings, augmenter, classes of a resumed run, previous
│       │   training moved aside, paths, loaders and their check, sample images, model, pretrained components,
│       │   optimizer, metric tracker, printouts, restoration.
│       │
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter()
│       │       The run name starts with SSL and ends with the head (PAVG.<layers>-PBN) and the method
│       │       (SIMCLR-T<temperature> or BT-OD<off-diagonal weight>). The DataAugmenter stays off.
│       ├── restore_run_label_dictionary() / move_previous_run_folder() / setup_training_paths()
│       │       A resumed run keeps the classes of its label_dictionary.json, a training from scratch renames the
│       │       run folder of a previous training of the same settings v<n>_<run name>, then the run folder is
│       │       created with the configuration and label_dictionary.json (classification_workflow.md, Step 2.6).
│       ├── build_dataloaders()
│       │   │   Builds the three loaders and names the classes after the categories of the three manifests.
│       │   │
│       │   └── get_classification_dataloaders(self_supervised=True)
│       │       ├── ClassificationDataset.__init__(output_mode="multi_class", keep_unlabelled_samples=True)
│       │       │       Same dataset as the classification; an image without a mapped category is kept, label -1.
│       │       ├── build_grouped_batch_sampler()
│       │       │       With Data.grouped_batches on, the GroupedBatchSampler of the training and validation splits
│       │       │       whose samples hold its keys: batches of blocks of images sharing an attribute of their groups.
│       │       └── build_native_loader()
│       │               drop_last for the training and validation loaders, so that every batch has batch_size images,
│       │               or the GroupedBatchSampler in place of the batch size, the shuffling and drop_last.
│       ├── check_loaders_give_batches()          refuses a split whose loader gives no batch
│       ├── get_test_sample_classes() + choose_sample_identifiers()
│       │       The 8 fixed test images of the sample images, spread over the classes.
│       ├── build_model()
│       │   └── SelfSupervisedModel.__init__()
│       │       ├── build_backbone()              the shared backbone
│       │       ├── global_pooling                AdaptiveAvgPool2d(1) + Flatten: the representation used downstream
│       │       └── ProjectionHead()              FullyConnectedBlock + embedding_layer: the embedding of the loss,
│       │                                         thrown away after training
│       ├── initialise_pretrained_components()    only the Backbone can come from another run
│       ├── build_optimizer()
│       ├── get_tracked_metrics() + MetricTracker.__init__()
│       │       SimCLR: total loss, InfoNCE loss, positive pair accuracy, embedding standard deviation.
│       │       Barlow Twins: total loss, Barlow Twins loss, diagonal and off-diagonal correlations, embedding
│       │       standard deviation.
│       ├── print_experiment_information() -> describe_split() -> count_split_samples()
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       │       The summary adds a Projection Head panel.
│       ├── view_input_batches()  (only with view_input_images, then exits)
│       │       The two random views of a training image, each next to the transformations drawn for it.
│       └── restore_last_model() -> get_last_checkpoint_iteration() / load_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── setup_tensorboard_writers() / initialize_training_loop_state() / start_experiment_tracking()
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch()                    a batch failing to load is dropped and the next one taken
        │   │   └── ClassificationDataset.__getitem__() -> read_image()
        │   │           images [B, 3, H, W] uint8, labels [B] int64 (unused by the training), sample_identifiers [B]
        │   ├── set_training_mode()
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()                 images to float32, labels to int64 on the device
        │   │   ├── augment_batch()                 returns the batch unchanged: the views are drawn in forward_pass
        │   │   ├── forward_pass()
        │   │   │   ├── ViewAugmenter.create_views()        two random views of every image, in float32
        │   │   │   └── SelfSupervisedModel.forward(input_tensor, second_view_tensor)
        │   │   │       ├── encode()                       the 2N views in one pass: backbone, pooling, projection head
        │   │   │       ├── compute_method_loss()          compute_info_nce_loss() or compute_barlow_twins_loss()
        │   │   │       └── compute_embedding_standard_deviation()   the collapse watch
        │   │   ├── compute_batch_metrics()         the metrics returned by the model, nothing recomputed
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()
        │   ├── save_model() on the multiples of weight_saving_iterations and intermediate_weight_saving_iterations
        │   ├── validation pass every validation_iteration_ratio iterations (new random views, no gradient)
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary() + report_dropped_batches()
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations, last    -> [STAGE 4]
        ├── save_model() of the steps taken since the last checkpoint, if any (nothing evaluated after the loop)
        └── close the writers, finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
├── create_evaluation_accumulator()
│       Empty lists of representations, labels and sample identifiers.
├── for each test batch:
│   ├── prepare_batch()
│   ├── predict_batch()
│   │       SelfSupervisedModel.forward() without second view: the representations and embeddings of the images
│   │       themselves, as in deployment.
│   ├── accumulate_evaluation_batch()
│   │       Keeps the representations (float32, CPU), the labels and the identifiers.
│   └── save_sample_images() -> build_sample_image()
│           The image and two seeded random views, captioned with the class and the cosine similarity of the
│           representations of both views.
├── compute_evaluation_results()
│   ├── compute_effective_rank()             RankMe effective rank of all the representations
│   ├── compute_embedding_standard_deviation()   spread of all the representations
│   └── compute_nearest_neighbour_results()  (with at least two labelled test images)
│       ├── compute_nearest_neighbour_predictions()   leave-one-out weighted k-NN vote among the labelled images
│       ├── compute_confusion_matrix() + compute_metrics_from_confusion_matrix()
│       └── write_class_metrics_file()                class_metrics.csv
└── report_evaluation() -> report_evaluation_results() + track_evaluation()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task: loads training_configuration.yaml, builds the SelfSupervisedTrainer in the output folder, then
    evaluate_checkpoints() -> start_evaluation_tracking() + summarise_checkpoint_evaluations() -> run_test_evaluation()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/self_supervised/configuration_synthetic_shapes.yaml`.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1): `parse_configuration_path()` reads the path of the YAML file from the command line (`--configuration_path`, absolute or relative to the current folder, `parser.error()` stopping with "No configuration file at ..." when it is not a file), then `main()` loads it, looks up the trainer of `ExperimentConfiguration.task` and starts its training loop. Nothing in it is specific to the self-supervised task.

```bash
python -m vision_training.main --configuration_path vision_training/configurations/self_supervised/configuration_synthetic_shapes.yaml
```

### Step 1.2 — `load_experiment_configuration()`
It splits the YAML into the two dictionaries read by every later stage, converts the values YAML reads with the wrong type (`1e10` is read as a string), adds `tracking_root` (the root folder of the MLflow backend, the `experiment_folder` of the configuration), adds `label_mode` and `label_dictionary` to `Data` (Step 1.4) and runs the checks (Step 1.5). The self-supervised task has one section of its own, `ProjectionHead`, and a `Data.view_augmentation` subsection that describes the random views.

**Example: the sections and the converted values**
```text
raw keys:               ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'ProjectionHead']
raw number_iterations:  '1e10'                       -> after loading: 10000000000
experiment keys:        task, experiment_folder, train_data_folder, validation_data_folder, test_data_folder, batch_size,
                        information_dump, weight_saving_iterations, intermediate_weight_saving_iterations,
                        number_iterations, validation_iteration_ratio, evaluation, tracking, learning_rate,
                        resume_training, diagnosis_modes, tracking_root
model keys:             ['Optimization', 'Data', 'Backbone', 'ProjectionHead']
task:                   'self_supervised'
experiment_folder:      PosixPath('.../visual-language-model-spark/experiments')
learning_rate: 0.0005   batch_size: 128   information_dump: 10   weight_saving_iterations: 250
ProjectionHead:         {'global_pooling': 'average', 'fully_connected': [512, 128], 'fully_connected_normalization': 'batch',
                         'batch_normalization_momentum': 0.1, 'method': 'simclr', 'simclr': {'temperature': 0.2},
                         'barlow_twins': {'off_diagonal_weight': 0.005}}
```

### Step 1.3 — `load_configuration()`
Plain YAML reader, as for every task: no conversion and no check.

**Example output**
```text
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '.../classifier_data/synthetic_shapes/label_mapping.json'}
```

### Step 1.4 — `build_label_dictionary()`
Reads the mapping file of `Data.label_mapping` (in `vision_training/data/label_mapping.py`) and gives the class index of every category identifier. Unlike the detection and the segmentation there is no background class. The labels are never read by the training: they only name the classes of the k-nearest neighbour evaluation of Stage 4.

**Example output**
```text
label_mode:       multi_class
label_dictionary: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
```

### Step 1.5 — `check_configuration()`
The common checks, then `check_self_supervised_configuration()` (in `vision_training/configuration_processing/self_supervised_checks.py`):
- **data**: `multi_class` label mode, classes 0 to K-1, `Data.data_augmentation` off, since the random views replace it, and no sequence settings (`check_image_task_sequence_settings()`).
- **views**: the `random_resized_crop` scale is a list `[low, high]` with `0 < low <= high <= 1` and a valid aspect ratio, which together give crops of at least `MINIMUM_CROP_SIDE` (2) pixels on each side at the input size (`check_crop_sides`), every probability is in [0, 1] or a list of two of them `[first view, second view]`, the rotation `maximum_degrees` is at most `MAXIMUM_ROTATION_DEGREES`, the perspective `distortion_scale` is in [0, 1], the jigsaw `grid` holds at least two pieces and divides the input size, the colour jitter strengths are not negative and the hue shift at most `MAXIMUM_HUE_SHIFT`, the Gaussian blur kernel is odd and below twice the input size, its sigma range valid, the motion blur kernel sizes are odd and at least 3 and its `maximum_angle` at most `MAXIMUM_MOTION_BLUR_DEGREES`, the noise `standard_deviation` is not negative, and the ranges of `VIEW_TRANSFORMATION_RANGES` (gamma, haze, linear illumination, plasma shadow, motion blur direction, solarization, random erasing) are within the bounds kornia accepts, the solarization addition strictly within (-0.5, 0.5).
- **grouped batches**: `check_grouped_batches_configuration()`: `Data.grouped_batches.activated` is True or False, `block_key` a non-empty string, `batch_key` null or a non-empty string, and `images_per_block` an integer of at least 2 that divides the batch size when the section is on.
- **head**: `check_projection_head_configuration()`: pooling, layers, normalization, the method (`simclr` or `barlow_twins`) and its settings.
- **batch and evaluation**: a batch size of at least 2, and `number_neighbours` above 0.

Every error is listed before the program stops.

**Example output: a configuration with three errors**
```text
------------------------------------------------------
Invalid Configuration File : file:///.../configuration_wrong.yaml
------------------------------------------------------
  - 'Data.view_augmentation.random_resized_crop.scale' must be a list of two numbers [low, high] with 0 < low <= high <= 1, got [0.6, 1.4].
  - 'ProjectionHead.method' must be one of ['simclr', 'barlow_twins'], got 'byol'.
  - 'ExperimentConfiguration.batch_size' must be at least 2 for the self-supervised task: the losses compare the views of the images of a batch with each other.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
<class 'vision_training.trainers.self_supervised_trainer.SelfSupervisedTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on `configurations/self_supervised/configuration_synthetic_shapes.yaml` (256 pixel images, batch size 128, 2 loading workers), with the run folder redirected to a temporary folder. The whole construction took 7.0 s on CPU.

### Step 2.1 — `SelfSupervisedTrainer.__init__()`
Before calling `BaseTrainer.__init__()`, it stores what its hooks read: `self.method` (`ProjectionHead.method`), `self.number_neighbours` (`evaluation.number_neighbours`, the k of the k-NN evaluation) and `self.view_augmenter`, the `ViewAugmenter` built from `Data.view_augmentation` at the input image size. Its class attributes name the task for the shared code, `DROPS_LAST_TRAINING_BATCH` telling it that the training loader drops its last incomplete batch (Step 2.19).

**Example output**
```text
method: simclr   number_neighbours: 20
TASK_NAME: self_supervised   SAMPLE_NAME: Images   COUNTS_FRAMES: False   DROPS_LAST_TRAINING_BATCH: True
```

### Step 2.2 — `BaseTrainer.__init__()`
The constructor of every task (see `classification_workflow.md`, Step 2.2).

**Example: the settings it stores**
```text
device: cpu   per_class_report: True   number_sample_images: 8   input_image_size: 256   batch_size: 128
mixed_precision: True   start_iteration: 1   dtype: torch.float32
```

### Step 2.3 — `get_device()`
**Example output**
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The run name gives the task (`SSL`), the input, the backbone, then the head built by `build_self_supervised_sections()` (in `vision_training/configuration_processing/run_name.py`): `PAVG` for the average pooling, the layers of the head, `PBN` for its batch normalization, and the method, `SIMCLR-T<temperature>` or `BT-OD<off-diagonal weight>`. There is no class count, since the training never reads a label.

**Example output**
```text
SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2
```

### Step 2.5 — `DataAugmenter()` and `ViewAugmenter()`
The `DataAugmenter` of the other tasks is built but stays off (`Data.data_augmentation.activated: False`, enforced by Step 1.5). The `ViewAugmenter` (in `vision_training/data/view_augmentations.py`) holds one Kornia chain per view in `view_transformations`, built by `build_view_transformations(view_index)` with the probabilities of that view (a probability given as `[first view, second view]` differs between the two chains), every transformation being drawn independently for every image. The haze is `RandomHaze`, a Kornia intensity transformation of the module, since Kornia has none:
- `draw_view(images, view_index)` flips the BGR images to RGB, scales them to [0, 1], applies the chain of the view, clamps the result and converts it back to BGR in [0, 255].
- `create_views(images, describe_views=False)` runs in float32, with the autocast off, draws the first view with the first chain and the second view with the second chain, and returns `(first_views, second_views, descriptions)`, the descriptions (`describe_view_transformations()`) being `None` unless asked for by the input viewer.

**Example: the attributes of both augmenters (the two chains differ by the probabilities of the Gaussian blur, the solarization and the black-out)**
```text
DataAugmenter: {'activated': False, 'probability': 0.4, 'independent_frames': False, 'shuffle_frames_activated': False, ...}
ViewAugmenter.view_transformations[0]: Sequential(
  (0): RandomResizedCrop(scale=(0.6, 1.0), resize_to=(0.75, 1.3333), output_size=(256, 256), p=1.0, ..., resample=bilinear)
  (1): RandomHorizontalFlip(p=0.5, ...)
  (2): RandomVerticalFlip(p=0.5, ...)
  (3): RandomRotation(degrees=180.0, p=0.5, ..., resample=bilinear)
  (4): RandomPerspective(distortion_scale=0.2, p=0.5, ..., resample=bilinear)
  (5): RandomJigsaw(grid=(2, 2), p=0.0, ...)
  (6): ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8, ...)
  (7): RandomGrayscale(p=0.2, ...)
  (8): RandomGamma(gamma_factor=(0.7, 1.5), gain_factor=(1.0, 1.0), p=0.3, ...)
  (9): RandomHaze(strength=(0.0, 0.5), airlight=(0.6, 1.0), p=0.3, ...)
  (10): RandomLinearIllumination(gain=(0.05, 0.25), sign=(-1.0, 1.0), p=0.3, ...)
  (11): RandomPlasmaShadow(roughness=(0.1, 0.7), shade_intensity=(-0.5, 0.0), shade_quantity=(0.0, 0.5), p=0.2, ...)
  (12): RandomGaussianBlur(sigma=(0.1, 2.0), p=1.0, ..., kernel_size=(9, 9), separable=True, border_type=reflect)
  (13): RandomMotionBlur(kernel_size=(3, 11), angle=180.0, direction=(-1.0, 1.0), p=0.1, ..., border_type=constant, resample=nearest)
  (14): RandomSolarize(thresholds=(0.5, 0.5), additions=(0.0, 0.0), p=0.0, ...)
  (15): RandomGaussianNoise(p=0.2, ..., mean=0.0, std=0.03)
  (16): RandomBlackOut(p=0.0, ...)
  (17): RandomErasing(scale=(0.02, 0.15), resize_to=(0.3, 3.3), value=0.0, p=0.25, ...)
)
ViewAugmenter.view_transformations[1]: Sequential(
  ...
  (12): RandomGaussianBlur(sigma=(0.1, 2.0), p=0.1, ..., kernel_size=(9, 9), separable=True, border_type=reflect)
  ...
  (14): RandomSolarize(thresholds=(0.5, 0.5), additions=(0.0, 0.0), p=0.2, ...)
  ...
  (16): RandomBlackOut(p=0.3, ...)
  ...
)
```
(Kornia prints the aspect ratio range of the crop and of the erased rectangles under the name `resize_to`.)

### Step 2.6 — `setup_training_paths()`
Creates the run folder, copies the YAML file into it and writes the label dictionary of the run (`label_dictionary.json`), then the constructor writes the data record once the loaders exist. The copy is a file copy, so it holds the file as written, not the values changed in memory after loading. Before it, `restore_run_label_dictionary()` keeps the classes of a resumed run and `move_previous_run_folder()` renames the folder of a previous training of the same settings when the run starts from scratch, as for every task (see `classification_workflow.md`, Step 2.6).

**Example: the run folder after the construction**
```text
Tensorboard/                    folder
Weights/                        folder
data.json                       17,000,995 bytes
label_dictionary.json           the class index of every category of the run
training_configuration.yaml     23,809 bytes
```

### Step 2.7 — `build_dataloaders()`
Calls `get_classification_dataloaders(..., self_supervised=True)` (Step 2.8), then merges the categories of the three manifests by identifier, so that a class absent from the test manifest keeps its name, and stores `class_names` in class index order.

**Example output**
```text
class_names: ['square', 'circle', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'plus_sign', 'star', 'octagon']
```

### Step 2.8 — `get_classification_dataloaders(self_supervised=True)`
The loaders of the classification (in `vision_training/data/loaders.py`), with three differences: the output mode is always `multi_class` (there is no `ClassificationHead` section), the datasets keep the images without a mapped category (Step 2.9), and the training and validation loaders drop their last incomplete batch, so that every batch of the losses has `batch_size` images. The test loader keeps every image.

With `Data.grouped_batches` on, `build_grouped_batch_sampler()` (in `vision_training/data/grouped_batch_sampler.py`) builds the `GroupedBatchSampler` of the training and validation splits from the `groups` of their samples, given to `build_native_loader()` (Step 2.10). The sampler rebuilds the batches at every pass: the images grouped by (batch value, block value) are shuffled and cut into blocks of `images_per_block`, the blocks of every batch value are joined into full batches, and the images left over are shuffled into full random batches, the last incomplete one being dropped, so that a pass still gives `floor(N / batch_size)` batches and the iterations per epoch do not change. A split whose samples hold none of the keys keeps random batches with a yellow warning, and a split where only some samples hold them stops the launch with the number of samples missing each key.

**Example: the three splits**

| Split | Images | Batches | drop_last |
|---|---|---|---|
| train | 500000 | 3906 | True |
| validation | 50000 | 390 | True |
| test | 50000 | 391 | False |

**Example output: grouped batches of a training split of 10 images in two cameras, with batches of 4 images and blocks of 2, and a validation split without groups**
```text
Grouped batches of the 'train_data_folder' split : block key 'camera' (2 values), 2 grouped and 0 leftover batches per pass
No sample of the 'validation_data_folder' split holds the keys ['camera'] of 'Data.grouped_batches': its batches stay random.
```

### Step 2.9 — `ClassificationDataset.__init__()`
The dataset of the classification (in `vision_training/data/classification/classification_dataset.py`), with `keep_unlabelled_samples=True`: an image without a mapped category is kept with the label -1 (`IGNORED_LABEL_INDEX`) instead of being dropped, and a yellow line counts them (see Step 5.6). Every record also keeps the optional `groups` of its sample, an empty dictionary without them, read by the grouped batches (Step 2.8). It shuffles the manifest once, so the first kept sample is not the first of the file.

**Example: the manifest, the first kept sample and `dataset[0]`**
```text
manifest keys: ['info', 'categories', 'samples']   500000 samples, 9 categories
manifest sample:  {'id': 1, 'file_name': 'train_000000.png', 'category_id': 6}
first kept:       {'identifier': 145458, 'file_name': 'train_145457.png', 'class_indices': [5], 'groups': {}}
dataset[0]:       images [3, 256, 256] torch.uint8 in [14, 231]   labels () torch.int64 = 5   sample_identifiers 145458
```

### Step 2.10 — `build_native_loader()`
Same torch DataLoader as the classification. With grouped batches, the DataLoader receives the `GroupedBatchSampler` as its `batch_sampler` instead of the batch size, the shuffling and `drop_last`, which it refuses next to a batch sampler: it then reads `batch_size None` and `drop_last False`, its length being the number of batches of the sampler.

**Example: the training loader**
```text
len 3906   batch_size 128   num_workers 2   prefetch_factor 2   pin_memory True   persistent_workers True   drop_last True
```

**Example: a training loader of grouped batches (10 images in two cameras, batches of 4 images, blocks of 2)**
```text
len 2   batch_size None   num_workers 2   prefetch_factor 2   pin_memory True   persistent_workers True   drop_last False   batch_sampler GroupedBatchSampler
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
`get_test_sample_classes()` gives the class of every test image (`get_sample_class_indices()` of the dataset, the unlabelled images forming their own group), so that `choose_sample_identifiers()` spreads the fixed sample images over the classes.

**Example output**
```text
test sample classes: 50000 entries, e.g. {19952: [6], 20782: [5], 35797: [4], 10507: [3], ...}
sample identifiers:  {1, 2, 6, 7, 9, 11, 13, 15}
                     test_000000, test_000001, test_000005, test_000006, test_000008, test_000010, test_000012, test_000014
                     classes [7], [6], [5], [1], [0], [3], [2], [4]                       (no octagon among the 8)
```

### Step 2.12 — `build_model()`
Creates the `SelfSupervisedModel` from the model registry (`TASK_MODEL_CLASSES["self_supervised"]`).

### Step 2.13 — `SelfSupervisedModel.__init__()`
The model (in `vision_training/models/self_supervised/self_supervised_model.py`) has three parts:
- `backbone`: the shared backbone (Step 2.14).
- `global_pooling`: the pooling of the last feature map into the **representation** that is kept after training (for a classifier or a retrieval index).
- `projection_head`: the `ProjectionHead` that turns the representation into the **embedding** compared by the loss, thrown away after training.

Its methods:
- `represent()`: backbone then pooling, `[N, 128]`.
- `encode()`: returns `representations` and `embeddings`.
- `forward(input_tensor, second_view_tensor=None)`: without second view it only encodes the images (Stage 4); with it, it concatenates both views into one batch of 2N images, splits the embeddings back, calls `compute_method_loss()` and adds the `embedding_standard_deviation` of the first views (Step 3.15).

**Example: the components**

| Component | Contents | Parameters |
|---|---|---|
| backbone | Backbone | 438,368 |
| global_pooling | Sequential(AdaptiveAvgPool2d(1), Flatten) | 0 |
| projection_head | FullyConnectedBlock(Linear 128→512, FeatureBatchNormalization(512, momentum 0.1), ReLU) + embedding_layer Linear 512→128 | 132,736 |
| Total | representation size 128, embedding size 128 | 571,104 |

### Step 2.14 — `build_backbone()`
The backbone of every task (see `classification_workflow.md`).

**Example output** (256 pixels)
```text
Block outputs: [1, 16, 128, 128], [1, 32, 64, 64], [1, 32, 32, 32], [1, 64, 16, 16], [1, 64, 8, 8], [1, 128, 8, 8]
--------------------------------------------
Final Backbone output shape: [1, 128, 8, 8]
--------------------------------------------
```

### Step 2.15 — `ProjectionHead.__init__()` and its panel
The head (in `vision_training/models/self_supervised/projection_head.py`) is a `FullyConnectedBlock` holding one linear layer, normalization and ReLU for every size of `fully_connected` except the last, then a plain linear `embedding_layer` to the last size, with no normalization nor activation so that the embedding can take any direction. `print_projection_head_summary()` adds its panel to the model summary.

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
The tracked metrics depend on the method (`METHOD_TRACKED_METRICS` of the trainer): the loss of the method, the batch metrics it returns and the embedding standard deviation, which watches the collapse of every method.

**Example: the tracked metrics of both methods**
```text
simclr:       {'total_loss': 'Total Loss', 'info_nce_loss': 'InfoNCE Loss', 'positive_pair_accuracy': 'Positive Pair Accuracy',
               'embedding_standard_deviation': 'Embedding Standard Deviation'}
barlow_twins: {'total_loss': 'Total Loss', 'barlow_twins_loss': 'Barlow Twins Loss', 'cross_correlation_diagonal': 'Diagonal Correlation',
               'cross_correlation_off_diagonal': 'Off-Diagonal Correlation', 'embedding_standard_deviation': 'Embedding Standard Deviation'}
```

### Step 2.19 — `print_experiment_information()` (→ `describe_split()` → `count_split_samples()`)
`describe_split()` counts the images and the categories of every manifest.

**Example output**
```text
--------------------------------------
=== Experiment Launch Information ===
--------------------------------------
- Run Name                : SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2
- Experiment Folder       : SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2
- Tensorboard Directory   : Tensorboard
- Weights Directory       : Weights
- Configuration File      : training_configuration.yaml
- Data Record             : data.json
- Train Data Folder       : train (500000 images, 9 categories)
- Validation Data Folder  : validation (50000 images, 9 categories)
- Test Data Folder        : test (50000 images, 9 categories)
- Iterations Per Epoch    : 3906
- Batch Size              : 128
- Total Iterations        : 10000000000
- Save And Evaluate Every : 250 iterations
- Save Only Every         : 125 iterations
- Learning Rate           : 0.0005
- Optimizer               : adam (weight decay 0.0)
- Learning Rate Schedule  : constant (warm-up 0 iterations)
- Gradient Clipping Norm  : None
- Weight Moving Average   : Off
- Mixed Precision         : True
- Resume Training         : True
- Validation Iter. Ratio  : 5
- MLflow Tracking         : Off
```
The 3906 iterations per epoch are 500000 / 128 rounded down: the self-supervised trainer sets `DROPS_LAST_TRAINING_BATCH`, since its training loader drops its last incomplete batch, while the other tasks round up, except the classification and sequence tasks whose training loader drops it too, for a batch normalized classification head or a view loss (see `classification_workflow.md`, Step 2.10).

### Step 2.20 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The backbone panel, the Projection Head panel of Step 2.15, the receptive field and the size of every component.

**Example output** (shortened)
```text
╭──────────────────────────────────── Backbone Receptive Field ────────────────────────────────────╮
│ Input Image Size : 256x256                                                                       │
│ ...                                                                                              │
│ Block 6 - Convolution 2  : Output 8x8x128 | Stride 32 | Receptive Field 282                      │
│                                                                                                  │
│ Squeeze-and-Excitation adds global image context.                                                │
│ This is not captured by the convolutional receptive field above.                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 438,368                       │
│   - Trainable Parameters : 438,368                       │
│   - Estimated Size       : 1.76 MB                       │
│ Projection Head                                          │
│   - Parameters           : 132,736                       │
│   - Trainable Parameters : 132,736                       │
│   - Estimated Size       : 0.54 MB                       │
│ Total Model                                              │
│   - Parameters           : 571,104                       │
│   - Trainable Parameters : 571,104                       │
│   - Estimated Size       : 2.29 MB                       │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.21 — `view_input_batches()`
Only with `diagnosis_modes.view_input_images`, then the program exits. `show_input_samples()` draws the two views of a training image with `create_views(describe_views=True)` and shows them in two windows, "View 1" and "View 2", each next to a black information panel holding the class of the image and the transformations drawn for the view with their values (the lines of Step 3.12). The validation images get their views drawn the same way, so only the training image is shown. It needs a display and was not run for this document.

### Step 2.22 — `restore_last_model()` → `get_last_checkpoint_iteration()` / `load_model()`
**Example output** (a new run folder)
```text
-----------------------------------------------------------
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.
-----------------------------------------------------------
```

---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of `configuration_synthetic_shapes.yaml` shortened to 64 pixel images, batch size 8, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, a test split of 1000 images and the MLflow tracking on. Its run name is `SSL_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2`. The construction took 4.8 s and the whole run 27.5 s. The Barlow Twins examples come from the same run with `ProjectionHead.method: barlow_twins` (run name ending `_BT-OD0.005`).

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1): from `start_iteration` to `number_iterations`, the total of the run, each iteration takes its training step, then saves its checkpoint, runs its validation batch and prints its summary on the iterations set for each, and evaluates its checkpoint last.

**Example: what happens at each iteration of the run**, in the order of the rows within an iteration, after its training step

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | ✔ | nothing: iteration 12 is saved |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |
| Test evaluation | | | | | | ✔ | | | | | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()`
**Example: the curves after the run**
```text
Tensorboard/Train/        Total Loss, InfoNCE Loss, Positive Pair Accuracy, Embedding Standard Deviation, Learning Rate   (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, InfoNCE Loss, Positive Pair Accuracy, Embedding Standard Deviation                  (iteration 10)
```

### Step 3.3 — `initialize_training_loop_state()` + `MetricTracker.create_trackers()`
**Example: the trackers of an interval**
```text
{'start_time': ..., 'interval_batches': 0, 'total_loss': 0.0, 'info_nce_loss': 0.0, 'positive_pair_accuracy': 0.0,
 'embedding_standard_deviation': 0.0}
```

### Step 3.4 — `start_experiment_tracking()`
The MLflow run is created in the experiment `self_supervised` (the task name), in the local database of the experiments root folder.

**Example output**
```text
MLflow Run Created In The Experiment 'self_supervised' : 73a1f377fd644212b312f006de1614c6 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.5 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12, so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6). The checkpoint holds the whole model, projection head included, so that the training can resume; only the backbone is transferred to other runs.

**Example: the checkpoints and the keys of the last one**
```text
Checkpoint Folder :
 - Iteration_3
 - Model Saved In : model_000003.pt

Weights/Iteration_3/model_000003.pt   6,967,751 bytes      Weights/Iteration_6/model_000006.pt   6,967,751 bytes
Weights/Iteration_9/model_000009.pt   6,967,751 bytes      Weights/Iteration_12/model_000012.pt  6,967,751 bytes
Weights/full-checkpoint               207 bytes: model_checkpoint_path: "Iteration_12", all_model_checkpoint_paths 3, 6, 9, 12
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone']
model_state: 108 tensors, the first 'backbone.convolution_blocks.0.squeeze_and_excitation_layers.0.first_adaptive_recalibrator.weight'
format_version: 2   task: 'self_supervised'   pretrained_backbone: None
```

### Step 3.6 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration, after its checkpoint and its summary (see `classification_workflow.md`, Step 3.7), see Stage 4.

### Step 3.7 — `get_next_batch()` → `ClassificationDataset.__getitem__()`
The classification dataset reads every image with `read_image()` and resizes it (see `classification_workflow.md`, Steps 3.8 and 3.9): a batch whose image cannot be read is dropped, its error printed once, and the next batch is taken. The labels come with the batch but the training never reads them.

**Example: the first batch of the run**
```text
images               [8, 3, 64, 64] torch.uint8 in [0, 252]
labels               [8] torch.int64                  [6, 8, 1, 1, 0, 1, 8, 3]
sample_identifiers   [8] torch.int64                  [48833, 387539, 251512, 104290, 77064, 235375, 394844, 284659]
```

### Step 3.8 — `set_training_mode()`
`model.train()`: the batch normalization layers of the backbone and of the head use the statistics of the batch (see Step 4.9 for what this changes at evaluation).

### Step 3.9 — `run_model_iteration()`
The same function for the training and the validation batches (see `classification_workflow.md`, Step 3.11): prepare, augment, forward, sum of the `_loss` outputs into `total_loss`, batch metrics, tracking. With a single loss, the total loss is the loss of the method.

**Example: the first training batch**
```text
info_nce_loss = 3.5625   total_loss = 3.5625   positive_pair_accuracy = 0.0
```

### Step 3.10 — `prepare_batch()`
**Example: before and after**
```text
images  [8, 3, 64, 64] torch.uint8   -> [8, 3, 64, 64] torch.float32
labels  [8] torch.int64              -> [8] torch.int64
```

### Step 3.11 — `augment_batch()`
Returns the batch unchanged: the views are drawn in `forward_pass()` (Step 3.12), so that the validation batches, which skip `augment_batch()`, get random views too.

### Step 3.12 — `forward_pass()` → `ViewAugmenter.create_views()`
`forward_pass()` draws two random views of every image, then calls `SelfSupervisedModel.forward(input_tensor=first_views, second_view_tensor=second_views)`. Every transformation is drawn independently for each image and each view (but the kernel size of the motion blur, drawn once per batch, the configured upper bound being passed to kornia raised by 2 so that it is drawn too), so the two views of an image differ in crop, geometry, colours, lighting, blur, noise and erasing, the first view being nearly always blurred and only the second one solarized.

**Example: the views of the first batch and the transformations drawn for two images**
```text
first_views, second_views: [8, 3, 64, 64] torch.float32 in [0, 255]

image 0, view 1: random_resized_crop : 56x61 pixels at x 7, y 0 (83 % of the area)
                 perspective : corners moved by up to 5.5 pixels
                 colour_jitter : brightness x0.66, contrast x1.23, saturation x1.28, hue +24.8 degrees
                 gamma : gamma 1.09
                 linear_illumination : gradient up to +0.09
                 gaussian_blur : sigma 0.42
                 motion_blur : kernel 5, angle +141.8 degrees, direction +0.58
                 random_erasing : 21x19 pixels at x 13, y 4
image 0, view 2: random_resized_crop : 61x46 pixels at x 2, y 12 (69 % of the area)
                 up_down_flip
                 colour_jitter : brightness x0.77, contrast x1.01, saturation x1.18, hue -15.4 degrees
                 gamma : gamma 1.14
                 gaussian_noise
                 random_erasing : 16x14 pixels at x 10, y 42
image 1, view 1: random_resized_crop : 59x51 pixels at x 5, y 12 (73 % of the area)
                 left_right_flip
                 up_down_flip
                 grayscale
                 linear_illumination : gradient up to -0.24
                 gaussian_blur : sigma 0.71
image 1, view 2: random_resized_crop : 60x63 pixels at x 4, y 0 (92 % of the area)
                 haze : strength 0.33, airlight 0.79
                 solarize : threshold 0.50, addition +0.00
                 random_erasing : 22x8 pixels at x 5, y 16
```

### Step 3.13 — `SelfSupervisedModel.forward()` → `encode()`
Both views go through the model as one batch of 2N = 16 images, so that the batch normalization sees both views together; the embeddings are then split back into the first and second views. Under the bfloat16 autocast of mixed precision, the outputs are bfloat16.

**Example: the outputs of the components on the 16 views**
```text
backbone           [16, 128, 2, 2] torch.bfloat16
global_pooling     [16, 128] torch.bfloat16 in [0, 0.8]       the representations
projection_head    [16, 128] torch.bfloat16                   the embeddings, split into [8, 128] + [8, 128]
```

### Step 3.14 — `compute_method_loss()` → `compute_info_nce_loss()` (SimCLR)
In `vision_training/losses/self_supervised_losses.py`, computed in float32 outside the autocast. The 2N embeddings are L2-normalized and compared by cosine similarity, divided by the temperature. Every view is a classification among the 2N - 1 other views (its own similarity is masked), whose right answer is the other view of its image, at index i + N for a first view and i - N for a second view. The loss is the cross entropy of these 2N classifications: about ln(2N - 1) for random embeddings, here ln 15 = 2.7081. The positive pair accuracy is the share of views whose most similar view is their partner.

**Example: the first iteration**
```text
first and second embeddings: [8, 128] torch.bfloat16
view 0: cosine with its partner 0.15, mean cosine with the 14 other views 0.40, highest cosine 0.75 (view 7)
        -> its partner is not its most similar view, and its own row of the loss is about 4.06
info_nce_loss = 3.5625   positive_pair_accuracy = 0.0   (ln 15 = 2.7081 for random embeddings)
```

**Example: the 12 training iterations** (the learning rate stays at 0.0005)

| Iteration | InfoNCE loss | Positive pair accuracy | Gradient norm |
|---|---|---|---|
| 1 | 3.5625 | 0.0 | 58.49 |
| 2 | 2.3762 | 0.4375 | 50.39 |
| 3 | 2.2249 | 0.375 | 45.64 |
| 4 | 2.1200 | 0.375 | 42.02 |
| 5 | 2.4303 | 0.125 | 54.76 |
| 6 | 1.6858 | 0.5 | 29.31 |
| 7 | 1.6953 | 0.5 | 36.09 |
| 8 | 1.9359 | 0.375 | 32.87 |
| 9 | 2.2775 | 0.125 | 37.79 |
| 10 | 2.4104 | 0.3125 | 35.98 |
| 11 | 1.7270 | 0.375 | 27.31 |
| 12 | 1.7833 | 0.375 | 19.97 |

### Step 3.15 — `compute_embedding_standard_deviation()`
The standard deviation over the batch of every feature of the L2-normalized embeddings of the first views, averaged over the features and multiplied by the square root of the dimension D. Embeddings spread over the whole sphere give about 1, collapsed embeddings (the same embedding whatever the image) give 0. It is computed without gradient and only watches the training.

**Example: the training batches**
```text
embedding_standard_deviation between 0.73 and 0.78 over the 12 iterations
```

### Step 3.16 — `compute_method_loss()` → `compute_barlow_twins_loss()` (Barlow Twins)
Every embedding feature is normalized over the batch (zero mean, unit variance, `BARLOW_TWINS_NORMALIZATION_EPSILON` added to the variance), then the `[D, D]` cross-correlation C of the features of the first views with those of the second views is computed. The loss pushes its diagonal to 1 (every feature takes the same value on both views of an image) and its off-diagonal terms to 0 (the features carry different information):

```text
barlow_twins_loss = Σ_i (C_ii - 1)² + off_diagonal_weight · Σ_i≠j C_ij²
```

The diagonal correlation is the mean of the diagonal, and the off-diagonal correlation the mean absolute value of the D(D - 1) other terms.

**Example: the first iterations of the Barlow Twins run**
```text
cross-correlation C: [128, 128]
iteration 1: barlow_twins_loss = 150.8647   cross_correlation_diagonal = 0.0103   cross_correlation_off_diagonal = 0.2927
             embedding_standard_deviation = 0.7651   gradient norm = 4980.37
iterations 2 to 4: barlow_twins_loss = 108.62, 122.39, 159.85
```
With a diagonal near 0 the invariance term alone is close to D = 128, which explains the size of the loss and of its gradients compared with InfoNCE.

### Step 3.17 — `compute_batch_metrics()`
Returns the batch metrics given by the model with its loss (positive pair accuracy, or diagonal and off-diagonal correlations, and the embedding standard deviation), without any extra computation.

### Step 3.18 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
Adds the metrics of the batch to the trackers of the interval and writes them to `Tensorboard/Train` at every iteration.

### Step 3.19 — `training_loss.backward()` + `step_optimizer()`
As for every task (see `classification_workflow.md`). No gradient clipping is set here, so the gradient norms of Step 3.14 are applied as they are.

### Step 3.20 — validation pass
Every 10 iterations, one validation batch runs through `run_model_iteration()` without gradient, with new random views, the model in evaluation mode.

**Example: the validation batch of iteration 10**
```text
info_nce_loss = 2.708 (= ln 15)   positive_pair_accuracy = 0.1875   embedding_standard_deviation = 0.0012
```
In evaluation mode the batch normalization uses its running statistics, which are not yet meaningful after 10 updates: all the embeddings are almost the same (standard deviation 0.0012), so every view is as similar to all the others and the loss is exactly the one of random embeddings. Step 4.9 explains it.

### Step 3.21 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()` + `report_dropped_batches()`
The sums of the interval are divided by the number of batches it ran. After the summary, the batches dropped since the last one because their loading failed are counted by split, only when there are some (see `classification_workflow.md`, Step 3.24).

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 10.2836, 'info_nce_loss': 10.2836, 'positive_pair_accuracy': 1.1875, 'embedding_standard_deviation': 2.9867}
after:  {'interval_batches': 4, 'total_loss': 2.5709, 'info_nce_loss': 2.5709, 'positive_pair_accuracy': 0.2969, 'embedding_standard_deviation': 0.7467}
MLflow, step 4: {'train/Total Loss': 2.5709, 'train/InfoNCE Loss': 2.5709, 'train/Positive Pair Accuracy': 0.2969,
                 'train/Embedding Standard Deviation': 0.7467, 'train/throughput': 4.6864,
                 'train/interval_seconds': 6.8282, 'learning_rate': 0.0005}
MLflow, step 12: train Total Loss 2.0495, Positive Pair Accuracy 0.2969, Embedding Standard Deviation 0.7493,
                 validation Total Loss 2.708, Positive Pair Accuracy 0.1875, Embedding Standard Deviation 0.0012,
                 throughput 11.5912, interval_seconds 2.7607
```

**Example output: iterations 8 and 12**
```text
========================================================================================================================
|     Iteration: 8     |      Total Images: 64     |    Interval Images: 32     |  Throughput: 4.0 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   1.9368  -0.6341 Best
Moving Average of Train InfoNCE Loss                   :   1.9368  -0.6341 Best
Moving Average of Train Positive Pair Accuracy         :   0.3750  +0.0781 Best
Moving Average of Train Embedding Standard Deviation   :   0.7506  +0.0039 Best
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:07.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
========================================================================================================================
...
========================================================================================================================
|    Iteration: 12     |      Total Images: 96     |    Interval Images: 32     |  Throughput: 11.6 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   2.0495  +0.1127
Moving Average of Valid Total Loss                     :   2.7080
Moving Average of Train InfoNCE Loss                   :   2.0495  +0.1127
Moving Average of Valid InfoNCE Loss                   :   2.7080
Moving Average of Train Positive Pair Accuracy         :   0.2969  -0.0781
Moving Average of Valid Positive Pair Accuracy         :   0.1875
Moving Average of Train Embedding Standard Deviation   :   0.7493  -0.0013
Moving Average of Valid Embedding Standard Deviation   :   0.0012
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:02.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```
At iteration 12 no metric beats its best value, so no line ends with "Best".

**Example output: the first interval of the Barlow Twins run** (throughput 3.7 Images Per Second)
```text
Moving Average of Train Total Loss                     : 135.4303
Moving Average of Train Barlow Twins Loss              : 135.4303
Moving Average of Train Diagonal Correlation           :   0.0914
Moving Average of Train Off-Diagonal Correlation       :   0.3017
Moving Average of Train Embedding Standard Deviation   :   0.7511
```

### Step 3.22 — end of the loop
The checkpoint of iteration 12 is already saved and evaluated inside its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 1000 Test Images · 2026-10-16 19:33 ───────────────      <- the evaluation of iteration 12, last in the iteration
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its test split (1000 images of 64×64 pixels, batch size 8), with `per_class_report: True`, `number_sample_images: 8` and `number_neighbours: 20`. The outputs of iteration 6 were captured when its checkpoint was still saved before its training step, after 5 steps; it now holds 6 steps.

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1). The test images are encoded as they are, without views, as in deployment.

**Example output: the progress bar** (1000 images, batch size 8)
```text
Test Evaluation Iteration 6: 100%|██████████| 125/125
```

### Step 4.2 — `create_evaluation_accumulator()`
The k-NN evaluation compares every test image with all the others, so the representations of the whole split are kept until the end of the loop.

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
images            [8, 3, 64, 64] torch.float32 in [0, 255]
representations   [8, 128] torch.float32 in [0, 0.1]
embeddings        [8, 128]
```

### Step 4.5 — `accumulate_evaluation_batch()`
Keeps the representations of the kept samples (`.detach().float().cpu()`), their labels and their identifiers. The embeddings are not kept: the representation is what is evaluated.

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
For each fixed test image of Step 2.11: the image next to two random views of it, captioned with its class (or `unlabelled`) and the cosine similarity of the representations of both views, which a backbone invariant to the views brings close to 1. The views are drawn from the seed `SAMPLE_VIEWS_SEED` (20000) plus the sample identifier, inside `torch.random.fork_rng()`, so that every evaluation compares the same two views and the random state of the training is left unchanged. `add_caption_bar()` adds a bar under the image.

**Example: the sample images of iteration 12**
```text
image and views: (64, 192, 3)  ->  with the caption bar: (88, 192, 3), wider for a long caption (88, 195, 3), (88, 196, 3)
captions:        'circle | Views Similarity 1.000', 'plus_sign | Views Similarity 1.000', 'pentagon | Views Similarity 1.000'
files:           Evaluation/Iteration_12/Samples/test_000000.png ... (8 images of 192×88 pixels)
TensorBoard:     Test images Samples/test_000000, ...
```
A similarity of 1.000 here does not mean the backbone is invariant: Step 4.9 shows that every image gets almost the same representation at this point.

### Step 4.7 — `compute_evaluation_results()`
Concatenates the representations and the labels of the split, then computes:
- the **effective rank** (Step 4.10) and the **representation standard deviation** (Step 3.15 applied to the representations) of every test image, labelled or not;
- with at least two labelled test images, the **k-NN metrics** (Step 4.8), the headline metric being then the k-NN accuracy; without them, the k-NN is skipped and the headline metric is the effective rank (Step 5.6).

**Example: the global metrics of both checkpoints**

| Iteration | k-NN Accuracy | k-NN Balanced Accuracy | k-NN Macro F1 | Effective Rank | Representation Standard Deviation |
|---|---|---|---|---|---|
| 6 | 0.0930 | 0.0911 | 0.0910 | 1.0118 | 0.0002 |
| 12 | 0.1070 | 0.1047 | 0.1021 | 1.0535 | 0.0012 |

```text
notes:         'Labelled Test Images : 1000 of 1000', 'k-NN : 20 neighbours among the other labelled test images',
               'Sample Images : 8 fixed test images'
class_scalars: 27 (k-NN Precision, Recall and F1 of the 9 classes)
```
With 9 classes, chance is about 0.11.

### Step 4.8 — `compute_nearest_neighbour_results()` → `compute_nearest_neighbour_predictions()`
Classifies every labelled test image by its neighbours among the other labelled test images, then scores it like a classifier (in `vision_training/evaluation/self_supervised/representation_metrics.py`):
- The representations are L2-normalized and compared by cosine similarity, in chunks of `SIMILARITY_CHUNK_SIZE` (1024) rows, the similarity of an image with itself being excluded (leave-one-out).
- The k = min(20, N - 1) most similar images vote for their class with the weight exp(similarity / `NEAREST_NEIGHBOUR_TEMPERATURE`), the temperature being 0.07, the votes being summed per class (`scatter_add`); the class with the most votes is the prediction.
- `compute_confusion_matrix()` and `compute_metrics_from_confusion_matrix()` give the accuracy, the balanced accuracy, the macro F1 and the precision, recall and F1 of every class; `write_class_metrics_file()` saves them in `class_metrics.csv`.

**Example: image 0 of the test split at iteration 12** (label 8, octagon)
```text
top 20 similarities: all 1.0
neighbour labels:    [2, 5, 2, 2, 8, 3, 6, 4, 1, 7, 8, 1, 1, 7, 7, 6, 2, 0, 3, 0]
votes per class:     [3.20e6, 4.80e6, 6.40e6, 3.20e6, 1.60e6, 1.60e6, 3.20e6, 4.80e6, 3.20e6]   (exp(1 / 0.07) = 1.6e6 per neighbour)
prediction:          2 (triangle), wrong
accuracy over the split: 0.107   samples per class: [91, 118, 125, 97, 99, 117, 126, 113, 114]
```
With all similarities at 1.0, every neighbour weighs the same and the vote is a plain count among neighbours chosen almost at random.

**Example: `class_metrics.csv` of iteration 12** (first rows)
```text
class_index,class,samples,precision,recall,f1
0,square,91,0.141176,0.131868,0.136364
...
```

### Step 4.9 — why the representations look collapsed
The test images are encoded in evaluation mode, so the batch normalization layers use their running statistics instead of the statistics of the batch. After 12 iterations they have been updated only 13 times (`num_batches_tracked` 13) at momentum 0.1, so they are still far from the statistics of the data (the running variance of the last backbone layer averages 0.27), and the normalized features of all images end up nearly identical:

**Example: one test batch at iteration 12, in both modes**
```text
evaluation mode (running statistics):  mean pairwise cosine 1.0     feature standard deviation 3.8e-5
training mode (batch statistics):      mean pairwise cosine 0.739   feature standard deviation 0.115
```
This explains the effective rank near 1, the views similarity of 1.000, the validation loss of exactly ln 15 and the k-NN near chance. The representations being positive after the ReLU and the pooling of the backbone also brings their directions closer. A longer training gives the running statistics time to settle.

### Step 4.10 — `compute_effective_rank()`
The RankMe effective rank of the `[N, F]` representations: the singular values are normalized into a distribution p, and the effective rank is the exponential of its entropy (`EFFECTIVE_RANK_EPSILON` 1e-12 avoiding log 0, NaN when the result is not finite). It goes from 1, when all representations lie along one direction, to F when they use every direction equally.

**Example: iteration 12**
```text
representations:  [1000, 128] in [0, 0.137], mean 0.0245
singular values:  12.79, 0.0152, 0.0081, 0.0030, 0.0027, ...   (sum 12.85)
p_1 = 12.79 / 12.85 = 0.9947   entropy = 0.0521   effective rank = exp(0.0521) = 1.0535
```

### Step 4.11 — the per-class table and scalars
The table "Per Class k-NN Metrics" lists the classes with their number of labelled test images, precision, recall and F1, the worst F1 first. Each class also gives three TensorBoard and MLflow scalars, `k-NN Precision/<class>`, `k-NN Recall/<class>` and `k-NN F1/<class>`.

**Example: `Evaluation/Iteration_12/class_metrics.csv` and the scalars of one class** (the table itself is in the block of Step 4.12)
```text
class_index,class,samples,precision,recall,f1
0,square,91,0.141176,0.131868,0.136364
1,circle,118,0.136364,0.127119,0.131579
2,triangle,125,0.123223,0.208000,0.154762
3,rectangle,97,0.039474,0.030928,0.034682
...
8,octagon,114,0.049383,0.035088,0.041026

Tensorboard/Test tags of circle:  k-NN Precision/circle, k-NN Recall/circle, k-NN F1/circle
k-NN Precision/circle:            step 6 0.0642, step 12 0.1364
```
The file keeps the class order of the label dictionary, while the console table sorts the classes by F1.

### Step 4.12 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 1000 Test Images · 2026-10-16 19:33 ─────────────────────────────────────────────────────────────────────
  k-NN Accuracy                       0.1070   k-NN Balanced Accuracy   0.1047
  k-NN Macro F1                       0.1021   Effective Rank           1.0535
  Representation Standard Deviation   0.0012
  Labelled Test Images : 1000 of 1000
  k-NN : 20 neighbours among the other labelled test images
  Sample Images : 8 fixed test images

  Per Class k-NN Metrics
   Class       Samples   Precision   Recall       F1
  ───────────────────────────────────────────────────
   rectangle        97      0.0395   0.0309   0.0347
   octagon         114      0.0494   0.0351   0.0410
   star            113      0.0744   0.0796   0.0769
   pentagon         99      0.0842   0.0808   0.0825
   plus_sign       126      0.1043   0.0952   0.0996
   circle          118      0.1364   0.1271   0.1316
   square           91      0.1412   0.1319   0.1364
   triangle        125      0.1232   0.2080   0.1548
   hexagon         117      0.1698   0.1538   0.1614
  k-nearest neighbour classification (20 neighbours) of the labelled test images by the other ones. Worst F1 first, ties
  in class order. — : no labelled test image of the class.
  - Results : file:///.../Results.txt
  - Class Metrics : file:///.../Evaluation/Iteration_12/class_metrics.csv
  - Sample Images : file:///.../Evaluation/Iteration_12/Samples
```

**Example: the run folder, TensorBoard and MLflow after the two evaluations**
```text
runs/SSL_IN-KAR-64_..._SIMCLR-T0.2/
    data.json  label_dictionary.json  mlflow_run.json  Results.txt  training_configuration.yaml
    Evaluation/Iteration_6/ and Iteration_12/   class_metrics.csv  Samples/ (8)
    Tensorboard/Train, Validation, Test
    Weights/Iteration_3, 6, 9, 12/  full-checkpoint
Tensorboard/Test: 32 scalars (5 global + 27 per class) at steps 6 and 12, 8 images Samples/test_...
    scalar k-NN Accuracy    6: 0.093, 12: 0.107
    scalar Effective Rank   6: 1.0118, 12: 1.0535
MLflow (runs/mlflow.db): 44 metric keys, 91 rows: train/*, validation/*, test/<global metric>, test/best_k-NN Accuracy,
        27 test_class/k-NN <Metric>/<class>, learning_rate
    parameters: the flattened configuration, e.g. Data.view_augmentation.rotation.maximum_degrees = 180.0
    tags: best_iteration 12, git_commit, git_dirty, host, pretrained_backbone None, run_folder, run_name, task
    artifacts: configuration/training_configuration.yaml, data/data.json, evaluation/iteration_6/ and iteration_12/
               class_metrics.csv, metric_names.json, Results.txt
```
With the moving average of the weights on (`Optimization.exponential_moving_average`), the averaged weights are evaluated the same way after the trained ones, in `Evaluation/Iteration_<n>/EMA/` and `Tensorboard/Test_EMA`, with the same sample views.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12, captured when those of iterations 3, 6 and 9 were still saved before their training step) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of 200 test images (`test_200`). A second evaluation of the checkpoint of iteration 12 uses the same 200 images without their categories (`test_unlabelled_200`).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5): set `run_name` to the self-supervised run, and `test_data_folder` to another split if needed. The split does not need labels: without them, only the effective rank and the representation standard deviation are computed (Step 5.6).

### Step 5.2 — trainer construction
The script loads `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it, points it to the output folder `evaluations/`, the evaluated split and the batch size, calls `disable_pretrained_settings()` (nothing to change here, the run had no pretrained backbone) and builds a `SelfSupervisedTrainer`.

**Example output** (shortened)
```text
- Test Data Folder        : test_200 (200 images, 9 categories)
- Iterations Per Epoch    : 31250
- Batch Size              : 16
- MLflow Tracking         : sqlite:///.../runs/mlflow.db
```
The panel names the backend of the evaluated run, where the evaluation is tracked (Step 5.4): `tracking_root` keeps the experiments root folder of the run while `experiment_folder` points to the output folder.

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
The MLflow run of the evaluation, named `<run name>-evaluation`, is created in the backend of the evaluated run and tagged with the training run it evaluates.

**Example output**
```text
MLflow Run Created In The Experiment 'self_supervised' : 6e8ccfa7c4f843079a00356e7c5d084c (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
tags: evaluation_of 73a1f377fd644212b312f006de1614c6, evaluated_iterations "12, 9, 6, 3", test_data_folder, best_iteration 12
metrics: 33 keys, 132 rows
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output**
```text
Evaluation Summary Of 4 Checkpoints : file:///.../evaluations/SSL_IN-KAR-64_..._SIMCLR-T0.2/Evaluation/evaluation_summary.csv
  - Best Checkpoint : iteration 12 (k-NN Accuracy 0.1200)

iteration,k-NN Accuracy,k-NN Balanced Accuracy,k-NN Macro F1,Effective Rank,Representation Standard Deviation, ...
12,0.12,0.10986599953991256,0.10130809773488435,1.0506236098178765,0.0011394398752599955, ...
9,0.105,0.0937...,0.0876...,1.0196...,0.000338..., ...
6,0.07,0.0640...,0.0577...,1.0112...,0.000159..., ...
3,0.105,0.0874...,0.0641...,1.0074...,0.000112..., ...
```
The file has 33 columns: the iteration, the 5 global metrics and the 27 per-class metrics.

**Example: the output folder**
```text
evaluations/SSL_IN-KAR-64_..._SIMCLR-T0.2/
    data.json  label_dictionary.json  Results.txt  training_configuration.yaml  Weights/ (empty)
    Evaluation/evaluation_summary.csv
    Evaluation/Iteration_3, 6, 9, 12/   class_metrics.csv  Samples/
    Tensorboard/Test
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 12 on the 200 images**
```text
-----------------------------
Evaluating : model_000012.pt
-----------------------------
Iteration 12 · 200 Test Images · 2026-10-16 19:38 ──────────────────────────────────────────────────────────────────────
  k-NN Accuracy                       0.1200   k-NN Balanced Accuracy   0.1099
  k-NN Macro F1                       0.1013   Effective Rank           1.0506
  Representation Standard Deviation   0.0011
  Labelled Test Images : 200 of 200
  k-NN : 20 neighbours among the other labelled test images
  Sample Images : 8 fixed test images

  Per Class k-NN Metrics
   Class       Samples   Precision   Recall       F1
  ───────────────────────────────────────────────────
   rectangle        14      0.0000   0.0000   0.0000
   octagon          21      0.0833   0.0476   0.0606
   triangle         28      0.0571   0.0714   0.0635
   circle           20      0.1111   0.0500   0.0690
   square           16      0.1250   0.0625   0.0833
   pentagon         23      0.1250   0.1739   0.1455
   plus_sign        27      0.2500   0.1111   0.1538
   hexagon          27      0.1277   0.2222   0.1622
   star             24      0.1333   0.2500   0.1739
  ...
```

**Example: the global metrics of the four checkpoints**

| Iteration | k-NN Accuracy | k-NN Balanced Accuracy | k-NN Macro F1 | Effective Rank | Representation Standard Deviation |
|---|---|---|---|---|---|
| 12 | 0.1200 | 0.1099 | 0.1013 | 1.0506 | 0.0011 |
| 9 | 0.1050 | 0.0937 | 0.0876 | 1.0196 | 0.0003 |
| 6 | 0.0700 | 0.0640 | 0.0577 | 1.0112 | 0.0002 |
| 3 | 0.1050 | 0.0874 | 0.0641 | 1.0074 | 0.0001 |

The effective rank and the spread of the representations grow with the training, as the running statistics of the batch normalization settle (Step 4.9).

**Example output: the same 200 images without labels**
```text
200 samples without a single mapped category kept without class : labels.json        <- yellow, while building the loaders
...
Iteration 12 · 200 Test Images · 2026-10-16 19:39 ──────────────────────────────────────────────────────────────────────
  Effective Rank   1.0506   Representation Standard Deviation   0.0011
  Labelled Test Images : 0 of 200
  k-NN : skipped, fewer than two labelled test images
  Sample Images : 8 fixed test images
  - Results : file:///.../Results.txt
  - Sample Images : file:///.../Evaluation/Iteration_12/Samples
-------------------------------------------------------------
Evaluation Summary Of 1 Checkpoints : file:///.../Evaluation/evaluation_summary.csv
-------------------------------------------------------------
  - Best Checkpoint : iteration 12 (Effective Rank 1.0506)
```
The images are the same as in `test_200`, so the effective rank is the same; without labels there is no class table nor `class_metrics.csv`, the sample images are captioned `unlabelled`, and the MLflow run of the evaluation has 3 metric keys.
