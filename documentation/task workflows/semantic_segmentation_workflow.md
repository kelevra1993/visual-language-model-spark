# Semantic Segmentation Workflow

This document follows one semantic segmentation run from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`; here they keep a short explanation and the real output of the segmentation run, and the steps specific to the segmentation are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training:
- **Stages 1 and 2**: `configurations/semantic_segmentation/configuration_synthetic_shapes.yaml` (synthetic shapes, 9 shape classes plus the background, 512 pixel images, batch size 16), with the run folder redirected to a temporary folder. The dataset of `configuration_example.yaml` is not on this machine.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel images, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the predicted masks saved and the MLflow tracking turned on. After 12 iterations the network mostly predicts the background, so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

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
│   trainer of its task, builds it and starts the training loop. Give a semantic_segmentation configuration to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, SegmentationDecoder, SegmentationLoss), converts the types, adds label_mode and
│   │   label_dictionary to 'Data' and runs the checks.
│   │
│   ├── load_configuration(configuration_path)
│   │       Plain YAML reader, no conversion.
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {mask value: class index} and adds 'background': 0, since the segmenter predicts
│   │       the background as class 0 (like the object detector). "binary" mode gives {background: 0, foreground: 1}.
│   └── check_configuration()
│       ├── common checks                         experiment, data, backbone, optimization (see classification)
│       └── check_segmentation_configuration()
│               ├── data:       Native loader, classes 0..K-1 with the background at 0, categories that are mask values
│               │               (1 to 254), no detection block, tiling scale > 0 and overlap smaller than a tile
│               ├── no sequence settings (frame shuffling, sequence normalization)
│               ├── decoder:    upsampling (transposed/bilinear), one level per skip connection of the backbone,
│               │               modules, normalization, head dropout, deep supervision weights, pretrained settings
│               ├── loss:       non-negative weights of the cross entropy, Dice, Lovász and boundary losses (not all
│               │               the region losses at 0), focal gamma, label smoothing, class weighting, boundary distance
│               └── evaluation: save_predicted_masks, original_resolution, boundary_tolerance
│
├── get_trainer_class(task)  -> SegmentationTrainer
│
├── SegmentationTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   Stores the segmentation settings its hooks read: the SegmentationLoss section, the deep supervision weights,
│   │   Data.tiling, and the evaluation settings save_predicted_masks, original_resolution and boundary_tolerance.
│   │
│   └── BaseTrainer.__init__()
│       │   Same order as for every task: device, run name, settings, augmenter, paths, loaders, sample images, model,
│       │   pretrained components, optimizer, metric tracker, printouts, restoration.
│       │
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter() / setup_training_paths()
│       │       The run name ends with the decoder (upsampling, normalization, D<layers>x<channels> of every level) and
│       │       the classes (MC<foreground classes>), plus TL<scale> when the run trains on tiles.
│       ├── build_dataloaders()
│       │   │   Builds the three loaders, keeps class_names (background first) and class_colours (one BGR colour per
│       │   │   class, for the overlays and the sample images).
│       │   │
│       │   └── get_segmentation_dataloaders()
│       │       │   One SegmentationDataset per split from <folder>/labels.json, <folder>/data/ and <folder>/masks/.
│       │       │   With tiling, the training split cuts random tiles, the validation split fixed tiles, and the test
│       │       │   split keeps whole scaled images, one per batch.
│       │       │
│       │       ├── SegmentationDataset.__init__() -> build_class_lookup_table()
│       │       │       Reads the manifest (every sample has a file_name and a mask_file_name), shuffles it once, and
│       │       │       builds the [256] table turning every mask value into a class index (0 background, 255 ignored).
│       │       └── build_native_loader()
│       │               Same torch DataLoader as the classification.
│       ├── get_test_sample_classes() + choose_sample_identifiers()
│       │       Counting the classes of every test mask would be slow, so every test image gets an empty class list,
│       │       and the sample images are evenly spread over the test split.
│       ├── build_model()
│       │   │   Creates the SemanticSegmenter from the registry, then copies the class weights of its cross entropy.
│       │   │
│       │   ├── SemanticSegmenter.__init__()
│       │   │   ├── build_backbone()              the shared backbone, the downsampling path of the U-Net
│       │   │   ├── SegmentationDecoder()         one upsampling layer and one ConvolutionBlock per level, from the
│       │   │   │                                 deepest skip connection to the input resolution
│       │   │   ├── segmentation_head             Dropout2d + 1x1 convolution to one logit per class
│       │   │   ├── deep_supervision_heads        1x1 convolutions on the levels of deep_supervision_weights (none here)
│       │   │   └── class_weights buffer          ones, overwritten below
│       │   └── compute_class_weights()
│       │           Ones, or with class_weighting "balanced" the weights of get_class_pixel_counts(), which counts the
│       │           pixels of every class in the training masks once and caches them in class_pixel_counts.json.
│       ├── initialise_pretrained_components()    Backbone and SegmentationDecoder can come from other runs
│       ├── build_optimizer()
│       ├── get_tracked_metrics() + MetricTracker.__init__()
│       │       total loss, the losses whose weight is not 0 (cross entropy, Dice, Lovász, boundary), the deep
│       │       supervision loss when there are heads, pixel accuracy and mean IoU.
│       ├── print_experiment_information() -> describe_split() -> count_split_samples()
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       │       The summary adds a decoder panel: the bottleneck, then for every level its skip connection,
│       │       convolutions and output, the head dropout and the size of the class logits.
│       ├── view_input_batches()  (only with view_input_images, then exits)
│       │       Each image with its mask painted over it, captioned with the classes it holds.
│       └── restore_last_model() -> get_last_checkpoint_iteration() / load_model()
│               load_model() is overridden to set the class weights of the current configuration again after loading,
│               since the loss settings may change between two resumptions.
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── setup_tensorboard_writers() / initialize_training_loop_state() / start_experiment_tracking()
        ├── for each training_iteration:
        │   ├── save_model() + run_test_evaluation() when due                    -> [STAGE 4]
        │   ├── get_next_batch()
        │   │   └── SegmentationDataset.__getitem__()
        │   │           Reads the image and its mask, resizes the image (bilinear) and the mask (nearest, padding
        │   │           at 255) with the same letterbox, or cuts a tile with tiling, then converts the mask values to
        │   │           class indices through the lookup table. Returns images [B, 3, H, W] uint8, masks [B, H, W] int64,
        │   │           sample_identifiers [B].
        │   ├── set_training_mode()
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()                 images to float32, masks to int64 on the device
        │   │   ├── augment_batch()                 the masks follow every geometric augmentation (nearest
        │   │   │                                   interpolation, the pixels uncovered by a rotation ignored)
        │   │   ├── forward_pass() -> SemanticSegmenter.forward(masks=...)
        │   │   │   ├── Backbone.forward(return_pre_pooling_outputs=True)   final feature map + skip connections
        │   │   │   ├── SegmentationDecoder.forward()                       upsample, concatenate the skip, refine
        │   │   │   ├── segmentation_head                                   class_logits [B, K, H, W]
        │   │   │   └── compute_mask_losses()
        │   │   │           compute_segmentation_cross_entropy() (weighted, focal factor, label smoothing),
        │   │   │           compute_dice_loss(), compute_lovasz_softmax_loss(), compute_boundary_loss(), each times
        │   │   │           its weight and left out at 0; the same for every deep supervision head.
        │   │   ├── compute_batch_metrics()
        │   │   │       compute_pixel_confusion_matrix() of the argmax of the logits against the masks, then
        │   │   │       compute_segmentation_metrics(): the pixel accuracy and the mean IoU of the batch.
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()
        │   ├── validation pass every validation_iteration_ratio iterations
        │   └── average_trackers() + log_interval_metrics() + print_interval_summary()
        ├── final save_model() + run_test_evaluation()
        └── close the writers, finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
├── create_evaluation_accumulator()
│       A [K, K] pixel confusion matrix and [K, 6] boundary counts, both on the device, the number of images, and the
│       Predicted_Masks folder of the checkpoint.
├── for each test batch:
│   ├── prepare_batch()
│   ├── predict_batch()
│   │       SemanticSegmenter.forward() without masks: class_logits and class_probabilities. A run trained on tiles
│   │       segments every whole scaled image tile by tile (segment_image_by_tiles), averaging the overlaps.
│   ├── accumulate_evaluation_batch()
│   │       Adds compute_pixel_confusion_matrix() and compute_boundary_counts() of the batch (at the network
│   │       resolution, or with original_resolution against every original mask after
│   │       restore_prediction_to_original_image()), and with save_predicted_masks writes one class index PNG per image.
│   └── save_sample_images() -> build_sample_image()
│           Four panels per fixed image: the input, the true mask and the predicted mask painted over it, the error
│           map, with the pixel accuracy of the image and a legend of its classes.
├── compute_evaluation_results()
│   ├── compute_segmentation_metrics()       pixel accuracy, IoU and Dice of every class, mean IoU, mean Dice,
│   │                                        frequency-weighted IoU, precision and recall
│   ├── compute_boundary_metrics()           boundary IoU and boundary F1 of every class, and their means
│   ├── build_segmentation_class_rows() + write_class_metrics_file()           class_metrics.csv
│   ├── save_confusion_matrix_table()                                          confusion_matrix.csv (pixel counts)
│   ├── save_confusion_matrix_image()   (only with per_class_report)          confusion_matrix.png (shares only)
│   └── build_segmentation_class_table() + build_segmentation_class_scalars()
└── report_evaluation() -> report_evaluation_results() + track_evaluation()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task: loads training_configuration.yaml, builds the SegmentationTrainer in the output folder, then
    evaluate_checkpoints() -> start_evaluation_tracking() + summarise_checkpoint_evaluations() -> run_test_evaluation()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/semantic_segmentation/configuration_synthetic_shapes.yaml`.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1): `parse_configuration_path()` reads the path of the YAML file from the command line (`--configuration_path`), then it loads the file, looks up the trainer of `ExperimentConfiguration.task` and starts its training loop. Nothing in it is specific to the segmentation.

### Step 1.2 — `load_experiment_configuration()`
It splits the YAML into the two dictionaries read by every later stage, converts the values YAML reads with the wrong type, adds `label_mode` and `label_dictionary` to `Data` (Step 1.4) and runs the checks (Step 1.5). The segmentation has two sections of its own, `SegmentationDecoder` (the upsampling path of the U-Net) and `SegmentationLoss`, its `Data` section has a `tiling` subsection, and its `evaluation` settings differ from the classification ones (no top-k, no thresholds, no error images).

**Example: the sections and the evaluation settings**
```text
model_configuration keys: Optimization, Data, Backbone, SegmentationDecoder, SegmentationLoss
Data keys after loading:  DALI, loading, image_settings, tiling, label_mapping, data_augmentation, label_mode, label_dictionary
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'save_predicted_masks': False,
                           'original_resolution': False, 'boundary_tolerance': 2}
SegmentationDecoder:      {'upsampling': 'transposed', 'convolutions': {'5': [2, 64], '4': [2, 64], '3': [2, 32], '2': [2, 32],
                           '1': [1, 16]}, 'modules': {'squeeze_and_excite': False, 'residual_nets': False},
                           'feature_map_normalization': 'batch', 'dropout_rate': 0.0, 'deep_supervision_weights': {}, ...}
SegmentationLoss:         {'cross_entropy_weight': 1.0, 'focal_gamma': 0.0, 'label_smoothing': 0.0, 'class_weighting': 'none',
                           'dice_weight': 1.0, 'lovasz_weight': 0.0, 'boundary_weight': 0.0, 'boundary_distance': 3}
```

### Step 1.3 — `load_configuration()`
Plain YAML reader, as for every task: no conversion and no check.

**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'SegmentationDecoder', 'SegmentationLoss']
number_iterations: '1e10'      <- still a string
learning_rate:     0.001
```

### Step 1.4 — `build_label_dictionary()`
The mapping file gives the class index of every category, a category being the value of its pixels in the mask files (1 to 254: 0 is the background and 255 the unlabelled pixels). Since the segmenter predicts the background as class 0, `build_label_dictionary()` adds `'background': 0` when the file does not map it. Several categories can be merged into one class by giving them the same index. In `"binary"` mode every category becomes the foreground class 1.

**Example: the label dictionary of the synthetic shapes**
```text
{1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 'background': 0}
10 entries, 10 distinct classes -> 10 output channels (9 shapes + the background)
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_segmentation_configuration()`: the data settings (Native loader, background at 0, categories that are mask values, tiling), no sequence settings, the decoder (one level per skip connection of the backbone), the losses and the evaluation settings. Every problem is reported together, then the program exits.

**Example: output with `upsampling: "nearest"` and `dice_weight: -1.0` in the YAML**
```text
-----------------------------------------------------------------------------------------------
Invalid Configuration File : file:///.../configuration_wrong.yaml
-----------------------------------------------------------------------------------------------
  - 'SegmentationDecoder.upsampling' must be one of ['transposed', 'bilinear'], got 'nearest'.
  - 'SegmentationLoss.dice_weight' must be a number >= 0, got -1.0.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="semantic_segmentation") -> <class 'vision_training.trainers.segmentation_trainer.SegmentationTrainer'>
```


---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on `configurations/semantic_segmentation/configuration_synthetic_shapes.yaml` (512 pixel images, batch size 16), with the run folder redirected to a temporary folder. The whole construction took 0.3 s on CPU.

### Step 2.1 — `SegmentationTrainer.__init__()`
Before calling the shared constructor, the trainer stores the settings its hooks read: the `SegmentationLoss` section (the losses, the class weighting), the deep supervision weights of the decoder, the tiling settings, and the three evaluation settings: `save_predicted_masks` (a class index PNG per test image), `original_resolution` (score the predictions against the original masks instead of the resized ones) and `boundary_tolerance` (the distance in pixels within which a contour pixel is right).

**Example: the attributes set before `super().__init__()`**
```text
self.loss_configuration         = {'cross_entropy_weight': 1.0, 'focal_gamma': 0.0, 'label_smoothing': 0.0, 'class_weighting': 'none',
                                   'dice_weight': 1.0, 'lovasz_weight': 0.0, 'boundary_weight': 0.0, 'boundary_distance': 3}
self.deep_supervision_weights   = {}
self.tiling_configuration       = {'activated': False, 'scale': 1.0, 'overlap': 128}
self.save_predicted_masks       = False
self.original_resolution        = False
self.boundary_tolerance         = 2
TASK_NAME = 'semantic_segmentation'   SAMPLE_NAME = 'Images'   COUNTS_FRAMES = False
```

### Step 2.2 — `BaseTrainer.__init__()`
The shared constructor (see `classification_workflow.md`, Step 2.2) checks the task, then runs the steps below in the same order as for every task.

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
The input and backbone tokens are those of every image task. `build_segmentation_sections()` adds the decoder: `UTC` (transposed convolutions, `UBL` for bilinear), the modules (`DSE`, `DRES`), the normalization (`DBN`), then `D<layers>x<channels>` of every level from the deepest one to the input resolution, `AUX<levels>` for deep supervision heads and the transfer tokens. Then come the classes: `MC9` counts the foreground classes, the background left out (`BIN` in binary mode). A run trained on tiles gets a `TL<scale>` section. The loss settings, the head dropout and the evaluation settings give no token, so they can change between two resumptions of a run.

**Example output**
```text
SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9
```

### Step 2.5 — `DataAugmenter()`
The same augmenter as the classification. Here the masks follow the images (Step 3.13).

**Example: the settings of the configuration**
```text
activated True   probability 0.4
6 of 18 types activated: random_crop, random_left_right_flip, random_rotation, brightness_contrast_saturation,
                         random_noise, random_haze
```

### Step 2.6 — `setup_training_paths()`
**Example: the run folder after this step**
```text
SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9/
├── Tensorboard/
├── Weights/
└── training_configuration.yaml    4,289 bytes
```
With `class_weighting: "balanced"`, the run folder also receives `class_pixel_counts.json` at Step 2.18.

### Step 2.7 — `build_dataloaders()`
This hook builds the loaders of the three splits (Step 2.8), keeps `class_names` (background first, then the categories of the training manifest) and `class_colours`, one BGR colour per class (black for the background), used to paint the masks in the input viewer and the sample images.

**Example output**
```text
class_names   = ['background', 'square', 'circle', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'plus_sign', 'star', 'octagon']
class_colours = [[0, 0, 0], [0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0], [255, 170, 0],
                 [255, 0, 0], [255, 0, 170], [170, 0, 255]]          (BGR: background black, square red, circle orange, ...)
```

### Step 2.8 — `get_segmentation_dataloaders()`
For each split it builds a `SegmentationDataset` (Step 2.9) from `<folder>/labels.json`, the images of `<folder>/data/` and the masks of `<folder>/masks/`, then wraps it in the Native loader (Step 2.10). Every image and mask is resized to the square network input, so the default collate stacks them. With `Data.tiling` activated, the splits instead cut their samples out of the images scaled by `tiling.scale`: a random tile of the network size for the training, a tile fixed per image for the validation (so that the passes stay comparable), and the whole scaled image for the test, one image per batch, segmented tile by tile in Stage 4.

**Example: size of the three splits**
```text
train       2000 samples -> 125 batches of 16 (one epoch = 125 iterations)
validation   200 samples ->  13 batches of 16
test         200 samples ->  13 batches of 16
```

### Step 2.9 — `SegmentationDataset.__init__()` → `build_class_lookup_table()`
It reads the manifest of the split: every sample gives its image file and its mask file, a single-channel `uint8` PNG of the size of its image whose every pixel holds the category of the image pixel (0 background, 255 unlabelled). The `category_ids` the generator now writes in every sample (the categories visible in its mask) are not read by the dataset, only by the record of the data (`data.json`). It shuffles the samples once and keeps them by identifier for the evaluation outputs. `build_class_lookup_table()` prepares a table of 256 class indices, one per possible mask value, so that a whole mask is converted at once with `table[mask]`: 0 stays the background, 255 becomes the ignored index (-100), and every category becomes the class of the label dictionary (an unmapped category becomes background). No image is read here.

**Example: a manifest sample, and the first sample after shuffling**
```text
labels.json:  {'id': 1, 'file_name': 'train_000000.png', 'mask_file_name': 'train_000000.png'}     (2000 samples, 9 categories)
categories:   {'id': 1, 'name': 'square'}, ...
kept:         {'identifier': 1351, 'file_name': 'train_001350.png', 'mask_file_name': 'train_001350.png'}
dataset[0]:   images [3, 512, 512] torch.uint8, masks [512, 512] torch.int64 holding the classes [0, 9], sample_identifiers 1351
```

### Step 2.10 — `build_native_loader()`
The same PyTorch `DataLoader` as for the classification (see `classification_workflow.md`, Step 2.10).

**Example output**
```text
len(loader) = 125   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
Spreading the sample images over the classes would need the classes of every test mask, which means reading all of them, so the hook gives every test image an empty class list. `choose_sample_identifiers()` then puts them in one group and takes `number_sample_images` images evenly spread over the sorted identifiers.

**Example output** (8 images among 200)
```text
get_test_sample_classes() -> 200 entries, e.g. {40: [], 61: [], 55: [], 59: []}
sample_image_identifiers  -> [1, 29, 58, 86, 115, 143, 172, 200]
names                     -> test_000000, test_000028, test_000057, test_000085, test_000114, test_000142, test_000171, test_000199
```

### Step 2.12 — `build_model()`
This hook creates the `SemanticSegmenter` from the registry `TASK_MODEL_CLASSES["semantic_segmentation"]` (Step 2.13), then copies the weights of the classes in the cross entropy into `model.class_weights` (Step 2.18). `BaseTrainer` then moves the model to the device and the dtype.

**Example: the components of the model**
```text
model.backbone                 Backbone                   438,368 parameters
model.segmentation_decoder     SegmentationDecoder        345,984 parameters
model.segmentation_head        Sequential                     170 parameters
model.deep_supervision_heads   ModuleDict                       0 parameters      (no deep supervision)
model.class_weights            buffer [10] torch.float32 [1.0, 1.0, ...]
model.number_classes           10
```

### Step 2.13 — `SemanticSegmenter.__init__()`
The segmenter is a U-Net: the backbone is the downsampling path, and the output of each of its blocks before its pooling is kept as a skip connection; the decoder brings the final feature map back to the input resolution through these skip connections; the segmentation head gives the logits of every class at every pixel. The head is kept apart from the decoder, so that a decoder transferred from another run can serve other classes. Deep supervision heads, when `deep_supervision_weights` names decoder levels, add 1×1 convolutions on those levels, trained with the same losses times their weight and left out of the predictions and exports. The number of classes counts the background.

### Step 2.14 — `build_backbone()`
The backbone shared by every task (see `classification_workflow.md`, Step 2.14). For the segmentation, its forward pass also returns the output of every block before its pooling (Step 3.15).

**Example: the backbone panel of `print_summary()` for a 512×512 image** (shortened)
```text
│  - Convolution Block 1                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 3, 512, 512]                                         │
│    - Output Shape : [1, 16, 256, 256]                                        │
│  ...                                                                         │
│  - Convolution Block 6                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 64, 16, 16]                                         │
│    - Output Shape : [1, 128, 16, 16]                                         │
Final Backbone output shape: [1, 128, 16, 16]
```

### Step 2.15 — `SegmentationDecoder.__init__()` (→ `compute_skip_block_numbers()`)
`compute_skip_block_numbers()` lists the backbone blocks whose output is a skip connection: every block before the last one (and the last one too when it ends with a pooling), from the deepest to the finest. For each of them the decoder builds one level: an upsampling layer that doubles the resolution to the channels of the level (a learned 2×2 transposed convolution, or a bilinear upsampling followed by a 1×1 convolution), then a `ConvolutionBlock` without pooling that refines the concatenation of the skip connection and of the upsampled features. The levels are keyed by the backbone block they read, as in `SegmentationDecoder.convolutions`, so that two decoders of the same levels share their weight names.

**Example: the levels and the upsampling layers**
```text
skip blocks = [5, 4, 3, 2, 1]    skip channels = {'5': 64, '4': 64, '3': 32, '2': 32, '1': 16}    output channels = 16
ModuleDict(
  (5): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))
  (4): ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
  (3): ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2))
  (2): ConvTranspose2d(32, 32, kernel_size=(2, 2), stride=(2, 2))
  (1): ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2))
)
```

### Step 2.16 — segmentation head and deep supervision heads
The head is a `Dropout2d` (kept at a rate of 0 so that the weight names do not depend on the rate) and a 1×1 convolution from the channels of the decoder output to one logit per class. Without `deep_supervision_weights`, the `ModuleDict` of deep supervision heads is empty.

**Example output**
```text
Sequential(
  (0): Dropout2d(p=0.0, inplace=False)
  (1): Conv2d(16, 10, kernel_size=(1, 1), stride=(1, 1))
)
deep_supervision_heads = ModuleDict()
```

### Step 2.17 — the decoder panel of `print_summary()`
The segmenter adds a panel to the backbone summary: the bottleneck, then for every level the size of its skip connection, its convolutions and its output, so that one can check that the last level is at the input resolution, then the head dropout and the size of the class logits.

**Example output** (shortened)
```text
╭─────────────────────── Segmentation Decoder And Head ────────────────────────╮
│ Upsampling    : transposed                                                   │
│ Normalization : batch                                                        │
│ Bottleneck    : 16x16x128                                                    │
│                                                                              │
│  - Level Of Block 5                                                          │
│    - Skip Connection : 32x32x64                                              │
│    - Convolutions    : 2 x 64 channels                                       │
│    - Output          : 32x32x64                                              │
│  ...                                                                         │
│  - Level Of Block 1                                                          │
│    - Skip Connection : 512x512x16                                            │
│    - Convolutions    : 1 x 16 channels                                       │
│    - Output          : 512x512x16                                            │
│                                                                              │
│ Head Dropout Rate : 0.0                                                      │
│ Class Logits      : 512x512x10                                               │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 2.18 — `compute_class_weights()` (→ `get_class_pixel_counts()`)
With `SegmentationLoss.class_weighting: "none"` (this run) every class weighs 1 in the cross entropy. With `"balanced"`, the weight of a class is `total pixels / (number of classes × pixels of the class)`, so the small shapes weigh more than the background. Counting the pixels reads every training mask, so `get_class_pixel_counts()` saves the counts in `class_pixel_counts.json` of the run folder and reads them back when the run resumes, as long as the training split and the label mapping did not change. The weighting gives no token to the run name, so `load_model()` sets the weights of the current configuration again after loading a checkpoint.

**Example output** (`class_weighting: "none"`)
```text
model.class_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

### Step 2.19 — `initialise_pretrained_components()`
The segmentation can transfer the backbone and the decoder of other runs. The segmentation head and the deep supervision heads stay outside the decoder, so that a decoder serves runs of other classes; with `SegmentationDecoder.pretrained.load_class_layers` they are loaded too (`load_segmentation_class_layers()`), the classes of the source run being the first ones of this run, the new classes keeping the random rows the layers of this run were built with, and the deep supervision levels the source run does not have keeping their random weights.

**Example: a run from scratch**
```text
pretrained_configurations['Backbone']            = {'experiment_folder': None, 'iteration': None, 'freeze': False, ...}
pretrained_configurations['SegmentationDecoder'] = {'experiment_folder': None, 'iteration': None, 'freeze': False,
                                                    'load_class_layers': False, 'learning_rate_multiplier': 1.0}
pretrained_checkpoint_paths                      = {'Backbone': None, 'SegmentationDecoder': None}
```

### Step 2.20 — `build_optimizer()`
**Example output**
```text
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
```

### Step 2.21 — `get_tracked_metrics()` + `MetricTracker.__init__()`
The tracked losses are those whose weight is not 0, so the console, TensorBoard and MLflow only show what is trained, then the pixel accuracy and the mean IoU of every batch (Step 3.19).

**Example output** (cross entropy and Dice on, Lovász and boundary off)
```text
{'total_loss': 'Total Loss', 'segmentation_loss': 'Cross Entropy Loss', 'dice_loss': 'Dice Loss',
 'pixel_accuracy': 'Pixel Accuracy', 'mean_iou': 'Mean IoU'}
```

### Step 2.22 — `print_experiment_information()` (→ `describe_split()` → `count_split_samples()`)
**Example output**
```text
- Run Name                : SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9
- Experiment Folder       : file:///.../example_runs/SEG_IN-KAR-512_..._MC9
...
- Train Data Folder       : file:///home/.../segmentation_data/synthetic_shapes/train (2000 images, 9 categories)
- Validation Data Folder  : file:///home/.../segmentation_data/synthetic_shapes/validation (200 images, 9 categories)
- Test Data Folder        : file:///home/.../segmentation_data/synthetic_shapes/test (200 images, 9 categories)
- Iterations Per Epoch    : 125
- Batch Size              : 16
- Total Iterations        : 10000000000
- Save And Evaluate Every : 1000 iterations
- Save Only Every         : 500 iterations
- Learning Rate           : 0.001
...
- Validation Iter. Ratio  : 25
- MLflow Tracking         : Off
```

### Step 2.23 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The backbone and decoder panels are shown in Steps 2.14 and 2.17. The size panel lists the decoder and the head on their own lines.

**Example output: the size panel**
```text
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 438,368                       │
│ Segmentation Decoder                                     │
│   - Parameters           : 345,984                       │
│ Segmentation Head                                        │
│   - Parameters           : 170                           │
│ Total Model                                              │
│   - Parameters           : 784,522                       │
│   - Estimated Size       : 3.15 MB                       │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.24 — `view_input_batches()`
With `diagnosis_modes.view_input_images`, the training image (augmented, with its augmented mask) and the validation image are shown with their masks painted over them in the class colours, each followed by an information panel giving the classes it holds (`Classes : square, star`, or `background only`) and, for the training image, the augmentations drawn with their values. It needs a display, so it has no console output to show here.

### Step 2.25 — `restore_last_model()` → `get_last_checkpoint_iteration()` / `load_model()`
The same restoration as every task (see `classification_workflow.md`, Steps 2.26 to 2.28), except that `load_model()` sets the class weights of the current configuration again after loading the checkpoint.

**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of `configuration_synthetic_shapes.yaml` shortened to 64 pixel images, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the predicted masks saved and the MLflow tracking on. Its run name is `SEG_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9`.

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
Tensorboard/Train/        Total Loss, Cross Entropy Loss, Dice Loss, Pixel Accuracy, Mean IoU, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Cross Entropy Loss, Dice Loss, Pixel Accuracy, Mean IoU                   (iteration 10)
```

### Step 3.3 — `initialize_training_loop_state()` + `MetricTracker.create_trackers()`
**Example: the trackers of an interval**
```text
{'start_time': ..., 'interval_batches': 0, 'total_loss': 0.0, 'segmentation_loss': 0.0, 'dice_loss': 0.0,
 'pixel_accuracy': 0.0, 'mean_iou': 0.0}
```

### Step 3.4 — `start_experiment_tracking()`
The MLflow run is created in the experiment `semantic_segmentation` (the task name), in the local database of the experiments root folder.

**Example output**
```text
MLflow Run Created In The Experiment 'semantic_segmentation' : 6a88ad94d2d2440f932cb52cc5b8249b (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.5 — `save_model()` → `save_checkpoint()`
The checkpoint stores the source runs of the two transferable components of the segmentation.

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   9,588,813 bytes      Weights/Iteration_6/model_000006.pt   9,588,813 bytes
Weights/Iteration_9/model_000009.pt   9,588,813 bytes      Weights/Iteration_12/model_000012.pt  9,588,813 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'mlflow_run_id',
 'pretrained_backbone', 'pretrained_segmentation_decoder']
model_state: 175 tensors, the first 'class_weights', then 'backbone.convolution_blocks.0. ...'
```

### Step 3.6 — `run_test_evaluation()`
At iterations 6 and 12 (the final checkpoint), see Stage 4.

### Step 3.7 — `get_next_batch()` → `SegmentationDataset.__getitem__()`
`__getitem__()` reads the image (BGR, `uint8`) and its mask, and checks that they have the same size. Without tiling, `preprocess_image()` resizes the image with the bilinear interpolation and the mask with the nearest interpolation (so that every pixel keeps a category), with the same letterbox: the padding of the image is black and the padding of the mask is 255, the unlabelled value. With tiling, `cut_tile()` scales both and cuts the tile of the split. The lookup table of Step 2.9 then turns the mask values into class indices.

**Example: the first batch of the run**
```text
images               [4, 3, 64, 64] torch.uint8
masks                [4, 64, 64] torch.int64       class index of every pixel, -100 on the padding and the unlabelled pixels
sample_identifiers   [4] torch.int64               [120, 1237, 1626, 1323]
```

### Step 3.8 — `set_training_mode()`
`model.train()`: in training mode the segmenter returns only the logits and the losses, no softmax of every pixel.

### Step 3.9 — `run_model_iteration()`
The same function for the training and the validation batches (see `classification_workflow.md`, Step 3.11): prepare, augment, forward, sum of the `_loss` outputs, batch metrics, tracking.

**Example: the first training batch**
```text
segmentation_loss = 2.0932   dice_loss = 0.9559   total_loss = 3.0491   pixel_accuracy = 0.2665   mean_iou = 0.0424
```

### Step 3.10 — `prepare_batch()`
**Example: before and after**
```text
images  [4, 3, 64, 64] torch.uint8   -> [4, 3, 64, 64] torch.float32
masks   [4, 64, 64]    torch.int64   -> [4, 64, 64]    torch.int64
```

### Step 3.11 — `augment_batch()` → `DataAugmenter.augment_batch(masks=...)`
The segmentation trainer overrides this hook so that the masks are augmented with their images: every geometric augmentation (crop, flips, rotations) is applied to the mask with the nearest interpolation, and the pixels uncovered by a rotation become ignored; the photometric augmentations leave the mask untouched. A mask that did not follow its image would teach the network wrong pixels.

**Example: which images and masks of the first batch were changed** (`probability: 0.4`)
```text
images: changed per sample [False, True, True, False]
masks:  changed per sample [False, True, True, False]      <- both images got a geometric augmentation
```

### Step 3.12 — `forward_pass()` → `SemanticSegmenter.forward(masks=...)`
The trainer passes the images and the masks, so that the segmenter returns its losses with its logits.

**Example: the dictionary returned in training mode**
```text
class_logits        [4, 10, 64, 64] torch.float32      one logit per class at every pixel
segmentation_loss   2.0932 (scalar)
dice_loss           0.9559 (scalar)
```

### Step 3.13 — `Backbone.forward(return_pre_pooling_outputs=True)`
The backbone returns its final feature map and the output of every block before its pooling, keyed by block number: the skip connections of the decoder.

**Example: shapes for a batch of 4 images of 64 pixels**
```text
final feature map [4, 128, 2, 2]
outputs before pooling  '1': [4, 16, 64, 64]   '2': [4, 32, 32, 32]   '3': [4, 32, 16, 16]   '4': [4, 64, 8, 8]   '5': [4, 64, 4, 4]
                        '6': [4, 128, 2, 2]    <- the last block has no pooling: it is the final map, not a skip connection
```

### Step 3.14 — `SegmentationDecoder.forward()` + segmentation head
From the deepest level to the finest, each level upsamples its input to twice its resolution, concatenates the skip connection of its block (resized by one pixel if the input size is not a power of 2), and refines the result with its convolution block. The head turns the `[B, 16, 64, 64]` output into the `[B, 10, 64, 64]` class logits.

**Example: shapes through the decoder**
```text
[4, 128, 2, 2] -> level 5 [4, 64, 4, 4] -> 4 [4, 64, 8, 8] -> 3 [4, 32, 16, 16] -> 2 [4, 32, 32, 32] -> 1 [4, 16, 64, 64]
-> head [4, 10, 64, 64]
```

### Step 3.15 — `compute_mask_losses()`
Each loss of `SegmentationLoss` whose weight is not 0 is computed on the labelled pixels (the ignored ones give no loss) and multiplied by its weight:
- **cross entropy** (`segmentation_loss`): weighted by `class_weights`, with the label smoothing and the focal factor `(1 - p)^gamma`, which lowers the loss of the pixels already well classified; normalised by the weights of the labelled pixels.
- **Dice** (`dice_loss`): one minus the mean soft Dice (`2 × intersection / (predicted + true)`) of the classes present in the batch, which keeps the small objects as important as the background.
- **Lovász-Softmax** (`lovasz_loss`): a direct surrogate of the IoU of every class.
- **boundary** (`boundary_loss`): the errors weighted by their closeness to the true edges (`boundary_distance` pixels).

With deep supervision, the same losses are computed for the upsampled logits of every head and summed, times the weight of its level, into `deep_supervision_loss`. A model that gives every class the same probability has a cross entropy of ln 10 = 2.30.

**Example output** (cross entropy and Dice, weight 1.0 each)
```text
segmentation_loss = 2.0932      dice_loss = 0.9559      total_loss = 3.0491
```

### Step 3.16 — `compute_batch_metrics()` (→ `compute_pixel_confusion_matrix()` + `compute_segmentation_metrics()`)
The predicted class of every pixel is the argmax of its logits. `compute_pixel_confusion_matrix()` counts the labelled pixels of every pair (true class, predicted class) with one `bincount`, and `compute_segmentation_metrics()` gives the pixel accuracy (share of the labelled pixels given their class) and the mean IoU of the classes present in the batch. No second forward pass is run.

**Example: the first batch**
```text
pixel_accuracy = 0.2665      mean_iou = 0.0424
```

### Step 3.17 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
**Example: the Train curves of the first iterations**
```text
Total Loss      1: 3.0491, 2: 2.9684, 3: 2.9365, 4: 2.9298, ...
Pixel Accuracy  1: 0.2665, 2: 0.4165, 3: 0.4551, 4: 0.5035, ...
```

### Step 3.18 — `training_loss.backward()` + `step_optimizer()`
**Example: the gradient norm of the first steps and the learning rate**
```text
iteration 1: 1.8968    iteration 2: 1.7794    iteration 3: 1.8824        learning rate 0.001 (constant)
```

### Step 3.19 — validation pass
**Example: iteration 10**
```text
Tensorboard/Validation at iteration 10: Total Loss 2.9886, Cross Entropy Loss 2.0299, Dice Loss 0.9587,
                                        Pixel Accuracy 0.9136, Mean IoU 0.1126
```

### Step 3.20 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The sums of the interval are divided by the number of batches it ran; the batch mean IoU is thus averaged over the batches, each batch having its own classes.

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 11.8839, 'segmentation_loss': 8.085, 'dice_loss': 3.7988, 'pixel_accuracy': 1.6417, 'mean_iou': 0.2646}
after:  {'interval_batches': 4, 'total_loss': 2.971, 'segmentation_loss': 2.0213, 'dice_loss': 0.9497, 'pixel_accuracy': 0.4104, 'mean_iou': 0.0661}
MLflow, step 4: {'train/Total Loss': 2.971, 'train/Cross Entropy Loss': 2.0213, 'train/Dice Loss': 0.9497,
                 'train/Pixel Accuracy': 0.4104, 'train/Mean IoU': 0.0661, 'train/throughput': 4.6914,
                 'train/interval_seconds': 3.4105, 'learning_rate': 0.001}
```

**Example output: iterations 8 and 12**
```text
========================================================================================================================
|     Iteration: 8     |      Total Images: 32     |    Interval Images: 16     |  Throughput: 1.5 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   2.8497  -0.1213 Best
Moving Average of Train Cross Entropy Loss             :   1.9076  -0.1137 Best
Moving Average of Train Dice Loss                      :   0.9421  -0.0076 Best
Moving Average of Train Pixel Accuracy                 :   0.4850  +0.0746 Best
Moving Average of Train Mean IoU                       :   0.0781  +0.0120 Best
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:10.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
========================================================================================================================
...
========================================================================================================================
|    Iteration: 12     |      Total Images: 48     |    Interval Images: 16     |  Throughput: 8.1 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   2.7779  -0.0718 Best
Moving Average of Valid Total Loss                     :   2.9886
Moving Average of Train Cross Entropy Loss             :   1.8413  -0.0663 Best
Moving Average of Valid Cross Entropy Loss             :   2.0299
Moving Average of Train Dice Loss                      :   0.9366  -0.0055 Best
Moving Average of Valid Dice Loss                      :   0.9587
Moving Average of Train Pixel Accuracy                 :   0.4567  -0.0284               <- red: lower than at iteration 8
Moving Average of Valid Pixel Accuracy                 :   0.9136
Moving Average of Train Mean IoU                       :   0.0829  +0.0048 Best
Moving Average of Valid Mean IoU                       :   0.1126
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:01.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```

### Step 3.21 — end of the loop
**Example output**
```text
Model successfully saved at iteration 12. Exiting Training.
Iteration 12 · 200 Test Images · 2026-10-09 10:04 ───────────────      <- the evaluation of the final checkpoint
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (200 images of 64×64 pixels), with `per_class_report: True`, `number_sample_images: 8` and `save_predicted_masks: True`.

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1). The `evaluation_context()` of the segmentation changes nothing.

**Example output: the progress bar** (200 images, batch size 4)
```text
Test Evaluation Iteration 6: 100%|██████████| 50/50
```

### Step 4.2 — `create_evaluation_accumulator()`
Storing the probabilities of every pixel of the test split would take a lot of memory, so the segmentation accumulates counts instead, on the device: the `[K, K]` pixel confusion matrix and the `[K, 6]` boundary counts of every class, with the number of evaluated images and the `Predicted_Masks` folder of the checkpoint.

**Example: the accumulator at the end of the loop**
```text
{'confusion_matrix': [10, 10] torch.int64, 'boundary_counts': [10, 6] torch.int64, 'number_images': 200,
 'predicted_masks_folder': PosixPath('.../Evaluation/Iteration_6/Predicted_Masks')}
```

### Step 4.3 — `prepare_batch()`
Same function as in training (Step 3.10).

### Step 4.4 — `predict_batch()`
Without tiling, it runs `SemanticSegmenter.forward()` without masks on the whole batch; in `eval()` mode the segmenter also returns the softmax of every pixel. With tiling, every whole scaled test image is segmented tile by tile by `segment_image_by_tiles()`: tiles of the network size, overlapping by `Data.tiling.overlap` pixels, are run in batches of `batch_size` tiles, and the probabilities of the overlapping pixels are averaged.

**Example: the predictions of a test batch**
```text
class_logits          [4, 10, 64, 64] torch.float32
class_probabilities   [4, 10, 64, 64] torch.float32
```

### Step 4.5 — `accumulate_evaluation_batch()` (→ `compute_pixel_confusion_matrix()` + `compute_boundary_counts()`)
The predicted class map of every kept image (the argmax of its probabilities) is compared with its mask: at the network resolution (the padding and the unlabelled pixels being ignored), or with `original_resolution: True` with the original mask of every image, its probabilities being first cut out of the letterbox and resized back to the original size (`restore_prediction_to_original_image()`). The pixel confusion matrix and the boundary counts of the batch are added to the accumulator. `compute_boundary_counts()` counts, for every class, the intersection and the union of the bands within `boundary_tolerance` pixels of the true and predicted contours, the predicted edge pixels with those within the tolerance of a true edge, and the true edge pixels with those within the tolerance of a predicted edge. With `save_predicted_masks`, the class map of every image is written as a single-channel PNG named after the image.

**Example: the predicted masks of the checkpoint of iteration 6**
```text
Evaluation/Iteration_6/Predicted_Masks/    200 files: test_000000.png, test_000001.png, ...   (pixel value = predicted class index)
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
For every fixed sample image found in the batch, `build_sample_image()` lays out four titled panels side by side: the input, the true mask painted over it, the predicted mask painted over it, and the error map. A caption bar above gives the pixel accuracy of the image over its labelled pixels, and a legend bar below names the classes painted in the image. The panels of a large image segmented tile by tile are reduced to 1024 pixels on their longest side. The image is saved in `Samples/` and sent to TensorBoard under `Samples/<image name>`.

**Example: `Evaluation/Iteration_12/Samples/test_000028.png`** (268 x 112 pixels)
```text
caption: "Pixel Accuracy 0.9006"
panels:  Input | Ground Truth | Prediction | Errors      (64 x 64 each; the titles are cut at this small size: "Ground Tru")
legend:  square (red), circle (orange), rectangle (green)
```

### Step 4.7 — `compute_evaluation_results()`
Once the loop is done, it computes the metrics from the accumulated counts (Steps 4.8 and 4.9), writes the evaluation files (Steps 4.10 to 4.12) and organizes the report (Step 4.13). The notes give the number of labelled pixels and the resolution they were scored at, the boundary tolerance and the predicted masks. The headline metric, used by Stage 5 and MLflow to choose the best checkpoint, is the mean IoU.

**Example: the results of iteration 12**
```text
evaluated_samples = '200 Test Images'
metrics           = {'Mean IoU': 0.079, 'Mean Dice': 0.0939, 'Mean Boundary IoU': 0.0195, 'Mean Boundary F1': 0.0608,
                     'Pixel Accuracy': 0.7222, 'Frequency Weighted IoU': 0.6748}
notes             = ['Labelled Pixels : 819200 at the network resolution, the padding and the unlabelled pixels being ignored',
                     'Boundary Tolerance : 2 pixels', 'Predicted Masks : 200 class index images', 'Sample Images : 8 fixed test images']
headline_metric   = 'Mean IoU'
files             = Class Metrics, Confusion Matrix Table, Confusion Matrix Image, Predicted Masks, Sample Images
class_scalars     = 60: IoU/background, Dice/background, Boundary IoU/background, Boundary F1/background, Precision/background, ...
```
The 819,200 labelled pixels are the 200 test images × 64 × 64: the synthetic images are square, so there is no padding.

### Step 4.8 — `compute_segmentation_metrics()`
All the region metrics come from the pixel confusion matrix (rows true classes, columns predicted classes). For a class: its **IoU** is the right pixels over the union of its true and predicted pixels, its **Dice** is the F1 of its pixels (twice the intersection over their sum), its **precision** and **recall** are those of the classification. The **pixel accuracy** is the diagonal over all the labelled pixels, the **mean IoU** and **mean Dice** are the means over the classes present in the test masks, and the **frequency-weighted IoU** weights the IoU of every class by its share of the pixels, which the background dominates.

**Example: `Evaluation/Iteration_12/confusion_matrix.csv`** (pixel counts; the network predicts only background, square and star)
```text
true \ predicted,background,square,circle,triangle,rectangle,pentagon,hexagon,plus_sign,star,octagon
background,587460,63058,0,0,0,0,0,0,89288,0
square,8986,3208,0,0,0,0,0,0,3689,0
circle,5109,2349,0,0,0,0,0,0,2754,0
...
octagon,7166,2334,0,0,0,0,0,0,2574,0
```

**Example: how the numbers of iteration 12 are obtained**
```text
background  true pixels 739,806   predicted pixels 634,726 (column sum)   right 587,460
            IoU = 587,460 / (739,806 + 634,726 - 587,460) = 0.7464    precision = 587,460 / 634,726 = 0.9255    recall = 0.7941
pixel accuracy         = (587,460 + 3,208 + 942) / 819,200                  = 0.7222
mean IoU               = (0.7464 + 0.0350 + 0.0086 + 7 x 0.0) / 10           = 0.0790
frequency-weighted IoU = (739,806 x 0.7464 + 15,883 x 0.0350 + 5,654 x 0.0086) / 819,200 = 0.6748
```

### Step 4.9 — `compute_boundary_metrics()`
The region metrics hardly see an error on the contours: an object predicted 3 pixels too wide on every side keeps a high IoU while none of its predicted contour is right. From the boundary counts of Step 4.5, the **boundary IoU** of a class is the IoU of the bands within the tolerance of its true and predicted contours, and its **boundary F1** is the F1 of the share of the predicted contour within the tolerance of the true one (precision) and the share of the true contour within the tolerance of the predicted one (recall). Their means over the classes are headline metrics.

**Example: iteration 12** (`boundary_tolerance: 2`)
```text
background   boundary IoU 0.1702   boundary F1 0.4931
square       boundary IoU 0.0173   boundary F1 0.0730
Mean Boundary IoU 0.0195   Mean Boundary F1 0.0608
```

### Step 4.10 — `build_segmentation_class_rows()` + `write_class_metrics_file()`
One row per class, in class index order, with its true pixels and all its metrics, written as `class_metrics.csv` at every evaluation, whatever `per_class_report`.

**Example: `Evaluation/Iteration_12/class_metrics.csv`**
```text
class_index,class,pixels,iou,dice,boundary_iou,boundary_f1,precision,recall
0,background,739806,0.746387,0.854778,0.170248,0.493107,0.925533,0.794073
1,square,15883,0.034965,0.067568,0.017344,0.073005,0.040570,0.201977
2,circle,10212,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
...
8,star,5654,0.008555,0.016965,0.007485,0.041788,0.008937,0.166608
9,octagon,12074,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
```

### Step 4.11 — `save_confusion_matrix_table()` + `save_confusion_matrix_image()`
The table is the pixel confusion matrix of Step 4.8 (`confusion_matrix.csv`, written whatever `per_class_report`). With `per_class_report`, the image is drawn like the one of the classification, titled `Pixel Confusion Matrix · Iteration N · Mean IoU ... · Pixel Accuracy ...`, its cells showing only the share of every true class, since the pixel counts are too large to print (`show_counts=False`).

### Step 4.12 — the notes and the predicted masks
The notes name what was scored: the number of labelled pixels and the resolution (`network resolution`, `original resolution of the images`, or `images scaled by <scale>, segmented tile by tile`), and the boundary tolerance. With `save_predicted_masks`, the `Predicted_Masks` folder and its number of images are added.

### Step 4.13 — `build_segmentation_class_table()` + `build_segmentation_class_scalars()`
The per-class table sorts the classes by IoU, worst first; the per-class scalars are the 6 metrics of every class (`IoU/<class>`, `Dice/<class>`, `Boundary IoU/<class>`, `Boundary F1/<class>`, `Precision/<class>`, `Recall/<class>`): 60 scalars for 10 classes, shown only with `per_class_report`.

### Step 4.14 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 200 Test Images · 2026-10-09 10:04 ──────────────────────────────────────────────────────────────────────
  Mean IoU            0.0790   Mean Dice                0.0939
  Mean Boundary IoU   0.0195   Mean Boundary F1         0.0608
  Pixel Accuracy      0.7222   Frequency Weighted IoU   0.6748
  Labelled Pixels : 819200 at the network resolution, the padding and the unlabelled pixels being ignored
  Boundary Tolerance : 2 pixels
  Predicted Masks : 200 class index images
  Sample Images : 8 fixed test images

  Per Class Metrics
   Class        Pixels      IoU     Dice   Boundary IoU   Boundary F1   Precision   Recall
  ─────────────────────────────────────────────────────────────────────────────────────────
   circle        10212   0.0000   0.0000         0.0000        0.0000      0.0000   0.0000
   triangle       2221   0.0000   0.0000         0.0000        0.0000      0.0000   0.0000
   rectangle      6571   0.0000   0.0000         0.0000        0.0000      0.0000   0.0000
   pentagon       8031   0.0000   0.0000         0.0000        0.0000      0.0000   0.0000
   hexagon        9485   0.0000   0.0000         0.0000        0.0000      0.0000   0.0000
   plus_sign      9263   0.0000   0.0000         0.0000        0.0000      0.0000   0.0000
   octagon       12074   0.0000   0.0000         0.0000        0.0000      0.0000   0.0000
   star           5654   0.0086   0.0170         0.0075        0.0418      0.0089   0.1666
   square        15883   0.0350   0.0676         0.0173        0.0730      0.0406   0.2020
   background   739806   0.7464   0.8548         0.1702        0.4931      0.9255   0.7941
  Worst IoU first, ties in class order. Boundary IoU and F1 : contours within the boundary tolerance. — : no pixel (or
  no edge for the boundary metrics) of the class in the test masks.
  - Results : file:///.../Results.txt
  - Class Metrics : file:///.../Evaluation/Iteration_12/class_metrics.csv
  - Confusion Matrix Table : file:///.../Evaluation/Iteration_12/confusion_matrix.csv
  - Confusion Matrix Image : file:///.../Evaluation/Iteration_12/confusion_matrix.png
  - Predicted Masks : file:///.../Evaluation/Iteration_12/Predicted_Masks
  - Sample Images : file:///.../Evaluation/Iteration_12/Samples
```

**Example: the run folder, TensorBoard and MLflow after the two evaluations**
```text
Evaluation/Iteration_6/ and Iteration_12/   Predicted_Masks/ (200)  Samples/ (8)  class_metrics.csv  confusion_matrix.csv  confusion_matrix.png
Tensorboard/Test: 66 scalars (6 global + 60 per class), 9 images (8 samples + Confusion Matrix)
    scalar Mean IoU          6: 0.0024, 12: 0.0790
    scalar Pixel Accuracy    6: 0.0196, 12: 0.7222
MLflow: test/Mean IoU, test/Mean Dice, ..., test/best_Mean IoU (steps 6, 12), 60 test_class/<metric>/<class>,
        artifacts evaluation/iteration_6/ and iteration_12/: class_metrics.csv, confusion_matrix.png
```
The checkpoint of iteration 6 predicted `square` almost everywhere (pixel accuracy 0.0196, square recall 0.9966); the one of iteration 12 learned that most pixels are background.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 40 images of the test split (its `labels.json` reduced, its `data/` and `masks/` folders linked).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5): set `run_name` to the segmentation run, and `test_data_folder` to another labelled split (holding `labels.json`, `data/` and `masks/`) if needed.

### Step 5.2 — trainer construction
The script loads `training_configuration.yaml` of the run, points it to the output folder, the reduced split and the batch size, and builds a `SegmentationTrainer`. With `class_weighting: "balanced"` it would count the pixels of the training masks again in the output folder, since the counts are cached in the run folder of the trainer.

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'semantic_segmentation' : 342ebb90469341998a33bbefd91e4b09 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output**
```text
Evaluation Summary Of 4 Checkpoints : file:///.../runs/evaluations/SEG_IN-KAR-64_..._MC9/Evaluation/evaluation_summary.csv
  - Best Checkpoint : iteration 12 (Mean IoU 0.0713)

iteration,Mean IoU,Mean Dice,Mean Boundary IoU,Mean Boundary F1,Pixel Accuracy,Frequency Weighted IoU,IoU/background, ...
12,0.07127240554791028,0.08981075670647244,0.015183896521748261,0.05434968979463565,0.642840576171875,0.5973665654951408, ...
9,0.054483986969393404,0.0767030110689235,0.017650408620012898,0.05856697293708574,0.473822021484375,0.4409209823322503, ...
6,0.003934161162990151,0.007616101260174676,0.002960294035671728,0.02332027384646323,0.03636474609375,0.0016963343355225918, ...
3,0.003597412109375,0.006944984505166907,0.0,0.0,0.03597412109375,0.0012941373884677886, ...
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 12 on the 40 images** (shortened: the classes absent from these images are left out)
```text
Evaluating : file:///.../runs/SEG_IN-KAR-64_..._MC9/Weights/Iteration_12/model_000012.pt
Iteration 12 · 40 Test Images · 2026-10-09 10:04 ───────────────────────────────────────────────────────────────────────
  Mean IoU            0.0713   Mean Dice                0.0898
  Mean Boundary IoU   0.0152   Mean Boundary F1         0.0543
  Pixel Accuracy      0.6428   Frequency Weighted IoU   0.5974
  Labelled Pixels : 163840 at the network resolution, the padding and the unlabelled pixels being ignored
  Boundary Tolerance : 2 pixels
  Predicted Masks : 40 class index images
  Sample Images : 8 fixed test images
  ...
   star            569   0.0049   0.0097         0.0054        0.0360      0.0049   0.2671
   square         5894   0.0496   0.0945         0.0203        0.0993      0.0617   0.2019
   background   148235   0.6583   0.7939         0.1261        0.4082      0.9144   0.7015
```
The sample images were chosen from the 40 images of this split, so they are not the same images as in Stage 4.
