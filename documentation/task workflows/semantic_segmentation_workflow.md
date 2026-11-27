# Semantic Segmentation Workflow

This document follows one semantic segmentation run from `main()` to the evaluation of a checkpoint, one step per function, each with a **real** output. Steps shared by every task are detailed in `classification_workflow.md` and stay short here.

The examples come from CPU runs, so the GPU stayed free, as the table below shows.

| Outputs | Source |
|---|---|
| Stages 1 and 2 | `vision_training/configurations/semantic_segmentation/configuration_synthetic_shapes.yaml` (512 pixels, batch size 16) on the 9-class synthetic dataset of 2026-10-21 (a dataset generated now holds 35 shapes, runs ending in `MC35`) |
| Stages 3, 4 and 5 | the same configuration cut to 64 pixels, batch size 4 and 12 iterations (Stage 3); the network mostly predicts background, so the metrics are poor on purpose |

## Read This First

The semantic segmentation task (`task: "semantic_segmentation"`, run name code `SEG`) learns the class of every pixel. The model is a U-Net. The shared `Backbone` is the encoder; its block outputs before pooling are the skip connections. The `SegmentationDecoder` brings the last feature map back to the input resolution level by level. A 1x1 head gives K logits per pixel, background being class 0 (K = 10 for 9 shapes). Labels are single-channel uint8 PNG masks holding the category of every pixel, 255 for pixels left out of losses and metrics.

Choose SEG when every pixel needs a class and objects need not be told apart (two touching squares form one region). Choose INS for the mask of every object, DET for boxes, KPT for named points, CLS for one label per image. Large images can be trained on tiles (`Data.tiling`) and scored at their original resolution (`evaluation.original_resolution`).

### The Task At A Glance

So that the stages below can be followed on one page, the diagram traces the task from the files of a split to the exported segmenter, with the shapes of the synthetic configuration.

```text
on disk        <split>/data/*.png, <split>/masks/*.png (uint8 category per pixel: 0 background, 255 unlabelled),
               <split>/labels.json (samples with file_name and mask_file_name, categories), label_mapping.json
sample         SegmentationDataset: letterbox to 512 (image bilinear, mask nearest-exact, padding 255), lookup table
               {0: 0, category: class, 255: -100}: images [3, 512, 512] uint8, masks [512, 512] int64
batch (16)     images [16, 3, 512, 512], masks [16, 512, 512]; augment_batch moves the masks with the image
               (nearest-exact, uncovered pixels -100), the photometric types change the images only
encoder        Backbone, outputs before pooling: '1' [16, 16, 512, 512] '2' [16, 32, 256, 256] '3' [16, 32, 128, 128]
               '4' [16, 64, 64, 64] '5' [16, 64, 32, 32], bottleneck '6' [16, 128, 16, 16]
decoder        per level 5 -> 1: 2x2 transposed conv (x2) -> cat(skip, upsampled) -> ConvolutionBlock:
               [16, 64, 32, 32] [16, 64, 64, 64] [16, 32, 128, 128] [16, 32, 256, 256] [16, 16, 512, 512]
head           Dropout2d(0.0) + 1x1 conv 16 -> 10: class_logits [16, 10, 512, 512], softmax in eval mode only
losses         1.0 x cross entropy + 1.0 x soft Dice (Lovász and boundary at 0), the -100 pixels left out
batch metrics  argmax -> pixel confusion matrix -> Pixel Accuracy, Mean IoU (console, TensorBoard, MLflow)
evaluation     summed on the device over the test split: confusion [10, 10], boundary counts [10, 6], histograms
               [10, 2, 10000] -> Mean IoU (headline), Mean Dice, Mean Boundary IoU / F1, Pixel Accuracy,
               Frequency Weighted IoU, Macro ROC AUC, Macro Average Precision, the same per class, the 95 %
               interval of every recall (Korn-Graubard, bootstrap over the test images)
export         InferenceSegmenter: images [B, 3, 512, 512] -> class_map [B, 512, 512] int64,
               class_probabilities [B, 10, 512, 512]; the input contract of the run (keep_ratio, 512, BGR, version 2)
               in preprocessing.json next to the model and in the metadata of model.onnx
```

### Files To Know

If you are looking for the code of a step, the table lists the files of the task and their main functions.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/segmentation_trainer.py` | `SegmentationTrainer` and its hooks |
| Dataset and loaders | `vision_training/data/segmentation/segmentation_dataset.py`, `vision_training/data/loaders.py` | `SegmentationDataset`, `build_class_lookup_table`, `get_segmentation_dataloaders` |
| Model | `vision_training/models/segmentation/semantic_segmenter.py`, `segmentation_decoder.py` | `SemanticSegmenter` (`compute_mask_losses`), `SegmentationDecoder`, `compute_skip_block_numbers` |
| Losses | `vision_training/losses/segmentation_losses.py` | `compute_segmentation_cross_entropy`, `compute_dice_loss`, `compute_lovasz_softmax_loss`, `compute_boundary_loss` |
| Metrics and report | `vision_training/evaluation/segmentation/segmentation_metrics.py`, `boundary_metrics.py`, `segmentation_report.py` | `compute_pixel_confusion_matrix`, `compute_segmentation_metrics`, `compute_boundary_metrics`, `build_segmentation_class_table` |
| Recall intervals | `vision_training/evaluation/recall_intervals.py` | `compute_bootstrap_recall_intervals`, `build_recall_interval_table` |
| Curves and confusion matrix | `vision_training/evaluation/classification/score_curves.py`, `confusion_matrix.py` | `compute_histogram_score_curves`, `save_score_curve_images`, `save_confusion_matrix_image` |
| Tiles and original resolution | `vision_training/models/segmentation/sliding_window.py`, `vision_training/data/preprocessing.py` | `segment_image_by_tiles`, `restore_prediction_to_original_image` |
| Sample images | `vision_training/utilities/visualization.py` | `build_segmentation_sample_grid`, `build_segmentation_error_map` |
| Checks and run name | `vision_training/configuration_processing/segmentation_checks.py`, `run_name.py` | `check_segmentation_configuration`, `build_segmentation_sections` |
| Export | `vision_training/models/segmentation/inference_segmenter.py`, `vision_training/deployment/input_contract.py` | `InferenceSegmenter` (model name `semantic_segmenter`, deep supervision heads left out), `build_input_contract` |
| Standalone evaluation | `scripts/evaluation/evaluate_checkpoint.py`, `vision_training/evaluation/checkpoint_evaluation.py` | `evaluate_checkpoints`, `summarise_checkpoint_evaluations` |

### Run It

In order to generate, train, evaluate and export the synthetic shapes segmenter from scratch, run these commands from the repository root. Before training, set `experiment_folder`, the three data folders, `Data.label_mapping.mapping_file_path` and `number_iterations` (1e10 in this file; 2000 for the 9-class run of `experiments/readme_screenshots`).

```bash
# Generate the dataset: 50,000 / 200 / 200 images of 512 pixels, up to 6 of the 35 shapes, with class masks
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=semantic_segmentation --output_folder="$HOME/Data/segmentation_data/synthetic_shapes"

# Train (Ctrl+C saves a checkpoint)
uv run python -m vision_training.main --configuration_path="vision_training/configurations/semantic_segmentation/configuration_synthetic_shapes.yaml"

# Evaluate every checkpoint into <run>/Checkpoint_Evaluation/, without error images (Step 5.1)
uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder <run folder> --iteration all --maximum_error_images_per_class_pair None

# Export (CUDA only; run without arguments for the help)
uv run python -m vision_training.deployment.export_model --experiment_folder <run folder> --iteration <iteration>
```

The export writes the input contract in `<model>/<version>/preprocessing.json` and the metadata of `model.onnx`; served models are replaced only once every format matches the eager model. On the 35-class dataset of 2026-11-19 this recipe (batch 32, shape pre-trainer backbone) reached only 4.44 % mean IoU at 1000 iterations; the 35-shape recipe is still to be studied.

### Configuration Keys To Know First

Before changing the synthetic configuration, look at these keys: they change the model, the losses or the evaluation the most.

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `SegmentationDecoder.convolutions` | `{"5": [2, 64], "4": [2, 64], "3": [2, 32], "2": [2, 32], "1": [1, 16]}` | one level per backbone block before the last, `[layers, channels]` from the deepest (`D2x64.2x64.2x32.2x32.1x16`) |
| `SegmentationDecoder.upsampling` | `"transposed"` | a learned 2x2 transposed convolution of stride 2 (`UTC`); `"bilinear"` is bilinear then 1x1 convolution (`UBL`) |
| `SegmentationDecoder.modules` | `squeeze_and_excite: False`, `residual_nets: False` | decoder block modules (Pitfalls) |
| `SegmentationDecoder.deep_supervision_weights` | `{}` | 1x1 class heads on intermediate levels (`AUX<levels>`), same losses times their weight, not exported |
| `SegmentationLoss.cross_entropy_weight`, `dice_weight` | `1.0`, `1.0` | the two losses of the total; `lovasz_weight` and `boundary_weight` are 0; the region losses cannot all be 0 |
| `SegmentationLoss.class_weighting`, `focal_gamma`, `label_smoothing` | `"none"`, `0.0`, `0.0` | `"balanced"` weights a class by total pixels / (K x class pixels), cached in `class_pixel_counts.json` |
| `Data.tiling` | `activated: False`, `scale: 1.0`, `overlap: 128` | training on tiles (`TL<scale>`, Step 2.8) |
| `ExperimentConfiguration.evaluation.boundary_tolerance` | `2` | distance in pixels within which a contour pixel is right, for boundary IoU and F1 |
| `ExperimentConfiguration.evaluation.save_predicted_masks`, `original_resolution` | `False`, `False` | class index PNG of every test image in `Predicted_Masks/`; scores against the original masks |
| `ExperimentConfiguration.evaluation.per_class_report` | `True` | per-class scalars, confusion matrix image and curves (Step 4.13) |

### What A Run Writes

Whenever you look for a result of the run, the table lists what training and evaluation write.

| Output | Content |
|---|---|
| `Results.txt` | one block per checkpoint: the headline metrics (Step 4.7), notes, the per-class table and the `Per Class Recall` table (Step 4.13); rates as `7.90 %`, files and scalars between 0 and 1 |
| `Evaluation/Iteration_<n>/` | `class_metrics.csv` (Step 4.10), `confusion_matrix.csv`; with `per_class_report` `confusion_matrix.png`, `roc_curves.png`, `precision_recall_curves.png`; `Samples/` (2060x560 at 512 pixels); `Predicted_Masks/` with `save_predicted_masks`; no `EMA/` (moving average off) |
| `class_pixel_counts.json` | with `class_weighting: "balanced"` |
| TensorBoard | `Train` and `Validation`: losses, `Pixel Accuracy`, `Mean IoU`; `Test`: 88 scalars and 11 images in the 9-class run of `experiments/readme_screenshots` (Step 4.13) |
| `Weights/Iteration_<n>/model_<n>.pt` | about 9.6 MB: 3.15 MB of weights and the two Adam moments |

### Pitfalls

Should a result of the task look wrong, check these behaviours first.

- **`residual_nets: True` often adds no shortcut.** A decoder level's input (skip plus upsampled) never has its output channels, so the first shortcut closes at the third layer; levels of 1 or 2 layers add none (`count_residual_additions()`, Step 2.14).
- **Mean IoU counts falsely painted classes.** A class painted without true pixels counts 0 (Step 4.8). Before 2026-10-25 such classes were left out, so older evaluations can be higher.
- **Runs before 2026-11-20 learned shifted masks.** Masks were resized with plain `cv2.INTER_NEAREST`, about half a pixel to the bottom right (another half with `random_crop`, `random_aspect_crop`, `random_zoom_out`). They now use pixel centres (`cv2.INTER_NEAREST_EXACT`, `NEAREST_EXACT`). Older runs may have lower boundary scores (not measured).
- **Some settings give no run name token.** `SegmentationLoss`, head dropout, deep supervision weights and tile overlap: changing them resumes the same run folder (Step 2.4).
- **A tiled run is exported with one tile as input.** The client scales and tiles with `segment_image_by_tiles()` ([tools reference, Export](../reference/tools_reference.md#export)); `run_inference.py` refuses such a run (`load_inference_inputs()`). The input contract version 2 describes the tiles ([tools reference, Input Contract](../reference/tools_reference.md#input-contract)): with `scale: 0.5` and `overlap: 128` it writes `input_resize` `tiles`, `input_interpolation` `area`, `input_padding_placement` `bottom_right`, `input_tile_scale` `0.5`, `input_tile_overlap` `128`, against `keep_ratio`, `linear`, `centred` and `none` here.

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

To find your way in the stages below, the tree lists every function in the order it runs.

```text
main()                                                 give a semantic_segmentation --configuration_path
├── load_experiment_configuration()                    [STAGE 1] build_label_dictionary, check_segmentation_configuration
├── get_trainer_class(task)  -> SegmentationTrainer
├── SegmentationTrainer(...) -> BaseTrainer.__init__() [STAGE 2]
│   ├── get_run_name() -> build_segmentation_sections()
│   ├── build_dataloaders() -> get_segmentation_dataloaders() -> SegmentationDataset + build_class_lookup_table
│   ├── build_model() -> SemanticSegmenter (backbone, SegmentationDecoder, head) + compute_class_weights()
│   ├── initialise_pretrained_components() / build_optimizer() / get_tracked_metrics() / print_summary()
│   └── restore_last_model() -> load_model()           class weights of the current configuration set again
└── run_training_loop() -> run_training_iterations()   [STAGE 3]
    ├── get_next_batch() -> SegmentationDataset.__getitem__()   letterbox or tile, lookup table
    ├── run_model_iteration()
    │   ├── prepare_batch() / augment_batch()          masks follow every geometric type
    │   ├── forward_pass() -> Backbone (+ skips) -> SegmentationDecoder -> head -> compute_mask_losses()
    │   └── compute_batch_metrics()                    pixel accuracy and mean IoU of the batch
    ├── backward() / step_optimizer() / save_model() / validation pass / summaries
    └── run_test_evaluation()                          [STAGE 4]
        ├── per test batch: predict_batch() (tile by tile when tiled), accumulate_evaluation_batch(),
        │                   save_sample_images()
        ├── compute_evaluation_results()               region, boundary, curve metrics, recall intervals, files
        └── report_evaluation()

scripts/evaluation/evaluate_checkpoint.py              [STAGE 5]
└── evaluate_checkpoints() -> run_test_evaluation() for every checkpoint -> summarise_checkpoint_evaluations()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below come from the loader run on `configuration_synthetic_shapes.yaml`.

### Step 1.1 — `main()`
The entry point of every task (`classification_workflow.md`, Step 1.1): `parse_configuration_path()` reads `--configuration_path`, then `main()` loads the file, looks up the trainer of `ExperimentConfiguration.task` and starts its loop.

### Step 1.2 — `load_experiment_configuration()`
It splits the YAML into the two dictionaries, converts types, adds `label_mode` and `label_dictionary` to `Data` (Step 1.4) and runs the checks (Step 1.5). The segmentation has its own `SegmentationDecoder` and `SegmentationLoss` sections and a `Data.tiling` subsection. Its `evaluation` has no class thresholds or error images, but predicted masks, original resolution and boundary tolerance.

**Example: the sections and the evaluation settings**
```text
model_configuration keys: Optimization, Data, Backbone, SegmentationDecoder, SegmentationLoss
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'save_predicted_masks': False,
                           'original_resolution': False, 'boundary_tolerance': 2}
SegmentationDecoder:      {'upsampling': 'transposed', 'convolutions': {'5': [2, 64], ..., '1': [1, 16]},
                           'modules': {...}, 'feature_map_normalization': 'batch', 'dropout_rate': 0.0,
                           'deep_supervision_weights': {}, 'pretrained': {...}}
SegmentationLoss:         {'cross_entropy_weight': 1.0, 'focal_gamma': 0.0, 'label_smoothing': 0.0, 'class_weighting': 'none',
                           'dice_weight': 1.0, 'lovasz_weight': 0.0, 'boundary_weight': 0.0, 'boundary_distance': 3}
```

### Step 1.3 — `load_configuration()`
A plain YAML reader, as for every task: `number_iterations` is still the string `'1e10'` here, `learning_rate` 0.001.

### Step 1.4 — `build_label_dictionary()`
The mapping file gives the class of every category, a category being its pixel value in the masks (1 to 254; 0 background, 255 unlabelled). `'background': 0` is added when missing. Categories sharing an index are merged. In `"binary"` mode every category becomes class 1.

**Example: the label dictionary of the synthetic shapes**
```text
{1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 'background': 0}
10 entries, 10 distinct classes -> 10 output channels (9 shapes + the background)
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_segmentation_configuration()`: background at 0, categories that are mask values, tiling, no sequence settings, no `random_perspective` or `random_jigsaw` (`check_image_only_augmentation_types()`), one decoder level per skip connection, the losses and the evaluation settings. All problems are reported together, then the program exits.

**Example: output with `upsampling: "nearest"` and `dice_weight: -1.0` in the YAML**
```text
------------------------------------------------------
Invalid Configuration File : configuration_wrong.yaml
------------------------------------------------------
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

> The examples below come from building the trainer on `configuration_synthetic_shapes.yaml` (512 pixels, batch size 16), run folder in a temporary folder. The construction took 0.3 s on CPU.

### Step 2.1 — `SegmentationTrainer.__init__()`
Before the shared constructor, the trainer stores what its hooks read: `SegmentationLoss`, the deep supervision weights, the tiling settings, `save_predicted_masks`, `original_resolution` and `boundary_tolerance` (see Configuration Keys).

**Example: some attributes set before `super().__init__()`**
```text
self.tiling_configuration       = {'activated': False, 'scale': 1.0, 'overlap': 128}
self.boundary_tolerance         = 2
TASK_NAME = 'semantic_segmentation'   SAMPLE_NAME = 'Images'   COUNTS_FRAMES = False
```

### Step 2.2 — `BaseTrainer.__init__()`
The shared constructor (`classification_workflow.md`, Step 2.2) checks the task, then runs the steps below in the usual order (`input_image_size` 512, `batch_size` 16 here).

### Step 2.3 — `get_device()`
It prints `For Training We Will Be Using device: cpu` here (forced on the CPU; a normal run prints `cuda`).

### Step 2.4 — `get_run_name()` → `build_run_name()`
`build_segmentation_sections()` adds the decoder and class tokens below to the usual image tokens.

| Token | Meaning |
|---|---|
| `UTC` (`UBL`) | transposed (bilinear) upsampling |
| `DSE`, `DRES`, `DBN` | decoder modules and normalization |
| `D<layers>x<channels>` | every level, deepest first |
| `AUX<levels>` | deep supervision heads |
| `MC9` (`BIN`) | foreground classes, background left out (binary mode) |
| `TL<scale>` | between backbone and decoder for a tiled run (`..._B1x16.2x32.2x32.2x64.2x64.2x128_TL0.5_UTC-DBN-...`) |

Transfer tokens follow the decoder. Loss, head dropout and evaluation settings give no token, so they can change between resumptions.

**Example output**
```text
SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9
```

### Step 2.5 — `DataAugmenter()`
The augmenter of the classification; here the masks follow the images (Step 3.11). The configuration turns on 6 of 28 types (`random_crop`, `random_left_right_flip`, `random_rotation`, `brightness_contrast_saturation`, `random_noise`, `random_haze`) with probability 0.4.

### Step 2.6 — `setup_training_paths()`
As for every task (`classification_workflow.md`, Step 2.6): a resumed run keeps the classes of its `label_dictionary.json`, then the run folder gets `Tensorboard/`, `Weights/`, `label_dictionary.json` and `training_configuration.yaml`. With `class_weighting: "balanced"` it also gets `class_pixel_counts.json` (Step 2.18).

### Step 2.7 — `build_dataloaders()`
This hook builds the loaders (Step 2.8), the validation one only when the run reads it (`classification_workflow.md`, Step 2.7). It keeps `class_names` (background first) and `class_colours`, one BGR colour per class (black background), for the input viewer and the samples.

**Example output**
```text
class_names   = ['background', 'square', 'circle', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'plus_sign', 'star', 'octagon']
class_colours = [[0, 0, 0], [0, 0, 255], [0, 170, 255], ...]          (BGR: background black, square red, circle orange, ...)
```

### Step 2.8 — `get_segmentation_dataloaders()`
For each split it builds a `SegmentationDataset` (Step 2.9) from `<folder>/labels.json`, `<folder>/data/` and `<folder>/masks/`, wrapped in the Native loader (Step 2.10). Every image and mask is resized to the square input, so the default collate stacks them. With `Data.tiling`, samples are cut from images scaled by `tiling.scale`, as listed below.
- training: a random tile of the network size;
- validation: a tile fixed per image, so passes stay comparable;
- test: the whole scaled image, one per batch, segmented tile by tile in Stage 4.

Here the training split gives 2000 samples in 125 batches of 16 (one epoch = 125 iterations); validation and test give 200 samples in 13 batches each.

### Step 2.9 — `SegmentationDataset.__init__()` → `build_class_lookup_table()`
It reads the manifest: every sample gives its image file and its mask file, a single-channel `uint8` PNG of the image size (0 background, 255 unlabelled). The samples are shuffled once and packed (`SerializedRecords`, `classification_workflow.md`, Step 2.9), so loader workers never copy them. `build_class_lookup_table()` builds a 256-entry table converting a whole mask at once with `table[mask]`: 0 stays background, 255 becomes the ignored index (-100), every category becomes its class (unmapped ones become background). No image is read here.

**Example: a manifest sample, and the first sample after shuffling**
```text
labels.json:  {'id': 1, 'file_name': 'train_000000.png', 'mask_file_name': 'train_000000.png'}     (2000 samples, 9 categories)
kept:         {'identifier': 1351, 'file_name': 'train_001350.png', 'mask_file_name': 'train_001350.png'}
dataset[0]:   images [3, 512, 512] torch.uint8, masks [512, 512] torch.int64 holding the classes [0, 9], sample_identifiers 1351
```

### Step 2.10 — `build_native_loader()`
The PyTorch `DataLoader` of the classification (`classification_workflow.md`, Step 2.10).

**Example output** (the training loader; recorded with pinned memory, now used only on a discrete GPU)
```text
len(loader) = 125   batch_size = 16   num_workers = 1   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
Spreading samples over classes would mean reading every test mask, so every test image gets an empty class list. `choose_sample_identifiers()` then takes `number_sample_images` images evenly spread over the sorted identifiers.

**Example output** (8 images among 200)
```text
sample_image_identifiers  -> [1, 29, 58, 86, 115, 143, 172, 200]
names                     -> test_000000, test_000028, test_000057, ..., test_000199
```

### Step 2.12 — `build_model()`
This hook creates the `SemanticSegmenter` from `TASK_MODEL_CLASSES["semantic_segmentation"]` (Step 2.13), then copies the class weights into `model.class_weights` (Step 2.18). `BaseTrainer` moves the model to the device and dtype.

**Example: the components of the model**
```text
model.backbone                 Backbone                   438,368 parameters
model.segmentation_decoder     SegmentationDecoder        345,984 parameters
model.segmentation_head        Sequential                     170 parameters
model.deep_supervision_heads   ModuleDict                       0 parameters      (no deep supervision)
model.class_weights            buffer [10] torch.float32 [1.0, 1.0, ...]
```

### Step 2.13 — `SemanticSegmenter.__init__()`
The U-Net of `Read This First`: backbone, decoder (Step 2.15) and head (Step 2.16). The head is kept apart from the decoder, so a transferred decoder can serve other classes. Deep supervision heads add 1×1 convolutions on the levels named by `deep_supervision_weights`, left out of predictions and exports. The class count includes the background.

### Step 2.14 — `build_backbone()`
The backbone of every task (`classification_workflow.md`, Step 2.14); here its forward pass also returns every block output before pooling (Step 3.13). Its panel lists residual shortcuts only for a block adding one (`count_residual_additions()`): blocks 3 and 5 here. Blocks 2, 4 and 6 change their channels (16 → 32, 32 → 64, 64 → 128), which shifts their residual pairs, so with 2 layers they add none; block 1 never has shortcuts.

### Step 2.15 — `SegmentationDecoder.__init__()` (→ `compute_skip_block_numbers()`)
`compute_skip_block_numbers()` lists the blocks whose output is a skip connection: every block before the last (and the last too when it ends with pooling), deepest first. Each gets one level: an upsampling layer doubling the resolution (a 2×2 transposed convolution, or bilinear then 1×1 convolution), then a `ConvolutionBlock` without pooling refining the concatenated skip and upsampled features. Levels are keyed by their backbone block, so two decoders of the same levels share weight names.

**Example: the levels and the upsampling layers**
```text
skip blocks = [5, 4, 3, 2, 1]    skip channels = {'5': 64, '4': 64, '3': 32, '2': 32, '1': 16}    output channels = 16
(5): ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))  ...  (1): ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2))
```

### Step 2.16 — segmentation head and deep supervision heads
The head is a `Dropout2d` (built even at rate 0, so weight names do not depend on the rate) and a 1×1 convolution to one logit per class. Without `deep_supervision_weights`, the deep supervision `ModuleDict` is empty.

Here the head is `Dropout2d(p=0.0)` then `Conv2d(16, 10, kernel_size=(1, 1))`.

### Step 2.17 — the decoder panel of `print_summary()`
So that one can check the last level reaches the input resolution, the segmenter adds a panel: the bottleneck, each level's skip, convolutions and output, the head dropout and the class logits.

**Example output** (shortened)
```text
╭─────────────────────── Segmentation Decoder And Head ────────────────────────╮
│ Upsampling    : transposed                                                   │
│ Bottleneck    : 16x16x128                                                    │
│  - Level Of Block 5 : skip 32x32x64, 2 x 64 channels, output 32x32x64        │
│  ...                                                                         │
│  - Level Of Block 1                                                          │
│    - Output          : 512x512x16                                            │
│ Head Dropout Rate : 0.0                                                      │
│ Class Logits      : 512x512x10                                               │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 2.18 — `compute_class_weights()` (→ `get_class_pixel_counts()`)
With `class_weighting: "none"` (this run) every class weighs 1 in the cross entropy. With `"balanced"`, a class weighs `total pixels / (number of classes × pixels of the class)`, so small shapes weigh more than background. `get_class_pixel_counts()` caches the counts in `class_pixel_counts.json`, reused on resume while the training split and mapping are unchanged. `load_model()` sets the current weights again after loading.

### Step 2.19 — `initialise_pretrained_components()`
The backbone and the decoder can come from other runs. The head and deep supervision heads stay outside the decoder so a decoder can serve other classes. With `SegmentationDecoder.pretrained.class_layers` (named `load_class_layers` before 2026-11-24) they load too (`load_segmentation_class_layers()`). Every source class must be a class of this run, matched by order or name; other classes keep random rows, and missing deep supervision levels keep random weights. A deeper backbone loads its first blocks only (`loaded_backbone_blocks`, `classification_workflow.md`, Step 2.19).

In a run from scratch, `pretrained_checkpoint_paths` is `{'Backbone': None, 'SegmentationDecoder': None}`.

### Step 2.20 — `build_optimizer()`
Adam, 1 parameter group, learning rate 0.001, weight decay 0.0.

### Step 2.21 — `get_tracked_metrics()` + `MetricTracker.__init__()`
Only losses with a non-zero weight are tracked, so the console, TensorBoard and MLflow show what is trained, plus the batch pixel accuracy and mean IoU (Step 3.16). `get_percent_tracked_metrics()` returns `['pixel_accuracy', 'mean_iou']`, printed as percentages with changes as `+0.52 %`; TensorBoard and MLflow keep them between 0 and 1 (Step 3.20).

Here they are `total_loss`, `segmentation_loss` (Cross Entropy Loss), `dice_loss`, `pixel_accuracy` and `mean_iou`.

### Step 2.22 — `print_experiment_information()` (→ `describe_split()` → `count_split_samples()`)
Every path is a terminal link (`format_path_link()`). The splits print as `train (2000 images, 9 categories)`.

### Step 2.23 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The backbone and decoder panels are in Steps 2.14 and 2.17. The size panel lists the decoder and head on their own lines. The Export Sizes panel follows (PyTorch, TorchScript, ONNX, TensorRT and checkpoint sizes).

**Example output: the size panel** (condensed: each component prints parameters, trainable parameters and size on three lines, all trainable here)
```text
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone               438,368 parameters    1.76 MB     │
│ Segmentation Decoder   345,984 parameters    1.39 MB     │
│ Segmentation Head          170 parameters    0.00 MB     │
│ Total Model            784,522 parameters    3.15 MB     │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.24 — `view_input_batches()`
In case you turn on `diagnosis_modes.view_input_images`, a training image (augmented) and a validation image are shown with their masks painted in class colours, a panel naming their classes (`Classes : square, star`) and the augmentations drawn. It needs a display.

### Step 2.25 — `restore_last_model()` → `get_last_checkpoint_iteration()` / `load_model()`
The restoration of every task (`classification_workflow.md`, Steps 2.26 to 2.28), except that `load_model()` sets the current class weights again. A new run prints `No Initiation Model Weights Will Be Used...` (`start_iteration = 1`).

---

## Stage 3 — Training Loop

> The examples below come from a CPU run cut to 64 pixels, batch size 4 and 12 iterations, with a checkpoint every 6 (intermediate every 3), a summary every 4, a validation batch every 10, predicted masks saved and MLflow on. Its run name is `SEG_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9`.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (`classification_workflow.md`, Step 3.1). The table shows what each iteration does after its training step.

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | ✔ | nothing: iteration 12 is saved |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |
| Test evaluation | | | | | | ✔ | | | | | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()`
After the run, `Tensorboard/Train/` holds `Total Loss`, `Cross Entropy Loss`, `Dice Loss`, `Pixel Accuracy`, `Mean IoU` and `Learning Rate` (iterations 1 to 12), and `Tensorboard/Validation/` the same without the learning rate (iteration 10).

### Step 3.3 — `initialize_training_loop_state()` + `MetricTracker.create_trackers()`
It creates one zero tracker per tracked metric (`total_loss`, `segmentation_loss`, `dice_loss`, `pixel_accuracy`, `mean_iou`) with `interval_batches` and `start_time`.

### Step 3.4 — `start_experiment_tracking()`
The MLflow run is created in the experiment `semantic_segmentation`, in the local database of the experiments root.

**Example output**
```text
MLflow Run Created In The Experiment 'semantic_segmentation' : 6a88ad94d2d2440f932cb52cc5b8249b (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.5 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12, so checkpoint N holds N steps (`classification_workflow.md`, Step 3.6). It stores the label dictionary, the source runs of the two transferable components and `pretrained_backbone_blocks` (`None` unless loaded from a shorter backbone; absent from this older run).

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt ... Weights/Iteration_12/model_000012.pt   9,588,813 bytes each
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_segmentation_decoder']
model_state: 175 tensors, the first 'class_weights', then 'backbone.convolution_blocks.0. ...'
```

### Step 3.6 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration (`classification_workflow.md`, Step 3.7). See Stage 4.

### Step 3.7 — `get_next_batch()` → `SegmentationDataset.__getitem__()`
`get_next_batch()` drops a batch that fails to load (`classification_workflow.md`, Step 3.8). `__getitem__()` does the work below.
- Reads the image with `read_image()` (BGR, `uint8`) and the mask (refused unless single-channel `uint8`), and checks both sizes match.
- Without tiling, `preprocess_image()` letterboxes both: the image bilinear with black padding, the mask with `cv2.INTER_NEAREST_EXACT` and padding 255.
- With tiling, `cut_tile()` scales both the same way and cuts the tile of the split.
- The lookup table of Step 2.9 turns mask values into class indices.

**Example: the first batch of the run**
```text
images               [4, 3, 64, 64] torch.uint8
masks                [4, 64, 64] torch.int64       class index of every pixel, -100 on the padding and the unlabelled pixels
sample_identifiers   [4] torch.int64               [120, 1237, 1626, 1323]
```

### Step 3.8 — `set_training_mode()`
`model.train()`: the segmenter then returns only logits and losses, no per-pixel softmax.

### Step 3.9 — `run_model_iteration()`
The same function for training and validation batches (`classification_workflow.md`, Step 3.11): prepare, augment, forward, sum of the `_loss` outputs, batch metrics, tracking.

### Step 3.10 — `prepare_batch()`
The images become `torch.float32` on the device; the masks stay `[4, 64, 64]` `torch.int64`.

### Step 3.11 — `augment_batch()` → `DataAugmenter.augment_batch(masks=...)`
The trainer overrides this hook so masks follow their images. Every geometric type moves the mask with nearest interpolation on pixel centres (`InterpolationMode.NEAREST_EXACT`); pixels uncovered by a rotation become ignored. Photometric types leave the mask untouched.

**Example: which images and masks of the first batch were changed** (`probability: 0.4`)
```text
images: changed per sample [False, True, True, False]
masks:  changed per sample [False, True, True, False]      <- both images got a geometric augmentation
```

### Step 3.12 — `forward_pass()` → `SemanticSegmenter.forward(masks=...)`
With the masks passed, the segmenter returns its losses with its logits.

It returns `class_logits` `[4, 10, 64, 64]` with `segmentation_loss` 2.0932 and `dice_loss` 0.9559.

### Step 3.13 — `Backbone.forward(return_pre_pooling_outputs=True)`
The backbone returns its final map and every block output before pooling, keyed by block number: the skip connections.

**Example: shapes for a batch of 4 images of 64 pixels**
```text
final feature map [4, 128, 2, 2]
outputs before pooling  '1': [4, 16, 64, 64]   '2': [4, 32, 32, 32]   '3': [4, 32, 16, 16]   '4': [4, 64, 8, 8]   '5': [4, 64, 4, 4]
                        '6': [4, 128, 2, 2]    <- the last block has no pooling: it is the final map, not a skip connection
```

### Step 3.14 — `SegmentationDecoder.forward()` + segmentation head
From the deepest level to the finest, each level doubles its input, concatenates the skip of its block before the upsampled features, and refines with its convolution block. When the two differ by a pixel, the upsampled map is resized bilinearly to the skip. The head turns `[B, 16, 64, 64]` into `[B, 10, 64, 64]` class logits.

**Example: shapes through the decoder**
```text
[4, 128, 2, 2] -> level 5 [4, 64, 4, 4] -> 4 [4, 64, 8, 8] -> 3 [4, 32, 16, 16] -> 2 [4, 32, 32, 32] -> 1 [4, 16, 64, 64]
-> head [4, 10, 64, 64]
```

### Step 3.15 — `compute_mask_losses()`
Each `SegmentationLoss` loss with a non-zero weight is computed on the labelled pixels and multiplied by its weight. The table lists the four losses.

| Loss | Output key | What it measures |
|---|---|---|
| cross entropy | `segmentation_loss` | weighted by `class_weights`, with label smoothing and the focal factor `(1 - p)^gamma` (lower loss for well classified pixels), normalised by the labelled pixel weights |
| Dice | `dice_loss` | one minus the mean soft Dice (`2 × intersection / (predicted + true)`) of the classes in the batch, so small objects count as much as background |
| Lovász-Softmax | `lovasz_loss` | a direct surrogate of the IoU of every class |
| boundary | `boundary_loss` | errors weighted by their closeness to the true edges (`boundary_distance` pixels) |

With deep supervision, the same losses on every head's upsampled logits, times their level weight, sum into `deep_supervision_loss`. Equal probabilities for every class give a cross entropy of ln 10 = 2.30; the first batch gives `segmentation_loss` 2.0932, `dice_loss` 0.9559, `total_loss` 3.0491.

### Step 3.16 — `compute_batch_metrics()` (→ `compute_pixel_confusion_matrix()` + `compute_segmentation_metrics()`)
The predicted class of a pixel is the argmax of its logits. `compute_pixel_confusion_matrix()` counts the labelled pixels of every (true, predicted) pair with one `bincount`. `compute_segmentation_metrics()` gives the pixel accuracy and the mean IoU of the classes present or predicted in the batch. No second forward pass runs. The first batch gives `pixel_accuracy` 0.2665 and `mean_iou` 0.0424.

### Step 3.17 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
The Train curves start at `Total Loss` 3.0491, 2.9684, 2.9365, 2.9298 and `Pixel Accuracy` 0.2665, 0.4165, 0.4551, 0.5035 (iterations 1 to 4).

### Step 3.18 — `training_loss.backward()` + `step_optimizer()`
The optimizer steps with unclipped gradients at the schedule's learning rate (`classification_workflow.md`, Step 3.22). The gradient norm, measured for this document, is 1.8968, 1.7794 and 1.8824 at iterations 1 to 3, at a constant learning rate of 0.001.

### Step 3.19 — validation pass
**Example: iteration 10**
```text
Tensorboard/Validation at iteration 10: Total Loss 2.9886, Cross Entropy Loss 2.0299, Dice Loss 0.9587,
                                        Pixel Accuracy 0.9136, Mean IoU 0.1126
```

### Step 3.20 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The interval sums are divided by its batch count, so the batch mean IoU is averaged over batches, each with its own classes. These outputs predate 2026-10-25, when the mean IoU still left out classes predicted but absent; a new run can show lower `Mean IoU` lines.

**Example output: iteration 12**
```text
|    Iteration: 12    |      Total Images: 48      |    Interval Images: 16    |   Throughput: 8.1 Images Per Second   |
Moving Average of Train Total Loss                     :     2.7779    -0.0718 Best
Moving Average of Valid Total Loss                     :     2.9886
...
Moving Average of Train Pixel Accuracy                 :    45.67 %    -2.84 %           <- red: lower than at iteration 8
Moving Average of Train Mean IoU                       :     8.29 %    +0.48 % Best
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
```

### Step 3.21 — end of the loop
Iteration 12 is saved and evaluated inside its iteration, so the loop saves and evaluates nothing more (`classification_workflow.md`, Step 3.25). It ends with `Last Checkpoint Saved At Iteration 12. Exiting Training.`

---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below come from checkpoints 6 and 12 of the Stage 3 run, on its whole test split (200 images of 64×64), with `per_class_report: True` and `save_predicted_masks: True`. Iteration 6 then held 5 steps (now 6). They predate the per-class recall intervals (2026-11-01) and the ROC and precision-recall curves; what those add is listed in each step.

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (`classification_workflow.md`, Step 4.1); the segmentation `evaluation_context()` changes nothing. The progress bar reads `Test Evaluation Iteration 6: 100%|██████████| 50/50` (200 images, batch 4).

### Step 4.2 — `create_evaluation_accumulator()`
Keeping every pixel probability would take too much memory, so the accumulator keeps counts on the device: the `[K, K]` pixel confusion matrix, the `[K, 6]` boundary counts, with `per_class_report` the `[K, 2, 10000]` `score_histograms` (Step 4.5), the per-image `image_true_positive_pixels` and `image_true_pixels`, the image count and the `Predicted_Masks` folder.

**Example: the accumulator at the end of the loop** (before the curves; it now also holds `'score_histograms': [10, 2, 10000] torch.int64`)
```text
{'confusion_matrix': [10, 10] torch.int64, 'boundary_counts': [10, 6] torch.int64, 'number_images': 200,
 'predicted_masks_folder': PosixPath('.../Evaluation/Iteration_6/Predicted_Masks')}
```

### Step 4.3 — `prepare_batch()`
The same function as in training (Step 3.10).

### Step 4.4 — `predict_batch()`
Without tiling, it runs `SemanticSegmenter.forward()` without masks; in `eval()` mode it also returns the per-pixel softmax (`class_logits` and `class_probabilities`, both `[4, 10, 64, 64]` float32). With tiling, `segment_image_by_tiles()` segments every whole scaled test image in network-size tiles overlapping by `Data.tiling.overlap` pixels, `batch_size` tiles at a time, averaging the overlapping probabilities.

### Step 4.5 — `accumulate_evaluation_batch()` (→ `compute_pixel_confusion_matrix()` + `compute_boundary_counts()`)
The predicted class map (argmax) of every image is compared with its mask at the network resolution, padding and unlabelled pixels ignored. With `original_resolution: True` it is compared with the original mask, the probabilities first cut out of the letterbox and resized back (`restore_prediction_to_original_image()`). The list below gives what is added per batch.
- The pixel confusion matrix and the boundary counts.
- `compute_image_class_pixel_counts()`: per image and class, the labelled pixels and those predicted as the class, for the recall interval (Step 4.10).
- `compute_boundary_counts()`: per class, the bands within `boundary_tolerance` of the true and predicted contours (intersection, union) and the edge pixels near the other contour.
- With `per_class_report`, `compute_pixel_score_histograms()`: per class, a 10,000-bin histogram (`SCORE_HISTOGRAM_BINS`) of the class probability, positives apart from negatives.
- With `save_predicted_masks`, a single-channel PNG of the class map per image (`Predicted_Masks/`, 200 files here).

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
For every fixed sample in the batch, `build_sample_image()` lays out four panels: the input, the true mask, the predicted mask and the error map, with the pixel accuracy as caption. The legend names the classes, then the error colours below (`build_segmentation_error_map()`).
- `Correct` (green): a pixel of a class given its class.
- `Wrong Class` (red): another class, or a class where the truth is background.
- `Missed` (yellow): a pixel of a class predicted as background.
- Ignored pixels are gray; background predicted as background stays dark.

Large tiled images are reduced to 1024 pixels. The image goes to `Samples/` and TensorBoard.

For example, `Evaluation/Iteration_12/Samples/test_000028.png` is 268 x 112 pixels, four 64 x 64 panels captioned `Pixel Accuracy 90.06 %` (titles cut at this size).

### Step 4.7 — `compute_evaluation_results()`
After the loop it computes the metrics (Steps 4.8 and 4.9), writes the files (Steps 4.10 to 4.12) and organizes the report (Step 4.13). The headline metric, used by Stage 5 and MLflow to pick the best checkpoint, is the mean IoU.

**Example: the metrics of iteration 12**
```text
metrics = {'Mean IoU': 0.079, 'Mean Dice': 0.0939, 'Mean Boundary IoU': 0.0195, 'Mean Boundary F1': 0.0608,
           'Pixel Accuracy': 0.7222, 'Frequency Weighted IoU': 0.6748}
```
The 819,200 labelled pixels are 200 × 64 × 64 (no padding). With the curves, `metrics` adds `'Macro ROC AUC'` and `'Macro Average Precision'`, with their files and scalars (Step 4.13).

### Step 4.8 — `compute_segmentation_metrics()`
All region metrics come from the pixel confusion matrix (rows true, columns predicted), as the table defines.

| Metric | Definition |
|---|---|
| IoU of a class | right pixels over the union of its true and predicted pixels |
| Dice of a class | F1 of its pixels (twice the intersection over their sum) |
| precision, recall | as in classification |
| pixel accuracy | diagonal over all labelled pixels |
| mean IoU, mean Dice | means over classes present or predicted (painted without true pixels counts 0; neither present nor predicted is left out) |
| frequency-weighted IoU | IoU weighted by each class's share of pixels, dominated by background |

**Example: `Evaluation/Iteration_12/confusion_matrix.csv`** (pixel counts; the network predicts only background, square and star)
```text
true \ predicted,background,square,circle,triangle,rectangle,pentagon,hexagon,plus_sign,star,octagon
background,587460,63058,0,0,0,0,0,0,89288,0
square,8986,3208,0,0,0,0,0,0,3689,0
...
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
Region metrics hardly see contour errors: an object predicted 3 pixels too wide keeps a high IoU while none of its contour is right. From the counts of Step 4.5, the **boundary IoU** of a class is the IoU of the bands within the tolerance of its true and predicted contours. The **boundary F1** combines the share of the predicted contour near the true one (precision) and of the true contour near the predicted one (recall). Their class means are headline metrics.

**Example: iteration 12** (`boundary_tolerance: 2`)
```text
background   boundary IoU 0.1702   boundary F1 0.4931
square       boundary IoU 0.0173   boundary F1 0.0730
Mean Boundary IoU 0.0195   Mean Boundary F1 0.0608
```

### Step 4.10 — `build_segmentation_class_rows()` + `write_class_metrics_file()`
Pixels of an image are correlated (a missed object loses all its pixels), so an exact interval over pixels would be far too narrow. `compute_bootstrap_recall_intervals()` instead gives a Korn-Graubard interval per class: 1000 resamples of whole test images (fixed seed) give the recall variance, hence an effective sample size `p (1 - p) / variance`, and the bounds are the Clopper-Pearson interval at that size. A recall of 0 or 1, or a single image holding the class, falls back to the exact interval over those images.

The bounds join the row as `recall_lower` and `recall_upper` after `recall`, empty for a class without true pixels. `class_metrics.csv` holds one row per class, in index order, at every evaluation. With `per_class_report`, `compute_histogram_score_curves()` turns the histograms into one-against-rest ROC and precision-recall curves, exact at the bin edges. Each class gets `roc_auc` and `average_precision`, and their means (`compute_macro_curve_metrics()`) become `Macro ROC AUC` and `Macro Average Precision`. A class without pixels has neither; a class covering every labelled pixel has no ROC AUC. The example predates these two columns.

**Example: `Evaluation/Iteration_12/class_metrics.csv`**
```text
class_index,class,pixels,iou,dice,boundary_iou,boundary_f1,precision,recall
0,background,739806,0.746387,0.854778,0.170248,0.493107,0.925533,0.794073
1,square,15883,0.034965,0.067568,0.017344,0.073005,0.040570,0.201977
2,circle,10212,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
...
8,star,5654,0.008555,0.016965,0.007485,0.041788,0.008937,0.166608
```

### Step 4.11 — `save_confusion_matrix_table()` + `save_confusion_matrix_image()`
`confusion_matrix.csv` is the pixel confusion matrix of Step 4.8. With `per_class_report`, the image (`Pixel Confusion Matrix · Iteration N · Mean IoU ... · Pixel Accuracy ...`) shows only the share of every true class (`74.64 %`), counts being too large. `save_score_curve_images()` then draws the curves as for the classification (`classification_workflow.md`, Step 4.15) into `roc_curves.png` and `precision_recall_curves.png`, also logged to TensorBoard and MLflow.

### Step 4.12 — the notes and the predicted masks
The notes name what was scored: the labelled pixel count and the resolution (`network resolution`, `original resolution of the images`, or `images scaled by <scale>, segmented tile by tile`), and the boundary tolerance. With `save_predicted_masks`, the `Predicted_Masks` folder and its image count are added.

### Step 4.13 — `build_segmentation_class_table()` + `build_segmentation_class_scalars()`
The per-class table sorts classes by IoU, worst first. `build_recall_interval_table()` adds the `Per Class Recall` table (true pixels, recall, `95 % Lower`, `95 % Upper`), worst recall first. Rates print as percentages (`percent_columns`, `percent_metrics`). With `per_class_report` only, each class with pixels gets 6 scalars (`IoU`, `Dice`, `Boundary IoU`, `Boundary F1`, `Precision`, `Recall`): 60 for 10 classes. `ROC AUC` and `Average Precision` raise this to 80, plus 8 headline metrics, in the 512 pixel run of `experiments/readme_screenshots`.

### Step 4.14 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (also appended to `Results.txt`)
```text
Iteration 12 · 200 Test Images · 2026-10-09 10:04 ──────────────────────────────────────────────────────────────────────
  Mean IoU            7.90 %    Mean Dice                9.39 %
  Mean Boundary IoU   1.95 %    Mean Boundary F1         6.08 %
  Pixel Accuracy      72.22 %   Frequency Weighted IoU   67.48 %
  ...
  Per Class Metrics
                                             Boundary   Boundary
   Class        Pixels       IoU      Dice        IoU         F1   Precision    Recall
   circle        10212    0.00 %    0.00 %     0.00 %     0.00 %      0.00 %    0.00 %
   ...
   square        15883    3.50 %    6.76 %     1.73 %     7.30 %      4.06 %   20.20 %
   background   739806   74.64 %   85.48 %    17.02 %    49.31 %     92.55 %   79.41 %
  - Results : Results.txt
  ...
```
File names are terminal links. With the curves, the block adds the macro curve line, two columns and the two curve images.

**Example: the run folder, TensorBoard and MLflow after the two evaluations**
```text
Evaluation/Iteration_6/ and Iteration_12/   Predicted_Masks/ (200)  Samples/ (8)  class_metrics.csv  confusion_matrix.csv  confusion_matrix.png
Tensorboard/Test: 66 scalars (6 global + 60 per class), 9 images (8 samples + Confusion Matrix)
    scalar Mean IoU          6: 0.0024, 12: 0.0790
    scalar Pixel Accuracy    6: 0.0196, 12: 0.7222
MLflow: test/Mean IoU, ..., test/best_Mean IoU (steps 6, 12), 60 test_class/<metric>/<class>,
        artifacts evaluation/iteration_6/ and iteration_12/: class_metrics.csv, confusion_matrix.png
```
Iteration 6 predicted `square` almost everywhere (pixel accuracy 1.96 %, square recall 99.66 %); iteration 12 learned that most pixels are background.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below come from `scripts/evaluation/evaluate_checkpoint.py` on the Stage 3 run (checkpoints 3, 6, 9 and 12, the first three saved before their training step) with `iteration = "all"`, `batch_size = 16`, on the first 40 test images. They predate the per-class recall intervals and the curves.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (`classification_workflow.md`, Stage 5): give the run as `--experiment_folder`, and `--test_data_folder` for another labelled split (`labels.json`, `data/`, `masks/`). `--maximum_error_images_per_class_pair` unless it is `None`, and `--save_predictions`, `--target_precisions`, `--nms_criterion` and `--nms_distance_exponent` when given, are skipped with a yellow warning: the task has no error images, predictions file, class thresholds or `Detector`. The evaluation goes to `Checkpoint_Evaluation/` of the run (`Checkpoint_Evaluation_split_<parent folder>_<split folder>/` for another split).

### Step 5.2 — trainer construction
The script loads the run's `training_configuration.yaml` with the classes of its `label_dictionary.json`, points it to `Checkpoint_Evaluation/`, the split and the batch size, and builds a `SegmentationTrainer`. With `class_weighting: "balanced"` it would count the training mask pixels again, the cache living in the trainer's folder.

### Step 5.3 — `get_checkpoint_iterations()`
With `iteration = "all"` it returns `[12, 9, 6, 3]`.

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'semantic_segmentation' : 342ebb90469341998a33bbefd91e4b09 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
**Example output** (the file name is a terminal link to `runs/<run>/Checkpoint_Evaluation/evaluation_summary.csv`)
```text
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
  - Best Checkpoint : iteration 12 (Mean IoU 7.13 %)

iteration,Mean IoU,Mean Dice,...
12,0.07127240554791028,0.08981075670647244, ...
9,0.054483986969393404,0.0767030110689235, ...
```
The current summary also has `Macro ROC AUC` and `Macro Average Precision` after `Frequency Weighted IoU`, and the per-class curve columns.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of iteration 12 on the 40 images** (shortened)
```text
Evaluating : model_000012.pt
...
Iteration 12 · 40 Test Images · 2026-10-09 10:04 ───────────────────────────────────────────────────────────────────────
  Mean IoU            7.13 %    Mean Dice                8.98 %
  Pixel Accuracy      64.28 %   Frequency Weighted IoU   59.74 %
  ...
   square         5894    4.96 %    9.45 %     2.03 %     9.93 %      6.17 %   20.19 %
   background   148235   65.83 %   79.39 %    12.61 %    40.82 %     91.44 %   70.15 %
```
The sample images come from these 40 images, so they differ from those of Stage 4.
