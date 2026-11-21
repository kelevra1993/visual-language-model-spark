# Semantic Segmentation Workflow

This document follows one semantic segmentation run from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`; here they keep a short explanation and the real output of the segmentation run, and the steps specific to the segmentation are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training:
- **Stages 1 and 2**: `vision_training/configurations/semantic_segmentation/configuration_synthetic_shapes.yaml` (synthetic shapes, 9 shape classes plus the background, 512 pixel images, batch size 16), with the run folder redirected to a temporary folder. The dataset of `configuration_example.yaml` is not on this machine. The examples read the 9-class synthetic dataset generated on 2026-10-21, before the 26 other shapes were added; a dataset generated with the generator of the repository holds the 35 shapes, and its runs end with `MC35`.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel images, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers (a setting then, now fixed at 1), with the predicted masks saved and the MLflow tracking turned on. After 12 iterations the network mostly predicts the background, so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

## Read This First

The semantic segmentation task (`task: "semantic_segmentation"`, run name code `SEG`) learns the class of every pixel of one image. The model is a U-Net: the shared `Backbone` is the encoder, its block outputs before their pooling are the skip connections, the `SegmentationDecoder` brings its last feature map back to the input resolution level by level, and a 1x1 head gives K logits per pixel, the background being class 0 (K = 10 for the 9 synthetic shapes). The labels are single-channel uint8 PNG masks holding the category of every pixel, 255 for the pixels left out of the losses and metrics. Choose it when every pixel needs a class and the objects need not be told apart (two touching squares form one region); choose INS for the mask of every object, DET for boxes, KPT for named points, CLS for one label per image. Images much larger than the network input can be trained on tiles (`Data.tiling`) and scored at their original resolution (`evaluation.original_resolution`).

### The Task At A Glance

So that the stages below can be followed on one page, the whole task fits in this diagram, from the files of a split to the exported segmenter, with the shapes of the synthetic configuration.

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
files          Evaluation/Iteration_<n>/ class_metrics.csv, confusion_matrix.csv and .png, roc_curves.png,
               precision_recall_curves.png, Samples/, Predicted_Masks/; Results.txt
export         InferenceSegmenter: images [B, 3, 512, 512] -> class_map [B, 512, 512] int64,
               class_probabilities [B, 10, 512, 512]; the input contract of the run (keep_ratio, 512, BGR, version 2)
               in preprocessing.json next to the model and in the metadata of model.onnx
```

### Files To Know

If you are looking for the code of a step, these are the files of the task and their main functions.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/segmentation_trainer.py` | `SegmentationTrainer`: `build_dataloaders`, `build_model`, `compute_class_weights`, `augment_batch`, `compute_batch_metrics`, `predict_batch`, `accumulate_evaluation_batch`, `build_sample_image`, `compute_evaluation_results` |
| Dataset and loaders | `vision_training/data/segmentation/segmentation_dataset.py`, `vision_training/data/serialized_records.py`, `vision_training/data/loaders.py` | `SegmentationDataset` (`cut_tile`, `count_class_pixels`), `SerializedRecords` (`find_record`), `build_class_lookup_table`, `get_segmentation_dataloaders`, `build_native_loader` |
| Model | `vision_training/models/segmentation/semantic_segmenter.py`, `vision_training/models/segmentation/segmentation_decoder.py`, `vision_training/models/backbones/backbone.py` | `SemanticSegmenter` (`compute_mask_losses`), `SegmentationDecoder`, `compute_skip_block_numbers`, `Backbone.forward(return_pre_pooling_outputs=True)` |
| Losses | `vision_training/losses/segmentation_losses.py`, `vision_training/utilities/mask_morphology.py` | `compute_segmentation_cross_entropy`, `compute_dice_loss`, `compute_lovasz_softmax_loss`, `compute_boundary_loss`, `compute_edge_distance_weights` |
| Metrics and evaluation | `vision_training/evaluation/segmentation/segmentation_metrics.py`, `boundary_metrics.py`, `segmentation_report.py` | `compute_pixel_confusion_matrix`, `compute_segmentation_metrics`, `compute_image_class_pixel_counts`, `compute_pixel_score_histograms`, `compute_boundary_counts`, `compute_boundary_metrics`, `build_segmentation_class_rows`, `build_segmentation_class_table`, `build_segmentation_class_scalars` |
| Recall intervals | `vision_training/evaluation/recall_intervals.py` | `compute_bootstrap_recall_intervals`, `describe_bootstrap_interval`, `build_recall_interval_table` |
| Curves and confusion matrix | `vision_training/evaluation/classification/score_curves.py`, `confusion_matrix.py` | `compute_histogram_score_curves`, `compute_macro_curve_metrics`, `save_score_curve_images`, `save_confusion_matrix_table`, `save_confusion_matrix_image` |
| Tiles and original resolution | `vision_training/models/segmentation/sliding_window.py`, `vision_training/data/preprocessing.py` | `segment_image_by_tiles`, `restore_prediction_to_original_image` |
| Sample images and input viewer | `vision_training/utilities/visualization.py` | `build_segmentation_sample_grid`, `build_segmentation_error_map`, `overlay_segmentation_mask` |
| Configuration checks | `vision_training/configuration_processing/segmentation_checks.py` | `check_segmentation_configuration` |
| Run name | `vision_training/configuration_processing/run_name.py` | `build_run_name`, `build_segmentation_sections` |
| Synthetic data generator | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset`, `build_segmentation_labels` |
| Export | `vision_training/models/segmentation/inference_segmenter.py`, `vision_training/deployment/export_specifications.py`, `vision_training/deployment/input_contract.py` | `InferenceSegmenter` (model name `semantic_segmenter`, the deep supervision heads left out), `compare_segmentations`, `build_input_contract` (the tile keys of a run trained on tiles) |
| Export sizes at launch | `vision_training/deployment/export_size_estimates.py` | `estimate_export_sizes`, `probe_inference_graph`, `print_export_size_summary` |
| Standalone evaluation | `scripts/evaluation/evaluate_checkpoint.py`, `vision_training/evaluation/checkpoint_evaluation.py` | `evaluate_checkpoints`, `summarise_checkpoint_evaluations` |

### Run It

In order to generate, train, evaluate and export the synthetic shapes segmenter from scratch, run these commands from the root of the repository, after setting the values of their main blocks.

```bash
# 1. Synthetic dataset: in the main block of vision_training/data/synthetic/create_synthetic_dataset.py, set
#    task = "semantic_segmentation" and output_folders["semantic_segmentation"] (50,000 / 200 / 200 images of 512
#    pixels with up to 6 shapes and their class masks, the 35 shapes and the background)
uv run python -m vision_training.data.synthetic.create_synthetic_dataset

# 2. Training: set experiment_folder, the three data folders and Data.label_mapping.mapping_file_path first, and
#    number_iterations, 1e10 in this file (2000 for the 9-class run of experiments/readme_screenshots, trained on
#    2026-10-21); Ctrl+C saves a checkpoint
uv run python -m vision_training.main --configuration_path="vision_training/configurations/semantic_segmentation/configuration_synthetic_shapes.yaml"

# 3. Evaluation of checkpoints: set run_name, experiment_folder, iteration (None, an integer or "all"),
#    test_data_folder, batch_size, track_evaluation and maximum_error_images_per_class_pair = None (Step 5.1) in the
#    main block of scripts/evaluation/evaluate_checkpoint.py, which writes in <run>/Checkpoint_Evaluation/
uv run python -m scripts.evaluation.evaluate_checkpoint

# 4. Export (CUDA only): set experiment_folder, iteration and formats in the main block of
#    vision_training/deployment/export_model.py; every model gets the input contract of the run in
#    <model>/<version>/preprocessing.json and, for ONNX, in the metadata of model.onnx (README, Input Contract);
#    the served models are replaced only once every format matches the eager model, a failed check leaving them as
#    they were
uv run python -m vision_training.deployment.export_model
```

The recipe of this configuration, with batch size 32 and the backbone of a shape pre-trainer, reached a mean IoU of 4.44 % at 1000 iterations on the 35-class dataset generated on 2026-11-19, predicting mostly the background; that run was stopped at about 1100 iterations, and the recipe for the 35 shapes is still to be studied.

### Configuration Keys To Know First

Before changing the synthetic configuration, these are the keys that change the model, the losses or the evaluation the most.

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `SegmentationDecoder.convolutions` | `{"5": [2, 64], "4": [2, 64], "3": [2, 32], "2": [2, 32], "1": [1, 16]}` | one level per backbone block before the last, keyed by that block, `[layers, channels]` from the deepest level (`D2x64.2x64.2x32.2x32.1x16`) |
| `SegmentationDecoder.upsampling` | `"transposed"` | a learned 2x2 transposed convolution of stride 2 (`UTC`); `"bilinear"` is a bilinear upsampling followed by a 1x1 convolution (`UBL`) |
| `SegmentationDecoder.modules` | `squeeze_and_excite: False`, `residual_nets: False` | Squeeze-and-Excitation and residual pairs of the decoder blocks; the residual pairs add nothing to the levels of at most 2 layers of this file (pitfall below) |
| `SegmentationDecoder.deep_supervision_weights` | `{}` | 1x1 class heads on intermediate levels (`AUX<levels>`), trained with the same losses times their weight, left out of the inference and the exports |
| `SegmentationLoss.cross_entropy_weight`, `SegmentationLoss.dice_weight` | `1.0`, `1.0` | the two losses of the total loss; `lovasz_weight` and `boundary_weight` are 0, and the region losses cannot all be 0 |
| `SegmentationLoss.class_weighting`, `focal_gamma`, `label_smoothing` | `"none"`, `0.0`, `0.0` | `"balanced"` weights a class by total pixels / (K x pixels of the class), counted once and cached in `class_pixel_counts.json` |
| `Data.tiling` | `activated: False`, `scale: 1.0`, `overlap: 128` | random tiles of the scaled images for the training, fixed ones for the validation, whole test images segmented tile by tile (`TL<scale>`) |
| `ExperimentConfiguration.evaluation.boundary_tolerance` | `2` | distance in pixels within which a contour pixel is right, for the boundary IoU and F1 |
| `ExperimentConfiguration.evaluation.save_predicted_masks`, `original_resolution` | `False`, `False` | the class index PNG of every test image in `Predicted_Masks/`; the scores against the original masks after `restore_prediction_to_original_image()` |
| `ExperimentConfiguration.evaluation.per_class_report` | `True` | per-class table, per-class recall table and scalars, confusion matrix image, ROC and precision-recall curves from the 10,000 bin histograms of the pixel probabilities |

### What A Run Writes

Whenever you look for a result of the run, these are the files and curves its training and the evaluations of its checkpoints write.

- `Results.txt`: one block per evaluated checkpoint with `Mean IoU` (the headline metric), `Mean Dice`, `Mean Boundary IoU`, `Mean Boundary F1`, `Pixel Accuracy`, `Frequency Weighted IoU`, with `per_class_report` `Macro ROC AUC` and `Macro Average Precision`, the notes (labelled pixels and their resolution, boundary tolerance, predicted masks, sample images) the per-class table sorted by IoU, worst first, and the `Per Class Recall` table (the pixel recall of every class with its 95 % confidence interval, a Korn-Graubard interval from a bootstrap over the test images) sorted by recall, worst first. Every rate of the block is written as a percentage with two decimals (`7.90 %`), the pixel counts keeping their values; the files, TensorBoard and MLflow keep every rate between 0 and 1.
- `Evaluation/Iteration_<n>/`: `class_metrics.csv` (pixels, IoU, Dice, boundary IoU and F1, precision, recall with the bounds of its 95 % confidence interval, `recall_lower` and `recall_upper`, and with `per_class_report` ROC AUC and average precision) and `confusion_matrix.csv` (pixel counts) at every evaluation; `confusion_matrix.png` (shares of every true class), `roc_curves.png` and `precision_recall_curves.png` with `per_class_report`; `Samples/<image>.png` (Input, Ground Truth, Prediction and Errors panels, 2060x560 for 512 pixel images); `Predicted_Masks/<image>.png` with `save_predicted_masks`. No `EMA/` folder: the moving average of the weights is off in this configuration.
- `class_pixel_counts.json` in the run folder with `class_weighting: "balanced"`.
- `Tensorboard/Train` and `Validation`: `Total Loss`, `Cross Entropy Loss`, `Dice Loss`, `Pixel Accuracy`, `Mean IoU` (and `Learning Rate`); `Tensorboard/Test`: 88 scalars in the 9-class run of `experiments/readme_screenshots` trained on 2026-10-21 (the 8 headline metrics and, for each of the 10 classes, `IoU`, `Dice`, `Boundary IoU`, `Boundary F1`, `Precision`, `Recall`, `ROC AUC` and `Average Precision`) and 11 images (the 8 samples, `Confusion Matrix`, `ROC Curves`, `Precision Recall Curves`).
- `Weights/Iteration_<n>/model_<n>.pt`: about 9.6 MB, the 3.15 MB of the weights and the two Adam moments, with the source runs of `Backbone` and `SegmentationDecoder`.

### Pitfalls

Should a result of the task look wrong, check these behaviours first.

- `residual_nets: True` adds no shortcut in the backbone blocks 2, 4 and 6, which change the channel count with 2 layers, nor in the decoder levels of this file: the input of a level (skip plus upsampled channels) never has the channels of its output, so its residual pairs are shifted by one layer and the first shortcut closes at the third layer, which no level of 1 or 2 layers has (`ConvolutionBlock.residual_parity` and `count_residual_additions()` of `convolution_block.py`). A decoder level of 3 or 4 layers adds one shortcut. The backbone summary lists the shortcuts for blocks 3 and 5 only (Step 2.14).
- `Mean IoU` and `Mean Dice` average over the classes present in the true masks or predicted, a class painted without any true pixel counting with 0 (`compute_segmentation_metrics()` of `segmentation_metrics.py`). Before 2026-10-25 they left out the classes predicted but absent, so older evaluations can be higher. The batch `Mean IoU` of the console averages the classes of each batch, then the batches (Step 3.20).
- Before 2026-11-20 the masks were resized with the plain nearest interpolation (`cv2.INTER_NEAREST`), about half a pixel to the bottom right of their images, `random_crop`, `random_aspect_crop` and `random_zoom_out` adding another half pixel, and the restored class maps of the inference clients were shifted alike. They are now resized on the pixel centres like the images (`cv2.INTER_NEAREST_EXACT` in `segmentation_dataset.py`, `InterpolationMode.NEAREST_EXACT` in `augmentations.py`, `nearest-exact` in `restore_prediction_to_original_image()`), so a run trained earlier learned masks shifted by up to about a pixel, which can lower its boundary scores (not measured).
- The `SegmentationLoss` settings, the head dropout, the deep supervision weights and the tile overlap give no run name token, so changing them resumes the same run folder, `load_model()` setting the class weights of the new configuration again (`build_segmentation_sections()` of `run_name.py`; Step 2.18).
- A run trained on tiles is exported with one tile as input, the client scaling and tiling the images with `segment_image_by_tiles()` ([tools reference, Export](../reference/tools_reference.md#export)), while `run_inference.py` refuses such a run, as its requests would hold one letterboxed image instead of the many tiles a photo needs (`load_inference_inputs()` of `vision_training/deployment/deployment_utilities.py`). The export describes the tiles in the input contract of the model, version 2 (`build_input_contract()` of `vision_training/deployment/input_contract.py`, [tools reference, Input Contract](../reference/tools_reference.md#input-contract)): with `Data.tiling` at `activated: True`, `scale: 0.5` and `overlap: 128`, it writes `input_resize` `tiles`, `input_interpolation` `area`, `input_padding_placement` `bottom_right`, `input_tile_scale` `0.5` and `input_tile_overlap` `128`, against `keep_ratio`, `linear`, `centred` and `none` for this file.

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

To find your way in the stages below, the whole run fits in this tree: every function in the order it runs, with what it does.

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
│               ├── data:       classes 0..K-1 with the background at 0, categories that are mask values (1 to 254),
│               │               tiling scale > 0 and overlap smaller than a tile
│               ├── no sequence normalization of the backbone, since images have no frames (the frame shuffling
│               │               and independent frames settings are refused by check_removed_settings)
│               ├── no image-only augmentation type (random_perspective, random_jigsaw), which the masks cannot follow
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
│       │   Same order as for every task: device, run name, settings, augmenter, classes of a resumed run, paths,
│       │   loaders and their check, sample images, model, pretrained components, optimizer, metric tracker,
│       │   printouts, restoration.
│       │
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter()
│       │       The run name ends with the decoder (upsampling, normalization, D<layers>x<channels> of every level) and
│       │       the classes (MC<foreground classes>), after a TL<scale> section when the run trains on tiles.
│       ├── restore_run_label_dictionary() / setup_training_paths()
│       │       A resumed run keeps the classes of its label_dictionary.json, then the run folder is created with
│       │       the configuration and label_dictionary.json (classification_workflow.md, Step 2.6).
│       ├── build_dataloaders()
│       │   │   Builds the loaders of the three splits (the validation one only when the training reads it), keeps
│       │   │   class_names (background first) and class_colours (one BGR colour per class, for the overlays and the
│       │   │   sample images).
│       │   │
│       │   └── get_segmentation_dataloaders()
│       │       │   One SegmentationDataset per split from <folder>/labels.json, <folder>/data/ and <folder>/masks/.
│       │       │   With tiling, the training split cuts random tiles, the validation split fixed tiles, and the test
│       │       │   split keeps whole scaled images, one per batch.
│       │       │
│       │       ├── SegmentationDataset.__init__() -> build_class_lookup_table()
│       │       │       Reads the manifest (every sample has a file_name and a mask_file_name), shuffles once and
│       │       │       packs the samples (SerializedRecords), and builds the [256] table turning every mask value
│       │       │       into a class index (0 background, 255 ignored).
│       │       └── build_native_loader()
│       │               Same torch DataLoader as the classification.
│       ├── check_loaders_give_batches()          refuses a split whose loader gives no batch, and a validation
│       │                                         loader built against the validation setting
│       ├── save_data_record()                    data.json
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
│       │       supervision loss when there are heads, pixel accuracy and mean IoU, these two printed as
│       │       percentages in the console summary (get_percent_tracked_metrics).
│       ├── print_experiment_information() -> describe_split() -> count_split_samples()
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       │       The summary adds a decoder panel: the bottleneck, then for every level its skip connection,
│       │       convolutions and output, the head dropout and the size of the class logits.
│       ├── estimate_export_sizes() + print_export_size_summary()
│       │       The Export Sizes panel: sizes of the PyTorch, TorchScript, ONNX, TensorRT and checkpoint files.
│       ├── view_input_batches()  (only with view_input_images, then exits)
│       │       Each image with its mask painted over it, captioned with the classes it holds.
│       └── restore_last_model() -> get_last_checkpoint_iteration() / load_model()
│               load_model() is overridden to set the class weights of the current configuration again after loading,
│               since the loss settings may change between two resumptions.
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── setup_tensorboard_writers() / initialize_training_loop_state() / start_experiment_tracking()
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch()                    a batch failing to load is dropped and the next one taken
        │   │   └── SegmentationDataset.__getitem__()
        │   │           Reads the image (read_image()) and its mask, resizes the image (bilinear) and the mask
        │   │           (nearest-exact, padding at 255) with the same letterbox, or cuts a tile with tiling, then
        │   │           converts the mask values to class indices through the lookup table. Returns images
        │   │           [B, 3, H, W] uint8, masks [B, H, W] int64, sample_identifiers [B].
        │   ├── run_training_step()  set_training_mode() to step_optimizer() below, the step also timed by
        │   │       the speed benchmark (diagnosis_modes.benchmark_speed)
        │   ├── set_training_mode()
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()                 images to float32, masks to int64 on the device
        │   │   ├── augment_batch()                 the masks follow every geometric augmentation (nearest
        │   │   │                                   interpolation on the pixel centres, the pixels uncovered by a
        │   │   │                                   rotation ignored)
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
        │   ├── save_model() on the multiples of weight_saving_iterations and intermediate_weight_saving_iterations
        │   ├── validation pass every validation_iteration_ratio iterations
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary() + report_dropped_batches()
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations, last    -> [STAGE 4]
        ├── save_model() of the steps taken since the last checkpoint, if any (nothing evaluated after the loop)
        └── close the writers, shutdown_loader_workers(), finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
├── create_evaluation_accumulator()
│       A [K, K] pixel confusion matrix and [K, 6] boundary counts, with per_class_report [K, 2, 10000] score
│       histograms, all on the device, the lists of the per-image true positive and true pixels of every class,
│       the number of images, and the Predicted_Masks folder of the checkpoint.
├── for each test batch:
│   ├── prepare_batch()
│   ├── predict_batch()
│   │       SemanticSegmenter.forward() without masks: class_logits and class_probabilities. A run trained on tiles
│   │       segments every whole scaled image tile by tile (segment_image_by_tiles), averaging the overlaps.
│   ├── accumulate_evaluation_batch()
│   │       Adds compute_pixel_confusion_matrix() and compute_boundary_counts() of the batch (at the network
│   │       resolution, or with original_resolution against every original mask after
│   │       restore_prediction_to_original_image()), keeps compute_image_class_pixel_counts() of every image, with
│   │       per_class_report compute_pixel_score_histograms() of the same pixels, and with save_predicted_masks
│   │       writes one class index PNG per image.
│   └── save_sample_images() -> build_sample_image()
│           Four panels per fixed image: the input, the true mask and the predicted mask painted over it, the error
│           map, with the pixel accuracy of the image and a legend of its classes and of the error colours.
├── compute_evaluation_results()
│   ├── compute_segmentation_metrics()       pixel accuracy, IoU and Dice of every class, mean IoU, mean Dice,
│   │                                        frequency-weighted IoU, precision and recall
│   ├── compute_boundary_metrics()           boundary IoU and boundary F1 of every class, and their means
│   ├── compute_histogram_score_curves() + compute_macro_curve_metrics()   (only with per_class_report)
│   │                                        ROC AUC and average precision of the pixels of every class, their means
│   ├── compute_bootstrap_recall_intervals()
│   │                                        95 % interval of the recall of every class, 1000 resamples of the
│   │                                        test images
│   ├── build_segmentation_class_rows() + write_class_metrics_file()           class_metrics.csv
│   ├── save_confusion_matrix_table()                                          confusion_matrix.csv (pixel counts)
│   ├── save_confusion_matrix_image()   (only with per_class_report)          confusion_matrix.png (shares only)
│   ├── save_score_curve_images()   (only with per_class_report)    roc_curves.png, precision_recall_curves.png
│   └── build_segmentation_class_table() + build_recall_interval_table() + build_segmentation_class_scalars()
└── report_evaluation() -> report_evaluation_results() + track_evaluation()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task: loads training_configuration.yaml, builds the SegmentationTrainer writing in
    <run>/Checkpoint_Evaluation/, then
    evaluate_checkpoints() -> start_evaluation_tracking() + summarise_checkpoint_evaluations() -> run_test_evaluation()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `vision_training/configurations/semantic_segmentation/configuration_synthetic_shapes.yaml`.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1): `parse_configuration_path()` reads the path of the YAML file from the command line (`--configuration_path`), then it loads the file, looks up the trainer of `ExperimentConfiguration.task` and starts its training loop. Nothing in it is specific to the segmentation.

### Step 1.2 — `load_experiment_configuration()`
It splits the YAML into the two dictionaries read by every later stage, converts the values YAML reads with the wrong type, adds `label_mode` and `label_dictionary` to `Data` (Step 1.4) and runs the checks (Step 1.5). The segmentation has two sections of its own, `SegmentationDecoder` (the upsampling path of the U-Net) and `SegmentationLoss`, its `Data` section has a `tiling` subsection, and its `evaluation` settings differ from the classification ones (no target precision of class thresholds, no error images, but the predicted masks, the original resolution and the boundary tolerance).

**Example: the sections and the evaluation settings**
```text
model_configuration keys: Optimization, Data, Backbone, SegmentationDecoder, SegmentationLoss
Data keys after loading:  image_settings, tiling, label_mapping, data_augmentation, label_mode, label_dictionary
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'save_predicted_masks': False,
                           'original_resolution': False, 'boundary_tolerance': 2}
SegmentationDecoder:      {'upsampling': 'transposed', 'convolutions': {'5': [2, 64], '4': [2, 64], '3': [2, 32], '2': [2, 32],
                           '1': [1, 16]}, 'modules': {'squeeze_and_excite': False, 'residual_nets': False},
                           'feature_map_normalization': 'batch', 'dropout_rate': 0.0,
                           'deep_supervision_weights': {}, 'pretrained': {...}}
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
The common checks run first, then `check_segmentation_configuration()`: the data settings (background at 0, categories that are mask values, tiling), no sequence settings, no augmentation type moving the image only (`random_perspective` and `random_jigsaw`, which the masks cannot follow, `check_image_only_augmentation_types()`), the decoder (one level per skip connection of the backbone), the losses and the evaluation settings. Every problem is reported together, then the program exits.

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

> The examples below are real outputs from building the trainer on `vision_training/configurations/semantic_segmentation/configuration_synthetic_shapes.yaml` (512 pixel images, batch size 16), with the run folder redirected to a temporary folder. The whole construction took 0.3 s on CPU.

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
self.batch_size = 16   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The input and backbone tokens are those of every image task. `build_segmentation_sections()` adds the decoder: `UTC` (transposed convolutions, `UBL` for bilinear), the modules (`DSE`, `DRES`), the normalization (`DBN`), then `D<layers>x<channels>` of every level from the deepest one to the input resolution, `AUX<levels>` for deep supervision heads and the transfer tokens. Then come the classes: `MC9` counts the foreground classes, the background left out (`BIN` in binary mode). A run trained on tiles gets a `TL<scale>` section between the backbone and the decoder (`..._B1x16.2x32.2x32.2x64.2x64.2x128_TL0.5_UTC-DBN-...` with `scale: 0.5`). The loss settings, the head dropout and the evaluation settings give no token, so they can change between two resumptions of a run.

**Example output**
```text
SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9
```

### Step 2.5 — `DataAugmenter()`
The same augmenter as the classification. Here the masks follow the images (Step 3.11).

**Example: the settings of the configuration**
```text
activated True   probability 0.4
6 of 28 types activated: random_crop, random_left_right_flip, random_rotation, brightness_contrast_saturation,
                         random_noise, random_haze
```

### Step 2.6 — `setup_training_paths()`
As for every task (see `classification_workflow.md`, Step 2.6): a resumed run first keeps the classes of its `label_dictionary.json` (`restore_run_label_dictionary()`), then the run folder is created with the copy of the configuration and `label_dictionary.json`, the class index of every category, background included.

**Example: the run folder after this step**
```text
SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9/
├── Tensorboard/
├── Weights/
├── label_dictionary.json          the class index of every category of the run
└── training_configuration.yaml    the copy of the launched file, its comments included
```
With `class_weighting: "balanced"`, the run folder also receives `class_pixel_counts.json` at Step 2.18.

### Step 2.7 — `build_dataloaders()`
This hook builds the loaders of the three splits (Step 2.8), the validation one only when the run reads it (`validation_iteration_ratio > 0` or a diagnosis mode ending the program, `classification_workflow.md`, Step 2.7), keeps `class_names` (background first, then the categories of the training manifest) and `class_colours`, one BGR colour per class (black for the background), used to paint the masks in the input viewer and the sample images.

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
It reads the manifest of the split: every sample gives its image file and its mask file, a single-channel `uint8` PNG of the size of its image whose every pixel holds the category of the image pixel (0 background, 255 unlabelled). The `category_ids` the generator now writes in every sample (the categories visible in its mask) are not read by the dataset, only by the record of the data (`data.json`). It shuffles the samples once, packed like the classification samples (`SerializedRecords`, Step 2.9 of `classification_workflow.md`), so that the loader workers never copy them, every sample being found by its identifier (`samples.find_record()`) for its original mask in the evaluation and for the evaluation outputs. `build_class_lookup_table()` prepares a table of 256 class indices, one per possible mask value, so that a whole mask is converted at once with `table[mask]`: 0 stays the background, 255 becomes the ignored index (-100), and every category becomes the class of the label dictionary (an unmapped category becomes background). No image is read here.

**Example: a manifest sample, and the first sample after shuffling**
```text
labels.json:  {'id': 1, 'file_name': 'train_000000.png', 'mask_file_name': 'train_000000.png'}     (2000 samples, 9 categories)
categories:   {'id': 1, 'name': 'square'}, ...
kept:         {'identifier': 1351, 'file_name': 'train_001350.png', 'mask_file_name': 'train_001350.png'}
dataset[0]:   images [3, 512, 512] torch.uint8, masks [512, 512] torch.int64 holding the classes [0, 9], sample_identifiers 1351
```

### Step 2.10 — `build_native_loader()`
The same PyTorch `DataLoader` as for the classification (see `classification_workflow.md`, Step 2.10).

**Example output** (the training loader; the validation and test loaders have the same settings, 13 batches each; recorded with pinned memory, which the loaders now use only on a discrete GPU)
```text
len(loader) = 125   batch_size = 16   num_workers = 1   prefetch_factor = 2   pin_memory = True   persistent_workers = True
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
The backbone shared by every task (see `classification_workflow.md`, Step 2.14). For the segmentation, its forward pass also returns the output of every block before its pooling (Step 3.13). The panel lists the residual shortcuts of a block only when it adds one (`ConvolutionBlock.count_residual_additions()`): blocks 3 and 5 keep their channel count and add one, while blocks 2, 4 and 6 change it (16 → 32, 32 → 64, 64 → 128), which shifts their residual pairs by one layer, so with 2 layers they add none; block 1 never has shortcuts.

**Example: the backbone panel of `print_summary()` for a 512×512 image** (the model of the configuration rebuilt on CPU with the current code)
```text
------------------------------
Backbone Architecture Summary
------------------------------
------------------------------------
Input image shape: [1, 3, 512, 512]
------------------------------------
╭─────────────────────────── Backbone Architecture ────────────────────────────╮
│ Normalization Strategy: Batch                                                │
│                                                                              │
│  - Convolution Block 1                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 3, 512, 512]                                         │
│    - Output Shape : [1, 16, 256, 256]                                        │
│                                                                              │
│  - Convolution Block 2                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 16, 256, 256]                                        │
│    - Output Shape : [1, 32, 128, 128]                                        │
│                                                                              │
│  - Convolution Block 3                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 32, 128, 128]                                        │
│    - Output Shape : [1, 32, 64, 64]                                          │
│                                                                              │
│  - Convolution Block 4                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 32, 64, 64]                                          │
│    - Output Shape : [1, 64, 32, 32]                                          │
│                                                                              │
│  - Convolution Block 5                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 64, 32, 32]                                          │
│    - Output Shape : [1, 64, 16, 16]                                          │
│                                                                              │
│  - Convolution Block 6                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 64, 16, 16]                                          │
│    - Output Shape : [1, 128, 16, 16]                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
----------------------------------------------
Final Backbone output shape: [1, 128, 16, 16]
----------------------------------------------
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
The segmentation can transfer the backbone and the decoder of other runs. The segmentation head and the deep supervision heads stay outside the decoder, so that a decoder serves runs of other classes; with `SegmentationDecoder.pretrained.load_class_layers` they are loaded too (`load_segmentation_class_layers()`), the classes of the source run being the first ones of this run, the new classes keeping the random rows the layers of this run were built with, and the deep supervision levels the source run does not have keeping their random weights. A backbone deeper than its source backbone loads its first blocks only, as for every task (`loaded_backbone_blocks`, see `classification_workflow.md`, Step 2.19).

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
The tracked losses are those whose weight is not 0, so the console, TensorBoard and MLflow only show what is trained, then the pixel accuracy and the mean IoU of every batch (Step 3.16). `get_percent_tracked_metrics()` gives the keys the console summary prints as percentages, `['pixel_accuracy', 'mean_iou']`, with their changes in percentage points written `+0.52 %`, the losses keeping 4 decimals; TensorBoard and MLflow keep them between 0 and 1 (Step 3.20).

**Example output** (cross entropy and Dice on, Lovász and boundary off)
```text
{'total_loss': 'Total Loss', 'segmentation_loss': 'Cross Entropy Loss', 'dice_loss': 'Dice Loss',
 'pixel_accuracy': 'Pixel Accuracy', 'mean_iou': 'Mean IoU'}
```

### Step 2.22 — `print_experiment_information()` (→ `describe_split()` → `count_split_samples()`)
Every path is a terminal link showing the name of its file or folder (`format_path_link()` of `vision_training/utilities/os_utilities.py`).

**Example output**
```text
--------------------------------------
=== Experiment Launch Information ===
--------------------------------------
- Run Name                : SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9
- Experiment Folder       : SEG_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9
- Tensorboard Directory   : Tensorboard
- Weights Directory       : Weights
- Configuration File      : training_configuration.yaml
- Data Record             : data.json
- Train Data Folder       : train (2000 images, 9 categories)
- Validation Data Folder  : validation (200 images, 9 categories)
- Test Data Folder        : test (200 images, 9 categories)
- Iterations Per Epoch    : 125
- Batch Size              : 16
- Total Iterations        : 10000000000
- Save And Evaluate Every : 1000 iterations
- Save Only Every         : 500 iterations
- Learning Rate           : 0.001
- Optimizer               : adam (weight decay 0.0)
- Learning Rate Schedule  : constant (warm-up 0 iterations)
- Weight Moving Average   : Off
- Validation Iter. Ratio  : 25
- MLflow Tracking         : Off
```

### Step 2.23 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The backbone and decoder panels are shown in Steps 2.14 and 2.17. The size panel lists the decoder and the head on their own lines. The trainer then prints the Export Sizes panel: the parameters the forward pass of the inference model reads, then the size of the PyTorch weights, the TorchScript file, the ONNX files and the TensorRT engines the model is turned into and of the training checkpoint ([Export Sizes](../../README.md#export-sizes)).

**Example output: the size panel**
```text
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 438,368                       │
│   - Trainable Parameters : 438,368                       │
│   - Estimated Size       : 1.76 MB                       │
│                                                          │
│ Segmentation Decoder                                     │
│   - Parameters           : 345,984                       │
│   - Trainable Parameters : 345,984                       │
│   - Estimated Size       : 1.39 MB                       │
│                                                          │
│ Segmentation Head                                        │
│   - Parameters           : 170                           │
│   - Trainable Parameters : 170                           │
│   - Estimated Size       : 0.00 MB                       │
│                                                          │
│ Total Model                                              │
│   - Parameters           : 784,522                       │
│   - Trainable Parameters : 784,522                       │
│   - Estimated Size       : 3.15 MB                       │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.24 — `view_input_batches()`
With `diagnosis_modes.view_input_images`, the training image (augmented, with its augmented mask) and the validation image are shown with their masks painted over them in the class colours, each followed by an information panel giving the classes it holds (`Classes : square, star`, or `background only`) and, for the training image, the augmentations drawn with their values. It needs a display, so it has no console output to show here.

### Step 2.25 — `restore_last_model()` → `get_last_checkpoint_iteration()` / `load_model()`
The same restoration as every task (see `classification_workflow.md`, Steps 2.26 to 2.28), except that `load_model()` sets the class weights of the current configuration again after loading the checkpoint.

**Example output: a new run** (`start_iteration = 1`)
```text
-----------------------------------------------------------
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.
-----------------------------------------------------------
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of `configuration_synthetic_shapes.yaml` shortened to 64 pixel images, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the predicted masks saved and the MLflow tracking on. Its run name is `SEG_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_UTC-DBN-D2x64.2x64.2x32.2x32.1x16_MC9`.

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
Saved right after the training step of iterations 3, 6, 9 and 12, so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6). The checkpoint stores the label dictionary of the run, the source runs of the two transferable components of the segmentation and the number of backbone blocks loaded from a shorter source backbone (`pretrained_backbone_blocks`, `None` otherwise).

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   9,588,813 bytes      Weights/Iteration_6/model_000006.pt   9,588,813 bytes
Weights/Iteration_9/model_000009.pt   9,588,813 bytes      Weights/Iteration_12/model_000012.pt  9,588,813 bytes
Weights/full-checkpoint               model_checkpoint_path: "Iteration_12", all_model_checkpoint_paths 3, 6, 9, 12
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_segmentation_decoder']
model_state: 175 tensors, the first 'class_weights', then 'backbone.convolution_blocks.0. ...'
```
This run was saved before the checkpoints held `pretrained_backbone_blocks`: a checkpoint of the current code also has that key after `pretrained_segmentation_decoder`, `None` for this run.

### Step 3.6 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration, after its checkpoint, and at iteration 12 after its summary too, the summaries coming every 4 iterations (see `classification_workflow.md`, Step 3.7), see Stage 4.

### Step 3.7 — `get_next_batch()` → `SegmentationDataset.__getitem__()`
`get_next_batch()` drops a batch whose loading fails and takes the next one (see `classification_workflow.md`, Step 3.8). `__getitem__()` reads the image with `read_image()` (BGR, `uint8`, a `ValueError` naming a file that cannot be read, a file of more than 8 bits per channel read with 8 bits after a yellow warning) and its mask (a mask that is not a single-channel `uint8` image is refused with a `ValueError`), and checks that they have the same size. Without tiling, `preprocess_image()` resizes the image with the bilinear interpolation and the mask with the nearest interpolation on the pixel centres (`cv2.INTER_NEAREST_EXACT`, so that every pixel keeps a category and stays on its image pixel, where the plain `cv2.INTER_NEAREST` shifts the mask by about half a pixel to the bottom right), with the same letterbox: the padding of the image is black and the padding of the mask is 255, the unlabelled value. With tiling, `cut_tile()` scales both, the mask with the same interpolation, and cuts the tile of the split. The lookup table of Step 2.9 then turns the mask values into class indices.

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
The segmentation trainer overrides this hook so that the masks are augmented with their images: every geometric augmentation (crop, flips, rotations) is applied to the mask with the nearest interpolation on the pixel centres (the crops and the zoom-out resize it with `InterpolationMode.NEAREST_EXACT`, the resized crop of torchvision taking the plain nearest for masks), and the pixels uncovered by a rotation become ignored; the photometric augmentations leave the mask untouched. A mask that did not follow its image would teach the network wrong pixels.

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
From the deepest level to the finest, each level upsamples its input to twice its resolution, concatenates the skip connection of its block before the upsampled features (the upsampled map being resized bilinearly to the skip connection when they differ by a pixel, which the configuration check, every map entering a pooling being even, leaves for the inputs that bypass it), and refines the result with its convolution block. The head turns the `[B, 16, 64, 64]` output into the `[B, 10, 64, 64]` class logits.

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
The predicted class of every pixel is the argmax of its logits. `compute_pixel_confusion_matrix()` counts the labelled pixels of every pair (true class, predicted class) with one `bincount`, and `compute_segmentation_metrics()` gives the pixel accuracy (share of the labelled pixels given their class) and the mean IoU of the classes present in the masks of the batch or predicted in it. No second forward pass is run.

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
The optimizer steps with the gradients as the backward pass left them, never clipped, at the learning rate of the schedule (see `classification_workflow.md`, Step 3.22).

**Example: the gradient norm of the first steps, measured for this document, and the learning rate**
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

> These outputs were recorded before 2026-10-25, when the mean IoU still left out the classes predicted but absent: a batch whose masks lack a class the network paints now gives a lower batch mean IoU, so the `Mean IoU` lines can be lower in a new run.

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
Moving Average of Train Total Loss                     :     2.8497    -0.1213 Best
Moving Average of Train Cross Entropy Loss             :     1.9076    -0.1137 Best
Moving Average of Train Dice Loss                      :     0.9421    -0.0076 Best
Moving Average of Train Pixel Accuracy                 :    48.50 %    +7.46 % Best
Moving Average of Train Mean IoU                       :     7.81 %    +1.20 % Best
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:10.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
========================================================================================================================
...
========================================================================================================================
|    Iteration: 12     |      Total Images: 48     |    Interval Images: 16     |  Throughput: 8.1 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     2.7779    -0.0718 Best
Moving Average of Valid Total Loss                     :     2.9886
Moving Average of Train Cross Entropy Loss             :     1.8413    -0.0663 Best
Moving Average of Valid Cross Entropy Loss             :     2.0299
Moving Average of Train Dice Loss                      :     0.9366    -0.0055 Best
Moving Average of Valid Dice Loss                      :     0.9587
Moving Average of Train Pixel Accuracy                 :    45.67 %    -2.84 %           <- red: lower than at iteration 8
Moving Average of Valid Pixel Accuracy                 :    91.36 %
Moving Average of Train Mean IoU                       :     8.29 %    +0.48 % Best
Moving Average of Valid Mean IoU                       :    11.26 %
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:01.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```

### Step 3.21 — end of the loop
The checkpoint of iteration 12 is already saved and evaluated inside its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 200 Test Images · 2026-10-09 10:04 ───────────────      <- the evaluation of iteration 12, last in the iteration
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (200 images of 64×64 pixels), with `per_class_report: True`, `number_sample_images: 8` and `save_predicted_masks: True`. The outputs of iteration 6 were captured when its checkpoint was still saved before its training step, after 5 steps; it now holds 6 steps. They were also captured before the per-class recall intervals were added (2026-11-01): the accumulator now also holds the per-image pixel counts of Step 4.5, `class_metrics.csv` the `recall_lower` and `recall_upper` columns after `recall`, and with `per_class_report` the report a `Per Class Recall` table after `Per Class Metrics` (Steps 4.10 and 4.13).

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1). The `evaluation_context()` of the segmentation changes nothing.

**Example output: the progress bar** (200 images, batch size 4)
```text
Test Evaluation Iteration 6: 100%|██████████| 50/50
```

### Step 4.2 — `create_evaluation_accumulator()`
Storing the probabilities of every pixel of the test split would take a lot of memory, so the segmentation accumulates counts instead, on the device: the `[K, K]` pixel confusion matrix and the `[K, 6]` boundary counts of every class, and with `per_class_report` the `[K, 2, 10000]` `score_histograms` of the pixel probabilities (Step 4.5), the lists of the `[B, K]` true positive and true pixels of every image and class (`image_true_positive_pixels`, `image_true_pixels`), next to the number of evaluated images and the `Predicted_Masks` folder of the checkpoint.

**Example: the accumulator at the end of the loop**
```text
{'confusion_matrix': [10, 10] torch.int64, 'boundary_counts': [10, 6] torch.int64, 'number_images': 200,
 'predicted_masks_folder': PosixPath('.../Evaluation/Iteration_6/Predicted_Masks')}
```
This run was captured before the ROC and precision-recall curves of the per-class report: with `per_class_report: True`, the current accumulator also holds `'score_histograms': [10, 2, 10000] torch.int64`.

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
The predicted class map of every kept image (the argmax of its probabilities) is compared with its mask: at the network resolution (the padding and the unlabelled pixels being ignored), or with `original_resolution: True` with the original mask of every image, its probabilities being first cut out of the letterbox and resized back to the original size (`restore_prediction_to_original_image()`). The pixel confusion matrix and the boundary counts of the batch are added to the accumulator, and `compute_image_class_pixel_counts()` keeps, for every image and class, its labelled pixels and those predicted as the class, for the Korn-Graubard interval of the recall (Step 4.10). `compute_boundary_counts()` counts, for every class, the intersection and the union of the bands within `boundary_tolerance` pixels of the true and predicted contours, the predicted edge pixels with those within the tolerance of a true edge, and the true edge pixels with those within the tolerance of a predicted edge. With `per_class_report`, `compute_pixel_score_histograms()` counts the same labelled pixels, class by class, in a histogram of 10,000 bins (`SCORE_HISTOGRAM_BINS`) of their probability of the class, the pixels of the class (positives) apart from the others (negatives), added to the `[K, 2, 10000]` `score_histograms` of the accumulator on the device: the probabilities of all the test pixels are far too many to keep for the ROC and precision-recall curves. With `save_predicted_masks`, the class map of every image is written as a single-channel PNG named after the image.

**Example: the predicted masks of the checkpoint of iteration 6**
```text
Evaluation/Iteration_6/Predicted_Masks/    200 files: test_000000.png, test_000001.png, ...   (pixel value = predicted class index)
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
For every fixed sample image found in the batch, `build_sample_image()` lays out four titled panels side by side: the input, the true mask painted over it, the predicted mask painted over it, and the error map. A caption bar above gives the pixel accuracy of the image over its labelled pixels as a percentage, and a legend bar below names the classes of its true and predicted masks (background left out), then the colours of the error map: `Correct` (green, a pixel of a class given its class), `Wrong Class` (red, another class, or a class where the truth is the background) and `Missed` (yellow, a pixel of a class predicted as background), the ignored pixels being gray and the background predicted as background staying dark (`build_segmentation_error_map()` of `vision_training/utilities/visualization.py`). The panels of a large image segmented tile by tile are reduced to 1024 pixels on their longest side. The image is saved in `Samples/` and sent to TensorBoard under `Samples/<image name>`.

**Example: `Evaluation/Iteration_12/Samples/test_000028.png`** (268 x 112 pixels)
```text
caption: "Pixel Accuracy 90.06 %"
panels:  Input | Ground Truth | Prediction | Errors      (64 x 64 each; the titles are cut at this small size: "Ground Tru")
legend:  square (red), circle (orange), rectangle (green), ...
```
The legend then names the three error colours, written by `build_segmentation_sample_grid()` after the classes of every sample.

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
The 819,200 labelled pixels are the 200 test images × 64 × 64: the synthetic images are square, so there is no padding. This run was captured before the ROC and precision-recall curves of the per-class report: with `per_class_report: True`, the current code also gives `'Macro ROC AUC'` and `'Macro Average Precision'` in `metrics`, the files `ROC Curves Image` and `Precision Recall Curves Image` after `Confusion Matrix Image`, and the `ROC AUC/<class>` and `Average Precision/<class>` scalars of the classes defining them (Step 4.13).

### Step 4.8 — `compute_segmentation_metrics()`
All the region metrics come from the pixel confusion matrix (rows true classes, columns predicted classes). For a class: its **IoU** is the right pixels over the union of its true and predicted pixels, its **Dice** is the F1 of its pixels (twice the intersection over their sum), its **precision** and **recall** are those of the classification. The **pixel accuracy** is the diagonal over all the labelled pixels, the **mean IoU** and **mean Dice** are the means over the classes present in the test masks or predicted (a class painted without any true pixel counts with 0, a class neither present nor predicted is left out), and the **frequency-weighted IoU** weights the IoU of every class by its share of the pixels, which the background dominates.

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
The pixels of an image are correlated (a missed object loses all its pixels together), so an exact interval over the pixels would be far too narrow. Before the rows are built, the per-image counts of Step 4.5 are checked against the rows of the confusion matrix, then `compute_bootstrap_recall_intervals()` of `vision_training/evaluation/recall_intervals.py` computes the Korn-Graubard interval of every class: 1000 resamples of the test images with replacement (fixed seed), every image kept whole, give the variance of its pixel recall (a resample without pixels of the class not counting for it), the effective sample size `p (1 - p) / variance`, capped at its pixels, is reduced for the degrees of freedom of the images holding the class, and the bounds are the exact (Clopper-Pearson) interval of the recall at that size. When the recall of the class is 0 or 1 (no image finding any of it, or every image finding all of it), or a single image holds it, the bounds are the exact interval over those images, each counted as one sample, instead of an interval of zero width. The bounds join the row as `recall_lower` and `recall_upper` after `recall`, empty for a class without true pixels.

One row per class, in class index order, with its true pixels and all its metrics, written as `class_metrics.csv` at every evaluation, whatever `per_class_report`. With `per_class_report`, `compute_histogram_score_curves()` first turns the histograms of Step 4.5 into the ROC and precision-recall curves of every class, one against the rest, the lower edge of every bin holding pixels being a threshold (the curves are exact at the bin edges, the pixels of a bin moving them together like tied scores): the ROC AUC and average precision of every class join its row as `roc_auc` and `average_precision`, and their means over the classes defining them (`compute_macro_curve_metrics()`) follow the headline metrics as `Macro ROC AUC` and `Macro Average Precision`. A class without pixels has neither, a class covering every labelled pixel no ROC AUC. The example below comes from a run made before the curves, without these two columns.

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
The table is the pixel confusion matrix of Step 4.8 (`confusion_matrix.csv`, written whatever `per_class_report`). With `per_class_report`, the image is drawn like the one of the classification, titled `Pixel Confusion Matrix · Iteration N · Mean IoU ... · Pixel Accuracy ...` with both rates as percentages (`Mean IoU 7.90 %`), its cells showing only the share of every true class as a percentage (`74.64 %`, `in_percent=True`), since the pixel counts are too large to print (`show_counts=False`). `save_score_curve_images()` then draws the curves of Step 4.10 as for the classification (`classification_workflow.md`, Step 4.15), titled `ROC Curves · Pixels · Iteration N` and `Precision-Recall Curves · Pixels · Iteration N`, the macro mean (`Macro ROC AUC ...`, `Macro Average Precision ...`) on the second line of the title so that it stays within the image, the rates of the titles and legends as percentages and the axes from `0 %` to `100 %` (`in_percent=True`), without thresholds, and saves `roc_curves.png` and `precision_recall_curves.png`, also logged to TensorBoard and MLflow.

### Step 4.12 — the notes and the predicted masks
The notes name what was scored: the number of labelled pixels and the resolution (`network resolution`, `original resolution of the images`, or `images scaled by <scale>, segmented tile by tile`), and the boundary tolerance. With `save_predicted_masks`, the `Predicted_Masks` folder and its number of images are added.

### Step 4.13 — `build_segmentation_class_table()` + `build_segmentation_class_scalars()`
The per-class table sorts the classes by IoU, worst first. `build_recall_interval_table()` adds the `Per Class Recall` table after it: the true pixels, the recall and the bounds of its interval (`95 % Lower`, `95 % Upper`, both headers on two lines) of every class, sorted by recall, worst first. Both tables write their rates as percentages (`percent_columns`), the per-class table under the headers of `CLASS_COLUMN_HEADERS` of `segmentation_report.py`, `Boundary IoU`, `Boundary F1` and `Average Precision` on two lines so that its ten columns fit in the report, and the results list every headline metric in `percent_metrics`, the best checkpoint line of Stage 5 following them. The per-class scalars are the 6 metrics of every class with pixels (`IoU/<class>`, `Dice/<class>`, `Boundary IoU/<class>`, `Boundary F1/<class>`, `Precision/<class>`, `Recall/<class>`), 60 scalars for 10 classes, written only with `per_class_report`, which also adds the `ROC AUC` and `Average Precision` columns and scalars of the classes defining them: 80 per-class scalars for 10 classes, next to the 8 headline metrics, in the 512 pixel run of `experiments/readme_screenshots`.

### Step 4.14 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 200 Test Images · 2026-10-09 10:04 ──────────────────────────────────────────────────────────────────────
  Mean IoU            7.90 %    Mean Dice                9.39 %
  Mean Boundary IoU   1.95 %    Mean Boundary F1         6.08 %
  Pixel Accuracy      72.22 %   Frequency Weighted IoU   67.48 %
  Labelled Pixels : 819200 at the network resolution, the padding and the unlabelled pixels being ignored
  Boundary Tolerance : 2 pixels
  Predicted Masks : 200 class index images
  Sample Images : 8 fixed test images

  Per Class Metrics
                                             Boundary   Boundary
   Class        Pixels       IoU      Dice        IoU         F1   Precision    Recall
  ─────────────────────────────────────────────────────────────────────────────────────
   circle        10212    0.00 %    0.00 %     0.00 %     0.00 %      0.00 %    0.00 %
   triangle       2221    0.00 %    0.00 %     0.00 %     0.00 %      0.00 %    0.00 %
   rectangle      6571    0.00 %    0.00 %     0.00 %     0.00 %      0.00 %    0.00 %
   pentagon       8031    0.00 %    0.00 %     0.00 %     0.00 %      0.00 %    0.00 %
   hexagon        9485    0.00 %    0.00 %     0.00 %     0.00 %      0.00 %    0.00 %
   plus_sign      9263    0.00 %    0.00 %     0.00 %     0.00 %      0.00 %    0.00 %
   octagon       12074    0.00 %    0.00 %     0.00 %     0.00 %      0.00 %    0.00 %
   star           5654    0.86 %    1.70 %     0.75 %     4.18 %      0.89 %   16.66 %
   square        15883    3.50 %    6.76 %     1.73 %     7.30 %      4.06 %   20.20 %
   background   739806   74.64 %   85.48 %    17.02 %    49.31 %     92.55 %   79.41 %
  Worst IoU first, ties in class order. Boundary IoU and F1 : contours within the boundary tolerance. — : no pixel (or
  no edge for the boundary metrics) of the class in the test masks.
  - Results : Results.txt
  - Class Metrics : class_metrics.csv
  - Confusion Matrix Table : confusion_matrix.csv
  - Confusion Matrix Image : confusion_matrix.png
  - Predicted Masks : Predicted_Masks
  - Sample Images : Samples
```
Every file name is a terminal link to the file in the evaluation folder of the checkpoint, `Results.txt` being in the run folder. This block was captured before the ROC and precision-recall curves of the per-class report. With `per_class_report: True` the current code adds a line `Macro ROC AUC ... Macro Average Precision ...` under the frequency-weighted IoU, the `ROC AUC` and `Average Precision` columns of the per-class table, `, or no pixel of another class for the ROC AUC.` at the end of its legend, and the lines `ROC Curves Image : roc_curves.png` and `Precision Recall Curves Image : precision_recall_curves.png` after the confusion matrix image, as in the `Results.txt` and `semantic_segmentation.log` of the 512 pixel run of `experiments/readme_screenshots`.

**Example: the run folder, TensorBoard and MLflow after the two evaluations**
```text
Evaluation/Iteration_6/ and Iteration_12/   Predicted_Masks/ (200)  Samples/ (8)  class_metrics.csv  confusion_matrix.csv  confusion_matrix.png
Tensorboard/Test: 66 scalars (6 global + 60 per class), 9 images (8 samples + Confusion Matrix)
    scalar Mean IoU          6: 0.0024, 12: 0.0790
    scalar Pixel Accuracy    6: 0.0196, 12: 0.7222
MLflow: test/Mean IoU, test/Mean Dice, ..., test/best_Mean IoU (steps 6, 12), 60 test_class/<metric>/<class>,
        artifacts evaluation/iteration_6/ and iteration_12/: class_metrics.csv, confusion_matrix.png
```
The checkpoint of iteration 6 predicted `square` almost everywhere (pixel accuracy 1.96 %, square recall 99.66 %); the one of iteration 12 learned that most pixels are background. With the curves, every evaluation folder also holds `roc_curves.png` and `precision_recall_curves.png`, and `Tensorboard/Test` of the 512 pixel run of `experiments/readme_screenshots` holds 88 scalars (8 headline metrics + 80 per class) and 11 images (8 samples, `Confusion Matrix`, `ROC Curves`, `Precision Recall Curves`).


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12, captured when those of iterations 3, 6 and 9 were still saved before their training step) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 40 images of the test split (its `labels.json` reduced, its `data/` and `masks/` folders linked). They were also captured before the per-class recall intervals (2026-11-01, Stage 4).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5): set `run_name` and `experiment_folder` to the segmentation run, `test_data_folder` to another labelled split (holding `labels.json`, `data/` and `masks/`) if needed. The segmentation configurations have no `evaluation.error_dumps` section, so `maximum_error_images_per_class_pair` is skipped with a yellow warning: the segmentation evaluation saves no error image. `save_predictions`, `target_precisions`, `nms_criterion` and `nms_distance_exponent`, None in the main block, would be skipped the same way when set, the task having no predictions file (`evaluation.save_predictions`), no class thresholds (`evaluation.target_precision`) and no `Detector` section. The evaluation is written in the `Checkpoint_Evaluation/` folder of the run, next to the `Evaluation/` folder of the training (`Checkpoint_Evaluation_split_<parent folder>_<split folder>/` for another labelled split).

### Step 5.2 — trainer construction
The script loads `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it, points it to the `Checkpoint_Evaluation/` folder of the run, the reduced split and the batch size, and builds a `SegmentationTrainer`. With `class_weighting: "balanced"` it would count the pixels of the training masks again in the `Checkpoint_Evaluation/` folder, since the counts are cached in the folder of the trainer.

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
**Example output** (the file name is a terminal link to `runs/<run>/Checkpoint_Evaluation/evaluation_summary.csv`)
```text
-------------------------------------------------------------
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv
-------------------------------------------------------------
  - Best Checkpoint : iteration 12 (Mean IoU 7.13 %)

iteration,Mean IoU,Mean Dice,Mean Boundary IoU,Mean Boundary F1,Pixel Accuracy,Frequency Weighted IoU,IoU/background, ...
12,0.07127240554791028,0.08981075670647244,0.015183896521748261,0.05434968979463565,0.642840576171875,0.5973665654951408, ...
9,0.054483986969393404,0.0767030110689235,0.017650408620012898,0.05856697293708574,0.473822021484375,0.4409209823322503, ...
6,0.003934161162990151,0.007616101260174676,0.002960294035671728,0.02332027384646323,0.03636474609375,0.0016963343355225918, ...
3,0.003597412109375,0.006944984505166907,0.0,0.0,0.03597412109375,0.0012941373884677886, ...
```
These evaluations were run before the curves of the per-class report: the current summary also has the `Macro ROC AUC` and `Macro Average Precision` columns after `Frequency Weighted IoU`, and the `ROC AUC/<class>` and `Average Precision/<class>` columns of the classes defining them.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 12 on the 40 images** (shortened: the classes absent from these images are left out)
```text
-----------------------------
Evaluating : model_000012.pt
-----------------------------
...
Iteration 12 · 40 Test Images · 2026-10-09 10:04 ───────────────────────────────────────────────────────────────────────
  Mean IoU            7.13 %    Mean Dice                8.98 %
  Mean Boundary IoU   1.52 %    Mean Boundary F1         5.43 %
  Pixel Accuracy      64.28 %   Frequency Weighted IoU   59.74 %
  Labelled Pixels : 163840 at the network resolution, the padding and the unlabelled pixels being ignored
  Boundary Tolerance : 2 pixels
  Predicted Masks : 40 class index images
  Sample Images : 8 fixed test images
  ...
   star            569    0.49 %    0.97 %     0.54 %     3.60 %      0.49 %   26.71 %
   square         5894    4.96 %    9.45 %     2.03 %     9.93 %      6.17 %   20.19 %
   background   148235   65.83 %   79.39 %    12.61 %    40.82 %     91.44 %   70.15 %
```
The sample images were chosen from the 40 images of this split, so they are not the same images as in Stage 4.
