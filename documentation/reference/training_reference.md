# Training Reference

Details of the training runs, completing the [Training](../../README.md#training) and [Transfer Between Runs](../../README.md#transfer-between-runs) sections of the README.

## Table Of Content
- [Launching A Run](#launching-a-run)
- [Run Folder](#run-folder)
- [Console Output](#console-output)
- [Stopping And Resuming](#stopping-and-resuming)
- [Data Augmentation](#data-augmentation)
- [Viewing The Input Images](#viewing-the-input-images)
- [Checking The Anchor Coverage](#checking-the-anchor-coverage)
- [Benchmarking The Speed](#benchmarking-the-speed)
- [TensorBoard Monitoring](#tensorboard-monitoring)
- [Moving Average Of The Weights](#moving-average-of-the-weights)
- [Transfer Between Runs](#transfer-between-runs)
  - [Adding Classes](#adding-classes)

## Launching A Run
- A `--configuration_path` leading to no file, or a configuration with problems, stops the program before anything starts, with the list of every problem. A setting removed from the configuration files is refused with a message naming what replaced it ([Fixed Settings](configuration_reference.md#fixed-settings)).
- The training, validation and test loaders each read their batches with 1 worker process, preparing 2 batches in advance, the validation loader being built only when `validation_iteration_ratio` is above 0 (or for the diagnosis modes ending the program); the batches are copied to pinned memory only with a discrete CUDA GPU (`uses_pinned_memory` of `vision_training/data/loaders.py`), an integrated GPU such as that of the DGX Spark sharing one memory with the processor, and the training and validation forward passes always run in bfloat16 (mixed precision), the weights and the test evaluation staying in float32.
- The launch adds `expandable_segments:True` to `PYTORCH_CUDA_ALLOC_CONF`, keeping its other settings. Without it, the changing shapes of the sequence batches make the allocator reserve more and more GPU memory.

## Run Folder

```text
<experiment_folder>/<run name>/
├── training_configuration.yaml        # Copy of the configuration of the last start of the run
├── training_configuration_until_iteration_<n>.yaml  # Configuration that trained the run up to iteration n, kept by a resume that changed it
├── data.json                          # Classes counted in every split, and data folder, SHA-256 of labels.json and images of every split
├── data_until_iteration_<n>.json      # Data that trained the run up to iteration n, kept by a resume whose data changed
├── label_dictionary.json              # Class index of every category of the label mapping, the classes of the run
├── anchor_recall.csv                  # Anchor recall of the training objects (DET, INS), written at every launch
├── Weights/
│   ├── full-checkpoint                # Registry of the saved checkpoints, the last one first
│   └── Iteration_<n>/model_<n>.pt     # Weights, optimizer state, moving average of the weights and metadata (classes included) of iteration n
├── Tensorboard/
│   ├── Train/                         # Training curves, every iteration
│   ├── Validation/                    # Validation curves
│   ├── Test/                          # Test metrics and sample images of every evaluation
│   └── Test_EMA/                      # Same for the moving average of the weights, when it is on
├── Results.txt                        # Report of every evaluated checkpoint
├── Evaluation/
│   └── Iteration_<n>/                 # Files of the evaluation of iteration n
│       └── EMA/                       # Files of the evaluation of the moving average of the weights, when it is on
├── Checkpoint_Evaluation/             # Evaluations of scripts/evaluation/evaluate_checkpoint.py, written after the training
│   ├── training_configuration.yaml    # Configuration of the evaluation, with its split and settings
│   ├── data.json, label_dictionary.json # Data of the evaluated split and classes of the run
│   ├── Results.txt, Tensorboard/Test/ # Report and test curves of the evaluated checkpoints
│   ├── evaluation_summary.csv         # One row per evaluated checkpoint, every checkpoint with iteration = "all"
│   └── Iteration_<n>/[EMA/]           # Files of the evaluation of iteration n, as in Evaluation/
├── Checkpoint_Evaluation_nms_<name>/  # The same for a DET or INS run evaluated with another final NMS (nms_criterion)
└── mlflow_run.json                    # MLflow run of the training, with the tracking on
```

- **Checkpoints**: one every `intermediate_weight_saving_iterations` iterations (0 for none), and one evaluated on the test split every `weight_saving_iterations`. `Iteration_<n>` holds n training steps, and its evaluation runs last in the iteration. The evaluation files are listed in [Evaluation Outputs](configuration_reference.md#evaluation-outputs).
- **`label_dictionary.json`**: the class index of every category, also stored in every checkpoint. The evaluation, export and model loading tools read it instead of the mapping file, so editing or moving that file never changes the classes of a run.
- **`data.json`**: written at the start of the run and sent to MLflow (about 1.5 seconds and 70 MB for the 1.3M images of ImageNet-1k). For every split, it holds the data folder, the SHA-256 of `labels.json` and the images in manifest order. Its `classes` are listed in class index order, the background first for DET, INS and SEG, each with its mapped categories and its count in every split (`null` for an SSL dataset without categories).

```json
"classes": [
  {"class_index": 0, "class_name": "square", "category_ids": [1], "images": {"train": 55540, "validation": 5581, "test": 5543}},
  {"class_index": 1, "class_name": "circle", "category_ids": [2], "images": {"train": 55373, "validation": 5753, "test": 5533}}
]
```

| Task | Counted per class |
|---|---|
| CLS, SSL | `images`, a multi-label image counting once per class |
| DET, INS, KPT | `images` holding the class, and its `annotations` |
| SEG | `images` whose mask holds the class, from the `category_ids` of `labels.json` |
| M2O | `sequences` |
| M2M, SEL | `frames` |
| CMP | `pairs` |

## Console Output
- **At launch**: the run name with clickable links to its files, the changes since the last start of a resumed run, the transferred components, the splits and their sizes, the main settings, the model summary and the model size. `diagnosis_modes.compute_receptive_field` adds the receptive field. A transferred component shows its source and learning rate:

```text
- Pretrained Backbone     : file:///…/experiments/CLS_IN-KAR-256_…_HFLAT.512.256-HLN-CWB_MC1000
    - Source Iteration    : last saved checkpoint
    - Frozen              : False
    - Norm. Layers Frozen : False
    - Learning Rate       : 0.0005 (multiplier 0.5 x 0.001)
```

- **Every `information_dump` iterations**: the averages of the interval (samples seen, throughput, real frames per second of the sequences, losses, metrics, duration). A run resumed between two summaries has a shorter first interval. The rates are written as percentages, `94.03 %`, their change since the last summary in percentage points, `+0.52 %`: the accuracy of CLS, M2O, M2M, CMP and SEL, the pixel accuracy and mean IoU of SEG and the positive pair accuracy of SimCLR (SSL); the losses, the correlations and standard deviation of Barlow Twins keep 4 decimals, and DET, INS and KPT follow only losses. The decimal points of the values and of the changes and the `Best` markers fall in one column each (`format_metric_line` of `metric_tracking.py`): a percentage with 2 decimals and its sign has as many characters after its decimal point as a 4-decimal value, so that values and changes right-aligned in 10 characters (`METRIC_COLUMN_WIDTH`) share their decimal point column, up to the losses in the thousands of Barlow Twins.
- **Dropped batches**: a batch that fails to load is skipped. Every distinct error is printed once in yellow, the first with its traceback, and the drops are counted under every summary:

```text
Training Batch Dropped : The image cannot be read : <file>
- Dropped Batches Since The Last Summary : Training 2, Validation 1 (their loading failed, see the errors printed above)
```

- **Every `weight_saving_iterations` iterations**: the evaluation report, also appended to `Results.txt`, then the `EMA Weights` report when the moving average is on. The reports of every task write their rates as percentages, the thresholds, ranks, pixel errors, effective rank and standard deviations keeping 4 decimals and the counts their whole numbers; `class_metrics.csv`, TensorBoard and MLflow keep every rate between 0 and 1.

## Stopping And Resuming

**Stopping**
- Ctrl+C saves a checkpoint of the last step. A `weight_saving_iterations` checkpoint left unevaluated is announced (`The Checkpoint Of Iteration <n> Was Not Evaluated`), to evaluate with [evaluate_checkpoint.py](tools_reference.md#evaluating-checkpoints).
- A failing iteration is reported in yellow and the training goes on; an iteration fails after 10 failed batch fetches in a row. More than 5 failing iterations in a row stop the training, with a checkpoint.
- A split giving no batch (empty, or emptied by the label mapping) is refused at launch.

**Resuming**
- A run always resumes: launching the same configuration again continues from the last checkpoint of its run folder, with its optimizer state. `number_iterations` is the total of the run: from 5000 with 8000, it trains iterations 5001 to 8000.
- Only the settings without a [run name](configuration_reference.md#run-names) token may change; any other change starts a new run folder. A `run_name_suffix` starts a separate run of the same settings.
- The run keeps the classes of its `label_dictionary.json`, warning `The Label Mapping Changed Since The Run Started, The Run Keeps Its Own Classes` when the mapping file now gives others.
- The changed settings are printed, and the previous configuration is kept as `training_configuration_until_iteration_<n>.yaml`, never overwritten:

```text
- Previous Configuration  : file:///…/training_configuration_until_iteration_72500.yaml
- Changed Settings        : Since Iteration 72500
    - ExperimentConfiguration.learning_rate        : 0.0002 -> 0.0001
    - Backbone.pretrained.learning_rate_multiplier : 0.5 -> 1.0
```

- A `labels.json` whose SHA-256 differs from `data.json` stops the resume, unless `ExperimentConfiguration.allow_data_change_on_resume: true`. The previous record is then kept as `data_until_iteration_<n>.json`, as when only the data folder moved.

**Restarting**: to train the same settings from scratch, set another `run_name_suffix` or `experiment_folder`, or move the run folder away. The new run folder starts a new MLflow run.

## Data Augmentation
- `Data.data_augmentation.activated` turns it on, `probability` is the share of training images augmented, and every type has its own `activated` and `probability`.
- It runs on the device before the forward pass, in the order of the table. When several histogram equalizations are drawn for an image, one is kept.
- The geometric types move the boxes, masks and keypoints. `random_perspective` and `random_jigsaw` move the image only, and are refused for DET, INS, SEG and KPT. The photometric types keep the targets, so `random_black_out` and `random_erasing` can hide a labelled object.
- A sequence gets the same draws on all its frames, unless `independent_frames` draws the photometric types per frame ([Sequence Tasks](tasks_reference.md#sequence-tasks-m2o-m2m-cmp-sel)).
- The strengths are fixed in `SUPERVISED_AUGMENTATION_STRENGTHS` and in `SELF_SUPERVISED_AUGMENTATION_STRENGTHS` for the [SSL views](tasks_reference.md#self-supervised-learning-ssl). Intensities are levels of [0, 255] images.

| Type | Supervised tasks | Self-supervised task |
|---|---|---|
| `random_crop` | 75 % to 100 % of each side, same aspect ratio, resized back to the input size | 20 % to 100 % of the area, aspect ratio x1/2 to x2 |
| `random_aspect_crop` | 75 % to 100 % of the width and, drawn on its own, of the height | same |
| `random_zoom_out` | image shrunk to 50 % to 100 % of each side on a black canvas | same |
| `random_translation` | shift of up to 25 % of the width, of the height or of both | same |
| `random_left_right_flip`, `random_up_down_flip` | mirror | same |
| `random_rotation` | -5 to 5 degrees | -10 to 10 degrees |
| `random_90_rotation` | -90 or 90 degrees, square images only | same |
| `random_perspective` | every corner moved inwards by up to 5 % of half the image side | same |
| `random_jigsaw` | 2x2 pieces in a random order; `Data.image_settings.size` must be a multiple of 2 | same |
| `brightness_contrast_saturation` | brightness +-40 levels, contrast and saturation x0.7 to x1.0 | brightness, contrast and saturation x0.6 to x1.2 |
| `random_hue` | hue turned by up to 18 degrees | up to 36 degrees |
| `clahe_augment` | CLAHE, clip limit 2.0 on 4x4 tiles | same |
| `sharp_augment` | Laplacian of Gaussian unsharp mask, sigma 1.5, strength 3 | same |
| `equalize_histogram_augment` | equalization of the luminance | same |
| `equalize_bgr_histogram_augment` | equalization of one channel or of all three | same |
| `random_noise` | Gaussian noise, standard deviation 20 levels | 30 levels |
| `gray_scale_augment` | grayscale on three channels | same |
| `random_colour_augmentation` | blue and red channels swapped | same |
| `random_gamma` | gamma 0.8 to 1.25 | same |
| `random_linear_illumination` | ramp of 12.75 to 38.25 levels growing towards one side | same |
| `random_plasma_shadow` | pixels of a fractal map (roughness 0.1 to 0.7) below a quantity of 0 to 0.5 darkened by up to 76.5 levels | quantity of 0 to 0.3, up to 51 levels |
| `random_blur` | 9x9 Gaussian kernel, sigma 2.5 | same |
| `random_motion_blur` | odd line of 5 to 15 pixels, angle 0 to 180 degrees, direction -1 to 1 | same |
| `random_haze` | airlight of 180 to 255 levels taking 20 % to 60 % of every pixel | 160 to 255 levels taking 20 % to 70 % |
| `random_solarize` | pixels at or above 127.5 levels inverted | pixels at or above 191.25 levels, second view only |
| `random_black_out` | every cell of a `grid_size` x `grid_size` grid blacked out with a probability of 0.2 | probability of 0.35, second view only |
| `random_erasing` | black rectangle of 2 % to 15 % of the area, height / width 0.3 to 3.3 | same |

Every type is computed on the device (`vision_training/data/device_image_operations.py`), so the host only waits for the GPU to update the boxes of DET, INS and KPT.

`scripts/data/visualize_augmentation_strengths.py` writes one figure per type, `<type>.png`: a few images of a split with the type applied alone at both strengths, the drawn values written under every image. `random_crop_aspect_ratio_explained.png` and `random_noise_levels_explained.png` explain the crop aspect ratios and the noise levels. Set the configuration, split and output folder in its main block, then run:

```bash
uv run python -m scripts.data.visualize_augmentation_strengths
```

## Viewing The Input Images
`ExperimentConfiguration.diagnosis_modes.view_input_images: True` opens a Training and a Validation window. Every image comes with a panel listing its targets and, for a training sample, the augmentations drawn with their values, the objects they removed and the frame order.
- The SSL runs show the two views of every image. With `Data.grouped_batches` on, every training batch is first shown whole, cut into its blocks.
- SPACE shows the next samples, ENTER exits without training.
- It needs a display and creates no run folder.

## Checking The Anchor Coverage
`ExperimentConfiguration.diagnosis_modes.compute_anchor_coverage: { activated: True, iou_threshold: 0.5 }` (object detection and instance segmentation) prints, for the training, validation and test splits, the share of the objects whose best anchor overlaps them by at least `iou_threshold`, per detection level and object size, then exits without training.
- Every split is read from its annotations with a progress bar, no image being loaded.
- The best anchor of every object is the grid anchor nearest its centre for every anchor shape, on the cell corners, exact and computed on the CPU.
- It creates no run folder and writes no file; the training launches measure the training split alone at `RegionProposal.foreground_iou_threshold.min` (`anchor_recall.csv`).

## Benchmarking The Speed
Before training an architecture, `ExperimentConfiguration.diagnosis_modes.benchmark_speed: True` measures how fast it trains and infers when the data never waits, then exits without training (`BaseTrainer.benchmark_speed`, helpers in `vision_training/trainers/speed_benchmark.py`). It is the only setting: the timing, the batch sizes and the memory margin are fixed in the code.
- **Batch**: one batch of the training loader, its loader workers stopped at once; `resize_batch` repeats its samples in their order to any batch size, indexing every tensor and list whose first dimension is the batch (images, labels, boxes, masks, sequence masks). The batch stays where the loader hands it over: pinned host memory for a discrete GPU, the GPU for an integrated one, whose memory is the host memory.
- **Memory**: `limit_process_memory` caps the allocator of PyTorch (`torch.cuda.set_per_process_memory_fraction`) at what the process holds plus the free memory minus a margin, 40 GB on an integrated GPU (`INTEGRATED_GPU_MEMORY_MARGIN_BYTES`, the other trainings of the GB10 slowing down once little memory is free) and 2 GB on a discrete one (`DISCRETE_GPU_MEMORY_MARGIN_BYTES`). With less free memory than the margin it stops before anything runs. ONNX Runtime gets the budget PyTorch leaves (`gpu_mem_limit`, TensorRT workspace).
- **Largest batch size**: `find_largest_batch_size` doubles from the configured batch size while two probe passes fit (halving while they do not), then bisects until the gap is at most 1/16 of the batch size found, up to `MAXIMUM_BENCHMARK_BATCH_SIZE` (2048). Only an out of memory error of PyTorch or ONNX Runtime ends a size; any other error is raised.
- **Inference**, first: the weights of the last checkpoint of the run folder are loaded when there is one (read only, the optimizer state left out), since the post-processing of the detection tasks depends on the detections. The largest batch size of the test evaluation is searched, then every format is timed at a batch of 1, at the configured batch size and at the largest one, on inputs already on the GPU: `PyTorch Evaluation` (`predict_batch` within `evaluation_context`, float32, as the test evaluation runs it), `PyTorch Inference Model` (the network every export traces), `TorchScript` (`export_torchscript`, loaded by `torch.jit.load`) and the ONNX file (`export_onnx`) in ONNX Runtime CUDA and ONNX Runtime TensorRT FP32 and FP16, its inputs and outputs bound on the GPU. The ONNX formats need `onnxruntime-gpu`; a format that cannot open or fails is reported with its reason. The sequence comparison gives both members of its pairs (`build_inference_inputs`).
- **Training**: the largest training batch size is searched, then `run_training_step`, the step of the training loop (batch moved, augmented, bfloat16 forward pass, backward pass, optimizer step with the moving average), is timed at the configured batch size and at the largest one, with the trackers of the training loop. The training time of `number_iterations` at the configured batch size is a lower bound: the loading, validation passes, checkpoints and test evaluations are left out.
- **Timing**: `measure_iteration_time` runs at least 2 seconds and 3 iterations untimed, then 5 windows lasting 5 seconds in all, timed between two synchronisations of the device; a row gives the median window in milliseconds, the samples (and real frames) per second, the window spread (slowest minus fastest, over the median) and the peak memory reserved by PyTorch (not measured for ONNX Runtime).
- **Report**: the GPU, the memory budget, the batch, the weights and the other processes on the GPU (from nvidia-smi, the daemon of the Multi-Process Service left out), then the inference and training tables with their notes. Nothing is written to the run folder.

## TensorBoard Monitoring

| Folder | Written | Content |
|---|---|---|
| `Train`, `Validation` | every iteration | losses and metrics |
| `Test` | every evaluation | test scalars, and the sample images `Samples/<sample>` with a slider over the evaluations |
| `Test_EMA` | every evaluation, with the moving average on | the `Test` tags of the averaged weights, drawn on the same charts |

## Moving Average Of The Weights
- After every optimizer step: `ema_weights = decay × ema_weights + (1 − decay) × current_weights`, the batch normalization statistics included.
- It follows roughly the last 1 / (1 − decay) iterations: about 1,000 at 0.999, 5,000 at 0.9998 and 10,000 at 0.9999. The n-th update uses min(decay, (1 + n) / (10 + n)), so the random starting weights fade quickly.
- It costs the memory of one more model, and a small fraction of a training step.
- It is saved in every checkpoint and restored on resume. It gives no run name token: turned on mid-run, it starts from the weights of the checkpoint.
- The validation curves and the [export](tools_reference.md#export) use the trained weights, the export taking the averaged ones with `use_averaged_weights`.

Every test evaluation scores the trained weights, then the averaged weights of the same checkpoint:

| Output | Trained weights | Averaged weights |
|---|---|---|
| Console and `Results.txt` header | `Iteration <n> · <evaluated samples> · <date and time>` | `Iteration <n> · <evaluated samples> · EMA Weights · <date and time>` |
| Evaluation files | `Evaluation/Iteration_<n>/` (`Checkpoint_Evaluation/Iteration_<n>/` for `evaluate_checkpoint.py`) | `Evaluation/Iteration_<n>/EMA/` (`Checkpoint_Evaluation/Iteration_<n>/EMA/`) |
| TensorBoard | `Test` | `Test_EMA` |
| MLflow | `test/`, `test_class/`, `test/best_<headline metric>` (tag `best_iteration`) | `test_ema/`, `test_ema_class/`, `test_ema/best_<headline metric>` (tag `best_iteration_test_ema`) |
| `evaluation_summary.csv` of `evaluate_checkpoint.py` | one column per scalar | the same columns prefixed by `ema_` |

## Transfer Between Runs
The `pretrained` block of a section initialises its component from a trained run, of the same task or of another one:

| Section | Component | From the tasks |
|---|---|---|
| `Backbone` | backbone | any task |
| `TemporalModel` | temporal module | M2O, M2M, CMP, SEL |
| `ClassificationHead` | classification head, with its class layer | CLS, M2O, M2M, CMP, SEL |
| `SegmentationDecoder` | U-Net decoder, with its class layers when `load_class_layers` is True | SEG |
| `RegionProposal` | region proposal heads of every detection level | DET, INS |
| `Detector` | detection heads of every detection level | DET, INS |
| `MaskHead` | mask heads of every detection level | INS |
| `HRNet` | high resolution network | KPT |
| `KeypointHead` | heatmap and tag head, with its higher resolution branch | KPT |

The settings defining the layers must match, which is checked at launch. The image size may differ, and so may the `batch_normalization_momentum` of a source run written before the momentum was fixed.

| Component | Settings that must match |
|---|---|
| Backbone | `input_channels`, `convolutions`, `modules`, `normalization`, `last_max_pooling`; when one backbone's `convolutions` are the first blocks of the other's (a shorter stem of a keypoint detector, or a deeper backbone), the shared blocks are loaded and `last_max_pooling` is free |
| Classification head | `fully_connected`, `fully_connected_normalization`, and the image size after a `flatten` pooling |
| Transformer with positional embedding | `maximum_sequence_length` |
| Region proposal heads | detection levels, `region_proposal` and anchors per position of every level, normalization, `Neck.type` `none` in both runs |
| Detection heads | detection levels, `detector` of every level, normalization, modules, `Neck.type` `none` in both runs |
| Mask heads | detection levels, `mask_head` of every level, `mask_resolution`, `upsampling`, `skip_blocks_at_pool_size`, `image_branch_convolutions`, `skip_blocks_at_mask_resolution`, `fusion_convolutions` (a source run written before them counting as without), normalization, modules, `Neck.type` `none` in both runs; the image branch is not transferred, so the run receiving the heads keeps it `[0, 0]` |
| High resolution network | `base_width`, `modules_per_stage`, `blocks_per_branch`, `modules`, `normalization`, and the channels of the last stem block |
| Keypoint head | `KeypointHead.higher_resolution` and `hidden_width`, the `HRNet.base_width`, and with the higher resolution branch its `blocks_per_branch`, `modules` and `normalization` (the normalization also with a hidden width above 0) |

- An object detection run can thus initialise an instance segmentation run, whose mask heads train alone with the other components frozen.
- A backbone deeper than its source backbone starts with the source blocks; its next blocks keep their random weights and learn at the learning rate of the run, in the parameter group of the new layers. `freeze` (with the batch statistics), `freeze_normalization_parameters` and `learning_rate_multiplier` apply to the loaded blocks only. The console prints `Starts With The <n> Blocks Of The Pretrained Backbone` and a `Loaded Blocks` line, and every checkpoint stores the number of loaded blocks (`pretrained_backbone_blocks`), so that a resumed run freezes and groups the same blocks even once its source run is gone.
- A resumed run does not load the pretrained component again.
- The averaged weights of the source are loaded when the source kept a [moving average](#moving-average-of-the-weights) and the new run has one on. The console says which weights were loaded.
- In the run name, `PT` marks a pretrained component and `FRZ` a frozen one, prefixed by the letter of the section (`RPT`, `RFRZ`); `load_class_layers` adds `DCL`. The `learning_rate_multiplier` gives no token.

### Adding Classes
A component with class layers can initialise a run with more classes than its source:
- The source classes must come first in the new run, in the same order, the background staying first for DET, INS and SEG: append the new classes at the end of the label mapping file.
- The class names are compared from the `data.json` of both runs ([Run Folder](#run-folder)). A renamed, reordered or missing class stops the run.
- The rows of the source classes are copied into the class layers. The new rows keep their random initialisation, which may disturb the source classes for the first iterations. The console prints the kept and added classes.
- With `load_class_layers`, a deep supervision level missing from the source keeps random weights, with a warning.

| Layer | Section |
|---|---|
| Class layer of the classification head | `ClassificationHead` |
| Classification and box regression layers of the detection heads | `Detector` |
| Mask prediction layer of the mask heads | `MaskHead` |
| Final class layer and deep supervision layers of the decoder | `SegmentationDecoder` with `load_class_layers` |

On the synthetic shapes (9 classes extended to 20, stage 14 of `documentation/architecture_benchmarking/experiment_plan.md`), the new classes reached an accuracy of 75.13 % at iteration 250 (trained weights), and the source classes kept 99.86 % (averaged weights).
