# Classification Workflow

This document follows one classification run from `main()` to the evaluation of a checkpoint. It shows which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation and a **real** output.

To read the examples, know which run each comes from; this table lists them.

| Stages | Run |
|---|---|
| 1 and 2 | `configuration_example.yaml` on CPU when it read ImageNet-1k (1000 classes, 256 pixels, batch 256, flattened head); it reads ImageNet-22k (`MC21841`) since 2026-11-11 |
| 3, 4 and 5 | `configuration_synthetic_shapes.yaml` on CPU, 2026-10-09, 9-class dataset, shortened to 64 pixels, batch 8, 12 iterations (settings in the Stage 3 note); near chance on purpose (1/9 = 11.11 %, loss near ln 9 = 2.197) |
| current outputs (Step 4.18), current MLflow run (Step 3.5) | README run `CLS_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128-PT_HFLAT.128-HLN-CWB_MC35` of `~/Data/tutorial_runs/readme_screenshots/` (outside the repository): 2,000 iterations, batch 64, shape pre-trained backbone, 35-class dataset (500,000 / 50,000 / 50,000), 81.76 % accuracy, 99.12 % top-5 |
| curves, class metrics, TensorBoard at iteration 1500 | 9-class README run of 2026-10-23 (`..._HFLAT.128-HLN-CWB_MC9`, same folder, 1,500 iterations from scratch) |

The CPU run and the 9-class README run show `Top-2 Accuracy` (k was a setting); the current code writes `Top-5 Accuracy`. Console rates are percentages with two decimals (`10.96 %`, changes in points); files, TensorBoard and MLflow keep 0 to 1. Console paths are terminal hyperlinks (`format_path_link()`): Stages 1 and 2 show the names, Stages 3 to 5 the `file://` URI.

Every other task has a workflow document in this folder. The steps shared by every task (training loop, checkpoints, MLflow tracking, standalone evaluation) are explained in most detail here.

## Read This First

The classification task (`task: "classification"`, run name code `CLS`) learns the class of a whole image. A BGR image resized to `Data.image_settings.size` (padded to a square with `keep_ratio`) goes through the shared backbone, a global pooling or flattening, and a fully connected head giving K logits. The logits are read with a softmax (`ClassificationHead.output_mode: "multi_class"`, one class per image) or one sigmoid per class (`"multi_label"`).

Choose it when the label belongs to the whole image. Choose DET or INS to locate objects, SEG for a class per pixel, KPT for points, the sequence tasks (M2O, M2M, CMP, SEL) when the decision needs several frames, and SSL to pre-train the backbone without labels. The sequence trainers extend `ClassificationTrainer` and reuse its head, loss and evaluation. A backbone trained here initialises any other task (`Backbone.pretrained`).

### The Task At A Glance

To find your way in the stages below, this diagram shows the whole task, from the manifest of a split to the exported classifier.

```text
<split>/labels.json  {"id": 1, "file_name": "train_000000.png", "category_id": 26}   label_mapping.json {"1": 0, ..}
<split>/data/train_000000.png ─► ClassificationDataset: read_image (BGR) + preprocess_image (512, keep_ratio)
                                 sample {images [3, 512, 512] uint8, labels: class index, sample_identifiers}
DataLoader (batch 64) ─► prepare_batch: images [64, 3, 512, 512] float32 in [0, 255], labels [64] int64
                      ─► DataAugmenter.augment_batch (training batches only, labels unchanged)
ImageClassifier: InputNormalizer x / 255 - 0.5 ─► Backbone, 6 ConvolutionBlocks (SE, shortcuts in blocks 3 and 5)
   B1 [64, 16, 256, 256] B2 [64, 32, 128, 128] B3 [64, 32, 64, 64] B4 [64, 64, 32, 32] B5 [64, 64, 16, 16]
   B6 [64, 128, 16, 16] (no pooling after the last block)
   ─► flatten [64, 32768] ─► head: Dropout, Linear 128, LayerNorm, ReLU, Dropout, Linear 35
   ─► class_logits [64, 35] ─► softmax ─► class_probabilities [64, 35]
loss: classification_loss = cross entropy (balanced class weights, no label smoothing) = total_loss
batch metric: accuracy (argmax == label) ─► TensorBoard Train / Validation every iteration, console every 10
every weight_saving_iterations: test split ─► confusion matrix, accuracy, balanced accuracy, macro F1, top-5,
   per class precision, recall with its 95 % interval, F1, ROC AUC, average precision, threshold at precision 0.99
   ─► Results.txt block, Tensorboard/Test, Evaluation/Iteration_<n>/ (confusion_matrix.csv/.png, class_metrics.csv,
      class_thresholds.json, roc_curves.png, precision_recall_curves.png, precision_recall_thresholds.png,
      temperature_explorer.html, class_logits.npz, predictions.csv, Samples/; Errors/ with error_dumps)
temperature: serve_temperature_explorer.py, http://127.0.0.1:8091/ ─► a typed T ─► Save Thresholds At This Temperature
   ─► saved_temperature_thresholds.json next to the page ─► Export At T ─► <run>/Exports/Iteration_<n>[_EMA]_T<T>/
export: InferenceClassifier, input images [B, 3, 512, 512] float32 BGR ─► class_logits / logit temperature ─► softmax
   ─► class_probabilities [B, 35], preprocessing.json (input contract) and the saved thresholds as class_thresholds.json
      next to the model
```

### Files To Know

If you are looking for the code of a step, this table lists the files of the task and their main functions.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/classification_trainer.py` | `ClassificationTrainer`: `build_dataloaders`, `build_model`, `compute_batch_metrics`, `accumulate_evaluation_batch`, `dump_errors`, `build_sample_image`, `compute_evaluation_results` |
| Shared training loop | `vision_training/trainers/base_trainer.py` | `BaseTrainer`: `run_training_iterations`, `run_model_iteration`, `step_optimizer`, `save_model`, `run_test_evaluation`, `restore_last_model` |
| Dataset and loaders | `vision_training/data/classification/classification_dataset.py`, `vision_training/data/serialized_records.py`, `vision_training/data/loaders.py` | `ClassificationDataset`, `SerializedRecords` (`find_record`), `get_classification_dataloaders`, `build_native_loader` |
| Model | `vision_training/models/classification/image_classifier.py`, `vision_training/models/backbones/backbone.py`, `vision_training/models/components/convolution_block.py` | `ImageClassifier`, `build_backbone`, `Backbone`, `ConvolutionBlock` |
| Head | `vision_training/models/classification/classification_head.py`, `vision_training/models/components/fully_connected_block.py` | `build_global_pooling`, `compute_pooled_features`, `ClassificationHead`, `build_fully_connected_block` |
| Loss | `vision_training/losses/classification_losses.py` | `compute_classification_loss`, `convert_logits_to_probabilities`, `compute_balanced_class_weights`, `compute_decorrelation_loss` |
| Metrics and evaluation | `vision_training/evaluation/classification/` | `compute_confusion_matrix`, `compute_metrics_from_confusion_matrix`, `compute_top_k_accuracy`, `find_class_thresholds`, `compute_class_score_curves`, `save_confusion_matrix_image`, `save_score_curve_images`, `build_class_metrics_table` |
| Recall intervals | `vision_training/evaluation/recall_intervals.py` | `compute_clopper_pearson_intervals`, `build_recall_interval_table` |
| Logit temperature | `vision_training/evaluation/classification/temperature_explorer.py`, `scripts/evaluation/serve_temperature_explorer.py`, `vision_training/deployment/explorer_export.py` | `save_temperature_explorer`, `save_class_logits`, `compute_temperature_thresholds`, `compute_score_ranges`, `serve_temperature_explorers`, `prepare_export_job`, `run_export_job` |
| Configuration checks | `vision_training/configuration_processing/classification_checks.py` | `check_classification_configuration` |
| Run name | `vision_training/configuration_processing/run_name.py` | `build_run_name`, `build_classification_sections` |
| Synthetic data generator | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset`, `build_classification_labels` (one shape per image, or with `multi_label` the distinct classes of the counted shapes images of `configuration_synthetic_counted_shapes.yaml`) |
| ImageNet-22k download | `vision_training/data/classification/imagenet_download.py` | `download_imagenet_22k` (one held-out image per class of at least 50 images as both validation and test, `--held_out_images_per_class`, `--minimum_held_out_class_size`), `write_class_folder_manifests` |
| Export | `vision_training/models/classification/inference_classifier.py`, `vision_training/deployment/export_model.py`, `vision_training/deployment/input_contract.py` | `InferenceClassifier` (model name `image_classifier`), `export_experiment`, `build_input_contract`, `write_preprocessing_file` |
| Export sizes at launch | `vision_training/deployment/export_size_estimates.py` | `estimate_export_sizes`, `probe_inference_graph`, `print_export_size_summary` |
| Standalone evaluation | `scripts/evaluation/evaluate_checkpoint.py`, `vision_training/evaluation/checkpoint_evaluation.py` | `evaluate_checkpoints`, `summarise_checkpoint_evaluations` |
| Figures of a research report | `scripts/evaluation/draw_report_figures.py` | `draw_report_figures`, `read_report_thresholds`, `draw_precision_recall_figure`, `compare_seen_and_never_seen_groups` |
| Scoring a large split | `scripts/evaluation/score_image_split.py` | `build_scoring_trainer`, `score_dataset`, `read_scored_samples`, `compare_precisions` |

### Run It

In order to train and evaluate the synthetic shapes classifier from scratch, run these commands from the repository root.

```bash
# Synthetic dataset (500,000 / 50,000 / 50,000 images of 512 pixels by default, other sizes with --split_sizes)
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=classification --output_folder="$HOME/Data/classifier_data/synthetic_shapes"

# Training: set experiment_folder, the three data folders and Data.label_mapping.mapping_file_path first
uv run python -m vision_training.main --configuration_path="vision_training/configurations/classification/configuration_synthetic_shapes.yaml"

# Evaluation of every checkpoint into <run>/Checkpoint_Evaluation/ (other arguments in Step 5.1)
uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder <run folder> --iteration all

# Temperature and export: open http://127.0.0.1:8091/, save the thresholds of a temperature, export from that page
uv run python -m scripts.evaluation.serve_temperature_explorer

# Report figures into <run>/Checkpoint_Evaluation/Iteration_<n>[/EMA]/Report_Figures/ (--weights ema for the averaged weights)
uv run python -m scripts.evaluation.draw_report_figures --experiment_folder ~/Data/<run> --iteration 2000
```

### Configuration Keys To Know First

Before changing the synthetic configuration, this table shows the keys that change the model, the loss or the evaluation the most.

| Key | Synthetic value | Effect |
|---|---|---|
| `ClassificationHead.output_mode` | `"multi_class"` | softmax, cross entropy, headline `Accuracy`; `"multi_label"`: sigmoids, binary cross entropy, `Subset Accuracy`, `Micro F1`, `Macro F1`, no confusion matrix nor error images |
| `ClassificationHead.global_pooling` | `"flatten"` | the [128, 16, 16] map becomes 32,768 features, so the head depends on the image size (`HFLAT`); `"average"` or `"max"` give 128 |
| `ClassificationHead.fully_connected` | `[128]` | hidden layers; the first holds 4,194,432 of the 4,637,571 parameters of the 35-class model |
| `ClassificationHead.fully_connected_normalization` | `"layer"` | LayerNorm in every hidden layer (`HLN`); `"batch"` (`HBN`) also makes the training loader drop its last incomplete batch |
| `ClassificationHead.class_weighting` | `"balanced"` | class weights N / (K x N_k) in the cross entropy (`CWB`) |
| `ClassificationHead.label_smoothing` | `0.0` | above 0 gives `LS<rate>`; with balanced weights it favours the rare classes (pitfall below) |
| `ClassificationHead.decorrelation` | `activated: False` | adds the squared off-diagonal correlations of one hidden layer to the loss, no run name token |
| `ExperimentConfiguration.evaluation.target_precision` | `0.99` | threshold of every class in `class_thresholds.json` and the `@ P99 %` columns; a list such as `[0.99, 0.95, 0.90]` adds the other precisions (Step 4.13) |
| `ExperimentConfiguration.evaluation.per_class_report` | `True` | per-class tables and scalars, confusion matrix image, ROC and precision-recall curves, temperature explorer with its logits |

### What A Run Writes

Each time a checkpoint is evaluated, the run folder, TensorBoard and MLflow receive the outputs of this table.

| Place | Content |
|---|---|
| `Results.txt` | one block per checkpoint, `Iteration <n> · <N> Test Images · <date>`: `Accuracy`, `Balanced Accuracy`, `Macro F1`, `Top-5 Accuracy`; with `per_class_report` also `Macro ROC AUC`, `Macro Average Precision`, the per-class table (by F1, worst first) and the `Per Class Recall` table (recall with its 95 % interval, worst first). Rates as percentages (`81.76 %`), thresholds between 0 and 1. |
| `Evaluation/Iteration_<n>/` | always `confusion_matrix.csv` (or `confusion_cells.csv`), `class_metrics.csv`, `class_thresholds.json`; with `per_class_report` the confusion matrix and curve images, `temperature_explorer.html`, `class_logits.npz`; `saved_temperature_thresholds.json` once saved; `predictions.csv`, `Errors/`, `Samples/` and `EMA/` when switched on (Steps 4.6 to 4.16) |
| `Evaluation/evaluation_summary.csv` | one row per evaluated iteration with every scalar (`ema_` columns for averaged weights); `evaluation_curves.png` next to `Results.txt` redrawn after each evaluation (Step 4.18) |
| `Tensorboard/Train`, `Tensorboard/Validation` | `Total Loss`, `Classification Loss`, `Accuracy` (`Learning Rate` in Train) |
| `Tensorboard/Test` | headline metrics, per-class scalars, sample, confusion matrix and curve images |
| `Exports/Iteration_<n>[_EMA]_T<temperature>/` | one Triton model repository per export from the temperature explorer, with `export_record.json` and `export_log.txt` (Step 4.13) |
| MLflow | `test/<metric>`, `test_class/<metric>/<class>`, `test/best_Accuracy`, tag `best_iteration`, files of `evaluation/iteration_<n>/` |

Files, TensorBoard and MLflow keep every rate between 0 and 1.

### Pitfalls

Should a result of the task look wrong, check these behaviours first.

- Blocks 2, 4 and 6 of the synthetic configuration change the channel count (16 to 32, 32 to 64, 64 to 128, 2 layers each), so they add no residual shortcut. Only blocks 3 and 5 list `Residual Shortcuts` (`ConvolutionBlock.residual_parity`, `count_residual_additions()` of `convolution_block.py`).
- `label_smoothing` above 0 with `class_weighting: "balanced"` trains towards the rare classes: PyTorch also weights the smoothing term. With weights [1, 20, 1000] and smoothing 0.1, the loss of a class-0 image is lowest at [0.027, 0.019, 0.954] (`compute_classification_loss()`; latent, every shipped configuration has smoothing 0).
- The `flatten` pooling ties the head to `Data.image_settings.size` and `Backbone.last_max_pooling`. Another image size changes the first hidden layer input and the run name; a head transfers only to a run of the same image size (`compute_pooled_features()`; [training reference, Transfer Between Runs](../reference/training_reference.md#transfer-between-runs)).
- The synthetic dataset holds the 35 classes of `SHAPE_CATEGORIES` (run ends with `MC35`) and is drawn without a seed, so a regenerated dataset holds other images. The CPU run and the 9-class README run used the older 9-class dataset ([tasks reference, Synthetic Datasets](../reference/tasks_reference.md#synthetic-datasets)).
- `Iterations Per Epoch` and the sample counts sent to MLflow count the manifest samples before the label mapping, so they are too high when the mapping leaves categories out (`count_split_samples()` of `split_description.py`).

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

To see the whole run on one page, this tree lists every function in calling order with one line on its role; each step below gives the details.

```text
main()                                         entry point of every task: --configuration_path, expandable CUDA segments
├── parse_configuration_path()                 argparse; "No configuration file at ..." when the path is not a file
├── load_experiment_configuration() -> prepare_experiment_configuration()   splits the YAML into the two dictionaries
│   ├── load_configuration()                   yaml.safe_load into one raw dictionary, FileNotFoundError if missing
│   ├── check_configuration_keys()
│   │   ├── check_removed_settings()           a removed setting reported alone, with what replaced it
│   │   └── check_unknown_keys()               every key the task does not read (KNOWN_KEY_PATHS)
│   ├── build_label_dictionary()               "multi_class": {category: index} from Data.label_mapping.mapping_file_path
│   └── check_configuration()                  gathers ALL errors, printed in red, exit code 1
│       ├── check_experiment_configuration()   task, suffix, loop settings, per-class report, samples, MLflow
│       ├── check_data_configuration()         augmentations, label mapping mode, positive even image size
│       ├── check_backbone_configuration()     blocks "1".."N", 3 input channels, size divisible by the poolings
│       ├── check_optimization_configuration() optimizer, weight decay, schedule, moving average decay
│       └── check_classification_configuration()  classes 0..K-1, head, decorrelation, evaluation settings
├── get_trainer_class(task)                    TASK_TRAINER_CLASSES["classification"] -> ClassificationTrainer
├── ClassificationTrainer(...)  [STAGE 2]      stores output_mode, class_weighting, evaluation settings, top-k = 5
│   └── BaseTrainer.__init__()                 everything independent of the task, task hooks in ClassificationTrainer
│       ├── get_device()                       cuda, else mps, else cpu
│       ├── get_run_name() -> build_run_name() folder name from the settings that define the model
│       ├── DataAugmenter()                    stores the 28 augmentation types, draws happen in Stage 3
│       ├── restore_run_label_dictionary()     a resumed run keeps its own classes
│       ├── check_resumed_data()               a changed labels.json stops a resume (allow_data_change_on_resume)
│       ├── setup_training_paths()             Tensorboard/, Weights/, YAML copy, label_dictionary.json
│       ├── build_dataloaders()                hook: three loaders (validation None when not needed), class_names
│       │   └── get_classification_dataloaders()
│       │       ├── ClassificationDataset.__init__()  manifest read, categories mapped, list shuffled once
│       │       └── build_native_loader()      DataLoader, 1 persistent worker, 2 batches ahead
│       ├── check_loaders_give_batches()       refuses a split that gives no batch
│       ├── save_data_record()                 data.json: classes, folders, SHA-256 of every labels.json
│       ├── get_test_sample_classes() + choose_sample_identifiers()  fixed sample images for every evaluation
│       ├── build_model()                      hook: ImageClassifier, balanced class weights
│       │   ├── ImageClassifier.__init__()
│       │   │   ├── build_backbone()           InputNormalizer + one ConvolutionBlock per entry
│       │   │   ├── compute_pooled_features()  C, or C x H x W for flatten
│       │   │   ├── build_global_pooling()     average, max or flatten
│       │   │   └── ClassificationHead.__init__() -> build_fully_connected_block()
│       │   └── compute_balanced_class_weights()   N / (K x N_k)
│       ├── initialise_pretrained_components() loads and freezes transferred components
│       ├── build_optimizer()                  Adam, AdamW or SGD; moving average of the weights
│       ├── get_tracked_metrics() + MetricTracker.__init__()
│       ├── print_experiment_information() -> describe_split() -> count_split_samples()
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       ├── estimate_export_sizes() + print_export_size_summary()
│       ├── view_input_batches()               only with view_input_images, then exits
│       ├── benchmark_speed()                  only with benchmark_speed, then exits
│       └── restore_last_model()               start_iteration: 1, or last checkpoint + 1
│           ├── get_last_checkpoint_iteration()
│           └── load_model() -> load_checkpoint()
└── trainer.run_training_loop()  [STAGE 3]     closes MLflow as FINISHED, KILLED or FAILED
    └── run_training_iterations()              start_iteration to number_iterations (the run total)
        ├── save_input_samples()               Input_Samples/: first 10 images of every split
        ├── setup_tensorboard_writers()        Train, Validation (Test at the first evaluation)
        ├── initialize_training_loop_state() -> MetricTracker.create_trackers()
        ├── start_experiment_tracking()        MLflow run, background thread
        ├── for each training_iteration:
        │   ├── get_next_batch() -> ClassificationDataset.__getitem__() -> read_image() + preprocess_image()
        │   ├── MetricTracker.count_frames()   frames of the sequence tasks; nothing for images
        │   ├── run_training_step()            set_training_mode() to step_optimizer(), timed by benchmark_speed
        │   ├── set_training_mode()            model.train(), frozen components in eval
        │   ├── run_model_iteration()          one training or validation batch, returns the total loss
        │   │   ├── prepare_batch()            float32 images, int64 labels on the device
        │   │   ├── augment_batch() -> DataAugmenter.augment_batch()   training only
        │   │   ├── forward_pass() -> ImageClassifier.forward()          bfloat16 autocast
        │   │   │   ├── Backbone.forward()
        │   │   │   ├── global_pooling + ClassificationHead.forward()
        │   │   │   ├── convert_logits_to_probabilities()   softmax, or sigmoids for multi_label
        │   │   │   └── compute_classification_loss()       (binary) cross entropy
        │   │   ├── compute_batch_metrics()    accuracy from the logits
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward()
        │   ├── step_optimizer() -> compute_learning_rate_factor()
        │   ├── save_model() -> save_checkpoint()          multiples of the saving frequencies
        │   ├── validation pass: run_model_iteration() without augmentation
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary() + report_dropped_batches()
        │   └── run_test_evaluation()  -> [STAGE 4]        multiples of weight_saving_iterations
        ├── save_model() of the steps taken since the last checkpoint, if any
        └── close TensorBoard writers; run_training_loop() then calls shutdown_loader_workers(), finish_experiment_tracking()

run_test_evaluation(iteration) -> evaluate_test_split()  [STAGE 4]   whole test split, then again on the EMA weights when on
├── create_evaluation_accumulator()
├── for each test batch:
│   ├── prepare_batch()
│   ├── predict_batch() -> ImageClassifier.forward() without labels
│   ├── accumulate_evaluation_batch() -> dump_errors() -> reserve_error_image() + build_input_image()
│   └── save_sample_images() -> build_sample_image()
├── compute_evaluation_results()               EvaluationResults dictionary for the reporter
│   ├── compute_confusion_matrix()             [K, K], sparse above 1,000 classes
│   ├── compute_metrics_from_confusion_matrix() -> compute_precision_recall_f1()
│   ├── compute_top_k_accuracy()               (compute_multi_label_metrics in multi_label mode)
│   ├── save_confusion_matrix_table()          confusion_matrix.csv or confusion_cells.csv
│   ├── find_class_thresholds()                class_thresholds.json
│   ├── compute_class_score_curves() + compute_macro_curve_metrics()   per_class_report only
│   ├── compute_recall_intervals() -> compute_clopper_pearson_intervals()
│   ├── build_class_metrics_rows() + write_class_metrics_file()        class_metrics.csv
│   ├── save_confusion_matrix_image()          per_class_report, up to 1,000 classes
│   ├── save_score_curve_images()              per_class_report
│   ├── save_temperature_explorer() + save_class_logits()              per_class_report
│   ├── write_predictions_file()               save_predictions only
│   └── build_class_metrics_table() + build_recall_interval_table() + build_class_metric_scalars()
└── report_evaluation()
    ├── report_evaluation_results()            TensorBoard Test, Results.txt, console, file links
    ├── track_evaluation()                     MLflow scalars, best value, small files
    └── update_evaluation_curves()             evaluation_summary.csv, evaluation_curves.png

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]   command line arguments, writes <run>/Checkpoint_Evaluation/
├── list_evaluated_iterations() -> get_checkpoint_iterations() / get_last_checkpoint_iteration()
├── load_experiment_configuration() + get_trainer_class() + trainer construction   (same as Stages 1-2)
└── evaluate_checkpoints()
    ├── start_evaluation_tracking()            MLflow run '<run name>-evaluation'
    └── summarise_checkpoint_evaluations()     best iteration per kind of weights
        └── run_test_evaluation()  for each checkpoint  -> [STAGE 4]

scripts/evaluation/draw_report_figures.py  [STEP 5.7]   report figures of a kept checkpoint
scripts/evaluation/score_image_split.py  [STEP 5.8]     scores a large split with one checkpoint
```

---

## Stage 1 — Entry Point and Configuration Loading

> The outputs below come from the loader on `configurations/classification/configuration_example.yaml` when it read ImageNet-1k (1000 classes); it reads ImageNet-22k (21,841 classes) since 2026-11-11.

### Step 1.1 — `main()`
Every task starts here.
- `enable_expandable_cuda_segments()` (`os_utilities.py`) adds `expandable_segments:True` to `PYTORCH_CUDA_ALLOC_CONF` before any GPU use.
- `parse_configuration_path()` reads `--configuration_path`; `parser.error()` stops with "No configuration file at ..." for a missing file.
- Loading and trainer construction run inside `read_manifests_once()` (`vision_training/data/manifest_cache.py`): every `labels.json` is parsed once by `read_manifest()`, shared, and freed before the workers fork. `evaluate_checkpoint.py` builds its trainer the same way.

The block below shows the body of `main()`.

```python
enable_expandable_cuda_segments()                           # before the first CUDA call
experiment_configuration_path = parse_configuration_path()  # absolute path of --configuration_path
with read_manifests_once():                                 # every labels.json parsed once, freed at the end
    experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=...)
    trainer_class = get_trainer_class(task=experiment_configuration["task"])  # -> ClassificationTrainer
    trainer = trainer_class(experiment_configuration=..., model_configuration=..., configuration_path=...)
trainer.run_training_loop()
```

### Step 1.2 — `load_experiment_configuration()`
Through `prepare_experiment_configuration()`, it returns the two dictionaries every later stage reads: `experiment_configuration` (the `ExperimentConfiguration` section) and `model_configuration` (every other section).
- It checks the keys as written (Step 1.5), then fixes types YAML reads wrong: folders to `Path`, counts such as `1e10` to `int` (a non-integer count is left to the checks), the learning rate, optimization numbers without a dot such as `weight_decay: 1e-4`.
- It adds `tracking_root` (the `experiment_folder`, MLflow root), `run_name` (None, Step 2.4) and `standalone_evaluation_folder` (None, set by Step 5.2), and `label_mode` and `label_dictionary` to `Data` (Step 1.4), then runs the checks. Every model is built in float32 (`MODEL_DTYPE` of `vision_training/models/model_registry.py`).

The block below shows how the keys are split.
```text
experiment_configuration keys: task, experiment_folder, run_name_suffix, train_data_folder, validation_data_folder,
                               test_data_folder, batch_size, information_dump, weight_saving_iterations,
                               intermediate_weight_saving_iterations, number_iterations, validation_iteration_ratio,
                               evaluation, tracking, learning_rate, allow_data_change_on_resume,
                               diagnosis_modes, tracking_root, run_name, standalone_evaluation_folder
model_configuration keys:      Optimization, Data, Backbone, ClassificationHead
Data keys after loading:       image_settings, label_mapping, data_augmentation, label_mode, label_dictionary
```

This table shows three values before and after conversion.

| Key | Raw YAML value | After conversion |
|---|---|---|
| `number_iterations` | `'1e10'` (string) | `10000000000` (int) |
| `train_data_folder` | `'/home/.../imagenet-1k/manifests/train'` (string) | `PosixPath('/home/.../imagenet-1k/manifests/train')` |
| `learning_rate` | `0.0002` (`2.0e-4` has a dot, so YAML reads a float) | `0.0002` (float) |

### Step 1.3 — `load_configuration()`
A plain file reader. It raises `FileNotFoundError` if the YAML is missing, otherwise parses it with `yaml.safe_load` into one nested dictionary. Nothing is converted: `1e10` is still a string, hence Step 1.2.

### Step 1.4 — `build_label_dictionary()`
This function decides which class index each dataset category is trained as.
- The mode must be `"multi_class"`. It reads the JSON at `Data.label_mapping.mapping_file_path` and turns digit keys into integers.
- The result goes into `model_configuration["Data"]["label_dictionary"]`. Datasets map raw categories with it; the classifier counts its distinct values to size its output layer.
- Several categories mapped to the same index are merged.
- A configuration read from a run folder holding `label_dictionary.json` (evaluation, export and model loading tools, Step 5.2) takes that file's classes (`read_run_label_dictionary()`).

The example shows the mapping file and the resulting dictionary.
```text
{'1': 0, '2': 1, '3': 2, '4': 3, ...}      ->   {1: 0, 2: 1, 3: 2, 4: 3, ...}
                                                1000 entries, 1000 distinct classes -> 1000 output neurons
```

### Step 1.5 — `check_configuration()`
This safety net runs before loaders, model or GPU memory exist.
- Before the conversions of Step 1.2, `check_configuration_keys()` reports alone the removed settings (`check_removed_settings()`: `'<key>' was removed: <what replaced it>. Delete it from the configuration file.`). Examples: `resume_training`, `mixed_precision`, `Data.loading`, `evaluation.top_k`, `Data.view_augmentation`, `Optimization.gradient_clipping_norm`.
- Then, alone too, every key the task does not read (`check_unknown_keys()` against `KNOWN_KEY_PATHS` of `configuration_keys.py`): `'<key>' is not a configuration key (did you mean '<closest key>'?): no code reads it, so its value would be ignored. Correct it or delete it from the configuration file.`
- `check_configuration()` runs the common checks, then the task checks of `TASK_CONFIGURATION_CHECKS`. Each returns a list of messages, so every problem is reported together. Any message: printed in red under a link to the YAML (`print_configuration_problems()`, called by `load_experiment_configuration()`), exit code 1.

This table lists what each check covers.

| Check | Covers |
|---|---|
| `check_experiment_configuration()` | known task, run name suffix, loop settings, per-class report, number of sample images, MLflow tracking |
| `check_data_configuration()` | augmentation types and probabilities, label mapping mode, image size a positive even integer |
| `check_backbone_configuration()` | blocks numbered "1".."N", each [layers, channels], 3 input channels, image size a multiple of 2 ** poolings with every map entering a pooling even |
| `check_optimization_configuration()` | optimizer, weight decay, warm-up / cosine schedule, moving average decay |
| `check_classification_configuration()` | at least 2 classes indexed 0..K-1 without gaps; no sequence normalization; head settings (pooling, hidden layers, dropout and label smoothing in [0, 1), output mode, weighting, pretrained, batches of at least 2 with batch normalized layers); decorrelation; `target_precision`, `save_predictions`, error images only in multi_class mode |

The output below comes from `global_pooling: "median"` and `dropout_rate: 1.5` (the file name is a link).
```text
------------------------------------------------------
Invalid Configuration File : configuration_wrong.yaml
------------------------------------------------------
  - 'ClassificationHead.global_pooling' must be one of ['average', 'max', 'flatten'], got 'median'.
  - 'ClassificationHead.dropout_rate' must be a number within [0, 1), got 1.5.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
A lookup in `TASK_TRAINER_CLASSES`. It returns the class, not an object; `main()` builds the object with both dictionaries and the YAML path (Stage 2). An unknown task raises a `KeyError` listing the valid tasks, though Step 1.5 already catches it.

The output below shows a hit and a miss.
```text
get_trainer_class(task="classification")
-> <class 'vision_training.trainers.classification_trainer.ClassificationTrainer'>

get_trainer_class(task="segmentation")
-> KeyError: "No trainer is registered for the task 'segmentation', expected one of ['object_detection',
   'instance_segmentation', 'classification', 'semantic_segmentation', 'sequence_classification', 'sequence_labelling',
   'sequence_comparison', 'frame_selection', 'self_supervised', 'keypoint_detection']."
```

---

## Stage 2 — Trainer Initialization

> The outputs below come from building the trainer on `configurations/classification/configuration_example.yaml` when it read ImageNet-1k (1000 classes, 256 pixel images), run folder redirected to a temporary folder. Nothing is trained here. The construction took 12.3 s on CPU; a normal run on this machine prints `cuda` where the example prints `cpu`.

### Step 2.1 — `ClassificationTrainer.__init__()`
Before calling `BaseTrainer.__init__()`, the trainer stores the settings its hooks read, since the base constructor already calls `build_dataloaders()`, `build_model()` and `get_tracked_metrics()`.
- From `ClassificationHead`: `output_mode`, `class_weighting`, `decorrelation_activated` (always False for the sequence tasks sharing this trainer).
- `top_k`, fixed at 5 (`TOP_K_ACCURACY_CLASSES`).
- From `evaluation`: `target_precisions` (a list), `target_precision` (its first), `save_predictions`, `error_dumps`.
- `TASK_NAME` is `"classification"`.

The block below shows these attributes.
```text
self.output_mode                = 'multi_class'
self.class_weighting            = 'balanced'
self.decorrelation_activated    = False
self.top_k                      = 5
self.target_precisions          = [0.99]
self.target_precision           = 0.99
self.save_predictions           = False
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
```

### Step 2.2 — `BaseTrainer.__init__()`
The constructor shared by every task. It first refuses a configuration of another task, then runs, in order:
- device (2.3), run name (2.4), evaluation and tracking settings, augmenter (2.5);
- classes of a resumed run, refusal of changed labels, paths (2.6);
- loaders, the check that each gives a batch, `data.json` (2.7–2.10), sample images (2.11);
- model (2.12–2.18), pretrained components (2.19), optimizer and weight moving average (2.20), metric tracker (2.21);
- printouts (2.22–2.24), input viewer (2.25), `analyse_training_data()` (nothing for classification), checkpoint restoration (2.26–2.28).

The task-dependent steps are hooks of `ClassificationTrainer`; `start_iteration` tells the loop where to begin. The MLflow tracker is a `DisabledTracker` (does nothing) until Step 3.5, so the viewer, the speed benchmark and a standalone evaluation never create a training MLflow run.

The error and attributes below come from this constructor.
```text
ValueError: ClassificationTrainer trains the task 'classification', but the configuration is for the task 'object_detection'.

self.per_class_report       = False                 <- evaluation.per_class_report, 1000 classes are too many to list
self.number_sample_images   = 16                    <- evaluation.number_sample_images
self.experiment_tracker     = <DisabledTracker>     <- replaced at the start of the training loop
self.exponential_moving_average = None              <- Optimization.exponential_moving_average.activated is False
self.start_iteration        = 1
```

### Step 2.3 — `get_device()`
It returns CUDA if an NVIDIA GPU is visible, else MPS (Apple), else CPU. Model, batches and checkpoints go to this device. The trainer also keeps `dtype`: `MODEL_DTYPE`, `torch.float32` for every model, with mixed precision computing the forward passes in bfloat16 (Step 3.11).

### Step 2.4 — `get_run_name()` → `build_run_name()`
The run folder is `experiment_folder / <run name>`, built from the settings that define the model: task, input, backbone, head, number of classes, optimizer. The same settings give the same folder, so a stopped run resumes; an architecture change starts a new folder. Batch size, learning rate and schedule, augmentation and dropout give no token, since they may change between resumptions.

This table reads the run name of `configuration_example.yaml` token by token.

| Token | Meaning |
|---|---|
| `CLS` | task (`DET`, `INS`, `CLS`, `SEG`, `M2O`, `M2M`, `CMP`, `SEL`, `SSL`, `KPT`) |
| `IN` | `input_normalization`: pixels mapped to [-0.5, 0.5] |
| `KAR` | `keep_ratio`: aspect ratio kept, image padded to a square |
| `256` | input image size |
| `SE`, `RES`, `BN` | squeeze-and-excitation, residual shortcuts, batch normalization in the backbone |
| `B1x64.2x64.2x64.4x64.5x64.5x64` | the 6 backbone blocks, `<convolution layers>x<filters>` each |
| `HFLAT.512.256` | head: `flatten` pooling, hidden layers of 512 and 256 units |
| `HLN` | `fully_connected_normalization: "layer"` (`HBN` for `"batch"`) |
| `CWB` | `class_weighting: "balanced"` |
| `MC1000` | `multi_class` output, 1000 classes (`MC21841` with the current file) |

```text
CLS_IN-KAR-256_SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64_HFLAT.512.256-HLN-CWB_MC1000
```
There is no optimizer token (plain Adam, no weight decay) and no suffix (`run_name_suffix` null). `run_name_suffix: "v1"` would add `_v1`, so a run of the same settings with another batch size, augmentation or learning rate gets its own folder.

### Step 2.5 — `DataAugmenter()`
It stores the settings of `Data.data_augmentation`: the global `activated` flag and `probability` (chance that an image is augmented at all), and the `activated` flag and `probability` of each of the 28 types (10 geometric, 18 photometric).
- Strengths are not settings: supervised tasks use `SUPERVISED_AUGMENTATION_STRENGTHS`, the self-supervised views the stronger `SELF_SUPERVISED_AUGMENTATION_STRENGTHS` ([training reference, Data Augmentation](../reference/training_reference.md#data-augmentation)).
- The draws happen in Step 3.13. `independent_frames` and `shuffle_frames` exist only in sequence configurations.

The block below shows the settings of `configuration_example.yaml`.
```text
activated: True    probability: 0.3
types:     12 of the 28 types activated, random_left_right_flip with probability 0.5 and the 11 others with 0.25
           (random_aspect_crop, random_zoom_out, random_translation, random_up_down_flip, random_90_rotation,
           random_perspective, random_jigsaw, random_hue, random_colour_augmentation, random_gamma,
           random_linear_illumination, random_plasma_shadow, random_motion_blur, random_solarize, random_black_out and
           random_erasing are off)
expected:  0.3 x (0.5 + 8 x 0.25 + (1 - 0.75^3)) = 0.92 augmentations per image, on average (clahe_augment,
           equalize_histogram_augment and equalize_bgr_histogram_augment drawn together keep only one of them)
attributes of the augmenter: activated, probability, self_supervised (False), strengths
                             (SUPERVISED_AUGMENTATION_STRENGTHS), independent_frames, shuffle_frames_activated,
                             shuffle_frames_probability, type_settings, black_out_grid_size
```

### Step 2.6 — `setup_training_paths()`
Two checks run first, only for a resumed run whose folder holds a checkpoint.
- `restore_run_label_dictionary()`: when the mapping file gives other classes than the saved `label_dictionary.json`, the run keeps its own and says so in yellow.
- `check_resumed_data()`: a `labels.json` whose SHA-256 differs from `data.json` stops the launch with a `ValueError` before anything is replaced, unless `ExperimentConfiguration.allow_data_change_on_resume` is true. A moved data folder with the same labels resumes.

A run always resumes from its last checkpoint. To train the same settings from scratch, set another `run_name_suffix` or `experiment_folder`, or move the run folder away.

`setup_training_paths()` then returns `Tensorboard/`, `Weights/` and `training_configuration.yaml`.
- For a training it creates both folders, copies the YAML (read by Stage 5) and writes `label_dictionary.json`. With `view_input_images` or `benchmark_speed` it creates nothing.
- A resumed run compares the YAML with the saved copy (`compare_with_saved_configuration()`); changed settings are printed at Step 2.22 and the old copy is kept as `training_configuration_until_iteration_<n>.yaml`.

The tree below shows the run folder after this step.
```text
<experiment_folder>/CLS_IN-KAR-256_SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64_HFLAT.512.256-HLN-CWB_MC1000/
├── Tensorboard/
├── Weights/
├── label_dictionary.json          the class index of every category of the run, 15,785 bytes
└── training_configuration.yaml    28,187 bytes, the comments of the file included
```

### Step 2.7 — `build_dataloaders()`
This hook builds the loaders of the three splits (Step 2.8) and keeps `class_names` (one per class index, from the training manifest) for reports, error images and sample images. `BaseTrainer` stores them as `train_dataloader`, `validation_dataloader` and `test_dataloader`.
- The validation loader is built only when read: `builds_validation_loader` is True when `validation_iteration_ratio > 0` or for a diagnosis mode ending the program (`view_input_images`, `benchmark_speed`, `compute_anchor_coverage` and `view_training_targets` of the detection tasks). Otherwise the loader is `None` and no validation dataset or worker exists. The validation folder is still described and recorded in `data.json`.
- `check_loaders_give_batches()` raises a `ValueError` (split, kept samples, folder) when the training or test loader, or the validation loader with `validation_iteration_ratio > 0`, gives no batch. It also refuses a validation loader that does not follow `builds_validation_loader`.

The first class names of ImageNet-1k are its folder names (wnids).
```text
trainer.class_names[:5] -> ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475']
```

`save_data_record()` then writes `data.json` (hook `record_split()` of `BaseTrainer`, `build_class_records()` and `build_split_record()` of `vision_training/data/data_record.py`).
- `classes`: `class_index`, `class_name` (`list_class_names()`, Step 2.19), mapped `category_ids`, `images` per split. Per split: data folder, SHA-256 of `labels.json`, every `file_name`, so a model traces back to its exact images.
- A resume keeps an unchanged record; a changed one is kept as `data_until_iteration_<n>.json` before the new one is written. MLflow receives it in its `data` folder.

The start of `data.json` for ImageNet-1k is below (2.4 seconds, 69 MB for 1,271,167 training images; one class per line here).
```text
{
  "created": "2026-10-24 01:57:36",
  "classes": [
    {"class_index": 0, "class_name": "n01440764", "category_ids": [1], "images": {"train": 1290, "validation": 10, "test": 50}},
    ...
  ],
  "splits": {
    "train": {
      "data_folder": "/home/.../imagenet-1k/manifests/train",
      "labels_sha256": "973d0172c405b0f1705b5cc04a45d24ab28f676def1fb067d2d37c9e746108f4",
      "number_images": 1271167,
      "images": [
        "n01440764/n01440764_10026_n01440764.JPEG",
        ...
```

### Step 2.8 — `get_classification_dataloaders()`
For each data folder it builds a `ClassificationDataset` (Step 2.9) from `<folder>/labels.json` and `<folder>/data/`, then a loader (Step 2.10). The splits share image size, `keep_ratio`, label dictionary and worker count. With `build_validation_loader=False` the validation folder is skipped. Only the training split teaches; validation gives the Stage 3 curves, test is used by Stage 4.

The sizes of the three splits are below.
```text
train       1271167 samples -> 4966 batches of 256 (one epoch = 4966 iterations)
validation    10000 samples ->   40 batches of 256
test          50000 samples ->  196 batches of 256
```

### Step 2.9 — `ClassificationDataset.__init__()`
It reads the `labels.json` of a split and maps every category to a class index with `label_dictionary`.
- Unmapped samples are left out and counted in `number_unlabelled_samples`, printed for the three splits in one yellow block, `Samples Of Unmapped Categories, Left Out :`.
- In `multi_label` mode a sample keeps each class of its mapped categories once, ignores unmapped ones, is left out when none is mapped, and is kept as a negative of every class when it lists no category.
- The list is shuffled once and the class names are built. Each sample keeps its optional `groups` (read only by the grouped batches of the self-supervised task).
- The samples are packed as pickled records in one read-only NumPy byte buffer (`SerializedRecords` of `vision_training/data/serialized_records.py`), so forked workers copy no memory pages. `samples[index]` unpickles one; `samples.find_record()` finds one by identifier.
- No image is opened here (Step 3.9).

The example shows a manifest sample and a kept sample after mapping and shuffling.
```text
labels.json:   {'id': 1, 'file_name': 'n01440764/n01440764_10026_n01440764.JPEG', 'category_id': 1}
categories[0]: {'id': 1, 'name': 'n01440764'}
kept (shuffled order, its raw category is mapped to the class index 948, 'groups' empty since the manifest has none):
               {'identifier': 1205097, 'file_name': 'n07742313/n07742313_4294_n07742313.JPEG', 'class_indices': [948],
                'groups': {}}
```

### Step 2.10 — `build_native_loader()`
It wraps the dataset in a PyTorch `DataLoader`.
- Reshuffled every epoch; 1 persistent worker (`NUMBER_LOADER_WORKERS` of `vision_training/data/loaders.py`, one OpenCV thread) preparing 2 batches ahead (`PREFETCH_BATCHES`).
- Pinned memory only with a discrete GPU (`uses_pinned_memory()`); an integrated GPU such as the GB10 of the DGX Spark shares its memory.
- `number_workers` 0 (embedding clustering script, tests) loads in the main process.
- The training loader drops its last incomplete batch (`drop_last`) when its training needs batch statistics (`ClassificationTrainer.drops_last_training_batch()`), i.e. with `fully_connected_normalization: "batch"`: a one-image batch has no batch statistics. An epoch is then `floor(training samples / batch size)` and a training split smaller than a batch is refused. Validation and test loaders always keep it.

The training loader below was recorded with 8 workers, when this was a setting.
```text
len(loader) = 4966   batch_size = 256   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
Every evaluation draws a few fixed test images with their true and predicted classes (Step 4.7), so their predictions can be followed over training. They are chosen once, from identifiers and classes only, so a resume follows the same images.
- `get_test_sample_classes()` gives the class of every test image.
- `choose_sample_identifiers()` gives one image to each class in turn until `evaluation.number_sample_images` are chosen, spread evenly over each class's sorted identifiers. `0` turns the sample images off.

With 16 images and 1000 classes, the first 16 classes get one image each (the manifest lists 50 images per class).
```text
get_test_sample_classes() -> 50000 entries, e.g. {43211: [864], 43537: [870], 3875: [77], ...}
sample_image_identifiers  -> [1, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501, 551, 601, 651, 701, 751]
                             1 n01440764 (n01440764_ILSVRC2012_val_00000293_n01440764), 51 n01443537 (...), ...
```

### Step 2.12 — `build_model()`
This hook creates an `ImageClassifier` through `TASK_MODEL_CLASSES["classification"]` (Step 2.13). With `class_weighting: "balanced"` it stores balanced weights in the model (Step 2.18). `BaseTrainer.__init__()` moves it to the device and `dtype`. Weights are random; Steps 2.19 and 2.28 may replace them.

### Step 2.13 — `ImageClassifier.__init__()`
It builds, in order, the backbone (2.14), the pooled vector size (2.15), the pooling layer (2.16) and the head (2.17). The number of classes is the number of distinct label dictionary values. `class_weights` is a buffer of ones (saved, not learned), overwritten by Step 2.18. The forward pass is image → backbone → pooling → head → logits.

The attributes of the model are below.
```text
model.backbone               Backbone (6 convolution blocks)
model.global_pooling         Sequential(Flatten)
model.classification_head    ClassificationHead (4096 -> 512 -> 256 -> 1000)
model.class_weights          buffer [1000] torch.float32
model.number_classes         1000
```

### Step 2.14 — `build_backbone()` (→ `InputNormalizer`, `ConvolutionBlock`)
It builds the feature extractor shared by all tasks from the `Backbone` section.
- `InputNormalizer`, parameter-free: `x / 255 - 0.5` (`input_normalization`) and/or per-image standardization (`per_sample_normalization`).
- One `ConvolutionBlock` per entry of `Backbone.convolutions`: N convolutions (3×3, stride 1, padding 1), each with the normalization of `Backbone.normalization.feature_map_normalization` (batch, layer or none) and ReLU, then a 2×2 max pooling. `last_max_pooling: False` removes the last pooling.
- Residual shortcuts start at block 2, one every pair of layers, added before the ReLU closing the pair; a block changing the channel count starts its pairs at its second layer (`residual_parity` 1, see Pitfalls). Squeeze-and-excitation follows every convolution; inception modules are also available.
- `configuration_example.yaml` keeps 64 channels from block 2: one shortcut in blocks 2 and 3, two in blocks 4 to 6.
- `Backbone.modules.residual_nets_v2: True` (`RES2`) uses pre-activation ("Identity Mappings in Deep Residual Networks") in blocks with shortcuts: normalization, ReLU, then convolution, one normalization and ReLU at the end of the block. The summary says `Pre-Activation Residual Shortcuts`.

The backbone panel of `print_summary()` (Step 2.24) for a 256×256 image is below, shortened.
```text
╭─────────────────────────── Backbone Architecture ────────────────────────────╮
│ Normalization Strategy: Batch                                                │
│                                                                              │
│  - Convolution Block 1                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 3, 256, 256]                                         │
│    - Output Shape : [1, 64, 128, 128]                                        │
│  ...                                                                         │
│  - Convolution Block 6                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 64, 8, 8]                                            │
│    - Output Shape : [1, 64, 8, 8]          <- no pooling: last_max_pooling   │
╰──────────────────────────────────────────────────────────────────────────────╯
Final Backbone output shape: [1, 64, 8, 8]
```

### Step 2.15 — `compute_pooled_features()`
It gives the input size of the first head layer without running an image: the size formula of every convolution and pooling yields `C` and `H×W` of the last map. The result is `C` for `average` and `max`, `C × H × W` for `flatten`. Hence a flattened head is tied to `Data.image_settings.size` and `Backbone.last_max_pooling`.

The same backbone gives these sizes with the three poolings.
```text
final feature map [64, 8, 8]
average / max -> 64 features
flatten       -> 64 x 8 x 8 = 4096 features   (the "Input Features" of the head panel)
```

### Step 2.16 — `build_global_pooling()`
It returns the layer turning `[B, C, H, W]` into `[B, F]`: `AdaptiveAvgPool2d(1)` + `Flatten` (`F = C`), `AdaptiveMaxPool2d(1)` + `Flatten` (`F = C`), or `Flatten` alone (`F = C × H × W`, keeping where each feature was found).

### Step 2.17 — `ClassificationHead.__init__()` (→ `build_fully_connected_block()`)
`build_fully_connected_block()` stacks, for every size of `ClassificationHead.fully_connected`: Dropout (always present, an identity at rate 0, so weight names do not depend on the rate), `Linear`, the normalization of `fully_connected_normalization` (`FeatureBatchNormalization` for `"batch"`, `LayerNorm` for `"layer"`, none for `"none"`) and ReLU. A last Dropout and a `Linear` with one output per class give the logits. The sequence tasks share this head, so it works on the last dimension of its input.

The head below has `fully_connected: [512, 256]`, `"layer"`, `dropout_rate: 0.2`, 1000 classes.
```text
ClassificationHead(
  (fully_connected_block): FullyConnectedBlock(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=4096, out_features=512, bias=True)
    (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True, bias=True)
    (3): ReLU(inplace=True)
    ...                                          the same four layers for 256 units
  )
  (dropout): Dropout(p=0.2, inplace=False)
  (classifier): Linear(in_features=256, out_features=1000, bias=True)
)
```

### Step 2.18 — `compute_balanced_class_weights()`
Only with `class_weighting: "balanced"`. Weight of a class = `total samples / (number of classes × samples of the class)`: rare classes weigh more, a balanced dataset gives 1 everywhere, a class without samples counts as one. Counts come from `ClassificationDataset.get_class_counts()`; the weights are copied into `model.class_weights`.

The ImageNet-1k training split (1271167 images, 722 to 1290 per class) gives:
```text
class with 1290 images: 1271167 / (1000 x 1290) = 0.9854     <- model.class_weights[:5] = tensor([0.9854, 0.9854, ...])
class with  722 images: 1271167 / (1000 x  722) = 1.7606     <- the largest weight of model.class_weights
```

### Step 2.19 — `initialise_pretrained_components()`
This is the transfer learning step, for `Backbone` and `ClassificationHead` sections whose `pretrained.experiment_folder` is not `null`.
- It loads the other run's weights and freezes the component (or only the backbone normalization scale and shift) if asked; every parameter frozen raises an error. A resume from its own checkpoint does not reload the source.
- The source's averaged weights are read when it kept a moving average and this run has `Optimization.exponential_moving_average.activated`, its trained weights otherwise (`read_checkpoint_model_state()`).
- A backbone deeper than its source loads its first blocks only (`loaded_backbone_blocks`, from the source configuration for a new run, from `pretrained_backbone_blocks` of the last checkpoint on resume). Only these are frozen (`get_transferred_part()`).
- `ClassificationHead.pretrained.class_layer: False` (default): `keep_target_class_layers()` loads the hidden layers only, whatever the classes; the class layer stays random and is never frozen. The console prints `Classification Head : Loaded Without Its Class Layer, Which Keeps The Random Initialisation Of This Run For Its <n> Classes (The Source Run Predicts <m> Classes).`
- `class_layer: True`: `prepare_class_layers()` compares class names (`read_source_class_names()` from the source `data.json`). Every source class must exist in this run (`compare_class_names()`), as its first classes or under the same name anywhere; a missing class stops the run. `extend_class_layers()` copies the source rows to their classes (e.g. ImageNet-1k classes moved to the index of their name in the merged ImageNet); other classes keep random rows.

The output below shows a 9-class head initialising a 20-class run with `class_layer: True`.
```text
Classification Head : 9 Classes Of The Source Run Kept, 11 New Classes Added After Them, Randomly Initialised.
Classification Head Initialised From The Averaged Weights Of : model_002000.pt
                                         <- the link opens .../run_37/CLS_IN-KAR-256_..._MC9/Weights/Iteration_2000/model_002000.pt
```

### Step 2.20 — `build_optimizer()`
It creates `Optimization.optimizer` (`adam`, `adamw`, or `sgd` with Nesterov momentum 0.9) with the run learning rate and `Optimization.weight_decay`.
- Every transferred, unfrozen component gets its own parameter group at learning rate × `learning_rate_multiplier`: only the loaded blocks of a deeper backbone, only the hidden layers of a head loaded without its class layer. The group exists even at multiplier 1, so the multiplier can change on resume.
- `base_learning_rates` keeps each group's rate for the schedule (Step 3.22).
- With `Optimization.exponential_moving_average.activated`, an `ExponentialMovingAverage` (`vision_training/trainers/weight_averaging.py`) is created after the transfer: a copy of the model in `eval()` without gradients, updated after every step with `ema_weights = decay × ema_weights + (1 − decay) × current_weights`, evaluated next to the trained weights (Step 4.1).

### Step 2.21 — `get_tracked_metrics()` + `MetricTracker.__init__()`
`get_tracked_metrics()` lists the keys followed on every training and validation batch with their display names; `decorrelation_loss` and `hidden_feature_correlation` are added when `ClassificationHead.decorrelation` is on. `MetricTracker` stores them with `batch_size`, `validation_iteration_ratio`, the sample kind (`"Images"`) and the keys printed as percentages (`get_percent_tracked_metrics()`: `['accuracy']`; losses keep 4 decimals). It averages per console interval (Step 3.24), writes TensorBoard (Step 3.20), and keeps previous and best values for the changes.

### Step 2.22 — `print_experiment_information()`
It prints the launch panel so the settings can be checked before anything long starts.
- Run name and clickable folders (with a diagnosis mode, `- Run Folder Not Created  : The Diagnosis Mode Exits Without Training` in yellow); for a resume, the changed settings and data.
- One block per transferred component (`print_pretrained_components()`), one line per split (Step 2.23), iterations per epoch (`ceil`, or `floor` when the last batch is dropped), checkpoint frequencies, optimization settings, MLflow backend. Split sizes and iterations per epoch are kept for MLflow.

The panel below is the ImageNet-1k launch (folder and split names are links).
```text
--------------------------------------
=== Experiment Launch Information ===
--------------------------------------
- Run Name                : CLS_IN-KAR-256_SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64_HFLAT.512.256-HLN-CWB_MC1000
...
- Train Data Folder       : train (1271167 images, 1000 categories)
- Validation Data Folder  : validation (10000 images, 1000 categories)
- Test Data Folder        : test (50000 images, 1000 categories)
- Iterations Per Epoch    : 4966
- Batch Size              : 256
- Total Iterations        : 10000000000
- Save And Evaluate Every : 10000 iterations
- Save Only Every         : 2500 iterations
- Learning Rate           : 0.0002
...
- MLflow Tracking         : Off
```

### Step 2.23 — `describe_split()` → `count_split_samples()`
Called once per split. `count_split_samples()` reads the split's `labels.json` (`read_manifest()`, already parsed, Step 1.1) and returns its samples and categories, counted before the label mapping. The training size gives the iterations per epoch.

### Step 2.24 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
These panels describe the model before training.
- `print_summary()` (always): every backbone block traced with a dummy image (Step 2.14), then the head panel (`print_classification_head_summary()`).
- `print_receptive_field_summary()` (only with `diagnosis_modes.compute_receptive_field`): the receptive field of every layer.
- `print_model_size_summary()` (always): parameters and size of backbone, head and model.
- Then the Export Sizes panel (`print_export_size_summary()` on `estimate_export_sizes()`): the parameters `InferenceClassifier` reads and the size of every export format and of the checkpoint, with its method (a margin, `traced` for TorchScript written to a temporary `model.pt`, `range` for TensorRT). A failed probe prints `failed` in red with the error and training goes on.

The size panel of the ImageNet-1k model is below.
```text
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 746,752                       │
│   - Trainable Parameters : 746,752                       │
│   - Estimated Size       : 3.00 MB                       │
│ ...                                                      │
│ Total Model                                              │
│   - Parameters           : 3,234,280                     │
│   - Trainable Parameters : 3,234,280                     │
│   - Estimated Size       : 12.95 MB                      │
╰──────────────────────────────────────────────────────────╯
```

The Export Sizes panel below is the 35-class `configuration_synthetic_shapes.yaml` model (Adam, no moving average, CPU, 2026-11-25); the classifier exports every parameter.
```text
╭────────────────────── Export Sizes ──────────────────────╮
│ Exported Parameters : 4,637,571 of 4,637,571             │
│                                                          │
│    Format                            Size  Method        │
│    PyTorch FP32 Weights          18.60 MB  ± 1 %         │
│    PyTorch FP16 Weights           9.32 MB  ± 1 %         │
│    TorchScript FP32              18.75 MB  traced        │
│    TorchScript FP16               9.47 MB  ± 2 %         │
│    ONNX                          18.58 MB  ± 3 %         │
│    TensorRT FP32 Engine  18.84 - 19.84 MB  range         │
│    TensorRT FP16 Engine   9.57 - 10.57 MB  range         │
│    Training Checkpoint           55.76 MB  ± 1 %         │
╰────────────────────── 0.5 seconds ───────────────────────╯
```

With `diagnosis_modes.benchmark_speed`, `benchmark_speed()` then times the inference of every format and the training step on one training batch tiled to the configured and largest fitting batch sizes, and exits; no run folder is created.

### Step 2.25 — `view_input_batches()`
Only with `diagnosis_modes.view_input_images: True`. It shows what the network receives.
- Two windows, `Training` (an augmented image, Step 3.13) and `Validation`, drawn by `build_input_windows()` and shown by `show_input_samples()` → `view_input_data()`.
- A black information panel to the right of each image (`build_viewer_window()` → `build_information_panel()`) gives the class (`Class : n04456115`) and, for training, the drawn augmentations with their values (`describe_input_augmentations()`, e.g. `random_rotation : <angle> degrees` within 5 degrees either way); validation gives `Augmentations : none (validation sample)`.
- The terminal prints `Press SPACE To Navigate, Press ENTER To Stop Viewing And Exit`; without a display, `No display available (DISPLAY is not set). Skipping the input image viewer.` in yellow.
- ENTER closes the viewer and `sys.exit()` ends the program before training, with no run folder. The same windows are written at every training launch (`save_input_samples()`, Step 3.1).

### Step 2.26 — `restore_last_model()`
It always runs, since a run always resumes from its last checkpoint.
- Without a checkpoint (Step 2.27 gives 0) it returns 1: random weights, or the components of Step 2.19, which it names.
- Otherwise it loads the checkpoint (Step 2.28) and returns iteration + 1 as `start_iteration`. Training continues up to `number_iterations`, the run total: raise it to train further.
- Then `release_freed_heap_memory()` (`malloc_trim`, Linux only) gives the freed heap back before the workers fork.

The output below shows a new run, then the Stage 3 run rebuilt after its 12 iterations.
```text
-----------------------------------------------------------
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
-----------------------------------------------------------

------------------------------------------------------------
We Loaded A Model That Was Previously Saved At Iteration 12           -> start_iteration = 13
------------------------------------------------------------
```

### Step 2.27 — `get_last_checkpoint_iteration()`
It reads the first line of `Weights/full-checkpoint`, which names the last checkpoint (rewritten by Step 3.6). When the registry is missing or unreadable (empty or cut lines, announced in yellow), it takes the newest `Iteration_N` folder with a complete `model_<N on 6 digits>.pt`; 0 without any. Stage 5 uses it too.

The registry below belongs to the Stage 3 run (checkpoints at 3, 6, 9 and 12).
```text
model_checkpoint_path: "Iteration_12"          <- first line: the last checkpoint, read by this function
all_model_checkpoint_paths: "Iteration_3"
all_model_checkpoint_paths: "Iteration_6"
all_model_checkpoint_paths: "Iteration_9"
all_model_checkpoint_paths: "Iteration_12"     <- saved right after the step of iteration 12 (Step 3.6)

get_last_checkpoint_iteration(weights_directory=...) -> 12
```

### Step 2.28 — `load_model()` → `load_checkpoint()`
`load_checkpoint()` reads `Weights/Iteration_N/model_<N on 6 digits>.pt` with `torch.load`, mapped to the current device. `load_model()` puts `model_state` into the model and `optimizer_state` into the optimizer (weights and Adam moving averages). With the weight moving average on, `load_exponential_moving_average()` restores `exponential_moving_average_state`; a checkpoint without it restarts the average from the loaded weights, with a yellow message. The checkpoint also stores the task, run name, `label_dictionary` (keys as text), MLflow run and transfer sources, which tools such as the model transfer check.

The keys of a Stage 3 checkpoint are below. Checkpoints written since a deeper backbone can start from a shorter one (Step 2.19) also hold `pretrained_backbone_blocks` (`None` here).
```text
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_classification_head']
iteration = 12    format_version = 2    task = 'classification'    mlflow_run_id = '9676967287064c1784b8a7c3bee0c1dc'
pretrained_backbone = None    pretrained_classification_head = None
```

---

## Stage 3 — Training Loop

> The outputs below come from a CPU run of `configurations/classification/configuration_synthetic_shapes.yaml` on the 9-class synthetic dataset of 2026-10-09: 64 pixel images, batch size 8, 12 iterations, a checkpoint every 6 (an intermediate one every 3), a summary every 4, validation every 10, error images and MLflow on. Its run name is `CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9` (`average` pooling, plain hidden layers, no class weighting, `learning_rate: 1.0e-3`, 2.0e-4 now). The same shortening of the current file gives `CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HFLAT.128-HLN-CWB_MC9`. Step 3.9 uses ImageNet images, whose sizes show the padding better.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
This is the training itself. `run_training_loop()` runs `run_training_iterations()` and closes the MLflow run with its status: `FINISHED`, `KILLED` (Ctrl+C) or `FAILED` (too many errors, a failed save or test evaluation, too many dropped batches, or an unexpected error, raised again once the run is closed).
- The loop runs from `start_iteration` to `number_iterations`, the run total, so a resume trains only what is left. A run already at `number_iterations` trains nothing and prints in green `The Run Is Already Complete: Its Last Checkpoint Is At Iteration <n> And 'number_iterations' Is <N>, The Total Of The Run. Raise 'number_iterations' To Train It Further.`
- Each iteration, in order: fetch a batch (3.8), one training step (3.10–3.22), checkpoint if due (3.6), validation batch if due (3.23), summary if due (3.24), test evaluation if due (3.7). The checkpoint of iteration N holds the weights after N steps; the evaluation runs last so a failure finds the rest done.
- A failing iteration is printed in yellow (`report_iteration_error()`: full traceback the first time an error occurs, one line later) and the loop goes on; a NaN or infinite loss is skipped without step (3.11). More than 5 failures in a row end the loop (`FAILED`), Ctrl+C too (`KILLED`).
- A failed save (3.6), failed evaluation (3.7) or too many dropped batches (3.8) end it at once (`FAILED`, `Critical Error At Iteration <n>, Exiting: <error>` in red).

This table shows what happens at each iteration of the run (`weight_saving_iterations: 6`, `intermediate_weight_saving_iterations: 3`, `information_dump: 4`, `validation_iteration_ratio: 10`), rows in their order within the iteration.

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | ✔ | nothing: iteration 12 is saved |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |
| Test evaluation | | | | | | ✔ | | | | | | ✔ | nothing |

Before the first iteration, `save_input_samples()` (`vision_training/trainers/input_samples.py`) writes `Input_Samples/`, so the network input can be checked without a display.
- The first 10 samples by identifier of each split (`NUMBER_INPUT_SAMPLES`), the training ones augmented as in training on copies of the random states (`draw_input_split()`, `fork_random_state()`), drawn by `build_input_windows()` into `<split>/sample_<nn>.png`.
- `samples.json` gives per sample its split, file, identifier, manifest `source`, panel `labels` and `augmentations`, and `image_size`.
- A resume with unchanged configuration and data keeps the folder; otherwise the old one becomes `Input_Samples_until_iteration_<n>`. Without validation loader no `Validation/` is written.

The console line below comes from a tiny synthetic run (224 pixels, 2026-11-18).
```text
- Input Samples           : file:///…/Input_Samples (30 images written in 0.26 seconds)
```

### Step 3.2 — `setup_tensorboard_writers()`
The writers are created right before the loop, so a trainer that never trains leaves no empty log. `Train` always, `Validation` when `validation_iteration_ratio > 0`; `Test` at the first evaluation (Step 4.18).

The folders below hold these curves after the run.
```text
Tensorboard/Train/        Total Loss, Classification Loss, Accuracy, Learning Rate      (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Accuracy                     (iteration 10)
Tensorboard/Test/         created by the first evaluation (Stage 4)
Tensorboard/Test_EMA/     created by the first evaluation of the moving average of the weights, when it is on
```

### Step 3.3 — `initialize_training_loop_state()`
It prepares the loop body: an iterator over the training loader, one over the validation loader when `validation_iteration_ratio > 0` (`None` otherwise), a validation flag, the trackers of Step 3.4 (validation `None` without validation) and the consecutive error counter.

### Step 3.4 — `MetricTracker.create_trackers()`
It returns one sum per tracked metric (Step 2.21), `interval_batches` and the interval start time; one for training, one for validation. `accumulate()` fills them (3.20); `average_trackers()` divides them at the end of the interval (3.24). The sequence tasks also count `interval_frames`.

The empty tracker is below.
```text
{'start_time': 1791532505.3603294, 'interval_batches': 0, 'total_loss': 0.0, 'classification_loss': 0.0, 'accuracy': 0.0}
```

### Step 3.5 — `start_experiment_tracking()`
With `ExperimentConfiguration.tracking.activated: True`, `create_experiment_tracker()` creates the MLflow tracker after the loader iterators exist, so the workers fork before its thread starts.
- Backend: `tracking.tracking_uri`, or the local `mlflow.db` of the experiments root (artifacts in `mlartifacts/`); experiment named after the task unless `tracking.experiment_name` is set.
- Sent: flattened parameters with parameter count, split sizes, iterations per epoch, device and library versions; tags (task, run name and folder, transfer sources, host, git commit, dirty tree); the configuration; and, when this launch wrote them, `data.json` and the input samples (`Input_Samples/` or `Input_Samples_resume_<n>/`).
- Calls only queue events; a background thread writes in batches and retries, so MLflow never slows or stops training, an error printing one warning.
- `mlflow_run.json` lets a resume reopen the same MLflow run.

The line below is printed once the thread opened the run (here during the first evaluation), followed by `mlflow_run.json`.
```text
MLflow Run Created In The Experiment 'classification' : 9676967287064c1784b8a7c3bee0c1dc (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)

{
    "tracking_uri": "sqlite:///.../runs/mlflow.db",
    "experiment_id": "1",
    "run_id": "9676967287064c1784b8a7c3bee0c1dc"
}
```

The block below summarises the MLflow run after training.
```text
experiment: classification   run name: CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9   status: FINISHED
tags:       task, run_name, run_folder, pretrained_backbone: 'None', pretrained_classification_head: 'None', host,
            git_commit, git_dirty, best_iteration: '6'
parameters: 119, e.g. Backbone.convolutions.1: '[1, 16]', Backbone.convolutions.2: '[2, 32]', ...
metrics:    train/Total Loss, train/Classification Loss, train/Accuracy, train/throughput, train/interval_seconds,
            learning_rate                                         steps [4, 8, 12]     (Step 3.24)
            validation/Total Loss, validation/Classification Loss,
            validation/Accuracy                                   steps [12]           (the only interval with validation)
            test/Accuracy, test/Balanced Accuracy, test/Macro F1, test/Top-2 Accuracy,
            test/best_Accuracy, 27 test_class/<metric>/<class>    steps [6, 12]        (Step 4.19)
artifacts:  Results.txt, configuration/training_configuration.yaml, metric_names.json,
            evaluation/iteration_6/ and iteration_12/: class_metrics.csv, class_thresholds.json, confusion_matrix.png
```

This run predates the data record and the curves. The current code also sends `data/data.json`, `Input_Samples/`, `evaluation_summary.csv`, `evaluation_curves.png`, `test/Macro ROC AUC`, `test/Macro Average Precision`, 5 `test_class/<metric>/<class>` per class, `test/Top-5 Accuracy` and, per iteration, the three curve images (see the 35-class README run `d6faf784de66494f8e7f18f8d82392ed` in `~/Data/tutorial_runs/readme_screenshots/mlflow.db`).

### Step 3.6 — `save_model()` → `save_checkpoint()`
It runs right after the step of an iteration that is a multiple of `weight_saving_iterations` (checkpoint then evaluated, Step 3.7) or `intermediate_weight_saving_iterations` (checkpoint only).
- `save_checkpoint()` writes `Weights/Iteration_N/model_<N on 6 digits>.pt` with the keys of Step 2.28; the MLflow run is `None` without tracking or while the thread is still opening it.
- Non-finite weights are refused (`find_non_finite_tensors()`, `ValueError`). The file is written to a temporary name and renamed, so it is complete or absent; Ctrl+C is ignored meanwhile.
- The iteration is then recorded in `Weights/full-checkpoint` (Step 2.27). A failed save stops training.

The checkpoint of iteration N holds the weights after N steps; a resume starts at N + 1. The output and files of the run are below.
```text
Checkpoint Folder :
 - file:///.../CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9/Weights/Iteration_3
 - Model Saved In : model_000003.pt

Weights/Iteration_3/model_000003.pt     5,579,851 bytes    mlflow_run_id None        <- the MLflow run was not open yet
Weights/Iteration_6/model_000006.pt     5,579,851 bytes    mlflow_run_id None
Weights/Iteration_9/model_000009.pt     5,579,915 bytes    mlflow_run_id '9676967287064c1784b8a7c3bee0c1dc'
Weights/Iteration_12/model_000012.pt    5,579,915 bytes    mlflow_run_id '9676967287064c1784b8a7c3bee0c1dc'
Weights/full-checkpoint
```

### Step 3.7 — `run_test_evaluation()`
Last in an iteration that is a multiple of `weight_saving_iterations`, the checkpoint saved after its step is evaluated on the whole test split (Stage 4). A failing evaluation (unreadable test image, full disk) stops training, since it would fail again; the end of the loop names the checkpoint to evaluate apart (3.25). The evaluation switches to `eval()`, hence Step 3.10.

### Step 3.8 — `get_next_batch()`
It takes the next batch from the loader iterator and returns the batch and the (possibly new) iterator.
- At the end of an epoch the iterator is recreated, which reshuffles; two passes in a row without a batch raise a `RuntimeError`.
- A batch whose loading fails (Step 3.9) is dropped, counted for the next summary (3.24), and its error printed once in yellow (`Training Batch Dropped : <last line of the error>`, the first with the worker traceback).
- 10 failed fetches in a row raise a `RuntimeError`, an error of the iteration. More than 20 dropped (`MAXIMUM_DROPPED_BATCHES_IN_WINDOW`) among the last 200 fetches of a split (`DROPPED_BATCH_WINDOW`) stop the training; 1 % unreadable images in batches of 64 already drop about 47 % of batches.
- The loop then calls `MetricTracker.count_frames()`, which counts frames for the sequence tasks and does nothing for images.

The first training batch (batch size 8, 64 pixels) is below.
```text
images               [8, 3, 64, 64] torch.uint8     (BGR, values from 0 to 255)
labels               [8] torch.int64                [1, 7, 6, 8, 3, 5, 8, 2]
sample_identifiers   [8] torch.int64                [16905, 9751, 89490, 1234, 75284, 91572, 3544, 93323]
```

### Step 3.9 — `ClassificationDataset.__getitem__()` → `read_image()` + `preprocess_image()`
These run in the loader workers, once per sample, in parallel with the training step.
- `read_image()` (OpenCV, BGR, `uint8`) raises `ValueError("The image cannot be read : <file link>")` for a missing or corrupted file, so the dropped batch names it.
- `preprocess_image()` resizes the longest side to `image_size`; with `keep_ratio: True` the other side is padded with centred black bars, otherwise the image is stretched.
- The image becomes a `[3, H, W]` tensor returned with its class index and identifier; the default collate function stacks the batch.

Three ImageNet images with `image_size: 256` and `keep_ratio: True` give:
```text
n01440764_10026  original (250, 250, 3) -> (256, 256, 3)   image rows 0 to 255    (square: no padding)
n01440764_10027  original (150, 200, 3) -> (256, 256, 3)   image rows 32 to 223   (resized to 192 x 256, 32 black rows above and below)
n01440764_10029  original (375, 500, 3) -> (256, 256, 3)   image rows 32 to 223
```

### Step 3.10 — `set_training_mode()`
`model.train()` before every step: dropout on, batch normalization uses batch statistics and updates its running averages. Needed because evaluation (3.7) and validation (3.23) switch to `eval()`. The batch normalization of a frozen transferred component stays in `eval()`, keeping its source statistics.

### Step 3.11 — `run_model_iteration()`
One function runs a training or validation batch; only `apply_augmentation` differs.
- Prepare (3.12), augment for training (3.13), forward in `bfloat16` autocast (3.14–3.18).
- Every output whose key ends with `_loss` is summed into the total loss (classification, plus decorrelation when on).
- Batch metrics (3.19), accumulate and log (3.20).
- A NaN or infinite total loss raises `NonFiniteLossError` before tracking, giving every loss: the batch is skipped without backward or step, a failing iteration (3.1).
- It returns the total loss (back-propagated for training) and the outputs.

The first training batch gave:
```text
total_loss = classification_loss = 2.2124      accuracy = 0.0
```

### Step 3.12 — `prepare_batch()`
It moves the batch to the device: images to `float32` (still BGR, [0, 255]; normalization is the first backbone layer, 3.15), labels to `int64` class indices (`float` multi-hot in `multi_label`). Identifiers are kept for the evaluation.

The types before and after are below.
```text
images  [8, 3, 64, 64] torch.uint8   -> [8, 3, 64, 64] torch.float32 (cpu in this run, cuda on a GPU)
labels  [8]            torch.int64   -> [8]            torch.int64
```

### Step 3.13 — `augment_batch()` → `DataAugmenter.augment_batch()`
Training batches only; validation and test stay original.
- Each image is augmented with probability `Data.data_augmentation.probability`.
- Geometric types first (crop, aspect crop, zoom-out, translation, flips, rotations, perspective, jigsaw), then photometric (brightness, contrast, saturation, hue, CLAHE, sharpening, histogram equalizations, noise, grayscale, colour swap, gamma, linear illumination, plasma shadow, blur, motion blur, haze, solarization, black out, erasing), each with its own probability. Of the three histogram equalizations drawn together (CLAHE, luminance, BGR channels) one is kept at random.
- Labels do not change; values stay in [0, 255].
- Every type runs on the batch device from Python numbers (rotation, CLAHE and YUV conversions of torchvision and kornia ported in `vision_training/data/device_image_operations.py`).
- It also returns the report of drawn types per image, read only by the input viewer (2.25) and the input samples (3.1).

The first batch (`probability: 0.4`, 8 images) had these images changed.
```text
changed per image: [False, False, True, True, True, True, False, False]      pixel range after: 0.0 to 252.0      labels unchanged
```

### Step 3.14 — `forward_pass()` → `ImageClassifier.forward()`
It calls the model with images and labels, so the loss is returned too. Steps 3.15 to 3.18 happen inside the model. A test batch uses `predict_batch()` without labels (4.4).

The dictionary below was recorded before the forward pass always ran in bfloat16 autocast; logits now come out as `torch.bfloat16` (probabilities too on the CPU), the loss as `torch.float32`.
```text
class_logits          [8, 9] torch.float32
class_probabilities   [8, 9] torch.float32
classification_loss   2.2124 (scalar torch.float32)
```

### Step 3.15 — `Backbone.forward()`
The `InputNormalizer` maps pixels to [-0.5, 0.5] and/or standardizes each image, then every `ConvolutionBlock` runs, each pooling halving height and width. It returns the final map and a dictionary of intermediate maps, used only by the detector, the instance segmentation mask heads reading finer maps and the segmentation decoder; empty for a classifier.

A batch of 8 images of 64 pixels (last pooling removed) has these shapes.
```text
[8, 3, 64, 64] -> block 1 [8, 16, 32, 32] -> 2 [8, 32, 16, 16] -> 3 [8, 32, 8, 8] -> 4 [8, 64, 4, 4] -> 5 [8, 64, 2, 2] -> 6 [8, 128, 2, 2]
```

### Step 3.16 — `global_pooling` + `ClassificationHead.forward()`
The pooling gives `[B, F]` (2.16); the head applies its hidden layers and last `Linear`: one logit per class, any real number, not a probability.

The run of this stage (`"average"`, `fully_connected: [128]`) gives these shapes.
```text
feature map [8, 128, 2, 2] -> average pooling [8, 128] -> hidden layer (128) -> [8, 128] -> classifier -> logits [8, 9]
```
The current file (`"flatten"`, `"layer"`) gives `[8, 128, 2, 2] -> flatten [8, 512] -> Linear, LayerNorm, ReLU [8, 128] -> logits [8, 9]`, and at 512 pixels `[8, 128, 16, 16] -> flatten [8, 32768]`.

### Step 3.17 — `convert_logits_to_probabilities()`
`multi_class`: a softmax, each row summing to 1. `multi_label`: an independent sigmoid per class. Stage 4 stores and thresholds these probabilities.

The first images of the batch give:
```text
class_probabilities.sum(dim=1)[:3] = [0.9999998807907104, 1.0, 1.0]
class_logits.argmax(dim=-1)        = [2, 2, 2, 2, 2, 0, 2, 8]     <- predicted class of each of the 8 images
```

### Step 3.18 — `compute_classification_loss()`
- `multi_class`: cross entropy against the class indices, weighted by `class_weights` (2.18), softened by `label_smoothing` (`torch.nn.functional.cross_entropy`, which also weights the smoothing term, see Pitfalls).
- `multi_label`: binary cross entropy with logits against multi-hot labels smoothed towards 0.5, each class weighted by its class weight.
- A model giving every class the same probability scores ln K, the start of a training.
- With `ClassificationHead.decorrelation` on (off here), the head runs `ClassificationHead.compute_logits(kept_hidden_layer=...)`, which also returns the features of the chosen hidden layer as its ReLU receives them. `compute_decorrelation_loss()` adds `decorrelation_loss` (weight × squared off-diagonal correlations over the batch) and `hidden_feature_correlation` (mean absolute correlation, tracked by 3.19).

An untrained 9-class network gives:
```text
classification_loss = 2.2124        (ln 9 = 2.1972)
```

### Step 3.19 — `compute_batch_metrics()`
Accuracy from the logits already computed, without a second forward pass or gradients: the share of images whose highest logit is the true class (`multi_label`: the share of right class decisions at 0.5). Monitoring only; the loss is optimized. With the decorrelation on it passes on `hidden_feature_correlation`.

The batch of Step 3.8 gives:
```text
labels      = [1, 7, 6, 8, 3, 5, 8, 2]
predictions = [2, 2, 2, 2, 2, 0, 2, 8]      no image is right
accuracy    = 0 / 8 = 0.0
```

### Step 3.20 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
`accumulate()` adds every tracked metric to the interval sums and counts the batch; untracked outputs are ignored. `log_to_tensorboard()` writes the batch metrics at every iteration to the split's writer.

The console value is the mean of the TensorBoard points of the interval.
```text
Tensorboard/Train Accuracy at iterations 1 to 4:  0.0, 0.25, 0.125, 0.125      sum = 0.5      mean = 0.125
console at iteration 4:  Moving Average of Train Accuracy : 12.50 %
Tensorboard/Train Accuracy at iterations 5 to 8:  0.0, 0.0, 0.125, 0.25        sum = 0.375    mean = 0.09375
console at iteration 8:  Moving Average of Train Accuracy : 9.38 %
```

### Step 3.21 — `training_loss.backward()`
Back-propagation: the gradient of the total loss for every trainable parameter, in `parameter.grad`, after `optimizer.zero_grad()`. Training batches only; validation and test run without gradients.

The gradient norm at the first three steps:
```text
iteration 1: gradient norm = 3.09      iteration 2: 2.36      iteration 3: 2.09
```

### Step 3.22 — `step_optimizer()` → `compute_learning_rate_factor()`
- Each group's rate is set to `base_learning_rate × factor(iteration)`, then `optimizer.step()` applies the gradients as they are.
- The factor depends only on the iteration, so a resume needs no scheduler state: linear warm-up over `warmup_iterations`, then constant or a cosine decay to `minimum_learning_rate_ratio` over `decay_iterations`.
- The weight moving average, when on, moves the parameters and batch normalization running statistics with `min(decay, (1 + n) / (10 + n))` at its n-th update; other buffers are copied.
- The first group's rate goes to TensorBoard.

In this run (`constant`, `learning_rate: 1.0e-3`), the first Adam step moves each weight by about the learning rate.
```text
Tensorboard/Train Learning Rate at iterations 1 to 12 = 0.001
largest change of a weight of the classifier after the first step = 0.0010000
```

This table shows the factor of a cosine schedule (`warmup_iterations: 1000`, `decay_iterations: 100000`, `minimum_learning_rate_ratio: 0.01`).

| Iteration | 1 | 500 | 1000 | 1001 | 51000 | 101000 | 200000 |
|---|---|---|---|---|---|---|---|
| Factor | 0.001 | 0.5 | 1.0 | 1.0 | 0.505 | 0.01 | 0.01 |

### Step 3.23 — validation pass: `run_model_iteration()` without augmentation
Every `validation_iteration_ratio` iterations (10 here): `model.eval()`, no gradients, the next validation batch, `run_model_iteration()` with `apply_augmentation=False`. Metrics go to the validation trackers and `Tensorboard/Validation`. One batch only, so cheap; the whole-split evaluation is Stage 4.

Iteration 10 of the run gave:
```text
Tensorboard/Validation at iteration 10: Total Loss 2.1964, Classification Loss 2.1964, Accuracy 0.125
console at iteration 12: Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
```

### Step 3.24 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()` + `report_dropped_batches()`
Every `information_dump` iterations:
- `average_trackers()` divides the sums by the batches really run (fewer in the first interval of a resume between summaries; validation depends on the ratio).
- `log_interval_metrics()` sends to MLflow `train/<name>` and, when a validation batch ran, `validation/<name>`, with throughput, interval duration and learning rate.
- `print_interval_summary()` prints throughput and the mean of every metric, then fresh trackers start. Each value has its change since the previous summary, green when better, red when worse, and `Best` for the best of the run. Accuracy is a percentage with its change in points (`9.38 %    -3.12 %`), losses have 4 decimals; TensorBoard and MLflow keep 0 to 1.
- Counts and throughputs are grouped by three (`Total Images: 6 656 000`); the header stays 120 characters wide up to 999 999 iterations. The duration includes evaluations and checkpoints.
- `report_dropped_batches()` prints in yellow the batches dropped since the last summary (3.8), e.g. `- Dropped Batches Since The Last Summary : Training 2, Validation 1 (their loading failed, see the errors printed above)`, and nothing otherwise.

The summary of iteration 12 is below (at iteration 8 the train loss showed `-0.0154 Best` in green and the accuracy `9.38 %    -3.12 %` in red; that interval took 48 s because the evaluation of iteration 6 ran inside it).
```text
========================================================================================================================
|    Iteration: 12    |      Total Images: 96      |    Interval Images: 32    |   Throughput: 31.8 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     2.2103    +0.0208            <- red: the loss increased
Moving Average of Valid Total Loss                     :     2.1964                       <- first Valid value: no change
Moving Average of Train Classification Loss            :     2.2103    +0.0208
Moving Average of Valid Classification Loss            :     2.1964
Moving Average of Train Accuracy                       :     9.38 %    +0.00 %            <- line colour: no change
Moving Average of Valid Accuracy                       :    12.50 %
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:01.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```

### Step 3.25 — end of the loop: last checkpoint, closing
However the loop ends, it saves the last stepped iteration if not saved yet (a save failing again raises its error); iteration 12 here was already saved. It prints in green `Last Checkpoint Saved At Iteration <n>. Exiting Training.`
- Nothing is evaluated after the loop. If the last `weight_saving_iterations` checkpoint of this launch was left unevaluated (Ctrl+C between save and evaluation, or a failed evaluation), a yellow line says `The Checkpoint Of Iteration <n> Was Not Evaluated: scripts/evaluation/evaluate_checkpoint.py Evaluates It.` (Stage 5).
- The Train and Validation writers close; `run_training_loop()` stops the loader workers (`shutdown_loader_workers()`), sends `Results.txt` to MLflow and closes the MLflow run (`FINISHED` here).

The end of iteration 12 and of the loop are below, with the form of a failed iteration message.
```text
Checkpoint Folder :
 - file:///.../CLS_IN-KAR-64_..._MC9/Weights/Iteration_12
 - Model Saved In : model_000012.pt
...                                                                             <- the summary of iteration 12 (Step 3.24)
Iteration 12 · 5000 Test Images · 2026-10-09 09:53 ────────────────────────     <- its evaluation, last in the iteration
...
---------------------------------------------------------
Last Checkpoint Saved At Iteration 12. Exiting Training.
---------------------------------------------------------

Error encountered during iteration 7. Continuing... (Consecutive Errors: 1/5)      <- the form of the message of a failed iteration
```

---

## Stage 4 — Test Evaluation of a Checkpoint

> The outputs below come from evaluating the checkpoints of iterations 6 and 12 of the Stage 3 run on its whole test split (5,000 images, 9 classes), with `per_class_report: True`, `number_sample_images: 8`, `save_predictions: True`, `error_dumps` on and MLflow on. After 12 steps the network predicts one class for every image, so metrics are near chance, but every file and line is real. Iteration 6 was captured when its checkpoint was saved before its step (after 5 steps; it now holds 6, Step 3.6).
> These outputs predate the curves and the recall intervals (2026-11-01). With `per_class_report`, the current code adds `Macro ROC AUC` and `Macro Average Precision`, per-class `ROC AUC` and `Average Precision`, `roc_curves.png` and `precision_recall_curves.png` (shown with the README run in Steps 4.8, 4.14, 4.18, 4.19), `recall_lower` and `recall_upper` in `class_metrics.csv`, and a `Per Class Recall` table (Step 4.17).

### Step 4.1 — `run_test_evaluation()`
It scores the current weights on the whole test split and reports (`evaluate_test_split()`, once per kind of weights). The training loop calls it at every `weight_saving_iterations` checkpoint (3.7); Stage 5 calls it per checkpoint.
- `eval()` mode, accumulator (4.2), test loader without gradients inside the task's `evaluation_context()` (nothing here), results (4.8), report (4.18). Each test image counts once (identifiers already seen are skipped).
- It returns the results, with those of the averaged weights (else `None`), for Stage 5.
- With the weight moving average on and updated at least once, `use_averaged_weights()` swaps in the averaged model for a second evaluation: files in `Evaluation/Iteration_N/EMA/`, `Tensorboard/Test_EMA`, header `Iteration N · <evaluated samples> · EMA Weights · <date and time>`, MLflow `test_ema/`, `test_ema_class/`, `test_ema/best_<headline metric>`.

The evaluated weights of iteration N are those after N steps (3.6). The progress bar (5,000 images, batch 8, CPU) is below.
```text
Test Evaluation Iteration 6: 100%|██████████| 625/625 [00:47<00:00, 13.15it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
It creates the storage filled during the test loop, so metrics are computed on the whole split at once: four lists (probabilities, labels, identifiers, and logits filled only when the temperature explorer is written), the error image folder and an empty error count per class pair.

The accumulator below was recorded before the `logits` list.
```text
{'probabilities': [], 'labels': [], 'sample_identifiers': [],
 'errors_folder': PosixPath('.../Evaluation/Iteration_6/Errors'), 'error_counts': {}}
```

### Step 4.3 — `prepare_batch()`
The same function as Step 3.12. Test images are never augmented: the model is scored on images as a deployment gives them.

### Step 4.4 — `predict_batch()` → `ImageClassifier.forward()` without labels
`model(input_tensor=images)` as in deployment: no labels, no loss, `eval()` mode, `torch.no_grad()`. It returns logits and probabilities.

A batch of 8 images, 9 classes gives:
```text
class_logits          [8, 9] torch.float32
class_probabilities   [8, 9] torch.float32      (no 'classification_loss' key)
```

### Step 4.5 — `accumulate_evaluation_batch()`
It moves the kept samples (the `sample_indices` of 4.1) to the CPU and appends them, freeing GPU memory after every batch: probabilities as `float32` (half the memory of `float64`), labels, identifiers, and logits as `float32` with `per_class_report` (4.13). Identifiers let the predictions file and error images name each image. With `error_dumps.activated` it calls `dump_errors()` while the images are still there.

The accumulator at the end of the loop is below.
```text
'probabilities':      625 tensors, the first [8, 9] torch.float32      5,000 images / batch of 8
'labels':             625 tensors, the first [8] torch.int64
'sample_identifiers': 625 tensors, the first [8] torch.int64
'error_counts':       8 pairs, square-TO-triangle, circle-TO-triangle, ..., 20 images each
```

### Step 4.6 — `dump_errors()` → `reserve_error_image()` + `build_input_image()`
For every wrong prediction, `reserve_error_image()` counts one more image for `<TRUE>-TO-<PREDICTED>` and refuses it past `maximum_per_class_pair`, so the first ones are kept and the disk does not fill. `build_input_image()` rebuilds the network input (BGR, `uint8`), a black caption bar gives both classes and the predicted probability, and the PNG is named after the sample and that score.

The error images of iteration 6 (every image predicted `triangle`) are below.
```text
Evaluation/Iteration_6/Errors/
├── circle-TO-triangle/       20 images, e.g. test_000233_0.116.png    caption: "circle -> triangle 0.116"
├── hexagon-TO-triangle/      20 images
...                           8 pairs x 20 images = 160 images, the number in the report line "Error Images : 160 in 8 class pairs"
```

### Step 4.7 — `save_sample_images()` → `build_sample_image()`
For every fixed sample of Step 2.11 in the batch, `build_sample_image()` draws the network input with a caption bar: true class, predicted class and its probability, green when right, red when wrong (in `multi_label`, the true and predicted classes with probabilities). A small image is padded on the right to fit the bar. `save_sample_images()` writes `Evaluation/Iteration_N/Samples/<sample name>.png` and keeps it in RGB for TensorBoard tag `Samples/<sample name>`, whose slider shows each prediction over training.

The 8 samples of iteration 12 (one per class for the first 8 classes) are below.
```text
Evaluation/Iteration_12/Samples/
test_000000.png  test_000001.png  test_000002.png  test_000003.png  test_000006.png  test_000010.png  test_000016.png  test_000017.png

test_000002.png: 152 x 88 pixels, the red caption "rectangle -> triangle 0.117" on a black bar above a 64 x 64 pink image holding a small red rectangle
note of the report: "Sample Images : 8 fixed test images"
```

### Step 4.8 — `compute_evaluation_results()`
After the loop it concatenates the batches, computes every metric, writes the files (4.9–4.16) and prepares the display (4.17). It displays nothing: it returns an `EvaluationResults` dictionary for `report_evaluation()` (4.18) and Stage 5. Notes are the task's extra notes (`describe_extra_notes()`, none here) then the error images line. Files go to `Evaluation/Iteration_N/` whatever `per_class_report`, except the confusion matrix image, the three curve images, the temperature explorer and its logits, which need it. In `multi_label` mode, `compute_multi_label_metrics` replaces Steps 4.9 to 4.12: `Subset Accuracy`, `Micro F1` and `Macro F1` of the class decisions at probability 0.5, without top-k accuracy or confusion matrix.

The headline part of iteration 6 is below.
```text
evaluated_samples = '5000 Test Images'
metrics           = {'Accuracy': 0.1096, 'Balanced Accuracy': 0.1111, 'Macro F1': 0.0219, 'Top-2 Accuracy': 0.2244}
notes             = ['Error Images : 160 in 8 class pairs, at most 20 per pair', 'Sample Images : 8 fixed test images']
headline_metric   = 'Accuracy'        (used by Stage 5 and MLflow to choose the best checkpoint)
files             = Confusion Matrix Table, Class Thresholds, Class Metrics, Confusion Matrix Image, Predictions,
                    Error Images, Sample Images
images            = Confusion Matrix        sample_images = Samples/test_000010, Samples/test_000017, ... (8)
class_scalars     = 27: Precision/square, Recall/square, F1/square, Precision/circle, ...
tables            = one table: title, columns, rows, sort_column, per_class, legend
```
With `per_class_report` the current code adds `Macro ROC AUC`, `Macro Average Precision`, the three curve images and 5 scalars per class. The results also hold `percent_metrics` (the global metrics written as percentages: every headline metric of the classification) and each table's `percent_columns`; `metrics` and scalars stay between 0 and 1.

### Step 4.9 — `compute_confusion_matrix()`
It counts test images for every (true, predicted) pair: a `[K, K]` table, rows true, columns predicted, correct predictions on the diagonal, from one `bincount` over `true × K + predicted`. Multi-class mode only.
- Above 1,000 classes (`MAXIMUM_DENSE_CONFUSION_CLASSES` of `classification_metrics.py`), e.g. the 21,841 of ImageNet-22k, a dense int64 matrix would take 3.8 GB. With `sparse=True`, `torch.unique` counts the distinct values and returns a sparse `[K, K]` COO tensor of the non-zero cells, sorted by true then predicted class; out-of-range classes are refused as in the dense path.
- Up to 1,000 classes (ImageNet-1k, 8 MB) nothing changes. The k-NN evaluation of the self-supervised trainer follows the same rule.

### Step 4.10 — `compute_metrics_from_confusion_matrix()` → `compute_precision_recall_f1()`
Every multi-class metric comes from the matrix; this table gives their definitions.

| Metric | Definition |
|---|---|
| support | row sum (test images of the class) |
| precision | diagonal / column sum; 0 for a class never predicted |
| recall | diagonal / row sum |
| F1 | harmonic mean of precision and recall |
| accuracy | diagonal sum / all images |
| balanced accuracy | mean recall over the classes present in the test set |
| macro F1 | mean F1 over classes present or predicted (a predicted class without test image counts with F1 0, as in scikit-learn) |
| true positives | the diagonal, returned with the support for the recall interval (4.14) |

From a sparse matrix the sums and diagonal are added from its non-zero cells (`index_add`), giving the same metrics. The numbers of iteration 6 are obtained like this:
```text
accuracy          = 548 / 5000                           = 0.1096   (only the triangle images are right)
triangle          precision = 548 / 5000 = 0.1096        recall = 548 / 548 = 1.0      F1 = 2 x 0.1096 x 1.0 / (0.1096 + 1.0) = 0.1975
the 8 other classes   precision = 0 (never predicted)    recall = 0 / support = 0.0    F1 = 0.0
balanced accuracy = (8 x 0.0 + 1.0) / 9                  = 0.1111
macro F1          = (8 x 0.0 + 0.1975) / 9               = 0.0219
```

### Step 4.11 — `compute_top_k_accuracy()`
The share of test images whose true class is among the 5 highest probabilities (`TOP_K_ACCURACY_CLASSES` of `classification_metrics.py`, fixed for every classification-like task). It tells whether the right class is at least close. With fewer classes than k, k is clipped and the result is 1.

The output below was recorded with `top_k: 2`.
```text
Top-2 Accuracy = 0.2244 at iteration 6, 0.2276 at iteration 12      (about 2 / 9 = 0.222: chance level for 9 classes)
```

### Step 4.12 — `save_confusion_matrix_table()`
It writes the matrix as `confusion_matrix.csv` with class names as headers, at every multi-class evaluation whatever `per_class_report`. Above 1,000 classes it writes `confusion_cells.csv` instead: one row `true_class,predicted_class,count` per non-zero cell, in dense row order, counts adding up to the test images. For ImageNet-22k the dense table would take about 955 MB; the long table holds at most one row per test image (0.13 MB for 5,000 images).

The `confusion_matrix.csv` of iteration 6 is below: every image was predicted as `triangle`.
```text
true \ predicted,square,circle,triangle,rectangle,pentagon,hexagon,plus_sign,star,octagon
square,0,0,579,0,0,0,0,0,0
circle,0,0,544,0,0,0,0,0,0
triangle,0,0,548,0,0,0,0,0,0
...
octagon,0,0,590,0,0,0,0,0,0
```

The first rows of a `confusion_cells.csv` (5,000 test images of random labels over 21,842 classes named `class_<index>`) are below.
```text
true_class,predicted_class,count
class_00000,class_02897,1
class_00000,class_10297,1
class_00011,class_19444,1
```

### Step 4.13 — `find_class_thresholds()`
A deployment may trust a prediction only above a probability threshold.
- For every class, one against the rest, it sorts the test images by decreasing probability and finds the threshold whose precision stays at or above `target_precision` (0.99 here) with the highest recall; among those, the highest threshold. It also keeps the recall at that threshold. `null` when no threshold reaches the target or the class has no sample.
- Thresholds go to `class_thresholds.json` for deployment. With a list, e.g. `target_precision: [0.99, 0.95, 0.90]`, the first stays the main target (`target_precision`, `classes`); the others follow in `additional_targets`, a list of `{"target_precision": ..., "classes": {...}}`.
- Probabilities are the softmax of the logits at temperature 1, recorded as `logit_temperature`. A cross-entropy classifier is often overconfident, pushing thresholds to 0.999 or above. A temperature above 1 spreads the scores without changing the predicted class: with two classes only the threshold moves, with more the recall can move a little.

**Temperature explorer.** With `per_class_report`, `save_temperature_explorer()` writes `temperature_explorer.html`, a self-contained page.
- Its slider divides the logits by temperatures from 0.5 to 10. For each it shows, per class, the histogram of its score over its own images with a dashed line at every target threshold, and the thresholds and recalls (`Threshold @ P99 %`, `Recall @ P99 %`, recall changes from T = 1 in points, e.g. `(+0.36 points from T = 1)`).
- At most 100 classes are drawn, in class index order (the lowest F1 above 100). Up to 100 classes, `save_class_logits()` writes `class_logits.npz` next to it.

**Serving and saving.** In order to bring a temperature to deployment, run the serve script.

```bash
# Serve every evaluation of the experiments folders at http://127.0.0.1:8091/
uv run python -m scripts.evaluation.serve_temperature_explorer
```

- It serves the evaluations (`<run>/Evaluation/Iteration_<n>[/EMA]` and `<run>/Checkpoint_Evaluation/Iteration_<n>[/EMA]`) of the repository folders whose name holds `experiments`, or of those given by `--experiments_roots` (`--port` for another port than 8091). They are indexed by run, then iteration (newest first), with weights, evaluation kind, temperatures, saved temperature and exports.
- Every page is rebuilt with the current template. With `class_logits.npz` any temperature from 0.05 to 100 can be typed (`check_logit_temperature()`, `LOGIT_TEMPERATURE_RANGE`); without it, only the slider ones (`Grid only`).
- A page opens on its checkpoint's global metrics from the folder's `evaluation_summary.csv` (`describe_model_performance()`), with the best iteration by the first metric.
- With `class_logits.npz`, a score ranges table gives per class the share of its images whose own score falls between 1 to 10 typed cut points in (0, 1), at the shown temperature (`compute_score_ranges()`, route `/api/score_ranges`; default `0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99`, `DEFAULT_SCORE_CUT_POINTS`).
- **Save Thresholds At This Temperature** writes `saved_temperature_thresholds.json` next to the page, in this file's format (`build_class_thresholds_file()`), with the temperature and the thresholds of `compute_temperature_thresholds()`, checked against `class_thresholds.json` (`check_saved_thresholds()`). Opened as a plain file, the page downloads it instead; `export_model.py --use_saved_temperature` reads it once placed next to the page.
- **Export At T = …** exports the checkpoint at the saved temperature with the formats, Triton version, maximum batch size, TensorRT precision and TorchScript precision chosen on the page, in a separate process whose console the page shows; one export at a time. With a TensorRT precision, its engine is checked by a temporary Triton container, so Docker is needed ([TensorRT Check](../reference/tools_reference.md#tensorrt-check)).
- `explorer_export.py` writes `<run>/Exports/Iteration_<n>[_EMA]_T<temperature>/`, its own Triton model repository, through a `.partial` folder, with `export_log.txt` and `export_record.json`. It calls `export_experiment()` with `use_saved_temperature` and `saved_thresholds_path` (checked by `check_saved_thresholds_location()`).
- `export_experiment()` checks every format against the eager model in a staging folder and installs them only when all match (`install_staged_models()`); otherwise `The <format> Export Does Not Match The Eager Model, The Served Models Left Unchanged`.

The `class_thresholds.json` of iteration 6 is below (only `hexagon` reaches 0.99, with one image).
```text
{
    "logit_temperature": 1.0,
    "target_precision": 0.99,
    "classes": {
        "square": null,
        ...
        "hexagon": {
            "threshold": 0.11630892753601074,
            "recall": 0.0017421602787456446          <- 1 of the 574 hexagons
        },
        "plus_sign": null,
        ...
```

### Step 4.14 — `build_class_metrics_rows()` + `write_class_metrics_file()`
- **Recall intervals.** `compute_recall_intervals()` gives each class the exact (Clopper-Pearson) 95 % interval of its recall (`compute_clopper_pearson_intervals()` of `vision_training/evaluation/recall_intervals.py`), from its true positives and support; none without test image.
- **Curves** (only with `per_class_report`). `compute_class_score_curves()` gives every class, one against the rest, its ROC curve and ROC AUC and its precision-recall curve and average precision, as scikit-learn computes them. `compute_macro_curve_metrics()` averages them into `Macro ROC AUC` and `Macro Average Precision`; rows then hold `roc_auc` and `average_precision` after `f1`.
- **Rows.** `build_class_metrics_rows()` merges per class, in class index order: support, precision, recall, `recall_lower`, `recall_upper`, F1, threshold and recall at the target precision, then `threshold_at_precision_<p>` and `recall_at_precision_<p>` for every other target. With several targets, `build_class_thresholds_table()` adds the `Per Class Thresholds` table.
- `write_class_metrics_file()` writes `class_metrics.csv` (6 decimals, empty cell when missing) at every evaluation, whatever `per_class_report`, so all classes stay on disk.

The Stage 3 run (before the curves) wrote this `class_metrics.csv` at iteration 6.
```text
class_index,class,samples,precision,recall,f1,threshold_at_target_precision,recall_at_target_precision
0,square,579,0.000000,0.000000,0.000000,,
2,triangle,548,0.109600,1.000000,0.197549,,
5,hexagon,574,0.000000,0.000000,0.000000,0.116309,0.001742
...
```

The 9-class README run of 2026-10-23 wrote these first rows at iteration 1500, with the curves.
```text
class_index,class,samples,precision,recall,f1,roc_auc,average_precision,threshold_at_target_precision,recall_at_target_precision
0,square,5543,0.820935,0.621144,0.707199,0.974022,0.823801,0.894590,0.000722
1,circle,5533,0.992339,0.702331,0.822521,0.998298,0.986904,0.426778,0.776071
2,triangle,5527,1.000000,0.893975,0.944020,0.999655,0.998243,0.055343,0.985345
3,rectangle,5431,0.682449,0.859878,0.760958,0.972337,0.702261,,
...
```

### Step 4.15 — `save_confusion_matrix_image()` + `save_score_curve_images()`   (only with `per_class_report`)
`save_score_curve_images()` writes `roc_curves.png` and `precision_recall_curves.png` and returns them for TensorBoard (`ROC Curves`, `Precision Recall Curves`, 4.18) and MLflow (4.19).
- Up to 10 classes one plot, above one panel per class (10 per row, the 100 lowest above 100 classes). Precision-recall curves mark every target precision and the threshold point of 4.13.
- `precision_recall_thresholds.png` (`Precision Recall Thresholds`) zooms on these points and writes every threshold, rounded to the fewest decimals selecting the same images.

`save_confusion_matrix_image()` writes `confusion_matrix.png` and returns it for TensorBoard.
- Each cell shows its count and share of the true class (white to dark blue); a right column gives recall, a bottom row precision (grey when never predicted), the corner accuracy, all as percentages.
- Counts and shares are shown up to 20 classes, counts up to 40, names up to 60. Above 1,000 classes it is not drawn (sparse matrix, 3.8 GB of dense float64 otherwise).

The `confusion_matrix.png` of iteration 12 (1470 x 1432 pixels with the current cells) reads:
```text
title:    Confusion Matrix · Iteration 12 · Accuracy 10.96 % · Balanced Accuracy 11.11 %
matrix:   9 x 9 cells, rows = true class, columns = predicted class
          column triangle: dark blue cells "579 / 100.00 %", "544 / 100.00 %", ... (every image was predicted as triangle)
          diagonal cells of the other classes: "0 / 0.00 %" in bold; the other zeros are left blank
right:    Recall column: 0.00 % for the 8 other classes, 100.00 % for triangle
bottom:   Precision row: grey for the 8 classes that are never predicted, 10.96 % for triangle
corner:   Accuracy 10.96 %
colour bar: 0 % to 100 %
```

### Step 4.16 — `write_predictions_file()`   (only with `save_predictions`)
It writes `predictions.csv`, one row per test image: file name (`describe_sample()`), true and predicted classes, correctness, probability of every class. Errors and thresholds can then be studied without the model. Its size is test images × classes, hence optional.

The first rows of iteration 6 (5,000 rows, 585,792 bytes, columns shortened) are below.
```text
sample,true_class,predicted_class,correct,probability_square,probability_circle,probability_triangle, ... ,probability_octagon
test_000612.png,triangle,triangle,1,0.111758,0.105057,0.116420, ... ,0.115801
test_000662.png,plus_sign,triangle,0,0.111758,0.105057,0.116420, ... ,0.115801
```
The current code writes every probability with nine significant digits, so that thresholds computed again from the file are those of `class_thresholds.json`.

### Step 4.17 — `build_class_metrics_table()` + `build_class_metric_scalars()`
These prepare the display without displaying. The reporter shows their outputs only with `per_class_report`.
- `build_class_metrics_table()`: the per-class table (with `per_class_report`, `ROC AUC` and `Average Precision` after F1), sorted by F1, worst first, ties in class order, a class without sample last with a dash. With `in_percent`, its `percent_columns` (all but class, samples and threshold) print as percentages and the target as `99 %`; the rows keep 0 to 1 so the sort is unchanged.
- `build_recall_interval_table()` (`recall_intervals.py`): the `Per Class Recall` table, after the per-class table and before `Per Class Thresholds`. Class, samples, recall, `95 % Lower`, `95 % Upper` as percentages, from `class_metrics.csv`, worst recall first, with a legend naming the interval.
- `build_class_metric_scalars()`: one TensorBoard scalar per class and metric, plus `ROC AUC/<class>` and `Average Precision/<class>` with `per_class_report`.
The 9 classes of the Stage 3 run give 27 scalars (3 metrics); with the curves, the 9-class README run writes 45 (9 x 5), the 35-class README run 175.

### Step 4.18 — `report_evaluation()` → `report_evaluation_results()`
The results go to four places.
- `Tensorboard/Test`: global scalars, sample images, and with `per_class_report` the per-class scalars and images.
- `Results.txt` and the console: the report block (header, global metrics, notes, tables), percentages by `format_percentage()` of `report_formatting.py`, then in the console the file links. With `per_class_report: False` the block stops after the global metrics and notes.
- MLflow (4.19). Then `update_evaluation_curves()` writes the iteration's row of `Evaluation/evaluation_summary.csv`, redraws `evaluation_curves.png` next to `Results.txt`, prints its link and sends both to MLflow ([training reference, Run Folder](../reference/training_reference.md#run-folder)).

The block of iteration 2000 of the README run (2026-11-19, error images off, tables cut) is below; the same text, without the links, goes to `Results.txt`. The CPU run's blocks show `Top-2 Accuracy` and an `Error Images : 160 in 8 class pairs, at most 20 per pair` note.
```text
Iteration 2000 · 50000 Test Images · 2026-11-19 20:20 ──────────────────────────────────────────────────────────────────
  Accuracy        81.76 %   Balanced Accuracy         81.65 %
  Macro F1        81.11 %   Top-5 Accuracy            99.12 %
  Macro ROC AUC   99.42 %   Macro Average Precision   86.51 %
  Sample Images : 8 fixed test images

  Per Class Metrics
                                                                                 Average   Threshold    Recall
   Class                  Samples   Precision    Recall        F1    ROC AUC   Precision     @ P99 %   @ P99 %
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
   trapezoid                 1387     38.52 %   17.66 %   24.22 %    97.11 %     38.31 %           —         —
   rhombus                   1425     52.00 %   36.49 %   42.89 %    97.86 %     50.69 %      0.7118    0.14 %
   ...                     (35 rows, worst F1 first)
  Worst F1 first, ties in class order. — : no samples of the class, no negative for the ROC AUC, or the precision 99 %
  is never reached.

  Per Class Recall
                                                 95 %      95 %
   Class                  Samples    Recall     Lower     Upper
  ──────────────────────────────────────────────────────────────
   trapezoid                 1387   17.66 %   15.69 %   19.77 %
   rhombus                   1425   36.49 %   33.99 %   39.05 %
   ...                     (35 rows, worst recall first)
  Worst Recall first, ties in class order. 95 % Lower and Upper : exact (Clopper-Pearson) interval, the counted samples
  being independent. — : no samples of the class.
  - Results : Results.txt
  - Confusion Matrix Table : confusion_matrix.csv
  ...                                (one link per file of the evaluation)
  - Evaluation Curves : evaluation_curves.png
```

The block below shows the run folder and the Test curves after the two evaluations of the Stage 3 run.
```text
Results.txt                                   one block per evaluation (iterations 6 and 12)
Evaluation/Iteration_6/                       Errors/  Samples/  class_metrics.csv  class_thresholds.json  confusion_matrix.csv
                                              confusion_matrix.png  predictions.csv
Evaluation/Iteration_12/                      the same files for the checkpoint of iteration 12

Tensorboard/Test: 31 scalars (4 global + 27 per class), 9 images
    scalar Accuracy          6: 0.1096, 12: 0.1096
    scalar Top-2 Accuracy    6: 0.2244, 12: 0.2276
    ...
    image  Samples/test_000010 ... Samples/test_000006    steps [6, 12]     (the 8 sample images)
    image  Confusion Matrix                               steps [6, 12]
```
The 9-class README run holds 51 scalars (6 global + 45 per class) and 11 images (8 samples, `Confusion Matrix`, `ROC Curves`, `Precision Recall Curves`) at steps 500, 1000 and 1500. The current code also writes the `Precision Recall Thresholds` image, `Evaluation/evaluation_summary.csv` and `evaluation_curves.png`.

### Step 4.19 — `track_evaluation()`
With MLflow on, it sends at the checkpoint iteration:
- global scalars as `test/<name>`; per-class scalars as `test_class/<name>` only with `per_class_report` (thousands for 1000 classes otherwise);
- the best headline value so far as `test/best_<metric>` with tag `best_iteration` (`test_ema/best_<metric>` and `best_iteration_test_ema` for averaged weights);
- the small files (`TRACKED_EVALUATION_FILES` of `base_trainer.py`: `class_metrics.csv`, `class_thresholds.json`, and with `per_class_report` `confusion_matrix.png`, `roc_curves.png`, `precision_recall_curves.png`, `precision_recall_thresholds.png`) under `evaluation/iteration_<n>/` (`ema/` below it for averaged weights).

Predictions, error and sample images stay in the run folder only. Without tracking it does nothing. What MLflow received at iteration 6 (before the curves) is below.
```text
step 6: {'test/Accuracy': 0.1096, 'test/Balanced Accuracy': 0.1111, 'test/Macro F1': 0.0219, 'test/Top-2 Accuracy': 0.2244,
         'test_class/Precision/square': 0.0, 'test_class/Recall/square': 0.0, ... (27 values)}
test/best_Accuracy: step 6 -> 0.1096, step 12 -> 0.1096      tag best_iteration = '6' (iteration 12 is not better)
artifacts: evaluation/iteration_6/class_metrics.csv, class_thresholds.json, confusion_matrix.png
```

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The outputs below come from this script on the Stage 3 run (checkpoints at 3, 6, 9 and 12, captured when 3, 6 and 9 were saved before their step), with `--iteration all` and a 300-image test folder taken from the test split (`--test_data_folder`). They predate the recall intervals (2026-11-01: the block of Step 5.6 now adds the `Per Class Recall` table, `class_metrics.csv` the `recall_lower` and `recall_upper` columns) and the folder per split (2026-11-20: this 300-image folder would now go to `Checkpoint_Evaluation_split_<parent folder>_test_300/`).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
If one wishes to compare checkpoints, pick one to deploy, or score a run on another labelled split, this script evaluates checkpoints of a trained (or training) run without training.
- Its settings are command line arguments (`parse_evaluation_arguments()`, passed to `evaluate_run()`); without any, it prints its help.
- The run's files are only read. Everything goes to `<run>/Checkpoint_Evaluation/`, next to the training's `Evaluation/`, or for another labelled split to `Checkpoint_Evaluation_split_<parent folder>_<split folder>/` (e.g. `Checkpoint_Evaluation_split_synthetic_shapes_validation`), so two splits never mix.

```bash
# Evaluate every checkpoint of a run, the last saved first
uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder <run folder> --iteration all
```

This table lists the arguments of the script; only `--experiment_folder` is required.

| Argument | Role |
|---|---|
| `--experiment_folder` | the trained run folder, with `Weights/` and `training_configuration.yaml` (required) |
| `--iteration` | `all`: every checkpoint of the registry, last first; an integer: one checkpoint; the last one by default |
| `--test_data_folder` | another labelled split instead of the run's test split (the default) |
| `--batch_size` | evaluation batch size, `128` by default (no gradients, so it can exceed the training one); `None` keeps the run's |
| `--track_evaluation` | `True`/`False` for an MLflow run of its own; `tracking.activated` by default |
| `--save_predictions` | `True`/`False` for `predictions.csv` per iteration (4.16), read by the seen and never seen groups chart of `draw_report_figures.py` (5.7); `evaluation.save_predictions` by default (`apply_save_predictions()`) |
| `--maximum_error_images_per_class_pair` | turns error images on with at most this many per class pair, the first met, `100000` by default (a number above the test samples saves every error); `None` keeps `evaluation.error_dumps` |
| `--target_precisions` | precision of the class thresholds, or several distinct precisions whose first is the main target (e.g. `0.99 0.95 0.90`, 4.13), checked like the configuration (`apply_target_precisions()`); `evaluation.target_precision` by default |
| `--nms_criterion`, `--nms_distance_exponent` | final non-maximum suppression of the detection tasks; skipped with a warning for a classifier |

### Step 5.2 — `load_experiment_configuration()` + `get_trainer_class()` + trainer construction (same as Stages 1–2)
- It loads the run's saved `training_configuration.yaml` (2.6) through the Stage 1 loader, classes from the `label_dictionary.json` next to it.
- When given, it sets `test_data_folder` (`apply_test_data_folder()`, which gives the `_split_<parent folder>_<split folder>` suffix for another split), `batch_size`, `tracking.activated`, `evaluation.save_predictions`, `evaluation.error_dumps` (on with `maximum_error_images_per_class_pair`) and `evaluation.target_precision` (`target_precisions`).
- `point_to_checkpoint_evaluation()` sets `experiment_folder` to the folder holding the run, `run_name` to the evaluated folder name (a built name would lose its `PT` tokens) and `standalone_evaluation_folder` to `Checkpoint_Evaluation`. No `Weights/` is created.
- `check_evaluation_folder_split()` raises a `ValueError` when the folder's `data.json` records another test split. `disable_pretrained_settings()` turns off the transfer.
- It builds the Stage 2 trainer, which writes the configuration copy, `data.json` and `label_dictionary.json` into the folder.

The construction prints the launch panel of Step 2.22 with the evaluation folder. No weights are loaded here (Step 5.5 does it).
```text
tracking in the saved configuration: {'activated': True, 'tracking_uri': None, 'experiment_name': None}
- Run Name                : CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9
- Experiment Folder       : file:///.../runs/CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9/Checkpoint_Evaluation
...
- Test Data Folder        : file:///.../test_300 (300 images, 9 categories)
- Batch Size              : 128
...
- MLflow Tracking         : sqlite:///.../runs/mlflow.db
-----------------------------------------------------------
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.
-----------------------------------------------------------
```
`MLflow Tracking` names the trained run's database: `tracking_root` keeps the experiments root of the run's configuration, so a moved or archived run folder is still evaluated in its training database.

### Step 5.3 — `get_checkpoint_iterations()` / `get_last_checkpoint_iteration()`
`list_evaluated_iterations()` calls them on the registry `Weights/full-checkpoint` (2.27) before the configuration is loaded, so nothing is written for a run that cannot be evaluated.
- `"all"`: `get_checkpoint_iterations()` lists every recorded iteration, intermediate ones included, reversed so the most trained weights come first.
- `None`: `get_last_checkpoint_iteration()`. An integer is used as given.
- When none of the iterations has its weight file (a run stopped before its first save, an integer without checkpoint), a `FileNotFoundError` names the `Weights/` folder and the registry iterations.

The Stage 3 registry gives:
```text
get_checkpoint_iterations(weights_directory=...)      -> [3, 6, 9, 12]
get_last_checkpoint_iteration(weights_directory=...)  -> 12
iteration = "all"  -> evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
With tracking on, `start_evaluation_tracking()` creates an MLflow run `<run name>-evaluation` in the trained run's backend and experiment, tagged with the evaluated folder, split, iterations and training run (`evaluation_of`, from `mlflow_run.json`). The training MLflow run is never modified. `evaluate_checkpoints()` runs Step 5.5 and closes the run with its status.

The MLflow run of the evaluation is below.
```text
MLflow Run Created In The Experiment 'classification' : 6b1db68d415d41d692a29f7b87fd4b69 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)

run name: CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9-evaluation   status: FINISHED
tags:     evaluated_experiment_folder, test_data_folder: '.../test_300', evaluated_iterations: '12, 9, 6, 3',
          evaluation_of: '9676967287064c1784b8a7c3bee0c1dc', best_iteration: '9', ...
metrics:  test/Accuracy, test/Balanced Accuracy, test/Macro F1, test/Top-2 Accuracy, test/best_Accuracy,
          27 test_class/<metric>/<class>                                             steps [12, 9, 6, 3]
artifacts: evaluation/iteration_12/, iteration_9/, iteration_6/, iteration_3/ (3 files each), evaluation_summary.csv,
           metric_names.json
```
`test/best_Accuracy` is the best so far in evaluation order (0.1000 at step 12, then 0.1133). MLflow shows a metric's value at the highest step, so its runs table gives 0.1000 while `best_iteration` is 9.

### Step 5.5 — `summarise_checkpoint_evaluations()`
For each iteration, in order:
- It loads `Weights/Iteration_N/model_<N on 6 digits>.pt` into the trainer's model. A missing file is skipped with a warning; when every file is missing, a `FileNotFoundError` closes the MLflow run as failed.
- It evaluates (5.6) and keeps the headline metric.
- Each evaluation adds its row to `Checkpoint_Evaluation/evaluation_summary.csv` (rows in increasing iteration order, per-class columns only with `per_class_report`) and redraws `evaluation_curves.png` (`BaseTrainer.update_evaluation_curves`, [training reference, Run Folder](../reference/training_reference.md#run-folder)).

At the end it prints the summary, the curves and the best iteration by headline metric, ties listed (`Best Checkpoint : iterations 12, 9 tie (Accuracy <value>)`). With the weight moving average on, `ema_<scalar>` columns and `ema_number_updates` follow, and a line `Best Checkpoint (EMA Weights) : iteration <n> (Accuracy <value>)` is printed.

The output after the four evaluations is below.
```text
Evaluation Summary Of 4 Checkpoints : file:///.../runs/CLS_IN-KAR-64_..._MC9/Checkpoint_Evaluation/evaluation_summary.csv
  - Evaluation Curves : file:///.../runs/CLS_IN-KAR-64_..._MC9/Checkpoint_Evaluation/evaluation_curves.png
  - Best Checkpoint : iteration 9 (Accuracy 11.33 %)

evaluation_summary.csv (32 columns: iteration, the 4 global scalars, the 27 per-class scalars), rows in increasing
iteration order:
iteration,Accuracy,Balanced Accuracy,Macro F1,Top-2 Accuracy,Precision/square,Recall/square,F1/square, ...
3,0.09,0.1111111111111111,0.018348623853211007,0.19,0.0,0.0,0.0, ...
6,0.1,0.1111111111111111,0.020202020202020204,0.24666666666666667,0.0,0.0,0.0, ...
9,0.11333333333333333,0.1111111111111111,0.02262142381902861,0.21333333333333335,0.0,0.0,0.0, ...
12,0.1,0.1111111111111111,0.020202020202020204,0.21333333333333335,0.0,0.0,0.0, ...
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
The Stage 4 function, so the console and `Results.txt` block, `Tensorboard/Test`, sample images and `Iteration_N/` files are the same as in training, only in `<run>/Checkpoint_Evaluation/` and tracked in the evaluation MLflow run.
- Another split gets `<run>/Checkpoint_Evaluation_split_<parent folder>_<split folder>/` (`apply_test_data_folder()`); a folder holding another split is refused (`check_evaluation_folder_split()`).
- Evaluating the same split again reuses its folder: new `Results.txt` blocks, rewritten `Iteration_N` files, replaced error images.
- With `per_class_report`, a classifier also gets `temperature_explorer.html` and `class_logits.npz`, listed by `serve_temperature_explorer.py` (4.13). Thresholds saved there land in this `Iteration_N` folder, one of the two places `export_model.py` searches with `use_saved_temperature` (`find_saved_thresholds_path()`, which takes the file of `Checkpoint_Evaluation/` with a yellow line when `Evaluation/` holds one for the checkpoint too).

The block of iteration 9 on the 300 images and the folder tree below predate the curves and the temperature explorer.
```text
----------------------------------------------------------------------------------------------------------
Evaluating : file:///.../runs/CLS_IN-KAR-64_..._MC9/Weights/Iteration_9/model_000009.pt
----------------------------------------------------------------------------------------------------------
Iteration 9 · 300 Test Images · 2026-10-09 09:53 ─────────────────────────────────────────────────
  Accuracy   11.33 %   Balanced Accuracy   11.11 %
  Macro F1   2.26 %    Top-2 Accuracy      21.33 %
  Error Images : 160 in 8 class pairs, at most 20 per pair
  Sample Images : 8 fixed test images
  ...

CLS_IN-KAR-64_..._MC9/Checkpoint_Evaluation/
├── Iteration_12/, Iteration_9/, Iteration_6/ and Iteration_3/
│                                                  Errors/  Samples/  class_metrics.csv  class_thresholds.json
│                                                  confusion_matrix.csv  confusion_matrix.png  predictions.csv
├── evaluation_summary.csv
├── Results.txt                                    the four blocks
├── Tensorboard/Test/
├── data.json                                      the record of the data of the evaluation (Step 2.7), its test
│                                                  split being the evaluated one
├── label_dictionary.json                          the classes of the trained run
└── training_configuration.yaml                    copy of the configuration of the trained run
```

### Step 5.7 — `scripts/evaluation/draw_report_figures.py`
Should a research report of a kept classifier need its figures again (a new evaluation, a new saved temperature), this script draws them into `Report_Figures/` of the evaluation folder of the kept iteration and weights (`--weights`). `--experiment_folder` and `--iteration` are required; the other arguments are in the tools reference (Drawing The Figures Of A Research Report) and in its help, printed without any. The folder is `<run>/Checkpoint_Evaluation/Iteration_<n>[/EMA]/` by default (`--evaluation_folder_name Evaluation` for the training's).

```bash
# Draw the report figures of the trained weights of iteration 2000
uv run python -m scripts.evaluation.draw_report_figures --experiment_folder <run> --iteration 2000
```

- `draw_report_figures()` checks every input before writing: `class_logits.npz` (one class per sample), `confusion_matrix.csv` with the same classes, the thresholds, and the balanced accuracy of trained and EMA weights from `Tensorboard/Test` and `Tensorboard/Test_EMA`.
- `read_report_thresholds()` takes `saved_temperature_thresholds.json` (it must match the thresholds recomputed from the logits), or the thresholds at temperature 1 with a yellow warning.
- It writes `confusion_matrix.png` (`save_confusion_matrix_image()`, accuracy and balanced accuracy in percent in the title), `precision_recall_curves.png` (`draw_precision_recall_figure()`, curves at the thresholds' temperature with the threshold of every target precision), `training_curves.png` (kept checkpoint ringed) and `report_values.json` (every number shown).
- Given a JSON Lines manifest of every sample's group (`--group_manifest_path`, e.g. the truck of every image), it reads the evaluation's `predictions.csv` (`save_predictions`, 5.1) and the training `labels.json`, and draws `seen_and_never_seen_groups.png`: balanced accuracy on groups seen and never seen in training (`compare_seen_and_never_seen_groups()`).

### Step 5.8 — `scripts/evaluation/score_image_split.py`
In case a kept classifier must find the images of a rare class among a million unlabelled ones, this script scores every image of a split with one checkpoint, without metrics, error images or second weights. Its command line options are in the [tools reference, Scoring A Large Split](../reference/tools_reference.md#scoring-a-large-split).
- Forward pass only, of the trained weights or their moving average (`--use_averaged_weights`), in `float32` or `bfloat16` autocast (`--precision`; `--compare_precisions` first compares both on the run's labelled test split).
- Images are decoded by many loader workers through the run's test dataset, in the order of the split's `labels.json`.
- Batch by batch it appends the `sample` and `probability_<class>` columns of `predictions.csv` (4.16) to `--output_path`, so a launch stopped by `--time_budget_hours` or interrupted resumes on the same file.
- Its trainer writes its configuration copy, `label_dictionary.json`, `data.json` and `Tensorboard/` into `<run>/Checkpoint_Evaluation_split_<parent folder>_<split folder>/`, so no file of the run or its evaluations changes.

```bash
# Score a split with the checkpoint of iteration 2000 in bfloat16, for at most 3 hours
uv run python -m scripts.evaluation.score_image_split --experiment_folder <run> --iteration 2000 \
    --test_data_folder <split> --output_path <predictions.csv> --precision bfloat16 --time_budget_hours 3
```
