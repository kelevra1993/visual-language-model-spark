# Self-Supervised Workflow

This document follows one self-supervised run from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`; here they keep a short explanation and the real output of the self-supervised run, and the steps specific to the self-supervised task (the two random views, the projection head, the InfoNCE and Barlow Twins losses, the k-nearest neighbour evaluation and the effective rank) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for other trainings:
- **Stages 1 and 2**: `configurations/self_supervised/configuration_synthetic_shapes.yaml` (synthetic shapes, 9 shape classes, 256 pixel images, batch size 128, SimCLR), with the run folder redirected to a temporary folder and 2 loading workers.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel images, batch size 8, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, the first 1000 images of the test split as test split and the MLflow tracking turned on, the random generators of Python, NumPy and PyTorch seeded with 0. A second run of 4 iterations, identical except for `ProjectionHead.method: barlow_twins` and without MLflow tracking, gives the Barlow Twins outputs of Stage 3, and Stage 5 evaluates the checkpoints on the first 200 test images. After 12 iterations the representations are almost random, so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

The outputs of Stages 1 and 2 were computed again on 2026-10-24 with the current code, and those of Steps 1.2 and 2.1 on 2026-11-12, once the loader added the key of the standalone evaluation folder and the trainer chose the reference images of the k-NN, except the grouped batches examples of Steps 2.8 and 2.10, checked against the code only, and the first kept sample of Step 2.9, a draw of the unseeded shuffle of the manifest; those of Stages 3 to 5 come from the runs of 2026-10-23. There is no run of this task in `experiments/readme_screenshots/`, and the GPU run of the tutorial videos (`tutorials/run_configurations/self_supervised_synthetic_shapes_tutorial.yaml`, 20,000 iterations on the 9-class dataset generated on 2026-10-11, trained again from 2026-11-15 to 2026-11-16 into `~/Data/tutorial_runs/tutorials/`, outside the repository) is not used by this document. A retrain of that run at 512 pixels on the 35 classes of the dataset generated on 2026-11-18 was stopped on 2026-11-19 at about 11,800 iterations, with a k-NN accuracy of 14.19 % at 10,000, and the videos keep the run of 2026-11-15.

## Read This First

The self-supervised task (`task: "self_supervised"`, run name code `SSL`) trains the backbone without labels (one backbone design for every task). Every training and validation image is drawn twice by the data augmentation (`Data.data_augmentation`, with the hard-coded `SELF_SUPERVISED_AUGMENTATION_STRENGTHS`); both views go through the backbone, a global pooling giving the **representation** of the image and a projection head giving its **embedding**, and the loss of `ProjectionHead.method` pulls the two views of an image together: InfoNCE over the 2N views of the batch (`simclr`) or the Barlow Twins cross-correlation pushed to the identity (`barlow_twins`). The labels of the `labels.json` manifests are only read by the test evaluation, a k-nearest neighbour classification of the labelled test images among a fixed sample of labelled training images by their representations, next to the effective rank that watches the collapse. Choose it to pre-train the backbone on images too many to label, then initialise any other task with `Backbone.pretrained` (the projection head is dropped); choose CLS when the images are labelled. The task reuses the dataset layout, the dataset and the loaders of CLS, and its metrics code for the k-NN scores; the exported encoder (`image_encoder`) returns L2-normalized representations for similarity searches.

### The Task At A Glance

To find your way in the stages below, this block follows an image of the synthetic configuration from its manifest to its two views, the loss, the k-NN evaluation and the exported encoder.

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
| Trainer | `vision_training/trainers/self_supervised_trainer.py` | `SelfSupervisedTrainer`: `build_dataloaders`, `build_reference_dataloader`, `forward_pass`, `predict_batch`, `compute_batch_metrics`, `encode_reference_images`, `compute_nearest_neighbour_results`, `compute_evaluation_results`, `build_sample_image`, `augment_input_batch`, `build_input_windows`, `show_input_samples`, `show_input_batch` |
| Shared training loop | `vision_training/trainers/base_trainer.py` | `BaseTrainer`: `run_training_iterations`, `run_model_iteration`, `save_model`, `run_test_evaluation`, `evaluate_test_split` |
| Dataset and loaders | `vision_training/data/classification/classification_dataset.py`, `vision_training/data/serialized_records.py`, `vision_training/data/loaders.py`, `vision_training/data/grouped_batch_sampler.py` | `ClassificationDataset(keep_unlabelled_samples=True)`, `ClassificationDataset.select_samples`, `SerializedRecords` (`find_record`), `get_classification_dataloaders(self_supervised=True)`, `GroupedBatchSampler`, `build_grouped_batch_sampler` |
| Views | `vision_training/data/augmentations.py` | `DataAugmenter.create_views`, `describe_report`, `SELF_SUPERVISED_AUGMENTATION_STRENGTHS`, `SECOND_VIEW_ONLY_TYPES` |
| Model | `vision_training/models/self_supervised/self_supervised_model.py`, `vision_training/models/backbones/backbone.py`, `vision_training/models/classification/classification_head.py` | `SelfSupervisedModel` (`represent`, `encode`, `compute_method_loss`, `forward`), `build_backbone`, `build_global_pooling`, `compute_pooled_features` |
| Head | `vision_training/models/self_supervised/projection_head.py`, `vision_training/models/components/fully_connected_block.py` | `ProjectionHead`, `print_projection_head_summary`, `build_fully_connected_block` |
| Losses | `vision_training/losses/self_supervised_losses.py` | `compute_info_nce_loss`, `compute_barlow_twins_loss`, `compute_feature_cross_correlation`, `compute_embedding_standard_deviation` |
| Metrics and evaluation | `vision_training/evaluation/self_supervised/representation_metrics.py`, `vision_training/evaluation/classification/`, `vision_training/evaluation/recall_intervals.py`, `vision_training/evaluation/evaluation_report.py` | `select_reference_samples`, `compute_nearest_neighbour_predictions`, `compute_effective_rank`, `compute_confusion_matrix`, `compute_metrics_from_confusion_matrix`, `build_class_metrics_table`, `compute_clopper_pearson_intervals`, `build_recall_interval_table`, `write_class_metrics_file` |
| Configuration checks | `vision_training/configuration_processing/self_supervised_checks.py` | `check_self_supervised_configuration`, `check_projection_head_configuration`, `check_grouped_batches_configuration` |
| Run name | `vision_training/configuration_processing/run_name.py` | `build_run_name`, `build_self_supervised_sections` |
| Synthetic data generator | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset`, `build_classification_labels` (this configuration), `build_shape_count_labels` (the counted shapes) |
| Export | `vision_training/models/self_supervised/inference_encoder.py`, `vision_training/deployment/export_specifications.py`, `vision_training/deployment/export_model.py`, `vision_training/deployment/input_contract.py` | `InferenceEncoder` (model name `image_encoder`, input `images`, output `representations`), `compare_representations`, `export_experiment`, `build_input_contract` (the input contract of the run, in the ONNX metadata and `preprocessing.json` next to every exported model) |
| Export sizes at launch | `vision_training/deployment/export_size_estimates.py` | `estimate_export_sizes`, `probe_inference_graph`, `print_export_size_summary` |
| Standalone evaluation | `scripts/evaluation/evaluate_checkpoint.py`, `vision_training/evaluation/checkpoint_evaluation.py` | `evaluate_checkpoints`, `summarise_checkpoint_evaluations` |

### Run It

In order to train and evaluate the task on the synthetic shapes from scratch, run these commands from the root of the repository, after setting the values of their main blocks and of the configuration.

```bash
# 1. Synthetic dataset: in the main block of vision_training/data/synthetic/create_synthetic_dataset.py, set
#    task = "classification" and output_folders["classification"] (500,000 / 50,000 / 50,000 images of 512 pixels),
#    the dataset of configuration_synthetic_shapes.yaml; task = "self_supervised" creates the counted shapes of
#    configuration_counted_shapes_simclr.yaml and configuration_counted_shapes_barlow_twins.yaml
uv run python -m vision_training.data.synthetic.create_synthetic_dataset

# 2. Training: set experiment_folder, the three data folders and Data.label_mapping.mapping_file_path first;
#    diagnosis_modes.view_input_images: True shows the two views of the training images instead (needs a display)
uv run python -m vision_training.main --configuration_path="vision_training/configurations/self_supervised/configuration_synthetic_shapes.yaml"

# 3. Evaluation of checkpoints: in the main block of scripts/evaluation/evaluate_checkpoint.py set run_name,
#    experiment_folder (the run folder), iteration (None, an integer or "all"), test_data_folder, batch_size and
#    track_evaluation; maximum_error_images_per_class_pair (and save_predictions, target_precisions, nms_criterion
#    and nms_distance_exponent when set) is skipped with a yellow warning, the task saving no error image, no
#    prediction file, no class threshold and having no detector (Step 5.1); the evaluation is written in
#    <run>/Checkpoint_Evaluation/
uv run python -m scripts.evaluation.evaluate_checkpoint
```

The dataset on disk, generated on 2026-11-18, holds the 35 classes of `SHAPE_CATEGORIES` that the comment of the configuration names (a k-NN chance level of 1/35); the examples of this document have the 9 classes of the datasets generated before the other classes were added.

### Configuration Keys To Know First

Before changing the synthetic configuration, these are the keys that change the views, the loss or the evaluation the most.

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `ProjectionHead.method` | `"simclr"` | InfoNCE (`SIMCLR-T<temperature>` run name token) or `"barlow_twins"` (`BT-OD<weight>`); also selects the batch metrics tracked |
| `ProjectionHead.simclr.temperature`, `ProjectionHead.barlow_twins.off_diagonal_weight` | `0.2`, `0.005` | divides the cosine similarities of InfoNCE; weighs the off-diagonal terms of Barlow Twins; both are checked (> 0) whatever the method |
| `ProjectionHead.global_pooling` | `"average"` | the representation compared by the k-NN and exported: 128 values; `"flatten"` gives the 8,192 values of the [128, 8, 8] map and ties it to the image size |
| `ProjectionHead.fully_connected` | `[512, 128]` | hidden layers, then the embedding size (linear only), the `PAVG.512.128` token |
| `ProjectionHead.fully_connected_normalization` | `"batch"` | `PBN`; `"layer"` gives `PLN`; must be `"none"` when the list holds the embedding layer only |
| `Data.data_augmentation.activated` | `True` | must be True, its types drawing the views; its global `probability` is ignored |
| `Data.data_augmentation.types.random_crop` | `activated: True, probability: 1.0` | random resized crop of 20 % to 100 % of the area, aspect ratio x1/2 to x2; the checks refuse another probability |
| `ExperimentConfiguration.batch_size` | `128` | N: every view is compared with the 2N - 1 = 255 others; at least 2, and a training or validation split smaller than a batch stops the launch |
| `ExperimentConfiguration.evaluation.number_neighbours` | `20` | k of the k-NN, reduced to the number of reference training images when larger |
| `ExperimentConfiguration.evaluation.number_reference_images_per_class` | `50` | reference training images of the k-NN per class: those of smallest file name MD5 hash, the same at every evaluation |
| `Data.grouped_batches.activated` | `False` | True builds the batches from blocks of `images_per_block` (32) images sharing `groups[block_key]` (`"camera"`), the hard negatives |

### What A Run Writes

If you are looking for an output of a run, this is where the training and every evaluated checkpoint write it.

- `Results.txt`: one block per evaluated checkpoint, `Iteration <n> · <N> Test Images · <date>`, with `k-NN Accuracy` (the headline metric), `k-NN Balanced Accuracy`, `k-NN Macro F1`, `Effective Rank` and `Representation Standard Deviation`, the notes `Labelled Test Images : <l> of <N>`, `k-NN : 20 neighbours among <M> labelled training images of <K> classes (50 per class at most)` (followed, when it applies, by a note counting the test classes without labelled training image, also printed in yellow at launch, and one counting the training classes without labelled test image) and `Sample Images : 8 fixed test images`, and with `per_class_report` the table `Per Class k-NN Metrics`, worst F1 first, then the table `Per Class k-NN Recall` (the recall of every class with its exact 95 % confidence interval over its labelled test images), worst recall first. The k-NN metrics, the precisions, recalls and F1 of the per-class table and the recalls and bounds of the recall table are written as percentages with two decimals (`12.10 %`), like in the classification report, the best checkpoint line included, while `Effective Rank` and `Representation Standard Deviation` keep 4 decimals; `class_metrics.csv`, `evaluation_summary.csv`, TensorBoard and MLflow keep every rate between 0 and 1. A test split without labelled image gives the two collapse metrics only, `Effective Rank` being then the headline metric.
- `Evaluation/Iteration_<n>/` (`Checkpoint_Evaluation/Iteration_<n>/` for the standalone evaluation, Step 5.5): `class_metrics.csv` (`class_index,class,samples,precision,recall,recall_lower,recall_upper,f1`, whatever `per_class_report`, when the k-NN is computed) and `Samples/<sample>.png` (the test image and two views drawn from a seed of the image, side by side below a 24 pixel caption bar `<class> | Views Similarity <cosine>`); `EMA/` with the moving average of the weights. There is no confusion matrix, threshold, prediction file, ROC or precision-recall curve nor error image.
- `evaluation_curves.png`, next to `Results.txt`, and `Evaluation/evaluation_summary.csv`, written again after every evaluation (`update_evaluation_curves()`): the curve of the headline metric, the k-NN accuracy (the effective rank for a test split without labelled image), and one row per evaluated iteration with every test scalar.
- `Input_Samples/`, written before the first iteration of a launch: the two views of the first 10 training and validation images with the augmentations drawn for each, and the first 10 test images whole, with `samples.json` (Step 2.21).
- `Tensorboard/Train` and `Tensorboard/Validation`: `Total Loss`, `InfoNCE Loss`, `Positive Pair Accuracy`, `Embedding Standard Deviation` (Barlow Twins: `Barlow Twins Loss`, `Diagonal Correlation`, `Off-Diagonal Correlation`, `Embedding Standard Deviation`), and `Learning Rate` in Train; `Tensorboard/Test`: the 5 global metrics, `k-NN Precision/<class>`, `k-NN Recall/<class>` and `k-NN F1/<class>` with `per_class_report`, and the images `Samples/<sample>`.
- `Weights/Iteration_<n>/model_<n>.pt` (e.g. `Weights/Iteration_250/model_000250.pt`): the whole model, projection head included, so that the training resumes; only the backbone initialises other runs.
- With the MLflow tracking: `test/<metric>`, `test_class/k-NN <Metric>/<class>`, `test/best_k-NN Accuracy` with the tag `best_iteration`, `class_metrics.csv` under `evaluation/iteration_<n>/`, `evaluation_summary.csv` and `evaluation_curves.png` at the root of the artifacts, and the input samples under `Input_Samples/`.

### Pitfalls

Suppose a self-supervised run behaves unexpectedly: these behaviours of the task are the first to check.

- The first checkpoints look collapsed whatever the training: the test images and the validation batches are encoded in evaluation mode, whose batch normalization uses running statistics updated at momentum 0.1 (`BATCH_NORMALIZATION_MOMENTUM`, fixed for the backbone and the projection head). After 12 iterations every test image got almost the same representation (mean pairwise cosine 1.0, against 0.774 in training mode), hence an effective rank of 1.05, a views similarity of 1.000 and a validation InfoNCE of exactly ln(2N - 1) (Step 4.9 and Step 3.20 of this document).
- A cue shared by both views lets the loss fall without learning the content: the colour of a shape when the colour types are off, or its position when the crops are too large. A `Positive Pair Accuracy` near 100 % (1 in TensorBoard) long before the k-NN accuracy rises shows it, and the input viewer shows the views with their drawn augmentations before a training ([tasks reference, Self-Supervised Learning](../reference/tasks_reference.md#self-supervised-learning-ssl), Views and Training metrics).
- With crops down to a fifth of the area (`crop_area_range` of `SELF_SUPERVISED_AUGMENTATION_STRENGTHS`), a view can miss a small object entirely: the two views the sample image of `test_000006.png` of the 9-class dataset generated on 2026-10-11 (a circle near the left border) draws with the synthetic configuration are crops of 22 % and 24 % of the area that both start to the right of the circle, so the loss pulls two backgrounds together. The input viewer (`diagnosis_modes.view_input_images`) shows how often it happens on a dataset.
- The views, the batch size, the learning rate and the grouped batches give no run name token: a run of the same settings with stronger views or another batch size resumes the previous run folder, a run always resuming, unless `run_name_suffix` is set (`build_self_supervised_sections()` of `run_name.py`; the comment of `run_name_suffix` in the configuration).
- Blocks 2, 4 and 6 of the synthetic configuration (16 to 32, 32 to 64 and 64 to 128 channels, 2 layers each) add no residual shortcut, so only blocks 3 and 5 list `Residual Shortcuts` in the backbone summary, and a backbone transferred to another task keeps this layout (`ConvolutionBlock.residual_parity` and `count_residual_additions()` of `convolution_block.py`).
- The k-NN only predicts the classes of its reference images: a test class without labelled training image is never classified correctly (a yellow note at launch and in every evaluation), and a training split without any labelled image next to a labelled test split stops the trainer. The reference images are encoded again at every evaluation, up to `number_reference_images_per_class` times the number of classes: 1,750 images for the 35 classes of the synthetic configuration (450 in the examples of this document, on 9 classes), about 218,000 for the 10 per class of the ImageNet-22k example (`configuration_example.yaml`).

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
│   ├── check_configuration_keys()                on the file as written: the removed settings and sections
│   │                                             first (resume_training, Data.loading, Data.view_augmentation...),
│   │                                             then the keys the self-supervised code does not read (see
│   │                                             classification)
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category identifier: class index}, without background. The classes only serve
│   │       the k-nearest neighbour evaluation: the training never reads a label.
│   └── check_configuration()
│       ├── common checks                         experiment, data (with the flag and probability of every
│       │                                         augmentation type), backbone, optimization (see classification)
│       └── check_self_supervised_configuration()
│               ├── data:       multi_class label mode, classes 0..K-1, Data.data_augmentation on (its types draw the
│               │               views) and random_crop at probability 1.0 when it is activated
│               ├── no sequence normalization (check_image_task_sequence_settings)
│               ├── grouped:    Data.grouped_batches: activated flag, block_key a string, batch_key null or a string,
│               │               images_per_block >= 2, dividing the batch size when the section is on
│               ├── head:       check_projection_head_configuration(): pooling, layers, normalization, method
│               │               (simclr or barlow_twins), and the temperature and the off-diagonal weight above 0,
│               │               both checked whatever the method
│               └── batch:      batch size of at least 2, number_neighbours > 0,
│                               number_reference_images_per_class > 0
│
├── get_trainer_class(task)  -> SelfSupervisedTrainer
│
├── SelfSupervisedTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   Stores the method, the number of neighbours of the k-NN evaluation and its number of reference images per
│   │   class. SELF_SUPERVISED_AUGMENTATION makes the general trainer build its DataAugmenter with the self-supervised
│   │   strengths: it draws the two views. After BaseTrainer.__init__(), build_reference_dataloader() chooses the
│   │   reference training images of the k-NN (select_reference_samples) and builds their loader.
│   │
│   └── BaseTrainer.__init__()
│       │   Same order as for every task: device, run name, settings, augmenter, classes of a resumed run, refusal
│       │   of a resume on changed labels, paths, loaders and their check, record of the data, sample images,
│       │   model, pretrained components, optimizer and moving average of the weights,
│       │   metric tracker, printouts, input viewer, analysis of the training data (nothing for this task),
│       │   restoration.
│       │
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter()
│       │       The run name starts with SSL and ends with the head (PAVG.<layers>-PBN), the method
│       │       (SIMCLR-T<temperature> or BT-OD<off-diagonal weight>) and the run_name_suffix if not null.
│       │       The DataAugmenter takes SELF_SUPERVISED_AUGMENTATION_STRENGTHS: it draws the two random views
│       │       of every image.
│       ├── restore_run_label_dictionary() / check_resumed_data() / setup_training_paths()
│       │       A resumed run keeps the classes of its label_dictionary.json and is refused when the labels.json of
│       │       a split changed since its data.json, then the run folder is created with the configuration and
│       │       label_dictionary.json (classification_workflow.md, Step 2.6).
│       ├── build_dataloaders()
│       │   │   Builds the loaders (the validation one only when the run reads it, builds_validation_loader) and
│       │   │   names the classes after the categories of the three manifests.
│       │   │
│       │   └── get_classification_dataloaders(self_supervised=True)
│       │       ├── ClassificationDataset.__init__(output_mode="multi_class", keep_unlabelled_samples=True)
│       │       │       Same dataset as the classification; an image without a mapped category is kept, with the
│       │       │       label IGNORED_LABEL_INDEX (-100).
│       │       ├── build_grouped_batch_sampler()
│       │       │       With Data.grouped_batches on, the GroupedBatchSampler of the training and validation splits
│       │       │       whose samples hold its keys: batches of blocks of images sharing an attribute of their groups.
│       │       └── build_native_loader()
│       │               drop_last for the training and validation loaders, so that every batch has batch_size images,
│       │               or the GroupedBatchSampler in place of the batch size, the shuffling and drop_last.
│       ├── check_loaders_give_batches()          refuses a split whose loader gives no batch, and a validation
│       │                                         loader not following builds_validation_loader
│       ├── save_data_record()                    data.json: the classes, counted on the labelled images of every
│       │                                         split, and the images of every split (classification_workflow.md,
│       │                                         Step 2.7)
│       ├── get_test_sample_classes() + choose_sample_identifiers()
│       │       The 8 fixed test images of the sample images, spread over the classes.
│       ├── build_model()
│       │   └── SelfSupervisedModel.__init__()
│       │       ├── build_backbone()              the backbone
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
│       ├── estimate_export_sizes() + print_export_size_summary()
│       │       The Export Sizes panel: sizes of the PyTorch, TorchScript, ONNX, TensorRT and checkpoint
│       │       files; its exported parameters leave out the projection head, unused by the exported encoder.
│       ├── view_input_batches()  (only with view_input_images, then exits)
│       │       The two random views of a training image, each next to the augmentations drawn for it, every
│       │       batch of the GroupedBatchSampler first shown whole (show_input_batch -> build_batch_mosaic).
│       └── restore_last_model() -> get_last_checkpoint_iteration() / load_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── save_input_samples()                Input_Samples/: the first 10 images of every split, the training and
        │                                       validation ones as their two views, the test ones whole (Step 2.21)
        ├── setup_tensorboard_writers() / initialize_training_loop_state() / start_experiment_tracking()
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch()                    a batch failing to load is dropped and the next one taken
        │   │   └── ClassificationDataset.__getitem__() -> read_image()
        │   │           images [B, 3, H, W] uint8, labels [B] int64 (unused by the training), sample_identifiers [B]
        │   ├── run_training_step()  set_training_mode() to step_optimizer() below, the step also timed by
        │   │       the speed benchmark (diagnosis_modes.benchmark_speed)
        │   ├── set_training_mode()
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()                 images to float32, labels to int64 on the device
        │   │   ├── augment_batch()                 returns the batch unchanged: the views are drawn in forward_pass
        │   │   ├── forward_pass()
        │   │   │   ├── DataAugmenter.create_views()        two random views of every image, in float32
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
        └── close the writers, shutdown_loader_workers(), finish_experiment_tracking()

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
│   └── compute_nearest_neighbour_results()  (with at least one labelled test image)
│       ├── encode_reference_images()                 reference training images, encoded with the evaluated weights
│       ├── compute_nearest_neighbour_predictions()   weighted k-NN vote of the reference images
│       ├── compute_confusion_matrix() + compute_metrics_from_confusion_matrix()   sparse above 1,000 classes
│       ├── compute_clopper_pearson_intervals()       exact 95 % interval of the recall of every class
│       ├── build_class_metrics_table() + build_recall_interval_table()   Per Class k-NN Metrics and Recall tables
│       └── write_class_metrics_file()                class_metrics.csv
└── report_evaluation() -> report_evaluation_results() + track_evaluation() + update_evaluation_curves()
                                       the row of the iteration in evaluation_summary.csv, evaluation_curves.png

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]   (uv run python -m scripts.evaluation.evaluate_checkpoint)
└── same as every task, maximum_error_images_per_class_pair skipped with a warning (the configuration has no
    error_dumps section), as save_predictions when set: loads training_configuration.yaml, builds the
    SelfSupervisedTrainer writing in <run>/Checkpoint_Evaluation/ (inside read_manifests_once()), then
    evaluate_checkpoints() -> start_evaluation_tracking() + summarise_checkpoint_evaluations() -> run_test_evaluation()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/self_supervised/configuration_synthetic_shapes.yaml`.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1): `parse_configuration_path()` reads the path of the YAML file from the command line (`--configuration_path`, absolute or relative to the current folder, `parser.error()` stopping with "No configuration file at ..." when it is not a file), then `main()` loads it, looks up the trainer of `ExperimentConfiguration.task` and starts its training loop. The loading and the construction of the trainer run inside `read_manifests_once()` (`vision_training/data/manifest_cache.py`), so that every `labels.json` is parsed once per launch and shared by its readers (`read_manifest()`: the datasets, `data.json`, the split descriptions, the validation categories of Step 2.7). Nothing in it is specific to the self-supervised task.

```bash
python -m vision_training.main --configuration_path vision_training/configurations/self_supervised/configuration_synthetic_shapes.yaml
```

### Step 1.2 — `load_experiment_configuration()`
It splits the YAML into the two dictionaries read by every later stage, converts the values YAML reads with the wrong type (`1e10` is read as a string), adds `tracking_root` (the root folder of the MLflow backend, the `experiment_folder` of the configuration), `run_name` (None, the name of the run folder then being built from the settings, Step 2.4) and `standalone_evaluation_folder` (None for a training, `Checkpoint_Evaluation` set by the standalone evaluation, Step 5.2), adds `label_mode` and `label_dictionary` to `Data` (Step 1.4) and runs the checks (Step 1.5). The self-supervised task has one section of its own, `ProjectionHead`; its random views are drawn by the types of `Data.data_augmentation`, the section of every task.

**Example: the sections and the converted values**
```text
raw keys:               ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'ProjectionHead']
raw number_iterations:  '1e10'                       -> after loading: 10000000000
experiment keys:        task, experiment_folder, run_name_suffix, train_data_folder, validation_data_folder,
                        test_data_folder, batch_size, information_dump, weight_saving_iterations,
                        intermediate_weight_saving_iterations, number_iterations, validation_iteration_ratio, evaluation,
                        tracking, learning_rate, allow_data_change_on_resume, diagnosis_modes,
                        tracking_root, run_name, standalone_evaluation_folder
model keys:             ['Optimization', 'Data', 'Backbone', 'ProjectionHead']
task:                   'self_supervised'
experiment_folder:      PosixPath('.../visual-language-model-spark/experiments')
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
Reads the mapping file of `Data.label_mapping` (in `vision_training/data/label_mapping.py`) and gives the class index of every category identifier. Unlike the detection and the segmentation there is no background class. The labels are never read by the training: they only name the classes of the k-nearest neighbour evaluation of Stage 4. A configuration file read from a run folder holding a `label_dictionary.json` (the copy of the configuration read by the standalone evaluation, Step 5.2) takes the classes of that file instead of the mapping file (`read_run_label_dictionary()`).

**Example output**
```text
label_mode:       multi_class
label_dictionary: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
```

### Step 1.5 — `check_configuration()`
The common checks, then `check_self_supervised_configuration()` (in `vision_training/configuration_processing/self_supervised_checks.py`):
- **data**: `multi_class` label mode, classes 0 to K-1, `Data.data_augmentation.activated` True, since its types draw the two views of every image, which would otherwise both be the image itself, `random_crop`, the random resized crop of the views, at a probability of 1.0 when it is activated (drawn for every view, or turned off to keep the whole images), and no sequence settings (`check_image_task_sequence_settings()`). The flag and the probability of every type are checked as for every task (`check_data_configuration()`); the strengths are hard-coded.
- **grouped batches**: `check_grouped_batches_configuration()`: `Data.grouped_batches.activated` is True or False, `block_key` a non-empty string, `batch_key` null or a non-empty string, and `images_per_block` an integer of at least 2 that divides the batch size when the section is on.
- **head**: `check_projection_head_configuration()`: the pooling, the layers (a non-empty list of positive integers, the embedding size last), the normalization (`"none"` when the list holds the embedding layer only; its batch momentum is fixed at 0.1 in the code), the method (`simclr` or `barlow_twins`), and the settings of both methods (`simclr.temperature` and `barlow_twins.off_diagonal_weight` above 0), checked whatever the method so that a run can switch method by one key.
- **batch and evaluation**: a batch size of at least 2, and `number_neighbours` and `number_reference_images_per_class` positive integers (`check_self_supervised_evaluation_configuration()`).

Every error is listed before the program stops.

**Example output: a configuration with three errors**
```text
------------------------------------------------------
Invalid Configuration File : configuration_wrong.yaml
------------------------------------------------------
  - 'Data.data_augmentation.types.random_crop.probability' must be 1.0 for the self-supervised task when 'Data.data_augmentation.types.random_crop.activated' is True, got 0.5: the random resized crop is drawn for every view, or turned off to keep the whole images.
  - 'ProjectionHead.method' must be one of ['simclr', 'barlow_twins'], got 'byol'.
  - 'ExperimentConfiguration.batch_size' must be at least 2 for the self-supervised task: the losses compare the views of the images of a batch with each other.
SystemExit code 1
```

A configuration still holding `Data.view_augmentation`, the section of the views before they were drawn by the data augmentation, is refused alone, before any other check, by `check_removed_settings()` (run by `check_configuration_keys()`), as are the settings fixed in the code since (`resume_training`, `mixed_precision`, `Data.loading`...). A key the self-supervised code does not read, such as a `ClassificationHead` section left from a classification configuration, is refused next, alone too (`check_unknown_keys()`).

**Example output: the configuration of the synthetic shapes before the merge of the two augmentations** (recorded before the fixed settings were removed: the same file now also gets a line for every fixed setting it holds)
```text
--------------------------------------------------------------------------------
Invalid Configuration File : configuration_before_single_data_augmentation.yaml
--------------------------------------------------------------------------------
  - 'Data.view_augmentation' was removed: the self-supervised task draws its two views with the types of 'Data.data_augmentation', whose strengths are hard-coded (SELF_SUPERVISED_AUGMENTATION_STRENGTHS of vision_training/data/augmentations.py), so its probabilities go to those types. Delete it from the configuration file.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
<class 'vision_training.trainers.self_supervised_trainer.SelfSupervisedTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on `configurations/self_supervised/configuration_synthetic_shapes.yaml` (256 pixel images, batch size 128, 2 loading workers), with the run folder redirected to a temporary folder. The whole construction took 14.3 s on CPU, at the lowest priority next to other trainings.

### Step 2.1 — `SelfSupervisedTrainer.__init__()`
Before calling `BaseTrainer.__init__()`, it stores what its hooks read: `self.method` (`ProjectionHead.method`), `self.number_neighbours` (`evaluation.number_neighbours`, the k of the k-NN evaluation) and `self.number_reference_images_per_class` (`evaluation.number_reference_images_per_class`). After it, `build_reference_dataloader()` chooses the reference images of the k-NN among the labelled training images (`select_reference_samples()`: in every class, the images of smallest file name MD5 hash, so the same images whatever the shuffling of the dataset) and builds their loader, read in order without augmentation; a test split with labelled images and a training split without any stops with a `ValueError`, and a test class without labelled training image is announced in yellow. Its class attributes name the task for the shared code, `DROPS_LAST_TRAINING_BATCH` telling it that the training loader drops its last incomplete batch (Step 2.19), and `SELF_SUPERVISED_AUGMENTATION` that its `DataAugmenter` draws the views with the self-supervised strengths (Step 2.5).

**Example output**
```text
method: simclr   number_neighbours: 20   number_reference_images_per_class: 50
TASK_NAME: self_supervised   SAMPLE_NAME: Images   COUNTS_FRAMES: False   DROPS_LAST_TRAINING_BATCH: True
SELF_SUPERVISED_AUGMENTATION: True
reference images: 450 in 4 batches of 128
reference_notes: ['k-NN : 20 neighbours among 450 labelled training images of 9 classes (50 per class at most)']
```

### Step 2.2 — `BaseTrainer.__init__()`
The constructor of every task (see `classification_workflow.md`, Step 2.2).

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
The run name gives the task (`SSL`), the input, the backbone, then the head built by `build_self_supervised_sections()` (in `vision_training/configuration_processing/run_name.py`): `PAVG` for the average pooling, the layers of the head, `PBN` for its batch normalization, and the method, `SIMCLR-T<temperature>` or `BT-OD<off-diagonal weight>`. There is no class count, since the training never reads a label. A `run_name_suffix` that is not null ends the name, e.g. `_v1`: the views, the batch size and the learning rate give no token, so a run of the same settings with stronger views or larger batches needs it to start in a folder of its own instead of resuming the previous run.

**Example output**
```text
SSL_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2
```

### Step 2.5 — `DataAugmenter()`
The general trainer builds the `DataAugmenter` of every task (in `vision_training/data/augmentations.py`) from `Data.data_augmentation`, here with `self_supervised=True` since the trainer sets `SELF_SUPERVISED_AUGMENTATION`: the types keep the activated flags and the probabilities of the configuration and take the hard-coded `SELF_SUPERVISED_AUGMENTATION_STRENGTHS`, different from the `SUPERVISED_AUGMENTATION_STRENGTHS` of the other tasks (a crop of 20 % to 100 % of the area at an aspect ratio of x1/2 to x2, a colour jitter of x0.6 to x1.2, a hue turned by up to 36 degrees..., the table of the [training reference, Data Augmentation](../reference/training_reference.md#data-augmentation)). Its `create_views(images)` draws the two views of a batch (Step 3.12):
- every image gets its two views: the global `probability` is ignored, and every activated type is drawn for each view of each image with its own probability;
- `random_solarize` and `random_black_out` (`SECOND_VIEW_ONLY_TYPES`) are drawn for the second view only, as the asymmetric views of BYOL and Barlow Twins, while the supervised tasks draw them for every image;
- the views are computed in float32 with the autocast off, so that the interpolations and the blurs keep their precision under the bfloat16 autocast of mixed precision, and returned with the augmentation report of every view.

**Example: the settings of the augmenter**
```text
activated: True   probability: 1.0 (ignored by the views)   self_supervised: True
strengths is SELF_SUPERVISED_AUGMENTATION_STRENGTHS: True
18 of 28 types activated: {'random_crop': 1.0, 'random_left_right_flip': 0.5, 'random_up_down_flip': 0.5,
    'random_rotation': 0.5, 'random_perspective': 0.5, 'brightness_contrast_saturation': 0.8, 'random_hue': 0.8,
    'random_noise': 0.2, 'gray_scale_augment': 0.2, 'random_gamma': 0.3, 'random_linear_illumination': 0.3,
    'random_plasma_shadow': 0.2, 'random_blur': 0.5, 'random_motion_blur': 0.1, 'random_haze': 0.3,
    'random_solarize': 0.2, 'random_black_out': 0.3, 'random_erasing': 0.25}
black_out_grid_size: 8   second view only: ['random_solarize', 'random_black_out']
  crop_area_range: (0.2, 1.0)
  crop_aspect_ratio_range: (0.5, 2.0)
  maximum_rotation_degrees: 10.0
  brightness_factor_range: (0.6, 1.2)
  maximum_hue_shift: 0.1
  blur_sigma_range: (2.5, 2.5)
  noise_standard_deviation: 30.0
```

### Step 2.6 — `setup_training_paths()`
Creates the run folder, copies the YAML file into it and writes the label dictionary of the run (`label_dictionary.json`), then the constructor writes the data record once the loaders exist. The copy is a file copy, so it holds the file as written, not the values changed in memory after loading. Before it, `restore_run_label_dictionary()` keeps the classes of a resumed run, `check_resumed_data()` refuses the resume of a run whose `labels.json` changed since its `data.json` (unless `allow_data_change_on_resume` is true), as for every task (see `classification_workflow.md`, Step 2.6).

**Example: the run folder after the construction**
```text
Tensorboard/                    folder
Weights/                        folder
data.json                       17,002,746 bytes
label_dictionary.json           110 bytes, the class index of every category of the run
training_configuration.yaml     28,255 bytes
```
The size of `training_configuration.yaml` is the one of the configuration file on 2026-10-24: its comments were shortened and its fixed settings removed since, the file holding 16,041 bytes on 2026-11-20.

### Step 2.7 — `build_dataloaders()`
Calls `get_classification_dataloaders(..., self_supervised=True)` (Step 2.8), then merges the categories of the three manifests by identifier, so that a class absent from the test manifest keeps its name, and stores `class_names` in class index order. The validation loader is built only when the run reads it (`validation_iteration_ratio > 0` or a diagnosis mode ending the program, `classification_workflow.md`, Step 2.7); without it, the categories of the validation manifest are read from its `labels.json`, so that the class names never depend on the validation of the run.

**Example output**
```text
class_names: ['square', 'circle', 'triangle', 'rectangle', 'pentagon', 'hexagon', 'plus_sign', 'star', 'octagon']
```

### Step 2.8 — `get_classification_dataloaders(self_supervised=True)`
The loaders of the classification (in `vision_training/data/loaders.py`), with three differences: the output mode is always `multi_class` (there is no `ClassificationHead` section), the datasets keep the images without a mapped category (Step 2.9), and the training and validation loaders drop their last incomplete batch, so that every batch of the losses has `batch_size` images. The test loader keeps every image. Once the three splits are built, it prints the images without a mapped category of every split (Step 2.9) in one yellow block, then the keys and batches of the grouped batches in one blue block, one line per split and key.

With `Data.grouped_batches` on, `build_grouped_batch_sampler()` (in `vision_training/data/grouped_batch_sampler.py`) builds the `GroupedBatchSampler` of the training and validation splits from the `groups` of their samples, given to `build_native_loader()` (Step 2.10). The sampler rebuilds the batches at every pass: the images grouped by (batch value, block value) are shuffled and cut into blocks of `images_per_block`, the blocks of every batch value are joined into full batches, and the images left over are shuffled into full random batches, the last incomplete one being dropped, so that a pass still gives `floor(N / batch_size)` batches and the iterations per epoch do not change. A split whose samples hold none of the keys keeps random batches with a yellow warning, and a split where only some samples hold them stops the launch with the number of samples missing each key.

**Example: the three splits**

| Split | Images | Batches | drop_last |
|---|---|---|---|
| train | 500000 | 3906 | True |
| validation | 50000 | 390 | True |
| test | 50000 | 391 | False |

**Example output: grouped batches of a training split of 10 images in two cameras, with batches of 4 images and blocks of 2, and a validation split without groups**
```text
No sample of the 'validation_data_folder' split holds the keys ['camera'] of 'Data.grouped_batches': its batches stay random.
Grouped Batches :
  - Train block key : camera (2 values)
  - Train batches per pass : 2 grouped, 0 leftover
```

### Step 2.9 — `ClassificationDataset.__init__()`
The dataset of the classification (in `vision_training/data/classification/classification_dataset.py`), with `keep_unlabelled_samples=True`: an image without a mapped category is kept with the label -100 (`IGNORED_LABEL_INDEX` of `vision_training/core/contracts.py`) instead of being dropped, and counted in `number_unlabelled_samples`, which the loaders print for the three splits (Step 2.8, Step 5.6). Every record also keeps the optional `groups` of its sample, an empty dictionary without them, read by the grouped batches (Step 2.8). It reads the manifest with `read_manifest()` (Step 1.1) and shuffles the kept samples once, so the first kept sample is not the first of the file, then packs them in that order into `SerializedRecords` (`vision_training/data/serialized_records.py`, `classification_workflow.md`, Step 2.9): `samples[index]` gives one record, a pass over `samples` reads them all (`select_reference_samples()` of Step 2.1), and `samples.find_record()` finds a record by its identifier (the groups of a batch in the input viewer, Step 2.21).

**Example: the manifest, the first kept sample and `dataset[0]`**
```text
manifest keys: ['info', 'categories', 'samples']   500000 samples, 9 categories
manifest sample:  {'id': 1, 'file_name': 'train_000000.png', 'category_id': 6}
first kept:       {'identifier': 145458, 'file_name': 'train_145457.png', 'class_indices': [5], 'groups': {}}
dataset[0]:       images [3, 256, 256] torch.uint8 in [14, 231]   labels () torch.int64 = 5   sample_identifiers 145458
```

### Step 2.10 — `build_native_loader()`
Same torch DataLoader as the classification. With grouped batches, the DataLoader receives the `GroupedBatchSampler` as its `batch_sampler` instead of the batch size, the shuffling and `drop_last`, which it refuses next to a batch sampler: it then reads `batch_size None` and `drop_last False`, its length being the number of batches of the sampler.

**Example: the training loader** (recorded with 2 workers, when the number of workers was a setting, and pinned memory; every loader now has 1 worker and pins its batches only on a discrete GPU)
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
- `backbone`: the backbone (Step 2.14).
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
The backbone of every task (see `classification_workflow.md`, Step 2.14). `Backbone.print_summary()` traces a zero image through the blocks in evaluation mode and lists the modules each block uses (`ConvolutionBlock.get_active_module_names()`). Blocks 2, 4 and 6 change the channel count with 2 layers, so their pairs start at their second layer and they add no residual shortcut (`count_residual_additions()` gives 0, 0, 1, 0, 1, 0 for the six blocks): only blocks 3 and 5 list `Residual Shortcuts` (`ConvolutionBlock.residual_parity`).

**Example output** (256 pixels, printed by `print_summary()`, Step 2.20)
```text
------------------------------
Backbone Architecture Summary
------------------------------
------------------------------------
Input image shape: [1, 3, 256, 256]
------------------------------------
╭─────────────────────────── Backbone Architecture ────────────────────────────╮
│ Normalization Strategy: Batch                                                │
│                                                                              │
│  - Convolution Block 1                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 3, 256, 256]                                         │
│    - Output Shape : [1, 16, 128, 128]                                        │
│                                                                              │
│  - Convolution Block 2                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 16, 128, 128]                                        │
│    - Output Shape : [1, 32, 64, 64]                                          │
│                                                                              │
│  - Convolution Block 3                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 32, 64, 64]                                          │
│    - Output Shape : [1, 32, 32, 32]                                          │
│                                                                              │
│  - Convolution Block 4                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 32, 32, 32]                                          │
│    - Output Shape : [1, 64, 16, 16]                                          │
│                                                                              │
│  - Convolution Block 5                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 64, 16, 16]                                          │
│    - Output Shape : [1, 64, 8, 8]                                            │
│                                                                              │
│  - Convolution Block 6                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 64, 8, 8]                                            │
│    - Output Shape : [1, 128, 8, 8]                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
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
The tracked metrics depend on the method (`METHOD_TRACKED_METRICS` of the trainer): the loss of the method, the batch metrics it returns and the embedding standard deviation, which watches the collapse of every method. `get_percent_tracked_metrics()` gives the rates of the method from `METHOD_PERCENT_TRACKED_METRICS`, printed as percentages with their changes in percentage points (`+0.52 %`) in the console summary (Step 3.21): the positive pair accuracy of SimCLR, and none for Barlow Twins, whose diagonal and off-diagonal correlations keep 4 decimals like the losses and the embedding standard deviation.

**Example: the tracked metrics of both methods**
```text
simclr:       {'total_loss': 'Total Loss', 'info_nce_loss': 'InfoNCE Loss', 'positive_pair_accuracy': 'Positive Pair Accuracy',
               'embedding_standard_deviation': 'Embedding Standard Deviation'}
barlow_twins: {'total_loss': 'Total Loss', 'barlow_twins_loss': 'Barlow Twins Loss', 'cross_correlation_diagonal': 'Diagonal Correlation',
               'cross_correlation_off_diagonal': 'Off-Diagonal Correlation', 'embedding_standard_deviation': 'Embedding Standard Deviation'}
get_percent_tracked_metrics(): simclr ['positive_pair_accuracy'], barlow_twins []
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
- Weight Moving Average   : Off
- Validation Iter. Ratio  : 5
- MLflow Tracking         : Off
```
The 3906 iterations per epoch are 500000 / 128 rounded down: the self-supervised trainer sets `DROPS_LAST_TRAINING_BATCH`, since its training loader drops its last incomplete batch, while the other tasks round up, except the classification and sequence tasks whose training loader drops it too, for a batch normalized classification head (see `classification_workflow.md`, Step 2.10).

### Step 2.20 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The backbone panel of Step 2.14, the Projection Head panel of Step 2.15, the receptive field (with `diagnosis_modes.compute_receptive_field`) and the size of every component, always printed. The trainer then prints the Export Sizes panel: the parameters the forward pass of the exported encoder reads, which leave out the projection head, then the size of the PyTorch weights, the TorchScript file, the ONNX files and the TensorRT engines the model is turned into and of the training checkpoint ([Export Sizes](../../README.md#export-sizes)).

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
│                                                          │
│ Projection Head                                          │
│   - Parameters           : 132,736                       │
│   - Trainable Parameters : 132,736                       │
│   - Estimated Size       : 0.54 MB                       │
│                                                          │
│ Total Model                                              │
│   - Parameters           : 571,104                       │
│   - Trainable Parameters : 571,104                       │
│   - Estimated Size       : 2.29 MB                       │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.21 — `view_input_batches()`
Only with `diagnosis_modes.view_input_images`, then the program exits. `augment_input_batch()` draws the two views of the training images with `create_views()`, and `build_input_windows()` draws those of one image as two windows, "View 1" and "View 2", each next to a black information panel holding the class of the image and, under "View augmentations", the augmentations drawn for the view with their values (`describe_report()`, the lines of Step 3.12). The validation images get their views drawn the same way, so `show_input_samples()` shows the training image only. The input samples written at the start of every training launch (`classification_workflow.md`, Step 3.1) draw the training and validation images as these two views (`AUGMENTED_INPUT_SPLITS`), and the test images whole, as the evaluation encodes them.

With `Data.grouped_batches` on, `show_input_batch()` first shows every training batch whole in a "Training Batch" window, before the views of its images: `build_batch_mosaic()` (in `vision_training/utilities/visualization.py`) lays out the thumbnails of its images in the order of the batch, cut into blocks of `images_per_block` images, each under a caption bar, up to 16 thumbnails per row, below a header drawn like the information panels. `describe_input_batch()` writes the header from the `grouped_batch_flags` the `GroupedBatchSampler` keeps for the batches of its current pass: the position of the batch in the pass, whether it is grouped or a random batch of the leftover pool, and the grouped and random batches of a pass; a grouped batch adds its batch value and its blocks, every block captioned with its block value; a random batch counts the images of every batch and block value, most frequent first, and captions every image with its values. Random batches (the section off, or a training split without groups) show no mosaic. SPACE goes from the mosaic to the views of the first image of the batch, ENTER stops. The terminal only receives the title of the viewer in red and its keys in green, as for every task (see `classification_workflow.md`, Step 2.25), and no run folder is created. It needs a display and was not run for this document.

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

> The examples below are real outputs from a run on CPU of `configuration_synthetic_shapes.yaml` shortened to 64 pixel images, batch size 8, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, the first 1000 test images and the MLflow tracking on, seeded with 0. Its run name is `SSL_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_PAVG.512.128-PBN_SIMCLR-T0.2`. The Barlow Twins examples come from the first 4 iterations of the same run with `ProjectionHead.method: barlow_twins` (run name ending `_BT-OD0.005`).

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
The MLflow run is created in the experiment `self_supervised` (the task name), in the local database of the experiments root folder, by a background thread, so that a slow database never delays the training.

**Example output**
```text
MLflow Run Created In The Experiment 'self_supervised' : a42373adfbf94d839fa79699e0103b2b (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```
In this run the message came after the console summary of iteration 4: the checkpoint of iteration 3, saved while the run was still being created, holds `mlflow_run_id: None` (Step 3.5), and the interval metrics of iteration 4, queued before, were written once the run was open.

### Step 3.5 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12, so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6). The checkpoint holds the whole model, projection head included, so that the training can resume; only the backbone is transferred to other runs.

**Example: the checkpoints and the keys of the last one**
```text
Checkpoint Folder :
 - Iteration_3
 - Model Saved In : model_000003.pt

Weights/Iteration_3/model_000003.pt   6,967,879 bytes      Weights/Iteration_6/model_000006.pt   6,967,943 bytes
Weights/Iteration_9/model_000009.pt   6,967,943 bytes      Weights/Iteration_12/model_000012.pt  6,967,943 bytes
Weights/full-checkpoint               207 bytes: model_checkpoint_path: "Iteration_12", all_model_checkpoint_paths 3, 6, 9, 12
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone']
model_state: 108 tensors, the first 'backbone.convolution_blocks.0.squeeze_and_excitation_layers.0.first_adaptive_recalibrator.weight'
format_version: 2   task: 'self_supervised'   pretrained_backbone: None
mlflow_run_id: None in the checkpoint of iteration 3 (the MLflow run was not open yet), 'a42373adfbf94d839fa79699e0103b2b' in those of 6, 9 and 12
```
A checkpoint written now also holds `pretrained_backbone_blocks` (`LOADED_BACKBONE_BLOCKS_METADATA_KEY` of `vision_training/models/model_transfer.py`), the number of backbone blocks loaded from a shorter source backbone, read back by a resumed run; it would be `None` for this run.

### Step 3.6 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration, after its checkpoint and its summary (see `classification_workflow.md`, Step 3.7), see Stage 4.

### Step 3.7 — `get_next_batch()` → `ClassificationDataset.__getitem__()`
The classification dataset reads every image with `read_image()` and resizes it (see `classification_workflow.md`, Steps 3.8 and 3.9): a batch whose image cannot be read is dropped, its error printed once, and the next batch is taken. The labels come with the batch but the training never reads them.

**Example: the first batch of the run**
```text
images               [8, 3, 64, 64] torch.uint8 in [0, 253]
labels               [8] torch.int64                  [5, 8, 2, 7, 8, 2, 6, 0]
sample_identifiers   [8] torch.int64                  [261621, 86285, 422277, 300679, 496458, 349760, 109902, 396162]
```

### Step 3.8 — `set_training_mode()`
`model.train()`: the batch normalization layers of the backbone and of the head use the statistics of the batch (see Step 4.9 for what this changes at evaluation).

### Step 3.9 — `run_model_iteration()`
The same function for the training and the validation batches (see `classification_workflow.md`, Step 3.11): prepare, augment, forward, sum of the `_loss` outputs into `total_loss`, batch metrics, tracking. With a single loss, the total loss is the loss of the method.

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
Returns the batch unchanged: the views are drawn in `forward_pass()` (Step 3.12), so that the validation batches, which skip `augment_batch()`, get random views too.

### Step 3.12 — `forward_pass()` → `DataAugmenter.create_views()`
`forward_pass()` draws two random views of every image, then calls `SelfSupervisedModel.forward(input_tensor=first_views, second_view_tensor=second_views)`. Every activated type is drawn independently for each image and each view with its own probability, the global `probability` being ignored, so the two views of an image differ in crop, geometry, colours, lighting, blur, noise and erasing, and only the second one can be solarized or blacked out (`SECOND_VIEW_ONLY_TYPES`).

**Example: the views of a training batch and the augmentations drawn for two images** (the first training batch of the shortened configuration, drawn on CPU with the seeds 0 outside the run, `describe_report()` giving the lines of every view)
```text
first_views:  [8, 3, 64, 64] torch.float32 in [0, 249]
second_views: [8, 3, 64, 64] torch.float32 in [0, 245]

image 0, view 1:
  Geometric :
    random_crop : 57x63 pixels (88 % of the area), at top 1 and left 7 pixels
    random_left_right_flip
    random_rotation : -0.5 degrees
  Photometric :
    random_hue : hue +16.5 degrees
    random_linear_illumination : -36 levels growing towards the left
image 0, view 2:
  Geometric :
    random_crop : 59x45 pixels (65 % of the area), at top 5 and left 0 pixels
    random_left_right_flip
    random_perspective : corners moved by up to 1.9 pixels
  Photometric :
    brightness_contrast_saturation : brightness x0.99, contrast x0.88, saturation x0.86
    random_noise : standard deviation 30 levels
    random_gamma : gamma 1.19
image 1, view 1:
  Geometric :
    random_crop : 46x60 pixels (67 % of the area), at top 1 and left 17 pixels
    random_left_right_flip
    random_rotation : +6.1 degrees
  Photometric :
    brightness_contrast_saturation : brightness x0.71, contrast x0.94, saturation x0.74
    random_hue : hue +33.7 degrees
    random_linear_illumination : +33 levels growing towards the bottom
image 1, view 2:
  Geometric :
    random_crop : 46x61 pixels (69 % of the area), at top 1 and left 4 pixels
    random_left_right_flip
    random_up_down_flip
    random_rotation : -8.1 degrees
  Photometric :
    brightness_contrast_saturation : brightness x0.78, contrast x1.10, saturation x0.78
    random_hue : hue -28.2 degrees
    random_noise : standard deviation 30 levels
    random_plasma_shadow : intensity -34 levels, quantity 0.14, roughness 0.44
    random_blur : 9x9 kernel, sigma 2.5
```

### Step 3.13 — `SelfSupervisedModel.forward()` → `encode()`
Both views go through the model as one batch of 2N = 16 images, so that the batch normalization sees both views together; the embeddings are then split back into the first and second views. Under the bfloat16 autocast of mixed precision, the outputs are bfloat16.

**Example: the outputs of the components on the 16 views**
```text
backbone           [16, 128, 2, 2] torch.bfloat16
global_pooling     [16, 128] torch.bfloat16 in [0, 0.96]      the representations
projection_head    [16, 128] torch.bfloat16                   the embeddings, split into [8, 128] + [8, 128]
```

### Step 3.14 — `compute_method_loss()` → `compute_info_nce_loss()` (SimCLR)
In `vision_training/losses/self_supervised_losses.py`, computed in float32 outside the autocast. The 2N embeddings are L2-normalized and compared by cosine similarity, divided by the temperature. Every view is a classification among the 2N - 1 other views (its own similarity is masked), whose right answer is the other view of its image, at index i + N for a first view and i - N for a second view. The loss is the cross entropy of these 2N classifications: about ln(2N - 1) for random embeddings, here ln 15 = 2.7081. The positive pair accuracy is the share of views whose most similar view is their partner.

**Example: the first iteration**
```text
first and second embeddings: [8, 128] torch.bfloat16
view 0: cosine with its partner 0.13, mean cosine with the 14 other views 0.39, highest cosine 0.82 (view 13)
        -> its partner is not its most similar view, and its own row of the loss is about 4.72
info_nce_loss = 3.0699   positive_pair_accuracy = 0.0   (ln 15 = 2.7081 for random embeddings)
```

**Example: the 12 training iterations** (the learning rate stays at 0.0005; the gradient norm was measured for this document, the trainer neither printing nor clipping it)

| Iteration | InfoNCE loss | Positive pair accuracy | Gradient norm |
|---|---|---|---|
| 1 | 3.0699 | 0.0 | 68.36 |
| 2 | 2.6210 | 0.1875 | 40.08 |
| 3 | 2.7429 | 0.1875 | 43.46 |
| 4 | 2.3775 | 0.1875 | 37.75 |
| 5 | 2.7915 | 0.125 | 30.52 |
| 6 | 3.1341 | 0.125 | 40.13 |
| 7 | 2.5209 | 0.125 | 33.69 |
| 8 | 2.8691 | 0.0 | 28.89 |
| 9 | 2.6249 | 0.0625 | 26.62 |
| 10 | 2.6380 | 0.125 | 19.07 |
| 11 | 2.7711 | 0.125 | 24.63 |
| 12 | 2.7278 | 0.0 | 20.53 |

### Step 3.15 — `compute_embedding_standard_deviation()`
The standard deviation over the batch of every feature of the L2-normalized embeddings of the first views, averaged over the features and multiplied by the square root of the dimension D. Embeddings spread over the whole sphere give about 1, collapsed embeddings (the same embedding whatever the image) give 0. It is computed without gradient and only watches the training.

**Example: the training batches**
```text
embedding_standard_deviation between 0.50 and 0.73 over the 12 iterations: 0.73 at iteration 1, 0.50 at iteration 12
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
iteration 1: barlow_twins_loss = 165.0846   cross_correlation_diagonal = -0.0430   cross_correlation_off_diagonal = 0.3012
             embedding_standard_deviation = 0.7320   gradient norm = 6000.94
iterations 2 to 4: barlow_twins_loss = 95.06, 172.97, 93.75
```
With a diagonal near 0 the invariance term alone is close to D = 128, which explains the size of the loss and of its gradients compared with InfoNCE.

### Step 3.17 — `compute_batch_metrics()`
Returns the batch metrics given by the model with its loss (positive pair accuracy, or diagonal and off-diagonal correlations, and the embedding standard deviation), without any extra computation.

### Step 3.18 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
Adds the metrics of the batch to the trackers of the interval and writes them to `Tensorboard/Train` at every iteration.

### Step 3.19 — `training_loss.backward()` + `step_optimizer()`
As for every task (see `classification_workflow.md`): the gradients of Step 3.14 are applied as they are.

### Step 3.20 — validation pass
Every 10 iterations, one validation batch runs through `run_model_iteration()` without gradient, with new random views, the model in evaluation mode.

**Example: the validation batch of iteration 10**
```text
info_nce_loss = 2.7081 (= ln 15)   positive_pair_accuracy = 0.0   embedding_standard_deviation = 0.0005
```
In evaluation mode the batch normalization uses its running statistics, which are not yet meaningful after 10 updates: all the embeddings are almost the same (standard deviation 0.0005), so every view is as similar to all the others and the loss is exactly the one of random embeddings. Step 4.9 explains it.

### Step 3.21 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()` + `report_dropped_batches()`
The sums of the interval are divided by the number of batches it ran. After the summary, the batches dropped since the last one because their loading failed are counted by split, only when there are some (see `classification_workflow.md`, Step 3.24).

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 10.8112, 'info_nce_loss': 10.8112, 'positive_pair_accuracy': 0.5625, 'embedding_standard_deviation': 2.8716}
after:  {'interval_batches': 4, 'total_loss': 2.7028, 'info_nce_loss': 2.7028, 'positive_pair_accuracy': 0.1406, 'embedding_standard_deviation': 0.7179}
MLflow, step 4: {'train/Total Loss': 2.7028, 'train/InfoNCE Loss': 2.7028, 'train/Positive Pair Accuracy': 0.1406,
                 'train/Embedding Standard Deviation': 0.7179, 'train/throughput': 5.3163,
                 'train/interval_seconds': 6.0192, 'learning_rate': 0.0005}
MLflow, step 12: train Total Loss 2.6904, Positive Pair Accuracy 0.0781, Embedding Standard Deviation 0.5425,
                 validation Total Loss 2.7081, Positive Pair Accuracy 0.0, Embedding Standard Deviation 0.0005,
                 throughput 11.2821, interval_seconds 2.8363
```

**Example output: iterations 8 and 12**
```text
========================================================================================================================
|     Iteration: 8     |      Total Images: 64     |    Interval Images: 32     |  Throughput: 2.8 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     2.8289    +0.1261
Moving Average of Train InfoNCE Loss                   :     2.8289    +0.1261
Moving Average of Train Positive Pair Accuracy         :     9.38 %    -4.69 %
Moving Average of Train Embedding Standard Deviation   :     0.6455    -0.0723
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:11.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
========================================================================================================================
...
========================================================================================================================
|    Iteration: 12     |      Total Images: 96     |    Interval Images: 32     |  Throughput: 11.3 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     2.6904    -0.1385 Best
Moving Average of Valid Total Loss                     :     2.7081
Moving Average of Train InfoNCE Loss                   :     2.6904    -0.1385 Best
Moving Average of Valid InfoNCE Loss                   :     2.7081
Moving Average of Train Positive Pair Accuracy         :     7.81 %    -1.56 %
Moving Average of Valid Positive Pair Accuracy         :     0.00 %
Moving Average of Train Embedding Standard Deviation   :     0.5425    -0.1030
Moving Average of Valid Embedding Standard Deviation   :     0.0005
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:02.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```
The positive pair accuracy is printed as a percentage with its change in percentage points, the losses and the embedding standard deviation with 4 decimals. At iteration 8 no metric beats its best value, so no line ends with "Best"; at iteration 12 the loss beats the 2.7028 of iteration 4. The interval of iteration 8 also holds the test evaluation of iteration 6, the trackers of an interval starting at the summary before it, hence its throughput of 2.8 images per second.

**Example output: the first interval of the Barlow Twins run** (throughput 12.2 Images Per Second; no rate, every value with 4 decimals)
```text
Moving Average of Train Total Loss                     :   131.7183
Moving Average of Train Barlow Twins Loss              :   131.7183
Moving Average of Train Diagonal Correlation           :     0.1465
Moving Average of Train Off-Diagonal Correlation       :     0.3666
Moving Average of Train Embedding Standard Deviation   :     0.7479
```

### Step 3.22 — end of the loop
The checkpoint of iteration 12 is already saved and evaluated inside its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 1000 Test Images · 2026-10-23 02:19 ───────────────      <- the evaluation of iteration 12, last in the iteration
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its test split (1000 images of 64×64 pixels, batch size 8), with `per_class_report: True`, `number_sample_images: 8` and `number_neighbours: 20`. They were captured before the per-class recall intervals were added (2026-11-01): `class_metrics.csv` now holds `recall_lower` and `recall_upper` after `recall`, and the report shows a `Per Class k-NN Recall` table after `Per Class k-NN Metrics` (Step 4.11). They were also captured when the k-NN compared every test image with the other test images (leave-one-out), before it compared them with reference training images (2026-11-11): the k-NN numbers below are those of the leave-one-out vote, the note of the k-NN now reads `k-NN : 20 neighbours among <M> labelled training images of <K> classes (50 per class at most)`, the legend of the per-class table now ends `of the labelled test images by labelled training images.`, and a second progress bar, `k-NN Reference Images`, follows the test loop (Step 4.1). They were also captured before the evaluation summary and curves (2026-11-18), with which every evaluation, after its report, adds the row of its iteration to `Evaluation/evaluation_summary.csv`, draws `evaluation_curves.png` again next to `Results.txt` and prints its link, `- Evaluation Curves : evaluation_curves.png`, after those of the files (Step 4.12).

### Step 4.1 — `run_test_evaluation()`
The evaluation loop of every task (see `classification_workflow.md`, Step 4.1). The test images are encoded as they are, without views, as in deployment. Once the split is encoded, the k-NN encodes the reference training images with the same weights, under a progress bar of its own, `k-NN Reference Images` (`(EMA Weights)` added for the averaged weights; Step 4.8).

**Example output: the progress bar** (1000 images, batch size 8)
```text
Test Evaluation Iteration 6: 100%|██████████| 125/125
```

### Step 4.2 — `create_evaluation_accumulator()`
The effective rank and the representation standard deviation are computed over the whole split, and the k-NN compares every labelled test image with the reference images once the split is encoded, so the representations of the whole split are kept on the CPU until the end of the loop.

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
Keeps the representations of the kept samples (`.detach().float().cpu()`), their labels and their identifiers. The embeddings are not kept: the representation is what is evaluated.

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
For each fixed test image of Step 2.11: the image next to two random views of it, captioned with its class (or `unlabelled`) and the cosine similarity of the representations of both views, which a backbone invariant to the views brings close to 1. The views are drawn from the seed `SAMPLE_VIEWS_SEED` (20000) plus the sample identifier, inside `torch.random.fork_rng()`, so that every evaluation compares the same two views and the random state of the training is left unchanged (Python's `random` state is saved and restored around the draw). `add_caption_bar()` writes the caption in a black bar of 24 pixels (`CAPTION_BAR_HEIGHT`) above the image and its views, padding them with gray on the right when the caption is wider.

**Example: the sample images of iteration 12**
```text
image and views: (64, 192, 3)  ->  with the caption bar: (88, 192, 3), wider for a long caption (88, 195, 3), (88, 196, 3)
captions:        'pentagon | Views Similarity 1.000', 'star | Views Similarity 1.000', 'plus_sign | Views Similarity 1.000'
files:           Evaluation/Iteration_12/Samples/test_000000.png, test_000001.png, test_000005.png ... (8 images of 192 to 196 × 88 pixels)
TensorBoard:     Test images Samples/test_000000, Samples/test_000001, Samples/test_000005, ...
```
A similarity of 1.000 here does not mean the backbone is invariant: Step 4.9 shows that every image gets almost the same representation at this point.

### Step 4.7 — `compute_evaluation_results()`
Concatenates the representations and the labels of the split, then computes:
- the **effective rank** (Step 4.10) and the **representation standard deviation** (Step 3.15 applied to the representations) of every test image, labelled or not;
- with at least one labelled test image, the **k-NN metrics** (Step 4.8), the headline metric being then the k-NN accuracy; without them, the k-NN is skipped and the headline metric is the effective rank (Step 5.6).

**Example: the global metrics of both checkpoints** (the `metrics` of the results, between 0 and 1)

| Iteration | k-NN Accuracy | k-NN Balanced Accuracy | k-NN Macro F1 | Effective Rank | Representation Standard Deviation |
|---|---|---|---|---|---|
| 6 | 0.1080 | 0.1053 | 0.1030 | 1.0141 | 0.0002 |
| 12 | 0.1210 | 0.1165 | 0.1113 | 1.0458 | 0.0008 |

```text
notes:         'Labelled Test Images : 1000 of 1000', 'k-NN : 20 neighbours among the other labelled test images',
               'Sample Images : 8 fixed test images'
class_scalars: 27 (k-NN Precision, Recall and F1 of the 9 classes)
```
With 9 classes, chance is about 0.11. The results list the three k-NN metrics in `percent_metrics` (only when the k-NN is computed), so that the report (Step 4.12) and the best checkpoint line write them as percentages (`12.10 %`), the effective rank and the standard deviation keeping 4 decimals; the per-class and recall tables list their rates in `percent_columns`.

### Step 4.8 — `compute_nearest_neighbour_results()` → `compute_nearest_neighbour_predictions()`
Classifies every labelled test image by its neighbours among the reference images, then scores it like a classifier (in `vision_training/evaluation/self_supervised/representation_metrics.py`):
- `encode_reference_images()` encodes the reference training images chosen at the start (Step 2.1, `build_reference_dataloader()`) with the weights being evaluated, trained or averaged, through `predict_batch()` like the test images. A test image is never compared with the other test images, so a test split of one image per class (ImageNet-22k) is scored like any other. Every class of the training split votes, a class without labelled test image being a possible wrong prediction, as for a classifier of every class.
- The representations are L2-normalized and compared by cosine similarity, in chunks of `SIMILARITY_CHUNK_SIZE` (1024) test images.
- The k = min(20, M) most similar reference images vote for their class with the weight exp(similarity / `NEAREST_NEIGHBOUR_TEMPERATURE`), the temperature being 0.07, the votes being summed per class (`scatter_add`); the class with the most votes is the prediction.
- `compute_confusion_matrix()` and `compute_metrics_from_confusion_matrix()` give the accuracy, the balanced accuracy, the macro F1 and the precision, recall and F1 of every class, from the non-zero cells of the matrix only above 1,000 classes (`MAXIMUM_DENSE_CONFUSION_CLASSES`, e.g. the 21,841 classes of ImageNet-22k, whose dense matrix would take 3.8 GB), with the same values (`classification_workflow.md`, Step 4.9); `compute_clopper_pearson_intervals()` of `vision_training/evaluation/recall_intervals.py` gives the exact (Clopper-Pearson) 95 % confidence interval of every recall from the true positives and the labelled test images of the class, the images being independent (`classification_workflow.md`, Step 4.14); `write_class_metrics_file()` saves them in `class_metrics.csv`, the bounds as `recall_lower` and `recall_upper` after `recall`.

**Example: image 0 of the test split at iteration 12** (label 6, plus_sign; captured with the leave-one-out vote, its neighbours being other test images)
```text
top 20 similarities: all 1.0
neighbour labels:    [2, 7, 4, 5, 6, 8, 2, 5, 5, 7, 2, 2, 0, 0, 6, 3, 3, 4, 4, 5]
votes per class:     [3.20e6, 0, 6.40e6, 3.20e6, 4.80e6, 6.40e6, 3.20e6, 3.20e6, 1.60e6]   (exp(1 / 0.07) = 1.6e6 per neighbour)
prediction:          2 (triangle), wrong
accuracy over the split: 0.121   samples per class: [91, 118, 125, 97, 99, 117, 126, 113, 114]
```
With all similarities at 1.0, every neighbour weighs the same and the vote is a plain count among neighbours chosen almost at random: here triangle and hexagon tie with 4 neighbours each, and `argmax` keeps the first one.

**Example: `class_metrics.csv` of iteration 12** (first rows)
```text
class_index,class,samples,precision,recall,f1
0,square,91,0.100000,0.065934,0.079470
...
```

### Step 4.9 — why the representations look collapsed
The test images are encoded in evaluation mode, so the batch normalization layers use their running statistics instead of the statistics of the batch. After 12 iterations they have been updated only 12 times (`num_batches_tracked` 12) at momentum 0.1, so they are still far from the statistics of the data (the running variance of the last backbone layer averages 0.30), and the normalized features of all images end up nearly identical:

**Example: one test batch at iteration 12, in both modes**
```text
evaluation mode (running statistics):  mean pairwise cosine 1.0     feature standard deviation 2.2e-5
training mode (batch statistics):      mean pairwise cosine 0.774   feature standard deviation 0.114
```
This explains the effective rank near 1, the views similarity of 1.000, the validation loss of exactly ln 15 and the k-NN near chance. The representations being positive after the ReLU and the pooling of the backbone also brings their directions closer. A longer training gives the running statistics time to settle.

### Step 4.10 — `compute_effective_rank()`
The RankMe effective rank of the `[N, F]` representations: the singular values are normalized into a distribution p, and the effective rank is the exponential of its entropy (`EFFECTIVE_RANK_EPSILON` 1e-12 avoiding log 0, NaN when the result is not finite). It goes from 1, when all representations lie along one direction, to F when they use every direction equally.

**Example: iteration 12**
```text
representations:  [1000, 128] in [0, 0.084], mean 0.0171
singular values:  9.2822, 0.0077, 0.0033, 0.0026, 0.0019, ...   (sum 9.3237)
p_1 = 9.2822 / 9.3237 = 0.99555   entropy = 0.0448   effective rank = exp(0.0448) = 1.0458
```

### Step 4.11 — the per-class table and scalars
The table "Per Class k-NN Metrics" lists the classes with their number of labelled test images, precision, recall and F1, the worst F1 first. The table "Per Class k-NN Recall" (`build_recall_interval_table()`) follows it: the number of labelled test images, the recall and the bounds of its interval (`95 % Lower`, `95 % Upper`) of every class, the worst recall first. Both tables write their rates as percentages (`15.32 %`), the number of labelled test images as a count. Each class also gives three TensorBoard and MLflow scalars, `k-NN Precision/<class>`, `k-NN Recall/<class>` and `k-NN F1/<class>`.

**Example: `Evaluation/Iteration_12/class_metrics.csv` and the scalars of one class** (the table itself is in the block of Step 4.12)
```text
class_index,class,samples,precision,recall,f1
0,square,91,0.100000,0.065934,0.079470
1,circle,118,0.153153,0.144068,0.148472
2,triangle,125,0.150685,0.264000,0.191860
3,rectangle,97,0.121951,0.103093,0.111732
...
8,octagon,114,0.087379,0.078947,0.082949

Tensorboard/Test tags of circle:  k-NN Precision/circle, k-NN Recall/circle, k-NN F1/circle
k-NN Precision/circle:            step 6 0.1017, step 12 0.1532
```
The file keeps the class order of the label dictionary, while the console table sorts the classes by F1.

### Step 4.12 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
Whenever a checkpoint is evaluated, its block goes to the console and `Results.txt`, its scalars and sample images to TensorBoard, its metrics and `class_metrics.csv` to MLflow, then `update_evaluation_curves()` writes its row into `Evaluation/evaluation_summary.csv` and draws `evaluation_curves.png` again from every row.

**Example output: the block of iteration 12** (the same text is appended to `Results.txt`; the three file names at the end are clickable links to the files)
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
   square           91     10.00 %    6.59 %    7.95 %
   octagon         114      8.74 %    7.89 %    8.29 %
   plus_sign       126     10.14 %   11.11 %   10.61 %
   rectangle        97     12.20 %   10.31 %   11.17 %
   hexagon         117     12.17 %   11.97 %   12.07 %
   star            113     12.31 %   14.16 %   13.17 %
   circle          118     15.32 %   14.41 %   14.85 %
   triangle        125     15.07 %   26.40 %   19.19 %
  k-nearest neighbour classification (20 neighbours) of the labelled test images by the other ones. Worst F1 first, ties
  in class order. — : no labelled test image of the class.
  - Results : Results.txt
  - Class Metrics : class_metrics.csv
  - Sample Images : Samples
```

**Example: the run folder, TensorBoard and MLflow after the two evaluations**
```text
runs/SSL_IN-KAR-64_..._SIMCLR-T0.2/
    data.json  label_dictionary.json  mlflow_run.json  Results.txt  training_configuration.yaml
    Evaluation/Iteration_6/ and Iteration_12/   class_metrics.csv  Samples/ (8)
    Tensorboard/Train, Validation, Test
    Weights/Iteration_3, 6, 9, 12/  full-checkpoint
Tensorboard/Test: 32 scalars (5 global + 27 per class) at steps 6 and 12, 8 images Samples/test_...
    scalar k-NN Accuracy    6: 0.108, 12: 0.121
    scalar Effective Rank   6: 1.0141, 12: 1.0458
MLflow (runs/mlflow.db): 44 metric keys, 91 rows: train/*, validation/*, test/<global metric>, test/best_k-NN Accuracy,
        27 test_class/k-NN <Metric>/<class>, learning_rate
    parameters: the flattened configuration, e.g. Data.data_augmentation.types.random_crop.probability = 1.0
    tags: best_iteration 12, git_commit, git_dirty, host, pretrained_backbone None, run_folder, run_name, task
    artifacts: configuration/training_configuration.yaml, data/data.json, evaluation/iteration_6/ and iteration_12/
               class_metrics.csv, metric_names.json, Results.txt
```
A run started after 2026-11-18 also holds `Input_Samples/` (Step 2.21), `evaluation_curves.png` and `Evaluation/evaluation_summary.csv`, the last two written again after every evaluation, and sends all three to MLflow.

With the moving average of the weights on (`Optimization.exponential_moving_average`), the averaged weights are evaluated the same way after the trained ones, in `Evaluation/Iteration_<n>/EMA/` and `Tensorboard/Test_EMA`, with the same sample views.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> Like those of Stage 4, the k-NN outputs of this stage were captured when the k-NN compared every test image with the other test images; the k-NN now compares them with reference training images, and its note names them.

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 200 test images (`test_200`). A second evaluation of the checkpoint of iteration 12 uses the same 200 images without their categories (`test_unlabelled_200`). They were captured before the per-class recall intervals were added (2026-11-01), whose `Per Class k-NN Recall` table now follows `Per Class k-NN Metrics` in the labelled blocks, and when the script wrote into `evaluations/<run name>/` next to the run: the link of Step 5.5 and the evaluation folder below are written for the current `<run>/Checkpoint_Evaluation/` folder, which also holds `evaluation_curves.png` since 2026-11-18.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5), configured by the variables of its main block and run with `uv run python -m scripts.evaluation.evaluate_checkpoint`: set `run_name` to the self-supervised run and `experiment_folder` to its run folder, `iteration` to `None` (the last checkpoint), an integer or `"all"`, and `test_data_folder` to another split if needed. The self-supervised configurations have no `evaluation.error_dumps` section, so `maximum_error_images_per_class_pair` is skipped with a yellow warning: the self-supervised evaluation saves no error image. `save_predictions`, `target_precisions`, `nms_criterion` and `nms_distance_exponent`, when set, are skipped the same way, the task having no `evaluation.save_predictions` setting (no `predictions.csv`), neither class thresholds nor a detector. A run saved before the k-NN took reference training images lacks `evaluation.number_reference_images_per_class` in its `training_configuration.yaml`, which the checks refuse (`Missing configuration key 'ExperimentConfiguration.evaluation.number_reference_images_per_class'`): add the key to it before evaluating it again. The split does not need labels: without them, only the effective rank and the representation standard deviation are computed (Step 5.6).

### Step 5.2 — trainer construction
The script loads `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it, points it to the `Checkpoint_Evaluation/` folder of the run with `point_to_checkpoint_evaluation()` (`experiment_folder` set to the folder holding the run, `run_name` to the name of the run folder and `standalone_evaluation_folder` to `Checkpoint_Evaluation`, so that the files go to `<run name>/Checkpoint_Evaluation/`, or to `<run name>/Checkpoint_Evaluation_split_<parent folder>_<split folder>/` for a split other than the test split of the run, such as the `test_200` folder of the example below), the evaluated split and the batch size, calls `disable_pretrained_settings()` (nothing to change here, the run had no pretrained backbone) and builds a `SelfSupervisedTrainer` inside `read_manifests_once()`, as `main()` does (Step 1.1).

**Example output** (shortened)
```text
- Test Data Folder        : test_200 (200 images, 9 categories)
- Iterations Per Epoch    : 31250
- Batch Size              : 16
- MLflow Tracking         : sqlite:///.../runs/mlflow.db
```
The panel names the backend of the evaluated run, where the evaluation is tracked (Step 5.4): the script moves the trainer to the `Checkpoint_Evaluation/` folder of the run, but `tracking_root` keeps the experiments root folder of the run.

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
The MLflow run of the evaluation, named `<run name>-evaluation`, is created in the backend of the evaluated run and tagged with the training run it evaluates.

**Example output**
```text
MLflow Run Created In The Experiment 'self_supervised' : fea7baf32e174d2eb2c5f3583ad3cd74 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
tags: evaluation_of a42373adfbf94d839fa79699e0103b2b, evaluated_iterations "12, 9, 6, 3", test_data_folder, best_iteration 3
metrics: 33 keys, 132 rows
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
So that the checkpoint to keep can be chosen, every evaluation adds the row of its iteration to `Checkpoint_Evaluation/evaluation_summary.csv`, the rows in increasing iteration order with those of the previous launches into the folder, and draws `Checkpoint_Evaluation/evaluation_curves.png` again (`update_evaluation_curves()`); the best checkpoint of the k-NN accuracy is printed after the last one.

**Example output** (the rows of the captured summary are shown in the increasing iteration order written since 2026-11-18, with the `Evaluation Curves` line printed since then)
```text
Evaluation Summary Of 4 Checkpoints : evaluation_summary.csv          <- a clickable link to SSL_IN-KAR-64_..._SIMCLR-T0.2/Checkpoint_Evaluation/evaluation_summary.csv
  - Evaluation Curves : evaluation_curves.png
  - Best Checkpoint : iteration 3 (k-NN Accuracy 12.50 %)

iteration,k-NN Accuracy,k-NN Balanced Accuracy,k-NN Macro F1,Effective Rank,Representation Standard Deviation, ...
3,0.125,0.1118...,0.1045...,1.0072...,0.000103..., ...
6,0.095,0.0848...,0.0840...,1.0131...,0.000190..., ...
9,0.075,0.0631...,0.0522...,1.0232...,0.000390..., ...
12,0.11,0.09638447971781304,0.08247118900298986,1.0432647447459944,0.0008116378448903561, ...
```
The file has 33 columns: the iteration, the 5 global metrics and the 27 per-class metrics.

**Example: the evaluation folder**
```text
SSL_IN-KAR-64_..._SIMCLR-T0.2/Checkpoint_Evaluation/
    data.json  label_dictionary.json  Results.txt  training_configuration.yaml
    evaluation_summary.csv  evaluation_curves.png
    Iteration_3, 6, 9, 12/   class_metrics.csv  Samples/
    Tensorboard/Test
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 12 on the 200 images**
```text
-----------------------------
Evaluating : model_000012.pt
-----------------------------
Iteration 12 · 200 Test Images · 2026-10-23 02:19 ──────────────────────────────────────────────────────────────────────
  k-NN Accuracy                       11.00 %   k-NN Balanced Accuracy   9.64 %
  k-NN Macro F1                       8.25 %    Effective Rank           1.0433
  Representation Standard Deviation   0.0008
  Labelled Test Images : 200 of 200
  k-NN : 20 neighbours among the other labelled test images
  Sample Images : 8 fixed test images

  Per Class k-NN Metrics
   Class       Samples   Precision    Recall        F1
  ─────────────────────────────────────────────────────
   square           16      0.00 %    0.00 %    0.00 %
   rectangle        14      0.00 %    0.00 %    0.00 %
   pentagon         23      0.00 %    0.00 %    0.00 %
   octagon          21      4.35 %    4.76 %    4.55 %
   circle           20      8.33 %    5.00 %    6.25 %
   hexagon          27     11.11 %   11.11 %   11.11 %
   plus_sign        27     11.54 %   11.11 %   11.32 %
   triangle         28     16.22 %   21.43 %   18.46 %
   star             24     17.02 %   33.33 %   22.54 %
  ...
```

**Example: the global metrics of the four checkpoints** (as in `evaluation_summary.csv`, between 0 and 1)

| Iteration | k-NN Accuracy | k-NN Balanced Accuracy | k-NN Macro F1 | Effective Rank | Representation Standard Deviation |
|---|---|---|---|---|---|
| 12 | 0.1100 | 0.0964 | 0.0825 | 1.0433 | 0.0008 |
| 9 | 0.0750 | 0.0632 | 0.0523 | 1.0233 | 0.0004 |
| 6 | 0.0950 | 0.0848 | 0.0840 | 1.0132 | 0.0002 |
| 3 | 0.1250 | 0.1118 | 0.1045 | 1.0073 | 0.0001 |

The effective rank and the spread of the representations grow with the training, as the running statistics of the batch normalization settle (Step 4.9). The k-NN accuracy of these almost collapsed representations only moves around chance (0.11) on 200 images, which makes iteration 3 the best checkpoint here.

**Example output: the same 200 images without labels**
```text
Samples Without A Single Mapped Category, Kept Without Class :        <- yellow, once the loaders are built
  - Test : 200 samples (labels.json)
...
Iteration 12 · 200 Test Images · 2026-10-23 02:20 ──────────────────────────────────────────────────────────────────────
  Effective Rank   1.0433   Representation Standard Deviation   0.0008
  Labelled Test Images : 0 of 200
  k-NN : skipped, no labelled test image
  Sample Images : 8 fixed test images
  - Results : Results.txt
  - Sample Images : Samples
-------------------------------------------------------------
Evaluation Summary Of 1 Checkpoints : evaluation_summary.csv
-------------------------------------------------------------
  - Evaluation Curves : evaluation_curves.png
  - Best Checkpoint : iteration 12 (Effective Rank 1.0433)
```
The images are the same as in `test_200`, so the effective rank is the same; without labels there is no class table nor `class_metrics.csv`, the sample images are captioned `unlabelled`, and the MLflow run of the evaluation has 3 metric keys. Since 2026-11-18 the evaluation summary of the folder keeps the rows of the previous launches: written into the same `Checkpoint_Evaluation/` folder as the labelled evaluation, the row of iteration 12 replaces the labelled one, its k-NN cells left empty, and the rows of iterations 3, 6 and 9 stay.
