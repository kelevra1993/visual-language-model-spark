# Tutorial Backlog

Before rebuilding or extending the tutorials, this list tells which implemented features the videos of `tutorials/` do not show yet, or show only in part, with where the feature is documented and where a scene could take it.

> This list is not exhaustive: it only holds the features added to it when they were implemented, so a feature missing from it may still be missing from the tutorials. An entry is removed once a rebuilt chapter shows the feature.

## Contents
To find your way in the backlog, the sections are:

- [Adding An Entry](#adding-an-entry)
- [Speed Benchmark Diagnosis Mode](#speed-benchmark-diagnosis-mode)
- [Deeper Backbone From A Shorter Pretrained Backbone](#deeper-backbone-from-a-shorter-pretrained-backbone)
- [Sparse Confusion Matrix Above 1,000 Classes](#sparse-confusion-matrix-above-1000-classes)
- [Docker Image Of The Trainer](#docker-image-of-the-trainer)

## Adding An Entry
Whenever a change lands on `main` with a behaviour the tutorials do not show, add a section with these four lines, then remove it once the chapter showing it is rebuilt:

- **Added**: the merge commit on `main` and its date.
- **Documented in**: the README section and the reference document describing it.
- **Tutorial status**: what the storyboards and the built videos say about it today.
- **Where it would fit**: the series and chapter that could show it, and what the scene needs (a real run, a figure, an idle GPU).

## Speed Benchmark Diagnosis Mode
To be able to compare the training and inference speeds of a configuration before a long run, `ExperimentConfiguration.diagnosis_modes.benchmark_speed: True` times the inference of every format (PyTorch evaluation, PyTorch inference model, TorchScript, ONNX Runtime CUDA, ONNX Runtime TensorRT FP32 and FP16) and the training step on one training batch tiled to any batch size, searches the largest batch sizes that fit up to 2048 while keeping a margin of memory free (40 GB on the integrated GPU of the DGX Spark, 2 GB on a discrete GPU), prints the report and exits without training (`vision_training/trainers/speed_benchmark.py`, `BaseTrainer.benchmark_speed`).

- **Added**: `main` 8744426, 2026-11-16.
- **Documented in**: [README, Augmentation And Diagnosis Modes](../README.md#augmentation-and-diagnosis-modes) and [training reference, Benchmarking The Speed](reference/training_reference.md#benchmarking-the-speed).
- **Tutorial status**: not mentioned in any storyboard. The other diagnosis modes are shown: `view_input_images` in `01_foundations/03_data_pipeline.yaml`, `compute_receptive_field` in `01_foundations/04_backbone_and_modules.yaml`, `compute_anchor_coverage` in `03_object_detection/01_data_levels_and_anchors.yaml`.
- **Where it would fit**: the foundations series, next to the other diagnosis modes or in `01_foundations/06_after_the_training.yaml` with the export, showing the report of a real run. The run needs an idle machine (on the DGX Spark the benchmark refuses to start unless more than 40 GB are free) and `onnxruntime-gpu` for the ONNX Runtime and TensorRT rows, which the environment of the repository does not have yet.

## Deeper Backbone From A Shorter Pretrained Backbone
If one wishes to start a deeper backbone from a shorter trained one, `Backbone.pretrained` loads the source backbone into the first blocks of the backbone of the run, the blocks after them keeping their random initialisation and learning at the learning rate of the run; `freeze`, `freeze_normalization_parameters` and `learning_rate_multiplier` apply to the loaded blocks only, and every checkpoint stores their number (`pretrained_backbone_blocks`) so that a resumed run freezes the same blocks once its source run is gone (`vision_training/models/model_transfer.py`, `BaseTrainer.get_transferred_part`).

- **Added**: `main` 4a76053, 2026-11-17.
- **Documented in**: [README, Transfer Between Runs](../README.md#transfer-between-runs) and [training reference, Transfer Between Runs](reference/training_reference.md#transfer-between-runs).
- **Tutorial status**: the narration of the "Starting from a trained run" scene of `01_foundations/04_backbone_and_modules.yaml` and the transfer box and sentence of `11_self_supervised/03_judging_without_a_classifier.yaml` were updated in the storyboards to say that a deeper backbone keeps its extra blocks random, but these chapters are not rebuilt yet. No scene shows that freezing, the fixed normalization and the multiplier cover the loaded blocks only, the `Loaded Blocks` line of the console, or the resume without the source run.
- **Where it would fit**: the "Starting from a trained run" scene of the foundations backbone chapter, with the console of a run of 6 blocks started from a run of 4 (the case of `tests/trainers/test_backbone_transfer_training.py`), then a rebuild of the foundations and self-supervised series.

## Sparse Confusion Matrix Above 1,000 Classes
Should an evaluation cover more than 1,000 classes (`MAXIMUM_DENSE_CONFUSION_CLASSES`, e.g. the 21,842 classes of imagenet-merged), the classification and k-NN evaluations count only the non-zero cells of the confusion matrix, with the same metrics, write them to `confusion_cells.csv` (`true_class,predicted_class,count`) instead of `confusion_matrix.csv`, and draw no confusion image.

- **Added**: `main` b45c72b, 2026-11-17.
- **Documented in**: [README, Evaluating Checkpoints](../README.md#evaluating-checkpoints) and [configuration reference, Evaluation Outputs](reference/configuration_reference.md#evaluation-outputs).
- **Tutorial status**: not shown; the classification chapters use the 9 classes of the synthetic shapes and their dense matrix.
- **Where it would fit**: one sentence in the "Confusions over the training" scene of `02_classification/02_judging_and_shipping.yaml`, no run needed.

## Docker Image Of The Trainer
In order to train in a container, on the amd64 GPU pods of a cloud Kubernetes cluster or on the arm64 DGX Spark, the multi-stage `Dockerfile` of the repository root builds the trainer image.

- **Added**: `main` f184436, 2026-11-16.
- **Documented in**: [README, Docker Image](../README.md#docker-image).
- **Tutorial status**: not shown; `01_foundations/01_repository_and_first_run.yaml` installs with `uv sync` only, and `06_after_the_training.yaml` uses Docker for Triton only.
- **Where it would fit**: one scene after the `uv sync` scene of `01_foundations/01_repository_and_first_run.yaml` (stages, build and run commands), no run needed.
