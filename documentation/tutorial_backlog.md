# Tutorial Backlog

Before rebuilding or extending the tutorials, this list tells which implemented features the videos of `tutorials/` do not show yet, or show only in part, with where the feature is documented and where a scene could take it.

> This list is not exhaustive: it only holds the features added to it when they were implemented, so a feature missing from it may still be missing from the tutorials. An entry is removed once a rebuilt chapter shows the feature.

## Contents
To find your way in the backlog, the sections are:

- [Adding An Entry](#adding-an-entry)
- [Speed Benchmark Diagnosis Mode](#speed-benchmark-diagnosis-mode)
- [Report Figures Of A Kept Classifier](#report-figures-of-a-kept-classifier)
- [Results Of The Finer Inputs Of The Mask Heads](#results-of-the-finer-inputs-of-the-mask-heads)
- [ImageNet-22k Download](#imagenet-22k-download)
- [Converting Configurations After A Removed Setting](#converting-configurations-after-a-removed-setting)

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

## Report Figures Of A Kept Classifier
Should a research report need the figures of a kept classifier again, `scripts/evaluation/draw_report_figures.py` writes them into `Checkpoint_Evaluation/Iteration_<n>[/EMA]/Report_Figures/` from the run folder: the confusion matrix, the precision-recall curves at the saved temperature, the training curves with the kept checkpoint, and `report_values.json`.

- **Added**: `main` 1e6daec, 2026-11-14.
- **Documented in**: [README, Evaluating Checkpoints](../README.md#evaluating-checkpoints) and [tools reference, Drawing The Figures Of A Research Report](reference/tools_reference.md#drawing-the-figures-of-a-research-report).
- **Tutorial status**: not mentioned in any storyboard; `02_classification/02_judging_and_shipping.yaml` stops at the temperature explorer and the export.
- **Where it would fit**: one scene after the explorer scenes of `02_classification/02_judging_and_shipping.yaml`, drawn from the classification tutorial run, leaving out the trucks-seen-or-unseen panel (it needs a manifest of image groups); no new training.

## Results Of The Finer Inputs Of The Mask Heads
In order to show what the finer inputs of the instance segmentation mask heads bring (U-Net skips of backbone blocks before their pooling, an image branch on the crop of the input image, fusion convolutions), a chapter needs the results of stage 3b of the instance segmentation benchmark and the design figure of `scripts/models/instance_segmentation/draw_mask_head_designs.py`.

- **Added**: `main` 0ef5cbd, 2026-11-15; skip rules `main` d536467, 2026-11-18; design A recorded `main` 45fbc73, 2026-11-17; design B recorded `main` e493e63, 2026-11-18.
- **Documented in**: [tasks reference](reference/tasks_reference.md), [configuration reference](reference/configuration_reference.md) and stage 3b of the [instance segmentation plan](architecture_benchmarking/instance_segmentation_experiment_plan.md).
- **Tutorial status**: `04_instance_segmentation/01_from_boxes_to_masks.yaml` names the settings, off in the run shown, and states the skip rules; `04_instance_segmentation/02_scoring_and_shipping_masks.yaml` says in one sentence, without figure or number, that the image branch and the backbone skips both drew finer borders. No chapter shows the stage 3b figures or the design figure (design A reaches a mask mAP of 0.8865 and design B 0.8859, a tie, against 0.8145 for the heads of stage 3).
- **Where it would fit**: the mask head scene of `04_instance_segmentation/01_from_boxes_to_masks.yaml` or the benchmark sentence of chapter 2, with the stage 3b figures and `mask_head_designs.png`, once the control is run (user decision); no new tutorial run.

## ImageNet-22k Download
Should you need the pretrained backbones of the classification or self-supervised example configurations, `vision_training/data/classification/imagenet_download.py` downloads ImageNet-22k (`timm/imagenet-22k-wds` at a pinned revision, 21,841 classes).

- **Added**: `main` a8a8ff2, 2026-11-11.
- **Documented in**: [README, ImageNet-22k](../README.md#imagenet-22k).
- **Tutorial status**: no storyboard mentions ImageNet or the script; the tutorials train on synthetic data only.
- **Where it would fit**: one terminal scene after the `uv sync` scene of `01_foundations/01_repository_and_first_run.yaml`; no download needed.

## Converting Configurations After A Removed Setting
Whenever a launch is refused with `'<key>' was removed: <what replaced it>. Delete it from the configuration file.`, `scripts/configuration_processing/convert_fixed_settings.py` removes the fixed settings from the older configuration files and run folders and adds the settings a task now requires, with a backup of every converted file.

- **Added**: `main` 8b5e190, 2026-10-29.
- **Documented in**: [configuration reference, Fixed Settings](reference/configuration_reference.md#fixed-settings).
- **Tutorial status**: no storyboard mentions the script, the refusal message or `REMOVED_SETTINGS`.
- **Where it would fit**: one terminal scene after the "One file defines the run" scene of `01_foundations/01_repository_and_first_run.yaml`, with `write_files = False` listing the files to convert; no training.
