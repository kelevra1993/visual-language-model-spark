# Bugs And Standards

> Audit of the whole repository carried out on 2026-10-20 at commit `165339b`, to go through together. Only the items
> still open are kept: the fixed ones and those dropped by the user are deleted from this file.
> Nine reviewers each read one area end to end (configuration, data generation, data loading, core
> models, detection and keypoint models, training loop, task trainers and losses, evaluation and tracking, deployment
> and scripts) and reproduced their findings with small scripts on the synthetic datasets, the dummy task or hand-made
> inputs. Every medium and high finding was then read again in the code. The coding standards of the global
> `CLAUDE.md` and of the project memory were checked by a script over the 266 Python files of `vision_training/`,
> `scripts/` and `tests/`, then by reading every hit.
>
> The items of the [Repository Audit For Instance Segmentation](suggestions.md#repository-audit-for-instance-segmentation)
> (2026-10-10, commit `84ac325`) are not repeated here: [Items Of suggestions.md Still Open](#items-of-suggestionsmd-still-open)
> gives the status of each open one, with what this audit adds to it.
>
> **Identifiers:** B1 to B59 for the bugs, ordered by severity, then along the pipeline (configuration, data,
> models, training, evaluation, deployment).
>
> **Severity:** **high** = a supported launch, training or evaluation crashes, or silently trains or reports something
> other than what was asked. **medium** = wrong or misleading results, or lost work, in some supported settings.
> **low** = edge cases, misleading output, documentation and duplicated code.
>
> **Status:** **confirmed** = reproduced, or read end to end in the code. **likely** = the mechanism was read in the
> code but not run.

## Table Of Content
- [Summary](#summary)
- [Bugs](#bugs)
  - [Medium](#medium)
  - [Low](#low)
- [Items Of suggestions.md Still Open](#items-of-suggestionsmd-still-open)
- [Coding Standards](#coding-standards)
  - [Respected Everywhere](#respected-everywhere)
  - [Accepted Exceptions](#accepted-exceptions)
- [Suggested Order](#suggested-order)
- [How It Was Checked](#how-it-was-checked)

## Summary

To find an open bug by area, the table links each one: 3 medium and 16 low bugs are open.

| Area | Medium | Low |
|---|---|---|
| Configuration |  | [B22](#b22-read_json-swallows-json-parse-errors-and-returns-none), [B24](#b24-stale-run-name-and-checker-documentation) |
| Data generation |  | [B28](#b28-comments-left-from-the-top-down-keypoint-detector-contradict-the-code), [B29](#b29-the-readme-gives-the-wrong-number-of-objects-per-synthetic-keypoint-image) |
| Core models | [B8](#b8-a-causal-sequence-labeller-or-frame-selector-with-sequence-normalization-uses-the-future-frames) | [B32](#b32-backbonecompute_detection_input_information-dry-runs-in-training-mode-with-gradients), [B33](#b33-model-builders-silently-fall-back-on-misspelled-option-strings) |
| Detection and keypoint models | [B10](#b10-a-keypoint-head-transfer-compares-only-the-class-names-so-keypoints-in-another-order-land-on-the-wrong-channels) | [B35](#b35-the-eager-evaluation-ignores-the-valid-proposal-mask-so-padding-copies-can-become-duplicate-detections), [B37](#b37-get_box_dimensions_and_centers-ends-the-process-with-exit-and-status-0-on-a-wrong-box-shape), [B39](#b39-hrnet-with-feature_map_normalization-none-builds-every-convolution-without-bias), [B40](#b40-the-batch_assign_targets_to_anchors-docstring-promises-a-positive-anchor-for-every-ground-truth), [B42](#b42-training-and-validation-forwards-of-the-detector-do-inference-only-work-with-one-host-synchronisation-per-image) |
| Task trainers and losses |  | [B49](#b49-an-evaluation-score-threshold-above-the-deployment-threshold-silently-removes-detections-from-the-confusion-matrix-and-the-images) |
| Evaluation and tracking |  | [B54](#b54-a-standalone-evaluation-is-tracked-in-another-mlflow-database-than-the-one-the-launch-panel-prints) |
| Deployment and scripts | [B19](#b19-standalone-evaluations-of-another-split-overwrite-and-mix-with-the-earlier-ones-in-the-same-output-folder) | [B57](#b57-a-run-without-checkpoint-registry-is-evaluated-on-nothing-and-the-script-ends-normally), [B58](#b58-a-misspelled-export-format-is-silently-exported-as-onnx-and-its-tensorrt-setting-dropped), [B59](#b59-load_inference_inputs-claims-the-preprocessing-of-the-loaders-but-a-tiled-segmentation-run-gets-a-whole-image-squeezed-into-one-tile) |
| **Total** | **3** | **16** |

## Bugs

### Medium
#### B8. A causal sequence labeller or frame selector with sequence normalization uses the future frames
- **Area:** core models · **Severity:** medium · **Status:** confirmed (measured by the reviewer: outputs change when
  later frames change) · **Category:** bug
- **Where:** `vision_training/models/components/sequence_normalization.py:70-104`,
  `models/sequence/frame_encoder.py:43-55`; the checker refuses causal plus bidirectional (`sequence_checks.py:92-99`)
  but not causal plus sequence normalization; README line 670; the causality test
  (`tests/models/sequence/test_sequence_labeller.py:45-60`) only uses batch normalization.
- **Problem:** sequence normalization takes the mean and variance of every channel over all frames of the sequence, in
  training and inference, in every backbone layer, so every frame depends on the frames after it whatever the causal
  mask or unidirectional GRU of the temporal module (README: with `causal: True` "every frame only sees the frames
  before it, as in a live stream").
- **Failure scenario:** a causal M2M or SEL run with `feature_map_normalization: "sequence"`: the test evaluation
  overstates the live-stream accuracy, and on a growing stream the earlier frames get new predictions at every new frame
  (random weights: changes of 1e-3 to 7e-3).
- **Fix:** refuse causal (and unidirectional recurrent) models with sequence normalization in
  `check_temporal_model_configuration`, or use past-only cumulative statistics; add this case to the causality test.

#### B10. A keypoint head transfer compares only the class names, so keypoints in another order land on the wrong channels
- **Area:** detection and keypoint models · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer;
  code read again) · **Category:** bug
- **Where:** `vision_training/models/model_transfer.py:975-981` (class name check, skipped when either side has no
  names), `498-537` (`compare_keypoint_heads`, docstring 505-507); `models/keypoints/keypoint_detector.py:57-89`
  (channels built from `Data.keypoint_categories`); `base_trainer.py:828` (the saved YAML holds no keypoint categories).
- **Problem:** the heatmap and tag channels follow the keypoint lists of the classes, read from the training
  labels.json, but the transfer only compares the class names and the widths: the same number of keypoints in another
  order, other keypoints, or the same total split differently between classes give tensors of the same shapes, copied as
  is. The docstring says this case is "checked on the class names of the runs and on the weights themselves".
- **Failure scenario:** source triangle keypoints `[top, left, right]`, target `[left, right, top]`, same class names:
  "Keypoint Head Initialised", and channel 0 trained on triangle.top is read as triangle.left.
- **Fix:** record `KeypointDetector.channel_names` in the data record or the checkpoint and refuse a transfer when they
  differ (or permute the rows by name); refuse instead of skipping when the names are unknown.

#### B19. Standalone evaluations of another split overwrite and mix with the earlier ones in the same output folder
- **Area:** deployment and scripts · **Severity:** medium · **Status:** likely (traced by the reviewer, not run) ·
  **Category:** bug
- **Where:** `scripts/evaluation/evaluate_checkpoint.py:40, 47-52`; `base_trainer.py:137-140, 1747, 1772-1773`;
  `classification_trainer.py:366, 684`; `scripts/evaluation/collect_experiment_results.py:135-137`.
- **Problem:** the output folder is always `<output_folder>/<run name>` whatever `test_data_folder` is (README: "another
  labelled split" is the main use) and nothing clears the evaluation folder: thresholds, class metrics and data.json are
  rewritten, error images are added into the same `Errors/<TRUE>-TO-<PREDICTED>/` folders, and a second TensorBoard Test
  value is written at the same step.
- **Failure scenario:** a run evaluated on its test split, then on another split: the error folders mix both splits and
  `compare_best_evaluations.py` silently reports whichever split ran last.
- **Fix:** include the evaluated split in the output folder name, or refuse a folder whose data.json records another
  split; clear `Errors/` of the iteration first.

### Low
#### B22. read_json swallows JSON parse errors and returns None
- **Area:** configuration · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) · **Category:**
  error-handling
- **Where:** `vision_training/utilities/os_utilities.py:26-45`; callers `base_trainer.py:893, 941`, `data_record.py:105,
  226`, `split_description.py`, `model_transfer.py:626`.
- **Problem:** `JSONDecodeError` is caught and None returned, every caller then indexes the result, losing the file name
  and the parse position.
- **Failure scenario:** a truncated data.json makes the resume fail with `TypeError: 'NoneType' object is not
  subscriptable`.
- **Fix:** let the error propagate, or raise it again with the path.

#### B24. Stale run name and checker documentation
- **Area:** configuration · **Severity:** low · **Status:** confirmed (the reviewer built every run name) ·
  **Category:** documentation
- **Where:** README lines 1293-1306 and 1175; the header comments of
  `classification/configuration_synthetic_shapes.yaml:15` and `classification/configuration_example.yaml:11`;
  `configuration_loader.py:31-44`.
- **Problem:** the README run name examples "built from the configurations of the repository" are stale (CLS shows
  `HAVG.128_MC4`, the file gives `HFLAT.128-HLN-CWB` with the class count of the mapping on disk; M2O and CMP show `T8`,
  the files give `T50`), and so are the examples in the header comments of the two classification files (the example
  file shows `B1x16...HAVG.256-CWB_MC12` for a `B1x64.2x64.2x64.4x64.5x64.5x64` backbone, an `HFLAT.512.256-HLN` head
  and ImageNet-1k); line 1175 names `Evaluation.keypoint_sigmas` instead of
  `ExperimentConfiguration.evaluation.keypoint_sigmas`; the `load_configuration` docstring promises `yaml.YAMLError` but
  an unchained ValueError is raised.
- **Fix:** regenerate the examples, fix the key path and the docstring.

#### B28. Comments left from the top-down keypoint detector contradict the code
- **Area:** data generation · **Severity:** low · **Status:** confirmed (history traced by the reviewer) · **Category:**
  documentation
- **Where:** `vision_training/data/synthetic/create_synthetic_dataset.py:302-304, 121`.
- **Problem:** line 303 speaks of "the class indices of the detector that will give its boxes" (the keypoint model is
  now bottom-up and the categories are renumbered); line 121 says a horizontal flip swaps the left and right keypoints,
  while `augmentations.py:253` and the keypoint workflow say a flip never renames a keypoint.
- **Failure scenario:** a user trusting the comment turns flips on and expects swapped names, getting inconsistent
  targets.
- **Fix:** rewrite both comments for the bottom-up detector, the renumbered mapping and the never-renamed keypoints.

#### B29. The README gives the wrong number of objects per synthetic keypoint image
- **Area:** data generation · **Severity:** low · **Status:** confirmed · **Category:** documentation
- **Where:** README line 125; `create_synthetic_dataset.py:2710` (`"keypoint_detection": 5`).
- **Problem:** the KPT row says "2 to 4 objects per 512 pixel image"; the generator,
  `configuration_synthetic_shapes.yaml` and the keypoint workflow say 2 to 5 (commit 95456e3 raised it without the
  README).
- **Fix:** "2 to 5 objects".

#### B32. Backbone.compute_detection_input_information dry-runs in training mode with gradients
- **Area:** core models · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) · **Category:** bug
- **Where:** `vision_training/models/backbones/backbone.py:226-257` (compare `print_summary`, 191-213, in eval mode and
  no_grad); caller `two_stage_detector.py:74`.
- **Problem:** the shape probe sends a zero image through the blocks in training mode, changing every BatchNorm buffer
  at construction (running_var 1.0 → 0.9, num_batches_tracked 1), and raising when a block meets a 1x1 map although
  training on batches of 2 works.
- **Failure scenario:** a detector whose last block receives a 1x1 input cannot be built ("Expected more than 1 value
  per channel").
- **Fix:** wrap the probe in `eval()` and `torch.no_grad()` and restore the mode, as `print_summary` does.

#### B33. Model builders silently fall back on misspelled option strings
- **Area:** core models · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) · **Category:**
  error-handling
- **Where:** `temporal_modules.py:452-457` (unknown type → TemporalConvolutionModule), `member_fusion.py:49, 127-129`
  (unknown fusion → "difference"), `segmentation_decoder.py:77-85` (anything but "transposed" → bilinear),
  `fully_connected_block.py:85-90` (unknown normalization → none), `classification_head.py:57-59` (anything but
  "flatten" → channel count).
- **Problem:** if/else defaults instead of errors: only the checkers stop bad values, so a model built from an unchecked
  dictionary (scripts, tests, older saved configurations) gets another architecture without message, against the
  fail-loudly rule.
- **Failure scenario:** `TemporalModel.type: "transformr"` reaching the builder trains a temporal convolution.
- **Fix:** end every builder with a `raise ValueError(...)` listing the valid values, as `decode_frame_selection` and
  `ConvolutionBlock` do.

#### B35. The eager evaluation ignores the valid proposal mask, so padding copies can become duplicate detections
- **Area:** detection and keypoint models · **Severity:** low (the reviewer said medium: the shipped NMS threshold 0.5
  removes identical copies, so only a threshold of 1.0 or zero-area boxes are affected) · **Status:** confirmed
  (reproduced by the reviewer; code read again) · **Category:** bug
- **Where:** `vision_training/models/detection/two_stage_detector.py:731-739` (`torch.ones_like` passed as
  `valid_proposal_mask`); `region_proposer.py:303-337` (padding with copies of random survivors);
  `postprocessing.py:24-28` ("Both keep the same detections"); `configuration_values.py:185`
  (`Detector.nms_iou_threshold` accepted up to 1.0).
- **Problem:** the eager forward relies on the final NMS to remove the padding copies, which fails when the IoU of
  identical boxes is not above the threshold (threshold 1.0) or undefined (zero-area boxes, 0/0), while the deployed
  `InferenceDetector` masks them; this is the root cause of the zero-area duplicates of item L7 of suggestions.md.
- **Failure scenario:** RPN NMS 0.05 (25 of 50 proposals valid) and `Detector.nms_iou_threshold: 1.0`: the test
  evaluation sees 250 detections per image (125 distinct, the copies counted as false positives), the deployed model
  125.
- **Fix:** carry `valid_proposal_mask` through the per-level slicing and the per-image rearrangement like the scores,
  and pass it to `postprocess_detections`.

#### B37. get_box_dimensions_and_centers ends the process with exit() and status 0 on a wrong box shape
- **Area:** detection and keypoint models · **Severity:** low · **Status:** confirmed (reproduced by the reviewer; the
  other `exit` calls of the repository checked) · **Category:** error-handling
- **Where:** `vision_training/models/detection/box_operations.py:113-119`.
- **Problem:** `exit()` raises `SystemExit(None)`, which no `except Exception` catches and which ends the process with
  code 0, so a crash looks like a success to scripts and schedulers (the other calls are deliberate:
  `base_trainer.py:273` after the input viewer, `configuration_loader.py:178` after the configuration errors).
- **Failure scenario:** `apply_regression_predictions` given boxes of shape `[2, 5]`: a red message, then exit code 0.
- **Fix:** raise a ValueError with the same message.

#### B39. HRNet with feature_map_normalization "none" builds every convolution without bias
- **Area:** detection and keypoint models · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** configuration
- **Where:** `vision_training/models/keypoints/hrnet.py:47, 221, 409-417` (`bias=False`);
  `configuration_processing/keypoint_checks.py:123-127` (accepts "none").
- **Problem:** the hard-coded `bias=False` assumes a normalization follows; with "none" the 17 convolutions of the high
  resolution network have neither bias nor normalization, while the backbone always keeps its bias.
- **Failure scenario:** `HRNet.normalization.feature_map_normalization: "none"` silently trains activations that cannot
  shift their thresholds.
- **Fix:** `bias=True` when the normalization is "none", or refuse "none" for HRNet in the checker.

#### B40. The batch_assign_targets_to_anchors docstring promises a positive anchor for every ground truth
- **Area:** detection and keypoint models · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** documentation
- **Where:** `vision_training/models/detection/target_assignment.py:129-130` (docstring), `186-212`.
- **Problem:** the best anchor of a ground truth becomes positive, but it is assigned to the ground truth it overlaps
  most (or the last one resurrecting it in strict mode), so a small object sharing its best anchor with a larger one, or
  with a best IoU of 0, gets no anchor.
- **Failure scenario:** anchors `[[0,0,100,100],[300,300,400,400]]` and ground truths `[[0,0,30,30],[0,0,100,90]]`:
  ground truth 0 has no positive anchor in both modes.
- **Fix:** describe the actual behaviour (as in torchvision).

#### B42. Training and validation forwards of the detector do inference-only work, with one host synchronisation per image
- **Area:** detection and keypoint models · **Severity:** low · **Status:** likely (read by the reviewer, not timed) ·
  **Category:** performance
- **Where:** `vision_training/models/detection/two_stage_detector.py:724-728, 366-414` (one `torch.where` per image),
  `731-739`; `models/instance_segmentation/instance_segmenter.py:123-124`.
- **Problem:** every training step rearranges the predictions per image for outputs nothing reads in training, and
  validation (eval mode with ground truth) runs the postprocessing, and the mask prediction for INS, whose detections
  `run_model_iteration` discards.
- **Failure scenario:** a batch of 16 on CUDA: 16 extra host synchronisations per training step and an unused batched
  NMS per validation batch.
- **Fix:** skip the rearrangement and the postprocessing when ground truth is given (the fix the audit of 2026-10-10
  proposed for its item H3, fixed since by padding the candidates), or rearrange with one argsort and gather.

#### B49. An evaluation score threshold above the deployment threshold silently removes detections from the confusion matrix and the images
- **Area:** task trainers and losses · **Severity:** low · **Status:** confirmed (code path read by the reviewer;
  latent, the shipped DET and INS files use 0.05 and 0.5) · **Category:** configuration
- **Where:** `vision_training/trainers/detection_trainer.py:250-265` (`evaluation_context`), `331, 630` (filter at the
  deployment threshold), `543-548, 640` (captions); `configuration_processing/detection_checks.py:389-392`.
- **Problem:** the evaluation replaces the detector threshold by `evaluation.score_threshold`, then the confusion
  matrix, error and sample images keep the detections above `Detector.inference.score_threshold`; they only match the
  deployed model when the evaluation threshold is the lower one, and nothing checks it.
- **Failure scenario:** `evaluation.score_threshold: 0.5` with an inference threshold of 0.3: detections between 0.3 and
  0.5, which the deployed model returns, are counted as missed objects in a confusion matrix captioned "score >= 0.30".
- **Fix:** set the model threshold to the minimum of both in `evaluation_context`, or refuse an evaluation threshold
  above the inference one in the checker.

#### B54. A standalone evaluation is tracked in another MLflow database than the one the launch panel prints
- **Area:** evaluation and tracking · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** configuration
- **Where:** `vision_training/evaluation/checkpoint_evaluation.py:45` against `base_trainer.py:1023-1028` and
  `configuration_loader.py:108-112`.
- **Problem:** `start_evaluation_tracking` resolves the backend from `experiment_folder.parent`, while the panel and
  commit ab3e786 use `experiment_configuration["tracking_root"]`.
- **Failure scenario:** a moved or archived run folder: the panel names `experiments/mlflow.db`, the evaluation run is
  created in `archive/mlflow.db`, away from the training run its `evaluation_of` tag points to.
- **Fix:** pass `tracking_root` (or the run record's `tracking_uri`) to `create_experiment_tracker`.

#### B57. A run without checkpoint registry is "evaluated" on nothing and the script ends normally
- **Area:** deployment and scripts · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** error-handling
- **Where:** `scripts/evaluation/evaluate_checkpoint.py:74-79`; `checkpointing.py:58-60`;
  `checkpoint_evaluation.py:116-120, 150-161`.
- **Problem:** without registry, `get_last_checkpoint_iteration` gives 0 and `get_checkpoint_iterations` gives `[]`,
  used as is; the export tools raise FileNotFoundError for the same case.
- **Failure scenario:** a run that crashed before its first save: one yellow "Iteration 0 Not Found, Skipped" line (or
  nothing with "all"), an empty `evaluation_summary.csv` and an MLflow evaluation run closed as FINISHED.
- **Fix:** raise FileNotFoundError naming the Weights folder when the registry is missing or empty.

#### B58. A misspelled export format is silently exported as ONNX, and its TensorRT setting dropped
- **Area:** deployment and scripts · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** configuration
- **Where:** `vision_training/deployment/export_model.py:231-249`; `triton_configuration.py:37-39, 62, 87`.
- **Problem:** every check is `== "torchscript"` with an else branch, so "TorchScript", "tensorrt" or "trt" take the
  ONNX path (platform `onnxruntime_onnx`), and the TensorRT block is only added for exactly "onnx".
- **Failure scenario:** `formats = ["tensorrt"]` serves plain ONNX Runtime without TensorRT, without message.
- **Fix:** check that `formats` is a non-empty subset of {"torchscript", "onnx"} and `tensorrt_precision` in {None,
  "FP16", "FP32"} at the start of `export_experiment`.

#### B59. load_inference_inputs claims the preprocessing of the loaders, but a tiled segmentation run gets a whole image squeezed into one tile
- **Area:** deployment and scripts · **Severity:** low · **Status:** likely (traced by the reviewer) · **Category:**
  documentation
- **Where:** `vision_training/deployment/deployment_utilities.py:38-40, 62-70`; `segmentation_dataset.py:186-191, 234`.
- **Problem:** with `Data.tiling` on, `image_settings.size` is the tile size and the loaders scale then cut tiles, while
  `load_inference_inputs` letterboxes the whole image into one tile.
- **Failure scenario:** `run_inference.py` on a tiled run times one tile per request instead of the many a real request
  needs, overstating the Images Per Second.
- **Fix:** refuse tiled runs in `load_inference_inputs` or tile them like the loaders, and correct the docstring.

## Items Of suggestions.md Still Open

The status of the items of the [Repository Audit For Instance Segmentation](suggestions.md#repository-audit-for-instance-segmentation)
that are still open, read in the code at `165339b` and updated on 2026-11-14. Their descriptions and fixes are in that
file; the fixed and dropped ones were removed from both files.
- **Open:** the defect described is still in the code.
- **Partly fixed:** only the part given in the note remains.

| Item | Severity there | Status | Note |
|---|---|---|---|
| H6. An unreadable image silently drops its batch | high | **partly fixed** | on 2026-10-21: unreadable images named, their batches dropped, counted and reported without stopping the training; nothing stops it above a dropped-batch ratio, and an unreadable test image fails every evaluation |
| M5. Masks resized with plain nearest are shifted | medium | **partly fixed** | the RLE masks use `INTER_NEAREST_EXACT`; the SEG masks, the masks of the crops and the zoom-out and the class maps restored by `preprocessing.py` still use plain nearest |
| M13. Decoded image size never compared with the annotations | medium | **partly fixed** | see below; `coco_dataset.py:249` still never compares the decoded shape with the JSON size |
| M16. The DET and INS export comparison compares padding | medium | **open** | |
| L3. The threshold finder returns the lowest valid threshold | low | **open** | `threshold_finder.py:46-48`, which also sets the DET, INS, KPT and SEL class thresholds |
| L4. Balanced weights do not balance positives in multi-label | low | **open** | `classification_losses.py:37-40`, not documented either |
| L7. Zero-area proposals and detections never removed | low | **partly fixed** | on 2026-11-12: `"per_level"` leaves out the empty proposals; `"joint"` (instance segmentation) keeps them, and the final detections get no filter |
| L8. DET boxes only cleaned on augmented samples | low | **open** | DET and KPT (`coco_dataset.py:118-121`); INS takes its boxes from the masks |
| L9. Multi-label keeps all-unmapped samples as negatives | low | **open** | `classification_dataset.py:80-85`; the class docstring still says they are left out |
| L11. M2M batches without labelled frame | low | **open** | see below |
| L17. Ragged last batch and averages over batches | low | **open** | |

| Statement contradicting the code | Status | Note |
|---|---|---|
| README "Stopping And Resuming", `build_run_name` docstring: "Changing any other setting gives another run name" | **open** | README line 249, `run_name.py:641-646` (see C4) |
| Every YAML with `residual_nets`: "Shortcuts are added every two convolution layers" | **open** | H4 dropped on 2026-11-05: the comment should say that the channel-changing blocks get no shortcut |
| `RegionProposal.foreground_iou_threshold.max` (YAML) | **open** | still read only by the checker |
| `mean_average_precision.py:16` "COCO" | **partly fixed** | still "COCO-style mAP@[0.5:0.95]" at line 22 and "(COCO)" in `average_precision_report.py:53` |
| `region_proposer.py:34` aspect ratio = width / height | **open** | now line 126; `anchors.py` uses height / width |
| `preprocessing.py:124`, `deployment_utilities.py:3-4` "used by the inference clients" | **partly fixed** | the restore functions now serve the evaluations, but no inference client of the repository calls them |
| README CLS run name `..._HAVG.128_MC4` | **open** | see [B24](#b24-stale-run-name-and-checker-documentation) |
| `sequence_classification_trainer.py:66-72` frame shuffle warning | **open** | still warns for every transformer, whatever `positional_embedding` and `causal` |
| Loader: `int(float(...))` truncation | **open** | `configuration_loader.py:126, 150` |
| Loader: unknown or misspelled keys ignored | **open** | only the augmentation types are checked for unknown names (`common_checks.py:189-195`) |

### What This Audit Adds

- **H6:**
  - The four datasets still pass a `None` image to `preprocess_image`: `classification_dataset.py:136-137`,
    `coco_dataset.py:249, 267`, `segmentation_dataset.py:226-228` and `sequence_dataset.py:62-63`.
  - So does the inference input loader, `deployment_utilities.py:68`, a file H6 does not list.
  - That call fails in `preprocessing.py:64` or `222` without naming the file.
- **M5 (partly fixed):**
  - The RLE masks are now resized with `INTER_NEAREST_EXACT` (`coco_masks.py:324-327`).
  - `random_crop` and `random_zoom_out` (`augmentations.py:592-595, 643-647`) and `SegmentationDataset`
    (`segmentation_dataset.py:191, 238-239`) still use plain nearest. After a crop or zoom-out the masks move up to 0.86
    pixel, against 0.07 for the boxes, which the tests do not catch with their 2-pixel tolerance.
- **M13 (partly fixed):**
  - The RLE size is now compared with the image, but only at the original resolution (`coco_masks.py:316-327`).
  - At the network resolution, a mismatched RLE is still letterboxed with its own size.
  - So training learns misaligned masks, and only the test evaluation at the original resolution fails.
- **L7:** [B35](#b35-the-eager-evaluation-ignores-the-valid-proposal-mask-so-padding-copies-can-become-duplicate-detections) is its root cause. The eager evaluation passes no valid proposal mask, so all the padding
  copies of a zero-area box survive the NMS.
- **L11:** both halves remain:
  - `sequence_losses.py:29-31` gives NaN for a batch without labelled frame, while the segmentation losses clamp their
    normaliser;
  - `sequence_metrics.py:28-40` counts a sequence whose frames are all unmapped as an exact match. One sequence with
    every frame wrong plus two unlabelled ones give a Sequence Exact Match of 0.667 with a frame accuracy of 0.0, and
    the "N Test Sequences" header counts them.
- **Statements contradicting the code:** the README CLS run name is still `..._HAVG.128_MC4`. [B24](#b24-stale-run-name-and-checker-documentation) adds the
  stale M2O and CMP examples.
- **Robustness gaps of the configuration loader:** the `int(float(...))` truncation remains
  (`configuration_loader.py:126, 150`).

## Coding Standards

The rules of the global `CLAUDE.md` and of the project memory, checked on the 266 Python files of `vision_training/`,
`scripts/` and `tests/`, the 48 tracked configuration files and the README. The violations found were all corrected and
are removed from this file.

### Respected Everywhere
- A module docstring at the top of every Python file.
- No function or class defined inside another function.
- Imports at the top of the files. The only exception is the optional MLflow tracker, which its factory loads with
  `importlib`, as the memory asks.
- No bare `except` and no `except: pass`.
- `Args:` docstrings, never `:param`.
- No mention of the inspiration code.
- Console units in full words.
- Configuration values read with `["key"]`, with one deliberate exception (see
  [Accepted Exceptions](#accepted-exceptions)).
- Printed paths given through `format_path_link`.
- YAML files commented in the instructional style (the keys flagged without comment were false positives).
- The README starts with its table of contents, then the description and the prerequisites, without praise, and its
  images point to `README/`.
- Skipped tests: only 3, all conditional (Linux, GPU), and they run on this machine.

### Accepted Exceptions
- **`tmp_path` and `tmp_path_factory`** (235 test parameters): pytest fixture names, which cannot be renamed.
- **`last_max_pooling`, `info_nce_loss` and `info_nce_outputs`:** "max pooling" and "InfoNCE" are the names of the
  methods.
- **`x` and `y` for point coordinates:** `create_synthetic_dataset.py:1904`, `visualization.py:947` and three tests.
- **One argument or parameter per line where no two fit in 120 characters:** 25 places:
  - source: `two_stage_detector.py` (`forward_detection_stages`, `self.region_proposal_filter`,
    `compute_region_proposal_losses`), `detection_losses.py` (the signature of `compute_region_proposal_losses`,
    `batch_assign_targets_to_anchors`, `sample_positive_and_negative_training_targets`, the smooth L1 loss of the
    proposals), `mean_average_precision.py` (`match_detections_to_ground_truth`, `find_class_score_thresholds`),
    `keypoint_detector.py` (both `compute_peak_offsets`), `export_specifications.py` (both `match_detections`),
    `frame_selection_metrics.py` (`find_class_thresholds`), `inference_detector.py` (`self.filter_proposals`) and
    `sequence_labelling_trainer.py` (`group_sequence_predictions`);
  - tests: `test_augmentations.py` (two `augment_batch`), `test_mean_average_precision.py`
    (`match_detections_to_ground_truth`), `test_self_supervised_trainer.py` (`SimpleNamespace`),
    `test_cluster_image_embeddings.py` (four signatures) and `test_os_utilities.py` (one signature).
- **`previous_settings.get(setting_key, MISSING_SETTING)`** (`configuration_changes.py:73`): it compares a saved
  configuration, with a sentinel for the settings that configuration was saved without.
- **Broad `except` without `raise`:**
  - The 8 in `experiment_tracker.py:301` and `mlflow_tracker.py:314, 329, 347, 364, 376, 415, 476` are deliberate:
    tracking never stops a training, and each prints its error.
  - `os_utilities.py:199` falls back to an empty namespace in the debugging helper of `print_tensor_shape`.
  - The other two, `base_trainer.py:1316` and `1591`, are deliberate since 2026-10-21: a batch that fails to load is
    dropped and counted, a `RuntimeError` following 10 failed fetches in a row (H6), and a
    failing training iteration is printed and skipped, the training stopping after 5 in a row.
- **The 4 tests without `assert`:** they assert through `check_nn_module_method` (`torch.allclose` against the `.pt`
  references) or through the `SystemExit` of the checker.
- **`imagenet_download.py:474`** (`check_split_sizes`, moved from the ImageNet-1k download on 2026-11-11): prints a
  count, not a path.

## Suggested Order

1. **The findings that give silently wrong results:** [B8](#b8-a-causal-sequence-labeller-or-frame-selector-with-sequence-normalization-uses-the-future-frames) (causal models seeing later frames),
   [B10](#b10-a-keypoint-head-transfer-compares-only-the-class-names-so-keypoints-in-another-order-land-on-the-wrong-channels) (a keypoint head loaded onto the wrong channels) and [B19](#b19-standalone-evaluations-of-another-split-overwrite-and-mix-with-the-earlier-ones-in-the-same-output-folder) (evaluations of two splits mixed in
   one folder).
2. **The open items of suggestions.md** that guard long runs: H6 and M13, then M5, L7 and L11.
3. **The low bugs, area by area**, starting with the misleading reports and the silent fallbacks: [B54](#b54-a-standalone-evaluation-is-tracked-in-another-mlflow-database-than-the-one-the-launch-panel-prints),
   [B57](#b57-a-run-without-checkpoint-registry-is-evaluated-on-nothing-and-the-script-ends-normally), [B58](#b58-a-misspelled-export-format-is-silently-exported-as-onnx-and-its-tensorrt-setting-dropped), [B33](#b33-model-builders-silently-fall-back-on-misspelled-option-strings) and [B37](#b37-get_box_dimensions_and_centers-ends-the-process-with-exit-and-status-0-on-a-wrong-box-shape).

## How It Was Checked

- **Reviews:** each reviewer read its area end to end and reproduced every finding it could with a script, kept
  outside the repository. The areas, with what was checked and found correct when the reviewer reported it:
  - **Configuration** (`vision_training/configuration_processing/`): the loader, the checks of every task, the run
    names, rebuilt for every shipped configuration, and the comparison of a resumed configuration.
  - **Data generation** (`vision_training/data/synthetic/`, the image downloaders, `scripts/data/`):
    - the keypoint labels against the drawn objects: 0 of 8,384 visible keypoints more than 1 pixel off;
    - the shape counts of the sequence labelling and self-supervised images at 512 pixels;
    - the worker generators: 600 images without duplicate;
    - the Open Images rotation convention;
    - the shape placement: no failure in 400 images of 10 shapes;
    - the ImageNet extraction, hold-out and resume logic.
  - **Data loading** (the datasets, loaders, augmentations, label mapping and data record):
    - the polygon rasteriser: 2,000 random polygons against a brute-force reference;
    - compressed RLE decoding: 3,000 round trips;
    - `GroupedBatchSampler`: 300 random layouts;
    - boxes and keypoints under every geometric augmentation: within 0.07 pixel;
    - every augmentation on odd sizes down to 5x7;
    - the seeding of the workers.
  - **Core models** (backbones, components, and the classification, segmentation, sequence and self-supervised models):
    - the 7 temporal modules, with and without padding;
    - the comparator masks;
    - `SequenceBatchNormalization` against a reference;
    - 1,536 `ConvolutionBlock` option combinations in float32 and bfloat16;
    - 264 sequence model and 228 classifier training variants, view losses and decorrelation included;
    - the U-Net decoder and flatten pooling at odd sizes;
    - the sliding window coverage;
    - the 209 tests of the area pass.
  - **Detection and keypoint models** (`vision_training/models/detection/`, `instance_segmentation/`, `keypoints/` and
    `model_transfer.py`): anchors, box operations, region proposals, target assignment, heads, postprocessing,
    inference models, HRNet and the keypoint grouping.
  - **Training loop:** `base_trainer.py`, checkpointing, metric tracking, weight averaging and the learning rate
    schedule.
  - **Task trainers and losses** (the eight task trainers and the seven loss modules): the Dice, Lovász and boundary
    losses return 0 on fully ignored masks.
  - **Evaluation and tracking** (`vision_training/evaluation/`, `vision_training/tracking/`): accuracy, balanced
    accuracy, the per-class metrics and top-k match scikit-learn exactly; the MLflow behaviour was reproduced on a
    scratch database.
  - **Deployment and scripts:** `vision_training/deployment/` and `scripts/`.
- **Second reading:** every medium and high finding was read again in the code before being kept, and the focal loss,
  solarize and motion blur findings were run again. One finding was lowered from medium to low ([B35](#b35-the-eager-evaluation-ignores-the-valid-proposal-mask-so-padding-copies-can-become-duplicate-detections),
  hidden by the shipped NMS threshold). The findings that are items of suggestions.md were moved to
  [their section](#items-of-suggestionsmd-still-open).
- **Standards:** a script parsed the 266 Python files (syntax tree and tokens) for:
  - module and function docstrings, `Args:` sections, argument and return annotations;
  - nested definitions and late imports;
  - abbreviated and single-letter names;
  - positional arguments to repository functions, and one argument or parameter per line;
  - lines over 120 characters, and stretches of 20 lines without comment;
  - bare or silent `except`, and `.get` on configurations;
  - printed paths without link, and abbreviated console units;
  - tests without assertion, and mentions of the inspiration code.

  Each rule was first shown to fire on a module breaking it and to stay silent on a compliant one, then every hit was
  read. Numbered comments, README images, debug prints and short docstrings were searched separately.
- **Configurations:** 42 of the 48 tracked configuration files load and pass the checker. The other 6 stop on label
  mapping files that are not on this machine: the two frame selection files and the examples of SEG, M2O, M2M and CMP.
- **Not covered:**
  - the training speed and the GPU memory;
  - the real datasets, beyond the COCO 2017 validation keypoints and the local Open Images photos used by two
    findings;
  - the Triton server itself.
- **Scratch files:** the review scripts and their outputs are in the folder of this session's job
  (`~/.claude/jobs/6a755f62/tmp/audit/`), outside the repository, and are deleted with the job.
