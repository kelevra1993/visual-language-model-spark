# Bugs And Standards

> Audit of the whole repository carried out on 2026-10-20 at commit `165339b`, to go through together. The coding
> standards S1 to S10 and the bugs B7, B20, B34, B47 and B50 were fixed on 2026-10-20 (not committed yet). On
> 2026-10-21, C3, C4 (the anchors and `last_max_pooling`), H6, M6, M7, M8, M17, M20, L13, B9, B11, B30 and B44
> were fixed, with the host waits of `random_black_out`. The items dropped by the user are deleted from this file,
> the other items are still open. Nine reviewers each read one area end to end (configuration, data generation, data loading, core
> models, detection and keypoint models, training loop, task trainers and losses, evaluation and tracking, deployment
> and scripts) and reproduced their findings with small scripts on the synthetic datasets, the dummy task or hand-made
> inputs. Every medium and high finding was then read again in the code. The coding standards of the global
> `CLAUDE.md` and of the project memory were checked by a script over the 266 Python files of `vision_training/`,
> `scripts/` and `tests/`, then by reading every hit.
>
> The items of the [Repository Audit For Instance Segmentation](suggestions.md#repository-audit-for-instance-segmentation)
> (2026-10-10, commit `84ac325`) are not repeated here. That section keeps its items after they are fixed, so
> [Items Of suggestions.md Still Open](#items-of-suggestionsmd-still-open) gives the status of each one at this commit,
> with what this audit adds to it.
>
> **Identifiers:** B1 to B59 for the bugs, ordered by severity, then along the pipeline (configuration, data,
> models, training, evaluation, deployment), and S1 to S11 for the standards.
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
  - [Not Respected](#not-respected)
  - [Accepted Exceptions](#accepted-exceptions)
- [Suggested Order](#suggested-order)
- [How It Was Checked](#how-it-was-checked)

## Summary

- **No new high bug.** The most serious problems found open at this commit were items of the audit of 2026-10-10, each
  checked again in the code (see [Items Of suggestions.md Still Open](#items-of-suggestionsmd-still-open)), all fixed
  on 2026-10-21 except H4 and M9:
  - **C3:** DET and INS count their classes as label dictionary keys and nothing checks their mapping, so merged
    categories crash the first evaluation (found again by three reviewers).
  - **C4 and M17:** run names collide for settings that change the weights (`last_max_pooling`, the anchors, the
    pretrained source, the content of the label mapping), so a different configuration resumes another run.
  - **H4, H6, M6 to M9 and M20:** missing shortcuts in channel-changing blocks, swallowed data errors,
    iterations counted from the resume point, fresh starts mixing two runs, and an empty split looping forever.
- **57 new bugs:** 17 medium and 40 low, listed by area below. Two come from the commits of
  2026-10-20: [B7](#b7-motion-blur-of-the-views-the-largest-kernel-size-is-never-drawn-and-the-viewer-describes-kernels-kornia-does-not-apply) (the motion blur kernel sizes of the views) and the solarize bound of [B20](#b20-the-view-augmentation-checks-accept-values-kornia-rejects).
- **Coding standards:** most rules are respected everywhere (module docstrings, imports, no nested functions, no bare
  `except`, `Args:` docstrings, strict configuration access, linked paths, YAML comments, README layout). The
  violations S1 to S10 (formatting of the detection code, a few names and docstrings, duplicated code) were
  corrected on 2026-10-20; S11, the README screenshots to take from real runs, is still open.
- **Fixed on 2026-10-20:** B7 and B20 (the views), and B34, B47 and B50 (the duplicated code of S10).
- **Tests at this commit:** the full suite passes (1,415 tests in 5 minutes 26 seconds, run on 2026-10-20 in a clean worktree), a baseline for the fixes.

| Area | Medium | Low |
|---|---|---|
| Configuration | [B1](#b1-a-missing-key-discards-every-problem-already-found-and-a-mode-typo-is-reported-as-a-missing-datalabel_dictionary) | [B20](#b20-the-view-augmentation-checks-accept-values-kornia-rejects), [B21](#b21-the-loader-coerces-malformed-values-before-the-checks-and-some-malformed-sections-crash-the-checker), [B22](#b22-read_json-swallows-json-parse-errors-and-returns-none), [B23](#b23-the-det-checks-require-mask-head-settings-whenever-a-maskhead-section-is-present), [B24](#b24-stale-run-name-and-checker-documentation) |
| Data generation | [B2](#b2-a-sequence-split-subset-silently-goes-stale-when-its-source-split-is-regenerated), [B3](#b3-duplicates-are-merged-only-on-identical-hashes-resized-copies-of-one-photo-stay-in-both-train-and-test) | [B25](#b25-failed-image-writes-are-ignored-so-the-labels-can-list-images-never-written), [B26](#b26-image-tasks-crash-with-an-obscure-numpy-error-on-images-smaller-than-34-pixels), [B27](#b27-two-cached-files-are-written-in-place-so-an-interrupted-write-is-trusted-on-the-next-run), [B28](#b28-comments-left-from-the-top-down-keypoint-detector-contradict-the-code), [B29](#b29-the-readme-gives-the-wrong-number-of-objects-per-synthetic-keypoint-image) |
| Data loading | [B4](#b4-datajson-crashes-at-launch-on-a-self-supervised-split-mixing-labelled-and-unlabelled-images), [B5](#b5-datajson-crashes-at-launch-on-a-coco-split-without-annotations-which-the-loaders-accept), [B6](#b6-datajson-overstates-the-keypoint-objects-it-counts-the-objects-the-keypoint-loader-drops), [B7](#b7-motion-blur-of-the-views-the-largest-kernel-size-is-never-drawn-and-the-viewer-describes-kernels-kornia-does-not-apply) | [B30](#b30-a-category-mapped-to-0-to-exclude-it-names-the-background-class-and-is-counted-under-it-in-datajson), [B31](#b31-get_image_paths-ignores-tif-and-webp-images) |
| Core models | [B8](#b8-a-causal-sequence-labeller-or-frame-selector-with-sequence-normalization-uses-the-future-frames), [B9](#b9-a-batch-normalized-classification-head-crashes-on-a-training-batch-of-a-single-vector) | [B32](#b32-backbonecompute_detection_input_information-dry-runs-in-training-mode-with-gradients), [B33](#b33-model-builders-silently-fall-back-on-misspelled-option-strings), [B34](#b34-sequenceclassifier-and-sequencelabeller-are-near-copies) |
| Detection and keypoint models | [B10](#b10-a-keypoint-head-transfer-compares-only-the-class-names-so-keypoints-in-another-order-land-on-the-wrong-channels) | [B35](#b35-the-eager-evaluation-ignores-the-valid-proposal-mask-so-padding-copies-can-become-duplicate-detections), [B36](#b36-the-rpn-classification-loss-is-nan-when-no-anchor-of-the-batch-is-sampled), [B37](#b37-get_box_dimensions_and_centers-ends-the-process-with-exit-and-status-0-on-a-wrong-box-shape), [B38](#b38-a-detection-head-transfer-from-a-run-saved-before-detectionlevels-fails-with-a-bare-keyerror), [B39](#b39-hrnet-with-feature_map_normalization-none-builds-every-convolution-without-bias), [B40](#b40-the-batch_assign_targets_to_anchors-docstring-promises-a-positive-anchor-for-every-ground-truth), [B41](#b41-detectionheadforward-documents-a-tensor_to_concatenate-argument-it-never-reads), [B42](#b42-training-and-validation-forwards-of-the-detector-do-inference-only-work-with-one-host-synchronisation-per-image) |
| Training loop | [B11](#b11-ten-failed-batch-fetches-in-a-row-leave-through-exit1-the-final-checkpoint-is-never-saved), [B12](#b12-ctrlc-during-a-periodic-save-is-ignored-by-the-trainer-but-kills-the-persistent-loader-workers-the-run-fails-5-iterations-later) | [B43](#b43-resuming-by-launching-the-runs-own-training_configurationyaml-crashes-with-samefileerror), [B44](#b44-evaluated-checkpoints-are-overwritten-or-left-unevaluated-at-stop-and-resume-boundaries), [B45](#b45-the-console-summary-treats-the-correlation-metrics-as-higher-is-better-a-rising-off-diagonal-correlation-prints-green-with-best), [B46](#b46-iterations-per-epoch-and-the-sample-counts-logged-to-mlflow-come-from-the-manifest-before-the-label-mapping), [B47](#b47-the-self-supervised-k-nn-report-re-implements-the-per-class-rows-table-and-scalars-of-class_metrics_reportpy) |
| Task trainers and losses | | [B48](#b48-the-det-and-ins-input-viewer-captions-the-boxes-with-raw-category-ids-instead-of-class-names), [B49](#b49-an-evaluation-score-threshold-above-the-deployment-threshold-silently-removes-detections-from-the-confusion-matrix-and-the-images), [B50](#b50-keypointtrainer-copies-four-detectiontrainer-methods-and-frameselectiontrainer-redefines-the-describe_split-it-inherits) |
| Evaluation and tracking | [B15](#b15-in-mlflow-testbest_metric-of-a-standalone-all-evaluation-shows-the-last-evaluated-value-not-the-best), [B16](#b16-one-artifact-write-failing-with-a-permanent-oserror-blocks-every-later-mlflow-write-and-leaves-the-run-running) | [B51](#b51-keypoint-matched-objects-and-mean-pixel-error-count-predictions-far-from-the-object), [B52](#b52-tied-probabilities-give-every-tied-frame-rank-1-inflating-top-k-selection-accuracy-and-mean-rank), [B53](#b53-mean-iou-mean-dice-and-macro-f1-leave-out-the-classes-predicted-but-absent-from-the-ground-truth), [B54](#b54-a-standalone-evaluation-is-tracked-in-another-mlflow-database-than-the-one-the-launch-panel-prints), [B55](#b55-a-detection-test-split-without-objects-reports-map-and-mar-00-instead-of-no-value) |
| Deployment and scripts | [B17](#b17-an-exported-frame-selector-with-averaged-ema-weights-gets-the-confidence-thresholds-of-the-trained-weights), [B18](#b18-a-maximum-batch-size-of-1-crashes-the-onnx-export-and-at-2-the-dynamic-batch-is-never-checked), [B19](#b19-standalone-evaluations-of-another-split-overwrite-and-mix-with-the-earlier-ones-in-the-same-output-folder) | [B56](#b56-evaluate_checkpointpys-default-error-image-setting-stops-seg-kpt-and-ssl-runs-with-a-bare-keyerror), [B57](#b57-a-run-without-checkpoint-registry-is-evaluated-on-nothing-and-the-script-ends-normally), [B58](#b58-a-misspelled-export-format-is-silently-exported-as-onnx-and-its-tensorrt-setting-dropped), [B59](#b59-load_inference_inputs-claims-the-preprocessing-of-the-loaders-but-a-tiled-segmentation-run-gets-a-whole-image-squeezed-into-one-tile) |
| **Total** | **17** | **40** |

## Bugs

### Medium

#### B1. A missing key discards every problem already found, and a mode typo is reported as a missing Data.label_dictionary
- **Area:** configuration · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer) · **Category:**
  error-handling
- **Where:** `vision_training/configuration_processing/configuration_checker.py:57-73`; `configuration_values.py:58-61`;
  `detection_checks.py:308-309`.
- **Problem:** `get_configuration_value` raises at the first missing key and the error escapes `check_configuration`,
  dropping the errors already collected (README line 155: "the list of every problem found"); the detection checks read
  the injected `Data.label_dictionary` unconditionally.
- **Failure scenario:** a DET file with `mode: "multiclass"` and `batch_size: 0` reports only "Missing configuration key
  'Data.label_dictionary'."
- **Fix:** wrap each check to collect its configuration errors, and skip the dictionary checks when the mode is invalid.

#### B2. A sequence split subset silently goes stale when its source split is regenerated
- **Area:** data generation · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer) · **Category:**
  bug
- **Where:** `scripts/data/create_sequence_split_subset.py:46-55`;
  `vision_training/data/synthetic/create_synthetic_dataset.py:2372-2389` (`save_frames`), `2529-2530`
  (`generate_split`), `2672` (main block: existing images and labels overwritten).
- **Problem:** the subset copies the labels of the drawn sequences, but its `data` is a symlink to the live frames of
  the source split, which the generator regenerates in place without seed and without clearing `data/`: the old labels
  then point at new frames, or at a mix of new and stale frames when a sequence got shorter, and nothing fails since
  every listed file exists.
- **Failure scenario:** `test_10000` (the test split of the 12 sequence_labelling run-*.yaml files) is built from
  `synthetic_changing_shapes/test`; regenerating that dataset makes every later test evaluation score labels of
  sequences that no longer exist (reproduced: 3 of 6 and 5 of 5 frames contradicting their subset label).
- **Fix:** store the SHA-256 of the source labels.json in the subset and refuse or warn when it changes; or empty
  `data/` in `generate_split` and accept a seed.

#### B3. Duplicates are merged only on identical hashes: resized copies of one photo stay in both train and test
- **Area:** data generation · **Severity:** medium · **Status:** confirmed (measured by the reviewer on local Open
  Images photos) · **Category:** bug
- **Where:** `vision_training/data/classification/foundation_model_image_downloader.py:1145-1176`
  (`merge_duplicate_samples`), `1222-1227` (`build_manifests`), `320-333` (docstring).
- **Problem:** samples are grouped by exact perceptual hash, while the docstring promises that no picture "appears in
  both the training and the test images" and that the hash survives resizing; copies of one photo at another resolution
  often differ by 1 to 3 bits, and only the ImageNet-1k exclusion uses the 3-bit tolerance.
- **Failure scenario:** a Flickr photo in the Tencent ImageNet train list and, larger, in the Open Images validation
  list: 1 bit apart, it lands in both train and test manifests (57 of 300 resized copies would not be merged; how many
  such copies the lists hold was not measured).
- **Fix:** keep exact-hash label merging, then drop the train and validation samples within `EXCLUSION_MAXIMUM_DISTANCE`
  of a test sample (`build_hash_chunk_index`, `is_near_copy`), and correct the docstrings (a 3-bit tolerance also
  matches 4 pairs of unrelated simple pictures among 41,606, so label merging should stay exact).

#### B4. data.json crashes at launch on a self-supervised split mixing labelled and unlabelled images
- **Area:** data loading · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer, code read again) ·
  **Category:** bug
- **Where:** `vision_training/data/data_record.py:174-182` (`count_manifest_classes`), called by
  `BaseTrainer.save_data_record` at every launch.
- **Problem:** once any sample has a category, every sample without `category_ids` is read with `sample["category_id"]`,
  while `ClassificationDataset(keep_unlabelled_samples=True)` accepts samples with neither key (README: "with its
  `category_id`, its `category_ids` ... or neither").
- **Failure scenario:** a self-supervised training split where only some images carry `category_id`, with a non-empty
  label mapping: the loaders build, then the launch stops with `KeyError: 'category_id'`.
- **Fix:** count only the samples that have a category (`[]` when neither key is present).

#### B5. data.json crashes at launch on a COCO split without "annotations", which the loaders accept
- **Area:** data loading · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer, code read again) ·
  **Category:** bug
- **Where:** `vision_training/data/data_record.py:165-170`; the loader accepts it at `coco_dataset.py:197-203`,
  `split_description.py:29` counts 0.
- **Problem:** `labels["annotations"]` is read without the check that `CocoDetectionDataset` and `split_description.py`
  make for unlabelled splits.
- **Failure scenario:** `test_data_folder` pointing to an unlabelled COCO split (the default "test" split of
  `data_retriever.py`): a DET, INS or KPT launch stops with `KeyError: 'annotations'`.
- **Fix:** iterate over `labels["annotations"] if "annotations" in labels else []`.

#### B6. data.json overstates the keypoint objects: it counts the objects the keypoint loader drops
- **Area:** data loading · **Severity:** medium · **Status:** confirmed (measured by the reviewer on COCO 2017
  validation) · **Category:** documentation
- **Where:** `vision_training/data/data_record.py:164-170`; the loader drops them at `coco_dataset.py:341-347`; README
  line 198 and `build_class_records` say the counts are the trained data.
- **Problem:** for KPT every manifest annotation of a mapped category is counted, while `add_keypoints` removes the
  objects without a labelled keypoint (crowds, persons without keypoints); these counts also feed
  `compare_best_evaluations.py`.
- **Failure scenario:** COCO 2017 validation: data.json gives person 11004 annotations in 2693 images, the loader keeps
  6352 objects in 2346 images (+73 %).
- **Fix:** count, for KPT, only the annotations with a visible keypoint (the loader rule), or document that the counts
  are manifest annotations.

#### B7. Motion blur of the views: the largest kernel size is never drawn, and the viewer describes kernels kornia does not apply
- **Area:** data loading · **Severity:** medium · **Status:** fixed on 2026-10-20 (found confirmed, reproduced by the reviewer; kornia 0.8.3 source
  read again), the views being drawn by the types of `Data.data_augmentation` since 2026-10-22, `view_augmentations.py`
  removed · **Category:** bug (added on 2026-10-20 with the harder views)
- **Where:** `vision_training/data/view_augmentations.py:180-183` (construction), `252-254` (description); kornia
  `random_generator/_2d/motion_blur.py:84-100` and `_2d/intensity/motion_blur.py:113-122`.
- **Problem:** kornia draws `Uniform(low // 2, high // 2).int() * 2 + 1`, so `[3, 9]` only gives 3, 5 or 7, and it blurs
  the whole batch with one kernel size (`ksize_factor[idx]`), the angle and direction staying per image; the description
  writes the per-image `ksize_factor`, which is not the applied kernel.
- **Failure scenario:** every self-supervised configuration (`kernel_size: [3, 9]`): kernel 9 never applied, and the
  `view_input_images` panel shows e.g. "kernel 5" on an image blurred with 7 (reproduced: the applied blur matches
  kornia's `motion_blur` with 7 exactly).
- **Fix:** pass `(low, high + 2)` (or document the exclusive top), describe the batch kernel `ksize_factor[idx]` (drawn
  per batch), or blur image by image.
- **Done:** `build_view_transformations` passes `(low, high + 2)` to kornia, so that every odd size of
  `kernel_size` is drawn, both bounds included, and `describe_view_parameters` writes the kernel size
  kornia applies to the batch, `ksize_factor[idx]`. The docstrings, the comments of the 20
  configurations, the README and the self-supervised workflow say that the kernel size is drawn once per
  batch. `test_motion_blur_draws_every_kernel_size_and_describes_the_applied_one` (it failed before
  the fix on both counts) checks the four sizes and the view rebuilt from the described values.

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

#### B9. A batch-normalized classification head crashes on a training batch of a single vector
- **Area:** core models · **Severity:** medium · **Status:** fixed on 2026-10-21 (found confirmed, reproduced by
  the reviewer) · **Category:** bug
- **Where:** `vision_training/models/components/fully_connected_block.py:35, 85-88`; `classification_head.py:181-189`;
  `loaders.py:244` (`drop_last` only for SSL), `359-361` (sequence loaders);
  `configuration_processing/classification_checks.py:97-112` (any batch size accepted).
- **Problem:** `fully_connected_normalization: "batch"` is valid for CLS, M2O, CMP, M2M and SEL, but BatchNorm1d raises
  in training on one vector, and the training loaders keep their last incomplete batch (unlike item L17 of
  suggestions.md, this fails at any image size).
- **Failure scenario:** a training split with `len(split) % batch_size == 1` (9 images with a batch of 8): the last
  batch of every pass raises "Expected more than 1 value per channel when training", the trainer skips it with a yellow
  error, and that sample is never trained on; `batch_size: 1` fails at every step.
- **Fix:** `drop_last=True` for the training loaders when the head or backbone has batch statistics, and refuse
  `batch_size: 1` with this head in the checker.

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

#### B11. Ten failed batch fetches in a row leave through exit(1): the final checkpoint is never saved
- **Area:** training loop · **Severity:** medium · **Status:** fixed on 2026-10-21 (found confirmed, reproduced
  by the reviewer, code read again) · **Category:** error-handling
- **Where:** `vision_training/trainers/base_trainer.py:1215-1231` (`get_next_batch`, `exit(1)` at 1230), called at
  1387-1388 and 1410-1411; final save at 1466; `run_training_loop` 1327-1334.
- **Problem:** `exit(1)` raises `SystemExit`, which neither the loop's `except Exception` nor `run_training_loop` (only
  `KeyboardInterrupt`) catches, so the final `save_model` of every other way out of the loop never runs (the
  classification workflow says "Whatever the reason, the loop ends with a last save").
- **Failure scenario:** the data mount drops or the dataset folder is regenerated during a run: every step since the
  last periodic checkpoint is lost (up to `intermediate_weight_saving_iterations`).
- **Fix:** raise an ordinary exception (or break out of the loop) so that the final save runs, printing the exception
  with its traceback.

#### B12. Ctrl+C during a periodic save is ignored by the trainer but kills the persistent loader workers: the run fails 5 iterations later
- **Area:** training loop · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer with a SIGINT to
  the process group during `torch.save`) · **Category:** error-handling
- **Where:** `vision_training/trainers/checkpointing.py:151-170` (`SIG_IGN` during `save_checkpoint`),
  `vision_training/data/loaders.py:64-67` (`persistent_workers=True`), torch `utils/data/_utils/worker.py` (a worker
  ends on `KeyboardInterrupt`).
- **Problem:** a terminal Ctrl+C reaches the whole process group; the main process drops it while saving instead of
  deferring it, but the workers exit, so training continues on dead workers: every iteration fails with "DataLoader
  worker ... exited unexpectedly", the run stops after 5 errors as FAILED instead of KILLED, and saves and evaluates
  stale weights under new iteration numbers in between (README: "Ctrl+C stops the training and saves a checkpoint of the
  current iteration").
- **Failure scenario:** Ctrl+C during the save of iteration 4: iterations 4 to 8 fail, checkpoints 4, 6, 8 and 9 hold
  the same weights, Results.txt holds an "Iteration 8" evaluation of them, and a resume starts at 10.
- **Fix:** record the SIGINT during the save and raise `KeyboardInterrupt` once the save is complete; optionally ignore
  SIGINT in the workers (`worker_init_fn`) so that only the main process shuts them down.

#### B15. In MLflow, test/best_<metric> of a standalone "all" evaluation shows the last evaluated value, not the best
- **Area:** evaluation and tracking · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer on a
  scratch database) · **Category:** bug
- **Where:** `vision_training/tracking/mlflow_tracker.py:775-810` (`add_best_metric`),
  `scripts/evaluation/evaluate_checkpoint.py:75` (`[::-1]`), `checkpoint_evaluation.py:116-129`.
- **Problem:** the best value so far is logged at the step of each evaluated iteration, and MLflow summarises a metric
  by its value at the highest step, while the standalone evaluation goes from the highest iteration down.
- **Failure scenario:** 300 (0.90), 200 (0.95), 100 (0.80) evaluated in that order: tag `best_iteration` 200, but the
  runs table shows `test/best_Accuracy` 0.9.
- **Fix:** evaluate in increasing iteration order before tracking, or log the best value once at close at the step of
  the best iteration.

#### B16. One artifact write failing with a permanent OSError blocks every later MLflow write and leaves the run RUNNING
- **Area:** evaluation and tracking · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer for
  PermissionError; HTTP 4xx traced) · **Category:** error-handling
- **Where:** `vision_training/tracking/mlflow_tracker.py:505` (every OSError retried), `459-470` (operations in turn),
  `845-865` (`write_artifact`).
- **Problem:** every OSError is treated as "MLflow unavailable" and retried until close, including permanent ones
  (PermissionError on a read-only artifact folder, `requests.HTTPError` for an HTTP 4xx such as a 413 on a large
  confusion matrix), so the worker stays on that artifact and every later metric batch waits and is abandoned at close,
  `set_terminated` never running.
- **Failure scenario:** a read-only artifact folder: the configuration artifact logged at start blocks the run, no
  metric reaches MLflow and the run stays RUNNING.
- **Fix:** drop an artifact after a few failures or treat PermissionError and HTTP 4xx as permanent, and do not make
  metrics wait behind artifact copies.

#### B17. An exported frame selector with averaged (EMA) weights gets the confidence thresholds of the trained weights
- **Area:** deployment and scripts · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer, code read
  again) · **Category:** bug
- **Where:** `vision_training/deployment/frame_selection_deployment.py:68` (`build_decoding_settings`),
  `export_model.py:265-269`; the averaged weights are evaluated in `Evaluation/Iteration_<n>/EMA/`
  (`base_trainer.py:1747-1749`).
- **Problem:** with `use_averaged_weights=True` the moving average is exported, but `build_decoding_settings` always
  reads `Evaluation/Iteration_<n>/class_thresholds.json`, the thresholds of the trained weights; the
  `EMA/class_thresholds.json` of the exported weights is never read.
- **Failure scenario:** a frame selection run with EMA exported with averaged weights: the Triton decoder's
  `decoding_settings.json` holds thresholds of other weights, so the deployed `low_confidence` flags are wrong, without
  warning.
- **Fix:** pass `use_averaged_weights` to `build_decoding_settings` and read the `EMA/` thresholds for averaged weights,
  warning when they are missing.

#### B18. A maximum batch size of 1 crashes the ONNX export, and at 2 the dynamic batch is never checked
- **Area:** deployment and scripts · **Severity:** medium · **Status:** confirmed (reproduced by the reviewer, code read
  again) · **Category:** bug
- **Where:** `vision_training/deployment/export_specifications.py:363-365, 398-399`; `export_model.py:151-160, 181-182,
  223-227`.
- **Problem:** the traced batch (2) and the checked batch (3) are both capped at `maximum_batch_size`: at 1,
  `torch.export.Dim("batch", min=1, max=1)` is refused by torch after the TorchScript model and the ONNX folder were
  written; at 2, traced and checked batches are equal, so the "batch of another size" check silently does not happen
  (`maximum_sequence_length = 1` fails the same way).
- **Failure scenario:** a large sequence model exported with `maximum_batch_size = 1` dies with an AssertionError and
  leaves an empty ONNX model folder in the served repository.
- **Fix:** validate the setting up front (static batch or clear error at 1) and always check a batch size different from
  the traced one.

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

#### B20. The view augmentation checks accept values kornia rejects
- **Area:** configuration · **Severity:** low · **Status:** fixed on 2026-10-20 (found confirmed, reproduced by the reviewer; the solarize bound
  checked again in kornia), the view augmentation checks removed on 2026-10-22 with the views, now drawn by the types of
  `Data.data_augmentation` with hard-coded strengths · **Category:** configuration (the solarize bound was added on
  2026-10-20)
- **Where:** `vision_training/configuration_processing/common_checks.py:47, 61` (solarize addition), `620-628` (blur
  kernel), `572-576` (crop aspect ratio).
- **Problem:** kornia's `solarize` requires an addition strictly inside (-0.5, 0.5) (`kornia/enhance/adjust.py:724`) but
  the check accepts ±0.5; a Gaussian blur kernel larger than the image and extreme crop aspect ratios are accepted too.
  All crash at the first view (the shipped configurations use none of these values).
- **Failure scenario:** `solarize.addition: [0.5, 0.5]` raises an AssertionError; `gaussian_blur.kernel_size: 65` at 32
  pixels raises a padding RuntimeError.
- **Fix:** exclusive addition bounds, a blur kernel bounded by the image size, a capped aspect ratio range.
- **Done:** `check_number_range` takes an excluded maximum, and the solarize addition must lie
  strictly within (-0.5, 0.5). The Gaussian blur kernel must be below twice the input size.
  `check_crop_sides` refuses a crop `scale` and `aspect_ratio` that can give crops thinner than
  `MINIMUM_CROP_SIDE` (2) pixels at the input size, from the draw and the centre crop fallback of
  kornia. Over 180 crop settings at 32 and 256 pixels, none of the 97 accepted crashed and 71 of the
  83 refused did. The checker test has a new case (it failed before the fix), and the expected
  message of the solarize addition shows the excluded bounds.

#### B21. The loader coerces malformed values before the checks, and some malformed sections crash the checker
- **Area:** configuration · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) · **Category:**
  error-handling
- **Where:** `vision_training/configuration_processing/configuration_loader.py:119-127`; `common_checks.py:126-128,
  190-192, 254-261, 312-320`; `segmentation_checks.py:107`; `run_name.py:150`.
- **Problem:** the loader applies `int(float())` to the iteration counts and `float()` to the learning rate before the
  checks: the boolean rejection never fires, fractions are truncated and null gives a TypeError; `Backbone.convolutions`
  or `SegmentationDecoder.convolutions` written as lists, `data_augmentation.types: null` or `inception_starting_block:
  2.0` crash the checker, and `inception_starting_block: True` passes and names the run `INCTrue`.
- **Failure scenario:** `learning_rate: True` trains at 1.0; `number_iterations: null` ends with a traceback.
- **Fix:** convert only numeric strings in the loader, check container types before `.keys()`, `.items()` and `set()`,
  validate `inception_starting_block` with `check_positive_integers`.

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

#### B23. The DET checks require mask head settings whenever a MaskHead section is present
- **Area:** configuration · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) · **Category:**
  configuration
- **Where:** `vision_training/configuration_processing/detection_checks.py:118-119, 420`.
- **Problem:** the checks look at the presence of `MaskHead` instead of the task (README line 1090: a section only used
  by another task is ignored).
- **Failure scenario:** a DET file with a leftover `MaskHead` section is refused with demands for `mask_head` levels and
  a `mask_loss` weight the detector never reads.
- **Fix:** pass the task (or a mask head flag) to `check_detection_configuration`.

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

#### B25. Failed image writes are ignored, so the labels can list images never written
- **Area:** data generation · **Severity:** low · **Status:** confirmed (reproduced by the reviewer with a read-only
  folder) · **Category:** error-handling
- **Where:** `vision_training/data/synthetic/create_synthetic_dataset.py:1741, 1750, 1757, 2388`.
- **Problem:** `cv2.imwrite` returns False instead of raising, and no call checks it; the only trace is an OpenCV
  warning among the output of 8 workers.
- **Failure scenario:** a disk filling during a 500,000 image generation: success reported with missing images
  (reproduced: labels list 5 images, 0 on disk).
- **Fix:** check every `cv2.imwrite` and raise an OSError naming the path.

#### B26. Image tasks crash with an obscure numpy error on images smaller than 34 pixels
- **Area:** data generation · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) · **Category:** bug
- **Where:** `vision_training/data/synthetic/create_synthetic_dataset.py:1584-1585` (`max(33, image_size // 3)`),
  `1453`, `1524-1525`.
- **Problem:** the smallest shape canvas is 34 pixels (17 for the hollow sequence shapes), larger than such images, so
  `integers(high <= 0)` raises inside a worker without mentioning the image size.
- **Failure scenario:** `create_synthetic_dataset(image_size=32, task="classification")` (and DET, INS, SEG; M2M and CMP
  at 16): `ValueError: high <= 0`.
- **Fix:** validate the image size against the canvas of every task, or cap the shape size at `image_size - 1`.

#### B27. Two cached files are written in place, so an interrupted write is trusted on the next run
- **Area:** data generation · **Severity:** low · **Status:** confirmed (reproduced by the reviewer with truncated
  files) · **Category:** error-handling
- **Where:** `vision_training/data/classification/foundation_model_image_downloader.py:457-470` (`np.save`), `981-990`
  (`write_text`).
- **Problem:** every other cache of the module is written to `.part` and renamed, these two directly, and only their
  existence is checked next time.
- **Failure scenario:** an interruption while saving the 86 MB wanted keys array: every rerun stops with `ValueError:
  mmap length is greater than file size` until `progress/imagenet_wanted_keys.npy` is found and deleted.
- **Fix:** write both to `.part` and rename, as `download_file` and `write_records` do.

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

#### B30. A category mapped to 0 to exclude it names the background class and is counted under it in data.json
- **Area:** data loading · **Severity:** low · **Status:** fixed on 2026-10-21 (found confirmed, reproduced by the
  reviewer) · **Category:** bug
- **Where:** `vision_training/data/label_mapping.py:59-61, 83-97` (`get_class_names`), `data_record.py:132-139`; compare
  `coco_dataset.py:101-104`.
- **Problem:** a category mapped to 0 is left out of training, but `get_class_names` names class 0 after it and
  `map_categories_to_classes` counts its annotations as class 0, while README line 186 says the background has no
  categories.
- **Failure scenario:** DET mapping `{"1": 1, "2": 0, "3": 2}`: class names `{0: 'bicycle', 1: 'person', 2: 'car'}`;
  reports, viewers and data.json call the background "bicycle"; a transfer comparing class names sees class 0 renamed.
- **Fix:** for the tasks with a background class, always name class 0 "background" and leave the categories mapped to 0
  out of the names and counts.

#### B31. get_image_paths ignores .tif and .webp images
- **Area:** data loading · **Severity:** low · **Status:** confirmed (read in the code) · **Category:** bug
- **Where:** `vision_training/utilities/os_utilities.py:228` (`IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp",
  ".tiff"}`).
- **Problem:** `.tif` (the usual spelling of TIFF) and `.webp`, both read by OpenCV, are not listed, and the function
  serves the manifest builders, the inference inputs and the clustering script.
- **Failure scenario:** a folder of `.tif` images: "No image found", although the images are readable.
- **Fix:** add `.tif` and `.webp` (the suffixes are already compared in lower case, line 237).

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

#### B34. SequenceClassifier and SequenceLabeller are near copies
- **Area:** core models · **Severity:** low · **Status:** fixed on 2026-10-20 (found confirmed, diff of the two files) · **Category:** duplication
- **Where:** `vision_training/models/sequence/sequence_classifier.py:31-80, 110-130` against
  `sequence_labeller.py:33-82, 114-134`; the same backbone, pooling, head and class weights set-up in
  `image_classifier.py:48-80` and `sequence_comparator.py:57-102`.
- **Problem:** `__init__`, `print_summary` and `get_size_components` are identical except `keep_time_axis`, so a fix to
  one must be repeated by hand in the other.
- **Fix:** one shared base (or builder) parameterised by `keep_time_axis`, keeping only `forward` per task.
- **Done:** `SequenceFrameModel` (`models/sequence/sequence_model.py`) builds the backbone, pooling,
  temporal module, head and class weights from `keep_time_axis`, with `compute_class_outputs`,
  `print_summary` and `get_size_components`; the classifier and the labeller only keep their loss
  in `forward`. The state_dict keys and the outputs of the classifier, labeller and frame selector are
  the same as before (63, 63 and 64 keys, outputs equal).

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

#### B36. The RPN classification loss is NaN when no anchor of the batch is sampled
- **Area:** detection and keypoint models · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** bug
- **Where:** `vision_training/models/detection/target_assignment.py:165-176, 190-194, 229-231`;
  `losses/detection_losses.py:58-63` (mean over an empty selection).
- **Problem:** with `RegionProposal.background_iou_threshold.min` above 0 (accepted with a warning), anchors overlapping
  no object are ignored and an object whose best anchor IoU is 0 resurrects none, so a batch of such objects (zero-width
  boxes, see L8, or boxes outside the anchors) samples no anchor; the localisation loss already clamps its count.
- **Failure scenario:** background min 0.2 and one image whose only box is `[100, 100, 100, 180]`: the RPN
  classification loss and the total loss are NaN (the gradients stay finite).
- **Fix:** sum the BCE and divide by the sampled count clamped to 1, or label every anchor background in an image where
  no object has a positive best IoU.

#### B37. get_box_dimensions_and_centers ends the process with exit() and status 0 on a wrong box shape
- **Area:** detection and keypoint models · **Severity:** low · **Status:** confirmed (reproduced by the reviewer; the
  other `exit` calls of the repository checked) · **Category:** error-handling
- **Where:** `vision_training/models/detection/box_operations.py:113-119`.
- **Problem:** `exit()` raises `SystemExit(None)`, which no `except Exception` catches and which ends the process with
  code 0, so a crash looks like a success to scripts and schedulers (the other calls are deliberate:
  `base_trainer.py:273` after the input viewer, `configuration_loader.py:178` after the configuration errors, and
  `base_trainer.py:1230` is [B11](#b11-ten-failed-batch-fetches-in-a-row-leave-through-exit1-the-final-checkpoint-is-never-saved)).
- **Failure scenario:** `apply_regression_predictions` given boxes of shape `[2, 5]`: a red message, then exit code 0.
- **Fix:** raise a ValueError with the same message.

#### B38. A detection head transfer from a run saved before DetectionLevels fails with a bare KeyError
- **Area:** detection and keypoint models · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** error-handling
- **Where:** `vision_training/models/model_transfer.py:423-424, 433, 443, 456, 968`.
- **Problem:** `compare_detection_heads` and the anchor message read `source_configuration["DetectionLevels"]` strictly,
  while the other components report a setting missing from an older source run with the instruction to add it (lines
  277-281, 343-347); the section exists since a10bb7a (2026-10-15).
- **Failure scenario:** `Detector.pretrained.experiment_folder` pointing to an older run: `KeyError: 'DetectionLevels'`
  without hint (the 8 local detection runs already have the section).
- **Fix:** return the same "missing from the source run" difference as the other `compare_*` functions.

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

#### B41. DetectionHead.forward documents a tensor_to_concatenate argument it never reads
- **Area:** detection and keypoint models · **Severity:** low · **Status:** confirmed (grep: no reader, no caller) ·
  **Category:** documentation
- **Where:** `vision_training/models/detection/detection_head.py:93, 107`.
- **Problem:** extra features passed by a caller are dropped silently.
- **Fix:** remove the parameter, or implement the concatenation with a test.

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
- **Fix:** skip the rearrangement and the postprocessing when ground truth is given (also the fix of item H3 of
  suggestions.md), or rearrange with one argsort and gather.

#### B43. Resuming by launching the run's own training_configuration.yaml crashes with SameFileError
- **Area:** training loop · **Severity:** low · **Status:** confirmed (reproduced by the reviewer, code read again) ·
  **Category:** bug
- **Where:** `vision_training/trainers/base_trainer.py:812, 829` (`copyfile` of the launched file onto `<run
  folder>/training_configuration.yaml`).
- **Problem:** when the launched file is the run's own copy, `shutil.copyfile` raises `SameFileError`, after
  `compare_with_saved_configuration` has run.
- **Failure scenario:** `--configuration_path <run>/training_configuration.yaml` (resumed or not) stops at launch.
- **Fix:** skip the copy when `os.path.samefile(initial, destination)`.

#### B44. Evaluated checkpoints are overwritten, or left unevaluated, at stop and resume boundaries
- **Area:** training loop · **Severity:** low · **Status:** fixed on 2026-10-21 (found confirmed, reproduced by the
  reviewer, two cases) ·
  **Category:** bug / documentation
- **Where:** `vision_training/trainers/base_trainer.py:1366-1367, 1379-1384` (comments and the `training_iteration !=
  self.start_iteration` test), 1466 (final save).
- **Problem:** (a) a Ctrl+C after the training step of a weight saving iteration (during its validation or summary)
  makes the final save overwrite the checkpoint just evaluated with weights one step later; (b) a run resumed from a
  checkpoint at a weight saving iteration minus one saves the next weight saving checkpoint at its first iteration and
  never evaluates it.
- **Failure scenario:** Results.txt, TensorBoard and MLflow "Iteration k" describe other weights than
  `Iteration_k/model_k.pt`, which the best-checkpoint tools pick; in case (b) a checkpoint is missing from the
  evaluation curve.
- **Fix:** save and evaluate after the step (also item M20 of suggestions.md); until then, never overwrite a checkpoint
  already saved at the current iteration and evaluate the first checkpoint of a resumed run when it falls on a weight
  saving iteration.

#### B45. The console summary treats the correlation metrics as higher-is-better: a rising off-diagonal correlation prints green with "Best"
- **Area:** training loop · **Severity:** low · **Status:** confirmed (reproduced by the reviewer, code read again) ·
  **Category:** bug (misleading output)
- **Where:** `vision_training/trainers/metric_tracking.py:31-33` (`LOWER_IS_BETTER_SUFFIX = "_loss"`), 185-197, 244-255;
  tracked keys at `self_supervised_trainer.py:38-41`, `classification_trainer.py:51-55, 206-207`.
- **Problem:** only keys ending in `_loss` improve when they fall, but `cross_correlation_off_diagonal` (Barlow Twins,
  also in the view losses) and `hidden_feature_correlation` (decorrelation) are mean absolute off-diagonal correlations
  that should fall.
- **Failure scenario:** a degrading redundancy reduction is shown as progress, the "Best" marker following the worst
  value.
- **Fix:** let the tasks declare the direction of every tracked metric (or keep an explicit set of lower-is-better
  keys).

#### B46. "Iterations Per Epoch" and the sample counts logged to MLflow come from the manifest before the label mapping
- **Area:** training loop · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) · **Category:** bug
  (misleading output)
- **Where:** `vision_training/trainers/classification_trainer.py:225-236` (`describe_split` → `count_split_samples`),
  `base_trainer.py:995-1000, 1496-1502`.
- **Problem:** the classification loader leaves out the samples of unmapped categories, so the printed epoch length and
  the MLflow parameters `number_*_samples` and `iterations_per_epoch` are overstated when the mapping keeps a subset of
  the categories.
- **Failure scenario:** 18 of 35 categories mapped on 40 training images: "Iterations Per Epoch : 10" against 7 real
  batches (batch 4, 25 kept samples).
- **Fix:** compute the epoch length and the counts from the built datasets (`len(dataset)`).

#### B47. The self-supervised k-NN report re-implements the per-class rows, table and scalars of class_metrics_report.py
- **Area:** training loop · **Severity:** low · **Status:** fixed on 2026-10-20 (found likely, code paths compared by the reviewer) · **Category:**
  duplication
- **Where:** `vision_training/trainers/self_supervised_trainer.py:286-306` against
  `vision_training/evaluation/classification/class_metrics_report.py:17-92`.
- **Problem:** a parallel implementation differing only by the absent threshold columns and a "k-NN" prefix, against the
  rule of extending existing functions.
- **Failure scenario:** a fix of the per-class report reaches the classification but not the self-supervised task.
- **Fix:** make the thresholds optional and the prefix a parameter of the three helpers, and call them from the
  self-supervised trainer.
- **Done:** `build_class_metrics_rows` takes optional thresholds, `build_class_metrics_table` an
  optional target precision with a title, a legend introduction and a support description, and
  `build_class_metric_scalars` a tag prefix; the self-supervised trainer calls them. Its metrics,
  table, scalars and class_metrics.csv are the same as before on fixed representations.

#### B48. The DET and INS input viewer captions the boxes with raw category ids instead of class names
- **Area:** task trainers and losses · **Severity:** low · **Status:** confirmed (reproduced by the reviewer on the
  synthetic shapes mapping) · **Category:** bug
- **Where:** `vision_training/trainers/detection_trainer.py:685-686` (used at 688-704, inherited by
  `InstanceSegmentationTrainer`); `tests/trainers/test_input_viewer_panels.py:93`.
- **Problem:** `label_names` inverts `self.label_dictionary`, whose keys are the raw category ids, so the box captions,
  the "Objects : n (...)" line and the removed-object lines show "1", "2", ... (with merged categories the inversion
  keeps only the last id), against the comment "so that each box is captioned with its class name"; `KeypointTrainer`
  uses `self.class_names`, and the DET test passes only because it uses a name-keyed dictionary.
- **Failure scenario:** the input viewer of a DET run on the synthetic shapes captions "1", "2", "3" instead of
  "square", "circle", "triangle"; with `{"1": 1, "2": 1}`, class 1 is captioned "2".
- **Fix:** `label_names = self.class_names`, as `KeypointTrainer` does, and an id-keyed dictionary in the test.

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

#### B50. KeypointTrainer copies four DetectionTrainer methods, and FrameSelectionTrainer redefines the describe_split it inherits
- **Area:** task trainers and losses · **Severity:** low · **Status:** fixed on 2026-10-20 (found confirmed, AST comparison of the method bodies by
  the reviewer) · **Category:** duplication
- **Where:** `vision_training/trainers/detection_trainer.py:118, 189, 562, 572` against `keypoint_trainer.py:106, 137,
  338, 348` (`list_class_names`, `describe_split`, `get_test_sample_classes`, `get_test_sample_name`; the test manifest
  reading of `build_dataloaders` too, 97 against 88); `frame_selection_trainer.py:134` against
  `sequence_classification_trainer.py:106`.
- **Problem:** identical bodies kept in two places, against the rule of extending existing functions: a fix to one copy
  misses the other.
- **Fix:** move the COCO split helpers into a module or mixin shared by both trainers, and delete
  `FrameSelectionTrainer.describe_split`.
- **Done:** `CocoSplitTrainer` (`trainers/coco_split_trainer.py`) holds `read_test_manifest`,
  `list_class_names`, `describe_split`, `get_test_sample_classes` and `get_test_sample_name`,
  inherited by the detection and keypoint trainers, and `FrameSelectionTrainer.describe_split` is
  deleted.

#### B51. Keypoint "Matched Objects" and mean pixel error count predictions far from the object
- **Area:** evaluation and tracking · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** bug
- **Where:** `vision_training/evaluation/keypoints/object_keypoint_similarity.py:164` (match when OKS > 0), `299-302`.
- **Problem:** a ground truth is matched whenever its float64 OKS is above 0, which only fails on underflow (about 3.9
  box sides away), so a far prediction counts as matched and its distance enters "Mean Keypoint Error (pixels)".
- **Failure scenario:** a 20x20 box with its only prediction 60 pixels away: Matched Objects 1.0, Mean Keypoint Error
  60.0, PCK 0.0.
- **Fix:** require a minimum OKS for a match (0.5 as in AP@0.5, or a setting).

#### B52. Tied probabilities give every tied frame rank 1, inflating top-k selection accuracy and mean rank
- **Area:** evaluation and tracking · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** bug
- **Where:** `vision_training/evaluation/sequence/frame_selection_metrics.py:90-125`, ranks from
  `models/sequence/frame_selection_decoding.py:231`.
- **Problem:** tied frames take the best of their ranks (documented in `compute_frame_ranks`), so a collapsed model, or
  frames whose float32 softmax saturates at 1.0, score perfectly although the decoding must pick one.
- **Failure scenario:** uniform probabilities on 5 sequences: Top-1 Selection Accuracy 1.0, Mean True Frame Rank 1.0.
- **Fix:** break ties pessimistically or use the average rank of the tie.

#### B53. Mean IoU, mean Dice and macro F1 leave out the classes predicted but absent from the ground truth
- **Area:** evaluation and tracking · **Severity:** low · **Status:** confirmed (measured against the standard
  definitions and scikit-learn) · **Category:** bug
- **Where:** `vision_training/evaluation/segmentation/segmentation_metrics.py:65-71`;
  `classification/classification_metrics.py:74-79, 127-131`.
- **Problem:** documented, but unlike mmseg and Cityscapes (every class with a non-empty union) and scikit-learn's macro
  F1 (labels of y_true ∪ y_pred), a hallucinated class is not penalised in the headline.
- **Failure scenario:** masks of classes 0 and 1, class 2 painted on a quarter of the pixels: IoU [0.5, 1, 0], Mean IoU
  reported 0.75 where the standard gives 0.5; macro F1 differs from scikit-learn by up to 0.28 on random inputs
  (accuracy, balanced accuracy, per-class metrics and top-k match exactly).
- **Fix:** average over the classes with TP + FP + FN > 0, or name these metrics "over present classes".

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

#### B55. A detection test split without objects reports mAP and mAR 0.0 instead of no value
- **Area:** evaluation and tracking · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** bug
- **Where:** `vision_training/evaluation/detection/mean_average_precision.py:366-368` against `494-496`.
- **Problem:** without ground truth in any class, the overall mAP and mAR are 0.0 while the per-size results are None
  ("—").
- **Failure scenario:** a negatives-only split given to `evaluate_checkpoint.py`: `mAP@0.5 area = 0.0` reported, logged
  and compared as a real value.
- **Fix:** return None and skip the scalars without ground truth, handling None in the best checkpoint comparison.

#### B56. evaluate_checkpoint.py's default error image setting stops SEG, KPT and SSL runs with a bare KeyError
- **Area:** deployment and scripts · **Severity:** low · **Status:** confirmed (reproduced by the reviewer) ·
  **Category:** error-handling
- **Where:** `scripts/evaluation/evaluate_checkpoint.py:36, 59-62`.
- **Problem:** the default `maximum_error_images_per_class_pair = 100000` is applied by reading
  `experiment_configuration["evaluation"]["error_dumps"]`, which the segmentation, keypoint and self-supervised
  configurations do not have.
- **Failure scenario:** the script pointed at a SEG, KPT or SSL run without editing that setting stops with `KeyError:
  'error_dumps'`.
- **Fix:** skip the override with a message when the section is missing, or raise a ValueError naming the setting.

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

The status at `165339b` of every item of the
[Repository Audit For Instance Segmentation](suggestions.md#repository-audit-for-instance-segmentation). Their
descriptions and fixes stay in that file.
- **Open:** the defect described is still in the code.
- **Partly fixed:** only the part given in the note remains.
- **Fixed:** as listed at the top of `suggestions.md`. The items dropped by the user are deleted from the table.

Statuses come from reading the code at this commit. The items re-found by the reviewers were also reproduced.

| Item | Severity there | Status | Note |
|---|---|---|---|
| C2. Label smoothing with class weights favours the rarest class | critical | **open** (latent) | see below |
| C3. DET and INS count their classes as dictionary keys | critical | fixed | on 2026-10-21: classes counted as distinct values, detection mapping checked |
| C4. Run names collide for settings that change the weights | critical | fixed | on 2026-10-21: the anchors (`A<counts>.<smallest>-<largest>.<hash>`) and `last_max_pooling` (`LMP`) give tokens, and M17 keeps the classes of a run; its `dtype` and `input_channels` points are moot |
| H1. The RPN never trains on plain background | high | fixed | `background_iou_threshold.min: 0.0` in the four DET and INS files, warning above 0 (`detection_checks.py:226-234`) |
| H2. Ground truth boxes never added to the proposals | high | fixed (`a10bb7a`) | `add_ground_truth_proposals`, `two_stage_detector.py:456-461` |
| H3. Validation crashes when samples x classes < maximum_detections | high | fixed (`a10bb7a`) | candidates padded before `topk` (`postprocessing.py:92-100`); validation still runs the unused postprocessing ([B42](#b42-training-and-validation-forwards-of-the-detector-do-inference-only-work-with-one-host-synchronisation-per-image)) |
| H4. `residual_nets` does nothing in channel-changing blocks | high | **open** | see below |
| H6. An unreadable image silently drops its batch | high | fixed | on 2026-10-21: unreadable images named, their batches dropped, counted and reported without stopping the training |
| M1. Squeeze-and-excitation gates the residual sum | medium | fixed (`f66c7a3`) | |
| M2. The shortcut is added after the ReLU | medium | fixed (`57eae97`) | |
| M3. Proposals selected jointly over the levels | medium | **open** | `region_proposer.py:88-103`, shared by `InferenceDetector` |
| M4. The two NMS thresholds look swapped | medium | fixed | 0.7 and 0.5 in the four DET and INS files |
| M5. Masks resized with plain nearest are shifted | medium | **partly fixed** | see below; also `preprocessing.py:200-202` restores class maps with `mode="nearest"` |
| M6. `number_iterations` counts from the resume point | medium | fixed | on 2026-10-21: `number_iterations` is the total of the run |
| M7. `resume_training: false` reuses the run folder | medium | fixed | on 2026-10-21: the previous training is moved to `v<n>_<run name>` |
| M8. An empty split loops forever | medium | fixed | on 2026-10-21: refused at launch, two empty passes raise |
| M9. Training step errors swallowed, stale final save | medium | **open** | |
| M10. Weight decay on every parameter | medium | **open** (latent) | one `weight_decay` for every group (`base_trainer.py:606-613`); 0.0 in the files read |
| M13. Decoded image size never compared with the annotations | medium | **partly fixed** | see below; `coco_dataset.py:249` still never compares the decoded shape with the JSON size |
| M14. Objectness ranked in bfloat16 | medium | **open** (latent) | `region_proposer.py:88` without `.float()`; the shipped DET and INS files keep `mixed_precision: False` |
| M15. A failed export check leaves the new model served | medium | **open** | |
| M16. The DET and INS export comparison compares padding | medium | **open** | |
| M17. The label mapping file is not part of the run | medium | fixed | on 2026-10-21: `label_dictionary.json` and checkpoint metadata |
| M18. CMP member frames ordered as text | medium | **open** | `os_utilities.py:237`; the frames of sequence inference requests too (`deployment_utilities.py:67`) |
| M20. `Iteration_k` holds k-1 steps | medium | fixed | on 2026-10-21, with [B44](#b44-evaluated-checkpoints-are-overwritten-or-left-unevaluated-at-stop-and-resume-boundaries) |
| M21. TensorRT FP16 never validated, engine cache | medium | **open** | |
| L1. Anchors centred on the cell corners | low | **open** | `anchors.py:76` |
| L2. Export crashes for some valid configurations | low | fixed | top-k clamped to the anchors (`region_proposer.py:103`, `inference_detector.py:135`), `dtype` removed, 3 channels enforced (`common_checks.py:273-277`) |
| L3. The threshold finder returns the lowest valid threshold | low | **open** | `threshold_finder.py:46-48`, which also sets the DET, INS, KPT and SEL class thresholds |
| L4. Balanced weights do not balance positives in multi-label | low | **open** | `classification_losses.py:37-40`, not documented either |
| L5. The box-loss normalisation contradicts its comment | low | **open** | see below |
| L7. Zero-area proposals and detections never removed | low | **open** | see below |
| L8. DET boxes only cleaned on augmented samples | low | **open** | DET and KPT (`coco_dataset.py:118-121`); INS takes its boxes from the masks |
| L9. Multi-label keeps all-unmapped samples as negatives | low | **open** | `classification_dataset.py:80-85`; the class docstring still says they are left out |
| L10. Nothing checks one label per frame | low | **open** | a count mismatch now raises in `collate_padded_sequences` (`sequence_dataset.py:138`), and `get_next_batch` drops that batch silently (read, not reproduced) |
| L11. M2M batches without labelled frame | low | **open** | see below |
| L13. The checkpoint registry is rewritten in place | low | fixed | on 2026-10-21: written to a temporary file, synced and replaced |
| L14. `run_benchmarking_loop` crashes at its first step | low | **open** | |
| L15. Loader workers forked from a multi-threaded process | low | **open** | `base_trainer.py:1353` (writers), 1361 (MLflow thread), 1799 (test workers) |
| L16. A host-GPU synchronisation per step | low | **open** | `metric_tracking.py:133, 165` |
| L17. Ragged last batch and averages over batches | low | **open** | see below |
| L18. Extreme inputs | low | **open** | `preprocessing.py:33` without `max(1, ...)` |
| L19. INS sends full-resolution masks through shared memory | low | **open** | `coco_dataset.py:384-389` |
| L20. GPU memory peaks in the original-resolution SEG evaluation | low | **open** | `segmentation_trainer.py:366-376` |
| L22. The README recipe squeezes INS masks at the border | low | **partly fixed** | the INS evaluation pastes into unclipped boxes; README line 1043 still restores the boxes without `clip=False` |

| Statement contradicting the code | Status | Note |
|---|---|---|
| README "Stopping And Resuming", `build_run_name` docstring: "Changing any other setting gives another run name" | **open** | README line 249, `run_name.py:641-646` (see C4) |
| Every YAML with `residual_nets`: "Shortcuts are added every two convolution layers" | **open** | see H4 |
| `RegionProposal.foreground_iou_threshold.max` (YAML) | **open** | still read only by the checker |
| `RegionProposal.strict_fallback_assignment` (YAML) | **open** | the YAML comments only; the function docstring is now correct |
| `detection_losses.py:65-76` "standard RPN normalization" | **open** | see L5 |
| `mean_average_precision.py:16` "COCO" | **partly fixed** | still "COCO-style mAP@[0.5:0.95]" at line 22 and "(COCO)" in `average_precision_report.py:53` |
| `region_proposer.py:34` aspect ratio = width / height | **open** | now line 126; `anchors.py` uses height / width |
| `backbone.py:42` "Missing keys default to disabled modules" | fixed | |
| `augmentations.py:530-531` brightness and contrast | fixed | |
| `preprocessing.py:124`, `deployment_utilities.py:3-4` "used by the inference clients" | **partly fixed** | the restore functions now serve the evaluations, but no inference client of the repository calls them |
| README CLS run name `..._HAVG.128_MC4` | **open** | see [B24](#b24-stale-run-name-and-checker-documentation) |
| `Backbone.input_channels` comment | fixed (`a10bb7a`) | |
| `sequence_classification_trainer.py:66-72` frame shuffle warning | **open** | still warns for every transformer, whatever `positional_embedding` and `causal` |
| `base_trainer.py:984-986` MLflow after the workers | **open** | now lines 1359-1360, true for the training and validation workers only (L15) |
| Loader: `dtype: "float16"` bare `KeyError` | fixed | the key is gone, and an old `dtype` key is now ignored silently |
| Loader: `int(float(...))` truncation | **open** | see [B21](#b21-the-loader-coerces-malformed-values-before-the-checks-and-some-malformed-sections-crash-the-checker) |
| Loader: unknown or misspelled keys ignored | **open** | only the augmentation types are checked for unknown names (`common_checks.py:189-195`) |

### What This Audit Adds

- **C2:** unchanged in `classification_losses.py:34-35`, `segmentation_losses.py:38-40` and `sequence_losses.py:29-31`.
  No shipped configuration smooths the labels, but `tests/trainers/test_segmentation_trainer.py:129` trains with
  "balanced" weights and a smoothing of 0.1.
- **C3:** found again by three reviewers.
  - The class count is now at `detection_trainer.py:454`, `instance_segmentation_trainer.py:305, 310`,
    `two_stage_detector.py:60` and `detection_checks.py:308-309`. `keypoint_trainer.py:305` already counts
    `len(self.class_names)`.
  - The `MC` token of `run_name.py:261-264, 625` is also wrong for KPT: a mapping merging 3 categories into one class
    is named `MC3`.
  - Two more failing mappings:
    - the gap `{"1": 1, "2": 3}` fails the first training step with `IndexError: Target 3 is out of bounds`;
    - `{"1": 1, "2": 0, "3": 2, "4": 0}` (two categories left out) crashes the first evaluation with `KeyError: 3`.
  - A category mapped to 0 also gives its name to the background class ([B30](#b30-a-category-mapped-to-0-to-exclude-it-names-the-background-class-and-is-counted-under-it-in-datajson)).
- **C4:**
  - `last_max_pooling` still has no token (`build_backbone_tokens`, `run_name.py:127-161`), although
    `model_transfer.py:49` lists it among the architecture keys. With `flatten` pooling the resume fails on a size
    mismatch (32,768 against 8,192 head inputs). With `average` or `max` pooling the checkpoint loads silently into the
    other architecture.
  - The anchor token (`run_name.py:224-229`) and the `PT` token (`107-124`) are unchanged.
  - Every shipped configuration sets `resume_training: true`, so the second launch always resumes the first run.
- **H4:** unchanged (`convolution_block.py:154, 329-331, 394-397` and `segmentation_decoder.py:88-96`).
  `keypoint_detection/configuration_synthetic_shapes.yaml` ends up with no shortcut at all.
- **H6:**
  - The four datasets still pass a `None` image to `preprocess_image`: `classification_dataset.py:136-137`,
    `coco_dataset.py:249, 267`, `segmentation_dataset.py:226-228` and `sequence_dataset.py:62-63`.
  - So does the inference input loader, `deployment_utilities.py:68`, a file H6 does not list.
  - That call fails in `preprocessing.py:64` or `222` without naming the file.
  - After 10 failed fetches in a row, the run ends through [B11](#b11-ten-failed-batch-fetches-in-a-row-leave-through-exit1-the-final-checkpoint-is-never-saved).
- **M5 (partly fixed):**
  - The RLE masks are now resized with `INTER_NEAREST_EXACT` (`coco_masks.py:324-327`).
  - `random_crop` and `random_zoom_out` (`augmentations.py:592-595, 643-647`) and `SegmentationDataset`
    (`segmentation_dataset.py:191, 238-239`) still use plain nearest. After a crop or zoom-out the masks move up to 0.86
    pixel, against 0.07 for the boxes, which the tests do not catch with their 2-pixel tolerance.
- **M7:** unchanged (`base_trainer.py:136-140, 283, 817, 896-902` and `checkpointing.py:68-116`). The error message of
  `check_resumed_data` recommends a fresh start with the same settings, which is exactly the launch that reuses the
  folder.
- **M13 (partly fixed):**
  - The RLE size is now compared with the image, but only at the original resolution (`coco_masks.py:316-327`).
  - At the network resolution, a mismatched RLE is still letterboxed with its own size.
  - So training learns misaligned masks, and only the test evaluation at the original resolution fails.
- **M17:** unchanged. The run folder keeps the YAML but not the mapping file:
  - `data.json` keeps the old class records, which `model_transfer.py:626-631` reads as the classes of the run;
  - `load_trained_model` (`model_loading.py:38-39`) re-reads today's mapping for the exports and evaluations of older
    runs.
- **L5:** the comment at `detection_losses.py:67-69` is unchanged.
- **L7:** [B35](#b35-the-eager-evaluation-ignores-the-valid-proposal-mask-so-padding-copies-can-become-duplicate-detections) is its root cause. The eager evaluation passes no valid proposal mask, so all the padding
  copies of a zero-area box survive the NMS.
- **L11:** both halves remain:
  - `sequence_losses.py:29-31` gives NaN for a batch without labelled frame, while the segmentation losses clamp their
    normaliser;
  - `sequence_metrics.py:28-40` counts a sequence whose frames are all unmapped as an exact match. One sequence with
    every frame wrong plus two unlabelled ones give a Sequence Exact Match of 0.667 with a frame accuracy of 0.0, and
    the "N Test Sequences" header counts them.
- **L17:** [B9](#b9-a-batch-normalized-classification-head-crashes-on-a-training-batch-of-a-single-vector) is a sharper case: a batch-normalized classification head fails on a last batch of one
  vector at any image size.
- **Statements contradicting the code:** the README CLS run name is still `..._HAVG.128_MC4`. [B24](#b24-stale-run-name-and-checker-documentation) adds the
  stale M2O and CMP examples.
- **Robustness gaps of the configuration loader:** the `int(float(...))` truncation remains. [B21](#b21-the-loader-coerces-malformed-values-before-the-checks-and-some-malformed-sections-crash-the-checker) adds the
  other coercions and the sections that crash the checker.

## Coding Standards

The rules of the global `CLAUDE.md` and of the project memory, checked on the 266 Python files of `vision_training/`,
`scripts/` and `tests/`, the 48 tracked configuration files and the README.

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

### Not Respected

S1 to S10 were corrected on 2026-10-20, in the worktree `audit-bug-standards` (not committed yet). The standards sweep
run again on the corrected files finds no line over 120 characters, no stretch of 20 lines without a comment, no
missing annotation or docstring and no abbreviated name outside the [Accepted Exceptions](#accepted-exceptions). S11
is still open.

#### S1. Missing type hints
- **Fixed.** `RegionProposal.forward` (`region_proposer.py`) is annotated, and `test_convolution_block` returns `None`.

#### S2. Abbreviated and single-letter names
- **Fixed.** `num_proposals` became `number_proposals`, `e` `error`, `k` and `v` `key` and `value`, the parameter `input`
  of `unsqueeze_tensor` `input_tensor` (its loop variable `_`), the `p` of `RandomHaze` `probability` (passed on to
  kornia as `p=probability`), the `min` and `max` of the threshold conditions of `target_assignment.py` `minimum` and
  `maximum`, and the box coordinates of `visualization.py` `x_minimum`, `y_minimum`, `x_maximum` and `y_maximum`.
- `img_tensor` in `tests/evaluation/test_evaluation_report.py` stays: it is the keyword of the TensorBoard
  `add_image` the fake writer replaces, which `evaluation_report.py` calls by name.

#### S3. Missing, generic or foreign docstrings
- **Fixed.** `ConvolutionBlock.__init__` holds the `Args:` section that was in the class docstring, and the generic or
  foreign docstrings (`TwoStageDetector.__init__`, `ChannelLayerNormalizer.__init__`, `InputNormalizer.__init__`,
  `get_random_color`, `print_tensor_shape`) give their place in the pipeline. The two tests without docstring have one,
  and the keypoint test of `test_coco_dataset.py` its `Args:` section.

#### S4. Repository functions called with positional arguments
- **Fixed.** `load_configuration` and every `print_tensor_shape` call take keyword arguments.

#### S5. One parameter or one argument per line
- **Fixed** wherever the grouped items fit in 120 characters. The 25 places left hold arguments or parameters of which
  no two fit on one line, and stay one per line (a 26th, `RandomPlasmaShadow`, was removed on 2026-10-22 with
  `view_augmentations.py`):
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

#### S6. Lines over 120 characters
- **Fixed.** The 45 lines are wrapped; no Python line of the repository is longer than 120 characters.

#### S7. Numbered inline comments
- **Fixed.** The comments of `DetectionHead.forward` lost their numbers.

#### S8. Long stretches of code without comment
- **Fixed.** The 32 stretches have comments on their blocks, comment lines only being added.

#### S9. Leftovers
- **Fixed.** The debug prints of `test_convolution_block.py` and `test_region_proposer.py` are removed,
  `README/random_image.png` is deleted, and the input viewer ends the program with `sys.exit()`. The bare `exit()` of
  `box_operations.py` is still bug [B37](#b37-get_box_dimensions_and_centers-ends-the-process-with-exit-and-status-0-on-a-wrong-box-shape).

#### S10. Duplicated code
- **Fixed** with the three bugs listing these copies:
  - [B34](#b34-sequenceclassifier-and-sequencelabeller-are-near-copies): `SequenceClassifier` and `SequenceLabeller`;
  - [B47](#b47-the-self-supervised-k-nn-report-re-implements-the-per-class-rows-table-and-scalars-of-class_metrics_reportpy): the k-NN report of the self-supervised task;
  - [B50](#b50-keypointtrainer-copies-four-detectiontrainer-methods-and-frameselectiontrainer-redefines-the-describe_split-it-inherits): the keypoint and frame selection trainers.

#### S11. README screenshots not made yet
- Still open: the README shows 20 images of `README/` (the task overview, the architectures, a sample of every task,
  the console output, TensorBoard, MLflow) that do not exist. They are the placeholders written on 2026-10-08
  (`50c8dcf`), each with a description of the screenshot to take. Until they are made, the page shows 20 broken
  images. They have to be taken from real runs and cannot be generated from the code.

### Accepted Exceptions
- **`tmp_path` and `tmp_path_factory`** (235 test parameters): pytest fixture names, which cannot be renamed.
- **`last_max_pooling`, `info_nce_loss` and `info_nce_outputs`:** "max pooling" and "InfoNCE" are the names of the
  methods.
- **`x` and `y` for point coordinates:** `create_synthetic_dataset.py:1904`, `visualization.py:947` and three tests.
- **`img_tensor`** (`tests/evaluation/test_evaluation_report.py:43`): the keyword of the TensorBoard `add_image`
  that the fake writer of the test replaces.
- **One argument or parameter per line where no two fit in 120 characters:** the 25 places listed in S5.
- **`max_pool_connections`** (`foundation_model_image_downloader.py:650`): the botocore setting it is passed to.
- **`previous_settings.get(setting_key, MISSING_SETTING)`** (`configuration_changes.py:73`): it compares a saved
  configuration, with a sentinel for the settings that configuration was saved without.
- **Broad `except` without `raise`:**
  - The 8 in `experiment_tracker.py:301` and `mlflow_tracker.py:314, 329, 347, 364, 376, 415, 476` are deliberate:
    tracking never stops a training, and each prints its error.
  - `os_utilities.py:199` falls back to an empty namespace in the debugging helper of `print_tensor_shape`.
  - The other two, `base_trainer.py:1222` and `1447`, are items H6 and M9 and bug [B11](#b11-ten-failed-batch-fetches-in-a-row-leave-through-exit1-the-final-checkpoint-is-never-saved).
- **The 4 tests without `assert`:** they assert through `check_nn_module_method` (`torch.allclose` against the `.pt`
  references) or through the `SystemExit` of the checker.
- **`imagenet_download.py:289`:** prints a count, not a path.

## Suggested Order

1. **Training safety and run identity, for all tasks:**
   - C3 (the class count of DET and INS), with [B30](#b30-a-category-mapped-to-0-to-exclude-it-names-the-background-class-and-is-counted-under-it-in-datajson);
   - C4 and M17 (run names and the label mapping);
   - M7 (a fresh start reusing a run folder);
   - then H6, M6, M8, M9, M20 and L13 of suggestions.md, with [B11](#b11-ten-failed-batch-fetches-in-a-row-leave-through-exit1-the-final-checkpoint-is-never-saved), [B12](#b12-ctrlc-during-a-periodic-save-is-ignored-by-the-trainer-but-kills-the-persistent-loader-workers-the-run-fails-5-iterations-later),
     [B44](#b44-evaluated-checkpoints-are-overwritten-or-left-unevaluated-at-stop-and-resume-boundaries) and [B9](#b9-a-batch-normalized-classification-head-crashes-on-a-training-batch-of-a-single-vector).

   A run name token added to every run would stop the running ImageNet runs from resuming in their folders, so the
   new tokens of C4 should only appear for non-default values.
2. **The new findings that give silently wrong results:**
   - [B2](#b2-a-sequence-split-subset-silently-goes-stale-when-its-source-split-is-regenerated) and [B3](#b3-duplicates-are-merged-only-on-identical-hashes-resized-copies-of-one-photo-stay-in-both-train-and-test) (stale test labels, test images leaking into training);
   - [B8](#b8-a-causal-sequence-labeller-or-frame-selector-with-sequence-normalization-uses-the-future-frames) (causal models seeing later frames);
   - [B10](#b10-a-keypoint-head-transfer-compares-only-the-class-names-so-keypoints-in-another-order-land-on-the-wrong-channels) (a keypoint head loaded onto the wrong channels);
   - [B17](#b17-an-exported-frame-selector-with-averaged-ema-weights-gets-the-confidence-thresholds-of-the-trained-weights) (thresholds taken from other weights);
   - [B19](#b19-standalone-evaluations-of-another-split-overwrite-and-mix-with-the-earlier-ones-in-the-same-output-folder), [B15](#b15-in-mlflow-testbest_metric-of-a-standalone-all-evaluation-shows-the-last-evaluated-value-not-the-best) and [B6](#b6-datajson-overstates-the-keypoint-objects-it-counts-the-objects-the-keypoint-loader-drops) (evaluations or counts mixed or misreported);
3. **The launches and runs that crash or hang:** [B1](#b1-a-missing-key-discards-every-problem-already-found-and-a-mode-typo-is-reported-as-a-missing-datalabel_dictionary), [B4](#b4-datajson-crashes-at-launch-on-a-self-supervised-split-mixing-labelled-and-unlabelled-images), [B5](#b5-datajson-crashes-at-launch-on-a-coco-split-without-annotations-which-the-loaders-accept),
   [B18](#b18-a-maximum-batch-size-of-1-crashes-the-onnx-export-and-at-2-the-dynamic-batch-is-never-checked) and [B16](#b16-one-artifact-write-failing-with-a-permanent-oserror-blocks-every-later-mlflow-write-and-leaves-the-run-running).
4. **The low bugs, area by area**, starting with the misleading reports: [B51](#b51-keypoint-matched-objects-and-mean-pixel-error-count-predictions-far-from-the-object) to [B55](#b55-a-detection-test-split-without-objects-reports-map-and-mar-00-instead-of-no-value),
   [B45](#b45-the-console-summary-treats-the-correlation-metrics-as-higher-is-better-a-rising-off-diagonal-correlation-prints-green-with-best) and [B46](#b46-iterations-per-epoch-and-the-sample-counts-logged-to-mlflow-come-from-the-manifest-before-the-label-mapping).
5. **The standards:** S1 to S10 are done (2026-10-20); S11 needs the screenshots of real runs.

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
- **Tests:** the full suite passes (1,415 tests in 5 minutes 26 seconds, run on 2026-10-20 in a clean worktree), a baseline for the fixes.
- **Not covered:**
  - the training speed and the GPU memory;
  - the real datasets, beyond the COCO 2017 validation keypoints and the local Open Images photos used by two
    findings;
  - the Triton server itself.
- **Scratch files:** the review scripts and their outputs are in the folder of this session's job
  (`~/.claude/jobs/6a755f62/tmp/audit/`), outside the repository, and are deleted with the job.
