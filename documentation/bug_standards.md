# Bugs And Standards

> Audit of the whole repository carried out on 2026-10-20 at commit `165339b`, to go through together. Every bug it
> found (B1 to B59) is fixed or was dropped by the user, the last ones on 2026-11-20, and is deleted from this file,
> which keeps the open items of the earlier audit, the coding standards and how the audit was made.
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

## Table Of Content
- [Items Of suggestions.md Still Open](#items-of-suggestionsmd-still-open)
- [Coding Standards](#coding-standards)
  - [Respected Everywhere](#respected-everywhere)
  - [Accepted Exceptions](#accepted-exceptions)
- [Suggested Order](#suggested-order)
- [How It Was Checked](#how-it-was-checked)

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
| Every YAML with `residual_nets`: "Residual shortcuts every two convolutions from block 2" | **open** | H4 dropped on 2026-11-05: the comment should say that the channel-changing blocks get no shortcut |
| `RegionProposal.foreground_iou_threshold.max` (YAML) | **open** | still read only by the checker |
| `mean_average_precision.py:16` "COCO" | **partly fixed** | still "COCO-style mAP@[0.5:0.95]" at line 22 and "(COCO)" in `average_precision_report.py:53` |
| `region_proposer.py:34` aspect ratio = width / height | **open** | now line 126; `anchors.py` uses height / width |
| `preprocessing.py:124`, `deployment_utilities.py:3-4` "used by the inference clients" | **partly fixed** | the restore functions now serve the evaluations, but no inference client of the repository calls them |
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
- **L7:** since 2026-11-20 the eager evaluation passes the valid proposal mask of the region proposal filter, so the
  padding copies of a zero-area box no longer survive the NMS; the empty proposals of the joint selection and the
  zero-area final detections remain.
- **L11:** both halves remain:
  - `sequence_losses.py:29-31` gives NaN for a batch without labelled frame, while the segmentation losses clamp their
    normaliser;
  - `sequence_metrics.py:28-40` counts a sequence whose frames are all unmapped as an exact match. One sequence with
    every frame wrong plus two unlabelled ones give a Sequence Exact Match of 0.667 with a frame accuracy of 0.0, and
    the "N Test Sequences" header counts them.
- **Robustness gaps of the configuration loader:** the `int(float(...))` truncation remains
  (`configuration_loader.py:126, 150`).

## Coding Standards

The rules of the global `CLAUDE.md` and of the project memory, checked on the 266 Python files of `vision_training/`,
`scripts/` and `tests/`, the 48 tracked configuration files and the README, then on 2026-11-20 on the 394 tracked Python
files (with `tutorials/` and `documentation/presentation/`) and the 120 tracked YAML files, by a sweep of every rule
followed by thirteen reviewers. The violations found were corrected (syntax trees compared before and after, so that no
formatting change altered the code) and are removed from this file; the sweep keeps 115 hits, all listed below as
accepted.

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
- **Imports after `matplotlib.use("Agg")`** in the figure scripts of `documentation/presentation/figures/`: the backend
  is chosen before `pyplot` is imported.
- **Library names kept:** `state_dict` and `load_state_dict` of `weight_averaging.py` (the PyTorch module interface),
  `obj` and `f` of `write_partial_file_then_fail` in `test_checkpointing.py` (it stands in for `torch.save`).
- **Dictionaries that are not configurations read with `.get`:** the Triton model configuration of
  `benchmark_serving_formats.py` and the planned file options of `make_run_snapshots.py`.
- **`predict_tile_probabilities(tiles)`** (`sliding_window.py:69`): a callable passed as an argument, not a repository
  function.
- **"px" drawn in an image** of `tutorials/figures/keypoint_detection/evaluation_figures.py`: a table label of a figure,
  not a console unit.
- **One argument per line where no two fit in 120 characters:** 80 calls, mostly in the tests, besides the places
  listed above.

## Suggested Order

Should you pick up the open items of suggestions.md, start with those that guard long runs: H6 and M13, then M5, L7
and L11.

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
  solarize and motion blur findings were run again. One finding was lowered from medium to low (B35,
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
