# Detection NMS Design

## Table Of Contents
- [Purpose](#purpose)
- [What The Benchmark Can Show](#what-the-benchmark-can-show)
- [Sources Read](#sources-read)
- [The DIoU-NMS](#the-diou-nms)
- [Implementation Against The Sources](#implementation-against-the-sources)
- [Decisions](#decisions)
- [Implementation Plan](#implementation-plan)
- [Configuration And Run Names](#configuration-and-run-names)
- [Tests](#tests)
- [Cost](#cost)
- [Expected Effect](#expected-effect)
- [Stage 5b Protocol](#stage-5b-protocol)
- [Open Decisions](#open-decisions)
- [Left Out](#left-out)

## Purpose
Suppose stage 5 of `architecture_benchmarking/object_detection_experiment_plan.md` has kept its box loss: this design prepares stage 5b, the DIoU-NMS of Zheng et al. (AAAI 2020) in place of the greedy non-maximum suppression of the final detections (`suggestions.md` item 38). The final per-class NMS of `postprocess_detections` removes the lower scoring of two boxes of the same class and image whose IoU exceeds `Detector.nms_iou_threshold` (0.5). DIoU-NMS first subtracts from that IoU the squared distance between the centres over the squared diagonal of the smallest box enclosing both, so that two boxes whose centres are far apart, likely two objects, are both kept.

The user asked for it on 2026-11-08, after the CIoU box loss ("Add diounms as something to be tested too. After ciou box loss, start at least the detailled design plan of it's implementation"). This document is that design, every configuration keeping the greedy NMS. Shown it the same day, the user chose to have Part A of the [Implementation Plan](#implementation-plan) written at once, off in every configuration, and to add an occlusion test split to the stage ([Stage 5b Protocol](#stage-5b-protocol)). Part A was implemented the same day, `"iou"` staying in every configuration. Shown that a first sample of the occlusion split held almost no pair of objects of a class overlapping above 0.5, the user chose, still the same day, to have the generator place part of the objects next to an object of their class, a sample being counted first and the 5,000 images generated only after their go; that placement was written, and the split generated with the settings the user chose, the same day ([Implementation Plan](#implementation-plan), [Stage 5b Protocol](#stage-5b-protocol) step 2).

## What The Benchmark Can Show
Before reading the published gains, know what this benchmark lets DIoU-NMS change. The generator of the synthetic data places every object so that it overlaps each object already placed by an IoU of 0.3 at most (`MAXIMUM_OBJECT_INTERSECTION_OVER_UNION = 0.3`, `vision_training/data/synthetic/create_synthetic_dataset.py:222-225`). Measured on 2026-11-08 on test_10000 (10,000 images, 85,143 objects, 35 classes), over every pair of objects of an image:

| Pairs | Number | Largest IoU | IoU above 0.1 | Above 0.2 | Above 0.3 |
|---|---|---|---|---|---|
| Same class | 9,953 | 0.2991 | 367 | 145 | 0 |
| Any classes | 346,093 | 0.3000 | 12,287 | 4,389 | 0 |

The two suppressions differ only on the pairs whose IoU exceeds the threshold: the DIoU of a pair is never above its IoU, so a pair below the threshold is kept by both. At 0.5 no pair of objects of this data is concerned, at 0.45 neither, whatever the exponent of the distance term. The boxes DIoU-NMS keeps in addition overlap a better box of their class by more than 0.5, which two objects of this data never do, so they are mostly second detections of an object already found, false positives for the mean average precision. On test_10000 the stage can show that DIoU-NMS does no harm, and what it costs, but not the published gain on occluded objects, so it also evaluates an occlusion test split, generated with larger overlaps ([Stage 5b Protocol](#stage-5b-protocol)).

## Sources Read
To check every step of the implementation, these sources were read on 2026-11-08:
- **Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression", AAAI 2020** (arXiv 1911.08287): the distance term (Eq. 6), the section "Non-Maximum Suppression using DIoU" (Eq. 13), Tables 1 to 3 (rows of the CIoU loss with and without DIoU-NMS), the section "Discussion on DIoU-NMS" and Fig. 9.
- **Zheng et al., "Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation", IEEE TCYB** (arXiv 2005.03572): Section IV (Cluster-NMS, Algorithm 2), IV-B2 (Cluster-NMS with the distance term is DIoU-NMS; Eq. 14 with its exponent 0.6 for the score penalty variant), IV-C (the proof that the iterations give the result of the greedy NMS), Table V (YOLOv3, NMS methods) and Table VII.
- **The authors' code of their Faster R-CNN experiments**, Zzh-tju/DIoU-pytorch-detectron: `lib/utils/cython_nms.pyx:89-151` (`diounms`), `lib/utils/boxes.py:326-330`, `lib/core/test.py:732-794` (`box_results_with_nms_and_limit`, the choice of the NMS at 764-768), `lib/core/config.py:222, 227, 357-365`, `lib/modeling/generate_proposals.py:157-164` (the region proposals), `configs/baselines/e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml:31-34` and `README.md:54-58`.
- **The authors' darknet**, Zzh-tju/DIoU-darknet: `README.md:63-80` (the exponent for YOLOv3, SSD and Faster R-CNN).
- **The authors' Cluster-NMS code**, Zzh-tju/CIoU (YOLACT): `layers/functions/detection.py:346-379` (`cluster_diounms`) and `:32`, `layers/box_utils.py:82-126` (`diou`).
- **darknet**, the code of YOLOv4 (AlexeyAB/darknet): `src/box.c:195-231` (`box_diou`, `box_diounms`), `src/box.c:868-930` (`diounms_sort`), `src/parser.c:619-627` (`nms_kind`, `beta_nms`), `src/detector.c:1003` and `1076-1077` (the mAP evaluation) and `cfg/yolov4.cfg:978-979`.
- **YOLOv4** (Bochkovskiy et al., arXiv 2004.10934): Section 2.2 (the post-processing methods), Section 3.4 (DIoU-NMS among the bag of specials of the detector) and Table 5 (the bag of specials studied).
- **torchvision 0.29** (installed): `torchvision/ops/boxes.py:20-128` (`nms`, `batched_nms`) and `:472-515` (`distance_box_iou`).
- **mmcv**: `mmcv/ops/nms.py`, whose suppressions are `nms`, `soft_nms`, `batched_nms`, `nms_match`, `nms_rotated` and `nms_quadri`, without DIoU-NMS.
- **The video supplied by the user** (https://youtu.be/b148nt9P8J0), [1:03] to [1:04]: DIoU-NMS reduces the false suppressions of occluded objects, the IoU of a box with the best box being reduced by the distance penalty of the DIoU loss, so that a box overlapping above the threshold but whose centre is far is not taken for a duplicate.

## The DIoU-NMS
Say two boxes of a class overlap: with `M` the box of the highest score and `B_i` another one, Eq. 13 of the AAAI paper keeps the score `s_i` when `IoU(M, B_i) - R_DIoU(M, B_i) < ε` and sets it to 0 otherwise, where `ε` is the NMS threshold and `R_DIoU = ρ²(b, b_i) / c²` (Eq. 6): `ρ` the distance between the two centres, `c` the diagonal of the smallest box enclosing both. As in the greedy NMS, the boxes of a class are taken by decreasing score, every kept box suppressing the lower ones its criterion exceeds `ε` with.

- **The exponent β:** the authors' code raises the distance term to a power, `IoU - R_DIoU^β`. The term being at most 1, a β below 1 enlarges it and keeps more boxes: more recall, less precision (journal Fig. 8, for the score penalty variant).
- **Matrix form:** Cluster-NMS (journal Algorithm 2) takes the upper triangular matrix of the criterion between the boxes of a class sorted by score, and repeats `kept_j = max_i (kept_i × X_ij) ≤ ε` until the kept boxes stop changing; the result is that of the greedy NMS (Section IV-C), after a few iterations in practice and at most as many as there are boxes. With the criterion of DIoU it is DIoU-NMS (Section IV-B2), and it runs on the GPU without a loop over the boxes.
- **What the distance term can change:** along each axis, the centres of two overlapping boxes are at most `(C - I) / 2` apart, `C` the extent of the enclosing box and `I` that of the overlap, and `I / C` is at least the IoU, so `R_DIoU < (1 - IoU)² / 4`, approached by flat boxes. At the threshold 0.5, DIoU-NMS can therefore keep apart only pairs whose IoU is below 0.5505 (`3 - √6`) with β = 1, and 0.6314 with β = 0.6; above, it suppresses like the greedy NMS. A random search of 16 million pairs per range of side ratios (2026-11-08) found no term above the bound: at an IoU of 0.5 the largest was 0.039 for squares, 0.051 for side ratios within 1/2 and 2, and 0.061 within 1/100 and 100, the bound being 0.0625.

## Implementation Against The Sources
In order to choose each detail, compare the implementations read:

| Step | Paper (AAAI Eq. 13) | Authors' Faster R-CNN code | Authors' Cluster-NMS (YOLACT) | darknet (YOLOv4) | Proposed here |
|---|---|---|---|---|---|
| Criterion | `IoU - R_DIoU` | `IoU - R^β1` (`cython_nms.pyx:147`) | `IoU - D^0.9` (`box_utils.py:124-125`) | `IoU - (d / c)^β` (`box.c:225`), d and c squared | `IoU - R^β`, `Detector.nms_distance_exponent` |
| Exponent | 1; 0.6 in Eq. 14 (score penalty variant) | 1.0 by default, 0.9 best in their search (`config.py:365`, `README.md:56-58`); their darknet: 0.6 for YOLOv3, 1.0 for SSD and Faster R-CNN (`README.md:73-80`) | 0.9, written in the code | 0.6: `beta_nms` by default (`parser.c:619`), and `greedynms` calls `box_diou` whose 0.6 is written in (`box.c:206`, `916`) | 1.0, and 0.6 evaluated at stage 5b |
| Threshold | the one of the greedy NMS: 0.5 for Faster R-CNN, 0.45 for YOLOv3 and SSD | `TEST.NMS` 0.5 (`e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml:34`) | the `nms_thresh` of the detector | 0.45 for the mAP (`detector.c:1003`) | `Detector.nms_iou_threshold`, unchanged (0.5) |
| Suppression | criterion `≥ ε` | `ovr >= thresh` (`cython_nms.pyx:148`) | kept when the column maximum is `≤ ε` (`detection.py:360, 364`) | `> thresh` (`box.c:920`) | criterion `> ε`, as torchvision's `nms` |
| Coordinates | continuous | IoU with the +1 of the old Detectron (`cython_nms.pyx:96, 144-145`), enclosing diagonal without | continuous | continuous | continuous, as our boxes and IoU |
| Empty enclosing box | not stated | no guard | `+ 1e-7` on `c²` (`box_utils.py:124`) | IoU alone when `c = 0` (`box.c:221-223`) | `+ 1e-7` on `c²`, as the CIoU loss and torchvision (`boxes.py:503`) |
| Applied to | the detections | the final detections only (`test.py:764-768`); the region proposals keep the greedy NMS (`generate_proposals.py:157-164`) | the final detections | the final detections of every `[yolo]` layer (`detector.c:1076-1077`) | the final detections only (`postprocess_detections`) |
| Groups | per class | per class, after the score threshold 0.05 (`test.py:750-753`, `config.py:227`), 100 detections per image after (`config.py:222`) | per class, 200 best per class before, score above 0.01 after (`detection.py:347-349, 364`) | per class (`box.c:883-887`) | per image and class, after `Detector.inference.score_threshold` (0.05 in the evaluation, `detection_trainer.py:294`), `maximum_detections` after |
| Algorithm | greedy | greedy loop (cython) | Cluster-NMS, at most 200 iterations | greedy loop | Cluster-NMS repeated until it stops changing, exact |
| Default | not applicable | on (`config.py:362`) | off (`detection.py:32`) | `greedynms` in `yolov4.cfg:978-979`, that is DIoU-NMS with 0.6 | off (`"iou"` in every configuration) |

torchvision 0.29 has `distance_box_iou` (`boxes.py:472-515`, `IoU - ρ² / (c² + 1e-7)`, with leading dimensions but without exponent) and no DIoU-NMS; mmcv has none either.

## Decisions
Should you need to change one of these, the reasons were:
- **The final detections only:** as the authors' Faster R-CNN code, YOLACT and YOLOv4. The final NMS runs only in evaluation and inference (`two_stage_detector.py:730`), so DIoU-NMS changes no trained weight and stage 5b needs no training. On the region proposals (`suggestions.md` item 38, "possibly the proposals") it would change the training, with no published result for a region proposal network: the authors keep the greedy NMS there.
- **Exponent 1.0 by default, 0.6 evaluated too:** 1.0 is Eq. 13 and the default of the authors' Faster R-CNN code, "good enough" in their README; 0.6 is the one of YOLOv4. Each costs one evaluation per launch at stage 5b, not a training.
- **The threshold unchanged:** every source keeps the threshold of its greedy NMS, and Fig. 9 of the AAAI paper finds DIoU-NMS better than the greedy NMS at every threshold from 0.43 to 0.48, so the threshold is not tuned on the test split.
- **Suppression above the threshold, strictly:** as torchvision's `nms` and darknet, so that boxes with the same centre (no distance term) give exactly the result of `batched_nms`, which a test pins. The authors' cython code suppresses at the threshold too; the two differ only for a criterion exactly equal to it.
- **Continuous coordinates:** without the +1 of the old Detectron IoU, which our boxes and `get_intersection_over_union` do not use.
- **`1e-7` added to `c²`:** as the CIoU loss, whose distance term it shares (`ENCLOSING_DIAGONAL_EPSILON`, `box_operations.py:30`), torchvision and the authors' YOLACT code, so that two boxes reduced to the same point give a distance term of 0.
- **Exact Cluster-NMS:** the candidates of every (image, class) group padded into one `[G, M, M]` matrix, the iteration repeated until it stops changing (at most `M` times), in a single call for the whole batch like the `batched_nms` call it replaces. The prototype of 2026-11-08 gives the same boxes as a greedy loop written from the authors' code and darknet, on 200 random draws of clustered boxes with exponents 1.0 and 0.6 and thresholds 0.45, 0.5 and 0.7. The padded matrix holds `G × M²` criteria, `M` the largest group: a proposal reaches the probability 0.05 for 20 classes at most, so an image of 250 proposals gives 5,000 candidates at most and a group 250, far above the usual sizes, and the groups are processed by decreasing size in chunks of at most 2^22 pairs (`MAXIMUM_DISTANCE_NMS_PAIRS`), each padded to its largest group. The criterion is computed in float32 at least, the validation running the detector in bfloat16.
- **One distance term for the loss and the NMS:** the term of the CIoU loss broadcasts over its leading dimensions, so it moved to `get_normalized_centre_distance` (`box_operations.py:102-126`) and serves both, the pairs of the loss and the matrices of the NMS.
- **No export until kept:** `InferenceDetector` refuses a model whose `Detector.nms_criterion` is `"distance_iou"` until a stage keeps it, so that a deployed model never suppresses differently from its evaluation; the exportable form is written only then (Part B below).

## Implementation Plan
Whenever the implementation starts, these are the changes, file by file. Part A is what stage 5b needs:

1. `vision_training/models/detection/box_operations.py`: `get_normalized_centre_distance(boxes_1, boxes_2)`, the squared distance between the centres over the squared diagonal of the enclosing box plus `1e-7`, broadcasting over the leading dimensions; `get_distance_intersection_over_union(boxes_1, boxes_2, distance_exponent)`, the `[..., N, M]` matrix of `get_intersection_over_union` minus that term (of the boxes unsqueezed to `[..., N, 1, 4]` and `[..., 1, M, 4]`) to the power of the exponent.
2. `vision_training/losses/detection_losses.py`: `compute_complete_iou_loss` takes its distance term from `get_normalized_centre_distance`, its values unchanged (the tests against torchvision pin them).
3. `vision_training/models/detection/postprocessing.py`: `NMS_CRITERIA = ["iou", "distance_iou"]`; `batched_distance_nms(boxes, scores, group_indices, iou_threshold, distance_exponent)`, the exact Cluster-NMS of the [Decisions](#decisions) returning the kept indices by decreasing score like `batched_nms`; `postprocess_detections` gains `nms_criterion` and `nms_distance_exponent`, its eager path calling `batched_nms` or `batched_distance_nms`, its export path (`suppress_candidates_only` False) raising a `ValueError` for `"distance_iou"`.
4. `vision_training/models/detection/two_stage_detector.py`: `Detector.nms_criterion` and `Detector.nms_distance_exponent` read next to `nms_iou_threshold` (line 67) and passed to `postprocess_detections` (lines 727-734); the instance segmenter inherits them.
5. `vision_training/models/detection/inference_detector.py`: a `ValueError` in `__init__` for a model whose criterion is `"distance_iou"`, naming the setting, which `InferenceInstanceSegmenter` inherits.
6. `vision_training/configuration_processing/detection_checks.py`: `check_nms_criterion_configuration`, the criterion one of `NMS_CRITERIA` and the exponent a positive number, called next to the checks of `Detector.nms_iou_threshold` (line 314).
7. The six tracked DET and INS configuration files: `nms_criterion: "iou"` and `nms_distance_exponent: 1.0` after `nms_iou_threshold`, with their comments (below).
8. `scripts/configuration_processing/convert_fixed_settings.py`: both settings in `ADDED_SETTINGS` for the saved runs and launch files, with the greedy NMS, backups `_before_nms_criterion`.
9. `scripts/evaluation/evaluate_checkpoint.py`: `nms_criterion` and `nms_distance_exponent` among its settings (None keeps the ones of the run), written into `model_configuration["Detector"]` before the trainer is built, and an output folder of its own for a criterion other than the run's, so that the greedy and DIoU evaluations of a run never share `evaluations/<run>`.
10. `vision_training/data/synthetic/create_synthetic_dataset.py`: `MAXIMUM_OBJECT_INTERSECTION_OVER_UNION` (0.3) becomes the default of a `maximum_object_intersection_over_union` argument of `create_synthetic_dataset`, passed down to the placement of every object (line 1732), so that every existing dataset is drawn as before and the occlusion test split of the [Stage 5b Protocol](#stage-5b-protocol) can allow larger overlaps.
11. Documentation: `configuration_reference.md` (the two settings), `object_detection_workflow.md` (the final post-processing), `suggestions.md` item 38, the plan of stage 5b and this design; `tutorials/build_tutorials.sh --check-references`, since the storyboards quote detection files.

Part A was written on 2026-11-08 as listed (the line numbers of the list are those of the code before it), with these details:
- `postprocessing.py`: the suppression of the eager path goes through `suppress_overlapping_detections`, which calls `batched_nms` or `batched_distance_nms` and refuses an unknown criterion; the iterations of every chunk are in `keep_cluster_boxes`.
- `detection_checks.py`: the exponent must be a finite positive number, checked whatever the criterion, since every detection file holds it.
- `evaluate_checkpoint.py`: `apply_nms_criterion` checks the evaluated settings like the configuration file, warns and skips them for a task without `Detector`, and suffixes the output subfolder with the evaluated NMS when it is not the run's: `<run folder>_nms_distance_iou_1.0`, `_nms_distance_iou_0.6`, or `_nms_iou` for a run trained with DIoU-NMS evaluated with the greedy NMS.
- `create_synthetic_dataset.py`: the cap is checked within [0, 1] and refused when it differs from 0.3 for the tasks whose images `generate_image` does not draw (the sequence and counted shapes tasks).

After Part A, the same day, the same-class placement the occlusion split needs ([Stage 5b Protocol](#stage-5b-protocol) step 2) was written in `create_synthetic_dataset.py`:
- `create_synthetic_dataset` takes `same_class_neighbour_probability` (0 by default), checked within [0, 1] and refused for the tasks other than the object detection and the instance segmentation and with a cap below 0.5, and passes it through `generate_split` and the chunks of the workers to `generate_image`.
- In `generate_image`, every object after the first is placed, with that probability, next to the object placed just before it, unless that object was itself placed that way: the objects of a class overlapping above 0.5 come in pairs, each pair one decision of the NMS, instead of the chains of three or more objects of a first version. `place_same_class_neighbour` draws up to 5 shapes of the class of that object (`SAME_CLASS_NEIGHBOUR_DRAW_ATTEMPTS`) at 0.8 to 1 times its size (`SAME_CLASS_NEIGHBOUR_SIZE_RATIO_RANGE`) and gives each one to `place_shape` with the box of that object (`neighbour_box`).
- At each of its 50 attempts, `place_shape` then draws an IoU uniformly between 0.5 (`SAME_CLASS_NEIGHBOUR_MINIMUM_INTERSECTION_OVER_UNION`) and the cap and a direction uniformly, and finds by bisection the distance between the centres giving that IoU (`find_neighbour_centre_distance`, 40 steps), the IoU of two boxes never rising as one moves away from the centre of the other along a direction. The box clipped to the image must keep an IoU of 0.5 at least with that object and stay under the cap with every placed box, as for any object; an object for which no shape fits is placed as any other.
- With the probability 0, the generator makes no other random draw: 48 images (easy, hard, keypoint, and hard with the cap 0.6, 12 seeds each, hashed with the next draw of the generator) are the same bit for bit before and after the change.

Part B, written only if stage 5b keeps DIoU-NMS:

12. `postprocess_detections`, export path: every (image, class) group holds exactly the `P` pairs of its proposals, so the matrix has the static shape `[B, C - 1, P, P]` without padding, and the suppression is the exact scan of the `P` ranks (`P` steps of operations over all the groups at once), without control flow depending on the data; the TorchScript trace and the dynamo ONNX export (opset 18, `export_model.py:73-88, 138-175`) equal to the eager path, as `test_torchscript_round_trip` checks for the greedy NMS.
13. `InferenceDetector` accepts `"distance_iou"` again.

## Configuration And Run Names
For a run or an evaluation to use DIoU-NMS, set the criterion of the final NMS, after `nms_iou_threshold`:

```yaml
Detector:
  # IoU threshold of the final per-class non-maximum suppression.
  nms_iou_threshold: 0.5
  # Criterion of the final NMS: "iou", or "distance_iou" (the IoU minus the normalized distance of the centres,
  # https://arxiv.org/abs/1911.08287), compared with nms_iou_threshold.
  nms_criterion: "iou"
  # Exponent of the distance term of "distance_iou": 1.0 as in the paper, 0.6 in YOLOv4 (more boxes kept).
  nms_distance_exponent: 1.0
```

No run name token: the final NMS runs in evaluation and inference only and never changes the trained weights, like the score threshold of the evaluation, so a run keeps its folder whatever its criterion. Stage 5b changes the criterion in `evaluate_checkpoint.py` only, whose evaluations go to a folder of their own (`evaluations/<run folder>_nms_distance_iou_1.0`).

## Tests
So that the implementation stays checked against the sources, the tests are:
- `tests/models/detection/test_postprocessing.py`: `batched_distance_nms` equal to a greedy loop written from `diounms_sort` of darknet and `diounms` of the authors' cython code, on 40 random clustered draws of 10 to 69 boxes in 1 to 7 groups, with exponents 1.0 and 0.6 and thresholds 0.45, 0.5 and 0.7; boxes of a group sharing one centre (no distance term) giving exactly the result of `batched_nms`, two of them at an IoU of exactly 0.5 both kept; the case of the paper, a box of 10 pixels and a lower scoring box of its class overlapping it above 0.5, both kept by the two exponents at an IoU of 0.526 (centres 4.5 pixels apart), by the exponent 0.6 only at an IoU of 0.6 (DIoU 0.571 and 0.480), by neither when the centres coincide, where `batched_nms` keeps one; groups never suppressing each other, no candidate, a single candidate, equal scores; chunks of one group or of a few, and bfloat16 boxes, keeping the same boxes. The design first gave the case of the paper with an IoU of 0.6 kept apart by β = 1, which the bound of [The DIoU-NMS](#the-diou-nms) rules out.
- `test_box_operations.py`: the distance term on pairs computed by hand (two boxes reduced to one point give 0), and the Distance-IoU equal to torchvision's `distance_box_iou` for the exponent 1.0, its distance term to the power of 0.6 for 0.6; the CIoU loss tests unchanged, its distance term now taken from `box_operations.py`.
- `test_inference_detector.py` and `test_two_stage_detector.py`: `postprocess_detections` with `"distance_iou"` in the eager path, its export path and an unknown criterion refused, `InferenceDetector` and `InferenceInstanceSegmenter` refusing such a model, and the detector suppressing its evaluated detections with the criterion and the exponent of its configuration.
- The checks (unknown criterion, exponent 0 or infinite), the run name unchanged, the conversion adding both settings, the override of `evaluate_checkpoint.py` (suffixes, refusals, skip), and the cap of the generator (reaching every placement, a split with the cap 0 holding no overlapping pair, the invalid caps refused).
- Each of 30 mutations of the code covered (the strict suppression, the float32 criterion, the exponent, the iterations, the groups, the order, the chunks, the epsilon, the halved distance, the wiring of the detector, the refusals, the checks, the conversion, the evaluation override and the cap of the generator) makes a test fail, and the full suite passes (594 tests, 7 skipped).
- End to end on a real detector (2026-11-08, on the CPU): a copy of run_9 of stage 3, converted by `convert_fixed_settings.py` (5 lines added after `nms_iou_threshold`), its 15,000 checkpoint evaluated through the main block of `evaluate_checkpoint.py` on the first 16 images of test_10000 (151 objects), in about 12 seconds per NMS. The trained weights got the same metrics with the three NMS, and the averaged weights mAP@[0.5:0.95] 0.8398 with the greedy NMS against 0.8395 with DIoU-NMS at either exponent (mAP@0.5 0.9544 against 0.9542), written to `<run folder>_nms_distance_iou_1.0` and `_0.6`: a few duplicates kept, as [Expected Effect](#expected-effect) foresees for data without occlusions, on too few images to measure the effect.
- The same-class placement (`tests/data/synthetic/test_create_synthetic_dataset.py`): `find_neighbour_centre_distance` giving the target IoU (0.1 and 0.55) along 16 directions, and None for a target out of reach; on 8 hard images with the cap 0.6 and the probability 1, every attempt next to a neighbour aiming at the object placed last, itself placed as any other object, every object placed that way with its class, 0.8 to 1 times its size and an IoU with it from 0.5 to 0.6, no two objects above the cap, the pixel counts those of the shapes placed, and 37 of the 79 objects placed that way; a detection split of 8 images holding 8 pairs of a class above 0.5 at least with the probability 1 (the sum over 8 images never below 18 in 100,000 resamples of 400 measured images) and fewer without it (0.01 per image), none above the cap; the invalid probabilities refused (1.5, NaN, the keypoint detection, a cap of 0.3). 20 of 22 mutations of the placement make a test fail; the two others change no label: a target IoU drawn above the cap, which the cap check rejects before the next attempt, and the probability drawn at 0 too, which only shifts the random stream (the 48 hashed images cover it). The full suite passes (602 tests, 7 skipped).

## Cost
If you are looking for the time it adds, the prototype was timed on 2026-11-08 on the CPU (4 threads, lowest priority beside the stage 4 trainings) against torchvision's `batched_nms`, at the size of an evaluation batch (64 images × 35 classes = 2,240 groups):

| Candidates | Per group | torchvision `batched_nms` | Cluster DIoU-NMS |
|---|---|---|---|
| 13,440 | 6 | 118.4 ms | 7.8 ms |
| 67,200 | 30 | 195.1 ms | 68.3 ms |
| 224,000 | 100 | 408.1 ms | 460.5 ms |

The implementation, timed the same way on 2026-11-08 beside the same trainings, took 7.2 ms against 128.8 ms for `batched_nms` on 13,440 candidates, 62.1 ms against 489.3 ms on 67,200 and 437.6 ms against 416.3 ms on 224,000, the times of `batched_nms` varying with the load of the CPU. On the CPU `batched_nms` loops over the groups above 1,000 boxes (`boxes.py:86`), while the Cluster form runs every group at once. The evaluation runs on the GPU, where `batched_nms` is a single kernel up to 25,000 boxes: the GPU cost is still to be measured, the GPU being busy with stage 4 on 2026-11-08. Either way it is small beside the evaluation itself, about 8.3 minutes per checkpoint on test_10000.

## Expected Effect
In case the stage needs a reference, the published effects of DIoU-NMS against the greedy NMS at the same threshold, all with the CIoU loss, are:
- **Faster R-CNN, ResNet-50-FPN, MS COCO 2017 val** (AAAI Table 3): AP 38.65 to 38.71, AP75 41.96 to 42.07, AP small 21.32 to 21.37, medium 41.83 to 41.93, large 51.51 to 51.60.
- **YOLOv3, PASCAL VOC 2007 test** (AAAI Table 1): AP 49.21 to 49.32, AP75 54.28 to 54.74. **SSD** (Table 2): AP 51.44 to 51.63, AP75 56.16 to 56.34.
- **YOLOv3 pretrained, MS COCO 2017 val** (journal Table V, Cluster-NMS with the distance term): AP 43.2 to 43.3, AP75 46.5 to 47.1, AR100 62.7 to 63.6, for 16.5 ms against 14.5 ms with torchvision's NMS.
- **YOLOv4:** DIoU-NMS is in the bag of specials of its detector (Section 3.4) without a row of its own in Table 5, and its release runs it with the exponent 0.6.

The gains are at the stricter IoU thresholds and in the recall, from occluded objects. On this benchmark no two objects overlap above 0.3 ([What The Benchmark Can Show](#what-the-benchmark-can-show)), so no gain is expected: on clustered random draws the prototype keeps 1.4 % more boxes than the greedy NMS with the exponent 1.0 and 5.2 % more with 0.6, which here would mostly be duplicates, so the mAP should stay level or fall slightly, more with 0.6. On the occlusion split, objects of a class overlapping above 0.5 are there to keep apart, so the gain can appear, on detectors that never saw such overlaps in training, for the pairs below an IoU of 0.5505 with β = 1 or 0.6314 with β = 0.6, the only ones the distance term can keep apart ([The DIoU-NMS](#the-diou-nms)).

## Stage 5b Protocol
Once stage 5 has kept its pair (the CIoU pair or its reference), stage 5b runs without training:
1. Merge Part A of the [Implementation Plan](#implementation-plan), `"iou"` staying in every configuration.
2. The occlusion test split, generated on 2026-11-08: 5,000 hard shape images drawn like test_10000 (1024 pixels, 5 to 12 objects, the same classes and label mapping) but with `maximum_object_intersection_over_union` 0.6 and `same_class_neighbour_probability` 0.3, in a folder of their own (`create_synthetic_dataset(output_folder="/home/robert_kelevra/Data/object_detection_data/synthetic_hard_shapes_occlusion", image_size=1024, split_sizes={"test": 5000}, maximum_objects=12, minimum_frames=1, maximum_frames=1, number_workers=2, task="object_detection", hard_objects=True, maximum_object_intersection_over_union=0.6, same_class_neighbour_probability=0.3)`). It must hold enough pairs of a class overlapping above an IoU of 0.5, the pairs DIoU-NMS can keep apart, which random placements rarely give: a first sample of 2026-11-08, 300 hard 1024 pixel images with the cap 0.6 alone (`maximum_objects` 12, generated in 8.6 seconds), held 2,504 objects and 10,019 pairs, the largest IoU 0.5664, 13 pairs above 0.5 of any classes, and 307 pairs of a class, 4 of them above 0.3 and 1 above 0.5, about 17 pairs DIoU-NMS could act on in 5,000 images. Part of the objects are therefore placed next to an object of their class, at an IoU drawn between 0.5 and the cap (0.5505 and 0.6314 being the largest IoUs the two exponents can keep apart), as the [Implementation Plan](#implementation-plan) describes. With the probability 0.3, a sample of 300 images (generated in 8.1 seconds) held 2,451 objects, the largest IoU 0.5997, 525 pairs above 0.5 of any classes and 745 pairs of a class, 512 of them above 0.5 (1,024 objects, 42 % of them), of which DIoU-NMS at 0.5 keeps apart 100 with the exponent 1.0 and 403 with 0.6. Shown it, the user had the split generated with these settings (122 seconds, 439 MB, its `test` split and `label_mapping.json`): 42,151 objects, the largest IoU 0.6000, 169,733 pairs, 8,883 of them above 0.5 of any classes, and 13,281 pairs of a class, 8,675 of them above 0.5, of which DIoU-NMS at 0.5 keeps apart 1,728 with the exponent 1.0 and 7,079 with 0.6, the 35 classes labelled 1,131 to 1,283 times each, the other objects placed as in test_10000. A neighbour takes 0.8 to 1 times the size of its partner instead of the smaller size drawn for it, an IoU of 0.5 needing the smaller box to have half the area of the larger at least, so the objects of the split are a little larger than those of test_10000: box sizes (square root of the area) of geometric mean 102.3 against 94.4 pixels and 26.4 % against 24.1 % above 256 pixels, for 8.43 against 8.51 objects per image. Its label mapping is that of the benchmark, byte for byte.
3. Evaluate the 15,000 checkpoint of both launches of the kept pair with `evaluate_checkpoint.py` (its trained and averaged weights, `checkpoint_evaluation.py:132-139`), on test_10000 and on the occlusion split (its `test_data_folder` `.../synthetic_hard_shapes_occlusion/test`, with an `output_folder` of its own such as `evaluations_occlusion`, the subfolder of an evaluation naming the run and the NMS but not the split), with `nms_criterion: "distance_iou"` and the exponent 1.0, then 0.6, and once more with `"iou"`, so that every comparison is between two evaluations of the same weights by the same code: twelve evaluations, about 8.3 minutes each on test_10000 and half of it on the occlusion split, 75 minutes in all.
4. The weights being the same, the difference comes from the NMS alone, so the launch gap rule of the training stages does not apply: DIoU-NMS is kept if it raises the Test_EMA mAP@[0.5:0.95] of both launches on the occlusion split without lowering that of either launch on test_10000 by more than 0.001. The greedy NMS stays otherwise, and Part B is written only if DIoU-NMS is kept.
5. Read beside it mAP@0.5, @0.75 and @0.95, the recall, the sizes of the objects and the number of detections per image.

## Open Decisions
If one wishes to change the stage before it runs, these are the choices left to the user:
- **Evaluation only or a training pair:** the final NMS changes no weight, so an evaluation of the kept checkpoints measures it exactly; a training pair would add the noise of training without anything to measure.
- **Exponents:** 1.0 and 0.6, or 1.0 only.
- **The occlusion split** (added by the user on 2026-11-08 and generated the same day with the cap 0.6, 5,000 images and the probability 0.3 they chose, [Stage 5b Protocol](#stage-5b-protocol) step 2): the tolerance of 0.001 on test_10000 in the rule.
- **The export (Part B):** only if the stage keeps DIoU-NMS, or at once.

## Left Out
To keep the stage to one change, these stay out:
- **DIoU-NMS on the region proposals:** it changes the training, the authors keep the greedy NMS there, and no result is published for it.
- **Soft-NMS, Weighted-NMS and the score penalty and weighted variants of Cluster-NMS** (journal Sections IV-B1 and IV-B3): the 39.0 AP of Faster R-CNN in the journal (Table VII) is Cluster-NMS with weighted coordinates and the distance term, not DIoU-NMS.
- **Tuning the threshold** on the test split, which also selects the stages.
- **Matrix NMS and adaptive NMS**, of other papers.
- **The +1 of the old Detectron IoU.**
