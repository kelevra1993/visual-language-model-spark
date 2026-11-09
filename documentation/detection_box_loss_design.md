# Detection Box Loss Design

## Table Of Contents
- [Purpose](#purpose)
- [Sources Read](#sources-read)
- [The CIoU Loss](#the-ciou-loss)
- [Implementation Against The Sources](#implementation-against-the-sources)
- [Decisions](#decisions)
- [Configuration And Run Names](#configuration-and-run-names)
- [Tests](#tests)
- [Cost](#cost)
- [Expected Effect](#expected-effect)
- [Stage 5 Pair](#stage-5-pair)
- [Left Out](#left-out)

## Purpose
This design is for whoever runs stage 5 of `architecture_benchmarking/object_detection_experiment_plan.md`, which compares the smooth L1 on the regression offsets with the Complete-IoU (CIoU) loss on the boxes they decode to (`suggestions.md` item 36). The detector finds the boxes but does not fit them tightly: the feature pyramid kept by stage 3 reaches 0.8079 Test_EMA mAP@[0.5:0.95] but 0.126 mAP@0.95 (run_9 and run_10).

The user asked on 2026-11-08 for the loss without turning it on ("Implement Ciou loss as well but it will not be activated for the next run"). Every configuration keeps `box_loss: "smooth_l1"` in both stages, so stage 4 (run_11 and run_12) trains as stages 1 to 3 did.

## Sources Read
To check every step of the implementation, these sources were read in full on 2026-11-08:
- **Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression", AAAI 2020** (arXiv 1911.08287): section "Complete IoU Loss" (Eq. 8 to 12) and Table 3 (Faster R-CNN on MS COCO 2017).
- **Zheng et al., "Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation", IEEE TCYB** (arXiv 2005.03572), the journal version: Section III-B (Eq. 7 to 10), Section V-B3 and Table VII.
- **The official code of their Faster R-CNN experiments**, Zzh-tju/DIoU-pytorch-detectron: `lib/utils/net.py` (`bbox_transform` 16-36, `compute_ciou` 80-128), `lib/modeling/fast_rcnn_heads.py:59-82`, `lib/modeling/FPN.py:422-458` (region proposal losses of the FPN models), `lib/core/config.py:420-437`, `configs/baselines/e2e_faster_rcnn_R-50-FPN_ciou_1x.yaml:5-6` and the `README.md`.
- **darknet**, the code of YOLOv4: `src/box.c:430-449`, the hand-written gradient of the CIoU loss.
- **torchvision 0.29** (installed): `torchvision/ops/ciou_loss.py` and `diou_loss.py`, taken from detectron2.
- **mmdetection**: `mmdet/models/losses/iou_loss.py:207-268` (`ciou_loss`) and `configs/faster_rcnn/faster-rcnn_r50_fpn_ciou_1x_coco.py`.
- **YOLOv4** (Bochkovskiy et al., arXiv 2004.10934): Section 2.2 (the box regression losses among the bag of freebies), Section 3.4 (CIoU loss in the detector of YOLOv4) and Table 4 of Section 4.3.
- **The video supplied by the user** (https://youtu.be/b148nt9P8J0), [0:40] to [0:47]: the MSE loss is not scale invariant and gives equal losses to boxes of different overlap; the IoU loss is flat for every box that does not overlap; GIoU adds the empty area of the enclosing box but stays flat when one box contains the other; DIoU adds the distance between the centres over the diagonal of the enclosing box; CIoU adds the aspect ratio term for boxes with the same centre and IoU but other shapes.

## The CIoU Loss
Suppose a predicted box B and its target B_gt, of widths w, w_gt and heights h, h_gt, with centres b and b_gt, and C the smallest box enclosing both, of diagonal c. The journal version defines (Eq. 7 to 10):

```text
L_CIoU = 1 - IoU + rho^2(b, b_gt) / c^2 + alpha * v
v      = 4 / pi^2 * (arctan(w_gt / h_gt) - arctan(w / h))^2
alpha  = 0                          if IoU < 0.5
         v / ((1 - IoU) + v)        if IoU >= 0.5
```

The AAAI version (Eq. 10 and 11) has the same loss without the condition on alpha. The journal adds it because "when two boxes are not well matched, the consistency of aspect ratio is less important" (Section III-B), the loss becoming the DIoU loss below an IoU of 0.5. The three terms are ratios of areas or lengths, so the loss does not change with the scale of the boxes, and it stays below 2: below an IoU of 0.5 it is 1 - IoU plus a distance term below 1, and from 0.5 the aspect ratio term alpha * v stays below 0.0071, the width over height ratios of two boxes overlapping that much differing by a factor of 2.25 at most (two boxes crossing at their centres; a random search of 20 million pairs found 0.0070).

`compute_complete_iou_loss` of `vision_training/losses/detection_losses.py` computes this loss for boxes given as `[x_min, y_min, x_max, y_max]`, and `compute_localisation_loss` chooses between the two box losses for both stages:
- **`smooth_l1`:** the smooth L1 (beta 1/9) between the predicted and target offsets, summed, the code of every run so far.
- **`complete_iou`:** the predicted and target offsets of the sampled positives are both decoded from their anchors (region proposal network) or proposals (detection heads, the offsets of the true class) by `apply_regression_predictions`, the target offsets giving back the matched ground truth boxes, then 1 - CIoU is summed over the positives.

Both callers divide the sum by the number of positives, as before.

## Implementation Against The Sources
Before choosing, compare every step with the five implementations read:

| Step | Paper (journal) | Official Faster R-CNN code | darknet (YOLOv4) | torchvision 0.29 | mmdetection | `detection_losses.py` |
|---|---|---|---|---|---|---|
| Boxes compared | the predicted and target boxes | both decoded from a unit box at the origin, divided by `BBOX_REG_WEIGHTS` (10, 10, 5, 5), the proposal frame (`net.py:16-36, 85-86`) | the boxes in image fractions | given boxes | the boxes decoded in image pixels (`reg_decoded_bbox=True`) | both decoded in image pixels from the anchor or proposal (weights 1, 1, 1, 1) |
| Union and enclosing diagonal | | + 1e-7 each (`net.py:113, 116`) | | + 1e-7 each | + eps each | + 1e-7 each |
| Aspect ratios | arctan(w / h) | no epsilon (`net.py:119`) | no epsilon | no epsilon | eps added to the heights | 1e-7 added to the heights |
| Alpha | 0 below IoU 0.5 (Eq. 10) | no condition, no epsilon (`net.py:120-122`) | no condition, + 0.000001 | no condition, + eps | `(ious > 0.5) * v / (1 - ious + v)` | 0 below IoU 0.5, + 1e-7 |
| Alpha in the gradient | | constant (`torch.no_grad`) | constant (hand-written gradient) | constant (`torch.no_grad`) | constant (`torch.no_grad`) | constant (`torch.no_grad`) |
| Gradient of v | 1 / (w^2 + h^2) dropped for coordinates in [0, 1] (AAAI, after Eq. 12) | autograd, exact | dropped (`box.c:435-436`) | autograd | autograd | autograd, through the log offsets |
| Clamp | | | | | CIoU within [-1, 1] | none, the loss staying below 2 |
| Stages | | detection heads only, the region proposal network keeping the smooth L1 (`FPN.py:455-458`, README "only on final bounding box refinement layer") | | | detection heads only (`roi_head`) | `RegionProposal.box_loss` and `Detector.box_loss` set apart |
| Normalization and weight | | summed over the positives, divided by all sampled RoIs (`net.py:126`), weight 12 (`ciou_1x.yaml:6`, README) | | | weight 12 | summed over the positives, divided by their number, weight of `Optimization.loss_weights` (1.0) |

The official code computes the loss in the frame of each proposal, which stretches its width and height to 1. The IoU does not change, but the distance and aspect ratio terms then compare the boxes relative to the proposal: a box of the proposal's shape counts as square. The paper defines the loss on the boxes themselves, as mmdetection, torchvision and detectron2 compute it.

`tests/losses/test_detection_losses.py` compares the values and gradients with those of torchvision on 2000 pairs: equal to the CIoU loss from an IoU of 0.5 and to the DIoU loss below it, within 1e-6 in float64.

## Decisions
Should you need to change one of these, the reasons were:
- **Image pixels, not the proposal frame:** the definition of the paper and the boxes of the evaluation, as in mmdetection, torchvision and detectron2.
- **Alpha 0 below an IoU of 0.5:** the later definition of the same authors (journal Eq. 10), also used by mmdetection; `>=` as in the paper, where mmdetection writes `>`, the two differing only at exactly 0.5.
- **Alpha a constant of the gradient:** as in every implementation read.
- **No change of the gradient of v:** the paper drops the factor 1 / (w^2 + h^2) because it explodes for coordinates in [0, 1]. Here the gradient reaches the offsets through w = w_anchor * exp(dw), so the derivative of arctan(w / h) with respect to dw is w h / (w^2 + h^2), at most 1/2 whatever the size of the box.
- **Epsilon 1e-7** on the union, on the squared diagonal, on the heights and in alpha: finite values and gradients for flat or empty boxes (tested), as in torchvision.
- **Target boxes decoded from the target offsets:** the matched ground truth boxes within the 1e-6 guards of `turn_boxes_to_transformation_targets`, as the official code does, so both stages share one function.
- **Float32 decoding:** the offsets of the heads, in bfloat16 under the mixed precision of the trainers, are cast to the float32 of the anchors and proposals before decoding.
- **Normalization and weight unchanged:** the sum over the positives divided by their number, as the smooth L1 here, so that a launch changes the loss only. The literature weighs it 12 with a normalization by all sampled RoIs, which comes to 12 x positives / sampled per positive, at most 6 for the half of positives of `Detector` (64 of 128 in `run-1.yaml`); a weight can be given by `Optimization.loss_weights` (see [Stage 5 Pair](#stage-5-pair)).
- **One setting per stage:** the published Faster R-CNN changes the detection heads only, while the plan of 2026-11-07 said both stages; both are possible.
- **The predicted box clamp:** `apply_regression_predictions` clamps dw and dh at log(1000 / 16) before the exponential, as torchvision and Detectron, so a prediction beyond it gets no gradient on that offset.

## Configuration And Run Names
For a launch to use the loss, set the `box_loss` of either section, after its `total_training_samples`:

```yaml
RegionProposal:
  # Box loss of the positive anchors: "smooth_l1" on the offsets, or "complete_iou" (1 - CIoU of the
  # decoded boxes, https://arxiv.org/abs/2005.03572).
  box_loss: "smooth_l1"
Detector:
  # Box loss of the positive proposals: "smooth_l1" on the offsets, or "complete_iou" (1 - CIoU of the
  # decoded boxes, https://arxiv.org/abs/2005.03572).
  box_loss: "smooth_l1"
```

- **Checks:** `check_box_loss_configuration` of `detection_checks.py` refuses any other value; a file without the setting is refused with the missing key.
- **Run names:** `complete_iou` adds `RCIOU` after the region proposal layers and `DCIOU` after the detector layers (`BOX_LOSS_CODES` of `run_name.py`); `smooth_l1` adds nothing, so every earlier run keeps its folder.
- **Saved runs:** `scripts/configuration_processing/convert_fixed_settings.py` writes `box_loss: "smooth_l1"` with its comment after the `total_training_samples` of both sections of the saved runs and launch files, which trained with it (`ADDED_SETTINGS`, backups `_before_box_loss`).
- **Loss names:** the curves keep `region_proposal_localisation_loss` and `detection_regression_loss`, whatever the box loss.

## Tests
So that the comparison with the sources stays checked, the tests are:
- `tests/losses/test_detection_losses.py`: the values and gradients against torchvision on both sides of an IoU of 0.5; zero for equal boxes, the same for scaled and translated boxes, below 2, finite for flat, empty and distant boxes; `compute_localisation_loss` (smooth L1 unchanged, the CIoU of the anchors for zero offsets, zero for the target offsets, lower after a gradient step, float32 from bfloat16 offsets, unknown losses refused); the region proposal loss decoding from the anchors of every image, and the detection loss reading the offsets of the true class from the proposals without their batch index.
- `tests/models/detection/test_two_stage_detector.py::test_complete_iou_box_loss_trains_both_detection_stages`: with the same seed the classification losses do not change while the box losses of both stages do, and their gradients reach the box regression layers of both stages.
- The configuration checker, the run names and the conversion of the saved runs have their own cases.

Each of 13 mutations of the code (no condition on alpha, alpha in the gradient, the centre distance without its quarter, another beta, either stage ignoring its setting, the batch index read as a coordinate, another normalization, no check, no run name token, the conversion writing elsewhere or for another task) makes at least one of them fail.

## Cost
To weigh the loss against the training time, it was timed on the CPU for 2048 positives, forward and backward: 0.63 ms against 0.018 ms for the smooth L1, a few elementwise operations on the sampled positives only. A training iteration of the benchmark takes about 3.5 seconds (15,000 iterations in 14.45 hours in stage 3), so the loss costs nothing measurable; it adds no parameter, and the exported model does not change.

## Expected Effect
In case the stage needs a reference, the published effects are:
- **Faster R-CNN, ResNet-50-FPN, MS COCO 2017 val** (AAAI Table 3, journal Table VII), against the IoU loss: AP 37.93 to 38.65, AP75 40.79 to 41.96, AP small 21.58 to 21.32, AP medium 40.82 to 41.83, AP large 50.14 to 51.51. The journal explains the moderate gain by the proposals already overlapping their objects, where IoU, GIoU and DIoU all regress well, CIoU fitting the boxes "more perfectly". The baseline is the IoU loss, not the smooth L1.
- **YOLOv4** (Table 4, CSPResNeXt50-PANet-SPP at 512 x 512): with the grid sensitivity fix, MSE 37.7, GIoU 39.4, DIoU 39.1, CIoU 39.6 AP, AP50 falling from 59.9 to 59.2 while AP75 rises from 40.5 to 42.6; with the other freebies, MSE 40.3, GIoU 42.4, CIoU 42.4.

The gains come at the high IoU thresholds (AP75) and on the large objects, where this detector loses most (0.126 mAP@0.95). A small loss on the small objects is possible, as in Table 3.

## Stage 5 Pair
Before stage 5 is launched, the reference is the feature pyramid (run_9 and run_10), stage 4 having tied on 2026-11-09 (path aggregation 0.8038 against 0.8079), and the new pair takes the same settings with the CIoU loss. Of the two choices of the stage, the user made one on 2026-11-09 and left the other open until they had seen the comparison of stage 4:
- **Stages, open:** the detection heads only (`Detector.box_loss`, the published Faster R-CNN, run name `DCIOU`), or both stages (the plan of 2026-11-07, `RCIOU` and `DCIOU`).
- **Weight, about 6:** a `detection_regression_loss` weight at the published balance (12 x positives / sampled, about 6 for 64 positives of 128), chosen by the user against 1.0, the weight of the smooth L1 runs, which would measure the loss alone.

The launcher writes the setting into the files built from `run-1.yaml` (`run_benchmark_stages.py`, `SETTING_KEYS`), whose `box_loss` lines were added after the merge.

## Left Out
To keep the stage to one change, these stay out:
- **GIoU and DIoU options:** torchvision has both; DIoU is what the CIoU loss computes below an IoU of 0.5.
- **The IoU loss of the official baseline** and a change of the weights of `BOX_REGRESSION_WEIGHTS`, kept at (1, 1, 1, 1).
- **DIoU non-maximum suppression and Cluster-NMS** (`suggestions.md` item 38), which change the inference, not the training: designed for stage 5b in `detection_nms_design.md`.
- **A box loss for the mask heads**, which have none.
