# Detection Proposal Selection Design

## Table Of Contents
- [Purpose](#purpose)
- [Sources Read](#sources-read)
- [The Current Selection](#the-current-selection)
- [Selection In The Sources](#selection-in-the-sources)
- [Measurements On run_9](#measurements-on-run_9)
- [Decisions](#decisions)
- [Configuration And Run Names](#configuration-and-run-names)
- [Export](#export)
- [Tests](#tests)
- [Stage 7 Protocol](#stage-7-protocol)
- [Preview And Export Checks](#preview-and-export-checks)
- [Left Out](#left-out)

## Purpose
Suppose stage 6 of `architecture_benchmarking/object_detection_experiment_plan.md` has kept batch normalization in the detection heads: this design prepares stage 7, the selection of the region proposals per pyramid level (audit M3 and item 3 of `suggestions.md`) together with the removal of the proposals clamped to an empty box (audit L7). The proposal filter keeps the proposals the detection heads classify. Today it ranks the anchors of all the levels together, so the finest level, which holds most of the anchors, can take most of the slots before and after the non-maximum suppression.

The user asked on 2026-11-11 for the plan of stage 7 and its implementation ("ok write the plan for the stage 7 and then you can implement it."), as a setting the benchmark can compare, every configuration keeping the current selection.

## Sources Read
To compare every step with the references, these sources were read on 2026-11-11:
- **torchvision 0.29** (installed in `.venv`): `torchvision/models/detection/rpn.py:181` (`self.min_size = 1e-3`), `:231-240` (`_get_top_n_idx`, the top-k of every level), `:242-295` (`filter_proposals`); `torchvision/ops/boxes.py:86` (the switch of `batched_nms` to a loop over the groups), `:149-150` (`remove_small_boxes`); `torchvision/models/detection/roi_heads.py:723-724` (empty final detections removed); `torchvision/models/detection/faster_rcnn.py:179-182` (the default counts).
- **Detectron2** (facebookresearch/detectron2, main, downloaded on 2026-11-11): `detectron2/modeling/proposal_generator/proposal_utils.py:22-130` (`find_top_rpn_proposals`), `detectron2/structures/boxes.py:212` (`Boxes.nonempty`), `detectron2/config/defaults.py:167, 234-235, 242-245` and `configs/Base-RCNN-FPN.yaml:14-20`.
- **This repository**: `select_best_proposals`, `suppress_overlapping_proposals` and `RegionProposalFilter` of `vision_training/models/detection/region_proposer.py`, `InferenceDetector.filter_proposals` of `inference_detector.py`, `aggregate_proposals_and_anchors` and `forward_detection_stages` of `two_stage_detector.py`, `postprocess_detections` of `postprocessing.py`, the export of `vision_training/deployment/export_model.py`.

## The Current Selection
Before changing the filter, know what it does. `select_best_proposals` (`region_proposer.py`) serves the `RegionProposalFilter` of the training and evaluation and the `InferenceDetector` of the exports, for the whole batch at once:
1. the sigmoid of the objectness of all the anchors of every level, concatenated in level order, in the precision of the boxes;
2. one `topk(pre_nms_proposals)` per image over all the anchors, the boxes clamped to the image;
3. one non-maximum suppression per image at `RegionProposal.nms_iou_threshold` over these boxes, whatever their level (`suppress_overlapping_proposals`: a torchvision `nms` call per image in the eager model, a single `batched_nms` with the image as the group in the exports);
4. one `topk(min(post_nms_proposals, pre_nms_proposals))` of the survivors per image, the suppressed proposals scored -1 so that they only fill the slots left, marked invalid.

The training filter then fills the invalid slots with copies of random survivors of the image (`_pad_proposals_to_target_size`), the exported detector masks them. No proposal is removed for its size: a proposal regressed outside the image and clamped to a zero-width box stays a candidate (`suggestions.md` L7).

## Selection In The Sources
In order to choose each detail, compare the selections read:

| Step | torchvision `filter_proposals` | Detectron2 `find_top_rpn_proposals` | Here, `"joint"` (current) | Here, `"per_level"` (proposed) |
|---|---|---|---|---|
| Top-k before the NMS | per level: `objectness.split(num_anchors_per_level, 1)`, then `ob.topk(min(pre_nms_top_n, anchors of the level))` (`rpn.py:234-238`) | per level: `logits_i.topk(min(Hi_Wi_A, pre_nms_topk))` (`proposal_utils.py:72-79`); "this number is per feature map" (`:43-45`) | over all the anchors | per level, `min(pre_nms_proposals, anchors of the level)` |
| Clipping | `clip_boxes_to_image` (`rpn.py:277`) | `boxes.clip(image_size)` (`:114`) | clamp to the image | the same |
| Empty boxes | removed when a side is below `min_size = 1e-3` (`rpn.py:181, 280`; `boxes.py:149-150`, `(ws >= min_size) & (hs >= min_size)`) | removed when a side is not above `MIN_SIZE` (`boxes.nonempty`, `widths > threshold`, `boxes.py:212`), 0 by default (`defaults.py:167`) | kept | removed below 1e-3 pixel, as torchvision |
| Non-finite values | not handled | removed, refused in training (`:105-113`) | not handled | not handled |
| NMS | `batched_nms(boxes, scores, lvl, nms_thresh)`, "independently done per level" (`rpn.py:288-289`) | `batched_nms(boxes.tensor, scores_per_img, lvl, nms_thresh)` (`:121`) | one per image over all the levels | one per image and level |
| Top-k after the NMS | `keep[: post_nms_top_n]` over all the levels (`rpn.py:292`) | `keep[:post_nms_topk]`; "this number is total, over all feature maps" (`:46-48`, `:129`) | `topk` of the survivors over all the levels | the same, over the survivors of every level |
| Counts | pre 2000 / 1000 per level, post 2000 / 1000 (`faster_rcnn.py:179-182`) | pre 2000 / 1000 per level, post 1000 / 1000 (`Base-RCNN-FPN.yaml:14-20`); 12000 / 6000 and 2000 / 1000 without FPN (`defaults.py:234-243`) | pre 1000 / 500, post 500 / 250 in the benchmark | the same numbers, pre now per level |

Both references take the top-k of every level and run the NMS of every level on its own, then keep the best survivors of all the levels together. The final detections of torchvision also drop the boxes with a side below 1e-2 pixel before their NMS (`roi_heads.py:723-724`).

## Measurements On run_9
So that decisions c and e rest on numbers, the 15,000 averaged weights of run_9 (feature pyramid, the reference of stage 7) were run on the CPU on the first 300 images of test_10000 (2,530 objects) on 2026-11-11, with the current selection and with a prototype of the per-level one, both on the same proposals of every batch, with the inference counts of the run (500 / 250) and its training counts (1000 / 500). The levels of the run have 24,576, 6,144, 1,536 and 1,536 anchors (levels 4 to 7, levels 6 and 7 both 16 x 16 cells).

| Counts | Selection | Proposals reaching the NMS | Kept, level 4 | Level 5 | Level 6 | Level 7 | Kept, valid |
|---|---|---|---|---|---|---|---|
| Inference, 500 / 250 | joint | 500 | 211.0 | 3.1 | 3.2 | 5.2 | 222.5 |
| Inference, 500 / 250 | per level | 2,000 | 235.3 | 4.1 | 4.7 | 5.9 | 250.0 |
| Training, 1000 / 500 | joint | 1,000 | 477.5 | 5.1 | 3.7 | 6.2 | 492.5 |
| Training, 1000 / 500 | per level | 4,000 | 481.9 | 6.1 | 5.2 | 6.8 | 500.0 |

Per image. The test evaluation of the same checkpoint at stage 5b (all 10,000 images) gives the same split: 224.8 proposals per image, 213.2 of level 4 and 3.3, 3.2 and 5.1 of levels 5 to 7.

- **Per-level counts (decision e):** level 4 keeps about 94 % of the slots with both selections. The post-NMS top-k ranks the survivors of all the levels by score, so per-level selection does not give the upper levels a share; it lets their candidates reach the suppression and fills every slot: the joint NMS leaves 27.5 of the 250 inference slots without survivor (7.5 of 500 in training), the per-level one none, levels 5 to 7 gaining about one proposal each. The starvation of `object_detection_workflow.md` Step 3.13 (`{5: 484, 6: 6, 7: 10}`) is that of the synthetic configuration at 5000 iterations, three levels of 32 to 512 pixel anchors.
- **Proposal recall** (best valid proposal of every object): small objects (539) 0.9518 and 0.9221 at IoU 0.5 and 0.7 with the joint selection, 0.9536 and 0.9239 per level; medium (723) 0.9945 and 0.9862, then 0.9945 and 0.9876; large (1,268) 1.0000 and 0.9921, then 1.0000 and 0.9929.
- **Final detections** (evaluation score threshold 0.05): 5,067 with the joint selection, 5,458 per level.
- **Empty boxes (decision c):** no kept proposal has a side below 1e-3 pixel in any of the four rows, and none of the 500 or 1,000 proposals reaching the joint NMS. Among the 1,200,000 per-level candidates of the training counts, 41 have such a side, all suppressed or ranked out. No final detection has a side below 1e-2 pixel, with either selection.

## Decisions
To keep the stage a clean comparison of one setting, these choices are made:
- **a. The switch:** `RegionProposal.proposal_selection`, `"joint"` (the current selection) or `"per_level"`, `"joint"` in every configuration, the instance segmentation included (its mask heads read the same proposals, `"per_level"` working there too, never measured). `"per_level"` adds the run name token `RPL` after the region proposal heads, `"joint"` none, so that the existing runs keep their names. `check_region_proposal_configuration` refuses another value. The converter writes `"joint"` into the saved runs and launch files, which were trained with it.
- **b. The counts:** with `"per_level"`, `pre_nms_proposals` is the number of every level (at most its anchors) and `post_nms_proposals` the number of every image over all the levels, as in Detectron2 and torchvision. The benchmark keeps 1000 / 500 and 500 / 250, so 4,000 proposals reach the training NMS of an image and 2,000 the inference one, against 1,000 and 500. The check that `post_nms_proposals` does not exceed `pre_nms_proposals` stays for both values: stricter than needed per level, it holds for every configuration.
- **c. Empty proposals:** removed by `"per_level"` only, below 1e-3 pixel on either side after the clamping (torchvision's `min_size`), so that `"joint"` stays bit-identical to the current code and runs. Removing them before the NMS, as torchvision does, would give a data-dependent number of boxes; instead they enter the NMS with a score of -1, below every probability, so that they come last and suppress no other proposal, then are marked out of the survivors, which keeps exactly the proposals torchvision keeps. (Marking them out after an NMS on their own scores would not: a box of width w suppresses at an IoU above 0.7 only boxes narrower than w / 0.7, which reaches 1.43e-3 pixel, above the limit. The prototype of the measurements did this; it changes no count above.) The final detections get no filter: none of the 5,067 and 5,458 detections measured is empty, and a filter there would change the evaluation of `"joint"`.
- **d. One selection, one form:** `select_best_proposals` takes the anchor count of every level and the selection, and ranks and suppresses every group of anchors on its own: one group of all the anchors for `"joint"`, one group per level for `"per_level"`. The group sizes are plain numbers taken from the anchors at construction, so that the loop has a fixed number of steps and fixed slice sizes in the TorchScript trace and the dynamo ONNX export, and every group calls `suppress_overlapping_proposals` as today. `find_nonempty_proposals` marks the proposals kept after the clamping, all of them for `"joint"`, which runs the same operations on the same values as before (one group, every proposal kept), as the behaviour baseline of `tests/models/detection/test_detector_baseline.py` checks.
- **Cost:** timed on the GPU on 2026-11-11 with the real proposals of a batch of 64 test images (run_9, bfloat16 autocast as in the training), median of 20 runs: 17.9 ms joint and 55.7 ms per level with the training counts, 12.8 and 46.2 ms with the inference ones, the eager path running one `nms` call per image and level (256 for the batch instead of 64). Against 3.35 seconds per iteration of a launch training beside another (19.1 images per second at batch 64, run_9), the training gains about 38 ms, 1.1 %.

## Configuration And Run Names
Each time a detection or instance segmentation file is written, the setting sits in the `RegionProposal` section, after its normalization, before the counts it changes the meaning of:

```yaml
RegionProposal:
  normalization:
    # "batch", "layer" (over channels) or "none"
    feature_map_normalization: "batch"
  # "joint" (one top-k and NMS over all the levels) or "per_level" (top-k and NMS of every level, then the best
  # over all the levels, as torchvision, empty proposals dropped), which makes pre_nms_proposals a count per level.
  proposal_selection: "joint"
  training:
    # Top raw proposals kept per image (per level with "per_level") before non-maximum suppression in training.
    pre_nms_proposals: 1000
```

| Value | Token | Example |
|---|---|---|
| `"joint"` | none | `..._RBN-R2x32_DSE-...` |
| `"per_level"` | `RPL` | `..._RBN-R2x32-RPL_DSE-...` |

`convert_fixed_settings.py` gets `RegionProposal.proposal_selection` in `ADDED_SETTINGS`, written after the `RegionProposal.normalization` block with its comment line, value `"joint"`, its backups suffixed `_before_proposal_selection`. An unconverted run stops at load with the missing key, as for the earlier added settings.

## Export
Should the per-level selection be exported, the form matters: `batched_nms` runs `_batched_nms_vanilla`, a loop over the distinct groups (`torch.unique`, data dependent), when `boxes.numel() > (4000 if boxes.device.type == "cpu" else 100_000)` outside a TorchScript trace (`torchvision/ops/boxes.py:86`). A single call over all the levels of a batch, the group being the image and the level, would give it 8,000 coordinates per image with the benchmark counts, above the GPU limit from 13 images on (the exports of `export_model.py` run on the GPU with a dynamic batch up to 16), and above the CPU one from the first image. The loop of decision d gives every call the boxes of one level of every image, at most `pre_nms_proposals` per image (500 at inference), the size the single call of `"joint"` already has, so `"per_level"` reaches no limit `"joint"` does not.

## Tests
Before trusting the switch, these tests have to fail without it:
- `select_best_proposals` with `"per_level"` keeps the best proposals of every level when one level would take every slot jointly, removes an empty proposal that `"joint"` keeps, and suppresses only within a level; with `"joint"` it keeps its current result.
- The eager suppression and the export one keep the same per-level proposals on random batches.
- `RegionProposalFilter` reads the setting and refuses proposals whose number differs from the anchors of its levels.
- The detector and `InferenceDetector` give the same detections with `"per_level"`, and the TorchScript and ONNX exports of a per-level detector reproduce the eager one.
- The run name gets `RPL` for `"per_level"` only, the check refuses another value, and the converter adds `"joint"` after the normalization of the detection and instance segmentation files.
- The behaviour baseline of the detector keeps its recorded values with `"joint"`.

## Stage 7 Protocol
Once the code is merged, stage 7 trains one pair against the reference kept so far, run_9 and run_10 (feature pyramid, smooth L1 box loss, batch normalization in the detection heads):
1. **Launches:** run_21 and run_22, the settings of run_9 and run_10 with `RegionProposal.proposal_selection: "per_level"`, 15,000 iterations, two side by side, compared on the `Test_EMA` mAP@[0.5:0.95] at 15,000 with the tie rule of the plan. The driver needs `RegionProposal.proposal_selection` among its setting keys and swept settings.
2. **Preview, not the verdict:** the 15,000 checkpoint of run_9 evaluated with `"per_level"` at inference only, on test_10000 with its greedy final NMS, so that the effect of the selection without retraining is known before the launches: +0.0018, see [Preview And Export Checks](#preview-and-export-checks).
3. **Cost:** about 14.5 hours for the pair, plus about 1 %.

## Preview And Export Checks
Before the pair trains, these runs of 2026-11-11 show what the selection changes without retraining and that the exports keep it.

**Preview, not the stage result.** The 15,000 checkpoint of run_9, trained with `"joint"`, was evaluated on test_10000 with `"per_level"` at inference only and the greedy final NMS of the run (`run_9/evaluations/<run name>_proposal_selection_per_level`), against its greedy evaluation of stage 5b (`run_9/evaluations/<run name>`). As a control, the same evaluation with `"joint"` gave 0.80984 with the code of this stage and 0.80985 with the code of the commit before it (0f2f6ed), against 0.80984 at stage 5b: evaluations on the GPU differ from run to run by about 1e-5 in mAP and 0.0007 proposals per image whatever the code, far below the gain, and the behaviour baseline test shows `"joint"` unchanged exactly on the CPU.

| Selection at inference | Test_EMA mAP@[0.5:0.95] | mAP@0.5 / 0.75 / 0.95 | Small / medium / large | Proposal recall at 0.5 / 0.7 | Proposals per image, levels 4 / 5 / 6 / 7 | Test (trained) |
|---|---|---|---|---|---|---|
| joint | 0.8098 | 0.9545 / 0.9257 / 0.1307 | 0.7110 / 0.8359 / 0.8317 | 0.9881 / 0.9722 | 213.2 / 3.3 / 3.2 / 5.1 (224.8) | 0.7355 |
| per level | 0.8116 (+0.0018) | 0.9560 / 0.9285 / 0.1259 | 0.7123 / 0.8362 / 0.8355 | 0.9894 / 0.9740 | 235.2 / 4.3 / 4.6 / 5.8 (250.0) | 0.7360 (+0.0005) |

- Every one of the 250 slots holds a survivor, against 224.8 jointly, as the 300 images of the measurements showed.
- The share of the objects a proposal of a level covers at 0.5 rises in every level: 0.300 to 0.341 for level 4, 0.271 to 0.383 for level 5, 0.271 to 0.405 for level 6, 0.226 to 0.261 for level 7. A proposal of one level no longer suppresses that of another level on the same object.
- The gain, 0.0018, is below the gap between run_9 and run_10 (0.0038); the large objects gain the most (+0.0038), mAP@0.95 loses 0.0048. Whether training with the selection changes more is what run_21 and run_22 measure.

**Export.** `export_experiment` exported the same averaged weights with `"per_level"` to TorchScript and ONNX (GPU, dynamic batch up to 16), each export matching the eager model on its random batches. On 4 test_10000 images with the final DIoU-NMS 0.6 of the object detection configurations, the evaluation path, `InferenceDetector`, the TorchScript file and ONNX Runtime gave the same detections (12, 11, 11 and 6 per image, 0 differences at 1e-2 pixel and 1e-3 in score), while `"joint"` gave different ones on 3 of the 4 images, so the exported graphs run the per-level selection. With the greedy final NMS of run_9, the TorchScript export matched, but the dynamo ONNX export stopped on a guard of `batched_nms` (`batch <= 20/7`): the exports give it every pair of the 250 proposals and 35 classes, 35,000 coordinates per image, above the 100,000 of `torchvision/ops/boxes.py:86` from 3 images on. The code of the base commit (0f2f6ed, before the setting) stops the same way on the same checkpoint, so this limit predates the selection and does not depend on it.

## Left Out
In case a later stage needs them, these parts of the references are not taken:
- **Non-finite proposals** (Detectron2 refusing them in training): the training already stops on a non-finite loss.
- **The empty final detections** of torchvision (`roi_heads.py:724`): none measured, see decision c.
- **A score threshold on the proposals** (torchvision `score_thresh`, 0.0 by default): no effect at its default.
- **A different count per level** (Detectron2 can give the training and test counts only, the same on every level).
