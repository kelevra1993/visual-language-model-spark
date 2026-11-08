# Detection Neck Design

## Table Of Contents
- [Purpose](#purpose)
- [Current Hand-Off Between Backbone And Heads](#current-hand-off-between-backbone-and-heads)
- [Neck Module](#neck-module)
- [Configuration](#configuration)
- [Code Changes](#code-changes)
- [Tests](#tests)
- [Cost](#cost)
- [Open Decisions](#open-decisions)
- [Order Of Work](#order-of-work)
- [Path Aggregation Design (Stage 4)](#path-aggregation-design-stage-4)
  - [Sources Read](#sources-read)
  - [The Bottom-Up Path Of PANet](#the-bottom-up-path-of-panet)
  - [Implementation Against The Sources](#implementation-against-the-sources)
  - [Cost Of The Bottom-Up Path](#cost-of-the-bottom-up-path)
  - [Expected Effect](#expected-effect)
  - [Stage 4 Pair](#stage-4-pair)
  - [Left Out Of Stage 4](#left-out-of-stage-4)

## Purpose
This design is for whoever implements stages 3 and 4 of `architecture_benchmarking/object_detection_experiment_plan.md`, which compare three heads inputs: the backbone blocks as today (no neck, every level's heads reading its block directly), a top-down feature pyramid (`suggestions.md` item 1), then path aggregation, a bottom-up path after the pyramid (`suggestions.md` item 37, PANet and the neck of YOLOv4). The user moved the neck to the first stages on 2026-11-07, both types being merged before the feature pyramid pair.

The user's choices of 2026-11-07, after the review of this design against the code:
- **Channels:** 32 on every level, those of the backbone blocks, so that the heads keep their size and the stage measures the neck alone.
- **Normalization:** batch normalization, as the backbone and the heads.
- **Transfers:** region proposal or detection heads trained after a neck are refused in another run until the neck is transferable (see [Open Decisions](#open-decisions)), since they would otherwise read the outputs of a new, untrained neck.

## Current Hand-Off Between Backbone And Heads
To find where the neck goes, note that the backbone gives every head the same dictionary of feature maps keyed by detection level (`{"4": ..., "7": ...}`, the blocks of `DetectionLevels`):
- **Training and evaluation:** `TwoStageDetector.forward_detection_stages` (`two_stage_detector.py:659`) calls `self.backbone(input_tensor=...)` and passes `backbone_output_tensor_dictionary` to `_forward_region_proposals`, `_forward_detections` and, through the returned dictionary, the mask heads of `InstanceSegmenter.forward` (`instance_segmenter.py:111`).
- **Export:** `InferenceDetector` (`inference_detector.py:86`) calls the backbone the same way; `InferenceInstanceSegmenter` reads the `feature_maps` it returns.
- **Head input channels:** every region proposal, detection and mask head takes its `input_channels` and `feature_map_size` from `self.detection_input_information`, the dry run of `Backbone.compute_detection_input_information` (`two_stage_detector.py:74, 126, 169`, `instance_segmenter.py:75`).
- **Levels of run-1.yaml at 1024 pixels** (dry run): block 4 64 x 64, block 5 32 x 32, block 6 16 x 16, block 7 16 x 16, all with 32 channels. Block 7 has no pooling (`last_max_pooling: False`), so the two coarsest levels share their resolution.

The neck is therefore a module taking that dictionary and returning one with the same keys, applied at the two call sites, and reporting its output channels in place of `detection_input_information`. The heads, anchors, proposal filter, target assignment and losses do not change.

## Neck Module
In order to keep the detector readable, the neck lives in a new file `vision_training/models/detection/feature_neck.py`, class `FeatureNeck(nn.Module)`:
- **Constructor:** `FeatureNeck(neck_type, level_information, output_channels, feature_map_normalization, batch_normalization_momentum, device, dtype)`, with `level_information` the `detection_input_information` of the backbone (levels ordered finest to coarsest by block number) and `neck_type` one of `"none"`, `"feature_pyramid"`, `"path_aggregation"`.
- **Layers**, every one a `ConvolutionBlock` with `number_layers=1` and the module flags off (no residual, inception or squeeze and excitation, stride allowed), normalization from `Neck.normalization` and `BATCH_NORMALIZATION_MOMENTUM`:
  - **Laterals:** one 1x1 convolution per level from its backbone channels to `output_channels`, without activation (as in FPN).
  - **Top-down outputs:** one 3x3 convolution per level on the sum, without activation.
  - **Bottom-up (path aggregation only):** for every level but the finest, one 3x3 downsampling convolution of the finer output, with ReLU, and one 3x3 convolution on the sum, with ReLU.
- **Forward**, `forward(feature_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`, looping over a list of level names fixed in the constructor so that the TorchScript trace of the export holds:
  - `none`: returns the dictionary unchanged.
  - Top-down: `merged[coarsest] = lateral(coarsest)`, then from coarse to fine `merged[level] = lateral(level) + interpolate(merged[coarser], size=size of level, mode="nearest")`, and `pyramid[level] = output_convolution(merged[level])`. Between blocks 6 and 7 the interpolation keeps the size (16 to 16), so the two are added directly.
  - Bottom-up: `aggregated[finest] = pyramid[finest]`, then from fine to coarse `aggregated[level] = smoothing(pyramid[level] + downsampling(aggregated[finer]))`.
- **Resampling from the real sizes:** the stride of each downsampling convolution is `finer size // level size` (2 between blocks 4 and 5 or 5 and 6, 1 between 6 and 7), and padding 1. The constructor raises a `ValueError` when a finer size is not a whole multiple of the coarser one, naming the two levels and their sizes.
- **`compute_output_information() -> Dict[str, Dict[str, int]]`:** the same dictionary as `compute_detection_input_information` with `number_channels` set to `output_channels`, or the backbone one unchanged for `none`. The heads are built from it.
- **Merge by addition**, as PANet; YOLOv4 concatenates instead (see [Open Decisions](#open-decisions)).

With `none`, the module has no parameters and no buffers, so the state dictionary, the recorded baselines of `test_detector_baseline.py` and the checkpoints of every earlier run stay the same.

## Configuration
So that the stage can switch the neck from the launch files, a new `Neck` section sits between `DetectionLevels` and `RegionProposal` in every DET and INS configuration:

```yaml
# Feature neck between the DetectionLevels blocks and the heads.
Neck:
  # "none" (heads read the blocks), "feature_pyramid" (1x1 laterals and top-down path) or "path_aggregation"
  # (feature pyramid then bottom-up path, https://arxiv.org/abs/1803.01534).
  type: "none"
  # Channels of every level after the neck, read by the heads; ignored with "none".
  output_channels: 32
  normalization:
    # "batch", "layer" (over channels) or "none"; ignored with "none".
    feature_map_normalization: "batch"
```

- **Checks:** a `check_neck_configuration` in `detection_checks.py`, called from `check_detection_configuration` (so the instance segmentation checks get it), refusing a `type` outside the three values, a non-positive `output_channels` (`check_positive_integers`) and a normalization outside `VALID_FEATURE_MAP_NORMALIZATIONS`.
- **Files to edit:** the tracked `configuration_example.yaml` and `configuration_synthetic_shapes.yaml` of `object_detection/`, the two of `instance_segmentation/`, the DET and INS run configurations of `documentation/task workflows/run_configurations/`, and the test fixture `get_detection_model_configuration` (`tests/fixtures/configuration_fixtures.py`). The launch base `run-1.yaml` and the user's `detection-run-configuration-defaults.yaml` are edited in the user's checkout after the merge: `run-1.yaml` holds uncommitted changes there, which a merge touching it would refuse to overwrite.
- **Saved runs:** the configuration values are read strictly, so the saved configuration of a run written before the neck lacks the key. `convert_fixed_settings.py` only removes settings and refuses any other change, so it is extended to add the sections a task now requires (`Neck` with `type: "none"` before `RegionProposal`, for DET and INS), its check accepting exactly these additions. The 27 saved DET and INS runs of the repository are converted, and the transfer reads a missing `Neck` of a source run as `none`, like `BACKBONE_MODULE_DEFAULTS` (`model_transfer.py:54`).

## Code Changes
To be able to follow every file the change touches, here they are in the order of the data:
1. **`feature_neck.py`** (new): `FeatureNeck` as above.
2. **`two_stage_detector.py`:** read `configuration["Neck"]`, build `self.neck` after `self.detection_input_information`, replace `self.detection_input_information` by `self.neck.compute_output_information()` before the heads are built, apply `self.neck` right after the backbone in `forward_detection_stages` (the returned `backbone_output_tensor_dictionary` then holds the neck outputs, so the mask heads read them), add `"Neck"` to `get_size_components`, and print the neck in `print_summary`.
3. **`inference_detector.py`:** keep `model.neck` and apply it after `self.backbone(input_tensor=images)` in `detect_objects`, whose returned `feature_maps` are then the neck outputs, read by the mask heads of `InferenceInstanceSegmenter`.
4. **`instance_segmenter.py`:** nothing beyond the inherited `detection_input_information`, which already carries the neck channels for the mask heads.
5. **`run_name.py`, `build_detection_sections`:** after the levels, a `NFPN<channels><normalization code>` or `NPAN<channels><normalization code>` token, and none for `none`, so that every earlier run name stays the same.
6. **`model_transfer.py`:** `compare_detection_heads` refuses heads trained after a neck (a source `Neck.type` other than `none`, a missing source `Neck` counting as `none`). Today the channel counts are only compared on the weights, so heads trained after a neck of 32 channels would load into a run without neck whose blocks also have 32 channels, or into a run whose own neck is new. The neck itself is not transferable in the first version (see [Open Decisions](#open-decisions)), so it falls into the first optimizer group of `base_trainer.build_optimizer` with the rest of the model.
7. **`detection_checks.py`:** `check_neck_configuration`.
8. **Converter:** the `Neck` addition for the saved runs, with its test.
9. **Documentation:** the plan document of stages 3 and 4, `suggestions.md` items 1 and 37, `tasks_reference.md`, `configuration_reference.md` (the sections of the detection tasks and the run name tokens), `training_reference.md` (the transfer tables), and the README, whose object detection configuration line and section table list `DetectionLevels`, `RegionProposal` and `Detector`. The comments of `two_stage_detector.py` calling the levels a feature pyramid network are corrected, the code having no pyramid before the neck.
10. **Driver:** the git-ignored `run_benchmark_stages.py` gets the neck type in `SETTING_KEYS` and `SWEPT_SETTINGS`.

The model summary's receptive field table keeps reading the backbone blocks only, so it understates the context the neck brings to the fine levels; it is left as it is.

## Tests
Before the stage starts, the following tests show that the neck works and that runs without it are untouched:
- **`tests/models/detection/test_feature_neck.py`** (new): output keys and shapes of the three types on the levels 4 to 7 of run-1 (including the 16 to 16 pair) and on the levels 5 to 7 of the fixture; no parameter for `none`; the `ValueError` on sizes that are not whole multiples; a gradient reaching every lateral; since 2026-11-08, the building block of PANet (see [Implementation Against The Sources](#implementation-against-the-sources)).
- **`test_detector_baseline.py`:** unchanged recorded losses and detections with `none`.
- **`test_two_stage_detector.py`** and **`test_instance_segmenter.py`:** the detection stages and the mask loss run with `path_aggregation`.
- **`test_inference_detector.py`:** `test_inference_detector_matches_the_eager_model` and `test_torchscript_round_trip` with `path_aggregation`.
- **`test_run_name.py`:** the `NFPN` and `NPAN` tokens, and no token for `none`.
- **`test_configuration_checker.py`:** refusal of an unknown type, of zero channels and of an unknown normalization.
- **`test_model_transfer.py`:** heads trained after a neck are refused, and a source run without `Neck` transfers into a run with `type: "none"`.
- **`test_convert_fixed_settings.py`:** the `Neck` section added to a DET file lacking it, nothing added to a file of another task or one already holding it.

## Cost
Should you need the size of the neck before launching, at 32 channels on the four levels of run-1:

| Neck | Convolutions | Weights |
|---|---|---|
| none | none | 0 |
| feature pyramid | 4 laterals 1x1, 4 outputs 3x3 | 41,728 |
| path aggregation | the pyramid, 3 downsampling 3x3, 3 smoothing 3x3 | 97,600 |

The heaviest convolution is a 3x3 on the 64 x 64 level, about 38 million multiply-adds per image, small next to the backbone. The implementation is about one day of work: some 150 lines for the module, the rest for the configuration, the run name, the transfer and the tests.

## Open Decisions
In case the stage results call for it, these choices are left for later:
- **Addition or concatenation:** PANet adds, YOLOv4 concatenates then fuses with five convolutions, the first a 1x1 bringing the channels back (see [Left Out Of Stage 4](#left-out-of-stage-4)). Addition is cheaper and keeps the channels; concatenation can be a later value of `Neck.type`.
- **Transferable neck:** a `Neck.pretrained` block would let a detection run initialise the neck of an instance segmentation run, as for the heads; it needs entries in `TRANSFERABLE_COMPONENTS` (appended last, its order setting the optimizer groups), `PRETRAINED_COMPONENT_NAMES`, `PRETRAINED_TOKEN_PREFIXES` and `COMPONENT_NAMES`. Left out until a transfer needs it.
- **One head shared by every level:** with a common channel count the levels could share one region proposal and one detection head (`suggestions.md` item 1); out of this stage, which keeps one head per level.
- **A finer level:** the pyramid makes block 3 (128 x 128) worth adding to `DetectionLevels` for the small objects (`suggestions.md` items 1 and 5), a separate experiment.

## Order Of Work
After stage 2b, the neck goes in on its own branch:
1. `FeatureNeck` and its test.
2. The detector, export and instance segmenter wiring, with the baseline test unchanged under `none`.
3. The configuration section, its checks, the configuration files, the fixture and the converter of the saved runs.
4. The run name and the transfer comparison, with their tests.
5. The full suite against the baseline, then the stage 3 launches (the feature pyramid against run_3 and run_4), then stage 4 (path aggregation against the better of the two). Stage 3 kept the feature pyramid on 2026-11-08 (+0.0096 mAP@[0.5:0.95]), so stage 4 trains path aggregation against run_9 and run_10. The path aggregation was checked against the PANet paper and its official code the same day ([Path Aggregation Design](#path-aggregation-design-stage-4)) and needs no other code; its launcher is written and dry-run.

## Path Aggregation Design (Stage 4)
Say you want to know whether stage 4 trains the path aggregation the papers describe: this section checks the `path_aggregation` neck merged on 2026-11-07 against the PANet paper, its official code and the variants that came after it, then gives its cost, the effect to expect and the comparison of the stage. The review of 2026-11-08 found the neck to be the bottom-up path of the official PANet code, batch normalization standing for group normalization, so stage 4 needs no other code; `test_path_aggregation_follows_the_panet_building_block` (`tests/models/detection/test_feature_neck.py`) now holds that order.

### Sources Read
If you are looking for where a statement of this section comes from, these sources were read in full on 2026-11-08, the papers as text from their arXiv HTML and the code from the official repositories:

| Source | Read for |
|---|---|
| PANet: Liu, Qi, Qin, Shi and Jia, "Path Aggregation Network for Instance Segmentation", CVPR 2018, [arXiv 1803.01534](https://arxiv.org/abs/1803.01534) | Sections 3.1 (bottom-up path augmentation), 3.2 (adaptive feature pooling) and 3.3 (fully-connected fusion), Section 4.2 with Tables 3 and 4 (ablations on COCO val-2017) |
| PANet official code, [ShuLiu1993/PANet](https://github.com/ShuLiu1993/PANet) | `lib/modeling/FPN.py` (class `fpn` with `panet_buttomup`) and `configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml` (the released model) |
| FPN: Lin, Dollár, Girshick, He, Hariharan and Belongie, "Feature Pyramid Networks for Object Detection", CVPR 2017, [arXiv 1612.03144](https://arxiv.org/abs/1612.03144) | Section 3 (the top-down building block, the extra layers without non-linearity) |
| YOLOv4: Bochkovskiy, Wang and Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection", 2020, [arXiv 2004.10934](https://arxiv.org/abs/2004.10934), with `cfg/yolov4.cfg` of [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Sections 3.1 and 3.3 with Figure 6 (modified PAN), Table 5 of Section 4.3, layers 136 to 161 of the configuration |
| EfficientDet: Tan, Pang and Le, "EfficientDet: Scalable and Efficient Object Detection", CVPR 2020, [arXiv 1911.09070](https://arxiv.org/abs/1911.09070) | Section 3 (BiFPN, fast normalized fusion) and Table 5 (feature networks compared) |
| mmdetection, [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | `mmdet/models/necks/pafpn.py`, `configs/pafpn` and the Faster R-CNN model zoo, a common reimplementation |
| The YOLOv4 explainer the user supplied, [youtu.be/b148nt9P8J0](https://youtu.be/b148nt9P8J0), its transcript read in full | Path aggregation at 1:00 to 1:02, the layers of the YOLOv4 neck at 1:10 to 1:11 |

### The Bottom-Up Path Of PANet
For the neck to be that of PANet and not a look-alike, its operations follow Section 3.1 of the paper, on the levels of the feature pyramid (`P2` to `P5` there, `P4` to `P7` here, the blocks of `DetectionLevels`):
- **Finest level:** "Note that N2 is simply P2, without any processing."
- **Building block:** "each building block takes a higher resolution feature map N_i and a coarser map P_{i+1} through lateral connection and generates the new feature map N_{i+1}. Each feature map N_i first goes through a 3×3 convolutional layer with stride 2 to reduce the spatial size. Then each element of feature map P_{i+1} and the down-sampled map are added through lateral connection. The fused feature map is then processed by another 3×3 convolutional layer to generate N_{i+1} for following sub-networks."
- **Activation and width:** "we consistently use channel 256 of feature maps. All convolutional layers are followed by a ReLU."
- **Readers:** "The feature grid for each proposal is then pooled from new feature maps, i.e., {N2, N3, N4, N5}."

With the batch normalization chosen for the neck, the path is therefore `N4 = P4`, then `N(i+1) = ReLU(BN(conv3x3(P(i+1) + ReLU(BN(strided conv3x3(N(i)))))))` for levels 5, 6 and 7, each step reading the previous output `N(i)`. The pyramid maps `P` themselves are those of the FPN paper, `P(i) = conv3x3(lateral(C(i)) + upsample(merged(i+1)))` with "no non-linearities in these extra layers". The motivation of the paper is localisation: "high response to edges or instance parts is a strong indicator to accurately localize instances", and the bottom-up path gives the low levels a route of "less than 10 layers" to the top one, against the "100+ layers" of the backbone. Here, with the feature pyramid alone, the level 7 output reads block 7 only, so the 64 x 64 detail of block 4 reaches the heads of levels 6 and 7 (anchors of 128 to 768 pixels) through the backbone blocks 5 to 7; the bottom-up path adds a route of six convolutions from `P4` to `N7`. The video describes the same motivation (1:00 to 1:01: "the path from lower level to top level features is a very long one consisting of a lot of layers").

### Implementation Against The Sources
To keep the deliberate departures apart from accidental ones, the table compares every step of the bottom-up path in the paper, the released PANet model, the PAFPN of mmdetection, YOLOv4 and `feature_neck.py`:

| Step | PANet paper | PANet official code (released model) | mmdetection PAFPN | YOLOv4 (darknet) | Ours |
|---|---|---|---|---|---|
| Pyramid `P` | FPN: 1x1 laterals, nearest 2x upsampling, addition, 3x3 per level, no activation | 1x1 laterals and 3x3 outputs with GN, no ReLU, nearest `scale_factor=2` (`FPN.py:102-131, 344-351`) | the same, no normalization or activation by default | 1x1 convolutions, upsampling, concatenation and five convolutions (layers 117 to 126 and 127 to 136, the top-down path of YOLOv3) | 1x1 laterals and 3x3 outputs with BN, no activation, nearest interpolation to the size of the level, none between levels 6 and 7 (`feature_neck.py:106-118, 151-171`) |
| Finest output | `N2 = P2`, unprocessed | `fpn_output_blobs.append(fpn_middle_blobs[-1])` (`FPN.py:288`) | `outs.append(inter_outs[0])` | the finest map of the top-down path (layer 136, read again by route 140) | `aggregated_maps[finest] = pyramid_maps[finest]` (`feature_neck.py:176`) |
| Downsampling | 3x3, stride 2, of `N(i)` | 3x3 stride 2, GN, ReLU, of `N(i)` (`FPN.py:141-145, 290`) | 3x3 stride 2 of the sum before its 3x3 convolution, not of `N(i)` (`pafpn.py:122-124`) | 3x3 stride 2, BN, leaky ReLU (layers 141 and 152) | 3x3, stride of the size ratio (2, 2, then 1 between the two 16 x 16 levels), BN, ReLU, of `N(i)` (`feature_neck.py:126-130, 178`) |
| Merge | element-wise addition with `P(i+1)` | addition (`FPN.py:292`) | addition | concatenation (route layers 142 and 153; Section 3.3: "replace shortcut connection of PAN to concatenation") | addition (`feature_neck.py:180`) |
| After the merge | 3x3, giving `N(i+1)` | 3x3, GN, ReLU, carried to the next step (`FPN.py:147-151, 293-294`) | 3x3 on the outputs only, the next step reading the sum (`pafpn.py:126-130`) | five convolutions, 1x1 and 3x3 alternating, BN, leaky ReLU (layers 143-147 and 154-158) | 3x3, BN, ReLU, carried to the next step (`feature_neck.py:131-134, 179-180`) |
| Normalization | batch normalization after every new layer in the rows of Table 3 with synchronized batch normalization, the path row included (Section 4.2: "make all new layers followed by a batch normalization layer") | GN (`USE_GN: True`) | none in the model zoo PAFPN (`configs/pafpn/faster-rcnn_r50_pafpn_1x_coco.py` sets no `norm_cfg`) | BN | BN, the user's choice of 2026-11-07 |
| Channels | 256 | 256 | 256 | 128, 256 and 512 by level | 32, those of the blocks |
| Extra level | not stated | `P6`, a max pool of kernel 1 and stride 2 of `N5`, after the path (`FPN.py:171, 296-297`) | max pool of the last output | none | none, the levels being those of `DetectionLevels` |
| Initialisation | not stated | XavierFill, uniform within `sqrt(3/fan_in)` (`lib/nn/init.py:11-16`), zero biases (`FPN.py:194-200`) | not read | not read | PyTorch default, uniform within `1/sqrt(fan_in)`, as every convolution of the model |
| Readers | the proposals, pooled from `N2` to `N5` | the region proposal, box and mask heads (`MULTILEVEL_RPN`, `MULTILEVEL_ROIS`) | the region proposal and box heads | the three YOLO heads | the region proposal, detection and mask heads of every level |

The departures are decisions:
- **Batch instead of group normalization:** the user chose batch normalization for the neck on 2026-11-07, as in the backbone and the heads; the paper's own path row in Table 3 used it too.
- **Stride 1 between levels 6 and 7:** block 7 has no pooling, both levels being 16 x 16, so the downsampling of that step is a plain 3x3 convolution; PANet's levels always halve.
- **32 channels:** those of the blocks, so that the heads keep their size and the stage measures the neck alone.
- **No extra level:** the anchors of levels 4 to 7 (16 to 768 pixels) already cover the objects.
- **Initialisation:** the batch normalization after every convolution cancels the scale of its weights and its bias in the forward pass; the PyTorch default being smaller than XavierFill (bound `1/sqrt(fan_in)` against `sqrt(3/fan_in)`, a third of the variance), the neck takes larger relative steps at the start, as every other convolution of the model.

The PAFPN of mmdetection is the one variant whose order differs: its next step reads the sum before the 3x3 convolution, so the chain of the bottom-up path runs through additions and strided convolutions only. The new test fails on that order, and also when the path does not chain, when its ReLU is removed or when the finest level is processed.

### Cost Of The Bottom-Up Path
Whenever the result of stage 4 is weighed against its price, these are the counts at the levels of the benchmark (blocks 4 to 7 at 64, 32, 16 and 16 cells, 32 channels), measured on the `FeatureNeck` built as the detector builds it:

| Neck | Convolutions | Parameters | Multiply-adds per image |
|---|---|---|---|
| feature pyramid | 8 | 41,728 | 57.7 million |
| path aggregation | 14 | 97,600 | 86.0 million |
| bottom-up path alone | 6 | 55,872 (0.22 MB) | 28.3 million, 18.9 of them on the 32 x 32 level |

In stage 3, the feature pyramid cost 4.3 % of wall time per launch (14.45 hours against 13.85), and 1.5 % per iteration from 5000 on, once the machine was no longer busier; the bottom-up path adds half its multiply-adds. Before stage 3, 40 iterations gave 41.6 and 42.6 images per second with path aggregation, against 41.0 and 45.2 with the feature pyramid, too few to measure a cost, and the same peak GPU memory (17.3 GB). The wall times of run_11 and run_12 will measure the cost as for stage 3.

### Expected Effect
Suppose the synthetic shapes respond as COCO did: the published gains of the bottom-up path are about one point of box AP, largest on the large objects:
- **PANet, Table 3** (ResNet-50, independently trained detector, COCO val-2017): box AP 37.1 to 38.0 (+0.9), large 47.1 to 49.4 (+2.3), medium 42.5 to 42.9 (+0.4), small 24.2 to 23.3 (-0.9); with adaptive feature pooling, 37.9 to 39.0 (+1.1); mask AP +0.7 and +0.6. The paper: "The improvement on instances with large scale is most significant. This verifies usefulness of information from lower feature levels."
- **EfficientDet, Table 5:** repeated FPN with PANet 44.08 against repeated top-down FPN 42.29, at the same parameters and FLOPs (1.0x), every path repeated and every convolution depthwise separable.
- **mmdetection model zoo:** Faster R-CNN R-50 1x, 37.5 box AP with PAFPN against 37.4 with FPN (17.2 against 21.4 images per second), a reimplementation without normalization or activation in the neck and with the order of the table above.
- **YOLOv4** uses PANet "instead of the FPN used in YOLOv3" (Section 3.1) without measuring the change alone: every row of its Table 5 holds PANet, and the video (1:01) gives no number either.

Stage 3 showed where the neck helps here: the feature pyramid gained most on the tight boxes (mAP@0.95 +0.018) and on the medium and large objects (+0.012, +0.013), the small ones about even (+0.002). The bottom-up path brings the detail of block 4 to the levels of the large anchors, so mAP@0.75, mAP@0.95 and the large objects are the results to read beside the decision metric. The published gains are about a point of AP, from necks of 256 channels on COCO, so a tie within the launch gaps (0.0038 for run_9 and run_10) is a plausible outcome.

### Stage 4 Pair
Once the user gives the go, stage 4 trains run_11 and run_12 with the settings of run_9 and run_10 (background IoU minimum 0.1, detector batch normalization, learning rate 1e-3, 15,000 iterations, 1 loader worker) and `Neck.type: "path_aggregation"`, run name token `L4567-NPAN32BN`:
- **Launcher:** the git-ignored `hyperparameter_experiments/object_detection/launch_runs_11_and_12_path_aggregation.py` writes the two configuration files as `run_benchmark_stages.py` does, drops the page cache and starts the two trainings 30 seconds apart. Its dry run (`--dry-run`, 2026-11-08) loaded both files through the configuration checks: they differ from `run-9.yaml` and `run-10.yaml` by their header comment, their experiment folder and `Neck.type` only.
- **Decision:** Test_EMA mAP@[0.5:0.95]/area at 15,000, pair means, with `compare_detection_pairs.py --challenger run_11 run_12 --reference run_9 run_10`; path aggregation is kept if it beats 0.8079 by more than the larger launch gap of the two pairs (0.0038 for run_9 and run_10), a tie or a loss keeping the feature pyramid as the reference of stage 5. Both types stay in the code either way, `Neck.type` choosing between them.
- **Duration:** about 15 hours for the pair, run side by side, as stage 3.

### Left Out Of Stage 4
If one wishes to go further than the bottom-up path, these parts of PANet and of the necks that followed it stay out of the stage, each a later stage of its own:
- **Adaptive feature pooling (PANet Section 3.2):** every proposal pooled from all levels, each grid through its own first layer, then fused by max; Table 4 gives 37.9 box AP fusing after the first layer, 37.3 before it, 37.1 without. The detector routes every proposal to the head of its origin level (`two_stage_detector.py:581-600`), so this needs one detection head reading every level (see the head shared by every level in [Open Decisions](#open-decisions)).
- **Fully-connected fusion (PANet Section 3.3):** a fully connected branch in the mask head, +0.7 mask AP; for the instance segmentation task, not the detection benchmark.
- **Concatenation (YOLOv4):** the downsampled map concatenated with the pyramid map, then five convolutions of alternating 1x1 and 3x3 (darknet layers 141 to 147 and 152 to 158) with batch normalization and leaky ReLU; heavier, and with no published comparison against the addition.
- **Weighted fusion (EfficientDet):** `O = sum_i(w_i / (epsilon + sum_j(w_j)) * I_i)`, every `w_i` kept non-negative by a ReLU and `epsilon = 0.0001`, inside a repeated bidirectional path (BiFPN); Table 5: 44.39 weighted against 43.94 unweighted.
- **Group normalization:** that of the released PANet model, against the batch normalization of the neck, the backbone and the heads.
- **Spatial pyramid pooling (YOLOv4):** a block after the backbone, not part of the path (`suggestions.md` item 39).
