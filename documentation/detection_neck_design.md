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
- **`tests/models/detection/test_feature_neck.py`** (new): output keys and shapes of the three types on the levels 4 to 7 of run-1 (including the 16 to 16 pair) and on the levels 5 to 7 of the fixture; no parameter for `none`; the `ValueError` on sizes that are not whole multiples; a gradient reaching every lateral.
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
| feature pyramid | 4 laterals 1x1, 4 outputs 3x3 | about 41,000 |
| path aggregation | the pyramid, 3 downsampling 3x3, 3 smoothing 3x3 | about 96,000 |

The heaviest convolution is a 3x3 on the 64 x 64 level, about 38 million multiply-adds per image, small next to the backbone. The implementation is about one day of work: some 150 lines for the module, the rest for the configuration, the run name, the transfer and the tests.

## Open Decisions
In case the stage results call for it, these choices are left for later:
- **Addition or concatenation:** PANet adds, YOLOv4 concatenates then reduces with a 1x1 convolution. Addition is cheaper and keeps the channels; concatenation can be a later value of `Neck.type`.
- **Transferable neck:** a `Neck.pretrained` block would let a detection run initialise the neck of an instance segmentation run, as for the heads; it needs entries in `TRANSFERABLE_COMPONENTS` (appended last, its order setting the optimizer groups), `PRETRAINED_COMPONENT_NAMES`, `PRETRAINED_TOKEN_PREFIXES` and `COMPONENT_NAMES`. Left out until a transfer needs it.
- **One head shared by every level:** with a common channel count the levels could share one region proposal and one detection head (`suggestions.md` item 1); out of this stage, which keeps one head per level.
- **A finer level:** the pyramid makes block 3 (128 x 128) worth adding to `DetectionLevels` for the small objects (`suggestions.md` items 1 and 5), a separate experiment.

## Order Of Work
After stage 2b, the neck goes in on its own branch:
1. `FeatureNeck` and its test.
2. The detector, export and instance segmenter wiring, with the baseline test unchanged under `none`.
3. The configuration section, its checks, the configuration files, the fixture and the converter of the saved runs.
4. The run name and the transfer comparison, with their tests.
5. The full suite against the baseline, then the stage 3 launches (the feature pyramid against run_3 and run_4), then stage 4 (path aggregation against the better of the two). Stage 3 kept the feature pyramid on 2026-11-08 (+0.0096 mAP@[0.5:0.95]), so stage 4 trains path aggregation against run_9 and run_10.
