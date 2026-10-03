# Implementation Plan: The Sequence Comparison Task And The Transfer Of The Temporal Module And Head

> Plan validated on 2026-10-03 and implemented phase by phase, each phase committed and pushed. The implementation
> notes (differences from the plan, acceptance results) are added at the end once the last phase is done.

## Table Of Contents
1. Context
2. Decisions
3. The TensorFlow Comparison Project
4. Standards Of vision_training To Follow
5. Design
6. Phases
7. Tests
8. Acceptance
9. Risks And Open Points
10. Verification
11. Implementation Notes

---

## 1. Context
- `previous_classifiers_comparison.md`, section 4.5:
  - item 20: a comparison task classifying a pair of sequences, a "before" sequence and an "after" sequence;
  - item 18: the transfer of the temporal module and the head between runs, plus `freeze_normalization_parameters`.
- Use: classify the change between two visits (e.g. an object lost between the before and the after photos), and
  start such a model from a trained sequence classifier, as the TensorFlow project did with `restore_aggregator`.

## 2. Decisions
**Taken with the user:**
- **Scope:** item 20 (the comparison task, export included) plus item 18.
- **Fusions:** `concatenation` `[a, b]`, `difference` `[a, b, b - a]`, and `cross_attention`: the frames of each member
  attend to the frames of the other member, then `[a, b, b - a]`.
- **Member names:** `before` / `after` in the manifest, the model arguments, the exports and the configuration.

**Taken in this plan:**
- **Task:** name `sequence_comparison`, run name code `CMP`.
- **Combined batch layout:** a pair batch is `images` [B, 2L, 3, S, S] + `sequence_mask` [B, 2L], split at
  L = `sequence_length`.
  - Every generic BaseTrainer path (augmentation, viewer, evaluation) stays unchanged.
  - Both members get the same flips and rotations, as in the TensorFlow project.
  - The photometric draws are shared too.
- **Members:** each member is one window of at most L frames: evenly spaced frames when it is longer, padding when it
  is shorter. `window_stride` is not used by this task.
- **Frozen components:** only their batch norm layers are set to eval, so their statistics stay fixed.
  - Setting a whole frozen GRU/LSTM to eval crashes: the cuDNN recurrent backward only runs in training mode, and
    gradients still flow through it.
  - For the backbone this is identical to today, since it has no dropout.
- **Not included:** error dumps (item 2 of the comparison document covers them for every task).

## 3. The TensorFlow Comparison Project (`previous_classifiers/comparison`, v2.0.0)
- **Data:** one pair list JSON per split, `{"source": folder, "data": [{"before", "after", "label", "sequence"}]}`.
  `before` and `after` are frame folders under `source`, and `sequence` names the pair.
- **Records:** TFRecords with the frames of both members.
  - Frames are read by a thread pool in completion order, so their order is lost.
  - `sequence_shuffling` also shuffles them at every iteration.
- **Model:**
  - The same CNN runs on both members, with batch norm moments computed over the concatenation of both.
  - The same aggregator (BiLSTM, squeeze-and-excite weights, mean, max or k-min-max) runs on both members.
  - The two feature vectors are concatenated, then go through FC1, FC2 and the classifier.
- **Augmentation:** applied to 40 % of the pairs. Brightness, crop and noise are drawn per frame; flips, grayscale and
  rotations use one draw for both members.
- **Transfer:** `start_from_scratch: {activated, restore_aggregator}` restores the CNN, and optionally the
  aggregator, of a many-to-one run.
- **Synthetic data:** the before member has 2–4 shapes; the after member has the same count or one fewer
  (p = 0.5), giving `loss_of_object` / `same_number_of_objects`. Members have 2–5 independent frames, and the
  validation shapes are larger.
- **Reports and deployment:** a confusion matrix, Excel files and the threshold finder; error dump folders
  `<TRUE>-TO-<PREDICTED>/<pair>/{Before,After}/`; a frozen `.pb` with the two placeholders `Before-Input` and
  `After-Input`.

## 4. Standards Of vision_training To Follow
- **Registries to fill for a task:**
  - `core/task_registry.py` TASK_TRAINER_CLASSES;
  - `models/model_registry.py` TASK_MODEL_CLASSES;
  - `configuration/configuration_checker.py` TASK_CONFIGURATION_CHECKS;
  - `configuration/run_name.py` TASK_CODES and TASK_RUN_NAME_SECTIONS;
  - `deployment/export_specifications.py` INFERENCE_MODEL_CLASSES, TASK_MODEL_NAMES and SEQUENCE_TASKS;
  - `data/synthetic/create_synthetic_dataset.py` TASK_CATEGORIES.

  `tests/core/test_task_registry.py` requires the first four to hold the same tasks, so they are filled in one step.
- **Contracts (`core/contracts.py`):**
  - batches carry `images`, `labels`, `sequence_mask` and `sample_identifiers`;
  - every `*_loss` key the model returns is summed;
  - images are uint8 BGR in [0, 255] in the loaders, cast to float in `prepare_batch`, and normalized inside
    `Backbone.forward`.
- **Configuration:**
  - every key is required and read with `["key"]`; a section a task does not use is not read;
  - checks return lists of messages;
  - migrations insert text under a section header, keeping the comments, and do nothing when run twice.
- **Code style:**
  - type hints everywhere;
  - "Args:" docstrings that start with the role of the function in the pipeline;
  - fully spelled-out names, and keyword arguments in calls;
  - no nested functions or lambdas;
  - a comment above every block, and constants at the top of the module with a comment;
  - `print_color` with `file://` links;
  - extend existing functions instead of adding parallel ones;
  - no mention of the TensorFlow code in docstrings, comments or YAML comments.
- **Workflow:**
  - tests for every feature;
  - an acceptance training on synthetic data per phase;
  - the README `## Implementation Plan` holds the checklist of the current phase and is removed at its end;
  - one commit per phase, `[TAG] Description.`, without attribution, then a push;
  - the reference `.pt` files are never regenerated.

---

## 5. Design

### 5.1 Data
- **Manifest** (`<split>/labels.json`, frames under `<split>/data/`):
  ```json
  {"categories": [{"id": 1, "name": "loss_of_object"}, {"id": 2, "name": "same_number_of_objects"}],
   "pairs": [{"id": 1, "name": "train_000000",
              "before": ["train_000000/before/000000.png", "train_000000/before/000001.png"],
              "after": ["train_000000/after/000000.png"], "category_id": 1}]}
  ```
- **`data/sequence/sequence_dataset.py`:**
  - extract the frame-reading loop of `SequenceDataset.__getitem__` into
    `read_window_frames(data_directory: str, frame_files: List[str], sequence_length: int, image_size: int, keep_ratio: bool) -> Tuple[torch.Tensor, torch.Tensor]`,
    which returns uint8 frames [L, 3, S, S] (black padding last) and a bool mask [L];
  - `SequenceDataset` calls it.
- **New `data/sequence/sequence_pair_dataset.py`:**
  - `PAIR_MEMBERS = ["before", "after"]`.
  - `compute_member_frame_indices(number_frames: int, sequence_length: int, short_sequences: str) -> List[int]`.
    Each member becomes one window, after the frame stride:
    - empty member, or shorter than L with `"skip"`: `[]`, which leaves the pair out;
    - at most L frames: every frame, padded;
    - more than L: L evenly spaced frames, `((2 * index + 1) * number_frames) // (2 * sequence_length)`.

    Members are never empty, since cross attention over a fully masked member gives NaN.
  - `split_pair_members(pair_tensor: torch.Tensor, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]`.
  - `SequencePairDataset(data_directory, labels_file, image_size, keep_ratio, label_dictionary, sequence_settings, shuffle=True)`:
    - an item is `{"images": uint8 [2L, 3, S, S] (before window, then after window), "sequence_mask": bool [2L], "labels": int64, "sample_identifiers": int}`;
    - unmapped categories and left-out pairs produce one yellow warning with a `file://` link;
    - attributes `class_names` and `number_classes`, and `get_class_counts()`.
- **`data/loaders.py`:** `get_sequence_dataloaders` builds `SequencePairDataset` when `label_type == "pair"`.
- **`data/split_description.py`:** `count_split_pairs(split_data_folder, sequence_settings) -> (pairs, frames, kept pairs)`.
- **Converter for existing pair list datasets:**
  - extract `link_folder(link_path, target_folder)` into `utilities/os_utilities.py`;
  - extract `write_label_mapping(categories, output_folder) -> Path` into `data/label_mapping.py`, both shared with
    `class_folder_manifests.py`;
  - new `data/sequence/pair_list_manifests.py::create_pair_list_manifests(pair_list_files: Dict[str, str], output_folder: str) -> None`
    reads one pair list JSON per split, links `data` to the source folder, and writes `labels.json` (shared
    alphabetical categories, frames sorted by name) and `label_mapping.json`.

### 5.2 Model
- **New `models/sequence/member_fusion.py`:**
  - `FUSION_TYPES = ["concatenation", "difference", "cross_attention"]`;
  - `class MemberFusion(nn.Module)`:
    - `__init__(self, fusion_type: str, frame_features: int, member_features: int, number_heads: int, dropout_rate: float, device, dtype)`;
    - for cross attention only, one `nn.MultiheadAttention(embed_dim=frame_features, num_heads=..., dropout=..., batch_first=True)`
      shared by both directions, plus a `LayerNorm`;
    - `output_features` is 2 or 3 × member features;
    - `attend_across_members(before_features, after_features, before_mask, after_mask)` computes
      `LayerNorm(x + attention(x, other, other, key_padding_mask=~other_mask))` for each member, then zeroes the
      padding positions again. It returns the inputs unchanged for the other fusion types;
    - `combine_member_vectors(before_vectors, after_vectors)` returns `[a, b]` or `[a, b, b - a]`.
  - Cross attention runs on the frame features, before the temporal module. The temporal module input therefore
    stays F (the channels of the last backbone block), so it can be transferred from a sequence classifier (5.7).
- **New `models/sequence/sequence_comparator.py`, `class SequenceComparator(TaskModel)`:**
  - **Attributes:** the same names as `SequenceClassifier`, so the weight prefixes match for the transfer:
    `backbone`, `global_pooling`, `temporal_module` (`keep_time_axis=False`, `maximum_length=L`), `member_fusion`,
    `classification_head` (input `member_fusion.output_features`), buffer `class_weights`, `number_classes`,
    `sequence_length`, `output_mode`, `label_smoothing`.
  - **`forward(self, before_images, after_images, before_mask=None, after_mask=None, labels=None) -> Dict[str, Any]`:**
    - with masks (training and evaluation): both members are stacked into [2B, L], then one `encode_frames` call
      (batch norm sees both members) and one temporal module call; L must be the same for both members;
    - without masks (deployment): each member is encoded and summarised separately, so the two members may have
      different numbers of frames, with no shape-dependent branch (it would be baked into the export);
    - giving exactly one of the two masks raises a ValueError;
    - returns `class_logits`, `class_probabilities`, and `classification_loss` when labels are given.
  - **Helpers:** `encode_members`, `summarise_members`, `print_summary` (backbone, temporal module, fusion, head),
    `get_size_components`.
  - **Summary panel:** generalise `print_temporal_module_summary` into
    `print_sequence_module_summary(title, module_configuration, output_features, border_style)`. It skips the
    `pretrained` block, its callers are updated, and the fusion panel uses it.

### 5.3 Trainer
- **`trainers/metric_tracking.py`:** `THROUGHPUT_UNITS` gains `"Pairs": "pairs/sec"`. Without it, every console dump
  raises a KeyError and training stops after 5 dumps.
- **New `trainers/sequence_comparison_trainer.py`, `class SequenceComparisonTrainer(SequenceClassificationTrainer)`:**
  - `TASK_NAME = "sequence_comparison"`, `SAMPLE_NAME = "Pairs"`, `LABEL_TYPE = "pair"`;
  - `get_frames_per_sample` returns 2 × L;
  - `describe_split` writes e.g. "8 pairs, 37 frames, 8 kept pairs of 2 x 4 frames";
  - `split_pair_batch(batch)` returns the model arguments: before/after images and masks;
  - `forward_pass` and `predict_batch` call the model with them;
  - `show_input_samples` keeps two windows (Training, Validation). Each window stacks the before frame grid above
    the after frame grid, the frames captioned "before" / "after", and the window captioned with the class.
- **Inherited unchanged:** the evaluation (accuracy, balanced accuracy, macro F1, top-k, per-class P/R/F1, confusion
  matrix, thresholds; "N Test Pairs"), balanced class weights, `prepare_batch`, the augmentation.
- **Docs:** describe the pair layout in `core/contracts.py`, and update the task lists in the docstrings of
  `base_trainer.py` and `sequence_classification_trainer.py`.

### 5.4 Configuration
- **New YAMLs `configurations/sequence_comparison/configuration_example.yaml` and `configuration_synthetic_pairs.yaml`:**
  - sections: ExperimentConfiguration (`task: "sequence_comparison"`), Optimization, Data, Backbone, TemporalModel,
    Fusion, ClassificationHead;
  - `Data.sequence_settings` holds `sequence_length` (maximum frames per member), `frame_stride` and
    `short_sequences`, with no `window_stride`;
  - new section `Fusion: {type, number_heads, dropout_rate}`, commented like the other sections;
  - the example uses GRU and difference;
  - the synthetic configuration:
    - uses mean pooling, difference and `top_k: 1`;
    - turns on flips and 90° rotations, which keep every object;
    - turns off crop, small rotation and grayscale, which can hide objects and change the count;
    - uses `global_pooling: "average"`, since max pooling discards counts;
  - the run name examples in the comments are produced with `build_run_name`.
- **`configuration/sequence_checks.py`:**
  - `check_sequence_settings(model_configuration, cuts_windows: bool)`: `window_stride` is only read when
    `cuts_windows` is True (True for M2O and M2M);
  - the causal message of `check_temporal_model_configuration` becomes task-neutral: "'TemporalModel.causal' must be
    False for this task, which summarises whole sequences (...)";
  - `check_fusion_configuration`:
    - `type` must be in `FUSION_TYPES`, `number_heads` an integer ≥ 1, and `dropout_rate` in [0, 1);
    - for cross attention, the channels of the last backbone block must be a multiple of `number_heads`;
  - `check_sequence_comparison_configuration`: the classification data, head and evaluation checks, the sequence
    settings without windows, the 7 temporal types with no causal module, and the fusion check.
- **`configuration/run_name.py`:**
  - split `build_sequence_sections` into `build_window_tokens` and `build_temporal_tokens`; the M2O and M2M names
    stay byte-identical;
  - add `build_fusion_tokens`: `FUSION_TYPE_CODES = {"concatenation": "FCAT", "difference": "FDIF", "cross_attention": "FXA"}`,
    with `FXA<heads>` and `FDO<rate>` when the rate is above 0;
  - add `build_sequence_comparison_sections`;
  - example: `CMP_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T8_MEAN_FDIF_HAVG.64-DO0.2_MC2_AUG-BS16-LR1e-3`.

### 5.5 Synthetic Dataset (`data/synthetic/create_synthetic_dataset.py`)
- **Shape helpers, extracted from the existing code and shared:**
  - `draw_shape_descriptor`: a shape class, size and contrasting colour, with its mask;
  - `place_shape`: up to 50 random positions, with a maximum IoU and a minimum gap to the placed boxes;
  - `generate_image` (detection, classification) is rewritten on them, keeping its behaviour.
- **Object-loss pairs:**
  - `generate_object_loss_pair(image_size, number_frames, maximum_objects, random_generator)`;
  - per pair, one background and N ∈ [2, maximum_objects] objects drawn once, with sizes S/8 to S/5;
  - the after member keeps every object or loses one at random (p = 0.5);
  - each member has [2, number_frames] frames, and every object is placed at a random position in every frame;
  - `draw_shapes_frame` places the objects without overlap and with a 2 px gap. It redraws the whole frame when an
    object does not fit (up to 20 times, then a RuntimeError), so every frame shows the exact count.
- **Plumbing:**
  - `PAIR_CATEGORIES = {"loss_of_object": 1, "same_number_of_objects": 2}`;
  - `TASK_MANIFEST_KEYS` (`sequences` or `pairs`) and `TASK_SAMPLE_NAMES` replace the hard-coded "sequences";
  - `generate_split` passes `maximum_objects` to the sequence workers;
  - `save_frames` is shared by every sequence task;
  - `__main__`:
    - output folder `/home/robert_kelevra/Data/sequence_data/synthetic_object_loss_pairs`;
    - 128 px, splits 2000 / 200 / 200;
    - 4 objects at most and 6 frames per member at most.

### 5.6 Deployment (several named inputs)
- **Export specification (`deployment/export_specifications.py`):**
  - the single-input keys (`input_dimensions`, `dynamic_frames`, `maximum_frames`, `example_shape`, `check_shape`)
    become a list in forward-argument order:
    `"inputs": [{"name", "dimensions", "data_type", "dynamic_frames", "maximum_frames", "example_shape", "check_shape"}]`;
  - new registry `TASK_INPUT_NAMES`: `["images"]` for the four existing tasks, `["before_images", "after_images"]`
    for the comparison;
  - new `build_input_specification(...)` and `create_random_inputs(specification, shape_name, maximum_batch_size, device, generator) -> Dict[str, Tensor]`;
  - the check frame counts are `max(1, L - 1 - input_index)`, so the before and after inputs are checked with
    different frame counts;
  - comparison entries in INFERENCE_MODEL_CLASSES, TASK_MODEL_NAMES (`"sequence_comparator"`), SEQUENCE_TASKS and
    TASK_INPUT_NAMES.
- **New `models/sequence/inference_sequence_comparator.py`:** `InferenceSequenceComparator.forward(before_images, after_images)`
  returns `class_probabilities` [B, K], with no masks.
- **`deployment/export_model.py`:**
  - `export_torchscript(inference_model, example_inputs: Dict[str, Tensor], output_path)` traces with the inputs
    passed positionally in specification order. The traced schema keeps the argument names, which Triton's PyTorch
    backend matches;
  - `export_onnx(inference_model, example_inputs, output_path, specification, maximum_batch_size)`:
    - one `Dim("batch")` shared by every input, and one `Dim("<name>_frames")` per sequence input;
    - before each export, clear `torch.ops.aten.gru.input._dispatch_cache` and
      `torch.ops.aten.lstm.input._dispatch_cache`, behind a `hasattr` check. Otherwise a second recurrent export
      in the same process silently writes a fixed frame count;
    - after the export, the new `find_fixed_dynamic_dimensions(model_proto, specification)` raises if a declared
      dynamic dimension came out fixed;
  - `export_experiment`: random named inputs, the eager reference `inference_model(**check_inputs)`, the TorchScript
    rerun with keywords, and ONNX Runtime with a dictionary of named arrays;
  - `hide_known_export_warnings` also hides the shared-batch "axis name" warning and the onnxscript
    constant-folding log.
- **`deployment/triton_configuration.py`:** one input line per input, joined with `",\n"`. With one input the
  output is identical, so the detection `config.pbtxt` stays byte-identical.
- **`deployment/deployment_utilities.py` and `run_inference.py`:**
  - each request is a dictionary of named arrays, sent as one `InferInput` per input;
  - shared memory uses one region with one offset per input;
  - `load_inference_inputs` builds the comparison windows from the same folder, the after window shifted by L
    frames (for throughput measurements), with the helper `build_frame_window`;
  - local and Triton throughput take a list of dictionaries.

### 5.7 Transfer Of The Temporal Module And Head, And `freeze_normalization_parameters`
- **Configuration:**
  - `TemporalModel.pretrained` (sequence tasks) and `ClassificationHead.pretrained` (classification and sequence
    tasks) mirror `Backbone.pretrained`: `{experiment_folder, iteration, freeze, learning_rate_multiplier}`;
  - `Backbone.pretrained.freeze_normalization_parameters` keeps the scale and shift of the backbone's batch or layer
    normalization fixed while its convolutions train; the batch norm statistics keep updating.
- **`models/backbones/backbone_transfer.py` moves to `models/model_transfer.py`** (`git mv`; tests moved to
  `tests/models/test_model_transfer.py`):
  - `TRANSFERABLE_COMPONENTS = {"Backbone": "backbone", "TemporalModel": "temporal_module", "ClassificationHead": "classification_head"}`.
    Its order is the optimizer group order, and the weight prefix is the attribute name;
  - two levels of verification:
    - a configuration comparison per section, with readable messages:
      - backbone: unchanged;
      - temporal module: `type`, then the keys that define its weights for that type, the width of the last
        backbone block, and `sequence_length` for the transformer;
      - head: `fully_connected`, and whether the dropout is zero, which shifts the layer indices;
    - before the strict load, `compare_state_dictionaries` lists missing, unexpected and wrong-shape tensors by name.
      For example, a sequence classifier head loaded into a comparator head, whose input is 2 or 3 × F';
  - a source run without the section is refused;
  - `load_pretrained_component(component, section_name, model_configuration, device) -> Path` replaces
    `load_pretrained_backbone`, and keeps the backbone wording;
  - also `freeze_component`, `freeze_normalization_parameters`, and `disable_pretrained_settings` (used by
    `scripts/evaluation/evaluate_checkpoint.py`);
  - the checks do not import this module, which would create an import cycle.
- **BaseTrainer:**
  - `initialise_pretrained_components()` replaces `initialise_backbone()`, for every section present and transferred,
    with no reload on resume;
  - `pretrained_checkpoint_paths: Dict[str, Optional[Path]]` replaces `pretrained_backbone_path`;
  - frozen components keep only their batch norm in eval;
  - a run where no parameter still learns is refused at startup;
  - optimizer groups: `[rest]`, then one group per transferred component whose multiplier is not 1, in the order
    backbone, temporal module, head. Existing backbone-only runs keep their layout, so resuming still works;
  - checkpoint metadata gains one `pretrained_<attribute>` key per section present; `pretrained_backbone` is
    unchanged;
  - the startup lines list every transferred component.
- **Checks:**
  - `check_pretrained_configuration(model_configuration, section_name)` generalises the backbone check, keeping its
    messages;
  - `freeze_normalization_parameters` must be a boolean, False without a source run, False when `freeze` is True, and
    is refused with `feature_map_normalization: "none"`;
  - a temporal source run is refused for mean and max pooling, which have no weights.
- **Run name:** `build_pretrained_tokens` generalises PT/FRZ/BLR. Existing names are unchanged.
  - backbone: `PT`, then `FRZ` or `BLR<m>`, then `FBN`;
  - temporal module: `TPT`, then `TFRZ` or `TLR<m>`;
  - head: `HPT`, then `HFRZ` or `HLR<m>`.
- **Migration:**
  - `add_pretrained_settings_to_configuration(path, section_name)` replaces
    `add_pretrained_backbone_to_configuration`. It returns False when the section is absent or already has the block;
  - `add_freeze_normalization_parameters_to_configuration(path)` inserts the key after the `    freeze:` line of
    the Backbone `pretrained:` block. This works for blocks at the end of the section and for blocks right below the
    header;
  - both are chained in `migrate_run_folder`;
  - all YAMLs are updated, the comparison ones included.

---

## 6. Phases
Each phase: README checklist, implementation, tests (`uv run pytest -q`, `ruff` name checks), acceptance, README
checklist removed, one commit, push.

| Phase | Content | Commit tag |
|---|---|---|
| 0 | `implementation_comparison.md` (this plan) + `.gitignore` exception | `[DOCS]` |
| 1 | Comparison task: 5.1–5.5, registries, tests, acceptance | `[COMPARISON]` |
| 2 | Deployment with several inputs (5.6), recurrent export fix and guard, tests, Triton acceptance | `[DEPLOYMENT]` |
| 3 | Transfer (5.7): module move, components, checks, run name, trainer, migration, YAMLs, tests, acceptance | `[TRANSFER]` |
| 4 | Implementation notes and acceptance results in `implementation_comparison.md`; items 18 and 20 marked as done in the (untracked) `previous_classifiers_comparison.md` | `[DOCS]` |

## 7. Tests
**Phase 1**
- **New `tests/data/sequence/test_sequence_pair_dataset.py`:**
  - member frame indices (pad, skip, empty, subsampling);
  - the `[2L]` layout, the masks and the frame stride;
  - unmapped or left-out pairs, with the warning;
  - class counts;
  - `count_split_pairs`;
  - the collated batch split into members.
- **New `tests/data/sequence/test_pair_list_manifests.py`:**
  - categories, sorted frames, the data link and the label mapping;
  - re-running replaces the outputs;
  - the converted split loads in `SequencePairDataset`.
- **New `tests/models/sequence/test_member_fusion.py`:**
  - exact `[a, b]` / `[a, b, b - a]` values;
  - cross attention ignores padding keys and zeroes padding queries.
- **New `tests/models/sequence/test_sequence_comparator.py`:**
  - shapes, probabilities and the loss, for every fusion and temporal type;
  - white padding frames change nothing;
  - the mask-free path equals the masked path in float64, with before 3/4 and after 2/4 real frames;
  - in training mode batch norm sees both members;
  - NaN padding stays finite;
  - giving a single mask is refused.
- **New `tests/trainers/test_sequence_comparison_trainer.py`:** a 5-iteration smoke run on generated pairs, checking:
  - `CMP_..._T4_GRU1x128-BI_FXA4_`;
  - the class names;
  - checkpoints at 2, 4 and 5;
  - "Iteration 4 : 6 Test Pairs" and the evaluation files;
  - resuming;
  - `pairs/sec` and `frames/sec`, and no errors.
- **Existing files:**
  - `test_create_synthetic_dataset.py`:
    - exact object counts per frame (connected components), loss vs same;
    - member lengths and the mapping;
    - frames that cannot fit fail loudly;
  - `test_configuration_checker.py`:
    - the task list message (5 tasks) and the new causal message;
    - the CMP YAMLs are consistent;
    - fusion errors;
    - M2O still requires `window_stride`;
  - `test_run_name.py`: CMP names with FCAT, FDIF, FXA4, FXA4-FDO0.1 and T8S2; M2O and M2M unchanged;
  - `test_metric_tracking.py`: pairs/sec.

**Phase 2**
- **`tests/deployment/test_export_specifications.py`:**
  - the dictionary API;
  - comparator GRU and transformer exports match the eager model with different before/after frame counts. The GRU
    case runs after the existing GRU case, so it also tests the recurrent fix;
  - the TorchScript argument names equal the specification names;
  - a double recurrent export keeps dynamic frames;
  - `find_fixed_dynamic_dimensions` reports a model exported with fixed frames;
  - the comparator's mask-free path equals its masked path.
- **`tests/deployment/test_triton_configuration.py`:** the detection files stay byte-identical; the comparator
  configuration has two input lines in forward order.
- **New `tests/deployment/test_deployment_utilities.py`:** the named inputs and the shift of the comparison windows.
- **`tests/core/test_task_registry.py`:** the deployment registries also cover every task.

**Phase 3**
- **`tests/models/test_model_transfer.py`** (moved). The existing tests are kept, plus:
  - temporal module and head between sequence classifiers;
  - temporal module from a sequence classifier into a labeller and into a comparator;
  - refused differences, parameterised: type, hidden size, bidirectional, transformer heads, kernel size, sequence
    length, last backbone width, head dropout numbering;
  - wrong-shape head weights listed by name;
  - a source run without the section;
  - `freeze_normalization_parameters` with batch norm and with layer norm;
  - `disable_pretrained_settings`.
- **`tests/trainers/test_backbone_transfer_training.py`:**
  - the existing test, with `pretrained_checkpoint_paths`;
  - a sequence run with a frozen GRU temporal module, the head at 0.1 and the backbone at 0.5, checking:
    - `TPT-TFRZ` and `HPT-HLR0.1`;
    - the groups `[rest, backbone, head]` with learning rates `[1e-3, 5e-4, 1e-4]`;
    - the GRU stays fixed while the backbone learns, on CUDA, which catches the cuDNN eval-mode crash;
    - the checkpoint metadata;
    - resuming without reloading;
  - `-PT-FBN`;
  - a model with every component frozen is refused.
- **Configuration tests:**
  - run name tokens;
  - the new checks;
  - migration of the 5-key backbone block, `freeze_normalization_parameters` in both layouts, the temporal and head
    blocks, detection files untouched, and a full `migrate_run_folder` on an old sequence run;
  - `tests/fixtures/dummy_task.py` gains the new key.

## 8. Acceptance
- **Phase 1:**
  - train on the synthetic pairs (128 px, L = 8, mean pooling) with each fusion;
  - target ≥ 95 % test accuracy, reporting the first checkpoint that reaches it;
  - a single member caps at 0.5 + 0.5/(M − 1) = 66.7 % for M = 4, so reaching the target shows the model uses
    both members.
- **Phase 2:** the TorchScript and ONNX exports of a trained comparison run match the eager model with different
  before/after frame counts, load in Triton (`deployment/triton/docker-compose.yml`), and `run_inference.py`
  measures their local and Triton throughput.
- **Phase 3:**
  - train a synthetic sequence classifier (moving shapes, GRU1x128-BI, same backbone as the synthetic pairs);
  - start a comparison run (GRU1x128-BI) from its backbone and temporal module;
  - the loaded weights must equal the source weights, and the training curves are compared with training from
    scratch, as in phase 8 of `major_refactoring.md`.

## 9. Risks And Open Points
- The recurrent export fix relies on a private PyTorch attribute (`_dispatch_cache`). The post-export guard turns
  any change after a PyTorch upgrade into a clear error.
- A frozen component with a multiplier other than 1 still gets its own optimizer group, while the run name shows only
  FRZ. This is the existing backbone behaviour, kept for compatibility.
- The checks cannot see: the same number of classes in a different order, or causal vs non-causal differences
  (a warning only). A temporal module transferred into the comparator with cross attention receives fused frame
  features.
- Older classification and sequence runs must be migrated (`scripts/configuration/migrate_configuration.py`) before
  they are evaluated or exported.
- The photometric draws are shared by both members, and long members are always subsampled the same way. Independent
  draws and random sampling could be later options.
- The converter orders frames by file name, so frame names need zero padding for the ordered temporal modules.
- Triton batches comparison requests only when both frame counts match, and TensorRT may build an engine per shape
  pair (the engine cache helps).

## 10. Verification
- `uv run pytest -q` passes after every phase.
- The M2O and M2M run names, the existing run names and the detection Triton configurations are unchanged.
- `NO_COLOR=1 uvx ruff check --no-cache --select F401,F811,F821,F841 vision_training tests scripts`.
- The acceptance trainings and exports of section 8, with the results recorded in `implementation_comparison.md`.

---

## 11. Implementation Notes

Every phase is implemented, one commit per phase (`[COMPARISON]`, `[DEPLOYMENT]`, `[TRANSFER]`). This section lists
what differs from the plan above, what is left out, and the acceptance results.

### Differences from the plan
1. **Synthetic detection and classification images:** `generate_image` is rewritten on the shape helpers shared with
   the object loss pairs. It draws the size and the colour of an object once and only retries its position (the
   previous code drew a new size at every attempt), so the same seed no longer gives the same images as before.
2. **Exported frame dimensions:** every sequence input has its own `Dim("<input name>_frames")`, so the ONNX exports
   of the sequence classifier and labeller name their frame dimension `images_frames` instead of `frames`. Triton
   does not read these names.
3. **Comparison of the heads:** the dropout of two heads is only compared when they have hidden layers, a head without
   hidden layers numbering its layers the same with or without dropout.
4. **Startup lines:** the transferred components are printed as `Pretrained Backbone`, `Pretrained Temp. Module` and
   `Pretrained Class. Head`, which keeps the column of the experiment information aligned.

### Left out
- Error dumps of the misclassified pairs, left to item 2 of the comparison document for every task.
- Independent photometric draws for the two members of a pair, and random frame sampling of the long members (see
  section 9): both members share every draw, and a long member is always subsampled the same way.

### Migration of the runs trained before
`scripts/configuration/migrate_configuration.py` now also adds the `pretrained` settings of `TemporalModel` and
`ClassificationHead` and `Backbone.pretrained.freeze_normalization_parameters`, all off. The classification and
sequence runs trained before phase 3 must be migrated before they are resumed, evaluated or exported. Their folder
keeps its name, the settings of a run without transfer adding no run name token.

### Acceptance results (synthetic datasets)
| Phase | Check | Result |
|---|---|---|
| 1 | Object loss pairs (128 px, 8 frames per member, mean pooling), concatenation / difference / cross attention; one member alone caps at 66.7 % | 97.0 / 98.5 / 96.0 % test accuracy at iteration 500 (target 95 %), 100 / 100 / 99.0 % at their best, 100 / 100 / 94.5 % at the last checkpoint (iteration 3000, constant learning rate) |
| 2 | Exports of the cross attention comparator | TorchScript and ONNX match the eager model with 7 before and 6 after frames, load in Triton with two inputs of dynamic frames, and the served outputs match the local model, also with 8 before and 5 after frames: TorchScript to 3e-7 against full FP32, ONNX with TensorRT within the TF32 noise (1.6e-3 at most) |
| 2 | Throughput, one pair of 2 x 8 frames of 128 px per request | local 407 pairs/s, Triton TorchScript 131 pairs/s (157 with shared memory), ONNX with TensorRT 183 pairs/s (241 with shared memory) |
| 2 | Second recurrent ONNX export in one process | fixed numbers of frames without the reset of the dispatch caches (caught by the guard), dynamic with it |
| 3 | Backbone and GRU1x128-BI of a trained synthetic sequence classifier (98.0 % test accuracy) loaded into a comparator | the 89 transferred tensors equal those of the source checkpoint |
| 3 | Frozen GRU temporal module trained on the GPU | stays fixed while the backbone before it learns (test); putting the whole frozen module in evaluation mode fails at every step with "cudnn RNN backward can only be called in training mode" |
| 3 | Comparator from the transferred weights vs from scratch | validation accuracy slightly ahead after the first 100 iterations (98.75 vs 96.25 % at iteration 400, 100 vs 98.75 % at 700), test accuracy comparable: 97.5 / 98.0 / 97.0 / 99.0 / 99.0 / 100 % vs 99.5 / 98.5 / 99.0 / 99.5 / 99.0 / 100 % at iterations 500 to 3000. The comparison task learns within 500 iterations from scratch, which leaves little to gain on this dataset |

### Observations
- **Plateau of the synthetic moving shapes:** the sequence classifier first trained as the transfer source never left
  the chance level in 3000 iterations (21 to 28 % test accuracy, training loss at ln 4). A second run of the same
  configuration left it around iteration 700 (98.5 % at iteration 1000), while the code before phase 1, run at the
  same time, stayed on it for 1500 iterations. The runs draw different random initialisations (no seed is set), so
  this plateau, and not a regression, explains the difference with the 100 % at iteration 1500 of phase 6 of
  `major_refactoring.md`. The transfer acceptance uses the run that learned.
- **Transfer from an untrained source:** a comparator initialised by the run stuck on the plateau reached 96.5 % test
  accuracy at iteration 500 (99.5 % from scratch) and 100 % at iteration 1000.
