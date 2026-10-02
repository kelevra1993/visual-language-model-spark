# Major Refactoring: From An Object Detector To A General Vision Training Framework

## Table Of Contents
1. Context And Goals
2. What Exists Today
3. Design Principles
4. Target Repository Tree
5. File Migration Map
6. Shared Contracts (Batches, Model Outputs, Evaluation Results)
7. The General Trainer (`BaseTrainer`)
8. Task Implementations
9. Models
10. Data Layer
11. Losses
12. Evaluation
13. Configuration
14. Backbone Transfer
15. Deployment
16. Scripts And Tests
17. Migration Phases
18. Backward Compatibility
19. Risks
20. Decisions Taken

---

## 1. Context And Goals

The repository trains one model type today: a two-stage object detector (backbone, per-level region proposal
networks, RoIAlign detection heads), with its own data loaders, evaluation (mAP), export (TorchScript / ONNX /
Triton) and configuration checker. The trainer was written around dictionaries (batch dictionary, model output
dictionary with `*_loss` keys, `tracked_metrics_mapping`, metric trackers), so most of it already does not depend on
the task.

**Goal:** turn the repository into a general framework for training vision networks, where adding a task means
writing the task-specific pieces only. Targets:

| Task | Input | Target | Output |
|---|---|---|---|
| Object detection (existing) | one image `[B, C, H, W]` | boxes + labels per image | boxes, scores, labels |
| Classification, one to one | one image `[B, C, H, W]` | one label per image | class probabilities `[B, K]` |
| Sequence classification, many to one | a sequence of images `[B, T, C, H, W]` | one label per sequence | class probabilities `[B, K]` |
| Sequence labelling, many to many | a sequence of images `[B, T, C, H, W]` | one label per image of the sequence | class probabilities `[B, T, K]` |

**Secondary goal:** reuse a backbone trained in one run to initialise another run (for example a backbone trained
for classification used to start a detection training).

**Non-goals of this refactoring:** changing the detector's behaviour or accuracy, new detection architectures (FPN,
one-stage detectors), mixed precision and `torch.compile` (listed as optional later work).

---

## 2. What Exists Today

### Already generic (moves to the general trainer almost unchanged)
- `Trainer.run_training_loop`: iteration loop, save / intermediate save schedule, validation gating with
  `validation_iteration_ratio`, consecutive error handling, Ctrl+C handling, final save.
- `Trainer.get_next_batch`, `initialize_training_loop_state`, `run_benchmarking_loop`.
- Metric tracking: `get_metric_trackers`, `console_log_update_tracker`, `log_metrics_to_tensorboard`, driven by
  `tracked_metrics_mapping`.
- Checkpointing: `save_model` (atomic, Ctrl+C-safe), `dump_in_checkpoint`, `extract_last_model_iteration`,
  `restore_last_model`, `load_model`.
- Run folders: `setup_training_paths`, `build_run_name` (`utilities/os_utilities.py`), `print_experiment_information`.
- The loss aggregation rule of `run_model_iteration`: every model output key containing `_loss` is summed.
- `DataAugmenter` (`data/augmentations.py`): already supports images without boxes (classification) and sequences
  `[B, T, C, H, W]` with one shared random augmentation per sequence (tested in `tests/data/test_augmentations.py`).
- `Backbone` (`architecture_modules/backbone.py`) and its building blocks (`ConvolutionBlock`, squeeze-and-excitation,
  inception, residual shortcuts, input normalization): `forward` returns the final feature map **and** the
  per-level feature maps, so a classifier can use the first and a detector the second.
- `print_model_size_summary`, `print_receptive_field_summary`, `format_time_interval`, console helpers.

### Detection-specific (becomes the detection implementation)
- `Trainer.prepare_input_data` (boxes and labels lists), the forward call with ground truth in
  `run_model_iteration`, `run_test_evaluation` (forward without ground truth, score threshold swap, DALI duplicate
  removal, mAP), `report_test_evaluation`, `view_training_and_validation_images` (draws boxes), the hard-coded
  `tracked_metrics_mapping`.
- `model/model.py` (966 lines): detector graph, target assignment calls, loss computation, post-processing, summaries.
- `model/inference_model.py` (`InferenceDetector`), `utilities/model/model_utilities.py` (box operations, target
  assignment, sampling, post-processing, drawing).
- `data/data_loader.py` (COCO `NativeDataset`, DALI COCO pipeline), `utilities/data_utilities.py` (box preprocessing).
- `evaluation/mean_average_precision.py`, `deployment/export_model.py` (4 detector outputs, detection comparison).
- `utilities/model/configuration_checker.py`: one file mixing common checks (experiment, data, backbone) with
  detection checks (levels, anchors, region proposal, detector, sampling).
- `build_run_name`: common sections (input, backbone, training) mixed with detection sections (levels, anchors, heads).

### Structural problems to fix on the way
- The package folder is named `object_detection/`, and configurations, tests and scripts assume detection.
- Mixed import styles: most files use bare imports (`from model.model import Model`, resolved through
  `pythonpath = [".", "object_detection"]`), 13 test imports use `from object_detection....`.
- `tests/model/test_inference_model.py` imports its configuration from `scripts/utilities/debugging_utilities.py`
  (test support code living in debug scripts).
- No test covers the trainer itself (loop, checkpointing, resume).
- `configuration_checker.py` lives in `utilities/model/` although it validates the whole file.

---

## 3. Design Principles

- **Template method:** a `BaseTrainer` owns everything that never changes (loop, logging, checkpoints, evaluation
  reporting). Tasks subclass it and override a small, documented set of hooks. Hooks that every task must provide are
  `@abstractmethod`; optional hooks have a default implementation in `BaseTrainer`.
- **Composition for the reusable parts:** losses, metrics, heads, temporal modules, datasets and export specifications
  are plain functions / modules, so a task class stays short and the hierarchy stays shallow (at most two levels
  below `BaseTrainer`).
- **Dictionaries as contracts:** keep the existing dictionary approach, but document every shared key once
  (`core/contracts.py`, with `TypedDict` definitions for type hints) instead of relying on implicit names.
- **One backbone class for every task**, always stored under the attribute name `backbone`, so weights move between
  tasks with a prefix filter on the state dictionary.
- **Behaviour first, then generality:** the first phases move and split code without changing a single number;
  detection outputs, losses and checkpoints must stay identical before any new task is added.
- **Existing rules stay in force:** strict configuration access with `["key"]`, required configuration arguments,
  keyword arguments in every call, type hints and "Args" docstrings, `file://` links in console output, extending an
  existing function instead of writing a parallel one, commits by feature without attribution.

---

## 4. Target Repository Tree

The package is renamed from `object_detection/` to `vision_training/` (decided, see section 20). Tests and scripts
live at the repository root and mirror the package tree.

```
visual-language-model-spark/
├── pyproject.toml                       # pytest pythonpath = ["."], package-qualified imports
├── uv.lock
├── README.md
├── suggestions.md
├── major_refactoring.md
├── README/                              # images used by README.md
├── vision_training/
│   ├── main.py                          # loads the configuration, picks the trainer of its task, trains
│   ├── core/
│   │   ├── contracts.py                 # documented batch / output / evaluation keys and TypedDicts
│   │   └── task_registry.py             # task name → trainer class, model builder, configuration checks
│   ├── configurations/
│   │   ├── object_detection/
│   │   │   ├── configuration_example.yaml
│   │   │   └── configuration_synthetic_shapes.yaml
│   │   ├── classification/
│   │   │   ├── configuration_example.yaml
│   │   │   └── configuration_synthetic_shapes.yaml
│   │   ├── sequence_classification/
│   │   │   ├── configuration_example.yaml
│   │   │   └── configuration_synthetic_sequences.yaml
│   │   └── sequence_labelling/
│   │       ├── configuration_example.yaml
│   │       └── configuration_synthetic_sequences.yaml
│   ├── configuration/
│   │   ├── configuration_loader.py      # load_configuration, load_experiment_configuration
│   │   ├── configuration_values.py      # ConfigurationError, get_configuration_value, check_* helpers
│   │   ├── common_checks.py             # Experiment, Data, Backbone, Optimization sections
│   │   ├── detection_checks.py          # levels, anchors, region proposal, detector, sampling
│   │   ├── classification_checks.py     # classification head
│   │   ├── sequence_checks.py           # sequence settings, temporal model
│   │   └── run_name.py                  # build_run_name (common sections + task sections)
│   ├── trainers/
│   │   ├── base_trainer.py              # BaseTrainer: everything that does not depend on the task
│   │   ├── checkpointing.py             # save / load / registry / resume, used by BaseTrainer
│   │   ├── metric_tracking.py           # trackers, console log, TensorBoard scalars, used by BaseTrainer
│   │   ├── detection_trainer.py         # DetectionTrainer(BaseTrainer)
│   │   ├── classification_trainer.py    # ClassificationTrainer(BaseTrainer)
│   │   ├── sequence_classification_trainer.py   # SequenceClassificationTrainer(ClassificationTrainer)
│   │   └── sequence_labelling_trainer.py        # SequenceLabellingTrainer(SequenceClassificationTrainer)
│   ├── models/
│   │   ├── task_model.py                # TaskModel(nn.Module, ABC): forward contract, summaries, size report
│   │   ├── model_loading.py             # get_checkpoint_path, load_trained_model (any task, via the registry)
│   │   ├── components/                  # task-independent building blocks
│   │   │   ├── convolution_block.py
│   │   │   ├── squeeze_and_excitation.py
│   │   │   ├── inception.py
│   │   │   ├── input_normalization.py
│   │   │   └── fully_connected_block.py # extracted from DetectionHead, shared with classification heads
│   │   ├── backbones/
│   │   │   ├── backbone.py
│   │   │   └── backbone_transfer.py     # loads a trained backbone from another run
│   │   ├── detection/
│   │   │   ├── anchors.py
│   │   │   ├── region_proposer.py
│   │   │   ├── detection_head.py
│   │   │   ├── box_operations.py        # area, IoU, clamp, regression decoding, box dimensions
│   │   │   ├── target_assignment.py     # anchor / proposal targets, sampling, transformation targets
│   │   │   ├── postprocessing.py        # postprocess_detections
│   │   │   ├── two_stage_detector.py    # the current Model, renamed TwoStageDetector
│   │   │   └── inference_detector.py    # InferenceDetector (export path)
│   │   ├── classification/
│   │   │   ├── classification_head.py   # global pooling + fully connected block + logits
│   │   │   ├── image_classifier.py      # Backbone + ClassificationHead
│   │   │   └── inference_classifier.py  # probabilities output for export
│   │   └── sequence/
│   │       ├── frame_encoder.py         # Backbone applied to every frame → one feature vector per frame
│   │       ├── temporal_modules.py      # pooling, attention pooling, GRU / LSTM, transformer, temporal convolution
│   │       ├── sequence_classifier.py   # many to one
│   │       ├── sequence_labeller.py     # many to many
│   │       └── inference_sequence_model.py
│   ├── losses/
│   │   ├── detection_losses.py          # region proposal + detection head losses (moved out of the model)
│   │   ├── classification_losses.py     # cross entropy (label smoothing, class weights), multi-label BCE
│   │   └── sequence_losses.py           # cross entropy over the valid frames only
│   ├── data/
│   │   ├── loaders.py                   # get_dataloaders: dispatch by task and loader backend
│   │   ├── split_description.py         # images / samples / sequences / annotations per split (startup print)
│   │   ├── label_mapping.py             # binary / multi-class label dictionaries, per task
│   │   ├── preprocessing.py             # resize + padding, box preprocessing and restoration
│   │   ├── augmentations.py             # DataAugmenter (unchanged)
│   │   ├── data_retriever.py
│   │   ├── detection/
│   │   │   ├── coco_dataset.py          # NativeDataset → CocoDetectionDataset + collate
│   │   │   └── coco_dali_pipeline.py    # dali_pipeline + DALIDataloaderWrapper
│   │   ├── classification/
│   │   │   ├── classification_dataset.py
│   │   │   └── classification_dali_pipeline.py   # later phase
│   │   ├── sequence/
│   │   │   ├── sequence_dataset.py      # frames, padding, masks, sliding windows
│   │   │   └── sequence_collate.py
│   │   └── synthetic/
│   │       └── create_synthetic_dataset.py      # extended with classification and sequence modes
│   ├── evaluation/
│   │   ├── evaluation_report.py         # lines + scalars + files → console, Tensorboard/Test, Results.txt
│   │   ├── detection/
│   │   │   └── mean_average_precision.py
│   │   ├── classification/
│   │   │   ├── classification_metrics.py   # accuracy, per-class precision / recall / F1, top-k
│   │   │   ├── confusion_matrix.py         # matrix + PNG / CSV
│   │   │   └── threshold_finder.py         # per-class threshold for a target precision
│   │   └── sequence/
│   │       └── sequence_metrics.py      # frame accuracy, sequence exact match, segment metrics
│   ├── deployment/
│   │   ├── export_model.py              # generic export driver
│   │   ├── export_specifications.py     # per-task inputs / outputs / example inputs / comparison
│   │   ├── triton_configuration.py      # config.pbtxt from an export specification
│   │   ├── deployment_utilities.py
│   │   ├── run_inference.py
│   │   ├── inference_data/
│   │   └── triton/
│   │       └── docker-compose.yml
│   └── utilities/
│       ├── os_utilities.py              # read_json, get_image_paths, print_color, print_dictionary
│       ├── tensor_utilities.py
│       └── visualization.py             # add_bounding_boxes, visualise_anchors, view_input_data, frame strips
├── scripts/                             # mirrors vision_training/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   │   ├── components/
│   │   ├── detection/
│   │   └── sequence/
│   └── deployment/
└── tests/                               # mirrors vision_training/
    ├── fixtures/
    │   ├── configuration_fixtures.py    # small configurations of every task (replaces get_model_configuration)
    │   └── dummy_task.py                # tiny CPU task used to test BaseTrainer on its own
    ├── configuration/
    ├── data/
    ├── evaluation/
    ├── losses/
    ├── models/
    │   ├── components/
    │   │   └── reference_values/        # the 28 existing .pt files (moved only with your confirmation)
    │   ├── detection/
    │   ├── classification/
    │   └── sequence/
    ├── trainers/
    └── testing_utilities.py
```

---

## 5. File Migration Map

| Current file | New location | Notes |
|---|---|---|
| `main.py` | `vision_training/main.py` | picks the trainer class of `ExperimentConfiguration.task` from the registry |
| `trainer/trainer.py` | `trainers/base_trainer.py`, `trainers/checkpointing.py`, `trainers/metric_tracking.py`, `trainers/detection_trainer.py`, `evaluation/evaluation_report.py` | generic parts vs detection parts, see section 7 |
| `model/model.py` | `models/detection/two_stage_detector.py` + `losses/detection_losses.py` | class `Model` renamed `TwoStageDetector`; attribute names kept so checkpoints still load |
| `model/inference_model.py` | `models/detection/inference_detector.py` + `models/model_loading.py` | `get_checkpoint_path` / `load_trained_model` become task-independent |
| `architecture_modules/backbone.py` | `models/backbones/backbone.py` | unchanged |
| `architecture_modules/convolution_block.py`, `squeeze_and_excitation.py`, `inception.py`, `input_normalization.py` | `models/components/` | unchanged |
| `architecture_modules/anchors.py`, `region_proposer.py`, `detector.py` | `models/detection/` | `detector.py` → `detection_head.py`; `_build_fully_connected_block` moves to `components/fully_connected_block.py` |
| `utilities/model/model_utilities.py` | `models/detection/box_operations.py`, `target_assignment.py`, `postprocessing.py`, `utilities/visualization.py` | split by role |
| `utilities/model/configuration_checker.py` | `configuration/configuration_values.py`, `common_checks.py`, `detection_checks.py` | per-task checks registered in the task registry |
| `utilities/os_utilities.py` | `configuration/configuration_loader.py`, `configuration/run_name.py`, `data/label_mapping.py`, `utilities/os_utilities.py` | console and file helpers stay |
| `utilities/data_utilities.py` | `data/preprocessing.py` + `utilities/visualization.py` | `view_input_data` extended for captions and sequences |
| `utilities/tensor_utilities.py` | `utilities/tensor_utilities.py` | unchanged |
| `data/data_loader.py` | `data/detection/coco_dataset.py`, `data/detection/coco_dali_pipeline.py`, `data/loaders.py`, `data/split_description.py` | `count_split_images_and_annotations` generalised per task |
| `data/augmentations.py` | `data/augmentations.py` | unchanged |
| `data/create_synthetic_dataset.py` | `data/synthetic/create_synthetic_dataset.py` | extended with classification / sequence modes |
| `data/data_retriever.py` | `data/data_retriever.py` | unchanged |
| `evaluation/mean_average_precision.py` | `evaluation/detection/mean_average_precision.py` | unchanged |
| `deployment/export_model.py` | `deployment/export_model.py`, `export_specifications.py`, `triton_configuration.py` | detection comparison becomes the detection specification |
| `deployment/deployment_utilities.py`, `run_inference.py`, `inference_data/`, `triton/` | same names under `vision_training/deployment/` | throughput helpers take any inference model |
| `configurations/*.yaml` | `configurations/object_detection/` | plus the `task` key (section 13) |
| `scripts/utilities/debugging_utilities.py` | `scripts/models/detection/detection_debugging_utilities.py` + `tests/fixtures/configuration_fixtures.py` | test support leaves `scripts/` |
| other `scripts/*` | `scripts/` mirroring the new tree | |
| `tests/*` | `tests/` mirroring the new tree | `.pt` files: see section 16 |
| `.gitignore` (`object_detection/deployment/triton/served_models/`) | path updated | |

---

## 6. Shared Contracts (Batches, Model Outputs, Evaluation Results)

`core/contracts.py` documents the keys and defines one `TypedDict` per batch type, used in the type hints of the hooks.

### Batch dictionary (produced by every loader, moved to the device by `prepare_batch`)

| Key | Detection | Classification | Many to one | Many to many |
|---|---|---|---|---|
| `images` | `[B, C, H, W]` | `[B, C, H, W]` | `[B, T, C, H, W]` | `[B, T, C, H, W]` |
| `sequence_mask` | — | — | `[B, T]` bool, True for real frames | `[B, T]` bool |
| `bounding_boxes` | list of `[N_i, 4]` | — | — | — |
| `labels` | list of `[N_i]` | `[B]` (or `[B, K]` multi-label) | `[B]` | `[B, T]`, padded frames = `-100` |
| `sample_identifiers` | `[B]` | `[B]` | `[B]` | `[B]` |

- Pixel values stay float BGR in `[0, 255]`; normalization happens inside the backbone (`InputNormalizer`), as today.
- `sample_identifiers` generalises the DALI `image_ids`, so `BaseTrainer` removes the duplicated samples that the
  DALI loaders add to their last batch, for every task.
- `-100` is PyTorch's default `ignore_index`, so padded frames never contribute to the many-to-many loss.

### Model output dictionary
- Every key ending in `_loss` is a scalar loss; `BaseTrainer` sums them into `total_loss` (current rule, kept).
- Prediction keys are task-defined and documented in the task model: `final_detection_*` (detection),
  `class_logits` / `class_probabilities` (classification, many to one), `frame_logits` / `frame_probabilities`
  (many to many).

### Evaluation results (returned by the task, reported by `BaseTrainer`)
```python
class EvaluationResults(TypedDict):
    lines: List[str]              # printed and appended to Results.txt
    scalars: Dict[str, float]     # written to Tensorboard/Test, e.g. "mAP@0.5/area", "Accuracy", "F1/circle"
    files: Dict[str, Path]        # extra outputs saved by the task (confusion matrix PNG, predictions JSON)
```

---

## 7. The General Trainer (`BaseTrainer`)

### Responsibilities kept in `BaseTrainer` (written once)
- Reading the common configuration (device, dtype, iterations, saving schedule, validation ratio, batch size,
  diagnosis modes), building the run name and the run folder, copying the configuration file.
- Building loaders, model and optimizer **through hooks**, printing the experiment information, diagnosis modes,
  resuming.
- `run_training_loop`, `run_benchmarking_loop`, `get_next_batch`, `initialize_training_loop_state`.
- `run_model_iteration`: the generic step (prepare → augment → forward → sum losses → batch metrics → trackers →
  TensorBoard).
- `run_test_evaluation`: the generic evaluation loop (eval mode, `torch.no_grad`, evaluation context, duplicate
  sample removal, accumulation, results) and `report_evaluation` (console, `Tensorboard/Test`, `Results.txt`).
- Checkpointing and resume (delegated to `trainers/checkpointing.py`), metric tracking and console log (delegated
  to `trainers/metric_tracking.py`).
- `view_input_batches`: the generic viewer loop (zips the training and validation loaders, augments the training
  batch through the hook, delegates the drawing of one sample to the task).

### Hooks

| Hook | Kind | Detection | Classification | Many to one | Many to many |
|---|---|---|---|---|---|
| `build_model()` | abstract | `TwoStageDetector` | `ImageClassifier` | `SequenceClassifier` | `SequenceLabeller` |
| `build_dataloaders()` | abstract | COCO (Native / DALI) | classification manifest | sequence manifest | sequence manifest |
| `get_tracked_metrics()` | abstract | 4 losses + total | loss, total, accuracy | same | loss, total, frame accuracy |
| `prepare_batch(batch)` | abstract | images, boxes, labels | images, labels | + `sequence_mask` | + `sequence_mask` |
| `forward_pass(batch)` | abstract | forward with ground truth | `model(images, labels)` | `model(images, sequence_mask, labels)` | same |
| `predict_batch(batch)` | abstract | forward **without** ground truth | probabilities | probabilities | per-frame probabilities |
| `accumulate_evaluation_batch(...)` | abstract | detections + ground truth lists | probabilities + labels | same | valid frames only |
| `compute_evaluation_results(accumulator)` | abstract | mAP (3 methods) | accuracy, P/R/F1, confusion | same | frame + sequence metrics |
| `draw_input_sample(batch, sample_index)` | abstract | boxes + class names | caption with the class | frame strip + caption | frame strip + per-frame captions |
| `describe_datasets()` | abstract | images + annotations | samples + class counts | sequences + frames | sequences + frames |
| `get_task_run_name_sections()` | abstract | levels, anchors, heads, classes | head, classes | temporal module, head, classes | same |
| `augment_batch(batch)` | default: images only | images + boxes + labels | default | default (sequence-consistent) | default |
| `compute_batch_metrics(batch, outputs)` | default: `{}` | default | accuracy from the training logits | same | frame accuracy |
| `evaluation_context()` | default: no-op | swaps the detection score threshold | default | default | default |
| `build_optimizer()` | default: Adam | default | default | default | default |
| `set_training_mode()` | default: `model.train()`, frozen backbone kept in eval mode | default | default | default | default |

Classification-like tasks share most of their code, hence the hierarchy:
```
BaseTrainer
├── DetectionTrainer
└── ClassificationTrainer
    └── SequenceClassificationTrainer          (many to one: adds sequence masks, frame strips)
        └── SequenceLabellingTrainer           (many to many: per-frame labels, losses and metrics)
```

### Skeleton (signatures and the generic step only)
```python
class BaseTrainer(ABC):
    """
    Generic training pipeline shared by every task: run folder, loop, validation, logging, checkpoints, resume and
    evaluation reporting. A task subclasses it and implements the hooks that depend on its data and model.
    """

    def __init__(self, experiment_configuration: Dict[str, Any], model_configuration: Dict[str, Any],
                 configuration_path: Path) -> None: ...

    # Task hooks
    @abstractmethod
    def build_model(self) -> TaskModel: ...
    @abstractmethod
    def build_dataloaders(self) -> Tuple[Iterable, Iterable, Iterable]: ...
    @abstractmethod
    def get_tracked_metrics(self) -> Dict[str, str]: ...
    @abstractmethod
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]: ...
    @abstractmethod
    def forward_pass(self, batch: Dict[str, Any]) -> Dict[str, Any]: ...
    @abstractmethod
    def predict_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]: ...
    # ... accumulate_evaluation_batch, compute_evaluation_results, draw_input_sample, describe_datasets,
    #     get_task_run_name_sections

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["images"], _, _ = self.data_augmenter.augment_batch(images=batch["images"])
        return batch

    def compute_batch_metrics(self, batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {}

    # Generic step, identical for every task
    def run_model_iteration(self, batch: Dict[str, Any], writer: Optional[SummaryWriter], iteration: int,
                            tracker_dictionary: Optional[Dict[str, Any]], apply_augmentation: bool,
                            tracker_divisor: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch = self.prepare_batch(batch=batch)
        if apply_augmentation:
            batch = self.augment_batch(batch=batch)
        model_outputs = self.forward_pass(batch=batch)

        # Sum every loss of the model, then add the task's batch metrics for the trackers and TensorBoard
        metric_dictionary = {key: value for key, value in model_outputs.items() if key.endswith("_loss")}
        total_loss = sum(metric_dictionary.values())
        metric_dictionary["total_loss"] = total_loss
        metric_dictionary.update(self.compute_batch_metrics(batch=batch, model_outputs=model_outputs))
        self.metric_tracker.update(tracker_dictionary=tracker_dictionary, metric_dictionary=metric_dictionary,
                                   tracker_divisor=tracker_divisor)
        self.metric_tracker.log_to_tensorboard(writer=writer, iteration=iteration,
                                               metric_dictionary=metric_dictionary)
        return total_loss, model_outputs
```

### Small generic improvements made while extracting
- Throughput printed in samples per second, plus frames per second for sequence tasks.
- Checkpoints also store `task`, `run_name` and `format_version` (section 18).
- `print_model_size_summary` moves to `TaskModel` and reads the components from `get_size_components()`, so every
  model gets it without copying the code.

---

## 8. Task Implementations

### 8.1 Object detection (existing, behaviour unchanged)
- `DetectionTrainer` contains: `prepare_input_data` → `prepare_batch`, the ground-truth forward → `forward_pass`,
  the evaluation loop body → `predict_batch` / `accumulate_evaluation_batch` / `compute_evaluation_results`,
  `report_test_evaluation`'s text → `EvaluationResults`, the box viewer → `draw_input_sample`, the score threshold
  swap → `evaluation_context`.
- `TwoStageDetector` keeps its attributes (`backbone`, `region_proposer_dictionary`, `detector_heads_dictionary`,
  `region_proposal_filter`) so existing checkpoints load without remapping.
- `_compute_region_proposal_losses` / `_compute_detection_losses` become functions of `losses/detection_losses.py`
  called by the model.

### 8.2 Classification, one to one
- **Model:** `ImageClassifier` = `Backbone` (final feature map) → `ClassificationHead` (global average or max pooling
  → `FullyConnectedBlock` with dropout → logits `[B, K]`). Outputs `class_logits`, `class_probabilities` and, with
  labels, `classification_loss`.
- **Classes:** no background class; labels are mapped to `0 … K-1`. Multi-class (softmax) by default, multi-label
  (sigmoid + BCE, `labels` `[B, K]`) as a configuration option.
- **Batch metric:** accuracy from the training logits (free, no extra forward pass).
- **Evaluation:** accuracy, balanced accuracy, top-k accuracy, per-class precision / recall / F1, confusion matrix
  (PNG + CSV in `<run>/Evaluation/Iteration_<n>/`), per-class threshold for a target precision, optional error dumps
  and predictions JSON (suggestions A2–A5).

### 8.3 Sequence classification, many to one
- **Model:** `FrameEncoder` reshapes `[B, T, C, H, W]` to `[B·T, C, H, W]`, runs the shared `Backbone`, pools every
  frame to a vector and reshapes back to `[B, T, F]`. A temporal module turns `[B, T, F]` into `[B, F']` while
  ignoring padded frames (`sequence_mask`), then the `ClassificationHead` gives `[B, K]`.
- **Temporal modules** (`models/sequence/temporal_modules.py`): masked mean pooling, masked max pooling, attention
  pooling, GRU / LSTM (last valid state, optional bidirectional), transformer encoder (learned positional encoding,
  masked mean or class token), temporal convolution.
- **Batch size** counts sequences: one step processes `B × T` images, so the image size of sequence configurations is
  usually smaller (e.g. 224 or 256).
- **Evaluation:** the classification metrics, at sequence level.

### 8.4 Sequence labelling, many to many
- **Model:** `FrameEncoder` → temporal module that keeps the time axis (`[B, T, F']`): GRU / LSTM (bidirectional for
  offline use, unidirectional when `causal: True` for online use), transformer encoder (causal attention mask
  option), temporal convolution (causal padding option) → `ClassificationHead` applied to every frame → `[B, T, K]`.
- **Loss:** cross entropy over the valid frames only (`ignore_index = -100` on padded frames).
- **Evaluation:** frame accuracy, per-class precision / recall / F1 over valid frames, frame confusion matrix,
  sequence exact match (every frame right), optional segment metrics (edit score, segment F1 at IoU thresholds) for
  long sequences.

---

## 9. Models

- **`TaskModel` base class** (`models/task_model.py`): `forward(...) -> Dict[str, Any]` following the output contract,
  `print_summary()`, `get_size_components() -> Dict[str, nn.Module]`, and the shared `print_model_size_summary()`.
- **Components:** the current building blocks unchanged, plus `FullyConnectedBlock` extracted from
  `DetectionHead._build_fully_connected_block` and reused by the classification head (extend, do not duplicate).
- **Backbone:** unchanged class. For classification and sequence tasks `detection_convolution_block_indices` is not
  used; the backbone gets an empty list there, and its per-level dictionary is simply empty.
- **Receptive field summary:** the generic part (per-layer receptive field) moves to `Backbone`; the detector keeps
  the overlay of its anchor scales.

---

## 10. Data Layer

### Split folder convention (kept for every task)
Every split stays `<split>/data/…` + `<split>/labels.json`, so the startup description, the loaders and the data
retriever share one convention.

### Label files

**Detection:** COCO, unchanged.

**Classification** (`labels.json`):
```json
{"categories": [{"id": 1, "name": "square"}, {"id": 2, "name": "circle"}],
 "samples": [{"id": 1, "file_name": "image_000001.png", "category_id": 2},
             {"id": 2, "file_name": "image_000002.png", "category_ids": [1, 2]}]}
```
(`category_ids` only for multi-label datasets.)

**Sequences** (`labels.json`, frames stored in `data/<sequence name>/`):
```json
{"categories": [{"id": 1, "name": "moving_left"}, {"id": 2, "name": "moving_right"}],
 "sequences": [{"id": 1, "name": "sequence_000001",
                "frames": ["sequence_000001/000000.png", "sequence_000001/000001.png"],
                "category_id": 2,
                "frame_category_ids": [1, 2]}]}
```
(`category_id` for many to one, `frame_category_ids` for many to many.)

### Sequence settings (`Data.sequence_settings`)
- `sequence_length`: frames per training sample.
- `frame_stride`: take one frame every N.
- `window_stride`: long sequences are cut into sliding windows.
- `short_sequences`: `"pad"` (padding + mask) or `"skip"`.

### Loaders
- `data/loaders.py` keeps the current `get_dataloaders` signature and dispatches on the task and on
  `Data.DALI.activated`.
- The new tasks start with Native loaders (OpenCV + `torch.utils.data.DataLoader`). DALI versions come later:
  - classification with `fn.readers.file`;
  - sequences with `fn.readers.sequence` or an external source.

### Label mapping (`data/label_mapping.py`)
The binary / multi-class modes stay. The dictionary building moves out of `load_experiment_configuration` and becomes
task-aware:
- detection keeps index 0 for the background;
- classification and sequence tasks map to contiguous indices `0 … K-1` without a background class.

### Augmentation
`DataAugmenter` is unchanged. Detection passes boxes and labels, classification passes images only, sequence tasks
pass `[B, T, C, H, W]` and get one augmentation per sequence (already implemented and tested).

### Synthetic datasets
`create_synthetic_dataset` is extended with a `task` argument, not duplicated. It reuses the shape drawing functions:
- **Classification:** one shape per image, label = shape.
- **Many to one:** a shape moving across `T` frames, label = movement direction. This can only be learned from several
  frames, which checks that the temporal module is used.
- **Many to many:** a shape that changes class during the sequence, label per frame = visible shape.

### Startup description (`data/split_description.py`)
`count_split_images_and_annotations` is generalised per task:
- images + annotations (detection);
- samples + count per class (classification);
- sequences + frames + count per class (sequence tasks).

---

## 11. Losses

- **`losses/detection_losses.py`:** region proposal classification (BCE) and smooth L1 localisation; detection head
  cross entropy and smooth L1 regression. Moved verbatim out of `model.py`, same numbers.
- **`losses/classification_losses.py`:** cross entropy with `label_smoothing` and optional class weights (from the
  training class counts or the configuration); BCE with logits for multi-label.
- **`losses/sequence_losses.py`:** cross entropy over `[B, T, K]` with `ignore_index = -100`, averaged over valid
  frames only.

---

## 12. Evaluation

- **`evaluation/evaluation_report.py`:** the generic writer of `EvaluationResults` (console block, `Tensorboard/Test`
  scalars, `Results.txt` append, `file://` links to the saved files). It replaces the reporting half of
  `report_test_evaluation`.
- **Detection:** `mean_average_precision.py` unchanged; `DetectionTrainer.compute_evaluation_results` formats its
  output as `EvaluationResults` with the exact current lines.
- **Classification:** `classification_metrics.py`, `confusion_matrix.py`, `threshold_finder.py`.
- **Sequences:** `sequence_metrics.py` (frame metrics through the classification functions applied to the valid
  frames, sequence exact match, segment metrics).
- **Standalone evaluation:** `scripts/evaluation/evaluate_checkpoint.py` builds the trainer of the task saved in the
  run's configuration, so it works for any task.

---

## 13. Configuration

### Common sections
```yaml
ExperimentConfiguration:
  # Task trained by this configuration, selecting the trainer, the model and the task-specific checks.
  # One of "object_detection", "classification", "sequence_classification" (many to one) or
  # "sequence_labelling" (many to many).
  task: "classification"
  experiment_folder: ...            # unchanged keys: data folders, batch size, saving, validation, dtype, resume
  evaluation: { ... }               # task-specific contents (detection: score_threshold)
  diagnosis_modes: { ... }
Optimization:                       # later phase, learning_rate moves here
  optimizer: "adam"
  learning_rate: 2.0e-4
  weight_decay: 0.0
  backbone_learning_rate_multiplier: 1.0
Data:                               # unchanged + sequence_settings for sequence tasks
Backbone:                           # unchanged + pretrained (section 14)
```

### Task sections
- **Detection:** `RegionProposal`, `Anchors`, `Detector` (unchanged).
- **Classification and sequence tasks:**
  ```yaml
  ClassificationHead:
    # Global pooling of the final backbone feature map, "average" or "max"
    global_pooling: "average"
    # Fully connected layers between the pooled features and the class logits
    fully_connected: [256]
    # Dropout applied before every fully connected layer
    dropout_rate: 0.5
    # "multi_class" (softmax, one label) or "multi_label" (sigmoid, several labels)
    output_mode: "multi_class"
    # Label smoothing of the cross entropy, 0 to disable
    label_smoothing: 0.0
  ```
- **Sequence tasks:**
  ```yaml
  TemporalModel:
    # "mean_pooling", "max_pooling", "attention_pooling", "gru", "lstm", "transformer" or "temporal_convolution"
    type: "gru"
    hidden_size: 256
    number_layers: 2
    # Reads the sequence in both directions (offline use), must be False when causal
    bidirectional: True
    # Many to many only: each frame only sees the frames before it (online use)
    causal: False
  ```

### Checker
- `configuration/common_checks.py` checks the common sections.
- `core/task_registry.py` maps each task to its check functions; `check_configuration` runs the common checks, then
  the checks of the configured task.
- Every key stays required and read with `["key"]`; a section that a task doesn't use isn't read for that task.

### Run names
- `build_run_name` keeps its common sections and asks the task for its sections (`get_task_run_name_sections`):
  - task code first: `DET`, `CLS`, `M2O`, `M2M`;
  - detection: the current levels / anchors / heads / classes sections;
  - classification: `HAVG.256-DO0.5`;
  - sequences: also `T16_GRU2x256-BI`.
- Example many to one name:
  `M2O_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.4x64.4x64_T16_GRU2x256-BI_HAVG.256-DO0.5_MC4_AUG-BS8-LR2e-4`.

---

## 14. Backbone Transfer

### Configuration
Every key is required; `null` disables the transfer.
```yaml
Backbone:
  pretrained:
    # Run folder whose trained backbone initialises this run, null to start from random weights
    experiment_folder: null
    # Checkpoint iteration of that run, null for its last saved checkpoint
    iteration: null
    # Keeps the loaded backbone fixed (no gradient, batch normalization statistics frozen)
    freeze: False
```

### Loading (`models/backbones/backbone_transfer.py`)
1. Read the source run's `training_configuration.yaml` with `load_experiment_configuration`.
2. Compare the architecture-defining backbone keys of both runs:
   - must match: `input_channels`, `convolutions`, `modules`, `normalization`, `last_max_pooling`; otherwise stop with
     the list of differences;
   - may differ: `detection_convolution_block_indices` (it only selects outputs) and the image size (convolutions
     don't depend on it). A different image size prints a warning.
3. Load the checkpoint, keep the `model_state` keys starting with `backbone.`, strip the prefix,
   `backbone.load_state_dict(strict=True)`, and print the source run and checkpoint as `file://` links.
4. When `freeze` is set, the backbone parameters stop requiring gradients, and `BaseTrainer.set_training_mode` keeps
   the backbone in eval mode so its batch normalization statistics stay fixed.

### Why it works between all tasks
Every task builds the same `Backbone` class under the same `backbone` attribute, so any direction works:
classification → detection, detection → classification, classification → sequence tasks.

### Run name
`PT` is added to the backbone section when pretrained, and `FRZ` when frozen. The source run is printed at startup
and saved in the checkpoint metadata.

### Learning rate
`Optimization.backbone_learning_rate_multiplier` puts the backbone in its own optimizer parameter group, so a
transferred backbone can be fine-tuned more slowly than the new heads.

---

## 15. Deployment

- **Inference model per task:** each task provides an exportable inference model:
  - `InferenceDetector` (existing);
  - `InferenceClassifier`, which returns probabilities `[B, K]`;
  - `InferenceSequenceModel`, which returns `[B, K]` or `[B, T, K]`.
- **Export specification per task** (`deployment/export_specifications.py`):
  - input name and dimensions (sequences: dynamic `T`, Triton dims `[ -1, 3, H, W ]`, TensorRT shape profile);
  - output names, dimensions and types;
  - example inputs;
  - a comparison function (`compare_detections` for detection, `allclose` for probabilities).
- **`triton_configuration.py`:** writes `config.pbtxt` from a specification. The detection specification reproduces
  the current file exactly.
- **`export_model.py`:** loads the run with `load_trained_model`, which builds the right model from the saved task.
  It keeps the TF32 handling, then exports, compares and writes the configurations.
- **Throughput:** `deployment_utilities.py` / `run_inference.py` measure any inference model, with example inputs
  from the specification.

---

## 16. Scripts And Tests

### Scripts
Scripts mirror the package. The detection debug scripts move to `scripts/models/detection/` and
`scripts/models/components/`, and the data loader benchmark to `scripts/data/detection/`.

### Test fixtures
- `tests/fixtures/configuration_fixtures.py` gives one small configuration per task. It replaces the import of
  `get_model_configuration` from `scripts/`.
- `tests/fixtures/dummy_task.py` holds a tiny CPU task (linear model on random tensors) to test `BaseTrainer` alone.

### New tests
- **`BaseTrainer`**, with the dummy task:
  - loss summing and tracker averaging, including the validation divisor;
  - save → resume continues at the next iteration;
  - the `full-checkpoint` registry;
  - duplicate sample removal in the evaluation;
  - `Results.txt` / TensorBoard reporting.
- **Detection equivalence:** before the old files are deleted, the old and new detector give identical outputs and
  losses on a fixed batch.
- **Classification:**
  - model shapes;
  - loss decreasing on a tiny batch;
  - metrics against hand-computed values;
  - label mapping without a background class;
  - dataset manifest loading.
- **Sequences:**
  - padding and masks;
  - masked loss ignoring padded frames;
  - each temporal module (shapes, masks respected, causal modules not looking ahead);
  - sliding windows.
- **Backbone transfer:**
  - round trip between two tasks;
  - architecture mismatch rejected with the list of differences;
  - a frozen backbone keeping its batch normalization statistics.
- **Configuration:** the checks of every task; the `task` key required.
- **Export:** the detection Triton configuration unchanged; each task's inference model traced and compared.

### Reference files
The 28 `.pt` files of `tests/architecture_modules/reference_values/` are **never regenerated**. Confirmed: in Phase 1
they move to `tests/models/components/reference_values/` with `git mv`, contents unchanged, and the tests load them
from the new folder.

---

## 17. Migration Phases

Each phase leaves the repository working: all tests pass, a short training on the synthetic shapes runs, and existing
checkpoints still load. During each phase, the README `## Implementation Plan` section holds that phase's checklist,
and it's removed when the phase is done. Commits are by feature, without attribution.

| Phase | Content | Behaviour change | Size |
|---|---|---|---|
| 0 | Safety net | none | S |
| 1 | Package rename and new tree | none | M |
| 2 | Split the large files | none | M |
| 3 | `BaseTrainer` + `DetectionTrainer` | none (detection) | L |
| 4 | Configuration v2 (`task`, per-task checks, run name task code) | configuration format | M |
| 5 | Classification (one to one) | new task | L |
| 6 | Sequence classification (many to one) | new task | L |
| 7 | Sequence labelling (many to many) | new task | M |
| 8 | Backbone transfer | new feature | M |
| 9 | Deployment for every task | new feature | M |
| 10 | Optional: DALI loaders for new tasks, `Optimization` section, mixed precision, documentation | — | M |

### Phase 0: safety net
- Fix the configuration checker test that depends on the example configuration's saving values. It currently fails
  because `weight_saving_iterations` is 5000 instead of 10000; it should read the value instead of hard-coding it.
- Add the `BaseTrainer`-independent smoke test: a 5-iteration training on a generated synthetic dataset, a save,
  then a resume.
- Record, in a test, the detector's losses on a fixed seeded batch. These are plain numbers in the test file, not
  `.pt` files.
- **Done when:** the full suite is green and the smoke test passes.

### Phase 1: package rename and new tree
- `object_detection/` → `vision_training/` with `git mv`, so file history is kept. Files move to the tree of section
  4 without editing their content beyond imports.
- Package-qualified imports everywhere (`from vision_training.models.backbones.backbone import Backbone`), and
  `pythonpath = ["."]`.
- Update `.gitignore` (served models path), `docker-compose.yml` paths and the `scripts/` paths.
- **Done when:** the suite is green, old checkpoints load, and the export of an existing run still matches.

### Phase 2: split the large files
- `model.py` → `two_stage_detector.py` + `detection_losses.py`.
- `model_utilities.py` → box operations / target assignment / post-processing / visualization.
- `data_loader.py` → COCO dataset / DALI pipeline / loaders / split description.
- `os_utilities.py` → configuration loader / run name / label mapping.
- `configuration_checker.py` → configuration values / common checks / detection checks.
- **Done when:** the suite is green and the detector equivalence test still gives the same numbers.

### Phase 3: `BaseTrainer` + `DetectionTrainer`
- Extract the generic code into `base_trainer.py`, `checkpointing.py`, `metric_tracking.py`,
  `evaluation_report.py`, and write `DetectionTrainer` with the detection hooks.
- Introduce `core/contracts.py`, `sample_identifiers`, `TaskModel`, and the checkpoint metadata.
- Write the `BaseTrainer` tests with the dummy task.
- **Done when:** a detection training gives the same losses as before for the same seed, the evaluation writes the
  same `Results.txt` lines, and resume works.

### Phase 4: configuration v2
- Add the `task` key and `core/task_registry.py`; move the per-task checks; add the task code to the run name.
- Write `scripts/configuration/migrate_configuration.py`, which upgrades the saved `training_configuration.yaml` of
  older runs.
- **Done when:** both detection configurations pass; old runs pass after migration; existing run folders are renamed
  (section 18).

### Phase 5: classification
- `ClassificationHead`, `ImageClassifier`, classification losses, dataset + Native loader, label mapping without
  background, synthetic classification mode, `ClassificationTrainer`, metrics, confusion matrix, threshold finder,
  configurations, run name sections, viewer captions, tests.
- **Done when:** the synthetic shapes classification reaches near 100 % test accuracy in a short training.

### Phase 6: sequence classification (many to one)
- Sequence dataset + collate (padding, masks, windows), `FrameEncoder`, temporal modules, `SequenceClassifier`,
  `SequenceClassificationTrainer`, synthetic moving shapes, configurations, frame strip viewer, tests.
- **Done when:** the movement-direction dataset is learned with a temporal module, and is **not** learned with a
  single-frame baseline (sanity check that time is used).

### Phase 7: sequence labelling (many to many)
- Per-frame labels, masked sequence loss, time-preserving temporal modules (causal options), `SequenceLabeller`,
  `SequenceLabellingTrainer`, frame and sequence metrics, configurations, tests.
- **Done when:** the changing-shape dataset gets high frame accuracy, and the causal modules pass the no-look-ahead
  test.

### Phase 8: backbone transfer
- `backbone_transfer.py`, the `Backbone.pretrained` keys, freeze handling in `set_training_mode`, the run name
  tokens, the optimizer parameter groups, tests.
- **Done when:** a classification backbone initialises a detection run, and the first detection losses are lower than
  from random weights on the synthetic shapes.

### Phase 9: deployment for every task
- Inference models for the new tasks, export specifications, the generic Triton configuration writer, throughput
  measurement for any task.
- **Done when:** the detection export is unchanged, and the classification and sequence exports match their eager
  models and load in Triton.

### Phase 10: optional
- DALI loaders for classification and sequences.
- The `Optimization` section (optimizer choice, weight decay, warm-up and cosine schedule, gradient clipping).
- bf16 autocast and `torch.compile` in `BaseTrainer`.
- README sections per task.
- Removal of `classifier_insipiration_code/` once its features are covered.

---

## 18. Backward Compatibility

- **Checkpoints:** module attribute names are kept, so `model_state` keys don't change. New checkpoints add `task`,
  `run_name` and `format_version: 2`. A checkpoint without `format_version` is read as an object detection
  checkpoint (version 1). That's runtime metadata, not configuration.
- **Saved configurations:** older runs' `training_configuration.yaml` lack `task` (and any renamed key). The migration
  script upgrades them explicitly, instead of the loader guessing defaults, which keeps the strict-configuration rule.
- **Run folders:** adding the task code to run names changes the folder of existing runs. A one-off rename of the
  existing detection run folders (prefixing `DET_`) keeps resuming possible; it's done when no training is running.
- **The training currently running** in `experiments/IN-KAR-1024_SE-RES-…` isn't affected by code moves, since its
  Python process has already loaded its modules. Its folder must not be renamed while it runs. Resuming it with the
  new code requires the folder rename and the configuration migration of Phase 4.
- **Triton repository:** the exported detection models and their `config.pbtxt` stay identical (checked in Phase 9).

---

## 19. Risks

| Risk | Mitigation |
|---|---|
| A silent change of detection behaviour during the moves | the equivalence test on a fixed batch (Phase 0), compared at the end of Phases 1–3 |
| Import breakage across ~95 tracked files | one mechanical phase (1) with package-qualified imports, the suite run after every group of moves |
| `BaseTrainer` becoming a new monolith | checkpointing, metric tracking and evaluation reporting kept in their own modules |
| Hooks too detection-shaped to fit sequences | the hook list designed with all four tasks (section 7), checked with the dummy task first |
| GPU memory of sequence batches (`B × T` images) | smaller image sizes in sequence configurations, an optional frozen backbone, gradient checkpointing as a later option |
| DALI support for new tasks | Native loaders first, DALI as an optional phase |
| Moving the `.pt` reference files | confirmed as a pure `git mv`, contents never regenerated |

---

## 20. Decisions Taken

1. **Package name:** `vision_training/`.
2. **Import style:** package-qualified imports (`from vision_training.models.backbones.backbone import Backbone`),
   with `pythonpath = ["."]`.
3. **Tests and scripts location:** at the repository root, mirroring the package.
4. **Reference `.pt` files:** moved with `git mv` to `tests/models/components/reference_values/`, contents unchanged.
5. **Classification label format:** a JSON manifest in `labels.json`, same split convention as detection
   (multi-label through `category_ids`).
6. **Sequence format:** frames stored as image files in one folder per sequence, listed in `labels.json`, with
   `category_id` (many to one) or `frame_category_ids` (many to many).
7. **Run name task code:** `DET` / `CLS` / `M2O` / `M2M` at the start of every run name; existing detection run
   folders get a one-off `DET_` prefix, done when no training is running.
8. **Phase order:** backbone transfer after the sequence tasks (Phase 8), as planned.
