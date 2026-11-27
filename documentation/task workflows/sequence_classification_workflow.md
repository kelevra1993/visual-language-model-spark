# Sequence Classification Workflow

This document follows one sequence classification run (many to one, M2O: one class per sequence of frames) from `main()` to the evaluation of a checkpoint: which function runs when, what it receives and what it returns. The **Call Tree Overview** gives the whole run on one page. Each stage then has one step per function, with a short explanation and a **real** output. The steps shared by every task are explained in `classification_workflow.md`; the evaluation is the one of the classification without its temperature explorer. The steps specific to sequences (whole sequences, padding, frame encoder, temporal module, frame grids) are explained in full.

Every sequence is one sample, read whole: there is no window. A sequence longer than `Data.sequence_settings.maximum_sequence_length` is left out of its split with a warning. The sequences of a batch are padded with black frames to the longest one.

The examples come from real runs on CPU:
- **Stages 1 and 2**: `vision_training/configurations/sequence_classification/configuration_synthetic_sequences.yaml` (4 classes `moving_left`, `moving_right`, `moving_up`, `moving_down`, `maximum_sequence_length: 50`, 128 pixel frames, batch size 16), on a generated dataset of 2000 training, 200 validation and 200 test sequences of 2 to 12 frames. The dataset the configuration points to (`/home/robert_kelevra/Data/sequence_data/synthetic_moving_shapes`, 2026-11-18) has the same counts at 512 pixels.
- **Stages 3, 4 and 5**: the same configuration shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 (an intermediate one every 3), a summary every 4, a validation batch every 10, error images and MLflow on. It predicts `moving_down` for every sequence (about 25 % accuracy, a loss near ln 4 = 1.386): the metrics are poor on purpose.

This short run dates from 2026-10-18, before the ROC and precision-recall curves, the `flatten` pooling (it used `average`), the fixed top-5 accuracy (it shows `Top-2 Accuracy`) and the bfloat16 autocast. Each step says where the current code prints more. Rates are shown as the current code writes them (`23.50 %`); files, TensorBoard and MLflow keep 0 to 1. The current outputs come from the README runs of `~/Data/tutorial_runs/readme_screenshots/` (`M2O_IN-KAR-256_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HFLAT.64_MC4` at 256 pixels, an `average` run and a `flatten` run at 512 pixels) and from the 128 pixel flatten README run of 2026-10-23 evaluated again on 2026-11-12 (Steps 4.10 to 4.13). Console paths are terminal hyperlinks (`format_path_link()`); the examples write the `file://` URI they open.

## Read This First

The sequence classification task (`task: "sequence_classification"`, run name code `M2O`) learns one class for a whole sequence of frames. Every real frame goes through the backbone and the frame pooling of `ClassificationHead.global_pooling` (`average`, `max`, or `flatten`, which keeps the whole last feature map). The temporal module of `TemporalModel.type` summarises the frame vectors into one vector. The fully connected head of CLS gives K logits read with a softmax (`output_mode` must be `multi_class`). Padding frames are kept out of the model by a mask.

Choose it rather than CLS when the class needs several frames (a direction of motion that no single frame shows). Choose M2M or SEL for a class or a choice per frame, and CMP to compare a before and an after sequence. `SequenceClassificationTrainer` extends `ClassificationTrainer`: the loss, the batch accuracy and the evaluation are those of CLS on sequences, without the temperature explorer and its `class_logits.npz` (`saves_temperature_explorer()`). `SequenceLabellingTrainer` (M2M) and `SequenceComparisonTrainer` (CMP) extend it in turn.

### The Task At A Glance

To find your way in the stages below, this block follows one batch of the synthetic configuration from the manifests to the export.

```text
<split>/labels.json {"id": 1, "name": "train_000000", "frames": ["train_000000/000000.png", ...], "category_id": 2}
<split>/data/train_000000/000000.png ... (2 to 12 frames from the generator)   label_mapping.json {"1": 0, .., "4": 3}
SequenceDataset: a sequence longer than maximum_sequence_length (50) left out; read_image + preprocess_image (128)
   sample {images [T_i, 3, 128, 128] uint8, sequence_mask [T_i] all True, labels: class index, sample_identifiers}
collate_padded_sequences: black frames up to the longest sequence T of the batch, mask False on them
   ─► images [16, T, 3, 128, 128], sequence_mask [16, T], labels [16] ─► prepare_batch (float32 in [0, 255])
   ─► augment_batch: one draw per sequence, shared by all its frames (training batches only)
SequenceClassifier.forward(input_tensor, sequence_mask, labels)
   encode_frames: the N real frames of the batch as one image batch ─► Backbone (SE, shortcuts in blocks 3 and 5)
      B1 [N, 16, 64, 64]  B2 [N, 32, 32, 32]  B3 [N, 32, 16, 16]  B4 [N, 64, 8, 8]  B5 [N, 64, 8, 8]
      ─► flatten [N, 4096] ─► put back at their frame positions, zero vectors on the padding: [16, T, 4096]
   RecurrentTemporalModule: GRU(4096, 128, bidirectional) on packed sequences ─► final states [16, 256]
   ClassificationHead: Dropout, Linear 64, ReLU, Dropout, Linear 4 ─► class_logits [16, 4] ─► softmax
loss: classification_loss = cross entropy (no class weights, no label smoothing) = total_loss; batch accuracy
every weight_saving_iterations (1000): the whole test split ─► accuracy, balanced accuracy, macro F1, top-5,
   macro ROC AUC, macro average precision, per-class metrics (recall with its 95 % interval) and thresholds at
   precision 0.99
   ─► Results.txt, Tensorboard/Test, Evaluation/Iteration_<n>/ (confusion matrix, class_metrics.csv, curves,
      class_thresholds.json, predictions.csv, Samples/ and Errors/ as grids of the real frames, 4 columns)
export: InferenceSequenceModel, images [B, T, 3, 128, 128] float32 BGR, every frame real ─► class_probabilities [B, 4]
   with the input contract of the run in preprocessing.json and the ONNX metadata (keep_ratio, 128, BGR, NTCHW, 0-255)
```

### Files To Know

Should you need to change one step of the task, this table names the file and the functions that hold it.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/sequence_classification_trainer.py` | `SequenceClassificationTrainer`: `prepare_batch`, `forward_pass`, `augment_batch`, `describe_split`, `build_input_image` |
| Loss, batch metric, evaluation | `vision_training/trainers/classification_trainer.py` | `ClassificationTrainer` (inherited) |
| Dataset and loaders | `vision_training/data/sequence/sequence_dataset.py`, `vision_training/data/loaders.py` | `SequenceDataset`, `read_sequence_frames`, `collate_padded_sequences`, `get_sequence_dataloaders` |
| Model | `vision_training/models/sequence/sequence_classifier.py`, `frame_encoder.py`, `temporal_modules.py` | `SequenceClassifier`, `encode_frames`, `build_temporal_module`, `RecurrentTemporalModule` and the other modules |
| Frame pooling and head | `vision_training/models/classification/classification_head.py` | `build_global_pooling`, `compute_pooled_features`, `ClassificationHead` |
| Sequence augmentation | `vision_training/data/augmentations.py` | `DataAugmenter.augment_batch`, `DataAugmenter.shuffle_frames` |
| Checks, run name | `vision_training/configuration_processing/sequence_checks.py`, `run_name.py` | `check_sequence_classification_configuration`, `build_sequence_sections` |
| Synthetic data | `vision_training/data/synthetic/create_synthetic_dataset.py` | `generate_moving_shape_sequence`, `build_sequence_labels` |
| Export | `vision_training/models/sequence/inference_sequence_model.py` | `InferenceSequenceModel` (model name `sequence_classifier`) |
| Standalone evaluation | `scripts/evaluation/evaluate_checkpoint.py`, `vision_training/evaluation/checkpoint_evaluation.py` | `evaluate_checkpoints`, `summarise_checkpoint_evaluations` |

### Run It

Before a first run of the task, generate the moving shapes once, then train the run and evaluate its checkpoints.

```bash
# Generate 2,000 / 200 / 200 sequences of 2 to 12 frames of 512 pixels, one shape moving left, right, up or down
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=sequence_classification --output_folder="$HOME/Data/sequence_data/synthetic_moving_shapes"

# Train: set experiment_folder, the three data folders and Data.label_mapping.mapping_file_path first
uv run python -m vision_training.main --configuration_path="vision_training/configurations/sequence_classification/configuration_synthetic_sequences.yaml"

# Evaluate every checkpoint into <run>/Checkpoint_Evaluation/ (split, batch size, tracking, predictions,
# error images and target precisions are arguments too)
uv run python -m scripts.evaluation.evaluate_checkpoint --experiment_folder <run folder> --iteration all
```

### Configuration Keys To Know First

If one wishes to change what the run learns from the frames or how long its sequences may be, these keys come first.

| Key | Synthetic value | Effect |
|---|---|---|
| `Data.sequence_settings.maximum_sequence_length` | `50` | the longest sequence kept (`T50`); longer ones are left out of every split with a red warning; the positions of a transformer embedding |
| `TemporalModel.type` | `"gru"` | 7 types: `mean_pooling`, `max_pooling` and `attention_pooling` ignore the frame order; `gru`, `lstm`, `transformer` and `temporal_convolution` follow it |
| `TemporalModel.hidden_size`, `number_layers` | `128`, `1` | `GRU1x128`; for `gru` and `lstm`, `dropout_rate` only acts between stacked layers |
| `TemporalModel.bidirectional` | `True` | both directions concatenated, 256 features (`BI`); `causal: True` is refused for M2O |
| `ClassificationHead.global_pooling` | `"flatten"` | the 64 x 8 x 8 map of a 128 pixel frame gives 4,096 features that keep where the shape is (`HFLAT`); `average` and `max` give 64 |
| `Data.image_settings.size` | `128` | the frame size; with `flatten` it sets the GRU input (65,536 features and 50.4 million GRU weights at 512 pixels) |
| `ClassificationHead.fully_connected` | `[64]` | the hidden layer of the head (`.64` of `HFLAT.64`) |
| `Data.data_augmentation.independent_frames` | `False` | same draws for every frame; True draws the photometric types frame by frame |
| `Data.data_augmentation.shuffle_frames` | `activated: False, probability: 0.25` | permutes the real frames of training sequences; a yellow warning with an ordered temporal module |

### What A Run Writes

Each time a run is launched, saved or evaluated, it writes the files of this table into its run folder.

| Output | Contents |
|---|---|
| `Results.txt` | one block per checkpoint, `Iteration <n> · <N> Test Sequences · <date>`, as CLS: `Accuracy` (headline), `Balanced Accuracy`, `Macro F1`, `Top-5 Accuracy`, with `per_class_report` the curves metrics, the per-class table and `Per Class Recall` (Step 4.13) |
| `Evaluation/Iteration_<n>/` | the files of CLS without `temperature_explorer.html` nor `class_logits.npz`; `predictions.csv` with `sequence`, `first_frame` and `last_frame` columns; `Samples/` and `Errors/<TRUE>-TO-<PREDICTED>/` as frame grids of 4 columns (Step 4.5) |
| `Evaluation/evaluation_summary.csv`, `evaluation_curves.png` | one row per evaluated iteration, the accuracy curve (Step 4.13) |
| `Input_Samples/` | the first 10 sequences of every split as frame grids, with `samples.json` (Step 3.1) |
| `data.json` | the `sequences` of every class in every split, and every frame of every split |
| `Tensorboard/`, MLflow | as CLS, plus `train/frame_throughput` (real frames per second) |

### Pitfalls

In case a run stays at chance, grows larger than expected or counts its epochs wrong, the cause is likely one of these.

- **`average` and `max` lose where the shape is**, so the moving shapes stay at chance: 24.50 % for the `average` README run at 512 pixels after 2,000 iterations, against 99.00 % at iteration 500 for `flatten` at 256 pixels ([tasks reference, Sequence Classification](../reference/tasks_reference.md#sequence-classification-m2o)).
- **`flatten` at 512 pixels is too large**: 65,536 features per frame, a GRU of 50.4 million weights, still 24.50 % after 1,000 iterations.
- **Lengths outside the training data.** The runs learn from 2 to 12 frames, but the export accepts up to 50 (`load_inference_inputs()` of `deployment_utilities.py`). Nothing warns about 13 to 50 frames.
- **`Iterations Per Epoch` counts sequences before the label mapping**, so it is too high when the mapping leaves categories out (`count_split_sequences()`).
- **`TemporalModel.dropout_rate` does nothing with one GRU or LSTM layer**: PyTorch dropout only acts between stacked layers.
- **`shuffle_frames` with an ordered temporal module** only prints a yellow warning and destroys the class of the moving shapes.

## Table of Contents
- [Read This First](#read-this-first)
- [Call Tree Overview](#call-tree-overview)
- [Stage 1 — Entry Point and Configuration Loading](#stage-1--entry-point-and-configuration-loading)
- [Stage 2 — Trainer Initialization](#stage-2--trainer-initialization)
- [Stage 3 — Training Loop](#stage-3--training-loop)
- [Stage 4 — Test Evaluation of a Checkpoint](#stage-4--test-evaluation-of-a-checkpoint)
- [Stage 5 — Standalone Checkpoint Evaluation (Optional)](#stage-5--standalone-checkpoint-evaluation-optional)

---

## Call Tree Overview

So that one run can be followed from its launch to the evaluation of a checkpoint, this tree gives every function in the order it runs.

```text
main() -> load_experiment_configuration() -> check_sequence_classification_configuration()
       -> SequenceClassificationTrainer(...)  [STAGE 2]
          BaseTrainer.__init__(): get_run_name(), build_dataloaders() -> get_sequence_dataloaders()
             (SequenceDataset, collate_padded_sequences), build_model() -> SequenceClassifier (backbone,
             global pooling, build_temporal_module(keep_time_axis=False), ClassificationHead),
             initialise_pretrained_components(), MetricTracker(counts_frames=True), printouts, restore_last_model()
       -> run_training_loop() -> run_training_iterations()  [STAGE 3], for each iteration:
             get_next_batch() (read_sequence_frames, padding), MetricTracker.count_frames(),
             run_model_iteration(): prepare_batch(), augment_batch() (one draw per sequence), forward_pass()
                -> encode_frames(), temporal_module, classification_head, compute_classification_loss()
             backward, step_optimizer(), save_model(), validation, print_interval_summary(),
             run_test_evaluation()  [STAGE 4]: the classification evaluation on whole sequences, frame grids for
                Samples/ and Errors/, report_evaluation() + update_evaluation_curves()
scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]: evaluate_checkpoints() -> run_test_evaluation()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs of the loader on `vision_training/configurations/sequence_classification/configuration_synthetic_sequences.yaml`.

### Step 1.1 — `main()`
The entry point of every task (`classification_workflow.md`, Step 1.1): `parse_configuration_path()` reads `--configuration_path`, then the file is loaded, the trainer of `ExperimentConfiguration.task` found and its loop started. Nothing here is specific to sequences.

### Step 1.2 — `load_experiment_configuration()`
It splits the YAML into the experiment and model dictionaries, converts the types, adds `label_mode` and `label_dictionary` to `Data` (Step 1.4) and runs the checks (Step 1.5). On top of the classification sections, the task has `TemporalModel` and `Data.sequence_settings` with one key, `maximum_sequence_length`. Every `TemporalModel` setting is required, even those its type does not read, so the type can change without adding keys.

**Example: the sections and the settings of the sequences** (shortened)
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, ClassificationHead
sequence_settings:        {'maximum_sequence_length': 50}
TemporalModel:            {'type': 'gru', 'hidden_size': 128, 'number_layers': 1, 'number_heads': 4, 'kernel_size': 3, 'dropout_rate': 0.0,
                           'bidirectional': True, 'causal': False, 'positional_embedding': True, 'pretrained': {...}}
ClassificationHead:       {'global_pooling': 'flatten', 'fully_connected': [64], 'output_mode': 'multi_class', ...}
```

### Step 1.3 — `load_configuration()`
Plain YAML reader, as for every task: no conversion and no check.

**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'TemporalModel', 'ClassificationHead']
number_iterations: '1e10'      <- still a string
```

### Step 1.4 — `build_label_dictionary()`
The mapping file gives the class index of every category, a category being the class of a whole sequence. Categories can be merged by giving them the same index. Sequences of a category left out of the mapping are left out of the splits (Step 2.9).

**Example: the label dictionary of the moving shapes**
```text
dict: {1: 0, 2: 1, 3: 2, 4: 3}
4 entries, 4 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_sequence_classification_configuration()`. This table lists its parts.

| Part | Checks |
|---|---|
| classification checks | the data, head and evaluation checks of CLS |
| `check_sequence_settings()` | `maximum_sequence_length` an integer of at least 1; the booleans `independent_frames` and `shuffle_frames.activated` with the `shuffle_frames` probability; a `multi_class` head (one category per sequence) |
| `check_temporal_model_configuration()` | type, sizes, dropout; not causal (causal modules, which see only past frames, are for M2M and SEL) |

Every global pooling is accepted, `flatten` included: the temporal module is built for the width given by `compute_pooled_features()` (Step 2.14). Every problem is reported together, then the program exits.

**Example: output with `causal: True` and `maximum_sequence_length: 0` in the YAML**
```text
Invalid Configuration File : file:///.../configuration_wrong.yaml
  - 'Data.sequence_settings.maximum_sequence_length' must be an integer >= 1, got 0.
  - 'TemporalModel.causal' must be False for this task, which summarises whole sequences (causal modules are for the sequence labelling task).
  - 'TemporalModel.bidirectional' must be False when 'TemporalModel.causal' is True: reading the sequence backwards looks at the future frames.
SystemExit code 1
```
The single `causal: True` gives two errors, since the configuration is also `bidirectional: True`.

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="sequence_classification") -> <class 'vision_training.trainers.sequence_classification_trainer.SequenceClassificationTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs of building the trainer on `configuration_synthetic_sequences.yaml` (128 pixel frames, `maximum_sequence_length: 50`, batch size 16), data and run folders in a temporary folder. The construction took 0.09 s on CPU.

### Step 2.1 — `SequenceClassificationTrainer.__init__()`
It keeps `Data.sequence_settings` and warns when `shuffle_frames` is on with an ordered temporal module (GRU, LSTM, transformer, temporal convolution). Then it runs the classification trainer constructor. This table lists the class attributes that describe the task to the shared code.

| Attribute | Value | Meaning |
|---|---|---|
| `LABEL_TYPE` | `"sequence"` | one class per sequence |
| `MEMBERS_PER_SAMPLE` | `1` | one sequence per sample (2 for the comparison pairs) |
| `SAMPLE_NAME` | `"Sequences"` | the unit of the console |
| `COUNTS_FRAMES` | `True` | the frame throughput (Step 3.8) |
| `SEQUENCE_AUGMENTATION` | `True` | the augmenter reads `independent_frames` and `shuffle_frames` (Step 2.5) |

The classification constructor then sets `top_k` 5 (`TOP_K_ACCURACY_CLASSES`), `save_predictions` and the error dumps settings.

### Step 2.2 — `BaseTrainer.__init__()`
The shared constructor (`classification_workflow.md`, Step 2.2), reached through `ClassificationTrainer.__init__()`. `input_image_size` is the size of every frame.


### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The input and backbone tokens are those of every image task. `build_sequence_sections()` adds `T<maximum_sequence_length>` (`build_sequence_length_tokens()`) and the temporal module (`build_temporal_tokens()`). The head (`HFLAT.64`) and the classes (`MC4`) end the name. The temporal tokens are `MEAN`, `MAX`, `ATT<hidden size>`, `GRU` or `LSTM<layers>x<hidden size>` (`BI` when bidirectional), `TRF<layers>x<hidden size>.<heads>` (`NPE` without positional embedding) and `TCN<layers>x<hidden size>.<kernel size>`. The temporal dropout gives no token.

**Example output**
```text
M2O_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HFLAT.64_MC4
```

### Step 2.5 — `DataAugmenter()`

The augmenter of the classification. Every sample is a sequence `[T, 3, H, W]`, augmented as a whole (Step 3.12). With `SEQUENCE_AUGMENTATION`, it reads `independent_frames` (photometric types drawn frame by frame) and `shuffle_frames`. The configuration activates 5 of the 28 types (`random_crop`, `random_rotation`, `brightness_contrast_saturation`, `random_noise`, `random_haze`) at probability 0.4, with both sequence settings off.

### Step 2.6 — `setup_training_paths()`
As for every task (`classification_workflow.md`, Step 2.6): `restore_run_label_dictionary()` and `check_resumed_data()` handle a resumed run. The folder holds the configuration copy and `label_dictionary.json`. `data.json` (the classes with their sequences in every split, the folder, `labels.json` hash and frames of every split) is written once the loaders are checked (`save_data_record()`).


### Step 2.7 — `build_dataloaders()`
It builds the three loaders with `label_type="sequence"` (Step 2.8), the validation one only when the run reads it (`classification_workflow.md`, Step 2.7). It keeps `class_names` from the training manifest.

**Example output**
```text
class_names = ['moving_left', 'moving_right', 'moving_up', 'moving_down']
```

### Step 2.8 — `get_sequence_dataloaders()`
For each split it builds a `SequenceDataset` (Step 2.9) and wraps it in the Native loader (Step 2.10); sequence tasks always read frames with OpenCV on the CPU. The collate is always `collate_padded_sequences()` (`members_per_sample=1`), which pads every batch with black frames to its longest sequence (Step 3.7). With a batch normalized head (`fully_connected_normalization: "batch"`), the training loader drops its last incomplete batch (`ClassificationTrainer.drops_last_training_batch()`); here every batch is kept. A split whose loader gives no batch is refused (`check_loaders_give_batches()`).

**Example: size of the three splits**
```text
train       2000 sequences -> 125 batches of 16 (one epoch = 125 iterations)
validation   200 sequences ->  13 batches of 16
test         200 sequences ->  13 batches of 16
collate     functools.partial(<function collate_padded_sequences>, members_per_sample=1)
```

### Step 2.9 — `SequenceDataset.__init__()`
It reads the `categories` and `sequences` of the manifest; each sequence gives its `id`, `name`, ordered `frames` (paths relative to `data/`) and `category_id`. Every sequence is one sample. Sequences of an unmapped category or without frames are left out first (one yellow note). Then every sequence longer than `maximum_sequence_length` is left out with a red warning, followed by a red total. The per-frame category check is for M2M and SEL only. The kept sequences are shuffled once and packed into `SerializedRecords` (`classification_workflow.md`, Step 2.9), found by identifier with `sequences.find_record()`. No frame is read here.

**Example: the lengths of the training sequences**
```text
frames: number of sequences   2: 178   3: 191   4: 187   5: 190   6: 200   7: 204   8: 165   9: 168   10: 185   11: 171   12: 161
```

**Example: a test split built with `maximum_sequence_length: 10` and the categories 1 to 3 only in the label dictionary**
```text
Sequence 'test_000004' Of The Test Split Has 12 Frames, More Than The Maximum Sequence Length (10): It Is Left Out Of The Test.      <- red, one line per sequence
...
28 Sequences Of The Test Split Are Longer Than The Maximum Sequence Length (10) And Are Left Out Of The Test.                       <- red
47 sequences of unmapped categories or without frames left out : labels.json                                                        <- yellow
len(dataset) = 125              (200 - 47 - 28)
```

**Example: the first sample the loader gets**
```text
dataset[0]: images (2, 3, 128, 128) torch.uint8, sequence_mask (2,) torch.bool [True, True], labels () torch.int64 2
get_class_counts() = [522, 484, 516, 478]           (sequences of every class, for the balanced class weights)
```

### Step 2.10 — `build_native_loader()`

The PyTorch `DataLoader` of the classification (`classification_workflow.md`, Step 2.10), with the collate of Step 2.8: 125 training batches of 16. It was recorded with 8 workers; every loader now has 1 worker and pins its batches only on a discrete GPU.

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
`get_sample_class_indices()` gives the class of every test sequence, so `choose_sample_identifiers()` gives the 8 sample sequences to the 4 classes in turn (2 each), taken at the ends of the sorted identifiers.

**Example output**
```text
sample_image_identifiers  -> [0, 1, 2, 4, 195, 197, 198, 199]
```

### Step 2.12 — `build_model()`
The classification hook creates the `SequenceClassifier` from `TASK_MODEL_CLASSES["sequence_classification"]` (Step 2.13). With `class_weighting: "balanced"` it sets `model.class_weights` from `get_class_counts()`.

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          RecurrentTemporalModule  3,245,568 parameters
model.classification_head      ClassificationHead          16,708 parameters
model.class_weights            buffer [4] torch.float32 [1.0, 1.0, 1.0, 1.0]
```

### Step 2.13 — `SequenceClassifier.__init__()`
The model turns every frame into one vector with the backbone and a global pooling (Step 2.14), summarises the vectors of a sequence with the temporal module (Step 2.15), and gives the class logits with the classification head (Step 2.16). The backbone is the one of the image tasks, so a backbone trained on images transfers to it.

### Step 2.14 — `build_backbone()` + `build_global_pooling()`
The backbone of every task (`classification_workflow.md`, Step 2.14), built for a frame of `image_settings.size`. The global pooling reduces its final map to one vector per frame, whose width `compute_pooled_features()` gives: 64 features for `average` and `max`, or 64 x 8 x 8 = 4096 for `flatten`, which follows the image size. Only blocks 3 and 5 list `Residual Shortcuts`: blocks 2 and 4 change the channel count at their first layer, and block 1 has a single layer.

**Example: the block output shapes for a 128×128 frame** (from the backbone panel)
```text
Block 1 [1, 16, 64, 64]   Block 2 [1, 32, 32, 32]   Block 3 [1, 32, 16, 16] (Residual Shortcuts)
Block 4 [1, 64, 8, 8]     Block 5 [1, 64, 8, 8] (Residual Shortcuts)
Final Backbone output shape: [1, 64, 8, 8]
```

### Step 2.15 — `build_temporal_module()` → `RecurrentTemporalModule`
It builds the module of `TemporalModel.type` with `keep_time_axis=False` (one vector per sequence), with `maximum_sequence_length` as its largest length. This table describes the types.

| Type | Order | What it does |
|---|---|---|
| `mean_pooling`, `max_pooling` | ignored | mean or maximum of the real frames, no weights |
| `attention_pooling` | ignored | weighted mean, the weight of every frame learned from its vector |
| `gru`, `lstm` | followed | final state of the last layer after the last real frame, both directions concatenated when bidirectional |
| `transformer` | followed | frame vectors projected to `hidden_size`, learned positional embedding unless `positional_embedding` is False, attention leaving out padding; outputs of real frames averaged |
| `temporal_convolution` | followed | 1D convolutions of `kernel_size` frames, dilation doubling at every layer, padding set to zero; outputs of real frames averaged |

Every module gives its output size in `output_features`, read by the head.

**Example: the module of the configuration** (4096 features per frame, hidden size 128, bidirectional)
```text
RecurrentTemporalModule(
  (recurrent_network): GRU(4096, 128, batch_first=True, bidirectional=True)
)
output_features = 256          (2 directions x 128)
```

### Step 2.16 — `ClassificationHead.__init__()`
The head of the classification (`classification_workflow.md`, Step 2.17), fed with the sequence vector.

**Example output**
```text
ClassificationHead(
  (fully_connected_block): FullyConnectedBlock(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=256, out_features=64, bias=True)
    (2): ReLU(inplace=True)
  )
  (dropout): Dropout(p=0.2, inplace=False)
  (classifier): Linear(in_features=64, out_features=4, bias=True)
)
```

### Step 2.17 — `initialise_pretrained_components()`
The backbone, the temporal module and the head can come from other runs (poolings without weights cannot). The rules are those of the classification (`classification_workflow.md`, Step 2.19):
- With `ClassificationHead.pretrained.class_layer: False` (the default) the head loads its hidden layers only. With `True`, every source class must be a class of this run (`compare_class_names()`), and `extend_class_layers()` copies its row.
- A temporal module transfers only to one of the same type and sizes, reading frame vectors of the same size (same last block channels, same pooling when one run flattens, same image size for `flatten`). A transformer with positional embedding also needs the same `maximum_sequence_length`.
- A backbone deeper than its source loads its first blocks only (`count_loaded_backbone_blocks()`).

**Example: a run from scratch**
```text
pretrained_checkpoint_paths = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
```

### Step 2.18 — `build_optimizer()`
**Example output**
```text
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
```

### Step 2.19 — `get_tracked_metrics()` + `get_percent_tracked_metrics()` + `MetricTracker.__init__()`
The metrics of the classification: total loss, classification loss and batch accuracy. The tracker has `counts_frames=True`, so it also counts the real frames (Step 3.8). The accuracy is its only rate, printed as a percentage (Step 3.21).

**Example output**
```text
{'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'accuracy': 'Accuracy'}
MetricTracker: batch_size 16, validation_iteration_ratio 25, sample_name 'Sequences', counts_frames True,
               percent_metric_keys ['accuracy']
```

### Step 2.20 — `print_experiment_information()` (→ `describe_split()` → `count_split_sequences()`)
`describe_split()` counts the sequences and frames of every split, and the sequences of at most `maximum_sequence_length` frames (`count_split_sequences()`). An epoch is the kept training sequences over the batch size. The count is made before the label mapping (see Pitfalls).

**Example output** (shortened)
```text
- Train Data Folder       : file:///tmp/.../synthetic_moving_shapes/train (2000 sequences, 13750 frames, 2000 sequences of at most 50 frames)
- Validation Data Folder  : file:///tmp/.../synthetic_moving_shapes/validation (200 sequences, 1357 frames, 200 sequences of at most 50 frames)
- Test Data Folder        : file:///tmp/.../synthetic_moving_shapes/test (200 sequences, 1377 frames, 200 sequences of at most 50 frames)
- Iterations Per Epoch    : 125
...
```

### Step 2.21 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The sequence classifier adds the Temporal Module panel (every setting of the section), then the head panel. The size panel lists the temporal module on its own line. Then the Export Sizes panel: its forward pass runs `InferenceSequenceModel` on one traced sequence (`probe_inference_graph()`). A GRU or LSTM counts as one weighted operator per layer, as in the ONNX graph, whatever its frames.

**Example output: the panels** (summarised)
```text
Temporal Module: Type gru, Hidden Size 128, Number Layers 1, Bidirectional True, Causal False, Output Features 256
Head panel: Global Pooling flatten, Input Features 256, Hidden Layers [64], Number Of Classes 4
Model Size: Backbone 183,648 (0.74 MB), Temporal Module 3,245,568 (12.98 MB), Classification Head 16,708 (0.07 MB),
            Total Model 3,445,924 (13.79 MB)
```

### Step 2.22 — `view_input_batches()` → `show_input_samples()`
With `diagnosis_modes.view_input_images`, the real frames of an augmented training sequence and of a validation sequence are laid out on numbered grids of 4 columns (`build_sample_frame_grid()` → `build_frame_grid()`). Each grid has a panel with the class (`Class : moving_up`), the real frames (`Real frames : 7`), the augmentations drawn (one line per frame with `independent_frames`) and, with `shuffle_frames`, the frame order (`describe_frame_order()`). The validation panel reads `Augmentations : none (validation sample)`. It needs a display; the same grids are written to `Input_Samples/` at every launch (Step 3.1).

### Step 2.23 — `restore_last_model()` → `get_last_checkpoint_iteration()` / `load_model()`
The restoration of every task (`classification_workflow.md`, Steps 2.26 to 2.28).

**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```

---

## Stage 3 — Training Loop

> The examples below are real outputs of the shortened CPU run (64 pixel frames, batch size 4, 12 iterations; an epoch is 500 iterations). Its run name is `M2O_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4`. It used the `average` pooling, so its frame vectors have 64 features instead of the 1024 of a flattened 64 x 4 x 4 map; the steps are the same.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (`classification_workflow.md`, Step 3.1). Before the first iteration, `save_input_samples()` (added on 2026-11-18) draws the first 10 sequences of every split as in Step 2.22 into `Input_Samples/Training`, `Validation` and `Test`, with `samples.json`. The 256 pixel README run printed `- Input Samples : Input_Samples (30 images written in 2.00 seconds)`.

This table shows what happens at each iteration of the run.

| Event | Iterations |
|---|---|
| Checkpoint saved | 3, 6, 9, 12 (nothing more after the loop) |
| Validation batch | 10 |
| Console summary | 4, 8, 12 |
| Test evaluation | 6, 12, last in the iteration |

### Step 3.2 — `setup_tensorboard_writers()`
**Example: the curves after the run**
```text
Tensorboard/Train/        Total Loss, Classification Loss, Accuracy, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Accuracy                   (iteration 10)
```

### Step 3.3 — `initialize_training_loop_state()` + `MetricTracker.create_trackers()`

The sequence trackers hold one more counter, `interval_frames`, the real frames of the interval (Step 3.8), next to `interval_batches`, `total_loss`, `classification_loss` and `accuracy`.

### Step 3.4 — `start_experiment_tracking()`
The MLflow run is created in the experiment `sequence_classification` and recorded in `mlflow_run.json`, so a resumed training continues it. A background thread opens it, so the line can appear during the first iterations (here iteration 2).

**Example output**
```text
MLflow Run Created In The Experiment 'sequence_classification' : 1489ef2ba2ed4370bdd5d228fc8544c5 (Browse : mlflow ui --backend-store-uri sqlite:///.../example_runs/mlflow.db)
```

### Step 3.5 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12 (`classification_workflow.md`, Step 3.6). The checkpoint stores the label dictionary and the source runs of the three transferable components. `Weights/full-checkpoint` records the last checkpoint and lists all of them. Current checkpoints also hold `pretrained_backbone_blocks` (Step 2.17).

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   4,289,735 bytes   (the same size at 6, 9 and 12)
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
model_state: 94 tensors, the first 'class_weights'   mlflow_run_id: '1489ef2ba2ed4370bdd5d228fc8544c5'
```
With `flatten` at 128 pixels the GRU holds 3,245,568 weights instead of 148,992: a checkpoint of the 128 pixel README run takes 41,450,653 bytes.

### Step 3.6 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration (`classification_workflow.md`, Step 3.7). See Stage 4.

### Step 3.7 — `get_next_batch()` → `SequenceDataset.__getitem__()` → `read_sequence_frames()` + `collate_padded_sequences()`
`get_next_batch()` drops a batch whose loading fails (`classification_workflow.md`, Step 3.8). `read_sequence_frames()` reads every frame with `read_image()` (OpenCV, BGR, `uint8`, a `ValueError` naming an unreadable frame) and resizes it (`preprocess_image()`), with an all-True mask. The label is the class of the sequence. In the loader workers, `collate_padded_sequences()` pads every sequence with black frames to the longest of the batch, the mask with False. The frame axis `T` of a batch is the length of its longest sequence.

**Example: the first batch of the run** (padded to 8 frames; the second batch to 12)
```text
images               [4, 8, 3, 64, 64] torch.uint8      [B, T, C, H, W]
sequence_mask        [4, 8] torch.bool                  real frames per sequence [4, 7, 7, 8]
labels               [4] torch.int64                    [3, 2, 2, 2]
```

### Step 3.8 — `MetricTracker.count_frames()`
Before the batch goes to the device, the tracker adds its real frames (the sum of the mask) to `interval_frames`. The padding is not counted, so the frame throughput does not depend on it.

**Example: the frames of the first interval**
```text
interval_frames after each batch: 26, 60, 86, 112          (26 = 4 + 7 + 7 + 8 real frames of the first batch)
```

### Step 3.9 — `set_training_mode()`
`model.train()`: dropout on, batch normalization statistics updated.

### Step 3.10 — `run_model_iteration()`
The same function for training and validation batches (`classification_workflow.md`, Step 3.11).

**Example: the first training batch**
```text
classification_loss = 1.3803   total_loss = 1.3803   accuracy = 0.25
```

### Step 3.11 — `prepare_batch()`
The classification hook moves the frames (`float32`, still [0, 255] BGR) and labels to the device; the sequence hook adds the mask.


### Step 3.12 — `augment_batch()` → `DataAugmenter.augment_batch()` + `shuffle_frames()`
The augmenter decides for every sequence whether it is augmented (`probability: 0.4`), then applies every activated type to the whole sequence, so all frames share the same crop, rotation and colour change and the motion stays coherent. With `independent_frames`, the photometric types are drawn again per frame. Then `shuffle_frames()` permutes the real frames when activated (off here), the padding staying last. The frame block length is that of the batch (`get_member_length()`).

**Example: which sequences of the first batch were changed**
```text
images:        changed per sample [False, True, False, False]
sequence_mask: changed per sample [False, False, False, False]
```

### Step 3.13 — `forward_pass()` → `SequenceClassifier.forward(sequence_mask=..., labels=...)`
The trainer passes the frames, the mask and the labels, so the model returns its loss with its logits. This run was in float32; since 2026-10-29 the forward passes run in bfloat16 autocast, giving bfloat16 `class_logits` and `class_probabilities` and a float32 loss.

**Example: the dictionary returned in training mode**
```text
class_logits          [4, 4] torch.float32
class_probabilities   [4, 4] torch.float32
classification_loss   1.3803 (scalar)                   <- close to ln 4 = 1.386, the loss of a model that does not know yet
```

### Step 3.14 — `encode_frames()`
The real frames of the batch are gathered into one image batch (`input_tensor[sequence_mask]`) and run through the backbone and pooling. Padding frames cost nothing and stay out of the batch normalization statistics. The vectors are put back at their frame positions, padding getting zero vectors. The sequence of every frame is given to the backbone for the `sequence` normalization. Without mask (deployment) the frames run as one fixed-shape batch, which the exports need.

**Example: the first batch of the run**
```text
images [4, 8, 3, 64, 64] -> backbone on the 26 real frames -> encode_frames -> [4, 8, 64]
zero vectors per sequence [4, 1, 1, 0]        (the padding frames of the sequences of 4, 7, 7 and 8 frames)
```

### Step 3.15 — temporal module: `RecurrentTemporalModule.forward()`
With a mask, the sequences are packed (`pack_padded_sequence()`), so the GRU stops at the last real frame. The sequence vector is the final state of the last layer in both directions, concatenated. The other modules use the mask too (Step 2.15). The padding changes nothing: four test sequences of 7, 7, 3 and 5 frames padded to 7 get the probabilities they get alone, to 1.5e-08 (random weights, CPU).

**Example: the first batch**
```text
[4, 8, 64] frame vectors -> temporal_module -> [4, 256] sequence vectors
```

### Step 3.16 — `ClassificationHead.forward()` + `compute_classification_loss()`
The head gives the class logits, then the loss of the classification (`classification_workflow.md`, Steps 3.16 to 3.18): cross entropy weighted by `class_weights`, with `label_smoothing`.

**Example: the first batch**
```text
[4, 256] -> classification_head -> class_logits [4, 4]      classification_loss = 1.3803
```

### Step 3.17 — `compute_batch_metrics()`
The batch accuracy, from the argmax of the logits.

**Example: the first batch**
```text
accuracy = 0.25          (1 sequence of 4 right)
```

### Step 3.18 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
**Example: the Train curves**
```text
Total Loss   1: 1.3803, 2: 1.4069, 3: 1.4105, 4: 1.4166, 5: 1.4618, 6: 1.3604, 7: 1.3056, 8: 1.4383, 9: 1.3274, 10: 1.3969, 11: 1.4032, 12: 1.3981
Accuracy     1: 0.2500, 2: 0.2500, 3: 0.0000, 4: 0.0000, 5: 0.2500, 6: 0.2500, 7: 0.5000, 8: 0.5000, 9: 0.7500, 10: 0.0000, 11: 0.2500, 12: 0.2500
```

### Step 3.19 — `training_loss.backward()` + `step_optimizer()`
**Example: the gradient norm of the first steps and the learning rate**
```text
iteration 1: 1.5764    iteration 2: 1.3961    iteration 3: 1.7311        learning rate 0.001 (constant)
```

### Step 3.20 — validation pass
**Example: iteration 10**
```text
Tensorboard/Validation at iteration 10: Total Loss 1.3548, Classification Loss 1.3548, Accuracy 0.2500
```

### Step 3.21 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The interval sums are divided by the number of batches; the frame counter is kept. The header gives `Sequences Per Second` and the real frames per second, sent to MLflow as `train/frame_throughput`. Accuracies print as percentages with their change in points, losses with 4 decimals.

**Example: the averages of the first interval** (MLflow `train/frame_throughput` 9.7405)
```text
after:  {'interval_batches': 4, 'total_loss': 1.4036, 'classification_loss': 1.4036, 'accuracy': 0.125, 'interval_frames': 112}
```

**Example output: iteration 12**
```text
|   Iteration: 12    |     Total Sequences: 48      |    Interval Sequences: 16   |   Throughput: 1.4 Sequences Per Second   |    9.5 Frames Per Second    |
--------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     1.3814    -0.0102 Best       <- green: lower than at iteration 8
Moving Average of Valid Total Loss                     :     1.3548
...
Moving Average of Train Accuracy                       :    31.25 %    -6.25 %
Moving Average of Valid Accuracy                       :    25.00 %
--------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:11.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
```
The interval of iteration 8 shows 0.3 Sequences Per Second (54 seconds), since it holds the evaluation of iteration 6.

### Step 3.22 — end of the loop
Iteration 12 is already saved and evaluated, so the loop does nothing more (`classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 200 Test Sequences · 2026-10-18 08:48 ───────────────      <- the evaluation of iteration 12
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```

---

## Stage 4 — Test Evaluation of a Checkpoint

> Steps 4.1 to 4.9 show the evaluation of iterations 6 and 12 of the Stage 3 run on its 200 test sequences (`per_class_report: True`, `top_k: 2`, `save_predictions: True`, error images on with at most 2 per pair). Both checkpoints predict `moving_down` everywhere. These outputs predate the curves, nine-digit probabilities, `logit_temperature` and the recall intervals. Steps 4.10 to 4.13 show the current files: iteration 500 of the 128 pixel README run (`M2O_IN-KAR-128_..._HFLAT.64_MC4`, since deleted; one `moving_left` sequence predicted `moving_down`), evaluated again on 2026-11-12 on CPU with `scripts/evaluation/evaluate_checkpoint.py` (batch size 16), so its files are those of `<run>/Checkpoint_Evaluation/`.

### Step 4.1 — `run_test_evaluation()`
The evaluation of the classification (`classification_workflow.md`, Stage 4), every sample a whole sequence, never augmented. The test loader shuffles, so the row order of `predictions.csv` changes between evaluations.

**Example output: the progress bar** (200 sequences, batch size 4)
```text
Test Evaluation Iteration 6: 100%|██████████| 50/50 [00:43<00:00,  1.16it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The lists of the classification: probabilities, labels and identifiers of every batch, the `Errors` folder and the error counter per class pair. Its `logits` list stays empty: only the image classification keeps them, for its temperature explorer.


### Step 4.3 — `prepare_batch()` + `predict_batch()`
The preparation of training (Step 3.11); `predict_batch()` runs the model with the mask and without labels, so no loss is computed.


### Step 4.4 — `accumulate_evaluation_batch()`
The classification hook keeps the probabilities (`float32`), labels and sequence identifiers on the CPU, and runs `dump_errors()` while the frames are still in the batch.

### Step 4.5 — `dump_errors()` → `reserve_error_image()` + `build_input_image()`
As for the classification, every wrong prediction is written in `Errors/<TRUE>-TO-<PREDICTED>/`, at most `maximum_per_class_pair` per pair. `build_input_image()` lays out the real frames on a numbered grid of 4 columns (4 pixel gray gap) under a caption bar with both classes and the predicted probability. The height follows the length (72 pixels for 1 row, 140 for 2, 208 for 3). The file is named after the sequence and that probability.

**Example: the error images of the checkpoint of iteration 12**
```text
build_input_image(sample_index=0) -> (72, 276, 3) uint8      1 row x 4 columns of 64 x 64 frames (a sequence of 2 frames)
Evaluation/Iteration_12/Errors/
├── moving_left-TO-moving_down/     test_000059_0.279.png   test_000192_0.278.png
├── moving_right-TO-moving_down/    test_000131_0.277.png   test_000175_0.279.png
└── moving_up-TO-moving_down/       test_000031_0.278.png   test_000053_0.279.png
image sizes: 96, 164 or 232 x 276 pixels (1, 2 or 3 rows of frames under the 24 pixel caption bar)
report line: "Error Images : 6 in 3 class pairs, at most 2 per pair"
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
For every fixed sample sequence of Step 2.11, the classification hook draws the frame grid of `build_input_image()` under a caption bar with the true class, the predicted class and its probability, green when right and red when wrong. It is saved as `Samples/<sequence name>.png` and sent to TensorBoard.

**Example: the sample images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Samples/   test_000000.png  test_000001.png  ...  test_000199.png   (8 images)
sizes: 96 x 276 (4 frames or less), 164 x 276, 232 x 276 (9 to 12 frames)
```

### Step 4.7 — `compute_evaluation_results()`
The results of the classification (`classification_workflow.md`, Step 4.8). The headline metric, used by Stage 5 and MLflow to choose the best checkpoint, is the accuracy. The current results also hold `Macro ROC AUC`, `Macro Average Precision`, the three curve images and 20 class scalars (`ROC AUC/<class>` and `Average Precision/<class>` added).

**Example: the results of iteration 6**
```text
evaluated_samples = '200 Test Sequences'
metrics           = {'Accuracy': 0.235, 'Balanced Accuracy': 0.25, 'Macro F1': 0.0951, 'Top-2 Accuracy': 0.515}
notes             = ['Error Images : 6 in 3 class pairs, at most 2 per pair', 'Sample Images : 8 fixed test sequences']
headline_metric   = 'Accuracy'
class_scalars     = 12: Precision/moving_left, Recall/moving_left, F1/moving_left, Precision/moving_right, ...
```

### Step 4.8 — `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()`
The confusion matrix counts the sequences of every (true, predicted) pair; the metrics come from it (`classification_workflow.md`, Steps 4.9 and 4.10). Above 1,000 classes (`MAXIMUM_DENSE_CONFUSION_CLASSES`) only the non-zero cells are kept, in `confusion_cells.csv`.

**Example: `Evaluation/Iteration_6/confusion_matrix.csv`** (the same at iteration 12)
```text
true \ predicted,moving_left,moving_right,moving_up,moving_down
moving_left,0,0,0,52
moving_right,0,0,0,45
moving_up,0,0,0,56
moving_down,0,0,0,47
```

**Example: how the numbers of iteration 6 are obtained**
```text
accuracy          = 47 / 200                                  = 0.235    (only the moving_down sequences are right)
moving_down       precision = 47 / 200 = 0.235    recall = 47 / 47 = 1.0    F1 = 2 x 0.235 x 1.0 / (0.235 + 1.0) = 0.3806
the 3 other classes   precision = 0 (never predicted)    recall = 0.0    F1 = 0.0
balanced accuracy = (3 x 0.0 + 1.0) / 4                       = 0.25
macro F1          = (3 x 0.0 + 0.3806) / 4                    = 0.0951
```

### Step 4.9 — `compute_top_k_accuracy()`
The share of sequences whose true class is among the 5 most probable (`TOP_K_ACCURACY_CLASSES`). With 5 classes or fewer, as here, every sequence counts as right.

**Example output** (recorded with `top_k: 2`)
```text
Top-2 Accuracy = 0.5150 at iteration 6, 0.4600 at iteration 12      (about 2 / 4 = 0.5: chance level for 4 classes)
```

### Step 4.10 — `find_class_thresholds()`
For every class, the probability threshold keeping a precision of at least `target_precision` with the highest recall, saved in `class_thresholds.json` (`build_class_thresholds_file()`). A list of target precisions gives the others under `additional_targets`. `logit_temperature` 1.0 says the thresholds come from the plain softmax: the sequence tasks have no temperature explorer.

**Example: `Evaluation/Iteration_6/class_thresholds.json`** (recorded before `logit_temperature`)
```text
{
    "target_precision": 0.99,
    "classes": {
        "moving_left": null,
        ...
        "moving_down": {
            "threshold": 0.2703714668750763,
            "recall": 0.02127659574468085            <- 1 of the 47 moving_down sequences
        }
    }
}
```
At iteration 12 only `moving_left` reaches the precision: threshold 0.22249144315719604, recall 0.019230769230769232 (1 of 52).

In the 128 pixel README run, the current file starts with `"logit_temperature": 1.0` and gives `moving_left` the threshold 0.21318694949150085 at recall 1.0.

### Step 4.11 — `build_class_metrics_rows()` + `write_class_metrics_file()` + `save_confusion_matrix_image()` + `save_score_curve_images()`
`class_metrics.csv` is written at every evaluation. With `per_class_report`, the evaluation also writes `confusion_matrix.png`, `roc_curves.png`, `precision_recall_curves.png` and `precision_recall_thresholds.png`, and adds the `roc_auc` and `average_precision` columns (`classification_workflow.md`, Steps 4.14 and 4.15). `recall_lower` and `recall_upper` hold the exact 95 % interval of every recall. The images write percentages; the CSV keeps 0 to 1.

**Example: the current layout, `Checkpoint_Evaluation/Iteration_500/class_metrics.csv` of the 128 pixel README run** (on CPU, thresholds within 0.0002 of its GPU evaluation)
```text
class_index,class,samples,precision,recall,recall_lower,recall_upper,f1,roc_auc,average_precision,threshold_at_target_precision,recall_at_target_precision
0,moving_left,49,1.000000,0.979592,0.891458,0.999483,0.989691,1.000000,1.000000,0.213187,1.000000
1,moving_right,52,1.000000,1.000000,0.931518,1.000000,1.000000,1.000000,1.000000,0.520205,1.000000
...
```

### Step 4.12 — `write_predictions_file()`   (only with `save_predictions`)
One row per test sequence, in test loader order. `describe_sample()` names a sequence by its name and its first and last frames. Probabilities have nine significant digits, so thresholds found again from the file match `class_thresholds.json`.

**Example: `Evaluation/Iteration_6/predictions.csv`** (200 rows; first rows; the checkpoint then held 5 steps, so the probabilities are almost the same)
```text
sequence,first_frame,last_frame,true_class,predicted_class,correct,probability_moving_left,probability_moving_right,probability_moving_up,probability_moving_down
test_000080,test_000080/000000.png,test_000080/000001.png,moving_up,moving_down,0,0.228245,0.248930,0.252467,0.270359
test_000077,test_000077/000000.png,test_000077/000001.png,moving_down,moving_down,1,0.228247,0.248922,0.252460,0.270371
```


### Step 4.13 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The report block and the file links (`classification_workflow.md`, Steps 4.18 and 4.19); the same text without links is appended to `Results.txt`. Then `update_evaluation_curves()` (added on 2026-11-18) writes the row of the iteration into `Evaluation/evaluation_summary.csv`, draws `evaluation_curves.png` again (accuracy per iteration, best ringed) and prints its link. The example below is the 128 pixel README run evaluated again with main fa41856, in the current layout.

**Example output: the block of iteration 500** (shortened)
```text
Iteration 500 · 200 Test Sequences · 2026-11-12 20:53 ──────────────────────────────────────────────────────────────────
  Accuracy        99.50 %    Balanced Accuracy         99.49 %
  Macro F1        99.47 %    Top-5 Accuracy            100.00 %
  Macro ROC AUC   100.00 %   Macro Average Precision   100.00 %
  Error Images : 1 in 1 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                                           Average   Threshold     Recall
   Class          Samples   Precision     Recall         F1    ROC AUC   Precision     @ P99 %    @ P99 %
  ────────────────────────────────────────────────────────────────────────────────────────────────────────
   moving_down         46     97.87 %   100.00 %    98.92 %   100.00 %    100.00 %      0.9532   100.00 %
   ...

  Per Class Recall
   ...
  - Results : file:///.../Checkpoint_Evaluation/Results.txt
  - Confusion Matrix Table : file:///.../Checkpoint_Evaluation/Iteration_500/confusion_matrix.csv
  ...
  - Sample Images : file:///.../Checkpoint_Evaluation/Iteration_500/Samples
```
The short run of Stage 3 gives at iteration 12 `Accuracy 23.50 %`, `Top-2 Accuracy 46.00 %` and only `moving_down` with a non-zero F1 (38.06 %). The current evaluation writes 26 scalars to `Tensorboard/Test` (6 global + 20 per class) and 12 images (8 samples, `Confusion Matrix`, `ROC Curves`, `Precision Recall Curves`, `Precision Recall Thresholds`), and sends the matching `test/` and `test_class/` metrics and figures to MLflow.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of `scripts/evaluation/evaluate_checkpoint.py` on the Stage 3 run (checkpoints 3, 6, 9 and 12), with `iteration = "all"`, `batch_size = 16` and the first 40 test sequences (`labels.json` reduced, `data/` linked). They were recorded on 2026-10-18 with `top_k: 2`, before the curves and the recall intervals. Paths and rates are shown as the current code writes them, in `<run>/Checkpoint_Evaluation/`.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The script of every task (`classification_workflow.md`, Stage 5). Give the run as `--experiment_folder` and, if needed, another labelled split as `--test_data_folder` (with `labels.json` and `data/`). `--maximum_error_images_per_class_pair` turns the error images on; `--target_precisions` (several such as `0.99 0.95 0.90`) replaces the target precision of the run; the NMS settings are for detection only. The evaluation is written in `Checkpoint_Evaluation/` (`Checkpoint_Evaluation_split_<parent folder>_<split folder>/` for another split). Sequences longer than the run's `maximum_sequence_length` are left out with the red warnings of Step 2.9.

### Step 5.2 — trainer construction
The script loads the run's `training_configuration.yaml` with the classes of its `label_dictionary.json`, points it to `Checkpoint_Evaluation/`, the reduced split and the batch size, keeps `run_name`, calls `disable_pretrained_settings()` and builds a `SequenceClassificationTrainer`.

**Example: the split line of the launch information**
```text
- Test Data Folder        : file:///.../test_reduced (40 sequences, 276 frames, 40 sequences of at most 50 frames)
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'sequence_classification' : 4338612f50f144f9bbaa372d56581b2e (Browse : mlflow ui --backend-store-uri sqlite:///.../example_runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
So that the checkpoint to keep can be chosen, every evaluation adds its row to `Checkpoint_Evaluation/evaluation_summary.csv` (increasing iteration order) and draws `evaluation_curves.png` (Step 4.13). The summary gives their links, then the best checkpoint by headline metric; on a tie every tied iteration is listed. The columns are the scalars of every evaluation (`build_summary_columns()`), the per-class ones only with `per_class_report`; the current code writes `Top-5 Accuracy`, `Macro ROC AUC`, `Macro Average Precision` and the per-class `ROC AUC` and `Average Precision`.

**Example output** (every checkpoint predicts `moving_down` for the 40 sequences)
```text
Evaluation Summary Of 4 Checkpoints : file:///.../example_runs/M2O_IN-KAR-64_..._MC4/Checkpoint_Evaluation/evaluation_summary.csv
  - Evaluation Curves : file:///.../example_runs/M2O_IN-KAR-64_..._MC4/Checkpoint_Evaluation/evaluation_curves.png
  - Best Checkpoint : iterations 12, 9, 6, 3 tie (Accuracy 12.50 %)

iteration,Accuracy,Balanced Accuracy,Macro F1,Top-2 Accuracy,Precision/moving_left,Recall/moving_left,F1/moving_left, ...
3,0.125,0.25,0.05555555555555555,0.425,0.0,0.0,0.0, ...
...
12,0.125,0.25,0.05555555555555555,0.45,0.0,0.0,0.0, ...
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
Each checkpoint prints a block like Step 4.13. The sample sequences are chosen from the 40 sequences of this split (identifiers 0, 1, 2, 4, 11, 37, 38 and 39).

**Example output: the block of the checkpoint of iteration 12 on the 40 sequences** (shortened)
```text
Iteration 12 · 40 Test Sequences · 2026-10-18 08:49 ────────────────────────────────────────────────────────────────────
  Accuracy   12.50 %   Balanced Accuracy   25.00 %
  Macro F1   5.56 %    Top-2 Accuracy      45.00 %
  Error Images : 6 in 3 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                             Threshold    Recall
   Class          Samples   Precision     Recall        F1     @ P99 %   @ P99 %
  ───────────────────────────────────────────────────────────────────────────────
   moving_left         10      0.00 %     0.00 %    0.00 %      0.2225   10.00 %
   moving_right        13      0.00 %     0.00 %    0.00 %           —         —
   moving_up           12      0.00 %     0.00 %    0.00 %      0.2452    8.33 %
   moving_down          5     12.50 %   100.00 %   22.22 %           —         —
```
