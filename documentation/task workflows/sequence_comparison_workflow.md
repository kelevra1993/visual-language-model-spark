# Sequence Comparison Workflow

This document follows one sequence comparison run (CMP: one class per pair of sequences, a before member and an after member) from `main()` to the evaluation of a checkpoint. Each stage has one step per function, with a short explanation and a **real** output. Steps shared with other tasks are explained in `sequence_classification_workflow.md` and `classification_workflow.md`. The steps specific to pairs are explained here in full.

The examples come from real runs on CPU, so that the GPU stayed free:
- **Stages 1 and 2**: `vision_training/configurations/sequence_comparison/configuration_synthetic_pairs.yaml` (2 classes `loss_of_object` and `same_number_of_objects`, at most 50 frames per member, 512 pixels, batch size 16, mean pooling, difference fusion). Its dataset (`/home/robert_kelevra/Data/sequence_data/synthetic_object_loss_pairs`) then held members of 2 to 6 frames of 128 pixels. It was generated again on 2026-11-18 (2 to 12 frames of 512 pixels, 2 to 6 objects in the before member), and the README run trained on it.
- **Stages 3, 4 and 5**: the same configuration at 64 pixels, batch size 4, 12 iterations, a checkpoint every 6 (an intermediate one every 3), a summary every 4, a validation batch every 10, error images and MLflow on. The network then predicts `same_number_of_objects` for every pair (loss near ln 2 = 0.693): the metrics are poor on purpose.
- **Differences from the current code**: the Stage 3 run (2026-10-10) shows `Top-1 Accuracy` (`top_k: 1`), where k is now fixed at 5 (`Top-5 Accuracy`, always 100.00 % for 2 classes). It ran in float32, where the code now runs bfloat16 autocast (Step 3.9). It predates the ROC and precision-recall curves. The current evaluation outputs come from the README run in `~/Data/tutorial_runs/readme_screenshots/`, `CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2`, trained for 1,000 iterations on 2026-11-19, checkpoints 500 and 1000 evaluated into its `Checkpoint_Evaluation` folder.
- Rates are shown as the code prints them (`46.00 %`); files, TensorBoard and MLflow keep values between 0 and 1.

Console paths are terminal hyperlinks (`format_path_link()` of `vision_training/utilities/os_utilities.py`), written here as `file://` URIs.

## Read This First

Before following the stages, this summary gives the purpose of the task, its code, how to run it and what to watch.

The task (`task: "sequence_comparison"`, run name code `CMP`) learns one class for a pair of sequences. In the synthetic data, the class says whether the after member lost one of the objects of the before member. Both members go through the same backbone, frame pooling and temporal module, so their vectors are comparable. `Fusion.type` combines the two member vectors into the input of the CLS head (see [Configuration Keys To Know First](#configuration-keys-to-know-first)).

Every member is read whole, up to `Data.sequence_settings.maximum_sequence_length` frames, and keeps its own length. In a batch, both members of every pair are padded to the longest member of the batch, stored one after the other along the frame axis, then split again just before the model.

Choose it rather than M2O when the class compares two separately recorded sequences (two visits), and rather than M2M or SEL when the output is one class for the pair. `SequenceComparisonTrainer` is `SequenceClassificationTrainer` with `LABEL_TYPE = "pair"` and `MEMBERS_PER_SAMPLE = 2`. The loss (cross entropy), batch accuracy and evaluation are those of CLS, computed on pairs.

### The Task At A Glance

To find your way in the stages below, this block follows the pairs of the synthetic configuration from their manifest to the probabilities of the export.

```text
<split>/labels.json {"id": 1, "name": "train_000000", "before": [9 frames], "after": [9 frames], "category_id": 2}
<split>/data/train_000000/before/000000.png ..., after/000000.png ...          label_mapping.json {"1": 0, "2": 1}
SequencePairDataset: a pair with a member over 50 frames left out; both members read and padded to the longer
   member T_i ─► images [2 T_i, 3, 512, 512] uint8 (before, then after), sequence_mask [2 T_i], labels
collate_padded_sequences(members_per_sample=2): every member padded to the longest member T of the batch
   ─► images [16, 2T, 3, 512, 512], sequence_mask [16, 2T] ─► prepare_batch ─► augment_batch (one draw per pair)
split_pair_batch ─► before_images, after_images [16, T, 3, 512, 512], before_mask, after_mask [16, T]
SequenceComparator.forward(before_images, after_images, before_mask, after_mask, labels)
   encode_members: both members as one batch of 32 sequences ─► encode_frames on the N real frames ─► Backbone
      B1 [N, 16, 256, 256]  B2 [N, 32, 128, 128]  B3 [N, 32, 64, 64]  B4 [N, 64, 32, 32]  B5 [N, 64, 32, 32]
      ─► average pooling [N, 64] ─► [32, T, 64], zero vectors on the padding
   attend_across_members: the cross attention of the frames of each member to the other (unchanged here)
   summarise_members: MaskedMeanPooling over the real frames of every member ─► before [16, 64], after [16, 64]
   combine_member_vectors (difference): [before, after, after - before] ─► [16, 192]
   ClassificationHead: Dropout, Linear 64, ReLU, Dropout, Linear 2 ─► class_logits [16, 2] ─► softmax
loss: cross entropy = total_loss; every 500 iterations the CLS evaluation on the 200 test pairs ─► Results.txt
export: InferenceSequenceComparator, before_images [B, T1, 3, 512, 512], after_images [B, T2, 3, 512, 512]
   ─► class_probabilities [B, 2], every member encoded and summarised on its own
```

### Files To Know

Should you need to change one part of the task, this table gives the file holding it and the functions to read first.

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/sequence_comparison_trainer.py` | `SequenceComparisonTrainer`: `describe_split`, `split_pair_batch`, `forward_pass`, `predict_batch`, `build_sample_frame_grid` |
| Inherited sequence hooks | `vision_training/trainers/sequence_classification_trainer.py` | `SequenceClassificationTrainer`: `build_dataloaders`, `prepare_batch`, `augment_batch`, `get_member_length`, `build_input_image`, `build_input_windows` |
| Loss, batch metric and evaluation (inherited) | `vision_training/trainers/classification_trainer.py` | `ClassificationTrainer`: `build_model`, `compute_batch_metrics`, `dump_errors`, `build_sample_image`, `compute_evaluation_results` |
| Dataset and loaders | `vision_training/data/sequence/sequence_pair_dataset.py`, `sequence_dataset.py`, `vision_training/data/serialized_records.py`, `vision_training/data/loaders.py`, `vision_training/data/split_description.py` | `SequencePairDataset`, `SerializedRecords` (`find_record`), `split_pair_members`, `PAIR_MEMBERS`, `read_sequence_frames`, `collate_padded_sequences`, `get_sequence_dataloaders`, `count_split_pairs` |
| Model | `vision_training/models/sequence/sequence_comparator.py`, `frame_encoder.py` | `SequenceComparator`: `encode_members`, `summarise_members`, `forward`; `encode_frames` |
| Fusion | `vision_training/models/sequence/member_fusion.py` | `MemberFusion`: `attend_across_members`, `attend_to_member`, `combine_member_vectors`; `FUSION_TYPES`, `SEQUENCE_NORMALIZATION_SCOPES` |
| Temporal modules | `vision_training/models/sequence/temporal_modules.py` | `build_temporal_module`, `MaskedMeanPooling` and the 6 other types |
| Frame pooling, head and loss | `vision_training/models/classification/classification_head.py`, `vision_training/losses/classification_losses.py` | `build_global_pooling`, `compute_pooled_features`, `ClassificationHead`, `compute_classification_loss` |
| Metrics and evaluation | `vision_training/evaluation/classification/` | the functions of CLS: `compute_confusion_matrix`, `find_class_thresholds`, `compute_class_score_curves`, ... |
| Configuration checks | `vision_training/configuration_processing/sequence_checks.py` | `check_sequence_comparison_configuration`, `check_fusion_configuration`, `check_sequence_settings`, `check_temporal_model_configuration` |
| Run name | `vision_training/configuration_processing/run_name.py` | `build_sequence_comparison_sections`, `build_fusion_tokens` |
| Synthetic data generator | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset`, `generate_object_loss_pair`, `draw_shapes_frame` |
| Export | `vision_training/models/sequence/inference_sequence_comparator.py`, `vision_training/deployment/export_specifications.py`, `export_model.py`, `input_contract.py` | `InferenceSequenceComparator` (model `sequence_comparator`, inputs `before_images` and `after_images`); `build_input_contract` (`preprocessing.json`, layout `NTCHW` for both inputs) |
| Export sizes at launch | `vision_training/deployment/export_size_estimates.py` | `estimate_export_sizes`, `probe_inference_graph`, `print_export_size_summary` |
| Standalone evaluation | `scripts/evaluation/evaluate_checkpoint.py`, `vision_training/evaluation/checkpoint_evaluation.py` | `evaluate_checkpoints`, `summarise_checkpoint_evaluations` |

### Run It

In order to train on the synthetic pairs, generate them, launch the training, then evaluate saved checkpoints again if needed.

```bash
# Generate the synthetic pairs (2,000 / 200 / 200 pairs, members of 2 to 12 frames of 512 pixels, 2 to 6 objects in the before member)
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=sequence_comparison --output_folder="$HOME/Data/sequence_data/synthetic_object_loss_pairs"

# Train (set experiment_folder, the three data folders and Data.label_mapping.mapping_file_path first)
uv run python -m vision_training.main --configuration_path="vision_training/configurations/sequence_comparison/configuration_synthetic_pairs.yaml"

# Evaluate checkpoints into <run>/Checkpoint_Evaluation/ (settings in the main block of scripts/evaluation/evaluate_checkpoint.py)
uv run python -m scripts.evaluation.evaluate_checkpoint
```

The main block of `evaluate_checkpoint.py` takes `run_name`, `experiment_folder` (the run folder), `iteration` (None, an integer or `"all"`), `test_data_folder`, `batch_size`, `track_evaluation`, `save_predictions` (None keeps the setting of the run), `maximum_error_images_per_class_pair` (turns error images on) and `target_precisions` (a list such as `[0.99, 0.95]`, the first one the main one).

### Configuration Keys To Know First

If one wishes to change what the comparator learns or how it reads the pairs, these keys of the synthetic configuration matter most.

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `Fusion.type` | `"difference"` | `[before, after, after - before]`, 192 features (`FDIF`); `"concatenation"` gives `[before, after]`, 128 features (`FCAT`); `"cross_attention"` adds an attention of the frames of each member to the other member before the temporal module (`FXA<heads>`) |
| `Fusion.number_heads`, `Fusion.dropout_rate` | `4`, `0.0` | read by the cross attention only, required for every type; with `"cross_attention"`, the channels of the last backbone block must be a multiple of the heads |
| `Fusion.sequence_normalization_scope` | `"member"` | with `Backbone.normalization.feature_map_normalization: "sequence"`, normalizes every member by its own frames, or both members of a pair together with `"pair"` (`SNP`); unused with the batch normalization of this configuration |
| `Data.sequence_settings.maximum_sequence_length` | `50` | the longest member: a pair with a longer member is left out of its split with a red warning (`T50`) |
| `TemporalModel.type` | `"mean_pooling"` | shared by both members (`MEAN`); any of the 7 types, never causal |
| `ClassificationHead.global_pooling` | `"average"` | 64 features per frame; with it the README run reaches a test accuracy of 88.50 % at iteration 500 and 95.00 % at iteration 1000 |
| `Data.image_settings.size` | `512` | the size the generator writes the frames at, so they are read without resizing |
| `Data.data_augmentation.shuffle_frames` | `activated: False, probability: 0.25` | when on, shuffles every member on its own, each with its own draw |

### What A Run Writes

Whenever a run is launched, or saves and evaluates a checkpoint, its run folder receives the files below.

| File or folder | Content |
|---|---|
| `Results.txt` | One block per evaluated checkpoint, headline `Accuracy` (Step 4.11). |
| `Evaluation/Iteration_<n>/` | The files of CLS (Steps 4.8 to 4.10), without `temperature_explorer.html` nor `class_logits.npz`, which only the image classification writes (`saves_temperature_explorer()` of `classification_trainer.py`). `predictions.csv` has one row per test pair. `Samples/<pair>.png` and `Errors/<TRUE>-TO-<PREDICTED>/<pair>_<score>.png` show the before grid above the after grid, 4 columns (Steps 4.5 and 4.6). |
| `data.json` | The `pairs` of every class in every split, and the frames of both members of every pair. |
| `Evaluation/evaluation_summary.csv`, `evaluation_curves.png` | One row per evaluated iteration; the accuracy curve with the best iteration ringed (`update_evaluation_curves()` of `base_trainer.py`, Step 4.11). |
| `Input_Samples/` | At every launch, the first 10 pairs of every split drawn as in Step 2.22 into `Training/`, `Validation/` and `Test/`, with `samples.json` (`{"pair": "train_000000"}`). |
| `Tensorboard/Test` | 16 scalars and 12 images (Step 4.11); `Train` and `Validation` as in M2O. |

The console summary gives the `Pairs Per Second` and the real `Frames Per Second` of both members, sent to MLflow as `train/throughput` and `train/frame_throughput`.

### Pitfalls

Suppose a comparison run behaves unexpectedly: these behaviours are the first to check.

- Training and evaluation encode both members as one padded batch; the export encodes every member on its own (Step 3.10). In evaluation mode both paths agree: four test pairs padded to 6 frames per member, or alone without masks, differ by 6e-08 (random weights, 128 pixels, CPU).
- The synthetic configuration keeps the left-right flip as its only geometric augmentation. A crop, zoom, shift or rotation can cut an object or push it out, changing the count of a pair (comment of `Data.data_augmentation.types` in `configuration_synthetic_pairs.yaml`). Grayscale stays off too: an object matching the luminance of the background would vanish.
- Every `Fusion` setting is required and checked for every fusion type, even when unused (`check_fusion_configuration()` of `sequence_checks.py`).
- Blocks 2 and 4 of the synthetic backbone change the channel count in their first layer and add no residual shortcut; only blocks 3 and 5 add one (`count_residual_additions()` of `convolution_block.py`).
- `Iterations Per Epoch` and the split descriptions count the pairs before the label mapping. A pair of an unmapped category is left out by the dataset only (`count_split_pairs()` of `split_description.py`, left as it is).
- The test loader shuffles the pairs, so the row order of `predictions.csv` changes between evaluations (`build_native_loader()`).

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

To be able to follow one comparison run from its launch to the evaluation of a checkpoint, this tree gives every function in the order it runs.

```text
main()                                    reads --configuration_path, builds the trainer, starts the loop
├── load_experiment_configuration()       load_configuration, build_label_dictionary, check_configuration
│       └── check_sequence_comparison_configuration()   + check_sequence_settings, check_fusion_configuration
├── get_trainer_class(task)  -> SequenceComparisonTrainer
├── SequenceComparisonTrainer(...)  [STAGE 2]   LABEL_TYPE = "pair", MEMBERS_PER_SAMPLE = 2, SAMPLE_NAME = "Pairs"
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() / DataAugmenter() / setup_training_paths()
│       ├── build_dataloaders() -> get_sequence_dataloaders(label_type="pair") -> SequencePairDataset
│       ├── build_model() -> SequenceComparator: backbone + pooling, temporal module, MemberFusion, ClassificationHead
│       ├── initialise_pretrained_components() / build_optimizer() / MetricTracker(counts_frames=True)
│       ├── print_experiment_information() / print_summary() (Member Fusion panel) / export sizes
│       └── view_input_batches() / restore_last_model()
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── save_input_samples() / TensorBoard writers / start_experiment_tracking()
        └── for each training_iteration up to number_iterations (the total of the run):
            ├── get_next_batch()   SequencePairDataset.__getitem__() + collate_padded_sequences(members_per_sample=2)
            ├── run_model_iteration()   prepare_batch, augment_batch (one draw per pair), forward_pass:
            │       split_pair_batch -> encode_members -> attend_across_members -> summarise_members
            │       -> combine_member_vectors -> classification_head -> compute_classification_loss
            ├── backward / step_optimizer / save_model / validation pass / interval summary
            └── run_test_evaluation() on the multiples of weight_saving_iterations   -> [STAGE 4]

run_test_evaluation(iteration)  [STAGE 4]   the evaluation of CLS on pairs, then update_evaluation_curves()
scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]   same as every task
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs of the loader on `configuration_synthetic_pairs.yaml`.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The comparison adds the `TemporalModel` and `Fusion` sections to those of the classification. Its `sequence_settings` hold only `maximum_sequence_length`: every member is fed whole, and a pair with a longer member is left out.

**Example: the sections and the fusion settings**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, Fusion, ClassificationHead
Fusion:                   {'type': 'difference', 'number_heads': 4, 'dropout_rate': 0.0, 'sequence_normalization_scope': 'member'}
```

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'TemporalModel', 'Fusion', 'ClassificationHead']
number_iterations: '1e10'      <- still a string
```

### Step 1.4 — `build_label_dictionary()`
**Example: the label dictionary of the object loss pairs**
```text
dict: {1: 0, 2: 1}       2 entries, 2 distinct classes
```

### Step 1.5 — `check_configuration()`
After the common checks, `check_sequence_comparison_configuration()` runs the checks of the sequence classification (Step 1.5 of `sequence_classification_workflow.md`), then `check_fusion_configuration()`.

**Example: `Fusion.type: "cross_attention"`, `number_heads: 3` and `sequence_normalization_scope: "frame"`**
```text
Invalid Configuration File : file:///.../configuration_wrong.yaml
  - 'Fusion.sequence_normalization_scope' must be one of ['member', 'pair'], got 'frame'.
  - The channels of the last backbone block (64) must be a multiple of 'Fusion.number_heads' (3) for the cross attention.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="sequence_comparison") -> <class 'vision_training.trainers.sequence_comparison_trainer.SequenceComparisonTrainer'>
```

---

## Stage 2 — Trainer Initialization

> The examples below are real outputs of building the trainer on `configuration_synthetic_pairs.yaml`. The construction took 0.6 s on CPU.

### Step 2.1 — `SequenceComparisonTrainer.__init__()`
No constructor of its own: the sequence classification trainer (`sequence_classification_workflow.md`, Step 2.1) with other class attributes. `LABEL_TYPE = "pair"` makes the loaders read pairs. `MEMBERS_PER_SAMPLE = 2` tells the shared code that every sample holds two equal-length members, before then after, along its frame axis.

Its other attributes: `SAMPLE_NAME = 'Pairs'`, `COUNTS_FRAMES = True`, `self.top_k = 5` (`TOP_K_ACCURACY_CLASSES`).

### Step 2.2 — `BaseTrainer.__init__()`
As for every task (`classification_workflow.md`, Step 2.2).

### Step 2.3 — `get_device()`
It prints `For Training We Will Be Using device: cpu` (forced here; a normal run on this machine prints `cuda`).

### Step 2.4 — `get_run_name()` → `build_run_name()`
`build_sequence_comparison_sections()` adds the maximum sequence length (`T50`), the temporal module (`MEAN`, see `sequence_classification_workflow.md`, Step 2.4), then the fusion: `FCAT`, `FDIF` or `FXA<heads>`, followed by `SNP` when the `sequence` normalization covers both members of a pair.

**Example output**
```text
CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2
```

### Step 2.5 — `DataAugmenter()`
**Example: the settings of the configuration**
```text
activated True   probability 0.4   independent_frames False   shuffle_frames {'activated': False, 'probability': 0.25}
4 of 28 types activated: random_left_right_flip, brightness_contrast_saturation, random_noise, random_haze
```

### Step 2.6 — `setup_training_paths()`
As for every task (`classification_workflow.md`, Step 2.6). `data.json` is written once the loaders exist (`save_data_record()`).

**Example: the run folder once the trainer is built**
```text
CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2/
├── Tensorboard/
├── Weights/
├── data.json                      807,388 bytes      the pairs of every class, then the frames of both members
├── label_dictionary.json
└── training_configuration.yaml    18,921 bytes
```

### Step 2.7 — `build_dataloaders()`
The hook of the sequence classification, with `label_type="pair"`. The class names are the categories of the training manifest: `['loss_of_object', 'same_number_of_objects']`.

### Step 2.8 — `get_sequence_dataloaders()`
Every split gets a `SequencePairDataset` (Step 2.9) and `collate_padded_sequences()` with `members_per_sample=2`. The validation split gets one only when the training reads it (`BaseTrainer.builds_validation_loader`: `validation_iteration_ratio` above 0, as the 25 here, or a diagnosis mode ending the program); otherwise its loader is `None`.

**Example: size of the three splits**
```text
train       2000 pairs -> 125 batches of 16 (one epoch = 125 iterations)
validation   200 pairs ->  13 batches of 16
test         200 pairs ->  13 batches of 16
```

### Step 2.9 — `SequencePairDataset.__init__()`
It reads the `categories` and `pairs` of the split manifest: `id`, `name`, the ordered `before` and `after` frames (relative to `data/`) and `category_id`. A member is never cut nor subsampled. This table gives the cases where a pair is left out.

| Case | Message |
|---|---|
| A member longer than `maximum_sequence_length` | a red warning per pair, then a red total for the split |
| An unmapped category or an empty member | a yellow note with their number and the manifest link |

The kept pairs are numbered in manifest order, shuffled once and packed in `SerializedRecords` (Step 2.9 of `classification_workflow.md`). With `maximum_sequence_length: 50`, every pair is kept.

**Example: `maximum_sequence_length: 5`** (pairs with a member of 6 frames left out; shortened)
```text
720 Pairs Of The Training Split Have A Member Longer Than The Maximum Sequence Length (5) And Are Left Out Of The Training.
- Train Data Folder       : file:///home/.../train (2000 pairs, 16096 frames, 1280 pairs of members of at most 5 frames)
- Iterations Per Epoch    : 80
```
The run name then holds `T5`, so the run gets its own folder.

**Example: the first pair after shuffling, and its sample**
```text
pair:       {'identifier': 1997, 'name': 'train_001997', 'before': [2 frames], 'after': [3 frames], 'label': 1}
dataset[0]: images (6, 3, 512, 512) torch.uint8        before then after, both padded to 3 frames
            sequence_mask (6,) [True, True, False, True, True, True]     <- 1 black padding frame in the before member
```

### Step 2.10 — `build_native_loader()`
**Example output** (recorded with 8 workers and pinned memory; every loader now has 1 worker and pins only on a discrete GPU)
```text
len(loader) = 125   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
The 8 sample pairs are shared by the 2 classes (4 each), evenly spread over their sorted identifiers.

**Example output**
```text
sample_image_identifiers  -> [0, 3, 65, 68, 123, 137, 198, 199]
```

### Step 2.12 — `build_model()`
The classification hook creates the `SequenceComparator` from `TASK_MODEL_CLASSES["sequence_comparison"]` (Step 2.13) and sets its class weights (`get_class_counts()`, with `class_weighting: "balanced"`).

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          MaskedMeanPooling                0 parameters
model.member_fusion            MemberFusion                     0 parameters
model.classification_head      ClassificationHead          12,482 parameters
model.class_weights            buffer [2] torch.float32 [1.0, 1.0]
```
With a mean pooling and a difference fusion, only the backbone and the head have weights.

### Step 2.13 — `SequenceComparator.__init__()`
It holds the shared backbone and pooling (one vector per frame), the member fusion, the shared temporal module (one vector per member) and the head (the logits of the pair). Shared weights make the difference of the two vectors measure a change. Its forward pass is Steps 3.9 to 3.13.

### Step 2.14 — `build_backbone()` + `build_global_pooling()`
The backbone and pooling of the sequence classification (`sequence_classification_workflow.md`, Step 2.14), here for 512 pixel frames with `average` pooling: 64 features per frame. Only blocks 3 and 5 add a residual shortcut (see Pitfalls). The block shapes are those of [The Task At A Glance](#the-task-at-a-glance); the final backbone output is `[1, 64, 32, 32]`.

### Step 2.15 — `build_temporal_module()` → `MaskedMeanPooling`
The temporal module of the sequence classification (`sequence_classification_workflow.md`, Step 2.15), shared by both members. It averages the real frame vectors of a member, ignoring their order: `output_features = 64`.

### Step 2.16 — `MemberFusion.__init__()`
It combines the two member vectors: 128 features for `"concatenation"`, 192 for `"difference"` and `"cross_attention"` (Read This First). With `"cross_attention"` it also builds one multi-head attention layer (`Fusion.number_heads`, `Fusion.dropout_rate`) with a layer normalization.

**Example output** (difference fusion, no weights)
```text
MemberFusion()   type difference   output_features 192
sequence_normalization = False   normalization_scope = member      (the backbone uses the batch normalization)
```

### Step 2.17 — `ClassificationHead.__init__()`
**Example output**
```text
Dropout(0.2) -> Linear(192, 64) -> ReLU -> Dropout(0.2) -> classifier Linear(64, 2)
```

### Step 2.18 — `initialise_pretrained_components()`
The backbone, temporal module and head can come from other runs. The fusion has no `pretrained` settings. A backbone deeper than its source loads its first blocks only, the next ones random and never frozen (`classification_workflow.md`, Step 2.19).

**Example: a run from scratch**
```text
pretrained_checkpoint_paths = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
```

### Step 2.19 — `build_optimizer()` + `get_tracked_metrics()` + `MetricTracker.__init__()`
**Example output**
```text
Adam, base_learning_rates [0.001]     tracked: Total Loss, Classification Loss, Accuracy     sample_name 'Pairs', counts_frames True
```

### Step 2.20 — `print_experiment_information()` (→ `describe_split()` → `count_split_pairs()`)
`count_split_pairs()` counts the pairs, the frames of both members and the kept pairs (both members non-empty and within `maximum_sequence_length`; the category is not checked). The kept pairs make an epoch.

**Example output**
```text
- Train Data Folder       : file:///home/.../synthetic_object_loss_pairs/train (2000 pairs, 16096 frames, 2000 pairs of members of at most 50 frames)
- Iterations Per Epoch    : 125
```

### Step 2.21 — `print_summary()` / `print_model_size_summary()`
The comparator adds the Member Fusion panel and a fusion line in the size panel. The Export Sizes panel follows: its forward pass is that of `InferenceSequenceComparator`, so backbone operators are counted once per member (once in all with the `pair` scope) and weights once.

**Example output** (shortened)
```text
Member Fusion: Type difference, Number Heads 4, Dropout Rate 0.0, Sequence Normalization Scope member, Output Features 192
Model Size:  Member Fusion 0 parameters   Total Model 196,130 parameters, 0.79 MB
```

### Step 2.22 — `view_input_batches()` → `show_input_samples()`
With `diagnosis_modes.view_input_images`, the overridden `build_sample_frame_grid()` puts the before frames above the after frames, captioned `0 before`, `0 after`. The panel gives the class, the real frames (`Real frames : 3 + 2`) and the frame order (`Frame order : shuffled, showing frames 2, 0, 1 | 5, 4`). It needs a display; `Input_Samples/` gets the same grids (Step 3.1).

### Step 2.23 — `restore_last_model()`
**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```

---

## Stage 3 — Training Loop

> The examples below are real outputs of the shortened CPU run described at the top (64 pixels, batch size 4, 12 iterations). Its run name is `CMP_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2`; with batches of 4 pairs, an epoch is 500 iterations.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (`classification_workflow.md`, Step 3.1). First, `save_input_samples()` writes `Input_Samples/` (the README run: `30 images written in 5.43 seconds`).

The table below shows what happens at each iteration of the run, after its training step.

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | ✔ | nothing: iteration 12 is saved |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |
| Test evaluation | | | | | | ✔ | | | | | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()` / `start_experiment_tracking()`
**Example output**
```text
trackers: {'start_time': ..., 'interval_batches': 0, 'total_loss': 0.0, 'classification_loss': 0.0, 'accuracy': 0.0, 'interval_frames': 0}
MLflow Run Created In The Experiment 'sequence_comparison' : 037372ca4aac463685e61176d80e825e
```

### Step 3.3 — `save_model()` → `save_checkpoint()`
Saved after the training step of iterations 3, 6, 9 and 12, so the checkpoint of iteration N holds N steps (`classification_workflow.md`, Step 3.6).

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt ... Weights/Iteration_12/model_000012.pt   2,441,735 bytes each
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
```
`mlflow_run_id` is `None` in checkpoints 3 and 6 (the MLflow run was not open yet). Current checkpoints also hold `pretrained_backbone_blocks`.

### Step 3.4 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration (`classification_workflow.md`, Step 3.7). See Stage 4.

### Step 3.5 — `get_next_batch()` → `SequencePairDataset.__getitem__()` + `collate_padded_sequences()`
`__getitem__()` reads each member with `read_sequence_frames()` (`sequence_classification_workflow.md`, Step 3.7), pads the shorter one with black frames and concatenates before and after. `collate_padded_sequences(members_per_sample=2)` pads both members to the longest member of the batch (mask `False`), so the batch splits into two equal halves.

**Example: the first batch** (the longest member has 5 frames)
```text
images               [4, 10, 3, 64, 64] torch.uint8      [B, 2 x 5, C, H, W]
sequence_mask        [4, 10] torch.bool
                     [[1, 1, 1, 0, 0,  1, 1, 1, 1, 0],        <- 3 before frames, 4 after frames
                      [1, 1, 0, 0, 0,  1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0,  1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1,  1, 1, 0, 0, 0]]        <- the longest member: 5 before frames
labels               [1, 1, 1, 0]
```

### Step 3.6 — `MetricTracker.count_frames()`
It counts the real frames of both members: `interval_frames = 122` for iterations 1 to 4 (16 pairs).

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
`prepare_batch()` of the sequence classification moves the frames, labels and `[B, 2T]` mask to the device.

**Example: the first training batch**
```text
images [4, 10, 3, 64, 64] torch.uint8 -> torch.float32
```

### Step 3.8 — `augment_batch()`
The augmentation of the sequence classification (`sequence_classification_workflow.md`, Step 3.12): one draw per sample, so both members get the same crop, flip and colour change. `shuffle_frames()` shuffles each member on its own (`get_member_length()`: `2T / MEMBERS_PER_SAMPLE`).

In the first batch (`probability: 0.4`), no pair was drawn.

### Step 3.9 — `forward_pass()` → `split_pair_batch()` → `SequenceComparator.forward()`
`split_pair_batch()` splits the frames and mask into two halves along the frame axis (`split_pair_members()`).

**Example: the first batch**
```text
before_images (4, 5, 3, 64, 64)   after_images (4, 5, 3, 64, 64)   before_mask (4, 5)   after_mask (4, 5)
```

**Example: the dictionary returned in training mode** (recorded in float32; bfloat16 autocast now gives bfloat16 `class_logits`, the loss staying float32)
```text
class_probabilities   [4, 2] torch.float32   [[0.4927, 0.5073], [0.5466, 0.4534], [0.5240, 0.4760], [0.4938, 0.5062]]
classification_loss   0.7293 (scalar)                    <- close to ln 2 = 0.693
```

### Step 3.10 — `encode_members()` → `encode_frames()`
This table shows how the members are encoded with and without masks.

| Case | Encoding |
|---|---|
| With masks (training, evaluation) | One batch of `2B` sequences through `encode_frames()` (`sequence_classification_workflow.md`, Step 3.14); padding frames get zero vectors. With the `sequence` normalization and `"pair"`, a before member and its after member share one group. |
| Without masks (deployment) | Each member on its own, so lengths may differ. With the `sequence` normalization and `"pair"`, both members are concatenated along one frame axis. |

**Example: the first batch**
```text
encode_members -> before (4, 5, 64), after (4, 5, 64)
zero vectors per member: before [2, 3, 2, 0], after [1, 3, 1, 3]      <- the padding frames of every member
```

### Step 3.11 — `member_fusion.attend_across_members()`
Only for `"cross_attention"`: the frames of each member attend to the real frames of the other, with a residual connection and a layer normalization; padding frames are set back to zero. Here the vectors pass unchanged.

### Step 3.12 — `summarise_members()`
The shared temporal module summarises both members as one batch of `2B` masked sequences, then splits the result back.

**Example: the first batch**
```text
summarise_members -> before (4, 64), after (4, 64)
```

### Step 3.13 — `member_fusion.combine_member_vectors()` + `ClassificationHead.forward()` + `compute_classification_loss()`
**Example: the first batch**
```text
combine_member_vectors -> (4, 192)      [before, after, after - before]
classification_head    -> (4, 2)        classification_loss = 0.7293
```

### Step 3.14 — `compute_batch_metrics()` + `MetricTracker.accumulate()` + `log_to_tensorboard()`
**Example: the Train curves**
```text
Total Loss   1: 0.7293, 2: 0.6944, ..., 11: 0.5966, 12: 0.5344
Accuracy     1: 0.2500, 2: 0.5000, ..., 11: 1.0000, 12: 1.0000
```

### Step 3.15 — `training_loss.backward()` + `step_optimizer()` + validation pass
**Example output**
```text
gradient norm: iteration 1: 1.7714   iteration 2: 1.1179   iteration 3: 1.4277        learning rate 0.001 (constant)
Tensorboard/Validation at iteration 10: Total Loss 0.6939, Classification Loss 0.6939, Accuracy 0.5000
```

### Step 3.16 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The header gives the pairs per second and the real frames of both members per second.

**Example output: iteration 12** (shortened)
```text
|   Iteration: 12    |     Total Pairs: 48      |    Interval Pairs: 16   |  Throughput: 12.2 Pairs Per Second   |    96.8 Frames Per Second   |
Moving Average of Train Accuracy                       :    87.50 %   +31.25 % Best
Moving Average of Valid Accuracy                       :    50.00 %
```

### Step 3.17 — end of the loop
Iteration 12 is already saved and evaluated, so the loop saves and evaluates nothing more (`classification_workflow.md`, Step 3.25).

It prints `Last Checkpoint Saved At Iteration 12. Exiting Training.`

---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of checkpoints 6 and 12 of the Stage 3 run on its 200 test pairs (`top_k: 1`, error dumps at 2 per class pair, MLflow on). Both predict `same_number_of_objects` for every pair; checkpoint 6 was then saved after 5 steps. They predate the curves and per-class recall intervals (`classification_workflow.md`, Steps 4.14 and 4.15); Steps 4.9 and 4.11 show the current layout from the README run.

### Step 4.1 — `run_test_evaluation()`
The evaluation of the classification (`classification_workflow.md`, Stage 4), every sample being a pair.

The progress bar reads `Test Evaluation Iteration 6: 100%|██████████| 50/50` (200 pairs, batch size 4).

### Step 4.2 — `create_evaluation_accumulator()`
At the end of the loop it holds 50 tensors each of `probabilities` (the first `[4, 2]`), `labels` and `sample_identifiers`.

### Step 4.3 — `prepare_batch()` + `predict_batch()`
`predict_batch()` runs the comparator on the split members with their masks and without labels. After 5 steps, every pair gets almost the same probabilities.

**Example: the first test batch** (members of 4 frames at most)
```text
images [4, 8, 3, 64, 64]   class_probabilities [4, 2], every row [0.4922, 0.5078]
```

### Step 4.4 — `accumulate_evaluation_batch()`
The classification hook keeps the probabilities, labels and pair identifiers on the CPU, and `dump_errors()` runs while the frames are in the batch.

### Step 4.5 — `dump_errors()` → `reserve_error_image()` + `build_input_image()`
`build_input_image()` draws the before grid above the after grid (Step 2.22). The file is named after the pair and the probability of the predicted class.

**Example: the error images of iteration 12**
```text
build_input_image(sample_index=0) -> (144, 276, 3) uint8      a before row and an after row of 64 x 64 frames
Evaluation/Iteration_12/Errors/loss_of_object-TO-same_number_of_objects/   test_000039_0.524.png   test_000073_0.524.png
report line: "Error Images : 2 in 1 class pairs, at most 2 per pair"
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
The frame grids of every sample pair under a caption bar (true class, predicted class, probability), green when right, red when wrong.

**Example: iteration 12**
```text
Evaluation/Iteration_12/Samples/   test_000000.png  test_000003.png  ...  test_000199.png    (8 files)
build_sample_image -> (168, 302, 3) uint8
```

### Step 4.7 — `compute_evaluation_results()`
The results of the classification. With `top_k: 1`, the top-k accuracy equals the accuracy; with k fixed at 5, every pair counts as right for 2 classes.

**Example: the results of iteration 6**
```text
evaluated_samples = '200 Test Pairs'
metrics           = {'Accuracy': 0.46, 'Balanced Accuracy': 0.5, 'Macro F1': 0.3151, 'Top-1 Accuracy': 0.46}
headline_metric   = 'Accuracy'
class_scalars     = 6: Precision/loss_of_object, Recall/loss_of_object, F1/loss_of_object, Precision/same_number_of_objects, ...
```
Current results also hold `Macro ROC AUC`, `Macro Average Precision`, the three curve images and 10 class scalars.

### Step 4.8 — `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()`
The matrix and metrics of CLS (`classification_workflow.md`, Steps 4.9 and 4.10); above 1,000 classes (`MAXIMUM_DENSE_CONFUSION_CLASSES`), only non-zero cells in `confusion_cells.csv`.

**Example: `Evaluation/Iteration_6/confusion_matrix.csv`**
```text
true \ predicted,loss_of_object,same_number_of_objects
loss_of_object,0,108
same_number_of_objects,0,92
```

**Example: how the numbers of iteration 6 are obtained**
```text
accuracy                = 92 / 200                                  = 0.46
same_number_of_objects  precision = 92 / 200 = 0.46    recall = 1.0    F1 = 2 x 0.46 x 1.0 / (0.46 + 1.0) = 0.6301
loss_of_object          precision = 0 (never predicted)   recall = 0.0   F1 = 0.0
balanced accuracy       = (0.0 + 1.0) / 2                           = 0.5
macro F1                = (0.0 + 0.6301) / 2                        = 0.3151
```

### Step 4.9 — `find_class_thresholds()` + `write_class_metrics_file()`
As for CLS (`classification_workflow.md`, Steps 4.13 and 4.14): `recall_lower` and `recall_upper` hold the exact (Clopper-Pearson) 95 % interval of each recall, and `class_thresholds.json` starts with `"logit_temperature": 1.0`.

**Example: `Evaluation/Iteration_6/class_thresholds.json`** (Stage 3 run)
```text
{"target_precision": 0.99, "classes": {"loss_of_object": {"threshold": 0.4922524094581604, "recall": 0.027777777777777776},
                                       "same_number_of_objects": null}}
```

**Example: the current layout, `Checkpoint_Evaluation/Iteration_500/class_metrics.csv` of the README run** (16 `loss_of_object` pairs predicted `same_number_of_objects` and 7 the other way; the threshold of `same_number_of_objects` stays empty, its precision never reaching 99 %)
```text
class_index,class,samples,precision,recall,recall_lower,recall_upper,f1,roc_auc,average_precision,threshold_at_target_precision,recall_at_target_precision
0,loss_of_object,98,0.921348,0.836735,0.748450,0.903701,0.877005,0.942177,0.959691,0.906770,0.714286
1,same_number_of_objects,102,0.855856,0.931373,0.863703,0.971965,0.892019,0.942177,0.885416,,
```

### Step 4.10 — `write_predictions_file()`   (only with `save_predictions`)
`describe_sample()` names a pair by its manifest name.

**Example: `Evaluation/Iteration_12/predictions.csv`** (200 rows, 15,039 bytes; first rows)
```text
pair,true_class,predicted_class,correct,probability_loss_of_object,probability_same_number_of_objects
test_000073,loss_of_object,same_number_of_objects,0,0.476044,0.523956
test_000148,same_number_of_objects,same_number_of_objects,1,0.476041,0.523959
```

### Step 4.11 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The report block and the links of the written files (`classification_workflow.md`, Steps 4.18 and 4.19). The block is also appended to `Results.txt`.

**Example: the current layout, the first block of `Checkpoint_Evaluation/Results.txt` of the README run** (trained on GPU, error images off; at iteration 1000 the accuracy reaches 95.00 %, all 10 errors being `loss_of_object` pairs predicted `same_number_of_objects`)
```text
Iteration 500 · 200 Test Pairs · 2026-11-19 21:34 ──────────────────────────────────────────────────────────────────────
  Accuracy        88.50 %   Balanced Accuracy         88.41 %
  Macro F1        88.45 %   Top-5 Accuracy            100.00 %
  Macro ROC AUC   94.22 %   Macro Average Precision   92.26 %
  Sample Images : 8 fixed test pairs

  Per Class Metrics
                                                                                  Average   Threshold    Recall
   Class                    Samples   Precision    Recall        F1   ROC AUC   Precision     @ P99 %   @ P99 %
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
   loss_of_object                98     92.13 %   83.67 %   87.70 %   94.22 %     95.97 %      0.9068   71.43 %
   same_number_of_objects       102     85.59 %   93.14 %   89.20 %   94.22 %     88.54 %           —         —

  Per Class Recall
                                                   95 %      95 %
   Class                    Samples    Recall     Lower     Upper
  ────────────────────────────────────────────────────────────────
   loss_of_object                98   83.67 %   74.84 %   90.37 %
   same_number_of_objects       102   93.14 %   86.37 %   97.20 %
```

The evaluation writes 16 scalars to `Tensorboard/Test` (6 global + 10 per class) and 12 images (8 samples, `Confusion Matrix`, `ROC Curves`, `Precision Recall Curves`, `Precision Recall Thresholds`). MLflow gets the `test/` metrics (with `test/best_Accuracy`), 10 `test_class/` scalars and the three curve images.

After the tracking, `update_evaluation_curves()` writes the row of the iteration into `Evaluation/evaluation_summary.csv` and draws `evaluation_curves.png` again (best iteration ringed). It prints `  - Evaluation Curves : file:///.../evaluation_curves.png` under the block and, with tracking on, sends both files to the root of the MLflow run.

---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of `scripts/evaluation/evaluate_checkpoint.py` on the Stage 3 run (checkpoints 3, 6, 9 and 12) with `iteration = "all"`, `batch_size = 16` and a test folder of the first 40 test pairs.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task, with the settings of [Run It](#run-it) (`classification_workflow.md`, Stage 5). `test_data_folder` may point to another labelled split (a pairs `labels.json` and `data/`). The output goes to `Checkpoint_Evaluation/` of the run, or `Checkpoint_Evaluation_split_<parent folder>_<split folder>/` for another split. A pair with a member over the `maximum_sequence_length` of the run is left out, with the red warnings of Step 2.9.

### Step 5.2 — trainer construction
The trainer is built from the `training_configuration.yaml` of the run, with the classes of its `label_dictionary.json`, the run folder name as `run_name` and pretrained transfer off (`disable_pretrained_settings()`, `classification_workflow.md`, Step 5.2).

**Example: the split line**
```text
- Test Data Folder        : file:///.../test_reduced (40 pairs, 339 frames, 40 pairs of members of at most 50 frames)
```

### Step 5.3 — `get_checkpoint_iterations()`
**Example output**
```text
evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
**Example output**
```text
MLflow Run Created In The Experiment 'sequence_comparison' : 7f5220908af54c13aaf92eb8ec7f147b
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
Every evaluation adds its row to `Checkpoint_Evaluation/evaluation_summary.csv` (increasing iteration order) and redraws `evaluation_curves.png` (Step 4.11). The summary gives their links, then the best checkpoint.

**Example output** (every checkpoint predicts `same_number_of_objects` for the 40 pairs, so the four tie)
```text
Evaluation Summary Of 4 Checkpoints : file:///.../runs/CMP_IN-KAR-64_..._MC2/Checkpoint_Evaluation/evaluation_summary.csv
  - Evaluation Curves : file:///.../runs/CMP_IN-KAR-64_..._MC2/Checkpoint_Evaluation/evaluation_curves.png
  - Best Checkpoint : iterations 12, 9, 6, 3 tie (Accuracy 52.50 %)
```
The columns are the scalars of every evaluation (`build_summary_columns()`), the per-class ones only with `per_class_report`.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
Each checkpoint prints `Evaluating : <checkpoint path>`, then the block of Step 4.11 (iteration 9 on the 40 pairs: `Accuracy 52.50 %`). The sample pairs come from these 40 pairs, so they differ from those of Stage 4.
