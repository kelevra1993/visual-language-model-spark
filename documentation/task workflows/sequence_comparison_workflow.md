# Sequence Comparison Workflow

This document follows one sequence comparison run (CMP: one class per pair of sequences, a before member and an after member) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The sequence comparison trainer is the sequence classification trainer with pairs instead of single sequences: the steps it shares with the sequence classification are explained in `sequence_classification_workflow.md`, and the steps shared by every task in `classification_workflow.md`; here they keep a short explanation and the real output of the comparison run, and the steps specific to the pairs (member length limit, member padding, pair batches, member split, member fusion) are explained in full.

The examples come from real runs on CPU, so that the GPU stayed free for another training:
- **Stages 1 and 2**: `vision_training/configurations/sequence_comparison/configuration_synthetic_pairs.yaml` (synthetic object loss pairs, 2 classes `loss_of_object` and `same_number_of_objects`, members of 2 to 6 frames, a maximum sequence length of 50 frames per member, frames read at 512 pixels, batch size 16, a mean pooling and a difference fusion), with the run folder redirected to a temporary folder. The dataset is the one the configuration points to (`/home/robert_kelevra/Data/sequence_data/synthetic_object_loss_pairs`) as it was when they were recorded, generated before the generator drew 2 to 12 frames of 512 pixels per member: its members held 2 to 6 frames of 128 pixels. It was generated again on 2026-11-18, with members of 2 to 12 frames of 512 pixels, 2 to 6 objects in the before member, textured or low contrast backgrounds and objects that may overlap or be cut by the border; the README run trained on it. The dataset of `configuration_example.yaml` is not on this machine.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers (a setting when these examples were recorded; the code fixes 1 worker per loader), with the error images and the MLflow tracking turned on. After 12 iterations the network predicts `same_number_of_objects` for every test pair (2 classes: a loss near ln 2 = 0.693), so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps. This run was recorded on 2026-10-10, before the ROC and precision-recall curves joined the evaluation (2026-10-20): the steps say where the current code prints more, and the README run of `~/Data/tutorial_runs/readme_screenshots/` (the tutorial runs folder of `tutorials/tutorial_project.yaml`; `CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2`, the synthetic configuration trained from scratch for 1,000 iterations on 2026-11-19 on the dataset generated on 2026-11-18, its checkpoints 500 and 1000 evaluated again right after its training into its `Checkpoint_Evaluation` folder) gives the outputs of the evaluation with the curves. The run of Stage 3 was recorded when the k of the top-k accuracy was a setting (`top_k: 1`), so its reports show `Top-1 Accuracy`, equal to the accuracy, where the current code, whose k is fixed at 5, writes `Top-5 Accuracy`, always 100.00 % for the 2 classes of the synthetic pairs. It also ran its training and validation forward passes in float32 (`mixed_precision: False`, a setting then), which run in bfloat16 autocast since 2026-10-29 (Step 3.9), as in the README run. The rates of the recorded console summaries, reports and best checkpoint lines are shown as the current code writes them, as percentages with two decimals (`46.00 %`, a change of the console summary in percentage points such as `+31.25 %`), the thresholds and losses keeping their decimals; the files, the result dictionaries, TensorBoard and MLflow keep their recorded values between 0 and 1.

In the console, every file or folder path is a terminal hyperlink (`format_path_link()` of `vision_training/utilities/os_utilities.py`) showing the name of the file or folder, underlined; the examples write the `file://` URI the link opens, except where they show that name.

## Read This First

Before following the stages, this summary tells what the task is for, where its code lives, how to run it and what to watch. The sequence comparison task (`task: "sequence_comparison"`, run name code `CMP`) learns one class for a pair of sequences, a before member and an after member (in the synthetic data, whether the after member lost one of the objects of the before member). Both members go through the same backbone, frame pooling and temporal module, so that their vectors are comparable, and `Fusion.type` combines the two member vectors into the input of the fully connected head of CLS: `[before, after]` for `concatenation`, `[before, after, after - before]` for `difference` and `cross_attention`, the cross attention also letting the frames of each member attend to the frames of the other member before the temporal module. Every member is read whole, up to `Data.sequence_settings.maximum_sequence_length` frames, and keeps its own length; in a batch, both members of every pair are padded to the longest member of the batch and stored one after the other along the frame axis, then split again just before the model. Choose it rather than M2O when the class compares two separately recorded sequences (two visits), which may have different lengths, and rather than M2M or SEL when the output is one class for the pair and not a class or a choice per frame. `SequenceComparisonTrainer` is `SequenceClassificationTrainer` with `LABEL_TYPE = "pair"` and `MEMBERS_PER_SAMPLE = 2`: the loss (cross entropy), the batch accuracy and the evaluation are those of CLS, computed on pairs.

### The Task At A Glance

To find your way in the stages below, this block follows the pairs of the synthetic configuration from their manifest to the probabilities of the export.

```text
<split>/labels.json {"id": 1, "name": "train_000000", "before": [9 frames], "after": [9 frames], "category_id": 2}
<split>/data/train_000000/before/000000.png ..., after/000000.png ...          label_mapping.json {"1": 0, "2": 1}
SequencePairDataset: a pair with a member longer than maximum_sequence_length (50) left out; every member read
   (read_sequence_frames, 512) and padded to the longer member T_i ─► images [2 T_i, 3, 512, 512] uint8 (before,
   then after), sequence_mask [2 T_i], labels: class index of the pair
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
loss: classification_loss = cross entropy = total_loss; batch accuracy; every 500 iterations the evaluation of
   CLS on the 200 test pairs ─► Results.txt (Accuracy), Evaluation/Iteration_<n>/ (predictions.csv: one row
   per pair; Samples/ and Errors/: the grid of the before frames above the grid of the after frames)
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
| Export | `vision_training/models/sequence/inference_sequence_comparator.py`, `vision_training/deployment/export_specifications.py`, `export_model.py`, `input_contract.py` | `InferenceSequenceComparator` (model name `sequence_comparator`, inputs `before_images` and `after_images`); `build_input_contract`, the input contract of the run written as `preprocessing.json` next to the model and into the metadata of `model.onnx`, its layout `NTCHW` for both inputs |
| Export sizes at launch | `vision_training/deployment/export_size_estimates.py` | `estimate_export_sizes`, `probe_inference_graph`, `print_export_size_summary` |
| Standalone evaluation | `scripts/evaluation/evaluate_checkpoint.py`, `vision_training/evaluation/checkpoint_evaluation.py` | `evaluate_checkpoints`, `summarise_checkpoint_evaluations` |

### Run It

In order to train on the synthetic pairs, generate them first, then launch the training; the third command evaluates saved checkpoints again.

```bash
# 1. Synthetic dataset: the sequence_comparison task (2,000 / 200 / 200 pairs, members of 2 to 12 frames of 512
#    pixels, 2 to 6 objects in the before member), its task and dataset folder being required
uv run python -m vision_training.data.synthetic.create_synthetic_dataset --task=sequence_comparison --output_folder="$HOME/Data/sequence_data/synthetic_object_loss_pairs"

# 2. Training: set experiment_folder, the three data folders and Data.label_mapping.mapping_file_path first
uv run python -m vision_training.main --configuration_path="vision_training/configurations/sequence_comparison/configuration_synthetic_pairs.yaml"

# 3. Evaluation of checkpoints: set run_name, experiment_folder (the run folder), iteration (None, an integer or
#    "all"), test_data_folder, batch_size, track_evaluation, save_predictions, maximum_error_images_per_class_pair
#    and target_precisions in the main block of scripts/evaluation/evaluate_checkpoint.py, which writes in
#    <run>/Checkpoint_Evaluation/
uv run python -m scripts.evaluation.evaluate_checkpoint
```

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

- `Results.txt`: one block per evaluated checkpoint, `Iteration <n> · <N> Test Pairs · <date>`, with `Accuracy` (the headline metric), `Balanced Accuracy`, `Macro F1`, `Top-5 Accuracy` and, with `per_class_report`, `Macro ROC AUC`, `Macro Average Precision`, the per-class table and the `Per Class Recall` table (the recall of every class with its exact 95 % confidence interval over the test pairs, worst recall first); the note `Sample Images : 8 fixed test pairs`. Every rate is written as a percentage with two decimals (`88.50 %`), as for CLS, the thresholds staying between 0 and 1.
- `Evaluation/Iteration_<n>/`: the files of CLS (`confusion_matrix.csv`, above 1,000 classes `confusion_cells.csv` with its non-zero cells only, `class_metrics.csv`, `class_thresholds.json`, and with `per_class_report` `confusion_matrix.png` (up to 1,000 classes), `roc_curves.png`, `precision_recall_curves.png`, `precision_recall_thresholds.png`), `class_thresholds.json` starting with `"logit_temperature": 1.0`, the temperature of the probabilities its thresholds were found on, and no `temperature_explorer.html` nor `class_logits.npz`, which only the image classification writes (`saves_temperature_explorer()` of `classification_trainer.py`); `predictions.csv` with `save_predictions`, one row per test pair, its `pair` name before the classes and probabilities; `Samples/<pair>.png` and `Errors/<TRUE>-TO-<PREDICTED>/<pair>_<score>.png`, the numbered grid of the real before frames (`0 before`, ...) above the grid of the real after frames, 4 columns, under the caption `true -> predicted probability` (2068 pixels wide and 1064, 1580, 2096 or 2612 pixels high for the samples of the README run, each member taking 1 to 3 rows of 512 pixel frames).
- `data.json`: the `pairs` of every class in every split, and the frames of both members of every pair of every split.
- `Evaluation/evaluation_summary.csv` and `evaluation_curves.png`: every evaluation adds the row of its iteration (every test scalar) to the summary and draws the curves again, the accuracy of every evaluated iteration with the best one ringed (`update_evaluation_curves()` of `base_trainer.py`, Step 4.11).
- `Input_Samples/`: written before the first iteration of every launch, the first 10 pairs of every split by identifier drawn as the input viewer draws them (Step 2.22) into `Training/`, `Validation/` and `Test/`, the training ones augmented, with `samples.json` naming every pair (`{"pair": "train_000000"}`) (Step 3.1).
- `Tensorboard/Test`: 6 headline scalars, `Precision`, `Recall`, `F1`, `ROC AUC` and `Average Precision` of both classes, the images `Samples/<pair>`, `Confusion Matrix`, `ROC Curves`, `Precision Recall Curves` and `Precision Recall Thresholds`; `Train` and `Validation` as in M2O.
- The console summary gives the `Pairs Per Second` and the real `Frames Per Second` of both members, sent to MLflow as `train/throughput` and `train/frame_throughput`.

### Pitfalls

Suppose a comparison run behaves unexpectedly: these behaviours of the task are the first to check.

- The training and the evaluation encode both members as one batch of 2B sequences padded to the same length, while the export encodes and summarises every member on its own; with `feature_map_normalization: "sequence"` and the `"pair"` scope, the deployed path concatenates the two members along one frame axis instead (`encode_members()` and `summarise_members()` of `sequence_comparator.py`). In evaluation mode both paths give the same probabilities: four test pairs padded to 6 frames per member, or alone without masks, differ by 6e-08 (a new model of the configuration at 128 pixels on the CPU, random weights).
- The synthetic configuration keeps the left-right flip as its only geometric augmentation: a crop, a zoom, a shift or a rotation can cut an object at the border or push it out, changing the count of a pair keeping its objects (comment of `Data.data_augmentation.types` in `configuration_synthetic_pairs.yaml`); grayscale stays off for the same reason, an object matching the luminance of the background vanishing.
- `Fusion.number_heads`, `Fusion.dropout_rate` and `Fusion.sequence_normalization_scope` are required for every fusion type and checked even when unused, and `SNP` only appears with the `sequence` normalization of the backbone (`check_fusion_configuration()` of `sequence_checks.py`, `build_fusion_tokens()` of `run_name.py`).
- Blocks 2 and 4 of the synthetic backbone (16 to 32 and 32 to 64 channels, 2 layers each) add no residual shortcut, their first layer changing the channel count, so only blocks 3 and 5 add one (`count_residual_additions()` of `convolution_block.py`).
- `Iterations Per Epoch` and the split descriptions count the pairs before the label mapping, a pair of an unmapped category being left out by the dataset only (`count_split_pairs()` of `split_description.py`, left as it is).
- `predictions.csv` names a pair by its name only, without its frames, and the test loader shuffles the pairs, so the order of its rows changes from one evaluation to the next (`SequencePairDataset.describe_sample()`, `build_native_loader()`).

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

To be able to follow one comparison run from its launch to the evaluation of a checkpoint, this tree gives every function in the order it runs, with what it does.

```text
main()
│   Entry point of every task: reads the path of the YAML file from --configuration_path, loads it, looks up the
│   trainer of its task, builds it and starts the training loop. Give a sequence_comparison configuration to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, TemporalModel, Fusion, ClassificationHead), converts the types, adds label_mode and
│   │   label_dictionary to 'Data' and runs the checks.
│   │
│   ├── load_configuration(configuration_path)
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category id: class index}; one category per pair.
│   └── check_configuration()
│       ├── common checks                         experiment, data, backbone, optimization (see classification)
│       └── check_sequence_comparison_configuration()
│               ├── the data, head and evaluation checks of the classification
│               ├── check_sequence_settings()             maximum_sequence_length: a positive integer, the
│               │                                         longest member a pair may have
│               ├── check_temporal_model_configuration()  any pooling or ordered type, not causal
│               └── check_fusion_configuration()          type, heads and dropout of the cross attention, the
│                                                         frames sharing the sequence normalization ("member", "pair")
│
├── get_trainer_class(task)  -> SequenceComparisonTrainer
│
├── SequenceComparisonTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   The SequenceClassificationTrainer with LABEL_TYPE = "pair", MEMBERS_PER_SAMPLE = 2 and SAMPLE_NAME = "Pairs".
│   │
│   └── BaseTrainer.__init__()
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter()
│       │       The run name adds the maximum sequence length (T<frames>), the temporal module and the fusion (FCAT, FDIF,
│       │       FXA<heads>, SNP when both members share the sequence normalization) before the head and the classes.
│       ├── restore_run_label_dictionary() / check_resumed_data() / setup_training_paths()
│       │       The classes of a resumed run kept from its label_dictionary.json, a resumed run refused when a
│       │       labels.json changed since its data.json, then the run folder with the configuration and
│       │       label_dictionary.json (classification_workflow.md, Step 2.6).
│       ├── build_dataloaders()
│       │   └── get_sequence_dataloaders(label_type="pair")
│       │       │   One SequencePairDataset per split (None for the validation split when the training reads
│       │       │   no validation batch), and the collate collate_padded_sequences (members_per_sample=2).
│       │       │
│       │       ├── SequencePairDataset.__init__()
│       │       │       Every member is kept whole. The pairs with a member longer than maximum_sequence_length are
│       │       │       left out with a red warning per pair and a total per split; the pairs of unmapped categories
│       │       │       or with an empty member are left out with a note. Shuffled once.
│       │       └── build_native_loader()
│       ├── check_loaders_give_batches()                               refuses a split whose loader gives no batch
│       ├── save_data_record()                                         data.json: the pairs of every class, the frames
│       │                                                              of both members of every pair
│       ├── get_test_sample_classes() + choose_sample_identifiers()   the class of every test pair
│       ├── build_model()
│       │   └── SequenceComparator.__init__()
│       │       ├── build_backbone() + build_global_pooling()   one vector per frame of both members
│       │       ├── build_temporal_module()                     shared by both members, one vector per member
│       │       ├── MemberFusion()                              [before, after] or [before, after, after - before],
│       │       │                                               with a cross attention between the frames for
│       │       │                                               "cross_attention"
│       │       ├── ClassificationHead()
│       │       └── class_weights buffer
│       ├── initialise_pretrained_components()  Backbone, TemporalModel and ClassificationHead can come from other runs
│       ├── build_optimizer() / get_tracked_metrics() + MetricTracker.__init__(counts_frames=True)
│       ├── print_experiment_information() -> describe_split() -> count_split_pairs()
│       │       "N pairs, F frames, K pairs of members of at most M frames"; an epoch is K pairs.
│       ├── print_summary()                     adds the Member Fusion panel
│       ├── estimate_export_sizes() + print_export_size_summary()   size of every export and of the checkpoint
│       ├── view_input_batches() -> show_input_samples()   the before grid above the after grid
│       └── restore_last_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── save_input_samples()                 Input_Samples/: the first 10 pairs of every split, the before grid
        │                                        above the after grid, the training ones augmented (Step 3.1)
        ├── setup_tensorboard_writers() / initialize_training_loop_state() / start_experiment_tracking()
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch()                 a batch failing to load is dropped and the next one taken
        │   │   ├── SequencePairDataset.__getitem__() -> read_sequence_frames() for each member
        │   │   │       The before member then the after member along the frame axis, both padded to the longer one.
        │   │   └── collate_padded_sequences(members_per_sample=2)
        │   │           Pads both members of every pair to the longest member of the batch: [B, 2T, ...].
        │   ├── MetricTracker.count_frames()     real frames of both members
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch() / augment_batch()   one augmentation draw per pair, for its 2T frames
        │   │   ├── forward_pass() -> split_pair_batch() -> SequenceComparator.forward()   in bfloat16 autocast
        │   │   │   ├── encode_members()                 both members as one batch of sequences -> encode_frames()
        │   │   │   ├── member_fusion.attend_across_members()   cross attention only
        │   │   │   ├── summarise_members()              the shared temporal module on both members as one batch
        │   │   │   ├── member_fusion.combine_member_vectors()
        │   │   │   ├── classification_head              class_logits [B, K]
        │   │   │   └── compute_classification_loss()
        │   │   └── compute_batch_metrics() / MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()
        │   ├── save_model() on the multiples of weight_saving_iterations and intermediate_weight_saving_iterations
        │   ├── validation pass
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary()   (pairs and Frames Per Second)
        │   │   + report_dropped_batches()
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations, last    -> [STAGE 4]
        └── save_model() of the steps taken since the last checkpoint, if any, then shutdown_loader_workers() and
                finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
└── the evaluation of the classification on pairs: the error and sample images show the before grid above the after
    grid, and predictions.csv names every pair; with per_class_report, the ROC and precision-recall curves, the zoom
    on the class thresholds (precision_recall_thresholds.png) and the Per Class Recall table too; then the row of the
    iteration in Evaluation/evaluation_summary.csv and evaluation_curves.png drawn again (update_evaluation_curves()).

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `vision_training/configurations/sequence_comparison/configuration_synthetic_pairs.yaml`.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1).

### Step 1.2 — `load_experiment_configuration()`
The comparison has the `TemporalModel` and `Fusion` sections on top of the sections of the classification. Its `sequence_settings` hold only `maximum_sequence_length`, the most frames a member may have: every member is fed whole, and a pair with a longer member is left out.

**Example: the sections and the settings of the pairs**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, Fusion, ClassificationHead
Data keys after loading:  image_settings, sequence_settings, label_mapping, data_augmentation, label_mode, label_dictionary
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'target_precision': 0.99,
                           'save_predictions': True, 'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
TemporalModel:            {'type': 'mean_pooling', 'hidden_size': 128, 'number_layers': 1, 'number_heads': 4, 'kernel_size': 3, ...}
Fusion:                   {'type': 'difference', 'number_heads': 4, 'dropout_rate': 0.0, 'sequence_normalization_scope': 'member'}
ClassificationHead:       {'global_pooling': 'average', 'fully_connected': [64], 'dropout_rate': 0.2, 'output_mode': 'multi_class', ...}
```
Every setting of `Fusion` is required, even those its type does not read (`number_heads` and `dropout_rate` are read by the cross attention only, `sequence_normalization_scope` with the `sequence` normalization of the backbone only).

### Step 1.3 — `load_configuration()`
**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'TemporalModel', 'Fusion', 'ClassificationHead']
number_iterations: '1e10'      <- still a string
```

### Step 1.4 — `build_label_dictionary()`
**Example: the label dictionary of the object loss pairs**
```text
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '/home/.../sequence_data/synthetic_object_loss_pairs/label_mapping.json'}
dict: {1: 0, 2: 1}
2 entries, 2 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_sequence_comparison_configuration()`: the checks of the sequence classification (Step 1.5 of `sequence_classification_workflow.md`), then `check_fusion_configuration()`: the fusion type, the heads and the dropout of the cross attention, the normalization scope, and, for the cross attention, frame vectors (as wide as the last backbone block) that can be split between its heads.

**Example: output with `Fusion.type: "cross_attention"`, `number_heads: 3` and `sequence_normalization_scope: "frame"` in the YAML**
```text
------------------------------------------------------
Invalid Configuration File : file:///.../configuration_wrong.yaml
------------------------------------------------------
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

> The examples below are real outputs from building the trainer on `vision_training/configurations/sequence_comparison/configuration_synthetic_pairs.yaml` (frames resized to 512 pixels, a maximum sequence length of 50 frames per member, batch size 16), with the run folder redirected to a temporary folder. The whole construction took 0.6 s on CPU, and 0.1 s when built again on 2026-11-12.

### Step 2.1 — `SequenceComparisonTrainer.__init__()`
The comparison trainer has no constructor of its own: it is the sequence classification trainer (`sequence_classification_workflow.md`, Step 2.1) with other class attributes. `LABEL_TYPE = "pair"` makes the loaders read pairs, and `MEMBERS_PER_SAMPLE = 2` tells the shared code that every sample holds two members of the same length one after the other along its frame axis, the before member then the after member: the collate pads both to the longest member of the batch (Step 3.5), the augmentation shuffles each on its own (Step 3.8) and the input viewer and the evaluation images draw both (Step 4.5). The losses, the metrics and the evaluation are those of the classification, a pair being classified as a whole.

**Example: the attributes after the construction**
```text
self.sequence_settings          = {'maximum_sequence_length': 50}
self.LABEL_TYPE                 = 'pair'
self.MEMBERS_PER_SAMPLE         = 2
self.top_k                      = 5                     <- TOP_K_ACCURACY_CLASSES
self.save_predictions           = True
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
SAMPLE_NAME = 'Pairs'   COUNTS_FRAMES = True   TASK_NAME = 'sequence_comparison'
```

### Step 2.2 — `BaseTrainer.__init__()`
**Example: some attributes set by the constructor**
```text
self.device = device(type='cpu')   self.per_class_report = True   self.number_sample_images = 8   self.input_image_size = 512
self.batch_size = 16   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
`build_sequence_comparison_sections()` gives the maximum sequence length (`T50`: the most frames of a member), the temporal module (`MEAN`, see `sequence_classification_workflow.md`, Step 2.4), then the fusion: `FCAT` (concatenation), `FDIF` (difference) or `FXA<heads>` (cross attention), followed by `SNP` when the backbone uses the `sequence` normalization over both members of a pair. The head and the classes end the name.

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
As for every task (see `classification_workflow.md`, Step 2.6): a resumed run first keeps the classes of its `label_dictionary.json` (`restore_run_label_dictionary()`), then the run folder is created with the copy of the configuration and `label_dictionary.json`; before that, a resumed run whose `labels.json` changed since its `data.json` is refused (`check_resumed_data()`). The loaders that follow are refused when one of them gives no batch (`check_loaders_give_batches()`), the training loader leaves out its last incomplete batch only with a batch normalized classification head (`sequence_classification_workflow.md`, Step 2.8), which is not the case here, and `data.json` is written once they exist (`save_data_record()`).

**Example: the run folder once the trainer is built**
```text
CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2/
├── Tensorboard/
├── Weights/
├── data.json                      807,388 bytes      the pairs of every class in every split, then the frames of
│                                                     both members of every pair of the three splits
├── label_dictionary.json                             the class index of every category of the run
└── training_configuration.yaml    18,921 bytes      a copy of the configuration file
```

### Step 2.7 — `build_dataloaders()`
The hook of the sequence classification, with `label_type="pair"`; the class names are the categories of the training manifest.

**Example output**
```text
class_names = ['loss_of_object', 'same_number_of_objects']
```

### Step 2.8 — `get_sequence_dataloaders()`
With `label_type="pair"` every split gets a `SequencePairDataset` (Step 2.9), and the collate is `collate_padded_sequences()` with `members_per_sample=2`, which pads both members of every pair to the longest member of the batch. The validation split gets one only when the training reads it (`BaseTrainer.builds_validation_loader`: `validation_iteration_ratio` above 0, as the 25 of this configuration, or a diagnosis mode ending the program); otherwise `None` takes the place of its loader, and no validation pair is read.

**Example: size of the three splits**
```text
train       2000 pairs -> 125 batches of 16 (one epoch = 125 iterations)
validation   200 pairs ->  13 batches of 16
test         200 pairs ->  13 batches of 16
collate     functools.partial(<function collate_padded_sequences>, members_per_sample=2)
```

### Step 2.9 — `SequencePairDataset.__init__()`
It reads the manifest of the split: its `categories` and its `pairs`, each giving its `id`, its `name`, the ordered frames of its `before` and `after` members (paths relative to `data/`) and its `category_id`. A member is never cut nor subsampled: every member is fed whole. A pair is left out when one of its members has more than `maximum_sequence_length` frames (a red warning names the pair, then a red total is given for the split), and when its category is unmapped or a member is empty (a yellow note gives their number and the link of the manifest). The kept pairs are numbered in the order of the manifest (`identifier`), shuffled once, and packed like the classification samples (`SerializedRecords`, Step 2.9 of `classification_workflow.md`), so that the loader workers never copy them, every pair being found by its identifier (`pairs.find_record()`) for the evaluation outputs.

**Example: a manifest pair**
```text
labels.json:  {'id': 1, 'name': 'train_000000', 'before': ['train_000000/before/000000.png', ... 5 frames],
               'after': ['train_000000/after/000000.png', ...], 'category_id': 2}
              (2000 pairs, members of 2 to 6 frames; 2 categories {'id': 1, 'name': 'loss_of_object'}, ...)
```
With `maximum_sequence_length: 50`, every pair of the three splits is kept.

**Example: the output with `maximum_sequence_length: 5` in the YAML** (the pairs with a member of 6 frames are left out; shortened)
```text
Pair 'train_000000' Of The Training Split Has A Member Of 6 Frames, More Than The Maximum Sequence Length (5): It Is Left Out Of The Training.
Pair 'train_000001' Of The Training Split Has A Member Of 6 Frames, More Than The Maximum Sequence Length (5): It Is Left Out Of The Training.
...
720 Pairs Of The Training Split Have A Member Longer Than The Maximum Sequence Length (5) And Are Left Out Of The Training.
Pair 'validation_000002' Of The Validation Split Has A Member Of 6 Frames, More Than The Maximum Sequence Length (5): It Is Left Out Of The Validation.
...
70 Pairs Of The Validation Split Have A Member Longer Than The Maximum Sequence Length (5) And Are Left Out Of The Validation.
...
67 Pairs Of The Test Split Have A Member Longer Than The Maximum Sequence Length (5) And Are Left Out Of The Test.
...
- Train Data Folder       : file:///home/.../sequence_data/synthetic_object_loss_pairs/train (2000 pairs, 16096 frames, 1280 pairs of members of at most 5 frames)
- Validation Data Folder  : file:///home/.../sequence_data/synthetic_object_loss_pairs/validation (200 pairs, 1586 frames, 130 pairs of members of at most 5 frames)
- Test Data Folder        : file:///home/.../sequence_data/synthetic_object_loss_pairs/test (200 pairs, 1590 frames, 133 pairs of members of at most 5 frames)
- Iterations Per Epoch    : 80
```
The run name then holds `T5`, so that this run gets a folder of its own.

**Example: the first pair after shuffling, and the sample the loader gets from it**
```text
pair:       {'identifier': 1997, 'pair_identifier': 1998, 'name': 'train_001997',
             'before': ['train_001997/before/000000.png', 'train_001997/before/000001.png'],
             'after': ['train_001997/after/000000.png', 'train_001997/after/000001.png', 'train_001997/after/000002.png'], 'label': 1}
dataset[0]: images (6, 3, 512, 512) torch.uint8        the before member then the after member, both padded to 3 frames (the longer member)
            sequence_mask (6,) torch.bool [True, True, False,        <- 2 real before frames and 1 black padding frame
                                           True, True, True]         <- 3 real after frames
            labels () torch.int64 1, sample_identifiers 1997
get_sample_name(1997) = 'train_001997'
describe_sample(1997) = {'pair': 'train_001997'}
get_class_counts()    = [992, 1008]
```

### Step 2.10 — `build_native_loader()`
**Example output** (recorded with 8 workers, when the number of workers was a setting, and pinned memory; every loader now has 1 worker and pins its batches only on a discrete GPU)
```text
len(loader) = 125   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
The class of every test pair, so that the 8 sample pairs are shared by the 2 classes (4 each), each evenly spread over its sorted identifiers.

**Example output**
```text
get_test_sample_classes() -> 200 entries, e.g. {160: [0], 56: [0], 42: [0], 113: [0]}
sample_image_identifiers  -> [0, 3, 65, 68, 123, 137, 198, 199]
names                     -> test_000000, test_000003, test_000065, test_000068, test_000123, test_000137, test_000198, test_000199
```

### Step 2.12 — `build_model()`
The classification hook creates the `SequenceComparator` from the registry `TASK_MODEL_CLASSES["sequence_comparison"]` (Step 2.13) and sets its class weights (`get_class_counts()`: the pairs of every class, with `class_weighting: "balanced"`).

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          MaskedMeanPooling                0 parameters
model.member_fusion            MemberFusion                     0 parameters
model.classification_head      ClassificationHead          12,482 parameters
model.class_weights            buffer [2] torch.float32 [1.0, 1.0]
model.number_classes           2
```
With a mean pooling and a difference fusion, only the backbone and the head have weights.

### Step 2.13 — `SequenceComparator.__init__()`
The comparator classifies two sequences together (here, whether an object was lost between them). The shared backbone and global pooling turn every frame of both members into a vector, the member fusion lets the frames of each member attend to the other member (cross attention only), the shared temporal module summarises every member into one vector, the member fusion combines the two member vectors, and the classification head gives the logits of the pair. Both members go through the same weights, so that the difference of their vectors measures a change. It also reads the normalization of the backbone: with `sequence` normalization, the statistics are computed over the frames of each member (`sequence_normalization_scope: "member"`) or of both members of a pair (`"pair"`).

### Step 2.14 — `build_backbone()` + `build_global_pooling()`
The backbone and pooling of the sequence classification (`sequence_classification_workflow.md`, Step 2.14), built here for a 512 pixel frame, with the `average` pooling: 64 features per frame. Blocks 3 and 5 add a residual shortcut; blocks 2 and 4, whose first layer changes the channel count, add none with their 2 layers and list no `Residual Shortcuts` (`ConvolutionBlock.count_residual_additions()`).

**Example: the backbone panel of `print_summary()` for a 512×512 frame** (shortened)
```text
│  - Convolution Block 2                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 16, 256, 256]                                        │
│    - Output Shape : [1, 32, 128, 128]                                        │
│                                                                              │
│  - Convolution Block 3                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 32, 128, 128]                                        │
│    - Output Shape : [1, 32, 64, 64]                                          │
│  ...                                                                         │
│  - Convolution Block 5                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 64, 32, 32]                                          │
│    - Output Shape : [1, 64, 32, 32]                                          │
Final Backbone output shape: [1, 64, 32, 32]
```

### Step 2.15 — `build_temporal_module()` → `MaskedMeanPooling`
The temporal module of the sequence classification (`sequence_classification_workflow.md`, Step 2.15), shared by both members. The mean pooling averages the vectors of the real frames of a member, ignoring their order.

**Example output**
```text
temporal module: MaskedMeanPooling()
output_features = 64
```

### Step 2.16 — `MemberFusion.__init__()`
It combines the two member vectors into the input of the head: `[before, after]` for `"concatenation"` (2 × 64 = 128 features), `[before, after, after - before]` for `"difference"` and `"cross_attention"` (3 × 64 = 192 features), the difference making the change between the members explicit. With `"cross_attention"` it also builds one multi-head attention layer (`Fusion.number_heads`, `Fusion.dropout_rate`) with a layer normalization, through which the frames of each member attend to the frames of the other member before the temporal module.

**Example output** (difference fusion, no weights)
```text
MemberFusion()   type difference   output_features 192
sequence_normalization = False   normalization_scope = member      (the backbone uses the batch normalization)
```

### Step 2.17 — `ClassificationHead.__init__()`
**Example output**
```text
ClassificationHead(
  (fully_connected_block): FullyConnectedBlock(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=192, out_features=64, bias=True)
    (2): ReLU(inplace=True)
  )
  (dropout): Dropout(p=0.2, inplace=False)
  (classifier): Linear(in_features=64, out_features=2, bias=True)
)
```

### Step 2.18 — `initialise_pretrained_components()`
The backbone, the temporal module and the head can come from other runs. The fusion has no `pretrained` settings. A backbone deeper than its source backbone loads its first blocks only, the next ones staying random and never frozen (`classification_workflow.md`, Step 2.19).

**Example: a run from scratch**
```text
pretrained_checkpoint_paths = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
```

### Step 2.19 — `build_optimizer()` + `get_tracked_metrics()` + `MetricTracker.__init__()`
**Example output**
```text
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
{'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'accuracy': 'Accuracy'}
MetricTracker: batch_size 16, validation_iteration_ratio 25, sample_name 'Pairs', counts_frames True
```

### Step 2.20 — `print_experiment_information()` (→ `describe_split()` → `count_split_pairs()`)
`count_split_pairs()` counts the pairs, the frames of both members and the kept pairs of every split: those whose two members hold frames and have at most `maximum_sequence_length` frames, as the dataset keeps them (the category is not checked here). The kept pairs are the samples of an epoch.

**Example output**
```text
- Run Name                : CMP_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2
...
- Train Data Folder       : file:///home/.../sequence_data/synthetic_object_loss_pairs/train (2000 pairs, 16096 frames, 2000 pairs of members of at most 50 frames)
- Validation Data Folder  : file:///home/.../sequence_data/synthetic_object_loss_pairs/validation (200 pairs, 1586 frames, 200 pairs of members of at most 50 frames)
- Test Data Folder        : file:///home/.../sequence_data/synthetic_object_loss_pairs/test (200 pairs, 1590 frames, 200 pairs of members of at most 50 frames)
- Iterations Per Epoch    : 125
- Batch Size              : 16
- Save And Evaluate Every : 500 iterations
- Save Only Every         : 250 iterations
...
- MLflow Tracking         : Off
```

### Step 2.21 — `print_summary()` / `print_model_size_summary()`
The comparator adds the Member Fusion panel between the temporal module and the head panels; the size panel lists the fusion on its own line. The trainer then prints the Export Sizes panel: the parameters the forward pass of the inference model reads, then the size of the PyTorch weights, the TorchScript file, the ONNX files and the TensorRT engines the model is turned into and of the training checkpoint (Export Sizes). Its forward pass is the one of `InferenceSequenceComparator`, which encodes every member alone: the operators of the backbone are counted once per member, as the exported graph holds them, and its weights once (the operators once too with the sequence normalization of the `pair` scope, which encodes both members together).

**Example output** (shortened)
```text
╭────────────────────────────── Temporal Module ───────────────────────────────╮
│ Type               : mean_pooling                                            │
│ ...                                                                          │
│ Output Features    : 64                                                      │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────── Member Fusion ────────────────────────────────╮
│ Type               : difference                                              │
│ Number Heads       : 4                                                       │
│ Dropout Rate       : 0.0                                                     │
│ Sequence Normalization Scope : member                                        │
│ Output Features    : 192                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 183,648                       │
│ Temporal Module                                          │
│   - Parameters           : 0                             │
│ Member Fusion                                            │
│   - Parameters           : 0                             │
│ Classification Head                                      │
│   - Parameters           : 12,482                        │
│ Total Model                                              │
│   - Parameters           : 196,130                       │
│   - Estimated Size       : 0.79 MB                       │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.22 — `view_input_batches()` → `show_input_samples()`
With `diagnosis_modes.view_input_images`, the overridden `build_sample_frame_grid()` lays out the real frames of the before member above those of the after member, every frame captioned with its member (`0 before`, `0 after`), and the information panel gives the class of the pair, the real frames of both members (`Real frames : 3 + 2`) and, for the training pair, the augmentations and the order of the frames of each member (`Frame order : shuffled, showing frames 2, 0, 1 | 5, 4`). It needs a display, so it has no console output to show here. The same grids and panels are written without a display to `Input_Samples/` at every launch (Step 3.1).

### Step 2.23 — `restore_last_model()`
**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of `configuration_synthetic_pairs.yaml` shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the error images and the MLflow tracking on. Its run name is `CMP_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_MEAN_FDIF_HAVG.64_MC2`; with batches of 4 pairs, an epoch is 500 iterations.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1): from `start_iteration` to `number_iterations`, the total of the run, each iteration takes its training step, then saves its checkpoint, runs its validation batch and prints its summary on the iterations set for each, and evaluates its checkpoint last.

Before its first iteration, `run_training_iterations()` calls `save_input_samples()` (added on 2026-11-18, after this run was recorded): the first 10 pairs of every split by identifier are drawn as in Step 2.22 into `Input_Samples/Training`, `Validation` and `Test` (`sample_00.png` to `sample_09.png`, the training ones augmented), with `samples.json` naming every pair. The README run printed
```text
- Input Samples           : Input_Samples (30 images written in 5.43 seconds)
```

**Example: what happens at each iteration of the run**, in the order of the rows within an iteration, after its training step

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | ✔ | nothing: iteration 12 is saved |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |
| Test evaluation | | | | | | ✔ | | | | | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()` / `initialize_training_loop_state()` / `start_experiment_tracking()`
**Example output**
```text
Tensorboard/Train/        Total Loss, Classification Loss, Accuracy, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Accuracy                   (iteration 10)
trackers: {'start_time': ..., 'interval_batches': 0, 'total_loss': 0.0, 'classification_loss': 0.0, 'accuracy': 0.0, 'interval_frames': 0}
MLflow Run Created In The Experiment 'sequence_comparison' : 037372ca4aac463685e61176d80e825e (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 3.3 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12, so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6), with the label dictionary of the run.

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   2,441,735 bytes      Weights/Iteration_6/model_000006.pt   2,441,735 bytes
Weights/Iteration_9/model_000009.pt   2,441,735 bytes      Weights/Iteration_12/model_000012.pt  2,441,735 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
model_state: 86 tensors, the first 'class_weights', then 'backbone.convolution_blocks.0. ...'
mlflow_run_id: None in the checkpoints of iterations 3 and 6 (the MLflow run was not open yet), '037372ca4aac463685e61176d80e825e' in those of 9 and 12
```
The checkpoints written by the current code also hold `pretrained_backbone_blocks` (the blocks loaded from a shorter source backbone, Step 2.18), which would be `None` for this run.

### Step 3.4 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration, after its checkpoint and its summary (see `classification_workflow.md`, Step 3.7), see Stage 4.

### Step 3.5 — `get_next_batch()` → `SequencePairDataset.__getitem__()` + `collate_padded_sequences()`
`get_next_batch()` drops a batch whose loading fails and takes the next one (see `classification_workflow.md`, Step 3.8). `__getitem__()` reads all the frames of each member with `read_sequence_frames()` (see `sequence_classification_workflow.md`, Step 3.7: every frame read with `read_image()`), the shorter member getting black padding frames up to the length of the longer one, and concatenates them along the frame axis: the before member, then the after member, each with its real frames first. `collate_padded_sequences(members_per_sample=2)` then pads both members of every pair to the longest member of the batch (black frames, mask `False`), so that both members keep the same number of frames and can be split in two halves before the model.

**Example: the first batch of the run** (the longest member of the batch has 5 frames, so every member is padded to 5 frames)
```text
images               [4, 10, 3, 64, 64] torch.uint8      [B, 2 x 5, C, H, W]
sequence_mask        [4, 10] torch.bool
                     [[1, 1, 1, 0, 0,  1, 1, 1, 1, 0],        <- 3 before frames, 4 after frames
                      [1, 1, 0, 0, 0,  1, 1, 0, 0, 0],
                      [1, 1, 1, 0, 0,  1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1,  1, 1, 0, 0, 0]]        <- the longest member: 5 before frames
labels               [4] torch.int64     [1, 1, 1, 0]
sample_identifiers   [4] torch.int64     [1002, 878, 629, 915]
```

### Step 3.6 — `MetricTracker.count_frames()`
The real frames of both members of every pair are counted.

**Example: the frames of the first two intervals**
```text
interval_frames = 122 (iterations 1 to 4), 121 (iterations 5 to 8)      16 pairs per interval, about 7.6 frames per pair
```

### Step 3.7 — `run_model_iteration()` → `prepare_batch()`
The same function as for every task. `prepare_batch()` of the sequence classification moves the frames, labels and the `[B, 2T]` sequence mask to the device.

**Example: the first training batch**
```text
images [4, 10, 3, 64, 64] torch.uint8 -> torch.float32      first batch: classification_loss = 0.7293   accuracy = 0.25
```

### Step 3.8 — `augment_batch()`
The augmentation of the sequence classification (`sequence_classification_workflow.md`, Step 3.12): one draw per sample, so both members of a pair get the same crop, flip and colour change. `shuffle_frames()` shuffles the real frames of each member on its own (`get_member_length()` gives the frames of one member: `2T / MEMBERS_PER_SAMPLE`).

**Example: which pairs of the first batch were changed** (`probability: 0.4`)
```text
images: changed per sample [False, False, False, False]      <- no pair drawn for this batch
```

### Step 3.9 — `forward_pass()` → `split_pair_batch()` → `SequenceComparator.forward()`
`split_pair_batch()` splits the frames and the mask of the batch in two halves along the frame axis (`split_pair_members()`), the arguments of the comparator.

**Example: the first batch**
```text
before_images (4, 5, 3, 64, 64)   after_images (4, 5, 3, 64, 64)   before_mask (4, 5)   after_mask (4, 5)
before_mask [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
after_mask  [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]]
```

**Example: the dictionary returned in training mode** (recorded in float32; the training forward pass now runs in bfloat16 autocast, which gives bfloat16 `class_logits`, the loss staying in float32)
```text
class_logits          [4, 2] torch.float32
class_probabilities   [4, 2] torch.float32   [[0.4927, 0.5073], [0.5466, 0.4534], [0.5240, 0.4760], [0.4938, 0.5062]]
classification_loss   0.7293 (scalar)                    <- close to ln 2 = 0.693
```

### Step 3.10 — `encode_members()` → `encode_frames()`
With masks (training and evaluation), both members have the same number of frames, so they are stacked as one batch of `2B` sequences and encoded together by `encode_frames()` (`sequence_classification_workflow.md`, Step 3.14): the batch normalization statistics are computed over the frames of both members, and the padding frames get zero vectors. With the `sequence` normalization and the `"pair"` scope, the before and after members of a pair form one normalization group (the before member of a pair and its after member, B rows later, get the same group). Without masks (deployment, every frame real), each member is encoded on its own, so the two members may have different numbers of frames; with the `sequence` normalization and the `"pair"` scope, the two members are then concatenated along one frame axis and encoded together instead, so that they still share their statistics.

**Example: the first batch through the trained model**
```text
encode_members -> before (4, 5, 64), after (4, 5, 64)
zero vectors per member: before [2, 3, 2, 0], after [1, 3, 1, 3]      <- the padding frames of every member
```

### Step 3.11 — `member_fusion.attend_across_members()`
Only for `"cross_attention"`: the frames of each member attend to the real frames of the other member (both directions computed from the vectors before any update, with a residual connection and a layer normalization), then the padding frames are set back to zero vectors. With the difference fusion of this run, the vectors are returned unchanged.

### Step 3.12 — `summarise_members()`
The shared temporal module summarises the two members as one batch of `2B` sequences with their masks, then the result is split back into the before and after vectors.

**Example: the first batch** (masked mean of the real frames of every member)
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
Total Loss   1: 0.7293, 2: 0.6944, 3: 0.6955, 4: 0.6334, 5: 0.6722, 6: 0.7452, 7: 0.6591, 8: 0.6701, 9: 0.6278, 10: 0.6093, 11: 0.5966, 12: 0.5344
Accuracy     1: 0.2500, 2: 0.5000, 3: 0.5000, 4: 1.0000, 5: 0.7500, 6: 0.0000, 7: 0.7500, 8: 0.7500, 9: 0.7500, 10: 0.7500, 11: 1.0000, 12: 1.0000
```

### Step 3.15 — `training_loss.backward()` + `step_optimizer()` + validation pass
**Example output**
```text
gradient norm: iteration 1: 1.7714   iteration 2: 1.1179   iteration 3: 1.4277        learning rate 0.001 (constant)
Tensorboard/Validation at iteration 10: Total Loss 0.6939, Classification Loss 0.6939, Accuracy 0.5000
```

### Step 3.16 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
The header gives the pairs per second and the real frames of both members per second.

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 2.7526, 'classification_loss': 2.7526, 'accuracy': 2.25, 'interval_frames': 122}
after:  {'interval_batches': 4, 'total_loss': 0.6882, 'classification_loss': 0.6882, 'accuracy': 0.5625, 'interval_frames': 122}
MLflow, step 4: {'train/Total Loss': 0.6882, 'train/Classification Loss': 0.6882, 'train/Accuracy': 0.5625,
                 'train/throughput': 6.0327, 'train/interval_seconds': 2.6522, 'learning_rate': 0.001,
                 'train/frame_throughput': 45.9996}
```

**Example output: iterations 4 and 12**
```text
================================================================================================================================================
|    Iteration: 4    |     Total Pairs: 16      |    Interval Pairs: 16   |   Throughput: 6.0 Pairs Per Second   |    46.0 Frames Per Second   |
------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     0.6882
Moving Average of Train Classification Loss            :     0.6882
Moving Average of Train Accuracy                       :    56.25 %
------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:02.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
================================================================================================================================================
...
================================================================================================================================================
|   Iteration: 12    |     Total Pairs: 48      |    Interval Pairs: 16   |  Throughput: 12.2 Pairs Per Second   |    96.8 Frames Per Second   |
------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :     0.5920    -0.0947 Best
Moving Average of Valid Total Loss                     :     0.6939
Moving Average of Train Classification Loss            :     0.5920    -0.0947 Best
Moving Average of Valid Classification Loss            :     0.6939
Moving Average of Train Accuracy                       :    87.50 %   +31.25 % Best
Moving Average of Valid Accuracy                       :    50.00 %
------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:01.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
================================================================================================================================================
```

### Step 3.17 — end of the loop
The checkpoint of iteration 12 is already saved and evaluated inside its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 200 Test Pairs · 2026-10-10 07:10 ───────────────      <- the evaluation of iteration 12, last in the iteration
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (200 pairs), with `per_class_report: True`, `number_sample_images: 8`, `top_k: 1` (a setting when these examples were recorded; the code fixes 5), `save_predictions: True`, `error_dumps` activated with `maximum_per_class_pair: 2`, and the MLflow tracking on. Both checkpoints predict `same_number_of_objects` for every test pair. The outputs of iteration 6 were captured when its checkpoint was still saved before its training step, after 5 steps; the code saves it after its 6 steps. They were recorded on 2026-10-10, before the ROC and precision-recall curves: with `per_class_report`, the current evaluation also prints `Macro ROC AUC` and `Macro Average Precision` after `Top-1 Accuracy`, adds the `ROC AUC` and `Average Precision` columns to the per-class table and to `class_metrics.csv`, and writes and logs `roc_curves.png` and `precision_recall_curves.png` (`classification_workflow.md`, Steps 4.14 and 4.15), and, since 2026-10-28, `precision_recall_thresholds.png`, the precision-recall curves zoomed on the threshold of every class. Steps 4.9 and 4.11 show the layout with the curves and the recall intervals from the README run, evaluated on 2026-11-19. The examples of the run of Stage 3 were also captured before the per-class recall intervals were added (2026-11-01): `class_metrics.csv` holds `recall_lower` and `recall_upper` after `recall` since then, and with `per_class_report` the report shows a `Per Class Recall` table after `Per Class Metrics`.

### Step 4.1 — `run_test_evaluation()`
The evaluation of the classification (see `classification_workflow.md`, Stage 4), every sample being a pair.

**Example output: the progress bar** (200 pairs, batch size 4)
```text
Test Evaluation Iteration 6: 100%|██████████| 50/50 [00:04<00:00, 10.45it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
**Example: the accumulator at the end of the loop**
```text
'probabilities': 50 tensors, the first [4, 2] torch.float32    'labels': 50 tensors, the first [4] torch.int64
'sample_identifiers': 50 tensors, the first [4] torch.int64    'errors_folder': PosixPath('.../Evaluation/Iteration_6/Errors')
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
`predict_batch()` runs the comparator on the split members with their masks and without labels. The checkpoint of iteration 6 of this example was trained for 5 steps only (captured before a checkpoint held the step of its iteration), and every pair gets almost the same probabilities.

**Example: the predictions of the first test batch** (members of 4 frames at most)
```text
images [4, 8, 3, 64, 64]   sequence_mask [4, 8]      (2 x 4 frames)
class_logits          [4, 2] torch.float32
class_probabilities   [4, 2] torch.float32   [[0.4922, 0.5078], [0.4922, 0.5078], [0.4922, 0.5078], [0.4922, 0.5078]]
```

### Step 4.4 — `accumulate_evaluation_batch()`
The classification hook: the probabilities, labels and pair identifiers are kept on the CPU, and `dump_errors()` runs while the frames are still in the batch.

### Step 4.5 — `dump_errors()` → `reserve_error_image()` + `build_input_image()`
`build_input_image()` of the sequence classification draws the overridden `build_sample_frame_grid()`: the grid of the real frames of the before member (4 columns) above the grid of the after member, every frame captioned with its member. The file is named after the pair and the probability of the predicted class.

**Example: the error images of the checkpoint of iteration 12**
```text
build_input_image(sample_index=0) -> (144, 276, 3) uint8      a before row and an after row of 64 x 64 frames
Evaluation/Iteration_12/Errors/
└── loss_of_object-TO-same_number_of_objects/     test_000039_0.524.png   test_000073_0.524.png
report line: "Error Images : 2 in 1 class pairs, at most 2 per pair"
```

### Step 4.6 — `save_sample_images()` → `build_sample_image()`
The frame grids of every fixed sample pair, under a caption bar giving the true class, the predicted class and its probability, in green when right and in red when wrong. The caption is wider than the grids here, so the image is padded on the right.

**Example: the sample images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Samples/
test_000000.png  test_000003.png  test_000065.png  test_000068.png  test_000123.png  test_000137.png  test_000198.png  test_000199.png
build_sample_image -> (168, 302, 3) uint8
note of the report: "Sample Images : 8 fixed test pairs"
```

### Step 4.7 — `compute_evaluation_results()`
The results of the classification. With the `top_k: 1` of the recorded run, the top-k accuracy is the accuracy; the k now fixed at 5 counts every pair as right with 2 classes.

**Example: the results of iteration 6**
```text
evaluated_samples = '200 Test Pairs'
metrics           = {'Accuracy': 0.46, 'Balanced Accuracy': 0.5, 'Macro F1': 0.3151, 'Top-1 Accuracy': 0.46}
notes             = ['Error Images : 2 in 1 class pairs, at most 2 per pair', 'Sample Images : 8 fixed test pairs']
headline_metric   = 'Accuracy'
files             = Confusion Matrix Table, Class Thresholds, Class Metrics, Confusion Matrix Image, Predictions, Error Images, Sample Images
class_scalars     = 6: Precision/loss_of_object, Recall/loss_of_object, F1/loss_of_object, Precision/same_number_of_objects, ...
```
The current results also hold `Macro ROC AUC` and `Macro Average Precision` in `metrics`, the files `ROC Curves Image`, `Precision Recall Curves Image` and `Precision Recall Thresholds Image` after `Confusion Matrix Image`, the images `ROC Curves`, `Precision Recall Curves` and `Precision Recall Thresholds`, and 10 class scalars, `ROC AUC/<class>` and `Average Precision/<class>` following the F1 of each class.

### Step 4.8 — `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()`
The matrix and the metrics of CLS (`classification_workflow.md`, Steps 4.9 and 4.10); above 1,000 classes (`MAXIMUM_DENSE_CONFUSION_CLASSES`), only its non-zero cells, written as `confusion_cells.csv` (Step 4.12 there).

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
The `recall_lower` and `recall_upper` columns of `class_metrics.csv`, after `recall`, hold the exact (Clopper-Pearson) 95 % confidence interval of the recall of every class, the test pairs being independent (`classification_workflow.md`, Step 4.14). The current `class_thresholds.json` starts with `"logit_temperature": 1.0`, the temperature of the probabilities its thresholds were found on (`classification_workflow.md`, Step 4.13), and, when `evaluation.target_precision` is a list of precisions, gives the thresholds of the others under `additional_targets`, `class_metrics.csv` holding a threshold and a recall column for each.

**Example: `Evaluation/Iteration_6/class_metrics.csv` and `class_thresholds.json`**
```text
class_index,class,samples,precision,recall,f1,threshold_at_target_precision,recall_at_target_precision
0,loss_of_object,108,0.000000,0.000000,0.000000,0.492252,0.027778
1,same_number_of_objects,92,0.460000,1.000000,0.630137,,

{"target_precision": 0.99, "classes": {"loss_of_object": {"threshold": 0.4922524094581604, "recall": 0.027777777777777776},
                                       "same_number_of_objects": null}}
```

**Example: the current layout, `Checkpoint_Evaluation/Iteration_500/class_metrics.csv` of the README run** (`CMP_IN-KAR-512_..._MC2`, its checkpoints evaluated right after its training on 2026-11-19 with the helpers of `scripts/evaluation/evaluate_checkpoint.py`, 16 `loss_of_object` pairs predicted `same_number_of_objects` and 7 the other way; the bounds of the 95 % interval follow `recall`, with `per_class_report` the `roc_auc` and `average_precision` columns follow `f1`, and the threshold of `same_number_of_objects` stays empty, its precision never reaching 99 %)
```text
class_index,class,samples,precision,recall,recall_lower,recall_upper,f1,roc_auc,average_precision,threshold_at_target_precision,recall_at_target_precision
0,loss_of_object,98,0.921348,0.836735,0.748450,0.903701,0.877005,0.942177,0.959691,0.906770,0.714286
1,same_number_of_objects,102,0.855856,0.931373,0.863703,0.971965,0.892019,0.942177,0.885416,,
```

### Step 4.10 — `write_predictions_file()`   (only with `save_predictions`)
`describe_sample()` of the pair dataset names a pair by its name in the manifest.

**Example: `Evaluation/Iteration_12/predictions.csv`** (200 rows, 15,039 bytes; first rows)
```text
pair,true_class,predicted_class,correct,probability_loss_of_object,probability_same_number_of_objects
test_000073,loss_of_object,same_number_of_objects,0,0.476044,0.523956
test_000148,same_number_of_objects,same_number_of_objects,1,0.476041,0.523959
test_000023,same_number_of_objects,same_number_of_objects,1,0.476021,0.523979
```

### Step 4.11 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The block of the report and the links of the files written by the evaluation (see `classification_workflow.md`, Steps 4.18 and 4.19). The first example was recorded before the code wrote the curves; the second one, from the README run trained and evaluated on 2026-11-19, has the layout the code writes: the curves, the top-5 accuracy and the `Per Class Recall` table after `Per Class Metrics`.

**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 200 Test Pairs · 2026-10-10 07:10 ───────────────────────────────────────────────────────────────────────
  Accuracy   46.00 %   Balanced Accuracy   50.00 %
  Macro F1   31.51 %   Top-1 Accuracy      46.00 %
  Error Images : 2 in 1 class pairs, at most 2 per pair
  Sample Images : 8 fixed test pairs

  Per Class Metrics
                                                                       Threshold    Recall
   Class                    Samples   Precision     Recall        F1     @ P99 %   @ P99 %
  ─────────────────────────────────────────────────────────────────────────────────────────
   loss_of_object               108      0.00 %     0.00 %    0.00 %      0.4760   19.44 %
   same_number_of_objects        92     46.00 %   100.00 %   63.01 %           —         —
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 99 % is never reached.
  - Results : file:///.../Results.txt
  - Confusion Matrix Table : file:///.../Evaluation/Iteration_12/confusion_matrix.csv
  - ...
  - Sample Images : file:///.../Evaluation/Iteration_12/Samples
```

**Example: the current layout, the first block of `Checkpoint_Evaluation/Results.txt` of the README run** (`CMP_IN-KAR-512_..._MC2`, trained on GPU with the error images off, its checkpoints evaluated right after its training on 2026-11-19 with the helpers of `scripts/evaluation/evaluate_checkpoint.py`; at iteration 1000 the accuracy reaches 95.00 %, every one of its 10 errors a `loss_of_object` pair predicted `same_number_of_objects`)
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
  Worst F1 first, ties in class order. — : no samples of the class, no negative for the ROC AUC, or the precision 99 %
  is never reached.

  Per Class Recall
                                                   95 %      95 %
   Class                    Samples    Recall     Lower     Upper
  ────────────────────────────────────────────────────────────────
   loss_of_object                98   83.67 %   74.84 %   90.37 %
   same_number_of_objects       102   93.14 %   86.37 %   97.20 %
  Worst Recall first, ties in class order. 95 % Lower and Upper : exact (Clopper-Pearson) interval, the counted samples
  being independent. — : no samples of the class.
```

**Example: TensorBoard and MLflow after the two evaluations**
```text
Tensorboard/Test: 10 scalars (4 global + 6 per class), 9 images (8 samples + Confusion Matrix)
    scalar Accuracy          6: 0.4600, 12: 0.4600
MLflow: test/Accuracy, test/Balanced Accuracy, test/Macro F1, test/Top-1 Accuracy, test/best_Accuracy (steps 6, 12),
        6 test_class/<metric>/<class>, train/frame_throughput (steps 4, 8, 12)
```
The current evaluation writes 16 scalars to `Tensorboard/Test` (6 global + 10 per class) and 12 images (the 8 samples, `Confusion Matrix`, `ROC Curves`, `Precision Recall Curves` and `Precision Recall Thresholds`), as `Checkpoint_Evaluation/Tensorboard/Test` of the README run holds, and also sends `test/Macro ROC AUC`, `test/Macro Average Precision`, 10 `test_class/` scalars, `roc_curves.png`, `precision_recall_curves.png` and `precision_recall_thresholds.png` to MLflow.

After the tracking, `update_evaluation_curves()` (added on 2026-11-18, after this run was recorded) writes the row of the iteration into `Evaluation/evaluation_summary.csv`, draws `evaluation_curves.png` of the run folder again (the accuracy of every evaluated iteration, the best one ringed), prints `  - Evaluation Curves : file:///.../evaluation_curves.png` under the block and, with the tracking on, sends both files to the root of the MLflow run, next to `Results.txt`.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12, captured when those of iterations 3, 6 and 9 were still saved before their training step) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 40 pairs of the test split (its `labels.json` reduced, its `data/` folder linked). They were recorded on 2026-10-10, before the ROC and precision-recall curves (Stage 4), and before the per-class recall intervals (2026-11-01), whose `Per Class Recall` table now follows `Per Class Metrics` in the block of Step 5.6.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5): set `run_name` and `experiment_folder` (the run folder, built from `run_name` in the main block) to the comparison run, and `test_data_folder` to another labelled split (holding a pairs `labels.json` and `data/`) if needed; `maximum_error_images_per_class_pair` turns the error images on with its own limit, `save_predictions` (True, False, or None to keep the `evaluation.save_predictions` of the run) writes or skips the `predictions.csv` of every evaluated checkpoint, `target_precisions` replaces the precision of the class thresholds (a list such as `[0.99, 0.95]` is accepted, its first precision the main one), and the evaluation is written in the `Checkpoint_Evaluation/` folder of the run (`Checkpoint_Evaluation_split_<parent folder>_<split folder>/` for another labelled split). A pair with a member longer than the `maximum_sequence_length` of the run is left out of the evaluation, with the red warnings of Step 2.9.

### Step 5.2 — trainer construction
The trainer is built from the `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it rather than those of the mapping file, the name of the run folder kept as its `run_name` and the transfer of the pretrained components turned off (`disable_pretrained_settings()`, `classification_workflow.md`, Step 5.2).

**Example: the split line of the launch information**
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
MLflow Run Created In The Experiment 'sequence_comparison' : 7f5220908af54c13aaf92eb8ec7f147b (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)
```

### Step 5.5 — `summarise_checkpoint_evaluations()`
Every evaluation has added its row to `Checkpoint_Evaluation/evaluation_summary.csv` and drawn `Checkpoint_Evaluation/evaluation_curves.png` (Step 4.11), the rows in increasing iteration order; the summary gives their links, then the best checkpoint. The example was recorded before the curves (2026-11-18), when the rows followed the evaluation order: they are shown in the order and with the `Evaluation Curves` line of the current code.

**Example output** (every checkpoint predicts `same_number_of_objects` for the 40 pairs, so the four iterations tie and are listed in their evaluation order)
```text
Evaluation Summary Of 4 Checkpoints : file:///.../runs/CMP_IN-KAR-64_..._MC2/Checkpoint_Evaluation/evaluation_summary.csv
  - Evaluation Curves : file:///.../runs/CMP_IN-KAR-64_..._MC2/Checkpoint_Evaluation/evaluation_curves.png
  - Best Checkpoint : iterations 12, 9, 6, 3 tie (Accuracy 52.50 %)

iteration,Accuracy,Balanced Accuracy,Macro F1,Top-1 Accuracy,Precision/loss_of_object,Recall/loss_of_object,F1/loss_of_object, ...
3,0.525,0.5,0.34426229508196726,0.525,0.0,0.0,0.0,0.525,1.0,0.6885245901639345
6,0.525,0.5,0.34426229508196726,0.525,0.0,0.0,0.0,0.525,1.0,0.6885245901639345
9,0.525,0.5,0.34426229508196726,0.525,0.0,0.0,0.0,0.525,1.0,0.6885245901639345
12,0.525,0.5,0.34426229508196726,0.525,0.0,0.0,0.0,0.525,1.0,0.6885245901639345
```
The columns are the scalars of every evaluation (`build_summary_columns()`), those of every class only with `per_class_report`: the current evaluation writes `Top-5 Accuracy` instead of `Top-1 Accuracy`, inserts `Macro ROC AUC` and `Macro Average Precision` after it, and `ROC AUC/<class>` and `Average Precision/<class>` after the F1 of each class. The README run evaluated again gives, in its `Checkpoint_Evaluation/evaluation_summary.csv` (shortened):
```text
iteration,Accuracy,Balanced Accuracy,Macro F1,Top-5 Accuracy,Macro ROC AUC,Macro Average Precision,Precision/loss_of_object,...
500,0.885,0.8840536214485795,0.8845120634681529,1.0,0.9421768707482994,0.9225531889718139,...
1000,0.95,0.9489795918367347,0.949753793588584,1.0,0.9832933173269308,0.982727272476577,...
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 9 on the 40 pairs**
```text
Evaluating : file:///.../runs/CMP_IN-KAR-64_..._MC2/Weights/Iteration_9/model_000009.pt
Iteration 9 · 40 Test Pairs · 2026-10-10 07:10 ─────────────────────────────────────────────────────────────────────────
  Accuracy   52.50 %   Balanced Accuracy   50.00 %
  Macro F1   34.43 %   Top-1 Accuracy      52.50 %
  Error Images : 2 in 1 class pairs, at most 2 per pair
  Sample Images : 8 fixed test pairs

  Per Class Metrics
                                                                       Threshold    Recall
   Class                    Samples   Precision     Recall        F1     @ P99 %   @ P99 %
  ─────────────────────────────────────────────────────────────────────────────────────────
   loss_of_object                19      0.00 %     0.00 %    0.00 %      0.4871   10.53 %
   same_number_of_objects        21     52.50 %   100.00 %   68.85 %      0.5129   19.05 %
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 99 % is never reached.
```
The sample pairs were chosen from the 40 pairs of this split, so they are not the same pairs as in Stage 4.
