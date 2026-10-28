# Sequence Classification Workflow

This document follows one sequence classification run (many to one, M2O: one class per sequence of frames) from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output. The steps shared by every task (configuration loading, checkpoints, training loop, MLflow tracking, standalone evaluation) are explained in detail in `classification_workflow.md`, and the evaluation of the sequence classification is the one of the classification: here they keep a short explanation and the real output of the sequence run, and the steps specific to the sequences (whole sequences, padding, frame encoder, temporal module, frame grids) are explained in full.

Every sequence is one sample, read whole: there is no window. A sequence longer than `Data.sequence_settings.maximum_sequence_length` is left out of its split with a warning, and the sequences of a batch are padded with black frames to the longest sequence of the batch.

The examples come from real runs on CPU, so that the GPU stayed free for another training:
- **Stages 1 and 2**: `vision_training/configurations/sequence_classification/configuration_synthetic_sequences.yaml` (synthetic moving shapes, 4 classes `moving_left`, `moving_right`, `moving_up`, `moving_down`, `maximum_sequence_length: 50`, 128 pixel frames, batch size 16), with the data and run folders redirected to a temporary folder. The dataset was generated with the synthetic generator (2000 training, 200 validation and 200 test sequences of 2 to 12 frames). The dataset the configuration points to on this machine (`/home/robert_kelevra/Data/sequence_data/synthetic_moving_shapes`, read by the README runs) was generated before the lengths became variable: every one of its sequences has 8 frames of 128 pixels. The dataset of `configuration_example.yaml` is not on this machine.
- **Stages 3, 4 and 5**: the same configuration, shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the error images and the MLflow tracking turned on. After 12 iterations the network predicts `moving_down` for every sequence (4 classes: about 25% accuracy, a loss near ln 4 = 1.386), so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps. This run was recorded on 2026-10-18, before the ROC and precision-recall curves joined the evaluation (2026-10-20) and before `flatten` became the pooling of the configuration (2026-10-23): the steps say where the current code prints more, and the README runs of `experiments/readme_screenshots/` give the current outputs (`M2O_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HFLAT.64_MC4`, the current synthetic configuration trained for 1,000 iterations, and `M2O_IN-KAR-512_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4`, the former `average` pooling at 512 pixels, 2,000 iterations).

In the console, every file or folder path is a terminal hyperlink (`format_path_link()` of `vision_training/utilities/os_utilities.py`) showing the name of the file or folder, underlined; the examples write the `file://` URI the link opens, except where they show that name.

## Read This First

The sequence classification task (`task: "sequence_classification"`, run name code `M2O`, many to one) learns one class for a whole sequence of frames. Every real frame goes through the shared backbone and the frame pooling of `ClassificationHead.global_pooling` (`average`, `max`, or `flatten`, which keeps the whole last feature map), the temporal module of `TemporalModel.type` summarises the frame vectors of the sequence into one vector, and the fully connected head of CLS gives K logits read with a softmax (`output_mode` must be `multi_class`). A sequence is read whole, up to `Data.sequence_settings.maximum_sequence_length` frames, and the sequences of a batch are padded with black frames that a mask keeps out of the model. Choose it rather than CLS when the class needs several frames (the synthetic moving shapes are labelled with a direction that no single frame shows), rather than M2M or SEL when the answer is one class for the whole sequence and not a class or a choice per frame, and rather than CMP when there is a single sequence and not a before and an after sequence to compare. `SequenceClassificationTrainer` extends `ClassificationTrainer`: the loss, the batch accuracy and the whole evaluation are those of CLS, computed on sequences, and `SequenceLabellingTrainer` (M2M) and `SequenceComparisonTrainer` (CMP) extend it in turn.

### The Task At A Glance

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
every weight_saving_iterations (1000): the whole test split ─► accuracy, balanced accuracy, macro F1, top-2,
   macro ROC AUC, macro average precision, per-class metrics and thresholds at precision 0.99
   ─► Results.txt, Tensorboard/Test, Evaluation/Iteration_<n>/ (confusion matrix, class_metrics.csv, curves,
      class_thresholds.json, predictions.csv, Samples/ and Errors/ as grids of the real frames, 4 columns)
export: InferenceSequenceModel, images [B, T, 3, 128, 128] float32 BGR, every frame real ─► class_probabilities [B, 4]
```

### Files To Know

| Role | File | Main functions or classes |
|---|---|---|
| Trainer | `vision_training/trainers/sequence_classification_trainer.py` | `SequenceClassificationTrainer`: `build_dataloaders`, `prepare_batch`, `forward_pass`, `predict_batch`, `augment_batch`, `get_member_length`, `describe_split`, `build_input_image`, `build_sample_frame_grid`, `show_input_samples` |
| Loss, batch metric and evaluation (inherited) | `vision_training/trainers/classification_trainer.py` | `ClassificationTrainer`: `build_model`, `compute_batch_metrics`, `dump_errors`, `build_sample_image`, `compute_evaluation_results`, `write_predictions_file` |
| Shared training loop | `vision_training/trainers/base_trainer.py` | `BaseTrainer`: `run_training_iterations`, `run_model_iteration`, `save_model`, `run_test_evaluation`, `restore_last_model` |
| Dataset and loaders | `vision_training/data/sequence/sequence_dataset.py`, `vision_training/data/loaders.py`, `vision_training/data/split_description.py` | `SequenceDataset`, `read_sequence_frames`, `collate_padded_sequences`, `get_sequence_dataloaders`, `build_native_loader`, `count_split_sequences` |
| Model | `vision_training/models/sequence/sequence_classifier.py`, `sequence_model.py`, `frame_encoder.py` | `SequenceClassifier`, `SequenceFrameModel.compute_class_outputs`, `encode_frames` |
| Temporal modules | `vision_training/models/sequence/temporal_modules.py` | `build_temporal_module`, `MaskedMeanPooling`, `MaskedMaxPooling`, `AttentionPooling`, `RecurrentTemporalModule`, `TransformerTemporalModule`, `TemporalConvolutionModule` |
| Frame pooling and head | `vision_training/models/classification/classification_head.py` | `build_global_pooling`, `compute_pooled_features`, `ClassificationHead` |
| Loss | `vision_training/losses/classification_losses.py` | `compute_classification_loss`, `compute_balanced_class_weights` |
| Metrics and evaluation | `vision_training/evaluation/classification/` | `compute_confusion_matrix`, `compute_metrics_from_confusion_matrix`, `compute_top_k_accuracy`, `find_class_thresholds`, `compute_class_score_curves`, `build_class_metrics_table` |
| Sequence augmentation | `vision_training/data/augmentations.py` | `DataAugmenter.augment_batch` (one draw per sequence), `DataAugmenter.shuffle_frames` |
| Configuration checks | `vision_training/configuration_processing/sequence_checks.py` | `check_sequence_classification_configuration`, `check_sequence_settings`, `check_temporal_model_configuration` |
| Run name | `vision_training/configuration_processing/run_name.py` | `build_sequence_sections`, `build_sequence_length_tokens`, `build_temporal_tokens` |
| Synthetic data generator | `vision_training/data/synthetic/create_synthetic_dataset.py` | `create_synthetic_dataset`, `generate_moving_shape_sequence`, `build_sequence_labels` |
| Export | `vision_training/models/sequence/inference_sequence_model.py`, `vision_training/deployment/export_specifications.py` | `InferenceSequenceModel` (model name `sequence_classifier`) |
| Standalone evaluation | `scripts/evaluation/evaluate_checkpoint.py`, `vision_training/evaluation/checkpoint_evaluation.py` | `evaluate_checkpoints`, `summarise_checkpoint_evaluations` |

### Run It

```bash
# 1. Synthetic dataset: in the main block of vision_training/data/synthetic/create_synthetic_dataset.py, set
#    task = "sequence_classification" and output_folders["sequence_classification"] (2,000 / 200 / 200 sequences of
#    2 to 12 frames of 512 pixels, one shape moving left, right, up or down across the frames)
uv run python -m vision_training.data.synthetic.create_synthetic_dataset

# 2. Training: set experiment_folder, the three data folders and Data.label_mapping.mapping_file_path first
uv run python -m vision_training.main --configuration_path="vision_training/configurations/sequence_classification/configuration_synthetic_sequences.yaml"

# 3. Evaluation of checkpoints: set run_name, experiment_folder (the run folder), iteration (None, an integer or
#    "all"), test_data_folder, batch_size, track_evaluation, maximum_error_images_per_class_pair and output_folder
#    in the main block of scripts/evaluation/evaluate_checkpoint.py
uv run python -m scripts.evaluation.evaluate_checkpoint
```

### Configuration Keys To Know First

| Key | Value in the synthetic configuration | Effect |
|---|---|---|
| `Data.sequence_settings.maximum_sequence_length` | `50` | the longest sequence kept: a longer one is left out of its split, the test split included, with a red warning (`T50`); the number of positions of the embedding of a transformer |
| `TemporalModel.type` | `"gru"` | any of the 7 types: `mean_pooling`, `max_pooling` and `attention_pooling` ignore the order of the frames, `gru`, `lstm`, `transformer` and `temporal_convolution` follow it |
| `TemporalModel.hidden_size`, `number_layers` | `128`, `1` | `GRU1x128`; for `gru` and `lstm`, `dropout_rate` only acts between stacked layers |
| `TemporalModel.bidirectional` | `True` | both directions concatenated, 256 features (`BI`); `causal: True` is refused for M2O |
| `ClassificationHead.global_pooling` | `"flatten"` | the 64 x 8 x 8 map of a 128 pixel frame becomes 4,096 features, which keep where the shape is (`HFLAT`); `average` and `max` give 64 |
| `Data.image_settings.size` | `128` | the frame size; with `flatten` it sets the GRU input (65,536 features and 50.4 million GRU weights at 512 pixels) |
| `ClassificationHead.fully_connected` | `[64]` | the hidden layer of the head on the 256 features of a sequence (`.64` of `HFLAT.64`) |
| `Data.data_augmentation.independent_frames` | `False` | every frame of a sequence gets the same draws; True draws the photometric types frame by frame |
| `Data.data_augmentation.shuffle_frames` | `activated: False, probability: 0.5` | permutes the real frames of the training sequences whose order carries nothing; a yellow warning with an ordered temporal module |
| `ExperimentConfiguration.evaluation.top_k` | `2` | the `Top-2 Accuracy` headline metric |

### What A Run Writes

- `Results.txt`: one block per evaluated checkpoint, `Iteration <n> · <N> Test Sequences · <date>`, with `Accuracy` (the headline metric), `Balanced Accuracy`, `Macro F1`, `Top-2 Accuracy` and, with `per_class_report`, `Macro ROC AUC`, `Macro Average Precision` and the per-class table sorted by F1, worst first; the notes `Error Images : ...` and `Sample Images : 8 fixed test sequences`.
- `Evaluation/Iteration_<n>/`: the files of CLS (`confusion_matrix.csv`, `class_metrics.csv`, `class_thresholds.json`, and with `per_class_report` `confusion_matrix.png`, `roc_curves.png`, `precision_recall_curves.png`); `predictions.csv` with `save_predictions`, one row per test sequence with its `sequence`, `first_frame` and `last_frame` before the classes and probabilities; `Samples/<sequence>.png` and `Errors/<TRUE>-TO-<PREDICTED>/<sequence>_<score>.png`, the real frames of the sequence on a numbered grid of 4 columns under the caption `true -> predicted probability` (532 pixels wide and 292 high for 8 frames of 128 pixels in the README run).
- `data.json`: the `sequences` of every class in every split, and every frame of every split.
- `Tensorboard/Train` and `Validation`: `Total Loss`, `Classification Loss`, `Accuracy` (and `Learning Rate`); `Tensorboard/Test`: 6 headline scalars, `Precision`, `Recall`, `F1`, `ROC AUC` and `Average Precision` of every class, the images `Samples/<sequence>`, `Confusion Matrix`, `ROC Curves` and `Precision Recall Curves`.
- The console summary gives the `Sequences Per Second` and the real `Frames Per Second` (padding frames left out), sent to MLflow as `train/throughput` and `train/frame_throughput`.

### Pitfalls

- `average` and `max` remove where the shape is from every frame vector, so the moving shapes stay at chance: the README run at 512 pixels with `average` kept its interval training loss between 1.386 and 1.391 for 2,000 iterations and a test accuracy of 0.23 (every sequence predicted `moving_down`), while the current `flatten` configuration at 128 pixels reached 0.995 at iteration 500 and 1.000 at iteration 1000 (`Results.txt` and `sequence_classification*.log` of `experiments/readme_screenshots/`; [tasks reference, Sequence Classification](../reference/tasks_reference.md#sequence-classification-m2o)). At 512 pixels `flatten` gives 65,536 features per frame and a GRU of 50.4 million weights, and stayed at 0.23 after 500 iterations (run `..._HFLAT.64_MC4_verification`; `suggestions.md` K31).
- The dataset the synthetic configuration points to on this machine predates the variable lengths: its sequences all have 8 frames of 128 pixels, so the README runs never padded a batch, while `create_synthetic_dataset.py` now writes 2 to 12 frames of 512 pixels (`suggestions.md` K32).
- Blocks 2 and 4 of the synthetic backbone (16 to 32 and 32 to 64 channels, 2 layers each) add no residual shortcut, their first layer changing the channel count, so only blocks 3 and 5 add one and the backbone summary lists `Residual Shortcuts` for them only (`count_residual_additions()` of `convolution_block.py`; `suggestions.md` H4, the projection shortcut still to do).
- `Iterations Per Epoch` and the split descriptions count the sequences of the manifests before the label mapping, so they are too high when the mapping leaves categories out (`count_split_sequences()` of `split_description.py`; `bug_standards.md` B46).
- `TemporalModel.dropout_rate` does nothing with a single GRU or LSTM layer: the dropout of PyTorch only acts between stacked layers (`RecurrentTemporalModule.__init__()` of `temporal_modules.py`).
- `shuffle_frames` with an ordered temporal module only prints a yellow warning and trains on the shuffled frames, which destroys the class of the moving shapes (`SequenceClassificationTrainer.__init__()`).

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


```text
main()
│   Entry point of every task: reads the path of the YAML file from --configuration_path, loads it, looks up the
│   trainer of its task, builds it and starts the training loop. Give a sequence_classification configuration to run this workflow.
│
├── load_experiment_configuration(configuration_path)
│   │   Splits the YAML into experiment_configuration ('ExperimentConfiguration') and model_configuration (Optimization,
│   │   Data, Backbone, TemporalModel, ClassificationHead), converts the types, adds label_mode and label_dictionary to
│   │   'Data' and runs the checks.
│   │
│   ├── load_configuration(configuration_path)
│   │       Plain YAML reader, no conversion.
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Reads the mapping file {category id: class index}; one category per sequence, no background class.
│   └── check_configuration()
│       ├── common checks                         experiment, data, backbone, optimization (see classification)
│       └── check_sequence_classification_configuration()
│               ├── the data, head and evaluation checks of the classification
│               ├── check_sequence_settings()             maximum_sequence_length (integer >= 1), a multi_class head
│               └── check_temporal_model_configuration()  any pooling or ordered type, sizes, dropout, not causal,
│                                                         not bidirectional when causal, pretrained settings
│
├── get_trainer_class(task)  -> SequenceClassificationTrainer
│
├── SequenceClassificationTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   Keeps Data.sequence_settings, warns when the frames are shuffled for an ordered temporal module, then runs the
│   │   ClassificationTrainer constructor (top_k, target_precision, save_predictions, error_dumps, output_mode).
│   │
│   └── BaseTrainer.__init__()
│       │   Same order as for every task: device, run name, settings, augmenter, classes of a resumed run, check of
│       │   the data of a resumed run, previous training moved aside, paths, loaders and their check, record of the
│       │   data, sample sequences, model, pretrained components, optimizer, metric tracker, printouts, restoration.
│       │
│       ├── get_device() / get_run_name() -> build_run_name() / DataAugmenter()
│       │       The run name adds the maximum sequence length (T<frames>) and the temporal module (e.g. GRU1x128-BI)
│       │       before the head and the classes.
│       ├── restore_run_label_dictionary() / check_resumed_data() / move_previous_run_folder() / setup_training_paths()
│       │       A resumed run keeps the classes of its label_dictionary.json and is refused when a labels.json
│       │       changed since its data.json, a training from scratch renames the run folder of a previous training of
│       │       the same settings v<n>_<run name>, then the run folder is created with the configuration and
│       │       label_dictionary.json (classification_workflow.md, Step 2.6).
│       ├── build_dataloaders()
│       │   │   Builds the three loaders and keeps class_names, taken from the categories of the training manifest.
│       │   │
│       │   └── get_sequence_dataloaders(label_type="sequence")
│       │       │   One SequenceDataset per split from <folder>/labels.json and the frames of <folder>/data/, and
│       │       │   the collate collate_padded_sequences (members_per_sample=1).
│       │       │
│       │       ├── SequenceDataset.__init__()
│       │       │       One sample per sequence. The sequences of unmapped categories or without frames are left out
│       │       │       (yellow note), then those longer than maximum_sequence_length (a red warning per sequence and a
│       │       │       red total per split). Shuffled once.
│       │       └── build_native_loader()
│       │               The training loader leaves out its last incomplete batch with a batch normalized
│       │               classification head (fully_connected_normalization "batch").
│       ├── check_loaders_give_batches()        refuses a split whose loader gives no batch
│       ├── save_data_record()                  data.json: the sequences of every class, the frames of every split
│       ├── get_test_sample_classes() + choose_sample_identifiers()
│       │       The class of every test sequence, so that the sample sequences are spread over the classes.
│       ├── build_model()
│       │   │   Creates the SequenceClassifier from the registry, then, with class_weighting "balanced", sets the
│       │   │   class weights of the loss.
│       │   │
│       │   └── SequenceClassifier.__init__()
│       │       ├── build_backbone()            the shared backbone, run on every frame
│       │       ├── build_global_pooling()      one vector per frame (compute_pooled_features() gives its width)
│       │       ├── build_temporal_module()     a pooling (mean, max, attention) or an ordered module (GRU, LSTM,
│       │       │                               transformer, temporal convolution), summarising the frame vectors of a
│       │       │                               sequence into one vector (keep_time_axis=False)
│       │       ├── ClassificationHead()        fully connected block and classifier, as for the images
│       │       └── class_weights buffer        ones, or balanced on the sequences of every class
│       ├── initialise_pretrained_components()  Backbone, TemporalModel and ClassificationHead can come from other runs
│       ├── build_optimizer()
│       ├── get_tracked_metrics() + MetricTracker.__init__(counts_frames=True)
│       ├── print_experiment_information() -> describe_split() -> count_split_sequences()
│       │       "S sequences, F frames, K sequences of at most M frames"; an epoch is K sequences.
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       │       The summary adds the Temporal Module panel; the size panel lists the temporal module on its own line.
│       ├── view_input_batches() -> show_input_samples()  (only with view_input_images, then exits)
│       └── restore_last_model() -> get_last_checkpoint_iteration() / load_model()
│
└── trainer.run_training_loop()  [STAGE 3]
    └── run_training_iterations()
        ├── setup_tensorboard_writers() / initialize_training_loop_state() / start_experiment_tracking()
        ├── for each training_iteration, from start_iteration to number_iterations (the total of the run):
        │   ├── get_next_batch()                 a batch failing to load is dropped and the next one taken
        │   │   ├── SequenceDataset.__getitem__() -> read_sequence_frames()
        │   │   │       Reads (read_image()) and resizes every frame of the sequence, with the sequence mask (all
        │   │   │       True) and the class of the sequence.
        │   │   └── collate_padded_sequences()
        │   │           Pads every sequence of the batch with black frames to its longest sequence before stacking.
        │   ├── MetricTracker.count_frames()     real frames of the batch, for the frame throughput
        │   ├── set_training_mode()
        │   ├── run_model_iteration()
        │   │   ├── prepare_batch()             frames to float32, labels and sequence mask on the device
        │   │   ├── augment_batch()             the same random augmentation for every frame of a sequence (photometric
        │   │   │                               types frame by frame with independent_frames), then shuffle_frames()
        │   │   ├── forward_pass() -> SequenceClassifier.forward(sequence_mask=..., labels=...)
        │   │   │   ├── encode_frames()          backbone + pooling on the real frames only, zeros for the padding
        │   │   │   ├── temporal_module          [B, T, F] frame vectors -> [B, F'] sequence vectors
        │   │   │   ├── classification_head      class_logits [B, K]
        │   │   │   └── compute_classification_loss()
        │   │   ├── compute_batch_metrics()      accuracy of the batch
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   ├── training_loss.backward() / step_optimizer()
        │   ├── save_model() on the multiples of weight_saving_iterations and intermediate_weight_saving_iterations
        │   ├── validation pass every validation_iteration_ratio iterations
        │   ├── average_trackers() + log_interval_metrics() + print_interval_summary()   (sequences and Frames Per Second)
        │   │   + report_dropped_batches()
        │   └── run_test_evaluation() on the multiples of weight_saving_iterations, last    -> [STAGE 4]
        ├── save_model() of the steps taken since the last checkpoint, if any (nothing evaluated after the loop)
        └── close the writers, finish_experiment_tracking()

run_test_evaluation(iteration)  [STAGE 4]
└── the evaluation of the classification (see classification_workflow.md, Stage 4), on whole sequences:
    ├── create_evaluation_accumulator() / prepare_batch() / predict_batch() / accumulate_evaluation_batch()
    ├── dump_errors() -> build_input_image()           the real frames of the sequence laid out on a grid
    ├── save_sample_images() -> build_sample_image()   the frame grid captioned with the true and predicted classes
    ├── compute_evaluation_results()                   confusion matrix, accuracy, balanced accuracy, macro F1, top-k,
    │                                                  thresholds, class_metrics.csv, predictions.csv (sequence, first
    │                                                  and last frame of every sequence); with per_class_report the ROC
    │                                                  and precision-recall curves (Macro ROC AUC, Macro Average
    │                                                  Precision, roc_curves.png, precision_recall_curves.png) and
    │                                                  confusion_matrix.png
    └── report_evaluation() -> report_evaluation_results() + track_evaluation()

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
└── same as every task: loads training_configuration.yaml, builds the SequenceClassificationTrainer in the output
    folder, then evaluate_checkpoints() -> start_evaluation_tracking() + summarise_checkpoint_evaluations()
    -> run_test_evaluation()
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `vision_training/configurations/sequence_classification/configuration_synthetic_sequences.yaml`.

### Step 1.1 — `main()`
The entry point is the same for every task (see `classification_workflow.md`, Step 1.1): `parse_configuration_path()` reads the path of the YAML file from the command line (`--configuration_path`), then it loads the file, looks up the trainer of `ExperimentConfiguration.task` and starts its training loop. Nothing in it is specific to the sequences.

### Step 1.2 — `load_experiment_configuration()`
It splits the YAML into the two dictionaries read by every later stage, converts the values YAML reads with the wrong type, adds `label_mode` and `label_dictionary` to `Data` (Step 1.4) and runs the checks (Step 1.5). The sequence classification has the `TemporalModel` section on top of the sections of the classification, its `Data` section has a `sequence_settings` subsection with a single key, `maximum_sequence_length` (the largest number of frames of a sequence, the longer ones being left out of the splits, and the size of the positional embedding of the transformer), and its evaluation settings are those of the classification.

**Example: the sections and the settings of the sequences**
```text
model_configuration keys: Optimization, Data, Backbone, TemporalModel, ClassificationHead
Data keys after loading:  loading, image_settings, sequence_settings, label_mapping, data_augmentation, label_mode, label_dictionary
sequence_settings:        {'maximum_sequence_length': 50}
evaluation:               {'per_class_report': True, 'number_sample_images': 8, 'top_k': 2, 'target_precision': 0.99,
                           'save_predictions': True, 'error_dumps': {'activated': False, 'maximum_per_class_pair': 20}}
TemporalModel:            {'type': 'gru', 'hidden_size': 128, 'number_layers': 1, 'number_heads': 4, 'kernel_size': 3, 'dropout_rate': 0.0,
                           'bidirectional': True, 'causal': False, 'positional_embedding': True, 'pretrained': {...}}
ClassificationHead:       {'global_pooling': 'flatten', 'fully_connected': [64], 'fully_connected_normalization': 'none',
                           'batch_normalization_momentum': 0.1, 'dropout_rate': 0.2, 'output_mode': 'multi_class',
                           'label_smoothing': 0.0, 'class_weighting': 'none', 'pretrained': {...}}
```
Every setting of `TemporalModel` is required, even those its type does not read (`number_heads` and `positional_embedding` for the transformer, `kernel_size` for the temporal convolution), so that the type can be changed without adding keys.

### Step 1.3 — `load_configuration()`
Plain YAML reader, as for every task: no conversion and no check.

**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'TemporalModel', 'ClassificationHead']
number_iterations: '1e10'      <- still a string
learning_rate:     0.001
```

### Step 1.4 — `build_label_dictionary()`
The mapping file gives the class index of every category of the manifest, a category being the class of a whole sequence. Several categories can be merged into one class by giving them the same index, and the sequences of a category left out of the mapping are left out of the splits (Step 2.9).

**Example: the label dictionary of the moving shapes**
```text
label_mapping: {'mode': 'multi_class', 'mapping_file_path': '/tmp/.../synthetic_moving_shapes/label_mapping.json'}
dict: {1: 0, 2: 1, 3: 2, 4: 3}
4 entries, 4 distinct classes
```

### Step 1.5 — `check_configuration()`
The common checks run first, then `check_sequence_classification_configuration()`: the data, head and evaluation checks of the classification, `check_sequence_settings()` (`maximum_sequence_length` an integer of at least 1, and a `multi_class` head since the manifests give one category per sequence), and `check_temporal_model_configuration()` (the type, the sizes, the dropout, and a module that is not causal: a causal module, which only sees the past frames, is for the sequence labelling). Every global pooling is accepted, `flatten` included: the temporal module is built for the width of the frame vector given by `compute_pooled_features()` (Step 2.14). Every problem is reported together, then the program exits.

**Example: output with `causal: True` and `maximum_sequence_length: 0` in the YAML**
```text
------------------------------------------------------
Invalid Configuration File : file:///.../configuration_wrong.yaml
------------------------------------------------------
  - 'Data.sequence_settings.maximum_sequence_length' must be an integer >= 1, got 0.
  - 'TemporalModel.causal' must be False for this task, which summarises whole sequences (causal modules are for the sequence labelling task).
  - 'TemporalModel.bidirectional' must be False when 'TemporalModel.causal' is True: reading the sequence backwards looks at the future frames.
SystemExit code 1
```
The single `causal: True` gives two errors, since the configuration also reads the sequences backwards (`bidirectional: True`).

### Step 1.6 — `get_trainer_class()`
**Example output**
```text
get_trainer_class(task="sequence_classification") -> <class 'vision_training.trainers.sequence_classification_trainer.SequenceClassificationTrainer'>
```


---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on `vision_training/configurations/sequence_classification/configuration_synthetic_sequences.yaml` (128 pixel frames, `maximum_sequence_length: 50`, batch size 16), with the data and run folders redirected to a temporary folder. The whole construction took 0.09 s on CPU.

### Step 2.1 — `SequenceClassificationTrainer.__init__()`
Before calling the constructor of the classification trainer, it keeps `Data.sequence_settings`, read by its hooks (the split descriptions), and prints a warning when the frames of the training sequences are shuffled (`data_augmentation.shuffle_frames`) while the temporal module models their order (GRU, LSTM, transformer, temporal convolution): shuffling is meant for the sequences whose frame order carries no information, with a pooling. The class attributes describe the task to the shared code: `LABEL_TYPE = "sequence"` (one class per sequence), `MEMBERS_PER_SAMPLE = 1` (one sequence per sample, 2 for the pairs of the comparison), `SAMPLE_NAME = "Sequences"` and `COUNTS_FRAMES = True` (the frame throughput, Step 3.8).

**Example: the attributes after the construction**
```text
self.sequence_settings          = {'maximum_sequence_length': 50}
self.LABEL_TYPE                 = 'sequence'
self.MEMBERS_PER_SAMPLE         = 1
self.top_k                      = 2                     <- the settings of the classification trainer
self.save_predictions           = True
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
SAMPLE_NAME = 'Sequences'   COUNTS_FRAMES = True   TASK_NAME = 'sequence_classification'
```

### Step 2.2 — `BaseTrainer.__init__()`
The shared constructor (see `classification_workflow.md`, Step 2.2), reached through `ClassificationTrainer.__init__()`, runs the steps below in the same order as for every task. `input_image_size` is the size of every frame.

**Example: some attributes set by the constructor**
```text
self.device = device(type='cpu')   self.per_class_report = True   self.number_sample_images = 8   self.input_image_size = 128
self.batch_size = 16   self.mixed_precision = False   self.start_iteration = 1
```

### Step 2.3 — `get_device()`
**Example output** (forced on the CPU; a normal run on this machine prints `cuda`)
```text
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The input and backbone tokens are those of every image task, the backbone being applied to every frame. `build_sequence_sections()` then adds the maximum sequence length, `T<maximum_sequence_length>` (`build_sequence_length_tokens()`), and the temporal module (`build_temporal_tokens()`): `MEAN`, `MAX`, `ATT<hidden size>`, `GRU` or `LSTM<layers>x<hidden size>` with `BI` when bidirectional, `TRF<layers>x<hidden size>.<heads>` (with `NPE` without positional embedding) or `TCN<layers>x<hidden size>.<kernel size>`, plus the transfer tokens. The head (`HFLAT.64`: the flattened feature map, one hidden layer of 64) and the classes (`MC4`) end the name. The temporal dropout gives no token, so that it can change between two resumptions of a run.

**Example output**
```text
M2O_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HFLAT.64_MC4
```

### Step 2.5 — `DataAugmenter()`
The same augmenter as the classification. Here every sample is a sequence `[T, 3, H, W]`, augmented as a whole (Step 3.12), and two settings are read for the sequences: `independent_frames` (the photometric types drawn frame by frame) and `shuffle_frames`.

**Example: the settings of the configuration**
```text
activated True   probability 0.4   independent_frames False   shuffle_frames {'activated': False, 'probability': 0.5}
5 of 28 types activated: random_crop, random_rotation, brightness_contrast_saturation, random_noise, random_haze
```

### Step 2.6 — `setup_training_paths()`
As for every task (see `classification_workflow.md`, Step 2.6): a resumed run first keeps the classes of its `label_dictionary.json` (`restore_run_label_dictionary()`), and a training from scratch renames the folder of a previous training of the same settings `v<n>_<run name>` (`move_previous_run_folder()`); before that, a resumed run whose `labels.json` changed since its `data.json` is refused (`check_resumed_data()`). Besides the copy of the configuration, the run folder holds `label_dictionary.json`, the class index of every category of the run. `data.json`, the record of the data of the run (the classes with their sequences in every split, and for every split its folder, the hash of its `labels.json` and its frames), is written next to them once the loaders are built and checked (`save_data_record()`, after Step 2.8).

**Example: the run folder once the trainer is built**
```text
M2O_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HFLAT.64_MC4/
├── Tensorboard/
├── Weights/
├── data.json
├── label_dictionary.json
└── training_configuration.yaml    31,113 bytes
```

### Step 2.7 — `build_dataloaders()`
This hook builds the loaders of the three splits with `label_type="sequence"` (Step 2.8), then keeps `class_names`, the names of the categories of the training manifest in class index order.

**Example output**
```text
class_names = ['moving_left', 'moving_right', 'moving_up', 'moving_down']
```

### Step 2.8 — `get_sequence_dataloaders()`
For each split it builds a `SequenceDataset` (Step 2.9) from `<folder>/labels.json` and the frames of `<folder>/data/`, then wraps it in the Native loader (Step 2.10): the sequence tasks always read their frames with OpenCV on the CPU. The sequences have different lengths, so the collate is always `collate_padded_sequences()` (with `members_per_sample=1`), which pads the sequences of every batch with black frames to its longest sequence (Step 3.7). The training loader leaves out the last incomplete batch of every epoch when the hidden layers of the head are batch normalized (`ClassificationHead.fully_connected_normalization: "batch"`), since a last batch of a single sequence gives no batch statistics (`ClassificationTrainer.drops_last_training_batch()`); an epoch is then `floor(training sequences / batch size)` iterations, printed at launch. Here the head is not batch normalized and every batch is kept. A split whose loader gives no batch is then refused (`check_loaders_give_batches()`, see `classification_workflow.md`, Step 2.7).

**Example: size of the three splits** (one sample per sequence, the sequences having 2 to 12 frames)
```text
train       2000 sequences -> 125 batches of 16 (one epoch = 125 iterations)
validation   200 sequences ->  13 batches of 16
test         200 sequences ->  13 batches of 16
collate     functools.partial(<function collate_padded_sequences>, members_per_sample=1)
```

### Step 2.9 — `SequenceDataset.__init__()`
It reads the manifest of the split: its `categories` and its `sequences`, each giving its `id`, its `name`, its ordered `frames` (paths relative to `data/`) and its `category_id`. Every sequence is one sample, fed whole to the model. The sequences of an unmapped category or without frames are left out first, counted in one yellow note printed once the split is read; then every other sequence longer than `maximum_sequence_length` is left out with a red warning naming it, followed by a red total for the split, before that note. Every kept sequence keeps its identifier, its name, its frames and the class of each frame (the class of the sequence), and the sequences are shuffled once and kept by identifier for the evaluation outputs. No frame is read here.

**Example: a manifest sequence, and the lengths of the training sequences**
```text
labels.json:  {'id': 1, 'name': 'train_000000', 'frames': ['train_000000/000000.png', 'train_000000/000001.png', ...], 'category_id': 1}
              (2000 sequences of 2 to 12 frames, the first of 7; 4 categories {'id': 1, 'name': 'moving_left'}, ...)
frames: number of sequences   2: 178   3: 191   4: 187   5: 190   6: 200   7: 204   8: 165   9: 168   10: 185   11: 171   12: 161
```

**Example: the messages of a test split built with `maximum_sequence_length: 10` and the categories 1 to 3 only in the label dictionary**
```text
Sequence 'test_000004' Of The Test Split Has 12 Frames, More Than The Maximum Sequence Length (10): It Is Left Out Of The Test.      <- red, one line per sequence
...
28 Sequences Of The Test Split Are Longer Than The Maximum Sequence Length (10) And Are Left Out Of The Test.                       <- red
47 sequences of unmapped categories or without frames left out : labels.json                                                        <- yellow
len(dataset) = 125              (200 - 47 - 28)
```

**Example: the first sequence after shuffling, and the sample the loader gets from it**
```text
sequence:   {'identifier': 1225, 'sequence_identifier': 1226, 'sequence_name': 'train_001225',
             'frames': ['train_001225/000000.png', 'train_001225/000001.png'], 'frame_labels': [2, 2]}
dataset[0]: images (2, 3, 128, 128) torch.uint8, sequence_mask (2,) torch.bool [True, True], labels () torch.int64 2,
            sample_identifiers 1225
get_sample_name(1225)  = 'train_001225'                 (the name of the sequence, '/' replaced by '_')
describe_sample(1225)  = {'sequence': 'train_001225', 'first_frame': 'train_001225/000000.png', 'last_frame': 'train_001225/000001.png'}
get_class_counts()     = [522, 484, 516, 478]           (sequences of every class, for the balanced class weights)
```

### Step 2.10 — `build_native_loader()`
The same PyTorch `DataLoader` as for the classification (see `classification_workflow.md`, Step 2.10), with the collate of Step 2.8.

**Example output**
```text
len(loader) = 125   batch_size = 16   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
`get_sample_class_indices()` of the test dataset gives the class of every test sequence, so `choose_sample_identifiers()` groups the sequences by class and gives the 8 sample sequences to the 4 classes in turn (2 each), each taken at the ends of its sorted identifiers.

**Example output** (the identifiers sorted here; `sample_image_identifiers` is a set)
```text
get_test_sample_classes() -> 200 entries, e.g. {161: [3], 10: [0], 53: [2], 24: [1]}
sample_image_identifiers  -> [0, 1, 2, 4, 195, 197, 198, 199]
names                     -> test_000000, test_000001, test_000002, test_000004,
                             test_000195, test_000197, test_000198, test_000199
```

### Step 2.12 — `build_model()`
The classification hook: it creates the `SequenceClassifier` from the registry `TASK_MODEL_CLASSES["sequence_classification"]` (Step 2.13), then, with `class_weighting: "balanced"`, sets `model.class_weights` from `get_class_counts()` of the training dataset (the sequences of every class). `BaseTrainer` then moves the model to the device and the dtype.

**Example: the components of the model**
```text
model.backbone                 Backbone                   183,648 parameters
model.global_pooling           Sequential                       0 parameters
model.temporal_module          RecurrentTemporalModule  3,245,568 parameters
model.classification_head      ClassificationHead          16,708 parameters
model.class_weights            buffer [4] torch.float32 [1.0, 1.0, 1.0, 1.0]
model.number_classes           4
```

### Step 2.13 — `SequenceClassifier.__init__()`
The model turns every frame into one vector with the shared backbone and a global pooling (Step 2.14), summarises the vectors of a sequence into one vector with the temporal module (Step 2.15), and gives the class logits with the head of the classification (Step 2.16). The backbone is the one of the image tasks, so a backbone trained on images can be transferred to it.

### Step 2.14 — `build_backbone()` + `build_global_pooling()`
The backbone shared by every task (see `classification_workflow.md`, Step 2.14), built for a frame of `image_settings.size`. The global pooling (`flatten` here, `average` or `max`) reduces its final feature map to one vector per frame, whose width `compute_pooled_features()` gives to the temporal module: as wide as the last backbone block for `average` and `max` (64 features for this backbone), or channels x height x width of the final feature map for `flatten` (64 x 8 x 8 = 4096 features per frame here), which then follows the image size.

The panel lists `Residual Shortcuts` only for the blocks adding at least one shortcut (`ConvolutionBlock.count_residual_additions()`): blocks 3 and 5. Blocks 2 and 4 change the channel count at their first layer, so their 2 layers close no residual pair and add none, and block 1 has a single layer (`suggestions.md` H4).

**Example: the backbone panel of `print_summary()` for a 128×128 frame** (shortened)
```text
│  - Convolution Block 1                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 3, 128, 128]                                         │
│    - Output Shape : [1, 16, 64, 64]                                          │
│                                                                              │
│  - Convolution Block 2                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 16, 64, 64]                                          │
│    - Output Shape : [1, 32, 32, 32]                                          │
│  ...                                                                         │
│  - Convolution Block 5                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 64, 8, 8]                                            │
│    - Output Shape : [1, 64, 8, 8]                                            │
Final Backbone output shape: [1, 64, 8, 8]
```

### Step 2.15 — `build_temporal_module()` → `RecurrentTemporalModule`
It builds the module of `TemporalModel.type` with `keep_time_axis=False`, so that it returns one vector per sequence, and with `maximum_sequence_length` as its largest length (the size of the positional embedding of the transformer):
- **poolings**, which ignore the frame order: `mean_pooling` and `max_pooling` (the mean or the maximum of the real frames, no weights) and `attention_pooling` (a weighted mean, the weight of every frame learned from its vector);
- **ordered modules**: `gru` and `lstm` (the final state of the last layer after the last real frame, both directions concatenated when bidirectional, `dropout_rate` acting between stacked layers only), `transformer` (the frame vectors projected to `hidden_size`, a learned positional embedding unless `positional_embedding` is False, pre-normalization encoder layers whose attention leaves out the padding frames) and `temporal_convolution` (stacked 1D convolutions of `kernel_size` frames whose dilation doubles at every layer, the padding frames set to zero before every convolution). For one vector per sequence, the transformer and the temporal convolution average the outputs of the real frames.

Every module gives the size of its output in `output_features`, read by the head.

**Example: the module of the configuration** (4096 features per frame, the flattened 64 x 8 x 8 map of a 128 pixel frame, hidden size 128, bidirectional)
```text
RecurrentTemporalModule(
  (recurrent_network): GRU(4096, 128, batch_first=True, bidirectional=True)
)
output_features = 256          (2 directions x 128)
```

### Step 2.16 — `ClassificationHead.__init__()`
The head of the classification (see `classification_workflow.md`, Step 2.17), fed with the sequence vector.

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
The sequence classification can transfer the backbone, the temporal module and the head of other runs (the poolings without weights cannot be transferred). The head may have more classes than its source run, appended after them as in the classification. A temporal module is only transferred to a module of the same type and sizes reading frame vectors of the same size (the channels of the last backbone block, the same global pooling when one of the runs flattens, and the same image size for a flattened feature map); a transformer with a positional embedding also needs the `maximum_sequence_length` of its source run, which gives the size of the embedding (a source run written before the setting existed is asked to add it to its `training_configuration.yaml`).

**Example: a run from scratch**
```text
pretrained_configurations['Backbone']           = {'experiment_folder': None, 'iteration': None, 'freeze': False,
                                                   'freeze_normalization_parameters': False, 'learning_rate_multiplier': 1.0}
pretrained_configurations['TemporalModel']      = {'experiment_folder': None, 'iteration': None, 'freeze': False, 'learning_rate_multiplier': 1.0}
pretrained_configurations['ClassificationHead'] = {'experiment_folder': None, 'iteration': None, 'freeze': False, 'learning_rate_multiplier': 1.0}
pretrained_checkpoint_paths                     = {'Backbone': None, 'TemporalModel': None, 'ClassificationHead': None}
```

### Step 2.18 — `build_optimizer()`
**Example output**
```text
Adam, 1 parameter group, base_learning_rates [0.001], weight decay 0.0
```

### Step 2.19 — `get_tracked_metrics()` + `MetricTracker.__init__()`
The metrics of the classification: the total loss, the classification loss and the accuracy of every batch. The tracker is created with `counts_frames=True`, so that it also counts the real frames of every interval (Step 3.8).

**Example output**
```text
{'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'accuracy': 'Accuracy'}
MetricTracker: batch_size 16, validation_iteration_ratio 25, sample_name 'Sequences', counts_frames True
```

### Step 2.20 — `print_experiment_information()` (→ `describe_split()` → `count_split_sequences()`)
`describe_split()` counts the sequences and the frames of every split from its manifest, and the sequences of at most `maximum_sequence_length` frames, those the dataset keeps (`count_split_sequences()`), so that the number of iterations per epoch is the number of kept training sequences over the batch size. The count is made before the label mapping: the sequences of an unmapped category are counted as kept, although the dataset leaves them out (Step 2.9).

**Example output**
```text
- Run Name                : M2O_IN-KAR-128_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HFLAT.64_MC4
- Experiment Folder       : file:///.../example_runs/M2O_IN-KAR-128_..._MC4
...
- Data Record             : file:///.../data.json
- Train Data Folder       : file:///tmp/.../synthetic_moving_shapes/train (2000 sequences, 13750 frames, 2000 sequences of at most 50 frames)
- Validation Data Folder  : file:///tmp/.../synthetic_moving_shapes/validation (200 sequences, 1357 frames, 200 sequences of at most 50 frames)
- Test Data Folder        : file:///tmp/.../synthetic_moving_shapes/test (200 sequences, 1377 frames, 200 sequences of at most 50 frames)
- Iterations Per Epoch    : 125
- Batch Size              : 16
- Total Iterations        : 10000000000
- Save And Evaluate Every : 1000 iterations
- Save Only Every         : 500 iterations
- Learning Rate           : 0.001
- Optimizer               : adam (weight decay 0.0)
- Learning Rate Schedule  : constant (warm-up 0 iterations)
- Gradient Clipping Norm  : None
- Weight Moving Average   : Off
- Mixed Precision         : False
- Resume Training         : True
- Validation Iter. Ratio  : 25
- MLflow Tracking         : Off
```

### Step 2.21 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
The backbone panel is shown in Step 2.14 and the receptive field panel is the one of the backbone. The sequence classifier adds the Temporal Module panel, which lists every setting of the section, then the head panel; the size panel lists the temporal module on its own line, every component with its parameters, its trainable parameters and its estimated size.

**Example output: the temporal module, head and size panels**
```text
╭────────────────────────────── Temporal Module ───────────────────────────────╮
│ Type               : gru                                                     │
│ Hidden Size        : 128                                                     │
│ Number Layers      : 1                                                       │
│ Number Heads       : 4                                                       │
│ Kernel Size        : 3                                                       │
│ Dropout Rate       : 0.0                                                     │
│ Bidirectional      : True                                                    │
│ Causal             : False                                                   │
│ Positional Embedding : True                                                  │
│ Output Features    : 256                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────── Classification Head ─────────────────────────────╮
│ Global Pooling     : flatten                                                 │
│ Input Features     : 256                                                     │
│ Hidden Layers      : [64]                                                    │
│ Normalization      : none                                                    │
│ Dropout Rate       : 0.2                                                     │
│ Output Mode        : multi_class                                             │
│ Number Of Classes  : 4                                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 183,648                       │
│   - Trainable Parameters : 183,648                       │
│   - Estimated Size       : 0.74 MB                       │
│                                                          │
│ Temporal Module                                          │
│   - Parameters           : 3,245,568                     │
│   - Trainable Parameters : 3,245,568                     │
│   - Estimated Size       : 12.98 MB                      │
│                                                          │
│ Classification Head                                      │
│   - Parameters           : 16,708                        │
│   - Trainable Parameters : 16,708                        │
│   - Estimated Size       : 0.07 MB                       │
│                                                          │
│ Total Model                                              │
│   - Parameters           : 3,445,924                     │
│   - Trainable Parameters : 3,445,924                     │
│   - Estimated Size       : 13.79 MB                      │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.22 — `view_input_batches()` → `show_input_samples()`
With `diagnosis_modes.view_input_images`, the real frames of an augmented training sequence and of a validation sequence are laid out on grids of 4 columns (`build_sample_frame_grid()` → `build_frame_grid()`), every frame numbered, each grid followed by an information panel giving the class of the sequence (`get_sample_caption()`, `Class : moving_up`), its real frames (`Real frames : 7`) and, for the training sequence, the augmentations drawn with their values (one line per real frame when `independent_frames` is on) and, when `shuffle_frames` is on, the order of its frames (`describe_frame_order()`, `Frame order : shuffled, showing frames 2, 0, 1, ...` or `kept`); the validation panel reads `Augmentations : none (validation sample)`. It needs a display, so it has no console output to show here.

### Step 2.23 — `restore_last_model()` → `get_last_checkpoint_iteration()` / `load_model()`
The same restoration as every task (see `classification_workflow.md`, Steps 2.26 to 2.28).

**Example output: a new run**
```text
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of `configuration_synthetic_sequences.yaml` shortened to 64 pixel frames, batch size 4, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, a validation batch every 10 iterations, with the error images and the MLflow tracking on. Its run name is `M2O_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64_T50_GRU1x128-BI_HAVG.64_MC4`; with batches of 4 sequences, an epoch is 500 iterations. It was recorded with the `average` pooling the configuration had before `flatten` became its default, so its frame vectors have 64 features instead of the 1024 of a flattened 64 x 4 x 4 map; the steps are the same.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
The loop of every task (see `classification_workflow.md`, Step 3.1): from `start_iteration` to `number_iterations`, the total of the run, each iteration takes its training step, then saves its checkpoint, runs its validation batch and prints its summary on the iterations set for each, and evaluates its checkpoint last.

**Example: what happens at each iteration of the run**, in the order of the rows within an iteration, after its training step

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | ✔ | nothing: iteration 12 is saved |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |
| Test evaluation | | | | | | ✔ | | | | | | ✔ | nothing |

### Step 3.2 — `setup_tensorboard_writers()`
**Example: the curves after the run**
```text
Tensorboard/Train/        Total Loss, Classification Loss, Accuracy, Learning Rate    (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Accuracy                   (iteration 10)
```

### Step 3.3 — `initialize_training_loop_state()` + `MetricTracker.create_trackers()`
The trackers of the sequence tasks hold one more counter, `interval_frames`, the real frames of the interval (Step 3.8).

**Example: the trackers of an interval**
```text
{'start_time': ..., 'interval_batches': 0, 'total_loss': 0.0, 'classification_loss': 0.0, 'accuracy': 0.0, 'interval_frames': 0}
```

### Step 3.4 — `start_experiment_tracking()`
The MLflow run is created in the experiment `sequence_classification` (the task name), in the local database of the experiments root folder, and recorded in `mlflow_run.json` of the run folder, so that a resumed training continues it. The run is opened by a background thread, so the line can appear in the middle of the first iterations (here during iteration 2).

**Example output**
```text
MLflow Run Created In The Experiment 'sequence_classification' : 1489ef2ba2ed4370bdd5d228fc8544c5 (Browse : mlflow ui --backend-store-uri sqlite:///.../example_runs/mlflow.db)
```

### Step 3.5 — `save_model()` → `save_checkpoint()`
Saved right after the training step of iterations 3, 6, 9 and 12, so that the checkpoint of iteration N holds N steps (see `classification_workflow.md`, Step 3.6). The checkpoint stores the label dictionary of the run and the source runs of the three transferable components. The `full-checkpoint` file of `Weights/` records the last checkpoint (`model_checkpoint_path: "Iteration_12"`) and lists all the saved ones (`all_model_checkpoint_paths`).

**Example: the checkpoints and the keys of the last one**
```text
Weights/Iteration_3/model_000003.pt   4,289,735 bytes      Weights/Iteration_6/model_000006.pt   4,289,735 bytes
Weights/Iteration_9/model_000009.pt   4,289,735 bytes      Weights/Iteration_12/model_000012.pt  4,289,735 bytes
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'label_dictionary',
 'mlflow_run_id', 'pretrained_backbone', 'pretrained_temporal_module', 'pretrained_classification_head']
model_state: 94 tensors, the first 'class_weights', then 'backbone.convolution_blocks.0. ...'
format_version 2   task 'sequence_classification'   pretrained_backbone / _temporal_module / _classification_head None
```
The MLflow run being open from iteration 2, the four checkpoints hold `mlflow_run_id: '1489ef2ba2ed4370bdd5d228fc8544c5'`. With the current `flatten` configuration at 128 pixels, the GRU holds 3,245,568 weights instead of 148,992, with their Adam moments: a checkpoint of the README run takes 41,450,653 bytes.

### Step 3.6 — `run_test_evaluation()`
At iterations 6 and 12, last in the iteration, after its checkpoint and its summary (see `classification_workflow.md`, Step 3.7), see Stage 4.

### Step 3.7 — `get_next_batch()` → `SequenceDataset.__getitem__()` → `read_sequence_frames()` + `collate_padded_sequences()`
`get_next_batch()` drops a batch whose loading fails and takes the next one (see `classification_workflow.md`, Step 3.8). `__getitem__()` reads the whole sequence: `read_sequence_frames()` reads every frame with `read_image()` (OpenCV, BGR, `uint8`, a `ValueError` naming a frame that cannot be read) and resizes it like an image of the other tasks (`preprocess_image()`), with the sequence mask, all True. The label is the class of the sequence (an `int64` scalar). `collate_padded_sequences()` then pads the frames of every sequence of the batch with black frames (zeros) up to its longest sequence, and the mask with False, before stacking them, so that the frame axis `T` of a batch is the length of its longest sequence. The frame labels would be padded with -100, but the class of a sequence has no frame axis. It runs in the loader workers.

**Example: the collate of three samples of 4, 5 and 3 frames**
```text
images          (3, 5, 3, 64, 64) torch.uint8      the padding frames black
sequence_mask   [[True, True, True, True, False], [True, True, True, True, True], [True, True, True, False, False]]
labels          (3,)    sample_identifiers (3,)
```

**Example: the first two batches of the run** (padded to their longest sequence, 8 and 12 frames)
```text
images               [4, 8, 3, 64, 64] torch.uint8      [B, T, C, H, W]
sequence_mask        [4, 8] torch.bool                  real frames per sequence [4, 7, 7, 8]
labels               [4] torch.int64                    [3, 2, 2, 2]
sample_identifiers   [4] torch.int64                    [993, 536, 1773, 440]
batch 2: images [4, 12, 3, 64, 64], real frames per sequence [5, 9, 8, 12]
```

### Step 3.8 — `MetricTracker.count_frames()`
Before the batch goes to the device, the tracker adds its real frames (the sum of the sequence mask) to the `interval_frames` of the interval, from which the console summary and MLflow get the frame throughput. The padding frames are not counted, so the throughput does not depend on the padding of the batches.

**Example: the frames of the first interval**
```text
interval_frames after each batch: 26, 60, 86, 112          (26 = 4 + 7 + 7 + 8 real frames of the first batch)
```

### Step 3.9 — `set_training_mode()`
`model.train()`: dropout on, batch normalization statistics updated.

### Step 3.10 — `run_model_iteration()`
The same function for the training and the validation batches (see `classification_workflow.md`, Step 3.11): prepare, augment, forward, sum of the `_loss` outputs, batch metrics, tracking.

**Example: the first training batch**
```text
classification_loss = 1.3803   total_loss = 1.3803   accuracy = 0.25
```

### Step 3.11 — `prepare_batch()`
The classification hook moves the frames (`float32`, still in [0, 255] BGR) and the labels to the device; the sequence hook adds the sequence mask, as booleans.

**Example: before and after**
```text
images         [4, 8, 3, 64, 64] torch.uint8   -> [4, 8, 3, 64, 64] torch.float32
sequence_mask  [4, 8] torch.bool               -> [4, 8] torch.bool
labels         [4] torch.int64                 -> [4] torch.int64
```

### Step 3.12 — `augment_batch()` → `DataAugmenter.augment_batch()` + `shuffle_frames()`
The augmenter decides for every sequence whether it is augmented (`probability: 0.4`), then applies every activated type to the whole sequence `[T, 3, H, W]`, so that all its frames share the same crop, rotation and colour change and the motion stays coherent; with `independent_frames`, the photometric types are drawn again for every frame. Then `shuffle_frames()` permutes the real frames of every sequence with the `shuffle_frames` probability when it is activated (off here), the padding frames staying last; for the sequence labelling the frame labels follow their frames. The length of the frame block given to it is that of the batch (`get_member_length()`: the frame axis of the sequence mask over `MEMBERS_PER_SAMPLE`), the length of its longest sequence.

**Example: which sequences of the first batch were changed**
```text
images:        changed per sample [False, True, False, False]
sequence_mask: changed per sample [False, False, False, False]
labels:        changed per sample [False, False, False, False]
```

### Step 3.13 — `forward_pass()` → `SequenceClassifier.forward(sequence_mask=..., labels=...)`
The trainer passes the frames, the mask and the labels, so that the model returns its loss with its logits.

**Example: the dictionary returned in training mode**
```text
class_logits          [4, 4] torch.float32
class_probabilities   [4, 4] torch.float32
classification_loss   1.3803 (scalar)                   <- close to ln 4 = 1.386, the loss of a model that does not know yet
```

### Step 3.14 — `encode_frames()`
The real frames of all the sequences of the batch are gathered into one batch of images (`input_tensor[sequence_mask]`) and run through the backbone and the global pooling, so that the padding frames cost no computation and stay out of the batch normalization statistics. The vectors are then put back at the position of their frame, the padding frames getting zero vectors. The sequence of every frame is also given to the backbone, for the `sequence` normalization of the feature maps, which computes its statistics over the frames of each sequence. Without mask (deployment sequences, all real) the frames run as one fixed-shape batch, which the exports need.

**Example: the first batch of the run**
```text
images [4, 8, 3, 64, 64] -> backbone on the 26 real frames -> encode_frames -> [4, 8, 64]
zero vectors per sequence [4, 1, 1, 0]        (the padding frames of the sequences of 4, 7, 7 and 8 frames)
```

### Step 3.15 — temporal module: `RecurrentTemporalModule.forward()`
The GRU reads the frame vectors of every sequence in order. With a mask, the sequences are packed (`pack_padded_sequence()` with the number of real frames of each), so that the GRU stops at the last real frame of every sequence and never reads its padding frames; the vector of the sequence is the final state of the last layer in both directions, concatenated. The other modules use the mask too: the poolings and the transformer leave the padding frames out, and the temporal convolution sets them to zero before every convolution and averages its outputs over the real frames. The padding thus changes nothing: in evaluation mode, four test sequences of 7, 7, 3 and 5 frames padded to 7 get the probabilities they get alone without mask, to 1.5e-08 (a new model of the configuration on the CPU, random weights).

**Example: the first batch**
```text
[4, 8, 64] frame vectors -> temporal_module -> [4, 256] sequence vectors
```

### Step 3.16 — `ClassificationHead.forward()` + `compute_classification_loss()`
The head turns every sequence vector into the class logits, then the loss of the classification is computed (see `classification_workflow.md`, Steps 3.16 to 3.18): the cross entropy of the softmax, weighted by `class_weights`, with `label_smoothing`.

**Example: the first batch**
```text
[4, 256] -> classification_head -> class_logits [4, 4]      classification_loss = 1.3803
```

### Step 3.17 — `compute_batch_metrics()`
The accuracy of the batch, from the argmax of the logits, without a second forward pass.

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
The sums of the interval are divided by the number of batches it ran, the frame counter is kept as it is. The header gives the sequences per second (`Sequences Per Second`) and the real frames per second, and MLflow receives the second one as `train/frame_throughput`.

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 5.6144, 'classification_loss': 5.6144, 'accuracy': 0.5, 'interval_frames': 112}
after:  {'interval_batches': 4, 'total_loss': 1.4036, 'classification_loss': 1.4036, 'accuracy': 0.125, 'interval_frames': 112}
MLflow, step 4: {'train/Total Loss': 1.4036, 'train/Classification Loss': 1.4036, 'train/Accuracy': 0.125,
                 'train/throughput': 1.3915, 'train/interval_seconds': 11.4984, 'learning_rate': 0.001,
                 'train/frame_throughput': 9.7405}
```

**Example output: iterations 4 and 12**
```text
===========================================================================================================================================================
|     Iteration: 4    |     Total Sequences: 16     |    Interval Sequences: 16    |   Throughput: 1.4 Sequences Per Second  |   9.7 Frames Per Second    |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   1.4036
Moving Average of Train Classification Loss            :   1.4036
Moving Average of Train Accuracy                       :   0.1250
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:11.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
...
===========================================================================================================================================================
|    Iteration: 12    |     Total Sequences: 48     |    Interval Sequences: 16    |   Throughput: 1.4 Sequences Per Second  |   9.5 Frames Per Second    |
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   1.3814  -0.0102 Best          <- green: lower than at iteration 8
Moving Average of Valid Total Loss                     :   1.3548
Moving Average of Train Classification Loss            :   1.3814  -0.0102 Best
Moving Average of Valid Classification Loss            :   1.3548
Moving Average of Train Accuracy                       :   0.3125  -0.0625
Moving Average of Valid Accuracy                       :   0.2500
-----------------------------------------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:11.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
===========================================================================================================================================================
```
The interval of iteration 8 shows 0.3 Sequences Per Second (54 seconds), since it holds the evaluation of iteration 6.

### Step 3.22 — end of the loop
The checkpoint of iteration 12 is already saved and evaluated inside its iteration, so the loop saves nothing more and evaluates nothing (see `classification_workflow.md`, Step 3.25).

**Example output**
```text
Iteration 12 · 200 Test Sequences · 2026-10-18 08:48 ───────────────      <- the evaluation of iteration 12, last in the iteration
...
Last Checkpoint Saved At Iteration 12. Exiting Training.
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (200 sequences of 2 to 12 frames of 64×64 pixels), with `per_class_report: True`, `number_sample_images: 8`, `top_k: 2`, `save_predictions: True`, `error_dumps` activated with `maximum_per_class_pair: 2`, and the MLflow tracking on. Both checkpoints predict `moving_down` for every sequence. The outputs of iteration 6 were captured when its checkpoint was still saved before its training step, after 5 steps; it now holds 6 steps. They were recorded on 2026-10-18, before the ROC and precision-recall curves: with `per_class_report`, the current evaluation also scores every class one against the rest with its ROC curve and precision-recall curve, prints `Macro ROC AUC` and `Macro Average Precision` after `Top-2 Accuracy`, adds the `ROC AUC` and `Average Precision` columns to the per-class table and to `class_metrics.csv`, and writes and logs `roc_curves.png` and `precision_recall_curves.png` (`classification_workflow.md`, Steps 4.14 and 4.15). Steps 4.11 and 4.13 show the current layout from the README run at 128 pixels.

### Step 4.1 — `run_test_evaluation()`
The evaluation of the classification (see `classification_workflow.md`, Stage 4), every sample being a whole sequence. The test sequences are never augmented. The test loader shuffles its sequences, so the order of the rows of `predictions.csv` changes from one evaluation to the next.

**Example output: the progress bar** (200 sequences, batch size 4)
```text
Test Evaluation Iteration 6: 100%|██████████| 50/50 [00:43<00:00,  1.16it/s]
```

### Step 4.2 — `create_evaluation_accumulator()`
The lists of the classification: the probabilities, labels and sample identifiers of every batch, the `Errors` folder and the counter of error images per pair of classes.

**Example: the accumulator at the end of the loop**
```text
'probabilities':      50 tensors, the first [4, 4] torch.float64
'labels':             50 tensors, the first [4] torch.int64
'sample_identifiers': 50 tensors, the first [4] torch.int64
'errors_folder':      PosixPath('.../Evaluation/Iteration_6/Errors')
```

### Step 4.3 — `prepare_batch()` + `predict_batch()`
The same preparation as in training (Step 3.11); `predict_batch()` runs the model with the sequence mask and without labels, so no loss is computed.

**Example: the predictions of a test batch** (padded to its longest sequence of 8 frames)
```text
class_logits          [4, 4] torch.float32
class_probabilities   [4, 4] torch.float32      (no 'classification_loss' key)
```

### Step 4.4 — `accumulate_evaluation_batch()`
The classification hook: the probabilities (as `float64`), the labels and the sequence identifiers are kept on the CPU, and `dump_errors()` runs while the frames are still in the batch.

### Step 4.5 — `dump_errors()` → `reserve_error_image()` + `build_input_image()`
As for the classification, every wrong prediction is written in `Errors/<TRUE>-TO-<PREDICTED>/`, at most `maximum_per_class_pair` per pair. The image is the one of `build_input_image()`: the real frames of the sequence laid out on a grid of 4 columns with a 4 pixel gray gap, every frame numbered, under a caption bar with both classes and the probability of the predicted class. The grid has one row per 4 frames, so its height depends on the length of the sequence (72 pixels for 1 row, 140 for 2, 208 for 3). The file is named after the sequence (`get_sample_name()`) and that probability.

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
For every fixed sample sequence of Step 2.11, the classification hook draws the frame grid of `build_input_image()` with a caption bar giving its true class, its predicted class and the probability of the predicted class, in green when right and in red when wrong. It is saved as `Samples/<sequence name>.png` and sent to TensorBoard under `Samples/<sequence name>`.

**Example: the sample images of the checkpoint of iteration 12**
```text
Evaluation/Iteration_12/Samples/
test_000000.png  test_000001.png  test_000002.png  test_000004.png
test_000195.png  test_000197.png  test_000198.png  test_000199.png
sizes: test_000000 and test_000199 (4 frames or less) 96 x 276, test_000001, test_000195 and test_000198 164 x 276,
       test_000002, test_000004 and test_000197 (9 to 12 frames) 232 x 276
note of the report: "Sample Images : 8 fixed test sequences"
```

### Step 4.7 — `compute_evaluation_results()`
The results of the classification (see `classification_workflow.md`, Step 4.8). The headline metric, used by Stage 5 and MLflow to choose the best checkpoint, is the accuracy.

**Example: the results of iteration 6**
```text
evaluated_samples = '200 Test Sequences'
metrics           = {'Accuracy': 0.235, 'Balanced Accuracy': 0.25, 'Macro F1': 0.0951, 'Top-2 Accuracy': 0.515}
notes             = ['Error Images : 6 in 3 class pairs, at most 2 per pair', 'Sample Images : 8 fixed test sequences']
headline_metric   = 'Accuracy'
files             = Confusion Matrix Table, Class Thresholds, Class Metrics, Confusion Matrix Image, Predictions, Error Images, Sample Images
images            = Confusion Matrix        sample_images = Samples/test_000000, ... (8)
class_scalars     = 12: Precision/moving_left, Recall/moving_left, F1/moving_left, Precision/moving_right, ...
```
The current results also hold `Macro ROC AUC` and `Macro Average Precision` in `metrics`, the files `ROC Curves Image` and `Precision Recall Curves Image` after `Confusion Matrix Image`, the images `ROC Curves` and `Precision Recall Curves`, and 20 class scalars, `ROC AUC/<class>` and `Average Precision/<class>` following the F1 of every class.

### Step 4.8 — `compute_confusion_matrix()` + `compute_metrics_from_confusion_matrix()`
The confusion matrix counts the sequences of every pair (true class, predicted class); the precision, recall, F1, accuracy, balanced accuracy and macro F1 come from it (see `classification_workflow.md`, Steps 4.9 and 4.10).

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
The share of sequences whose true class is among the `top_k` most probable classes.

**Example output** (`top_k: 2`)
```text
Top-2 Accuracy = 0.5150 at iteration 6, 0.4600 at iteration 12      (about 2 / 4 = 0.5: chance level for 4 classes)
```

### Step 4.10 — `find_class_thresholds()`
For every class, the lowest probability threshold keeping a precision of at least `target_precision`, with the recall kept, saved in `class_thresholds.json`.

**Example: `Evaluation/Iteration_6/class_thresholds.json`**
```text
{
    "target_precision": 0.99,
    "classes": {
        "moving_left": null,
        "moving_right": null,
        "moving_up": null,
        "moving_down": {
            "threshold": 0.2703714668750763,
            "recall": 0.02127659574468085            <- 1 of the 47 moving_down sequences
        }
    }
}
```
At iteration 12 only `moving_left` reaches the precision: threshold 0.22249144315719604, recall 0.019230769230769232 (1 of the 52 `moving_left` sequences).

### Step 4.11 — `build_class_metrics_rows()` + `write_class_metrics_file()` + `save_confusion_matrix_image()` + `save_score_curve_images()`
`class_metrics.csv` is written at every evaluation; `confusion_matrix.png`, `roc_curves.png` and `precision_recall_curves.png` only with `per_class_report`, which also adds the `roc_auc` and `average_precision` columns after `f1` (see `classification_workflow.md`, Steps 4.14 and 4.15). The example of this run predates the curves.

**Example: `Evaluation/Iteration_6/class_metrics.csv`**
```text
class_index,class,samples,precision,recall,f1,threshold_at_target_precision,recall_at_target_precision
0,moving_left,52,0.000000,0.000000,0.000000,,
1,moving_right,45,0.000000,0.000000,0.000000,,
2,moving_up,56,0.000000,0.000000,0.000000,,
3,moving_down,47,0.235000,1.000000,0.380567,0.270371,0.021277
```

**Example: the current layout, `Evaluation/Iteration_500/class_metrics.csv` of the README run at 128 pixels** (`M2O_IN-KAR-128_..._HFLAT.64_MC4` in `experiments/readme_screenshots/`, one `moving_left` sequence predicted `moving_down`)
```text
class_index,class,samples,precision,recall,f1,roc_auc,average_precision,threshold_at_target_precision,recall_at_target_precision
0,moving_left,49,1.000000,0.979592,0.989691,1.000000,1.000000,0.213127,1.000000
1,moving_right,52,1.000000,1.000000,1.000000,1.000000,1.000000,0.520245,1.000000
2,moving_up,53,1.000000,1.000000,1.000000,1.000000,1.000000,0.981040,1.000000
3,moving_down,46,0.978723,1.000000,0.989247,1.000000,1.000000,0.953328,1.000000
```

### Step 4.12 — `write_predictions_file()`   (only with `save_predictions`)
One row per test sequence, in the order of the test loader. `describe_sample()` of the sequence dataset names a sequence by its name and its first and last frames.

**Example: `Evaluation/Iteration_6/predictions.csv`** (200 rows, 24,296 bytes; first rows)
```text
sequence,first_frame,last_frame,true_class,predicted_class,correct,probability_moving_left,probability_moving_right,probability_moving_up,probability_moving_down
test_000080,test_000080/000000.png,test_000080/000001.png,moving_up,moving_down,0,0.228245,0.248930,0.252467,0.270359
test_000077,test_000077/000000.png,test_000077/000001.png,moving_down,moving_down,1,0.228247,0.248922,0.252460,0.270371
test_000007,test_000007/000000.png,test_000007/000004.png,moving_down,moving_down,1,0.229002,0.248709,0.252380,0.269909
```
The probabilities are almost the same for every sequence: the checkpoint of iteration 6 of this example was trained for 5 steps only, since it was captured when a checkpoint was still saved before the step of its iteration (it now holds 6 steps, `classification_workflow.md`, Step 3.6).

### Step 4.13 — `report_evaluation()` → `report_evaluation_results()` + `track_evaluation()`
The block of the report and the links of the files written by the evaluation (see `classification_workflow.md`, Steps 4.18 and 4.19). The first example predates the curves; the second one, from the README run at 128 pixels, has the current layout.

**Example output: the block of iteration 12** (the same text is appended to `Results.txt`)
```text
Iteration 12 · 200 Test Sequences · 2026-10-18 08:48 ───────────────────────────────────────────────────────────────────
  Accuracy   0.2350   Balanced Accuracy   0.2500
  Macro F1   0.0951   Top-2 Accuracy      0.4600
  Error Images : 6 in 3 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                          Threshold    Recall
   Class          Samples   Precision   Recall       F1     @ P0.99   @ P0.99
  ────────────────────────────────────────────────────────────────────────────
   moving_left         52      0.0000   0.0000   0.0000      0.2225    0.0192
   moving_right        45      0.0000   0.0000   0.0000           —         —
   moving_up           56      0.0000   0.0000   0.0000           —         —
   moving_down         47      0.2350   1.0000   0.3806           —         —
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 0.99 is never reached.
  - Results : file:///.../Results.txt
  - Confusion Matrix Table : file:///.../Evaluation/Iteration_12/confusion_matrix.csv
  - Class Thresholds : file:///.../Evaluation/Iteration_12/class_thresholds.json
  - Class Metrics : file:///.../Evaluation/Iteration_12/class_metrics.csv
  - Confusion Matrix Image : file:///.../Evaluation/Iteration_12/confusion_matrix.png
  - Predictions : file:///.../Evaluation/Iteration_12/predictions.csv
  - Error Images : file:///.../Evaluation/Iteration_12/Errors
  - Sample Images : file:///.../Evaluation/Iteration_12/Samples
```

**Example: the current layout, the first block of `Results.txt` of the README run at 128 pixels** (`M2O_IN-KAR-128_..._HFLAT.64_MC4`, trained on GPU with the error images off)
```text
Iteration 500 · 200 Test Sequences · 2026-10-23 04:38 ──────────────────────────────────────────────────────────────────
  Accuracy        0.9950   Balanced Accuracy         0.9949
  Macro F1        0.9947   Top-2 Accuracy            1.0000
  Macro ROC AUC   1.0000   Macro Average Precision   1.0000
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                                      Average   Threshold    Recall
   Class          Samples   Precision   Recall       F1   ROC AUC   Precision     @ P0.99   @ P0.99
  ──────────────────────────────────────────────────────────────────────────────────────────────────
   moving_down         46      0.9787   1.0000   0.9892    1.0000      1.0000      0.9533    1.0000
   moving_left         49      1.0000   0.9796   0.9897    1.0000      1.0000      0.2131    1.0000
   moving_right        52      1.0000   1.0000   1.0000    1.0000      1.0000      0.5202    1.0000
   moving_up           53      1.0000   1.0000   1.0000    1.0000      1.0000      0.9810    1.0000
  Worst F1 first, ties in class order. — : no samples of the class, no negative for the ROC AUC, or the precision 0.99
  is never reached.
```

**Example: the run folder, TensorBoard and MLflow after the two evaluations**
```text
Evaluation/Iteration_6/ and Iteration_12/   Errors/ (3 pairs x 2)  Samples/ (8)  class_metrics.csv  class_thresholds.json
                                            confusion_matrix.csv  confusion_matrix.png  predictions.csv
Tensorboard/Test: 16 scalars (4 global + 12 per class), 9 images (8 samples + Confusion Matrix)
    scalar Accuracy          6: 0.2350, 12: 0.2350
    scalar Top-2 Accuracy    6: 0.5150, 12: 0.4600
MLflow: test/Accuracy, test/Balanced Accuracy, test/Macro F1, test/Top-2 Accuracy, test/best_Accuracy (steps 6, 12),
        12 test_class/<metric>/<class>, train/frame_throughput (steps 4, 8, 12),
        artifacts configuration/training_configuration.yaml, data/data.json, metric_names.json, Results.txt,
        evaluation/iteration_6/ and iteration_12/: class_metrics.csv, class_thresholds.json, confusion_matrix.png
```
The current evaluation writes 26 scalars to `Tensorboard/Test` (6 global + 20 per class) and 11 images (the 8 samples, `Confusion Matrix`, `ROC Curves` and `Precision Recall Curves`), as the event files of the README runs hold, and also sends `test/Macro ROC AUC`, `test/Macro Average Precision`, 20 `test_class/` scalars, `roc_curves.png` and `precision_recall_curves.png` to MLflow.


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of the logic of `scripts/evaluation/evaluate_checkpoint.py` run on the run of Stage 3 (checkpoints 3, 6, 9 and 12, captured when those of iterations 3, 6 and 9 were still saved before their training step) with `iteration = "all"`, `batch_size = 16` and, to keep it short, a test folder of the first 40 sequences of the test split (its `labels.json` reduced, its `data/` folder linked). They were recorded on 2026-10-18, before the ROC and precision-recall curves (Stage 4).

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
The same script for every task (see `classification_workflow.md`, Stage 5): set `run_name` and `experiment_folder` (the run folder, built from `run_name` in the main block) to the sequence run, and `test_data_folder` to another labelled split (holding `labels.json` and `data/`) if needed; `maximum_error_images_per_class_pair` turns the error images on with its own limit, and `output_folder` (`<experiments>/evaluations` by default) receives the evaluation in a folder named after the run. The sequences longer than the `maximum_sequence_length` of the run are left out of the evaluation, with the red warnings of Step 2.9.

### Step 5.2 — trainer construction
The script loads `training_configuration.yaml` of the run, with the classes of the `label_dictionary.json` saved next to it, points it to the output folder, the reduced split and the batch size, keeps the name of the run folder (`run_name`), turns the transfer of the pretrained components off (`disable_pretrained_settings()`) and builds a `SequenceClassificationTrainer`.

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
The best checkpoint is the one with the highest headline metric; on a tie every tied iteration is listed, in the evaluation order, from the last checkpoint.

**Example output** (every checkpoint predicts `moving_down` for the 40 sequences, so they all have the same accuracy)
```text
Evaluation Summary Of 4 Checkpoints : file:///.../example_runs/evaluations/M2O_IN-KAR-64_..._MC4/Evaluation/evaluation_summary.csv
  - Best Checkpoint : iterations 12, 9, 6, 3 tie (Accuracy 0.1250)

iteration,Accuracy,Balanced Accuracy,Macro F1,Top-2 Accuracy,Precision/moving_left,Recall/moving_left,F1/moving_left, ...
12,0.125,0.25,0.05555555555555555,0.45,0.0,0.0,0.0, ...
9,0.125,0.25,0.05555555555555555,0.45,0.0,0.0,0.0, ...
6,0.125,0.25,0.05555555555555555,0.425,0.0,0.0,0.0, ...
3,0.125,0.25,0.05555555555555555,0.425,0.0,0.0,0.0, ...
```
The columns are the scalars of every evaluation (`build_summary_columns()`), those of every class only with `per_class_report`: the current evaluation inserts `Macro ROC AUC` and `Macro Average Precision` after `Top-2 Accuracy`, and `ROC AUC/<class>` and `Average Precision/<class>` after the F1 of every class.

### Step 5.6 — `run_test_evaluation()` for each checkpoint
**Example output: the block of the checkpoint of iteration 12 on the 40 sequences**
```text
Evaluating : file:///.../example_runs/M2O_IN-KAR-64_..._MC4/Weights/Iteration_12/model_000012.pt
Iteration 12 · 40 Test Sequences · 2026-10-18 08:49 ────────────────────────────────────────────────────────────────────
  Accuracy   0.1250   Balanced Accuracy   0.2500
  Macro F1   0.0556   Top-2 Accuracy      0.4500
  Error Images : 6 in 3 class pairs, at most 2 per pair
  Sample Images : 8 fixed test sequences

  Per Class Metrics
                                                          Threshold    Recall
   Class          Samples   Precision   Recall       F1     @ P0.99   @ P0.99
  ────────────────────────────────────────────────────────────────────────────
   moving_left         10      0.0000   0.0000   0.0000      0.2225    0.1000
   moving_right        13      0.0000   0.0000   0.0000           —         —
   moving_up           12      0.0000   0.0000   0.0000      0.2452    0.0833
   moving_down          5      0.1250   1.0000   0.2222           —         —
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 0.99 is never reached.
```
The sample sequences were chosen from the 40 sequences of this split (identifiers 0, 1, 2, 4, 11, 37, 38 and 39), so they are not all the same sequences as in Stage 4.
