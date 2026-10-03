# Implementation Plan: Evaluation Outputs, Sequence Augmentations And Sequence Normalization

> Plan of items 1, 2, 3 and 9 of `previous_classifiers_comparison.md` (section 4) and of its optional sequence
> normalization (section 4.6), written on 2026-10-03. Once validated, it is implemented phase by phase, each phase
> committed and pushed. The implementation notes (differences from the plan, acceptance results) are added at the end
> once the last phase is done.

## Table Of Contents
1. Context
2. Decisions
3. The Previous Projects
4. Current Behaviour Of vision_training
5. Design
6. Configuration Summary
7. Phases
8. Tests
9. Acceptance
10. Risks And Open Points
11. Verification

---

## 1. Context
- **Item 1, predictions file:** the probability of every class for every test sample, saved at each evaluation, so
  that errors and thresholds can be studied again without running the model.
- **Item 2, error dumps:** the misclassified test samples saved as images, sorted by true and predicted class.
- **Item 3, evaluation of every checkpoint:** every saved checkpoint of a run evaluated in one call, with a summary
  table to choose the checkpoint to deploy.
- **Item 9, frame shuffling and per-frame draws:** shuffle the frames of a window when their order carries no
  information (sets of photos), and draw the photometric augmentations frame by frame.
- **Sequence normalization:** batch normalization statistics computed over the frames of each window, at training
  and at inference, so that a frame is normalised by the frames it is seen with.

## 2. Decisions
**Taken with the user:**
- **Plan location:** this dedicated document, kept under version control.
- **Statistics of the comparison task:** both scopes, chosen in the configuration: `member` (every member window
  normalised by its own frames) or `pair` (both members normalised together).
- **Transfer:** a backbone trained with batch normalization can initialise a run with sequence normalization, and
  back.
- **Predictions file:** the probability of every class, with an off switch for large test sets.

**Taken in this plan:**
- **Tasks of items 1 and 2:** the classification-like tasks (classification, sequence classification, sequence
  labelling, sequence comparison). The detection versions stay in `suggestions.md` (A4).
- **Error dumps:** multi-class mode only, since a multi-label sample has no single true → predicted pair. The images
  are the network inputs (resized and padded, as the model saw them), not copies of the original files. Sequences and
  pairs are saved as frame grids.
- **Error dumps off by default** in the shipped configurations, predictions on.
- **Frame shuffling:** its own probability per window, independent of the global augmentation probability (the global
  `activated` flag still turns it off). The real frames are shuffled, the padding frames stay last, and each member of
  a pair is shuffled on its own.
- **Sequence normalization:**
  - statistics per window, the unit the sequence models see; a sequence cut into several windows gets the statistics
    of each window;
  - the running statistics are never used: evaluation and deployment also normalise with the statistics of the
    current window. They keep following the training batches like batch normalization, so that the backbone can go
    back to batch normalization;
  - only for the backbone of the sequence tasks, never for the image tasks or the detection heads.

---

## 3. The Previous Projects
- **Predictions (`master`):** `Validation_Predictions.json`, the scores of every validation sample at each checkpoint.
- **Error dumps:**
  - `master` copied the misclassified images into `LABEL-TO-PREDICTION` folders;
  - `many-to-one` copied every frame of a misclassified sequence into `LABEL-TO-PREDICTION/<sequence>/`, the score
    written in the file names.
- **Every checkpoint (`master`):** `infer.py` with `index_iteration: None` looped over the checkpoint registry.
- **Frame shuffling (`many-to-one`):** `sequence_shuffling` shuffled the frames of the training and validation
  sequences at every iteration.
- **Per-frame draws (`many-to-one`):** brightness, contrast, saturation, crop and noise drawn for every frame.
- **Sequence normalization (`many-to-one`):**
  - a batch was one sequence (its frames as the batch dimension);
  - the batch normalization phase was a placeholder defaulting to "training" and never fed, so the moments of the
    current sequence normalised it at training and at inference;
  - the moving averages were updated but never used.
- **Comparison (`comparison`):** a batch was one pair, so the moments were computed over the frames of both members.

## 4. Current Behaviour Of vision_training
- **Evaluation loop (`BaseTrainer.run_test_evaluation`):**
  - every test batch goes through `prepare_batch` and `predict_batch`;
  - samples already evaluated (repeated by the DALI loaders) are skipped by their `sample_identifiers`;
  - `accumulate_evaluation_batch` keeps the class probabilities and labels on the CPU;
  - `compute_evaluation_results` writes the metrics, the confusion matrix and the thresholds to
    `<run>/Evaluation/Iteration_<n>/`;
  - `report_evaluation` appends to `Results.txt` and writes the TensorBoard scalars.
  - The sequence labelling trainer keeps the labelled frames; the comparison trainer inherits the sequence
    classification trainer.
- **Sample names:**
  - the classification identifiers are the manifest `id`s, the window and pair identifiers their index in the dataset;
  - the names (`file_name`, sequence `name`, pair `name`) are in the manifests but not kept by the datasets.
- **`scripts/evaluation/evaluate_checkpoint.py`:** evaluates one checkpoint (the last one, or a given iteration) into
  a separate output folder, the pretrained settings disabled.
- **Augmentation (`DataAugmenter`):**
  - a sample is augmented with the global probability;
  - every type is then drawn with its own probability, geometric types first, photometric types after;
  - a sequence is transformed as a whole, one draw for all its frames.
- **Normalization:**
  - `ConvolutionBlock` builds `nn.BatchNorm2d`, `ChannelLayerNormalizer` or nothing, and is shared with the detection
    heads;
  - `encode_frames` runs the backbone on the real frames gathered from `[B, T]` (with masks) or on every frame (without
    masks);
  - the comparator encodes both members as one batch with masks, and member by member without masks.

---

## 5. Design

### 5.1 Predictions File (item 1)
- **File:** `Evaluation/Iteration_<n>/predictions.csv`, one row per evaluated image, window or pair, and one row per
  labelled frame for sequence labelling.
- **Columns:**
  - identification, given by the dataset:
    - classification: `sample` (file name);
    - sequence classification: `sequence`, `first_frame`, `last_frame` (file names of the window);
    - sequence labelling: `sequence`, `frame`;
    - sequence comparison: `pair`;
  - `true_class`, `predicted_class` and `correct` (1 or 0); in multi-label mode `true_classes` and
    `predicted_classes` (the decisions at 0.5, separated by `;`);
  - one `probability_<class name>` column per class.
- **Datasets:**
  - `ClassificationDataset`, `SequenceDataset` and `SequencePairDataset` keep the names of their samples (the
    sequence and pair records gain `name`);
  - new method `describe_sample(sample_identifier: int) -> Dict[str, str]` on each, giving the identification columns
    in order;
  - new method `SequenceDataset.get_window_frame_names(sample_identifier: int) -> List[str]` for the frame rows.
- **Trainers:**
  - `accumulate_evaluation_batch` also keeps the `sample_identifiers`, and the sequence labelling trainer the position
    of every labelled frame in its window;
  - new `ClassificationTrainer.build_prediction_rows(accumulator) -> List[Dict[str, Any]]`, overridden by the sequence
    labelling trainer for the frame rows;
  - new `ClassificationTrainer.save_predictions(accumulator, evaluation_folder) -> Path`, called by
    `compute_evaluation_results` and listed in its `files` ("Predictions").
- **Configuration:** `ExperimentConfiguration.evaluation.save_predictions: True`.
- **Size:** rows × classes numbers. For example, 10 000 ImageNet validation images × 1000 classes is about 100 MB per
  evaluation, hence the off switch.

### 5.2 Error Dumps (item 2)
- **Files:** `Evaluation/Iteration_<n>/Errors/<TRUE>-TO-<PREDICTED>/<sample>_<score>.png`.
  - The score is the probability of the predicted class with 3 decimals, e.g. `test_000012_0.912.png`.
  - The path separators and the extension of the sample name are replaced (`class/image.jpg` → `class_image`).
- **When:** during the evaluation loop, the images being only available batch by batch. When the dumps are on,
  `accumulate_evaluation_batch` calls the new `ClassificationTrainer.dump_errors(accumulator, batch, predictions,
  sample_indices) -> None`. The accumulator counts the dumped images of every class pair, and at most
  `maximum_per_class_pair` images are kept per pair, the first ones met.
- **Images:**
  - classification: the network input image, with a caption bar `<true> -> <predicted> <score>`;
  - sequence classification: the frame grid of the window (`build_sample_frame_grid`), with the same caption bar;
  - sequence comparison: the before grid above the after grid (its existing `build_sample_frame_grid`);
  - sequence labelling: one image per wrong frame, the grid of its window, every frame captioned `true/predicted` and
    the wrong frames marked; named `<sequence>_frame<k>_<score>.png`.
  - New hook `build_error_image(batch, sample_index, frame_index) -> np.ndarray`, overridden by the sequence trainers.
- **Report:** the `Errors` folder in the evaluation `files`, and the number of dumped errors in the evaluation lines.
- **Configuration:** `ExperimentConfiguration.evaluation.error_dumps: {activated: False, maximum_per_class_pair: 20}`.
  The checks refuse `activated: True` in multi-label mode.

### 5.3 Evaluation Of Every Checkpoint (item 3)
- **Script:** in `scripts/evaluation/evaluate_checkpoint.py`, `iteration` is an iteration, `None` (the last
  checkpoint) or `"all"`.
- **New `vision_training/evaluation/checkpoint_evaluation.py`:**
  - `evaluate_checkpoints(trainer, experiment_folder, iterations) -> Path`:
    - loads every checkpoint into the model of the trainer and runs `run_test_evaluation`;
    - writes `Evaluation/evaluation_summary.csv` in the output run folder: one row per iteration, one column per
      evaluation scalar;
    - prints the best iteration by the headline metric, with a link to the summary.
  - A checkpoint listed in the registry but deleted from the disk is skipped with a yellow line.
- **`trainers/checkpointing.py`:** new `get_checkpoint_iterations(weights_directory) -> List[int]`, every iteration
  of the registry history, sorted and unique (the intermediate checkpoints included).
- **`BaseTrainer.run_test_evaluation`** returns its `EvaluationResults` (it returned nothing).
- **Headline metric:** `EvaluationResults` (in `core/contracts.py`) gains `headline_metric`, the name of the scalar
  that compares checkpoints:
  - `Accuracy` (multi-class) or `Subset Accuracy` (multi-label) for the classification-like tasks;
  - the area mAP@0.5 for detection, whose scalars are all named `<metric>/<method>`.

### 5.4 Frame Shuffling And Per-Frame Draws (item 9)
- **Configuration (`Data.data_augmentation`, every task):**
  - `shuffle_frames: {activated: False, probability: 0.5}`: the probability of shuffling a training window;
  - `independent_frames: False`: the photometric types drawn frame by frame in the augmented sequences, the geometric
    types staying per sequence (a crop or a rotation per frame would make objects jump).
  - The checks refuse both for the image tasks. Shuffling with an ordered temporal module (GRU, LSTM, transformer,
    temporal convolution) only prints a yellow warning, since the order then carries no information.
- **`DataAugmenter`:**
  - `__init__` reads both settings;
  - `augment_sample` loops over the frames for the photometric types when `independent_frames` is True;
  - new `shuffle_frames(images, sequence_mask, frame_block_length) -> Tuple[torch.Tensor, torch.Tensor]` returns the
    shuffled images and the `[B, T]` frame order. In every block of `frame_block_length` frames (the window, or each
    member window of a pair), the real frames are permuted and the padding frames stay last.
- **`SequenceClassificationTrainer.augment_batch` (new override):**
  - augments, then shuffles with block length `sequence_length`;
  - the frame order is applied to the frame labels of sequence labelling (`[B, T]` labels);
  - the sequence mask is unchanged, since the real frames stay first.
  - The comparison trainer inherits it, so each member is shuffled within its own window.
  - The input viewer shows the shuffled windows, being built on `augment_batch`.
- **Run name:** `SHUF` and `IND` after `AUG` in the training section, e.g. `AUG-SHUF-BS16-LR1e-3`.

### 5.5 Sequence Normalization
- **Configuration:**
  - `Backbone.normalization.feature_map_normalization: "sequence"`, run name code `SN`, for the sequence tasks only;
  - new `Fusion.sequence_normalization_scope: "member"` or `"pair"` (comparison task), read only with sequence
    normalization. The run name gets `SNP` in the fusion section for `pair`, `member` adding nothing.
- **New `models/components/sequence_normalization.py`:**
  - `SequenceGrouping` (a `TypedDict` in `core/contracts.py`): `frames_per_sequence` (equal consecutive groups) or
    `frame_sequence_indices` with `number_sequences` (groups of any size);
  - `SequenceBatchNormalization(nn.BatchNorm2d)`:
    - the same parameters and buffers as `nn.BatchNorm2d`, so that the state dictionaries stay identical and the
      transfer works both ways;
    - `forward(input_tensor, sequence_grouping)` computes the mean and the biased variance of every channel over the
      frames of each group and their spatial positions. It uses the `[B, T, C, H, W]` reshape for equal groups (the
      path without masks, export friendly) and segment sums (`index_add_`) otherwise;
    - without grouping, the whole batch is one sequence (backbone summary, receptive field);
    - the statistics are computed in float32, also under bfloat16 autocast;
    - in training, the running statistics follow the statistics of the whole batch, as in batch normalization, but
      they are never used.
- **Grouping passed to the layers:**
  - `ConvolutionBlock` builds `SequenceBatchNormalization` for `"sequence"`;
  - `ConvolutionBlock.forward(input_tensor, sequence_grouping=None)` passes the grouping to these layers only;
  - `Backbone.forward(input_tensor, sequence_grouping=None)` passes it to its blocks;
  - the detection heads and the existing state dictionary keys are unchanged.
- **`encode_frames`:**
  - gains `sequence_groups: Optional[torch.Tensor]`, the `[B]` normalization group of every sequence of the batch
    (default: one group per sequence);
  - with masks, the grouping is the group of every real frame, from the mask and `sequence_groups`;
  - without masks, it is `frames_per_sequence = T`.
- **Comparator:**
  - `member`: with masks, every row of the `[2B, L]` batch is its own group; without masks, the members are encoded
    one by one as today;
  - `pair`:
    - with masks, rows `i` and `i + B` share group `i`;
    - without masks, the two members are concatenated along the frame axis, encoded with `T1 + T2` frames per
      sequence, then split again. The before and after inputs keep their independent dynamic numbers of frames.
- **Transfer (`models/model_transfer.py`):** `compare_backbone_architectures` treats `"batch"` and `"sequence"` as the
  same weights, and prints a yellow line when the source run used the other one.
- **Checks:**
  - `"sequence"` is accepted in `Backbone.normalization.feature_map_normalization` for the sequence tasks only;
  - it is refused for the image tasks and in `RegionProposal.normalization` and `Detector.normalization`;
  - `Fusion.sequence_normalization_scope` must be `"member"` or `"pair"`.
- **Deployment:**
  - the inference models run the path without masks, so the exports use the reshape;
  - a request is normalised by its own window, so Triton batching never mixes the statistics of two requests.

---

## 6. Configuration Summary
| Key | Default | Tasks |
|---|---|---|
| `ExperimentConfiguration.evaluation.save_predictions` | `True` | classification, M2O, M2M, CMP |
| `ExperimentConfiguration.evaluation.error_dumps` | `{activated: False, maximum_per_class_pair: 20}` | classification, M2O, M2M, CMP |
| `Data.data_augmentation.shuffle_frames` | `{activated: False, probability: 0.5}` | every task, only activated for M2O, M2M, CMP |
| `Data.data_augmentation.independent_frames` | `False` | every task, only True for M2O, M2M, CMP |
| `Backbone.normalization.feature_map_normalization` | new value `"sequence"` | M2O, M2M, CMP |
| `Fusion.sequence_normalization_scope` | `"member"` | CMP |

- **Configurations:** every YAML gets the new keys with the commenting style of the files.
- **Migration:** new `add_evaluation_outputs_to_configuration`, `add_sequence_augmentation_to_configuration` and
  `add_sequence_normalization_scope_to_configuration`, inserted as text like the existing migrations and chained in
  `migrate_run_folder`. A file already holding the keys is left untouched.
- **Run names:** unchanged for the existing configurations, the new tokens (`SHUF`, `IND`, `SN`, `SNP`) only
  appearing when the settings are used.

## 7. Phases
Each phase: README checklist, implementation, tests (`pytest`, `ruff` name checks), acceptance, README checklist
removed, one commit, push.

| Phase | Content | Commit tag |
|---|---|---|
| 0 | This plan and its `.gitignore` exception | `[DOCS]` |
| 1 | Predictions files (5.1) | `[EVALUATION]` |
| 2 | Error dumps (5.2) | `[EVALUATION]` |
| 3 | Evaluation of every checkpoint (5.3) | `[EVALUATION]` |
| 4 | Frame shuffling and per-frame draws (5.4) | `[AUGMENTATION]` |
| 5 | Sequence normalization (5.5) | `[NORMALIZATION]` |
| 6 | Implementation notes and acceptance results in this document | `[DOCS]` |

## 8. Tests
**Phase 1**
- `describe_sample` and `get_window_frame_names` of the three datasets.
- Classification, sequence classification and comparison smoke runs:
  - one row per evaluated sample, the probabilities summing to 1;
  - the accuracy recomputed from the file equals the one of `Results.txt`;
  - no file with `save_predictions: False`.
- Sequence labelling: one row per labelled frame.
- Checks of the new keys, migration and the run names unchanged.

**Phase 2**
- `dump_errors` on a batch with known predictions: the folders, the file names with their scores, the cap per pair,
  and no image for the right samples.
- Smoke runs with the dumps on: images only for the wrong samples; sequence labelling frame names.
- The checks refuse the dumps in multi-label mode.

**Phase 3**
- `get_checkpoint_iterations` on a registry with repeated entries.
- `evaluate_checkpoints` on a short run with 3 checkpoints: 3 summary rows, every scalar, the best iteration, and a
  deleted checkpoint skipped.
- `run_test_evaluation` returns the results with their headline metric.

**Phase 4**
- `shuffle_frames`:
  - only the real frames move and the padding stays last;
  - the two member windows of a pair are shuffled separately;
  - the frame labels follow the frames;
  - probability 0 gives the identity.
- `independent_frames`: the frames of a sequence get different photometric draws and the same geometric draw; off,
  every frame gets the same draw.
- Checks for the image tasks, the warning with the ordered modules, run name tokens and migration.

**Phase 5**
- `SequenceBatchNormalization`:
  - the statistics equal a manual computation, and the reshape and segment paths agree;
  - a frame depends on the other frames of its window, not on the other windows;
  - evaluation mode gives the training mode output, and the running statistics only change in training;
  - the state dictionary keys equal those of `nn.BatchNorm2d`.
- Backbone and models:
  - the masked and mask-free paths agree on full windows;
  - the summary runs without grouping.
- Comparator scopes: with `pair` a change of the after member changes the before features; with `member` it does not.
- Exports of a sequence classifier and of a comparator (both scopes): TorchScript and ONNX match the eager model with
  dynamic frames.
- Transfer of a batch normalization backbone into a sequence normalization run, and back.
- Checks, run names (`SN`, `SNP`) and migration.

## 9. Acceptance
- **Phases 1 and 2:** short synthetic classification and comparison runs:
  - `predictions.csv` agrees with `Results.txt`;
  - a few error dumps of each task inspected.
- **Phase 3:** every checkpoint of a comparison run evaluated (the intermediate ones included), with the summary and
  the best iteration.
- **Phase 4:**
  - comparison pairs with `shuffle_frames` (their frames are independent views, so the order carries no
    information): at least 95 % test accuracy;
  - moving shapes with `shuffle_frames` and a GRU: chance level, as expected without the order, which shows that the
    shuffling is applied;
  - comparison pairs with `independent_frames`: at least 95 %.
- **Phase 5:**
  - comparison pairs with sequence normalization, scopes `member` and `pair`, against batch normalization: at least
    95 % each;
  - a batch normalization backbone transferred into a sequence normalization run;
  - the exports of a comparator with sequence normalization match the eager model.

## 10. Risks And Open Points
- **Size of the outputs:** large test sets give large predictions files and many error images, hence the off switch
  of the predictions and the dumps off by default with a cap per pair.
- **Sequence normalization at deployment:** a frame depends on its window, so the deployed windows must be built like
  the training ones (same `sequence_length` and stride). A window of one frame is normalised over the spatial
  positions of that frame only.
- **Augmentation speed:** the per-frame draws loop over the frames in Python, which slows the augmentation of long
  windows.
- **Synthetic plateau:** the moving shapes classifier may stay at chance level for a number of iterations that depends
  on the run (no seed is set), which can blur the comparisons of phase 5. Its acceptance therefore uses the comparison
  pairs.
- **Older runs:** they must be migrated (`scripts/configuration/migrate_configuration.py`) before being evaluated or
  exported.

## 11. Verification
- `pytest` passes after every phase.
- The run names of the existing configurations and the detection Triton configurations are unchanged.
- `NO_COLOR=1 uvx ruff check --no-cache --select F401,F811,F821,F841 vision_training tests scripts`.
- The acceptance results of section 9 recorded in this document.
