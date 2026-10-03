# Implementation Plan: DALI Loaders For Classification, Sequence Classification And Sequence Comparison

> Plan written on 2026-10-03. Phase 1 is a benchmark that adds new files only and changes no existing code, so that
> the gain of every option is measured before anything is changed. The phases after it depend on its results and on a
> decision taken with the user (section 5.8). During the implementation, the README `## Implementation Plan` section
> holds the checklist of the current phase, as for the previous plans.

## Table Of Contents
1. Context
2. Decisions
3. Current Behaviour Of vision_training
4. Preliminary Probe
5. Phase 1: Benchmark
6. Design Of The Integration
7. Configuration Summary
8. Phases
9. Tests
10. Acceptance
11. Risks And Open Points
12. Verification
13. Benchmark Results

---

## 1. Context
- The classification, sequence classification (M2O), sequence labelling (M2M) and sequence comparison (CMP) trainers
  read their images with the Native loader: OpenCV decoding and resizing in 4 DataLoader worker processes. Only the
  object detection task has a DALI pipeline.
- ImageNet-1k pre-training (1,271,167 training JPEG images) is the first large workload of the classification task:
  one epoch takes 21 minutes at 1,000 images/s and 4 minutes at 5,000 images/s.
- Goal: measure how much faster training and evaluation would be with DALI loaders (decoding and resizing in C++
  threads and on the GPU) or with a tuned Native loader, then implement only what is worth it.

## 2. Decisions
**Taken with the user:**
- **Benchmark first:** it changes no existing file. New scripts import the existing loaders, datasets, configurations
  and models.
- **Tasks:** classification, sequence classification and sequence comparison. M2M uses the M2O loader with frame
  labels, so it is covered with it.
- **Our own DALI reader** where DALI has none for our formats (the windows of the sequences and the pairs).
- **Benchmark sequences:** there is no real sequence dataset yet, and the real sequences will be JPEG. The benchmark
  uses 1,000 sequences of 6 to 30 random ImageNet images copied into one folder per sequence, and pairs made of two of
  these sequences. Both were built on 2026-10-03 (section 5.3).

**Taken in this plan:**
- **Three loader families are compared:**
  - the current Native loader, exactly as the trainers build it;
  - a tuned Native loader (more workers, persistent workers, pinned memory, one OpenCV thread per worker);
  - DALI prototypes.

  A tuned Native loader close to DALI would be a much smaller change than DALI.
- **Gains are measured end to end:** loader, then augmentation, then the training step, against the GPU ceiling of
  each configuration. A faster loader gains nothing when the GPU is the limit, so loader speed alone does not decide.
- **The DALI outputs must reproduce the Native preprocessing:** BGR, linear resize without antialiasing, centred black
  padding and black padding frames. Deployment preprocesses with `preprocess_image` (OpenCV), so a model trained on
  DALI inputs must see nearly the same pixels. The benchmark measures the differences.
- **No duplicated code after the integration:** during phase 1 the prototypes live in the benchmark folder. If they are
  integrated, they move to `vision_training/data`, and the benchmark then imports the integrated loaders.
- **Plan location:** this dedicated document, kept under version control with a `.gitignore` exception. The benchmark
  results are recorded in section 13 before the decision.

## 3. Current Behaviour Of vision_training
- **Loaders (`vision_training/data/loaders.py`):**
  - `get_classification_dataloaders` and `get_sequence_dataloaders` build one
    `DataLoader(dataset, batch_size, shuffle=True, num_workers=number_of_workers)` per split;
  - there is no `pin_memory`, `persistent_workers` or `prefetch_factor` (default: 2 batches per worker), so the workers
    are started again at every epoch and at every evaluation pass;
  - `number_of_workers=4` is hard-coded in `ClassificationTrainer`, `SequenceClassificationTrainer` (inherited by M2M
    and CMP) and `DetectionTrainer`.
- **Datasets (`ClassificationDataset`, `SequenceDataset`, `SequencePairDataset`):**
  - every image or frame is read with `cv2.imread`, then resized by `preprocess_image`:
    - `INTER_LINEAR` resize keeping the ratio, the new size rounded with `round`;
    - centred black padding, with offsets `(size - new size) // 2`;
    - output: uint8 BGR CHW tensors;
  - a window holds its real frames first, then black padding frames, with a boolean mask;
    - M2M frame labels are padded with -100;
    - a pair is the before window followed by the after window, `[2L]` frames;
  - the sample list is shuffled once at creation, then the DataLoader permutes it at every epoch;
  - the trainers use the metadata of `loader.dataset`:
    - `class_names` and `number_classes`;
    - `get_class_counts()`, for the balanced class weights;
    - `describe_sample`, `get_sample_name` and `get_window_frame_names`, for the predictions files and error images.
- **OpenCV threads:** OpenCV uses 20 threads in each worker on this machine (`cv2.getNumThreads()`), so 4 workers can
  run up to 80 OpenCV threads on 20 cores. PyTorch limits its own threads to 1 per worker, OpenCV does not.
- **Trainer side (`BaseTrainer`):**
  - `get_next_batch` pulls one training batch per iteration, and one validation batch every
    `validation_iteration_ratio` iterations; at the end of an epoch it creates a new iterator;
  - `prepare_batch` moves the images to the device and casts them to float32;
  - `DataAugmenter.augment_batch` loops over the samples of the batch in Python, on the GPU;
  - `run_test_evaluation` reads the whole test split and skips the samples already evaluated, using their
    `sample_identifiers`, since a DALI loader may repeat samples to fill its last batch.
- **Detection DALI (`data/detection/coco_dali_pipeline.py`):**
  - the pipeline: `readers.coco` with `random_shuffle=True`, `decoders.image(device="mixed", output_type=BGR)`, then
    `resize(mode="not_larger")` and `crop_mirror_normalize` for the padding, with a float32 CHW output;
  - `DALIDataloaderWrapper` turns the padded boxes into lists;
  - `scripts/data/detection/dataloader_benchmarker.py` times epochs of both detection loaders, the loader alone.
- **Checks:** `check_classification_data_configuration` refuses `Data.DALI.activated: True` for classification, M2O, M2M
  and CMP, and the YAML comments say so.
- **Data on this machine:**
  - ImageNet-1k is prepared: 1,271,167 training JPEG images of about 500×375 pixels, 146 GB in total, more than the
    121 GB of memory;
  - the synthetic datasets are PNG: classification shapes at 256×256, moving shapes and object loss pairs at 128×128;
  - there is no real sequence dataset yet. The benchmark sequences and pairs made of ImageNet JPEG images are in
    `/home/robert_kelevra/Data/sequence_data/imagenet_sequences` and `imagenet_pairs` (section 5.3).

## 4. Preliminary Probe
**Setup (run on 2026-10-03):**
- **Hardware:** NVIDIA GB10, 20 Arm cores (10 Cortex-X925 and 10 Cortex-A725), 121 GB of unified memory.
- **Software:** torch 2.14 (CUDA 13.0), torchvision 0.29, DALI 2.3.0 (`nvidia-dali-cuda120`), nvImageCodec 0.9.0,
  OpenCV 5.0.

**No hardware JPEG decoding:**
- nvImageCodec's hardware backend cannot create its decoder.
- `nvidia-smi --query-gpu=utilization.jpeg` stays at 0 %, and `hw_decoder_load` changes nothing.
- This holds for both the CUDA 12 and the CUDA 13 DALI builds (`nvidia-dali-cuda130` 2.3.0 was tried in an isolated
  environment).
- DALI therefore decodes JPEG in hybrid mode: Huffman decoding on CPU threads, the rest on the GPU. PNG is decoded on
  CPU threads in every case.

**Decoding speed:** 3,072 ImageNet images read from the page cache and resized to 256 pixels with the ratio kept.

| Decoder | Images/s |
|---|---|
| OpenCV + `preprocess_image`, 1 process (20 OpenCV threads) | 912 |
| DALI mixed decoding, 4 threads | 3,506 |
| DALI mixed decoding, 12 threads | 5,076 |
| DALI CUDA 13 build, 12 threads | 4,850–5,020 |
| nvImageCodec hybrid batch decoding (decoding only, no resize) | 4,793 |
| torchvision `decode_jpeg` on the GPU, batches of 64 | 1,034 |

**Reading:**
- DALI decodes about 5 times faster than one OpenCV process, but the speed depends on the number of CPU threads.
- A Native loader with more workers may therefore close part of the gap, which is why the tuned Native variant is
  benchmarked.
- torchvision's GPU decoding is not competitive here.
- nvImageCodec alone is no faster than DALI and has none of its pipeline (resize, prefetching), so DALI remains the GPU
  option.

---

## 5. Phase 1: Benchmark

### 5.1 Questions Answered
1. How fast is the current Native loader of each task, alone and inside a training iteration?
2. How fast can the GPU train each configuration with no loading at all (the ceiling), in float32 as configured and
   in bfloat16?
3. How much of the gap does a tuned Native loader close?
4. How much do the DALI prototypes add on top, for JPEG and PNG, from the page cache and from the disk?
5. Do the DALI outputs match the Native ones: labels, identifiers and masks exactly, pixels closely?
6. How fast is the test evaluation pass? It is inference only, so the loader is more likely to be the limit.

### 5.2 Files
All files are new. No existing file is modified.

| File | Content |
|---|---|
| `scripts/data/loader_benchmark/run_loader_benchmark.py` | Entry point. The settings are in its main block: output folder, cases, variants, numbers of batches, repeats. Run with `.venv/bin/python -m scripts.data.loader_benchmark.run_loader_benchmark`. |
| `scripts/data/loader_benchmark/benchmark_datasets.py` | Builds the benchmark datasets and configurations, and controls the page cache. Its sequence and pair datasets are already written and built. |
| `scripts/data/loader_benchmark/dali_prototypes.py` | DALI pipelines and batch loaders of the three tasks, yielding the batch dictionaries of the Native loaders. |
| `scripts/data/loader_benchmark/benchmark_measurements.py` | The measured stages, the CPU and GPU sampling, the parity checks and the reports. |

**`benchmark_datasets.py`:**
- `create_benchmark_subset_manifest(source_split_folder: Path, output_split_folder: Path, number_samples: int, seed: int) -> Path`:
  - writes a classification split made of `number_samples` random ImageNet training images, keeping all categories;
  - the split has a `data` link to the source images and its own `labels.json`;
  - it fits in memory, so the warm runs measure decoding, not the disk.
- **Already written and run (2026-10-03):**
  - `draw_sequence_frames(image_paths: List[Path], split_sizes: Dict[str, int], minimum_length: int, maximum_length: int, random_generator: random.Random) -> Dict[str, List[List[Path]]]`:
    draws the length of every sequence uniformly between the two lengths, then distinct random images for all the
    frames.
  - `create_imagenet_sequence_dataset(imagenet_split_folder: Path, output_folder: Path, split_sizes: Dict[str, int], minimum_length: int, maximum_length: int, seed: int) -> None`:
    - copies the drawn ImageNet images, bytes unchanged, as `<split>/data/<split>_<index>/<frame>.jpg`;
    - writes `<split>/labels.json` with a random category per sequence: `category_id` for M2O and
      `frame_category_ids` for M2M, so the same dataset serves both;
    - writes `label_mapping.json` with `write_label_mapping`;
    - refuses to write into an existing split.
  - `create_sequence_pair_dataset(sequence_dataset_folder: Path, output_folder: Path, split_sizes: Dict[str, int], seed: int) -> None`:
    - every pair takes two different sequences of the same split as its before and after members, with a random
      category;
    - `<split>/data` links to the sequence frames, so no frame is copied twice.
  - Run with `.venv/bin/python -m scripts.data.loader_benchmark.benchmark_datasets`; the settings are in its main
    block (seed 0).
- `write_benchmark_configuration(template_path: Path, output_path: Path, split_folders: Dict[str, Path], overrides: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]`:
  - copies a YAML of the repository with the benchmark split folders and label mapping file;
  - loads it with `load_experiment_configuration`, so the label dictionary is injected exactly as in a training run.
- `load_into_page_cache(file_paths: List[Path]) -> None` and `evict_from_page_cache(file_paths: List[Path]) -> None`:
  - they read the files, or drop them from the page cache with `os.posix_fadvise(POSIX_FADV_DONTNEED)`;
  - this gives warm and cold runs without root rights.

**`dali_prototypes.py`:**
- `resize_and_pad_images(images: DataNode, image_size: int, keep_ratio: bool) -> DataNode`:
  - with the ratio kept: `fn.resize(mode="not_larger", interp_type=INTERP_LINEAR, antialias=False)`, then
    `fn.crop_mirror_normalize(dtype=UINT8, output_layout="CHW", crop=(size, size), crop_pos_x=0.5, crop_pos_y=0.5, out_of_bounds_policy="pad", fill_values=0)`;
  - without it: a direct resize to the square, then the same layout conversion.
- `classification_pipeline(file_paths: List[str], sample_indices: List[int], image_size: int, keep_ratio: bool, shuffle: bool)`,
  decorated with `@pipeline_def`:
  - `fn.readers.file(files=..., labels=sample_indices, shuffle_after_epoch=shuffle, pad_last_batch=True, name="Reader")`;
  - then `fn.decoders.image(device="mixed", output_type=BGR)` and `resize_and_pad_images`;
  - the "label" read by DALI is the position of the sample in the dataset, from which the loader looks up the real
    labels.
- `ClassificationDALILoader(dataset: ClassificationDataset, batch_size: int, number_threads: int, prefetch_queue_depth: int, shuffle: bool, device_id: int)`:
  - yields `{"images": uint8 [B, 3, S, S] on the GPU, "labels": int64 [B] or float32 multi-hot [B, K], "sample_identifiers": int64 [B]}`;
  - the labels and identifiers are looked up in GPU tables indexed by the sample position;
  - `LastBatchPolicy.PARTIAL`, so every sample is given exactly once per epoch;
  - `.dataset` (the Native dataset, for the metadata) and `__len__`.
- `WindowFrameSource`, a top-level class that DALI's reader processes can pickle:
  - `__init__(data_directory: str, sample_frames: List[List[Optional[str]]], frames_per_sample: int, batch_size: int, shuffle: bool, seed: int)`:
    - `sample_frames` holds the F frame slots of every sample, `None` for the padding slots;
    - for a pair, the L before slots come first, then the L after slots, as in `SequencePairDataset`;
  - `__call__(batch_information: BatchInfo) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]`:
    - returns the encoded bytes of the real frames of the batch, read with `np.fromfile`;
    - with each frame, its slot in the batch (sample position × F + frame position) and its sample index;
    - the padding slots are not decoded at all, which saves up to half of the work for pairs with short members;
    - it raises `StopIteration` at the end of the epoch;
    - the order is a permutation drawn from the seed and the epoch index, so it changes at every epoch and can be
      reproduced.
- `sequence_pipeline(source: WindowFrameSource, image_size: int, keep_ratio: bool)`, decorated with `@pipeline_def`:
  - `fn.external_source(source=source, num_outputs=3, batch=True, batch_info=True, parallel=True)`;
  - mixed decoding, then `resize_and_pad_images`;
  - pipeline arguments: `batch_size` = B × F (the largest number of real frames in a batch), `py_num_workers`,
    `py_start_method="spawn"`.
- `SequenceDALILoader(dataset: SequenceDataset | SequencePairDataset, batch_size: int, number_threads: int, number_reader_processes: int, shuffle: bool, device_id: int)`:
  - writes the decoded frames into a zero tensor `[B × F, 3, S, S]` at their slots, then reshapes it to
    `[B, F, 3, S, S]`, so the padding frames are exact zeros as in the Native loader;
  - builds the `sequence_mask` from the slots, and the labels and identifiers from per-sample tables taken from the
    Native dataset records: the window class, the frame classes with -100 on the padding, or the pair class;
  - `.dataset` and `__len__`.

**`benchmark_measurements.py`:**
- `limit_opencv_threads(worker_index: int) -> None`: the `worker_init_fn` of the tuned Native loaders, calling
  `cv2.setNumThreads(1)`.
- `run_training_step(model: TaskModel, batch: Dict[str, Any], task: str, augmenter: DataAugmenter, optimizer: torch.optim.Optimizer, mixed_precision: bool) -> None`:
  - the training iteration of `BaseTrainer`: cast to float, augmentation, forward pass with the labels, sum of the
    `*_loss` outputs, backward pass and optimizer step;
  - the model is called like the `forward_pass` hooks do:
    - `model(input_tensor=..., labels=...)` for classification;
    - `model(input_tensor=..., sequence_mask=..., labels=...)` for M2O and M2M;
    - `model(**split_pair_members(...), labels=...)` for CMP.
- `measure_loader(loader: Iterable, number_warmup_batches: int, number_measured_batches: int, device: torch.device, frames_per_sample: Optional[int]) -> Dict[str, float]`:
  - the time to the first batch, then samples/s, frames/s and batches/s;
  - the number of busy CPU cores;
  - the images of every batch are moved to the GPU and synchronised, as `prepare_batch` does.
- `measure_training_ceiling(model: TaskModel, batch: Dict[str, Any], task: str, augmenter: DataAugmenter, mixed_precision: bool, number_iterations: int) -> Dict[str, float]`:
  - one fixed batch kept on the GPU, so no loading at all;
  - measures the iterations/s of the augmentation alone, the step alone, and the augmentation plus the step.
- `measure_end_to_end(loader: Iterable, model: TaskModel, task: str, augmenter: DataAugmenter, mixed_precision: bool, number_iterations: int) -> Dict[str, float]`:
  - the loader followed by `run_training_step`;
  - iterations/s, samples/s and frames/s;
  - GPU and CPU utilisation.
- `measure_evaluation_pass(loader: Iterable, model: TaskModel, task: str, maximum_batches: int) -> Dict[str, float]`:
  inference without gradients on the test split.
- `compare_loader_outputs(native_dataset: Dataset, dali_loader: Iterable, number_samples: int, output_path: Path) -> Dict[str, float]`:
  - matches the samples by identifier;
  - checks that labels, masks and identifiers are exactly equal;
  - measures the mean and maximum absolute pixel difference, and the share of pixels differing by more than 8 levels;
  - writes a Native | DALI | difference × 10 image.
- `UtilisationSampler`: a top-level thread class sampling `utilization.gpu`, `utilization.jpeg` (`nvidia-smi`) and the
  CPU (`psutil`) every 0.5 s.
- `write_benchmark_reports(rows: List[Dict[str, Any]], output_folder: Path) -> Tuple[Path, Path]`:
  - `results.csv`, with one row per case, variant, stage and repeat;
  - `summary.md`, with the median of the repeats, the speedups against the current Native loader, and the headroom
    (loader rate divided by the GPU ceiling).

### 5.3 Benchmark Cases
| Case | Task | Data | Format → input | Settings (from the YAML) |
|---|---|---|---|---|
| C-IN | classification | ImageNet subset, 50,000 training images (5.5 GB) | JPEG ~500×375 → 256 | `classification/configuration_example.yaml`: batch 64 |
| C-SYN | classification | synthetic shapes, train split (4,000) | PNG 256×256 → 256 | `configuration_synthetic_shapes.yaml`: batch 64 |
| S-SYN | M2O | synthetic moving shapes, train split (2,000 × 8 frames) | PNG 128×128 → 128 | `configuration_synthetic_sequences.yaml`: L 8, batch 16 |
| S-JPG | M2O | `imagenet_sequences`, train split: 800 sequences of 6–30 copied ImageNet frames (14,088 frames, 1,424 windows of 16) | JPEG ~500×375 → 256 | `sequence_classification/configuration_example.yaml`: L 16, window stride 8, batch 8 |
| P-SYN | CMP | synthetic object loss pairs, train split (2,000) | PNG 128×128 → 128 | `configuration_synthetic_pairs.yaml`: L 8, batch 16 |
| P-JPG | CMP | `imagenet_pairs`, train split: 800 pairs of two of these sequences (members of 6–30 frames) | JPEG ~500×375 → 256 | `sequence_comparison/configuration_example.yaml`: L 16, batch 8 |
| D-COCO (optional) | detection, reference | COCO train | JPEG → 1024 | the existing DALI and Native detection loaders |

- **Page cache:** every case is measured warm (files already in memory). C-IN, S-JPG and P-JPG are also measured cold
  (files evicted before every repeat), since ImageNet is larger than the memory and every epoch reads it from the disk.
- **M2M labels:** the M2M frame labels are checked on S-JPG in the parity stage, with `label_type="frame"`.
- **Real data:** real sequence datasets are added to the case list of the main block once they exist.

### 5.4 Loader Variants
| Variant | Description |
|---|---|
| `native_current` | Exactly the loaders of the trainers: `get_classification_dataloaders` and `get_sequence_dataloaders` with 4 workers. |
| `native_tuned` | The same datasets in `DataLoader(num_workers ∈ {8, 12, 16}, prefetch_factor ∈ {2, 4}, persistent_workers=True, pin_memory=True, worker_init_fn=limit_opencv_threads)`. |
| `dali` | The prototypes, with `num_threads ∈ {4, 8, 12, 16}` and `prefetch_queue_depth ∈ {2, 3}`, plus 2 or 4 reader processes for the sequences. |

For `native_tuned` and `dali`, the best combination of stage A is the one kept for stages C and D.

### 5.5 Measured Stages
| Stage | What is measured | Cases × variants |
|---|---|---|
| A. Loader alone | 20 warm-up and 200 measured batches, moved to the GPU | all × all |
| B. GPU ceiling | Fixed GPU batch: augmentation alone, step alone, augmentation and step; float32 (as configured) and bfloat16 | all cases, once |
| C. End-to-end training | 20 warm-up and 200 iterations of loader, preparation, augmentation and step | all × {`native_current`, best `native_tuned`, best `dali`} |
| D. Evaluation pass | Inference on the test split, at most 100 batches, including the start of the loader | all × the same 3 |
| E. Parity | 256 samples, Native against DALI | all cases |
| F. Disk | Cold read rate of 20,000 ImageNet files outside the subset, with 8 threads | once |

- Every stage is repeated 3 times; the reports give the median and the range.
- The augmentation follows the YAML of the case. Stage B shows its cost separately, since a faster loader cannot help
  if the augmentation is the bottleneck.

### 5.6 Protocol
- **One process per case:** every case runs in its own process (`multiprocessing` with the `spawn` method), so that
  the DALI pipelines, the workers and the CUDA context of one case do not affect the next one.
- **Quiet machine:** nothing else runs on the machine during the benchmark (no training).
- **Environment report:** the GPU clocks, the versions and the CPU frequency policy are written to `environment.txt`.
- **Real models:** the ceilings and end-to-end stages use the models of the YAMLs, built from the model registry, with
  AdamW, so the numbers are those of a real training run without its logging.
- **Duration:** about 1 to 1.5 hours for every case and variant.

### 5.7 Outputs
- `<output folder>/<date>_<time>/results.csv`, `summary.md`, `environment.txt` and `Parity/<case>.png`, printed as
  `file://` links. The default output folder is `/home/robert_kelevra/Data/loader_benchmark/`, next to the benchmark
  datasets.
- The summary is copied into section 13 of this document.

### 5.8 Decision Gate
These thresholds are proposals: the user decides from the results, and no later phase starts without that decision.

- **Native loader settings (phase 2):** if `native_tuned` is at least 1.10 × `native_current` end to end on any case.
  It is a cheap change: a configuration block and a few lines.
- **DALI for classification (phase 3):** if `dali` is at least 1.25 × `native_tuned` end to end on C-IN (ImageNet), or
  at least 1.5 × on its evaluation pass.
- **DALI for the sequence tasks (phase 4):** if `dali` is at least 1.25 × `native_tuned` end to end on S-JPG or P-JPG,
  or on the real data once benchmarked. The synthetic and PNG cases are for information.
- **Parity, required before any integration:**
  - labels, identifiers and masks are equal;
  - the mean absolute pixel difference is at most 1 level;
  - at most 1 % of the pixels differ by more than 8 levels.

  Otherwise the prototype resize and padding are corrected first.

---

## 6. Design Of The Integration
Every subsection below applies only if the decision gate keeps it.

### 6.1 Native Loader Settings (phase 2)
- **New `Data.loading` block** (final keys and defaults from the benchmark):
  ```yaml
  Data:
    loading:
      # Number of worker processes of the Native loaders, and of CPU threads of the DALI pipelines
      number_workers: 12
      # Number of batches prepared in advance by every worker (Native) or by the pipeline (DALI)
      prefetch_batches: 2
  ```
- **`vision_training/data/loaders.py`:**
  - new `build_native_loader(dataset: Dataset, batch_size: int, loading_configuration: Dict[str, Any], collate_function: Optional[Callable] = None) -> DataLoader`,
    shared by the detection, classification and sequence loader functions instead of their three `DataLoader` calls;
  - it sets `persistent_workers` and `prefetch_factor`, `pin_memory` if it mattered, and
    `worker_init_fn=limit_opencv_threads`;
  - `limit_opencv_threads` moves from the benchmark to this module.
- **Trainers:** they read `Data.loading.number_workers` instead of the hard-coded 4. The detection DALI pipeline uses it
  for its threads.
- **Checks:** `number_workers` is an integer ≥ 0 (0 loads in the main process, for debugging), and `prefetch_batches`
  is an integer ≥ 1.
- **Migration:** `add_loading_settings_to_configuration` writes `number_workers: 4` and `prefetch_batches: 2`, the
  previous behaviour, into older runs.
- **YAMLs:** all 10 configurations are updated with the commenting style of the files.

### 6.2 DALI For Classification (phase 3)
- **New `vision_training/data/dali_loading.py`, the parts shared by the tasks:**
  - `resize_and_pad_images`, with an `output_dtype` argument, since the detection pipeline keeps its float32 output;
  - `DALIBatchLoader`, the base class: `.dataset`, `__len__`, `__iter__` restarting at every `iter()`, and the device
    index taken from the trainer device.
- **The detection pipeline** uses `resize_and_pad_images`, with its output unchanged (checked bit for bit by a test).
- **New `vision_training/data/classification/classification_dali_pipeline.py`:** `classification_dali_pipeline` and
  `ClassificationDALILoader`, taken from the prototypes.
- **`get_classification_dataloaders`:** with `Data.DALI.activated`, it builds the DALI loader of the three splits:
  - the training and validation splits are shuffled at every epoch (`shuffle_after_epoch`);
  - the test split is read in manifest order.
- **Metadata:** the DALI loaders build the `ClassificationDataset` (unshuffled) for the records and metadata, so
  `.dataset` keeps working for the trainer (class names, class counts, evaluation outputs).
- **Checks:** `check_classification_data_configuration` accepts DALI for classification, and the YAML comments are
  updated.
- **`prepare_batch` is unchanged:** moving a CUDA tensor to the GPU does nothing, and the cast to float32 stays.

### 6.3 DALI For The Sequence Tasks And The Comparison (phase 4)
- **New `vision_training/data/sequence/sequence_dali_pipeline.py`:** `WindowFrameSource`, `sequence_dali_pipeline` and
  `SequenceDALILoader`, taken from the prototypes. One loader serves windows and pairs, a pair being F = 2L slots.
- **`get_sequence_dataloaders`:** a DALI branch for the label types `"sequence"`, `"frame"` and `"pair"`.
- **Checks:** DALI is accepted for M2O, M2M and CMP.
- **Augmentation:** frame shuffling and per-frame draws stay in `DataAugmenter`, after the loading, unchanged.

## 7. Configuration Summary
| Key | Default | Tasks |
|---|---|---|
| `Data.loading.number_workers` | from the benchmark (4 in migrated runs) | every task: Native workers, DALI threads |
| `Data.loading.prefetch_batches` | 2 | every task: Native prefetch factor, DALI prefetch queue depth |
| `Data.DALI.activated` | `False` | now accepted for classification, M2O, M2M and CMP |
| `Data.DALI.number_reader_processes` | from the benchmark (e.g. 4) | M2O, M2M, CMP: the processes reading the frame files for DALI |

- **Run names:** unchanged, since the loader does not change the model.
- **Migrations:** the new keys are inserted as text, keeping the comments, and a file already holding them is left
  untouched.
- **Final keys:** only the settings that changed something in the benchmark become keys; the others keep fixed values.

## 8. Phases
Each phase follows the previous workflow:
- the README checklist of the phase;
- implementation;
- tests (`pytest` and the `ruff` name checks);
- acceptance;
- the README checklist removed;
- one commit, then a push.

| Phase | Content | Commit tag |
|---|---|---|
| 0 | This plan and its `.gitignore` exception | `[DOCS]` |
| 1 | Benchmark scripts (new files only), the runs, and the results recorded in section 13 | `[BENCHMARK]` |
| — | Decision gate with the user (5.8) | — |
| 2 | Native loader settings (6.1), if kept | `[DATA]` |
| 3 | Shared DALI parts and the classification DALI loader (6.2), if kept | `[DATA]` |
| 4 | DALI loader of the sequence tasks and the comparison (6.3), if kept | `[DATA]` |
| 5 | Benchmark switched to the integrated loaders, prototypes deleted, final benchmark run | `[BENCHMARK]` |
| 6 | Implementation notes and results in this document | `[DOCS]` |

## 9. Tests
**Phase 1**
- New `tests/scripts/data/test_loader_benchmark.py`: the benchmark runs on a tiny generated classification and pair
  dataset, with 2 batches per stage and one variant of each family, and writes its reports. It catches a change of
  the imported APIs breaking the script.
- The parity stage itself verifies the prototypes on the real cases.

**Phase 2**
- `build_native_loader` passes the settings (workers, prefetch, persistence, initialisation function).
- The trainers read `Data.loading.number_workers`.
- Checks and migration of the new block. The existing loader and trainer tests pass unchanged.

**Phase 3**
- **Epochs:** every sample exactly once per epoch, the training order changing between epochs, the test split in
  manifest order, and the last partial batch.
- **Labels:** multi-class and multi-label labels, and identifiers equal to those of the Native dataset.
- **Pixel parity** with the Native dataset, on generated JPEG and PNG images of several sizes and aspect ratios, with
  the ratio kept or not, within the thresholds of phase 1.
- **Interface:** `.dataset` metadata and `len`.
- **Detection:** the pipeline output is unchanged with the shared resize (bit-identical on generated images).
- **Trainer:** a classification smoke run with DALI (5 iterations): evaluation files, predictions agreeing with
  `Results.txt`.
- **Checks:** DALI is accepted for classification.

**Phase 4**
- **Windows:** padding, masks and padding frames that are exact zeros; frame labels with -100 on the padding.
- **Pairs:** before and after slots, with padding per member.
- **Epochs:** every window and pair exactly once per epoch.
- **Identifiers and pixel parity** with the Native datasets.
- **Trainers:** M2O, M2M and CMP smoke runs with DALI.
- **Checks:** DALI is accepted for the three tasks.

## 10. Acceptance
- **Phase 1:** the benchmark report with every case, recorded in section 13, and the gate decision.
- **Phase 2:** the training throughput printed in the console, before and after, on ImageNet classification and on
  the synthetic pairs, consistent with the benchmark gain.
- **Phase 3:**
  - synthetic shapes classification trained with DALI reaches the Native test accuracy within 1 point, with the same
    configuration;
  - the throughput of an ImageNet classification run, Native against DALI, over its first 1,000 iterations.
- **Phase 4:**
  - comparison pairs with DALI reach at least 95 % test accuracy;
  - moving shapes M2O trains with DALI as it does with Native;
  - the throughput on S-JPG and P-JPG.
- **Phase 5:** the benchmark run with the integrated loaders gives the numbers of the prototypes.

## 11. Risks And Open Points
- **No hardware JPEG decoder (section 4):** the gain is limited by the Huffman decoding on the CPU threads. DALI's
  advantage comes from its C++ threads, the resize on the GPU and the absence of copies between processes, and may be
  modest against a tuned Native loader. This is what the benchmark measures.
- **Unified memory (GB10):** pinned memory and host-to-device copies behave differently from a discrete GPU, so their
  effect is measured, not assumed.
- **Preprocessing parity:**
  - DALI's rounding of the resized size, the padding offset at `crop_pos 0.5`, and nvJPEG against libjpeg-turbo can
    differ from OpenCV by a level or a pixel;
  - models trained on DALI inputs are deployed with `preprocess_image` (OpenCV), so the parity thresholds bound this
    difference;
  - antialiasing must stay off (DALI turns it on by default when downscaling, OpenCV `INTER_LINEAR` does not use it).
- **Shuffling:** `readers.file(random_shuffle=True)` only shuffles within a buffer of 1,024 samples by default, while
  the ImageNet manifests are sorted by class, which would give batches of one or two classes. The classification
  pipeline must use `shuffle_after_epoch=True`, a full permutation at every epoch.
- **Reader processes:** `parallel=True` starts Python processes. With CUDA already initialised in the trainer, they
  must be spawned (`py_start_method="spawn"`), and the source must be a top-level class that can be pickled.
- **Iterator restart:** a DALI iterator left in the middle of an epoch (e.g. the input viewer stopped early) continues
  from there at the next `iter()`. This is harmless for training, and the test loader is always read to the end.
- **Memory:** DALI allocates its buffers in the unified memory shared with the model. Their size grows with the
  prefetch depth, the batch size and the number of frames.
- **Representativeness:**
  - there is no real sequence dataset yet, so ImageNet JPEG images stand in for the frames, in sequences of 6 to 30
    frames;
  - the real sequences will be JPEG too, but their resolution and lengths may differ from the ImageNet images (about
    500×375). The decoding cost grows with the number of pixels, so the real data must be benchmarked once it exists.
- **Augmentation cost:** if `augment_batch` (a Python loop over the samples) is the bottleneck, faster loading gains
  nothing. Stage B shows it, and a batched augmentation would be a separate plan.
- **DALI build:** the project depends on `nvidia-dali-cuda120` while torch uses CUDA 13. The CUDA 13 build gave the
  same speed and no hardware decoder, so the dependency is kept as it is.

## 12. Verification
- In phase 1, `git status` shows only new files.
- `pytest` passes after every phase, and the `ruff` name checks
  (`NO_COLOR=1 uvx ruff check --no-cache --select F401,F811,F821,F841 vision_training tests scripts`) report nothing
  new.
- The benchmark results and the gate decision are recorded in this document before phase 2.

## 13. Benchmark Results

### 13.1 Differences From The Plan
- **Preprocessing parity of DALI:** two settings were needed for the DALI images to match `preprocess_image`, found
  on 256 ImageNet images before writing the prototypes:
  - `fn.resize(..., subpixel_scale=False)`: by default DALI keeps the exact scale factor and samples a fractional
    region, while `cv2.resize` maps the image onto its rounded size; the content drifted by up to half a pixel towards
    the edges;
  - a crop position of `0.4999` instead of `0.5` for the padding: DALI rounds the centre position up, while
    `preprocess_image` puts the extra pixel of an odd padding at the bottom and on the right
    (`(size - new size) // 2`). With 0.5, 35 images out of 128 were shifted by one pixel (mean difference 8.3
    levels on those images).

  With both, DALI with CPU decoding matches OpenCV within one level (mean 0.12). With GPU decoding, the mean
  difference is 0.4 levels and 0.03 % of the pixels differ by more than 8 levels, from the nvJPEG decoder.
- **Start method of the case processes:** every case runs in a process created with the `spawn` method, which made
  `spawn` the default start method inside it, so the DataLoader workers were started as new interpreters (about
  0.8 s each, one after another) instead of being forked as in a training process (0.1 s). The case processes now
  force `fork` for the DataLoader workers; the DALI reader processes keep their explicit `spawn`.
- **GPU sampling:** the GPU utilisation is sampled by an `nvidia-smi --loop-ms` process instead of a thread. A
  DataLoader forking its workers while the sampling thread ran left the workers blocked forever (found in the dry
  run, in the evaluation pass of a tuned loader).
- **Iterations:** 100 measured iterations for the training iteration (stage C) and 30 for the GPU ceiling (stage
  B), instead of 200 and 50, the example models training at a few iterations per second.
- **DALI prefetch queue depth:** fixed at 2 instead of swept over {2, 3}, to keep the run within about 1.5 hours.
- **Evaluation pass:** the reports give the throughput with the start of the loader included (as planned), the
  throughput after the first batch, and the estimated duration of an evaluation of the whole test split.
- **Smoke test:** a classification case (also cold) and a sequence comparison case on generated datasets, plus a
  test of the frame slots and label tables of the DALI sequence loader against the Native windows and pairs (frame
  labels included). It takes about 40 s.
- **D-COCO:** the optional detection case was not run.

### 13.2 Results
Run of 2026-10-03, 15:08 to 16:10, 425 measurements, medians of 3 repeats. Full reports:
`/home/robert_kelevra/Data/loader_benchmark/20261003_150842/` (`summary.md`, `results.csv`, `Parity/`). The frames/s
count the frame slots of the windows and pairs, padding included.

**Training iteration (stage C), samples/s, against the GPU ceiling (stage B)**

| Case | Current | Best tuned Native | Best DALI | Ceiling float32 | Ceiling bfloat16 | Current / ceiling |
|---|---|---|---|---|---|---|
| C-IN | 364 | 366 (1.01×) | 363 (1.00×) | 388 | 633 | 94 % |
| C-SYN | 412 | 432 (1.05×) | 441 (1.07×) | 459 | 771 | 90 % |
| S-SYN | 189 | 198 (1.05×) | 196 (1.04×) | 213 | 373 | 89 % |
| S-JPG | 26 | 27 (1.03×) | 27 (1.03×) | 29 | 48 | 89 % |
| P-SYN | 180 | 192 (1.07×) | 202 (1.13×) | 193 | 353 | 93 % |
| P-JPG | 14 | 14 (1.03×) | 15 (1.05×) | 15 | 24 | 95 % |

**Loader alone (stage A, warm page cache), samples/s (frames/s)**

| Case | Current (4 workers) | Best tuned Native | Best DALI | Current / float32 ceiling |
|---|---|---|---|---|
| C-IN | 2,937 | 9,645, 16 workers (3.28×) | 5,419, 16 threads (1.85×) | 7.6 |
| C-SYN | 5,227 | 9,343, 16 workers (1.79×) | 8,896, 8 threads (1.70×) | 11.4 |
| S-SYN | 2,734 (21,871) | 4,322, 16 workers (1.58×) | 1,974, 4 threads (0.72×) | 12.8 |
| S-JPG | 188 (3,014) | 625 (10,007), 16 workers (3.32×) | 411 (6,573), 16 threads (2.18×) | 6.4 |
| P-SYN | 2,089 (33,416) | 2,681, 16 workers (1.28×) | 1,999, 4 threads (0.96×) | 10.9 |
| P-JPG | 100 (3,192) | 301 (9,648), 16 workers (3.02×) | 254 (8,130), 12 threads (2.55×) | 6.8 |

- **Cold page cache (JPEG cases):** C-IN 2,581 / 8,360 / 5,479, S-JPG 183 / 571 / 412, P-JPG 94 / 302 / 257
  samples/s (current / tuned / DALI), within 15 % of the warm runs.
- **Disk:** 2,229 MB/s and 19,269 files/s on cold ImageNet files (116 KB per file on average), about twice the
  1.1 GB/s read by the fastest loader.
- **CPU:** on the JPEG cases, the current loader keeps about 18 cores busy (4 workers with 20 OpenCV threads each)
  for 2,937 images/s, the tuned loader 17 cores for 9,645 images/s, DALI 8 cores for 5,419 images/s.

**Evaluation of the whole test split (stage D), seconds (current / tuned / DALI)**
- C-IN (50,000 images): 43.0 / 38.4 (1.12×) / 42.0 (1.02×), the inference itself running at about 1,200 images/s.
- S-JPG (186 windows): 3.0 / 2.4 / 2.3, P-JPG (100 pairs): 3.1 / 2.6 / 2.2. The synthetic test splits take less
  than a second with any loader.

**Parity (stage E):** labels, frame labels, masks and identifiers equal in every case. Mean pixel difference 0.40
levels on JPEG (at most 0.03 % of the pixels above 8 levels), 0 on PNG.

**Decision gate (5.8), measured**

| Gate | Threshold | Measured | Met |
|---|---|---|---|
| Native loader settings (phase 2) | tuned ≥ 1.10 × current end to end on a case | 1.01 to 1.07 × | No |
| DALI for classification (phase 3) | DALI ≥ 1.25 × tuned on C-IN, or ≥ 1.5 × on its evaluation | 0.99 × and 0.91 × | No |
| DALI for the sequence tasks (phase 4) | DALI ≥ 1.25 × tuned on S-JPG or P-JPG | 1.00 × and 1.02 × | No |
| Parity | ≤ 1 level on average, ≤ 1 % of the pixels above 8 levels | 0.40 levels, 0.03 % | Yes |

### 13.3 Observations
- **The GPU is the limit in every case:** the current loader already feeds 89 to 95 % of the GPU ceiling, and alone
  it is 6 to 13 times faster than the ceiling in float32 (4 to 7 times in bfloat16). The faster loaders only remove
  the small stalls of the current one (its worker restarts at every epoch, the copies between processes), hence the
  1 to 13 % gains, the largest on the small synthetic datasets whose epochs are short.
- **Tuned Native beats DALI on JPEG here:** without a hardware JPEG decoder, DALI's hybrid decoding is limited by its
  CPU threads (about 5,400 ImageNet images/s at 256 pixels), while 16 Native workers with one OpenCV thread each reach
  9,600 images/s. On the small synthetic PNG frames, the DALI sequence prototype is slower than the current loader
  (per-sample overhead of the pipeline and its reader processes).
- **OpenCV oversubscription:** the current loader runs 20 OpenCV threads in each of its 4 workers. One OpenCV
  thread per worker gives about 3.5 times more images per busy core (581 against 157 ImageNet images/s).
- **Mixed precision is the larger lever:** bfloat16 raises the GPU ceiling 1.6 to 1.8 times in every case
  (`Optimization.mixed_precision`, off in every configuration), far more than any loader. Its effect on accuracy
  would need its own check.
- **Augmentation:** 15 ms per batch of 64 ImageNet images (9 % of a float32 iteration, 15 % in bfloat16), 9 ms per
  batch of 8 ImageNet pairs, about 1 to 3 ms elsewhere: a batched augmentation would be the next lever after mixed
  precision for the classification task.
- **DALI build time:** 1.1 to 1.6 s for the classification pipeline, 3.9 to 7.0 s for the sequence pipeline, which
  spawns its reader processes.
- **Detection DALI pipeline:** it uses the default `subpixel_scale` and a crop position of 0.5, so its images can be
  one pixel off the Native loader and `preprocess_boxes`, which computes the boxes for the Native geometry. Left as
  it is in this phase; the shared resize of phase 3 would fix it.

### 13.4 Decision
Pending the user. With the thresholds of 5.8, none of phases 2 to 4 is kept.
