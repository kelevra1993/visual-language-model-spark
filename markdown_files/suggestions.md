# Suggestions

> This file is kept at the root of the project as a record of proposed improvements, so that we can come back to it
> later and decide what we want to implement. Nothing listed here has been implemented yet.

## A. From the classifier, not yet used

### Evaluation and analysis
The biggest gap: `run_test_evaluation` is still a stub, and nothing uses the test set.

1. **Full evaluation at each save-and-evaluate checkpoint**, like the classifier's `Results.txt`: COCO mAP (0.5:0.95,
   and at 0.5 and 0.75), plus precision and recall per class, one block of lines per checkpoint.
2. **A confusion matrix with a background row and column**, like the classifier's confusion-matrix PNG. It would show
   which shapes get confused with each other, as well as false positives (background detected as an object) and
   missed objects.
3. **A per-class score threshold**, the equivalent of `threshold_finder`: the score threshold that reaches a target
   precision (e.g. 99%) for each class. This would replace the single `score_threshold: 0.5`. The export could write
   one threshold per class, like the classifier's `threshold_<iteration>.txt`.
4. **Error dumps**, like `LABEL-TO-PREDICTION`: annotated images saved into `FALSE-POSITIVE/<class>`,
   `MISSED/<class>` and `<label>-TO-<prediction>` folders, drawn with the existing `add_bounding_boxes`.
5. **A predictions JSON in COCO results format**, the equivalent of `Validation_Predictions.json`, so a checkpoint's
   detections can be analysed again without re-running the model.
6. **An evaluate-any-checkpoint script**, like `infer.py`: all checkpoints, or one iteration, using the
   `full-checkpoint` registry. `get_checkpoint_path` already exists.

### Experiment bookkeeping
7. **Automatic run names built from the settings**, like the classifier's `BN-KAR-300-300-2(32)_…`, for example
   `SE-RES-KAR-1024-7B-L567-A32-512`. This would avoid stale experiment folders whose config no longer matches.
8. **`params.json` / `data.json` equivalents:** `training_configuration.yaml` is already saved. Adding a `data.json`
   listing the images of each split would record exactly what a model was trained on.
9. **Dataset sizes printed at startup:** the trainer still has a TODO for this, and the loaders already know the
   numbers.
10. **Weight histograms in TensorBoard**, like `keep_distributions`, as a config flag.
11. **Refusing to train without a GPU**, instead of silently falling back to the CPU.

### Model options
12. **Dropout before the classification layer:** the classifier used 0.5. The detection head's fully connected layers
    have none.
13. **Global average pooling instead of flattening in the detection head** (`average_pool`): cheaper, and less tied
    to the RoIAlign pool size.
14. **A choice of weight initialisation:** the classifier used a truncated normal with standard deviation 0.1, and
    biases set to 1. The detector uses PyTorch's defaults. It could be a config option to compare the two.

### README, modelled on the classifier's
The classifier README's sections were:
- installation;
- configuration;
- input pipelines;
- training settings, with the run naming scheme;
- model architecture;
- hyperparameters;
- TensorBoard;
- data augmentation, with images;
- validation / test and the Excel analysis;
- deployment (simple and Triton);
- additional settings: GPU monitoring, receptive field, model size;
- complex architectures: residual, inception, squeeze-and-excitation, attention, each with a picture;
- known bugs.

The README rules say it only has the table of contents, the description and the prerequisites, and that the other
sections are chosen by hand. Proposed outline for the detector, to be written once the sections are chosen:
- **Data:** COCO format, the synthetic dataset, DALI vs Native.
- **Configuration:** each section, and the checker.
- **Architecture:** backbone modules, the 3 levels, anchors, RPN → filtering → detection heads → post-processing,
  with the real numbers (13824 anchors, 500 / 250 proposals, NMS 0.5, RoIAlign 7×7, score 0.5, NMS 0.8, top 100).
- **Training:** save frequencies, resume, augmentation with before/after images.
- **Diagnostics:** receptive field, model size, input viewer.
- **Evaluation:** once it exists.
- **Deployment:** export, Triton, the client.
- **Known issues:** the intermittent DALI/CUDA crash, and resuming counting iterations from the resume point.

## B. Refactoring the object detector
1. **Reuse the batched proposal filter in `Model`.** `InferenceDetector.filter_proposals` does the same thing as
   `RegionProposalFilter`, without the per-image loop or the random-copy padding. Using it in both would remove
   duplicated code, speed up training, and make training and deployment filter identically.
2. **Split `model.py` (about 1000 lines).** The losses would move to `loss/`, and the post-processing to its own
   module. `Model` would keep only building the network and the forward pass.
3. **Split `data_loader.py`** into DALI, Native, and a factory. The label-mapping logic is still duplicated between
   `NativeDataset` and `DALIDataloaderWrapper`.
4. **Vectorise the per-image loops in training**, in target assignment, sampling and the DALI wrapper's un-padding.
   They are a likely part of the low throughput (about 10 images per second at batch 40 and 1024 px).
5. **Batch the augmentations:** they currently cost about 280 ms per batch of 40 images at 1024 px with all types on.
6. **Trainer fixes:**
   - A resumed run trains `number_iterations` more from the resume point, instead of stopping at the total.
   - Console output is buffered when written to a file. The trainer could flush its prints, or use Python's
     `logging` module.
   - A background run can't be stopped with Ctrl+C (SIGINT is ignored for background processes of non-interactive
     shells). The trainer could also handle SIGTERM, doing the same final save as for Ctrl+C.
7. **Group the per-level config in one place** (anchors + region proposal + detector per level). The YAML TODOs
   already point at this. The checker currently has to verify that four sections list the same levels.
8. **Find the cause of the DALI/CUDA crash** (`cudaErrorIllegalAddress`, with `Xid 13` / `Xid 31` GPU faults in the
   kernel log): run `compute-sanitizer` on a short run (e.g. 200 images, a few epochs), or run a long Native-loader
   training to see whether DALI is involved.

## C. New ideas for the object detector
1. **A top-down feature pyramid (FPN):** lateral 1×1 convolutions plus upsampling would give level 5 the deeper
   context of levels 6 and 7. All levels would then share the same channel count, and could share one detection
   head.
2. **Choosing the detection level by box size** (the FPN rule `k = k₀ + log₂(√(wh)/224)`), instead of by the level
   the proposal came from.
3. **NMS per level in the proposal filter:** use `batched_nms` with the level as the group, so that one strong level
   doesn't suppress the proposals of the others.
4. **Training setup:**
   - AdamW with weight decay;
   - a learning-rate warm-up, then cosine decay;
   - gradient clipping;
   - configurable weights for the four losses.
5. **Mixed precision:** bf16 `torch.autocast` on the GB10, plus `torch.compile`. Together with vectorising, this is
   the main way to speed up training.
6. **An EMA of the weights for evaluation and export:** a moving average of the weights, which is usually more stable
   than the last checkpoint.
7. **Box-aware mosaic / copy-paste augmentation:** combine 4 images into one, or paste objects from one image into
   another. They are very effective for detection, and fit in `augmentations.py`.
8. **Harder synthetic data:** textured or noisy backgrounds, and partially cropped shapes at the image borders. The
   current task is almost solved (validation loss 0.07), so it no longer tells models apart.

**Suggested order:** the evaluation (A1–A6) first, because without mAP none of the other changes can be compared.
After that, B1 plus B4–B5 for training speed, and then the FPN (C1).
