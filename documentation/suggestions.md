# Suggestions

> **Nothing is added to this file any more.** A new idea is tested right away instead of being written here. The file
> only gets smaller: each item is removed once implemented or dropped, and the file is deleted once empty.

The items were checked against the code on 2026-10-15, 2026-11-13, 2026-11-20 and 2026-11-27. Audit L19 (item 8) is a
finding of the repository audit of 2026-10-10. "Mask R-CNN" (item 8) means its Keras / TensorFlow 1
implementation (matterport, commit `3deaec5`), whose files `model.py`, `utils.py` and `config.py` are those of `mrcnn/`.

## Table Of Content
- [B. Instance segmentation](#b-instance-segmentation)
- [C. Evaluation and deployment](#c-evaluation-and-deployment)
- [G. Keypoint detection](#g-keypoint-detection)
- [H. Classification](#h-classification)
- [J. Self-supervised counted shapes: tests left to run](#j-self-supervised-counted-shapes-tests-left-to-run)

## B. Instance segmentation
8. **Ground truth masks cropped to their boxes ("mini masks")**:
   - **Mask R-CNN:** `USE_MINI_MASK` and `MINI_MASK_SHAPE` (56 × 56) (`config.py:103-104`): each mask is cropped to its
     box and resized (`minimize_mask`, `utils.py:517-534`), and the mask target of a RoI is cropped from that small
     mask, after expressing the RoI in the coordinates of the ground truth box (`model.py:584-600`).
   - **Here:** one `[N, S, S]` uint8 mask per object goes through the loader workers to the GPU. Audit L19 estimates
     about 4.7 GiB in flight at 1024 pixels, batch 40, 8 workers and prefetch 2. Since 2026-11-13,
     `Data.instance_index_maps` sends one `[S, S]` map per image instead when the masks never overlap (the synthetic
     datasets), expanded on the GPU; the mini masks remain the option for overlapping masks (COCO).
   - **Change:** an option to load the masks as `[N, m, m]` crops with their boxes (`m` configurable, 56 in Mask
     R-CNN, 112 for large objects). `build_mask_targets` would crop with RoIAlign in the coordinates of the crop (box
     offset and scale). The flips, the crop and the zoom-out act exactly on a box-relative crop; the ±5° rotation
     needs the crop resampled in the rotated box. The evaluation and the sample images paste the crops back with
     `paste_instance_masks`. The other fix that L19 suggests (sending polygons or RLE and rasterising on the device)
     keeps the full precision.
   - **Cost and risk:** medium to high (dataset, augmentations, targets, evaluation), with a small loss of precision
     for large objects. Only worth doing if memory or loader throughput becomes the limit.

## C. Evaluation and deployment
41. **What the TensorRT engine check leaves open**: the engine check of `export_model.py` accepts an FP32 or FP16
    engine within the measured roundings of its reduced precision (`REDUCED_PRECISION_DIFFERENCE_FACTOR` of
    `tensorrt_check.py`). Left:
    - **Only the batches of the check are built**: the check builds the engine of its batches of 3, and ONNX Runtime
      builds the engine of every larger batch at its first request in Triton, unchecked. Explicit
      `trt_profile_min_shapes`, `trt_profile_opt_shapes` and `trt_profile_max_shapes` in `config.pbtxt` would cover
      the largest batch by construction, at the cost of building the largest engine at every export (M2M 7.2 GiB at
      16 sequences of 50 frames, the sequence models at 512 pixels about 29 GiB estimated). Whether the ONNX Runtime
      backend of Triton 26.01 passes these keys is unchecked.
    - **Report the ONNX Runtime bug upstream**: its TensorRT execution provider (1.20.1 and 1.23.2) left the values of
      a top-k, a graph output beside indices with three consumers, out of its engine, which crashed Triton (signal 11).
      The reproduction is kept outside the repository, in
      `/home/robert_kelevra/.claude/jobs/a75eb28a/tmp/tensorrt_failures/topk_repro`, made by the `make_topk_repro.py`
      next to that folder; if the folder is gone, the reproduction has to be rebuilt with that script.

## G. Keypoint detection
16. **Ignored regions, experiment left to run** (`KeypointHead.ignore_unlabelled_objects`, implemented): as in
    HigherHRNet, the heatmap loss leaves out the cells at least half covered by the objects left out of the labels
    (the COCO crowds and the persons without any labelled keypoint), read from their `segmentation`. On in the COCO
    `configuration_example.yaml`, off in the synthetic one, whose shapes have no `segmentation`.
    - **Left:** the OKS AP of a COCO run with and without the setting, the crowds and the unlabelled persons being
      4652 of the 11004 persons of the validation split.
17. **Finer peaks, measure on COCO left to run** (`compute_peak_offsets`, implemented): every keypoint candidate is
    placed within its cell, along each axis, at the vertex of the parabola through the log-heatmap at its cell and at
    its two neighbours (the Taylor expansion of DARK without its cross derivative), instead of a quarter of a cell
    towards its higher neighbour.
    - **Measured** on 200 test images of the synthetic run (`keypoint_focal_synthetic`, iteration 6000, moving
      average, stride 2, 4290 matched keypoints): the mean error per axis is 0.363 pixel instead of 0.458 on the
      coordinates off the pixel centres (-21 %), and 0.293 instead of 0.127 on the pixel centres, where the quarter
      shift was exact. 59 % of the synthetic coordinates are pixel centres, while the resized COCO coordinates are
      continuous.
    - **Left:** the OKS AP of a COCO run decoded both ways, the published gain of DARK being largest for small
      objects.

## H. Classification
18. **Decorrelation of the hidden features (Barlow Twins redundancy term), experiment left to run**
    (`ClassificationHead.decorrelation`, implemented): a loss term added to the cross entropy so
    that the hidden features of the classification head carry different information. Every feature is standardised
    over the batch, and the squared off-diagonal terms of the [D, D] correlation matrix of the features are added to the
    loss with a weight. This is the redundancy reduction term of Barlow Twins (arXiv 2103.03230) without its invariance
    term, which needs two augmented views of every image: with the single view of the supervised training, it is the
    DeCov penalty (Cogswell et al., ICLR 2016, arXiv 1511.06068).
    - **Literature:**
      - DeCov penalises the off-diagonal covariance of the hidden activations of supervised networks, and reports a
        smaller gap between the training and validation accuracy, often better than dropout.
      - OrthoReg (Rodríguez et al., ICLR 2017, arXiv 1611.01967) decorrelates the convolution filters instead of the
        activations.
      - VCReg (arXiv 2306.13292) adds the variance and covariance terms of VICReg (arXiv 2105.04906) to the supervised
        training, the main gain being the transfer to other tasks rather than the accuracy on the training task.
      - Barlow Twins trained together with the cross entropy from scratch lowered the accuracy of a VQA model
        (arXiv 2203.03727), the two losses having very different ranges. Training with the cross entropy first, then
        fine-tuning with both, worked.
    - **Measured redundancy:** feature correlation of the ImageNet-1k Medium run (`configuration_medium_model.yaml`,
      Iteration 115000, EMA weights) on two batches of 256 test images (mean of both batches), each layer compared with
      independent Gaussian features of the same size, measured on 2026-10-18 with a temporary script:

      | Features | Units | Mean \|r\| | % pairs \|r\| > 0.3 | Largest \|r\| | Effective rank | Directions for 90 % of the variance |
      |---|---|---|---|---|---|---|
      | Backbone channels (spatial mean) | 128 | 0.098 | 2.1 | 0.53 | 43 (Gaussian 85) | 59 (Gaussian 86) |
      | Hidden layer 1, before its ReLU | 512 | 0.113 | 3.1 | 0.60 | 46 (Gaussian 171) | 61 (Gaussian 170) |
      | Hidden layer 1, after its ReLU | 512 | 0.082 | 0.5 | 0.58 | 80 (Gaussian 171) | 124 (Gaussian 170) |
      | Hidden layer 2, before its ReLU | 256 | 0.120 | 4.3 | 0.83 | 38 (Gaussian 128) | 44 (Gaussian 131) |
      | Hidden layer 2, after its ReLU | 256 | 0.100 | 1.6 | 0.66 | 51 (Gaussian 128) | 75 (Gaussian 131) |

      The independent features give a mean |r| of 0.050 (the sampling noise at N = 256) and no pair above 0.3. The
      pairwise correlations are low, but the effective rank is too: 90 % of the variance of the 256 units of the second
      hidden layer lie in 44 directions before its ReLU, against 131 for independent features. The redundancy comes from
      many small correlations, which the off-diagonal sum penalises all together. One unit of the second hidden layer
      never fires on these batches.
    - **Setting:** `ClassificationHead.decorrelation` (`activated`, `weight`, `hidden_layer`), off in every
      configuration file, tracked as `decorrelation_loss` and `hidden_feature_correlation`. It gives no run name
      token, so the warm-up of the cross entropy is a run resumed with it turned on. The second hidden layer before
      its ReLU, where the redundancy is highest, is the first candidate.
    - **Batch size:** the correlation matrix of N images has a rank of at most N − 1, so the term cannot reach 0 with
      more features than images: at the batch size of 256, the 512 units of the first hidden layer keep a floor, while
      the 256 units of the second fit. The sampling noise alone also gives a sum of the squared off-diagonal terms of
      about D(D − 1) / (N − 1), 256 for D = N = 256, below which the term only fits noise.
    - **Experiment:** the Small model with and without the term, for the same number of iterations, compared on the
      test top-1 and on the statistics above, with a weight giving the term about a tenth of the cross entropy at the
      start, turned on after a warm-up of the cross entropy as in the VQA result.

    Cost and risk: small to medium. The Medium run underfits (iterations 110000 to 115000: training accuracy 0.599 with
    the augmentation and the dropout, validation accuracy 0.613), and the papers report regularisation-sized gains, so
    the term may lower the top-1 as well as raise it. Decorrelated features also stay dependent in non-linear ways.

## J. Self-supervised counted shapes: tests left to run
The synthetic counted shapes dataset (`/home/robert_kelevra/Data/self_supervised_data/synthetic_counted_shapes`,
500,000 training, 10,000 validation and 10,000 test images of 5 to 10 shapes at 512 pixels, drawn like the sequence
labelling frames) and its two configurations (`self_supervised/configuration_counted_shapes_simclr.yaml` and
`configuration_counted_shapes_barlow_twins.yaml`, the backbone and flattened feature map of the launches of
`sequence_labelling_experiment_plan.md`) were written, but no run has been trained on them, and the dataset folder
was empty on 2026-11-20: the dataset is to be generated again first (`--task=self_supervised` of
`create_synthetic_dataset.py`). The goal is to see whether the self-supervised backbone learns not only how many shapes
an image holds but also which ones.

24. **The two self-supervised trainings**: SimCLR and Barlow Twins trained until the k-NN accuracy of the shape count
    (6 classes, `5_shapes` to `10_shapes`) stops rising, the input viewer checking the two views first.
    - **Watch:** `Embedding Standard Deviation` staying away from 0 (no collapse), and the positive pair accuracy of
      SimCLR not reaching 1 long before the k-NN accuracy rises, which would show a cue shared by the two views.
25. **Probe of which shapes an image holds**: the k-NN evaluation only reads the shape count, the label of every
    sample. The categories of its shapes are in `shape_category_ids` of `labels.json`, for a probe trained on the
    frozen representation: a multi-label linear classifier of the 35 shape classes, its macro F1 against the same probe
    on a backbone with random weights and on a supervised backbone.
26. **Initialisation of sequence labelling**: the self-supervised backbone given to the sequence labelling runs of the
    shape change experiment (`Backbone.pretrained`, the same architecture as their backbone), against the same runs from
    random weights, compared on the test Macro F1 of the averaged weights at the same number of iterations.
27. **Strength of the views**: `random_crop` is off in both configurations, so that both views keep every shape and
    the same count, with flips and small rotations. Turning it on (20 to 100 % of the area for the self-supervised
    views) makes the task harder but can change the number of shapes between the two views; to compare on the probes
    of items 25 and 26 rather than on the k-NN accuracy of the count. The harder views of 2026-10-20 add, among
    others, a random erasing of 2 to 15 % of the image (probability 0.25) and a plasma shadow darkening parts of it
    (probability 0.2), which can also hide a shape of one view and change its count: to compare with
    `random_erasing` and `random_plasma_shadow` off on the same probes.
