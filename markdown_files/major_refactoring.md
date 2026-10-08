# Major Refactoring: Remaining Work

The items of the general vision training framework that are not implemented yet, to review one by one. Everything
else of the original plan (package, `BaseTrainer`, the six tasks, transfers, deployment, optimization, MLflow) is
built. The full plan, with its design and implementation notes, is in the git history:
`git show bdaa51d:markdown_files/major_refactoring.md`.

Every item gives what it is, a concrete example, when it is worth doing, and a rough effort: **S** (a few hours, one
or two files), **M** (about a day, several files and tests), **L** (several days: a new configuration section or a
large feature).

## Table Of Contents
1. Sequence Labelling
2. Semantic Segmentation
3. Training Infrastructure
4. Documentation
5. Additions From The Previous Classifiers

---

## 1. Sequence Labelling

### 1. Segment metrics (S)
**What:** the sequence labelling task is scored frame by frame today (frame accuracy, per-class precision / recall /
F1, windows with every frame right). Segment metrics score it event by event: a segment is a run of consecutive frames
of the same class.

**Example:** 10 frames.
```
True:      A A A A A B B B B B   → 2 segments: A(0–4), B(5–9)
Predicted: A A B A A B B A B B   → 6 segments: A, B, A, B, A, B
```
- Frame accuracy: 80 %, only 2 frames are wrong.
- Edit score: the order of the segments, `A B` vs `A B A B A B`, needs 4 edits: (1 − 4/6) × 100 = 33 %.
- Segment F1 at 10 % overlap: 2 of the 6 predicted segments match a true one, F1 = 0.50. At 50 % overlap: none
  does, F1 = 0.

**Worth it if:** you care about when events start and stop (an action in a video). If every frame matters on its own,
the frame metrics are enough.

---

## 2. Semantic Segmentation

### 2. Boundary metrics and boundary losses (M)
**What:** a boundary metric only scores the pixels near the edges of the objects (e.g. boundary IoU, boundary F1
within 2 pixels); a boundary loss weights the errors by their distance to the true edge.

**Example:** a 100 × 100 pixel object predicted 3 pixels too wide on every side. The mean IoU stays high
(100² / 106² = 0.89), while every predicted edge pixel is wrong, so the boundary F1 is close to 0.

**Worth it if:** precise contours matter (measuring sizes, cutting out objects). Not needed if the rough location of
the regions is enough.

### 3. Deep supervision (M)
**What:** extra 1 × 1 segmentation heads on the intermediate levels of the decoder, each upsampled to the input size
and given its own loss (with smaller weights), so that the gradient reaches the early layers more directly. The extra
heads are dropped at inference, so the exports do not change.

**Example:** a decoder of 5 levels gets heads at 1/16, 1/8, 1/4 and 1/2 of the resolution, with loss weights 0.1, 0.2,
0.3, 0.4 next to the final loss of weight 1.

**Worth it if:** deep decoders train slowly or the early layers learn little. On the synthetic shapes the current
decoder already learns, so the gain is uncertain.

### 4. Lovász loss (S)
**What:** a loss that optimises the IoU of every class directly (Lovász-Softmax, Berman et al., CVPR 2018), next to or
instead of the cross entropy and the Dice loss.

**Example:** a class covering 1 % of the pixels. The cross entropy is dominated by the large classes, while the mean
IoU counts that class as much as the background. The Dice loss already partly fixes this; the Lovász loss is often used
for the last part of a training to push the IoU further.

**Worth it if:** the mean IoU is the number you are judged on and the Dice loss does not get it high enough. It
overlaps with the Dice loss already in place.

### 5. Sliding-window inference (M)
**What:** an image larger than the network input is cut into overlapping tiles processed at full resolution, whose
probabilities are averaged on the overlaps, instead of resizing the whole image to the network size.

**Example:** a 4000 × 3000 photo with a 512 pixel network. Resized, a 20 pixel wide object becomes 2.5 pixels wide and
disappears. With 512 pixel tiles overlapping by 128 pixels, it stays 20 pixels wide, for about 11 × 8 = 88 forward
passes per photo instead of 1.

**Worth it if:** your images are much larger than the network input and contain small objects. The training then also
needs crops at the same scale.

### 6. Evaluation at the original resolution (S)
**What:** the metrics are computed at the network resolution today, on the resized masks. The predicted probabilities
would be resized back to every original image (`restore_mask_to_original_image` already does it for deployment) and
compared with the original masks.

**Example:** a 1920 × 1080 image evaluated at 512 × 288. A 2 pixel wide line of the original mask is less than one
pixel wide after resizing, so it is partly lost from the ground truth itself, and the reported IoU of its class is not
the one of the deployed output.

**Worth it if:** you report the numbers of the deployed output, or have thin objects. For large regions the difference
is small.

### 7. Converter of per-class binary mask datasets (S)
**What:** a script turning datasets that store one black and white mask per class into the single-channel masks read
by the segmentation dataset (category of every pixel, 0 background, 255 unlabelled).

**Example:** `image_001.png` with `image_001_car.png` and `image_001_road.png` (0 or 255 each) becomes one
`image_001.png` mask holding 1 for the car pixels and 2 for the road pixels, with a rule for pixels claimed by two
classes (e.g. marked 255 and ignored).

**Worth it if:** your data comes in that format. Otherwise not needed.

### 8. GIF of the sample images (S)
**What:** a script stacking the sample images saved at every evaluation (`<run>/Evaluation/Iteration_<n>/Samples/`)
into one animation per sample. The sample images exist for every task, so it would work for all of them.

**Example:** a sample of a segmentation run evaluated at iterations 500, 1000, …, 5000 becomes a 10-frame GIF where the
predicted mask sharpens over the training.

**Worth it if:** you want to show a training to others. TensorBoard already has a slider over the same images.

### 9. Export of the class map alone (S)
**What:** the exported segmenter returns the class of every pixel and the probabilities of every class for every
pixel. An option would export the class map only.

**Example:** a 512 × 512 image with 150 classes. The probabilities weigh 150 × 512 × 512 × 4 bytes = 157 MB per image,
the class map 2 MB (0.26 MB stored as uint8). Sending the probabilities to and from Triton would limit the
throughput.

**Worth it if:** you have many classes and only need the map in deployment. With 10 classes the probabilities weigh
10 MB, usually fine.

### 10. Migration of `SegmentationDecoder.pretrained` (S)
**What:** the step of `migrate_configuration.py` adding the `pretrained` block to segmentation configurations written
before it existed.

**Example:** none: every segmentation run was trained with the block already in place.

**Worth it if:** only if an older segmentation configuration without the block shows up. Can be dropped.

---

## 3. Training Infrastructure

### 11. DALI loaders for the classification and sequence tasks (L)
**What:** decoding and resizing the images on the GPU (NVIDIA DALI) instead of the CPU worker processes (OpenCV), as
the detection task can already do.

**Example:** the benchmark of the Native and DALI loaders gave no gain: the training was limited by the GPU, so faster
loading did not make it faster. The loading only limits the training when the GPU waits, e.g. a small model on large
JPEG images, with the CPU workers at full load and the GPU well below 100 % of use.

**Worth it if:** a run shows the GPU waiting for the data while every worker is busy. Not the case so far.

### 12. `torch.compile` (M)
**What:** compiling the model into fused GPU kernels, often 10 to 40 % faster per iteration, measured case by case.

**Example:** two costs. The compiled model prefixes every weight name with `_orig_mod.`, so the checkpoints would have
to be saved from the uncompiled model to stay loadable. And every new input shape triggers a new compilation: with
`batch_padding: "longest"`, every batch of a sequence task can have another number of frames, so the compilation must
be told the frame axis is dynamic.

**Worth it if:** long trainings with fixed input shapes (classification, segmentation, detection) where 10 to 40 % of
the training time matters.

### 13. Gradient checkpointing (S)
**What:** instead of keeping every intermediate activation of the backbone for the backward pass, keep only those at
the boundaries of its blocks and recompute the rest during the backward pass. Less memory, about 30 % more compute.

**Example:** a many to one batch of 16 sequences of 50 frames runs 800 images through the backbone at once, and the
memory of their activations limits the batch size. With checkpointing, the same GPU fits a batch several times larger,
or longer sequences.

**Worth it if:** the sequence trainings run out of GPU memory, which becomes likely with `sequence_length` set to the
longest sequence of the dataset.

---

## 4. Documentation

### 14. README sections per task (M)
**What:** one README section per task, telling how to prepare its data (layout of `labels.json`, the label mapping
file), which configuration to copy, how to train, evaluate and export it.

**Example:** "Classification: put the images in `<split>/data/`, write `<split>/labels.json` with `categories` and
`samples`, create the label mapping, copy `configurations/classification/configuration_example.yaml`, set the folders,
point `main.py` to it and run `python -m vision_training.main`". The README has the configuration keys, the run names
and MLflow today, but no such walkthrough (its `## TODO` section is empty).

**Worth it if:** someone else will use the repository, or you come back to a task after months.

---

## 5. Additions From The Previous Classifiers

### 15. Sequence-level evaluation (M)
**What:** a sequence longer than `sequence_length` is cut into several windows, each evaluated as a separate sample.
The windows of each sequence would be merged into one prediction before computing the metrics.

**Example:** with `sequence_length: 50` and `window_stride: 25`, a 60-frame sequence becomes the windows 0–49 and 10–59,
so it counts twice in the accuracy, and its two windows can predict different classes. Averaging their probabilities
gives one prediction for the sequence. For many to many, a frame seen in two windows would also be counted once.

**Worth it if:** your sequences are longer than one window. With `sequence_length` at least the length of the longest
sequence, every sequence is one window and this item changes nothing. Can be dropped in that case.

### 16. Frame selection (L)
**What:** an option of the sequence labelling task picking, in every sequence, the best frame of each wanted class and
rejecting the other frames. Every wanted class gets exactly one frame.

**Example:** 8 photos of a product, wanted classes "left view", "middle view", "right view", the other 5 photos
rejected (blurred, duplicates). With the probabilities

| Frame | left | middle |
|---|---|---|
| 1 | 0.90 | 0.85 |
| 2 | 0.88 | 0.10 |

a greedy choice gives "left" to frame 1 (0.90), then "middle" to frame 2 (0.10), a bad pick. The optimal assignment
gives "middle" to frame 1 (0.85) and "left" to frame 2 (0.88).

**Details:**
- **Configuration:** an optional `Selection` section with `rejection_class` and `selection_loss_weight`.
- **Data check:** every wanted class exactly once per sequence; the offending sequences are listed at startup.
- **Windows:** the whole sequence in one window (`sequence_length` at least the length of the longest sequence).
- **Decoding:** the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`, a new `scipy` dependency), with the
  cost `-(log p(frame, class) - log p(frame, rejection))`; the frames left over are rejected.
- **Loss:** the per-frame cross entropy, plus, for every wanted class, a softmax of its logit over the frames trained
  on the index of its true frame ("which frame is the best for this class").
- **Metrics:** accuracy per wanted class after decoding, share of sequences with every selection right, recall of the
  rejection class, a CSV per sequence and a JSON of chosen vs true frames.
- **Augmentation:** left ↔ right view swaps on flips (the labels swapped with the flip), frame shuffling (already
  available with `shuffle_frames`), blur of the rejected frames only.
- **Inference:** an optional sharpness filter (Laplacian variance), applied without the labels.
- **Synthetic data:** rejected squares plus left, middle and right circles.

**Worth it if:** your task is choosing frames out of a sequence (best view per angle, best shot per step) rather than
classifying every frame.
