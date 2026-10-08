# Major Refactoring: Remaining Work

The items of the general vision training framework that are not implemented yet, to review one by one. Everything
else of the original plan (package, `BaseTrainer`, the six tasks, transfers, deployment, optimization, MLflow) is
built. The full plan, with its design and implementation notes, is in the git history:
`git show bdaa51d:markdown_files/major_refactoring.md`.

Every item gives what it is, a concrete example, when it is worth doing, and a rough effort: **S** (a few hours, one
or two files), **M** (about a day, several files and tests), **L** (several days: a new configuration section or a
large feature).

## Table Of Contents
1. Sequence Tasks
2. Semantic Segmentation
3. Training Infrastructure
4. Additions From The Previous Classifiers

---

## 1. Sequence Tasks

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

### 2. Transformer without positional embedding (S)
**What:** a `TemporalModel.positional_embedding` option of the transformer. The transformer adds a learned position
embedding to every frame, so it uses the order of the frames. Without it, it treats a window as a set: shuffling the
frames does not change the output of M2O and CMP (the mean of the frame outputs), and shuffles the outputs of M2M with
the frames.

**Example:** 10 photos of a product taken in any order, the case of `shuffle_frames`. With the embedding, the model has
to learn from shuffled training windows that the order means nothing; without it, the order cannot matter. The
transformer weights also stop depending on `sequence_length`, so a trained transformer transfers to a run with another
`sequence_length`, which the transfer refuses today.

**Details:** a run name token (e.g. `NPE`), the weights being different; the `sequence_length` transfer check skipped
without the embedding; a migration step adding `positional_embedding: True` to the existing sequence configurations.

**Worth it if:** your sequences are sets of photos in any order, for M2O, M2M, CMP and the frame selection task
(item 13).

---

## 2. Semantic Segmentation

### 3. Boundary metrics and boundary losses (M)
**What:** a boundary metric only scores the pixels near the edges of the objects (e.g. boundary IoU, boundary F1
within 2 pixels); a boundary loss weights the errors by their distance to the true edge.

**Example:** a 100 × 100 pixel object predicted 3 pixels too wide on every side. The mean IoU stays high
(100² / 106² = 0.89), while every predicted edge pixel is wrong, so the boundary F1 is close to 0.

**Worth it if:** precise contours matter (measuring sizes, cutting out objects). Not needed if the rough location of
the regions is enough.

### 4. Deep supervision (M)
**What:** extra 1 × 1 segmentation heads on the intermediate levels of the decoder, each upsampled to the input size
and given its own loss (with smaller weights), so that the gradient reaches the early layers more directly. The extra
heads are dropped at inference, so the exports do not change.

**Example:** a decoder of 5 levels gets heads at 1/16, 1/8, 1/4 and 1/2 of the resolution, with loss weights 0.1, 0.2,
0.3, 0.4 next to the final loss of weight 1.

**Worth it if:** deep decoders train slowly or the early layers learn little. On the synthetic shapes the current
decoder already learns, so the gain is uncertain.

### 5. Lovász loss (S)
**What:** a loss that optimises the IoU of every class directly (Lovász-Softmax, Berman et al., CVPR 2018), next to or
instead of the cross entropy and the Dice loss.

**Example:** a class covering 1 % of the pixels. The cross entropy is dominated by the large classes, while the mean
IoU counts that class as much as the background. The Dice loss already partly fixes this; the Lovász loss is often used
for the last part of a training to push the IoU further.

**Worth it if:** the mean IoU is the number you are judged on and the Dice loss does not get it high enough. It
overlaps with the Dice loss already in place.

### 6. Sliding-window inference (M)
**What:** an image larger than the network input is cut into overlapping tiles processed at full resolution, whose
probabilities are averaged on the overlaps, instead of resizing the whole image to the network size.

**Example:** a 4000 × 3000 photo with a 512 pixel network. Resized, a 20 pixel wide object becomes 2.5 pixels wide and
disappears. With 512 pixel tiles overlapping by 128 pixels, it stays 20 pixels wide, for about 11 × 8 = 88 forward
passes per photo instead of 1.

**Worth it if:** your images are much larger than the network input and contain small objects. The training then also
needs crops at the same scale.

### 7. Evaluation at the original resolution (S)
**What:** the metrics are computed at the network resolution today, on the resized masks. The predicted probabilities
would be resized back to every original image (`restore_mask_to_original_image` already does it for deployment) and
compared with the original masks.

**Example:** a 1920 × 1080 image evaluated at 512 × 288. A 2 pixel wide line of the original mask is less than one
pixel wide after resizing, so it is partly lost from the ground truth itself, and the reported IoU of its class is not
the one of the deployed output.

**Worth it if:** you report the numbers of the deployed output, or have thin objects. For large regions the difference
is small.

### 8. Converter of per-class binary mask datasets (S)
**What:** a script turning datasets that store one black and white mask per class into the single-channel masks read
by the segmentation dataset (category of every pixel, 0 background, 255 unlabelled).

**Example:** `image_001.png` with `image_001_car.png` and `image_001_road.png` (0 or 255 each) becomes one
`image_001.png` mask holding 1 for the car pixels and 2 for the road pixels, with a rule for pixels claimed by two
classes (e.g. marked 255 and ignored).

**Worth it if:** your data comes in that format. Otherwise not needed.

### 9. GIF of the sample images (S)
**What:** a script stacking the sample images saved at every evaluation (`<run>/Evaluation/Iteration_<n>/Samples/`)
into one animation per sample. The sample images exist for every task, so it would work for all of them.

**Example:** a sample of a segmentation run evaluated at iterations 500, 1000, …, 5000 becomes a 10-frame GIF where the
predicted mask sharpens over the training.

**Worth it if:** you want to show a training to others. TensorBoard already has a slider over the same images.

### 10. Export of the class map alone (S)
**What:** the exported segmenter returns the class of every pixel and the probabilities of every class for every
pixel. An option would export the class map only.

**Example:** a 512 × 512 image with 150 classes. The probabilities weigh 150 × 512 × 512 × 4 bytes = 157 MB per image,
the class map 2 MB (0.26 MB stored as uint8). Sending the probabilities to and from Triton would limit the
throughput.

**Worth it if:** you have many classes and only need the map in deployment. With 10 classes the probabilities weigh
10 MB, usually fine.

---

## 3. Training Infrastructure

### 11. `torch.compile` (M)
**What:** compiling the model into fused GPU kernels, often 10 to 40 % faster per iteration, measured case by case.

**Example:** two costs. The compiled model prefixes every weight name with `_orig_mod.`, so the checkpoints would have
to be saved from the uncompiled model to stay loadable. And every new input shape triggers a new compilation: with
`batch_padding: "longest"`, every batch of a sequence task can have another number of frames, so the compilation must
be told the frame axis is dynamic.

**Worth it if:** long trainings with fixed input shapes (classification, segmentation, detection) where 10 to 40 % of
the training time matters.

### 12. Gradient checkpointing (S)
**What:** instead of keeping every intermediate activation of the backbone for the backward pass, keep only those at
the boundaries of its blocks and recompute the rest during the backward pass. Less memory, about 30 % more compute.

**Example:** a many to one batch of 16 sequences of 50 frames runs 800 images through the backbone at once, and the
memory of their activations limits the batch size. With checkpointing, the same GPU fits a batch several times larger,
or longer sequences.

**Worth it if:** the sequence trainings run out of GPU memory, which becomes likely with `sequence_length` set to the
longest sequence of the dataset.

---

## 4. Additions From The Previous Classifiers

### 13. Frame selection, a 7th task (L)
**What:** a new task, `frame_selection` (run name code `SEL`), picking in every sequence the best frame of each wanted
class, every other frame being unselected. Every wanted class gets exactly `frames_per_class` frames (1 by default).
The wanted classes are whatever the dataset defines: one photo per viewing angle of a product, the best shot of every
step of a procedure, the sharpest frame showing each defect... The task reuses the model of the sequence labelling task
(one class per frame) and adds the decoding of the selection, its loss, its data check, its metrics and its images.

**Example:** 8 photos of a product and three wanted classes A, B and C: 3 photos are selected, the 5 others (blurred
photos, duplicates) are unselected. A duplicate is unselected because another photo of the same class is better, not
because it is bad.

**Decoding**, from the per-frame probabilities of the model:
- `"hungarian"`: the optimal assignment (`scipy.optimize.linear_sum_assignment`; scipy 1.18.1 is already installed as
  a dependency of another package, it would be declared in `pyproject.toml`). The gain of selecting frame f for class
  c is `log p(f, c) - log p(f, unselected)`, and the assignment with the largest total gain is the most probable
  labelling of the whole sequence with every wanted class chosen `frames_per_class` times (every class column being
  repeated `frames_per_class` times).
- `"greedy"`: the rule of the previous selector. Every frame takes its most probable class; a wanted class taken by too
  many frames keeps the most probable ones, the others becoming unselected; a wanted class taken by too few frames
  completes them with the unselected frames most probable for it, class by class.

With three frames and two wanted classes:

| Frame | A | B | unselected |
|---|---|---|---|
| 1 | 0.50 | 0.45 | 0.05 |
| 2 | 0.45 | 0.05 | 0.50 |
| 3 | 0.02 | 0.08 | 0.90 |

- Greedy: frame 1 takes A (0.50). No frame takes B, so it goes to the unselected frame most probable for it, frame 3
  (0.08), a bad pick.
- Hungarian: the gains are 2.30 (A) and 2.20 (B) for frame 1, -0.11 and -2.30 for frame 2, -3.81 and -2.42 for frame 3.
  B for frame 1 and A for frame 2 total 2.09, against -0.12 for the greedy choice.

Every evaluation scores both decodings, so that they are compared on the same checkpoint without training again.
`Selection.decoding` sets the one used by the sample images, the error images and the deployment.

**Details:**
- **Task:** its own trainer (built on the sequence labelling trainer), model registry entry, configuration checks, run
  name code `SEL`, configurations in `vision_training/configurations/frame_selection/` (example and synthetic) and
  export name `frame_selector`.
- **Configuration:** a `Selection` section with `unselected_category` (the name of the manifest category of the
  unselected frames), `decoding` (`"hungarian"` or `"greedy"`), `frames_per_class` and `selection_loss_weight`.
- **Data:** the manifest of the sequence labelling task (`frame_category_ids`). Every sequence must hold every wanted
  class exactly `frames_per_class` times: the sequences that do not are listed at startup and the training stops. A
  frame of an unmapped category counts as unselected, so that removing a wanted class from the label mapping turns its
  frames into unselected ones.
- **Windows:** the whole sequence in one window. `sequence_length` must be at least the length of the longest
  sequence, checked at startup, and there is no `window_stride` nor `short_sequences`.
- **Loss:** the per-frame cross entropy, plus, for every wanted class, a softmax of its logit over the frames of the
  sequence trained towards its true frames (1 / `frames_per_class` on each), "which frames are the best for this
  class", weighted by `selection_loss_weight` (0 keeps the per-frame cross entropy alone, as in the previous selector).
- **Metrics,** for both decodings: the selection accuracy of every wanted class (share of its true frames chosen), the
  share of sequences with every selection right, and the precision and recall of the unselected class. Before
  decoding: the top-k selection accuracy (share of the true frames among the k frames most probable for their class, k
  being `evaluation.top_k`), the mean rank of the true frames among the frames ordered by their probability for the
  class (1 when the true frame is the most probable), and the per-frame metrics of the sequence labelling task. A CSV
  gives the chosen and true frames of every sequence with their probabilities and ranks.
- **Confidence thresholds:** for every wanted class, the probability of the chosen frames reaching
  `evaluation.target_precision` (share of the chosen frames above it that are true frames), with the share of the true
  frames it keeps, saved in `class_thresholds.json`. A chosen frame below the threshold of its class is flagged as low
  confidence (the sequence may hold no good frame of that class), for a human to check.
- **Sample and error images:** the whole sequence as a grid of thumbnails (160 pixels, 8 per row: a 40-frame sequence
  gives about 1316 x 824 pixels), every frame numbered:
  - a solid green border on a frame chosen for its true class, a solid red border on a frame chosen for a class it does
    not hold, and a dashed yellow border on a true frame left unchosen, each labelled with the class and the
    probability of the frame for it (the colours of the segmentation error maps). A frame can carry a red and a yellow
    border, when it holds a class but was chosen for another;
  - the frames rightly unselected dimmed, so that the eye goes to the selections;
  - under the grid, a heat map of the probability of every frame (columns) for every wanted class (rows), the chosen
    frames of every class marked with a dot and its true frames outlined, which shows the frames competing for a class;
  - a caption bar above (`<right>/<wanted> selections right`, green or red) and a legend bar below.

  The sample images show the fixed test sequences at every evaluation. The error images show every test sequence with
  a wrong selection, in `Errors/<CLASS>/<sequence>.png` for every class picked wrong.
- **Augmentation:** the generic augmentations and `shuffle_frames` only. No swap of the classes on flips and no blur
  of the unselected frames, both being specific to one dataset.
- **Deployment:** the exported network returns the per-frame probabilities [B, T, K] like the sequence labeller. A
  Triton ensemble chains it with a Python backend model running the decoding of the run (the same function as in the
  evaluation and in `run_inference.py`), so that the clients receive the chosen frames of every wanted class, their
  probabilities, their low-confidence flags and the k best candidates of every class. The Python backend needs scipy
  in the Triton container, installed by a short Dockerfile on top of the Triton image of `docker-compose.yml`.
- **Synthetic data:** sequences of 4 to 12 frames and three wanted shapes (e.g. circle, triangle and star). Every
  wanted shape appears in one or more frames at random sizes, its largest copy being the frame to select; the smaller
  copies and the frames of other shapes are unselected, so that the choice depends on the other frames of the sequence.

**Worth it if:** your task is choosing frames out of a sequence (best view per angle, best shot per step) rather than
classifying every frame.
