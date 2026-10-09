# Major Refactoring: Remaining Work

The items of the general vision training framework that are not implemented yet, to review one by one. Everything
else of the original plan (package, `BaseTrainer`, the seven tasks, transfers, deployment, optimization, MLflow) is
built. The full plan, with its design and implementation notes, is in the git history:
`git show bdaa51d:markdown_files/major_refactoring.md`.

Every item gives what it is, a concrete example, when it is worth doing, and a rough effort: **S** (a few hours, one
or two files), **M** (about a day, several files and tests), **L** (several days: a new configuration section or a
large feature).

## Table Of Contents
1. Sequence Tasks
2. Semantic Segmentation
3. Training Infrastructure

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

---

## 2. Semantic Segmentation

### 2. Converter of per-class binary mask datasets (S)
**What:** a script turning datasets that store one black and white mask per class into the single-channel masks read
by the segmentation dataset (category of every pixel, 0 background, 255 unlabelled).

**Example:** `image_001.png` with `image_001_car.png` and `image_001_road.png` (0 or 255 each) becomes one
`image_001.png` mask holding 1 for the car pixels and 2 for the road pixels, with a rule for pixels claimed by two
classes (e.g. marked 255 and ignored).

**Worth it if:** your data comes in that format. Otherwise not needed.

### 3. GIF of the sample images (S)
**What:** a script stacking the sample images saved at every evaluation (`<run>/Evaluation/Iteration_<n>/Samples/`)
into one animation per sample. The sample images exist for every task, so it would work for all of them.

**Example:** a sample of a segmentation run evaluated at iterations 500, 1000, …, 5000 becomes a 10-frame GIF where the
predicted mask sharpens over the training.

**Worth it if:** you want to show a training to others. TensorBoard already has a slider over the same images.

### 4. Export of the class map alone (S)
**What:** the exported segmenter returns the class of every pixel and the probabilities of every class for every
pixel. An option would export the class map only.

**Example:** a 512 × 512 image with 150 classes. The probabilities weigh 150 × 512 × 512 × 4 bytes = 157 MB per image,
the class map 2 MB (0.26 MB stored as uint8). Sending the probabilities to and from Triton would limit the
throughput.

**Worth it if:** you have many classes and only need the map in deployment. With 10 classes the probabilities weigh
10 MB, usually fine.

---

## 3. Training Infrastructure

### 5. `torch.compile` (M)
**What:** compiling the model into fused GPU kernels, often 10 to 40 % faster per iteration, measured case by case.

**Example:** two costs. The compiled model prefixes every weight name with `_orig_mod.`, so the checkpoints would have
to be saved from the uncompiled model to stay loadable. And every new input shape triggers a new compilation: with
`batch_padding: "longest"`, every batch of a sequence task can have another number of frames, so the compilation must
be told the frame axis is dynamic.

**Worth it if:** long trainings with fixed input shapes (classification, segmentation, detection) where 10 to 40 % of
the training time matters.

### 6. Gradient checkpointing (S)
**What:** instead of keeping every intermediate activation of the backbone for the backward pass, keep only those at
the boundaries of its blocks and recompute the rest during the backward pass. Less memory, about 30 % more compute.

**Example:** a many to one batch of 16 sequences of 50 frames runs 800 images through the backbone at once, and the
memory of their activations limits the batch size. With checkpointing, the same GPU fits a batch several times larger,
or longer sequences.

**Worth it if:** the sequence trainings run out of GPU memory, which becomes likely with `sequence_length` set to the
longest sequence of the dataset.
