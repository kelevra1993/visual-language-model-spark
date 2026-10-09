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
