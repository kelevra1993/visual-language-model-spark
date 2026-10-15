# Self-Supervised Learning Task

State of the `self_supervised` task (run name code `SSL`) on the `self-supervised-learning` branch, and the work left to do once the code can be run with PyTorch and a GPU. The implementation plan was kept at the bottom of `README.md` during the implementation and removed once it was done.

## Table Of Content
- [Goal](#goal)
- [Implementation State](#implementation-state)
- [Tests Left To Write](#tests-left-to-write)
- [Checks Left To Run](#checks-left-to-run)
- [Open Points](#open-points)

## Goal
The backbone learns from images without labels: two random views of every image (crop, flips, rotation, colour jitter, grayscale, blur) go through the same backbone, global pooling and projection head, and the loss pulls the two views of an image together:
- `simclr`: InfoNCE over the 2N views of the batch, every view having to find its partner among the 2N - 1 others (cosine similarities divided by a temperature, cross-entropy whose target is the partner).
- `barlow_twins`: the cross-correlation matrix of the embedding features of both views, every feature normalized over the batch, pushed to the identity (diagonal to 1 for the invariance, off-diagonal to 0 for the redundancy reduction).

The trained backbone then initialises the backbone of the other tasks through `Backbone.pretrained`, the projection head being left out.

## Implementation State
Every item of the implementation plan, ticked once its code was committed:
- [x] 1. `vision_training/data/self_supervised/self_supervised_dataset.py`
- [x] 2. `vision_training/data/view_augmentations.py`
- [x] 3. `vision_training/data/loaders.py`: `get_self_supervised_dataloaders`
- [x] 4. `vision_training/core/contracts.py`: `SelfSupervisedBatch`
- [x] 5. `vision_training/losses/self_supervised_losses.py`
- [x] 6. `vision_training/models/self_supervised/projection_head.py`
- [x] 7. `vision_training/models/self_supervised/self_supervised_model.py`
- [x] 8. `vision_training/models/self_supervised/inference_encoder.py`
- [x] 9. `vision_training/evaluation/self_supervised/representation_metrics.py`
- [x] 10. `vision_training/trainers/self_supervised_trainer.py`
- [x] 11. `vision_training/configuration_processing/self_supervised_checks.py`
- [x] 12. Registration in the task, model, check, run name and export tables
- [x] 13. `vision_training/configurations/self_supervised/` configuration files
- [x] 14. `README.md` documentation of the task

## Tests Left To Write
1. `tests/data/self_supervised/test_self_supervised_dataset.py`: every sample is kept, the unlabelled and unmapped ones with `IGNORED_LABEL_INDEX`; class names, `get_sample_class_indices` (empty list for the unlabelled samples), `describe_sample` and `get_sample_name`.
2. `tests/data/test_view_augmentations.py`: the views keep the shape, dtype, BGR channel order and [0, 255] range of the images; two views of the same image differ; with every transformation off (crop scale [1, 1], probabilities 0) both views equal the input; every transformation runs on the CPU and on CUDA.
3. `tests/losses/test_self_supervised_losses.py`: InfoNCE equals the cross-entropy written by hand on a 3-image batch (targets i + N and i - N, diagonal masked); identical views give a positive pair accuracy of 1; random embeddings give a loss near log(2N - 1); Barlow Twins is near 0 for identical, decorrelated views and large for constant embeddings; the losses stay in float32 under bfloat16 autocast; the embedding standard deviation is about 1 for spread embeddings and 0 for collapsed ones.
4. `tests/models/self_supervised/test_self_supervised_model.py`: representation and embedding shapes for the `average`, `max` and `flatten` poolings; the loss and batch metric keys of each method with second views, none without; the projection head without hidden layer; `InferenceEncoder` gives L2-normalized representations.
5. `tests/evaluation/self_supervised/test_representation_metrics.py`: k-nearest neighbour accuracy of 1 on separated clusters, the image itself left out of its neighbours, the unlabelled images ignored, the per-class recall; effective rank equal to the dimension for orthogonal representations and to 1 for rank-1 ones; representation standard deviation near 0 for collapsed representations.
6. `tests/configuration_processing/test_configuration_checker.py`: both `self_supervised` configurations pass; refused: an unknown method, a temperature or off-diagonal weight not above 0, `Data.data_augmentation.activated` True, an empty `ProjectionHead.fully_connected`, a normalization without hidden layer, invalid view augmentation values, `number_neighbours` below 1.
7. `tests/configuration_processing/test_run_name.py`: `SSL_..._PAVG.2048.128-PBN_SIMCLR-T0.1`, `PFLAT`, `BT-OD0.005`, the view augmentation giving no token.
8. `tests/trainers/test_self_supervised_trainer.py`: short trainings of both methods on the synthetic classification dataset: checkpoints saved, test evaluation in `Results.txt` (k-NN accuracy, effective rank, representation standard deviation, per-class table), `class_metrics.csv`, sample images, tracked metrics of each method, resume with the same weights, averaged weights evaluated, no failed iteration; the same on a split without labels (headline metric `Effective Rank`).
9. `tests/trainers/test_backbone_transfer_training.py`: the backbone of a `self_supervised` run initialises a `classification` run.
10. `tests/core/test_task_registry.py` (existing): passes with `self_supervised` in every table.
11. `tests/deployment/test_export_specifications.py`: the TorchScript and ONNX exports of the encoder match the eager `InferenceEncoder`.
12. On the GPU: speed of the kornia views per batch, the embedding standard deviation staying away from 0 (no collapse), the k-NN accuracy rising on the synthetic shapes, then a classification run fine-tuned from the self-supervised backbone with 1 %, 10 % and 100 % of the labels against a run from scratch.

## Checks Left To Run
Done in the development environment (PyTorch on the CPU, no GPU), on 2026-10-15:
- Every new module imports, and the synthetic shapes configuration, pointed to a generated dataset, passes the checks.
- Short CPU trainings of both methods (64 pixel images, batches of 4, 5 iterations, moving average of the weights on) on a generated synthetic classification dataset: checkpoints, validation passes, evaluation of the trained and averaged weights (k-NN metrics, effective rank, representation standard deviation, per-class table, class_metrics.csv, sample images of the image and its two views), no failed iteration. The validation InfoNCE loss started at log(2 x 4 - 1) = 1.95, as expected for random embeddings.
- `tests/core`, `tests/configuration_processing` and `tests/deployment/test_export_specifications.py` pass, the expected list of tasks of the configuration checker test now holding `self_supervised`.
- The whole test suite: 931 passed, 1 skipped, 1 failed. The failure,
  `tests/models/instance_segmentation/test_instance_segmenter.py::test_inference_instance_segmenter_matches_the_eager_model[None]`,
  fails the same way on `main`: the boxes of the inference instance segmenter differ from the eager ones by a float32
  rounding beyond the default tolerance of `torch.allclose` on this CPU. It does not come from this task.

Left to do:
1. Run the whole test suite (`uv run pytest tests`) once more after the tests of the task are written.
2. Train both methods on the GPU on the synthetic shapes (`self_supervised/configuration_synthetic_shapes.yaml`) until the k-NN accuracy stops rising, watching `Embedding Std` stay away from 0 and the positive pair accuracy not reaching 1 long before the k-NN accuracy rises (a shortcut shared by the views).
3. Export the encoder of a trained run (`vision_training/deployment/export_model.py`) and check the `image_encoder` model in Triton.
4. Fine-tune a classification run from the self-supervised backbone, as described in item 12 of the tests above.

## Open Points
- The views are drawn on the device with kornia, batch by batch: their speed on large batches is to be measured on the GPU.
- `Data.data_augmentation` must stay off for this task, the views replacing it; the checks refuse it on.
- The k-nearest neighbour evaluation compares every labelled test image with the other test images (leave-one-out), not with the training images, so that the evaluation only reads the test split like the other tasks.
