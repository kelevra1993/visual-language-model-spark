# Self-Supervised Learning Task

State of the `self_supervised` task (run name code `SSL`), merged into `main`, and the work left to do on the GPU. The implementation plan was kept at the bottom of `README.md` during the implementation and removed once it was done.

## Table Of Content
- [Goal](#goal)
- [Implementation State](#implementation-state)
- [Tests](#tests)
- [Checks Left To Run](#checks-left-to-run)
- [Open Points](#open-points)

## Goal
The backbone learns from images without labels: two random views of every image (crop, flips, rotation, colour jitter, grayscale, blur) go through the same backbone, global pooling and projection head, and the loss pulls the two views of an image together:
- `simclr`: InfoNCE over the 2N views of the batch, every view having to find its partner among the 2N - 1 others (cosine similarities divided by a temperature, cross-entropy whose target is the partner).
- `barlow_twins`: the cross-correlation matrix of the embedding features of both views, every feature normalized over the batch, pushed to the identity (diagonal to 1 for the invariance, off-diagonal to 0 for the redundancy reduction).

The trained backbone then initialises the backbone of the other tasks through `Backbone.pretrained`, the projection head being left out.

## Implementation State
Every item of the implementation plan, ticked once its code was committed:
- [x] 1. `vision_training/data/classification/classification_dataset.py`: `keep_unlabelled_samples`, keeping the images without `category_id` or of an unmapped category with `IGNORED_LABEL_INDEX` (the separate self-supervised dataset of the first implementation duplicated this dataset and was removed)
- [x] 2. `vision_training/data/view_augmentations.py`
- [x] 3. `vision_training/data/loaders.py`: `get_classification_dataloaders(self_supervised=True)`, the training and validation loaders dropping their last incomplete batch and a split smaller than the batch size stopping the launch
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

## Tests
Written:
1. `tests/data/classification/test_classification_dataset.py`: the unlabelled and unmapped samples, and the multi-label samples of several mapped categories, kept with `IGNORED_LABEL_INDEX` and an empty class list, a multi-label sample of a single mapped category keeping its class; the training and validation loaders of the task dropping their last incomplete batch, the test loader keeping it; a split smaller than the batch size refused.
2. `tests/data/test_view_augmentations.py`: the views keep the shape, dtype and [0, 255] range of the images and differ from each other; with every transformation off both views equal the input; the grayscale follows the BGR channel order; the same seed gives the same views; the views stay in float32 under autocast.
3. `tests/losses/test_self_supervised_losses.py`: InfoNCE equals the cross-entropy written by hand; identical views give a positive pair accuracy of 1; random embeddings give a loss near log(2N - 1); Barlow Twins is near 0 for decorrelated views and 8 for collapsed ones; the losses stay in float32 under bfloat16 autocast; the embedding standard deviation is about 1 for spread embeddings, 0 for collapsed ones and between 0.5 and 0.9 after a ReLU.
4. `tests/models/self_supervised/test_self_supervised_model.py`: representation and embedding sizes for the `average`, `max` and `flatten` poolings; the loss and batch metric keys of each method and the gradients reaching every parameter; the projection head without hidden layer; `InferenceEncoder` giving the L2-normalized representations; an unknown method refused.
5. `tests/evaluation/self_supervised/test_representation_metrics.py`: k-nearest neighbour accuracy of 1 on separated clusters, the image itself left out of its neighbours, the weighted vote and its ties going to the lowest class index, the chunked similarities; effective rank equal to the dimension for orthogonal representations, 1 for rank-1 ones and NaN for non-finite ones.
6. `tests/configuration_processing/test_configuration_checker.py`: both `self_supervised` configurations pass; refused: invalid projection head settings, a normalization without hidden layer, `Data.data_augmentation.activated` True, invalid view augmentation values, invalid evaluation settings.
7. `tests/configuration_processing/test_run_name.py`: `SSL_..._PAVG.512.128-PBN_SIMCLR-T0.2`, `PFLAT` with `PLN` and `BT-OD0.005`, a head without hidden layer.
8. `tests/trainers/test_self_supervised_trainer.py`: short trainings of both methods on the synthetic classification dataset: checkpoints saved, tracked metrics of each method in TensorBoard, test evaluation in `Results.txt` (k-NN metrics, effective rank, representation standard deviation, per-class table), `class_metrics.csv`, sample images of the image and its two views, resume with the same weights, no failed iteration; the sample views drawn again identically without changing the random state of the training; a dataset without labels and an empty label mapping evaluated on the effective rank and the representation standard deviation only (headline metric `Effective Rank`); the classes named after the categories of the three manifests, a class absent from the test manifest keeping its name.
9. `tests/trainers/test_backbone_transfer_training.py`: the backbone of a `self_supervised` run initialises a `classification` run, which trains and saves.
10. `tests/deployment/test_export_specifications.py`: the export of the encoder matches the eager `InferenceEncoder`.

Left to write:
1. The views on CUDA: the tests of the views run on the CPU only.
2. The evaluation of the averaged weights (`Optimization.exponential_moving_average`) in the trainer test of the task.

## Checks Left To Run
Done in the development environment (PyTorch on the CPU, no GPU), on 2026-10-15:
- Every new module imports, and the synthetic shapes configuration, pointed to a generated dataset, passes the checks.
- Short CPU trainings of both methods (64 pixel images, batches of 4, 5 iterations, moving average of the weights on) on a generated synthetic classification dataset: checkpoints, validation passes, evaluation of the trained and averaged weights (k-NN metrics, effective rank, representation standard deviation, per-class table, class_metrics.csv, sample images of the image and its two views), no failed iteration. The validation InfoNCE loss started at log(2 x 4 - 1) = 1.95, as expected for random embeddings.

Left to do:
1. Train both methods on the GPU on the synthetic shapes (`self_supervised/configuration_synthetic_shapes.yaml`) until the k-NN accuracy stops rising, watching `Embedding Standard Deviation` stay away from 0 and the positive pair accuracy not reaching 1 long before the k-NN accuracy rises (a shortcut shared by the views).
2. Export the encoder of a trained run (`vision_training/deployment/export_model.py`) and check the `image_encoder` model in Triton.
3. Fine-tune a classification run from the self-supervised backbone with 1 %, 10 % and 100 % of the labels, against a run from scratch.

## Open Points
- The parameters of the views are drawn on the CPU and the images transformed on their device with kornia, batch by batch: their speed on large batches is to be measured on the GPU.
- `Data.data_augmentation` must stay off for this task, the views replacing it; the checks refuse it on.
- The k-nearest neighbour evaluation compares every labelled test image with the other test images (leave-one-out), not with the training images, so that the evaluation only reads the test split like the other tasks.
- The class names of the evaluation are taken from the categories of the three manifests. The samples of a multi-label manifest (`category_ids`) are read with their categories: one with a single mapped category is labelled for the k-NN evaluation, one with several mapped categories is evaluated like an unlabelled image, the k-NN classification giving one class per image.
