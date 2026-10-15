# Classification Workflow

This document follows one classification run from `main()` to the evaluation of a checkpoint, to show which function runs when, what it receives, what it returns and who uses the result next. The **Call Tree Overview** gives the whole run on one page. Each stage then has one section, with one step per function: a short explanation followed by a **real** output.

The examples come from two real runs, both on CPU so that the GPU stayed free for another training:
- **Stages 1 and 2**: `configurations/classification/configuration_example.yaml` (ImageNet-1k, 1000 classes, 256 pixel images, batch size 256, flattened head), with the run folder redirected to a temporary folder.
- **Stages 3, 4 and 5**: `configurations/classification/configuration_synthetic_shapes.yaml` (9 shape classes), shortened to 64 pixel images, batch size 8, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4, a validation batch every 10, 2 loading workers, with the error images and the MLflow tracking turned on. After 12 iterations the network is still close to chance (9 classes: 1/9 = 11.1% accuracy, a loss near ln 9 = 2.197), so the metrics of the examples are poor on purpose: the point is to see the shapes, the files and the order of the steps.

The other tasks have their own document, written the same way: `object_detection_workflow.md`, `semantic_segmentation_workflow.md`, `sequence_classification_workflow.md`, `sequence_labelling_workflow.md`, `sequence_comparison_workflow.md` and `frame_selection_workflow.md`. The steps shared by every task (the training loop, the checkpoints, the MLflow tracking, the standalone evaluation) are explained in most detail here.

## Table of Contents
- [Call Tree Overview](#call-tree-overview)
- [Stage 1 — Entry Point and Configuration Loading](#stage-1--entry-point-and-configuration-loading)
- [Stage 2 — Trainer Initialization](#stage-2--trainer-initialization)
- [Stage 3 — Training Loop](#stage-3--training-loop)
- [Stage 4 — Test Evaluation of a Checkpoint](#stage-4--test-evaluation-of-a-checkpoint)
- [Stage 5 — Standalone Checkpoint Evaluation (Optional)](#stage-5--standalone-checkpoint-evaluation-optional)

---

## Call Tree Overview


```text
main()
│   Entry point of the whole project, the same for every task. It hard-codes the path to one YAML file: to train
│   another configuration, of any task, change that path. It then does four things in order: load and validate the
│   YAML, look up the trainer of the task, build that trainer, start the training loop. (A commented-out
│   run_benchmarking_loop() is the alternative to training, for speed measurements.)
│
├── load_experiment_configuration(configuration_path)
│   │   Prepares the configuration that every later stage reads. The YAML has 5 top-level sections:
│   │   ExperimentConfiguration, Optimization, Data, Backbone, ClassificationHead. They are split in two:
│   │     - experiment_configuration = 'ExperimentConfiguration' alone: task, experiment/train/validation/test
│   │       folders, batch size, logging/checkpoint/validation frequencies, number of iterations, evaluation and
│   │       MLflow tracking settings
│   │     - model_configuration = all the other sections: how to load data, how to build the network, how to optimize
│   │   It then fixes the values YAML reads with the wrong type: folders become Path objects, "1e10" becomes an int,
│   │   learning rate / weight decay become floats, "float32" becomes torch.float32. Finally it adds two keys to
│   │   'Data' (label_mode, label_dictionary) and runs the checks. It returns both dictionaries to main().
│   │
│   ├── load_configuration(configuration_path)
│   │       Plain file reader: raises FileNotFoundError if the YAML is missing, otherwise parses it with
│   │       yaml.safe_load into one raw nested dictionary (an empty file gives {}). No conversion or validation here.
│   │
│   ├── build_label_dictionary(label_mapping_configuration, task)
│   │       Decides which class index each dataset category is trained as. For classification the mode must be
│   │       "multi_class": it reads the JSON file at 'Data.label_mapping.mapping_file_path' ({category: index}),
│   │       turning digit keys into ints. (The extra background class at 0 is for detection and segmentation only.)
│   │       The result is stored in model_configuration["Data"]["label_dictionary"]. Later, the datasets use it to
│   │       turn each image's raw category into a class index, and the classifier counts its distinct values to size
│   │       its output layer. An unknown mode returns None, which the checks below then report.
│   │
│   └── check_configuration(experiment_configuration, model_configuration)
│       │   Safety net run before anything expensive (data loaders, model, GPU) is created. Each check returns a list
│       │   of error messages instead of raising, so ALL problems are gathered. If the list is not empty, the loader
│       │   prints every message in red with a link to the YAML file and exits. It runs, in order:
│       │
│       ├── check_experiment_configuration()      task name is a known task, training loop settings, per-class
│       │                                         report, number of sample images, MLflow tracking settings
│       ├── check_data_configuration()            loader flags, label mapping mode, input image size is a power of 2
│       ├── check_backbone_configuration()        convolution blocks numbered "1".."N", each [layers, channels],
│       │                                         and the image still has at least 1 pixel after every pooling
│       ├── check_optimization_configuration()    optimizer, weight decay, warm-up / cosine schedule, gradient
│       │                                         clipping, mixed precision (requires float32 weights)
│       └── check_classification_configuration()  classification-specific checks:
│               ├── data:       no DALI loaders, "multi_class" mapping, at least 2 classes indexed 0..K-1 without
│               │               gaps, and no detection blocks in the backbone (the head uses the last feature map)
│               ├── no sequence settings (frame shuffling, sequence normalization), since images have no frames
│               ├── head:       global pooling (average/max/flatten), hidden layer sizes, dropout and label
│               │               smoothing in [0, 1), output mode (multi_class/multi_label), class weighting
│               └── evaluation: top_k > 0, target_precision in [0, 1], error images only in multi_class mode
│
├── get_trainer_class(task)
│       Lookup in the TASK_TRAINER_CLASSES registry, which maps each task name to its trainer class
│       (object_detection, classification, semantic_segmentation, sequence_classification, sequence_labelling,
│       sequence_comparison, frame_selection). For "classification" it returns the ClassificationTrainer class itself,
│       not an object. main() then builds the object by passing it both dictionaries and the YAML path (the trainer
│       copies the YAML into the run folder).
│
├── ClassificationTrainer(experiment_configuration, model_configuration, configuration_path)  [STAGE 2]
│   │   Receives both dictionaries and the YAML path from main(). Before calling the shared constructor, it stores the
│   │   classification settings that its hooks read later: output_mode and class_weighting (ClassificationHead), then
│   │   top_k, target_precision, save_predictions and error_dumps (evaluation). TASK_NAME is "classification"; the base
│   │   class refuses a configuration of another task.
│   │
│   └── BaseTrainer.__init__()
│       │   Does everything that does not depend on the task, in this order: read the settings, pick the device, name
│       │   the run folder, build the augmenter, the paths, the three loaders, choose the sample images, build the
│       │   model, the optimizer and the metric tracker, print the experiment information and the model summaries,
│       │   then restore the last checkpoint if resume_training is on. The steps that depend on the task (loaders,
│       │   classes of the test samples, model, tracked metrics, split description) are hooks that
│       │   ClassificationTrainer implements. Its attribute start_iteration tells the training loop where to begin.
│       │   The MLflow tracker is a DisabledTracker that does nothing until the training loop starts.
│       │
│       ├── get_device()
│       │       Returns torch.device("cuda") when an NVIDIA GPU is visible, else "mps" (Apple), else "cpu". The model,
│       │       the batches and the checkpoints are all moved to this device.
│       ├── get_run_name() -> build_run_name()
│       │       Builds the folder name from the settings that define the model: task code, input, backbone, head, number
│       │       of classes and optimizer. Batch size, learning rate, augmentation and dropout give no token. The run
│       │       folder is experiment_folder/<run name>: the same settings resume in the same folder, and any change of
│       │       the architecture starts a new one.
│       ├── DataAugmenter(augmentation_configuration)
│       │       Reads Data.data_augmentation: the global activated flag and probability, and the activated flag and
│       │       probability of each of the 14 augmentation types. It only stores them; the random draws happen later, on
│       │       the training batches (Stage 3, augment_batch).
│       ├── setup_training_paths()
│       │       Computes experiment_folder/Tensorboard, /Weights and /training_configuration.yaml. For a training it
│       │       creates the two folders and copies the YAML into the run folder, so that the run keeps the settings it
│       │       was launched with. A resumed run first compares the YAML with the saved copy
│       │       (compare_with_saved_configuration), lists the changed settings and keeps a changed copy as
│       │       training_configuration_until_iteration_<n>.yaml. With view_input_images it only returns the paths and
│       │       creates nothing.
│       ├── build_dataloaders()
│       │   │   Hook implemented by ClassificationTrainer. Builds the loaders of the three splits and keeps class_names
│       │   │   (one name per class index, read from the manifest of the training split) for the evaluation reports.
│       │   │   Returns the training, validation and test loaders.
│       │   │
│       │   └── get_classification_dataloaders()
│       │       │   For each of the train, validation and test folders: builds a ClassificationDataset from
│       │       │   <folder>/labels.json and <folder>/data/, then wraps it in a loader. The three splits share the same
│       │       │   image size, keep_ratio, label dictionary and loading settings.
│       │       │
│       │       ├── ClassificationDataset.__init__()
│       │       │       Reads the labels.json manifest, converts the raw category of every sample into a class index
│       │       │       with the label dictionary (samples of an unmapped category are dropped), shuffles the list once
│       │       │       and builds the class names. No image is read here: images are opened in __getitem__, when a
│       │       │       batch is requested.
│       │       └── build_native_loader()
│       │               Wraps the dataset in a torch DataLoader: batch_size from the YAML, reshuffling at every epoch,
│       │               Data.loading.number_workers persistent worker processes, prefetch_batches batches prepared
│       │               ahead, pinned memory for a fast copy to the GPU, one OpenCV thread per worker.
│       ├── save_data_record() -> record_split() -> build_split_record()
│       │       Writes data.json next to the YAML: the class names (list_class_names), then the data folder, the
│       │       SHA-256 of labels.json and the images of every split. A resumed run keeps an unchanged record, and keeps a changed one as
│       │       data_until_iteration_<n>.json before writing the new one. With view_input_images nothing is written.
│       ├── get_test_sample_classes() + choose_sample_identifiers()
│       │       The hook lists the class of every test image. choose_sample_identifiers then picks, once for the whole
│       │       run, the evaluation.number_sample_images test images drawn at every evaluation (Stage 4), spread over
│       │       the classes, so that the predictions of the same images can be followed over the training.
│       ├── build_model()
│       │   │   Hook implemented by ClassificationTrainer. Creates the ImageClassifier from the model registry
│       │   │   (TASK_MODEL_CLASSES["classification"]) and, with class_weighting "balanced", sets its class weights from
│       │   │   the training sample counts. BaseTrainer then moves the model to the device and the dtype.
│       │   │
│       │   ├── ImageClassifier.__init__()
│       │   │   │   Counts the classes (distinct values of the label dictionary), then builds in order: the backbone,
│       │   │   │   the size of the pooled vector, the pooling layer and the classification head. It also registers
│       │   │   │   class_weights, a vector of ones that the trainer overwrites when the weighting is balanced.
│       │   │   │
│       │   │   ├── build_backbone()
│       │   │   │   │   Builds the feature extractor from the 'Backbone' section: its input normalizer, then one
│       │   │   │   │   ConvolutionBlock per entry of Backbone.convolutions. Returns the Backbone module, whose weights
│       │   │   │   │   are random at this point.
│       │   │   │   │
│       │   │   │   ├── InputNormalizer(input_normalization, per_sample_normalization)
│       │   │   │   │       Parameter-free layer at the entrance of the backbone: maps the raw [0, 255] pixels to [-0.5,
│       │   │   │   │       0.5] (input_normalization) and/or standardizes every image to zero mean and unit variance
│       │   │   │   │       (per_sample_normalization).
│       │   │   │   └── ConvolutionBlock(...)   one per entry of Backbone.convolutions
│       │   │   │           N convolution layers (3x3, stride 1, padding 1), each followed by batch normalization and
│       │   │   │           ReLU, with the optional residual shortcuts, squeeze-and-excitation or inception modules,
│       │   │   │           then a 2x2 max pooling that halves the height and width (the pooling of the last block
│       │   │   │           depends on Backbone.last_max_pooling). With residual_nets_v2, the blocks with shortcuts
│       │   │   │           place them before the activations: normalization and ReLU before every convolution,
│       │   │   │           nothing after the addition, and one normalization and ReLU at the end of the block.
│       │   │   ├── compute_pooled_features()
│       │   │   │       Applies the convolution and pooling size formula to every layer (no image is run through the
│       │   │   │       network) to get the channels C and the size HxW of the last feature map. Returns C for the
│       │   │   │       average and max poolings, or C x H x W for flatten: the number of inputs of the first layer of
│       │   │   │       the head.
│       │   │   ├── build_global_pooling()
│       │   │   │       Returns the layer turning the [B, C, H, W] feature map into a [B, F] vector:
│       │   │   │       AdaptiveAvgPool2d(1) + Flatten, AdaptiveMaxPool2d(1) + Flatten, or only Flatten.
│       │   │   └── ClassificationHead.__init__()
│       │   │       │   Builds the hidden layers of ClassificationHead.fully_connected (Dropout, Linear, ReLU for each),
│       │   │       │   then a last Dropout and a Linear layer with one output (logit) per class.
│       │   │       │
│       │   │       └── build_fully_connected_block()
│       │   │               Stacks the hidden layers: for every size of the list, Dropout (an identity at rate 0),
│       │   │               Linear(previous size, size) and ReLU.
│       │   └── compute_balanced_class_weights()
│       │           Only called when ClassificationHead.class_weighting is "balanced". weight of a class = total samples
│       │           / (number of classes x samples of the class), so that rare classes weigh more in the loss; a class
│       │           without sample counts as having one. The weights are copied into model.class_weights.
│       ├── initialise_pretrained_components()
│       │       Only acts when Backbone.pretrained or ClassificationHead.pretrained name another run (experiment_folder
│       │       not null): loads its trained weights into the component and freezes it if asked. A run from scratch does
│       │       nothing here, and a run that resumes from its own checkpoint does not reload the source run.
│       ├── build_optimizer()
│       │       Creates Adam, AdamW or SGD (Nesterov, momentum 0.9) over the model parameters, with the learning rate
│       │       and the weight decay of the configuration. Every transferred component that is not frozen gets its own
│       │       parameter group at learning rate x learning_rate_multiplier. base_learning_rates keeps the starting rate
│       │       of every group for the schedule. With Optimization.exponential_moving_average.activated, the
│       │       ExponentialMovingAverage is then created: a copy of the model, never trained, averaging its weights.
│       ├── get_tracked_metrics() + MetricTracker.__init__()
│       │       get_tracked_metrics returns {total_loss, classification_loss, accuracy} with their display names.
│       │       MetricTracker stores them with batch_size, validation_iteration_ratio and the kind of sample ("Images"):
│       │       it will average them over every console interval, print them with their changes and write them to
│       │       TensorBoard.
│       ├── print_experiment_information()
│       │   │   Prints the launch panel: run name, clickable folders, the settings changed since the last start of a
│       │   │   resumed run, one block per transferred component, one line per split, iterations per epoch, batch
│       │   │   size, checkpoint frequencies, learning rate, optimizer, schedule, gradient clipping, weight
│       │   │   initialisation, moving average of the weights, precision, resume flag, validation ratio and MLflow
│       │   │   backend. Display only, except the split sizes, kept for MLflow.
│       │   │
│       │   └── describe_split() -> count_split_samples()
│       │           Reads the labels.json of a split and returns its number of samples and categories, e.g. '50000
│       │           images, 1000 categories'. The size of the training split gives the iterations per epoch.
│       ├── print_summary() / print_receptive_field_summary() / print_model_size_summary()
│       │       print_summary always runs: the input and output shape of every backbone block (traced with a dummy
│       │       image), then the head panel. The two others are switched by diagnosis_modes.compute_receptive_field and
│       │       compute_model_size: receptive field of every layer, and parameters and size of backbone, head and total.
│       ├── view_input_batches()  (only with diagnosis_modes.view_input_images, then exits)
│       │       Shows an augmented training image next to a validation image, each with its class. ENTER closes the
│       │       viewer and exit() stops the program before any training; no run folder was created.
│       └── restore_last_model()
│           │   Only called when resume_training is true. Returns the iteration where the loop starts: 1 for a new run,
│           │   or the last saved iteration + 1 after loading that checkpoint into the model and the optimizer.
│           │
│           ├── get_last_checkpoint_iteration()
│           │       Reads the first line of Weights/full-checkpoint (e.g. model_checkpoint_path: "Iteration_12") and
│           │       returns 12, or 0 when the file does not exist.
│           └── load_model() -> load_checkpoint()
│                   load_checkpoint reads Weights/Iteration_N/model_00000N.pt with torch.load, mapped to the current
│                   device. load_model puts its model_state in the model and its optimizer_state in the optimizer,
│                   and restores the moving average of the weights from its exponential_moving_average_state.
│
└── trainer.run_training_loop()  [STAGE 3]
    │   Runs the iterations (run_training_iterations) and closes the MLflow run whatever ends them: FINISHED, KILLED
    │   (Ctrl+C) or FAILED (too many errors, or an unexpected error, raised again once the run is closed).
    │
    └── run_training_iterations()
        │   Loops from start_iteration for number_iterations iterations. Each iteration: save (and evaluate) a
        │   checkpoint when due (never at the start of the last iteration), fetch a batch, take one training step,
        │   run one validation batch when due, and print a summary every information_dump iterations. A failing
        │   iteration is retried; more than 5 failures in a row, or Ctrl+C, end the loop. A last checkpoint is always
        │   saved when the loop ends, and evaluated when the run completes on a weight_saving_iterations iteration.
        │
        ├── setup_tensorboard_writers()
        │       Creates Tensorboard/Train and, when validation_iteration_ratio > 0, Tensorboard/Validation. The Test
        │       writer is only created at the first evaluation.
        ├── initialize_training_loop_state()
        │   │   Creates the iterators over the training and validation loaders, the two tracker dictionaries and the
        │   │   counter of consecutive errors.
        │   │
        │   └── MetricTracker.create_trackers()
        │           Returns {start_time, interval_batches: 0, total_loss: 0.0, classification_loss: 0.0,
        │           accuracy: 0.0}: the sums and the batch count that accumulate() fills during an interval.
        ├── start_experiment_tracking()
        │       With tracking.activated: creates the MLflow tracker (or reopens the run of mlflow_run.json for a
        │       resumed run), sends the parameters, the tags and the configuration file. A background thread writes
        │       everything, so MLflow never slows down or stops the training. Without tracking it does nothing.
        ├── for each training_iteration:
        │   │   One pass of the loop body, from start_iteration (1 for a new run).
        │   │
        │   ├── save_model() -> save_checkpoint()
        │   │       Called when the iteration is a multiple of weight_saving_iterations (checkpoint + evaluation) or of
        │   │       intermediate_weight_saving_iterations (checkpoint only), except the last iteration. Writes
        │   │       Weights/Iteration_N/model_00000N.pt (model, optimizer, task, run name, MLflow run, transfer
        │   │       sources, moving average of the weights when on) through a temporary file, then adds the
        │   │       iteration to Weights/full-checkpoint.
        │   ├── run_test_evaluation()   (every weight_saving_iterations)  -> [STAGE 4]
        │   │       Evaluates the checkpoint that was just saved on the whole test split, then its moving average
        │   │       of the weights when on. Not run for the first iteration of the run. See Stage 4.
        │   ├── get_next_batch()
        │   │   │   Takes the next batch from the loader iterator. At the end of an epoch the iterator is restarted,
        │   │   │   which reshuffles the samples; a loading error is retried up to 10 times in a row. Returns images
        │   │   │   [B, 3, H, W] uint8, labels [B] int64 and sample_identifiers [B].
        │   │   │
        │   │   └── ClassificationDataset.__getitem__() -> preprocess_image()
        │   │           Runs in the loader workers, once per sample: cv2.imread (BGR, uint8), resize to image_size with
        │   │           black bars when keep_ratio is true, conversion to a [3, H, W] tensor, label = class index. The
        │   │           default collate function stacks B samples into one batch.
        │   ├── MetricTracker.count_frames()
        │   │       Counts the real frames of the batch for the frame throughput of the sequence tasks; nothing for
        │   │       images.
        │   ├── set_training_mode()
        │   │       model.train(): dropout is active and the batch normalization layers update their statistics.
        │   │       Needed again after every evaluation and validation, which switch to model.eval().
        │   ├── run_model_iteration()
        │   │   │   The same function runs a training batch and a validation batch: prepare, augment (training only),
        │   │   │   forward, losses, metrics, tracking. It returns the total loss, which the loop then back-propagates.
        │   │   │
        │   │   ├── prepare_batch()
        │   │   │       Moves the images to the device as float32 (still in [0, 255], BGR) and the labels to the device
        │   │   │       as int64 class indices.
        │   │   ├── augment_batch() -> DataAugmenter.augment_batch()
        │   │   │       Training only. Each image is augmented with probability Data.data_augmentation.probability: the
        │   │   │       geometric types first (crop, flips, rotations), then the photometric ones (brightness, CLAHE,
        │   │   │       noise, blur...), each type with its own probability. The labels do not change.
        │   │   ├── forward_pass() -> ImageClassifier.forward()
        │   │   │   │   Calls model(input_tensor=images, labels=labels), in bfloat16 when mixed_precision is on.
        │   │   │   │   Returns class_logits [B, K], class_probabilities [B, K] and classification_loss (a scalar,
        │   │   │   │   since the labels are given).
        │   │   │   │
        │   │   │   ├── Backbone.forward()
        │   │   │   │       Normalizes the pixels (x / 255 - 0.5 with input_normalization), then runs the image through
        │   │   │   │       every ConvolutionBlock. Returns the final feature map [B, C, H', W'].
        │   │   │   ├── global_pooling + ClassificationHead.forward()
        │   │   │   │       The pooling (or flatten) gives a [B, F] vector; the head applies its hidden layers and the
        │   │   │   │       last Linear layer to give the [B, K] logits, one per class.
        │   │   │   ├── convert_logits_to_probabilities()
        │   │   │   │       Softmax over the classes in "multi_class" mode (each row sums to 1), independent sigmoids
        │   │   │   │       in "multi_label" mode.
        │   │   │   └── compute_classification_loss()
        │   │   │           Cross entropy of the logits against the class indices (with the class weights and the
        │   │   │           label smoothing), or binary cross entropy with logits against the multi-hot labels in
        │   │   │           "multi_label" mode.
        │   │   ├── compute_batch_metrics()
        │   │   │       Accuracy of the batch, from the logits already computed (no second forward pass): the share of
        │   │   │       images whose highest logit is the true class.
        │   │   └── MetricTracker.accumulate() + log_to_tensorboard()
        │   │           accumulate adds each tracked metric of the batch to the sums of the interval and counts the
        │   │           batch. log_to_tensorboard writes each metric of the batch at the current iteration.
        │   ├── training_loss.backward()
        │   │       Computes the gradient of the total loss for every trainable parameter. Training batches only.
        │   ├── step_optimizer() -> compute_learning_rate_factor()
        │   │       Clips the gradient norm when Optimization.gradient_clipping_norm is set, sets the learning rate of
        │   │       every parameter group to base rate x factor(iteration) (warm-up, then constant or cosine), calls
        │   │       optimizer.step(), updates the moving average of the weights when on, and writes the learning
        │   │       rate to TensorBoard.
        │   ├── validation pass: run_model_iteration() without augmentation
        │   │       Every validation_iteration_ratio iterations: model.eval(), no gradients, next validation batch,
        │   │       same run_model_iteration with apply_augmentation=False. Its metrics go to the validation trackers
        │   │       and to Tensorboard/Validation.
        │   └── average_trackers() + log_interval_metrics() + print_interval_summary()
        │           Every information_dump iterations: divides the training and validation sums by the number of
        │           batches each really ran, sends the averages, the throughput and the learning rate to MLflow, prints
        │           the summary (every value with its change since the previous summary and the best value of the run
        │           marked), then starts new trackers.
        ├── final save_model() + run_test_evaluation()
        │       After the last iteration (or an interruption, or too many errors): saves a last checkpoint and prints
        │       "Model successfully saved". When the run completed and its last iteration is a multiple of
        │       weight_saving_iterations, this final checkpoint is evaluated (Stage 4).
        └── close TensorBoard writers, then finish_experiment_tracking()
                Closes the Train and Validation writers; run_training_loop then sends Results.txt to MLflow and closes
                the MLflow run with its status.

run_test_evaluation(iteration)  [STAGE 4]
│   Scores the current weights on the whole test split, exactly once per image, then reports. It is called by the
│   training loop at every weight_saving_iterations checkpoint, and by Stage 5 for each evaluated checkpoint. It puts
│   the model in eval mode, runs the test loader without gradients, accumulates the predictions, draws the sample
│   images, computes the metrics, then prints, writes and logs them. With the moving average of the weights on, the
│   same evaluation runs again on the averaged weights, reported as "EMA Weights" in Evaluation/Iteration_N/EMA/ and
│   Tensorboard/Test_EMA.
│
├── create_evaluation_accumulator()
│       Returns the empty collection filled during the loop: lists for probabilities, labels and sample identifiers, the
│       folder Evaluation/Iteration_N/Errors for the error images and their counter per class pair.
├── for each test batch:
│   │   A progress bar "Test Evaluation Iteration N" follows the loader.
│   │
│   ├── prepare_batch()
│   │       Same function as in training: images to the device as float32, labels as int64.
│   ├── predict_batch() -> ImageClassifier.forward() without labels
│   │       Runs the model as in deployment: no labels, so no loss. Returns class_logits and class_probabilities.
│   ├── accumulate_evaluation_batch()
│   │   │   Moves the probabilities, labels and sample identifiers of the batch to the CPU and appends them to the
│   │   │   accumulator. With error_dumps activated it also saves the misclassified images of the batch.
│   │   │
│   │   └── dump_errors() -> reserve_error_image() + build_input_image()
│   │           For each wrong prediction: reserve_error_image counts one more image for the pair TRUE-TO-PREDICTED and
│   │           refuses it past maximum_per_class_pair; build_input_image rebuilds the network input; a caption bar
│   │           names both classes and the score; the PNG is written in Errors/<TRUE>-TO-<PREDICTED>/.
│   └── save_sample_images() -> build_sample_image()
│           For each of the fixed sample images found in the batch: the network input captioned with its true class,
│           its predicted class and its probability, green when right and red when wrong, saved in
│           Evaluation/Iteration_N/Samples/ and kept for TensorBoard.
├── compute_evaluation_results()
│   │   Runs once the whole split has been predicted. Concatenates the batches, computes every metric and writes the
│   │   evaluation files, then returns one EvaluationResults dictionary (header text, headline metrics, notes, tables,
│   │   scalars, class scalars, images, file links, headline metric) that the reporter displays.
│   │
│   ├── compute_confusion_matrix()
│   │       Counts, for each pair (true class, predicted class), the number of images: a [K, K] table, rows true,
│   │       columns predicted. Multi-class mode only.
│   ├── compute_metrics_from_confusion_matrix() -> compute_precision_recall_f1()
│   │       From the diagonal and the row and column sums: accuracy, balanced accuracy, macro F1 and, for each class,
│   │       precision, recall, F1 and support (number of test images of the class).
│   ├── compute_top_k_accuracy()
│   │       Share of images whose true class is among the top_k highest probabilities.
│   ├── save_confusion_matrix_table()
│   │       Writes the matrix to Evaluation/Iteration_N/confusion_matrix.csv, with the class names as headers.
│   ├── find_class_thresholds()
│   │       For each class (one against the rest), finds the lowest probability threshold whose precision stays at or
│   │       above target_precision, with the recall kept at that threshold, or None when no threshold reaches it. Saved
│   │       in class_thresholds.json.
│   ├── build_class_metrics_rows() + write_class_metrics_file()
│   │       Merges the per-class metrics and thresholds into one row per class, in class index order, and writes
│   │       class_metrics.csv. Written at every evaluation, whatever per_class_report.
│   ├── save_confusion_matrix_image()   (only with per_class_report)
│   │       Draws the confusion matrix with matplotlib (counts, shares, recall column, precision row, colour bar) and
│   │       saves confusion_matrix.png. The same image is later sent to TensorBoard.
│   ├── write_predictions_file()   (only with save_predictions)
│   │       Writes predictions.csv: one row per test image with its file name, true class, predicted class, whether it
│   │       is correct and the probability of every class.
│   └── build_class_metrics_table() + build_class_metric_scalars()
│           Prepare what is displayed: the per-class table (columns, rows, sort by F1, legend) and the per-class
│           TensorBoard scalars Precision/<class>, Recall/<class>, F1/<class>.
└── report_evaluation()
    │
    ├── report_evaluation_results()
    │       Writes the global scalars, the sample images (and with per_class_report the class scalars and the confusion
    │       matrix image) to Tensorboard/Test, builds the report block, appends its plain text to Results.txt, prints
    │       the coloured version in the console and prints the links of the files written by the evaluation.
    └── track_evaluation()
            With MLflow tracking: sends the global scalars (and with per_class_report the class scalars), the best
            value so far of the headline metric, and the small evaluation files to the MLflow run.

scripts/evaluation/evaluate_checkpoint.py  [STAGE 5]
│   Separate script, run on its own (python scripts/evaluation/evaluate_checkpoint.py): evaluates checkpoints of a
│   trained run without training. Its settings are variables at the top of its main block (no command line): the run
│   name, the iteration ("all" for every checkpoint from the last one, None for the last one, or an integer), an
│   optional other labelled test folder, the evaluation batch size, whether the evaluation is tracked in MLflow, and
│   the output folder.
│
├── load_experiment_configuration() + get_trainer_class() + trainer construction   (same as Stages 1-2)
│       Loads training_configuration.yaml from the trained run, points experiment_folder to the output folder (and
│       test_data_folder, batch_size and tracking.activated to the given values), turns off the transfer of pretrained
│       components, then builds the same trainer as for a training. Its run folder is created in the output folder.
├── get_checkpoint_iterations() / get_last_checkpoint_iteration()
│       Read Weights/full-checkpoint of the trained run: all the recorded iterations, evaluated from the last one
│       ("all"), or only the last one (None). An integer iteration is used as given.
└── evaluate_checkpoints()
    │   Opens the MLflow run of the evaluation, evaluates the checkpoints, sends the summary, and closes the run.
    │
    ├── start_evaluation_tracking()
    │       With tracking: a new MLflow run '<run name>-evaluation' in the backend of the trained run, tagged with the
    │       MLflow run of the training ('evaluation_of'). The MLflow run of the training is never modified.
    └── summarise_checkpoint_evaluations()
        │   For each iteration: skips it with a warning if its file is missing, loads its weights into the trainer's
        │   model, runs the test evaluation, and keeps the scalars. At the end writes Evaluation/evaluation_summary.csv
        │   (one row per iteration) and prints the best iteration of the headline metric.
        │
        └── run_test_evaluation()  for each checkpoint  -> [STAGE 4]
                The same function as in the training loop, so the same Results.txt block, TensorBoard Test curves,
                sample images and evaluation files.
```

---

## Stage 1 — Entry Point and Configuration Loading

> The examples below are real outputs from running the loader on `configurations/classification/configuration_example.yaml` (ImageNet-1k, 1000 classes).

### Step 1.1 — `main()`
This is the entry point of the whole project, the same for every task. It builds the path to the YAML file of the experiment (`configuration_large_model.yaml` in the repository at the moment): to train another configuration, of this task or of another one, change that path. It loads and validates that file (Step 1.2), asks the registry which trainer fits `experiment_configuration["task"]` (Step 1.6), builds that trainer with both dictionaries and the YAML path (Stage 2) and starts training (Stage 3). It returns nothing. Everything after this point happens inside the trainer.

```python
experiment_configuration_path = Path(__file__).parent / "configurations" / "classification" / "configuration_example.yaml"
experiment_configuration, model_configuration = load_experiment_configuration(configuration_path=...)
trainer_class = get_trainer_class(task=experiment_configuration["task"])  # -> ClassificationTrainer
trainer = trainer_class(experiment_configuration=..., model_configuration=..., configuration_path=...)
trainer.run_training_loop()
```

### Step 1.2 — `load_experiment_configuration()`
This function produces the two dictionaries that every later stage reads. It removes the `ExperimentConfiguration` section from the raw YAML and keeps it as `experiment_configuration`, which holds the run settings. Whatever is left becomes `model_configuration`, which holds the data, network and optimization settings. It then converts the values YAML reads with the wrong type, adds `label_mode` and `label_dictionary` to `Data` (Step 1.4), and runs the checks (Step 1.5).

**Example: how the dictionaries are split**
```text
experiment_configuration keys: task, experiment_folder, train_data_folder, validation_data_folder, test_data_folder,
                               batch_size, information_dump, weight_saving_iterations,
                               intermediate_weight_saving_iterations, number_iterations, validation_iteration_ratio,
                               evaluation, tracking, learning_rate, dtype, resume_training, diagnosis_modes
model_configuration keys:      Optimization, Data, Backbone, ClassificationHead
Data keys after loading:       DALI, loading, image_settings, label_mapping, data_augmentation, label_mode, label_dictionary
```

**Example: types before and after conversion**

| Key | Raw YAML value | After conversion |
|---|---|---|
| `number_iterations` | `'1e10'` (string) | `10000000000` (int) |
| `dtype` | `'float32'` (string) | `torch.float32` |
| `train_data_folder` | `'/home/.../imagenet-1k/manifests/train'` (string) | `PosixPath('/home/.../imagenet-1k/manifests/train')` |
| `learning_rate` | `0.0001` (`1.0e-4` has a dot, so YAML already reads a float) | `0.0001` (float) |

### Step 1.3 — `load_configuration()`
This is a plain file reader. It raises `FileNotFoundError` if the YAML file is missing. Otherwise it parses the file with `yaml.safe_load` into one nested dictionary that mirrors the file exactly. Nothing is converted or validated here. Numbers written without a dot, such as `1e10`, are still strings at this point, which is why Step 1.2 has to convert them.

**Example output**
```text
top-level keys:    ['ExperimentConfiguration', 'Optimization', 'Data', 'Backbone', 'ClassificationHead']
number_iterations: '1e10'      <- still a string
dtype:             'float32'   <- still a string
learning_rate:     0.0001      <- already a float
```

### Step 1.4 — `build_label_dictionary()`
This function decides which class index each dataset category is trained as. For classification the mode must be `"multi_class"`. It reads the JSON file at `Data.label_mapping.mapping_file_path` and turns digit keys into integers. The result goes into `model_configuration["Data"]["label_dictionary"]`. Later, the datasets use it to turn each image's raw category into a class index, and the classifier counts its distinct values to size its output layer. You can merge categories by mapping several of them to the same index.

**Example: `label_mapping.json` (input) → `label_dictionary` (output)**
```text
{'1': 0, '2': 1, '3': 2, '4': 3, ...}      ->   {1: 0, 2: 1, 3: 2, 4: 3, ...}
                                                1000 entries, 1000 distinct classes -> 1000 output neurons
```

### Step 1.5 — `check_configuration()`
This is the safety net that runs before anything expensive, such as the data loaders, the model or GPU memory, is created. It runs the common checks (experiment, data, backbone, optimization) first, then the checks of the task found in `TASK_CONFIGURATION_CHECKS` (for classification: data, sequence settings, head, evaluation). Each check returns a list of messages instead of raising an error, so every problem in the file is reported together. If there are any messages, `load_experiment_configuration()` prints them in red under a link to the YAML file and exits with code 1.

**Example: output with `global_pooling: "median"` and `dropout_rate: 1.5` in the YAML**
```text
-----------------------------------------------------------------------------------------------
Invalid Configuration File : file:///.../configuration_wrong.yaml
-----------------------------------------------------------------------------------------------
  - 'ClassificationHead.global_pooling' must be one of ['average', 'max', 'flatten'], got 'median'.
  - 'ClassificationHead.dropout_rate' must be a number within [0, 1), got 1.5.
SystemExit code 1
```

### Step 1.6 — `get_trainer_class()`
This is a lookup in `TASK_TRAINER_CLASSES`, which maps each task name to its trainer class. It returns the class itself, not an object. `main()` builds the object right after, which is where Stage 2 begins. An unknown task raises a `KeyError` that lists the valid tasks. In practice the checks in Step 1.5 already catch a wrong task name before this point.

**Example output**
```text
get_trainer_class(task="classification")
-> <class 'vision_training.trainers.classification_trainer.ClassificationTrainer'>

get_trainer_class(task="segmentation")
-> KeyError: "No trainer is registered for the task 'segmentation', expected one of ['object_detection', 'classification',
   'semantic_segmentation', 'sequence_classification', 'sequence_labelling', 'sequence_comparison', 'frame_selection']."
```


---

## Stage 2 — Trainer Initialization

> The examples below are real outputs from building the trainer on `configurations/classification/configuration_example.yaml` (ImageNet-1k, 1000 classes, 256 pixel images), with the run folder redirected to a temporary folder. Nothing is trained in this stage: it only prepares everything that Stage 3 needs. The whole construction took 14.2 s on CPU. On this machine a GPU is available, so a normal run prints `cuda` where the example below, run on CPU, prints `cpu`.

### Step 2.1 — `ClassificationTrainer.__init__()`
`main()` builds the trainer with both dictionaries from Stage 1 and the YAML path. Before calling the shared constructor of `BaseTrainer`, this one stores the classification settings that its hooks read later: `output_mode` and `class_weighting` from `ClassificationHead`, and `top_k`, `target_precision`, `save_predictions` and `error_dumps` from `evaluation`. They must be stored first because `BaseTrainer.__init__()` already calls the hooks (`build_dataloaders()`, `build_model()`), which use them.

**Example: the attributes set before `super().__init__()`**
```text
self.output_mode                = 'multi_class'
self.class_weighting            = 'balanced'
self.top_k                      = 5
self.target_precision           = 0.99
self.save_predictions           = False
self.error_dumps_configuration  = {'activated': False, 'maximum_per_class_pair': 20}
```

### Step 2.2 — `BaseTrainer.__init__()`
This is the constructor shared by every task. It first checks that the task of the configuration is the one of the trainer, then it runs the steps below in order: device (2.3), run name (2.4), the settings of the evaluation and of the tracking, augmenter (2.5), paths (2.6), loaders (2.7–2.10), sample images (2.11), model (2.12–2.18), pretrained components (2.19), optimizer (2.20), metric tracker (2.21), information printouts (2.22–2.24), input viewer (2.25) and checkpoint restoration (2.26–2.28). The steps that depend on the task (loaders, classes of the test samples, model, tracked metrics, split description) are abstract hooks that `ClassificationTrainer` implements. At the end, the attribute `start_iteration` tells the training loop where to begin. The MLflow tracker is a `DisabledTracker`, which accepts every call and does nothing, until the training loop opens the real one (Step 3.5): the input viewer, the benchmarking loop and a standalone evaluation never create an MLflow run for the training.

**Example: the error raised when the configuration is for another task**
```text
ValueError: ClassificationTrainer trains the task 'classification', but the configuration is for the task 'object_detection'.
```

**Example: some attributes set by the constructor**
```text
self.per_class_report       = False                 <- evaluation.per_class_report, 1000 classes are too many to list
self.number_sample_images   = 16                    <- evaluation.number_sample_images
self.experiment_tracker     = <DisabledTracker>     <- replaced at the start of the training loop
self.mixed_precision        = True                  <- Optimization.mixed_precision
self.exponential_moving_average = None              <- Optimization.exponential_moving_average.activated is False
self.start_iteration        = 1
```

### Step 2.3 — `get_device()`
This function picks the hardware used for the whole run: CUDA if an NVIDIA GPU is visible, otherwise MPS (Apple GPU), otherwise the CPU. The model, every batch and every checkpoint are moved to this device. The trainer also keeps `dtype` (`torch.float32` here, from the YAML) for the model weights and the images.

**Example output** (this run was forced on the CPU; a normal run on this machine prints `cuda`)
```text
get_device() -> device(type='cpu')
For Training We Will Be Using device: cpu
```

### Step 2.4 — `get_run_name()` → `build_run_name()`
The run folder is `experiment_folder / <run name>`. The name is built from the settings that define the model: task, input, backbone, head, number of classes and optimizer. The same settings always give the same folder, so a stopped run resumes where it was, and any change of the architecture starts a new folder. The batch size, the learning rate and its schedule, the gradient clipping, the mixed precision, the augmentation and the dropout rates give no token, because they can change between two resumptions of a run.

**Example: the run name of `configuration_example.yaml`, read token by token**

| Token | Meaning |
|---|---|
| `CLS` | task (`DET`, `CLS`, `SEG`, `M2O`, `M2M`, `CMP`, `SEL`) |
| `IN` | `input_normalization`: pixels mapped to [-0.5, 0.5] |
| `KAR` | `keep_ratio`: aspect ratio kept, the image is padded to a square |
| `256` | input image size |
| `SE`, `RES`, `BN` | squeeze-and-excitation, residual shortcuts and batch normalization are used in the backbone |
| `B1x64.2x64.2x64.4x64.5x64.5x64` | the 6 blocks of the backbone, `<convolution layers>x<filters>` each |
| `HFLAT.512.256` | head: `flatten` pooling, then hidden layers of 512 and 256 units |
| `CWB` | `class_weighting: "balanced"` |
| `MC1000` | `multi_class` output, 1000 classes |

```text
CLS_IN-KAR-256_SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64_HFLAT.512.256-CWB_MC1000
```
There is no optimizer section at the end because the run uses plain Adam without weight decay.

### Step 2.5 — `DataAugmenter()`
This creates the augmentation tool of the training batches from `Data.data_augmentation`. It only reads and stores the settings: the global `activated` flag and `probability` (the chance that an image is augmented at all), and for each of the 14 augmentation types (5 geometric, 9 photometric) its own `activated` flag and `probability`. The random draws happen later, in Step 3.13, on every training batch. The `independent_frames` and `shuffle_frames` settings only concern the sequence tasks.

**Example: the settings of `configuration_example.yaml`**
```text
activated: True    probability: 0.3    independent_frames: False    shuffle_frames: {activated: False, ...}
types:     13 of the 14 types activated, each with probability 0.5 (random_up_down_flip is off)
expected:  0.3 x 13 x 0.5 = 1.95 augmentations per image, on average
attributes of the augmenter: activated, probability, independent_frames, shuffle_frames_activated,
                             shuffle_frames_probability, type_settings
```

### Step 2.6 — `setup_training_paths()`
This function gives the three paths of the run folder: `Tensorboard/`, `Weights/` and `training_configuration.yaml`. For a training it creates the two folders and copies the YAML into the run folder, so that the run keeps the exact settings it was launched with (Stage 5 reads this copy). With `diagnosis_modes.view_input_images` it only returns the paths: the viewer exits without training and must not leave an empty run folder.

A run resumed from a checkpoint first compares the launched YAML with the copy saved by its previous start (`compare_with_saved_configuration()`, through `find_changed_settings()` of `vision_training/configuration_processing/configuration_changes.py`), since the settings that give no token in the run name (learning rate, `learning_rate_multiplier`, batch size, dropout, augmentation...) may have changed. It returns the changed settings, printed at Step 2.22, and keeps the saved copy as `training_configuration_until_iteration_<n>.yaml` when they differ, n being the iteration of the checkpoint the run resumes from. A file of that name is never replaced. A new run compares nothing.

**Example: what exists in the run folder after this step**
```text
<experiment_folder>/CLS_IN-KAR-256_SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64_HFLAT.512.256-CWB_MC1000/
├── Tensorboard/
├── Weights/
└── training_configuration.yaml    3,913 bytes
```

### Step 2.7 — `build_dataloaders()`
This hook of `ClassificationTrainer` builds the loaders of the three splits (Step 2.8) and keeps `class_names`: one name per class index, read from the categories of the manifest of the training split. The evaluation reports, the error images and the sample images use them. It returns the training, validation and test loaders, which `BaseTrainer` stores as `train_dataloader`, `validation_dataloader` and `test_dataloader`.

**Example: the first class names of ImageNet-1k (the folder names, called wnids)**
```text
trainer.class_names[:5] -> ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475']
```

Once the loaders exist, `BaseTrainer.__init__()` writes `data.json` in the run folder (`save_data_record()`, with `record_split()` and `build_split_record()` of `vision_training/data/data_record.py`). It starts with the class names of the run in class index order (`class_names`, given by `list_class_names()`), which a run adding classes to this one compares with its own (Step 2.19). For each split, it records the absolute data folder, the SHA-256 of `labels.json` (which identifies the images and their labels exactly) and the `file_name` of every sample in manifest order. A model can then be traced back to the exact images it saw, even after the data folders change. A resumed run compares the data folders and SHA-256 with the saved record. When they are unchanged, the record of the first start is kept with its date. When a split changed, the saved record is kept as `data_until_iteration_<n>.json` (never replaced, like the archived configurations) before the new record is written. The input viewer writes nothing. With MLflow on, the record is sent to the `data` folder of the run whenever it is written.

**Example: the start of `data.json` for ImageNet-1k (1.4 seconds to build, 65 MB for the 1,271,167 training images)**
```text
{
  "created": "2026-10-15 15:12:40",
  "class_names": ["n01440764", "n01443537", "n01484850", ...],
  "splits": {
    "train": {
      "data_folder": "/home/.../imagenet-1k/manifests/train",
      "labels_sha256": "973d0172c405b0f1705b5cc04a45d24ab28f676def1fb067d2d37c9e746108f4",
      "number_images": 1271167,
      "images": [
        "n01440764/n01440764_10026_n01440764.JPEG",
        "n01440764/n01440764_10027_n01440764.JPEG",
        ...
```

### Step 2.8 — `get_classification_dataloaders()`
For each of `train_data_folder`, `validation_data_folder` and `test_data_folder`, it builds one `ClassificationDataset` (Step 2.9) from `<folder>/labels.json` and the images in `<folder>/data/`, then wraps it in a loader (Step 2.10). The three splits share the image size, `keep_ratio`, the label dictionary of Stage 1 and the `Data.loading` settings. Only the training split is used to learn: the validation split gives the curves of Stage 3 and the test split is only used by Stage 4.

**Example: size of the three splits**
```text
train       1271167 samples -> 4966 batches of 256 (one epoch = 4966 iterations)
validation    10000 samples ->   40 batches of 256
test          50000 samples ->  196 batches of 256
```

### Step 2.9 — `ClassificationDataset.__init__()`
This reads the `labels.json` manifest of a split. For every sample it converts the raw category into a class index with the `label_dictionary` (a sample whose category is not mapped is left out, with a warning), shuffles the list once, and builds the class names from the categories. It also keeps every sample by identifier (`sample_records`), so that the evaluation can name the image of a prediction. No image is opened here: the files are only read when a batch is requested (Step 3.9), which is why a manifest of 1.27 million images is ready in seconds.

**Example: a manifest sample, and a kept sample after mapping and shuffling**
```text
labels.json:   {'id': 1, 'file_name': 'n01440764/n01440764_10026_n01440764.JPEG', 'category_id': 1}
categories[0]: {'id': 1, 'name': 'n01440764'}
kept (shuffled order, its raw category is mapped to the class index 501):
               {'identifier': 638979, 'file_name': 'n03045698/n03045698_3287_n03045698.JPEG', 'class_indices': [501]}
```

### Step 2.10 — `build_native_loader()`
This wraps the dataset in a PyTorch `DataLoader` with the `Data.loading` settings: the samples are reshuffled at every epoch, `number_workers` worker processes decode the images (each with a single OpenCV thread, kept alive between epochs and evaluations), `prefetch_batches` batches are prepared ahead of the model, and the batches are copied to pinned memory so that the transfer to the GPU is fast. With `number_workers: 0` the images are loaded in the main process, which is slower but easier to debug.

**Example: the training loader of `configuration_example.yaml`**
```text
len(loader) = 4966   batch_size = 256   num_workers = 8   prefetch_factor = 2   pin_memory = True   persistent_workers = True
```

### Step 2.11 — `get_test_sample_classes()` + `choose_sample_identifiers()`
The evaluation of every checkpoint draws a few fixed test images with their true and predicted classes (Step 4.7), so that the predictions of the same images can be followed over the training. They are chosen once, here, from the identifiers and the classes only, so that a resumed run follows the same images. The hook `get_test_sample_classes()` gives the class of every test image. `choose_sample_identifiers()` groups the images by class, gives one image to each class in turn, in class order, until `evaluation.number_sample_images` images are chosen, then takes the images of each class evenly spread over its sorted identifiers. With 1000 classes and 16 images, the 16 first classes get one image each. `number_sample_images: 0` turns the sample images off.

**Example output** (16 images, 1000 classes)
```text
get_test_sample_classes() -> 50000 entries, e.g. {43211: [864], 43537: [870], 3875: [77], ...}
sample_image_identifiers  -> [1, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501, 551, 601, 651, 701, 751]
                             1 n01440764 (n01440764_ILSVRC2012_val_00000293_n01440764), 51 n01443537 (...), ...
```
The test manifest lists its images class by class, 50 per class, so the first image of each of the first 16 classes is chosen.

### Step 2.12 — `build_model()`
This hook creates the network through the registry `TASK_MODEL_CLASSES["classification"]`, which gives an `ImageClassifier` (Step 2.13). With `class_weighting: "balanced"` it counts the training samples of every class and stores the balanced weights in the model (Step 2.18). Back in `BaseTrainer.__init__()`, the model is then moved to the device and cast to the `dtype`, every layer keeping the PyTorch initialisation. Its weights are random at this point: the transferred components (Step 2.19) and a resumed checkpoint (Step 2.28) replace them.

**Example: size of the model of `configuration_example.yaml`**
```text
Backbone              746,752 parameters
Classification Head 2,485,992 parameters
Total Model         3,232,744 parameters (12.94 MB)
```

### Step 2.13 — `ImageClassifier.__init__()`
This builds the `nn.Module` in this order: the backbone (2.14), the number of features of the pooled vector (2.15), the pooling layer (2.16) and the classification head (2.17). The number of classes is the number of distinct values of the label dictionary. It also registers `class_weights`, a vector of ones (a buffer: saved in the checkpoints but not learned) that Step 2.18 overwrites when the weighting is balanced. The forward pass chains these parts: image → backbone → pooling → head → logits.

**Example: the attributes of the model**
```text
model.backbone               Backbone (6 convolution blocks)
model.global_pooling         Sequential(Flatten)
model.classification_head    ClassificationHead (4096 -> 512 -> 256 -> 1000)
model.class_weights          buffer [1000] torch.float32
model.number_classes         1000
```

### Step 2.14 — `build_backbone()` (→ `InputNormalizer`, `ConvolutionBlock`)
This builds the feature extractor, shared by all the tasks, from the `Backbone` section. It starts with an `InputNormalizer` (parameter-free: pixels `x / 255 - 0.5`), then creates one `ConvolutionBlock` per entry of `Backbone.convolutions`. A block is `N` convolutions (3×3, stride 1, padding 1), each followed by batch normalization and ReLU, then a 2×2 max pooling that halves the height and width. The residual shortcuts start at block 2, and squeeze-and-excitation recalibrates the channels after every convolution. `last_max_pooling: False` removes the pooling of the last block.

With `Backbone.modules.residual_nets_v2: True` (run name token `RES2` instead of `RES`), the blocks with residual shortcuts use the pre-activation order of the paper "Identity Mappings in Deep Residual Networks": every layer is normalization, ReLU, then convolution, nothing follows the addition of the shortcut, so that the shortcuts form an identity path through the block, and the block ends with one normalization and ReLU, so that its output stays normalized and activated like a plain block. Block 1, without shortcut, keeps the original order. The summary names these blocks `Pre-Activation Residual Shortcuts`.

**Example: the backbone panel of `print_summary()` for a 256×256 image** (Step 2.24, shortened)
```text
╭─────────────────────────── Backbone Architecture ────────────────────────────╮
│ Normalization Strategy: Batch                                                │
│                                                                              │
│  - Convolution Block 1                                                       │
│    - Modules      : Squeeze-and-Excitation                                   │
│    - Input Shape  : [1, 3, 256, 256]                                         │
│    - Output Shape : [1, 64, 128, 128]                                        │
│                                                                              │
│  - Convolution Block 2                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 64, 128, 128]                                        │
│    - Output Shape : [1, 64, 64, 64]                                          │
│  ...                                                                         │
│  - Convolution Block 6                                                       │
│    - Modules      : Residual Shortcuts, Squeeze-and-Excitation               │
│    - Input Shape  : [1, 64, 8, 8]                                            │
│    - Output Shape : [1, 64, 8, 8]          <- no pooling: last_max_pooling   │
╰──────────────────────────────────────────────────────────────────────────────╯
Final Backbone output shape: [1, 64, 8, 8]
```

### Step 2.15 — `compute_pooled_features()`
This gives the number of inputs of the first layer of the head, which must be known before the head is built. It does not run an image through the network: it applies the size formula of every convolution and pooling to get the channels `C` and the size `H×W` of the last feature map. The result is `C` for the `average` and `max` poolings (whatever the image size), and `C × H × W` for `flatten`, which is why a flattened head is tied to `Data.image_settings.size` and `Backbone.last_max_pooling`.

**Example: the same backbone with the three poolings**
```text
final feature map [64, 8, 8]
average / max -> 64 features
flatten       -> 64 x 8 x 8 = 4096 features   (the "Input Features" of the head panel)
```

### Step 2.16 — `build_global_pooling()`
This returns the layer that turns the `[B, C, H, W]` feature map into a `[B, F]` vector, the shape the head expects: an adaptive average pooling to 1×1 then a flatten (`F = C`), an adaptive max pooling to 1×1 then a flatten (`F = C`), or only a flatten (`F = C × H × W`), which keeps where in the image each feature was found.

**Example output** (`global_pooling: "flatten"`)
```text
Sequential(
  (0): Flatten(start_dim=1, end_dim=-1)
)
```

### Step 2.17 — `ClassificationHead.__init__()` (→ `build_fully_connected_block()`)
This builds the fully connected part. `build_fully_connected_block()` stacks, for every size of `ClassificationHead.fully_connected`, a dropout (always present, an identity at a rate of 0, so that the names of the weights do not depend on the rate), a `Linear` layer and a ReLU. A last dropout and a `Linear` layer with one output per class follow: its outputs are the logits, the raw class scores. The same head is also used by the sequence tasks, which is why it works on the last dimension of its input.

**Example output** (`fully_connected: [512, 256]`, `dropout_rate: 0.1`, 1000 classes)
```text
ClassificationHead(
  (fully_connected_block): Sequential(
    (0): Dropout(p=0.1, inplace=False)
    (1): Linear(in_features=4096, out_features=512, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.1, inplace=False)
    (4): Linear(in_features=512, out_features=256, bias=True)
    (5): ReLU(inplace=True)
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=256, out_features=1000, bias=True)
)
```

### Step 2.18 — `compute_balanced_class_weights()`
This is only used with `class_weighting: "balanced"`. The weight of a class is `total samples / (number of classes × samples of the class)`, so a class with few images weighs more in the loss, and a perfectly balanced dataset gives a weight of 1 everywhere. A class without any sample counts as having one. The trainer gets the counts from `ClassificationDataset.get_class_counts()` and copies the weights into `model.class_weights`.

**Example: ImageNet-1k training split** (1271167 images, between 722 and 1290 per class)
```text
class with 1290 images: 1271167 / (1000 x 1290) = 0.9854     <- model.class_weights[:5] = tensor([0.9854, 0.9854, ...])
class with  722 images: 1271167 / (1000 x  722) = 1.7606     <- the largest weight of model.class_weights
```

### Step 2.19 — `initialise_pretrained_components()`
This is the transfer learning step. For each of the `Backbone` and `ClassificationHead` sections whose `pretrained.experiment_folder` is not `null`, it loads the trained weights of that other run into the component, and freezes it (or only the scale and shift of the normalization layers of the backbone) if asked. It raises an error if every parameter ends up frozen, since nothing would be learned. When a run resumes from its own checkpoint, the source run is not loaded again: the checkpoint already holds the components as they were trained.

A component is read from the averaged weights of the source checkpoint when the source run kept a moving average and this run has `Optimization.exponential_moving_average.activated`, and from its trained weights otherwise (`read_checkpoint_model_state()`), the console printing which ones. The head may also have more classes than its source run: `prepare_class_layers()` compares the `class_names` of both `data.json` files (the source classes must be the first ones of this run, in the same order), then `extend_class_layers()` copies the rows of the source classes and sets those of the new ones as chosen by `ClassificationHead.pretrained.new_class_initialisation`: with `"source"`, zero weights and a bias 3 below the mean source bias (`NEW_CLASS_BIAS_OFFSET`), so that they start quiet in the softmax; with `"random"`, the random rows the head of this run was built with, the source rows replacing its first rows.

**Example: the head of a 9-class run initialising a run of 20 classes**
```text
Classification Head : 9 Classes Of The Source Run Kept, 11 New Classes Added After Them, Built From The Source Classes.
Classification Head Initialised From The Averaged Weights Of : file:///.../run_37/CLS_IN-KAR-256_..._MC9/Weights/Iteration_2000/model_2000.pt
```

**Example: a run from scratch** (`configuration_example.yaml`)
```text
pretrained_configurations['Backbone']           = {'experiment_folder': None, 'iteration': None, 'freeze': False,
                                                   'freeze_normalization_parameters': False, 'learning_rate_multiplier': 1.0}
pretrained_configurations['ClassificationHead'] = {'experiment_folder': None, 'iteration': None, 'freeze': False,
                                                   'learning_rate_multiplier': 1.0, 'new_class_initialisation': 'source'}
pretrained_checkpoint_paths                     = {'Backbone': None, 'ClassificationHead': None}   <- nothing loaded
```

### Step 2.20 — `build_optimizer()`
This creates the optimizer chosen by `Optimization.optimizer` (`adam`, `adamw` or `sgd` with Nesterov momentum 0.9) over the parameters of the model, with the learning rate of the experiment and `Optimization.weight_decay`. Every transferred component that is not frozen gets its own parameter group at the learning rate times its `learning_rate_multiplier`, so it can learn at another speed. The group exists even with a multiplier of 1, so that the multiplier, which gives no token in the run name, can change when the run is resumed without changing the groups saved in its checkpoints. The trainer then keeps `base_learning_rates`, the rate of every group, which the schedule multiplies at every step (Step 3.22).

With `Optimization.exponential_moving_average.activated`, the trainer then creates an `ExponentialMovingAverage`: a copy of the model, kept in `eval()` mode without gradients, updated after every optimizer step (Step 3.22) with `ema_weights = decay × ema_weights + (1 − decay) × current_weights` and evaluated next to the trained weights (Step 4.1). It is created after the transferred components are loaded, so that it starts from their weights.

**Example output** (`optimizer: "adam"`, `learning_rate: 1.0e-4`, `weight_decay: 0.0`)
```text
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    ...
    eps: 1e-08
    lr: 0.0001
    weight_decay: 0.0
)
param groups: 1    base_learning_rates: [0.0001]
```

### Step 2.21 — `get_tracked_metrics()` + `MetricTracker.__init__()`
`get_tracked_metrics()` lists what is followed on every training and validation batch: the keys that the model or the trainer produce and their display names. The `MetricTracker` stores this list with `batch_size`, `validation_iteration_ratio` and the kind of sample (`"Images"`, which names the throughput). In Stage 3 it averages the metrics over every console interval and prints them (Step 3.24), and writes them to TensorBoard (Step 3.20). It also keeps the values of the previous summary and the best value of the run of every metric, empty for now, for the changes printed in the summaries.

**Example output**
```text
get_tracked_metrics() -> {'total_loss': 'Total Loss', 'classification_loss': 'Classification Loss', 'accuracy': 'Accuracy'}
MetricTracker: batch_size 256, validation_iteration_ratio 25, sample_name 'Images', counts_frames False,
               previous_values {'Train': {}, 'Valid': {}}, best_values {'Train': {}, 'Valid': {}}
```

### Step 2.22 — `print_experiment_information()`
This prints the launch panel, so the settings of the run can be checked before anything long starts: the run name, the clickable folders (`file://` links), the settings and the data changed since the last start of a resumed run (`print_configuration_changes()`, Step 2.6, and `print_data_record_changes()`, Step 2.7), one block per component initialised by another run with its source run folder, source iteration, freezes and learning rate times its multiplier (`print_pretrained_components()`), one line per split (Step 2.23), the number of iterations per epoch (`ceil(training samples / batch size)`), the checkpoint frequencies, the optimization settings and the MLflow backend (`Off`, the local database of the experiments root folder, or a server). It also keeps the number of samples of every split and the iterations per epoch, sent to MLflow as parameters.

**Example output**
```text
--------------------------------------
=== Experiment Launch Information ===
--------------------------------------
- Run Name                : CLS_IN-KAR-256_SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64_HFLAT.512.256-CWB_MC1000
- Experiment Folder       : file:///.../CLS_IN-KAR-256_SE-RES-BN-B1x64.2x64.2x64.4x64.5x64.5x64_HFLAT.512.256-CWB_MC1000
- Tensorboard Directory   : file:///.../CLS_IN-KAR-256_.../Tensorboard
- Weights Directory       : file:///.../CLS_IN-KAR-256_.../Weights
- Configuration File      : file:///.../CLS_IN-KAR-256_.../training_configuration.yaml
- Data Record             : file:///.../CLS_IN-KAR-256_.../data.json
- Train Data Folder       : file:///home/.../imagenet-1k/manifests/train (1271167 images, 1000 categories)
- Validation Data Folder  : file:///home/.../imagenet-1k/manifests/validation (10000 images, 1000 categories)
- Test Data Folder        : file:///home/.../imagenet-1k/manifests/test (50000 images, 1000 categories)
- Iterations Per Epoch    : 4966
- Batch Size              : 256
- Total Iterations        : 10000000000
- Save And Evaluate Every : 10000 iterations
- Save Only Every         : 2500 iterations
- Learning Rate           : 0.0001
- Optimizer               : adam (weight decay 0.0)
- Learning Rate Schedule  : constant (warm-up 0 iterations)
- Gradient Clipping Norm  : None
- Weight Moving Average   : Off
- Mixed Precision         : True
- Resume Training         : True
- Validation Iter. Ratio  : 25
- MLflow Tracking         : Off
```

### Step 2.23 — `describe_split()` → `count_split_samples()`
`print_experiment_information()` calls this hook once per split. `count_split_samples()` reads the `labels.json` of the split and returns its number of samples and categories, counted before the label mapping. The hook turns them into the text between brackets. The size of the training split also gives the number of iterations per epoch.

**Example output**
```text
describe_split(split_data_folder=".../manifests/test") -> (50000, '50000 images, 1000 categories')
number_samples_per_split = {'train_data_folder': 1271167, 'validation_data_folder': 10000, 'test_data_folder': 50000}
iterations_per_epoch     = 4966
```

### Step 2.24 — `print_summary()` / `print_receptive_field_summary()` / `print_model_size_summary()`
`print_summary()` always runs: for every backbone block, the input and output shape traced with a dummy image (as shown in Step 2.14), then a panel for the head. The two others are optional and switched by `diagnosis_modes.compute_receptive_field` and `diagnosis_modes.compute_model_size`: the receptive field of every layer (the area of the input image that one output value depends on), and the number of parameters and the size of the backbone, the head and the whole model.

**Example output: the head panel, the receptive field panel (shortened) and the size panel**
```text
╭──────────────────────────── Classification Head ─────────────────────────────╮
│ Global Pooling     : flatten                                                 │
│ Input Features     : 4096                                                    │
│ Hidden Layers      : [512, 256]                                              │
│ Dropout Rate       : 0.1                                                     │
│ Output Mode        : multi_class                                             │
│ Number Of Classes  : 1000                                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────── Backbone Receptive Field ────────────────────────────────────╮
│ Input Image Size : 256x256                                                                       │
│                                                                                                  │
│ Block 1 - Convolution 1  : Output 256x256x64 | Stride 1 | Receptive Field 3                      │
│ Block 1 - Max Pooling    : Output 128x128x64 | Stride 2 | Receptive Field 4                      │
│ ...                                                                                              │
│ Block 6 - Convolution 5  : Output 8x8x64 | Stride 32 | Receptive Field 602                       │
│                                                                                                  │
│ Squeeze-and-Excitation adds global image context.                                                │
│ This is not captured by the convolutional receptive field above.                                 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Model Size ───────────────────────╮
│ Backbone                                                 │
│   - Parameters           : 746,752                       │
│   - Trainable Parameters : 746,752                       │
│   - Estimated Size       : 3.00 MB                       │
│                                                          │
│ Classification Head                                      │
│   - Parameters           : 2,485,992                     │
│   - Trainable Parameters : 2,485,992                     │
│   - Estimated Size       : 9.94 MB                       │
│                                                          │
│ Total Model                                              │
│   - Parameters           : 3,232,744                     │
│   - Trainable Parameters : 3,232,744                     │
│   - Estimated Size       : 12.94 MB                      │
╰──────────────────────────────────────────────────────────╯
```

### Step 2.25 — `view_input_batches()`
This only runs with `diagnosis_modes.view_input_images: True`. It shows what the network will really receive: an augmented training image (Step 3.13) next to a validation image, in two windows titled `Training` and `Validation`, each captioned with its class (`Class : n04456115`). SPACE shows the next images and ENTER closes the viewer, after which `exit()` ends the program before any training, and no run folder was created (Step 2.6). It needs a display, so it has no console output to show here.

### Step 2.26 — `restore_last_model()`
This runs only when `resume_training` is `True`. It asks Step 2.27 for the last saved iteration. Without any checkpoint it returns 1 and the run starts from the random weights (or from the pretrained components of Step 2.19, which the message then names). Otherwise it loads that checkpoint (Step 2.28) and returns the iteration + 1, which becomes `start_iteration`: the training loop continues exactly where it stopped.

**Example output: a new run, then the run of Stage 3 built again after its 12 iterations**
```text
-----------------------------------------------------------
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.            -> start_iteration = 1
-----------------------------------------------------------

------------------------------------------------------------
We Loaded A Model That Was Previously Saved At Iteration 12           -> start_iteration = 13
------------------------------------------------------------
```

### Step 2.27 — `get_last_checkpoint_iteration()`
This reads the first line of the registry file `Weights/full-checkpoint`, which always names the last checkpoint saved (`save_model()` rewrites it, Step 3.6), and returns its iteration. It returns 0 when the file does not exist, i.e. for a run without any checkpoint. The standalone evaluation of Stage 5 uses it too.

**Example: `Weights/full-checkpoint` of the run of Stage 3, which saved checkpoints at iterations 3, 6, 9 and 12**
```text
model_checkpoint_path: "Iteration_12"          <- first line: the last checkpoint, read by this function
all_model_checkpoint_paths: "Iteration_3"
all_model_checkpoint_paths: "Iteration_6"
all_model_checkpoint_paths: "Iteration_9"
all_model_checkpoint_paths: "Iteration_12"     <- saved once, after the loop (Step 3.25)

get_last_checkpoint_iteration(weights_directory=...) -> 12
```

### Step 2.28 — `load_model()` → `load_checkpoint()`
`load_checkpoint()` reads `Weights/Iteration_N/model_00000N.pt` with `torch.load`, mapping every tensor to the current device. `load_model()` then puts `model_state` into the model and `optimizer_state` into the optimizer, which restores the weights and the moving averages of Adam. With the moving average of the weights on, `load_exponential_moving_average()` restores it from `exponential_moving_average_state`; a checkpoint saved without it (the average was just turned on) restarts the average from the loaded weights, with a yellow message. The checkpoint also stores the task, the run name, the MLflow run and the source runs of the transferred components, which tools such as the model transfer check.

**Example: the keys of a checkpoint of the run of Stage 3**
```text
['iteration', 'model_state', 'optimizer_state', 'format_version', 'task', 'run_name', 'mlflow_run_id',
 'pretrained_backbone', 'pretrained_classification_head']
iteration = 12    format_version = 2    task = 'classification'    mlflow_run_id = '9676967287064c1784b8a7c3bee0c1dc'
pretrained_backbone = None    pretrained_classification_head = None
```


---

## Stage 3 — Training Loop

> The examples below are real outputs from a run on CPU of `configurations/classification/configuration_synthetic_shapes.yaml` (9 shape classes) shortened to: 64 pixel images, batch size 8, 12 iterations, a checkpoint every 6 iterations (an intermediate one every 3), a console summary every 4 iterations, the validation every 10 iterations (`validation_iteration_ratio`), with the error images and the MLflow tracking turned on. Its run name is `CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9`. Step 3.9 uses real ImageNet images instead, because their sizes show the padding better.

### Step 3.1 — `trainer.run_training_loop()` → `run_training_iterations()`
This is the training itself. `run_training_loop()` runs `run_training_iterations()` and, whatever ends it, closes the MLflow run with its status: `FINISHED`, `KILLED` after a Ctrl+C, `FAILED` after too many errors (or an unexpected error, raised again once the run is closed). For every iteration from `start_iteration`, for `number_iterations` iterations, `run_training_iterations()` does in this order: save a checkpoint if due (3.6), evaluate it on the test split if due (3.7), fetch a batch (3.8), take one training step (3.10–3.22), run one validation batch if due (3.23), and print a summary if due (3.24). The last iteration saves nothing at its start: its checkpoint is the final one, saved after its training step (3.25), so that a checkpoint is never evaluated and then overwritten. An exception inside an iteration is printed and the iteration is skipped; more than 5 failures in a row, or Ctrl+C, end the loop. Whatever the reason, the loop ends with a last save.

**Example: what happens at each iteration of the run** (`weight_saving_iterations: 6`, `intermediate_weight_saving_iterations: 3`, `information_dump: 4`, `validation_iteration_ratio: 10`)

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | after the loop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Checkpoint saved | | | ✔ | | | ✔ | | | ✔ | | | | ✔ (iteration 12) |
| Test evaluation | | | | | | ✔ | | | | | | | ✔ (iteration 12) |
| Validation batch | | | | | | | | | | ✔ | | | |
| Console summary | | | | ✔ | | | | ✔ | | | | ✔ | |

### Step 3.2 — `setup_tensorboard_writers()`
This creates the TensorBoard writers right before the loop, not in the constructor, so that a trainer that never trains leaves no empty log file. There is one writer for the training curves and, if `validation_iteration_ratio > 0`, one for the validation curves, so that both appear as separate runs of the same charts. The `Test` writer is only created at the first evaluation (Step 4.18).

**Example: the folders and the curves they hold after the run**
```text
Tensorboard/Train/        Total Loss, Classification Loss, Accuracy, Learning Rate      (iterations 1 to 12)
Tensorboard/Validation/   Total Loss, Classification Loss, Accuracy                     (iteration 10)
Tensorboard/Test/         created by the first evaluation (Stage 4)
Tensorboard/Test_EMA/     created by the first evaluation of the moving average of the weights, when it is on
```

### Step 3.3 — `initialize_training_loop_state()`
This prepares what the loop body needs: an iterator over the training loader and one over the validation loader (the loop pulls batches by hand with `get_next_batch()`, see Step 3.8), a flag telling whether validation is enabled, the tracker dictionaries of Step 3.4 (the validation one is `None` without validation), and the counter of consecutive errors.

### Step 3.4 — `MetricTracker.create_trackers()`
This returns a dictionary with one sum per tracked metric (see Step 2.21), the number of batches added to it (`interval_batches`) and the start time of the interval. One is used for the training metrics and one for the validation metrics. `accumulate()` fills them batch by batch (Step 3.20), and at the end of the interval `average_trackers()` divides the sums by the batch count, `print_interval_summary()` prints them and the loop creates fresh ones (Step 3.24). The sequence tasks also count `interval_frames`.

**Example output**
```text
{'start_time': 1791532505.3603294, 'interval_batches': 0, 'total_loss': 0.0, 'classification_loss': 0.0, 'accuracy': 0.0}
```

### Step 3.5 — `start_experiment_tracking()`
With `ExperimentConfiguration.tracking.activated: True`, this creates the MLflow tracker of the run (`create_experiment_tracker()`), right after the loader iterators exist, so that the loader worker processes are forked before the tracker starts its thread. The tracker writes to the server of `tracking.tracking_uri`, or to the local database `mlflow.db` of the experiments root folder (its artifacts in `mlartifacts/` next to it), in the MLflow experiment named after the task unless `tracking.experiment_name` is set. It sends the parameters of the run (every configuration section flattened, plus the number of parameters, the split sizes, the iterations per epoch, the device and the library versions), the tags (task, run name, run folder, source runs of the transferred components, host, git commit) and the configuration file. Every call only puts an event in a queue: a background thread opens the MLflow run and writes in batches, retrying while MLflow is unavailable, so MLflow never slows down nor stops the training, and any MLflow error only prints one warning. The run is recorded in `mlflow_run.json` of the run folder, so that a resumed training reopens the same MLflow run. Without tracking, the tracker stays a `DisabledTracker`.

**Example: the line printed once the background thread opened the run** (here during the first evaluation, since creating the local database takes a few seconds), and `mlflow_run.json`
```text
MLflow Run Created In The Experiment 'classification' : 9676967287064c1784b8a7c3bee0c1dc (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)

{
    "tracking_uri": "sqlite:///.../runs/mlflow.db",
    "experiment_id": "1",
    "run_id": "9676967287064c1784b8a7c3bee0c1dc"
}
```

**Example: the MLflow run after the training**
```text
experiment: classification   run name: CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9   status: FINISHED
tags:       task, run_name, run_folder, pretrained_backbone: 'None', pretrained_classification_head: 'None', host,
            git_commit, git_dirty, best_iteration: '6'
parameters: 119, e.g. Backbone.convolutions.1: '[1, 16]', Backbone.convolutions.2: '[2, 32]', ...
metrics:    train/Total Loss, train/Classification Loss, train/Accuracy, train/throughput, train/interval_seconds,
            learning_rate                                         steps [4, 8, 12]     (Step 3.24)
            validation/Total Loss, validation/Classification Loss,
            validation/Accuracy                                   steps [12]           (the only interval with validation)
            test/Accuracy, test/Balanced Accuracy, test/Macro F1, test/Top-2 Accuracy,
            test/best_Accuracy, 27 test_class/<metric>/<class>    steps [6, 12]        (Step 4.19)
artifacts:  Results.txt, configuration/training_configuration.yaml, metric_names.json,
            evaluation/iteration_6/ and iteration_12/: class_metrics.csv, class_thresholds.json, confusion_matrix.png
```

### Step 3.6 — `save_model()` → `save_checkpoint()`
This runs at the start of an iteration that is a multiple of `weight_saving_iterations` (a checkpoint that will be evaluated, Step 3.7) or of `intermediate_weight_saving_iterations` (a checkpoint only), except the last iteration of the run. `save_checkpoint()` writes `Weights/Iteration_N/model_00000N.pt` with the model, the optimizer, the task, the run name, the MLflow run (`None` without tracking, or while the background thread is still opening the run) the source run of the transferred components and, when it is on, the moving average of the weights (`exponential_moving_average_state`: the averaged weights and the number of updates). It writes to a temporary file and renames it, so a checkpoint on disk is complete or absent, and it ignores Ctrl+C meanwhile. Then it records the iteration in `Weights/full-checkpoint`, the registry of Step 2.27.

**Good to know:** the checkpoint of iteration N is saved before the training step N, so it holds the weights after N−1 steps. The only exception is the last one, saved after the loop (Step 3.25).

**Example output and files**
```text
Checkpoint Folder :
 - file:///.../CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9/Weights/Iteration_3
 - Model Saved In : model_000003.pt

Weights/Iteration_3/model_000003.pt     5,579,851 bytes    mlflow_run_id None        <- the MLflow run was not open yet
Weights/Iteration_6/model_000006.pt     5,579,851 bytes    mlflow_run_id None
Weights/Iteration_9/model_000009.pt     5,579,915 bytes    mlflow_run_id '9676967287064c1784b8a7c3bee0c1dc'
Weights/Iteration_12/model_000012.pt    5,579,915 bytes    mlflow_run_id '9676967287064c1784b8a7c3bee0c1dc'
Weights/full-checkpoint
```

### Step 3.7 — `run_test_evaluation()`
At a checkpoint of `weight_saving_iterations`, right after the save, the checkpoint is evaluated on the whole test split: the progress bar, the metrics, the files and the report are described in Stage 4. It is skipped for the first iteration of a run (nothing was trained yet, or it is the checkpoint a resumed run just loaded and that was already evaluated). The evaluation puts the model in `eval()` mode, which is why Step 3.10 sets the training mode again before the next step.

### Step 3.8 — `get_next_batch()`
This takes the next batch from the loader iterator. At the end of an epoch the iterator raises `StopIteration` and is recreated, which reshuffles the samples, so the loop never ends because of the data. If the loader raises another exception (a corrupted image, for instance), it tries again, up to 10 times in a row, then exits. It returns the batch and the (possibly new) iterator. The loop then calls `MetricTracker.count_frames()`, which counts the real frames of a sequence batch for the frame throughput, and does nothing for images.

**Example: the first batch of the training loader** (batch size 8, 64 pixel images)
```text
images               [8, 3, 64, 64] torch.uint8     (BGR, values from 0 to 255)
labels               [8] torch.int64                [1, 7, 6, 8, 3, 5, 8, 2]
sample_identifiers   [8] torch.int64                [16905, 9751, 89490, 1234, 75284, 91572, 3544, 93323]
```

### Step 3.9 — `ClassificationDataset.__getitem__()` → `preprocess_image()`
These run inside the loader workers, in parallel with the training step, once per sample of the batch. `__getitem__()` reads the image file with OpenCV (BGR, `uint8`), calls `preprocess_image()` to get a square image, converts it to a `[3, H, W]` tensor and returns it with its class index and its identifier. `preprocess_image()` resizes the longest side to `image_size`; with `keep_ratio: True` the other side is padded with black bars (centred), otherwise the image is stretched to a square. The default collate function then stacks the samples of the batch.

**Example: three real ImageNet images with `image_size: 256` and `keep_ratio: True`**
```text
n01440764_10026  original (250, 250, 3) -> (256, 256, 3)   image rows 0 to 255    (square: no padding)
n01440764_10027  original (150, 200, 3) -> (256, 256, 3)   image rows 32 to 223   (resized to 192 x 256, 32 black rows above and below)
n01440764_10029  original (375, 500, 3) -> (256, 256, 3)   image rows 32 to 223
```

### Step 3.10 — `set_training_mode()`
This calls `model.train()` before every training step: dropout is active, and the batch normalization layers use the statistics of the current batch and update their running averages. It is needed at every iteration because the evaluation (Step 3.7) and the validation (Step 3.23) switch the model to `eval()`, where dropout is off and the batch normalization uses its running averages. The batch normalization of a frozen transferred component stays in `eval()` so that its statistics remain those of its source run.

### Step 3.11 — `run_model_iteration()`
One function runs a training batch and a validation batch, the only difference being `apply_augmentation`. It prepares the batch (3.12), augments it for training (3.13), runs the forward pass (3.14 to 3.18, in `bfloat16` if `mixed_precision` is on), sums every output whose key ends with `_loss` into the total loss, adds the batch metrics (3.19), then accumulates and logs the metrics (3.20). It returns the total loss, which the loop back-propagates for the training batch, and the model outputs.

**Example: what the first training batch of the run gave**
```text
total_loss = classification_loss = 2.2124      accuracy = 0.0
```

### Step 3.12 — `prepare_batch()`
This moves the batch to the device of the model and fixes the types: the images become `float32` (still BGR and in [0, 255]: the normalization is the first layer of the backbone, Step 3.15) and the labels become `int64` class indices (`float` multi-hot vectors in `multi_label` mode). The sample identifiers are kept for the evaluation.

**Example: before and after**
```text
images  [8, 3, 64, 64] torch.uint8   -> [8, 3, 64, 64] torch.float32 (cpu in this run, cuda on a GPU)
labels  [8]            torch.int64   -> [8]            torch.int64
```

### Step 3.13 — `augment_batch()` → `DataAugmenter.augment_batch()`
This is only called for training batches, so the model learns from varied images while the validation and the test stay on the original ones. Each image of the batch is drawn on its own: with probability `Data.data_augmentation.probability` it is augmented, otherwise it is left as it is. An augmented image gets the geometric types first (crop, left-right flip, up-down flip, rotations), then the photometric ones (brightness, contrast, saturation, CLAHE, sharpening, histogram equalizations, noise, grayscale, colour swap, blur), each type with its own probability. The labels do not change, since none of these changes the class. The result stays in [0, 255].

**Example: which images of the first batch were changed** (`probability: 0.4`, 8 images)
```text
changed per image: [False, False, True, True, True, True, False, False]      pixel range after: 0.0 to 252.0      labels unchanged
```

### Step 3.14 — `forward_pass()` → `ImageClassifier.forward()`
This calls the model with the images and the labels, so that it also returns the loss. The trainer only passes the batch: everything else happens inside the model, in the next four steps. For a test batch the trainer calls `predict_batch()` instead, which omits the labels (Step 4.4).

**Example: the dictionary returned**
```text
class_logits          [8, 9] torch.float32
class_probabilities   [8, 9] torch.float32
classification_loss   2.2124 (scalar torch.float32)
```

### Step 3.15 — `Backbone.forward()`
The images first go through the `InputNormalizer`, which maps the pixels to [-0.5, 0.5] (`input_normalization`) and/or standardizes each image (`per_sample_normalization`). Then they go through every `ConvolutionBlock`: the height and width are halved by each pooling while the channels are set by the block. It returns the final feature map and a dictionary of intermediate maps, which only the detector and the segmentation decoder use.

**Example: shapes for a batch of 8 images of 64 pixels** (6 blocks, the pooling of the last one removed)
```text
[8, 3, 64, 64] -> block 1 [8, 16, 32, 32] -> 2 [8, 32, 16, 16] -> 3 [8, 32, 8, 8] -> 4 [8, 64, 4, 4] -> 5 [8, 64, 2, 2] -> 6 [8, 128, 2, 2]
```

### Step 3.16 — `global_pooling` + `ClassificationHead.forward()`
The pooling turns the `[B, C, H, W]` feature map into `[B, F]` (Step 2.16). The head then applies its hidden layers and its last `Linear` layer: one raw score, the logit, per class. A logit is not a probability: it can be any real number.

**Example: shapes** (`global_pooling: "average"`, `fully_connected: [128]`)
```text
feature map [8, 128, 2, 2] -> average pooling [8, 128] -> hidden layer (128) -> [8, 128] -> classifier -> logits [8, 9]
```

### Step 3.17 — `convert_logits_to_probabilities()`
This turns the logits into probabilities. In `multi_class` mode, a softmax over the classes gives numbers between 0 and 1 that sum to 1 for each image: one class per image. In `multi_label` mode, an independent sigmoid per class lets an image have several classes at once. The probabilities are what Stage 4 stores and thresholds.

**Example: the first images of the batch**
```text
class_probabilities.sum(dim=1)[:3] = [0.9999998807907104, 1.0, 1.0]
class_logits.argmax(dim=-1)        = [2, 2, 2, 2, 2, 0, 2, 8]     <- predicted class of each of the 8 images
```

### Step 3.18 — `compute_classification_loss()`
This measures how wrong the logits are, as one number to minimize. In `multi_class` mode it is the cross entropy against the class indices, weighted by `class_weights` (Step 2.18) and softened by `label_smoothing`. In `multi_label` mode it is the binary cross entropy with logits against the multi-hot labels. A model that gives every class the same probability has a cross entropy of ln K, which is the starting point of a training.

**Example output** (9 classes, an untrained network)
```text
classification_loss = 2.2124        (ln 9 = 2.1972)
```

### Step 3.19 — `compute_batch_metrics()`
This computes the accuracy of the batch from the logits that were just computed, without a second forward pass and without gradients: the share of images whose highest logit is the true class (in `multi_label` mode, the share of class decisions at probability 0.5 that are right). It is a monitoring value only; the loss of Step 3.18 is what is optimized.

**Example: the batch of Step 3.8**
```text
labels      = [1, 7, 6, 8, 3, 5, 8, 2]
predictions = [2, 2, 2, 2, 2, 0, 2, 8]      no image is right
accuracy    = 0 / 8 = 0.0
```

### Step 3.20 — `MetricTracker.accumulate()` + `log_to_tensorboard()`
`accumulate()` adds every tracked metric of the batch to the sums of the interval and counts the batch in `interval_batches`; the outputs that are not tracked are ignored. The sums become means only at the end of the interval (Step 3.24), divided by the number of batches really added. `log_to_tensorboard()` writes the same metrics of the batch, at every iteration, under the name of Step 2.21, to the writer of the split (Train or Validation).

**Example: the console value is the mean of the TensorBoard points of the interval**
```text
Tensorboard/Train Accuracy at iterations 1 to 4:  0.0, 0.25, 0.125, 0.125      sum = 0.5      mean = 0.125
console at iteration 4:  Moving Average of Train Accuracy : 0.1250
Tensorboard/Train Accuracy at iterations 5 to 8:  0.0, 0.0, 0.125, 0.25        sum = 0.375    mean = 0.09375
console at iteration 8:  Moving Average of Train Accuracy : 0.0938
```

### Step 3.21 — `training_loss.backward()`
This is the back-propagation: PyTorch computes the gradient of the total loss with respect to every trainable parameter and stores it in `parameter.grad`. It only runs for training batches, after `optimizer.zero_grad()` has erased the gradients of the previous step. The validation and test batches never call it, and run without gradients.

**Example: the norm of all the gradients together, at the first three steps of the run**
```text
iteration 1: gradient norm = 3.09      iteration 2: 2.36      iteration 3: 2.09
```

### Step 3.22 — `step_optimizer()` → `compute_learning_rate_factor()`
This applies the gradients. If `Optimization.gradient_clipping_norm` is set, the gradients are first scaled down so that their total norm does not exceed it. Then the learning rate of every parameter group is set to `base_learning_rate × factor(iteration)` and `optimizer.step()` updates the weights, after which the moving average of the weights, when on, moves towards them: the parameters and the batch normalization running statistics with `min(decay, (1 + n) / (10 + n))` at its n-th update, the other buffers copied. The factor only depends on the iteration, so a resumed run continues the schedule without saving any scheduler state: a linear warm-up over `warmup_iterations`, then a constant rate or a cosine decay down to `minimum_learning_rate_ratio` over `decay_iterations`. The rate of the first group is written to TensorBoard.

**Example: this run** (`constant`, `learning_rate: 1.0e-3`): the first Adam step moves each weight by about the learning rate
```text
Tensorboard/Train Learning Rate at iterations 1 to 12 = 0.001
largest change of a weight of the classifier after the first step = 0.0010000
```

**Example: the factor of a cosine schedule** (`warmup_iterations: 1000`, `decay_iterations: 100000`, `minimum_learning_rate_ratio: 0.01`)

| Iteration | 1 | 500 | 1000 | 1001 | 51000 | 101000 | 200000 |
|---|---|---|---|---|---|---|---|
| Factor | 0.001 | 0.5 | 1.0 | 1.0 | 0.505 | 0.01 | 0.01 |

### Step 3.23 — validation pass: `run_model_iteration()` without augmentation
Every `validation_iteration_ratio` iterations (10 here), the loop evaluates one batch of the validation split to follow the generalization during the training: `model.eval()`, no gradients, the next validation batch, and the same `run_model_iteration()` with `apply_augmentation=False`. Its metrics go to the validation trackers and to `Tensorboard/Validation`. It is only one batch, not the whole split, so it is cheap; the whole-split evaluation is the test evaluation of Stage 4.

**Example: iteration 10 of the run**
```text
Tensorboard/Validation at iteration 10: Total Loss 2.1964, Classification Loss 2.1964, Accuracy 0.125
console at iteration 12: Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
```

### Step 3.24 — `average_trackers()` + `log_interval_metrics()` + `print_interval_summary()`
Every `information_dump` iterations, the loop divides the training and the validation sums by the number of batches each really ran (`interval_batches`). The training interval usually holds `information_dump` batches, but the first interval of a run resumed between two summaries holds fewer, and the number of validation batches depends on the ratio. `log_interval_metrics()` sends the averages to MLflow (`train/<name>` and, when a validation batch ran, `validation/<name>`), with the throughput, the duration of the interval and the learning rate. `print_interval_summary()` prints the summary, and the loop starts fresh trackers. The summary gives the throughput and the moving average of every metric for the training and, if a validation batch ran in the interval, for the validation. After each value comes its change since the previous summary of the run, in green when it improved (a loss decreasing, an accuracy increasing) and in red when it got worse, and `Best` when it is the best value of the run so far. The duration includes everything between the two summaries, evaluations and checkpoints included.

**Example: the sums and the averages of the first interval**
```text
before: {'interval_batches': 4, 'total_loss': 8.8198, 'classification_loss': 8.8198, 'accuracy': 0.5}
after:  {'interval_batches': 4, 'total_loss': 2.205,  'classification_loss': 2.205,  'accuracy': 0.125}
MLflow, step 4: {'train/Total Loss': 2.205, 'train/Classification Loss': 2.205, 'train/Accuracy': 0.125,
                 'train/throughput': 8.5719, 'train/interval_seconds': 3.7331, 'learning_rate': 0.001}
```

**Example output: iterations 4, 8 and 12** (the evaluation of the checkpoint of iteration 6 ran inside the interval of iteration 8, which is why it took 48 s)
```text
========================================================================================================================
|     Iteration: 4     |      Total Images: 32     |    Interval Images: 32     |  Throughput: 8.6 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   2.2050
Moving Average of Train Classification Loss            :   2.2050
Moving Average of Train Accuracy                       :   0.1250
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:03.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
========================================================================================================================
...
========================================================================================================================
|     Iteration: 8     |      Total Images: 64     |    Interval Images: 32     |  Throughput: 0.7 Images Per Second   |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   2.1895  -0.0154 Best        <- green: the loss decreased
Moving Average of Train Classification Loss            :   2.1895  -0.0154 Best
Moving Average of Train Accuracy                       :   0.0938  -0.0312             <- red: the accuracy decreased
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:48.
Validation Iterations In This Interval: 0 (Ratio: Every 10 Training Iterations).
========================================================================================================================
...
========================================================================================================================
|    Iteration: 12     |      Total Images: 96     |    Interval Images: 32     |  Throughput: 31.8 Images Per Second  |
------------------------------------------------------------------------------------------------------------------------
Moving Average of Train Total Loss                     :   2.2103  +0.0208             <- red: the loss increased
Moving Average of Valid Total Loss                     :   2.1964                      <- first Valid value: no change
Moving Average of Train Classification Loss            :   2.2103  +0.0208
Moving Average of Valid Classification Loss            :   2.1964
Moving Average of Train Accuracy                       :   0.0938  +0.0000             <- line colour: no change
Moving Average of Valid Accuracy                       :   0.1250
------------------------------------------------------------------------------------------------------------------------
Interval Of 4 Iterations Took 00:00:01.
Validation Iterations In This Interval: 1 (Ratio: Every 10 Training Iterations).
========================================================================================================================
```

### Step 3.25 — end of the loop: final `save_model()`, final evaluation, closing
After the last iteration, after a Ctrl+C, or after too many consecutive errors, the loop saves a last checkpoint with the iteration it reached and prints a confirmation. When the run completed and its last iteration is a multiple of `weight_saving_iterations` (12 here), this final checkpoint, which holds the weights after the last training step, is evaluated (Stage 4); a stopped run is not evaluated. The loop then closes the TensorBoard writers, and `run_training_loop()` sends `Results.txt` to MLflow and closes the MLflow run with its status (`FINISHED` here). An exception inside an iteration is printed with a counter and the loop moves on to the next iteration.

**Example output**
```text
Checkpoint Folder :
 - file:///.../CLS_IN-KAR-64_..._MC9/Weights/Iteration_12
 - Model Saved In : model_000012.pt
------------------------------------------------------------
Model successfully saved at iteration 12. Exiting Training.
------------------------------------------------------------

Iteration 12 · 5000 Test Images · 2026-10-09 09:53 ────────────────────────     <- the evaluation of the final checkpoint
...

Error encountered during iteration 7. Continuing... (Consecutive Errors: 1/5)      <- the form of the message of a failed iteration
```


---

## Stage 4 — Test Evaluation of a Checkpoint

> The examples below are real outputs of the evaluation of the checkpoints of iterations 6 and 12 of the run of Stage 3, on its whole test split (5,000 images, 9 classes), with `per_class_report: True`, `number_sample_images: 8`, `save_predictions: True`, `error_dumps` activated and the MLflow tracking on. The network was trained for 5 and 12 steps only, so it predicts the same class for every image: the metrics are close to chance, but every file and every line of the report is the real one.

### Step 4.1 — `run_test_evaluation()`
This function scores the current weights on the whole test split, which no training step ever sees, and reports the result. The training loop calls it at every `weight_saving_iterations` checkpoint (Steps 3.7 and 3.25), and Stage 5 calls it for every checkpoint it evaluates. It puts the model in `eval()` mode, creates the accumulator (4.2), runs the test loader without gradients within the `evaluation_context()` of the task (nothing for classification), then computes the results (4.8) and reports them (4.18). It also makes sure that every test image is counted exactly once: loaders that fill their last batch with images already given (the DALI loaders of the detector) would otherwise count them twice, so it keeps only the sample identifiers it has not seen yet. It adds the sample images (4.7) to the results, with a note and their folder, and returns the results, which Stage 5 uses to compare the checkpoints.

With the moving average of the weights on, `run_test_evaluation()` evaluates the trained weights first (`evaluate_test_split()`, everything above), then the averaged weights of the same checkpoint: `use_averaged_weights()` puts the averaged model in place of `self.model` and gives it back afterwards, even on an error. That second evaluation writes its files in `Evaluation/Iteration_N/EMA/`, its scalars and sample images in `Tensorboard/Test_EMA` with the same tags, so that both curves share the same charts, its report under the header `Iteration N · <evaluated samples> · EMA Weights · <date and time>`, and its MLflow metrics as `test_ema/` and `test_ema_class/`. The returned results, `test/best_<headline metric>` and the comparison of Stage 5 stay those of the trained weights.

**Example output: the progress bar of the evaluation** (5,000 images, batch size 8, CPU)
```text
Test Evaluation Iteration 6: 100%|██████████| 625/625 [00:47<00:00, 13.15it/s]
```

**Good to know:** the weights evaluated at iteration N are the ones saved at the start of iteration N (Step 3.6), i.e. after N−1 training steps, except the final checkpoint (iteration 12 here), saved and evaluated after its training step.

### Step 4.2 — `create_evaluation_accumulator()`
This creates the empty storage filled during the test loop. The predictions of all the batches are kept so that the metrics can be computed on the whole split at once: three lists (class probabilities, labels and sample identifiers, one tensor per batch), the folder where the error images will be written and an empty counter of error images per pair of classes.

**Example output**
```text
{'probabilities': [], 'labels': [], 'sample_identifiers': [],
 'errors_folder': PosixPath('.../Evaluation/Iteration_6/Errors'), 'error_counts': {}}
```

### Step 4.3 — `prepare_batch()`
This is the same function as in training (Step 3.12): the images go to the device as `float32` (still in [0, 255] BGR) and the labels as `int64` class indices. Test images are never augmented: the model must be scored on the images as a deployment would give them.

### Step 4.4 — `predict_batch()` → `ImageClassifier.forward()` without labels
This runs the model as in deployment: `model(input_tensor=images)`, without labels, so no loss is computed. The model is in `eval()` mode (dropout off, batch normalization using its running averages) and no gradient is kept (`torch.no_grad()`). It returns the logits and the probabilities.

**Example output** (batch of 8 images, 9 classes)
```text
class_logits          [8, 9] torch.float32
class_probabilities   [8, 9] torch.float32      (no 'classification_loss' key)
```

### Step 4.5 — `accumulate_evaluation_batch()`
This keeps the useful part of the batch on the CPU. The probabilities (as `float64`), the labels and the sample identifiers of the kept samples (the `sample_indices` of Step 4.1) are appended to the accumulator lists, so the GPU memory is released after every batch. The identifiers let the predictions file and the error images name the image of each prediction. If `error_dumps.activated` is true, it also calls `dump_errors()` while the images are still in the batch.

**Example: the accumulator at the end of the loop**
```text
'probabilities':      625 tensors, the first [8, 9] torch.float64      5,000 images / batch of 8
'labels':             625 tensors, the first [8] torch.int64
'sample_identifiers': 625 tensors, the first [8] torch.int64
'error_counts':       8 pairs, square-TO-triangle, circle-TO-triangle, ..., 20 images each
```

### Step 4.6 — `dump_errors()` → `reserve_error_image()` + `build_input_image()`
For every wrong prediction of the batch (predicted class ≠ true class), `reserve_error_image()` counts one more image for the pair `<TRUE>-TO-<PREDICTED>` and refuses it once the pair holds `maximum_per_class_pair` images, so the first ones met are kept and the disk does not fill up. `build_input_image()` rebuilds the image the network saw (BGR, `uint8`), a black bar above it is captioned with both classes and the probability of the predicted class, and the PNG is written, named after the sample and that score.

**Example: the error images of the checkpoint of iteration 6** (every image was predicted as `triangle`)
```text
Evaluation/Iteration_6/Errors/
├── circle-TO-triangle/       20 images, e.g. test_000233_0.116.png    caption: "circle -> triangle 0.116"
├── hexagon-TO-triangle/      20 images
├── octagon-TO-triangle/      20 images
...                           8 pairs x 20 images = 160 images, the number in the report line "Error Images : 160 in 8 class pairs"
```

### Step 4.7 — `save_sample_images()` → `build_sample_image()`
For every fixed sample image of Step 2.11 found in the test batch, the hook `build_sample_image()` draws the network input (Step 4.6) with a caption bar: its true class, its predicted class and the probability of the predicted class, in green when the prediction is right and in red when it is wrong (in `multi_label` mode, the true classes and the predicted classes with their probabilities). The caption bar is wider than a small image, so the image is padded on the right. `save_sample_images()` writes the image in `Evaluation/Iteration_N/Samples/<sample name>.png` and keeps it, in RGB, under the TensorBoard tag `Samples/<sample name>`. The same images are drawn at every evaluation, so the TensorBoard image slider shows how the prediction of each one changes over the training.

**Example: the 8 sample images of the checkpoint of iteration 12** (one image per class for the first 8 classes)
```text
Evaluation/Iteration_12/Samples/
test_000000.png  test_000001.png  test_000002.png  test_000003.png  test_000006.png  test_000010.png  test_000016.png  test_000017.png

test_000002.png: 152 x 88 pixels, the red caption "rectangle -> triangle 0.117" on a black bar above a 64 x 64 pink image holding a small red rectangle
note of the report: "Sample Images : 8 fixed test images"
```

### Step 4.8 — `compute_evaluation_results()`
Once the loop is done, this function concatenates the accumulated batches, computes every metric, writes the evaluation files (Steps 4.9 to 4.16) and organizes what must be displayed (Step 4.17). The notes are the extra notes of the task (`describe_extra_notes()`, none for classification) followed by the error images line. It does not display anything: it returns a dictionary, the `EvaluationResults`, that `report_evaluation()` shows (Step 4.18) and that Stage 5 keeps to compare the checkpoints. The files are written in `Evaluation/Iteration_N/` of the run, whatever `per_class_report`, except the confusion matrix image, which needs it.

**Example: the headline part of the results of iteration 6** (after Step 4.1 added the sample images)
```text
evaluated_samples = '5000 Test Images'
metrics           = {'Accuracy': 0.1096, 'Balanced Accuracy': 0.1111, 'Macro F1': 0.0219, 'Top-2 Accuracy': 0.2244}
notes             = ['Error Images : 160 in 8 class pairs, at most 20 per pair', 'Sample Images : 8 fixed test images']
headline_metric   = 'Accuracy'        (used by Stage 5 and MLflow to choose the best checkpoint)
files             = Confusion Matrix Table, Class Thresholds, Class Metrics, Confusion Matrix Image, Predictions,
                    Error Images, Sample Images
images            = Confusion Matrix        sample_images = Samples/test_000010, Samples/test_000017, ... (8)
class_scalars     = 27: Precision/square, Recall/square, F1/square, Precision/circle, ...
tables            = one table: title, columns, rows, sort_column, per_class, legend
```

### Step 4.9 — `compute_confusion_matrix()`
This counts, for every pair (true class, predicted class), the number of test images. The result is a `[K, K]` table (K = number of classes): the row is the true class, the column the predicted class, so the diagonal holds the correct predictions. It is computed with a single `bincount` over `true × K + predicted`. It is used in multi-class mode only: in multi-label mode an image has several classes, so no single matrix exists.

**Example output** (first rows of the matrix of iteration 6: every image was predicted as class 2, `triangle`)
```text
tensor([[  0,   0, 579,   0,   0,   0,   0,   0,   0],      <- the 579 squares, all predicted as triangle
        [  0,   0, 544,   0,   0,   0,   0,   0,   0],      <- the 544 circles
        [  0,   0, 548,   0,   0,   0,   0,   0,   0],      <- the 548 triangles, all right
        ...
```

### Step 4.10 — `compute_metrics_from_confusion_matrix()` → `compute_precision_recall_f1()`
All the multi-class metrics come from the matrix. For a class: the **support** is its row sum (its test images), the **precision** is the diagonal divided by its column sum (how many images predicted as the class are right), the **recall** is the diagonal divided by its row sum (how many images of the class were found) and the **F1** is their harmonic mean. A class that is never predicted gets a precision of 0. The **accuracy** is the diagonal sum over all the images, the **balanced accuracy** is the mean recall over the classes present in the test set, and the **macro F1** the mean F1 over the same classes: they do not depend on how many images each class has.

**Example: how the numbers of iteration 6 are obtained**
```text
accuracy          = 548 / 5000                           = 0.1096   (only the triangle images are right)
triangle          precision = 548 / 5000 = 0.1096        recall = 548 / 548 = 1.0      F1 = 2 x 0.1096 x 1.0 / (0.1096 + 1.0) = 0.1975
the 8 other classes   precision = 0 (never predicted)    recall = 0 / support = 0.0    F1 = 0.0
balanced accuracy = (8 x 0.0 + 1.0) / 9                  = 0.1111
macro F1          = (8 x 0.0 + 0.1975) / 9               = 0.0219
```

### Step 4.11 — `compute_top_k_accuracy()`
This is the share of test images whose true class is among the `top_k` classes with the highest probability (`evaluation.top_k`, 2 here, 5 for ImageNet). It is more forgiving than the accuracy and tells whether the right class is at least close. With `top_k` larger than the number of classes it is clipped, and the result is 1.

**Example output** (`top_k: 2`)
```text
Top-2 Accuracy = 0.2244 at iteration 6, 0.2276 at iteration 12      (about 2 / 9 = 0.222: chance level for 9 classes)
```

### Step 4.12 — `save_confusion_matrix_table()`
This writes the matrix of Step 4.9 as `confusion_matrix.csv`, with the class names as row and column headers, to read the exact counts or open them in a spreadsheet. It is written at every evaluation in multi-class mode, whatever `per_class_report`.

**Example: `Evaluation/Iteration_6/confusion_matrix.csv`**
```text
true \ predicted,square,circle,triangle,rectangle,pentagon,hexagon,plus_sign,star,octagon
square,0,0,579,0,0,0,0,0,0
circle,0,0,544,0,0,0,0,0,0
triangle,0,0,548,0,0,0,0,0,0
rectangle,0,0,537,0,0,0,0,0,0
pentagon,0,0,557,0,0,0,0,0,0
hexagon,0,0,574,0,0,0,0,0,0
plus_sign,0,0,545,0,0,0,0,0,0
star,0,0,526,0,0,0,0,0,0
octagon,0,0,590,0,0,0,0,0,0
```

### Step 4.13 — `find_class_thresholds()`
A deployment can ask for a prediction to be trusted only above a probability threshold. For every class, treated one against the rest, this function sorts the test images by decreasing probability of the class and looks for the lowest threshold where the precision stays at or above `target_precision` (0.99 here), together with the recall kept at this threshold. If no threshold reaches the target precision, or the class has no sample, the result is `null`. The thresholds are saved in `class_thresholds.json`, for the deployment of the checkpoint.

**Example: `Evaluation/Iteration_6/class_thresholds.json`** (only `hexagon` reaches it, with a single image)
```text
{
    "target_precision": 0.99,
    "classes": {
        "square": null,
        "circle": null,
        "triangle": null,
        "rectangle": null,
        "pentagon": null,
        "hexagon": {
            "threshold": 0.11630892753601074,
            "recall": 0.0017421602787456446          <- 1 of the 574 hexagons
        },
        "plus_sign": null,
        ...
```

### Step 4.14 — `build_class_metrics_rows()` + `write_class_metrics_file()`
`build_class_metrics_rows()` merges, for every class, its support, precision, recall, F1 and its threshold and recall at the target precision into one row, in class index order. `write_class_metrics_file()` saves them as `class_metrics.csv`, with 6 decimals and an empty cell for a missing value. It is written at every evaluation, whatever `per_class_report`, so that the details of all the classes (a 1000 class run included) stay on the disk even when they are not displayed.

**Example: `Evaluation/Iteration_6/class_metrics.csv`**
```text
class_index,class,samples,precision,recall,f1,threshold_at_target_precision,recall_at_target_precision
0,square,579,0.000000,0.000000,0.000000,,
1,circle,544,0.000000,0.000000,0.000000,,
2,triangle,548,0.109600,1.000000,0.197549,,
3,rectangle,537,0.000000,0.000000,0.000000,,
4,pentagon,557,0.000000,0.000000,0.000000,,
5,hexagon,574,0.000000,0.000000,0.000000,0.116309,0.001742
...
```

### Step 4.15 — `save_confusion_matrix_image()`   (only with `per_class_report`)
This draws the matrix with matplotlib and saves it as `confusion_matrix.png`; the same image array is returned to be logged to TensorBoard (Step 4.18). Rows are the true classes and columns the predicted ones. Each cell shows its count and its share of the true class, with a colour from white (0%) to dark blue (100%); a column on the right gives the recall of each class, a row at the bottom the precision of each class (grey for a class that is never predicted), and the corner the accuracy. The numbers and the class names disappear for many classes (counts and shares up to 20 classes, counts up to 40, names up to 60), which is why it depends on `per_class_report`.

**Example: `Evaluation/Iteration_12/confusion_matrix.png`** (1192 x 1155 pixels)
```text
title:    Confusion Matrix · Iteration 12 · Accuracy 0.1096 · Balanced Accuracy 0.1111
matrix:   9 x 9 cells, rows = true class, columns = predicted class
          column triangle: dark blue cells "579 / 100.0%", "544 / 100.0%", ... (every image was predicted as triangle)
          diagonal cells of the other classes: "0 / 0.0%" in bold; the other zeros are left blank
right:    Recall column: 0.0% for the 8 other classes, 100.0% for triangle
bottom:   Precision row: grey for the 8 classes that are never predicted, 11.0% for triangle
corner:   Accuracy 11.0%
```

### Step 4.16 — `write_predictions_file()`   (only with `save_predictions`)
This writes `predictions.csv`, one row per test image: its file name (`describe_sample()` of the dataset), its true and predicted classes, whether the prediction is right and the probability of every class. The errors and the thresholds can then be studied again without running the model. The file grows with the number of test images times the number of classes, which is why it is optional.

**Example: `Evaluation/Iteration_6/predictions.csv`** (5,000 rows, 585,792 bytes; first rows, columns shortened)
```text
sample,true_class,predicted_class,correct,probability_square,probability_circle,probability_triangle, ... ,probability_octagon
test_000612.png,triangle,triangle,1,0.111758,0.105057,0.116420, ... ,0.115801
test_000662.png,plus_sign,triangle,0,0.111758,0.105057,0.116420, ... ,0.115801
```

### Step 4.17 — `build_class_metrics_table()` + `build_class_metric_scalars()`
These prepare what is displayed, without displaying it. `build_class_metrics_table()` returns the per-class table: the columns, one row per class, the sort column (F1, worst first, ties in class order, a class without test sample last and shown with a dash), and the legend. `build_class_metric_scalars()` returns one TensorBoard scalar per class and metric. Both are marked as per-class outputs: the reporter shows them only with `per_class_report`.

**Example: the per-class scalars of the 9 classes**
```text
Precision/square, Recall/square, F1/square, Precision/circle, ... , F1/octagon       27 scalars (9 classes x 3 metrics)
```

### Step 4.18 — `report_evaluation()` → `report_evaluation_results()`
This shows the results in four places. The scalars (the global metrics, and with `per_class_report` the per-class scalars) are written to `Tensorboard/Test` at the iteration of the checkpoint, with the sample images and, with `per_class_report`, the confusion matrix image. The report block (header, global metrics, notes and tables) is appended as plain text to `Results.txt`, which accumulates one block per evaluation. The same block is printed in colour in the console, followed by the links of the files written by the evaluation. With `per_class_report: False`, the block stops after the global metrics and the notes, and the per-class scalars, the table and the image are left out; `class_metrics.csv`, `class_thresholds.json` and `confusion_matrix.csv` are still written. `report_evaluation()` then sends the evaluation to MLflow (Step 4.19).

**Example output: the block of iteration 6** (the same text, without the links, is appended to `Results.txt`)
```text
Iteration 6 · 5000 Test Images · 2026-10-09 09:53 ──────────────────────────────────────────────────────────────────
  Accuracy   0.1096   Balanced Accuracy   0.1111
  Macro F1   0.0219   Top-2 Accuracy      0.2244
  Error Images : 160 in 8 class pairs, at most 20 per pair
  Sample Images : 8 fixed test images

  Per Class Metrics
                                                       Threshold    Recall
   Class       Samples   Precision   Recall       F1     @ P0.99   @ P0.99
  ─────────────────────────────────────────────────────────────────────────
   square          579      0.0000   0.0000   0.0000           —         —
   circle          544      0.0000   0.0000   0.0000           —         —
   rectangle       537      0.0000   0.0000   0.0000           —         —
   pentagon        557      0.0000   0.0000   0.0000           —         —
   hexagon         574      0.0000   0.0000   0.0000      0.1163    0.0017
   plus_sign       545      0.0000   0.0000   0.0000           —         —
   star            526      0.0000   0.0000   0.0000           —         —
   octagon         590      0.0000   0.0000   0.0000           —         —
   triangle        548      0.1096   1.0000   0.1975           —         —
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 0.99 is never reached.
  - Results : file:///.../Results.txt
  - Confusion Matrix Table : file:///.../Evaluation/Iteration_6/confusion_matrix.csv
  - Class Thresholds : file:///.../Evaluation/Iteration_6/class_thresholds.json
  - Class Metrics : file:///.../Evaluation/Iteration_6/class_metrics.csv
  - Confusion Matrix Image : file:///.../Evaluation/Iteration_6/confusion_matrix.png
  - Predictions : file:///.../Evaluation/Iteration_6/predictions.csv
  - Error Images : file:///.../Evaluation/Iteration_6/Errors
  - Sample Images : file:///.../Evaluation/Iteration_6/Samples
```

**Example: what the run folder holds after the two evaluations, and the Test curves**
```text
Results.txt                                   one block per evaluation (iterations 6 and 12)
Evaluation/Iteration_6/                       Errors/  Samples/  class_metrics.csv  class_thresholds.json  confusion_matrix.csv
                                              confusion_matrix.png  predictions.csv
Evaluation/Iteration_12/                      the same files for the checkpoint of iteration 12

Tensorboard/Test: 31 scalars (4 global + 27 per class), 9 images
    scalar Accuracy          6: 0.1096, 12: 0.1096
    scalar Top-2 Accuracy    6: 0.2244, 12: 0.2276
    scalar Precision/square  6: 0.0000, 12: 0.0000
    ...
    image  Samples/test_000010 ... Samples/test_000006    steps [6, 12]     (the 8 sample images)
    image  Confusion Matrix                               steps [6, 12]
```

### Step 4.19 — `track_evaluation()`
With the MLflow tracking on, this sends the evaluation to the MLflow run at the iteration of the checkpoint: the global scalars as `test/<name>`, the per-class scalars as `test_class/<name>` only with `per_class_report` (they would be thousands for 1000 classes), the best value so far of the headline metric as `test/best_<metric>` with the tag `best_iteration` (for the averaged weights `test_ema/best_<metric>` with the tag `best_iteration_test_ema`), and the small evaluation files (`confusion_matrix.png`, `class_metrics.csv`, `class_thresholds.json`) under `evaluation/iteration_<n>/`. The predictions, the error images and the sample images grow with the test split, so they stay in the run folder only. Without tracking it does nothing.

**Example: what MLflow received at iteration 6**
```text
step 6: {'test/Accuracy': 0.1096, 'test/Balanced Accuracy': 0.1111, 'test/Macro F1': 0.0219, 'test/Top-2 Accuracy': 0.2244,
         'test_class/Precision/square': 0.0, 'test_class/Recall/square': 0.0, ... (27 values)}
test/best_Accuracy: step 6 -> 0.1096, step 12 -> 0.1096      tag best_iteration = '6' (iteration 12 is not better)
artifacts: evaluation/iteration_6/class_metrics.csv, class_thresholds.json, confusion_matrix.png
```


---

## Stage 5 — Standalone Checkpoint Evaluation (Optional)

> The examples below are real outputs of this script logic run on the run of Stage 3 (checkpoints saved at iterations 3, 6, 9 and 12), with `iteration = "all"` and, to keep it short, a test folder of 300 images taken from the test split (`test_data_folder`), instead of the 5,000 images of Stage 4.

### Step 5.1 — `scripts/evaluation/evaluate_checkpoint.py`
This separate script evaluates checkpoints of a run that has already been trained (or is still training) without training anything: to compare several checkpoints, to pick the one to deploy, or to score a run on another labelled split. It is launched alone (`python scripts/evaluation/evaluate_checkpoint.py`) and has no command line: its settings are the variables at the top of its `__main__` block. The run folder is only read; everything the evaluation writes goes to a separate folder.

**Example: the settings of the script** (`run_name` of the file names an ImageNet run, change it for another run)

| Variable | Role |
|---|---|
| `run_name` | the trained run, `experiments/<run_name>`, with its `Weights/` and its `training_configuration.yaml` |
| `iteration` | `"all"` (the default) for every checkpoint of the registry, the last saved first; `None` for the last checkpoint; an integer for one checkpoint |
| `test_data_folder` | another labelled split to evaluate instead of the test split of the run, `None` to keep the test split of the run |
| `batch_size` | batch size of the evaluation, `128` by default (no gradient is kept, so it can be larger than the training one), `None` to keep the one of the run |
| `track_evaluation` | `True` or `False` to track the evaluation in an MLflow run of its own, `None` to keep `tracking.activated` of the run |
| `output_folder` | root of the evaluations, `experiments/evaluations` by default, kept apart so the trained run is untouched |

### Step 5.2 — `load_experiment_configuration()` + `get_trainer_class()` + trainer construction (same as Stages 1–2)
The script loads `training_configuration.yaml` of the trained run, the copy that the training saved (Step 2.6), through the same loader as Stage 1 (conversion, label dictionary, checks). It then points `experiment_folder` to the output folder, sets `test_data_folder`, `batch_size` and `tracking.activated` when they are given, and calls `disable_pretrained_settings()`: the weights come from the evaluated checkpoint, so the source runs the training started from are neither loaded nor needed anymore. It builds the same trainer as for a training (Stage 2) with the same model architecture and the same sample images, which creates a run folder with the same run name inside the output folder. The trainer keeps a `DisabledTracker` until Step 5.4.

**Example output:** the construction prints the launch panel of Step 2.22 with the output folder. The saved configuration has `resume_training: True`, but the new run folder holds no weights, so no model is loaded here: the weights are loaded by Step 5.5.
```text
resume_training in the saved configuration: True
tracking in the saved configuration: {'activated': True, 'tracking_uri': None, 'experiment_name': None, 'log_artifacts': True}
- Run Name                : CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9
- Experiment Folder       : file:///.../runs/evaluations/CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9
...
- Test Data Folder        : file:///.../test_300 (300 images, 9 categories)
- Iterations Per Epoch    : 782
- Batch Size              : 128
...
- MLflow Tracking         : sqlite:///.../runs/evaluations/mlflow.db
-----------------------------------------------------------
No Initiation Model Weights Will Be Used...
We Generate A New Model That Will Be Trained From Scratch.
-----------------------------------------------------------
```
The `MLflow Tracking` line of the panel names the database of the output folder, but the MLflow run of the evaluation is written to the database of the trained run (Step 5.4).

### Step 5.3 — `get_checkpoint_iterations()` / `get_last_checkpoint_iteration()`
They read the registry `Weights/full-checkpoint` of the trained run (Step 2.27) to decide which iterations are evaluated: `get_checkpoint_iterations()` lists every recorded iteration, intermediate checkpoints included, and the script reverses the list (`iteration = "all"`), so that the most trained weights are evaluated first; `get_last_checkpoint_iteration()` gives the last one (`iteration = None`). An integer iteration is used as given.

**Example output** (registry of the run of Stage 3)
```text
get_checkpoint_iterations(weights_directory=...)      -> [3, 6, 9, 12]
get_last_checkpoint_iteration(weights_directory=...)  -> 12
iteration = "all"  -> evaluated_iterations = [12, 9, 6, 3]
```

### Step 5.4 — `evaluate_checkpoints()` → `start_evaluation_tracking()`
This function wraps the evaluations in an MLflow run of their own. `start_evaluation_tracking()` creates it when the tracking is on: a new run named `<run name>-evaluation`, in the database (or server) and the MLflow experiment of the trained run, with the parameters of the trainer and tags naming the evaluated run folder, the evaluated split, the evaluated iterations and, through `mlflow_run.json` of the trained run, its MLflow run (`evaluation_of`). The MLflow run of the training is only read, so that an evaluation on another split never mixes into its curves. `evaluate_checkpoints()` then runs Step 5.5, sends `evaluation_summary.csv` to the MLflow run, and closes it with its status, whatever ends the evaluation.

**Example: the MLflow run of the evaluation**
```text
MLflow Run Created In The Experiment 'classification' : 6b1db68d415d41d692a29f7b87fd4b69 (Browse : mlflow ui --backend-store-uri sqlite:///.../runs/mlflow.db)

run name: CLS_IN-KAR-64_SE-RES-BN-B1x16.2x32.2x32.2x64.2x64.2x128_HAVG.128_MC9-evaluation   status: FINISHED
tags:     evaluated_experiment_folder, test_data_folder: '.../test_300', evaluated_iterations: '12, 9, 6, 3',
          evaluation_of: '9676967287064c1784b8a7c3bee0c1dc', best_iteration: '9', ...
metrics:  test/Accuracy, test/Balanced Accuracy, test/Macro F1, test/Top-2 Accuracy, test/best_Accuracy,
          27 test_class/<metric>/<class>                                             steps [12, 9, 6, 3]
artifacts: evaluation/iteration_12/, iteration_9/, iteration_6/, iteration_3/ (3 files each), evaluation_summary.csv,
           metric_names.json
```
`test/best_Accuracy` is the best value of the checkpoints evaluated so far, in the order of the evaluation: 0.1000 at step 12, then 0.1133 at steps 9, 6 and 3.

### Step 5.5 — `summarise_checkpoint_evaluations()`
For each iteration, in order, this function loads the weights of `Weights/Iteration_N/model_00000N.pt` of the trained run into the model of the trainer (a missing file is skipped with a warning), evaluates it (Step 5.6) and keeps the scalars of the results. After the last checkpoint, it writes `Evaluation/evaluation_summary.csv`, one row per iteration with one column per scalar, and prints the iteration with the best headline metric (the accuracy for the classification, the higher the better). The per-class columns are only in the summary when `per_class_report` is on. With the moving average of the weights on, `run_test_evaluation()` also returns the results of the averaged weights: their scalars follow as `ema_<scalar>` columns, and a second line `Best Checkpoint (EMA Weights) : iteration <n> (Accuracy <value>)` is printed.

**Example output** (after the four evaluations)
```text
Evaluation Summary Of 4 Checkpoints : file:///.../runs/evaluations/CLS_IN-KAR-64_..._MC9/Evaluation/evaluation_summary.csv
  - Best Checkpoint : iteration 9 (Accuracy 0.1133)

evaluation_summary.csv (32 columns: iteration, the 4 global scalars, the 27 per-class scalars):
iteration,Accuracy,Balanced Accuracy,Macro F1,Top-2 Accuracy,Precision/square,Recall/square,F1/square, ...
12,0.1,0.1111111111111111,0.020202020202020204,0.21333333333333335,0.0,0.0,0.0, ...
9,0.11333333333333333,0.1111111111111111,0.02262142381902861,0.21333333333333335,0.0,0.0,0.0, ...
6,0.1,0.1111111111111111,0.020202020202020204,0.24666666666666667,0.0,0.0,0.0, ...
3,0.09,0.1111111111111111,0.018348623853211007,0.19,0.0,0.0,0.0, ...
```

### Step 5.6 — `run_test_evaluation()` for each checkpoint
This is exactly the function of Stage 4, so a checkpoint evaluated here gives the same block in the console and in `Results.txt`, the same TensorBoard curves (`Tensorboard/Test`), the same sample images and the same evaluation files (`Evaluation/Iteration_N/`) as during the training, only in the output folder, on the split chosen in the script, and tracked in the MLflow run of the evaluation. The output folder is named after the run settings, so evaluating the same run twice reuses it: `Results.txt` receives new blocks and the `Iteration_N` folders are rewritten.

**Example output: the block of the checkpoint of iteration 9 on the 300 images** (the console prints the same links as in Step 4.18, then the next checkpoint starts with its `Evaluating :` line)
```text
----------------------------------------------------------------------------------------------------------
Evaluating : file:///.../runs/CLS_IN-KAR-64_..._MC9/Weights/Iteration_9/model_000009.pt
----------------------------------------------------------------------------------------------------------
Iteration 9 · 300 Test Images · 2026-10-09 09:53 ─────────────────────────────────────────────────
  Accuracy   0.1133   Balanced Accuracy   0.1111
  Macro F1   0.0226   Top-2 Accuracy      0.2133
  Error Images : 160 in 8 class pairs, at most 20 per pair
  Sample Images : 8 fixed test images

  Per Class Metrics
                                                       Threshold    Recall
   Class       Samples   Precision   Recall       F1     @ P0.99   @ P0.99
  ─────────────────────────────────────────────────────────────────────────
   ...
   octagon          34      0.1133   1.0000   0.2036           —         —
  Worst F1 first, ties in class order. — : no samples of the class, or the precision 0.99 is never reached.

evaluations/CLS_IN-KAR-64_..._MC9/
├── Evaluation/Iteration_12/, Iteration_9/, Iteration_6/ and Iteration_3/
│                                                  Errors/  Samples/  class_metrics.csv  class_thresholds.json
│                                                  confusion_matrix.csv  confusion_matrix.png  predictions.csv
├── Evaluation/evaluation_summary.csv
├── Results.txt                                    the four blocks
├── Tensorboard/Test/
├── Weights/                                       empty
└── training_configuration.yaml                    copy of the configuration of the trained run
```
