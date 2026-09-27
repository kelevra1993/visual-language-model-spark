# Visual Language Model Spark

## Table Of Content
- Description
- Prerequisites

## Description
This repository contains an object detection pipeline built with PyTorch, supporting NVIDIA DALI and TFRecord data loading backends.

## Prerequisites
- PyTorch
- NVIDIA DALI
- OpenCV

## Implementation Plan
### Goal Description
The benchmarking loop relies on the model iteration method to execute the forward passes and backward passes. Currently, the model iteration method is populated with placeholder logic. We must route the unpacked batch tensors into the forward pass, aggregate the individual loss components into a singular loss tensor for backpropagation, and fortify the type hints and variable scoping to ensure crash-free execution.

### Proposed Changes

#### `object_detection/trainer/trainer.py`
1. **Type Hints Rectification:** Update the return type signature of `get_next_batch` to accurately reflect the data dictionary output instead of the temporary placeholder tuple.
   
2. **Forward Pass Integration:** Update the empty model call to explicitly map the unpacked data dictionary variables to the target input arguments.
   ```python
   # Execute the forward pass of the model by providing the images alongside the ground truth annotations
   model_outputs = self.model(input_tensor=images, ground_truth_bounding_boxes=bounding_boxes, ground_truth_labels=labels)
   ```

3. **Loss Aggregation:** Extract the individual loss components from the dictionary and compute their sum.
   ```python
   # Extract the individual loss components and sum them to construct the total loss graph for backpropagation
   total_loss = model_outputs["region_proposal_classification_loss"] + model_outputs["region_proposal_localisation_loss"] + model_outputs["detection_classification_loss"] + model_outputs["detection_regression_loss"]
   ```

4. **Iterator Scoping:** Move the validation dataloader iterator initialization strictly inside the validation block to prevent initialization crashes when validation is disabled.

### Verification Plan
Execute the primary entry point script. The benchmarking loop will initiate without crashing, successfully propagating gradients through the aggregated loss tensor.
