import torch

from typing import List, Union, Dict, Literal, Tuple, Any, Optional
from torchvision.ops import roi_align
from torch import nn

from architecture_modules.convolution_block import ConvolutionBlock
from utilities.model.model_utilities import batch_assign_targets_to_proposals, apply_regression_predictions
from utilities.tensor_utilities import print_tensor_shape, print_tensor_list


class DetectionHead(nn.Module):
    def __init__(self, detector_configuration: Dict[str, Any], number_classes: int, input_channels: int,
                 feature_map_normalization: str, input_image_size: int, feature_map_size: int, dtype: torch.dtype,
                 device: torch.device) -> None:
        """
        Initializes the Detection Head module for a specific feature map scale.
        
        This module receives pooled features from the ROI Align layer and processes them through
        convolutional blocks followed by fully connected layers. Finally, it outputs the
        class scores and bounding box regression offsets for the detected objects.
        
        Args:
            detector_configuration (Dict[str, Any]): The configuration dictionary for this specific head scale.
            number_classes (int): The total number of classes to predict (including background).
            input_channels (int): The number of channels in the input feature maps.
            feature_map_normalization (str): The chosen normalization layer type (e.g., 'batch', 'layer', 'none').
            dtype (torch.dtype): The tensor data type.
            device (torch.device): The computational device.
        """
        super(DetectionHead, self).__init__()

        self.dtype = dtype
        self.device = device
        self.number_classes = number_classes
        self.input_image_size = input_image_size
        self.feature_map_size = feature_map_size
        self.feature_map_normalization = feature_map_normalization

        # The spatial scale is the ratio used by RoIAlign to map coordinates from the original 
        # input image space down into the downsampled feature map coordinate space (e.g., 32 / 1024 = 1/32)
        self.spatial_scale = self.feature_map_size / self.input_image_size
        self.roi_align_pool_size = detector_configuration["roi_align_pool_size"]

        self.convolution_number_layers = detector_configuration["convolutions"][0]
        convolution_output_channels = detector_configuration["convolutions"][1]
        self.fully_connected_dimensions = detector_configuration["fully_connected"]

        # Convolutional feature extraction block
        self.convolutional_block = ConvolutionBlock(
            input_channels=input_channels, output_channels=convolution_output_channels,
            bias=True, number_layers=self.convolution_number_layers,
            kernel_size=3, stride=1, padding=1,
            feature_map_normalization=self.feature_map_normalization,
            activation=True, dropout_rate=0.0, add_pooling=False, device=self.device, dtype=self.dtype)

        # Fully connected layers
        flattened_dimension = self.roi_align_pool_size * self.roi_align_pool_size * convolution_output_channels
        self.fully_connected_block = self._build_fully_connected_block(
            input_dimension=flattened_dimension,
            dimensions=self.fully_connected_dimensions)

        # Final task-specific output layers
        # Object classification layer (outputs un-normalized logits for number_classes)
        self.classifier = nn.Linear(in_features=detector_configuration["fully_connected"][-1],
                                    out_features=self.number_classes, device=self.device, dtype=self.dtype)

        # Bounding box regression layer (outputs [dx, dy, dw, dh] per class)
        self.bounding_box_regressor = nn.Linear(in_features=detector_configuration["fully_connected"][-1],
                                                out_features=self.number_classes * 4,
                                                device=self.device, dtype=self.dtype)

    def _build_fully_connected_block(self, input_dimension: int, dimensions: List[int]) -> nn.Sequential:
        """
        Builds the fully connected layers for the Detection Head.
        
        Args:
            input_dimension (int): The flattened input dimension.
            dimensions (List[int]): The sequence of output dimensions for each FC layer.
            
        Returns:
            nn.Sequential: The constructed sequence of FC and ReLU layers.
        """
        fully_connected_layers = []
        current_dimension = input_dimension

        for next_dimension in dimensions:
            fully_connected_layers.append(nn.Linear(in_features=current_dimension, out_features=next_dimension,
                                                    device=self.device, dtype=self.dtype))
            fully_connected_layers.append(nn.ReLU(inplace=True))
            current_dimension = next_dimension

        return nn.Sequential(*fully_connected_layers)

    def forward(self, proposal_boxes: torch.Tensor, input_tensor: torch.Tensor,
                tensor_to_concatenate: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes the forward pass of the Detection Head.
        
        This method processes the sampled region proposals by cropping and aligning their corresponding 
        features from the backbone's feature map using RoIAlign. The pooled features are then passed 
        through a series of convolutional and fully connected layers to predict class probabilities 
        and bounding box refinements.
        
        Args:
            proposal_boxes (torch.Tensor): A tensor of shape [K, 5] representing the sampled proposals across the batch, 
                                           where the first column is the batch index and the rest are [x1, y1, x2, y2].
            input_tensor (torch.Tensor): Batched feature map from backbone of shape [Batch, Channels, Height, Width].
            tensor_to_concatenate (Optional[torch.Tensor]): Optional tensor to concatenate.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - classification_scores (torch.Tensor): The predicted class logits of shape [K, number_classes].
                - box_regressions (torch.Tensor): The predicted bounding box regressions of shape [K, number_classes, 4].
                - detection_boxes (torch.Tensor): The final bounding box coordinates after
                 applying regression offsets of shape [K, number_classes, 4].
        """

        # 1. RoIAlign: Extract and pool features for each proposal
        # Input Feature Map Shape: [Batch, Channels, Height, Width]
        # Output Pooled Shape: [K, Channels, Pool_Size, Pool_Size]
        pooled_features = roi_align(input=input_tensor, boxes=proposal_boxes,
                                    output_size=(self.roi_align_pool_size, self.roi_align_pool_size),
                                    spatial_scale=self.spatial_scale, aligned=True)

        # 2. Convolutional Block: Process the pooled features
        # Input Shape: [K, Channels, Pool_Size, Pool_Size]
        # Output Shape: [K, Conv_Out_Channels, Pool_Size, Pool_Size]
        enhanced_pooled_features = self.convolutional_block(input_tensor=pooled_features)

        # 3. Flatten the spatial dimensions for the fully connected layers
        # Input Shape: [K, Conv_Out_Channels, Pool_Size, Pool_Size]
        # Output Shape: [K, Flattened_Dimension]
        flattened_enhanced_pooled_features = enhanced_pooled_features.flatten(start_dim=1)

        # 4. Fully Connected Block: Deep feature extraction
        # Input Shape: [K, Flattened_Dimension] 
        # Output Shape: [K, FC_Out_Dimension]
        fully_connected_output_tensor = self.fully_connected_block(input=flattened_enhanced_pooled_features)

        # 5. Prediction Heads: Output classification and regression predictions
        # Output Classification Shape: [K, number_classes]
        classification_scores = self.classifier(input=fully_connected_output_tensor)

        # Output Regression Shape: [K, number_classes * 4] -> reshaped to [K, number_classes, 4]
        box_regressions = self.bounding_box_regressor(input=fully_connected_output_tensor)

        # Reshape box regressions to isolate the coordinates per class cleanly
        number_boxes = box_regressions.shape[0]
        box_regressions = box_regressions.reshape(number_boxes, self.number_classes, 4)

        # The proposal_boxes have shape [K, 5] where the first column is the batch index.
        # We slice [:, 1:] to extract the [K, 4] bounding box coordinates.
        detection_boxes = apply_regression_predictions(
            regression_predictions=box_regressions.detach(),
            boxes=proposal_boxes[:, 1:])

        return classification_scores, box_regressions, detection_boxes
