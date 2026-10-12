"""
Mask branch of the instance segmenter: the MaskHead of one detection level pools every box from the feature map of its
level with RoIAlign, runs the pooled features through a ConvolutionBlock, doubles their resolution with a transposed
convolution and predicts one mask logit map per class, as in Mask R-CNN. Built for every detection level by the
InstanceSegmenter, which trains it on the positive proposals and runs it on the final detections, and reused by the
InferenceInstanceSegmenter of the exports.
"""

import torch

from typing import Any, Dict, List
from torch import nn
from torchvision.ops import roi_align

from vision_training.models.components.convolution_block import ConvolutionBlock


class MaskHead(nn.Module):
    """
    Predicts the binary mask of the object of every box of one detection level, one [2P, 2P] logit map per class, the
    channel of the detected (or ground truth) class being read by the instance segmenter.
    """

    def __init__(self, head_configuration: List[int], roi_align_pool_size: int, number_classes: int,
                 input_channels: int, feature_map_normalization: str, batch_normalization_momentum: float,
                 input_image_size: int, feature_map_size: int, modules: Dict[str, Any], dtype: torch.dtype,
                 device: torch.device) -> None:
        """
        Builds the mask head of one detection level from the 'MaskHead' section of the configuration.

        Args:
            head_configuration (List[int]): The [number_layers, output_channels] of the convolution block of this level.
            roi_align_pool_size (int): The RoIAlign pool size P, the same on every level, the masks being 2P x 2P.
            number_classes (int): The number of classes, background included, one mask channel per class.
            input_channels (int): The number of channels of the feature map of this level.
            feature_map_normalization (str): The normalization of the convolution block ("batch", "layer" or "none").
            batch_normalization_momentum (float): The momentum of the running statistics of its batch normalization.
            input_image_size (int): The size of the square network input, giving the spatial scale of RoIAlign.
            feature_map_size (int): The size of the square feature map of this level.
            modules (Dict[str, Any]): The 'MaskHead.modules' flags ('squeeze_and_excite', 'residual_nets',
                'inception') of the convolution block.
            dtype (torch.dtype): The data type of the parameters.
            device (torch.device): The device of the parameters.
        """
        super(MaskHead, self).__init__()

        self.number_classes = number_classes
        self.feature_map_size = feature_map_size
        self.roi_align_pool_size = roi_align_pool_size
        self.mask_size = 2 * roi_align_pool_size
        self.convolution_number_layers, convolution_output_channels = head_configuration

        # Ratio mapping the boxes from network input pixels to feature map cells, as in the detection head
        self.spatial_scale = feature_map_size / input_image_size

        # Convolutional feature extraction on the pooled features, at the pool resolution
        self.convolutional_block = ConvolutionBlock(
            input_channels=input_channels, output_channels=convolution_output_channels, bias=True,
            number_layers=self.convolution_number_layers, kernel_size=3, stride=1, padding=1,
            feature_map_normalization=feature_map_normalization,
            batch_normalization_momentum=batch_normalization_momentum, activation=True, dropout_rate=0.0,
            add_pooling=False, device=device, dtype=dtype, squeeze_and_excite=bool(modules["squeeze_and_excite"]),
            residual=bool(modules["residual_nets"]), inception=bool(modules["inception"]))

        # Transposed convolution doubling the resolution, then one mask logit map per class
        self.upsampling = nn.ConvTranspose2d(in_channels=convolution_output_channels,
                                             out_channels=convolution_output_channels, kernel_size=2, stride=2,
                                             device=device, dtype=dtype)
        self.activation = nn.ReLU()
        self.mask_predictor = nn.Conv2d(in_channels=convolution_output_channels, out_channels=number_classes,
                                        kernel_size=1, device=device, dtype=dtype)

    def forward(self, proposal_boxes: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predicts the mask logits of every box from the feature map of this level.

        Args:
            proposal_boxes (torch.Tensor): The [R, 5] boxes, the image index of every box followed by its
                [x1, y1, x2, y2] corners in network input pixels.
            input_tensor (torch.Tensor): The [B, C, H, W] feature map of this level.

        Returns:
            torch.Tensor: The [R, number_classes, 2P, 2P] mask logits.
        """
        # Pool every box with the RoIAlign settings of the detection head, exported as the ONNX RoiAlign operator
        pooled_features = roi_align(input=input_tensor, boxes=proposal_boxes,
                                    output_size=(self.roi_align_pool_size, self.roi_align_pool_size),
                                    spatial_scale=self.spatial_scale, sampling_ratio=0, aligned=True)

        # Refine the pooled features, double their resolution and predict one mask per class
        enhanced_features = self.convolutional_block(input_tensor=pooled_features)
        upsampled_features = self.activation(input=self.upsampling(input=enhanced_features))
        return self.mask_predictor(input=upsampled_features)
