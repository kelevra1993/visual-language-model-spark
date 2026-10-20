"""
Small model configuration dictionaries of the detection (with or without mask heads), classification, semantic
segmentation, sequence, sequence comparison, self-supervised and keypoint detection models, with their device and
dtype, shared by the model, export and transfer tests. Every configuration keeps no moving average of the weights (the
'Optimization' setting read by the transfer of the components).
"""

import torch

from typing import Any, Dict, Tuple


def get_detection_model_configuration(mask_head: bool = False) -> Tuple[Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies a small detector configuration dictionary, with its device and dtype, shared by the detector tests:
    9 backbone blocks, levels 5 to 7 and 5 foreground classes. With mask_head, the 'MaskHead' section of the
    instance segmenter is added: 7 x 7 pooling (14 x 14 masks) and 2 layers of 32 channels on every level.

    Args:
        mask_head (bool): Whether to add the 'MaskHead' section of the instance segmentation task. Defaults to False.

    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The FPN F-RCNN configuration mapping, compute device, and
            tensor dtype.
    """
    # Images of 1024 pixels with the background and 5 foreground classes
    configuration = {'Data': {'image_settings': {'size': 1024},
                              'label_dictionary': {'background': 0,
                                                   'foreground_1': 1,
                                                   'foreground_2': 2,
                                                   'foreground_3': 3,
                                                   'foreground_4': 4,
                                                   'foreground_5': 5}},
                     # Backbone of 9 blocks of 2 convolutions, without optional module or normalization
                     'Backbone': {
                         'input_channels': 3,
                         'convolutions': {'1': [2, 16],
                                          '2': [2, 32],
                                          '3': [2, 32],
                                          '4': [2, 64],
                                          '5': [2, 64],
                                          '6': [2, 64],
                                          '7': [2, 64],
                                          '8': [2, 64],
                                          '9': [2, 64],
                                          },
                         'modules': {'squeeze_and_excite': False, 'residual_nets': False, 'residual_nets_v2': False,
                                     'inception': False, 'inception_starting_block': 3},
                         'last_max_pooling': True,
                         'normalization': {'feature_map_normalization': 'none', 'batch_normalization_momentum': 0.1,
                                           'input_normalization': False, 'per_sample_normalization': False}},
                     # Region proposal and detection heads on blocks 5 to 7, with 6 anchors per position
                     'DetectionLevels': {
                         level: {'anchors': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]},
                                 'region_proposal': {'convolutions': [2, 128]},
                                 'detector': {'roi_align_pool_size': 7, 'convolutions': [2, 64],
                                              'fully_connected': [256, 128]}}
                         for level in ['5', '6', '7']},
                     # Proposal filter and anchor target assignment of the region proposal network
                     'RegionProposal': {
                         'normalization': {'feature_map_normalization': 'batch', 'batch_normalization_momentum': 0.1},
                         'nms_iou_threshold': 0.7,
                         'training': {'pre_nms_proposals': 500, 'post_nms_proposals': 250},
                         'inference': {'pre_nms_proposals': 100, 'post_nms_proposals': 50},
                         'foreground_iou_threshold': {"min": 0.35, "max": 1.0},
                         'background_iou_threshold': {"min": 0.2, "max": 0.3},
                         'strict_fallback_assignment': False,
                         'number_training_positives': 128,
                         'total_training_samples': 256,
                         'localisation_loss_beta': 1 / 9
                     },
                     # Proposal sampling, losses and postprocessing of the detection heads
                     'Detector': {
                         'normalization': {'feature_map_normalization': 'batch', 'batch_normalization_momentum': 0.1},
                         'foreground_iou_threshold': 0.3,
                         'number_training_positives': 128,
                         'total_training_samples': 256,
                         'localisation_loss_beta': 1 / 9,
                         'modules': {'squeeze_and_excite': False, 'residual_nets': False, 'inception': False},
                         'nms_iou_threshold': 0.5,
                         'inference': {'score_threshold': 0.5, 'maximum_detections': 100},
                     },
                     'Optimization': {'exponential_moving_average': {'activated': False, 'decay': 0.9998}}}
    # Mask heads of the instance segmenter on the same levels as the detection heads
    if mask_head:
        for level_configuration in configuration['DetectionLevels'].values():
            level_configuration['mask_head'] = {'roi_align_pool_size': 7, 'convolutions': [2, 32]}
        configuration['MaskHead'] = {'mask_resolution': 14, 'upsampling': 'transposed',
                                     'normalization': {'feature_map_normalization': 'batch',
                                                       'batch_normalization_momentum': 0.1},
                                     'modules': {'squeeze_and_excite': False, 'residual_nets': True,
                                                 'inception': False},
                                     'inference': {'mask_threshold': 0.5}}
    device = torch.device(device="cpu")
    dtype = torch.float32

    return configuration, device, dtype


def get_classification_model_configuration(output_mode: str) -> Tuple[Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies a small image classifier configuration dictionary, with its device and dtype, shared by the
    classification model tests: 64 pixel images, 4 backbone blocks, one hidden layer and 3 classes.

    Args:
        output_mode (str): "multi_class" (one class per image) or "multi_label" (several classes per image).

    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The classifier configuration, compute device and dtype.
    """
    configuration = {'Data': {'image_settings': {'size': 64}, 'label_dictionary': {1: 0, 2: 1, 3: 2}},
                     'Backbone': {'input_channels': 3,
                                  'convolutions': {'1': [1, 8], '2': [1, 16], '3': [2, 16], '4': [2, 32]},
                                  'modules': {'squeeze_and_excite': True, 'residual_nets': True,
                                              'residual_nets_v2': False, 'inception': False,
                                              'inception_starting_block': 3},
                                  'last_max_pooling': False,
                                  'normalization': {'feature_map_normalization': 'batch',
                                                    'batch_normalization_momentum': 0.1, 'input_normalization': True,
                                                    'per_sample_normalization': False}},
                     'ClassificationHead': {'global_pooling': 'average', 'fully_connected': [16],
                                            'fully_connected_normalization': 'none',
                                            'batch_normalization_momentum': 0.1, 'dropout_rate': 0.0,
                                            'output_mode': output_mode, 'label_smoothing': 0.0,
                                            'class_weighting': 'none',
                                            'decorrelation': {'activated': False, 'weight': 0.001, 'hidden_layer': 1}},
                     'Optimization': {'exponential_moving_average': {'activated': False, 'decay': 0.9998}}}
    device = torch.device(device="cpu")
    dtype = torch.float32

    return configuration, device, dtype


def get_segmentation_model_configuration(upsampling: str) -> Tuple[Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies a small semantic segmentation configuration dictionary, with its device and dtype, shared by the
    segmentation model tests: the 4 backbone blocks of the image classifier configuration on 64 pixel images, a
    decoder of 3 levels (blocks 3, 2 and 1) and 3 classes, the background included, trained with the cross entropy
    and the Dice loss.

    Args:
        upsampling (str): The upsampling of the decoder levels, "transposed" or "bilinear".

    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The segmenter configuration, compute device and dtype.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    del configuration['ClassificationHead']
    configuration['Data']['label_dictionary'] = {'background': 0, 1: 1, 2: 2}
    configuration['SegmentationDecoder'] = {'upsampling': upsampling,
                                            'convolutions': {'3': [1, 16], '2': [1, 16], '1': [1, 8]},
                                            'modules': {'squeeze_and_excite': False, 'residual_nets': False},
                                            'feature_map_normalization': 'batch',
                                            'batch_normalization_momentum': 0.1, 'dropout_rate': 0.0,
                                            'deep_supervision_weights': {}}
    configuration['SegmentationLoss'] = {'cross_entropy_weight': 1.0, 'focal_gamma': 0.0, 'label_smoothing': 0.0,
                                         'class_weighting': 'none', 'dice_weight': 1.0, 'lovasz_weight': 0.0,
                                         'boundary_weight': 0.0, 'boundary_distance': 3}

    return configuration, device, dtype


def get_sequence_model_configuration(temporal_type: str) -> Tuple[Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies a small sequence model configuration dictionary, with its device and dtype, shared by the sequence model
    tests: the image classifier configuration with sequences of at most 4 frames and a small temporal module.

    Args:
        temporal_type (str): The type of the temporal module.

    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The sequence model configuration, compute device and dtype.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    configuration['Data']['sequence_settings'] = {'maximum_sequence_length': 4}
    configuration['TemporalModel'] = {'type': temporal_type, 'hidden_size': 16, 'number_layers': 1, 'number_heads': 2,
                                      'kernel_size': 3, 'dropout_rate': 0.0, 'bidirectional': False, 'causal': False,
                                      'positional_embedding': True}

    return configuration, device, dtype


def get_sequence_comparison_model_configuration(temporal_type: str, fusion_type: str) -> Tuple[
        Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies a small sequence comparison model configuration dictionary, with its device and dtype, shared by the
    comparison model tests: the sequence model configuration with members of at most 4 frames and the fusion of the
    members, its cross attention splitting the 32 channels of the last backbone block between 2 heads.

    Args:
        temporal_type (str): The type of the temporal module.
        fusion_type (str): The type of the fusion of the members.

    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The comparison model configuration, compute device and dtype.
    """
    configuration, device, dtype = get_sequence_model_configuration(temporal_type=temporal_type)
    configuration['Fusion'] = {'type': fusion_type, 'number_heads': 2, 'dropout_rate': 0.0,
                               'sequence_normalization_scope': 'member'}

    return configuration, device, dtype


def get_self_supervised_model_configuration(method: str, global_pooling: str) -> Tuple[
        Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies a small self-supervised model configuration dictionary, with its device and dtype, shared by the
    self-supervised model and export tests: the backbone of the image classifier configuration on 64 pixel images and
    a projection head of one batch normalized hidden layer of 16 units and an embedding of 8.

    Args:
        method (str): The self-supervised method, "simclr" or "barlow_twins".
        global_pooling (str): The reduction of the final feature map, "average", "max" or "flatten".

    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The self-supervised model configuration, compute device and
            dtype.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    del configuration['ClassificationHead']
    configuration['ProjectionHead'] = {'global_pooling': global_pooling, 'fully_connected': [16, 8],
                                       'fully_connected_normalization': 'batch', 'batch_normalization_momentum': 0.1,
                                       'method': method, 'simclr': {'temperature': 0.2},
                                       'barlow_twins': {'off_diagonal_weight': 0.005}}

    return configuration, device, dtype


def get_keypoint_model_configuration(higher_resolution: bool = False,
                                     hidden_width: int = 0) -> Tuple[Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies a small keypoint detector configuration dictionary, with its device and dtype, shared by the keypoint
    model and export tests: 64 x 64 whole images, a stem of 3 backbone blocks ending at a quarter of the image, a high
    resolution network of 2 stages (3 branches, the lowest at 1/16 of the image) of base width 8, and the keypoint
    categories injected by the configuration loader for 2 classes, a triangle of 3 keypoints and a rectangle of 4
    keypoints, so 7 heatmap and 7 tag channels.

    Args:
        higher_resolution (bool): Whether to add the higher resolution branch of heatmaps at half the image
            resolution. Defaults to False.
        hidden_width (int): The channels of the 3x3 convolution of the heads before their predictions, 0 for the 1x1
            predictions read on the features. Defaults to 0.

    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The keypoint detector configuration, compute device and
            dtype.
    """
    configuration, device, dtype = get_classification_model_configuration(output_mode="multi_class")
    del configuration['ClassificationHead']
    configuration['Data']['label_dictionary'] = {'background': 0, 1: 1, 2: 2}
    configuration['Data']['keypoint_categories'] = {
        1: {'name': 'triangle', 'keypoints': ['top', 'left', 'right'], 'skeleton': [[0, 1], [1, 2], [2, 0]]},
        2: {'name': 'rectangle', 'keypoints': ['top_left', 'top_right', 'bottom_right', 'bottom_left'],
            'skeleton': [[0, 1], [1, 2], [2, 3], [3, 0]]}}
    configuration['Backbone']['convolutions'] = {'1': [1, 8], '2': [1, 16], '3': [1, 16]}
    configuration['HRNet'] = {'base_width': 8, 'modules_per_stage': [1, 1], 'blocks_per_branch': 1,
                              'modules': {'squeeze_and_excite': True, 'residual_nets': True},
                              'normalization': {'feature_map_normalization': 'batch',
                                                'batch_normalization_momentum': 0.1}}
    configuration['KeypointHead'] = {'higher_resolution': higher_resolution, 'hidden_width': hidden_width,
                                     'heatmap_sigma': 2.0, 'ignore_unlabelled_objects': False,
                                     'tag_pull_weight': 0.001, 'tag_push_weight': 0.001,
                                     'inference': {'maximum_candidates': 5, 'keypoint_threshold': 0.1,
                                                   'tag_threshold': 1.0, 'object_score_threshold': 0.2}}

    return configuration, device, dtype
