import torch

from typing import Any, Dict, Tuple


def get_detection_model_configuration() -> Tuple[Dict[str, Any], torch.device, torch.dtype]:
    """
    Supplies a small detector configuration dictionary, with its device and dtype, shared by the detector tests and
    the detection debug scripts: 9 backbone blocks, levels 5 to 7 and 5 foreground classes.
    
    Returns:
        Tuple[Dict[str, Any], torch.device, torch.dtype]: The FPN F-RCNN configuration mapping, compute device, and tensor dtype.
    """
    configuration = {'Data': {'image_settings': {'size': 1024},
                              'label_dictionary': {'background': 0,
                                                   'foreground_1': 1,
                                                   'foreground_2': 2,
                                                   'foreground_3': 3,
                                                   'foreground_4': 4,
                                                   'foreground_5': 5}},
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
                         'modules': {'squeeze_and_excite': False, 'residual_nets': False, 'inception': False,
                                     'inception_starting_block': 3},
                         'last_max_pooling': True,
                         'normalization': {'feature_map_normalization': 'none', 'input_normalization': False,
                                           'per_sample_normalization': False},
                         'detection_convolution_block_indices': [5, 6, 7]},
                     'Anchors': {
                         'scales_and_ratios': {
                             '5': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]},
                             '6': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]},
                             '7': {'scales': [64, 128], 'aspect_ratios': [0.5, 1.0, 2.0]}
                         }},
                     'RegionProposal': {
                         'head_configurations': {'5': [2, 128], '6': [2, 128], '7': [2, 128]},
                         'normalization': {'feature_map_normalization': 'batch'},
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
                     'Detector': {
                         'head_configurations': {
                             '5': {'roi_align_pool_size': 7, 'convolutions': [2, 64], 'fully_connected': [256, 128]},
                             '6': {'roi_align_pool_size': 7, 'convolutions': [2, 64], 'fully_connected': [256, 128]},
                             '7': {'roi_align_pool_size': 7, 'convolutions': [2, 64], 'fully_connected': [256, 128]}
                         },
                         'normalization': {'feature_map_normalization': 'batch'},
                         'foreground_iou_threshold': 0.3,
                         'number_training_positives': 128,
                         'total_training_samples': 256,
                         'localisation_loss_beta': 1 / 9,
                         'modules': {'squeeze_and_excite': False, 'residual_nets': False, 'inception': False},
                         'nms_iou_threshold': 0.5,
                         'inference': {'score_threshold': 0.5, 'maximum_detections': 100},
                     }}
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
                                  'modules': {'squeeze_and_excite': True, 'residual_nets': True, 'inception': False,
                                              'inception_starting_block': 3},
                                  'last_max_pooling': False,
                                  'normalization': {'feature_map_normalization': 'batch', 'input_normalization': True,
                                                    'per_sample_normalization': False},
                                  'detection_convolution_block_indices': []},
                     'ClassificationHead': {'global_pooling': 'average', 'fully_connected': [16], 'dropout_rate': 0.0,
                                            'output_mode': output_mode, 'label_smoothing': 0.0,
                                            'class_weighting': 'none'}}
    device = torch.device(device="cpu")
    dtype = torch.float32

    return configuration, device, dtype
