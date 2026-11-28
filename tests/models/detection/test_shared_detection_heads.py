"""
Tests of the region proposal and detection heads shared by the levels ('RegionProposal.shared' and
'Detector.shared') of vision_training/models/detection/two_stage_detector.py, as in Mask R-CNN: a single head of each
kind reads every level, the anchors of every level reaching the shared region proposal head and the proposals of every
level reaching the shared detection head, so that the losses, the gradients and the detections are those of per-level
heads holding the same weights, with both proposal selections and in the instance segmenter. The deployed
InferenceDetector reproduces the eager model with the shared heads, and the levels that cannot share a head are
refused.
"""

import copy
import pytest
import torch

from typing import Any, Dict, List

from tests.fixtures.configuration_fixtures import get_detection_model_configuration
from tests.models.detection.test_detector_baseline import get_fixed_batch
from tests.models.instance_segmentation.test_instance_segmenter import build_box_masks
from tests.testing_utilities import build_test_model, build_whole_input_windows
from vision_training.deployment.export_specifications import compare_detections
from vision_training.models.detection.inference_detector import InferenceDetector
from vision_training.models.detection.two_stage_detector import SHARED_HEAD_KEY, TwoStageDetector
from vision_training.models.instance_segmentation.instance_segmenter import InstanceSegmenter

# Dictionaries of the heads a model shares between the levels, and the final detections compared between two models
SHARED_HEAD_DICTIONARIES = ["region_proposer_dictionary", "detector_heads_dictionary"]
DETECTION_OUTPUT_NAMES = ["final_detection_boxes", "final_detection_scores", "final_detection_labels",
                          "number_detections"]

# Anchor scales doubling from level to level, as in the shipped configurations, so that the objects of the fixed batch
# and their proposals spread over the levels
LEVEL_ANCHOR_SCALES = {"5": [32, 64], "6": [64, 128], "7": [128, 256]}


def copy_shared_heads_to_levels(shared_model: TwoStageDetector, level_model: TwoStageDetector) -> None:
    """
    Gives every per-level region proposal and detection head of a model the weights and statistics of the head of the
    same kind shared by the levels of another model, and the mask heads of the instance segmenters (one per level in
    both models) the same weights, so that the two models compute the same function.

    Args:
        shared_model (TwoStageDetector): The model sharing its region proposal and detection heads.
        level_model (TwoStageDetector): The model with one head of each kind per level, modified in place.
    """
    for dictionary_name in SHARED_HEAD_DICTIONARIES:
        shared_state = getattr(shared_model, dictionary_name)[SHARED_HEAD_KEY].state_dict()
        for level_head in getattr(level_model, dictionary_name).values():
            level_head.load_state_dict(state_dict=shared_state)
    if hasattr(level_model, "mask_heads_dictionary"):
        level_model.mask_heads_dictionary.load_state_dict(state_dict=shared_model.mask_heads_dictionary.state_dict())


def run_detection_stages(model: TwoStageDetector, instance_segmentation: bool) -> Dict[str, Any]:
    """
    Runs every stage of a model on the fixed batch with its ground truth, the random sampling of the targets drawing
    from a fixed seed, so that two models computing the same function sample the same anchors and proposals.

    Args:
        model (TwoStageDetector): The detector or instance segmenter, in its current mode.
        instance_segmentation (bool): Whether the model is an instance segmenter, given the box masks of the objects.

    Returns:
        Dict[str, Any]: The stages of forward_detection_stages (forward_instance_segmentation_stages for the instance
            segmenter).
    """
    images, boxes, labels = get_fixed_batch()
    images = images.to(dtype=next(model.parameters()).dtype)
    boxes = [image_boxes.to(dtype=images.dtype) for image_boxes in boxes]
    torch.manual_seed(seed=2)
    if instance_segmentation:
        return model.forward_instance_segmentation_stages(
            input_tensor=images, image_windows=build_whole_input_windows(images=images),
            ground_truth_bounding_boxes=boxes, ground_truth_labels=labels,
            ground_truth_masks=build_box_masks(boxes=boxes, image_size=model.input_image_size))
    return model.forward_detection_stages(input_tensor=images, image_windows=build_whole_input_windows(images=images),
                                          ground_truth_bounding_boxes=boxes, ground_truth_labels=labels)


def collect_losses(model_output_dictionary: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Gives the losses of a model output, the four losses of the detector and the mask loss of the instance segmenter.

    Args:
        model_output_dictionary (Dict[str, Any]): The model output dictionary of the stages.

    Returns:
        Dict[str, torch.Tensor]: The scalar losses, keyed by name.
    """
    return {name: value for name, value in model_output_dictionary.items() if name.endswith("_loss")}


def build_spread_level_model(shared: bool, proposal_selection: str, instance_segmentation: bool) -> TwoStageDetector:
    """
    Builds the test detector (or instance segmenter) of the fixture in float64 behind the path aggregation neck, with
    anchor scales doubling from level to level (LEVEL_ANCHOR_SCALES) and detection heads without normalization, so
    that in training mode the detection heads compute the same function on the proposals of every level whether they
    are run level by level or at once (the region proposal heads, applied to the map of every level apart, read the
    statistics of that map in both cases), from the same seed as build_test_model.

    Args:
        shared (bool): Whether the region proposal and detection heads are shared by the levels.
        proposal_selection (str): The 'RegionProposal.proposal_selection' of the model.
        instance_segmentation (bool): Whether to build the instance segmenter, with one mask head per level.

    Returns:
        TwoStageDetector: The model in training mode, on the CPU.
    """
    torch.manual_seed(seed=0)
    configuration, device, _ = get_detection_model_configuration(mask_head=instance_segmentation,
                                                                 neck_type="path_aggregation")
    configuration = copy.deepcopy(configuration)
    configuration["Data"]["image_settings"]["size"] = 512
    configuration["Detector"]["inference"]["score_threshold"] = 0.0
    configuration["Detector"]["normalization"]["feature_map_normalization"] = "none"
    configuration["Detector"]["shared"] = shared
    configuration["RegionProposal"].update({"proposal_selection": proposal_selection, "shared": shared})
    for level_name, scales in LEVEL_ANCHOR_SCALES.items():
        configuration["DetectionLevels"][level_name]["anchors"]["scales"] = scales
    model_class = InstanceSegmenter if instance_segmentation else TwoStageDetector
    model = model_class(configuration=configuration, device=device, dtype=torch.float64)
    return model.to(device=device, dtype=torch.float64).train()


@pytest.mark.parametrize(argnames=("proposal_selection", "instance_segmentation"),
                         argvalues=[("joint", False), ("per_level", False), ("per_level", True)])
def test_shared_heads_train_on_every_level_like_per_level_heads_of_the_same_weights(
        proposal_selection: str, instance_segmentation: bool) -> None:
    """
    Verifies that with 'RegionProposal.shared' and 'Detector.shared' the model holds one region proposal head and one
    detection head reading every level, a third of the parameters of the three per-level heads of each kind, and that
    in training mode its losses, with the anchors labelled and sampled over every level and the proposals filtered and
    sampled over every level, equal those of per-level heads holding the same weights (build_spread_level_model), the
    gradient of every shared parameter being the sum of the gradients of the per-level heads, and that its final
    detections in evaluation mode are the same. The sampled proposals spread over the levels in different numbers, so
    that a level missing from the shared heads or a shifted slice would change the losses.

    Args:
        proposal_selection (str): The 'RegionProposal.proposal_selection' of both models.
        instance_segmentation (bool): Whether to compare instance segmenters, with their mask loss.
    """
    shared_model = build_spread_level_model(shared=True, proposal_selection=proposal_selection,
                                            instance_segmentation=instance_segmentation)
    level_model = build_spread_level_model(shared=False, proposal_selection=proposal_selection,
                                           instance_segmentation=instance_segmentation)
    copy_shared_heads_to_levels(shared_model=shared_model, level_model=level_model)
    level_names = list(level_model.detection_input_information)

    # One head of each kind reading every level, with the parameters of one per-level head
    for dictionary_name in SHARED_HEAD_DICTIONARIES:
        shared_dictionary, level_dictionary = getattr(shared_model, dictionary_name), getattr(level_model,
                                                                                               dictionary_name)
        assert list(shared_dictionary) == [SHARED_HEAD_KEY] and list(level_dictionary) == level_names
        assert sum(parameter.numel() for parameter in level_dictionary.parameters()) == \
            len(level_names) * sum(parameter.numel() for parameter in shared_dictionary.parameters())
    assert shared_model.detector_heads_dictionary[SHARED_HEAD_KEY].level_names == level_names

    # Same backbone and neck in both models, built before the heads from the same seed
    for component_name in ["backbone", "neck"]:
        shared_state = getattr(shared_model, component_name).state_dict()
        for tensor_name, tensor_value in getattr(level_model, component_name).state_dict().items():
            assert torch.equal(input=tensor_value, other=shared_state[tensor_name]), tensor_name

    # Losses of both models on the same sampled targets, then their gradients
    shared_stages = run_detection_stages(model=shared_model, instance_segmentation=instance_segmentation)
    level_stages = run_detection_stages(model=level_model, instance_segmentation=instance_segmentation)
    shared_losses = collect_losses(model_output_dictionary=shared_stages["model_output_dictionary"])
    level_losses = collect_losses(model_output_dictionary=level_stages["model_output_dictionary"])
    assert sorted(shared_losses) == sorted(level_losses) and len(shared_losses) == 4 + int(instance_segmentation)
    for loss_name, loss_value in shared_losses.items():
        assert torch.allclose(input=loss_value, other=level_losses[loss_name], rtol=1e-10, atol=1e-12), loss_name
    sum(shared_losses.values()).backward()
    sum(level_losses.values()).backward()
    for dictionary_name in SHARED_HEAD_DICTIONARIES:
        level_dictionary = getattr(level_model, dictionary_name)
        for parameter_name, parameter in getattr(shared_model, dictionary_name)[SHARED_HEAD_KEY].named_parameters():
            level_gradients = [level_head.get_parameter(target=parameter_name).grad
                               for level_head in level_dictionary.values()]
            assert parameter.grad is not None and all(gradient is not None for gradient in level_gradients)
            assert torch.allclose(input=parameter.grad, other=torch.stack(tensors=level_gradients).sum(dim=0),
                                  rtol=1e-8, atol=1e-12), f"{dictionary_name}.{parameter_name}"

    # Sampled proposals on every level, in different numbers
    sampled_levels = shared_stages["training_targets"]["sampled_levels"]
    level_counts = [int((sampled_levels == int(level_name)).sum()) for level_name in level_names]
    assert all(count > 0 for count in level_counts) and len(set(level_counts)) > 1

    # Same final detections in evaluation mode, the per-level heads given again the running statistics of the shared
    # region proposal head, which the training step updated with the map of every level in turn
    copy_shared_heads_to_levels(shared_model=shared_model, level_model=level_model)
    images = get_fixed_batch()[0]
    shared_model.eval()
    level_model.eval()
    with torch.no_grad():
        shared_outputs = shared_model(input_tensor=images, image_windows=build_whole_input_windows(images=images))
        level_outputs = level_model(input_tensor=images, image_windows=build_whole_input_windows(images=images))
    assert int(shared_outputs["number_detections"].sum()) > 0
    for output_name in DETECTION_OUTPUT_NAMES:
        assert torch.allclose(input=shared_outputs[output_name].double(), other=level_outputs[output_name].double(),
                              rtol=1e-9, atol=1e-9), output_name


@pytest.mark.parametrize(argnames="proposal_selection", argvalues=["joint", "per_level"])
def test_inference_detector_with_shared_heads_matches_the_eager_model(proposal_selection: str) -> None:
    """
    Verifies that the export-friendly inference detector, whose shared detection head pools every proposal from every
    level, keeps the pooled features of its own level and runs once, reproduces the detections of the eager detector
    with shared heads in evaluation mode, with both proposal selections.

    Args:
        proposal_selection (str): The 'RegionProposal.proposal_selection' of the proposal filter.
    """
    model = build_test_model(dtype=torch.float32, neck_type="path_aggregation", proposal_selection=proposal_selection,
                             shared_region_proposal=True, shared_detector=True)
    detector = InferenceDetector(model=model).eval()
    assert detector.shared_region_proposal and detector.shared_detector and detector.level_names == ["5", "6", "7"]
    images = torch.rand(size=(2, 3, 512, 512), generator=torch.Generator().manual_seed(4)) * 255.0

    with torch.no_grad():
        eager_outputs = model(input_tensor=images, image_windows=build_whole_input_windows(images=images))
        inference_outputs = detector(images=images, image_windows=build_whole_input_windows(images=images))

    # Same numbers of detections and scores, every detection paired with one of the same label, score and box
    eager_detections = [eager_outputs[output_name].numpy() for output_name in DETECTION_OUTPUT_NAMES]
    assert int(inference_outputs[3].sum()) > 0
    assert torch.equal(input=inference_outputs[3], other=eager_outputs["number_detections"])
    assert torch.allclose(input=inference_outputs[1], other=eager_outputs["final_detection_scores"])
    assert compare_detections(reference=eager_detections, candidate=[output.numpy() for output in inference_outputs],
                              box_tolerance=1e-4, score_tolerance=1e-6) == []


def change_level_7(configuration: Dict[str, Any], changed_setting: str) -> None:
    """
    Changes one setting of level 7 of the detector configuration, so that its heads can no longer be shared with
    those of levels 5 and 6.

    Args:
        configuration (Dict[str, Any]): The detector configuration, modified in place.
        changed_setting (str): "anchors" for a third anchor scale, "region_proposal" or "detector" for other
            convolutions, "channels" for a backbone block of other channels read without neck.
    """
    level_configuration = configuration["DetectionLevels"]["7"]
    if changed_setting == "anchors":
        level_configuration["anchors"]["scales"] = [64, 128, 256]
    elif changed_setting == "channels":
        configuration["Backbone"]["convolutions"]["7"] = [2, 32]
    else:
        level_configuration[changed_setting]["convolutions"] = [1, 64]


@pytest.mark.parametrize(argnames=("section_name", "changed_setting", "expected_message"), argvalues=[
    ("RegionProposal", "anchors", "'RegionProposal.shared' needs the same number of anchors per position"),
    ("RegionProposal", "region_proposal", "'RegionProposal.shared' needs the same 'region_proposal'"),
    ("RegionProposal", "channels", "'RegionProposal.shared' needs the same channel count"),
    ("Detector", "detector", "'Detector.shared' needs the same 'detector'"),
    ("Detector", "channels", "'Detector.shared' needs the same channel count")])
def test_shared_heads_are_refused_for_levels_that_cannot_share_them(section_name: str, changed_setting: str,
                                                                    expected_message: str) -> None:
    """
    Verifies that the detector refuses to share a head between levels with different anchors per position (the
    outputs of the region proposal head), different head settings or different channel counts, while the same
    change leaves a detector of per-level heads buildable.

    Args:
        section_name (str): The section whose heads are shared.
        changed_setting (str): The setting of level 7 changed (see change_level_7).
        expected_message (str): The beginning of the error message.
    """
    configuration, device, dtype = get_detection_model_configuration()
    configuration = copy.deepcopy(configuration)
    configuration["Data"]["image_settings"]["size"] = 512
    change_level_7(configuration=configuration, changed_setting=changed_setting)
    TwoStageDetector(configuration=configuration, device=device, dtype=dtype)
    configuration[section_name]["shared"] = True
    with pytest.raises(expected_exception=ValueError, match=expected_message):
        TwoStageDetector(configuration=configuration, device=device, dtype=dtype)


def test_single_level_heads_refuse_the_maps_of_other_levels() -> None:
    """
    Verifies that a per-level detection head refuses to pool the proposals of a level it does not read, and that the
    shared detection head refuses the one-level forward pass, whose map would be ambiguous.
    """
    level_model = build_test_model(dtype=torch.float32)
    shared_model = build_test_model(dtype=torch.float32, shared_detector=True)
    proposals = torch.tensor(data=[[0.0, 10.0, 10.0, 100.0, 100.0]])
    feature_map = torch.zeros(size=(1, 64, 16, 16))
    with pytest.raises(expected_exception=ValueError, match="reads the levels \\['5'\\], not level '6'"):
        level_model.detector_heads_dictionary["5"].pool_box_features(proposal_boxes=proposals,
                                                                     input_tensor=feature_map, level_name="6")
    level_names: List[str] = ["5", "6", "7"]
    with pytest.raises(expected_exception=ValueError, match=f"reads the levels {level_names}".replace("[", "\\[")):
        shared_model.detector_heads_dictionary[SHARED_HEAD_KEY](proposal_boxes=proposals, input_tensor=feature_map)
