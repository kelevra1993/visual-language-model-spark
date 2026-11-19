"""
Mask figures of the instance segmentation tutorials, made by tutorials/figures/instance_segmentation/figures.py from
the INS run of tutorials/tutorial_sources.yaml and its test split: the run length encoding of the first annotation of
the test manifest with the mask it decodes to and the box read from that mask (the chapter on the masks and their
loading); the target of the mask loss built for a real proposal of the trained region proposal network around the
star of the test image of the series, cropped and resized by build_mask_targets (the chapter on the mask head and its
loss); and the masks the trained instance segmenter predicts for the detections of that image on the CPU,
pasted into the image at their boxes and thresholded by paste_instance_masks, against the ground truth masks, with
their mask overlaps (the chapter on scoring and shipping masks). The functions of the repository decode, crop, paste
and compare the masks, so that the figures show what the code computes. The inference is reused unless forced, and a
run read from the run snapshots (tutorials/run_snapshots, no checkpoint) takes the inference committed under
figure_reuse/instance_segmentation/masks of the snapshots by a build from the real runs; the geometry of the mask heads
and of the mask evaluation quoted by both chapters (levels, strides, kernels, image size, chunk size, object size
limits) is read at every build from the model built untrained from the configuration of the run and the constants of
the repository.
"""

import math

import cv2
import numpy
import torch

from fractions import Fraction
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Tuple

from tutorials.figures.figure_utilities import (SERIES_COLOURS, read_reused_facts, read_run_configuration,
                                                restore_committed_figures, round_fact, write_reused_facts)
from tutorials.figures.object_detection.detection_sources import (CHECKPOINT_ITERATION, SAMPLE_IMAGE_NAME,
                                                                  find_test_folder, hexadecimal_to_rgb,
                                                                  load_run_model, load_untrained_run_model,
                                                                  read_category_names, read_image_annotations,
                                                                  read_network_image, read_test_manifest,
                                                                  write_listing)
from tutorial_builder.layout import (CARD_COLOUR, DIM_TEXT_COLOUR, MONOSPACE_FONT_PATH,
                                     SANS_SEMIBOLD_FONT_PATH, TEXT_COLOUR)
from vision_training.data.detection.coco_masks import build_instance_mask, compute_mask_bounding_boxes
from vision_training.evaluation.detection.object_sizes import OBJECT_SIZE_RANGES
from vision_training.losses.detection_losses import LOCALISATION_LOSS_BETA
from vision_training.models.detection.box_operations import get_intersection_over_union
from vision_training.models.instance_segmentation.mask_operations import (build_mask_targets,
                                                                          compute_mask_intersection_over_union,
                                                                          paste_instance_masks)
from vision_training.trainers.checkpointing import get_checkpoint_file
from vision_training.trainers.instance_segmentation_trainer import (MASK_OVERLAP_CHUNK_SIZE,
                                                                    MASK_OVERLAP_REFERENCE_PIXELS)

# Object of the test image whose mask target the figures build, and the overlap range of the proposal chosen for it,
# a positive proposal (IoU from 0.5) visibly off the object (below 0.7)
TARGET_OBJECT_NAME = "star"
TARGET_PROPOSAL_IOU_RANGE = (0.5, 0.7)

# Run length values written in the listing of the first annotation
RUN_LENGTH_VALUES_SHOWN = 8

# Size of the figures shown on the whole card, and the bands of their headings and notes
FIGURE_WIDTH = 1720
FIGURE_HEIGHT = 700
HEADING_HEIGHT = 64
NOTE_HEIGHT = 84
PANEL_SIDE = 552
PANEL_GAP = 32

# Opacity of the masks drawn over the images
MASK_OPACITY = 0.6

# Folder of the mask figures in the figures folder of the series, also the family committed under figure_reuse/ of the
# run snapshots for the builds without the checkpoint
MASK_FAMILY_FOLDER = "masks"


def load_figure_fonts() -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """
    Loads the fonts of the mask figures: Inter semibold for the headings of the panels, Hack for their notes.

    Returns:
        Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]: The heading font and the note font.
    """
    return (ImageFont.truetype(font=str(SANS_SEMIBOLD_FONT_PATH), size=30),
            ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=20))


def overlay_masks(image: Image.Image, masks: numpy.ndarray, colours: List[Tuple[int, int, int]]) -> Image.Image:
    """
    Draws binary masks over an image, every mask filled with its colour at MASK_OPACITY, the way the sample images of
    the evaluation fill the masks of the objects.

    Args:
        image (Image.Image): The RGB image.
        masks (numpy.ndarray): The [N, H, W] binary masks, at the size of the image.
        colours (List[Tuple[int, int, int]]): The colour of every mask.

    Returns:
        Image.Image: The image with its masks.
    """
    pixels = numpy.array(object=image, dtype=numpy.float32)
    for mask, colour in zip(masks, colours):
        selected = mask.astype(bool)
        pixels[selected] = pixels[selected] * (1.0 - MASK_OPACITY) + numpy.array(object=colour) * MASK_OPACITY
    return Image.fromarray(obj=pixels.clip(min=0, max=255).astype(numpy.uint8))


def paste_panels(panels: List[Tuple[str, str, Image.Image]], output_path: Path) -> List[List[int]]:
    """
    Places square panels side by side on a figure of the height of the card (its width for three panels), every panel
    under its heading and above its note, and gives the box of every panel.

    Args:
        panels (List[Tuple[str, str, Image.Image]]): The heading, note and image of every panel.
        output_path (Path): The PNG file of the figure.

    Returns:
        List[List[int]]: The left, top, right and bottom of every panel image.
    """
    canvas = Image.new(mode="RGB", size=(len(panels) * PANEL_SIDE + (len(panels) - 1) * PANEL_GAP, FIGURE_HEIGHT),
                       color=CARD_COLOUR)
    draw = ImageDraw.Draw(im=canvas)
    heading_font, note_font = load_figure_fonts()
    panel_boxes = []
    for panel_index, (heading, note, panel_image) in enumerate(panels):
        left = panel_index * (PANEL_SIDE + PANEL_GAP)
        canvas.paste(im=panel_image.resize(size=(PANEL_SIDE, PANEL_SIDE), resample=Image.Resampling.NEAREST),
                     box=(left, HEADING_HEIGHT))
        draw.text(xy=(left, HEADING_HEIGHT // 2), text=heading, fill=TEXT_COLOUR, font=heading_font, anchor="lm")
        for line_index, note_line in enumerate(note.split("\n")):
            draw.text(xy=(left, HEADING_HEIGHT + PANEL_SIDE + 24 + line_index * 28), text=note_line,
                      fill=DIM_TEXT_COLOUR, font=note_font, anchor="lm")
        panel_boxes.append([left, HEADING_HEIGHT, left + PANEL_SIDE, HEADING_HEIGHT + PANEL_SIDE])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(fp=output_path)
    return panel_boxes


def read_rgb_image(test_folder: Path, image_name: str) -> Image.Image:
    """
    Reads a test image in RGB for the figures, the network reading it in BGR.

    Args:
        test_folder (Path): The test split.
        image_name (str): The file name of the image.

    Returns:
        Image.Image: The RGB image.
    """
    image, _ = read_network_image(test_folder=test_folder, image_name=image_name)
    return Image.fromarray(obj=cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB))


def draw_rectangle(image: Image.Image, box: List[float], colour: Tuple[int, int, int], width: int,
                   scale: float) -> Image.Image:
    """
    Draws the outline of a box on a copy of an image, the box given in the pixels of the network input.

    Args:
        image (Image.Image): The RGB image.
        box (List[float]): The [x1, y1, x2, y2] box.
        colour (Tuple[int, int, int]): The colour of the outline.
        width (int): Its width in pixels.
        scale (float): The scale of the image relative to the network input.

    Returns:
        Image.Image: The image with the box.
    """
    drawn_image = image.copy()
    ImageDraw.Draw(im=drawn_image).rectangle(xy=[coordinate * scale for coordinate in box], outline=colour,
                                             width=width)
    return drawn_image


def compute_overlap_chunk_size(image_height: int, image_width: int) -> int:
    """
    Gives the number of detection masks the mask evaluation compares at once at an image size, as
    InstanceSegmentationTrainer.compute_mask_overlaps computes it from MASK_OVERLAP_CHUNK_SIZE masks at
    MASK_OVERLAP_REFERENCE_PIXELS, so that the figures compare the masks of the test image by the chunks of the
    evaluation and the mask IoU scene quotes the chunk of the run.

    Args:
        image_height (int): The height of the images the masks are pasted into.
        image_width (int): Their width.

    Returns:
        int: The masks compared at once, at least one.
    """
    return max(1, MASK_OVERLAP_CHUNK_SIZE * MASK_OVERLAP_REFERENCE_PIXELS // (image_height * image_width))


def make_annotation_figure(test_folder: Path, output_folder: Path) -> Dict[str, Any]:
    """
    Decodes the run length encoding of the first annotation of the test manifest with build_instance_mask, as the
    loader does, writes its listing (size, counts, the runs they stand for), and draws its image with the decoded mask
    and the box read from the mask next to the mask alone; gives the bbox of the annotation against the box of its mask.

    Args:
        test_folder (Path): The test split.
        output_folder (Path): The figures folder of the series.

    Returns:
        Dict[str, Any]: The facts of the annotation, named annotation.*.
    """
    manifest = read_test_manifest(test_folder=test_folder)
    annotation = manifest["annotations"][0]
    image_entry = [entry for entry in manifest["images"] if entry["id"] == annotation["image_id"]][0]
    category_name = read_category_names(test_folder=test_folder)[annotation["category_id"]]
    segmentation = annotation["segmentation"]
    mask = build_instance_mask(segmentation=segmentation, image_width=image_entry["width"],
                               image_height=image_entry["height"], image_size=image_entry["width"], keep_ratio=True)
    mask_box = compute_mask_bounding_boxes(instance_masks=torch.from_numpy(mask).unsqueeze(dim=0))[0].tolist()

    # Listing: the segmentation as stored, then the runs it stands for, column by column from the top left pixel
    counts = segmentation["counts"]
    run_descriptions = [f"{'background' if run_index % 2 == 0 else 'object'} {value:,}"
                        for run_index, value in enumerate(counts[:5])]
    lines = [f"test/labels.json · annotations[0] · {category_name} of {image_entry['file_name']}",
             f"  bbox {annotation['bbox']}",
             f"  segmentation size {segmentation['size']}   (height, width)",
             f"  counts [{', '.join(str(value) for value in counts[:RUN_LENGTH_VALUES_SHOWN])}, ...]",
             f"         {len(counts)} values, sum {sum(counts):,} = {segmentation['size'][0]} x "
             f"{segmentation['size'][1]}",
             f"  runs, column by column: {' · '.join(run_descriptions)} ...",
             f"decoded mask: {int(mask.sum()):,} pixels   box from the mask {[int(value) for value in mask_box]}"]
    write_listing(lines=lines, output_path=output_folder / "listings" / "run_length_encoding.txt")

    # Figure: the image with the mask and its box, then the mask alone
    image = read_rgb_image(test_folder=test_folder, image_name=image_entry["file_name"])
    colour = hexadecimal_to_rgb(colour=SERIES_COLOURS[3])
    masked_image = draw_rectangle(image=overlay_masks(image=image, masks=mask[None], colours=[colour]), box=mask_box,
                                  colour=TEXT_COLOUR, width=3, scale=1.0)
    mask_image = Image.fromarray(obj=(mask * 255).astype(numpy.uint8)).convert(mode="RGB")
    panel_boxes = paste_panels(panels=[
        (f"{category_name} of {image_entry['file_name']}", "white: the box of the mask", masked_image),
        (f"decoded mask, {mask.shape[1]} × {mask.shape[0]}", f"{int(mask.sum()):,} pixels", mask_image)],
        output_path=output_folder / "masks" / "decoded_annotation.png")
    return {"annotation.category": category_name, "annotation.image_name": image_entry["file_name"],
            "annotation.bbox": annotation["bbox"], "annotation.mask_box": [int(value) for value in mask_box],
            "annotation.mask_pixels": int(mask.sum()),
            "annotation.run_length_values": len(counts), "annotation.run_length_sum": sum(counts),
            "annotation.first_background_run": counts[0], "annotation.first_object_run": counts[1],
            "annotation.panel_1_box": panel_boxes[0], "annotation.panel_2_box": panel_boxes[1]}


def read_object_masks(test_folder: Path) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[str]]:
    """
    Builds the ground truth masks of the objects of the test image of the series at the network resolution, with the
    boxes of the masks, as the loader of the instance segmentation does (build_instance_mask, then
    compute_mask_bounding_boxes), and the class of every object.

    Args:
        test_folder (Path): The test split.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[int], List[str]]: The [N, 512, 512] uint8 masks, their [N, 4] boxes,
            the classes and the class names.
    """
    image_entry, annotations = read_image_annotations(test_folder=test_folder, image_name=SAMPLE_IMAGE_NAME)
    category_names = read_category_names(test_folder=test_folder)
    masks = torch.from_numpy(numpy.stack(arrays=[build_instance_mask(
        segmentation=annotation["segmentation"], image_width=image_entry["width"], image_height=image_entry["height"],
        image_size=image_entry["width"], keep_ratio=True) for annotation in annotations]))
    classes = [annotation["category_id"] for annotation in annotations]
    return masks, compute_mask_bounding_boxes(instance_masks=masks).float(), classes, [category_names[class_index]
                                                                                       for class_index in classes]


def make_target_figure(object_masks: torch.Tensor, object_boxes: torch.Tensor, object_names: List[str],
                       proposals: torch.Tensor, mask_size: int, image: Image.Image,
                       output_path: Path) -> Dict[str, Any]:
    """
    Builds the mask target of a real positive proposal of the trained region proposal network around TARGET_OBJECT_NAME
    with build_mask_targets (RoIAlign of the full-size ground truth mask in the proposal, at the mask resolution of the
    run, thresholded at 0.5), and draws the object with its mask and the proposal, the mask inside the proposal, and
    the target.

    Args:
        object_masks (torch.Tensor): The [N, H, W] ground truth masks.
        object_boxes (torch.Tensor): Their [N, 4] boxes.
        object_names (List[str]): Their class names.
        proposals (torch.Tensor): The [P, 4] proposals of the image in evaluation.
        mask_size (int): The side of the masks of the run ('MaskHead.mask_resolution').
        image (Image.Image): The RGB test image.
        output_path (Path): The PNG file of the figure.

    Returns:
        Dict[str, Any]: The overlap of the proposal, the foreground of the target and the panel boxes, named target.*.

    Raises:
        ValueError: If no proposal of the image overlaps the object within TARGET_PROPOSAL_IOU_RANGE.
    """
    object_index = object_names.index(TARGET_OBJECT_NAME)
    overlaps = get_intersection_over_union(boxes_1=object_boxes[object_index:object_index + 1], boxes_2=proposals)[0]
    in_range = (overlaps >= TARGET_PROPOSAL_IOU_RANGE[0]) & (overlaps < TARGET_PROPOSAL_IOU_RANGE[1])
    if not bool(in_range.any()):
        raise ValueError(f"No proposal overlaps the {TARGET_OBJECT_NAME} between {TARGET_PROPOSAL_IOU_RANGE}.")
    proposal_index = int(torch.where(condition=in_range, input=overlaps,
                                     other=torch.zeros_like(input=overlaps)).argmax())
    proposal = proposals[proposal_index]

    # Target of the proposal, the matched ground truth being the object
    target = build_mask_targets(ground_truth_masks=[object_masks],
                                proposal_boxes=torch.cat(tensors=[torch.zeros(size=(1,)), proposal]).unsqueeze(dim=0),
                                matched_ground_truth_indices=torch.tensor(data=[object_index]),
                                mask_size=mask_size)[0]

    # Panels: the object with its mask and the proposal, the crop of the mask in the proposal, and the target
    colour = hexadecimal_to_rgb(colour=SERIES_COLOURS[4])
    object_mask = object_masks[object_index].numpy()
    with_proposal = draw_rectangle(image=overlay_masks(image=image, masks=object_mask[None], colours=[colour]),
                                   box=proposal.tolist(), colour=hexadecimal_to_rgb(colour=SERIES_COLOURS[2]),
                                   width=4, scale=1.0)
    left, top, right, bottom = [int(round(value)) for value in proposal.tolist()]
    crop = Image.fromarray(obj=(object_mask * 255).astype(numpy.uint8)).crop(box=(left, top, right, bottom))
    side = max(crop.width, crop.height)
    target_image = Image.fromarray(obj=(target.numpy() * 255).astype(numpy.uint8))
    panel_boxes = paste_panels(panels=[
        (f"{TARGET_OBJECT_NAME} and a positive proposal", f"proposal IoU {float(overlaps[proposal_index]):.2f}",
         with_proposal),
        ("ground truth mask in the proposal", f"{right - left} × {bottom - top} pixels",
         crop.resize(size=(side, side), resample=Image.Resampling.NEAREST).convert(mode="RGB")),
        (f"target, {mask_size} × {mask_size}", f"RoIAlign aligned, >= 0.5: {int(target.sum())} of "
                                                f"{mask_size ** 2} pixels",
         target_image.convert(mode="RGB"))], output_path=output_path)
    return {"target.object": TARGET_OBJECT_NAME,
            "target.proposal_iou": round_fact(value=float(overlaps[proposal_index])),
            "target.proposal_box": [round_fact(value=value) for value in proposal.tolist()],
            "target.foreground_pixels": int(target.sum()), "target.panel_1_box": panel_boxes[0],
            "target.panel_2_box": panel_boxes[1], "target.panel_3_box": panel_boxes[2]}


def make_prediction_figures(outputs: Dict[str, Any], object_masks: torch.Tensor, object_names: List[str],
                            category_names: Dict[int, str], mask_threshold: float, image: Image.Image,
                            output_folder: Path) -> Dict[str, Any]:
    """
    Draws the mask probabilities of the detections of the test image (final_detection_masks), each under its class
    and score; then the ground truth masks, the detection masks pasted into the image at their boxes and thresholded
    at the mask threshold (paste_instance_masks), and the best mask overlap of every detection
    (compute_mask_intersection_over_union, by the chunks of the evaluation), the computation of the mask evaluation.

    Args:
        outputs (Dict[str, Any]): The outputs of the instance segmenter in evaluation mode for the image.
        object_masks (torch.Tensor): The [N, H, W] ground truth masks.
        object_names (List[str]): Their class names.
        category_names (Dict[int, str]): The name of every class.
        mask_threshold (float): The mask threshold of the run ('MaskHead.inference.mask_threshold').
        image (Image.Image): The RGB test image.
        output_folder (Path): The folder of the figures.

    Returns:
        Dict[str, Any]: The detections with their scores, levels, mask and box overlaps, and the panel boxes, named
            predictions.*.
    """
    detection_count = int(outputs["number_detections"][0])
    boxes = outputs["final_detection_boxes"][0][:detection_count]
    masks = outputs["final_detection_masks"][0][:detection_count]
    labels = [category_names[int(label)] for label in outputs["final_detection_labels"][0][:detection_count]]
    scores = [float(score) for score in outputs["final_detection_scores"][0][:detection_count]]
    levels = torch.gather(input=outputs["sliced_origins"], dim=1,
                          index=outputs["final_detection_proposal_indices"])[0][:detection_count]

    # Pasted and thresholded masks, compared with the ground truth masks as the evaluation does
    pasted = paste_instance_masks(masks=masks, boxes=boxes, image_height=image.height, image_width=image.width)
    binary = pasted >= mask_threshold
    mask_overlaps = compute_mask_intersection_over_union(
        predicted_masks=binary, ground_truth_masks=object_masks,
        chunk_size=compute_overlap_chunk_size(image_height=image.height, image_width=image.width))
    box_overlaps = get_intersection_over_union(boxes_1=boxes, boxes_2=compute_mask_bounding_boxes(
        instance_masks=object_masks).float())

    # Figure of the masks in the frame of their boxes, one tile per detection, the probabilities from black to white
    heading_font, note_font = load_figure_fonts()
    tile_side, tile_gap = 300, 30
    canvas = Image.new(mode="RGB", size=(FIGURE_WIDTH, FIGURE_HEIGHT), color=CARD_COLOUR)
    draw = ImageDraw.Draw(im=canvas)
    tiles_left = (FIGURE_WIDTH - detection_count * tile_side - (detection_count - 1) * tile_gap) // 2
    tile_boxes = []
    for detection_index in range(detection_count):
        left = tiles_left + detection_index * (tile_side + tile_gap)
        tile = Image.fromarray(obj=(masks[detection_index].numpy() * 255).astype(numpy.uint8)).resize(
            size=(tile_side, tile_side), resample=Image.Resampling.NEAREST)
        canvas.paste(im=tile.convert(mode="RGB"), box=(left, 200))
        draw.text(xy=(left, 160), text=f"{labels[detection_index]} {scores[detection_index]:.2f}", fill=TEXT_COLOUR,
                  font=heading_font, anchor="lm")
        draw.text(xy=(left, 200 + tile_side + 30), text=f"block {int(levels[detection_index])}", fill=DIM_TEXT_COLOUR,
                  font=note_font, anchor="lm")
        tile_boxes.append([left, 140, left + tile_side, 200 + tile_side + 50])
    draw.text(xy=(tiles_left, 60), text=f"final_detection_masks {list(outputs['final_detection_masks'].shape)}: the "
                                        f"sigmoid of the detected class",
              fill=TEXT_COLOUR, font=heading_font, anchor="lm")
    canvas.save(fp=output_folder / "predicted_masks.png")

    # Figure of the pasted masks against the ground truth masks
    colours = [hexadecimal_to_rgb(colour=SERIES_COLOURS[index % len(SERIES_COLOURS)]) for index in range(
        max(detection_count, len(object_names)))]
    best_objects = mask_overlaps.argmax(dim=1)
    ground_truth_image = overlay_masks(image=image, masks=object_masks.numpy(), colours=colours[:len(object_names)])
    detection_image = overlay_masks(image=image, masks=binary.numpy(), colours=[colours[int(index)] for index in
                                                                                best_objects])
    pixel_probabilities = Image.fromarray(obj=(pasted.max(dim=0).values.numpy() * 255).astype(numpy.uint8))
    panel_boxes = paste_panels(panels=[
        ("ground truth masks", f"{len(object_names)} objects", ground_truth_image),
        ("pasted probabilities", "paste_instance_masks at the boxes", pixel_probabilities.convert(mode="RGB")),
        (f"masks >= {mask_threshold:g} · {detection_count} detections", "mask IoU " + " · ".join(
            f"{value:.2f}" for value in mask_overlaps.max(dim=1).values.tolist()), detection_image)],
        output_path=output_folder / "pasted_masks.png")

    facts = {"predictions.detection_count": detection_count, "predictions.object_count": len(object_names),
             "predictions.mask_shape": list(outputs["final_detection_masks"].shape)}
    for detection_index in range(detection_count):
        prefix = f"predictions.detection_{detection_index + 1}"
        facts.update({f"{prefix}.name": labels[detection_index], f"{prefix}.score": round_fact(
                          value=scores[detection_index]),
                      f"{prefix}.level": int(levels[detection_index]),
                      f"{prefix}.mask_iou": round_fact(value=float(mask_overlaps[detection_index].max())),
                      f"{prefix}.box_iou": round_fact(value=float(box_overlaps[detection_index].max())),
                      f"{prefix}.object": object_names[int(best_objects[detection_index])],
                      f"{prefix}.tile_box": tile_boxes[detection_index]})
    for panel_index, panel_box in enumerate(panel_boxes):
        facts[f"predictions.panel_{panel_index + 1}_box"] = panel_box
    return facts


def count_mask_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Gives the parameters of the mask heads of the trained instance segmenter and of the whole model, and the geometry
    of the mask head of every level.

    Args:
        model (torch.nn.Module): The trained InstanceSegmenter.

    Returns:
        Dict[str, Any]: The facts, named model.*.
    """
    head = model.mask_heads_dictionary["5"]
    return {"model.total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "model.mask_parameters": sum(parameter.numel() for parameter in model.mask_heads_dictionary.parameters()),
            "model.mask_level_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "model.detector_parameters": sum(parameter.numel() for name, parameter in model.named_parameters()
                                             if not name.startswith("mask_heads_dictionary")),
            "model.pool_size": head.roi_align_pool_size, "model.upsamplings": head.number_upsamplings,
            "model.mask_size": head.mask_size, "model.number_classes": model.number_classes}


def describe_mask_geometry(run_folder: Path) -> Dict[str, Any]:
    """
    Gives the geometry of the mask heads and of the mask evaluation that the chapters quote and that no configuration
    value holds as it is, read from the instance segmenter built untrained from the configuration of the run (not run,
    none of these values depends on the weights, so that no checkpoint is needed) and from the constants of the
    repository: the detection levels and the strides their mask heads pool from, the feature map of the first
    level in a training batch, the kernels of the upsampling and of the mask predictor, the pixels of a mask, the masks
    compared at once by compute_mask_overlaps at the network resolution of the run (and that resolution), the object
    size limits of the size tables and the beta of the smooth L1 box loss. Cheap, it runs at every build, outside the
    reused inference of make_mask_figures, so that its facts never come from an older build.

    Args:
        run_folder (Path): The folder of the INS run of tutorial_sources.yaml.

    Returns:
        Dict[str, Any]: The facts, named geometry.*.

    Raises:
        ValueError: If the mask heads double their features otherwise than by a transposed convolution, which the
            diagrams of the mask head draw.
    """
    run_configuration = read_run_configuration(run_folder=run_folder)
    image_size = run_configuration["Data"]["image_settings"]["size"]
    batch_size = run_configuration["ExperimentConfiguration"]["batch_size"]
    model = load_untrained_run_model(run_folder=run_folder)

    # Mask head of every detection level, in the order of the levels, and the strides of their feature maps
    level_names = list(model.mask_heads_dictionary.keys())
    first_head = model.mask_heads_dictionary[level_names[0]]
    strides = sorted({round(1.0 / head.spatial_scale) for head in model.mask_heads_dictionary.values()})
    stride_text = (", ".join(str(stride) for stride in strides[:-1]) + " and " + str(strides[-1]) if len(strides) > 1
                   else str(strides[0]))

    # Doubling layer of the first head, drawn as a transposed convolution by the diagrams
    if first_head.number_upsamplings == 0 or not isinstance(first_head.upsampling_layers[0], torch.nn.ConvTranspose2d):
        raise ValueError(f"The mask heads of {run_folder} do not double their features with a transposed "
                         f"convolution, which the mask head diagrams of the chapters show.")
    upsampling_layer = first_head.upsampling_layers[0]

    # Masks compared at once by the evaluation at the network resolution of the run
    overlap_chunk_size = compute_overlap_chunk_size(image_height=image_size, image_width=image_size)
    return {"geometry.level_names": ", ".join(level_names), "geometry.level_count": len(level_names),
            "geometry.image_size": image_size,
            "geometry.first_level": int(level_names[0]),
            "geometry.first_level_channels": first_head.convolutional_block.input_channels,
            "geometry.first_level_shape": f"[{batch_size}, {first_head.convolutional_block.input_channels}, "
                                          f"{first_head.feature_map_size}, {first_head.feature_map_size}]",
            "geometry.strides_text": stride_text,
            "geometry.upsampling_kernel": upsampling_layer.kernel_size[0],
            "geometry.upsampling_stride": upsampling_layer.stride[0],
            "geometry.predictor_kernel": first_head.mask_predictor.kernel_size[0],
            "geometry.mask_pixels": first_head.mask_size ** 2, "geometry.overlap_chunk_size": overlap_chunk_size,
            "geometry.small_object_limit": f"{int(math.sqrt(OBJECT_SIZE_RANGES['small'][1]))}²",
            "geometry.large_object_limit": f"{int(math.sqrt(OBJECT_SIZE_RANGES['large'][0]))}²",
            "geometry.box_loss_beta": str(Fraction(LOCALISATION_LOSS_BETA).limit_denominator(max_denominator=100))}


def make_mask_figures(output_folder: Path, run_folder: Path, force: bool) -> Dict[str, Any]:
    """
    Makes the annotation, target and prediction figures of the instance segmentation series, the inference of the
    trained instance segmenter on the CPU being reused from the previous build unless forced; a run without its
    checkpoint (read from the run snapshots) reuses the inference committed in the snapshots, copied into the figures
    folder by restore_committed_figures.

    Args:
        output_folder (Path): The figures folder of the series.
        run_folder (Path): The folder of the INS run of tutorial_sources.yaml.
        force (bool): Whether the inference is run again even when the previous build left its figures.

    Returns:
        Dict[str, Any]: The facts, named annotation.*, target.*, predictions.* and model.*.
    """
    figure_folder = output_folder / MASK_FAMILY_FOLDER
    facts_path = figure_folder / "mask_facts.json"
    test_folder = find_test_folder(run_configuration=read_run_configuration(run_folder=run_folder))
    facts = make_annotation_figure(test_folder=test_folder, output_folder=output_folder)

    # A run without the checkpoint (read from the run snapshots) takes the inference committed by a build from the real
    # runs, which cannot be made again, forced or not
    checkpoint_path = get_checkpoint_file(weights_directory=run_folder / "Weights", iteration=CHECKPOINT_ITERATION)
    checkpoint_missing = not checkpoint_path.exists()
    if checkpoint_missing:
        if force:
            print(f"No checkpoint {checkpoint_path}: --force cannot run the instance segmenter again, the committed "
                  f"mask figures of the run snapshots are reused")
        restore_committed_figures(output_folder=output_folder, family_folder=MASK_FAMILY_FOLDER)
    reused_facts = read_reused_facts(facts_path=facts_path, required_paths=[], force=force and not checkpoint_missing)
    if reused_facts is not None:
        facts.update(reused_facts)
        return facts
    if checkpoint_missing:
        raise FileNotFoundError(f"No checkpoint {checkpoint_path} and no committed mask figures for {facts_path}: "
                                f"build the instance_segmentation figures from the real runs, then write the snapshots "
                                f"again (tutorials/make_run_snapshots.py --series instance_segmentation).")

    # The trained instance segmenter on the test image of the series, in evaluation mode
    model = load_run_model(run_folder=run_folder, iteration=CHECKPOINT_ITERATION)
    object_masks, object_boxes, _, object_names = read_object_masks(test_folder=test_folder)
    image = read_rgb_image(test_folder=test_folder, image_name=SAMPLE_IMAGE_NAME)
    _, input_tensor = read_network_image(test_folder=test_folder, image_name=SAMPLE_IMAGE_NAME)
    with torch.no_grad():
        outputs = model(input_tensor=input_tensor)

    # Target of a real proposal, the predicted masks and their pasting
    inference_facts = make_target_figure(object_masks=object_masks, object_boxes=object_boxes,
                                         object_names=object_names, proposals=outputs["proposal_boxes"][0],
                                         mask_size=model.mask_size, image=image,
                                         output_path=figure_folder / "mask_target.png")
    inference_facts.update(make_prediction_figures(outputs=outputs, object_masks=object_masks,
                                                   object_names=object_names,
                                                   category_names=read_category_names(test_folder=test_folder),
                                                   mask_threshold=model.mask_threshold, image=image,
                                                   output_folder=figure_folder))
    inference_facts.update(count_mask_parameters(model=model))
    write_reused_facts(facts_path=facts_path, figure_paths=[figure_folder / "mask_target.png",
                                                            figure_folder / "predicted_masks.png",
                                                            figure_folder / "pasted_masks.png"],
                       facts=inference_facts)
    facts.update(inference_facts)
    return facts
