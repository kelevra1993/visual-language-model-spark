"""
DataAugmenter, the random geometric and photometric augmentations applied on the device to the training batches of
every task right before the forward pass, moving the bounding boxes of the object detector, the masks of the semantic
segmentation, the instance masks of the instance segmentation and the keypoints of the keypoint detection with the
images and handling the frame shuffling and per-frame draws of the sequence tasks.
Built by the base trainer from the Data.data_augmentation configuration, whose augmentation types are verified by the
common checks. Every augmented batch comes with a report of the types drawn for each sample and their values, which
the 'view_input_images' diagnosis mode writes in the information panel next to every training image.
"""

import random
import warnings
import torch

from typing import Any, Dict, List, Optional, Tuple, TypedDict
from torch.nn import functional as torch_functional
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as vision_functional
from torchvision.transforms.v2 import InterpolationMode

from vision_training.core.contracts import IGNORED_LABEL_INDEX
from vision_training.data.detection.coco_masks import compute_mask_bounding_boxes

# kornia decorates some of its functions with torch.jit.script when it is imported, which recent PyTorch versions
# report as deprecated; the notice is harmless and cannot be fixed on our side, so it is silenced for this import only
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
    from kornia.color import rgb_to_yuv, yuv_to_rgb
    from kornia.enhance import equalize_clahe

# Geometric augmentations, applied first, in this order. They move the bounding boxes and the masks along with the
# image. random_zoom_out follows random_crop, so that a sample drawing both is cropped first and then shrunk, its two
# draws being independent, and random_translation then shifts the result
GEOMETRIC_AUGMENTATION_TYPES = ["random_crop", "random_zoom_out", "random_translation", "random_left_right_flip",
                                "random_up_down_flip", "random_rotation", "random_90_rotation"]

# Photometric augmentations, applied after the geometric ones, in this order. They never move the bounding boxes and
# never change the masks
PHOTOMETRIC_AUGMENTATION_TYPES = ["brightness_contrast_saturation", "clahe_augment", "sharp_augment",
                                  "equalize_histogram_augment", "equalize_bgr_histogram_augment", "random_noise",
                                  "gray_scale_augment", "random_colour_augmentation", "random_blur"]

AUGMENTATION_TYPES = GEOMETRIC_AUGMENTATION_TYPES + PHOTOMETRIC_AUGMENTATION_TYPES

# Fixed augmentation strengths, documented in the data_augmentation section of the configuration file
MINIMUM_CROP_FRACTION = 0.75
MINIMUM_ZOOM_OUT_FRACTION = 0.5
MAXIMUM_TRANSLATION_FRACTION = 0.25
MAXIMUM_ROTATION_DEGREES = 5.0
BRIGHTNESS_MAXIMUM_DELTA = 40.0
CONTRAST_FACTOR_RANGE = (0.7, 1.0)
SATURATION_FACTOR_RANGE = (0.7, 1.0)
NOISE_STANDARD_DEVIATION = 40.0
BLUR_KERNEL_SIZE = 9
BLUR_SIGMA = 2.5
SHARPENING_SIGMA = 1.5
SHARPENING_STRENGTH = 3.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (4, 4)

# Axes a translation shifts the image along, one of them drawn with equal chances: the width only, the height only, or
# both at once, each shift being drawn independently up to MAXIMUM_TRANSLATION_FRACTION of its side
TRANSLATION_AXES = ["x", "y", "xy"]

# Boxes narrower or shorter than this many pixels after a geometric augmentation are removed with their label
MINIMUM_BOX_SIDE = 1.0

# Objects keeping less than this share of their area inside the image after a geometric augmentation (box area, or
# mask pixels for the instance segmentation) are removed with their label, so that the detector is never trained to
# find a fragment cut by the border of a crop or a rotation
MINIMUM_VISIBLE_AREA_FRACTION = 0.25

# Channel indices of the BGR images delivered by the data loaders
BGR_CHANNEL_INDICES = {"B": [0], "G": [1], "R": [2], "BGR": [0, 1, 2]}


class AugmentationReport(TypedDict):
    """
    Record of the augmentations drawn for one training sample by DataAugmenter.augment_batch, kept in the batch by the
    trainers and written by describe_report in the information panel of the input viewer ('view_input_images'), so
    that the user sees which augmentations every training image received and how strong they were. 'augmented' is
    False when the global probability of the sample was not drawn (or the augmentation is off). Every applied type is a
    (type name, drawn values) pair, the values being empty for the types without random values (e.g. a flip). The
    photometric types hold one list for the sample, or one per frame when the frames of a sequence are drawn on their
    own ('independent_frames'). 'removed_labels' gives the labels of the objects removed after the geometric
    augmentations, None for the tasks without boxes.
    """
    augmented: bool
    geometric_augmentations: List[Tuple[str, str]]
    photometric_augmentations: List[List[Tuple[str, str]]]
    removed_labels: Optional[torch.Tensor]


def create_empty_report() -> AugmentationReport:
    """
    Creates the report of a sample that received no augmentation, filled by DataAugmenter.augment_sample when the
    sample is augmented.

    Returns:
        AugmentationReport: The report without any applied type.
    """
    return {"augmented": False, "geometric_augmentations": [], "photometric_augmentations": [],
            "removed_labels": None}


class DataAugmenter:
    """
    Randomly augments training batches on their device (usually the GPU), right before the forward pass.

    It is shared by every network of the project: the object detector passes bounding boxes and labels, which are
    moved, clipped and filtered together with the geometric augmentations, the semantic segmenter passes the class
    masks of its images, moved with nearest interpolation so that every pixel keeps a class, the instance segmenter
    passes the binary mask of every object with its boxes, the boxes then becoming the tight boxes of the moved masks,
    the keypoint detector passes the keypoints of every object with its boxes, moved without being renamed, while a
    classifier simply omits them.
    Images are float BGR tensors in the [0, 255] range, either single images [B, C, H, W] or sequences
    [B, T, C, H, W] for the sequence networks, in which case every frame of a sequence receives exactly the same
    random augmentation so that the sequence stays coherent, unless the photometric types are drawn frame by frame
    ('independent_frames'). The real frames of the windows can also be shuffled ('shuffle_frames', see
    shuffle_frames) for the sequences whose frame order carries no information.
    """

    def __init__(self, augmentation_configuration: Dict[str, Any]) -> None:
        """
        Reads the 'Data.data_augmentation' configuration section.

        Args:
            augmentation_configuration (Dict[str, Any]): The section holding the global 'activated' flag, the
                'probability' of augmenting an image, the 'independent_frames' flag and the 'shuffle_frames' settings
                of the sequences and, for every augmentation type, its own 'activated' flag and 'probability' of
                being applied to an augmented image.
        """
        self.activated = augmentation_configuration["activated"]
        self.probability = augmentation_configuration["probability"]

        # Sequences only: the photometric types drawn frame by frame, and the shuffling of the frames of a window
        self.independent_frames = augmentation_configuration["independent_frames"]
        self.shuffle_frames_activated = augmentation_configuration["shuffle_frames"]["activated"]
        self.shuffle_frames_probability = augmentation_configuration["shuffle_frames"]["probability"]

        # Keep the activation flag and probability of every supported augmentation type
        types_configuration = augmentation_configuration["types"]
        self.type_settings = {name: (types_configuration[name]["activated"], types_configuration[name]["probability"])
                              for name in AUGMENTATION_TYPES}

    def augment_batch(self, images: torch.Tensor, bounding_boxes: Optional[List[torch.Tensor]] = None,
                      labels: Optional[List[torch.Tensor]] = None, masks: Optional[torch.Tensor] = None,
                      instance_masks: Optional[List[torch.Tensor]] = None,
                      keypoints: Optional[List[torch.Tensor]] = None) -> Tuple[
                      torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[torch.Tensor],
                      Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], List[AugmentationReport]]:
        """
        Augments every sample of a training batch independently, each one with probability 'probability', and reports
        the types drawn for every sample with their values, for the information panels of the input viewer.

        Args:
            images (torch.Tensor): Float BGR images in [0, 255], of shape [B, C, H, W] or [B, T, C, H, W].
            bounding_boxes (Optional[List[torch.Tensor]]): Per-image [N, 4] boxes in absolute [x1, y1, x2, y2]
                pixel coordinates, or None for networks without boxes. Defaults to None.
            labels (Optional[List[torch.Tensor]]): Per-image [N] labels matching the boxes. Defaults to None.
            masks (Optional[torch.Tensor]): The [B, H, W] int64 class index of every pixel of the images, or None
                for networks without masks. Defaults to None.
            instance_masks (Optional[List[torch.Tensor]]): Per-image [N, H, W] uint8 binary masks matching the boxes
                (instance segmentation), or None. Defaults to None.
            keypoints (Optional[List[torch.Tensor]]): Per-image [N, K, 3] float keypoints (x, y, visibility) of the
                objects of the boxes (keypoint detection), or None. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], Optional[torch.Tensor],
                Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], List[AugmentationReport]]: The augmented
                images, boxes, labels, masks, instance masks and keypoints, with the same structure as the inputs, and
                the augmentation report of every sample.

        Raises:
            ValueError: If boxes, masks or keypoints are given with sequences, boxes and labels are not given together,
                or instance masks or keypoints are given without boxes, the instance masks also with class masks.
        """
        if not self.activated:
            return (images, bounding_boxes, labels, masks, instance_masks, keypoints,
                    [create_empty_report() for sample_index in range(images.shape[0])])

        # Boxes and masks are only supported for single images, and the boxes always travel together with their labels
        if images.dim() == 5 and (bounding_boxes is not None or masks is not None or instance_masks is not None or
                                  keypoints is not None):
            raise ValueError("Bounding boxes, masks and keypoints are not supported for image sequences "
                             "[B, T, C, H, W].")
        if (bounding_boxes is None) != (labels is None):
            raise ValueError("Bounding boxes and labels must be given together.")

        # The instance masks belong to the boxes, and share the mask argument of the geometric methods
        if instance_masks is not None and (bounding_boxes is None or masks is not None):
            raise ValueError("Instance masks must be given with their bounding boxes and without class masks.")

        # The keypoints belong to the objects of the boxes, removed with them
        if keypoints is not None and bounding_boxes is None:
            raise ValueError("Keypoints must be given with the bounding boxes of their objects.")

        augmented_images, augmented_boxes, augmented_labels, augmented_masks = [], [], [], []
        augmented_instance_masks, augmented_keypoints, augmentation_reports = [], [], []

        # Decide independently for every sample whether it is augmented, an unaugmented sample keeping an empty report
        for sample_index in range(images.shape[0]):
            sample_image = images[sample_index]
            sample_boxes = bounding_boxes[sample_index] if bounding_boxes is not None else None
            sample_labels = labels[sample_index] if labels is not None else None
            sample_mask = masks[sample_index] if masks is not None else None
            sample_instance_masks = instance_masks[sample_index] if instance_masks is not None else None
            sample_keypoints = keypoints[sample_index] if keypoints is not None else None
            sample_report = create_empty_report()

            if random.random() < self.probability:
                (sample_image, sample_boxes, sample_labels, sample_mask, sample_instance_masks, sample_keypoints,
                 sample_report) = self.augment_sample(image=sample_image, bounding_boxes=sample_boxes,
                                                      labels=sample_labels, mask=sample_mask,
                                                      instance_masks=sample_instance_masks, keypoints=sample_keypoints)

            augmented_images.append(sample_image)
            augmented_boxes.append(sample_boxes)
            augmented_labels.append(sample_labels)
            augmented_masks.append(sample_mask)
            augmented_instance_masks.append(sample_instance_masks)
            augmented_keypoints.append(sample_keypoints)
            augmentation_reports.append(sample_report)

        output_boxes = augmented_boxes if bounding_boxes is not None else None
        output_labels = augmented_labels if labels is not None else None
        output_masks = torch.stack(tensors=augmented_masks) if masks is not None else None
        output_instance_masks = augmented_instance_masks if instance_masks is not None else None
        output_keypoints = augmented_keypoints if keypoints is not None else None

        return (torch.stack(tensors=augmented_images), output_boxes, output_labels, output_masks, output_instance_masks,
                output_keypoints, augmentation_reports)

    def augment_sample(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                       labels: Optional[torch.Tensor], mask: Optional[torch.Tensor],
                       instance_masks: Optional[torch.Tensor] = None,
                       keypoints: Optional[torch.Tensor] = None) -> Tuple[
                       torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
                       Optional[torch.Tensor], Optional[torch.Tensor], AugmentationReport]:
        """
        Applies every activated augmentation type to one sample, each with its own probability, and reports the
        applied types with their drawn values. A sequence [T, C, H, W] is transformed as a whole, so all its frames
        share the same random parameters, except the photometric types with 'independent_frames', drawn for every
        frame. The keypoints of the keypoint detection move with the image as boxes of zero size, so that every
        geometric type moves them exactly, and keep their names: a flip or a rotation never renames a keypoint, the
        user choosing the geometric types that suit the keypoints of the dataset.

        Args:
            image (torch.Tensor): A float BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] absolute [x1, y1, x2, y2] boxes, or None.
            labels (Optional[torch.Tensor]): The [N] labels of the boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] int64 class index of every pixel, or None.
            instance_masks (Optional[torch.Tensor]): The [N, H, W] uint8 binary masks of the boxes, or None. Defaults
                to None.
            keypoints (Optional[torch.Tensor]): The [N, K, 3] float keypoints (x, y, visibility) of the objects of the
                boxes, or None. Defaults to None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
                Optional[torch.Tensor], Optional[torch.Tensor], AugmentationReport]: The augmented image, boxes,
                labels, mask, instance masks and keypoints, and the report of the applied types.
        """
        report = create_empty_report()
        report["augmented"] = True

        # The keypoints travel as [x, y, x, y] boxes of zero size after the boxes of the objects, split again after the
        # geometric types
        number_objects = bounding_boxes.shape[0] if bounding_boxes is not None else 0
        if keypoints is not None:
            point_boxes = keypoints[..., :2].reshape(-1, 2).to(dtype=bounding_boxes.dtype).repeat(1, 2)
            bounding_boxes = torch.cat(tensors=[bounding_boxes, point_boxes], dim=0)

        # The instance masks go through the mask argument of the geometric methods, except for an image without objects,
        # their pixels counted first to measure the share of every object left inside the image afterwards
        moves_instance_masks = instance_masks is not None and instance_masks.shape[0] > 0
        if moves_instance_masks:
            mask = instance_masks
            original_mask_pixels = instance_masks.flatten(start_dim=1).sum(dim=1, dtype=torch.float32)

        # Apply the geometric augmentations, which move the boxes and the mask together with the image, the area scale
        # of the objects (the zoom of a crop or a zoom-out) multiplying along the way, every applied type being
        # reported with its drawn values
        geometry_changed = False
        area_scale = 1.0
        for augmentation_type in GEOMETRIC_AUGMENTATION_TYPES:
            if self.should_apply(augmentation_type=augmentation_type):
                augmentation_method = getattr(self, augmentation_type)
                image, bounding_boxes, mask, type_area_scale, drawn_values = augmentation_method(
                    image=image, bounding_boxes=bounding_boxes, mask=mask)
                report["geometric_augmentations"].append((augmentation_type, drawn_values))
                area_scale *= type_area_scale
                geometry_changed = True

        # The moved keypoints taken back from their boxes of zero size, with their visibilities
        if keypoints is not None:
            moved_points = bounding_boxes[number_objects:, :2].reshape(*keypoints.shape[:2], 2)
            keypoints = torch.cat(tensors=[moved_points.to(dtype=keypoints.dtype), keypoints[..., 2:]], dim=-1)
            bounding_boxes = bounding_boxes[:number_objects]

        # The moved instance masks give the exact boxes of their objects, tighter than the moved boxes, and the share of
        # the pixels every object kept inside the image, out of those it would have at the new scale
        visible_fractions = None
        if moves_instance_masks:
            instance_masks, mask = mask, None
            if geometry_changed:
                bounding_boxes = compute_mask_bounding_boxes(instance_masks=instance_masks).to(
                    dtype=bounding_boxes.dtype)
                visible_mask_pixels = instance_masks.flatten(start_dim=1).sum(dim=1, dtype=torch.float32)
                expected_mask_pixels = original_mask_pixels * area_scale
                visible_fractions = torch.where(expected_mask_pixels > 0.0,
                                                visible_mask_pixels / expected_mask_pixels.clamp(min=1e-6),
                                                torch.ones_like(input=visible_mask_pixels))

        # Clip the boxes to the image and remove the ones that left it or kept too small a share of their area inside
        # it, together with their labels, masks and keypoints, the labels of the removed objects being reported
        if bounding_boxes is not None:
            bounding_boxes, labels, instance_masks, keypoints, report["removed_labels"] = self.clean_bounding_boxes(
                bounding_boxes=bounding_boxes, labels=labels, image_height=image.shape[-2],
                image_width=image.shape[-1], instance_masks=instance_masks, visible_fractions=visible_fractions,
                keypoints=keypoints)

        # Apply the photometric augmentations, which never move the boxes nor change the mask, once for the sample or
        # frame by frame, the report keeping one list of applied types per draw
        if image.dim() == 4 and self.independent_frames:
            augmented_frames = [self.apply_photometric_augmentations(image=frame) for frame in image]
            image = torch.stack(tensors=[augmented_frame for augmented_frame, frame_types in augmented_frames])
            report["photometric_augmentations"] = [frame_types for augmented_frame, frame_types in augmented_frames]
        else:
            image, applied_types = self.apply_photometric_augmentations(image=image)
            report["photometric_augmentations"] = [applied_types]

        return image.clamp(min=0.0, max=255.0), bounding_boxes, labels, mask, instance_masks, keypoints, report

    def apply_photometric_augmentations(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[str, str]]]:
        """
        Applies every activated photometric augmentation type to an image, a sequence or one frame of a sequence, each
        with its own probability and random parameters.

        Args:
            image (torch.Tensor): A float BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, List[Tuple[str, str]]]: The augmented image or sequence, and the applied types with
                their drawn values, in order.
        """
        applied_types = []
        for augmentation_type in PHOTOMETRIC_AUGMENTATION_TYPES:
            if self.should_apply(augmentation_type=augmentation_type):
                image, drawn_values = getattr(self, augmentation_type)(image=image)
                applied_types.append((augmentation_type, drawn_values))
        return image, applied_types

    def shuffle_frames(self, images: torch.Tensor, sequence_mask: torch.Tensor,
                       frame_block_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Shuffles the real frames of the windows of a training batch of sequences, for the sequences whose frame order
        carries no information (e.g. sets of photos): every window is shuffled with the 'shuffle_frames' probability,
        independently of the global augmentation probability, its real frames being permuted and its padding frames
        staying last. The frames are shuffled within every block of frame_block_length frames, so that each member
        window of a pair is shuffled on its own.

        Args:
            images (torch.Tensor): The [B, T, C, H, W] frames of the batch.
            sequence_mask (torch.Tensor): The [B, T] booleans, True for the real frames, which come first in every
                block.
            frame_block_length (int): The number of frames of a window, T being a multiple of it.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The shuffled [B, T, C, H, W] frames and the [B, T] frame order, the
                index of the original frame now at every position, to move the frame labels along.
        """
        batch_size, number_frames = sequence_mask.shape
        frame_orders = torch.arange(number_frames, device=images.device).repeat(batch_size, 1)
        if not self.activated or not self.shuffle_frames_activated:
            return images, frame_orders

        # Every window drawn on its own, its real frames permuted among themselves
        for sample_index in range(batch_size):
            for block_start in range(0, number_frames, frame_block_length):
                if random.random() >= self.shuffle_frames_probability:
                    continue
                block_end = block_start + frame_block_length
                number_real_frames = int(sequence_mask[sample_index, block_start:block_end].sum())
                frame_orders[sample_index, block_start:block_start + number_real_frames] = torch.randperm(
                    n=number_real_frames, device=images.device) + block_start

        # Frames gathered in their new order, sample by sample
        sample_indices = torch.arange(batch_size, device=images.device)[:, None]
        return images[sample_indices, frame_orders], frame_orders

    def should_apply(self, augmentation_type: str) -> bool:
        """
        Draws whether an augmentation type is applied to the current sample.

        Args:
            augmentation_type (str): The name of the augmentation type.

        Returns:
            bool: True if the type is activated and its random draw falls below its probability.
        """
        is_activated, probability = self.type_settings[augmentation_type]
        return is_activated and random.random() < probability

    def describe_report(self, report: AugmentationReport, label_names: Optional[Dict[int, str]] = None,
                        real_frames: Optional[List[int]] = None) -> List[str]:
        """
        Writes the augmentation report of a training sample as the lines of its information panel in the input viewer
        ('view_input_images'): the applied geometric types, then the photometric ones (one line per frame when they are
        drawn frame by frame), each with its drawn values, and the objects removed by the geometric augmentations.

        Args:
            report (AugmentationReport): The report of the sample, given by augment_batch.
            label_names (Optional[Dict[int, str]]): The class name of every label index, to name the removed objects.
                Defaults to None, in which case the label indices are written.
            real_frames (Optional[List[int]]): The indices of the frames written in the frame by frame lines of a
                sequence, its real frames, so that its padding frames are left out. Defaults to None, in which case
                every frame is written.

        Returns:
            List[str]: The lines of the panel, the types indented under their group.
        """
        if not self.activated:
            return ["Augmentation off (data_augmentation.activated False)"]
        if not report["augmented"]:
            return [f"Not augmented (global probability {self.probability} not drawn)"]

        # Geometric types, shared by every frame of a sequence
        lines = ["Geometric :"]
        lines += [f"  {augmentation_type} : {drawn_values}" if drawn_values else f"  {augmentation_type}"
                  for augmentation_type, drawn_values in report["geometric_augmentations"]] or ["  none"]

        # Photometric types, drawn once for the sample or frame by frame
        photometric_augmentations = report["photometric_augmentations"]
        if len(photometric_augmentations) == 1:
            lines.append("Photometric :")
            lines += [f"  {augmentation_type} : {drawn_values}" if drawn_values else f"  {augmentation_type}"
                      for augmentation_type, drawn_values in photometric_augmentations[0]] or ["  none"]
        else:
            lines.append("Photometric, drawn frame by frame :")
            written_frames = real_frames if real_frames is not None else range(len(photometric_augmentations))
            for frame_index in written_frames:
                frame_descriptions = [f"{augmentation_type} ({drawn_values})" if drawn_values else augmentation_type
                                      for augmentation_type, drawn_values in photometric_augmentations[frame_index]]
                lines.append(f"  Frame {frame_index} : {', '.join(frame_descriptions) or 'none'}")

        # Objects removed after the geometric augmentations, for the tasks with boxes
        if report["removed_labels"] is not None:
            removed_labels = report["removed_labels"].tolist()
            removed_names = [label_names.get(label, str(label)) if label_names else str(label)
                             for label in removed_labels]
            lines.append(f"Removed objects : {len(removed_labels)} ({', '.join(removed_names)})" if removed_labels
                         else "Removed objects : none")
        return lines

    @staticmethod
    def clean_bounding_boxes(bounding_boxes: torch.Tensor, labels: torch.Tensor, image_height: int, image_width: int,
                             instance_masks: Optional[torch.Tensor] = None,
                             visible_fractions: Optional[torch.Tensor] = None,
                             keypoints: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                Optional[torch.Tensor],
                                                                                Optional[torch.Tensor], torch.Tensor]:
        """
        Clips the boxes to the image borders after the geometric augmentations and removes the boxes that became
        too small (mostly cropped or rotated out of the image) or that kept less than MINIMUM_VISIBLE_AREA_FRACTION of
        their area inside the image (a sliver of an object at the border of a crop, which would teach the detector to
        fire on fragments), so the detector only receives valid targets. The instance masks of the removed boxes are
        removed with them, and their labels are returned for the augmentation report. The keypoints that left the
        image become unlabelled (visibility 0, position 0), and an object keeping a labelled keypoint inside the image
        is never removed, since the keypoint detector would otherwise learn its visible keypoints as background.

        Args:
            bounding_boxes (torch.Tensor): The [N, 4] absolute [x1, y1, x2, y2] boxes, moved without clipping so that
                their part outside the image is known.
            labels (torch.Tensor): The [N] labels of the boxes.
            image_height (int): The height of the augmented image.
            image_width (int): The width of the augmented image.
            instance_masks (Optional[torch.Tensor]): The [N, H, W] binary masks of the boxes, or None. Defaults to
                None.
            visible_fractions (Optional[torch.Tensor]): The [N] share of every object left inside the image measured
                on its mask pixels (instance segmentation), or None to measure it on the boxes, as the clipped area
                over the area of the moved box. Defaults to None.
            keypoints (Optional[torch.Tensor]): The [N, K, 3] moved keypoints (x, y, visibility) of the boxes, or None.
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]: The kept
                boxes, their labels, their instance masks and their keypoints (None without them), and the labels of
                the removed boxes.
        """
        clipped_boxes = bounding_boxes.clone()
        clipped_boxes[:, 0::2] = clipped_boxes[:, 0::2].clamp(min=0.0, max=float(image_width))
        clipped_boxes[:, 1::2] = clipped_boxes[:, 1::2].clamp(min=0.0, max=float(image_height))

        # Keep only the boxes that still have a visible width and height
        box_widths = clipped_boxes[:, 2] - clipped_boxes[:, 0]
        box_heights = clipped_boxes[:, 3] - clipped_boxes[:, 1]
        kept_mask = (box_widths >= MINIMUM_BOX_SIDE) & (box_heights >= MINIMUM_BOX_SIDE)

        # Share of every box left inside the image, unless the masks measured it, and removal of the fragments
        if visible_fractions is None:
            moved_areas = ((bounding_boxes[:, 2] - bounding_boxes[:, 0]).clamp(min=0.0) *
                           (bounding_boxes[:, 3] - bounding_boxes[:, 1]).clamp(min=0.0))
            visible_fractions = box_widths.clamp(min=0.0) * box_heights.clamp(min=0.0) / moved_areas.clamp(min=1e-6)
        kept_mask &= visible_fractions >= MINIMUM_VISIBLE_AREA_FRACTION

        # Keypoints outside the image unlabelled, the objects keeping a labelled keypoint inside the image kept
        kept_keypoints = None
        if keypoints is not None:
            inside_image = ((keypoints[..., 0] >= 0.0) & (keypoints[..., 0] < image_width) &
                            (keypoints[..., 1] >= 0.0) & (keypoints[..., 1] < image_height))
            labelled = (keypoints[..., 2] > 0) & inside_image
            keypoints = torch.where(labelled[..., None], keypoints, torch.zeros_like(input=keypoints))
            kept_mask |= labelled.any(dim=1)
            kept_keypoints = keypoints[kept_mask]

        kept_instance_masks = instance_masks[kept_mask] if instance_masks is not None else None
        return clipped_boxes[kept_mask], labels[kept_mask], kept_instance_masks, kept_keypoints, labels[~kept_mask]

    @staticmethod
    def wrap_bounding_boxes(bounding_boxes: torch.Tensor, image: torch.Tensor) -> tv_tensors.BoundingBoxes:
        """
        Wraps plain boxes as torchvision BoundingBoxes so that torchvision's geometric functions transform them
        exactly like the image. The boxes are moved without being clipped to the image, so that clean_bounding_boxes
        can measure the share of every box left inside it.

        Args:
            bounding_boxes (torch.Tensor): The [N, 4] absolute [x1, y1, x2, y2] boxes.
            image (torch.Tensor): The image the boxes belong to, giving the canvas size.

        Returns:
            tv_tensors.BoundingBoxes: The wrapped boxes.
        """
        return tv_tensors.BoundingBoxes(data=bounding_boxes, format="XYXY",
                                        canvas_size=(image.shape[-2], image.shape[-1]), clamping_mode=None)

    def random_crop(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                    mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                           Optional[torch.Tensor], float, str]:
        """
        Crops a random region covering 75 to 100% of each side and resizes it back to the original size, which
        zooms slightly into the image. Boxes are shifted and scaled accordingly; boxes leaving the crop, or keeping
        too small a share of their area inside it, are removed afterwards by 'clean_bounding_boxes'. The mask is
        cropped and resized with the nearest interpolation.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The cropped image, boxes
                and mask, the area scale of the objects (the squared zoom, above 1) and the drawn crop for the
                augmentation report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]

        # Draw one crop fraction for both sides to keep the aspect ratio, and a random position for the crop
        crop_fraction = random.uniform(MINIMUM_CROP_FRACTION, 1.0)
        crop_height, crop_width = round(image_height * crop_fraction), round(image_width * crop_fraction)
        top, left = random.randint(0, image_height - crop_height), random.randint(0, image_width - crop_width)

        cropped_image = vision_functional.resized_crop(image, top=top, left=left, height=crop_height,
                                                       width=crop_width, size=[image_height, image_width],
                                                       antialias=True)

        # Crop the boxes and the mask exactly like the image, the mask with the nearest interpolation
        cropped_boxes, cropped_mask = None, None
        if bounding_boxes is not None:
            cropped_boxes = vision_functional.resized_crop(
                self.wrap_bounding_boxes(bounding_boxes=bounding_boxes, image=image), top=top, left=left,
                height=crop_height, width=crop_width, size=[image_height, image_width]).as_subclass(torch.Tensor)
        if mask is not None:
            cropped_mask = vision_functional.resized_crop(tv_tensors.Mask(mask), top=top, left=left,
                                                          height=crop_height, width=crop_width,
                                                          size=[image_height, image_width]).as_subclass(torch.Tensor)
        drawn_values = f"{crop_fraction * 100:.0f} % of each side, at top {top} and left {left} pixels"
        return (cropped_image, cropped_boxes, cropped_mask, (image_height * image_width) / (crop_height * crop_width),
                drawn_values)

    def random_zoom_out(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                        mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                               Optional[torch.Tensor], float, str]:
        """
        Shrinks the image to 50 to 100% of each side and places it at a random position of a black canvas of the
        original size, the padding of the data loaders, which shows the objects smaller than in the dataset, the
        opposite of the crop: the finest detection levels then also train on the objects that are large in the
        dataset. Boxes are scaled and shifted with the image and never leave it. The mask is shrunk with the nearest
        interpolation, the uncovered border being ignored in a class mask, like the padding of the loaders, and
        background (0) in the binary instance masks.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The zoomed out image,
                boxes and mask, the area scale of the objects (the squared shrink factor, below 1) and the drawn
                shrink for the augmentation report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]

        # Draw one shrink fraction for both sides to keep the aspect ratio, and a random position on the canvas
        zoom_fraction = random.uniform(MINIMUM_ZOOM_OUT_FRACTION, 1.0)
        shrunk_height, shrunk_width = round(image_height * zoom_fraction), round(image_width * zoom_fraction)
        top, left = random.randint(0, image_height - shrunk_height), random.randint(0, image_width - shrunk_width)

        # Shrink the image and place it on the black canvas
        zoomed_image = torch.zeros_like(input=image)
        zoomed_image[..., top:top + shrunk_height, left:left + shrunk_width] = vision_functional.resize(
            image, size=[shrunk_height, shrunk_width], antialias=True)

        # Scale and shift the boxes exactly like the image
        zoomed_boxes, zoomed_mask = None, None
        if bounding_boxes is not None:
            scales = torch.tensor(data=[shrunk_width / image_width, shrunk_height / image_height] * 2,
                                  dtype=bounding_boxes.dtype, device=bounding_boxes.device)
            offsets = torch.tensor(data=[left, top] * 2, dtype=bounding_boxes.dtype, device=bounding_boxes.device)
            zoomed_boxes = bounding_boxes * scales + offsets

        # Shrink the mask with the nearest interpolation onto its canvas, ignored around a class mask and background
        # around binary instance masks
        if mask is not None:
            mask_fill = IGNORED_LABEL_INDEX if mask.dtype == torch.int64 else 0
            zoomed_mask = torch.full_like(input=mask, fill_value=mask_fill)
            zoomed_mask[..., top:top + shrunk_height, left:left + shrunk_width] = vision_functional.resize(
                tv_tensors.Mask(mask), size=[shrunk_height, shrunk_width]).as_subclass(torch.Tensor)
        drawn_values = f"{zoom_fraction * 100:.0f} % of each side, at top {top} and left {left} pixels"
        return (zoomed_image, zoomed_boxes, zoomed_mask, (shrunk_height * shrunk_width) / (image_height * image_width),
                drawn_values)

    @staticmethod
    def shift_tensor(tensor: torch.Tensor, shift_x: int, shift_y: int, fill_value: float) -> torch.Tensor:
        """
        Moves the content of an image, a sequence or masks by whole pixels along their last two dimensions, for
        random_translation: the part pushed past a border is dropped, unlike torch.roll which would bring it back on the
        other side, and the uncovered border is filled, so that the pixels are moved exactly without any resampling.

        Args:
            tensor (torch.Tensor): The [..., H, W] image [C, H, W], sequence [T, C, H, W], class mask [H, W] or
                instance masks [N, H, W].
            shift_x (int): The shift towards the right in pixels, negative towards the left.
            shift_y (int): The shift downwards in pixels, negative upwards.
            fill_value (float): The value of the uncovered border.

        Returns:
            torch.Tensor: The shifted tensor, of the same shape and type.
        """
        height, width = tensor.shape[-2], tensor.shape[-1]
        shifted_tensor = torch.full_like(input=tensor, fill_value=fill_value)

        # Rows and columns that stay inside the image, nothing being kept when a shift reaches the whole side
        kept_height, kept_width = height - abs(shift_y), width - abs(shift_x)
        if kept_height <= 0 or kept_width <= 0:
            return shifted_tensor

        # Source rows and columns of the kept part, and the rows and columns they land on
        source_top, source_left = max(0, -shift_y), max(0, -shift_x)
        destination_top, destination_left = max(0, shift_y), max(0, shift_x)
        source_rows, source_columns = (slice(source_top, source_top + kept_height),
                                       slice(source_left, source_left + kept_width))
        destination_rows, destination_columns = (slice(destination_top, destination_top + kept_height),
                                                 slice(destination_left, destination_left + kept_width))
        shifted_tensor[..., destination_rows, destination_columns] = tensor[..., source_rows, source_columns]
        return shifted_tensor

    def random_translation(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                           mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                  Optional[torch.Tensor], float, str]:
        """
        Shifts the image along its width, its height or both (TRANSLATION_AXES, drawn with equal chances) by whole
        pixels, each shift being drawn up to MAXIMUM_TRANSLATION_FRACTION (25 %) of its side in either direction, so
        that every task, the classification included, sees its objects away from the positions of the dataset. The
        uncovered border is black like the padding of the data loaders. Boxes move by the same shifts without being
        clipped, so that 'clean_bounding_boxes' removes the objects pushed out of the image or keeping too small a share
        of their area inside it. The mask moves exactly, its uncovered border being ignored in a class mask and
        background (0) in the binary instance masks.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The shifted image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and the drawn shifts for the
                augmentation report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]

        # Draw the axes, then the shift along each of them, the other axis keeping a zero shift
        translation_axes = random.choice(TRANSLATION_AXES)
        maximum_shift_x = int(image_width * MAXIMUM_TRANSLATION_FRACTION)
        maximum_shift_y = int(image_height * MAXIMUM_TRANSLATION_FRACTION)
        shift_x = random.randint(-maximum_shift_x, maximum_shift_x) if "x" in translation_axes else 0
        shift_y = random.randint(-maximum_shift_y, maximum_shift_y) if "y" in translation_axes else 0

        # Shift the image onto a black canvas
        translated_image = self.shift_tensor(tensor=image, shift_x=shift_x, shift_y=shift_y, fill_value=0.0)

        # Shift the boxes by the same number of pixels, and the mask with its own border value
        translated_boxes, translated_mask = None, None
        if bounding_boxes is not None:
            translated_boxes = bounding_boxes + torch.tensor(data=[shift_x, shift_y] * 2, dtype=bounding_boxes.dtype,
                                                             device=bounding_boxes.device)
        if mask is not None:
            mask_fill = IGNORED_LABEL_INDEX if mask.dtype == torch.int64 else 0
            translated_mask = self.shift_tensor(tensor=mask, shift_x=shift_x, shift_y=shift_y, fill_value=mask_fill)

        # Report the shift along every drawn axis, positive towards the right and downwards
        axis_shifts = {"x": shift_x, "y": shift_y}
        drawn_values = ", ".join(f"{axis} {axis_shifts[axis]:+d} pixels" for axis in translation_axes)
        return translated_image, translated_boxes, translated_mask, 1.0, drawn_values

    def random_left_right_flip(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                      Optional[torch.Tensor], float, str]:
        """
        Mirrors the image from left to right, together with its boxes and its mask.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The flipped image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and no drawn value (empty).
        """
        flipped_image = vision_functional.horizontal_flip(image)
        flipped_boxes = None
        if bounding_boxes is not None:
            flipped_boxes = vision_functional.horizontal_flip(
                self.wrap_bounding_boxes(bounding_boxes=bounding_boxes, image=image)).as_subclass(torch.Tensor)
        flipped_mask = mask.flip(dims=(-1,)) if mask is not None else None
        return flipped_image, flipped_boxes, flipped_mask, 1.0, ""

    def random_up_down_flip(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                            mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                   Optional[torch.Tensor], float, str]:
        """
        Mirrors the image from top to bottom, together with its boxes and its mask.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The flipped image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and no drawn value (empty).
        """
        flipped_image = vision_functional.vertical_flip(image)
        flipped_boxes = None
        if bounding_boxes is not None:
            flipped_boxes = vision_functional.vertical_flip(
                self.wrap_bounding_boxes(bounding_boxes=bounding_boxes, image=image)).as_subclass(torch.Tensor)
        flipped_mask = mask.flip(dims=(-2,)) if mask is not None else None
        return flipped_image, flipped_boxes, flipped_mask, 1.0, ""

    def random_rotation(self, image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                        mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                               Optional[torch.Tensor], float, str]:
        """
        Rotates the image by a random angle between -5 and 5 degrees around its centre, filling the uncovered
        corners with black like the padding of the data loaders. A rotated box is no longer axis aligned, so it is
        replaced by the enclosing box of its rotated corners, which is slightly larger than the object. The mask is
        rotated with the nearest interpolation, its uncovered corners being ignored like its padding for a class mask,
        and background (0) for binary instance masks.

        Args:
            image (torch.Tensor): The image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The rotated image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and the drawn angle for the
                augmentation report.
        """
        angle = random.uniform(-MAXIMUM_ROTATION_DEGREES, MAXIMUM_ROTATION_DEGREES)

        rotated_image = vision_functional.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR,
                                                 fill=0.0)

        # Rotate the boxes and the mask by the same angle, the uncovered corners of a class mask being ignored and
        # those of binary instance masks being background
        rotated_boxes, rotated_mask = None, None
        if bounding_boxes is not None:
            rotated_boxes = vision_functional.rotate(
                self.wrap_bounding_boxes(bounding_boxes=bounding_boxes, image=image),
                angle=angle).as_subclass(torch.Tensor)
        if mask is not None:
            mask_fill = IGNORED_LABEL_INDEX if mask.dtype == torch.int64 else 0
            rotated_mask = vision_functional.rotate(tv_tensors.Mask(mask), angle=angle,
                                                    fill=mask_fill).as_subclass(torch.Tensor)
        return rotated_image, rotated_boxes, rotated_mask, 1.0, f"{angle:+.1f} degrees"

    @staticmethod
    def random_90_rotation(image: torch.Tensor, bounding_boxes: Optional[torch.Tensor],
                           mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor],
                                                                  Optional[torch.Tensor], float, str]:
        """
        Rotates a square image by 90 degrees clockwise or counter-clockwise, chosen at random. The rotation is an
        exact pixel permutation, so the boxes and the mask are transformed exactly.

        Args:
            image (torch.Tensor): The square image [C, H, W] or sequence [T, C, H, W].
            bounding_boxes (Optional[torch.Tensor]): The [N, 4] boxes, or None.
            mask (Optional[torch.Tensor]): The [H, W] class mask or [N, H, W] instance masks, or None.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, str]: The rotated image, boxes
                and mask, the area scale of the objects (1, the sizes are kept) and the drawn direction for the
                augmentation report.

        Raises:
            ValueError: If the image is not square, since the batch could then no longer be stacked.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]
        if image_height != image_width:
            raise ValueError(f"random_90_rotation requires square images, got {image_height}x{image_width}.")

        # Counter-clockwise rotation maps a point (x, y) to (y, W - x), clockwise maps it to (H - y, x)
        counter_clockwise = random.random() < 0.5
        rotated_image = torch.rot90(image, k=1 if counter_clockwise else -1, dims=(-2, -1))
        rotated_mask = torch.rot90(mask, k=1 if counter_clockwise else -1, dims=(-2, -1)) if mask is not None else None
        drawn_values = "counter-clockwise" if counter_clockwise else "clockwise"
        if bounding_boxes is None:
            return rotated_image, None, rotated_mask, 1.0, drawn_values

        x_1, y_1, x_2, y_2 = bounding_boxes.unbind(dim=-1)
        if counter_clockwise:
            rotated_boxes = torch.stack(tensors=[y_1, image_width - x_2, y_2, image_width - x_1], dim=-1)
        else:
            rotated_boxes = torch.stack(tensors=[image_height - y_2, x_1, image_height - y_1, x_2], dim=-1)
        return rotated_image, rotated_boxes, rotated_mask, 1.0, drawn_values

    @staticmethod
    def brightness_contrast_saturation(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Slightly changes the brightness (+-40 intensity levels), contrast and saturation (both scaled by 0.7 to 1.0)
        to make the network robust to lighting and camera differences.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The adjusted BGR image in [0, 255] and the drawn shift and factors for the
                augmentation report.
        """
        brightness_delta = random.uniform(-BRIGHTNESS_MAXIMUM_DELTA, BRIGHTNESS_MAXIMUM_DELTA)
        brightened_image = (image + brightness_delta).clamp(min=0.0, max=255.0)

        # Contrast and saturation weight the channels as RGB, so the BGR image is reversed before and after
        contrast_factor = random.uniform(*CONTRAST_FACTOR_RANGE)
        saturation_factor = random.uniform(*SATURATION_FACTOR_RANGE)
        rgb_image = brightened_image.flip(dims=(-3,)) / 255.0
        rgb_image = vision_functional.adjust_contrast(rgb_image, contrast_factor=contrast_factor)
        rgb_image = vision_functional.adjust_saturation(rgb_image, saturation_factor=saturation_factor)
        drawn_values = (f"brightness {brightness_delta:+.1f} levels, contrast x{contrast_factor:.2f}, "
                        f"saturation x{saturation_factor:.2f}")
        return rgb_image.flip(dims=(-3,)) * 255.0, drawn_values

    @staticmethod
    def clahe_augment(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Applies contrast-limited adaptive histogram equalization (clip limit 2.0 on a 4x4 grid of tiles) to every
        channel, enhancing local contrast in dark or washed-out regions.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The equalized BGR image in [0, 255] and its fixed settings for the augmentation
                report.
        """
        normalized_image = (image / 255.0).clamp(min=0.0, max=1.0)
        drawn_values = f"clip limit {CLAHE_CLIP_LIMIT}, {CLAHE_GRID_SIZE[0]}x{CLAHE_GRID_SIZE[1]} tiles"
        return (equalize_clahe(normalized_image, clip_limit=CLAHE_CLIP_LIMIT, grid_size=CLAHE_GRID_SIZE) * 255.0,
                drawn_values)

    @staticmethod
    def sharp_augment(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Sharpens the image with a Laplacian of Gaussian unsharp mask: the Laplacian of the blurred grayscale image
        highlights edges, and subtracting it (strength 3) makes them crisper.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The sharpened BGR image in [0, 255] and its fixed settings for the augmentation
                report.
        """
        image_height, image_width = image.shape[-2], image.shape[-1]

        # Grayscale luminance from the BGR channels, blurred to limit the noise amplified by the Laplacian
        luminance_weights = torch.tensor(data=[0.114, 0.587, 0.299], dtype=image.dtype, device=image.device)
        grayscale_image = (image * luminance_weights.view(3, 1, 1)).sum(dim=-3, keepdim=True)
        blurred_image = vision_functional.gaussian_blur(grayscale_image, kernel_size=[5, 5],
                                                        sigma=[SHARPENING_SIGMA, SHARPENING_SIGMA])

        # Apply the Laplacian kernel on every frame with reflected borders
        laplacian_kernel = torch.tensor(data=[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=image.dtype,
                                        device=image.device).view(1, 1, 3, 3)
        padded_image = torch_functional.pad(blurred_image.reshape(-1, 1, image_height, image_width),
                                            pad=(1, 1, 1, 1), mode="reflect")
        laplacian = torch_functional.conv2d(input=padded_image, weight=laplacian_kernel)

        # Subtract the scaled Laplacian to sharpen the edges and report the fixed strength and blur
        sharpened_image = (image - SHARPENING_STRENGTH * laplacian.reshape(grayscale_image.shape)).clamp(min=0.0,
                                                                                                         max=255.0)
        return sharpened_image, f"strength {SHARPENING_STRENGTH}, sigma {SHARPENING_SIGMA}"

    @staticmethod
    def equalize_histogram_augment(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Equalizes the histogram of the luminance (Y channel of YUV) only, which spreads the intensities over the
        full range while keeping the colours.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The equalized BGR image in [0, 255] and the equalized channel for the
                augmentation report.
        """
        yuv_image = rgb_to_yuv((image.flip(dims=(-3,)) / 255.0).clamp(min=0.0, max=1.0))

        # Equalize the luminance channel and keep the two chrominance channels unchanged
        equalized_luminance = vision_functional.equalize(yuv_image[..., 0:1, :, :].clamp(min=0.0, max=1.0))
        yuv_image = torch.cat(tensors=[equalized_luminance, yuv_image[..., 1:, :, :]], dim=-3)

        return (yuv_to_rgb(yuv_image).clamp(min=0.0, max=1.0) * 255.0).flip(dims=(-3,)), "luminance"

    @staticmethod
    def equalize_bgr_histogram_augment(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Equalizes the histogram of one randomly chosen channel (B, G or R) or of all three channels, which changes
        the colour balance of the image.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The equalized BGR image in [0, 255] and the drawn channels for the augmentation
                report.
        """
        channel_name = random.choice(seq=list(BGR_CHANNEL_INDICES))
        channel_indices = BGR_CHANNEL_INDICES[channel_name]

        # Equalize the drawn channels only, the other channels keeping their values
        equalized_image = image.clone()
        normalized_channels = (image[..., channel_indices, :, :] / 255.0).clamp(min=0.0, max=1.0)
        equalized_image[..., channel_indices, :, :] = vision_functional.equalize(normalized_channels) * 255.0
        return equalized_image, f"channels {channel_name}"

    @staticmethod
    def random_noise(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Adds Gaussian noise with a standard deviation of 40 intensity levels, simulating sensor noise. For a
        sequence every frame receives its own noise sample, with the same strength, as a real camera would.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The noisy image and its strength for the augmentation report.
        """
        return (image + torch.randn_like(input=image) * NOISE_STANDARD_DEVIATION,
                f"standard deviation {NOISE_STANDARD_DEVIATION:.0f} levels")

    @staticmethod
    def gray_scale_augment(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Converts the image to grayscale while keeping three channels, so the network does not rely on colour only.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The grayscale image replicated on three channels and no drawn value (empty).
        """
        return vision_functional.rgb_to_grayscale(image.flip(dims=(-3,)), num_output_channels=3), ""

    @staticmethod
    def random_colour_augmentation(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Swaps the blue and red channels (BGR to RGB), a strong colour change that keeps every structure intact.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The image with its channel order reversed and the swap for the augmentation
                report.
        """
        return image.flip(dims=(-3,)), "blue and red swapped"

    @staticmethod
    def random_blur(image: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Blurs the image with a 9x9 Gaussian kernel (sigma 2.5), simulating defocus and motion softness.

        Args:
            image (torch.Tensor): The BGR image [C, H, W] or sequence [T, C, H, W] in [0, 255].

        Returns:
            Tuple[torch.Tensor, str]: The blurred image and the kernel for the augmentation report.
        """
        blurred_image = vision_functional.gaussian_blur(image, kernel_size=[BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE],
                                                        sigma=[BLUR_SIGMA, BLUR_SIGMA])
        return blurred_image, f"{BLUR_KERNEL_SIZE}x{BLUR_KERNEL_SIZE} kernel, sigma {BLUR_SIGMA}"
