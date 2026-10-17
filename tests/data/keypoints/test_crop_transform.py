"""
Tests of vision_training/data/keypoints/crop_transform.py, the crop transform the keypoint trainer runs on the device
for whole batches: the crop region of every object (its box widened to the aspect ratio of the crop and enlarged, a
degenerate box keeping one pixel), the affine matrices between the resized image and the crop (inverse of each other,
the region centre at the crop centre, the flip mirroring x, a positive rotation turning anticlockwise on the screen),
the bilinear warp of the crops (pixels reproduced by an identity region, black outside the image, the image content
landing where transform_points moves it), the Gaussian heatmap targets and their weights, the random crop
augmentation within its ranges, and the lines of the crop augmentation in the input viewer panel.
"""

import torch

from vision_training.data.keypoints.crop_transform import (build_crop_affines, build_heatmap_targets,
                                                           compute_crop_regions, create_identity_augmentations,
                                                           describe_crop_augmentation, draw_crop_augmentations,
                                                           transform_points, warp_crops)

# Crop size of the tests, the 64 x 48 crops of the keypoint fixture, and the heatmap stride of the HRNet output
CROP_HEIGHT, CROP_WIDTH, HEATMAP_STRIDE = 64, 48, 4

# Crop augmentation section of the tests, the rotation limited to 30 degrees by the configuration
AUGMENTATION_CONFIGURATION = {"scale_range": [0.8, 1.2], "maximum_rotation_degrees": 30.0, "rotation_probability": 1.0,
                              "flip_probability": 0.5, "maximum_shift_fraction": 0.1}


def test_crop_regions_follow_the_crop_aspect_ratio() -> None:
    """
    Verifies that a tall box is widened to the 3:4 aspect ratio of the crop and a wide box heightened, both enlarged by
    the box enlargement around the box centre, and that a degenerate box still gives a region of one pixel enlarged.
    """
    bounding_boxes = torch.tensor(data=[[10.0, 20.0, 30.0, 100.0], [0.0, 0.0, 100.0, 10.0], [5.0, 5.0, 5.0, 5.0]])
    centres, sizes = compute_crop_regions(bounding_boxes=bounding_boxes, crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH,
                                          box_enlargement=1.25)
    assert torch.equal(input=centres, other=torch.tensor(data=[[20.0, 60.0], [50.0, 5.0], [5.0, 5.0]]))

    # Tall box: 80 pixels high, so 60 wide; wide box: 100 pixels wide, so 133.3 high; degenerate box: 1 x 4/3
    expected_sizes = torch.tensor(data=[[60.0, 80.0], [100.0, 400.0 / 3.0], [1.0, 4.0 / 3.0]]) * 1.25
    assert torch.allclose(input=sizes, other=expected_sizes, atol=1e-5)
    assert torch.allclose(input=sizes[:, 0] / sizes[:, 1], other=torch.full(size=(3,), fill_value=0.75), atol=1e-6)


def test_crop_affines_are_inverse_and_centre_the_region() -> None:
    """
    Verifies that the image to crop matrices and the crop to image matrices of random augmentations undo each other on
    points, that the shifted region centre lands on the crop centre, and that unaugmented crops scale the region onto
    the crop width without moving its centre.
    """
    generator = torch.Generator().manual_seed(0)
    centres = torch.rand(size=(5, 2), generator=generator, dtype=torch.float64) * 200.0
    sizes = 20.0 + torch.rand(size=(5, 2), generator=generator, dtype=torch.float64) * 100.0
    augmentations = {"scales": 0.8 + 0.4 * torch.rand(size=(5,), generator=generator),
                     "rotations": 360.0 * torch.rand(size=(5,), generator=generator) - 180.0,
                     "flips": torch.tensor(data=[True, False, True, False, True]),
                     "shifts": 0.2 * torch.rand(size=(5, 2), generator=generator) - 0.1}
    image_to_crop, crop_to_image = build_crop_affines(centres=centres, sizes=sizes, augmentations=augmentations,
                                                      crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)
    assert image_to_crop.shape == crop_to_image.shape == (5, 2, 3) and image_to_crop.dtype == torch.float64

    # Points of the image go into the crop and come back to their place
    points = torch.rand(size=(5, 7, 2), generator=generator, dtype=torch.float64) * 300.0
    crop_points = transform_points(points=points, affines=image_to_crop)
    assert torch.allclose(input=transform_points(points=crop_points, affines=crop_to_image), other=points, atol=1e-9)

    # The shifted region centre lands on the crop centre
    region_centres = centres + augmentations["shifts"].to(dtype=torch.float64) * sizes
    crop_centres = transform_points(points=region_centres[:, None, :], affines=image_to_crop)[:, 0]
    assert torch.allclose(input=crop_centres, other=torch.tensor(data=[[24.0, 32.0]], dtype=torch.float64).expand(5, 2),
                          atol=1e-9)

    # Without augmentation, the region width fills the crop width around the same centre
    image_to_crop, _ = build_crop_affines(centres=centres, sizes=sizes,
                                          augmentations=create_identity_augmentations(number_samples=5),
                                          crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)
    zooms = CROP_WIDTH / sizes[:, 0]
    assert torch.allclose(input=image_to_crop[:, 0, 0], other=zooms) and torch.allclose(input=image_to_crop[:, 1, 1],
                                                                                        other=zooms)
    assert torch.equal(input=image_to_crop[:, 0, 1], other=torch.zeros(size=(5,), dtype=torch.float64))


def test_flip_mirrors_and_rotation_turns_anticlockwise() -> None:
    """
    Verifies that a flipped crop sends a point right of the region centre to the left of the crop centre, and that a
    rotation of +90 degrees sends it above the crop centre (anticlockwise on the screen, y pointing down) while a point
    below the centre goes to its right, the region of 48 pixels wide being mapped at a zoom of 1.
    """
    centres = torch.tensor(data=[[100.0, 100.0], [100.0, 100.0]])
    sizes = torch.tensor(data=[[48.0, 64.0], [48.0, 64.0]])
    augmentations = create_identity_augmentations(number_samples=2)
    augmentations["flips"][0] = True
    augmentations["rotations"][1] = 90.0
    image_to_crop, _ = build_crop_affines(centres=centres, sizes=sizes, augmentations=augmentations,
                                          crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)

    # A point 10 pixels right of the centre and a point 10 pixels below it
    points = torch.tensor(data=[[[110.0, 100.0], [100.0, 110.0]]]).expand(2, 2, 2)
    crop_points = transform_points(points=points, affines=image_to_crop)
    assert torch.allclose(input=crop_points[0], other=torch.tensor(data=[[14.0, 32.0], [24.0, 42.0]]), atol=1e-5)
    assert torch.allclose(input=crop_points[1], other=torch.tensor(data=[[24.0, 22.0], [34.0, 32.0]]), atol=1e-5)


def test_warp_reproduces_the_pixels_and_blacks_the_outside() -> None:
    """
    Verifies that a region covering the whole image at the crop size reproduces its pixels exactly, in the data type of
    the images, and that a region shifted by its whole width to the right of the image gives a black crop.
    """
    images = torch.arange(end=2 * 3 * CROP_HEIGHT * CROP_WIDTH, dtype=torch.float32).reshape(2, 3, CROP_HEIGHT,
                                                                                            CROP_WIDTH) % 251.0
    centres = torch.tensor(data=[[CROP_WIDTH / 2.0, CROP_HEIGHT / 2.0]]).expand(2, 2)
    sizes = torch.tensor(data=[[float(CROP_WIDTH), float(CROP_HEIGHT)]]).expand(2, 2)
    augmentations = create_identity_augmentations(number_samples=2)
    _, crop_to_image = build_crop_affines(centres=centres, sizes=sizes, augmentations=augmentations,
                                          crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)
    crops = warp_crops(images=images, crop_to_image=crop_to_image, crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)
    assert crops.shape == images.shape and torch.allclose(input=crops, other=images, atol=1e-3)

    # The data type of the images is kept
    double_precision_crops = warp_crops(images=images.to(dtype=torch.float64), crop_to_image=crop_to_image,
                            crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)
    assert double_precision_crops.dtype == torch.float64

    # A region one width to the right of the image is outside it
    augmentations["shifts"][:, 0] = 1.0
    _, crop_to_image = build_crop_affines(centres=centres, sizes=sizes, augmentations=augmentations,
                                          crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)
    outside_crops = warp_crops(images=images, crop_to_image=crop_to_image, crop_height=CROP_HEIGHT,
                               crop_width=CROP_WIDTH)
    assert float(outside_crops.abs().max()) == 0.0


def test_warped_content_lands_where_the_points_go() -> None:
    """
    Verifies that the bright pixel of an image lands, in a rotated, scaled, flipped and shifted crop, on the crop pixel
    holding the place transform_points gives to the centre of that pixel, so that the keypoints and the crops stay
    aligned.
    """
    images = torch.zeros(size=(1, 1, 200, 200))
    images[0, 0, 90, 120] = 255.0
    centres, sizes = torch.tensor(data=[[110.0, 100.0]]), torch.tensor(data=[[60.0, 80.0]])
    augmentations = {"scales": torch.tensor(data=[1.1]), "rotations": torch.tensor(data=[35.0]),
                     "flips": torch.tensor(data=[True]), "shifts": torch.tensor(data=[[0.05, -0.05]])}
    image_to_crop, crop_to_image = build_crop_affines(centres=centres, sizes=sizes, augmentations=augmentations,
                                                      crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)
    crops = warp_crops(images=images, crop_to_image=crop_to_image, crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH)

    # The brightest crop pixel holds the moved centre of the bright pixel
    moved_point = transform_points(points=torch.tensor(data=[[[120.5, 90.5]]]), affines=image_to_crop)[0, 0]
    brightest_index = int(crops[0, 0].argmax())
    brightest_row, brightest_column = divmod(brightest_index, CROP_WIDTH)
    assert abs(brightest_column + 0.5 - float(moved_point[0])) <= 0.5
    assert abs(brightest_row + 0.5 - float(moved_point[1])) <= 0.5


def test_heatmap_targets_peak_on_the_keypoint_cell() -> None:
    """
    Verifies that a keypoint at the centre of a heatmap cell gives a Gaussian of peak 1 on that cell with the configured
    standard deviation, and that an unlabelled keypoint, a keypoint left of the crop and a keypoint on the right edge of
    the crop get a weight of 0 and an empty heatmap.
    """
    crop_keypoints = torch.tensor(data=[[[18.0, 26.0], [18.0, 26.0], [-1.0, 5.0], [float(CROP_WIDTH), 5.0]]])
    visibilities = torch.tensor(data=[[2, 0, 1, 2]])
    heatmaps, weights = build_heatmap_targets(crop_keypoints=crop_keypoints, visibilities=visibilities,
                                              crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH,
                                              heatmap_stride=HEATMAP_STRIDE, heatmap_sigma=2.0)
    assert heatmaps.shape == (1, 4, 16, 12) and heatmaps.dtype == weights.dtype == torch.float32
    assert weights.tolist() == [[1.0, 0.0, 0.0, 0.0]]

    # Crop pixel (18, 26) is the centre of the cell of column 4 and row 6, one cell away the Gaussian falls to
    # exp(-1 / 8)
    trained_heatmap = heatmaps[0, 0]
    assert float(trained_heatmap[6, 4]) == 1.0 and int(trained_heatmap.argmax()) == 6 * 12 + 4
    assert abs(float(trained_heatmap[6, 5]) - float(torch.exp(torch.tensor(data=-1.0 / 8.0)))) < 1e-6
    assert float(heatmaps[0, 1:].abs().max()) == 0.0


def test_crop_augmentations_stay_within_their_ranges() -> None:
    """
    Verifies that the drawn scales, shifts and rotations stay within their ranges, the rotation bounded by the
    configured limit, that the flips follow their probability, and that a rotation probability of 0 and a flip probability of 0 leave the crops unrotated and unflipped.
    """
    torch.manual_seed(seed=0)
    augmentations = draw_crop_augmentations(number_samples=2000, augmentation_configuration=AUGMENTATION_CONFIGURATION)
    assert float(augmentations["scales"].min()) >= 0.8 and float(augmentations["scales"].max()) <= 1.2
    assert float(augmentations["shifts"].abs().max()) <= 0.1 and augmentations["shifts"].shape == (2000, 2)
    assert augmentations["flips"].dtype == torch.bool and 0.4 < float(augmentations["flips"].float().mean()) < 0.6

    # Every crop turns by the 30 degrees of the configuration at most
    assert augmentations["rotations"].shape == (2000,)
    assert 29.0 < float(augmentations["rotations"].abs().max()) <= 30.0

    # Without rotation nor flip probability, the crops are neither rotated nor flipped
    unrotated_configuration = {**AUGMENTATION_CONFIGURATION, "rotation_probability": 0.0, "flip_probability": 0.0}
    unrotated_augmentations = draw_crop_augmentations(number_samples=2000,
                                                      augmentation_configuration=unrotated_configuration)
    assert float(unrotated_augmentations["rotations"].abs().max()) == 0.0
    assert not bool(unrotated_augmentations["flips"].any())


def test_crop_augmentation_lines_describe_the_crop() -> None:
    """
    Verifies the lines written in the input viewer panel for an augmented crop and for an unaugmented one, the
    negative zeros drawn without shift or for a class without rotation being written without a minus sign.
    """
    augmentations = {"scales": torch.tensor(data=[1.0, 1.137]), "rotations": torch.tensor(data=[-0.0, -12.34]),
                     "flips": torch.tensor(data=[False, True]),
                     "shifts": torch.tensor(data=[[-0.0, -0.0], [0.05, -0.08]])}
    assert describe_crop_augmentation(augmentations=augmentations, sample_index=1) == [
        "Crop :", "  scale : 1.14", "  rotation : -12.3 degrees", "  flip : yes", "  shift : +0.05 width, -0.08 height"]
    assert describe_crop_augmentation(augmentations=augmentations, sample_index=0) == [
        "Crop :", "  scale : 1.00", "  rotation : 0.0 degrees", "  flip : no", "  shift : +0.00 width, +0.00 height"]
    assert describe_crop_augmentation(augmentations=create_identity_augmentations(number_samples=1),
                                      sample_index=0) == [
        "Crop :", "  scale : 1.00", "  rotation : 0.0 degrees", "  flip : no", "  shift : +0.00 width, +0.00 height"]
