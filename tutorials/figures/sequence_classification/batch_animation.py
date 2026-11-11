"""
Animated padded batch of the sequence classification tutorials (chapters 1 and 2), made by
tutorials/figures/sequence_classification/figures.py: four sequences drawn by the current synthetic generator of the
repository (generate_moving_shape_sequence, 2 to 12 frames of 512 pixels, the dataset on disk holding only sequences
of 8 frames), saved and read back as the sequence dataset reads them (128 pixel frames), batched by the collate of the
loaders (collate_padded_sequences) and run through the frame encoder of the trained flatten run on the CPU. It writes
the frames, the frame vectors drawn as heat squares and the layout file of animations/07_sequence_classification/
frame_batches.js, whose phases are the samples, the padded batch, its mask, the real frames gathered for the backbone,
their vectors and the vectors scattered back; its facts are the real lengths and shapes the narration quotes.
"""

import cv2
import numpy
import torch

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import round_fact
from tutorials.figures.sequence_classification.sequence_figure_utilities import (write_animation_layout,
                                                                                 write_vector_square)
from vision_training.data.sequence.sequence_dataset import collate_padded_sequences, read_sequence_frames
from vision_training.data.synthetic.create_synthetic_dataset import (MINIMUM_SEQUENCE_FRAMES, MOVEMENT_CATEGORIES,
                                                                     generate_moving_shape_sequence, save_frames)
from vision_training.models.sequence.frame_encoder import encode_frames
from vision_training.models.task_model import TaskModel

# Frames drawn by the generator for the sequence classification, as set in the main block of
# vision_training/data/synthetic/create_synthetic_dataset.py: 512 pixels, 2 to 12 frames per sequence
GENERATOR_IMAGE_SIZE = 512
GENERATOR_MAXIMUM_FRAMES = 12

# Sequences of the animated batch, and the bounds of the example kept: distinct lengths, at most 10 frames so that the
# batch fits the scene, four directions; the first seed of the generator giving such a batch is taken
SEQUENCE_COUNT = 4
LONGEST_SHOWN_FRAMES = 10
MAXIMUM_SEED = 2000

# Layout of the scene in pixels of its content area: the batch grid of frames, the row of the gathered real frames
# under it and the mask on the right, every frame and vector drawn as a square tile
GRID_LEFT = 330
GRID_TOP = 96
TILE_SIDE = 72
TILE_STRIDE = 82
GATHER_TOP = 556
GATHER_SIDE = 40
GATHER_STRIDE = 44
MASK_LEFT = 1300
MASK_SIDE = 36
MASK_STRIDE = 40
GATHER_DELAY_SECONDS = 0.03

# Phases of the layout in their order, the scenes starting each of them with one of their sentences
LAYOUT_PHASES = ["sequences", "padded", "mask", "gather", "encode", "scatter"]


def draw_example_sequences(output_folder: Path) -> Dict[str, Any]:
    """
    Draws the sequences of the example batch with the generator of the repository, the same calls as one chunk of
    generate_sequence_chunk (the number of frames, then the sequence, from one random generator), from the first seed
    giving distinct lengths of at most LONGEST_SHOWN_FRAMES frames and four directions, and saves their frames with
    save_frames, the way the dataset is written.

    Args:
        output_folder (Path): The folder receiving the frame folders of the sequences.

    Returns:
        Dict[str, Any]: The 'seed', and the 'frame_files' and 'category_identifier' of every sequence.

    Raises:
        RuntimeError: If no seed below MAXIMUM_SEED gives such a batch.
    """
    for seed in range(MAXIMUM_SEED):
        random_generator = numpy.random.default_rng(seed=seed)
        sequences = []
        for _ in range(SEQUENCE_COUNT):
            number_frames = int(random_generator.integers(low=MINIMUM_SEQUENCE_FRAMES,
                                                          high=GENERATOR_MAXIMUM_FRAMES + 1))
            frames, category_identifier = generate_moving_shape_sequence(image_size=GENERATOR_IMAGE_SIZE,
                                                                         number_frames=number_frames,
                                                                         random_generator=random_generator)
            sequences.append({"frames": frames, "category_identifier": category_identifier})
        lengths = [len(sequence["frames"]) for sequence in sequences]
        categories = [sequence["category_identifier"] for sequence in sequences]
        if len(set(lengths)) == SEQUENCE_COUNT and max(lengths) <= LONGEST_SHOWN_FRAMES and \
                len(set(categories)) == SEQUENCE_COUNT:
            break
    else:
        raise RuntimeError(f"No seed below {MAXIMUM_SEED} gives {SEQUENCE_COUNT} sequences of distinct lengths of at "
                           f"most {LONGEST_SHOWN_FRAMES} frames and distinct directions.")

    # Frames saved like the generator saves them, every sequence in its own folder
    output_folder.mkdir(parents=True, exist_ok=True)
    for sequence_index, sequence in enumerate(sequences):
        sequence["frame_files"] = save_frames(data_folder=str(output_folder), frame_folder=f"sequence_{sequence_index}",
                                              frames=sequence["frames"])
    return {"seed": seed, "sequences": sequences}


def compute_block_shapes(model: TaskModel, gathered_frames: torch.Tensor) -> List[List[int]]:
    """
    Gives the shape of the output of every backbone block for the gathered real frames, running the blocks of the
    trained backbone the way Backbone.forward does (input normalization, then the layers and the pooling of every
    block), the batch normalization of the run needing no sequence grouping.

    Args:
        model (TaskModel): The trained sequence classifier.
        gathered_frames (torch.Tensor): The [N, 3, H, W] float real frames.

    Returns:
        List[List[int]]: The [N, C, H, W] shape after every block.
    """
    block_shapes = []
    with torch.no_grad():
        current_tensor = model.backbone.input_normalizer(input_tensor=gathered_frames)
        for block in model.backbone.convolution_blocks:
            current_tensor = block.apply_pooling(input_tensor=block.compute_layers(input_tensor=current_tensor,
                                                                                    sequence_grouping=None))
            block_shapes.append(list(current_tensor.shape))
    return block_shapes


def format_shape(shape: List[int]) -> str:
    """
    Writes a tensor shape the way the videos show it, e.g. "[4, 9, 3, 128, 128]".

    Args:
        shape (List[int]): The sizes of the tensor.

    Returns:
        str: The shape between brackets.
    """
    return "[" + ", ".join(str(size) for size in shape) + "]"


def build_batch_layout(lengths: List[int], class_names: List[str], frame_images: List[List[str]],
                       vector_images: List[List[str]], zero_vector_image: str,
                       shapes: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Places every tile and text of the animated batch at every phase: the frames of every sequence on its row of the
    batch grid, the black padding frames after them, the mask on the right, the real frames gathered on one row in
    the order of input_tensor[sequence_mask] (sequence after sequence), their vector squares, then the vectors back
    at their frame positions with zero squares on the padding.

    Args:
        lengths (List[int]): The real frames of every sequence.
        class_names (List[str]): The class of every sequence.
        frame_images (List[List[str]]): The frame images of every sequence, relative to the layout file.
        vector_images (List[List[str]]): The vector square of every real frame, relative to the layout file.
        zero_vector_image (str): The square of a zero vector, relative to the layout file.
        shapes (Dict[str, str]): The shapes written in the texts: 'images', 'mask', 'gathered', 'blocks',
            'vectors' and 'frame_features'.

    Returns:
        Dict[str, List[Dict[str, Any]]]: The 'tiles' and 'texts' of the layout.
    """
    longest_length = max(lengths)
    tiles, texts = [], []

    # Frame of the batch, outlined from the padded phase on, under the tiles
    grid_right = GRID_LEFT + (longest_length - 1) * TILE_STRIDE + TILE_SIDE
    grid_bottom = GRID_TOP + (len(lengths) - 1) * TILE_STRIDE + TILE_SIDE
    tiles.append({"image": None, "fill": "card", "outline": "accent",
                  "states": {"padded": {"box": [GRID_LEFT - 12, GRID_TOP - 12, grid_right - GRID_LEFT + 24,
                                                grid_bottom - GRID_TOP + 24], "opacity": 1.0}}})

    # Frames of every sequence on its row, gathered in the order of the boolean indexing, then back as vectors
    gathered_index = 0
    for row_index, length in enumerate(lengths):
        row_top = GRID_TOP + row_index * TILE_STRIDE
        for column_index in range(longest_length):
            grid_box = [GRID_LEFT + column_index * TILE_STRIDE, row_top, TILE_SIDE, TILE_SIDE]
            if column_index < length:
                gather_box = [GRID_LEFT + gathered_index * GATHER_STRIDE, GATHER_TOP, GATHER_SIDE, GATHER_SIDE]
                delay = round(gathered_index * GATHER_DELAY_SECONDS, 3)
                tiles.append({"image": frame_images[row_index][column_index],
                              "second_image": vector_images[row_index][column_index], "fill": "black",
                              "outline": "border",
                              "states": {"sequences": {"box": grid_box, "opacity": 1.0},
                                         "gather": {"box": gather_box, "opacity": 1.0, "delay": delay},
                                         "encode": {"box": gather_box, "opacity": 1.0, "face": 1, "delay": delay},
                                         "scatter": {"box": grid_box, "opacity": 1.0, "face": 1, "delay": delay}}})
                gathered_index += 1
            else:
                # Black padding frames, dimmed while the real frames are encoded, then zero vectors
                tiles.append({"image": None, "second_image": zero_vector_image, "fill": "black", "outline": "border",
                              "label": "pad",
                              "states": {"padded": {"box": grid_box, "opacity": 1.0},
                                         "gather": {"box": grid_box, "opacity": 0.3},
                                         "scatter": {"box": grid_box, "opacity": 1.0, "face": 1}}})

            # Mask cell of the frame, True for the real frames
            mask_box = [MASK_LEFT + column_index * MASK_STRIDE, row_top + (TILE_SIDE - MASK_SIDE) // 2, MASK_SIDE,
                        MASK_SIDE]
            tiles.append({"image": None, "fill": "aqua" if column_index < length else "box", "outline": "border",
                          "states": {"mask": {"box": mask_box, "opacity": 1.0}}})

        # Class and length of every sequence on the left, and the shape of its sample before the collate
        texts.append({"text": f"{class_names[row_index]}\n{length} frames", "position": [GRID_LEFT - 28,
                      row_top + TILE_SIDE // 2], "anchor": "rm", "font": "diagram_small", "colour": "text",
                      "phases": LAYOUT_PHASES})
        texts.append({"text": format_shape(shape=[length, 3, 128, 128]),
                      "position": [GRID_LEFT + length * TILE_STRIDE + 4, row_top + TILE_SIDE // 2], "anchor": "lm",
                      "font": "diagram_monospace_small", "colour": "dim", "phases": ["sequences"]})

    # Shapes of the batch, of its mask, of the gathered frames and of the frame vectors
    texts += [{"text": f"images {shapes['images']} uint8", "position": [GRID_LEFT - 12, 40], "anchor": "la",
               "font": "diagram_monospace_small", "colour": "key", "phases": ["padded", "mask", "gather", "encode"]},
              {"text": f"frame_features {shapes['frame_features']}, zeros on the padding",
               "position": [GRID_LEFT - 12, 40], "anchor": "la", "font": "diagram_monospace_small", "colour": "key",
               "phases": ["scatter"]},
              {"text": f"sequence_mask {shapes['mask']} bool", "position": [MASK_LEFT, 40], "anchor": "la",
               "font": "diagram_monospace_small", "colour": "key",
               "phases": ["mask", "gather", "encode", "scatter"]},
              {"text": f"input_tensor[sequence_mask]  {shapes['gathered']}", "position": [GRID_LEFT, GATHER_TOP - 44],
               "anchor": "la", "font": "diagram_monospace_small", "colour": "key", "phases": ["gather"]},
              {"text": f"backbone  {shapes['blocks']}", "position": [GRID_LEFT, GATHER_TOP - 44], "anchor": "la",
               "font": "diagram_monospace_small", "colour": "key", "phases": ["encode"]},
              {"text": f"flatten  {shapes['vectors']}: one vector per real frame", "position": [GRID_LEFT,
               GATHER_TOP + GATHER_SIDE + 22], "anchor": "la", "font": "diagram_monospace_small", "colour": "key",
               "phases": ["encode"]}]
    return {"tiles": tiles, "texts": texts}


def make_batch_animation(output_folder: Path, model: TaskModel, label_dictionary: Dict[int, int],
                         image_size: int) -> Dict[str, Any]:
    """
    Makes the frames, vector squares and layout file of the animated padded batch, and gives its facts: the seed and
    lengths of the sequences, the shapes of the batch, of its mask, of the gathered frames after every backbone block
    and of the frame vectors, the largest value on the padding vectors (zero), and the predictions of the trained run
    on these sequences of the current generator.

    Args:
        output_folder (Path): The figures folder of the series.
        model (TaskModel): The trained flatten run at 128 pixels, on the CPU.
        label_dictionary (Dict[int, int]): The class index of every category of the run.
        image_size (int): The frame size of the run, 'Data.image_settings.size'.

    Returns:
        Dict[str, Any]: The facts, named batch.*.
    """
    batch_folder = output_folder / "batch"
    example = draw_example_sequences(output_folder=batch_folder / "generated")
    category_names = {identifier: name for name, identifier in MOVEMENT_CATEGORIES.items()}

    # Samples read back as the sequence dataset reads them, then batched by the collate of the loaders
    samples = []
    for sequence_index, sequence in enumerate(example["sequences"]):
        frames, sequence_mask = read_sequence_frames(data_directory=str(batch_folder / "generated"),
                                                     frame_files=sequence["frame_files"],
                                                     number_frames=len(sequence["frame_files"]),
                                                     image_size=image_size, keep_ratio=True)
        samples.append({"images": frames, "sequence_mask": sequence_mask,
                        "labels": torch.tensor(data=label_dictionary[sequence["category_identifier"]]),
                        "sample_identifiers": sequence_index})
    batch = collate_padded_sequences(samples=samples, members_per_sample=1)
    sequence_mask = batch["sequence_mask"]
    input_tensor = batch["images"].float()

    # Frame vectors of the trained encoder, the shapes after every block, and the predictions of the run
    with torch.no_grad():
        frame_features = encode_frames(backbone=model.backbone, global_pooling=model.global_pooling,
                                       input_tensor=input_tensor, sequence_mask=sequence_mask)
        probabilities = model(input_tensor=input_tensor, sequence_mask=sequence_mask)["class_probabilities"]
    block_shapes = compute_block_shapes(model=model, gathered_frames=input_tensor[sequence_mask])

    # Frames as the batch holds them (BGR, network size) and the vector square of every frame, read as the last
    # feature map of the backbone it flattens
    lengths = [int(length) for length in sequence_mask.sum(dim=1).tolist()]
    map_shape = (block_shapes[-1][1], block_shapes[-1][2], block_shapes[-1][3])
    # Full blue at the 99th percentile of the positive values, the few largest ones saturating, so that the sparse
    # flattened maps stay visible
    positive_values = frame_features[frame_features > 0]
    maximum_value = float(torch.quantile(input=positive_values, q=0.99))
    frame_images, vector_images = [], []
    for sequence_index, length in enumerate(lengths):
        frame_images.append([])
        vector_images.append([])
        for frame_index in range(length):
            frame_name = f"frame_{sequence_index}_{frame_index:02d}.png"
            if not cv2.imwrite(filename=str(batch_folder / frame_name),
                               img=batch["images"][sequence_index, frame_index].permute(1, 2, 0).numpy()):
                raise OSError(f"The frame {batch_folder / frame_name} could not be written.")
            vector_name = f"vector_{sequence_index}_{frame_index:02d}.png"
            write_vector_square(vector=frame_features[sequence_index, frame_index].numpy(), map_shape=map_shape,
                                maximum_value=maximum_value, side_pixels=TILE_SIDE * 2,
                                output_path=batch_folder / vector_name)
            frame_images[-1].append(f"../batch/{frame_name}")
            vector_images[-1].append(f"../batch/{vector_name}")
    write_vector_square(vector=numpy.zeros(shape=frame_features.shape[-1]), map_shape=map_shape,
                        maximum_value=maximum_value, side_pixels=TILE_SIDE * 2,
                        output_path=batch_folder / "vector_zero.png")

    # Layout of the animation, its texts quoting the real shapes
    class_names = [category_names[sequence["category_identifier"]] for sequence in example["sequences"]]
    real_frames = int(sequence_mask.sum())
    shapes = {"images": format_shape(shape=list(batch["images"].shape)),
              "mask": format_shape(shape=list(sequence_mask.shape)),
              "gathered": format_shape(shape=list(input_tensor[sequence_mask].shape)),
              "blocks": "  ".join(format_shape(shape=shape) for shape in block_shapes),
              "vectors": format_shape(shape=[real_frames, frame_features.shape[-1]]),
              "frame_features": format_shape(shape=list(frame_features.shape))}
    layout = build_batch_layout(lengths=lengths, class_names=class_names, frame_images=frame_images,
                                vector_images=vector_images, zero_vector_image="../batch/vector_zero.png",
                                shapes=shapes)
    write_animation_layout(output_path=output_folder / "animations" / "padded_batch.json", phases=LAYOUT_PHASES,
                           tiles=layout["tiles"], texts=layout["texts"])

    # Facts: lengths, shapes, the zero padding vectors and the predictions of the run on these sequences
    index_names = {class_index: category_names[identifier] for identifier, class_index in label_dictionary.items()}
    predicted_names = [index_names[int(index)] for index in probabilities.argmax(dim=1)]
    padding_values = frame_features[~sequence_mask]
    facts = {"batch.seed": example["seed"], "batch.lengths": lengths, "batch.longest": max(lengths),
             "batch.sequence_count": len(lengths), "batch.real_frames": real_frames,
             "batch.padding_frames": int((~sequence_mask).sum()),
             "batch.padding_share": round_fact(value=float((~sequence_mask).float().mean())),
             "batch.classes": class_names, "batch.predicted_classes": predicted_names,
             "batch.correct_predictions": sum(int(predicted == true) for predicted, true in
                                              zip(predicted_names, class_names)),
             "batch.padding_vector_maximum": float(padding_values.abs().max()) if len(padding_values) else 0.0,
             "batch.frame_vector_features": int(frame_features.shape[-1])}
    for shape_name, shape_text in shapes.items():
        facts[f"batch.shape.{shape_name}"] = shape_text
    facts["batch.frame_channels"] = int(batch["images"].shape[2])
    for block_number, block_shape in enumerate(block_shapes, start=1):
        facts[f"batch.block_{block_number}_shape"] = format_shape(shape=block_shape)
        facts[f"batch.block_{block_number}_channels"] = int(block_shape[1])
        facts[f"batch.block_{block_number}_size"] = int(block_shape[2])
    for sequence_index, length in enumerate(lengths):
        facts[f"batch.length_{sequence_index}"] = length
    return facts
