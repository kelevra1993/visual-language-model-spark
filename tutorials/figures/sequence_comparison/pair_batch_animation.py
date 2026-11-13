"""
Animated batch of pairs of the sequence comparison tutorials (chapters 1 and 2), made by
tutorials/figures/sequence_comparison/figures.py from four real training pairs of the object loss pairs dataset of the
comparison run: the members read and padded to the longer member by the pair dataset (SequencePairDataset), batched by
the collate of the loaders (collate_padded_sequences with two members per sample), split back into the before and
after members (split_pair_members), stacked as 2B sequences and encoded together like encode_members of the trained
comparator, run on the CPU. It writes the frames, the frame vectors drawn as heat squares and the layout file of
animations/07_sequence_classification/frame_batches.js; its facts are the real lengths, shapes and padding of the batch,
and the difference between the masked path of the training and the per-member path of the export on these pairs.
"""

import json
import math

import cv2
import numpy
import torch

from pathlib import Path
from typing import Any, Dict, List

from tutorials.figures.figure_utilities import read_reused_facts, round_fact, write_reused_facts
from tutorials.figures.sequence_classification.batch_animation import format_shape
from tutorials.figures.sequence_classification.sequence_figure_utilities import (write_animation_layout,
                                                                                 write_vector_square)
from vision_training.data.sequence.sequence_dataset import collate_padded_sequences
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS, SequencePairDataset, split_pair_members
from vision_training.models.task_model import TaskModel

# Pairs of the animated batch: the window of consecutive training pairs, among the first ones of the manifest, whose
# longer members have the most distinct lengths, so that both levels of padding show
PAIR_COUNT = 4
SEARCHED_WINDOWS = 50

# Layout of the scene in pixels of its content area, every frame and vector drawn as a square tile
LABEL_RIGHT = 296
FRAMES_LEFT = 320
TILE_SIDE = 52
TILE_STRIDE = 58
MEMBER_GAP = 24
SPLIT_GAP = 150
MEMBERS_TOP = 40
PAIR_STRIDE = 150
MEMBER_STRIDE = 62
BATCH_TOP = 96
BATCH_ROW_STRIDE = 96
STACK_TOP = 60
STACK_ROW_STRIDE = 66
DELAY_SECONDS = 0.02

# Phases of the layout in their order, the scenes starting each of them with one of their sentences
LAYOUT_PHASES = ["members", "pair_padded", "batch", "split", "stack", "encode"]


def choose_pair_window(pairs: List[Dict[str, Any]]) -> int:
    """
    Chooses the first pair of the animated batch: the start of the window of PAIR_COUNT consecutive pairs of the
    manifest, among the first SEARCHED_WINDOWS, whose longer members have the most distinct numbers of frames, the
    earliest window winning a tie.

    Args:
        pairs (List[Dict[str, Any]]): The pairs of the training manifest, in its order.

    Returns:
        int: The index of the first pair of the window.
    """
    window_scores = []
    for start_index in range(min(SEARCHED_WINDOWS, len(pairs) - PAIR_COUNT + 1)):
        longer_members = {max(len(pairs[index][member]) for member in PAIR_MEMBERS)
                          for index in range(start_index, start_index + PAIR_COUNT)}
        window_scores.append((-len(longer_members), start_index))
    return min(window_scores)[1]


def build_pair_layout(member_lengths: List[List[int]], pair_names: List[str], class_names: List[str],
                      frame_images: List[List[List[str]]], vector_images: List[List[List[str]]],
                      zero_vector_image: str, shapes: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Places every tile and text of the animated pair batch at every phase: the two members of every pair on their own
    rows, the shorter member padded to the longer one, every pair on one row of the [B, 2T] batch with both members
    padded to the longest member of the batch, the two halves split into the before and after members, the members
    stacked as 2B sequences, and their frame vectors.

    Args:
        member_lengths (List[List[int]]): The real frames of the before and after member of every pair.
        pair_names (List[str]): The name of every pair.
        class_names (List[str]): The class of every pair.
        frame_images (List[List[List[str]]]): The frame images of every member of every pair, relative to the layout.
        vector_images (List[List[List[str]]]): The vector squares of the same frames, relative to the layout.
        zero_vector_image (str): The square of a zero vector, relative to the layout.
        shapes (Dict[str, str]): The shapes written in the texts.

    Returns:
        Dict[str, List[Dict[str, Any]]]: The 'tiles' and 'texts' of the layout.
    """
    pair_count = len(member_lengths)
    longest_member = max(max(lengths) for lengths in member_lengths)
    tiles, texts = [], []
    real_index = 0
    for pair_index, lengths in enumerate(member_lengths):
        pair_length = max(lengths)
        for member_index, member_length in enumerate(lengths):
            member_top = MEMBERS_TOP + pair_index * PAIR_STRIDE + member_index * MEMBER_STRIDE
            batch_left = FRAMES_LEFT + member_index * (longest_member * TILE_STRIDE + MEMBER_GAP)
            split_left = batch_left + member_index * SPLIT_GAP
            batch_top = BATCH_TOP + pair_index * BATCH_ROW_STRIDE
            stack_top = STACK_TOP + (member_index * pair_count + pair_index) * STACK_ROW_STRIDE
            for frame_index in range(longest_member):
                member_box = [FRAMES_LEFT + frame_index * TILE_STRIDE, member_top, TILE_SIDE, TILE_SIDE]
                batch_box = [batch_left + frame_index * TILE_STRIDE, batch_top, TILE_SIDE, TILE_SIDE]
                split_box = [split_left + frame_index * TILE_STRIDE, batch_top, TILE_SIDE, TILE_SIDE]
                stack_box = [FRAMES_LEFT + frame_index * TILE_STRIDE, stack_top, TILE_SIDE, TILE_SIDE]
                if frame_index < member_length:
                    # A real frame, moving with its member, then replaced by its vector
                    delay = round(real_index * DELAY_SECONDS, 3)
                    tiles.append({"image": frame_images[pair_index][member_index][frame_index],
                                  "second_image": vector_images[pair_index][member_index][frame_index],
                                  "fill": "black", "outline": "aqua" if member_index == 0 else "magenta",
                                  "states": {"members": {"box": member_box, "opacity": 1.0},
                                             "batch": {"box": batch_box, "opacity": 1.0, "delay": delay},
                                             "split": {"box": split_box, "opacity": 1.0},
                                             "stack": {"box": stack_box, "opacity": 1.0, "delay": delay},
                                             "encode": {"box": stack_box, "opacity": 1.0, "face": 1,
                                                        "delay": delay}}})
                    real_index += 1
                else:
                    # A padding frame of the pair dataset (up to the longer member) or of the collate (up to the
                    # longest member of the batch), then a zero vector
                    first_phase = "pair_padded" if frame_index < pair_length else "batch"
                    states = {"batch": {"box": batch_box, "opacity": 1.0}, "split": {"box": split_box,
                                                                                    "opacity": 1.0},
                              "stack": {"box": stack_box, "opacity": 1.0},
                              "encode": {"box": stack_box, "opacity": 1.0, "face": 1}}
                    if first_phase == "pair_padded":
                        states["pair_padded"] = {"box": member_box, "opacity": 1.0}
                    tiles.append({"image": None, "second_image": zero_vector_image, "fill": "black",
                                  "outline": "border", "label": "pad", "states": states})

            # Member labels of the stacked phases, and the shape of every member sample before the collate
            texts.append({"text": f"{pair_names[pair_index]} {PAIR_MEMBERS[member_index]}",
                          "position": [LABEL_RIGHT, stack_top + TILE_SIDE // 2], "anchor": "rm",
                          "font": "diagram_small", "colour": "aqua" if member_index == 0 else "magenta",
                          "phases": ["stack", "encode"]})
            texts.append({"text": f"{PAIR_MEMBERS[member_index]} · {member_length}",
                          "position": [FRAMES_LEFT + pair_length * TILE_STRIDE + 8, member_top + TILE_SIDE // 2],
                          "anchor": "lm", "font": "diagram_small", "colour": "aqua" if member_index == 0 else
                          "magenta", "phases": ["members", "pair_padded"]})

        # Pair labels of the first phases, beside its two members, then beside its row of the batch
        texts.append({"text": f"{pair_names[pair_index]}\n{class_names[pair_index]}",
                      "position": [LABEL_RIGHT, MEMBERS_TOP + pair_index * PAIR_STRIDE + (MEMBER_STRIDE + TILE_SIDE)
                                   // 2], "anchor": "rm", "font": "diagram_small", "colour": "text",
                      "phases": ["members", "pair_padded"]})
        texts.append({"text": f"{pair_names[pair_index]}\n{class_names[pair_index]}",
                      "position": [LABEL_RIGHT, BATCH_TOP + pair_index * BATCH_ROW_STRIDE + TILE_SIDE // 2],
                      "anchor": "rm", "font": "diagram_small", "colour": "text", "phases": ["batch", "split"]})

    # Shapes of the samples, the batch, its halves and the stacked members
    after_left = FRAMES_LEFT + longest_member * TILE_STRIDE + MEMBER_GAP
    batch_bottom = BATCH_TOP + (pair_count - 1) * BATCH_ROW_STRIDE + TILE_SIDE
    texts += [{"text": f"images {shapes['images']} uint8   sequence_mask {shapes['mask']}",
               "position": [FRAMES_LEFT, 40], "anchor": "la", "font": "diagram_monospace_small", "colour": "key",
               "phases": ["batch"]},
              {"text": "before", "position": [FRAMES_LEFT, batch_bottom + 24], "anchor": "la", "font": "diagram_small",
               "colour": "aqua", "phases": ["batch"]},
              {"text": "after", "position": [after_left, batch_bottom + 24], "anchor": "la", "font": "diagram_small",
               "colour": "magenta", "phases": ["batch"]},
              {"text": f"before_images {shapes['member_images']}\nbefore_mask {shapes['member_mask']}",
               "position": [FRAMES_LEFT, 28], "anchor": "la", "font": "diagram_monospace_small", "colour": "aqua",
               "phases": ["split"]},
              {"text": f"after_images {shapes['member_images']}\nafter_mask {shapes['member_mask']}",
               "position": [after_left + SPLIT_GAP, 28], "anchor": "la", "font": "diagram_monospace_small",
               "colour": "magenta", "phases": ["split"]},
              {"text": f"torch.cat([before, after])\n{shapes['stacked']}\n{shapes['stacked_count']} sequences, one "
                       f"call", "position": [FRAMES_LEFT + longest_member * TILE_STRIDE + 60, 200], "anchor": "la",
               "font": "diagram_monospace_small", "colour": "key", "phases": ["stack"]},
              {"text": f"encode_frames on the\n{shapes['real_frames']} real frames\n{shapes['features']}\nzeros on the "
                       f"padding", "position": [FRAMES_LEFT + longest_member * TILE_STRIDE + 60, 200],
               "anchor": "la", "font": "diagram_monospace_small", "colour": "key", "phases": ["encode"]}]
    return {"tiles": tiles, "texts": texts}


def make_pair_batch_animation(output_folder: Path, model: TaskModel, train_folder: Path,
                              label_dictionary: Dict[int, int], image_size: int, keep_ratio: bool,
                              force: bool) -> Dict[str, Any]:
    """
    Makes the frames, vector squares and layout file of the animated pair batch, and gives its facts: the pairs and
    their member lengths, the shapes of the samples, the batch, its halves and the stacked members, the real and
    padding frames, and the largest difference between the probabilities of the masked path (both members padded and
    encoded together) and the export path (every member alone, without mask). The encoding is reused from the
    previous build unless forced.

    Args:
        output_folder (Path): The figures folder of the series.
        model (TaskModel): The trained comparator, on the CPU.
        train_folder (Path): The training split of the run.
        label_dictionary (Dict[int, int]): The class index of every category of the run.
        image_size (int): The frame size of the run.
        keep_ratio (bool): Whether the run resizes its frames keeping their aspect ratio.
        force (bool): Whether the encoding runs again even when the previous build left it.

    Returns:
        Dict[str, Any]: The facts, named batch.*.
    """
    batch_folder = output_folder / "batch"
    layout_path = output_folder / "animations" / "pair_batch.json"
    facts_path = output_folder / "pair_batch_facts.json"
    facts = read_reused_facts(facts_path=facts_path, required_paths=[layout_path], force=force)
    if facts is not None:
        return facts

    # Pairs of the window, read and batched by the code of the loaders
    manifest = json.loads((train_folder / "labels.json").read_text(encoding="utf-8"))
    first_index = choose_pair_window(pairs=manifest["pairs"])
    dataset = SequencePairDataset(data_directory=str(train_folder / "data"), labels_file=str(train_folder /
                                  "labels.json"), image_size=image_size, keep_ratio=keep_ratio,
                                  label_dictionary=label_dictionary, sequence_settings={
                                      "maximum_sequence_length": 10 ** 6}, split_name="Training", shuffle=False)
    pair_indices = list(range(first_index, first_index + PAIR_COUNT))
    samples = [dataset[pair_index] for pair_index in pair_indices]
    batch = collate_padded_sequences(samples=samples, members_per_sample=len(PAIR_MEMBERS))
    before_images, after_images = split_pair_members(pair_tensor=batch["images"])
    before_mask, after_mask = split_pair_members(pair_tensor=batch["sequence_mask"])

    # Frame vectors of the stacked members, then the probabilities of the masked path and of the export path
    stacked_mask = torch.cat(tensors=[before_mask, after_mask], dim=0)
    with torch.no_grad():
        before_features, after_features = model.encode_members(
            before_images=before_images.float(), after_images=after_images.float(), before_mask=before_mask,
            after_mask=after_mask)
        masked_probabilities = model(before_images=before_images.float(), after_images=after_images.float(),
                                     before_mask=before_mask, after_mask=after_mask)["class_probabilities"]
        alone_probabilities = torch.cat(tensors=[model(
            before_images=before_images[index:index + 1, :int(before_mask[index].sum())].float(),
            after_images=after_images[index:index + 1, :int(after_mask[index].sum())].float())[
            "class_probabilities"] for index in range(PAIR_COUNT)])
    stacked_features = torch.cat(tensors=[before_features, after_features], dim=0)

    # Frames and vector squares of every member, the frames as the batch holds them
    batch_folder.mkdir(parents=True, exist_ok=True)
    pairs = [dataset.pairs[pair_index] for pair_index in pair_indices]
    member_lengths = [[len(pair[member]) for member in PAIR_MEMBERS] for pair in pairs]
    maximum_value = float(stacked_features.max())
    map_shape = (stacked_features.shape[-1], 1, 1)
    frame_images, vector_images = [], []
    for pair_number, lengths in enumerate(member_lengths):
        frame_images.append([])
        vector_images.append([])
        for member_index, (member_images, member_features) in enumerate([(before_images, before_features),
                                                                         (after_images, after_features)]):
            frame_images[-1].append([])
            vector_images[-1].append([])
            for frame_index in range(lengths[member_index]):
                frame_name = f"frame_{pair_number}_{PAIR_MEMBERS[member_index]}_{frame_index:02d}.png"
                small_frame = cv2.resize(src=member_images[pair_number, frame_index].permute(1, 2, 0).numpy(),
                                         dsize=(TILE_SIDE * 2, TILE_SIDE * 2), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(filename=str(batch_folder / frame_name), img=small_frame):
                    raise OSError(f"The frame {batch_folder / frame_name} could not be written.")
                vector_name = f"vector_{pair_number}_{PAIR_MEMBERS[member_index]}_{frame_index:02d}.png"
                write_vector_square(vector=member_features[pair_number, frame_index].numpy(), map_shape=map_shape,
                                    maximum_value=maximum_value, side_pixels=TILE_SIDE * 2,
                                    output_path=batch_folder / vector_name)
                frame_images[-1][-1].append(f"../batch/{frame_name}")
                vector_images[-1][-1].append(f"../batch/{vector_name}")
    write_vector_square(vector=numpy.zeros(shape=stacked_features.shape[-1]), map_shape=map_shape,
                        maximum_value=maximum_value, side_pixels=TILE_SIDE * 2,
                        output_path=batch_folder / "vector_zero.png")

    # Layout of the animation, its texts quoting the real shapes
    category_names = {category["id"]: category["name"] for category in manifest["categories"]}
    index_names = {class_index: category_names[identifier] for identifier, class_index in label_dictionary.items()}
    class_names = [index_names[pair["label"]] for pair in pairs]
    real_frames = int(stacked_mask.sum())
    shapes = {"images": format_shape(shape=list(batch["images"].shape)),
              "mask": format_shape(shape=list(batch["sequence_mask"].shape)),
              "member_images": format_shape(shape=list(before_images.shape)),
              "member_mask": format_shape(shape=list(before_mask.shape)),
              "stacked": format_shape(shape=[2 * PAIR_COUNT] + list(before_images.shape[1:])),
              "stacked_count": str(2 * PAIR_COUNT), "real_frames": str(real_frames),
              "features": format_shape(shape=list(stacked_features.shape))}
    layout = build_pair_layout(member_lengths=member_lengths, pair_names=[pair["name"] for pair in pairs],
                               class_names=class_names, frame_images=frame_images, vector_images=vector_images,
                               zero_vector_image="../batch/vector_zero.png", shapes=shapes)
    write_animation_layout(output_path=layout_path, phases=LAYOUT_PHASES, tiles=layout["tiles"],
                           texts=layout["texts"])

    # Facts: the pairs, their lengths, the shapes, the padding and the two paths of the model
    path_difference = float((masked_probabilities - alone_probabilities).abs().max())
    sample_lengths = [int(sample["sequence_mask"].shape[0]) // 2 for sample in samples]
    facts = {"batch.pairs": [pair["name"] for pair in pairs], "batch.classes": class_names,
             "batch.member_lengths": member_lengths, "batch.sample_member_lengths": sample_lengths,
             "batch.longest_member": int(before_images.shape[1]), "batch.pair_count": PAIR_COUNT,
             "batch.stacked_count": 2 * PAIR_COUNT, "batch.real_frames": real_frames,
             "batch.padding_frames": int((~stacked_mask).sum()),
             "batch.padding_share": round_fact(value=float((~stacked_mask).float().mean())),
             "batch.frame_features": int(stacked_features.shape[-1]),
             "batch.path_difference": path_difference,
             "batch.path_inverse_bound": 10 ** -math.ceil(math.log10(max(path_difference, 1e-15)))}
    for shape_name, shape_text in shapes.items():
        facts[f"batch.shape.{shape_name}"] = shape_text
    for pair_number, (pair, lengths) in enumerate(zip(pairs, member_lengths)):
        facts[f"batch.pair_{pair_number}.name"] = pair["name"]
        facts[f"batch.pair_{pair_number}.before"] = lengths[0]
        facts[f"batch.pair_{pair_number}.after"] = lengths[1]
    figure_paths = sorted(batch_folder.glob("*.png")) + [layout_path]
    write_reused_facts(facts_path=facts_path, figure_paths=figure_paths, facts=facts)
    return facts
