"""
When run, builds the JPEG sequence and pair datasets of the loader benchmark of implementation_dali_loaders.md (phase
1) out of ImageNet, while no real sequence dataset exists: random ImageNet training images copied into one folder per
sequence (6 to 30 frames by default), and pairs made of two of these sequences, in the split layout read by
SequenceDataset and SequencePairDataset of vision_training/data/sequence.
"""

import json
import random
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from vision_training.data.label_mapping import write_label_mapping
from vision_training.data.sequence.sequence_pair_dataset import PAIR_MEMBERS
from vision_training.utilities.os_utilities import link_folder, print_color

# Categories of the benchmark sequences and pairs. The frames being random images, every sample gets a random
# category: the categories only give the loaders and models of the sequence tasks the labels they expect
BENCHMARK_CATEGORIES = [{"id": 1, "name": "random_category_1"}, {"id": 2, "name": "random_category_2"}]

# Extension of the copied frames, whose bytes are the unchanged ImageNet JPEG files
FRAME_EXTENSION = ".jpg"

# Number of threads copying the frames
NUMBER_COPY_THREADS = 16


def draw_sequence_frames(image_paths: List[Path], split_sizes: Dict[str, int], minimum_length: int,
                         maximum_length: int, random_generator: random.Random) -> Dict[str, List[List[Path]]]:
    """
    Draws the frames of every benchmark sequence among the ImageNet images: the length of every sequence uniformly
    between 'minimum_length' and 'maximum_length' frames, then distinct random images for all the frames, so that no
    image appears twice in the dataset and the loaders decode as many different photographs as frames.

    Args:
        image_paths (List[Path]): The paths of the ImageNet images the frames are drawn from.
        split_sizes (Dict[str, int]): The number of sequences of every split, keyed by split name.
        minimum_length (int): The smallest number of frames of a sequence.
        maximum_length (int): The largest number of frames of a sequence.
        random_generator (random.Random): The seeded generator of the draws.

    Returns:
        Dict[str, List[List[Path]]]: The source image of every frame of every sequence, keyed by split name.

    Raises:
        ValueError: If the lengths are inconsistent, or if there are fewer images than frames to draw.
    """
    if not 1 <= minimum_length <= maximum_length:
        raise ValueError(f"The sequence lengths must satisfy 1 <= minimum_length <= maximum_length, got "
                         f"{minimum_length} and {maximum_length}.")

    # Length of every sequence of every split
    split_lengths = {split_name: [random_generator.randint(a=minimum_length, b=maximum_length)
                                  for _ in range(number_sequences)]
                     for split_name, number_sequences in split_sizes.items()}
    number_frames = sum(sum(lengths) for lengths in split_lengths.values())
    if number_frames > len(image_paths):
        raise ValueError(f"{number_frames} frames to draw from only {len(image_paths)} images.")

    # Distinct images for all the frames, handed out to the sequences in order
    drawn_images = random_generator.sample(population=image_paths, k=number_frames)
    split_sequences = {}
    image_index = 0
    for split_name, lengths in split_lengths.items():
        split_sequences[split_name] = []
        for length in lengths:
            split_sequences[split_name].append(drawn_images[image_index:image_index + length])
            image_index += length
    return split_sequences


def create_imagenet_sequence_dataset(imagenet_split_folder: Path, output_folder: Path, split_sizes: Dict[str, int],
                                     minimum_length: int, maximum_length: int, seed: int) -> None:
    """
    Builds the sequence dataset of the loader benchmark: every sequence is a folder of random ImageNet images copied
    as its frames ('<output_folder>/<split>/data/<split>_<index>/<frame>.jpg'), listed with a random category in
    '<output_folder>/<split>/labels.json', with the 'category_id' read by the sequence classification task and the
    'frame_category_ids' read by the sequence labelling task. It gives the sequence loaders the decoding cost of real
    photographs. The label mapping is written to '<output_folder>/label_mapping.json'.

    Args:
        imagenet_split_folder (Path): An ImageNet split prepared by imagenet_download.py ('data' folder and
            'labels.json' manifest), whose images are copied.
        output_folder (Path): The root folder of the created splits.
        split_sizes (Dict[str, int]): The number of sequences of every split, keyed by split name.
        minimum_length (int): The smallest number of frames of a sequence.
        maximum_length (int): The largest number of frames of a sequence.
        seed (int): The seed of the draws of the lengths, images and categories.

    Raises:
        FileExistsError: If a split folder already exists, so that an existing dataset is never mixed with a new one.
    """
    # Never write into an existing split, whose frames would be mixed with the new ones
    for split_name in split_sizes:
        if (output_folder / split_name).exists():
            raise FileExistsError(f"The split folder already exists, delete it to build the dataset again : "
                                  f"file://{(output_folder / split_name).absolute()}")

    # Images of the ImageNet split, from its manifest
    manifest = json.loads(s=(imagenet_split_folder / "labels.json").read_text())
    image_paths = [imagenet_split_folder / "data" / sample["file_name"] for sample in manifest["samples"]]
    print_color(color="blue", output=f"{len(image_paths)} ImageNet images : file://{imagenet_split_folder}")

    # Frames of every sequence, drawn with a seeded generator so that the dataset can be built again identically
    random_generator = random.Random(seed)
    split_sequences = draw_sequence_frames(image_paths=image_paths, split_sizes=split_sizes,
                                           minimum_length=minimum_length, maximum_length=maximum_length,
                                           random_generator=random_generator)

    for split_name, sequences in split_sequences.items():
        split_folder = output_folder / split_name

        # One record per sequence with its frame paths relative to the data folder, and the copies to make
        records, copies = [], []
        for sequence_index, source_paths in enumerate(sequences):
            sequence_name = f"{split_name}_{sequence_index:06d}"
            (split_folder / "data" / sequence_name).mkdir(parents=True)
            frames = [f"{sequence_name}/{frame_index:06d}{FRAME_EXTENSION}" for frame_index in range(len(source_paths))]
            category_identifier = random_generator.choice(seq=BENCHMARK_CATEGORIES)["id"]
            records.append({"id": sequence_index + 1, "name": sequence_name, "frames": frames,
                            "category_id": category_identifier,
                            "frame_category_ids": [category_identifier] * len(frames)})
            copies.extend(zip(source_paths, [split_folder / "data" / frame for frame in frames]))

        # Copy the frames with several threads, the copies being bound by the disk
        with ThreadPoolExecutor(max_workers=NUMBER_COPY_THREADS) as executor:
            futures = [executor.submit(shutil.copyfile, src=source_path, dst=destination_path)
                       for source_path, destination_path in copies]
            for future in tqdm(iterable=as_completed(fs=futures), total=len(futures),
                               desc=f"Copying {split_name} frames"):
                future.result()

        # Manifest of the split, read by the sequence dataset
        labels_path = split_folder / "labels.json"
        labels_path.write_text(data=json.dumps(obj={"categories": BENCHMARK_CATEGORIES, "sequences": records}))
        lengths = [len(record["frames"]) for record in records]
        print_color(color="green", output=f"{split_name} split : {len(records)} sequences, {sum(lengths)} frames "
                                          f"({min(lengths)} to {max(lengths)} per sequence) : file://{labels_path}")

    # Class index of every category, from 0 since the sequence models predict no background class
    write_label_mapping(categories=BENCHMARK_CATEGORIES, output_folder=output_folder)


def create_sequence_pair_dataset(sequence_dataset_folder: Path, output_folder: Path, split_sizes: Dict[str, int],
                                 seed: int) -> None:
    """
    Builds the pair dataset of the loader benchmark out of the benchmark sequences, without copying any frame: every
    pair takes two different sequences of the same split as its before and after members, with a random category.
    '<output_folder>/<split>/data' links to the data folder of the sequence split, '<output_folder>/<split>/labels.json'
    lists the pairs read by the sequence pair dataset, and '<output_folder>/label_mapping.json' maps the categories.

    Args:
        sequence_dataset_folder (Path): The root folder of the sequence dataset built by
            create_imagenet_sequence_dataset.
        output_folder (Path): The root folder of the created splits.
        split_sizes (Dict[str, int]): The number of pairs of every split, keyed by split name.
        seed (int): The seed of the draws of the members and categories.

    Raises:
        FileExistsError: If the 'data' path of a created split exists and is not a symbolic link.
    """
    random_generator = random.Random(seed)
    for split_name, number_pairs in split_sizes.items():
        sequence_split_folder = sequence_dataset_folder / split_name
        sequences = json.loads(s=(sequence_split_folder / "labels.json").read_text())["sequences"]

        # Link the frames of the sequence split as the data folder of the pair split
        split_folder = output_folder / split_name
        split_folder.mkdir(parents=True, exist_ok=True)
        link_folder(link_path=split_folder / "data", target_folder=sequence_split_folder / "data")

        # Two different sequences of the split as the before and after members of every pair
        pairs: List[Dict[str, Any]] = []
        for pair_index in range(number_pairs):
            member_sequences = random_generator.sample(population=sequences, k=len(PAIR_MEMBERS))
            pairs.append({"id": pair_index + 1, "name": f"{split_name}_{pair_index:06d}",
                          **{member_name: sequence["frames"] for member_name, sequence in zip(PAIR_MEMBERS,
                                                                                            member_sequences)},
                          "category_id": random_generator.choice(seq=BENCHMARK_CATEGORIES)["id"]})

        # Manifest of the split, read by the sequence pair dataset
        labels_path = split_folder / "labels.json"
        labels_path.write_text(data=json.dumps(obj={"categories": BENCHMARK_CATEGORIES, "pairs": pairs}))
        print_color(color="green", output=f"{split_name} split : {len(pairs)} pairs : file://{labels_path}")

    # Class index of every category, from 0 since the comparison model predicts no background class
    write_label_mapping(categories=BENCHMARK_CATEGORIES, output_folder=output_folder)


if __name__ == "__main__":
    # ImageNet training split prepared by vision_training/data/classification/imagenet_download.py, whose images are
    # copied as the frames of the benchmark sequences
    imagenet_split_folder = Path("/home/robert_kelevra/Data/object_detection_data/imagenet-1k/manifests/train")
    # Root folders of the created sequence and pair datasets, next to the synthetic sequence datasets
    sequence_dataset_folder = Path("/home/robert_kelevra/Data/sequence_data/imagenet_sequences")
    pair_dataset_folder = Path("/home/robert_kelevra/Data/sequence_data/imagenet_pairs")
    # Number of sequences of every split (1,000 in total), also used as the number of pairs of every split
    split_sizes = {"train": 800, "validation": 100, "test": 100}
    # Lengths of the sequences, drawn uniformly between these numbers of frames
    minimum_length = 6
    maximum_length = 30
    # Seed of every draw, so that the datasets can be built again identically
    seed = 0

    create_imagenet_sequence_dataset(imagenet_split_folder=imagenet_split_folder, output_folder=sequence_dataset_folder,
                                     split_sizes=split_sizes, minimum_length=minimum_length,
                                     maximum_length=maximum_length, seed=seed)
    create_sequence_pair_dataset(sequence_dataset_folder=sequence_dataset_folder, output_folder=pair_dataset_folder,
                                 split_sizes=split_sizes, seed=seed)
