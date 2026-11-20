"""
Reading of the 'labels.json' manifests of the data splits, parsed once per training or evaluation launch: the datasets
of the COCO tasks, the record of the data of the run (data_record.py), the description of the splits
(split_description.py), the test manifest of the COCO trainers, the keypoint definitions and the resume check read the
same files while the trainer is built, which parsed the manifest of a training split up to five times (each time about
30 seconds and 11 GB of memory at its peak for the 3.79 GB manifest of the 512,000 training images of the instance
segmentation benchmark dataset). Within read_manifests_once, used around the construction of the trainer by
vision_training/main.py and scripts/evaluation/evaluate_checkpoint.py, a manifest is parsed once and shared by its
readers, then freed with its heap memory before the first batch forks the loader workers; outside it, every read parses
its file. The readers never modify a shared manifest.
"""

import contextlib
import json

from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from vision_training.utilities.os_utilities import release_freed_heap_memory

# Manifests parsed during the launch in progress, keyed by the resolved path, size and modification time of their file,
# None outside read_manifests_once, every read then parsing its file
shared_manifests: Optional[Dict[Tuple[str, int, int], Dict[str, Any]]] = None


def read_manifest(labels_path: str | Path) -> Dict[str, Any]:
    """
    Parses a 'labels.json' manifest, or gives the one already parsed by another reader of the launch in progress
    (read_manifests_once) when its file has the same size and modification time, so that every manifest is parsed once
    per launch.

    Args:
        labels_path (str | Path): The 'labels.json' file of a split.

    Returns:
        Dict[str, Any]: The parsed manifest, shared with the other readers of the launch, which never modify it.
    """
    # The file identified by its resolved path, size and modification time, so that a rewritten file is parsed again
    resolved_path = Path(labels_path).resolve()
    file_status = resolved_path.stat()
    manifest_key = (str(resolved_path), file_status.st_size, file_status.st_mtime_ns)
    if shared_manifests is not None and manifest_key in shared_manifests:
        return shared_manifests[manifest_key]

    # Parsed from the file, and kept for the other readers of the launch in progress
    with open(file=resolved_path, mode="r") as file_handler:
        manifest = json.load(fp=file_handler)
    if shared_manifests is not None:
        shared_manifests[manifest_key] = manifest
    return manifest


@contextlib.contextmanager
def read_manifests_once() -> Iterator[None]:
    """
    Shares the manifests parsed during the launch of a training or an evaluation between their readers (read_manifest),
    then frees them and gives their heap memory back to the system once the launch ends, normally or by an error, so
    that the loader workers forked by the first batch never inherit them.

    Yields:
        None: The launch runs inside the block.
    """
    global shared_manifests
    shared_manifests = {}
    try:
        yield
    finally:
        # The parsed manifests freed with their last reference, then their memory given back
        shared_manifests = None
        release_freed_heap_memory()
