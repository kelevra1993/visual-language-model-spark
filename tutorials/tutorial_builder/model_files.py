"""
Downloads the models of the tutorial builder into its cache, each file checked against its SHA-256 checksum: the Kokoro
voice model and voices that narrate the videos (narration.py) and the faster-whisper base.en model that transcribes them
to check the narration (transcription_check.py). A file already in the cache with the right checksum is kept, so a
build downloads the models once per machine; a file whose checksum differs is downloaded again, and a download that
still differs stops the build.
"""

import hashlib
import urllib.request

from pathlib import Path
from typing import Dict

# Files of every model: their URL, their path in the models folder of the cache and their SHA-256 checksum. The
# faster-whisper files are pinned to one revision of the Hugging Face repository so that the checksums stay valid.
WHISPER_REVISION = "3d3d5dee26484f91867d81cb899cfcf72b96be6c"
WHISPER_URL = f"https://huggingface.co/Systran/faster-whisper-base.en/resolve/{WHISPER_REVISION}"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
MODEL_FILES = {
    "kokoro": {
        "kokoro-v1.0.onnx": {"url": f"{KOKORO_URL}/kokoro-v1.0.onnx",
                             "sha256": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"},
        "voices-v1.0.bin": {"url": f"{KOKORO_URL}/voices-v1.0.bin",
                            "sha256": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"}},
    "faster-whisper-base.en": {
        "config.json": {"url": f"{WHISPER_URL}/config.json",
                        "sha256": "f3bc3821e9fc76a27bae538e11ae5b677dcdd352b4600429ce7951d398569aeb"},
        "model.bin": {"url": f"{WHISPER_URL}/model.bin",
                      "sha256": "2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef"},
        "tokenizer.json": {"url": f"{WHISPER_URL}/tokenizer.json",
                           "sha256": "929c5252409436dce1b38a75d1abbcb5e132d170d8e324e4e04ed915fa2d22df"},
        "vocabulary.txt": {"url": f"{WHISPER_URL}/vocabulary.txt",
                           "sha256": "ff77588746d3a2595d32ab5b69ffd7b95ce2441ac57533cb66fc3eb575a115cf"}}}

# Size of the blocks read while hashing or downloading a file
READ_BLOCK_BYTES = 8 * 1024 * 1024


def compute_file_checksum(file_path: Path) -> str:
    """
    Computes the SHA-256 checksum of a model file, read block by block so that a large model is never held in memory.

    Args:
        file_path (Path): The file.

    Returns:
        str: The hexadecimal checksum.
    """
    checksum = hashlib.sha256()
    with open(file=file_path, mode="rb") as model_file:
        while block := model_file.read(READ_BLOCK_BYTES):
            checksum.update(block)
    return checksum.hexdigest()


def download_file(url: str, file_path: Path, expected_checksum: str) -> None:
    """
    Downloads one model file next to its final path, checks its checksum while it is written, then moves it into
    place, so that an interrupted download never leaves a partial file under the final name.

    Args:
        url (str): The URL of the file.
        file_path (Path): The final path of the file in the cache.
        expected_checksum (str): The SHA-256 checksum the file must have.

    Raises:
        ValueError: If the downloaded file does not have the expected checksum.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = file_path.with_name(f"{file_path.name}.part")
    checksum = hashlib.sha256()
    print(f"Downloading {url}")

    # Stream the response into the partial file, hashing every block
    with urllib.request.urlopen(url=url) as response, open(file=partial_path, mode="wb") as partial_file:
        while block := response.read(READ_BLOCK_BYTES):
            checksum.update(block)
            partial_file.write(block)
    if checksum.hexdigest() != expected_checksum:
        partial_path.unlink()
        raise ValueError(f"The download of {url} has the checksum {checksum.hexdigest()}, expected "
                         f"{expected_checksum}: the file changed upstream or the download was corrupted.")
    partial_path.replace(target=file_path)


def ensure_model_files(model_folder: Path) -> Dict[str, Path]:
    """
    Makes sure every model file is in the cache with its checksum, downloading the missing or changed ones, before the
    narration or the transcription loads them.

    Args:
        model_folder (Path): The models folder of the cache.

    Returns:
        Dict[str, Path]: The folder of every model, keyed by its name ("kokoro", "faster-whisper-base.en").
    """
    model_folders = {}
    for model_name, model_files in MODEL_FILES.items():
        for file_name, file_information in model_files.items():
            file_path = model_folder / model_name / file_name

            # Keep a cached file with the right checksum, download it otherwise
            if file_path.exists() and compute_file_checksum(file_path=file_path) == file_information["sha256"]:
                continue
            if file_path.exists():
                print(f"{file_path} has another checksum than {file_information['sha256']}, downloading it again")
            download_file(url=file_information["url"], file_path=file_path,
                          expected_checksum=file_information["sha256"])
        model_folders[model_name] = model_folder / model_name
    return model_folders
