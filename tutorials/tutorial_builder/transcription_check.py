"""
Check of the narration of a built tutorial video, since nobody listens to every video after every build: its audio is
transcribed back to text by faster-whisper (base.en, downloaded by model_files.py) on the CPU and compared word by word
with the sentences of its chapter record, giving the word error rate and every place where the heard words differ from
the script. A high rate points at words the voice says unclearly (pretrained read "patrained", dataset heard as "stair
set"), to be rewritten in the storyboard. The numbers written one way in the script and heard the other way (words
heard as digits, digits heard as words, digits grouped differently), the British spellings of the script heard in their
American spelling (labelled heard "labeled") and the words heard split or joined (tensorboard heard "tensor board") are
counted apart, being a difference of spelling rather than of speech.
"""

import difflib
import re
import subprocess

import imageio_ffmpeg
import numpy

from faster_whisper import WhisperModel
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Words of the numbers the narration writes out so that the voice reads them clearly, which the recognizer writes in
# digits ("five hundred and twelve" heard as "512")
NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
                "point", "and", "a", "hundredth", "tenth", "fifth", "half", "percent"}

# British spellings of the narration and their American spelling, the one the recognizer writes
BRITISH_SPELLINGS = {"labelled": "labeled", "labelling": "labeling", "labeller": "labeler", "modelled": "modeled",
                     "modelling": "modeling", "cancelled": "canceled", "travelled": "traveled", "colour": "color",
                     "colours": "colors", "coloured": "colored", "centre": "center", "centres": "centers",
                     "centred": "centered", "metre": "meter", "neighbour": "neighbor", "neighbours": "neighbors",
                     "neighbouring": "neighboring", "behaviour": "behavior", "favour": "favor", "grey": "gray",
                     "greyscale": "grayscale", "normalise": "normalize", "normalised": "normalized",
                     "normalisation": "normalization", "optimise": "optimize", "optimised": "optimized",
                     "initialise": "initialize", "initialised": "initialized", "recognise": "recognize",
                     "summarise": "summarize", "visualise": "visualize", "analyse": "analyze", "analysed": "analyzed"}


def decode_audio(media_path: Path) -> numpy.ndarray:
    """
    Decodes the audio track of a video or audio file into the 16 kHz mono float32 samples faster-whisper expects, with
    the ffmpeg of imageio-ffmpeg (the PyAV reader of faster-whisper fails with some PyAV versions).

    Args:
        media_path (Path): The video or audio file.

    Returns:
        numpy.ndarray: The samples in [-1, 1].
    """
    raw_samples = subprocess.run(args=[imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(media_path), "-ar",
                                       "16000", "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return numpy.frombuffer(buffer=raw_samples, dtype=numpy.float32)


def normalize_words(text: str) -> List[str]:
    """
    Lowers a text and keeps its letters and digits only, split into words, so that the script and the transcription
    are compared without punctuation or case.

    Args:
        text (str): The text.

    Returns:
        List[str]: The words.
    """
    return re.sub(pattern=r"[^a-z0-9 ]", repl=" ", string=text.lower().replace("-", " ")).split()


def is_number_token(word: str) -> bool:
    """
    Tells whether a word of the script or of the transcription belongs to the writing of a number: digits (with an
    ordinal ending) or one of NUMBER_WORDS.

    Args:
        word (str): The normalized word.

    Returns:
        bool: True for a word of a number.
    """
    return word in NUMBER_WORDS or re.fullmatch(pattern=r"\d+(st|nd|rd|th)?", string=word) is not None


def is_number_spelling(script_words: List[str], heard_words: List[str]) -> bool:
    """
    Tells whether a difference between the script and the transcription only writes a number two ways, in words on one
    side and in digits on the other, or in digits grouped differently ("1 000" and "1000"), which is no error of the
    voice: both sides are then made of the words and digits of numbers only.

    Args:
        script_words (List[str]): The words of the script in the difference.
        heard_words (List[str]): The words heard in the difference.

    Returns:
        bool: True for a number written two ways.
    """
    return (bool(script_words) and bool(heard_words) and all(is_number_token(word=word) for word in script_words)
            and all(is_number_token(word=word) for word in heard_words))


def is_word_spelling(script_words: List[str], heard_words: List[str]) -> bool:
    """
    Tells whether a difference between the script and the transcription only spells the same words another way: the
    British spelling of the script heard in its American spelling, or words heard split or joined ("tensorboard" and
    "tensor board"), which is no error of the voice.

    Args:
        script_words (List[str]): The words of the script in the difference.
        heard_words (List[str]): The words heard in the difference.

    Returns:
        bool: True for the same words spelled another way.
    """
    american_script_words = [BRITISH_SPELLINGS.get(word, word) for word in script_words]
    american_heard_words = [BRITISH_SPELLINGS.get(word, word) for word in heard_words]
    return bool(script_words) and bool(heard_words) and "".join(american_script_words) == "".join(american_heard_words)


def compare_with_script(script: str, transcription: str) -> Tuple[float, float, List[str]]:
    """
    Aligns the words heard in the narration with the words of the script and counts the words that differ, once with
    every difference and once without the spelling differences: the numbers written one way in the script and heard
    the other way, the British spellings heard in American and the words heard split or joined.

    Args:
        script (str): The sentences of the chapter.
        transcription (str): The text heard by faster-whisper.

    Returns:
        Tuple[float, float, List[str]]: The word error rate, the word error rate without the spelling differences, and
            one line per difference, the number and word spellings marked.
    """
    script_words = normalize_words(text=script)
    heard_words = normalize_words(text=transcription)
    matcher = difflib.SequenceMatcher(a=script_words, b=heard_words, autojunk=False)
    differences = []
    error_count = 0
    spelling_count = 0
    for tag, script_start, script_end, heard_start, heard_end in matcher.get_opcodes():
        if tag == "equal":
            continue

        # Every difference counted, the numbers and the words spelled two ways counted apart
        difference_size = max(script_end - script_start, heard_end - heard_start)
        error_count += difference_size
        script_part, heard_part = script_words[script_start:script_end], heard_words[heard_start:heard_end]
        spelling = ("number" if is_number_spelling(script_words=script_part, heard_words=heard_part) else
                    "spelling" if is_word_spelling(script_words=script_part, heard_words=heard_part) else None)
        spelling_count += difference_size if spelling else 0
        differences.append(f"{tag}: script {' '.join(script_part)!r} -> heard {' '.join(heard_part)!r}"
                           f"{f' ({spelling})' if spelling else ''}")
    word_count = max(len(script_words), 1)
    return error_count / word_count, (error_count - spelling_count) / word_count, differences


def check_video_narration(video_path: Path, record: Dict[str, Any], whisper_folder: Path,
                          thread_count: int) -> Dict[str, Any]:
    """
    Transcribes the narration of a built video and compares it with the sentences of its chapter record.

    Args:
        video_path (Path): The MP4 file.
        record (Dict[str, Any]): The record of the chapter written by make_video, with its scenes and sentences.
        whisper_folder (Path): The faster-whisper base.en folder of the models cache.
        thread_count (int): The threads of the build (--threads), given to faster-whisper.

    Returns:
        Dict[str, Any]: The word error rates (all differences, spelling differences excluded) and the differences.
    """
    model = WhisperModel(model_size_or_path=str(whisper_folder), device="cpu", compute_type="int8",
                         cpu_threads=thread_count)
    segments, _ = model.transcribe(audio=decode_audio(media_path=video_path), beam_size=5)
    transcription = " ".join(segment.text.strip() for segment in segments)
    script = " ".join(sentence["text"] for scene in record["scenes"] for sentence in scene["sentences"])
    word_error_rate, spelling_free_word_error_rate, differences = compare_with_script(script=script,
                                                                                      transcription=transcription)
    return {"word_error_rate": round(word_error_rate, 4),
            "word_error_rate_without_spellings": round(spelling_free_word_error_rate, 4), "differences": differences}
