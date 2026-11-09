"""
Narration of the tutorial videos: every sentence of a storyboard is synthesized on its own with the offline Kokoro voice
(downloaded by model_files.py) at the pace of the storyboard, and kept in the narration cache under a key made of the
voice, the pace, the model checksums and the text, so that a later build of an edited storyboard only synthesizes the
sentences that changed. The sentences are then placed on the time axis of the video (build_timeline), the timeline
driving the frames of frame_generation.py, the subtitles and the narration track encoded by video_encoding.py.
"""

import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import wave

import espeakng_loader
import numpy
import onnxruntime

from kokoro_onnx import Kokoro
from kokoro_onnx.config import EspeakConfig
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tutorial_builder.layout import FRAMES_PER_SECOND
from tutorial_builder.model_files import MODEL_FILES

# Spoken forms of the words the voice reads wrongly, given to Kokoro in place of the written word (whole words only)
# while the captions keep the written one: "ONNX" is read "onks", its usual pronunciation is "onyx". A doubtful
# word is checked by synthesizing it alone and transcribing it back before it is added here
SPOKEN_FORMS = {"ONNX": "onyx"}

# Decimal number of a sentence (0.5, 1,234.25, the 35.3 of 35.3%), neither inside a word (x1.05) nor part of a version
# (2.1.319): espeak-ng reads "0.5" of a sentence ending with a period as "zero. five", the voice pausing at that dot
DECIMAL_NUMBER_PATTERN = re.compile(r"(?<![\w.])(\d[\d,]*)\.(\d+)(?!\w|\.\d)")

# Words of the digits read one by one after the decimal point, as English reads 0.837 "zero point eight three seven"
DIGIT_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
               "8": "eight", "9": "nine"}

# Pauses of the narration at a pace of 1, divided by the pace of the storyboard: before the first sentence of a scene,
# between two sentences and after the last one
SCENE_LEAD_SECONDS = 0.6
SENTENCE_PAUSE_SECONDS = 0.45
SCENE_TAIL_SECONDS = 0.9

# Sample rate of the Kokoro voices
KOKORO_SAMPLE_RATE = 24000

# Longest path of the espeak-ng data espeak-ng reads, with the file names it appends: past its 160-character buffer it
# silently reads the folder it was compiled with, which does not exist ("Error processing file .../phontab")
ESPEAK_DATA_PATH_LIMIT = 140


def prepare_espeak_data(cache_folder: Path) -> Path:
    """
    Copies the espeak-ng data of espeakng-loader, which Kokoro uses to turn the sentences into phonemes, into the cache
    under a short path named after its version: inside an environment with a long path (a worktree), the data folder
    of espeakng-loader is longer than the buffer espeak-ng reads it into, and espeak-ng stops the process.

    Args:
        cache_folder (Path): The cache folder of the build.

    Returns:
        Path: The espeak-ng data folder of the cache.

    Raises:
        ValueError: If even the path of the cache is too long for espeak-ng.
    """
    data_folder = cache_folder / f"espeak-ng-data-{importlib.metadata.version('espeakng-loader')}"
    if len(str(data_folder)) + len("/espeak-ng-data") > ESPEAK_DATA_PATH_LIMIT:
        raise ValueError(f"The espeak-ng data path {data_folder} is too long for espeak-ng: give a shorter "
                         f"--cache-folder.")
    if not (data_folder / "phontab").exists():
        shutil.copytree(src=espeakng_loader.get_data_path(), dst=data_folder, dirs_exist_ok=True)
    return data_folder


def load_kokoro_voice(kokoro_folder: Path, espeak_data_folder: Path, thread_count: int) -> Kokoro:
    """
    Loads the Kokoro model narrating the videos, run by ONNX Runtime on the CPU only and on the threads of the build,
    so that the trainings keep the GPU and, with the default thread count, most of the processors.

    Args:
        kokoro_folder (Path): The Kokoro folder of the models cache, with kokoro-v1.0.onnx and voices-v1.0.bin.
        espeak_data_folder (Path): The espeak-ng data folder of prepare_espeak_data.
        thread_count (int): The threads of the build (--threads).

    Returns:
        Kokoro: The loaded model.
    """
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = thread_count
    session_options.inter_op_num_threads = 1
    session = onnxruntime.InferenceSession(path_or_bytes=str(kokoro_folder / "kokoro-v1.0.onnx"),
                                           sess_options=session_options, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(session=session, voices_path=str(kokoro_folder / "voices-v1.0.bin"),
                               espeak_config=EspeakConfig(data_path=str(espeak_data_folder)))


def convert_to_spoken_text(sentence: str) -> str:
    """
    Gives the text the voice reads for a sentence of the narration: the sentence with every word of SPOKEN_FORMS
    replaced by its spoken form and the decimal part of every number written as words after "point" (0.5 read "0 point
    five"), the captions and the narration check keeping the written sentence.

    Args:
        sentence (str): The sentence, with its facts already filled in.

    Returns:
        str: The text given to Kokoro.
    """
    spoken_text = sentence
    for written_word, spoken_word in SPOKEN_FORMS.items():
        spoken_text = re.sub(pattern=rf"\b{re.escape(written_word)}\b", repl=spoken_word, string=spoken_text)

    # Write the digits after the decimal point as words, so that no dot is left inside a number for espeak-ng to take
    # for the end of a sentence; the integer part stays in digits, which espeak-ng reads right
    return DECIMAL_NUMBER_PATTERN.sub(repl=spell_decimal_part, string=spoken_text)


def spell_decimal_part(match: re.Match) -> str:
    """
    Gives the spoken form of one decimal number of the narration for convert_to_spoken_text: its integer part in digits,
    then "point" and every digit of its decimal part as a word, so that the voice reads 0.837 "zero point eight three
    seven" without the pause espeak-ng puts at the dot of a sentence ending with a period.

    Args:
        match (re.Match): The number matched by DECIMAL_NUMBER_PATTERN, its integer and decimal parts as groups 1 and 2.

    Returns:
        str: The spoken form, e.g. "0 point eight three seven".
    """
    return f"{match.group(1)} point {' '.join(DIGIT_WORDS[digit] for digit in match.group(2))}"


def compute_sentence_key(text: str, voice_settings: Dict[str, Any], pace: float) -> str:
    """
    Gives the cache key of a synthesized sentence: the SHA-256 of its text, voice, language, pace and of the checksums
    of the Kokoro files, so that changing any of them synthesizes the sentence again.

    Args:
        text (str): The spoken text of the sentence (convert_to_spoken_text), with its facts already filled in.
        voice_settings (Dict[str, Any]): The voice of the storyboard, with its name and optional language.
        pace (float): The pace of the storyboard.

    Returns:
        str: The hexadecimal key.
    """
    key_fields = {"text": text, "voice": voice_settings["name"], "language": voice_settings.get("language", "en-us"),
                  "pace": round(pace, 4), "model": {file_name: file_information["sha256"]
                                                   for file_name, file_information in MODEL_FILES["kokoro"].items()}}
    return hashlib.sha256(json.dumps(obj=key_fields, sort_keys=True).encode(encoding="utf-8")).hexdigest()


def write_wav(samples: numpy.ndarray, sample_rate: int, wav_path: Path) -> None:
    """
    Writes int16 samples as a mono 16-bit WAV file: a cached sentence, or the narration track of a video.

    Args:
        samples (numpy.ndarray): The int16 samples.
        sample_rate (int): The sample rate.
        wav_path (Path): The WAV file.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(f=str(wav_path), mode="wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def read_wav(wav_path: Path) -> Tuple[numpy.ndarray, int]:
    """
    Reads a mono 16-bit WAV file written by write_wav, a sentence of the narration cache.

    Args:
        wav_path (Path): The WAV file.

    Returns:
        Tuple[numpy.ndarray, int]: The int16 samples and their sample rate.

    Raises:
        ValueError: If the file is not mono 16-bit.
    """
    with wave.open(f=str(wav_path), mode="rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError(f"{wav_path} is not a mono 16-bit WAV file of the narration cache.")
        sample_rate = wav_file.getframerate()
        samples = numpy.frombuffer(buffer=wav_file.readframes(wav_file.getnframes()), dtype=numpy.int16).copy()
    return samples, sample_rate


def synthesize_narration(storyboard: Dict[str, Any], kokoro_folder: Path, narration_folder: Path, cache_folder: Path,
                         thread_count: int) -> Tuple[List[List[numpy.ndarray]], int, Dict[str, int]]:
    """
    Gives the samples of every sentence of every scene, read from the narration cache or synthesized with the voice of
    the storyboard and cached, the Kokoro model being loaded only when a sentence is missing from the cache.

    Args:
        storyboard (Dict[str, Any]): The loaded storyboard, with its voice, pace and scenes.
        kokoro_folder (Path): The Kokoro folder of the models cache.
        narration_folder (Path): The narration folder of the cache.
        cache_folder (Path): The cache folder, receiving the espeak-ng data of Kokoro.
        thread_count (int): The threads of the build (--threads), given to Kokoro.

    Returns:
        Tuple[List[List[numpy.ndarray]], int, Dict[str, int]]: The int16 samples of every sentence of every scene,
            their sample rate, and the counts of cached and synthesized sentences.
    """
    voice_model: Optional[Kokoro] = None
    sentence_audio = []
    counts = {"cached": 0, "synthesized": 0}
    voice_settings = storyboard["voice"]
    for scene in storyboard["scenes"]:
        scene_audio = []
        for sentence in scene["narration"]:
            # The voice reads the spoken form of the sentence, which also keys its cache
            spoken_text = convert_to_spoken_text(sentence=sentence)
            sentence_key = compute_sentence_key(text=spoken_text, voice_settings=voice_settings,
                                                pace=storyboard["pace"])
            sentence_path = narration_folder / f"{sentence_key}.wav"

            # A cached sentence is read back, a missing one synthesized at the pace of the storyboard and cached
            if sentence_path.exists():
                samples, sample_rate = read_wav(wav_path=sentence_path)
                counts["cached"] += 1
            else:
                if voice_model is None:
                    voice_model = load_kokoro_voice(kokoro_folder=kokoro_folder,
                                                    espeak_data_folder=prepare_espeak_data(cache_folder=cache_folder),
                                                    thread_count=thread_count)
                float_samples, sample_rate = voice_model.create(text=spoken_text, voice=voice_settings["name"],
                                                                speed=storyboard["pace"],
                                                                lang=voice_settings.get("language", "en-us"))
                samples = (numpy.clip(a=float_samples, a_min=-1.0, a_max=1.0) * 32767).astype(numpy.int16)
                write_wav(samples=samples, sample_rate=sample_rate, wav_path=sentence_path)
                counts["synthesized"] += 1
            scene_audio.append(samples)
        sentence_audio.append(scene_audio)
    return sentence_audio, KOKORO_SAMPLE_RATE, counts


def build_timeline(scenes: List[Dict[str, Any]], sentence_audio: List[List[numpy.ndarray]], sample_rate: int,
                   pace: float) -> Tuple[List[Dict[str, Any]], numpy.ndarray, int]:
    """
    Places the scenes and their sentences on the time axis of the video, a lead before the first sentence of a scene,
    a pause between two sentences and a tail after the last one, all divided by the pace (a scene can lengthen the
    pause after one of its sentences with hold_after, to let an animation finish), and assembles the narration track,
    padded to the duration of the last video frame.

    Args:
        scenes (List[Dict[str, Any]]): The scenes of the storyboard.
        sentence_audio (List[List[numpy.ndarray]]): The samples of every sentence, from synthesize_narration.
        sample_rate (int): The sample rate of the voice.
        pace (float): The pace of the storyboard.

    Returns:
        Tuple[List[Dict[str, Any]], numpy.ndarray, int]: The start and end of every scene with the text, start and end
            of its sentences, the int16 narration track, and the number of video frames.
    """
    timeline = []
    placed_sentences = []
    current_time = 0.0
    for scene, scene_audio in zip(scenes, sentence_audio):
        scene_start = current_time
        current_time += SCENE_LEAD_SECONDS / pace
        sentences = []
        hold_after = scene.get("hold_after", {})
        for sentence_index, (sentence, samples) in enumerate(zip(scene["narration"], scene_audio)):
            sentence_start = current_time
            current_time += len(samples) / sample_rate
            sentences.append({"text": sentence, "start": sentence_start, "end": current_time})
            placed_sentences.append((sentence_start, samples))
            is_last_sentence = sentence_index == len(scene_audio) - 1
            current_time += (SCENE_TAIL_SECONDS if is_last_sentence else SENTENCE_PAUSE_SECONDS) / pace
            current_time += float(hold_after.get(sentence_index, 0.0))
        timeline.append({"start": scene_start, "end": current_time, "sentences": sentences})

    # Narration track as long as the frames, every sentence written at its start
    number_frames = math.ceil(current_time * FRAMES_PER_SECOND)
    narration = numpy.zeros(shape=(round(number_frames / FRAMES_PER_SECOND * sample_rate),), dtype=numpy.int16)
    for sentence_start, samples in placed_sentences:
        first_sample = round(sentence_start * sample_rate)
        narration[first_sample:first_sample + len(samples)] = samples[:len(narration) - first_sample]
    return timeline, narration, number_frames


def format_subtitle_time(seconds: float) -> str:
    """
    Formats a time of the video as a SubRip timestamp, hours:minutes:seconds,milliseconds.

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: The timestamp.
    """
    milliseconds = round(seconds * 1000)
    return (f"{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:{milliseconds // 1000 % 60:02d},"
            f"{milliseconds % 1000:03d}")


def write_subtitles(timeline: List[Dict[str, Any]], subtitles_path: Path) -> None:
    """
    Writes the sentences of the narration with their times as a SubRip file next to the video, for players or editors
    that show their own subtitles.

    Args:
        timeline (List[Dict[str, Any]]): The scenes of build_timeline.
        subtitles_path (Path): The .srt file.
    """
    entries = []
    sentences = [sentence for scene_timing in timeline for sentence in scene_timing["sentences"]]
    for sentence_index, sentence in enumerate(sentences):
        entries.append(f"{sentence_index + 1}\n{format_subtitle_time(seconds=sentence['start'])} --> "
                       f"{format_subtitle_time(seconds=sentence['end'])}\n{sentence['text']}\n")
    subtitles_path.write_text(data="\n".join(entries), encoding="utf-8")
