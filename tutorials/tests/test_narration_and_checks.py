"""
Tests of the narration and of the checks of the tutorial build: the timeline of the sentences (tutorial_builder/
narration.py), the cache key of a synthesized sentence, the WAV files of the narration cache, the comparison of the
transcribed narration with its script (transcription_check.py), and the index page of the output folder
(index_page.py), with the WebVTT subtitles it writes, and the palette shared by the frames and the HTML scenes.
"""

import json
import re

from pathlib import Path

import numpy

from tutorial_builder.index_page import write_index_page
from tutorial_builder.layout import FRAMES_PER_SECOND, NAMED_COLOURS
from tutorial_builder.narration import (SCENE_LEAD_SECONDS, SCENE_TAIL_SECONDS, SENTENCE_PAUSE_SECONDS, build_timeline,
                                        compute_sentence_key, convert_to_spoken_text, read_wav, write_wav)
from tutorial_builder.settings import ANIMATIONS_FOLDER
from tutorial_builder.transcription_check import compare_with_script

# Voice of the storyboards
VOICE_SETTINGS = {"name": "af_heart", "language": "en-us"}


def test_timeline_places_the_sentences_with_their_pauses() -> None:
    """
    Verifies that the timeline starts every scene with its lead, separates two sentences by the pause, ends the scene
    with its tail, all divided by the pace, lengthens a pause by its hold_after, and writes every sentence into the
    narration track at its start, the track as long as the frames.
    """
    sample_rate = 1000
    pace = 2.0
    scenes = [{"narration": ["first", "second"], "hold_after": {0: 1.0}}, {"narration": ["third"]}]
    sentence_audio = [[numpy.full(shape=(500,), fill_value=1, dtype=numpy.int16),
                       numpy.full(shape=(250,), fill_value=2, dtype=numpy.int16)],
                      [numpy.full(shape=(1000,), fill_value=3, dtype=numpy.int16)]]

    timeline, narration, number_frames = build_timeline(scenes=scenes, sentence_audio=sentence_audio,
                                                        sample_rate=sample_rate, pace=pace)
    first_start = SCENE_LEAD_SECONDS / pace
    second_start = first_start + 0.5 + SENTENCE_PAUSE_SECONDS / pace + 1.0
    assert timeline[0]["sentences"][0]["start"] == first_start
    assert abs(timeline[0]["sentences"][1]["start"] - second_start) < 1e-9
    assert abs(timeline[0]["end"] - (second_start + 0.25 + SCENE_TAIL_SECONDS / pace)) < 1e-9
    assert timeline[1]["start"] == timeline[0]["end"]
    assert number_frames == int(numpy.ceil(timeline[1]["end"] * FRAMES_PER_SECOND))
    assert narration[round(second_start * sample_rate)] == 2
    assert len(narration) == round(number_frames / FRAMES_PER_SECOND * sample_rate)


def test_sentence_key_follows_text_voice_and_pace(tmp_path: Path) -> None:
    """
    Verifies that the cache key of a sentence changes with its text and its pace and stays the same otherwise, and that
    the WAV files of the cache are read back unchanged.
    """
    key = compute_sentence_key(text="Hello.", voice_settings=VOICE_SETTINGS, pace=1.16)
    assert key == compute_sentence_key(text="Hello.", voice_settings=VOICE_SETTINGS, pace=1.16)
    assert key != compute_sentence_key(text="Hello!", voice_settings=VOICE_SETTINGS, pace=1.16)
    assert key != compute_sentence_key(text="Hello.", voice_settings=VOICE_SETTINGS, pace=1.0)

    samples = numpy.arange(start=-100, stop=100, dtype=numpy.int16)
    write_wav(samples=samples, sample_rate=24000, wav_path=tmp_path / "sentence.wav")
    read_samples, sample_rate = read_wav(wav_path=tmp_path / "sentence.wav")
    assert sample_rate == 24000 and numpy.array_equal(read_samples, samples)


def test_spoken_form_replaces_whole_words_only() -> None:
    """
    Verifies that the text given to the voice replaces the words of SPOKEN_FORMS by their spoken form, as whole words
    only, so that "ONNX" is read "onyx" while a longer word containing it is left as it is.
    """
    assert convert_to_spoken_text(sentence="Export to ONNX, then ONNX Runtime.") == "Export to onyx, then onyx Runtime."
    assert convert_to_spoken_text(sentence="ONNXRuntime stays.") == "ONNXRuntime stays."


def test_spoken_form_reads_the_decimals_digit_by_digit() -> None:
    """
    Verifies that the text given to the voice writes the decimal part of every number as words after "point", since
    espeak-ng reads "0.5" of a sentence ending with a period as "zero. five", the voice then pausing as at the end of a
    sentence, while versions, words holding digits and integers are left as they are.
    """
    assert (convert_to_spoken_text(sentence="A probability of 0.5, then 0.837 and 1,234.25.")
            == "A probability of 0 point five, then 0 point eight three seven and 1,234 point two five.")
    assert convert_to_spoken_text(sentence="It changes 35.3% of them, -0.5 below.") == ("It changes 35 point three% of "
                                                                                       "them, -0 point five below.")
    assert convert_to_spoken_text(sentence="Version 2.1.319 of x1.05, 1500 iterations.") == ("Version 2.1.319 of "
                                                                                           "x1.05, 1500 iterations.")


def test_transcription_comparison_counts_numbers_apart() -> None:
    """
    Verifies that the comparison of the transcription with the script counts every different word, and counts apart a
    number written in words in the script and heard in digits.
    """
    script = "The image is five hundred and twelve pixels wide."
    word_error_rate, number_free_rate, differences = compare_with_script(script=script,
                                                                         transcription="The image is 512 pixels wide.")
    assert word_error_rate > 0 and number_free_rate == 0
    assert any("(number)" in difference for difference in differences)
    word_error_rate, number_free_rate, _ = compare_with_script(script=script,
                                                               transcription="The image is five hundred and twelve "
                                                                             "pixels high.")
    assert word_error_rate == number_free_rate == 1 / 9


def test_transcription_comparison_counts_digits_heard_as_words_apart() -> None:
    """
    Verifies that a number written in digits in the script and heard in words, or heard with its digits grouped
    differently, is counted apart like a number written in words and heard in digits, while a word heard as another
    word stays an error.
    """
    script = "Block 3 keeps 1 000 images, two of them."
    word_error_rate, number_free_rate, _ = compare_with_script(script=script,
                                                               transcription="Block three keeps 1000 images, two of them.")
    assert word_error_rate > 0 and number_free_rate == 0
    _, number_free_rate, _ = compare_with_script(script=script,
                                                 transcription="Block 3 keeps 1 000 images, to of them.")
    assert number_free_rate == 1 / 9


def test_transcription_comparison_counts_word_spellings_apart() -> None:
    """
    Verifies that a British spelling of the script heard in its American spelling, or words heard split or joined, are
    counted apart like the numbers, while a word heard as another word stays an error.
    """
    script = "The labelled colour images go to tensorboard."
    word_error_rate, spelling_free_rate, differences = compare_with_script(
        script=script, transcription="The labeled color images go to tensor board.")
    assert word_error_rate > 0 and spelling_free_rate == 0
    assert any("(spelling)" in difference for difference in differences)
    _, spelling_free_rate, _ = compare_with_script(script=script,
                                                   transcription="The labelled collar images go to tensorboard.")
    assert spelling_free_rate == 1 / 7


def test_index_page_lists_the_chapters(tmp_path: Path) -> None:
    """
    Verifies that the index page lists a built chapter of a known series with its player, its scene timestamps and its
    recap, and writes its WebVTT subtitles.
    """
    series_folder = tmp_path / "01_foundations"
    series_folder.mkdir()
    record = {"series": "foundations", "series_folder": "01_foundations", "chapter": 1, "title": "The repository and a first run",
              "output_name": "foundations_01", "video": "foundations_01.mp4", "subtitles": "foundations_01.srt",
              "duration": 75.0, "summary": [{"text": "Iterations, not epochs"}],
              "scenes": [{"title": "Ten tasks", "start": 4.5, "code_reference": "vision_training/main.py · main",
                          "sentences": [{"text": "First sentence.", "start": 5.0, "end": 7.25}]}]}
    (series_folder / "foundations_01.json").write_text(data=json.dumps(obj=record), encoding="utf-8")

    index_text = write_index_page(output_folder=tmp_path).read_text(encoding="utf-8")
    assert "The repository and a first run" in index_text and 'data-seek="4.50"' in index_text
    assert "Iterations, not epochs" in index_text and "01_foundations/foundations_01.mp4" in index_text
    assert (series_folder / "foundations_01.vtt").read_text(encoding="utf-8").splitlines()[2:4] == [
        "00:00:05.000 --> 00:00:07.250", "First sentence."]


def test_html_scene_colours_match_the_frame_colours() -> None:
    """
    Verifies that the colour table of the HTML scenes (COLOURS of animations/common/tutorial.js) and the CSS tokens of
    animations/common/tutorial.css give the colours of NAMED_COLOURS of layout.py, so that a diagram captured from
    Chrome sits on the same card, in the same palette, as a frame drawn in Python.
    """
    script = (ANIMATIONS_FOLDER / "common" / "tutorial.js").read_text(encoding="utf-8")
    table = re.search(pattern=r"const COLOURS = \{(.*?)\};", string=script, flags=re.DOTALL).group(1)
    script_colours = {name: tuple(int(value) for value in values.split(","))
                      for name, values in re.findall(pattern=r"(\w+): \[(\d+, \d+, \d+)\]", string=table)}
    for name, colour in NAMED_COLOURS.items():
        assert script_colours[name] == tuple(colour), name
    style = (ANIMATIONS_FOLDER / "common" / "tutorial.css").read_text(encoding="utf-8")
    for token, name in {"background": "background", "card": "card", "card-border": "border", "accent": "accent"}.items():
        assert f"--{token}: #{'%02x%02x%02x' % NAMED_COLOURS[name]};" in style, token
