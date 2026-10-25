"""
Making of one tutorial video from its storyboard: the narration read from the cache or synthesized (narration.py), the
timeline of its sentences, the frames (frame_generation.py, with the HTML scenes captured by animation_rendering.py)
encoded with the narration into an MP4 file by the static ffmpeg of imageio-ffmpeg, the SubRip subtitles and a JSON
record of the chapter (its scenes, sentences and times) next to it in the output folder. The record feeds the index
page (index_page.py) and the checks of the build (transcription_check.py).
"""

import contextlib
import json
import subprocess
import time

import imageio_ffmpeg
import numpy

from pathlib import Path
from typing import Any, Dict, Iterator

from tutorial_builder.animation_rendering import HtmlSceneCapture
from tutorial_builder.frame_generation import generate_frames
from tutorial_builder.layout import FRAMES_PER_SECOND, VIDEO_HEIGHT, VIDEO_WIDTH, load_fonts
from tutorial_builder.model_files import ensure_model_files
from tutorial_builder.narration import build_timeline, synthesize_narration, write_subtitles, write_wav
from tutorial_builder.settings import THREAD_COUNT, BuildSettings
from tutorial_builder.storyboard_loading import load_storyboard


def encode_video(frames: Iterator[numpy.ndarray], narration_path: Path, video_path: Path, log_path: Path) -> None:
    """
    Encodes the frames and the narration into the MP4 file of the video with the ffmpeg of imageio-ffmpeg, on
    THREAD_COUNT threads: H.264 video in yuv420p for every player, the narration normalized to -16 LUFS and encoded in
    AAC at 48 kHz. The video is written under a temporary name and moved into place once complete.

    Args:
        frames (Iterator[numpy.ndarray]): The frames of generate_frames.
        narration_path (Path): The WAV narration track.
        video_path (Path): The MP4 file.
        log_path (Path): The file receiving the messages of ffmpeg.

    Raises:
        RuntimeError: If ffmpeg fails, with the end of its messages.
    """
    partial_path = video_path.with_name(f"{video_path.stem}.partial.mp4")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-threads", str(THREAD_COUNT),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
               "-r", str(FRAMES_PER_SECOND), "-i", "-", "-i", str(narration_path), "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-threads", str(THREAD_COUNT), "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(partial_path)]
    with open(file=log_path, mode="w", encoding="utf-8") as log_file:
        process = subprocess.Popen(args=command, stdin=subprocess.PIPE, stderr=log_file)
        try:
            for frame in frames:
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
            return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {log_path.read_text(encoding='utf-8')[-2000:]}")
    partial_path.replace(target=video_path)


def make_video(storyboard_path: Path, settings: BuildSettings) -> Dict[str, Any]:
    """
    Makes the video of a storyboard into the output folder: the narration (cached sentences reused), its WAV track,
    the subtitles, the frames encoded with the narration, and the record of the chapter.

    Args:
        storyboard_path (Path): The YAML storyboard.
        settings (BuildSettings): The settings of the build.

    Returns:
        Dict[str, Any]: The record of the chapter: series, chapter, title, files, duration, scenes with their titles
            and times, sentences with their times, and the counts of cached and synthesized sentences.
    """
    start_time = time.time()
    storyboard = load_storyboard(storyboard_path=storyboard_path, settings=settings)
    model_folders = ensure_model_files(model_folder=settings.model_folder)
    sentence_audio, sample_rate, sentence_counts = synthesize_narration(storyboard=storyboard,
                                                                        kokoro_folder=model_folders["kokoro"],
                                                                        narration_folder=settings.narration_folder,
                                                                        cache_folder=settings.cache_folder)
    timeline, narration, number_frames = build_timeline(scenes=storyboard["scenes"], sentence_audio=sentence_audio,
                                                        sample_rate=sample_rate, pace=storyboard["pace"])

    # Narration track and subtitles next to the video, then the frames encoded with the narration, the HTML scenes
    # captured by one Chrome page opened for the whole video
    series_folder = settings.output_folder / storyboard["series"]
    series_folder.mkdir(parents=True, exist_ok=True)
    output_name = storyboard["output_name"]
    narration_path = settings.cache_folder / "tracks" / f"{output_name}_narration.wav"
    write_wav(samples=narration, sample_rate=sample_rate, wav_path=narration_path)
    subtitles_path = series_folder / f"{output_name}.srt"
    write_subtitles(timeline=timeline, subtitles_path=subtitles_path)
    video_path = series_folder / f"{output_name}.mp4"
    has_html_scenes = any(scene["kind"] == "html" for scene in storyboard["scenes"])
    with HtmlSceneCapture(settings=settings) if has_html_scenes else contextlib.nullcontext() as html_capture:
        encode_video(frames=generate_frames(storyboard=storyboard, timeline=timeline, fonts=load_fonts(),
                                            number_frames=number_frames, html_capture=html_capture),
                     narration_path=narration_path, video_path=video_path,
                     log_path=settings.cache_folder / "tracks" / f"{output_name}_ffmpeg.log")

    # Record of the chapter, read by the index page and the transcription check
    record = {"series": storyboard["series"], "series_title": storyboard["series_title"],
              "chapter": storyboard["chapter"], "title": storyboard["title"], "output_name": output_name,
              "storyboard": str(storyboard_path), "video": video_path.name, "subtitles": subtitles_path.name,
              "duration": number_frames / FRAMES_PER_SECOND, "sentence_counts": sentence_counts,
              "build_seconds": round(time.time() - start_time, 1),
              "summary": next((scene["items"] for scene in storyboard["scenes"] if scene["kind"] == "summary"), []),
              "scenes": [{"title": scene.get("title", scene.get("heading")), "kind": scene["kind"],
                          "start": scene_timing["start"], "end": scene_timing["end"],
                          "code_reference": scene.get("code_reference"),
                          "sentences": scene_timing["sentences"]}
                         for scene, scene_timing in zip(storyboard["scenes"], timeline)]}
    (series_folder / f"{output_name}.json").write_text(data=json.dumps(obj=record, indent=2), encoding="utf-8")
    print(f"{video_path}: {record['duration']:.1f} s, {len(timeline)} scenes, "
          f"{sum(len(scene_timing['sentences']) for scene_timing in timeline)} sentences "
          f"({sentence_counts['synthesized']} synthesized), built in {record['build_seconds']:.0f} s")
    return record
