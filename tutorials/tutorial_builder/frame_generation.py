"""
Generation of the frames of a tutorial video from its loaded storyboard and the timeline of its narration: at every
frame, the scene of that moment is drawn in its state (the camera of a figure between two views, a curve being
uncovered, a command being typed, an HTML scene captured at that time), faded from the previous sentence of its scene,
crossfaded from the previous scene, with the caption of the sentence spoken and the progress bar. A frame where nothing
moves is drawn once per sentence and reused. The frames go to video_encoding.py; previews.py draws one settled frame per
sentence with the same functions.
"""

import numpy

from PIL import Image, ImageFont
from typing import Any, Dict, Iterator, List, Optional

from tutorial_builder.animation_rendering import HtmlSceneCapture
from tutorial_builder.layout import (ACCENT_COLOUR, CAPTION_BAND_HEIGHT, CAPTION_TOP, CONTENT_MARGIN, CONTENT_TOP,
                                     FRAMES_PER_SECOND, PROGRESS_BAR_HEIGHT, PROGRESS_TRACK_COLOUR, VIDEO_HEIGHT,
                                     VIDEO_WIDTH, create_scene_background)
from tutorial_builder.scene_rendering import (compute_typing_schedule, find_sentence_view, interpolate_view,
                                              render_caption_band, render_code_scene, render_image_scene,
                                              render_summary, render_terminal_frame, render_terminal_window,
                                              render_text_panel, render_title_card)

# Fades between two scenes and at both ends of the video, and over the start of a new sentence of a scene
CROSSFADE_SECONDS = 0.5
SENTENCE_FADE_SECONDS = 0.35

# Duration of a camera move between two views of a figure
CAMERA_MOVE_SECONDS = 0.9


def compute_settled_content_state(scene: Dict[str, Any], active_sentence_index: int) -> Dict[str, Any]:
    """
    Gives the state of the content of a scene once the animations of a sentence are finished: the view of that
    sentence, its reveal and the earlier ones complete and the later ones not started. It is the state of the previews
    and of every frame where nothing moves.

    Args:
        scene (Dict[str, Any]): The scene.
        active_sentence_index (int): The sentence of the scene, -1 before the first one.

    Returns:
        Dict[str, Any]: The view_box (None without views), reveal_progresses (None without reveals), and the
            animating and camera_moving flags, both False.
    """
    view_box = None
    reveal_progresses = None
    if scene["kind"] == "image" and scene.get("views"):
        view_box = find_sentence_view(scene=scene, sentence_index=active_sentence_index)
    if scene["kind"] == "image" and scene.get("reveals"):
        reveal_progresses = [1.0 if reveal["sentence"] <= active_sentence_index else 0.0 for reveal in scene["reveals"]]
    return {"view_box": view_box, "reveal_progresses": reveal_progresses, "animating": False, "camera_moving": False}


def compute_content_state(scene: Dict[str, Any], scene_timing: Dict[str, Any], frame_time: float,
                          active_sentence_index: int) -> Dict[str, Any]:
    """
    Gives the state of the content of an image scene at a frame of the video: the camera between the views of the
    previous and of the current sentence while it moves after the start of the sentence, and the progress of the
    reveal of the current sentence over its duration, with whether something moves at this frame.

    Args:
        scene (Dict[str, Any]): The scene.
        scene_timing (Dict[str, Any]): The start and end of the scene and of its sentences, from build_timeline.
        frame_time (float): The time of the frame in the video.
        active_sentence_index (int): The sentence being spoken, or last spoken, -1 before the first one.

    Returns:
        Dict[str, Any]: The state of compute_settled_content_state, moved to the frame time.
    """
    state = compute_settled_content_state(scene=scene, active_sentence_index=active_sentence_index)
    if active_sentence_index < 0:
        return state
    sentence = scene_timing["sentences"][active_sentence_index]
    elapsed_seconds = frame_time - sentence["start"]

    # Camera moving from the view of the previous sentence to the view of this one
    if state["view_box"] is not None and elapsed_seconds < CAMERA_MOVE_SECONDS:
        previous_view_box = find_sentence_view(scene=scene, sentence_index=active_sentence_index - 1)
        if previous_view_box != state["view_box"]:
            state["view_box"] = interpolate_view(start_box=previous_view_box, end_box=state["view_box"],
                                                 progress=max(0.0, elapsed_seconds) / CAMERA_MOVE_SECONDS)
            state["animating"] = True
            state["camera_moving"] = True

    # Reveal of this sentence uncovered over the time the sentence is spoken
    if state["reveal_progresses"] is not None:
        for reveal_index, reveal in enumerate(scene["reveals"]):
            if reveal["sentence"] == active_sentence_index:
                progress = min(1.0, max(0.0, elapsed_seconds / max(sentence["end"] - sentence["start"], 1e-6)))
                state["reveal_progresses"][reveal_index] = progress
                state["animating"] = state["animating"] or progress < 1.0
    return state


def render_scene_content(series_label: str, scene: Dict[str, Any], fonts: Dict[str, ImageFont.FreeTypeFont],
                         active_sentence_index: int, content_state: Dict[str, Any]) -> Image.Image:
    """
    Draws a scene drawn in Python (not a terminal nor an HTML scene) by its kind, in the state of its content: the
    figures with their camera and reveals, and the excerpts, panels, cards and recaps with the highlights of the
    sentence being spoken.

    Args:
        series_label (str): The series and chapter of the video.
        scene (Dict[str, Any]): The scene.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.
        active_sentence_index (int): The sentence of the scene being spoken, or last spoken, -1 before the first one.
        content_state (Dict[str, Any]): The state of compute_content_state or compute_settled_content_state.

    Returns:
        Image.Image: The frame of the scene, without caption.

    Raises:
        ValueError: If the kind of the scene is not drawn in Python.
    """
    if scene["kind"] == "image":
        return render_image_scene(series_label=series_label, scene=scene, fonts=fonts,
                                  active_sentence_index=active_sentence_index, view_box=content_state["view_box"],
                                  reveal_progresses=content_state["reveal_progresses"])
    if scene["kind"] in ("yaml", "code"):
        return render_code_scene(series_label=series_label, scene=scene, fonts=fonts,
                                 active_sentence_index=active_sentence_index)
    renderers = {"title_card": render_title_card, "text_panel": render_text_panel, "summary": render_summary}
    if scene["kind"] not in renderers:
        raise ValueError(f"The scene kind {scene['kind']!r} is not drawn by render_scene_content.")
    return renderers[scene["kind"]](series_label=series_label, scene=scene, fonts=fonts)


def compose_html_frame(series_label: str, scene: Dict[str, Any], fonts: Dict[str, ImageFont.FreeTypeFont],
                       content_image: Image.Image, background_cache: Dict[str, Image.Image]) -> Image.Image:
    """
    Places the content captured from an HTML scene on the background of its scene, under its title bar.

    Args:
        series_label (str): The series and chapter of the video.
        scene (Dict[str, Any]): The html scene.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.
        content_image (Image.Image): The content area captured from Chrome.
        background_cache (Dict[str, Image.Image]): The background of the scene, drawn once and kept under "background".

    Returns:
        Image.Image: The frame of the scene, without caption.
    """
    if "background" not in background_cache:
        background_cache["background"] = create_scene_background(series_label=series_label,
                                                                 scene_title=scene["title"],
                                                                 code_reference=scene.get("code_reference"),
                                                                 fonts=fonts)
    frame = background_cache["background"].copy()
    frame.paste(im=content_image, box=(CONTENT_MARGIN, CONTENT_TOP))
    return frame


def find_active_sentence(scene_timing: Dict[str, Any], frame_time: float) -> int:
    """
    Gives the sentence of a scene being spoken, or last spoken, at a time of the video.

    Args:
        scene_timing (Dict[str, Any]): The timing of the scene, from build_timeline.
        frame_time (float): The time of the frame.

    Returns:
        int: The index of the sentence, -1 before the first one.
    """
    started_sentences = [index for index, sentence in enumerate(scene_timing["sentences"])
                         if frame_time >= sentence["start"]]
    return started_sentences[-1] if started_sentences else -1


def generate_frames(storyboard: Dict[str, Any], timeline: List[Dict[str, Any]],
                    fonts: Dict[str, ImageFont.FreeTypeFont], number_frames: int,
                    html_capture: Optional[HtmlSceneCapture]) -> Iterator[numpy.ndarray]:
    """
    Yields every frame of the video in order: the frame of the current scene (drawn once per spoken sentence while
    nothing moves, at every frame for a terminal and while a camera, a reveal or an HTML element moves), a fade from
    the content of the previous sentence when a sentence starts (unless the camera moves, or the scene is an HTML
    scene, which eases its own changes), the caption band of the sentence spoken or last spoken in the scene, a
    crossfade from the previous scene (from black for the first one), a fade to black at the end, and the progress bar.

    Args:
        storyboard (Dict[str, Any]): The loaded storyboard, with its series label, pace and scenes.
        timeline (List[Dict[str, Any]]): The scenes of build_timeline.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.
        number_frames (int): The number of frames of the video.
        html_capture (Optional[HtmlSceneCapture]): The started capture of the HTML scenes, None when the storyboard
            has none.

    Yields:
        numpy.ndarray: The [VIDEO_HEIGHT, VIDEO_WIDTH, 3] uint8 frames.
    """
    series_label = storyboard["series_label"]
    scenes = storyboard["scenes"]
    empty_band = numpy.asarray(a=render_caption_band(text=None, fonts=fonts))
    previous_scene_frame = numpy.zeros(shape=(VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=numpy.uint8)
    last_frame = previous_scene_frame
    scene_index = -1
    total_seconds = number_frames / FRAMES_PER_SECOND
    for frame_index in range(number_frames):
        frame_time = frame_index / FRAMES_PER_SECOND

        # Scene of the frame, and what is prepared on entering it: its terminal window and typing times, its HTML
        # page, its caption bands, its still frames
        while scene_index + 1 < len(scenes) and frame_time >= timeline[scene_index + 1]["start"]:
            scene_index += 1
            scene = scenes[scene_index]
            context = f"{storyboard['output_name']}, scene {scene_index + 1} ({scene.get('title', scene['kind'])})"
            if scene["kind"] == "terminal":
                terminal_window = render_terminal_window(series_label=series_label, scene=scene, fonts=fonts)
                typing_schedule = compute_typing_schedule(lines=scene["lines"], pace=storyboard["pace"])
            if scene["kind"] == "html":
                html_capture.load_scene(scene=scene, scene_timing=timeline[scene_index], context=context)
            static_frames = {}
            background_cache = {}
            faded_sentence_index = -1
            fade_source_frame = None
            last_content_frame = None
            caption_bands = [numpy.asarray(a=render_caption_band(text=sentence["text"], fonts=fonts))
                             for sentence in timeline[scene_index]["sentences"]]
            previous_scene_frame = last_frame
        scene = scenes[scene_index]
        scene_timing = timeline[scene_index]
        elapsed_seconds = frame_time - scene_timing["start"]
        active_sentence_index = find_active_sentence(scene_timing=scene_timing, frame_time=frame_time)
        camera_moving = False

        # Content of the scene: a terminal typed at every frame, an HTML scene captured (once per sentence when still),
        # the other scenes drawn once per sentence while nothing moves and at every frame while something moves
        if scene["kind"] == "terminal":
            frame = numpy.array(object=render_terminal_frame(background=terminal_window, scene=scene, fonts=fonts,
                                                             elapsed_seconds=elapsed_seconds,
                                                             schedule=typing_schedule))
        elif scene["kind"] == "html":
            content_image, _ = html_capture.capture(scene_time=elapsed_seconds, context=context)
            frame = numpy.array(object=compose_html_frame(series_label=series_label, scene=scene, fonts=fonts,
                                                          content_image=content_image,
                                                          background_cache=background_cache))
        else:
            content_state = compute_content_state(scene=scene, scene_timing=scene_timing, frame_time=frame_time,
                                                  active_sentence_index=active_sentence_index)
            camera_moving = content_state["camera_moving"]
            if content_state["animating"]:
                frame = numpy.array(object=render_scene_content(series_label=series_label, scene=scene, fonts=fonts,
                                                                active_sentence_index=active_sentence_index,
                                                                content_state=content_state))
            else:
                if active_sentence_index not in static_frames:
                    static_frames[active_sentence_index] = numpy.asarray(a=render_scene_content(
                        series_label=series_label, scene=scene, fonts=fonts,
                        active_sentence_index=active_sentence_index, content_state=content_state))
                frame = static_frames[active_sentence_index].copy()

        # Fade from the content of the previous sentence over the start of a new one, for the scenes drawn in Python
        # whose camera does not move
        if scene["kind"] not in ("terminal", "html"):
            if active_sentence_index != faded_sentence_index:
                fade_source_frame = None if camera_moving else last_content_frame
                faded_sentence_index = active_sentence_index
            if fade_source_frame is not None and active_sentence_index >= 0:
                fade_seconds = frame_time - scene_timing["sentences"][active_sentence_index]["start"]
                if fade_seconds < SENTENCE_FADE_SECONDS:
                    blend = fade_seconds / SENTENCE_FADE_SECONDS
                    frame = (fade_source_frame * (1.0 - blend) + frame * blend).astype(numpy.uint8)
            last_content_frame = frame

        # Caption of the sentence spoken, or last spoken, in the scene
        frame[CAPTION_TOP:CAPTION_TOP + CAPTION_BAND_HEIGHT] = \
            caption_bands[active_sentence_index] if active_sentence_index >= 0 else empty_band

        # Crossfade from the previous scene, and fade to black over the last frames of the video
        if elapsed_seconds < CROSSFADE_SECONDS:
            blend = elapsed_seconds / CROSSFADE_SECONDS
            frame = (previous_scene_frame * (1.0 - blend) + frame * blend).astype(numpy.uint8)
        if total_seconds - frame_time < CROSSFADE_SECONDS:
            frame = (frame * ((total_seconds - frame_time) / CROSSFADE_SECONDS)).astype(numpy.uint8)

        # Progress bar along the bottom edge
        progress_width = round(VIDEO_WIDTH * (frame_index + 1) / number_frames)
        frame[VIDEO_HEIGHT - PROGRESS_BAR_HEIGHT:, :] = PROGRESS_TRACK_COLOUR
        frame[VIDEO_HEIGHT - PROGRESS_BAR_HEIGHT:, :progress_width] = ACCENT_COLOUR
        last_frame = frame
        yield frame
