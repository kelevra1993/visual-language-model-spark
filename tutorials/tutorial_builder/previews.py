"""
Previews of the storyboards, drawn in seconds before a video is synthesized and encoded: one frame per sentence of
every scene, settled (its animations finished, a terminal fully typed), with the caption of that sentence, written as
PNG files with contact sheets of small frames, so that the layout of a new or edited storyboard is checked at a glance.
The layout check lists what a reader would see as broken: a caption longer than three lines, texts of an HTML scene
reaching out of the card or overlapping each other; the build stops on them before rendering.
"""

import contextlib
import math

from pathlib import Path
from PIL import Image, ImageDraw
from typing import Any, Dict, List, Tuple

from tutorial_builder.animation_rendering import HtmlSceneCapture
from tutorial_builder.frame_generation import compose_html_frame, compute_settled_content_state, render_scene_content
from tutorial_builder.layout import CAPTION_TOP, DIM_TEXT_COLOUR, VIDEO_HEIGHT, VIDEO_WIDTH, load_fonts
from tutorial_builder.narration import SCENE_LEAD_SECONDS
from tutorial_builder.scene_rendering import (compute_typing_schedule, count_caption_lines, render_caption_band,
                                              render_terminal_frame, render_terminal_window)
from tutorial_builder.settings import BuildSettings
from tutorial_builder.storyboard_loading import load_storyboard

# Assumed time of a sentence in the previews, longer than every transition of the HTML scenes
PREVIEW_SENTENCE_SECONDS = 3.0
PREVIEW_SETTLE_SECONDS = 2.2

# Frames of a contact sheet: columns, and the width of a small frame
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_FRAME_WIDTH = 480
CONTACT_SHEET_FRAMES = 24

# Longest caption, in lines of the smaller caption size
MAXIMUM_CAPTION_LINES = 3


def build_preview_timing(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gives a scene the timing the previews assume: every sentence PREVIEW_SENTENCE_SECONDS long after the lead of the
    scene, so that an HTML scene can be drawn settled at every sentence without the narration.

    Args:
        scene (Dict[str, Any]): The scene.

    Returns:
        Dict[str, Any]: The timing of the scene, shaped like the timeline of build_timeline.
    """
    sentences = [{"text": sentence, "start": SCENE_LEAD_SECONDS + index * PREVIEW_SENTENCE_SECONDS,
                  "end": SCENE_LEAD_SECONDS + (index + 1) * PREVIEW_SENTENCE_SECONDS - 0.4}
                 for index, sentence in enumerate(scene["narration"])]
    return {"start": 0.0, "end": SCENE_LEAD_SECONDS + len(sentences) * PREVIEW_SENTENCE_SECONDS + 0.5,
            "sentences": sentences}


def render_preview_frame(storyboard: Dict[str, Any], scene: Dict[str, Any], scene_timing: Dict[str, Any],
                         sentence_index: int, html_capture: Any, context: str) -> Tuple[Image.Image, List[str]]:
    """
    Draws the settled frame of one sentence of a scene with its caption, and lists its layout problems.

    Args:
        storyboard (Dict[str, Any]): The loaded storyboard.
        scene (Dict[str, Any]): The scene.
        scene_timing (Dict[str, Any]): The preview timing of the scene, its HTML page already loaded with it.
        sentence_index (int): The sentence.
        html_capture (Any): The started capture of the HTML scenes, or None.
        context (str): The scene, for the messages.

    Returns:
        Tuple[Image.Image, List[str]]: The frame, and the layout problems of this sentence.
    """
    fonts = load_fonts()
    problems = []
    if scene["kind"] == "html":
        settled_time = scene_timing["sentences"][sentence_index]["start"] + PREVIEW_SETTLE_SECONDS
        content_image, _ = html_capture.capture(scene_time=settled_time, context=context)
        frame = compose_html_frame(series_label=storyboard["series_label"], scene=scene, fonts=fonts,
                                   content_image=content_image, background_cache={})
        problems += [f"{context}, sentence {sentence_index + 1}: {problem}"
                     for problem in html_capture.check_layout(scene_time=settled_time)]
    elif scene["kind"] == "terminal":
        schedule = compute_typing_schedule(lines=scene["lines"], pace=storyboard["pace"])
        frame = render_terminal_frame(background=render_terminal_window(series_label=storyboard["series_label"],
                                                                        scene=scene, fonts=fonts),
                                      scene=scene, fonts=fonts, elapsed_seconds=schedule[-1][1] + 0.1,
                                      schedule=schedule)
    else:
        frame = render_scene_content(series_label=storyboard["series_label"], scene=scene, fonts=fonts,
                                     active_sentence_index=sentence_index,
                                     content_state=compute_settled_content_state(scene=scene,
                                                                                 active_sentence_index=sentence_index))

    # Caption of the sentence, checked for its length
    sentence = scene["narration"][sentence_index]
    frame.paste(im=render_caption_band(text=sentence, fonts=fonts), box=(0, CAPTION_TOP))
    if count_caption_lines(text=sentence, fonts=fonts) > MAXIMUM_CAPTION_LINES:
        problems.append(f"{context}, sentence {sentence_index + 1}: caption longer than {MAXIMUM_CAPTION_LINES} "
                        f"lines, split the sentence")
    return frame, problems


def write_contact_sheets(preview_paths: List[Path], preview_folder: Path) -> List[Path]:
    """
    Gathers the previews into contact sheets of CONTACT_SHEET_FRAMES small frames, each under its file name, so that a
    whole chapter is looked over in a few images.

    Args:
        preview_paths (List[Path]): The preview frames, in order.
        preview_folder (Path): The folder receiving the sheets.

    Returns:
        List[Path]: The contact sheets.
    """
    fonts = load_fonts()
    frame_height = round(CONTACT_SHEET_FRAME_WIDTH * VIDEO_HEIGHT / VIDEO_WIDTH)
    sheet_paths = []
    for sheet_index in range(math.ceil(len(preview_paths) / CONTACT_SHEET_FRAMES)):
        sheet_previews = preview_paths[sheet_index * CONTACT_SHEET_FRAMES:(sheet_index + 1) * CONTACT_SHEET_FRAMES]
        rows = math.ceil(len(sheet_previews) / CONTACT_SHEET_COLUMNS)
        sheet = Image.new(mode="RGB", size=(CONTACT_SHEET_COLUMNS * (CONTACT_SHEET_FRAME_WIDTH + 10) + 10,
                                            rows * (frame_height + 40) + 10), color=(10, 10, 12))
        draw = ImageDraw.Draw(im=sheet)

        # Every small frame under its file name
        for preview_index, preview_path in enumerate(sheet_previews):
            left = 10 + (preview_index % CONTACT_SHEET_COLUMNS) * (CONTACT_SHEET_FRAME_WIDTH + 10)
            top = 10 + (preview_index // CONTACT_SHEET_COLUMNS) * (frame_height + 40)
            with Image.open(fp=preview_path) as preview:
                sheet.paste(im=preview.resize(size=(CONTACT_SHEET_FRAME_WIDTH, frame_height),
                                              resample=Image.Resampling.LANCZOS), box=(left, top + 30))
            draw.text(xy=(left, top + 4), text=preview_path.stem, fill=DIM_TEXT_COLOUR, font=fonts["file_label"])
        sheet_path = preview_folder / f"contact_sheet_{sheet_index + 1:02d}.png"
        sheet.save(fp=sheet_path)
        sheet_paths.append(sheet_path)
    return sheet_paths


def render_storyboard_previews(storyboard_path: Path, settings: BuildSettings, preview_folder: Path) -> Dict[str, Any]:
    """
    Draws the previews of every sentence of a storyboard and their contact sheets, and lists the layout problems.

    Args:
        storyboard_path (Path): The YAML storyboard.
        settings (BuildSettings): The settings of the build.
        preview_folder (Path): The folder receiving the previews and the contact sheets, emptied of earlier previews.

    Returns:
        Dict[str, Any]: The preview files, the contact sheets and the layout problems.
    """
    storyboard = load_storyboard(storyboard_path=storyboard_path, settings=settings)
    preview_folder.mkdir(parents=True, exist_ok=True)
    for earlier_preview in preview_folder.glob("*.png"):
        earlier_preview.unlink()
    preview_paths = []
    problems = []
    has_html_scenes = any(scene["kind"] == "html" for scene in storyboard["scenes"])
    with HtmlSceneCapture(settings=settings) if has_html_scenes else contextlib.nullcontext() as html_capture:
        for scene_index, scene in enumerate(storyboard["scenes"]):
            context = f"{storyboard_path.name}, scene {scene_index + 1} ({scene.get('title', scene['kind'])})"
            scene_timing = build_preview_timing(scene=scene)
            if scene["kind"] == "html":
                html_capture.load_scene(scene=scene, scene_timing=scene_timing, context=context)

            # One settled frame per sentence of the scene
            for sentence_index in range(len(scene["narration"])):
                frame, frame_problems = render_preview_frame(storyboard=storyboard, scene=scene,
                                                             scene_timing=scene_timing, sentence_index=sentence_index,
                                                             html_capture=html_capture, context=context)
                preview_path = preview_folder / f"scene{scene_index + 1:02d}_sentence{sentence_index + 1:02d}.png"
                frame.save(fp=preview_path)
                preview_paths.append(preview_path)
                problems += frame_problems
    sheet_paths = write_contact_sheets(preview_paths=preview_paths, preview_folder=preview_folder)
    return {"previews": preview_paths, "contact_sheets": sheet_paths, "problems": problems}
