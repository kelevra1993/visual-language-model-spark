"""
Renderers of the scenes of the tutorial videos drawn in Python with Pillow, every one returning the frame of its scene
without its caption: the title card opening a chapter, a figure or screenshot of a real run seen through a camera that
zooms from one part to the next (views), with curves uncovered while they are explained (reveals) and parts outlined
(highlights), a panel of text columns, an excerpt of a configuration or of the code read from the repository with the
lines being explained highlighted, a terminal typing the commands, and the recap closing a chapter. The diagrams and
animations are HTML scenes captured by animation_rendering.py; frame_generation.py calls both.
"""

import functools
import math

from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Optional, Tuple

from tutorial_builder.layout import (ACCENT_COLOUR, BACKGROUND_COLOUR, CAPTION_BAND_COLOUR, CAPTION_BAND_HEIGHT,
                                     CARD_BORDER_COLOUR, CARD_PADDING, CODE_FONT_SIZES, CODE_LINE_SPACING_RATIO,
                                     PANEL_FONT_SIZE, PANEL_LINE_SPACING_RATIO, SELECTABLE_FONT_SIZES,
                                     COMMENT_COLOUR, CONTENT_BOTTOM, CONTENT_MARGIN, CONTENT_TOP, DIM_TEXT_COLOUR,
                                     HIGHLIGHT_COLOUR, HIGHLIGHT_MARGIN, MAXIMUM_IMAGE_UPSCALE, PROMPT_COLOUR,
                                     TERMINAL_BOTTOM, TERMINAL_COLOUR, TERMINAL_HEADER_COLOUR, TERMINAL_TOP,
                                     TEXT_COLOUR, VIDEO_HEIGHT, VIDEO_WIDTH, VIEWPORT_HEIGHT, VIEWPORT_LEFT,
                                     VIEWPORT_TOP, VIEWPORT_WIDTH, WINDOW_BUTTON_COLOURS, create_scene_background,
                                     draw_card, draw_code_line, wrap_words)

# Largest enlargement of a figure by a view, so that a zoom never shows blurred pixels
MAXIMUM_VIEW_UPSCALE = 2.0

# Prompt of the terminal scenes, and the folder named in their window header
TERMINAL_PROMPT = "$ "
TERMINAL_TITLE = "bash — ~/Projects/visual-language-model-spark"

# Typing of the terminal scenes at a pace of 1, and the blinking of their cursor
TYPING_START_SECONDS = 0.4
TYPING_CHARACTERS_PER_SECOND = 30
LINE_PAUSE_SECONDS = 0.4
CURSOR_BLINK_SECONDS = 0.5


def render_title_card(series_label: str, scene: Dict[str, Any],
                      fonts: Dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    """
    Draws the opening card of a chapter: the series in the accent colour, the heading, an accent line, the
    subheading and the footnote.

    Args:
        series_label (str): The series and chapter of the video.
        scene (Dict[str, Any]): The title card scene, with its heading, subheading and footnote.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.

    Returns:
        Image.Image: The frame of the scene, without caption.
    """
    frame = create_scene_background(series_label=series_label, scene_title=None, code_reference=None, fonts=fonts)
    draw = ImageDraw.Draw(im=frame)
    centre_x = VIDEO_WIDTH // 2

    # Series above the heading, an accent line under it, then the subheading and the footnote
    draw.text(xy=(centre_x, 300), text=series_label.upper(), fill=ACCENT_COLOUR, font=fonts["series_label"],
              anchor="mm")
    draw.text(xy=(centre_x, 410), text=scene["heading"], fill=TEXT_COLOUR, font=fonts["heading"], anchor="mm")
    draw.rectangle(xy=(centre_x - 120, 490, centre_x + 120, 496), fill=ACCENT_COLOUR)
    draw.text(xy=(centre_x, 570), text=scene["subheading"], fill=TEXT_COLOUR, font=fonts["subheading"], anchor="mm")
    draw.text(xy=(centre_x, 640), text=scene.get("footnote", ""), fill=DIM_TEXT_COLOUR, font=fonts["footnote"],
              anchor="mm")
    return frame


@functools.lru_cache(maxsize=32)
def load_figure(image_path: str) -> Image.Image:
    """
    Opens a figure of a scene (a screenshot of a run, a README figure, a chart, a dataset image) once, in RGB, so that
    the frames animated by a view or a reveal do not read the file again. The returned image is shared and never
    modified: the drawing functions crop, resize or copy it.

    Args:
        image_path (str): The image file.

    Returns:
        Image.Image: The RGB image.
    """
    with Image.open(fp=image_path) as opened_image:
        return opened_image.convert(mode="RGB")


def compose_revealed_image(scene: Dict[str, Any], reveal_progresses: Optional[List[float]]) -> Image.Image:
    """
    Gives the figure of an image scene at a moment of its reveals: the base image, with every region of the overlay
    image (the same figure with more drawn on it, such as the curves of a chart over its empty axes) uncovered from its
    left edge up to the progress of its reveal, so that a curve is drawn while its sentence is spoken.

    Args:
        scene (Dict[str, Any]): The image scene, with its image_path and optional overlay_path and reveals (region in
            pixels of the figure, sentence during which it is uncovered).
        reveal_progresses (Optional[List[float]]): The progress of every reveal within [0, 1], None for no reveal.

    Returns:
        Image.Image: The figure to show, a copy whenever a region of the overlay is uncovered.
    """
    base_image = load_figure(image_path=scene["image_path"])
    if not scene.get("reveals") or reveal_progresses is None:
        return base_image

    # Every region, kept inside the figure, uncovered from its left edge up to the progress of its reveal
    composed_image = base_image.copy()
    overlay_image = load_figure(image_path=scene["overlay_path"])
    for reveal, progress in zip(scene["reveals"], reveal_progresses):
        region_left, region_top = max(reveal["region"][0], 0), max(reveal["region"][1], 0)
        region_right = min(reveal["region"][2], overlay_image.width)
        region_bottom = min(reveal["region"][3], overlay_image.height)
        uncovered_right = round(region_left + progress * (region_right - region_left))
        if uncovered_right > region_left:
            composed_image.paste(im=overlay_image.crop(box=(region_left, region_top, uncovered_right, region_bottom)),
                                 box=(region_left, region_top))
    return composed_image


def fit_view_to_viewport(view_box: List[float], image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    """
    Turns a view of a figure (the part a sentence talks about) into the box the viewport shows: widened or heightened
    around its centre to the proportions of the viewport, enlarged so that the figure is never magnified more than
    MAXIMUM_VIEW_UPSCALE, then moved inside the figure along every axis where it fits and centred on the figure where
    it does not, so that the camera shows as little empty card as possible.

    Args:
        view_box (List[float]): The left, top, right and bottom of the view, in pixels of the figure.
        image_size (Tuple[int, int]): The width and height of the figure.

    Returns:
        Tuple[float, float, float, float]: The box shown in the viewport, in pixels of the figure.
    """
    viewport_ratio = VIEWPORT_WIDTH / VIEWPORT_HEIGHT
    left, top, right, bottom = view_box
    width = max(right - left, (bottom - top) * viewport_ratio, VIEWPORT_WIDTH / MAXIMUM_VIEW_UPSCALE)
    height = width / viewport_ratio

    # Every axis of the box kept inside the figure when the figure is larger, centred on the figure otherwise
    fitted_extents = []
    for centre, size, image_extent in [((left + right) / 2, width, image_size[0]),
                                       ((top + bottom) / 2, height, image_size[1])]:
        start = centre - size / 2
        start = min(max(start, 0.0), image_extent - size) if size <= image_extent else (image_extent - size) / 2
        fitted_extents.append((start, start + size))
    return fitted_extents[0][0], fitted_extents[1][0], fitted_extents[0][1], fitted_extents[1][1]


def find_sentence_view(scene: Dict[str, Any], sentence_index: int) -> Tuple[float, float, float, float]:
    """
    Gives the box of the figure shown during a sentence of an image scene with views: the last of its views starting
    at or before that sentence (from_sentence -1 framing the opening of the scene), the whole figure before the first.

    Args:
        scene (Dict[str, Any]): The image scene, with its views in order (box in pixels of the figure, from_sentence).
        sentence_index (int): The sentence, -1 before the first one.

    Returns:
        Tuple[float, float, float, float]: The box shown in the viewport, fitted by fit_view_to_viewport.
    """
    image_size = load_figure(image_path=scene["image_path"]).size
    view_box = [0, 0, image_size[0], image_size[1]]
    for view in scene["views"]:
        if view["from_sentence"] <= sentence_index:
            view_box = view["box"]
    return fit_view_to_viewport(view_box=view_box, image_size=image_size)


def interpolate_view(start_box: Tuple[float, float, float, float], end_box: Tuple[float, float, float, float],
                     progress: float) -> Tuple[float, float, float, float]:
    """
    Gives the box of a camera move between two views at a progress of the move, eased at both ends, the centre moving
    linearly and the width geometrically so that a zoom looks steady.

    Args:
        start_box (Tuple[float, float, float, float]): The box at the start of the move.
        end_box (Tuple[float, float, float, float]): The box at its end.
        progress (float): The progress of the move within [0, 1].

    Returns:
        Tuple[float, float, float, float]: The box at that progress, in the proportions of the viewport.
    """
    eased_progress = progress * progress * (3.0 - 2.0 * progress)
    centre_x = ((start_box[0] + start_box[2]) * (1.0 - eased_progress) + (end_box[0] + end_box[2]) * eased_progress) / 2
    centre_y = ((start_box[1] + start_box[3]) * (1.0 - eased_progress) + (end_box[1] + end_box[3]) * eased_progress) / 2
    width = math.exp(math.log(start_box[2] - start_box[0]) * (1.0 - eased_progress) +
                     math.log(end_box[2] - end_box[0]) * eased_progress)
    height = width * VIEWPORT_HEIGHT / VIEWPORT_WIDTH
    return centre_x - width / 2, centre_y - height / 2, centre_x + width / 2, centre_y + height / 2


def render_image_scene(series_label: str, scene: Dict[str, Any], fonts: Dict[str, ImageFont.FreeTypeFont],
                       active_sentence_index: int, view_box: Optional[Tuple[float, float, float, float]],
                       reveal_progresses: Optional[List[float]]) -> Image.Image:
    """
    Draws a figure (a screenshot of a real run, a README figure, a chart) on a card, with the parts explained by the
    sentence being spoken outlined in the accent colour. Without views, the figure is cropped to its crop_box and
    scaled to fit the content area with its proportions kept (enlarged at most by MAXIMUM_IMAGE_UPSCALE so that its
    text stays sharp), the card wrapping it. With views, the card fills the content area and the viewport shows the
    view box of the moment, the camera zooming and panning from one part of the figure to the next. The regions of the
    overlay are uncovered by their reveals.

    Args:
        series_label (str): The series and chapter of the video.
        scene (Dict[str, Any]): The image scene, with its title, image_path, optional crop_box, views, overlay_path,
            reveals and highlights (box in pixels of the figure, sentences during which it is outlined).
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.
        active_sentence_index (int): The sentence of the scene being spoken, or last spoken, -1 before the first one.
        view_box (Optional[Tuple[float, float, float, float]]): The box of the figure shown in the viewport at this
            moment, None for a scene without views.
        reveal_progresses (Optional[List[float]]): The progress of every reveal, None for a scene without reveals.

    Returns:
        Image.Image: The frame of the scene, without caption.
    """
    frame = create_scene_background(series_label=series_label, scene_title=scene["title"],
                                    code_reference=scene.get("code_reference"), fonts=fonts)
    figure = compose_revealed_image(scene=scene, reveal_progresses=reveal_progresses)
    draw = ImageDraw.Draw(im=frame)

    # Without views: the figure cropped, scaled to fit the content area and centred on a card wrapping it
    if view_box is None:
        crop_box = scene.get("crop_box", [0, 0, figure.width, figure.height])
        figure = figure.crop(box=tuple(crop_box))
        available_width = VIDEO_WIDTH - 2 * CONTENT_MARGIN - 2 * CARD_PADDING
        available_height = CONTENT_BOTTOM - CONTENT_TOP - 2 * CARD_PADDING
        scale = min(available_width / figure.width, available_height / figure.height, MAXIMUM_IMAGE_UPSCALE)
        scaled_size = (round(figure.width * scale), round(figure.height * scale))
        left = (VIDEO_WIDTH - scaled_size[0]) // 2
        top = CONTENT_TOP + (CONTENT_BOTTOM - CONTENT_TOP - scaled_size[1]) // 2
        draw_card(draw=draw, box=(left - CARD_PADDING, top - CARD_PADDING, left + scaled_size[0] + CARD_PADDING,
                                  top + scaled_size[1] + CARD_PADDING))
        frame.paste(im=figure.resize(size=scaled_size, resample=Image.Resampling.LANCZOS), box=(left, top))
        origin_x, origin_y = left - crop_box[0] * scale, top - crop_box[1] * scale
        visible_box = (left, top, left + scaled_size[0], top + scaled_size[1])

    # With views: the card filling the content area, the viewport showing the part of the figure inside the view box,
    # resampled from the exact (fractional) box so that the camera moves smoothly
    else:
        draw_card(draw=draw, box=(CONTENT_MARGIN, CONTENT_TOP, VIDEO_WIDTH - CONTENT_MARGIN, CONTENT_BOTTOM))
        box_left, box_top, box_right, box_bottom = view_box
        scale = VIEWPORT_WIDTH / (box_right - box_left)
        source_box = (max(box_left, 0.0), max(box_top, 0.0), min(box_right, float(figure.width)),
                      min(box_bottom, float(figure.height)))
        if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
            target_size = (max(1, round((source_box[2] - source_box[0]) * scale)),
                           max(1, round((source_box[3] - source_box[1]) * scale)))
            frame.paste(im=figure.resize(size=target_size, resample=Image.Resampling.LANCZOS, box=source_box),
                        box=(round(VIEWPORT_LEFT + (source_box[0] - box_left) * scale),
                             round(VIEWPORT_TOP + (source_box[1] - box_top) * scale)))
        origin_x, origin_y = VIEWPORT_LEFT - box_left * scale, VIEWPORT_TOP - box_top * scale
        visible_box = (VIEWPORT_LEFT, VIEWPORT_TOP, VIEWPORT_LEFT + VIEWPORT_WIDTH, VIEWPORT_TOP + VIEWPORT_HEIGHT)

    # Parts explained by the sentence being spoken, outlined around their box once scaled, kept on the visible figure
    for highlight in scene.get("highlights", []):
        if active_sentence_index in highlight["sentences"]:
            box_left, box_top, box_right, box_bottom = highlight["box"]
            outline_box = (max(origin_x + box_left * scale - HIGHLIGHT_MARGIN, visible_box[0] - HIGHLIGHT_MARGIN),
                           max(origin_y + box_top * scale - HIGHLIGHT_MARGIN, visible_box[1] - HIGHLIGHT_MARGIN),
                           min(origin_x + box_right * scale + HIGHLIGHT_MARGIN, visible_box[2] + HIGHLIGHT_MARGIN),
                           min(origin_y + box_bottom * scale + HIGHLIGHT_MARGIN, visible_box[3] + HIGHLIGHT_MARGIN))
            if outline_box[2] > outline_box[0] and outline_box[3] > outline_box[1]:
                draw.rounded_rectangle(xy=outline_box, radius=8, outline=ACCENT_COLOUR, width=4)
    return frame


def render_text_panel(series_label: str, scene: Dict[str, Any],
                      fonts: Dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    """
    Draws columns of monospace text side by side on a card, each under its heading and in the colours of its syntax
    (a folder tree with its comments, a JSON manifest), with an optional footnote at the bottom of the card, the size
    of the lines given by the font_size of the scene (PANEL_FONT_SIZE by default), so that a short panel can fill its
    card.

    Args:
        series_label (str): The series and chapter of the video.
        scene (Dict[str, Any]): The text panel scene, with its columns (heading, lines and optional syntax), optional
            footnote and optional font_size (one of SELECTABLE_FONT_SIZES).
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.

    Returns:
        Image.Image: The frame of the scene, without caption.

    Raises:
        ValueError: If the font size is not one of SELECTABLE_FONT_SIZES, or the columns do not fit the card at it.
    """
    frame = create_scene_background(series_label=series_label, scene_title=scene["title"],
                                    code_reference=scene.get("code_reference"), fonts=fonts)
    draw = ImageDraw.Draw(im=frame)
    draw_card(draw=draw, box=(CONTENT_MARGIN, CONTENT_TOP, VIDEO_WIDTH - CONTENT_MARGIN, CONTENT_BOTTOM))

    # Font of the lines asked by the scene, and the room of the columns on the card above its footnote
    font_size = scene.get("font_size", PANEL_FONT_SIZE)
    if font_size not in SELECTABLE_FONT_SIZES:
        raise ValueError(f"Text panel {scene['title']!r}: font_size {font_size} is none of {SELECTABLE_FONT_SIZES}.")
    line_font = fonts[f"code_{font_size}"]
    line_spacing = round(font_size * PANEL_LINE_SPACING_RATIO)
    footnote_height = 110 if "footnote" in scene else 0
    block_height = 70 + max(len(column["lines"]) for column in scene["columns"]) * line_spacing
    column_widths = [round(max([fonts["column_heading"].getlength(text=column["heading"])] +
                               [line_font.getlength(text=line) for line in column["lines"]]))
                     for column in scene["columns"]]
    block_width = sum(column_widths) + 70 * (len(column_widths) - 1)
    available_height = CONTENT_BOTTOM - footnote_height - CONTENT_TOP
    if block_height > available_height or block_width > VIDEO_WIDTH - 2 * CONTENT_MARGIN - 100:
        raise ValueError(f"Text panel {scene['title']!r}: its columns ({block_width} x {block_height} pixels at "
                         f"{font_size} pixels) do not fit the card: ask for a smaller font_size or fewer lines.")

    # Columns centred in the height of the card above its footnote, each as wide as its heading or its longest line
    block_top = CONTENT_TOP + (CONTENT_BOTTOM - footnote_height - CONTENT_TOP - block_height) // 2
    column_left = CONTENT_MARGIN + 50
    for column, column_width in zip(scene["columns"], column_widths):
        draw.text(xy=(column_left, block_top), text=column["heading"], fill=ACCENT_COLOUR,
                  font=fonts["column_heading"])
        for line_index, line in enumerate(column["lines"]):
            draw_code_line(draw=draw, position=(column_left, block_top + 70 + line_index * line_spacing), text=line,
                           font=line_font, syntax=column.get("syntax", "plain"))
        column_left += column_width + 70

    # Footnote at the bottom of the card
    if "footnote" in scene:
        draw.text(xy=(CONTENT_MARGIN + 50, CONTENT_BOTTOM - 60), text=scene["footnote"], fill=DIM_TEXT_COLOUR,
                  font=fonts["footnote"], anchor="lm")
    return frame


def choose_code_font(lines: List[Dict[str, Any]], fonts: Dict[str, ImageFont.FreeTypeFont],
                     font_size: Optional[int] = None) -> Tuple[ImageFont.FreeTypeFont, int]:
    """
    Chooses the monospace size of a code excerpt: the size its scene asks for (font_size, larger for a short excerpt
    that would leave most of the card empty, smaller for a long one), or else the largest of CODE_FONT_SIZES at which
    its longest line fits the width of the card and all its lines fit its height, so that an excerpt read from the
    repository stays on the card when the file grows a little.

    Args:
        lines (List[Dict[str, Any]]): The lines of the excerpt, each with its text.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.
        font_size (Optional[int]): The size asked by the scene, one of SELECTABLE_FONT_SIZES, or None to choose it.
            Defaults to None.

    Returns:
        Tuple[ImageFont.FreeTypeFont, int]: The font and the spacing of the lines in pixels.

    Raises:
        ValueError: If the size asked is none of SELECTABLE_FONT_SIZES or does not fit, or the excerpt does not fit
            even at the smallest automatic size, so that the storyboard asks for less.
    """
    available_width = VIDEO_WIDTH - 2 * CONTENT_MARGIN - 100
    available_height = CONTENT_BOTTOM - CONTENT_TOP - 110
    if font_size is not None and font_size not in SELECTABLE_FONT_SIZES:
        raise ValueError(f"A code excerpt asks for font_size {font_size}, none of {SELECTABLE_FONT_SIZES}.")

    # The sizes tried: the one asked alone, else the automatic sizes from the largest
    for candidate_size in [font_size] if font_size is not None else CODE_FONT_SIZES:
        font = fonts[f"code_{candidate_size}"]
        line_spacing = round(candidate_size * CODE_LINE_SPACING_RATIO)
        longest_width = max((font.getlength(text=line["text"]) for line in lines), default=0.0)
        if longest_width <= available_width and len(lines) * line_spacing <= available_height:
            return font, line_spacing
    if font_size is not None:
        fitting_sizes = [size for size in SELECTABLE_FONT_SIZES
                         if max((fonts[f"code_{size}"].getlength(text=line["text"]) for line in lines), default=0.0)
                         <= available_width and len(lines) * round(size * CODE_LINE_SPACING_RATIO) <= available_height]
        raise ValueError(f"A code excerpt of {len(lines)} lines does not fit the card at font_size {font_size}: the "
                         f"largest size that fits is {fitting_sizes[0] if fitting_sizes else 'none'}.")
    raise ValueError(f"A code excerpt of {len(lines)} lines, the longest of "
                     f"{max(len(line['text']) for line in lines)} characters, does not fit the card even at "
                     f"{CODE_FONT_SIZES[-1]} pixels: narrow its source (start, end or keys) in the storyboard.")


def render_code_scene(series_label: str, scene: Dict[str, Any], fonts: Dict[str, ImageFont.FreeTypeFont],
                      active_sentence_index: int) -> Image.Image:
    """
    Draws an excerpt of a configuration (kind "yaml") or of the code (kind "code", with its syntax), read from the
    repository at build time, on a card under the path of its file, in the colours of its syntax, the lines being
    explained highlighted with an accent mark: always for a line marked highlight, and while their sentence is spoken
    for a line listing highlight_sentences.

    Args:
        series_label (str): The series and chapter of the video.
        scene (Dict[str, Any]): The excerpt scene, with its file_label, optional syntax ("yaml" by default) and lines
            (text, optional highlight, optional highlight_sentences).
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.
        active_sentence_index (int): The sentence of the scene being spoken, or last spoken, -1 before the first one.

    Returns:
        Image.Image: The frame of the scene, without caption.
    """
    frame = create_scene_background(series_label=series_label, scene_title=scene["title"],
                                    code_reference=scene.get("code_reference"), fonts=fonts)
    draw = ImageDraw.Draw(im=frame)
    draw_card(draw=draw, box=(CONTENT_MARGIN, CONTENT_TOP, VIDEO_WIDTH - CONTENT_MARGIN, CONTENT_BOTTOM))
    text_left = CONTENT_MARGIN + 50
    draw.text(xy=(text_left, CONTENT_TOP + 30), text=scene["file_label"], fill=DIM_TEXT_COLOUR,
              font=fonts["file_label"])
    font, line_spacing = choose_code_font(lines=scene["lines"], fonts=fonts, font_size=scene.get("font_size"))

    # Band and accent mark of every line highlighted now, all drawn before the text so that the band of a line never
    # covers the underscores of the line above it
    for line_index, line in enumerate(scene["lines"]):
        line_top = CONTENT_TOP + 85 + line_index * line_spacing
        if line.get("highlight", False) or active_sentence_index in line.get("highlight_sentences", []):
            draw.rectangle(xy=(CONTENT_MARGIN + 2, line_top - 3, VIDEO_WIDTH - CONTENT_MARGIN - 2,
                               line_top + line_spacing - 3), fill=HIGHLIGHT_COLOUR)
            draw.rectangle(xy=(CONTENT_MARGIN + 2, line_top - 3, CONTENT_MARGIN + 10, line_top + line_spacing - 3),
                           fill=ACCENT_COLOUR)

    # Text of every line of the excerpt in the colours of its syntax
    for line_index, line in enumerate(scene["lines"]):
        draw_code_line(draw=draw, position=(text_left, CONTENT_TOP + 85 + line_index * line_spacing), text=line["text"],
                       font=font, syntax=scene.get("syntax", "yaml"))
    return frame


def compute_typing_schedule(lines: List[Dict[str, str]], pace: float) -> List[Tuple[float, float]]:
    """
    Times the typing of the lines of a terminal scene from the start of the scene: every line is typed at
    TYPING_CHARACTERS_PER_SECOND times the pace once the previous one is finished, after a pause standing for its Enter
    key.

    Args:
        lines (List[Dict[str, str]]): The lines of the terminal scene, each with its text.
        pace (float): The pace of the storyboard, 1.1 typing 10 % faster.

    Returns:
        List[Tuple[float, float]]: The start and end of the typing of every line, in seconds from the scene start.
    """
    schedule = []
    line_start = TYPING_START_SECONDS / pace
    for line in lines:
        line_end = line_start + len(line["text"]) / (TYPING_CHARACTERS_PER_SECOND * pace)
        schedule.append((line_start, line_end))
        line_start = line_end + LINE_PAUSE_SECONDS / pace
    return schedule


def render_terminal_window(series_label: str, scene: Dict[str, Any],
                           fonts: Dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    """
    Draws the empty terminal window of a terminal scene, with its header bar, its three window buttons and the folder
    the commands run from, on which render_terminal_frame types the commands.

    Args:
        series_label (str): The series and chapter of the video.
        scene (Dict[str, Any]): The terminal scene, with its title.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.

    Returns:
        Image.Image: The scene background with the empty terminal window.
    """
    frame = create_scene_background(series_label=series_label, scene_title=scene["title"],
                                    code_reference=scene.get("code_reference"), fonts=fonts)
    draw = ImageDraw.Draw(im=frame)
    window_box = (CONTENT_MARGIN, TERMINAL_TOP, VIDEO_WIDTH - CONTENT_MARGIN, TERMINAL_BOTTOM)
    draw.rounded_rectangle(xy=window_box, radius=14, fill=TERMINAL_COLOUR, outline=CARD_BORDER_COLOUR, width=2)

    # Header bar with the three window buttons and the folder of the shell
    draw.rounded_rectangle(xy=(window_box[0], window_box[1], window_box[2], window_box[1] + 48), radius=14,
                           fill=TERMINAL_HEADER_COLOUR, corners=(True, True, False, False))
    for button_index, button_colour in enumerate(WINDOW_BUTTON_COLOURS):
        button_centre = (window_box[0] + 30 + button_index * 28, window_box[1] + 24)
        draw.ellipse(xy=(button_centre[0] - 8, button_centre[1] - 8, button_centre[0] + 8, button_centre[1] + 8),
                     fill=button_colour)
    draw.text(xy=(VIDEO_WIDTH // 2, window_box[1] + 24), text=TERMINAL_TITLE, fill=DIM_TEXT_COLOUR,
              font=fonts["terminal_title"], anchor="mm")
    return frame


def render_terminal_frame(background: Image.Image, scene: Dict[str, Any], fonts: Dict[str, ImageFont.FreeTypeFont],
                          elapsed_seconds: float, schedule: List[Tuple[float, float]]) -> Image.Image:
    """
    Draws a terminal scene at a moment of its typing: every started line after its prompt, typed up to the characters
    of that moment and wrapped at the width of the window like a terminal, with a block cursor after the last typed
    character, blinking once the typing is finished. Output lines (role "output") appear at once without prompt.

    Args:
        background (Image.Image): The terminal window of render_terminal_window, drawn on the scene background.
        scene (Dict[str, Any]): The terminal scene, with its lines (role "command", "comment" or "output", and text).
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.
        elapsed_seconds (float): The time since the start of the scene.
        schedule (List[Tuple[float, float]]): The typing times of compute_typing_schedule.

    Returns:
        Image.Image: The frame of the scene at that moment, without caption.
    """
    frame = background.copy()
    draw = ImageDraw.Draw(im=frame)
    font = fonts["terminal_text"]
    character_width = font.getlength(text="M")
    line_height = 44
    text_left = CONTENT_MARGIN + 40
    text_top = TERMINAL_TOP + 48 + 36
    maximum_columns = int((VIDEO_WIDTH - 2 * CONTENT_MARGIN - 80) // character_width)

    # Every line whose typing has started, its prompt in green and its text in the colour of its role
    visual_row = 0
    cursor_position = None
    for line, (line_start, line_end) in zip(scene["lines"], schedule):
        if elapsed_seconds < line_start:
            break
        is_output = line["role"] == "output"
        typing_progress = 1.0 if is_output else (elapsed_seconds - line_start) / max(line_end - line_start, 1e-6)
        typed_count = min(len(line["text"]), int(typing_progress * len(line["text"])))
        text_colour = COMMENT_COLOUR if line["role"] == "comment" else DIM_TEXT_COLOUR if is_output else TEXT_COLOUR
        prompt = "" if is_output else TERMINAL_PROMPT
        visible_text = prompt + line["text"][:typed_count]

        # Rows of the line, wrapped every maximum_columns characters, the prompt drawn in its own colour
        for row_start in range(0, max(len(visible_text), 1), maximum_columns):
            row_text = visible_text[row_start:row_start + maximum_columns]
            row_top = text_top + visual_row * line_height
            if row_start == 0 and prompt:
                draw.text(xy=(text_left, row_top), text=prompt, fill=PROMPT_COLOUR, font=font)
                draw.text(xy=(text_left + len(prompt) * character_width, row_top), text=row_text[len(prompt):],
                          fill=text_colour, font=font)
            else:
                draw.text(xy=(text_left, row_top), text=row_text, fill=text_colour, font=font)
            visual_row += 1
        cursor_column = len(visible_text) % maximum_columns
        cursor_row = visual_row - 1 if cursor_column else visual_row
        cursor_position = (text_left + cursor_column * character_width, text_top + cursor_row * line_height)

    # Block cursor, steady while typing and blinking once the last line is typed
    typing_finished = elapsed_seconds >= schedule[-1][1]
    cursor_visible = not typing_finished or int(elapsed_seconds / CURSOR_BLINK_SECONDS) % 2 == 0
    if cursor_position is not None and cursor_visible:
        draw.rectangle(xy=(cursor_position[0], cursor_position[1] + 4, cursor_position[0] + character_width - 2,
                           cursor_position[1] + 36), fill=TEXT_COLOUR)
    return frame


def render_summary(series_label: str, scene: Dict[str, Any], fonts: Dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    """
    Draws the recap closing a chapter on a card: the numbered points, each after a numbered accent disc, a command or a
    file in the monospace font, a long point wrapped on two lines.

    Args:
        series_label (str): The series and chapter of the video.
        scene (Dict[str, Any]): The recap scene, with its items (text and optional monospace flag), 7 at most.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.

    Returns:
        Image.Image: The frame of the scene, without caption.
    """
    frame = create_scene_background(series_label=series_label, scene_title=scene["title"],
                                    code_reference=scene.get("code_reference"), fonts=fonts)
    draw = ImageDraw.Draw(im=frame)
    draw_card(draw=draw, box=(CONTENT_MARGIN, CONTENT_TOP, VIDEO_WIDTH - CONTENT_MARGIN, CONTENT_BOTTOM))

    # Every point on its row, after its numbered disc, wrapped to the width of the card
    row_spacing = (CONTENT_BOTTOM - CONTENT_TOP) / (len(scene["items"]) + 1)
    disc_centre_x = CONTENT_MARGIN + 90
    text_left = disc_centre_x + 70
    for item_index, item in enumerate(scene["items"]):
        row_centre_y = round(CONTENT_TOP + row_spacing * (item_index + 1))
        draw.ellipse(xy=(disc_centre_x - 32, row_centre_y - 32, disc_centre_x + 32, row_centre_y + 32),
                     fill=ACCENT_COLOUR)
        draw.text(xy=(disc_centre_x, row_centre_y), text=str(item_index + 1), fill=BACKGROUND_COLOUR,
                  font=fonts["summary_number"], anchor="mm")
        item_font = fonts["summary_monospace"] if item.get("monospace", False) else fonts["summary_text"]
        item_lines = wrap_words(text=item["text"], font=item_font,
                                maximum_width=VIDEO_WIDTH - CONTENT_MARGIN - 60 - text_left)
        line_height = item_font.size + 10
        for line_index, item_line in enumerate(item_lines):
            line_centre_y = row_centre_y + (line_index - (len(item_lines) - 1) / 2) * line_height
            draw.text(xy=(text_left, line_centre_y), text=item_line, fill=TEXT_COLOUR, font=item_font, anchor="lm")
    return frame


def render_caption_band(text: Optional[str], fonts: Dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    """
    Draws the caption band of a sentence of the narration, the sentence wrapped on at most two centred lines, smaller
    when two lines of the normal size do not hold it, or the empty band between two scenes.

    Args:
        text (Optional[str]): The sentence, None for the empty band.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.

    Returns:
        Image.Image: The VIDEO_WIDTH x CAPTION_BAND_HEIGHT band.
    """
    band = Image.new(mode="RGB", size=(VIDEO_WIDTH, CAPTION_BAND_HEIGHT), color=CAPTION_BAND_COLOUR)
    if text is not None:
        draw = ImageDraw.Draw(im=band)

        # The normal size when the sentence holds on two lines, the smaller one otherwise
        font = fonts["caption"]
        lines = wrap_words(text=text, font=font, maximum_width=VIDEO_WIDTH - 2 * CONTENT_MARGIN)
        if len(lines) > 2:
            font = fonts["caption_small"]
            lines = wrap_words(text=text, font=font, maximum_width=VIDEO_WIDTH - 2 * CONTENT_MARGIN)
        line_height = font.size + 14
        first_line_centre = CAPTION_BAND_HEIGHT / 2 - (len(lines) - 1) * line_height / 2
        for line_index, line in enumerate(lines):
            draw.text(xy=(VIDEO_WIDTH // 2, round(first_line_centre + line_index * line_height)), text=line,
                      fill=TEXT_COLOUR, font=font, anchor="mm")
    return band


def count_caption_lines(text: str, fonts: Dict[str, ImageFont.FreeTypeFont]) -> int:
    """
    Counts the lines a sentence takes in the caption band at the smaller caption size, used by the layout check of
    previews.py: a sentence needing more than three lines no longer fits the band and must be split.

    Args:
        text (str): The sentence.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.

    Returns:
        int: The number of lines of the sentence at the smaller caption size.
    """
    return len(wrap_words(text=text, font=fonts["caption_small"], maximum_width=VIDEO_WIDTH - 2 * CONTENT_MARGIN))

