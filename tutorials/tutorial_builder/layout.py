"""
Layout, colours and fonts shared by every frame of the tutorial videos, and the drawing helpers the scene renderers of
scene_rendering.py build on: the 1920x1080 frame with its title bar (scene title on the left, series and chapter on the
right, and the file and function the scene is about), the content area holding a card, the caption band and the
progress bar at the bottom. The HTML scenes of animation_rendering.py are drawn in the content area with the same
palette (animations/common/tutorial.css), so that both kinds of scenes look alike.
"""

import functools
import re

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Any, Dict, List, Optional, Tuple

# Size and rate of the video
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FRAMES_PER_SECOND = 30

# Layout: title bar at the top, caption band and progress bar at the bottom, content area between them
TITLE_BAR_HEIGHT = 110
CAPTION_BAND_HEIGHT = 164
PROGRESS_BAR_HEIGHT = 6
CONTENT_MARGIN = 80
CONTENT_TOP = TITLE_BAR_HEIGHT + 30
CAPTION_TOP = VIDEO_HEIGHT - CAPTION_BAND_HEIGHT - PROGRESS_BAR_HEIGHT
CONTENT_BOTTOM = CAPTION_TOP - 30
CONTENT_WIDTH = VIDEO_WIDTH - 2 * CONTENT_MARGIN
CONTENT_HEIGHT = CONTENT_BOTTOM - CONTENT_TOP
CARD_PADDING = 20
MAXIMUM_IMAGE_UPSCALE = 1.5
TERMINAL_TOP = CONTENT_TOP + 110
TERMINAL_BOTTOM = CONTENT_BOTTOM - 110
HIGHLIGHT_MARGIN = 4

# Viewport of the figures shown through views: the content area less the padding of its card
VIEWPORT_LEFT = CONTENT_MARGIN + CARD_PADDING
VIEWPORT_TOP = CONTENT_TOP + CARD_PADDING
VIEWPORT_WIDTH = CONTENT_WIDTH - 2 * CARD_PADDING
VIEWPORT_HEIGHT = CONTENT_HEIGHT - 2 * CARD_PADDING

# Colours of the frames: a dark ink blue close to black, chosen by the user on 2026-10-25, with a cyan accent
BACKGROUND_COLOUR = (6, 10, 18)
TITLE_BAR_COLOUR = (10, 15, 26)
CARD_COLOUR = (12, 19, 32)
CARD_BORDER_COLOUR = (28, 40, 62)
CAPTION_BAND_COLOUR = (3, 5, 10)
PROGRESS_TRACK_COLOUR = (24, 32, 48)
ACCENT_COLOUR = (76, 201, 240)
TEXT_COLOUR = (232, 232, 232)
DIM_TEXT_COLOUR = (140, 150, 166)
TERMINAL_COLOUR = (4, 7, 13)
TERMINAL_HEADER_COLOUR = (20, 28, 44)
PROMPT_COLOUR = (120, 220, 120)
COMMENT_COLOUR = (125, 135, 145)
KEY_COLOUR = (125, 196, 255)
VALUE_COLOUR = (240, 200, 120)
HIGHLIGHT_COLOUR = (15, 48, 70)
WINDOW_BUTTON_COLOURS = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]

# Colours of the diagram elements by name: the colours of the frames, and the dark steps of the reference categorical
# palette (blue, aqua, yellow, magenta, violet, red, validated in this order against the card surface, its orange left
# out, the highlights having their own cyan accent) for the kinds of boxes of a diagram, always labelled by their text
NAMED_COLOURS = {"text": TEXT_COLOUR, "dim": DIM_TEXT_COLOUR, "accent": ACCENT_COLOUR, "key": KEY_COLOUR,
                 "value": VALUE_COLOUR, "comment": COMMENT_COLOUR, "prompt": PROMPT_COLOUR, "card": CARD_COLOUR,
                 "box": (21, 31, 50), "border": CARD_BORDER_COLOUR, "background": BACKGROUND_COLOUR,
                 "terminal": TERMINAL_COLOUR, "blue": (57, 135, 229), "aqua": (25, 158, 112), "yellow": (201, 133, 0),
                 "magenta": (213, 81, 129), "violet": (144, 133, 233), "red": (230, 103, 103)}

# Fonts of the frames, vendored in tutorials/fonts so that every machine draws the same frames: Inter for the text and
# Hack for the code, the commands and the shapes of the tensors
FONTS_FOLDER = Path(__file__).resolve().parents[1] / "fonts"
SANS_FONT_PATH = FONTS_FOLDER / "Inter-Regular.otf"
SANS_SEMIBOLD_FONT_PATH = FONTS_FOLDER / "Inter-SemiBold.otf"
SANS_BOLD_FONT_PATH = FONTS_FOLDER / "Inter-Bold.otf"
MONOSPACE_FONT_PATH = FONTS_FOLDER / "Hack-Regular.ttf"
MONOSPACE_BOLD_FONT_PATH = FONTS_FOLDER / "Hack-Bold.ttf"

# Sizes of the monospace font of the code excerpts, the largest one holding the longest line and every line being used,
# and every size a scene can ask for with its font_size (a short excerpt larger, a long one smaller)
CODE_FONT_SIZES = [26, 24, 22, 20, 18]
SELECTABLE_FONT_SIZES = list(range(34, 15, -2))
CODE_LINE_SPACING_RATIO = 1.42

# Size of the lines of a text panel when its scene asks for none, and spacing of its lines relative to their size
PANEL_FONT_SIZE = 26
PANEL_LINE_SPACING_RATIO = 40 / 26


@functools.lru_cache(maxsize=1)
def load_fonts() -> Dict[str, ImageFont.FreeTypeFont]:
    """
    Loads the fonts of every element drawn on the frames of the tutorial videos, from the title bar to the captions,
    once per process.

    Returns:
        Dict[str, ImageFont.FreeTypeFont]: The fonts keyed by the element they draw.

    Raises:
        FileNotFoundError: If a font file of tutorials/fonts is missing.
    """
    for font_path in [SANS_FONT_PATH, SANS_SEMIBOLD_FONT_PATH, SANS_BOLD_FONT_PATH, MONOSPACE_FONT_PATH,
                      MONOSPACE_BOLD_FONT_PATH]:
        if not font_path.exists():
            raise FileNotFoundError(f"Missing font file {font_path}: the fonts of the videos live in {FONTS_FOLDER}.")
    fonts = {"title": ImageFont.truetype(font=str(SANS_BOLD_FONT_PATH), size=44),
             "series": ImageFont.truetype(font=str(SANS_FONT_PATH), size=24),
             "chip": ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=20),
             "heading": ImageFont.truetype(font=str(SANS_BOLD_FONT_PATH), size=96),
             "subheading": ImageFont.truetype(font=str(SANS_FONT_PATH), size=44),
             "footnote": ImageFont.truetype(font=str(SANS_FONT_PATH), size=30),
             "series_label": ImageFont.truetype(font=str(SANS_SEMIBOLD_FONT_PATH), size=30),
             "column_heading": ImageFont.truetype(font=str(SANS_SEMIBOLD_FONT_PATH), size=30),
             "file_label": ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=22),
             "terminal_text": ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=28),
             "terminal_title": ImageFont.truetype(font=str(SANS_FONT_PATH), size=22),
             "summary_text": ImageFont.truetype(font=str(SANS_FONT_PATH), size=38),
             "summary_monospace": ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=30),
             "summary_number": ImageFont.truetype(font=str(SANS_BOLD_FONT_PATH), size=32),
             "caption": ImageFont.truetype(font=str(SANS_FONT_PATH), size=40),
             "caption_small": ImageFont.truetype(font=str(SANS_FONT_PATH), size=34)}

    # One monospace font per size a code excerpt or a text panel can take
    for font_size in SELECTABLE_FONT_SIZES:
        fonts[f"code_{font_size}"] = ImageFont.truetype(font=str(MONOSPACE_FONT_PATH), size=font_size)
    return fonts


def wrap_words(text: str, font: ImageFont.FreeTypeFont, maximum_width: float) -> List[str]:
    """
    Cuts a sentence into lines no wider than a width, word by word, for the captions of the narration and the recaps.

    Args:
        text (str): The sentence.
        font (ImageFont.FreeTypeFont): The font of the lines.
        maximum_width (float): The width of a line in pixels.

    Returns:
        List[str]: The lines.
    """
    lines = []
    current_line = ""
    for word in text.split():
        candidate_line = word if not current_line else f"{current_line} {word}"
        if font.getlength(text=candidate_line) <= maximum_width or not current_line:
            current_line = candidate_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def resolve_colour(colour: Any) -> Tuple[int, int, int]:
    """
    Gives the RGB colour of an element from its name in NAMED_COLOURS, a "#rrggbb" string or an RGB list, the three
    ways a storyboard names a colour.

    Args:
        colour (Any): The colour name, hexadecimal string or RGB list.

    Returns:
        Tuple[int, int, int]: The RGB colour.

    Raises:
        ValueError: If the colour is neither a known name, nor a hexadecimal colour, nor an RGB list.
    """
    if isinstance(colour, (list, tuple)) and len(colour) == 3:
        return int(colour[0]), int(colour[1]), int(colour[2])
    if isinstance(colour, str) and colour in NAMED_COLOURS:
        return NAMED_COLOURS[colour]
    if isinstance(colour, str) and re.fullmatch(pattern=r"#[0-9a-fA-F]{6}", string=colour):
        return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)
    raise ValueError(f"Unknown colour {colour!r}, expected one of {sorted(NAMED_COLOURS)}, a '#rrggbb' string or an "
                     f"RGB list.")


def create_scene_background(series_label: str, scene_title: Optional[str], code_reference: Optional[str],
                            fonts: Dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    """
    Draws the background shared by the scenes of a video: the dark frame and, for every scene but the title card, the
    title bar naming the scene on the left and the series and chapter on the right, above the file and function the
    scene is about when it gives one, so that a newcomer always knows where to look in the repository.

    Args:
        series_label (str): The series and chapter, such as "Object Detection · 2/3", shown on the right of the bar.
        scene_title (Optional[str]): The title of the scene, None for a frame without title bar.
        code_reference (Optional[str]): The file and function of the scene, such as
            "vision_training/trainers/base_trainer.py · step_optimizer", or None.
        fonts (Dict[str, ImageFont.FreeTypeFont]): The fonts of load_fonts.

    Returns:
        Image.Image: The RGB background.
    """
    background = Image.new(mode="RGB", size=(VIDEO_WIDTH, VIDEO_HEIGHT), color=BACKGROUND_COLOUR)
    draw = ImageDraw.Draw(im=background)
    if scene_title is None:
        return background

    # Title bar with an accent strip on its left edge, the series and chapter on the right
    draw.rectangle(xy=(0, 0, VIDEO_WIDTH, TITLE_BAR_HEIGHT), fill=TITLE_BAR_COLOUR)
    draw.rectangle(xy=(0, 0, 12, TITLE_BAR_HEIGHT), fill=ACCENT_COLOUR)
    right_edge = VIDEO_WIDTH - CONTENT_MARGIN
    draw.text(xy=(right_edge, TITLE_BAR_HEIGHT // 2), text=series_label, fill=DIM_TEXT_COLOUR, font=fonts["series"],
              anchor="rm")

    # Scene title on the left, raised above the file and function of the scene when it names them, the reference
    # shortened in its middle when it would reach the series label
    if code_reference is None:
        draw.text(xy=(CONTENT_MARGIN, TITLE_BAR_HEIGHT // 2), text=scene_title, fill=TEXT_COLOUR,
                  font=fonts["title"], anchor="lm")
        return background
    draw.text(xy=(CONTENT_MARGIN, 42), text=scene_title, fill=TEXT_COLOUR, font=fonts["title"], anchor="lm")
    available_width = right_edge - fonts["series"].getlength(text=series_label) - 60 - (CONTENT_MARGIN + 22)
    shown_reference = code_reference
    while fonts["chip"].getlength(text=shown_reference) > available_width and len(shown_reference) > 12:
        middle = len(shown_reference) // 2
        shown_reference = f"{shown_reference[:middle - 2]}…{shown_reference[middle + 1:]}".replace("……", "…")
    draw.rectangle(xy=(CONTENT_MARGIN, 82, CONTENT_MARGIN + 8, 96), fill=ACCENT_COLOUR)
    draw.text(xy=(CONTENT_MARGIN + 22, 89), text=shown_reference, fill=VALUE_COLOUR, font=fonts["chip"], anchor="lm")
    return background


def draw_card(draw: ImageDraw.ImageDraw, box: Tuple[float, float, float, float]) -> None:
    """
    Draws the rounded card holding the content of a scene: a figure, a text panel, a configuration or a recap.

    Args:
        draw (ImageDraw.ImageDraw): The drawing context of the frame.
        box (Tuple[float, float, float, float]): The left, top, right and bottom of the card.
    """
    draw.rounded_rectangle(xy=box, radius=18, fill=CARD_COLOUR, outline=CARD_BORDER_COLOUR, width=2)


def draw_code_line(draw: ImageDraw.ImageDraw, position: Tuple[float, float], text: str, font: ImageFont.FreeTypeFont,
                   syntax: str) -> None:
    """
    Draws one line of a configuration, code or manifest excerpt in the colours of its syntax: the skipped parts
    ('...') and the comments dimmed, the YAML keys before their colon, the Python names assigned before their equal sign
    and the JSON keys in the key colour, and the values or strings in the value colour.

    Args:
        draw (ImageDraw.ImageDraw): The drawing context of the frame.
        position (Tuple[float, float]): The left and top of the line.
        text (str): The line.
        font (ImageFont.FreeTypeFont): The monospace font of the excerpt.
        syntax (str): "yaml", "python", "json" or "plain".
    """
    stripped_text = text.strip()
    if stripped_text == "..." or stripped_text.startswith("#"):
        draw.text(xy=position, text=text, fill=COMMENT_COLOUR, font=font)
        return

    # Segments of the line with their colours, a trailing comment kept apart
    code_text, comment_separator, comment_text = text.partition("  #")
    segments = []
    if syntax == "yaml":
        key_text, separator, value_text = code_text.partition(":")
        segments = [(key_text + separator, KEY_COLOUR), (value_text, VALUE_COLOUR)]
    elif syntax in ("python", "json"):
        remainder = code_text
        assignment = re.match(pattern=r"^(\s*[A-Za-z_]\w*)(\s*=\s*)(.*)$", string=code_text) \
            if syntax == "python" else None
        if assignment is not None:
            segments = [(assignment.group(1), KEY_COLOUR), (assignment.group(2), TEXT_COLOUR)]
            remainder = assignment.group(3)

        # Strings in the value colour, a JSON string followed by a colon being a key
        parts = [part for part in re.split(pattern=r'("[^"]*")', string=remainder) if part]
        for part_index, part in enumerate(parts):
            is_string = len(part) >= 2 and part.startswith('"') and part.endswith('"')
            next_part = parts[part_index + 1] if part_index + 1 < len(parts) else ""
            is_json_key = syntax == "json" and is_string and next_part.lstrip().startswith(":")
            segments.append((part, KEY_COLOUR if is_json_key else VALUE_COLOUR if is_string else TEXT_COLOUR))
    else:
        segments = [(code_text, TEXT_COLOUR)]
    if comment_separator:
        segments.append((comment_separator + comment_text, COMMENT_COLOUR))

    # Segments drawn one after the other
    segment_left = position[0]
    for segment_text, segment_colour in segments:
        draw.text(xy=(segment_left, position[1]), text=segment_text, fill=segment_colour, font=font)
        segment_left += font.getlength(text=segment_text)


def to_frame_box(box: List[float]) -> Tuple[float, float, float, float]:
    """
    Converts a box given in pixels from the top left corner of the content area into frame pixels.

    Args:
        box (List[float]): The left, top, right and bottom of the box in the content area.

    Returns:
        Tuple[float, float, float, float]: The box in the frame.
    """
    return CONTENT_MARGIN + box[0], CONTENT_TOP + box[1], CONTENT_MARGIN + box[2], CONTENT_TOP + box[3]
