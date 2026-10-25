"""
Index page of the tutorial videos, written to the output folder after every build from the records of the chapters
(written by video_encoding.py next to every video): one self-contained HTML file listing the series in their order,
every chapter with its player, its subtitles (WebVTT, written next to the SubRip file), its scenes as timestamps that
seek the player, and its recap as a cheat sheet, with relative links only so that the folder of the videos can be
moved or shared as it is.
"""

import datetime
import html
import json
import operator
import subprocess

from pathlib import Path
from typing import Any, Dict, List

import yaml

from tutorial_builder.settings import REPOSITORY_ROOT, STORYBOARDS_FOLDER

# Look of the index page, the palette of the videos
PAGE_STYLE = """
:root { --background: #060a12; --card: #0c1320; --border: #1c283e; --text: #e8e8e8; --dim: #8c96a6;
        --accent: #4cc9f0; --value: #f0c878; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--background); color: var(--text); font: 16px/1.55 Inter, system-ui, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 30px; margin: 0 0 4px; }
.build { color: var(--dim); margin: 0 0 32px; font-size: 14px; }
section.series { margin: 0 0 40px; }
section.series > h2 { font-size: 22px; margin: 0 0 6px; border-left: 6px solid var(--accent); padding-left: 12px; }
section.series > p { color: var(--dim); margin: 0 0 16px; }
article.chapter { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px;
                  margin: 0 0 16px; display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 18px; }
article.chapter h3 { margin: 0 0 8px; font-size: 18px; }
article.chapter h3 .number { color: var(--accent); margin-right: 8px; }
video { width: 100%; border-radius: 8px; background: #000; }
.meta { color: var(--dim); font-size: 13px; margin: 6px 0 0; }
.meta a { color: var(--value); }
ol.scenes { margin: 0; padding-left: 0; list-style: none; max-height: 330px; overflow-y: auto; font-size: 14px; }
ol.scenes li { padding: 3px 0; border-bottom: 1px solid var(--border); }
ol.scenes button { background: none; border: none; color: var(--value); cursor: pointer; font: inherit;
                   padding: 0 8px 0 0; min-width: 56px; text-align: left; }
ol.scenes code { color: var(--dim); font-size: 12px; display: block; }
details { margin-top: 10px; font-size: 14px; }
summary { cursor: pointer; color: var(--accent); }
@media (max-width: 820px) { article.chapter { grid-template-columns: 1fr; } }
"""

# Seeking of the player of a chapter when a scene timestamp is clicked
PAGE_SCRIPT = """
document.querySelectorAll("button[data-seek]").forEach((button) => {
    button.addEventListener("click", () => {
        const video = document.getElementById(button.dataset.video);
        video.currentTime = Number(button.dataset.seek);
        video.play();
    });
});
"""


def format_duration(seconds: float) -> str:
    """
    Formats a duration or a time of a video as minutes:seconds.

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: The time, such as 7:37.
    """
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def format_web_subtitle_time(seconds: float) -> str:
    """
    Formats a time of the video as a WebVTT timestamp, hours:minutes:seconds.milliseconds.

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: The timestamp.
    """
    milliseconds = round(seconds * 1000)
    return (f"{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:{milliseconds // 1000 % 60:02d}."
            f"{milliseconds % 1000:03d}")


def write_web_subtitles(record: Dict[str, Any], subtitles_path: Path) -> None:
    """
    Writes the sentences of a chapter as a WebVTT file, the subtitle format browsers show under their player.

    Args:
        record (Dict[str, Any]): The record of the chapter.
        subtitles_path (Path): The .vtt file.
    """
    cues = ["WEBVTT", ""]
    for scene in record["scenes"]:
        for sentence in scene["sentences"]:
            cues += [f"{format_web_subtitle_time(seconds=sentence['start'])} --> "
                     f"{format_web_subtitle_time(seconds=sentence['end'])}", sentence["text"], ""]
    subtitles_path.write_text(data="\n".join(cues), encoding="utf-8")


def read_repository_commit() -> str:
    """
    Gives the commit of the repository the videos were built from, shown on the index page.

    Returns:
        str: The short hash and subject of the commit, or "unknown" outside a git checkout.
    """
    result = subprocess.run(args=["git", "-C", str(REPOSITORY_ROOT), "log", "-1", "--format=%h %s"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip()[:120] if result.returncode == 0 else "unknown"


def render_chapter(record: Dict[str, Any]) -> str:
    """
    Gives the HTML card of one chapter: its player with its subtitles, its files, its scenes as timestamps and its
    recap.

    Args:
        record (Dict[str, Any]): The record of the chapter.

    Returns:
        str: The HTML of the card.
    """
    video_identifier = f"video-{record['output_name']}"
    series_folder = record["series"]
    web_subtitles = Path(record["subtitles"]).with_suffix(".vtt").name

    # Scenes as timestamps seeking the player, under their file and function when they name one
    scene_items = []
    for scene in record["scenes"]:
        reference = f"<code>{html.escape(scene['code_reference'])}</code>" if scene.get("code_reference") else ""
        scene_items.append(f'<li><button data-video="{video_identifier}" data-seek="{scene["start"]:.2f}">'
                           f'{format_duration(seconds=scene["start"])}</button>{html.escape(str(scene["title"]))}'
                           f'{reference}</li>')
    recap_items = "".join(f"<li>{html.escape(item['text'])}</li>" for item in record.get("summary", []))
    recap = f"<details><summary>Recap</summary><ol>{recap_items}</ol></details>" if recap_items else ""
    return (f'<article class="chapter"><div><h3><span class="number">{record["chapter"]}</span>'
            f'{html.escape(record["title"])}</h3>'
            f'<video id="{video_identifier}" controls preload="metadata" '
            f'src="{series_folder}/{html.escape(record["video"])}">'
            f'<track kind="subtitles" srclang="en" label="English" src="{series_folder}/{web_subtitles}"></video>'
            f'<p class="meta">{format_duration(seconds=record["duration"])} · '
            f'<a href="{series_folder}/{html.escape(record["video"])}">MP4</a> · '
            f'<a href="{series_folder}/{html.escape(record["subtitles"])}">subtitles</a></p>{recap}</div>'
            f'<ol class="scenes">{"".join(scene_items)}</ol></article>')


def write_index_page(output_folder: Path) -> Path:
    """
    Writes the index page of the output folder from the records of every built chapter, in the order of the series of
    tutorials/storyboards/series.yaml and of their chapters, with the WebVTT subtitles of every chapter.

    Args:
        output_folder (Path): The output folder of the videos.

    Returns:
        Path: The index page.
    """
    series_list = yaml.safe_load((STORYBOARDS_FOLDER / "series.yaml").read_text(encoding="utf-8"))["series"]
    sections = []
    for series in series_list:
        record_paths = sorted((output_folder / series["name"]).glob("*.json"))
        records: List[Dict[str, Any]] = sorted((json.loads(record_path.read_text(encoding="utf-8"))
                                                for record_path in record_paths), key=operator.itemgetter("chapter"))
        if not records:
            continue

        # WebVTT subtitles of every chapter, then the section of the series
        for record in records:
            write_web_subtitles(record=record, subtitles_path=output_folder / series["name"] /
                                Path(record["subtitles"]).with_suffix(".vtt").name)
        total_duration = sum(record["duration"] for record in records)
        sections.append(f'<section class="series"><h2>{html.escape(series["title"])}</h2>'
                        f'<p>{html.escape(series.get("description", ""))} {len(records)} chapters, '
                        f'{format_duration(seconds=total_duration)}.</p>'
                        f'{"".join(render_chapter(record=record) for record in records)}</section>')
    built_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>VLMS Tutorials</title><style>{PAGE_STYLE}</style></head><body><main>'
            f'<h1>Visual Language Model Spark — tutorials</h1>'
            f'<p class="build">Built {built_at} from {html.escape(read_repository_commit())}</p>'
            f'{"".join(sections)}</main><script>{PAGE_SCRIPT}</script></body></html>')
    index_path = output_folder / "index.html"
    index_path.write_text(data=page, encoding="utf-8")
    return index_path
