"""
Tests of the previews of the tutorial build (tutorials/tutorial_builder/previews.py) on the smoke storyboard: every
sentence of every scene kind is drawn, the HTML diagram captured from Chrome, at the size of the video, and the layout
check finds nothing wrong with it. Skipped without a Chrome on the machine.
"""

from pathlib import Path

import pytest

from PIL import Image

from tutorial_builder.layout import VIDEO_HEIGHT, VIDEO_WIDTH
from tutorial_builder.previews import render_storyboard_previews
from tutorial_builder.settings import create_build_settings, find_chrome_executable

# Smoke storyboard quoting real files of the repository, one scene of every kind
SMOKE_STORYBOARD_PATH = Path(__file__).parent / "fixtures" / "smoke_storyboard.yaml"


@pytest.mark.skipif(condition=find_chrome_executable(requested_path=None) is None, reason="no Chrome on this machine")
def test_previews_of_every_scene_kind(tmp_path: Path) -> None:
    """
    Verifies that the previews of the smoke storyboard give one frame of the video size per sentence, gathered in
    contact sheets, without layout problem.
    """
    settings = create_build_settings(output_folder=str(tmp_path / "output"), cache_folder=str(tmp_path / "cache"),
                                     experiments_folder=str(tmp_path / "experiments"), repository_python=None,
                                     chrome_executable=None)
    result = render_storyboard_previews(storyboard_path=SMOKE_STORYBOARD_PATH, settings=settings,
                                        preview_folder=tmp_path / "previews")
    assert len(result["previews"]) == 14
    assert len(result["contact_sheets"]) == 1
    assert result["problems"] == []
    with Image.open(fp=result["previews"][0]) as first_preview:
        assert first_preview.size == (VIDEO_WIDTH, VIDEO_HEIGHT)
