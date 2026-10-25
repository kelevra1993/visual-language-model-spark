"""
Capture of the HTML scenes of the tutorial videos (kind html in a storyboard): the diagrams and animations of
tutorials/animations drawn by the engine of animations/common/tutorial.js in headless Chrome, driven by Playwright.
A local HTTP server serves, from one origin, the pages, the fonts, the figures of the cache, the repository and the
real runs, so that a page reads every image and font it names. For every frame of the scene, frame_generation.py asks
the page to draw the scene at the time of the frame and captures the content area; a frame where nothing moves is
captured once per sentence and reused, so that a still diagram costs one capture per sentence.
"""

import functools
import http.server
import io
import json
import shutil
import tempfile
import threading
import urllib.parse

from pathlib import Path
from PIL import Image
from playwright.sync_api import Browser, Page, Playwright, sync_playwright
from typing import Any, Dict, List, Optional, Tuple

from tutorial_builder.layout import CONTENT_HEIGHT, CONTENT_WIDTH, FONTS_FOLDER
from tutorial_builder.settings import ANIMATIONS_FOLDER, REPOSITORY_ROOT, BuildSettings
from tutorial_builder.storyboard_loading import PATH_PREFIXES, resolve_path

# Chrome flags of the captures: no GPU (the trainings keep it), the sRGB colours of the Python frames, no scrollbar
CHROME_ARGUMENTS = ["--disable-gpu", "--force-color-profile=srgb", "--hide-scrollbars", "--font-render-hinting=none"]

# Longest wait for a page and its setup, in milliseconds
PAGE_TIMEOUT_MILLISECONDS = 60000


class QuietRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Request handler of the local server of the captures, serving the site folder without printing every request.
    """

    def log_message(self, format: str, *arguments: Any) -> None:
        """
        Leaves the requests unprinted, the build printing its own progress.

        Args:
            format (str): The format of the message.
            *arguments (Any): The values of the message.
        """
        return


class HtmlSceneCapture:
    """
    Headless Chrome page and local server capturing the HTML scenes of one video, opened with `with` around the
    generation of its frames so that the browser and the server are always stopped.

    Attributes:
        settings (BuildSettings): The settings of the build.
        site_folder (Optional[Path]): The temporary folder of links served by the local server.
        roots (Dict[str, Path]): The folder behind every top-level path of the server.
    """

    def __init__(self, settings: BuildSettings) -> None:
        """
        Prepares the capture of the HTML scenes, started by __enter__.

        Args:
            settings (BuildSettings): The settings of the build, with the Chrome executable and the folders served.
        """
        self.settings = settings
        self.site_folder: Optional[Path] = None
        self.roots = {"animations": ANIMATIONS_FOLDER, "fonts": FONTS_FOLDER, "figures": settings.figures_folder,
                      "repository": REPOSITORY_ROOT, "runs": settings.experiments_folder}
        self.server: Optional[http.server.ThreadingHTTPServer] = None
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.still_frames: Dict[int, Image.Image] = {}
        self.page_errors = []

    def __enter__(self) -> "HtmlSceneCapture":
        """
        Starts the local server over a temporary folder of links to the served folders, then Chrome and its page,
        sized to the content area of the frames.

        Returns:
            HtmlSceneCapture: The started capture.
        """
        # Site folder of links, one per served folder, the folders missing on this machine left out
        self.settings.cache_folder.mkdir(parents=True, exist_ok=True)
        self.site_folder = Path(tempfile.mkdtemp(prefix="site_", dir=self.settings.cache_folder))
        for root_name, root_folder in self.roots.items():
            if root_folder.exists():
                (self.site_folder / root_name).symlink_to(target=root_folder.resolve(), target_is_directory=True)

        # Local server on a free port, in a background thread
        handler = functools.partial(QuietRequestHandler, directory=str(self.site_folder))
        self.server = http.server.ThreadingHTTPServer(server_address=("127.0.0.1", 0), RequestHandlerClass=handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

        # Chrome of the settings, or the Chromium of Playwright, with one page as large as the content area
        self.playwright = sync_playwright().start()
        executable_path = str(self.settings.chrome_executable) if self.settings.chrome_executable else None
        self.browser = self.playwright.chromium.launch(executable_path=executable_path, args=CHROME_ARGUMENTS)
        self.page = self.browser.new_page(viewport={"width": CONTENT_WIDTH, "height": CONTENT_HEIGHT},
                                          device_scale_factor=1)
        self.page.set_default_timeout(timeout=PAGE_TIMEOUT_MILLISECONDS)
        self.page.on("pageerror", self.record_page_error)
        return self

    def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:
        """
        Stops Chrome, Playwright and the local server, and removes the folder of links.

        Args:
            exception_type (Any): The type of the exception ending the block, or None.
            exception_value (Any): The exception, or None.
            traceback (Any): Its traceback, or None.
        """
        if self.browser is not None:
            self.browser.close()
        if self.playwright is not None:
            self.playwright.stop()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.site_folder is not None:
            shutil.rmtree(path=self.site_folder, ignore_errors=True)

    def record_page_error(self, error: Exception) -> None:
        """
        Keeps the JavaScript errors of the page, raised by load_scene and capture so that a broken scene stops the build.

        Args:
            error (Exception): The error thrown in the page.
        """
        self.page_errors.append(str(error))

    def raise_page_errors(self, context: str) -> None:
        """
        Stops the build when the page threw a JavaScript error.

        Args:
            context (str): The scene, for the message.

        Raises:
            RuntimeError: If the page threw an error.
        """
        if self.page_errors:
            raise RuntimeError(f"{context}: the HTML page threw {self.page_errors}.")

    def path_to_url(self, path: Path) -> str:
        """
        Gives the URL of a file under one of the served folders.

        Args:
            path (Path): The absolute path of the file.

        Returns:
            str: Its URL on the local server.

        Raises:
            ValueError: If the file is under none of the served folders.
        """
        resolved_path = path.resolve()
        for root_name, root_folder in self.roots.items():
            if root_folder.exists() and resolved_path.is_relative_to(root_folder.resolve()):
                relative_path = resolved_path.relative_to(root_folder.resolve()).as_posix()
                return f"http://127.0.0.1:{self.server.server_address[1]}/{root_name}/" \
                       f"{urllib.parse.quote(string=relative_path)}"
        raise ValueError(f"{path} is under none of the folders served to the HTML scenes: {self.roots}.")

    def convert_paths(self, value: Any) -> Any:
        """
        Turns every path of the data of a scene into the URL the page reads it from: the absolute paths existing on
        disk, and the prefixed paths (figures:, repository:, runs:, animations:) of a data file.

        Args:
            value (Any): A value of the scene data.

        Returns:
            Any: The value with URLs.
        """
        if isinstance(value, str) and value.startswith(PATH_PREFIXES):
            return self.path_to_url(path=resolve_path(path_text=value, settings=self.settings))
        if isinstance(value, str) and value.startswith("/") and Path(value).is_file():
            return self.path_to_url(path=Path(value))
        if isinstance(value, list):
            return [self.convert_paths(value=item) for item in value]
        if isinstance(value, dict):
            return {key: self.convert_paths(value=item) for key, item in value.items()}
        return value

    def load_scene(self, scene: Dict[str, Any], scene_timing: Dict[str, Any], context: str) -> None:
        """
        Opens the page of an HTML scene and sets it up with the scene (its data file read when it names one, its
        paths turned into URLs) and the times of its sentences from the scene start.

        Args:
            scene (Dict[str, Any]): The html scene of the storyboard, with its page_path, elements and optional data.
            scene_timing (Dict[str, Any]): The start and end of the scene and of its sentences, from build_timeline.
            context (str): The scene, for the error messages.
        """
        scene_data = {key: value for key, value in scene.items() if key not in ("narration", "page_path")}
        if isinstance(scene.get("data"), str):
            scene_data["data"] = json.loads(Path(scene["data"]).read_text(encoding="utf-8"))
        scene_data = self.convert_paths(value=scene_data)
        timing = {"duration": scene_timing["end"] - scene_timing["start"],
                  "sentences": [{"start": sentence["start"] - scene_timing["start"],
                                 "end": sentence["end"] - scene_timing["start"]}
                                for sentence in scene_timing["sentences"]]}

        # Page opened from the local server, set up once its fonts and images are loaded
        self.page_errors = []
        self.still_frames = {}
        self.page.goto(url=self.path_to_url(path=Path(scene["page_path"])), wait_until="load")
        self.page.evaluate(expression="([scene, timing]) => Tutorial.setupScene(scene, timing)",
                           arg=[scene_data, timing])
        self.raise_page_errors(context=context)

    def capture(self, scene_time: float, context: str) -> Tuple[Image.Image, bool]:
        """
        Draws the loaded scene at a time from its start and captures the content area; a frame where nothing moves
        is kept for the rest of its sentence, every change of an element starting with a sentence.

        Args:
            scene_time (float): The time from the start of the scene, in seconds.
            context (str): The scene, for the error messages.

        Returns:
            Tuple[Image.Image, bool]: The RGB content image, and whether something moves at this time.
        """
        frame_state = self.page.evaluate(expression="(time) => Tutorial.renderFrame(time)", arg=scene_time)
        sentence_index = frame_state["sentence"]
        if not frame_state["moving"] and sentence_index in self.still_frames:
            return self.still_frames[sentence_index], False
        screenshot = self.page.screenshot(type="png", clip={"x": 0, "y": 0, "width": CONTENT_WIDTH,
                                                            "height": CONTENT_HEIGHT})
        self.raise_page_errors(context=context)
        with Image.open(fp=io.BytesIO(initial_bytes=screenshot)) as captured_image:
            content_image = captured_image.convert(mode="RGB")
        if not frame_state["moving"]:
            self.still_frames[sentence_index] = content_image
        return content_image, bool(frame_state["moving"])

    def check_layout(self, scene_time: float) -> List[str]:
        """
        Lists the layout problems of the loaded scene at a time, for the previews: the shown texts reaching out of the
        card and the pairs of shown texts overlapping, found by Tutorial.checkLayout in the page.

        Args:
            scene_time (float): The time from the start of the scene, in seconds.

        Returns:
            List[str]: The problems, as messages naming the start of their texts.
        """
        return self.page.evaluate(expression="(time) => Tutorial.checkLayout(time)", arg=scene_time)
