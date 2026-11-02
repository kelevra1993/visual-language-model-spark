# Tutorials

## Table Of Content
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Rebuilding The Videos](#rebuilding-the-videos)
- [Updating The Tutorials](#updating-the-tutorials)
- [Writing A Chapter](#writing-a-chapter)
- [Scene Reference](#scene-reference)
- [Diagram Elements](#diagram-elements)
- [Figure Scripts](#figure-scripts)
- [Narration](#narration)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)

## Description
Narrated tutorial videos of the repository, for engineers who know deep learning and are new to the repository. An
overview video gives the whole repository in one chapter (the ten tasks, each told through a situation where it is the
right tool, how they differ and what they share), a foundations series covers what every task shares (the repository
and a first run, the configuration and the training loop, the data pipeline, the backbone, the optimization, the
evaluation and the deployment), then one series per task covers its data, model, targets, losses, evaluation outputs
and pitfalls.

The videos are not stored in the repository: everything needed to rebuild them is. A build reads the storyboards of
`storyboards/` (one YAML file per chapter), the configurations and code of the repository they quote, and the figures
made by `figures/` from the real runs listed in `tutorial_sources.yaml`, then synthesizes the narration offline,
draws or captures every frame, encodes every chapter with its subtitles, checks the narration by speech recognition,
and writes an index page. Everything runs on the CPU at the lowest priority, so that the trainings keep the GPU.

| Folder or file | Content |
|---|---|
| `build_tutorials.sh` | Launcher of the build, with the environment it needs |
| `tutorial_builder/` | The builder: storyboard loading, scene rendering, HTML capture, narration, encoding, checks, index page |
| `storyboards/` | `series.yaml` (the series in order) and one folder of chapters per series |
| `animations/` | HTML pages of the diagram scenes: `diagram.html` and the engine `common/tutorial.js` |
| `figures/` | Figure scripts of every series, run with the repository environment |
| `fonts/` | Inter and Hack, used by every frame |
| `tutorial_sources.yaml` | Real runs read by the figure scripts, and how to train them again |
| `tests/` | Tests of the builder |

## Prerequisites
- uv, to create the environment of this folder (`uv sync` in `tutorials/`) and the one of the repository (`uv sync` at its root): the figure scripts need PyTorch and the repository code.
- Chrome or Chromium, to capture the HTML scenes. The first `google-chrome`, `google-chrome-stable`, `chromium` or `chromium-browser` of the PATH is used, or the one given by `--chrome-executable` or `VLMS_TUTORIALS_CHROME`; without any, run `uv run --project tutorials playwright install chromium`.
- The real runs of `tutorial_sources.yaml` in the experiments folder, each with its dataset.
- About 1 GB for the caches: the voice and speech recognition models (downloaded on the first build and checked against their SHA-256), the narration of every sentence, the figures and the previews.

## Rebuilding The Videos
From the repository root:

```bash
# Every series: figures, reference check, previews, videos, narration check, report and index page
tutorials/build_tutorials.sh

# One series, or some chapters of it
tutorials/build_tutorials.sh --series object_detection
tutorials/build_tutorials.sh --series foundations --chapters 1 2

# Only check that every storyboard still matches the repository and the facts of the cache (seconds, no figure made)
tutorials/build_tutorials.sh --check-references

# Edit a storyboard and look at it: previews of every sentence and their contact sheets, reusing the figures
tutorials/build_tutorials.sh --series foundations --chapters 3 --previews-only --skip-figures
```

| Option | Effect |
|---|---|
| `--series`, `--chapters` | Series (names of `storyboards/series.yaml`) and chapters to build, all by default |
| `--output-folder` | Folder of the videos and of the index page (default `~/Data/tutorial_videos`, or `VLMS_TUTORIALS_OUTPUT`) |
| `--cache-folder` | Folder of the caches (default `~/.cache/vlms_tutorials`, or `VLMS_TUTORIALS_CACHE`) |
| `--experiments-folder` | Folder of the real runs (default the `experiments/` folder of the repository, or `VLMS_TUTORIALS_EXPERIMENTS`) |
| `--repository-python` | Python of the repository environment (default `.venv/bin/python` of the repository, or `VLMS_TUTORIALS_REPOSITORY_PYTHON`) |
| `--chrome-executable` | Chrome capturing the HTML scenes |
| `--threads` | Threads of the voice, the speech recognition, the encoding and the figure scripts (default 4, or `VLMS_TUTORIALS_THREADS`); 4 leaves the processors to the trainings, raise it when the machine is free |
| `--check-references` | Stop after the reference check |
| `--figures-only`, `--previews-only` | Stop after the figures, or after the previews and their layout check |
| `--skip-figures`, `--force-figures` | Reuse the figures of the cache, or make again the figures the scripts would reuse |
| `--skip-transcription` | Do not check the narration by speech recognition |
| `--allow-layout-problems` | Render even when the previews list layout problems |

The steps of a build, each stopping it on a problem:
1. **Figures**: the figure scripts of the series named by the storyboards (`figures:` key) run with the repository Python on the CPU, writing their PNG files and `facts.json` (the numbers the narration quotes) into the cache. They come first so that the next step reads the facts of this build: a fact added to a figure script exists only once its figures are made again. `--check-references` and `--skip-figures` make none.
2. **Reference check**: every storyboard is loaded, which reads every configuration and code excerpt it quotes, checks every highlight pattern and fills every fact, then the file and function of every scene chip, the modules and configurations of the terminal commands and the `references` of the storyboard are looked up. All the problems of all the storyboards are listed at once.
3. **Previews**: one settled frame per sentence and contact sheets, in `<cache>/previews/<number>_<series>/<storyboard>/`, with a layout check: captions longer than three lines, texts of HTML scenes out of their card or overlapping.
4. **Videos**: the narration of every sentence (read from the cache when its text, voice and pace did not change), the frames, the MP4 file, the SubRip subtitles and the record of the chapter (`<name>.json`).
5. **Narration check**: the audio transcribed by faster-whisper and compared with the script, word by word.
6. **Report and index**: `verification_report.md` (word error rate of every chapter, flagged above 5 %, and every difference; the numbers written one way and heard the other, the British spellings heard in American and the words heard split or joined are spelling differences, left out of the rate) and `index.html` (every chapter with its player, WebVTT subtitles, scenes as timestamps and recap) in the output folder.

The output folder, movable as it is:

```text
<output folder>/
├── index.html                  # Every series and chapter, with players and scene timestamps
├── verification_report.md      # Narration check of every chapter built (and its JSON version)
└── <number>_<series>/      # 01_foundations, numbered in the order of storyboards/series.yaml
    ├── <chapter>.mp4           # The video, 1920 x 1080, H.264 and AAC
    ├── <chapter>.srt / .vtt    # Its subtitles
    └── <chapter>.json          # Its record: scenes, sentences and their times
```

## Updating The Tutorials
Every two to four weeks, or after a change of the code they show:
1. `tutorials/build_tutorials.sh --check-references`: every key, function, file and command the storyboards quote that no longer exists is listed. Update the storyboards (their excerpts follow the files by themselves, only renamed keys and functions need an edit).
2. When a run of `tutorial_sources.yaml` was trained again (or must be, after a change of its task), update its folder there.
3. Read the narration of the chapters whose code changed: an excerpt follows the file, but the sentence explaining it does not.
4. Rebuild the changed series, then read `verification_report.md` and look at the contact sheets of the chapters you edited.

## Writing A Chapter
1. Copy a storyboard of the series and number it after the last chapter (`storyboards/<number>_<series>/<number>_<name>.yaml`).
2. List in `figures:` the figure series it uses, and in `references:` the files and functions it explains without quoting them.
3. Write the scenes (next section), every scene with its `narration`, a list of sentences.
4. Check it with `--previews-only --skip-figures`, look at the contact sheets, then build it.

The keys of a storyboard:

| Key | Content |
|---|---|
| `series`, `series_title` | Folder of the series and its title in the title bar |
| `chapter`, `title` | Number and title of the chapter |
| `output_name` | Name of the video files |
| `pace`, `voice` | Pace of the narration (1.0, the natural speed of the voice) and its voice (`{name: af_heart, language: en-us}`) |
| `figures` | Figure series whose figures and facts the chapter uses |
| `references` | Files, or `file · Class.method` and `file · Yaml.key` names, checked to exist |
| `scenes` | The scenes, in order |

Paths in a storyboard take a prefix: `figures:` (the figures of the cache, `figures:<series>/...`), `repository:` (a file of the repository), `runs:` (the experiments folder), `animations:` (this `animations/` folder). Texts quote the facts of the figure scripts as `{{series.name}}` or `{{series.name:format}}` (a Python format such as `.1%`); a value made of a placeholder alone takes the fact as it is, a list or a number, such as the box of a panel. Indices after the name take one item of a list fact: `{{series.panel_boxes[2]}}` (the box of the third panel, for a highlight) or `{{series.matrix[1][0]:.2f}}`; an index outside the list stops the build.

## Scene Reference
Every scene has a `kind`, a `title` (but the title card), a `narration`, an optional `code_reference` (shown under its title: `file · name`, checked by the reference check) and an optional `hold_after` (`{sentence index: seconds}`, a longer pause after a sentence).

| Kind | Keys | Drawn |
|---|---|---|
| `title_card` | `heading`, `subheading`, `footnote` | Opening card of a chapter |
| `yaml`, `code` | `source`, `highlights`, optional `syntax`, `file_label`, `font_size` | Excerpt read from the repository at build time |
| `terminal` | `lines` (`role`: command, comment, output; `text`) | Commands typed one after the other |
| `image` | `image`, optional `views`, `highlights`, `crop_box`, `overlay` with `reveals` | A figure, a camera moving between its views |
| `text_panel` | `columns` (`heading`, `lines`, optional `syntax`), optional `footnote`, `font_size` (26 by default) | Columns of monospace text |
| `html` | `page` (`diagram` or a page of `animations/`), `elements`, optional `data` | SVG diagram or animation captured from Chrome |
| `summary` | `items` (`text`, optional `monospace`), 7 at most | Recap of the chapter |

The `source` of an excerpt is a file with:
- `keys`: dotted YAML keys (`Optimization.learning_rate_schedule`), every block shown under its parent keys, `...` where lines are skipped, `comments: false` to leave the comments and the blank lines between them out;
- `function`: a function, class or constant of a Python file (`ConvolutionBlock.compute_layers`, `SUPERVISED_AUGMENTATION_STRENGTHS`), its docstring left out unless `docstring: true`, narrowed by `start` and `end` patterns (a constant keeping its first line, as a function its signature);
- or `start`, `end` and `count` for any text file.

A highlight is `{match: <regular expression>, sentences: [...]}` (or `always: true`, `first_only: true`) and must match a line, so a renamed key breaks the reference check instead of highlighting nothing. The font of an excerpt shrinks from 26 to 18 pixels to fit; longer excerpts stop the build. `font_size` (an even size from 16 to 34) forces the size of an excerpt or of the lines of a text panel, larger for a short one that would leave the card empty, smaller for a long one; a size at which the scene does not fit stops the build, naming the largest size that fits.

`views` of an image scene are `{box: [left, top, right, bottom], from_sentence: <index>}` in pixels of the figure; `highlights` are `{box, sentences}`; `reveals` uncover a region of the `overlay` image from its left edge while their `sentence` is spoken.

## Diagram Elements
An html scene with `page: diagram` draws its `elements` in the content area, 1760 x 740 pixels from its top left corner. Every element has an optional `appear` (the sentence it fades in with, -1 or absent for the whole scene), `until` (the last sentence it stays), `highlight` and `dim` (lists of sentences). An element rises 14 pixels as it fades in, and the elements appearing with the same sentence enter 0.07 seconds apart in the order of the list (0.35 seconds at most); arrows, bars, curves and grids grow instead of rising. A highlighted element turns to the accent colour with a glow.

| Type | Keys |
|---|---|
| `box` | `box`, `text`, `font`, `fill` (null for an outline), `outline`, `text_colour`, `radius`, `width` |
| `arrow` | `points`, `colour`, `width`, `head` (end, start, both, none), `label`, `label_position`, `label_anchor`, `flow` (dots flowing while highlighted), `dash` (`[dash, gap]` in pixels for a dashed arrow) |
| `text` | `position`, `text`, `font`, `colour`, `anchor` (`la`, `mm`, `ma`, `ls`, `rm`...), or `segments` (`text`, `colour`, `highlight`, `dim`) |
| `image` | `path`, `box`, `crop`, `scaling` (`pixel`), `border`, `caption` |
| `stack` | `box`, `count`, `offset`, `fill` or `fills`, `layer_opacities`, `label` |
| `bars` | `box`, `values`, `orientation`, `labels`, `value_labels`, `colour` or `colours`, `maximum` |
| `grouped_bars` | `box`, `groups` (their names), `series` (`name`, `colour`, `values` per group, optional `value_labels`), `orientation`, `maximum`, `legend` (false to leave the legend above the plot out), `group_gap`, `bar_gap`, `label_width` |
| `cuboid` | `box` (front face), `depth`, `fill`, `text`, `label` (a tensor and its shape) |
| `grid` | `box`, `values` (rows), `show_values`, `decimals`, `row_labels`, `column_labels`, `low_colour`, `high_colour`, `marks` (`row`, `column`, `sentences`) |
| `curve` | `box`, `series` (`name`, `colour`, `points`, `highlight`, `dim`), `x_range`, `y_range`, `x_ticks`, `y_ticks`, `x_label`, `y_label`; the names written at the last points are moved apart when they would overlap, so leave room right of the box |
| `listing` | `box`, `source` (like an excerpt), `highlights`, `font` |

Fonts: `diagram_heading`, `diagram_bold`, `diagram_text`, `diagram_small`, `diagram_tiny`, `diagram_monospace`, `diagram_monospace_small`, `diagram_monospace_tiny`, `code_text`. Colours: `text`, `dim`, `accent`, `key`, `value`, `box`, `border`, `card`, and the series colours `blue`, `aqua`, `yellow`, `magenta`, `violet`, `red`, always labelled by their text; or `#rrggbb`. The palette (a dark ink blue with a cyan accent) is defined in `tutorial_builder/layout.py` and copied in `animations/common/tutorial.js` and `tutorial.css`; a test fails when the copies differ, and the figure scripts take their colours from `layout.py`, so a new palette needs `--force-figures`. A page of its own (`animations/<number>_<series>/<name>.html`) loads `common/tutorial.js` and adds animated parts with `Tutorial.registerCustom(setup)`, the setup returning the function drawing them at a time.

## Figure Scripts
`figures/<series>/figures.py` holds `make_figures(output_folder, experiments_folder, force)`, called by `figures/make_figures.py` with the repository Python from the repository root, on the CPU. It reads its runs through `figures/figure_utilities.py` and `tutorial_sources.yaml`, writes its PNG files into the output folder, reuses its expensive figures unless forced, and returns its facts (numbers, and the pixel boxes of its panels for the highlights and reveals of the storyboards). Figures are drawn at the size they are shown (1720 x 700 for a figure filling the card) with the palette and fonts of the videos.

## Narration
- Write for an engineer with years of computer vision behind them: the repository specifics, the defaults and their measured effects, the pitfalls and where the code is; no textbook explanation.
- One idea per sentence, short enough for three caption lines; the caption follows the sentence being spoken.
- Frame a few sentences of every chapter as a situation the viewer might meet, so that the narration does not only state facts: "Suppose you want to...", "Say you need to...", "Imagine that...", "Picture a data set where...", "Should you need to...", "Faced with...", "If, for instance, you...". Two to four per chapter, never the same opener twice in a chapter; the facts of the sentence stay those of the code. The situations use the synthetic shapes shown on screen most of the time, sometimes everyday examples (cats, dogs, parcels on a shelf), never the data of a particular company.
- Show the real thing: excerpts read from the repository, real runs, real outputs. Never invented console output or numbers.
- Pronunciation of the voice: write "pre-trained" (not "pretrained"), "data set" (not "dataset"), avoid reading tool names such as pytest aloud; "weighting" and "waiting" sound alike. A doubtful word is checked by synthesizing it alone and transcribing it, and the narration check of every build flags the rest. A word the voice reads wrongly whatever its spelling gets a spoken form in `SPOKEN_FORMS` of `tutorial_builder/narration.py` ("ONNX" read "onyx"), the captions keeping the written word.

## Troubleshooting
- **"Error processing file .../espeak-ng-data/phontab"**: the espeak-ng data path was too long for espeak-ng; the builder copies the data into `<cache>/espeak-ng-data-<version>`, so give a shorter `--cache-folder` if the message names the cache.
- **Frames or figures take seconds each and the build uses its CPU in the kernel**: the transparent huge pages of numpy and PyTorch with fragmented memory; `build_tutorials.sh` sets `NUMPY_MADVISE_HUGEPAGE=0` and `THP_MEM_ALLOC_ENABLE=0` (the figure scripts get both too), run the build through it.
- **No Chrome**: install one, or run `uv run --project tutorials playwright install chromium`.
- **A figure script stops on a missing run**: the message quotes how the run was trained (`tutorial_sources.yaml`).

## Tests
```bash
uv run --project tutorials pytest tutorials/tests
```
