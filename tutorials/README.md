# Tutorials

## Table Of Content
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Rebuilding The Videos](#rebuilding-the-videos)
- [Updating The Tutorials](#updating-the-tutorials)
- [Writing A Chapter](#writing-a-chapter)
- [Figure Scripts](#figure-scripts)
- [Narration](#narration)
- [Updating The Builder](#updating-the-builder)
- [Troubleshooting](#troubleshooting)

## Description
Narrated tutorial videos of the repository, for engineers who know deep learning and are new to the repository. An
overview video gives the whole repository in one chapter (the ten tasks, each told through a situation where it is the
right tool, how they differ and what they share), a foundations series covers what every task shares (the repository
and a first run, the configuration and the training loop, the data pipeline, the backbone, the optimization, the
evaluation and the deployment), then one series per task covers its data, model, targets, losses, evaluation outputs
and pitfalls.

The videos are not stored in the repository: everything needed to rebuild them is. The builder is the private
repository [tutorial-video-builder](https://github.com/kelevra1993/tutorial-video-builder), installed in the environment
of this folder at a release tag (`pyproject.toml`); its README is the reference of the storyboard keys, the scenes, the
diagram elements and the build steps. A build reads the storyboards of `storyboards/` (one YAML file per chapter), the
configurations and code of the repository they quote, and the figures made by `figures/` from the real runs listed in
`tutorial_sources.yaml`, then synthesizes the narration offline, draws or captures every frame, encodes every chapter
with its subtitles, checks the narration by speech recognition, and writes an index page. Everything runs on the CPU at
the lowest priority, so that the trainings keep the GPU.

To find your way in this folder, what each file holds:

| Folder or file | Content |
|---|---|
| `build_tutorials.sh` | Launcher of the build: the builder of the environment of this folder run on `tutorial_project.yaml` |
| `tutorial_project.yaml` | The project read by the builder: name, repository root, terminal title, figures module, output and cache folders, experiments folder, repository Python |
| `pyproject.toml`, `uv.lock` | Environment of the build, the builder pinned to its release tag |
| `storyboards/` | `series.yaml` (the series in order) and one folder of chapters per series |
| `animations/` | HTML pages of the animated scenes of some chapters, using the engine `common/` of the builder |
| `figures/` | Figure scripts of every series, run with the repository environment |
| `tutorial_sources.yaml` | Real runs read by the figure scripts, and how to train them again |
| `run_configurations/` | Configurations of the runs trained for the tutorials only |

## Prerequisites
Before the first build, the machine needs:
- uv, to create the environment of this folder (`uv sync --project tutorials`) and the one of the repository (`uv sync` at its root): the figure scripts need PyTorch and the repository code.
- SSH access to GitHub for the private builder repository: `uv sync` clones it with `ssh://git@github.com/kelevra1993/tutorial-video-builder.git`.
- Chrome or Chromium, to capture the HTML scenes. The first `google-chrome`, `google-chrome-stable`, `chromium` or `chromium-browser` of the PATH is used, or the one given by `--chrome-executable` or `TUTORIAL_BUILDER_CHROME`; without any, run `uv run --project tutorials playwright install chromium`.
- The real runs of `tutorial_sources.yaml` in the experiments folder, each with its dataset.
- About 1 GB for the caches: the voice and speech recognition models (downloaded on the first build and checked against their SHA-256), the narration of every sentence, the figures and the previews.

## Rebuilding The Videos
Whenever the videos have to be made again, run the launcher from the repository root:

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

Should a build need other folders than those of `tutorial_project.yaml`, an option or a variable overrides them:

| Option | Effect |
|---|---|
| `--series`, `--chapters` | Series (names of `storyboards/series.yaml`) and chapters to build, all by default |
| `--output-folder` | Folder of the videos and of the index page (default `~/Data/tutorial_videos`, or `TUTORIAL_BUILDER_OUTPUT`) |
| `--cache-folder` | Folder of the caches (default `~/.cache/vlms_tutorials`, or `TUTORIAL_BUILDER_CACHE`) |
| `--experiments-folder` | Folder of the real runs (default the `experiments/` folder of the repository, or `TUTORIAL_BUILDER_EXPERIMENTS`) |
| `--repository-python` | Python of the repository environment (default `.venv/bin/python` of the repository, or `TUTORIAL_BUILDER_REPOSITORY_PYTHON`) |
| `--chrome-executable` | Chrome capturing the HTML scenes (or `TUTORIAL_BUILDER_CHROME`) |
| `--threads` | Threads of the voice, the speech recognition, the encoding and the figure scripts (default 4, or `TUTORIAL_BUILDER_THREADS`); 4 leaves the processors to the trainings, raise it when the machine is free |
| `--check-references` | Stop after the reference check |
| `--figures-only`, `--previews-only` | Stop after the figures, or after the previews and their layout check |
| `--skip-figures`, `--force-figures` | Reuse the figures of the cache, or make again the figures the scripts would reuse |
| `--skip-transcription` | Do not check the narration by speech recognition |
| `--allow-layout-problems` | Render even when the previews list layout problems |

The steps of a build (figures, reference check, previews and layout check, videos, narration check, report and index
page) and the layout of the output folder are described in the
[Rebuilding The Videos](https://github.com/kelevra1993/tutorial-video-builder#rebuilding-the-videos) section of the
builder. The output folder holds `index.html`, `verification_report.md` and one folder per series
(`01_foundations/...`) with the MP4 file, the subtitles and the record of every chapter.

## Updating The Tutorials
Every two to four weeks, or after a change of the code they show:
1. `tutorials/build_tutorials.sh --check-references`: every key, function, file and command the storyboards quote that no longer exists is listed. Update the storyboards (their excerpts follow the files by themselves, only renamed keys and functions need an edit).
2. When a run of `tutorial_sources.yaml` was trained again (or must be, after a change of its task), update its folder there.
3. Read the narration of the chapters whose code changed: an excerpt follows the file, but the sentence explaining it does not.
4. Rebuild the changed series, then read `verification_report.md` and look at the contact sheets of the chapters you edited.

## Writing A Chapter
Suppose a series needs one more chapter:
1. Copy a storyboard of the series and number it after the last chapter (`storyboards/<number>_<series>/<number>_<name>.yaml`).
2. List in `figures:` the figure series it uses, and in `references:` the files and functions it explains without quoting them.
3. Write the scenes, every scene with its `narration`, a list of sentences. The keys of a storyboard, the scene kinds, the path prefixes (`figures:`, `repository:`, `runs:`, `animations:`), the `{{series.fact}}` placeholders and the diagram elements are listed in the [Writing A Chapter](https://github.com/kelevra1993/tutorial-video-builder#writing-a-chapter), [Scene Reference](https://github.com/kelevra1993/tutorial-video-builder#scene-reference) and [Diagram Elements](https://github.com/kelevra1993/tutorial-video-builder#diagram-elements) sections of the builder.
4. Check it with `--previews-only --skip-figures`, look at the contact sheets, then build it.

A page of its own (`animations/<number>_<series>/<name>.html`) loads `../common/tutorial.js` and `../common/tutorial.css`,
served from the builder next to the pages of this folder, and adds animated parts with `Tutorial.registerCustom(setup)`.

## Figure Scripts
If you are looking for where a number of the narration comes from, it is a fact of a figure script.
`figures/<series>/figures.py` holds `make_figures(output_folder, experiments_folder, force)`, called by
`figures/make_figures.py` (`figures_module` of `tutorial_project.yaml`) with the repository Python from the repository
root, on the CPU. It reads its runs through `figures/figure_utilities.py` and `tutorial_sources.yaml`, writes its PNG
files into the output folder, reuses its expensive figures unless forced, and returns its facts (numbers, and the pixel
boxes of its panels for the highlights and reveals of the storyboards). Figures are drawn at the size they are shown
(1720 x 700 for a figure filling the card) with the palette and fonts of `tutorial_builder.layout`; the repository root,
the sources file and the threads of the build come from `tutorial_builder.figure_context`, both put on the import path
of the figure scripts by the builder, so a figure script runs through the build only
(`tutorials/build_tutorials.sh --series <series> --figures-only`).

## Narration
So that a chapter reads like the others, its narration follows these rules:
- Write for an engineer with years of computer vision behind them: the repository specifics, the defaults and their measured effects, the pitfalls and where the code is; no textbook explanation.
- One idea per sentence, short enough for three caption lines; the caption follows the sentence being spoken.
- Frame a few sentences of every chapter as a situation the viewer might meet, so that the narration does not only state facts: "Suppose you want to...", "Say you need to...", "Imagine that...", "Picture a data set where...", "Should you need to...", "Faced with...", "If, for instance, you...". Two to four per chapter, never the same opener twice in a chapter; the facts of the sentence stay those of the code. The situations use the synthetic shapes shown on screen most of the time, sometimes everyday examples (cats, dogs, parcels on a shelf), never the data of a particular company.
- Show the real thing: excerpts read from the repository, real runs, real outputs. Never invented console output or numbers.
- Pronunciation of the voice: write "pre-trained" (not "pretrained"), "data set" (not "dataset"), avoid reading tool names such as pytest aloud; "weighting" and "waiting" sound alike. A word the voice reads wrongly whatever its spelling gets a spoken form in `SPOKEN_FORMS` of `tutorial_builder/narration.py` of the builder ("ONNX" read "onyx"), the captions keeping the written word; decimals are written as usual (0.5, 35.3%), the builder giving the voice their decimal part as words. The [Narration](https://github.com/kelevra1993/tutorial-video-builder#narration) section of the builder has the rest.

## Updating The Builder
After a change of the builder, this folder takes it by moving its tag: set `tag = "v<version>"` of
`tutorial-video-builder` in `pyproject.toml`, run `uv sync --project tutorials`, then
`tutorials/build_tutorials.sh --check-references` and the previews of a series. The
[Updating The Builder](https://github.com/kelevra1993/tutorial-video-builder#updating-the-builder) section of the
builder explains how to tag a release and how to try a change before it is tagged.

## Troubleshooting
In case a build stops or crawls, the causes met here (the
[Troubleshooting](https://github.com/kelevra1993/tutorial-video-builder#troubleshooting) section of the builder lists
the others: espeak-ng data path, transparent huge pages, no Chrome):
- **A figure script stops on a missing run**: the message quotes how the run was trained (`tutorial_sources.yaml`).
- **A build from a git worktree**: the launcher of the worktree builds its own storyboards, but the worktree has no `.venv` and no `experiments/` of its own, and shares the cache of the main checkout. Give it the folders of the main checkout and a scratch cache: `tutorials/build_tutorials.sh --experiments-folder <main checkout>/experiments --repository-python <main checkout>/.venv/bin/python --cache-folder <scratch folder>`, after `uv sync --project tutorials` in the worktree.
