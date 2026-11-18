# Maintaining The Tutorials

## Table Of Content
- [Description](#description)
- [Prerequisites](#prerequisites)
- [What Lives Where](#what-lives-where)
- [Building On A New Machine](#building-on-a-new-machine)
- [Run Snapshots](#run-snapshots)
- [Keeping The Repository Self-Sufficient](#keeping-the-repository-self-sufficient)
- [Rules For Writing And Updating Chapters](#rules-for-writing-and-updating-chapters)
- [Review And Rebuild Workflow](#review-and-rebuild-workflow)
- [Builder Releases](#builder-releases)
- [Publishing](#publishing)
- [Known Gaps](#known-gaps)

## Description
If you are an agent or an engineer asked to create, update or rebuild the tutorial videos, this file holds the rules to
follow and the checks that prove the work. The build itself, its options and the storyboard syntax are described in
[README.md](README.md) and in the README of the builder
([tutorial-video-builder](https://github.com/kelevra1993/tutorial-video-builder)); this file does not repeat them.

The goal behind every rule: the videos are rebuilt from the repository alone, every number and every excerpt they show
comes from the code, a configuration or a real run, and nothing a rebuild needs lives only on one machine.

## Prerequisites
Before following this file, the machine needs what [README.md](README.md#prerequisites) lists, and:
- Python 3.13 for both environments: `uv sync --python 3.13` at the repository root (the `datasets` group, fiftyone, is not needed by the figure scripts: `uv sync --python 3.13 --no-group datasets`), and `uv sync --project tutorials`.
- About 6 GB of disk for a full build: the cache measured 5.2 GB after many builds (scene segments 1.5 GB, narration 1.2 GB, audio tracks 0.9 GB, chapter videos 0.8 GB, voice and speech models 0.5 GB), plus 1 GB of videos.
- Internet once, for the voice model (Kokoro, `model-files-v1.0` release), the speech recognition model (faster-whisper `base.en`) and the Playwright Chromium if no Chrome is installed; every downloaded file is checked against its SHA-256. On macOS Chrome is not on the PATH: pass `--chrome-executable "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"`.

## What Lives Where
To find your way between the repository and the machine, every input of a build and where it is kept:

| Input | Where | In git |
|---|---|---|
| Storyboards, one YAML file per chapter | `tutorials/storyboards/<number>_<series>/` | yes |
| Animated scenes | `tutorials/animations/<number>_<series>/` | yes |
| Figure scripts, one package per series | `tutorials/figures/<series>/` | yes |
| Runs read by the figure scripts, and how each was trained | `tutorials/tutorial_sources.yaml` (`runs`) | yes |
| Repository files read by the figure scripts (benchmark results) | `tutorials/tutorial_sources.yaml` (`repository_files`) | yes |
| Launch files of the runs trained for the tutorials | `tutorials/run_configurations/` | yes |
| Small copies of the runs, their dataset files and the figures that need a checkpoint | `tutorials/run_snapshots/`, written by `tutorials/make_run_snapshots.py` from `tutorials/figures/<series>/snapshot_files.yaml` | yes |
| The builder (fonts, palette, scene engine, voice and checks) | private repository, pinned by tag in `tutorials/pyproject.toml` | its own repository |
| The real runs, with their checkpoints | `~/Data/tutorial_runs` (`experiments_folder` of `tutorial_project.yaml`) | no, too large (9.5 GB, 7.3 GB of checkpoints) |
| The datasets of the runs | `~/Data/<task>_data/...`, paths written in the saved configurations | no |
| Caches: models, narration, figures, previews, scene segments | `~/.cache/vlms_tutorials` | no |
| The videos, subtitles, index page and verification report | `~/Data/tutorial_videos` | never: the videos are rebuilt, never edited by hand or committed |

## Building On A New Machine
Suppose a newcomer clones the repository on a machine without the runs: the snapshot folder takes the place of the
experiments folder, and every series builds from the repository alone.

```bash
uv sync --python 3.13 --no-group datasets      # repository environment, read by the figure scripts
uv sync --project tutorials                    # builder environment (needs SSH access to the builder repository)
tutorials/build_tutorials.sh --experiments-folder tutorials/run_snapshots
```

A build from the snapshots reads no checkpoint and no dataset of the machine: the figures that need a checkpoint come
from `run_snapshots/figure_reuse/` and the dataset files from `run_snapshots/datasets/`. The first build of a new
machine draws every scene once, since the scene fingerprints include the versions of Chrome, Playwright, Pillow,
FreeType, numpy and ffmpeg; a facts file that differs from the one of the last build points at a missing or stale
snapshot file.

## Run Snapshots
To be able to rebuild without the 9.5 GB of runs, `tutorials/run_snapshots/` holds, at the same relative paths as the
experiments folder, the small files the figure scripts and the storyboards read from every run (configurations, label
dictionaries, TensorBoard scalars, results and evaluation files, sample images, console logs), so that
`--experiments-folder tutorials/run_snapshots` needs no change of code or storyboard. Next to them:
- `datasets/<path under the recorded data root>`: the dataset files the figure scripts read (one image, a reduced manifest). `read_run_configuration` of `tutorials/figures/figure_utilities.py` points the dataset paths of a snapshot run there, and `resolve_dataset_path` does it for the paths read elsewhere.
- `figure_reuse/<series>/<family>/`: the outputs of the figure families that need a checkpoint (a forward pass of a trained model), copied from the figures folder of a build on the machine of the runs. Without the checkpoint, a family copies them into its figures folder (`restore_committed_figures`) and reuses them.
- `manifests/<series>.json`: every file of the series with its source, its reduction, and the sizes and SHA-256 of the source and of the copy.

Every series declares what it needs in `tutorials/figures/<series>/snapshot_files.yaml`: `runs` (a key of
`tutorial_sources.yaml`, dotted for comparison runs, with the paths relative to the run folder, globs and `..` allowed),
`datasets` (paths relative to the recorded data root) and `figure_reuse` (folders of the series' figures folder). A file
too large for git is reduced in that file:

| Reduction | What it keeps |
|---|---|
| `copy` (default) | the file unchanged |
| `tensorboard_scalars` | the scalar summaries of a TensorBoard log (optionally only some `tags`); sample images make most of the bytes of an evaluation log |
| `json_keys` | some top-level keys of a JSON file (`classes` of a 77 MB `data.json`) |
| `text_lines` | the lines of a log matching a regular expression |
| `csv_columns` | some columns of a CSV file |
| `manifest_subset` | a dataset manifest with the listed images whole and the other records slimmed, so the counts stay right |
| `folder_listing` | a listing (type, size, path) of a folder, for the scenes drawing the tree of a run |
| `compress: gzip` | added to any entry: the copy gzipped, with a fixed header |
| `module:function` | a reduction of one series, in its package |

So that git stays light, a snapshot file is at most 5 MB (the writer refuses more) and the whole folder stays under 200
MB (`maximum_file_megabytes` and `maximum_total_megabytes` of the `snapshots` section of `tutorial_sources.yaml`).
Checkpoints are never committed: a figure needing one either loses the dependency (a model built from the configuration
when only the architecture matters, as the export listings of the deployment series) or commits its outputs under
`figure_reuse/`. A file read by several series is declared identically in all of them (the writer refuses otherwise):
configurations, label dictionaries, checkpoint registries, images and CSV files are copied, TensorBoard logs reduced to
their scalars.

After a change of a run, of a figure script reading a new file, or of a scene showing a new run file, write the
snapshots again on the machine of the runs, from the repository root:

```bash
.venv/bin/python -m tutorials.make_run_snapshots                    # every series, replaces changed files
.venv/bin/python -m tutorials.make_run_snapshots --series foundations   # one series, never changes a file another series reads
.venv/bin/python -m tutorials.make_run_snapshots --prune            # every series, deletes files no manifest lists
.venv/bin/python -m tutorials.make_run_snapshots --check            # compares the folder with its manifests, reads no run
```

The `figure_reuse/` copies come from the figures folder of the cache (`--figures-folder` to take another): build the
figures of the series from the real runs first (`tutorials/build_tutorials.sh --series <series> --figures-only`), so
that the committed outputs are those of the current code. Until they are written again, a build from the snapshots shows
the outputs of the last snapshot, even if the code of their family changed since: refresh them with every change of a
family that needs a checkpoint.

## Keeping The Repository Self-Sufficient
Whenever a figure script, a storyboard or a run starts reading a file that is not in the repository, make it
reconstructible in the same change:
1. A file of a run (a new TensorBoard tag, an evaluation file, a sample image, a `runs:` path of a storyboard): add it to `snapshot_files.yaml` of the series, reduced if needed, and write the snapshot.
2. A dataset file: add it under `datasets:` and read its path through `read_run_configuration` or `resolve_dataset_path`, keeping the folder names (some scripts print or check `test` or `test_10000`).
3. An output of a trained model: avoid it when the architecture is enough; otherwise commit the family outputs under `figure_reuse:` and restore them when the checkpoint is absent.
4. A file of another folder of the machine (a benchmark output, an export): copy it next to its documentation (as the serving benchmark results next to `documentation/architecture_benchmarking/serving_benchmark_plan.md`) and name it under `repository_files` of `tutorial_sources.yaml`.
5. A new run: its entry in `tutorial_sources.yaml` says exactly how it was trained (configuration, every change, iterations, resumes, the commit when the code mattered), and its own `training_configuration.yaml` (and the `training_configuration_until_iteration_*.yaml` of a resumed run) goes into the snapshot, the exact record of the recipe.

Then prove it, on the machine of the runs:
```bash
.venv/bin/python -m tutorials.make_run_snapshots --series <series>
strace -f -e trace=openat,stat,newfstatat,statx -o /tmp/snapshot_trace.txt \
  tutorials/build_tutorials.sh --series <series> --figures-only --cache-folder <empty scratch folder> \
  --experiments-folder tutorials/run_snapshots
grep -c "$HOME/Data" /tmp/snapshot_trace.txt           # must be 0
diff -r <empty scratch folder>/figures/<series> ~/.cache/vlms_tutorials/figures/<series>   # must be identical
```
A facts or figure difference means a file the snapshot misses or a reduction that drops something read; fix the
declaration, not the comparison. The same build from the real runs into another empty scratch folder must also match,
so that the change did not alter what the real runs give.

## Rules For Writing And Updating Chapters
So that every chapter stays true and consistent, these rules apply on top of the
[Narration](README.md#narration) and [Numbers In The Storyboards](README.md#numbers-in-the-storyboards) sections:
- **Audience**: engineers with four years or more of computer vision; repository specifics, defaults and their measured effects, pitfalls and where the code is; no textbook explanation.
- **Scenarios**: two to four situation openers per chapter ("Suppose you…", "Should you need to…", "Faced with…"), never the same twice in a chapter; examples use the synthetic data shown on screen, sometimes everyday ones (cats, dogs, parcels), never company or client data. The same rule opens every section of the READMEs and of this file with a sentence saying when the reader needs it.
- **Stable names**: name the function or the file, never volatile identifiers (bug-list numbers, audit items, commit hashes).
- **Numbers from runs**: never type a number or a console output by hand; facts come from the figure scripts (`facts.json`), configuration values and constants from placeholders read at build time. A number of the method goes in `literal_numbers` with a comment naming its source.
- **Facts follow the screen**: a value the chapter shows from a configuration file is read from that file, not from an older run trained with other values.
- **Voice**: `voice: {name: af_heart, language: en-us}` and `pace: 1.0` in every storyboard (1.16 is too fast); keep them unless the owner asks for another voice. Synthesize a doubtful word alone and transcribe it back before adding a spoken form.
- **Summaries**: seven items at most.
- **Old runs**: a fact outdated only because its run used older code is reworded so that it does not contradict the code, and the run is listed for retraining or re-evaluation at the next rebuild; nothing is retrained or re-evaluated without the owner's approval, and runs trained only to verify code use very small models and few iterations.
- **Drift is silent**: the reference check catches renames and the placeholders follow the code, but a number kept in `literal_numbers`, a claim without a number and a new section are not checked. A commit changing a behaviour a tutorial shows also searches the storyboards and the workflow documents for the old value and updates them in the same change.
- **Backlog**: a feature that lands without tutorial coverage gets an entry in `documentation/tutorial_backlog.md` (see its Adding An Entry section), removed once a rebuilt chapter shows it.

## Review And Rebuild Workflow
Every two to four weeks, or before a rebuild, review the storyboards and the workflow documents against the code:
1. Split the review by series among parallel reviewers, each owning disjoint files (a storyboard folder and its figure package), each working in its own git worktree.
2. Run `tutorials/build_tutorials.sh --check-references`, then `--check-drift` to list the scenes a build would draw again and why.
3. Rebuild every series rather than choosing chapters by hand; the scene cache redraws only what changed. From a worktree, give the Python of the main checkout and a scratch cache (`--repository-python <main checkout>/.venv/bin/python --cache-folder <scratch>`), after `uv sync --project tutorials` in the worktree.
4. Look at the contact sheets of every edited chapter (`previews/` of the cache: captions over three lines, text out of its card, overlaps) and read `verification_report.md` (narration flagged above 5 % word error rate; number spellings are left out of the rate, so listen to the pauses around numbers separately).
5. Write the snapshots again when a run, a figure input or a family needing a checkpoint changed, and run the snapshot proof above for the series concerned.
6. Build on CPU only, at the lowest priority, with few threads (4 by default) while trainings use the machine.

## Builder Releases
In case the builder itself must change, the change is made in its repository, tested (`uv run pytest`), tagged and
pushed; a pushed tag is never moved or deleted, a tag put on the wrong commit being superseded by the next patch
version. This repository then moves `tag` in `tutorials/pyproject.toml`, runs `uv sync --project tutorials`, the
reference check and the previews of a series (see [Updating The Builder](README.md#updating-the-builder)). Each project
using the builder has its own output and cache folders, so that one build never overwrites the index page and
verification report of another.

## Publishing
Once a build is verified, the owner shares the videos by copying the series folders, `index.html` and
`verification_report.md` of the output folder to the Google Drive folder "Tutorials Vision Training" (with `rclone copy`
and `rclone check`); never create a public link to them.

## Known Gaps
Should you have to train a run again, these are the pieces the repository cannot give back yet:
- **Datasets are not reproducible.** `vision_training/data/synthetic/create_synthetic_dataset.py` draws every chunk from an unseeded `np.random.default_rng()`, and its class sets changed since several runs were trained (the classification shapes went from 9 to 35 classes, the instance segmentation shapes from 20 to 35, the keypoint shapes from 33 to 37, the segmentation shapes from 9 to 35; the moving shapes now draw 2 to 12 frames instead of 8). A run trained again on a regenerated dataset gets another run name (`MC<classes>`), other numbers and other sample images; the scenes naming a given image (`STAR_IMAGE_NAME = test_005452.png` and similar constants of the figure scripts) must then be reviewed. The snapshot keeps only the dataset files the figures read.
- **Untracked launch files.** The detection benchmark runs (`run_9`, `run_10`, `run_21`, `run_22`) were launched from launch files that are not in git; their exact recipe is the `training_configuration.yaml` of each run, kept in the snapshot.
- **Committed figures age.** The `figure_reuse/` outputs show the code of their last snapshot until written again from the real checkpoints.
