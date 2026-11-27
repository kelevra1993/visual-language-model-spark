# Reduced README File Ideas

Working document: ideas collected by eight parallel reviewers on 2026-11-27 for shrinking `README.md` and
`documentation/reference/tasks_reference.md` and leaning on illustrations. Nothing was changed in either file; we go
through this file together from top to bottom, decide each point, then rewrite. Line numbers are those of main
`1c4d3d63` (`README.md:123` is line 123 of the README). The file is deleted once the rewrite is done.

## Table Of Content
- [The Numbers](#the-numbers)
- [Decisions To Take First](#decisions-to-take-first)
- [Proposed Outline Of The Reduced README](#proposed-outline-of-the-reduced-readme)
- [Work Needed Before Any Figure](#work-needed-before-any-figure)
- [Side Findings](#side-findings)
- [Part 1: Title To Moving Average Of The Weights (README.md:1-307)](#part-1-title-to-moving-average-of-the-weights-readmemd1-307)
- [Part 2: Tasks And Label Files (README.md:308-730)](#part-2-tasks-and-label-files-readmemd308-730)
- [Part 3: Transfer Between Runs To Tutorials (README.md:731-935)](#part-3-transfer-between-runs-to-tutorials-readmemd731-935)
- [Part 4: Deployment Concepts And Number Formats (README.md:936-1275)](#part-4-deployment-concepts-and-number-formats-readmemd936-1275)
- [Part 5: Export To Repository Structure (README.md:1276-1579)](#part-5-export-to-repository-structure-readmemd1276-1579)
- [Part 6: Metrics Explained, CLS, DET And INS (README.md:1580-2010)](#part-6-metrics-explained-cls-det-and-ins-readmemd1580-2010)
- [Part 7: Metrics Explained, KPT To SSL (README.md:2011-2643)](#part-7-metrics-explained-kpt-to-ssl-readmemd2011-2643)
- [Part 8: Tasks Reference (tasks_reference.md, 441 lines)](#part-8-tasks-reference-tasks_referencemd-441-lines)

## The Numbers

To see at a glance where the length is, here are the sizes of each README range today and after the proposals of its part.

| Part | README range | Lines now | README lines after | Where the rest goes |
|---|---|---|---|---|
| 1 | Title to Moving Average (1-307) | 307 | ~138 | training_reference.md (6 screenshots, run folder details) |
| 2 | Tasks and Label Files (308-730) | 423 | ~32 | tasks_reference.md (label files, sequence pitfalls, flatten figures) |
| 3 | Transfer to Tutorials (731-935) | 205 | ~74 | tools_reference.md, configuration_reference.md, tutorials/README.md |
| 4 | Deployment concepts, Number Formats (936-1275) | 340 | ~7 | new teaching page(s), ~315 lines (decision D1) |
| 5 | Export to Repository Structure (1276-1579) | 304 | ~66 | new deployment document, training_reference.md (Docker, pods) |
| 6 | Metrics Explained: intro, CLS, DET, INS (1580-2010) | 431 | ~14 | new metrics_reference.md, ~190 lines |
| 7 | Metrics Explained: KPT to SSL (2011-2643) | 633 | 0 | new metrics_reference.md, ~274 lines |
| **README** | | **2,643 (48,190 words)** | **~331** | |
| 8 | tasks_reference.md | 441 (8,843 words) | ~356 (~3,900 words) | takes the README label files, gives its Evaluation and Configuration prose to links |

Metrics Explained alone (1580-2643) is about 26,000 words, more than half of the README; the deployment teaching
(936-1275) is another 7,500. Those two moves make most of the reduction; the rest comes from removing copies of
facts that already exist in the reference files.

## Decisions To Take First

Before going section by section, these choices shape every part; the reviewers disagreed on the first one and the
others decide where content lands.

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Home of the deployment teaching (README.md:936-1275) and of the measured serving sections (1384-1473) | (a) part 4: `documentation/explained/deployment_explained.md` (~230 lines) + `documentation/explained/number_formats_explained.md` (~85 lines), a teaching folder that could also hold Metrics Explained; (b) part 5: one `documentation/reference/deployment_reference.md` holding the concepts, number formats, the measured serving paths, Export Sizes and the whole Deployment section of tools_reference.md:200-312 | (b) one `deployment_reference.md`, with Number Formats as its own top-level section so training can link to it: one home for every deployment fact, and tools_reference.md stops being split in two |
| D2 | Home of Metrics Explained (README.md:1580-2643) | (a) `documentation/reference/metrics_reference.md` (parts 6, 7); (b) per task inside tasks_reference.md (rejected by part 6: the metrics are shared by several tasks and would triple the file); (c) the `documentation/explained/` folder of D1 (a) | (a), H1 kept as "Metrics Explained", images stay in `README/metrics_*`; the README keeps a 14-line table task → headline metric → link at its end |
| D3 | Who owns the label file format | README Label Files today (378 lines of commented JSON) versus tasks_reference.md | tasks_reference.md owns it (parts 2 and 8 agree): one compact commented example per task section; the README keeps one sentence and a link |
| D4 | Main illustration of a task | (a) part 2: a "label file atlas", two images (image tasks, sequence tasks), sample next to its `labels.json` record; (b) part 8: ten "task cards" of 4 panels (sample drawn with labels, record and mapping line, measured tensor shapes, outputs and headline metric) | (b) in tasks_reference.md, drawn by one script from `create_synthetic_dataset.py --task=<task>` output so it cannot drift; the README keeps `tasks_overview.png` only |
| D5 | The six per-task tables of the README (Description, Quick Start, Tasks, Run Names, Export, Metrics Explained) | keep one | keep the Tasks table as the single index (Task, Predicts, Headline metric, links to reference, workflow and tutorial), the Quick Start table keeps only the commands |
| D6 | Metric text inside tasks_reference.md (Evaluation blocks, ~2,900 words) | keep, or one line + link | one line per task (headline metric, report tables) + link to metrics_reference.md (parts 6, 7, 8 agree); the SEL decodings stay in the task reference as model behaviour |
| D7 | Configuration prose inside tasks_reference.md (benchmark history, key lists) | keep, or rely on the YAML comments and the workflow tables | cut, benchmark history to the experiment plans |
| D8 | Docker build details, multi-architecture push, Kubernetes pods (README.md:1487-1541) | README, training_reference.md, or a commented example pod manifest | README keeps the native build and `docker run` (~14 lines), the rest to training_reference.md |

## Proposed Outline Of The Reduced README

So that the reader follows one journey from clone to served model, parts 1, 3 and 5 converge on this outline
(about 330 lines):

1. Title and `tasks_overview.png`
2. Table Of Content (~16 lines, top-level only)
3. Description (what the repository does, the task table of D5, `architectures.png`)
4. Prerequisites (uv sync, `scripts_and_requirements.png`, GPU)
5. Quick Start With Synthetic Data (one command block, a new "what to point where" figure)
6. Your Own Data (folder layout figure, one link per task to tasks_reference.md; replaces Data Requirements and Label Files)
7. Training: Launching A Run (with Configurations), Run Folder (with Run Names), Stopping And Resuming, Following A Run (console, TensorBoard, MLflow), Checking Before A Long Run (diagnosis table), Moving Average, Starting From A Trained Run (Transfer Between Runs)
8. Evaluating Checkpoints (command, folder figure, an "Other tools" table: comparing launches, clustering, report figures, large splits)
9. Deployment (~30 lines: settings table, export command, `export_pipeline.png`, `docker compose up`, `run_inference`, a "which path" list, link to the deployment document)
10. Docker Image (~14 lines)
11. Tests (~5 lines)
12. Documentation Map (one table: references, workflows, tutorials, metrics, deployment, repository structure)
13. Metrics Explained pointer table (14 lines, D2)

## Work Needed Before Any Figure

Should we adopt the figure ideas, a few things have to happen first.
- The scripts that drew the `README/metrics_*` figures were never committed (c8486735 added the images only), and the
  README image helpers live outside the repository (`~/Data/tutorial_runs/readme_screenshots/helpers/`). Extending a
  figure means finding or rewriting its script; a tracked home such as `scripts/documentation/figures/` was proposed.
- `README/tests_layout.png` is stale (1,004 test cases drawn, about 1,266 today).
- `README/transfer_between_runs.png` has no tracked script; a "heads without class layers" row was proposed for it.
- Inbound links to re-point when sections move: `README.md#export-sizes` (12 places), `#export`,
  `#serving-benchmark-results`, `#data-requirements` (pretraining_reference.md), `#comparing-launches`,
  `#clustering-image-embeddings`, `#configurations`, `#run-names`, `#mlflow-tracking`, about 30 internal metric links
  (`#recall-intervals` alone 10 times); the storyboards and the deck to be grepped for README anchors.

## Side Findings

These came up during the review and are outside the reduction itself.
- "the README run" (README.md:2368-2485) is never defined; it is `tutorials/run_snapshots/readme_screenshots/SEL_..._FPC1-SLW1`.
- `readme_screenshots/SEG_..._MC9` (README.md:2184) lacks its `tutorials/run_snapshots/` prefix.
- The `uv python install` / `uv python pin` block (README.md:101-106) may be redundant since `uv sync` downloads the
  pinned Python; to check on a machine without Python 3.13 before removing it.
- The Triton container is named `object-detection-triton-server` (`vision_training/deployment/triton/docker-compose.yml:18`) although it serves every task.
- The serving benchmark DET rows of README.md:1442-1473 are out of date against `serving_benchmark_plan.md`.
- Facts that exist only in tasks_reference.md and must move, not vanish: the DeCov `decorrelation` explanation, the
  ImageNet-22k split counts and retry details, the synthetic constant names, `data_retriever.py`, the Lovász / Dice /
  boundary loss descriptions.
- Repeated facts to settle once: the detector FP16 overflow (7.88 points) told 5 times, "the export turns TF32 off"
  6 times, eager 2.2 ms against TorchScript 3.3 ms 4 times, the TensorRT fusion and engine costs twice, the
  "percentages with two decimals" sentence 8 times in tasks_reference.md.

---

The eight parts follow in the order of the README, then the task reference. Each part keeps its reviewer's
recommendation; where a part disagrees with the decisions above, the decision table wins once we take it.

## Part 1: Title To Moving Average Of The Weights (README.md:1-307)

Compared with documentation/reference/training_reference.md (291 lines), configuration_reference.md (295 lines) and
the Synthetic Datasets section of tasks_reference.md (tasks_reference.md:20-46). Line numbers are those of README.md on
main 1c4d3d63.

Principle used throughout: the README keeps the command, one scenario sentence and one figure per step; every
"how it works in detail" paragraph goes to (or already exists in) training_reference.md, which then also receives the
screenshots the README drops (training_reference.md has no image at all today).

---

#### Title and overview figure (README.md:1-7, 7 lines)
- **Keep in README**: the title and `README/tasks_overview.png` (one glance at every task), 5 lines.
- **Cut**: the caption's dates and dataset history ("trained on 2026-10-21 and 2026-11-15, on the datasets of those
  days", README.md:6). They date the page and tell a newcomer nothing. Replace with "One evaluation sample of every
  supervised task, trained on the synthetic datasets."
- **Move**: nothing.
- **Illustrate**: the existing figure is enough.
- **Estimated length after**: 5 lines.

#### Table Of Content (README.md:8-54, 47 lines)
- **Keep in README**: a table of contents of top-level sections only, plus the 5 Training subsections. Each section
  then fits on one screen, so deep entries are not needed.
- **Cut**: the 8 "Metrics Explained" sub-entries (README.md:46-53) and the 7 "Deployment" sub-entries
  (README.md:32-38). These sections move to their own documents (other parts of this brief), so their entries move
  with them.
- **Move**: the documentation list of README.md:81-87 becomes the last top-level section, "Documentation Map"
  (see Description below), so the TOC ends on a section that points everywhere.
- **Proposed outline of the reduced README** (reader journey clone → data → train → evaluate → deploy):

  ```markdown
  ## Table Of Content
  - [Description](#description)                          tasks table, architectures figure
  - [Prerequisites](#prerequisites)                      uv, GPU, uv sync
  - [Quick Start With Synthetic Data](#quick-start-with-synthetic-data)   generate, point the config, train
  - [Your Own Data](#your-own-data)                      folder layout, labels.json, label_mapping.json, converters (today "Tasks" + "Label Files")
  - [Training](#training)
    - [Launching A Run](#launching-a-run)
    - [Run Folder](#run-folder)
    - [Stopping And Resuming](#stopping-and-resuming)
    - [Following A Run](#following-a-run)               today "Console And TensorBoard" + MLflow pointer
    - [Checking Before A Long Run](#checking-before-a-long-run)   today "Augmentation And Diagnosis Modes"
  - [Starting From A Trained Run](#starting-from-a-trained-run)   today "Transfer Between Runs" + pretraining pointer
  - [Evaluating](#evaluating)                            evaluate_checkpoint, comparing launches, metrics document link
  - [Deployment](#deployment)                            export, Triton, link to the deployment guide
  - [Docker Image](#docker-image)
  - [Documentation Map](#documentation-map)              one table: need → document (references, workflows, tutorials, tests, repository structure, metrics, deployment guide)
  ```
  Configurations, Run Names, MLflow Tracking, Clustering, Tests, Repository Structure and Tutorials stop being
  top-level sections: each becomes one row of the Documentation Map (or one sentence inside Training), since a
  newcomer does not need them to go from clone to served model. Moving Average Of The Weights folds into
  "Following A Run" (it only changes what the reports show). About 16 lines.
- **Illustrate**: none; the outline itself is the journey. `README/scripts_and_requirements.png` (already in
  Prerequisites) is organised Data / Train / Evaluate / Compare / Deploy and doubles as the visual journey.
- **Estimated length after**: 16 lines.

#### Description (README.md:55-88, 34 lines)
- **Keep in README**: the opening sentence (README.md:56, shortened to its first half: nine tasks plus SSL, one
  backbone design, shared training loop / configuration format / report / export), the task table
  (README.md:58-69) and `README/architectures.png` (README.md:71-73). About 18 lines.
- **Cut**: the "Model" column of the task table (README.md:58-69). `architectures.png` draws the same models one
  line below, and the column holds the longest cells (CMP row, README.md:67). Without it, the table fits in the
  screen width.
- **Move**:
  - README.md:75-79 (`base_trainer.py`, `task_registry.py` and `README/shared_and_task_code.png`) is for
    someone changing the code, not running it. Move it to the Repository Structure content (wherever that section
    goes, e.g. a `documentation/reference/code_structure.md` or the Repository Structure section if it stays).
    Pointer in the Documentation Map: "Changing what a task computes → Repository structure".
  - README.md:81-87 (the documentation list: 7 long lines, the pretraining line alone is 60 words) becomes the
    "Documentation Map" table at the end of the README: columns "Should you need to ..." | "Read". One row per
    document, at most 12 words per row. The Description keeps one sentence: "This README takes you from
    clone to served model; the [Documentation Map](#documentation-map) lists where every detail lives."
- **Illustrate**: reuse `tasks_overview.png` + `architectures.png`. No new figure.
- **Estimated length after**: 18 lines.

#### Prerequisites: Package Requirements (README.md:91-121, 31 lines)
- **Keep in README**: the requirement sentence (README.md:92), the uv installer block (README.md:96-99), and
  `uv sync` + `uv run python --version` (README.md:110-113). One sentence on how scripts take their arguments plus
  `README/scripts_and_requirements.png`, whose legend already says the same as README.md:117. About 15 lines.
- **Cut**:
  - README.md:101-106, the `.python-version` explanation and the `uv python install` / `uv python pin` block.
    `pyproject.toml` sets no `python-downloads` option, so uv keeps its default (`automatic`) and `uv sync` itself
    downloads the Python 3.13 that `.python-version` pins. Shorten to half a sentence: "`uv sync` installs the
    Python 3.13 pinned by `.python-version` and the versions of `uv.lock`." (This should be checked once on a
    machine without Python 3.13 before the block is deleted.)
  - README.md:117: the paragraph repeats the figure's legend (blue = `--configuration_path`, orange = `--task` and
    `--output_folder`, green = main block variables). Keep a single sentence: "Every script runs as a module
    from the repository root; the figure shows which ones take arguments."
- **Move**: README.md:115 (the `dev` and `datasets` dependency groups left out of the Docker image) goes to the
  Docker Image section, the only place where it matters.
- **Illustrate**: keep `scripts_and_requirements.png`. It is already the right figure.
- **Estimated length after**: 15 lines.

#### Prerequisites: Data Requirements (README.md:123-130, 8 lines)
- **Keep in README**: one sentence and one figure. 5 lines.
- **Cut / merge**: `README/data_folder_layout.png` (README.md:127) and `README/tasks_dataset_layout.png` (Tasks,
  README.md:310-312) show the same three facts twice: the split folder tree, `label_mapping.json` turning
  categories into classes, and classes from 0 versus from 1 with the background. Keep ONE figure in the README, in
  "Your Own Data". `data_folder_layout.png` is the better one: it links each folder to its configuration key and
  has the labels.json sample per task family. Optionally add the "merged / left out" arrows of
  `tasks_dataset_layout.png` to it, then delete `tasks_dataset_layout.png`.
  Data Requirements then shrinks to: "Each time you train on your own data, every split needs `data/`
  and a `labels.json`, and the dataset a `label_mapping.json` ([Your Own Data](#your-own-data)); the synthetic
  datasets of the Quick Start write all of them."
- **Move**: README.md:130 (the pretraining pointer) goes to the "Starting From A Trained Run" section and the
  Documentation Map, since pretraining is not a prerequisite of a first run.
- **Alternative**: drop the Data Requirements subsection entirely and let Prerequisites be packages only. I
  recommend that: data is the next step of the journey (Quick Start / Your Own Data), not a prerequisite.
- **Estimated length after**: 0 lines (recommended) or 3 lines.

#### Quick Start With Synthetic Data (README.md:132-162, 31 lines)
- **Keep in README**: the scenario sentence, three short steps, the two-command block (README.md:138-141), the
  task → `--task` → configuration table, and `README/synthetic_datasets.gif`. About 20 lines.
- **Cut**:
  - In step 1 (README.md:134), the description of the console panels ("A panel describes each split ... closing
    panel ...") duplicates tasks_reference.md:46 (Console bullet). Also cut "the image size, split sizes and shapes
    per image of every task staying in its main block", already in tasks_reference.md:21. Step 1 becomes "Generate
    the dataset with `--task` and `--output_folder` (existing splits are overwritten)."
  - README.md:157 (how the shapes are drawn: sizes, octaves, polygon minimum sizes, textured backgrounds, low
    contrast, 85 % visibility) duplicates tasks_reference.md:37-40 almost word for word. Replace it with a link in the
    GIF caption: "... how the shapes are drawn: [Synthetic Datasets](documentation/reference/tasks_reference.md#synthetic-datasets)".
  - In the table: the "Synthetic dataset" column duplicates the Content column of tasks_reference.md:23-36. Keep
    only Task | `--task` | Configuration. The GIF shows what the datasets look like, and the content details live in
    the reference. The DET row's note about the 1024 pixel benchmark images (README.md:147) is benchmark history
    that already sits in tasks_reference.md:28, so cut it from the README.
- **Move**: nothing new (everything cut already exists in tasks_reference.md).
- **Illustrate**: a new figure replaces step 2 (README.md:135, the longest step: which five keys to edit, absolute
  paths, `~` not expanded). Proposed figure `README/quick_start_paths.png`, "What to point at the generated
  dataset". Left: the folder tree written by `create_synthetic_dataset.py --output_folder=$HOME/Data/.../synthetic_shapes`.
  Right: the first lines of `classification/configuration_synthetic_shapes.yaml` rendered as code, with
  `experiment_folder`, `train_data_folder`, `validation_data_folder`, `test_data_folder` and
  `Data.label_mapping.mapping_file_path` highlighted and arrows to their folder or file. A red note says
  "absolute paths: `~` and `$HOME` are not expanded". Drawn with Pillow/matplotlib from the real YAML file and a
  real generated tree. `data_folder_layout.png` already draws the key arrows on the folder side, so an alternative
  is to reuse it here and drop it from Your Own Data. I recommend the new figure, because it shows the YAML side the
  newcomer actually edits.
- **Estimated length after**: 20 lines.

#### Training: Launching A Run (README.md:166-178, 13 lines)
- **Keep in README**: the scenario sentence (README.md:167), the command (README.md:169-171) and
  `README/launch_flow.png` with its caption. 9 lines.
- **Cut**: README.md:173 (the three passes of the configuration check: removed settings, unread keys, values).
  It duplicates configuration_reference.md:16, the README Configurations section (README.md ~855-858, with
  `configuration_rejected.png`) and training_reference.md:20, and `launch_flow.png` already draws "Configuration
  check → Stops listing every problem". Keep half a sentence: "the configuration is checked first and every
  problem is listed ([configuration reference](documentation/reference/configuration_reference.md#configurations))."
- **Move**: if the README "Configurations" section dissolves into the Documentation Map (see TOC), its one useful
  newcomer fact, the `check_configuration` command, goes here as one more line under the training command.
- **Illustrate**: keep `launch_flow.png`. It explains the run name → run folder → new run / resume logic in one
  glance.
- **Estimated length after**: 10 lines.

#### Training: Run Folder (README.md:180-208, 29 lines)
- **Keep in README**: the scenario sentence, a reduced tree of 6 entries (`training_configuration.yaml`,
  `Weights/`, `Tensorboard/`, `Results.txt`, `evaluation_curves.png`, `Evaluation/`), the schedule sentence
  (README.md:204) and `README/run_folder_schedule.png`. About 14 lines.
- **Cut**:
  - The full tree (README.md:183-198) is a subset of the fuller tree at training_reference.md:26-61. Two trees
    to maintain means they already differ (the README lacks `anchor_recall.csv`, `*_until_iteration_<n>`,
    `full-checkpoint`). Keep the 6 entries a newcomer opens and link "every file: [Run Folder](training_reference.md#run-folder)".
  - README.md:200 (11 lines of prose on `evaluation_summary.csv` / `evaluation_curves.png`, warm-up windows,
    TensorBoard float32 back-fill) duplicates training_reference.md:64, which says all of it and more. Cut.
  - README.md:202 (Input_Samples) duplicates training_reference.md:84-90. Cut, and keep one tree comment:
    "`Input_Samples/  # first 10 samples of every split as the network sees them`".
- **Move**: nothing new. The reference already holds both paragraphs.
- **Illustrate**: new figure for training_reference.md (not the README), `README/evaluation_curves_annotated.png`:
  the real `evaluation_curves.png` of the README classifier run (35-class synthetic shapes, iteration 2000) with
  call-outs for the trained and EMA curves, the ringed best iterations, the orange EMA window and the grey
  configured window. It replaces most of the 300-word training_reference.md:64 bullet. Optional in the README:
  show the real `evaluation_curves.png` thumbnail instead of describing it, since it is the first file a user opens.
- **Estimated length after**: 14 lines.

#### Training: Stopping And Resuming (README.md:210-222, 13 lines)
- **Keep in README**: README.md:211 (Ctrl+C saves, the same configuration resumes) and `README/resume_rules.png`.
  4 lines.
- **Cut**: the bullets README.md:217-220. `resume_rules.png` draws each of them: "number_iterations is the total of
  the run", the two boxes of settings with and without a run name token, "labels.json changed →
  allow_data_change_on_resume", "Start over → run_name_suffix / experiment_folder / move the folder".
  training_reference.md:126-141 also has them in text.
- **Move**: README.md:221-222 (failing iterations, NaN losses, 5 failures in a row, FAILED status, dropped batches)
  is about failure handling, not resuming. training_reference.md:121-122 already says the same, so cut it from the README
  with a pointer: "What stops a training, and what is skipped: [Stopping And Resuming](training_reference.md#stopping-and-resuming)."
- **Illustrate**: `resume_rules.png` already does the job.
- **Estimated length after**: 5 lines.

#### Training: Console And TensorBoard (README.md:224-253, 30 lines)
- **Keep in README**: one scenario sentence, `README/evaluation_report.png` (what a result looks like), the
  TensorBoard command (README.md:246-248) and `README/tensorboard.png` with a one-line caption. About 12 lines.
  Rename to "Following A Run" and add one sentence + link for MLflow (today a separate top-level section,
  README.md:889-909).
- **Cut**:
  - README.md:237, the formatting of rates ("94.03 %", percentage points in reports, "+0.52 %" in summaries,
    files keep 0-1). It duplicates training_reference.md:107 and :115 and configuration_reference.md:110. Cut.
  - README.md:244: the list of TensorBoard folders duplicates the table at training_reference.md:212-216. Keep
    only the command.
  - Caption README.md:253 (smoothing 0 vs 0.6, the four evaluations, 81.65 %): shorten to "TensorBoard of the
    synthetic shapes classifier (smoothing set to 0)".
- **Move**: `README/experiment_launch_information.png` (README.md:227-229) and `README/interval_summary.png`
  (README.md:233-235) with their two sentences (README.md:225, :231) go to training_reference.md "Console Output"
  (training_reference.md:96), which describes exactly these two console blocks in text with no image. README
  pointer: "What every console block means: [Console Output](training_reference.md#console-output)."
- **Illustrate**: reuse `evaluation_report.png` and `tensorboard.png`. The moved screenshots give the reference its
  first images.
- **Estimated length after**: 12 lines.

#### Training: Augmentation And Diagnosis Modes (README.md:255-291, 37 lines)
- **Keep in README**: the scenario sentence (README.md:256), a 2-sentence summary of augmentation
  (`Data.data_augmentation`, on the GPU, geometric types move the targets) with `README/augmentation_types.jpg`, and
  a 3-row table of the diagnosis modes:

  | Mode | Use it when | Details |
  |---|---|---|
  | `view_input_images` | before a long run: see what the network receives | [Viewing The Input Images](training_reference.md#viewing-the-input-images) |
  | `compute_anchor_coverage` (DET, INS) | after changing image size, levels or anchors | [Checking The Anchor Coverage](training_reference.md#checking-the-anchor-coverage) |
  | `benchmark_speed` | before training an architecture: its training and serving speed | [Benchmarking The Speed](training_reference.md#benchmarking-the-speed) |

  plus "These modes exit without training and create no run folder" (README.md:291). One of the two input viewer
  images stays: `augmented_detection_samples.gif` (shows boxes following the augmentations, which is the point).
  Rename the section "Checking Before A Long Run". About 18 lines.
- **Cut**:
  - README.md:285-289 (speed benchmark, 5 dense lines: formats, batch search up to 2048, 40 GB / 2 GB margins,
    row contents) duplicates training_reference.md:200-208, which is more complete. Cut (the table row replaces it).
  - README.md:258's strength-table names (`SUPERVISED_AUGMENTATION_STRENGTHS`, refused types) duplicate
    training_reference.md:144-148. Keep only the visualize script command as a link.
- **Move**:
  - README.md:276-283 (anchor coverage: the mode, how to read the `All Levels` column, which scales to change) and
    `README/anchor_coverage.png` with its long caption go to training_reference.md "Checking The Anchor Coverage"
    (training_reference.md:194-198). The reference lacks both the figure and the reading guidance of README.md:283,
    so this is a real move, not a cut.
  - `README/input_viewer.png` (README.md:267-269) goes to training_reference.md "Viewing The Input Images"
    (training_reference.md:188), which has no image.
- **Illustrate**: keep `augmentation_types.jpg` and `augmented_detection_samples.gif`. No new figure needed; the table
  replaces 15 lines of prose.
- **Estimated length after**: 18 lines.

#### Training: Moving Average Of The Weights (README.md:293-306, 14 lines)
- **Keep in README**: one sentence (what `Optimization.exponential_moving_average` does, that every evaluation
  scores both weights as "EMA Weights" / `Test_EMA`, and that it can be switched on at resume) and
  `README/moving_average_of_the_weights.png` with a one-line caption ("on the detection benchmark the averaged
  weights lead at all 30 evaluations, +8 points at 15,000"). It is the figure that convinces the reader to turn it on.
  About 5 lines, inside "Following A Run" rather than as its own subsection.
- **Cut**: README.md:296 (the decay read as a window, warm-up formula, 0.9998 reached after 45,000 iterations)
  duplicates training_reference.md:221-222 almost exactly. The formula of README.md:294 duplicates
  training_reference.md:220.
- **Move**: `README/moving_average_window.png` with its 7-line caption (README.md:298-301) goes to
  training_reference.md "Moving Average Of The Weights" (training_reference.md:218), next to the window text it
  illustrates. Pointer: "How to choose the decay: [Moving Average Of The Weights](training_reference.md#moving-average-of-the-weights)."
- **Illustrate**: both figures already exist. Only their placement changes.
- **Estimated length after**: 5 lines.

---

### Cross-document notes
- Inbound links to keep working: training_reference.md:3 links `README.md#training`, tasks_reference.md:21 links
  `README.md#quick-start-with-synthetic-data`, pretraining_reference.md links `README.md#data-requirements` (which
  goes away under the recommended option, so repoint it to `#your-own-data` or `#prerequisites`). The tutorial
  storyboards and the deck should be grepped for README anchors before renaming "Console And TensorBoard" and
  "Augmentation And Diagnosis Modes".
- training_reference.md then gains 6 images (launch information, interval summary, input viewer, anchor coverage,
  moving average window, optionally the annotated evaluation curves). That matches the user's wish for references
  that lean on pictures.
- The README "Configurations" section (README.md:850-870) duplicates configuration_reference.md:15-16 and :96-106
  in the same way. That is part 2's range, noted here because it affects the TOC proposal.

### Summary of the part

| Section | Lines now | Lines after | Main action |
|---|---|---|---|
| Title and overview figure | 7 | 5 | shorten the dated caption |
| Table Of Content | 47 | 16 | top-level journey outline, sub-entries of Deployment/Metrics leave with their documents |
| Description | 34 | 18 | drop the Model column (architectures.png shows it), move the shared-code figure, documentation list → Documentation Map table |
| Package Requirements | 31 | 15 | drop the `uv python install/pin` block (uv sync downloads the pinned Python), figure replaces the arguments paragraph |
| Data Requirements | 8 | 0 | fold into Your Own Data; merge the two duplicate layout figures |
| Quick Start With Synthetic Data | 31 | 20 | cut the shape-drawing paragraph and Content column (in tasks_reference), new "what to point where" figure |
| Launching A Run | 13 | 10 | cut the three-pass check prose (launch_flow.png + configuration reference) |
| Run Folder | 29 | 14 | 6-entry tree, cut 2 paragraphs that duplicate training_reference |
| Stopping And Resuming | 13 | 5 | resume_rules.png replaces the bullets; failure handling only in the reference |
| Console And TensorBoard | 30 | 12 | keep report + TensorBoard, move 2 console screenshots to training_reference |
| Augmentation And Diagnosis Modes | 37 | 18 | 3-row diagnosis table; anchor coverage figure + guidance move to the reference |
| Moving Average Of The Weights | 14 | 5 | keep the benefit figure, move the window figure and the decay maths |
| **Total** | **304 (+3 blank/heading)** | **~138** | |

## Part 2: Tasks And Label Files (README.md:308-730)

Range read: README.md:308-730 (423 lines, of which "Label Files" is README.md:352-730, 379 lines). Also read in full:
documentation/reference/tasks_reference.md (441 lines), the "Read This First" / "Task At A Glance" / "Pitfalls"
parts of the ten task workflows, README.md:1-163 (Description, Data Requirements, Quick Start) and README.md:1580-1597
(Metrics Explained headline table), and the images README/tasks_overview.png, README/tasks_dataset_layout.png and
README/data_folder_layout.png.

### Main finding

Almost everything in README.md:308-730 already exists in tasks_reference.md, in a terser form:

| README text | Same fact in tasks_reference.md |
|---|---|
| README.md:309 (folder per split, `labels.json`, `label_mapping.json`, classes from 0 or 1) | tasks_reference.md:48-75 (Common Dataset Layout) |
| README.md:355-376 (mapping file, merged classes, left-out categories, `mode: "binary"`) | tasks_reference.md:66-86 (bullets plus the "category missing from the mapping" table) |
| README.md:378-405 (CLS samples) | tasks_reference.md:91-98 |
| README.md:407-427 (multi-label `category_ids`) | tasks_reference.md:89-91 |
| README.md:429-453 (SEG masks, 0 background, 255 ignored) | tasks_reference.md:299-306 |
| README.md:455-500 (DET COCO, `bbox`, unread keys) | tasks_reference.md:127-135 |
| README.md:502-524 (INS polygons or RLE) | tasks_reference.md:186-192 |
| README.md:526-592 (KPT triples, `v` flags, skeleton from 1) | tasks_reference.md:236-245 |
| README.md:594-624 (M2O `sequences`, `frames` in order) | tasks_reference.md:318-330 |
| README.md:626-657 (M2M `frame_category_ids`) | tasks_reference.md:344-349 |
| README.md:659-692 (SEL unselected category, `frames_per_class`) | tasks_reference.md:400 |
| README.md:694-730 (CMP `pairs`, `before` / `after`) | tasks_reference.md:362-372 |
| README.md:329 (KPT flips keep the names) | tasks_reference.md:255, keypoint_detection_workflow.md:106 (Pitfalls) |
| README.md:330 (sequences fed whole, padding, which temporal modules follow the order) | tasks_reference.md:320-322 |
| README.md:331, 338-350 (M2O motion needs `flatten`) | tasks_reference.md:336, sequence_classification_workflow.md:97 |
| README.md:315-326, Evaluation column | README.md:1585-1595 (headline metric table of Metrics Explained) and the Evaluation block of every task in tasks_reference.md |

What the README has and the reference lacks is the teaching: the commented `jsonc` examples (README.md:358-730) show a
merge, a left-out category, a subfolder kept in `file_name`, identifier gaps, a hidden versus a not labelled keypoint,
sequences of different lengths. That teaching should not be lost, but it belongs with the reference (one place for the
format), and most of it is better told by a picture of a real sample next to its record than by 370 lines of JSON.

Recommended split of responsibilities:
- **README** says that a dataset is three split folders plus one mapping file, shows it in two figures, and links each
  task to its reference section, workflow and tutorial series. No JSON in the README.
- **tasks_reference.md** owns the file format: the mapping rules, the left-out table, and for every task its label
  file panel (figure) plus one complete commented example (the current README `jsonc` blocks, moved, trimmed to one
  record per list).

---

#### Tasks (README.md:308-350, 43 lines)

- **Keep in README** (about 20 lines):
  - One scenario sentence replacing README.md:309, e.g. "Say you bring your own data: every task reads three split
    folders, each with its images in `data/` and a `labels.json`, and one `label_mapping.json` giving the class of every
    category; [Label Files](#label-files) shows the record of every task." The "from 1 for DET, INS, KPT and SEG" detail
    leaves this sentence: the figure of Label Files shows it.
  - The per-task table (README.md:315-326), slimmed to an index (12 lines). Proposed columns:
    `Task | labels.json holds | Converter for real data | Example configuration | Details`, the Details cell keeping
    `reference · workflow` and adding `tutorial` (the series number of `tutorials/storyboards/`, e.g. `05`, since the
    videos live outside the repository; optional). Example rows:
    - `CLS | samples, category_id (category_ids in multi-label) | class_folder_manifests.py, imagenet_download.py | classification/configuration_example.yaml | reference · workflow · 02`
    - `KPT | COCO, keypoints and named points per category | data_retriever.py (COCO 2017) | keypoint_detection/configuration_example.yaml | reference · workflow · 05`
    - `CMP | pairs, before and after frame lists | pair_list_manifests.py | ... | ...`
    "Each task starts from the `configuration_example.yaml` of its folder" (README.md:309) becomes this column.
  - One pointer line for the sequence tasks: "The sequence tasks (M2O, M2M, CMP, SEL) read every sequence whole and pad
    the batches; a class given by a motion needs `ClassificationHead.global_pooling: flatten`
    ([why](documentation/reference/tasks_reference.md#sequence-classification-m2o))." The flatten pitfall is the one
    fact worth a line here: a wrong pooling leaves M2O at chance without any error.
- **Cut**:
  - Table column "Own sections" (README.md:315-326): it lists configuration sections, which the configuration reference
    and the Configuration block of every task in tasks_reference.md already list; a newcomer opens the example
    configuration instead.
  - Table column "Evaluation": duplicated by the headline metric table of Metrics Explained (README.md:1585-1595) and by
    the Evaluation block of every task in tasks_reference.md.
  - Table column "Labels" in its current prose form: replaced by the short "labels.json holds" column above.
  - README.md:328-331 ("A few behaviours are easy to miss"): the KPT flip line is in tasks_reference.md:255 and in
    the Pitfalls of keypoint_detection_workflow.md:106; the padding and temporal-order line is tasks_reference.md:320-322;
    the flatten line becomes the single pointer line above.
- **Move**:
  - README.md:333-336 (`sequence_padded_batch.gif` and its caption) to tasks_reference.md "Sequence Tasks (M2O, M2M,
    CMP, SEL)", next to the bullet on padding (tasks_reference.md:322), which it illustrates exactly.
  - README.md:338-350 (the flatten explanation, `sequence_pooling_mechanism.gif`, `sequence_pooling_similarity.png` and
    their captions, 13 lines) to tasks_reference.md "Sequence Classification (M2O)", as a short "Why flatten" paragraph
    after its Configuration bullet (tasks_reference.md:336). This is the best illustrated explanation of the section
    and the reference currently states the rule without the reason. Also update the docstring of
    `documentation/presentation/figures/draw_pooling_similarity_animation.py:2` ("the animation of the Tasks section of
    the README") to name its new place.
  - Add one Pitfalls bullet to sequence_classification_workflow.md (it has no Pitfalls section, unlike the other
    workflows) pointing to the same "Why flatten" paragraph.
- **Illustrate**:
  - `README/tasks_dataset_layout.png` (README.md:311): its left panel (the folder tree) repeats
    `README/data_folder_layout.png` of Data Requirements (README.md:127). Redraw it as `README/label_mapping.png` with
    only its middle panel (categories to classes, merged, left out) and its right panel (classes from 0 versus 1 with
    the background), and move it under Label Files. Coordination with part 1: Data Requirements keeps
    `data_folder_layout.png` but its bottom panel "Samples of labels.json by task" is superseded by the label file
    atlas below, so it can be cropped to its two top panels.
  - No new figure for the per-task index: `README/tasks_overview.png` (README.md:4) already shows every task once.
- **Coordination note** (outside this range, for the parts owning README.md:55-163, 871-888, 1276-1310 and 1580-1597):
  the README holds six per-task tables (Description README.md:57-68, Quick Start README.md:143-155, Tasks
  README.md:315-326, Run Names README.md:881-885, Export README.md:1289, Metrics Explained README.md:1585-1595).
  Recommended: the Description table becomes the single per-task index (code, task, input, output, plus the Details
  links of this table), and the Tasks table keeps only what a person bringing data needs (label file shape, converter,
  example configuration). If part 1 prefers to keep the Description table short, the Details links stay here; either
  way, the links live in exactly one table.
- **Estimated length after**: 20 lines.

#### Label Files (README.md:352-730, 379 lines)

- **Keep in README** (about 12 lines):
  - One scenario sentence (README.md:353 shortened): "Before a first launch on your own data, write one `labels.json`
    per split and one `label_mapping.json` for the dataset; `class_folder_manifests.py` (one folder per class) and
    `pair_list_manifests.py` (CMP pair lists) write both for you."
  - The redrawn `README/label_mapping.png` (see Tasks above) with a one-line caption.
  - The new label file atlas (see Illustrate) with a one-line caption.
  - One pointer line: "The complete commented record of every task, the visibility flags of the keypoints and what a
    category left out of the mapping does are in the
    [tasks reference](documentation/reference/tasks_reference.md#common-dataset-layout)."
- **Cut**:
  - README.md:376 (what a left-out category does, task by task): it is the table of tasks_reference.md:77-86, word for
    word in substance.
  - The "the `//` comments only explain the examples" sentence (README.md:353): it goes with the examples to the
    reference.
  - Nothing else is deleted outright: the examples move.
- **Move** (to tasks_reference.md; the other agent restructures that file, this is what the README hands over):
  - README.md:355-374 (the two commented mapping examples: a merge into `square+rectangle` and a left-out star, then
    the same with classes from 1 and a category mapped to 0): to "Common Dataset Layout", replacing the bare
    `{"1": 0, "2": 1, "3": 1}` of tasks_reference.md:68-70. They show the three rules (merge, leave out, offset) in
    11 lines and are clearer than the four bullets there (tasks_reference.md:72-75), which can shrink to one line each.
  - README.md:378-730 (the ten commented `jsonc` blocks): one block per task section of tasks_reference.md, replacing
    the one-line JSON examples there (tasks_reference.md:93-98, 129-135, 188-190, 238-243, 301-306, 320-330, 340-342,
    366-372). Trim each to one record per list (the README shows two or three entries per list, e.g. three CLS
    samples README.md:385-403, three DET annotations README.md:479-497, two KPT categories README.md:567-591), which
    halves them. Keep the facts that only the README states:
    - CLS: a subfolder of `data/` stays in `file_name` (README.md:401-402);
    - DET: category identifiers may have gaps (README.md:492);
    - KPT: what each `v` flag means for training and scoring (hidden is learned and scored like visible, 0 keeps its
      place as `0, 0, 0` and is neither learned nor scored; README.md:528-530), which tasks_reference.md:236 reduces
      to "0 (not labelled), 1 (hidden) or 2 (visible)";
    - M2O and CMP: sequences and members may differ in length (README.md:594, 616, 708).
    Optionally wrap each full example in a `<details><summary>Full labels.json</summary>` block so the reference
    sections stay short and scannable, with the atlas panel visible above it.
  - SEL (README.md:659-692): the reference already has the rule (tasks_reference.md:400); its commented example moves
    under "Frame Selection (SEL)", **Data**, where the reference has no example today.
  - SSL: nothing to move (the README only says "the layout of CLS"); the reference already has the empty mapping `{}`
    and the `groups` example (tasks_reference.md:417-424).
- **Illustrate**:
  - **New figure `README/label_files_atlas.png`, "One sample and its record, task by task"** (replaces README.md:378-730).
    A grid of ten panels (CLS, CLS multi-label, SEG, DET, INS, KPT on the first two rows; M2O, M2M, SEL, CMP on the
    last two, the sequence panels being twice as wide). Each panel shows on the left a real synthetic sample drawn with
    its labels and on the right the matching record of its `labels.json`, pretty-printed in monospace and trimmed to
    the fields the loader reads, with colour links between the two:
    - CLS: the image, the sample record, and the `label_mapping.json` line taking its `category_id` to the class index
      (the three-file chain of `data_folder_layout.png`, one panel wide).
    - CLS multi-label: a counted shapes image, each distinct shape outlined in the colour of its identifier in
      `category_ids`.
    - SEG: image, then its `masks/` PNG in a palette with the raw pixel values written on the regions (`0` background,
      `2` square, `255` ignored band along a border), then the sample record with `mask_file_name`.
    - DET: the image with every box drawn, the four numbers of each `bbox` printed in the colour of its box, a dimension
      arrow on one box marking `x, y` at the top-left corner and `width, height`.
    - INS: the same image with the RLE masks filled; the `segmentation` shown as the truncated `counts` string (the
      synthetic data writes compressed RLE) and, in an inset, one polygon with its vertices numbered in the order of
      the flat `[x1, y1, x2, y2, ...]` list.
    - KPT: a square and a triangle with each keypoint name written next to its point, `v = 2` as a filled dot,
      `v = 1` as a ring under an overlapping shape, `v = 0` as a grey cross at the image border with `0, 0, 0` in the
      record; the `skeleton` drawn as lines labelled with their 1-based pairs.
    - M2O: a filmstrip of the frames of one sequence with the `frames` list beneath (one arrow per path) and the
      `category_id` as a banner over the whole strip.
    - M2M: the same filmstrip, the `frame_category_ids` entry printed under each frame, the frame where a shape appears
      highlighted.
    - SEL: a filmstrip with the frames to select framed in the colour of their wanted class and the unselected frames
      dimmed, the `frame_category_ids` under each frame, `frames_per_class: 1` written beside it.
    - CMP: two filmstrips (before over after) of different lengths, the `before` and `after` lists beside each, the
      pair `category_id` as the banner.
    Data source: a figure script (e.g. `documentation/presentation/figures/draw_label_files_atlas.py`, next to the
    existing figure scripts) that calls the generator functions of
    `vision_training/data/synthetic/create_synthetic_dataset.py` with a fixed seed and tiny split sizes into a
    temporary folder, then draws from the `labels.json` files actually written, so the records shown cannot drift from
    the loader format. Same script with `--panel <task>` writes one PNG per task (`README/label_file_<task>.png`), so
    every task section of tasks_reference.md shows its own panel above its full example, and the README shows the
    assembled atlas. What the reader learns in one glance: which keys a task needs and what each number means on the
    image.
  - Alternative (real trade-off): two images instead of one, `label_files_images.png` (6 panels) and
    `label_files_sequences.png` (4 panels), which stay legible at the 100 % width GitHub renders on a laptop; one
    atlas of ten panels risks small text. Recommended: the two-image split.
  - `README/label_mapping.png` (the redrawn middle and right panels of `tasks_dataset_layout.png`, see Tasks).
- **Estimated length after**: 12 lines.

---

### Summary of the part

| Section | Lines now | Lines after | Main action |
|---|---|---|---|
| Tasks (README.md:308-350) | 43 | 20 | Table slimmed to an index (label shape, converter, example configuration, reference · workflow · tutorial); "easy to miss" bullets cut (all in the reference); flatten explanation and its 2 figures plus the padding GIF moved to tasks_reference.md M2O and Sequence Tasks |
| Label Files (README.md:352-730) | 379 | 12 | Ten commented `jsonc` blocks and the two mapping examples moved to the task sections of tasks_reference.md (trimmed to one record per list); replaced in the README by a label file atlas (real synthetic sample next to its record, per task) and a mapping-only figure |
| **Total** | **422** | **32** | |

Hand-over list for the agent rewriting tasks_reference.md: the two mapping examples (README.md:358-374) into Common
Dataset Layout; one commented example per task from README.md:381-730 with the four README-only facts (subfolder in
`file_name`, identifier gaps, `v` flag semantics, different lengths); `sequence_padded_batch.gif` into Sequence Tasks;
the flatten paragraph with `sequence_pooling_mechanism.gif` and `sequence_pooling_similarity.png` into M2O; the per-task
atlas panels above each example. The README anchor `#label-files` is linked only from README.md:22, 124 and 309, so the
section keeps its heading and no outside link breaks; tasks_reference.md:3 links `README.md#tasks`, which stays.

## Part 3: Transfer Between Runs To Tutorials (README.md:731-935)

Range: Transfer Between Runs, Evaluating Checkpoints, Comparing Launches, Clustering Image Embeddings, Configurations, Run Names, MLflow Tracking, Tutorials (205 lines, 8 top-level sections).

Findings across the range:
- Every section already has a reference home that repeats most of its prose: `training_reference.md#transfer-between-runs` (training_reference.md:238-291), `tools_reference.md#evaluating-checkpoints` / `#comparing-launches` / `#clustering-image-embeddings` (tools_reference.md:27-198), `configuration_reference.md#configurations` / `#run-names` / `#mlflow-tracking` (configuration_reference.md:15-295), `tutorials/README.md` (build commands at tutorials/README.md:58-75). The README text is a second, slightly different copy, and has started to drift: the transfer table in README.md:734-743 says "classification and box regression layers" for `Detector` and merges `HRNet` and `KeypointHead` in one row, while training_reference.md:241-250 says "class layers" and splits them.
- The references hold no picture at all (`grep README/` gives 0 in tools, training and configuration references). The figures that leave the README can go with their text (`../../README/<file>.png`), so the moved sections stay illustrated.
- Four of the eight sections are not on the clone → train → evaluate → export path of a newcomer: research report figures, scoring a large split, comparing benchmark launches, clustering embeddings. They become one "Other tools" table of links.
- Three sections describe the training itself (configuration files, run names, MLflow) and the transfer is a training setting. They fit as subsections of `## Training`, next to text that already half-says the same thing (README.md:167 copies `configuration_example.yaml`, README.md:173 the check passes, README.md:181 links Run Names). The anchors stay valid when the heading text is kept (`#transfer-between-runs`, `#mlflow-tracking` and so on work at any heading level).
- Coordination with the part covering `## Training` (README.md:164-307): the merges below add about 35 lines to that section. Whoever writes that part should know about them.

#### Transfer Between Runs (README.md:731-765, 35 lines)
- **Keep in README**: the scenario sentence (README.md:732), the figure `README/transfer_between_runs.png` (README.md:745-747), the YAML block (README.md:749-762), and two sentences in place of the paragraph at README.md:764: "The settings that define the layers must match those of the source run (this is checked at launch). Heads load without their class layers by default (`class_layer` / `class_layers: False`), so a head trained on other classes can serve any run of its task. More classes, shorter or deeper backbones, the `Neck` limit: [training reference](...)". Make the section a `###` subsection of `## Training`, after Moving Average Of The Weights, since it is set in the training configuration. The anchor stays `#transfer-between-runs`, which training_reference.md:3 and pretraining_reference.md link to. About 20 lines.
- **Cut**: the component table (README.md:734-743). training_reference.md:241-250 has the same table, and the two copies already differ (see above). The first column of the YAML already shows that the block sits in a section.
- **Move**: nothing new to write. Every clause of README.md:764 is already in training_reference.md: the settings that must match (252-263), the deeper backbone (265), the averaged weights (267), `class_layer: False` (268), `class_layers: False` (269), `Neck.type: none` (table rows 257-261), adding classes by order or by name (274-291). Replace the paragraph with the pointer above.
- **Illustrate**: add a sixth row to `README/transfer_between_runs.png`, "Other classes, DET heads". It draws `Detector.pretrained` with `class_layers: False`: the convolutions and fully connected layers in blue (loaded), the classification and box regression layers dashed (random), with the caption "heads trained on other classes: everything but the class layers". The `class_layer(s)` rule is the hardest part of the prose to follow, and this row makes it visible. No tracked script draws this figure (`grep transfer_between_runs` finds no `.py` file), so the row has to be redrawn in the same block style (matplotlib/Pillow).
- **Estimated length after**: 20 lines (as a Training subsection).

#### Evaluating Checkpoints (README.md:766-811, 46 lines)
- **Keep in README**: the scenario sentence and the command (README.md:767-771). Shorten 767 to "set `run_name`, `experiment_folder`, `iteration` (`None`, a number or `"all"`) and `test_data_folder` in the main block". Keep the figure `README/checkpoint_evaluation_folder.png` (README.md:775-777) and one sentence: "It writes into `<run>/Checkpoint_Evaluation/` and leaves the training files untouched; settings and outputs: [tools reference](...)". About 12 lines. Then the **Other tools** table (below, about 10 lines) closes the section.
- **Cut**:
  - Most of README.md:773: the folder names, `evaluation_summary.csv` rows, `"all"`, the `FileNotFoundError`, `class_logits.npz` and `save_predictions`. tools_reference.md:31-42 repeats every clause, and the figure already shows `Checkpoint_Evaluation_nms_<criterion>/`, the summary, `"all"`, `class_logits.npz`, `predictions.csv` and `Errors/`.
  - README.md:779 (how to read the confusion matrix, `confusion_cells.csv`, `Errors/<true>-TO-<predicted>/`). The README already explains this in Metrics Explained (README.md:1610-1611, with `README/metrics_classification_confusion_matrix.png` at 1614), and configuration_reference.md:122-131 lists the files. This is a third copy.
- **Move**:
  - `README/evaluation_error_images.png` with its caption (README.md:785-788) → Metrics Explained, Classification, confusion matrix entry, next to the line "a strong off-diagonal cell ... its error images" (README.md:1611), or to wherever Metrics Explained goes in another part. It is the only picture of the `Errors/` folders, so keep it. `README/confusion_matrix.png` (README.md:781-783) → tools_reference.md "Drawing The Figures Of A Research Report", as the report confusion matrix (that caption already says "The confusion matrix of the report is the one shown above"). Otherwise drop it, since Metrics Explained has its own confusion matrix figure.
  - Research report figures (README.md:790-803: paragraph, command, both figures and the long caption) → tools_reference.md#drawing-the-figures-of-a-research-report (tools_reference.md:53-74). That section already lists every output; it gains the command and the two figures `README/report_precision_recall_curves.png` and `README/report_training_curves.png`. Pointer: one row of the Other tools table.
  - Scoring a large split (README.md:805-810: paragraph and command) → tools_reference.md#scoring-a-large-split (tools_reference.md:76-90). The reference has the options table but no command, so the command moves there. Pointer: one row of the table.
- **Illustrate**: the **Other tools** table replaces three sections and two sub-parts with clickable rows. Each row has a scenario cell, the script and a link:

  | Should you need to ... | Script | Details |
  |---|---|---|
  | draw the figures of a research report again | `scripts/evaluation/draw_report_figures.py` | tools reference |
  | score every image of a huge unlabelled split | `scripts/evaluation/score_image_split.py` | tools reference |
  | compare the launches of a hyperparameter stage | `collect_experiment_results.py`, `compare_best_evaluations.py`, `plot_experiment_comparisons.py` | tools reference |
  | pick diverse images among near-identical frames | `scripts/embedding/cluster_image_embeddings.py` | tools reference |
  | choose a logit temperature before export | `scripts/evaluation/serve_temperature_explorer.py` | tools reference |

  The last row is optional; it depends on what the Deployment part keeps. A figure version of this table ("what to do with a trained run": a run folder in the centre with arrows to each tool and its output folder) would be nice at the top of tools_reference.md. In the README, the table is better because it is clickable.
- **Estimated length after**: 22 lines (section and table).

#### Comparing Launches (README.md:812-835, 24 lines)
- **Keep in README**: nothing beyond its row in the Other tools table. These scripts serve the architecture benchmarks (`documentation/architecture_benchmarking`), not a newcomer training a model.
- **Cut**: the three bullets (README.md:813-816). tools_reference.md:92-138 describes each script in more detail.
- **Move**: to tools_reference.md#comparing-launches, which has no command and no figure yet. Move the command block (README.md:822-826), `README/comparing_launches_flow.png` (README.md:818-820, it sums up the three scripts at a glance and fits at the top of the reference section), and the "spread of the launches" sentence with `README/launch_comparison_stage_figure.png` (README.md:828-832) under "Drawing The Comparison Of Every Stage". Update tools_reference.md:3, which links `README.md#comparing-launches`, to the Evaluating Checkpoints section.
- **Illustrate**: the existing figures are enough. Move them, do not redraw them.
- **Estimated length after**: 0 lines (1 table row counted under Evaluating Checkpoints).

#### Clustering Image Embeddings (README.md:836-849, 14 lines)
- **Keep in README**: its row in the Other tools table.
- **Cut**: the summary at README.md:837. tools_reference.md:149-198 says the same (clusters, mosaics, three kept-image folders, stored features reused).
- **Move**: the command (README.md:839-841) and `README/embedding_clusters.png` with its caption (README.md:843-846) → top of tools_reference.md#clustering-image-embeddings. The reference has no command, and the figure is the best one-glance explanation of what the tool gives. Update the link at tools_reference.md:3.
- **Illustrate**: no new figure. Alternative only if the user wants one: a strip showing `kept_images_by_clustering/`, `kept_images_at_random/` and `kept_images_evenly_spaced_per_group/` side by side on the same 720 shapes. It would replace the bullet list of the three kept folders (tools_reference.md:184-188), which is part 3's reference, not the README.
- **Estimated length after**: 0 lines (1 table row).

#### Configurations (README.md:850-870, 21 lines)
- **Keep in README**: merge into `### Launching A Run`, which already says to copy `configuration_example.yaml` (README.md:167) and already lists the three check passes (README.md:173). Keep, once: `README/configuration_sections.png` (README.md:853-855; its top row is the same three-pass check, so README.md:173 can shrink to one sentence), and a two-line "check without training" with the command (README.md:863-867), which the Docker section repeats at README.md:1530 and pipelines need. Keep the heading text or an HTML anchor: README.md:117 and configuration_reference.md:3 link `#configurations`. About 9 lines added to Launching A Run.
- **Cut**:
  - "every task has a `configuration_example.yaml` and a `configuration_synthetic_*.yaml` ... each explained by its comment" (README.md:851). It repeats README.md:167 and configuration_reference.md:16.
  - The image size rule (README.md:857). configuration_reference.md:48 states it with the same example sizes, and it is one value rule among dozens, so the README has no reason to single it out.
  - The details of the exit codes and `--check_paths` (README.md:863). configuration_reference.md:96-106 has them.
- **Move**: `README/configuration_rejected.png` (README.md:859-861) → configuration_reference.md#checking-without-training, which describes the red block in words only ("An invalid file prints the red block ...").
- **Illustrate**: `configuration_sections.png` is enough. Its top row (load → check → trainer, with the red stop box) replaces README.md:173 and the rejected screenshot.
- **Estimated length after**: 9 lines (inside Launching A Run, about 5 lines fewer there too, once README.md:173 shrinks).

#### Run Names (README.md:871-888, 18 lines)
- **Keep in README**: merge into `### Run Folder`, whose first line (README.md:181) already says the folder is the run name built from the settings. Add one sentence: "the same settings resume in the same folder, changed settings start a new one, and `run_name_suffix` starts a separate run". Add `README/run_name_decoded.png` with its caption (README.md:874-877), since it decodes a name token by token and replaces the prose. Keep an anchor `#run-names` (README.md:181 and configuration_reference.md:3 link it). About 6 lines added.
- **Cut**: the separators and allowed characters (README.md:872, configuration_reference.md:161-164 and 250), and the 3-row example table (README.md:879-885). configuration_reference.md:255-265 gives the example of every task, the same three rows among them.
- **Move**: nothing. The reference already has everything.
- **Illustrate**: the existing `run_name_decoded.png`. The README never shows `launch_flow.png` (README.md:175-177, "The run name decides the folder ...") and `run_name_decoded.png` together, so placing them next to each other in Training tells the whole resume-or-new story in two pictures.
- **Estimated length after**: 6 lines (inside Run Folder).

#### MLflow Tracking (README.md:889-909, 21 lines)
- **Keep in README**: turn it into a short `### MLflow Tracking` subsection after `### Console And TensorBoard`, the three monitoring tools side by side. Keep the scenario sentence shortened to "turn on `ExperimentConfiguration.tracking.activated`: every run is a row in `<experiment_folder>/mlflow.db`", the `mlflow ui` command (README.md:892-894), `README/mlflow_runs.png` with its caption (README.md:896-899), and a link to the reference. About 9 lines.
- **Cut**: the clause "its files ... a resumed training continuing its MLflow run" (README.md:890) and all of README.md:908 (TensorBoard keeps the curves, checkpoints are not sent, the metrics of every task). configuration_reference.md:272-275 ("Kept", "Limits") already says this.
- **Move**: the run page paragraph and `README/mlflow_run_page.png` (README.md:901-906) → configuration_reference.md#mlflow-tracking. The reference does **not** list the run tags yet (folder, transferred components, git commit, uncommitted changes, best iteration; `grep` finds none of them there), so the move adds a "Tags" bullet there rather than losing the information. The `mlflow ui` command is also missing from the reference, so add it there as well, or link back to the README.
- **Illustrate**: no new figure. The two screenshots cover it.
- **Estimated length after**: 9 lines.

#### Tutorials (README.md:910-935, 26 lines)
- **Keep in README**: the scenario sentence (README.md:911), shortened to: 13 narrated series (overview, foundations, one per task, deployment); videos written to `~/Data/tutorial_videos`; to rebuild them, see [tutorials/README.md](tutorials/README.md). Keep `README/tutorial_scenes.gif` with its caption (README.md:913-916), the one animated preview of what the videos look like. Also add the tutorials as a line in the documentation map of the Description (README.md:80-86). A reader who prefers to watch should find them before the Training section, not after Run Names. About 8 lines.
- **Cut**: the build command block (README.md:918-928). tutorials/README.md:58-75 has the same commands with more options, and README.md:918 already says that file "describes the build".
- **Move**: the `index.html` sentence and `README/tutorial_index_page.png` (README.md:930-934) → tutorials/README.md "Rebuilding The Videos" (it mentions `index.html` at tutorials/README.md:101 with no picture). The image path from there is `../README/tutorial_index_page.png`.
- **Illustrate**: the existing GIF is enough.
- **Estimated length after**: 8 lines.

#### Summary of the part

| Section | Lines now | Lines after | Main action |
|---|---|---|---|
| Transfer Between Runs | 35 | 20 | Becomes a Training subsection; component table and long rules paragraph cut (copies of training_reference.md:241-291); add a "heads without class layers" row to the figure |
| Evaluating Checkpoints | 46 | 22 | Keep command + folder figure; report figures and large-split scoring move to tools_reference with their commands and images; confusion-matrix reading cut (Metrics Explained has it); add the Other tools table |
| Comparing Launches | 24 | 0 | Move commands + 2 figures to tools_reference#comparing-launches; 1 table row |
| Clustering Image Embeddings | 14 | 0 | Move command + figure to tools_reference#clustering-image-embeddings; 1 table row |
| Configurations | 21 | 9 | Merge into Training > Launching A Run; image size rule and repeated copy instruction cut; rejected screenshot to configuration_reference |
| Run Names | 18 | 6 | Merge into Training > Run Folder as one sentence + `run_name_decoded.png`; example table cut |
| MLflow Tracking | 21 | 9 | Training subsection next to TensorBoard; run page screenshot + tags (missing in the reference) move to configuration_reference |
| Tutorials | 26 | 8 | Command block cut (tutorials/README.md has it); index page image to tutorials/README.md; add a line to the documentation map |
| **Total** | **205** | **74** | 8 top-level sections become 2 (`Evaluating Checkpoints`, `Tutorials`), plus 4 Training subsections |

Links to update if adopted: tools_reference.md:3 (`#comparing-launches`, `#clustering-image-embeddings`), configuration_reference.md:3 (`#configurations`, `#run-names`, `#mlflow-tracking`, still valid if the heading texts are kept as subsections), README.md:117 (`#configurations`), and the README table of contents (README.md:23-30).

## Part 4: Deployment Concepts And Number Formats (README.md:936-1275)

Range: 340 lines, 7,511 words (about 16 % of the README words). Two teaching sections the user asked for:
"From A Checkpoint To A Served Model" (README.md:939-1172, 234 lines) and "Number Formats" (README.md:1174-1275,
102 lines), plus the 3-line Deployment intro.

### Recommended home

Move both teaching sections, nearly whole, out of the README into two new pages under one new folder that
can also take the "Metrics Explained" section (part handled elsewhere), so the three teaching pages sit together:

- `documentation/explained/deployment_explained.md`: README.md:939-1172 (eager, export, TorchScript, ONNX, ONNX
  Runtime, TensorRT, Triton, one image through every path, which one when, glossary).
- `documentation/explained/number_formats_explained.md`: README.md:1174-1275 (bits, every format, rounding
  examples, scales, weight memory). It is a separate page because training needs it too (BF16 autocast,
  README.md:287, README.md:809), not only deployment.

Alternative (a real trade-off): one page `documentation/explained/deployment_explained.md` holding both, one fewer
link but a 300-line page mixing serving paths and floating-point bits. Two pages are better because each
can be linked from its own place: the export settings link to the first, the `--precision bfloat16` option
and the serving precision link to the second.

Not `documentation/reference/`: the reference files list settings and outputs (tools_reference.md:200-300 already
holds the facts of the export); these pages teach ideas with worked examples. The deployment tutorial series
(tutorials/storyboards/12_deployment, 3 chapters: export and checks, Triton, serving benchmark) does not cover
these concepts (no scene on eager versus graph, TensorRT fusion or number formats), so the pages do not
duplicate it; they are a candidate for a later "12/00 concepts" chapter, to mention to the user only.

Links to repoint after the move (no file outside README.md links into this range; `git grep` finds only
`README.md#deployment`, `#export`, `#export-sizes`, `#serving-benchmark-results` from other files):
- README.md:1277 (`#torchscript`, `#onnx`, `#the-eager-model`, `#how-the-export-records-the-graph`),
  README.md:1385 (`#number-formats`), README.md:1427 (`#tensorrt`), README.md:1438 (`#torchscript`):
  point them at the new pages.
- Table of contents README.md:32-33: drop the two entries.
- Inside the moved text, links back to README sections (`#export`, `#input-contract`, `#triton-server`,
  `#serving-benchmark-results`, `#what-is-exported-today`, `#quick-start-with-synthetic-data`,
  `#tensorrt-or-torchscript-and-onnx-runtime-cuda`) become `../../README.md#...` (or the new homes chosen in
  part 5 for the serving sections).

---

#### Deployment (README.md:936-938, 3 lines)
- **Keep in README**: the heading, one scenario sentence giving the order of the steps (export, input
  contract, Triton, serving formats), and a second sentence: "Should eager, ONNX, TensorRT, Triton or FP16 be
  new to you, [Deployment Explained](documentation/explained/deployment_explained.md) follows one run from its
  checkpoint to a served model and [Number Formats Explained](documentation/explained/number_formats_explained.md)
  shows what reduced precision does to the values." (2 lines).
- **Illustrate**: keep `README/deployment_paths_map.png` (now README.md:953-956) here, directly under the
  Deployment heading, with its caption. It already says, per path, who makes it, the file and its size, who
  runs it, what the serving machine needs, whether the export checks it and where it runs: the one picture a
  newcomer needs before `export_model.py`. It is the only figure of this range that stays in the README.
- **Estimated length after**: 7 lines (heading, 2 sentences, 4 lines of figure and caption).

#### From A Checkpoint To A Served Model (README.md:939-957, 19 lines)
- **Keep in README**: nothing beyond the figure moved up into "Deployment" above.
- **Move**: the scenario sentence (README.md:940) and the word table with its analogies (README.md:942-951)
  become the opening of `deployment_explained.md`; the analogies (cook, film, universal recipe, restaurant)
  are the teaching the user asked for, keep them.
- **Cut**: README.md:940 "from which every file, output and timing below comes unless the serving benchmark is
  named" can shrink to "every number below comes from this run, exported on 2026-11-19".
- **Illustrate**: in the page, `deployment_paths_map.png` again right after the word table (the README and the
  page both show it; it is a figure, not a repeated fact).
- **Estimated length after**: 0 lines in README (page: about 16 lines).

#### Training Versus Serving (README.md:958-963, 6 lines)
- **Move**: whole to `deployment_explained.md`, as is (the three needs: a readable file, speed, the same answers).
- **Estimated length after**: 0 in README (page: 6).

#### The Eager Model (README.md:964-971, 8 lines)
- **Move**: to `deployment_explained.md` with `README/eager_trace_graph.gif` (README.md:968-970).
- **Cut**: the benchmark numbers "2.2 milliseconds ... against 3.3 for TorchScript" (README.md:965) are also in
  README.md:1117-1118, README.md:1423 and README.md:1447-1448; keep the claim "eager is not slow, kernel launches
  are asynchronous" with a link to Serving Benchmark Results instead of the numbers.
- **Estimated length after**: 0 in README (page: 8).

#### Checkpoint Versus Model File (README.md:972-980, 9 lines)
- **Move**: to `deployment_explained.md`, the table as is: it holds facts the paths map does not (106 tensors,
  37.1 MB of Adam state making the checkpoint 3 times the model files, 125 ONNX nodes, 5 constants).
- **Cut**: the File row sizes repeat `deployment_paths_map.png`; keep them, the table is short and exact.
- **Estimated length after**: 0 in README (page: 9).

#### How The Export Records The Graph (README.md:981-986, 6 lines, 347 words)
- **Move**: the TorchScript and ONNX bullets (README.md:983-984) to `deployment_explained.md`: tracing freezes the
  `if` branch, `torch.export` keeps the batch symbolic, batch normalization folded.
- **Move**: the `strip_exporter_metadata` detail (README.md:984 from "Last, ..." on: 87 to 94 % of `model.onnx`,
  1.37 MB versus 0.16 MB for run_9) to `documentation/reference/tools_reference.md` "### Export"
  (tools_reference.md:202), which lists what `export_model.py` does; the page keeps one clause "debugging
  notes removed (see the tools reference)".
- **Cut**: "The check" bullet (README.md:985): `README/export_pipeline.png` (README.md:1284, Export section, the
  two Check boxes, the batch of another size, TF32 off, tolerances 1e-4 / 0.01 pixel, install or RuntimeError)
  draws exactly this. One line in the page: "each file is then checked against the eager model, see
  [Export](../../README.md#export)".
- **Estimated length after**: 0 in README (page: 4 lines, about 180 words instead of 347).

#### TorchScript (README.md:987-989, 3 lines, 204 words)
- **Move**: to `deployment_explained.md`.
- **Cut**: the TF32 trace detail (README.md:988, "The trace also records the TF32 switch ... 1.85e-05 for the
  eager model") is said again in README.md:1199 (TF32 bullet), README.md:1239, README.md:1393 and README.md:1438.
  Keep it here only (this is where it is measured) and let the others link here. The deprecation warning and
  `hide_known_export_warnings` sentence goes to tools_reference.md "### Export".
- **Estimated length after**: 0 in README (page: 3, about 130 words).

#### ONNX (README.md:990-997, 8 lines)
- **Move**: whole to `deployment_explained.md`, with `README/onnx_graph_excerpt.png`.
- **Estimated length after**: 0 in README (page: 8).

#### ONNX Runtime And Its Execution Providers (README.md:998-1016, 19 lines)
- **Move**: whole to `deployment_explained.md` (provider table and the `get_available_providers` check, which
  prevents the "why does my ONNX run on the CPU" surprise).
- **Estimated length after**: 0 in README (page: 19).

#### TensorRT (README.md:1017-1029, 13 lines, 449 words)
- **Move**: README.md:1018-1026 (interpreter versus compiler, fusion, kernel auto-tuning, FP16, with
  `README/tensorrt_fusion_and_tuning.gif`) to `deployment_explained.md`: the worked example on the real engine.
- **Cut**: README.md:1028 (engine valid for one GPU and TensorRT version, built at the first requests, cache in
  `/tmp/tensorrt_engine_cache`, 19.1 / 10.3 MB, no `model.plan`) repeats README.md:1432-1435 and README.md:1440
  (the cost list of "TensorRT Or TorchScript And ONNX Runtime CUDA") and the TensorRT column of
  `deployment_paths_map.png`. Replace by one sentence: "the engine is tied to one GPU model and TensorRT version
  and built at the first requests; its costs are listed in [TensorRT Or TorchScript ...]".
- Also: the three bullets "Layer and tensor fusion / Kernel auto-tuning / Reduced precision" of README.md:1427-1429
  repeat these three bullets. Flag for part 5: keep the definitions here (with the measured engine) and let
  README.md:1426-1429 shrink to a link plus the measured gains.
- **Estimated length after**: 0 in README (page: 10).

#### Precision Of The Numbers (README.md:1030-1032, 3 lines)
- **Cut**: whole. It only bridges to Number Formats and repeats the detector FP16 story (7.88 points), which the
  range tells three more times (README.md:1201, README.md:1238) and part 5 twice (README.md:1452, README.md:1467).
  The word table row "Precision (FP32 / FP16)" (README.md:950) already defines the term; add to it a link to
  `number_formats_explained.md`.
- **Estimated length after**: 0.

#### Triton Inference Server (README.md:1033-1053, 21 lines)
- **Move**: the client snippet with its output (README.md:1036-1052) to `deployment_explained.md`: the worked
  example of a client needing only `tritonclient` and numpy.
- **Cut**: README.md:1034 (ports 8000 and 8001, `config.pbtxt`, 1 millisecond dynamic batching, backends) repeats
  `README/triton_request_flow.png` (README.md:1374, Triton Server section, which shows ports 8000 / 8001 / 8002,
  the dynamic batcher up to 16 requests and 1000 microseconds, the backends), README.md:1406-1417 and the
  glossary. Keep two sentences in the page: "a server that holds the model and answers over the network,
  see [Triton Server](../../README.md#triton-server)" and the "in the process" definition, which the paths
  need.
- **Estimated length after**: 0 in README (page: 17).

#### One Image Through Every Path (README.md:1054-1122, 69 lines)
- **Move**: the scenario, the snippet and its printed line (README.md:1055-1097) to `deployment_explained.md`:
  the strongest worked example of the range (run it, see 0.0 and 2.7e-06).
- **Cut**: the second table (README.md:1113-1122, 10 lines with its sentence): its "Python / PyTorch /
  repository code needed / tied to one GPU / checked" columns are the "Needs on the serving machine" and
  "Checked by the export" rows of `deployment_paths_map.png`, and its timing columns are the first two columns
  of Serving Benchmark Results (README.md:1445-1452) and the rows of What Is Exported Today (README.md:1390-1396).
  One line instead: "their latency and throughput are in [Serving Benchmark Results]".
- **Illustrate**: replace the 9-row table of probabilities (README.md:1099-1109) and its explanation
  (README.md:1111) by a new figure `README/one_image_every_path.png`, "How far each path moves from the eager
  model": one horizontal row per path (the 9 rows of the table), a dot at the largest difference over the 35
  probabilities on a log axis from 1e-8 to 1e-3 (the 0 of TorchScript drawn at the left edge as "identical"),
  a vertical dashed line at the export tolerance 1e-4, dots coloured by "checked by the export" (TorchScript and
  ONNX rows) or "never checked" (TensorRT rows), and the two TensorRT FP32 rows (first build, container
  started again) joined by a bracket "same file, engine built again". Data: the numbers of README.md:1101-1109
  themselves (no rerun needed). One glance: every path within 1e-4 except TensorRT FP16, which the export never
  checks, and a rebuilt engine changes its digits. Keep the hexagon / hollow_hexagon / pentagon digits out: the
  figure says "same order of the three top classes on every path" in its caption.
- **Estimated length after**: 0 in README (page: about 45 lines instead of 69).

#### Which One When (README.md:1124-1131, 8 lines)
- **Move**: to `deployment_explained.md` as its closing section, as is (5 situation bullets).
- **Cut**: nothing; it is short and decisive. Check against part 5: if "Serving Benchmark Results / In short"
  (README.md:1465-1472) stays in the README, these bullets should be the only "which path" advice in the
  teaching page and link there for the numbers.
- **Estimated length after**: 0 in README (page: 8).

#### Glossary (README.md:1132-1172, 42 lines, 834 words)
- **Move**: to the end of `deployment_explained.md`.
- **Cut**: 13 of the 37 rows. Six repeat the word table (README.md:942-951) one row each: Eager
  (README.md:1141), TorchScript (1145), ONNX (1147), ONNX Runtime (1149), TensorRT (1151), Triton Inference Server
  (1155); keep them in one place, the word table, and say "the eight words of the opening table are not repeated".
  The seven number formats (README.md:1166-1172) repeat `number_formats.png` and the Summary Of The Formats;
  replace them by one row "FP32, TF32, BF16, FP16, FP8, INT8, FP4: see Number Formats Explained".
- **Estimated length after**: 0 in README (page: about 29).

#### Number Formats (README.md:1174-1176, 3 lines)
- **Keep in README**: nothing; the pointer sits in the Deployment intro. Add the same link where the README
  mentions the bfloat16 training precision (README.md:287 or README.md:809) so training readers find it.
- **Move**: the intro (bit, finite set of values, fewer bits = faster and coarser) opens
  `number_formats_explained.md`.
- **Estimated length after**: 0 in README (page: 3).

#### What A Floating-Point Number Is (README.md:1177-1190, 14 lines)
- **Move**: whole, with `README/number_format_decoding.gif` (1000.3 decoded from 0x63D1). Core teaching, keep.
- **Estimated length after**: 0 in README (page: 14).

#### The Formats One By One (README.md:1191-1207, 17 lines, 855 words)
- **Move**: to `number_formats_explained.md` with `README/number_formats.png`.
- **Cut**: in each bullet, the bit counts, largest value and gap after 1 in parentheses (README.md:1198-1204) are
  drawn in `number_formats.png` right above (bits to scale, largest finite, smallest normal, gap after 1, bytes);
  drop the parentheses and keep full name, why it exists, where the repository uses it and when to choose it.
  The FP16 bullet's serving numbers ("within 0.03 points ... 1.9 times", README.md:1201) and the BF16 bullet's
  Triton refusal (README.md:1200) become links to Serving Benchmark Results and What Is Exported Today.
- **Estimated length after**: 0 in README (page: 15 lines, about 600 words).

#### The Same Values In Every Format (README.md:1208-1234, 27 lines)
- **Cut**: the 9-line table (README.md:1211-1219). `README/number_format_examples.png` (README.md:1224) shows the
  same six values in the same six formats with the stored value printed in bold next to each lane and the gap,
  plus inf / 448 / 0 arrows: the table is the figure in text form.
- **Move**: the sentence of README.md:1209 (how the values were computed, PyTorch dtypes, TF32 rounding rule)
  becomes the figure caption; keep `number_format_examples.png` and `number_format_rounding.png`.
- **Estimated length after**: 0 in README (page: about 14).

#### What These Numbers Mean (README.md:1235-1242, 8 lines, 386 words)
- **Move**: to `number_formats_explained.md` (the five lessons are the point of the page).
- **Cut**: in the FP16 bullet (README.md:1238), the account of the NMS bug (group offsets of about 1,025 pixels,
  the 64th group, `suppress_stacked_groups`, +0.21 points) is told again at README.md:1467 and in the serving
  benchmark plan and tutorial 12/03 ("Why the detector broke in FP16", "The fix"); keep the rule ("every sum,
  area or offset inside a model served in FP16 must stay below 65,504") and one clause "the exported NMS of the
  detector broke this rule, see Serving Benchmark Results". The TF32 bullet (README.md:1239) keeps its DET
  measurement and links to the TorchScript paragraph for the export switch instead of restating it.
- **Estimated length after**: 0 in README (page: 7 lines, about 260 words).

#### Scales For INT8 And FP8 (README.md:1243-1252, 10 lines)
- **Move**: whole (scale, zero point, per channel with the real block 2 weights, calibration, Q/DQ).
- **Illustrate** (optional, the text already carries a worked example): a small figure
  `README/int8_per_channel_scale.png`: the 32 output channels of `backbone.convolution_blocks.1` of the README
  classifier (BN folded) as horizontal bars of their largest absolute weight, with two scales drawn as level
  ticks: one tensor scale (2.73 / 127, the narrowest channel using 13 levels) and per-channel scales (127
  levels each). Data: the checkpoint of the README classifier. One glance: why weights get one scale per channel.
- **Cut**: README.md:1251 (none of this implemented: Q/DQ, calibration table, accepted `tensorrt_precision`
  values, no FP8 precision_mode, engine never checked) repeats What Is Exported Today (README.md:1397-1400) and
  tools_reference.md:203; one sentence with the link.
- **Estimated length after**: 0 in README (page: 9).

#### Summary Of The Formats (README.md:1253-1266, 14 lines)
- **Cut**: whole. Its Bits, Largest value and Gap after 1 columns are `number_formats.png`; Full name is the first
  words of each bullet of The Formats One By One.
- **Illustrate**: add a sixth column "used here for" to `number_formats.png` (FP32: weights, evaluation, export
  inputs and outputs; TF32: eager convolutions, TensorRT "FP32", off in the export; BF16: training forward
  passes; FP16: TensorRT FP16, TorchScript FP16; the rest: not implemented, greyed), so the figure becomes the
  summary the reader comes back to. Source: the Used column of README.md:1258-1265.
- **Estimated length after**: 0.

#### Weight Memory And Bandwidth (README.md:1267-1275, 9 lines)
- **Move**: to `number_formats_explained.md` with `README/model_weight_sizes.png`.
- **Cut**: the end of README.md:1274 ("TensorRT FP16 serves 1.9 times ... DET serves 189 images per second")
  repeats README.md:1466-1467; replace by a link to Serving Benchmark Results.
- **Estimated length after**: 0 in README (page: 7).

---

### Summary of the part

| Section | Lines now | Lines after (README) | Main action |
|---|---|---|---|
| Deployment (intro) | 3 | 7 | keep, add the two page links and `deployment_paths_map.png` |
| From A Checkpoint To A Served Model | 19 | 0 | move word table with analogies to `documentation/explained/deployment_explained.md` |
| Training Versus Serving | 6 | 0 | move |
| The Eager Model | 8 | 0 | move, drop benchmark numbers (link) |
| Checkpoint Versus Model File | 9 | 0 | move as is |
| How The Export Records The Graph | 6 | 0 | move; check bullet cut (export_pipeline.png); exporter metadata to tools_reference |
| TorchScript | 3 | 0 | move; single home of the TF32 trace fact |
| ONNX | 8 | 0 | move |
| ONNX Runtime And Its Execution Providers | 19 | 0 | move |
| TensorRT | 13 | 0 | move fusion/tuning example; engine costs cut (in README.md:1432-1440) |
| Precision Of The Numbers | 3 | 0 | cut (bridge, repeats the 7.88-point story) |
| Triton Inference Server | 21 | 0 | move client snippet; description cut (triton_request_flow.png) |
| One Image Through Every Path | 69 | 0 | move snippet; second table cut; new figure one_image_every_path.png replaces the probability table |
| Which One When | 8 | 0 | move as is |
| Glossary | 42 | 0 | move, drop 13 duplicated rows |
| Number Formats (intro) | 3 | 0 | move to `documentation/explained/number_formats_explained.md` |
| What A Floating-Point Number Is | 14 | 0 | move as is |
| The Formats One By One | 17 | 0 | move, drop the numbers number_formats.png shows |
| The Same Values In Every Format | 27 | 0 | table cut (number_format_examples.png shows it) |
| What These Numbers Mean | 8 | 0 | move, shorten the NMS story to the rule plus a link |
| Scales For INT8 And FP8 | 10 | 0 | move; optional per-channel scale figure |
| Summary Of The Formats | 14 | 0 | cut; add a "used here for" column to number_formats.png |
| Weight Memory And Bandwidth | 9 | 0 | move, serving numbers replaced by a link |
| **Total** | **340** | **7** | two teaching pages: about 230 lines (deployment) + 85 lines (number formats), down from 337 |

Repeated facts found in this range and elsewhere (for whoever writes the final plan, so each fact gets one home):
- Detector FP16 overflow, 7.88 points: README.md:1031, 1201, 1238 (this part), 1452, 1467 (part 5), tutorial 12/03.
- Export turns TF32 off, `model.pt` keeps FP32 convolutions: README.md:985, 988, 1199, 1239 (this part), 1393, 1438 (part 5).
- Eager 2.2 ms versus TorchScript 3.3 ms: README.md:965, 1117-1118 (this part), 1423, 1447-1448 (part 5).
- TensorRT fusion / auto-tuning / FP16 and engine costs: README.md:1018-1021, 1028 (this part), 1426-1440 (part 5), serving_formats.png.
- Path needs and checks: README.md:1115-1122 (this part), 1390-1396 (part 5), deployment_paths_map.png.

## Part 5: Export To Repository Structure (README.md:1276-1579)

Range: 304 lines, 5,891 words (the teaching part 936-1275 of the other agent holds 7,511 words). Sources compared:
`documentation/reference/tools_reference.md` (Deployment section, lines 200-312), `tutorials/storyboards/12_deployment/`
(01 export and its checks, 02 serving with Triton, 03 serving benchmark), `tutorials/storyboards/01_foundations/`
(01 repository and first run, 06 after the training), `documentation/architecture_benchmarking/serving_benchmark_plan.md`,
`documentation/suggestions.md` (item 41), the `Dockerfile` header.

### Proposed home of the deployment content (coordination with the part 936-1275)

Recommendation: one new document, **`documentation/reference/deployment_reference.md`**, named like the other references,
which takes (a) the teaching sections of README 936-1275 (other agent), (b) the measured serving sections of this part
(README 1384-1473, deduplicated), (c) the Export Sizes details (README 1310-1347) and (d) the whole Deployment section of
`tools_reference.md` (200-312), which then stops being split between two files. Order inside it, from "what the words
mean" to "what it costs":

1. Concepts: From A Checkpoint To A Served Model, Number Formats (other agent).
2. Export: settings and checks (tools_reference 202-225 + README 1277, 1302-1304), served model inputs and outputs (README 1289-1302), export at a logit temperature (README 1308, pointer to tools_reference Logit Temperature, which stays in tools_reference as an evaluation tool).
3. Input Contract (tools_reference 227-254; README 1350-1363 reduced to the figure).
4. Test Split Parity and FP16 TorchScript Trial (tools_reference 256-277).
5. Triton Server and Throughput Measurement (README 1366-1382 + tools_reference 279-280).
6. Serving Paths Measured (README 1384-1473 after the cuts below), then the Serving Benchmark drivers (tools_reference 282-295).
7. Export Sizes and Export Size Check (README 1310-1347 + tools_reference 297-312).

The README keeps one short **Deployment** section (about 30 lines, sketched at the end of this file) that gets a newcomer
from a trained run to a served, queried model and links into that document. `tools_reference.md` loses 113 lines and
its intro (line 3) drops "Deployment". Inbound links to update: `README.md#export-sizes` (12 occurrences: task
workflows, `training_reference.md:97`, `tools_reference.md:298`), `README.md#export` (`tools_reference.md:203`),
`README.md#serving-benchmark-results` (`serving_benchmark_plan.md:249`).

Alternative (real trade-off): keep the practical sections (Export, Input Contract, Triton Server) in the README at full
length and move only the teaching and the measurements. That keeps the README self-sufficient for deployment but leaves
about 90 lines here, mostly duplicating tools_reference; not recommended.

---

#### Export (README.md:1276-1309, 34 lines, 941 words)
- **Keep in README**: one scenario sentence, a 6-row settings table replacing the run-on sentence of README.md:1277 (`experiment_folder`, `iteration`, `use_averaged_weights`, `formats`, `maximum_batch_size`, `tensorrt_precision` / `torchscript_precision`, with their values), the command (1279-1281), `README/export_pipeline.png` (1283-1285, it already shows staging, both checks, install-or-raise and the `served_models/` tree, so the prose describing those can go), one line with the output folder (1287) and one line "Before serving a reduced precision, set `test_split_parity = True` (details)". About 15 lines.
- **Cut**:
  - README.md:1277 parenthetical on the keypoint detector crash and the sequence TensorRT FP32 engines: dated incident already recorded as item 41 of `documentation/suggestions.md:131-138`; keep a one-word pointer ("known issues: suggestions item 41") or nothing.
  - README.md:1277 "any other format or precision stopping the export before it starts": duplicated in tools_reference.md:203 (`check_export_settings`).
  - README.md:1304 (FP16 TorchScript, 120 words): nearly word for word in tools_reference.md:203 (`convert_to_half_precision`, `model_fp16.pt`, `default_model_filename`, `HALF_PRECISION_TOLERANCE_FACTOR`). Also in tutorial 12_deployment/01 "The settings of the export" / "Tolerances of the comparison".
  - README.md:1306 (test split parity, 200 words): the folder naming, files and "never decides the installation" are all in tools_reference.md:256-266. README keeps one line.
  - README.md:1308 (export at a temperature): steps 2-4 of tools_reference.md:44-51 (Logit Temperature) say the same, with the button names. README keeps nothing here; the Evaluating Checkpoints part of the README already points to Logit Temperature.
- **Move**:
  - The served-model table and its legend (README.md:1289-1302, 14 lines) to `deployment_reference.md#served-model-inputs-and-outputs`, merged with the post-processing paragraphs it already points to (tools_reference.md:205-225), so a client author reads inputs, outputs and post-processing in one place. Pointer: "Inputs, outputs and client post-processing of every task: [deployment reference](...)". Alternative: keep this table in the README (it is scannable and per task); it costs 14 lines.
  - Everything cut above lands in `deployment_reference.md#export` (from tools_reference, not from the README copy).
- **Illustrate**: reuse `README/export_pipeline.png` as the single explanation of the export flow (it makes 1277 and the install sentence redundant). Optional new figure for the moved I/O table: "What each task's served model returns", a 10-tile grid, one tile per task, a synthetic-shapes input with its decoded output drawn on it (boxes, masks, keypoints, class map, frame strip) and the tensor names and shapes as captions; data from the ten tutorial runs in `tutorials/run_snapshots`. Belongs in deployment_reference (or tasks_reference if the part-agent of tasks_reference wants one picture per task).
- **Estimated length after**: 15 lines.

#### Export Sizes (README.md:1310-1348, 39 lines, 816 words)
- **Keep in README**: one sentence, placed in the training launch part (it is a launch panel, not an export step): "Every launch prints an Export Sizes panel estimating each exported file before training ([deployment reference](...#export-sizes))". 1-2 lines.
- **Cut**: README.md:1329 (real sizes of run_9 file by file) and README.md:1340 timings of the estimates (0.1 to 0.8 s, 2 to 27 s, 5 s to 43 min): calibration evidence, not instructions; the check script reproduces them (tools_reference.md:312). README.md:1342-1347 command block duplicates tools_reference.md:300-303.
- **Move**: the panel example (1313-1327), the formula table (1331-1338) and the Exported Parameters paragraph (1340) to `deployment_reference.md#export-sizes`, joined with the Export Size Check section (tools_reference.md:297-312). Update the 12 links to `README.md#export-sizes`. Tutorial 12_deployment/01 scene "Export sizes known at launch" already narrates the same mechanism; link the chapter from there.
- **Illustrate**: replace the 7-row formula table (1331-1338) in the reference by "Estimated against real export sizes": log-log scatter, one point per tutorial run and format (PyTorch FP32/FP16, TorchScript FP32/FP16, ONNX, checkpoint), the y = x line with a ±3 % band, TensorRT engines as vertical range bars with the real engine size as a dot; data from `export_size_check.csv` of `~/Data/export_size_check` (written by `check_export_size_estimates.py`). One glance shows which formats are exact and why TensorRT is a range. The formula column stays as a 1-line caption per format.
- **Estimated length after**: 2 lines in the README (0 under this heading; the heading disappears).

#### Input Contract (README.md:1349-1364, 16 lines, 180 words)
- **Keep in README**: one sentence inside the Deployment section: "A server builds the input of a photo from `preprocessing.json` (also in the ONNX metadata), never by assuming a resize ([input contract](...))." 1-2 lines.
- **Cut**: README.md:1350 and the code block 1357-1361 and line 1363 repeat tools_reference.md:228, 248-252 and 254 almost verbatim.
- **Move**: `README/input_contract_resizes.png` and its caption (1352-1355) to `deployment_reference.md#input-contract`, above the key table of tools_reference.md:230-242. The figure is the best part of this section; it belongs with the key table it illustrates.
- **Illustrate**: existing `input_contract_resizes.png` (stretch / keep_ratio / tiles on one COCO photo). No new figure needed.
- **Estimated length after**: 2 lines (folded into Deployment, heading removed).

#### Triton Server (README.md:1365-1383, 19 lines, 191 words)
- **Keep in README**: the scenario sentence with the ports (1366), the `docker compose up` block (1368-1371) and one line for `run_inference` (command only). About 8 lines. This is the "serve and query" step a newcomer needs.
- **Cut**: README.md:1377 details (one request at a time, dynamic batcher never forms a batch, tiled SEG refused): duplicated in tools_reference.md:280 (Throughput Measurement) and tutorial 12_deployment/02 "Measuring a model with run_inference".
- **Move**: `README/triton_request_flow.png` (1373-1375) to deployment_reference Triton Server; the teaching section 1033-1053 (other agent) already explains the same request path in words, so this figure should replace part of that prose there. `README/run_inference_output.png` (1379-1382) to the same place; a console screenshot is not needed in the README quick path.
- **Illustrate**: reuse `triton_request_flow.png` once (in the reference). Note (out of scope): its container title `object-detection-triton-server` matches `container_name` of `vision_training/deployment/triton/docker-compose.yml:18`, a name left from a detection-only past while the server holds every task.
- **Estimated length after**: 8 lines.

#### Serving Formats And Precision, intro (README.md:1384-1386, 3 lines, 91 words)
- **Keep in README**: nothing.
- **Cut**: the Triton / CUDA / TensorRT version list duplicates `triton_request_flow.png` and the teaching part (TensorRT 1017-1029).
- **Move**: to `deployment_reference.md#serving-paths-measured` as its one-line intro (hardware + image version once, for the whole document).
- **Illustrate**: none.
- **Estimated length after**: 0 lines in the README.

#### What Is Exported Today (README.md:1387-1405, 19 lines, 336 words)
- **Keep in README**: nothing; the README's "which path" list (below) names the three exported paths.
- **Cut**: this 9-row table is the third copy of the same matrix: the "Precision (FP32 / FP16)" and TensorRT rows of the word table (README.md:945-952), `README/deployment_paths_map.png` (953-956), `README/serving_formats.png` (1402-1404), Precision Of The Numbers (1030-1032), and tutorial 12_deployment/02 scene "The serving paths", which draws the same rows with the same columns. Keep one table in the reference, in the Concepts or the Measured part, not both. The INT8 / BF16 / FP8 rows (1397-1400) duplicate the number formats summary of the other part (1253-1266); keep them in one of the two.
- **Move**: one table to `deployment_reference.md#what-is-exported-today`; coordinate with the other agent so that `deployment_paths_map.png` and `serving_formats.png` are merged into one figure.
- **Illustrate**: merge `serving_formats.png` and the table: add a "checked by the export" badge column (GPU check, CPU check, not checked) and a "status" colour (exported / served unchecked / not servable / not implemented) to `serving_formats.png`, and drop its bottom "Layer fusion" panel, which duplicates `README/tensorrt_fusion_and_tuning.gif` (teaching part, 1017-1029). The table then becomes the figure plus a 3-line note on float32 inputs and outputs.
- **Estimated length after**: 0 lines in the README.

#### Triton Or The Model In The Process (README.md:1406-1424, 19 lines, 472 words)
- **Keep in README**: nothing (one "which path" bullet covers it).
- **Cut**: rows "HTTP and gRPC", "metrics on port 8002" and "versions side by side" of the table (1409-1417) repeat Triton Inference Server of the teaching part (1033-1053) and the ports in Triton Server (1366); `dynamic_batching_timeline.png` (1419-1421) explains the batcher the teaching part also explains in words. README.md:1423 numbers (4.5 / 2.5 / 6.5 / 3.9 / 2.2 ms, 1,747 against 1,168, 166 against 153, "the 171 against 64 came from the NMS broken in batches") repeat the Serving Benchmark Results (1442-1473) and the plan (`serving_benchmark_plan.md` 83-174); the parenthesis about the broken NMS is history that belongs in the plan only.
- **Move**: the table (reduced to its "set by the export / it costs" columns, which are unique: `max_queue_delay_microseconds`, `instance_group`, `version_policy`, no model control mode, 3.1 / 12.6 MB per request) and `dynamic_batching_timeline.png` to `deployment_reference.md#triton-or-the-model-in-the-process`. Tutorial 12_deployment/02 "Changing batching and instances" covers how to change them.
- **Illustrate**: `dynamic_batching_timeline.png` stays as the one picture of the batcher (and the teaching part 1033-1053 should point to it rather than describing it in prose).
- **Estimated length after**: 0 lines in the README.

#### TensorRT Or TorchScript And ONNX Runtime CUDA (README.md:1425-1441, 17 lines, 722 words)
- **Keep in README**: nothing.
- **Cut**: the three bullets of 1427-1429 (fusion, auto-tuning, reduced precision) are the same three bullets as TensorRT in the teaching part (1017-1029, with real numbers there) and the bottom panel of `serving_formats.png`. The engine-cache explanation of 1434 repeats the end of the TensorRT teaching section (1029: "caches them in `/tmp/tensorrt_engine_cache` of the container under a prefix naming the export") and tools_reference.md:203 (`trt_engine_cache_prefix`). "FP32 is not exact FP32" (1438) repeats Precision Of The Numbers (1030) and the "What Is Exported Today" table. The throughput sentence of 1431 repeats the results table (1447-1452).
- **Move**: the cost list (1432-1438) is the one unique piece: keep it in `deployment_reference.md#what-tensorrt-costs` as 7 short bullets of one line each (engines built at first request, new engine per larger batch, cache not mounted, engines tied to GPU and TensorRT version, unsupported operators, 4 GiB workspace, TF32 in "FP32"), each with its fix where one exists (warm up, mount the cache, `trt_profile_*_shapes`). Tutorial 12_deployment/02 "Engines built at the first requests" draws the first one as a bar chart; reuse it.
- **Illustrate**: new figure "First requests with TensorRT", grouped bars of the seconds of the first request at batch 1, 4 and 16 for TorchScript, ONNX Runtime CUDA, TensorRT FP32 and FP16 on DET (11.9 / 17.1 / 34.4 s for TensorRT FP32 against at most 3.8 s), data from `serving_benchmark_results.csv`; the tutorial scene already computes it from the same data, so the figure script can share its numbers. Replaces the prose of 1432-1433.
- **Estimated length after**: 0 lines in the README.

#### Serving Benchmark Results (README.md:1442-1473, 32 lines, 1,011 words)
- **Keep in README**: fold the outcome into one "Which path" list of 4-5 bullets in the README Deployment section, merged with "Which One When" of the teaching part (README.md:1124-1131, other agent), so the recommendation lives once: eager to debug; TorchScript when the service links PyTorch; ONNX Runtime for portability; TensorRT FP16 for throughput after `test_split_parity` (3.5x the eager model on CLS and M2M at batch 16, DET 17x); dynamic batching only when requests overlap. About 6 lines, every number linked to the plan.
- **Cut**:
  - The 7-column table (1445-1455) duplicates `serving_benchmark_plan.md` Results (83-174) and `serving_benchmark_results.csv`; its DET cells of the ONNX rows are obsolete since the NMS fix (1457 says to read another table in their place).
  - The NMS-fix table (1459-1463) duplicates `serving_benchmark_plan.md:217-228` (Detector NMS Without Offsets).
  - The "In short" list (1465-1472) duplicates the plan's Findings (236-245) and tutorial 12_deployment/03 "What the benchmark decided", "Why the detector broke in FP16", "The fix".
  - The explanation of how the DET FP16 overflow happened (1467) is history: the plan keeps it.
  - The driver pointer (1443 last sentence) duplicates tools_reference.md:282-293.
- **Move**: one results figure (below) plus the decisions to `deployment_reference.md#serving-benchmark-results`; numbers stay in the plan and its CSV files only. Update `serving_benchmark_plan.md:249`, which links to this README anchor.
- **Illustrate**: new figure "Serving paths: speed against accuracy", three small panels (CLS, DET, M2M), one bar per path (eager, TorchScript in process, Triton TorchScript, ONNX Runtime CUDA, TensorRT FP32, TensorRT FP16) giving the batch-16 throughput relative to the eager model, the test-metric change in points printed on each bar (red when beyond 0.1 point), and the memory in GB as a small label; BF16 / native engine / INT8 as greyed "not servable / not implemented" stubs. Data: `serving_benchmark_results.csv`, `serving_benchmark_test_metrics.csv`, and `serving_benchmark_nms_fix_test_metrics.csv` for the DET ONNX rows (post-fix values only, so no "before/after" footnotes). It replaces two tables and eight bullets with one picture.
- **Estimated length after**: 6 lines in the README (the shared "Which path" list; 0 under this heading).

#### Tests (README.md:1474-1486, 13 lines, 59 words)
- **Keep in README**: the scenario sentence, `uv run pytest tests`, and the single-folder example inline (one sentence instead of a second paragraph). About 5 lines.
- **Cut**: nothing substantial; the tutorial 01_foundations/06 "The tests" scene repeats it, which is fine (tutorials may repeat).
- **Move**: `README/tests_layout.png` to `training_reference.md` or drop it: it states "1,004 test cases ... on 2026-11-20" while the suite now has 1,246 (commit 2991f2af), so it is stale and would need redrawing at every test change. Recommend dropping the image and keeping the sentence "folders of `tests/` mirror `vision_training/` and `scripts/`", which the Repository Structure tree already says.
- **Illustrate**: none (removing the stale image is the illustration change).
- **Estimated length after**: 5 lines.

#### Docker Image (README.md:1487-1497, 11 lines, 188 words)
- **Keep in README**: one scenario sentence ("Should a training run in a container or on Kubernetes, the `Dockerfile` builds the trainer for amd64 and arm64") with a link. 2 lines.
- **Cut**: the stage table (1490-1496) repeats the `Dockerfile` header (lines 7-12, same five stages, same wording) and tutorial 01_foundations/01 shows that header on screen. "Starts from Ubuntu 24.04, not a CUDA image" and the `.dockerignore` list are also in the Dockerfile header (lines 2-5).
- **Move**: the remaining intro facts to `training_reference.md#training-in-a-container` (new section) together with the three subsections below; a container is not on the newcomer's clone-to-model path (Prerequisites, README.md:92, already says Docker is optional).
- **Illustrate**: none needed; the Dockerfile header is the diagram.
- **Estimated length after**: 2 lines (this heading plus the subsections below collapse into one short README section).

#### Building The Image (README.md:1498-1518, 21 lines, 188 words)
- **Keep in README**: the native build command (1502) only, 3 lines with its fence, inside the short Docker section.
- **Cut**: the `--target test` command (1503) and the QEMU sentence (1517) repeat the Dockerfile header (lines 11, 16-18).
- **Move**: the registry push and `imagetools create` block (1508-1515) and the `GIT_COMMIT` labels sentence (1517) to `training_reference.md#training-in-a-container`.
- **Illustrate**: none.
- **Estimated length after**: 3 lines.

#### Running A Training (README.md:1519-1533, 15 lines, 188 words)
- **Keep in README**: the `docker run` block (1522-1528) with a one-sentence opener, as it shows the three mounts and `--ipc=host` a newcomer would get wrong. About 9 lines. Alternative: move it too and keep only the link (saves 9 lines).
- **Cut**: nothing duplicated elsewhere today.
- **Move**: the `check_configuration` in a pipeline paragraph (1530) and the SIGINT / `docker stop --timeout 120` paragraph (1532) to `training_reference.md#training-in-a-container`; the SIGINT behaviour is the same as Ctrl+C on the host, which the training reference should state once for both.
- **Illustrate**: none.
- **Estimated length after**: 9 lines.

#### Kubernetes Pods (README.md:1534-1541, 8 lines, 148 words)
- **Keep in README**: nothing.
- **Cut**: nothing.
- **Move**: to `training_reference.md#training-in-a-container`. Better still (idea for the user): turn the five bullets into a commented example manifest, e.g. `kubernetes/training_pod.yaml`, whose instructional YAML comments carry each rule (`args` not `command`, `nvidia.com/gpu`, `/dev/shm` emptyDir, `terminationGracePeriodSeconds`, `securityContext` 10001); the README and the reference then link the file, and the facts live in the one place an engineer copies from.
- **Illustrate**: the commented manifest replaces the prose.
- **Estimated length after**: 0 lines in the README.

#### Repository Structure (README.md:1542-1579, 38 lines, 360 words)
- **Keep in README**: the scenario sentence and `README/repository_layout.png` (1545-1547); a tree reduced to the top level and `documentation/` (about 10 lines: README, pyproject/uv.lock, Dockerfile, documentation/ with its five subfolders, vision_training/, scripts/, tests/, tutorials/). About 16 lines.
- **Cut**: the `vision_training/` subtree of the tree (1562-1574, 13 lines) repeats the boxes of `repository_layout.png` almost word for word ("loading, checks, run names" / "load, check, run name"; "metrics, reports and checkpoint evaluation"; etc.). Also 01_foundations/01 and 06 show the same tree.
- **Move**: nothing.
- **Illustrate**: extend `repository_layout.png` with a thin bottom row for the top-level files (`pyproject.toml`, `uv.lock`, `Dockerfile`, `documentation/` with its subfolders `reference/`, `task workflows/`, `architecture_benchmarking/`, `presentation/`), then the tree can go entirely (saves the remaining 10 lines). Placement idea for the coordinator: this section helps orientation, so it would serve better right after Prerequisites / Quick Start than at the end, before the reader meets file paths.
- **Estimated length after**: 16 lines (6 if the image is extended and the tree dropped).

---

### Sketch of the README Deployment section that replaces 936-1473 (for the coordinator)

About 30 lines, under one heading `## Deployment`:
1. Scenario sentence + link to `deployment_reference.md` for the concepts (eager, TorchScript, ONNX, TensorRT, Triton, number formats) and the measurements.
2. Settings table (6 rows) + `uv run python -m vision_training.deployment.export_model` + `export_pipeline.png`.
3. One line: inputs, outputs and client post-processing per task, input contract (`preprocessing.json`): links.
4. One line: `test_split_parity = True` before serving a reduced precision: link.
5. `docker compose up` block with ports + `run_inference` command.
6. "Which path" list (4-5 bullets, the merge of Which One When 1124-1131 and the benchmark findings), linking the results figure.

### Summary of the part

| Section | Lines now | Lines after | Main action |
|---|---|---|---|
| Export | 34 | 15 | Settings as a table, keep command + export_pipeline.png; FP16 TorchScript, parity, temperature paragraphs cut (in tools_reference); I/O table to deployment_reference |
| Export Sizes | 39 | 2 | Move to deployment_reference with the Export Size Check, one sentence in the training launch part; estimate-vs-real scatter figure |
| Input Contract | 16 | 2 | Verbatim duplicate of tools_reference; figure moves next to the key table |
| Triton Server | 19 | 8 | Keep compose command and ports; figures and run_inference details to the reference |
| Serving Formats And Precision (intro) | 3 | 0 | Versions once in the reference |
| What Is Exported Today | 19 | 0 | Third copy of the paths matrix; merge into one serving_formats figure with check badges |
| Triton Or The Model In The Process | 19 | 0 | Unique "set by the export / costs" columns to the reference; numbers live in the plan |
| TensorRT Or TorchScript And ONNX Runtime CUDA | 17 | 0 | Fusion/tuning bullets duplicate the teaching part; keep 7 one-line costs in the reference; first-request bar figure |
| Serving Benchmark Results | 32 | 6 | Tables and findings duplicate the plan; one speed-vs-accuracy figure in the reference, one shared "Which path" list in the README |
| Tests | 13 | 5 | Keep command; drop the stale tests_layout.png (1,004 vs 1,246 cases) |
| Docker Image (intro + stage table) | 11 | 2 | Stage table duplicates the Dockerfile header |
| Building The Image | 21 | 3 | Keep the native build; push/multi-arch to training_reference |
| Running A Training | 15 | 9 | Keep the docker run block; the rest to training_reference |
| Kubernetes Pods | 8 | 0 | To training_reference, or a commented example pod manifest |
| Repository Structure | 38 | 16 (6) | Drop the vision_training subtree that repeats repository_layout.png; move the section near the top |
| **Total** | **304** | **66** (56) | One deployment_reference.md absorbing README 1276-1473 details and tools_reference 200-312 |

## Part 6: Metrics Explained, CLS, DET And INS (README.md:1580-2010)

Range read: README.md:1580-2010, 431 lines and 11,070 words (23 % of the words of the whole README in 16 % of its lines,
the paragraphs being long). The whole "Metrics Explained" section (README.md:1580-2643) is 1,064 lines and about
26,000 words: more than half of the words of the README. Moving it out is the single largest cut available.

### Where Metrics Explained should live (decision)

**Recommended: one new document, `documentation/reference/metrics_reference.md`** (H1 "Metrics Explained" kept as the
user named it, next to the five other `*_reference.md` files), holding the eight families of README.md:1580-2643 in
the same order, with the figures still in `README/metrics_*` (referenced as `../../README/metrics_*.png`, as
tools_reference.md already links `../../README.md`; no image moves, no binary churn).

Why not per task inside `tasks_reference.md`:
- The metrics are shared, not per task: the classification block (README.md:1599-1788) serves CLS, M2O, CMP, M2M and
  the frame metrics of SEL (README.md:2457); the matching, AP and mAP of DET (README.md:1815-1880) serve INS and KPT
  (README.md:1790, "which the keypoint detection reuses for its OKS mAP"); the recall intervals (README.md:1754-1780)
  serve all ten tasks. Split per task, each would be duplicated three to five times or cross-linked back and forth.
- `tasks_reference.md` is 8,800 words and must shrink; the metrics would add about 26,000 words, three times its size.
- No file outside README.md links into these anchors today (checked: only `tools_reference.md` has its own
  `#logit-temperature`, a different section), so the move breaks no external link; the ~30 internal README links
  (`#recall-intervals` alone is linked 10 times) are rewritten to `documentation/reference/metrics_reference.md#...`.

What `tasks_reference.md` keeps about metrics (consequence for the part shortening it): its "Evaluation" blocks stop
defining metrics and listing files, columns and tags, which today duplicate this section:
- tasks_reference.md:114-122 (CLS, 9 bullets, about 800 words) duplicates README.md:1603, 1605-1717, 1739-1753:
  keep 2 lines (headline metric, report tables, link to `metrics_reference.md#classification-cls-m2o-cmp`).
- tasks_reference.md:165-176 (DET, 11 bullets) duplicates README.md:1829-1978: keep 2 lines + link.
- tasks_reference.md:205-208 (INS) duplicates README.md:1979-2010: keep 1 line + link.
- The "Display" bullet (percentages in reports, 0 to 1 in files) is repeated six times (tasks_reference.md:122, 175,
  208, 280, 317, 392) and twice here (README.md:1584, 1603): one sentence in the introduction of metrics_reference.md.
- The temperature effect is explained three times (README.md:1739-1753, tools_reference.md:45, tasks_reference.md:120):
  keep the explanation in metrics_reference.md, keep the explorer and export steps (tools_reference.md:47-51) in
  tools_reference.md, and replace tools_reference.md:45 by a one-line link.

### Ideas, top to bottom

#### Metrics Explained (README.md:1580-1598, 19 lines)
- **Keep in README**: a short "## Metrics Explained" section at the same place (the user asked for it at the bottom and
  the table of contents points there): one scenario sentence ("Before choosing between two checkpoints, …") and a
  pointer table of the ten tasks, about 14 lines:

  | Task | Headline metric (names the best checkpoint) | Other metrics of the report | Explained in |
  |---|---|---|---|
  | CLS | `Accuracy` (`Subset Accuracy` multi-label) | balanced accuracy, macro F1, top-5, per-class P/R/F1, ROC AUC, AP, thresholds at P99, recall intervals | metrics_reference.md#classification-cls-m2o-cmp |
  | M2O, CMP | `Accuracy` over sequences / pairs | as CLS | same + #sequences-pairs-and-frames |
  | DET | `mAP@0.5/area` | mAP at 4 IoU, by size, mAR, per-class recall, confusion matrix, proposal and anchor recall | #object-detection-and-instance-segmentation-det-ins |
  | INS | `Mask mAP@0.5/area` | DET metrics for masks and boxes | same + #mask-iou-and-mask-map |
  | KPT, SEG, M2M, SEL, SSL | (rows of README.md:1590-1594, part 7's range) | … | … |

  It is the table of README.md:1586-1595 with one column added ("Other metrics") so that a reader knows from the
  README alone which numbers exist, and the anchors pointing to the new document.
- **Cut**: README.md:1582 "This section takes them one by one … The figures and animations were drawn from these same
  toy examples" (describes the document, goes to its own introduction); README.md:1584 percentages-versus-0-to-1
  sentence (moves once to the metrics document); README.md:1597 recall-interval sentence (the table links it). The
  table of contents lines README.md:45-47 and the entries of the other families collapse into one line.
- **Move**: README.md:1582-1584 to the introduction of metrics_reference.md (where the headline metric is used:
  `evaluate_checkpoint.py` best line, ringed iteration of `evaluation_curves.png`, `best_` in MLflow). Alternative
  worth naming: put the pointer table in "## Evaluating Checkpoints" (README.md:766) where the reader first meets
  `Results.txt`, and drop the end section; rejected only because the user asked for the end section by name.
- **Illustrate**: none in the README; the table is the pointer.
- **Estimated length after**: 14 lines in the README (the whole section, all ten tasks, so part 7's range keeps 0).

#### Classification (CLS, M2O, CMP) (README.md:1599-1604, 6 lines, 214 words)
- **Keep in README**: nothing (moved).
- **Cut**: the headline-metric sentence of README.md:1603 ("its headline metric, `Accuracy` …, names the best
  checkpoint …, `test/best_Accuracy` in MLflow"), repeated at README.md:1585-1587 and README.md:1635; the percentage
  sentence (third copy, README.md:1584).
- **Move**: the rest of README.md:1603 (probabilities at temperature 1, argmax, `compute_evaluation_results`, which files
  always exist versus with `evaluation.per_class_report: True`) to metrics_reference.md, as the caption of the new
  overview figure below plus one "Where" table for the whole family.
- **Illustrate**: **new `metrics_classification_overview.png`** "From logits to every classification metric": a flow
  diagram, logits → (÷ T) → softmax / sigmoid → two branches. Branch 1 "most probable class": confusion matrix →
  accuracy, balanced accuracy, P/R/F1, macro F1, plus a "top-5" arrow from the ranking. Branch 2 "one class against
  the rest, every score a threshold": ROC curve → ROC AUC, PR curve → AP, threshold sweep → `Threshold @ P99 %`,
  `Recall @ P99 %` → `class_thresholds.json`. Each box carries its report name and file; boxes needing
  `per_class_report` drawn dashed; branch 2 tinted "changes with the temperature", branch 1 "never changes". Drawn
  with matplotlib boxes and arrows, no data needed. It replaces README.md:1603 and the "never changes / changes"
  bullets of the temperature (README.md:1745-1746), and tells at a glance why the accuracy and the thresholds answer
  different questions.
- **Estimated length after**: 4 lines in metrics_reference.md (scenario, figure, caption, one sentence on M2M frames).

#### Confusion Matrix (README.md:1605-1617, 13 lines, 285 words)
- **Keep in README**: nothing.
- **Cut** (textbook for the audience): "What it holds" (rows true, columns predicted, diagonal right) and "how to read a
  row and a column" (README.md:1609, 1611) reduce to one clause. Also README.md:1610's description of the 35-class
  `confusion_matrix.png` margins, which README.md:776 ("Evaluating Checkpoints") already gives.
- **Keep in metrics_reference.md** (repository-specific): `bincount` of `true × K + predicted`; above 1,000 classes
  (`MAXIMUM_DENSE_CONFUSION_CLASSES`) `confusion_cells.csv` instead; no confusion matrix in multi-label mode;
  `Errors/<TRUE>-TO-<PREDICTED>/`. Note README.md:776 says the same 1,000-class fact: keep it in one place
  (metrics_reference.md) and link from Evaluating Checkpoints.
- **Illustrate**: `README/metrics_classification_confusion_matrix.png` already carries recall, precision, accuracy,
  balanced accuracy and macro F1 with their fractions (21 / 30, 21 / 27, 78 of 100, mean of the 4 recalls). Its caption
  (README.md:1616) is enough; it becomes the single figure of the next four subsections.
- **Estimated length after**: 5 lines.

#### Precision, Recall And F1 (README.md:1618-1632, 15 lines, 320 words)
- **Cut**: the three formulas (README.md:1622-1626, textbook) and the "Toy example" bullet README.md:1630, which repeats
  the confusion matrix figure and caption (21 / 30, 21 / 27).
- **Keep**: the edge cases (never predicted → precision 0, no test sample → recall 0, `—` listed last), the report and
  CSV names (go into the family "Where" table), and the "How to read" real-run example (parallelogram 45.88 % precision
  for 63.83 % recall, README.md:1631) in one sentence.
- **Move**: every "Where" bullet of the family (README.md:1610, 1629, 1637, 1645, 1676, 1691, 1705, 1758) into **one
  Markdown table** at the top of the family: metric | report name | `class_metrics.csv` column | TensorBoard / MLflow
  tag | needs `per_class_report`. About 12 rows replace eight bullets of 40 to 80 words each.
- **Illustrate**: confusion matrix figure above.
- **Estimated length after**: 3 lines (+ its rows in the shared table).

#### Accuracy (README.md:1633-1640, 8 lines, 184 words)
- **Cut**: the definition (textbook), the headline sentence (third copy, see README.md:1603), the toy bullet
  README.md:1638 (in the figure), and the 90-circles example (README.md:1639), which the balanced accuracy figure shows.
- **Keep**: one repository-specific warning: the accuracy printed in the console during the training is that of the
  training and validation batches, not of the test split (README.md:1639).
- **Estimated length after**: 2 lines. Recommend merging Accuracy, Balanced Accuracy and Macro And Micro Averages into
  one subsection "Accuracy, Balanced Accuracy And Macro F1" around the balanced accuracy figure.

#### Balanced Accuracy (README.md:1641-1653, 13 lines, 316 words)
- **Cut**: the second half of the toy bullet README.md:1646 (90 circles / 10 stars), word for word in the caption of
  `metrics_classification_balanced_accuracy.png` (README.md:1652); "chance is 1 / K" (textbook).
- **Keep**: a class without test samples is left out of the mean while its wrong predictions still count (matches
  scikit-learn); the warning that it ignores precision; `draw_report_figures.py` keeps the checkpoint on it.
- **Illustrate**: `README/metrics_classification_balanced_accuracy.png` as is.
- **Estimated length after**: 3 lines (inside the merged subsection).

#### Macro And Micro Averages (README.md:1654-1662, 9 lines, 269 words)
- **Cut**: the definitions of macro and micro (textbook) and the toy bullet README.md:1660 (the macro F1 75.42 % and
  47.37 % are in both figures).
- **Keep**: the repository rules: `Macro F1` counts a class predicted without any test sample with F1 0 (as
  scikit-learn); the multi-class report has no micro metric because micro equals accuracy; the multi-label report
  gives `Micro F1`.
- **Estimated length after**: 2 lines (inside the merged subsection).

#### Top-k Accuracy (README.md:1663-1670, 8 lines, 222 words)
- **Cut**: the definition and the three-image toy bullet README.md:1668 (textbook, no figure needed).
- **Keep**: k fixed at 5 (`TOP_K_ACCURACY_CLASSES`) and clipped to the classes, no multi-label version; "with 5 classes
  or fewer it is always 100.00 % and says nothing" (M2O 4 classes, CMP 2 classes); the 81.76 % against 99.12 % gap of
  the 35-class report as the one reading example.
- **Estimated length after**: 2 lines.

#### ROC Curve And ROC AUC (README.md:1671-1679, 9 lines, 387 words)
- **Cut**: the construction of the curve and the AUC as the pairwise ranking probability (README.md:1675, textbook);
  the toy bullet README.md:1677 (the GIF caption README.md:1698 has the 97.40 %).
- **Keep**: one-against-the-rest, tied scores accepted together, `—` when undefined, `Macro ROC AUC` over the classes
  defining it, `score_curves.py` functions; the "How to read" trap that the ROC AUC hides rarity (trapezoid 97.11 % ROC
  AUC against 38.31 % AP, README.md:1678) and that with two classes (CMP) both classes share one ROC AUC.
- **Estimated length after**: 3 lines.

#### Average Precision Of A Class (README.md:1680-1699, 20 lines, 413 words)
- **Cut**: the AP formula block README.md:1686-1688 (scikit-learn's `average_precision_score`, textbook once named);
  the toy bullet README.md:1692 (in the GIF caption); "chance is the share of the class" can stay as half a sentence.
- **Keep**: no interpolation here, unlike the detection AP (one line, it is the confusion readers hit); the grid layout
  of `precision_recall_curves.png` (one plot up to 10 classes, panels above, 100 lowest above 100 classes).
- **Illustrate**: `README/metrics_classification_threshold_sweep.gif` stays, shared by ROC, AP and thresholds.
- **Estimated length after**: 3 lines + the GIF (4 lines of HTML).

#### Threshold And Recall At A Target Precision (README.md:1700-1717, 18 lines, 533 words)
- **Keep** (fully repository-specific, and read by the deployment): the selection rule (cuts reaching the target with
  at least one right sample, most recall, then the highest threshold), recall over every sample of the class,
  `null` / `—` for a class never reaching it, a list of targets with the first as the main one, the
  `class_thresholds.json` block (README.md:1708-1714, the deployment contract), and the pitfalls of README.md:1716
  (threshold can be below 0.5 one against the rest, holds only at its temperature, `null` classes must be handled by
  the server).
- **Cut**: "With fewer than 100 accepted samples, 99 % precision means that none is wrong" stays; the toy sentence
  README.md:1706 (already the last frame of the GIF) goes; the rounding rule of `precision_recall_thresholds.png`
  moves to the "Where" table note. tasks_reference.md:117 repeats the rule: keep it here only.
- **Illustrate**: extend the last frame of `metrics_classification_threshold_sweep.gif` with a text box stating the
  rule in one line ("precision ≥ 99 % with ≥ 1 star → most stars kept → highest such threshold: 0.4414") so the prose
  states it once and the frame shows it applied.
- **Estimated length after**: 10 lines (JSON block included).

#### Multi-Label Metrics (README.md:1718-1738, 21 lines, 398 words)
- **Keep**: decision at 0.5 (`MULTI_LABEL_THRESHOLD`), positive when label ≥ 0.5, `Subset Accuracy` as headline, what is
  absent (no confusion matrix, no top-k), and the console trap (README.md:1737: the console `accuracy` counts every class
  decision apart, far above the subset accuracy).
- **Cut**: the definitions of micro and macro F1 (README.md:1726-1727, textbook).
- **Illustrate**: **new `metrics_classification_multi_label.png`** replacing the 6-line table README.md:1729-1734 and
  the toy sentence README.md:1736: a 4 × 4 heatmap (images × triangle, square, circle, star) of the toy probabilities,
  each cell outlined green (right positive), red (missed label), orange (added class), the 0.5 decision marked, a right
  column ✓/✗ "every decision right", a bottom row of per-class F1, and a side panel: subset accuracy 50.00 %, micro F1
  2 × 5 / 13 = 76.92 %, macro F1 62.50 %. Data: the toy values already written in README.md:1731-1736, numbers from
  `compute_multi_label_metrics`.
- **Estimated length after**: 6 lines + figure.

#### Logit Temperature (README.md:1739-1753, 15 lines, 549 words)
- **Keep**: `softmax(logits / T)`, every evaluation at T = 1 (written in `class_thresholds.json`), which outputs are at
  T = 1 (report AUC, AP, thresholds) versus at the saved temperature (`draw_report_figures.py` curves), and that the
  server must divide by the same T.
- **Cut**: the explorer and export workflow sentence of README.md:1743 (it is tools_reference.md:44-51 already); the
  "What it never changes" list and most of "What it changes" (README.md:1744-1745), carried by the overview figure and
  the extended temperature figure; the toy bullet README.md:1746 (the left panel of the figure shows 0.998 / 0.950 /
  0.665).
- **Move**: tools_reference.md:45 (its own explanation of the temperature) becomes a link to this subsection, so the
  effect is explained once.
- **Illustrate**: extend `README/metrics_classification_temperature.png` with a strip under both panels: "same at every
  T: most probable class, confusion matrix, accuracy, balanced accuracy, P/R/F1, top-k, multi-label decisions at 0.5 |
  moves: scores, thresholds; with more than two classes also Recall @ P99 %, ROC AUC, AP". The right panel already shows
  the threshold and recall moving (0.6294 / 0.4414 / 0.3829, 75 % → 50 %).
- **Estimated length after**: 5 lines + figure.

#### Recall Intervals (README.md:1754-1780, 27 lines, 914 words)
- **Move inside the document**: it applies to all ten tasks, so it becomes its own top-level section of
  metrics_reference.md ("Recall Intervals (every task)") instead of sitting under Classification; the README pointer
  table links it once.
- **Cut** (textbook): the Clopper-Pearson formula block README.md:1763-1766 (name it and its function, the formula is in
  every statistics text) and the series of widths at 10 / 100 / 1,000 / 500 / 2,000 samples (README.md:1767), which the
  left panel of the figure plots.
- **Keep** (repository-specific, non-standard): which tasks use Clopper-Pearson and which use Korn-Graubard over groups
  (README.md:1760, 1768), `CONFIDENCE_LEVEL`, columns `recall_lower` / `recall_upper` and `<recall>_lower` for DET/KPT,
  not sent to TensorBoard, the bootstrap seed 0 and 1,000 resamples, the degenerate cases (variance 0, recall 0 or
  100 %, a single group), "the number that matters is the groups holding the class".
- **Illustrate**: replace the four numbered steps README.md:1769-1773 (the densest prose of the range) by extending
  `README/metrics_classification_recall_intervals.png` with a procedure strip on the left: (1) 20 sequences drawn as rows
  of frames, found / missed; (2) 1,000 resampled sets of rows → histogram of the class recall; (3) variance v →
  effective size `p(1 − p) / v × (z / t)²`, capped at the true samples; (4) Clopper-Pearson at that size, drawn as an
  interval bar next to the over-narrow bar of the frames taken as independent. Data: the three scenarios already in
  README.md:1774 (scattered, 5 in 20 sequences, 10 whole sequences: effective sizes 976, 227, 100), computed by
  `compute_bootstrap_recall_intervals`. The prose then keeps one sentence per step.
- **Estimated length after**: 11 lines + figure.

#### Sequences, Pairs And Frames (README.md:1781-1788, 8 lines, 261 words)
- **Keep**: what one sample is per task (M2O sequence, CMP pair, M2M labelled frame without padding and unmapped
  frames), which interval each uses, M2O without temperature explorer, M2M pointers to the edit score and segment F1.
  These are repository facts; the M2O and CMP bullets can become a 3-row table "task | one sample is | interval over".
- **Cut**: the tutorial-run scores (M2O 99.00 % / 99.04 %, CMP 95.00 %, README.md:1785-1786): they say nothing about the
  metric and go stale on every retraining; the `appearing_kite` 0.00 to 0.75 % example can stay as the reading example.
  The CMP "top-5 always 100 % and one shared ROC AUC" is already said in Top-k and ROC (README.md:1669, 1678): keep once.
- **Move**: tasks_reference.md:337, 348, 367 ("the metrics of CLS on the sequences / frames / pairs") become links here.
- **Estimated length after**: 5 lines (table).

#### Object Detection And Instance Segmentation (DET, INS) (README.md:1789-1791, 3 lines, 44 words)
- **Keep in README**: nothing (moved).
- **Illustrate**: **new `metrics_detection_overview.png`** "Which detections each detection metric sees": a left-to-right
  pipeline drawn with matplotlib boxes: anchors (anchor recall, training objects, printed at launch, threshold
  `foreground_iou_threshold.min`) → proposals (proposal recall @0.5/0.7, test objects, every evaluation) → kept
  detections (score ≥ `evaluation.score_threshold` 0.05, ≤ 100 per image, per-class NMS) → class-by-class VOC matching
  at 10 IoU thresholds → AP / mAP (area, 11-point, 101-point), by size, mAR, per-class recall; a lower branch from the
  detections with score ≥ `Detector.inference.score_threshold` 0.5 → two-pass matching at IoU 0.5 → confusion matrix
  and `Errors/` folders; an INS badge "same with the mask IoU". It replaces the three places where the two thresholds
  are explained again (README.md:1823, 1902, 1928-1930) and shows the diagnosis chain of the "How to read" bullets
  (low anchor recall → anchors; proposal recall ≫ mAR → heads; mAR ≫ mAP → ranking; mAR high but many missed in the
  confusion matrix → deployment threshold).
- **Estimated length after**: 4 lines (scenario, figure, caption).

#### Intersection Over Union Of Two Boxes (README.md:1792-1814, 23 lines, 406 words)
- **Cut** (textbook): the paragraph "answers how well this box sits on this object, 1 identical, 0 disjoint"
  (README.md:1795) and the formula block README.md:1798-1801; the toy sentence README.md:1806 (each panel of
  `metrics_detection_box_iou.png` prints its IoU and fraction, the caption repeats them a second time).
- **Keep** (repository-specific): continuous coordinates without "+ 1", the union clamped instead of an epsilon (exact
  0.75 reaches 0.75), float64 in the evaluation (exact 0.95 counts at 0.95), and where the IoU is used (matching, target
  assignment, NMS, anchor and proposal recall).
- **Illustrate**: extend `README/metrics_detection_box_iou.png` with a fifth panel for the "How to read" claim of
  README.md:1813: a 20 × 20 object with a box one pixel short (0.95) next to a 200 × 200 object one pixel short (0.995),
  so that "small objects fall first at strict thresholds" is seen, and the sentence shrinks to a clause.
- **Estimated length after**: 8 lines + figure.

#### Matching Detections To Ground Truths (README.md:1815-1828, 14 lines, 471 words)
- **Keep** (the most important repository-specific fact of the family): class by class, best ground truth computed once,
  greedy by score with a stable sort, the **PASCAL VOC rule and why the numbers are not those of the COCO tools**
  (README.md:1822), and which detections enter (`evaluation.score_threshold` 0.05, `maximum_detections` 100, after NMS).
- **Cut**: the two-parcel toy paragraph README.md:1825 as prose (moves into a figure), and the "How to read" pointer to
  the GIF (README.md:1827, the GIF follows anyway).
- **Illustrate**: **new `metrics_detection_voc_matching.png`**: the two parcels `[0, 0, 100, 100]` and
  `[40, 0, 140, 100]`, detection 1 (0.9) on parcel 1, detection 2 (0.8) overlapping parcel 1 by 0.74 and parcel 2 by
  0.60; left panel "this repository (VOC)": detection 2's best ground truth already taken → false positive, AP@0.5
  50.00 %, recall 50.00 %; right panel "COCO tools": detection 2 takes the free parcel 2 → true positive, AP@0.5
  100.00 %. Numbers from `evaluate_detections`. It gives the VOC-versus-COCO pitfall, mentioned again at README.md:1879
  and README.md:1910, one visual home and lets those repetitions become links.
- **Estimated length after**: 7 lines + figure.

#### Precision-Recall Curve And Average Precision Of A Class (README.md:1829-1859, 31 lines, 748 words)
- **Cut** (textbook): the precision_k / recall_k block README.md:1834-1838 and "every detection is one point"
  (README.md:1839); the descriptions of the three methods can each lose half their words (VOC 2010 area with envelope,
  VOC 2007 11-point, COCO 101-point are standard); the eight-detection walk-through README.md:1845 and the arithmetic
  block README.md:1846-1850 (the GIF shows every step with its colour and shades the area and samples; the caption
  README.md:1855 gives the three results).
- **Keep**: the three rows `Area`, `11 Point`, `101 Point` printed by the report with `Area` as headline, the edge cases
  (no detection → AP 0, no ground truth → `—` and out of every mean), the per-class table and its columns (into a family
  "Where" table), and the real-run reading (kite AP 32.81 % with recall 92.65 %, none reaching the deployment threshold,
  README.md:1858), the best example of the whole range of "high recall, low AP".
- **Move**: README.md:1856 (per-class table, `class_metrics.csv` columns, TensorBoard `AP@0.5/<class>`) and the
  equivalent "Where" bullets README.md:1869-1870, 1893, 1909, 1919-1920, 1933, 1953, 1967, 1987 into **one detection
  "Where" table**: metric | report table | `class_metrics.csv` / own CSV columns | TensorBoard tag | INS mask prefix.
  This also absorbs tasks_reference.md:166-176.
- **Illustrate**: `README/metrics_detection_matching.gif` as is.
- **Estimated length after**: 9 lines + GIF.

#### mAP@0.5, mAP@0.75, mAP@0.95 And mAP@[0.5:0.95] (README.md:1860-1880, 21 lines, 467 words)
- **Cut**: the definition block README.md:1864-1867 (textbook COCO-style), the headline sentence README.md:1870 (fourth
  copy of "names the best checkpoint", in the README pointer table), and the toy sentence README.md:1872 (every number
  is in `metrics_detection_iou_thresholds.png`, which even has a "Report" panel).
- **Keep**: macro mean (class without ground truth left out, never detected counts 0), the ten thresholds constant
  `INTERSECTION_OVER_UNION_THRESHOLDS`, the scalar names `mAP@0.5/area` / `.../101_point`, the real-run reading
  (75.25 / 68.29 / 0.49 / 55.05 % and the averaged weights gaining most at strict thresholds) and the VOC pitfall as a
  link to the matching figure.
- **Illustrate**: extend the "Report" panel of `README/metrics_detection_iou_thresholds.png` with the mAR row
  (mAR@0.5 80.00 %, mAR@0.75 60.00 %, mAR@0.95 20.00 %, mAR@[0.5:0.95] 52.00 %) and the circle's per-class recall
  with its interval (80.00 %, 28.36 % to 99.49 %): one figure then carries mAP, mAR and per-class recall of the same
  eight detections, and the toy paragraphs of README.md:1907 and README.md:1922 go.
- **Estimated length after**: 6 lines + figure.

#### AP And mAP By Object Size (README.md:1881-1898, 18 lines, 438 words)
- **Cut**: the COCO size ranges block README.md:1885-1889 (textbook; one inline clause "COCO small < 32², large ≥ 96²"),
  and the toy paragraph README.md:1891.
- **Keep**: the unit of the area per case (network input pixels for DET and INS boxes, mask pixels for the mask mAP,
  original pixels with `evaluation.original_resolution`), the COCO ignore rule as implemented by
  `select_object_size_detections` (other sizes ignored, a detection whose best ground truth of the size is taken stays a
  false positive), `—` for a size without objects, and the real-run reading (small 71.25 %, medium 80.09 %, large
  73.17 %, link to anchor recall by size).
- **Illustrate**: add the size label (S / M / L, with the 24 × 24, 60 × 60, 100 × 100 sizes) to the five circles of
  `metrics_detection_matching.gif` and a size table in its last frame (mAP@0.5 small 100.00 %, medium 55.56 %, large
  100.00 %), instead of a new figure.
- **Estimated length after**: 6 lines.

#### Mean Average Recall (README.md:1899-1914, 16 lines, 311 words)
- **Cut**: the formula block README.md:1903-1907 (textbook), the toy paragraph README.md:1909 (moved into the
  iou_thresholds report panel, above), "does not depend on the AP method" (obvious).
- **Keep**: counted over all kept detections from `evaluation.score_threshold`, close to COCO AR@100 but with the VOC
  matching, a very low `evaluation.score_threshold` raises it without changing the deployed detector, and the reading
  "mAR@0.5 96.65 % against mAP@0.5 75.25 %: found but ranked below false positives".
- **Estimated length after**: 4 lines.

#### Per-Class Recall At IoU 0.5 And 0.75 (README.md:1915-1925, 11 lines, 288 words)
- **Cut**: the toy paragraph README.md:1922 (moves into the iou_thresholds figure); the `class_metrics.csv` and report
  table bullets go to the detection "Where" table (they are repeated in tasks_reference.md:169).
- **Keep**: same matching as the mAR, true positives recovered as `round(recall × ground truths)`, Clopper-Pearson
  interval (link to the intervals section), INS prints `Mask` and `Box Per Class Recall`, and the reading "recall at
  0.75 far below 0.5 = found but poorly boxed" (L shape 94.18 % against 75.47 %).
- **Estimated length after**: 4 lines.

#### Detection Confusion Matrix (README.md:1926-1943, 18 lines, 580 words)
- **Keep** (repository-specific): computed at `Detector.inference.score_threshold` (0.50), not at the evaluation
  threshold; the two-pass matching at IoU 0.5 (own class first, then any class) of `match_image_detections`; the
  `background` row and column; the `Errors/FALSE-POSITIVE`, `Errors/MISSED`, `Errors/<TRUE>-TO-<PREDICTED>` folders; the
  report note format; INS by mask IoU; the identity right + wrong class + missed = test objects and the reading that
  most missed objects have a detection under 0.50 (README.md:1942).
- **Cut**: the toy paragraph README.md:1935 nearly whole: `metrics_detection_confusion_outcomes.png` draws the image,
  the matrix, the four counts and the AP of the same detections; keep one sentence ("the star at 0.30 is missed at the
  deployment threshold but finds its star in the mAP, which matches from 0.05").
- **Illustrate**: `README/metrics_detection_confusion_outcomes.png` as is (it is self-explanatory).
- **Estimated length after**: 8 lines + figure.

#### Proposal Recall (README.md:1944-1958, 15 lines, 367 words)
- **Keep**: fully repository-specific (no textbook equivalent): best IoU of every test object with the valid proposals
  at inference (padding left out), thresholds 0.5 and 0.7 (`PROPOSAL_RECALL_THRESHOLDS`), rows per size and per block,
  `proposal_recall.csv` columns, INS at network resolution even with `original_resolution`, and the diagnosis "proposal
  recall far above the mAR → heads; low → anchors, proposal counts or selection".
- **Cut**: the formula block README.md:1949-1952 (two lines of prose say it), the toy sentence README.md:1955 (it refers
  to the anchor figure without being drawn there), and half of the real-run numbers of README.md:1957 (keep 98.39 % /
  95.70 % overall and the small objects; the four block values and "238.69 of 250 proposals" go).
- **Illustrate**: the overview pipeline figure places it; optionally draw the five proposals on the anchor figure if
  the toy numbers are wanted (two of block 4, three of block 5), otherwise drop the toy numbers.
- **Estimated length after**: 6 lines.

#### Anchor Recall (README.md:1959-1978, 20 lines, 466 words)
- **Keep**: fully repository-specific: measured once per launch on the training objects from the annotations without
  augmentation (`analyse_training_data`), exact best-anchor IoU (`Anchors.compute_best_overlaps`, nearest grid anchor per
  scale and ratio), `All Levels`, console table and `anchor_recall.csv`, the diagnosis mode
  `compute_anchor_coverage`, "an object under the threshold everywhere still gets its single best anchor", and the
  reading (78.02 % overall, 43.99 % of the small objects) with the tuning advice (one sentence).
- **Cut**: the formula block README.md:1964-1967 (to a sentence), the toy paragraph README.md:1972: the figure already
  draws the grids, the five boxes with their IoU, and both tables (by size and per object).
- **Illustrate**: `README/metrics_detection_anchor_recall.png` as is. Its caption (README.md:1976) names
  `compute_best_anchor_overlaps` (region_recall.py:73) and the prose `Anchors.compute_best_overlaps` (anchors.py:101):
  both exist, name one of them only in the moved text.
- **Estimated length after**: 8 lines + figure.

#### Mask IoU And Mask mAP (README.md:1979-1998, 20 lines, 560 words)
- **Cut**: the formula (textbook) and the toy paragraph README.md:1990 (the figure and its caption give the hollow
  square 1.00 / 0.64 and the L shape 0.69 / 0.81); the headline sentence README.md:1988 (pointer table); the list of
  `Mask` table names README.md:1987 shrinks to "every DET table, prefixed `Mask`, printed before the `Box` tables" and
  goes into the detection "Where" table as a prefix column.
- **Keep**: how the 28 × 28 probabilities are pasted into the box at ground-truth mask size and thresholded at
  `MaskHead.inference.mask_threshold`, exact counts and float64, objects sized by mask area, box scalars keeping DET
  names so INS and DET compare, thresholds / confusion matrix / errors matched by masks, and the reading of
  README.md:1997 (masks better at 0.95, different size populations 8,080 against 5,778, a low mask IoU on hollow or thin
  classes points at the mask head). tasks_reference.md:206-208 repeats this: links here.
- **Illustrate**: `README/metrics_detection_mask_iou.png` as is.
- **Estimated length after**: 8 lines + figure.

#### Evaluation At The Original Resolution (README.md:1999-2010, 12 lines, 340 words)
- **Keep**: what changes with `evaluation.original_resolution` (objects rebuilt from the annotation at image size,
  only objects with no pixel removed, boxes restored by `restore_detection_boxes` and clipped for the box mAP but whole
  for pasting, sizes in original pixels, proposal recall and images staying at network resolution, more memory), and
  "compare runs evaluated at the same resolution only". This setting is better placed with the configuration: the
  configuration reference should hold the key and link here for its effect on the metrics.
- **Cut**: the toy sentence README.md:2007 as prose.
- **Illustrate**: **new `metrics_detection_original_resolution.png`**: a 1280 × 720 image with a 60 × 60 object (3,600
  square pixels, "medium") beside its 512-pixel letterboxed input where the object becomes the box `[240, 232, 264,
  256]`, 24 × 24 = 576 square pixels ("small"), with the padding bands shaded, plus a thin object that vanishes at the
  network resolution but stays (and counts as missed) at the original one. Data: `preprocess_boxes` on that one box.
- **Estimated length after**: 6 lines + figure.

### Textbook versus repository-specific (summary of the rule applied above)

| Cut to a name or one clause (textbook for 4+ years CV/ML) | Keep in full (repository behaviour, decides how to read a number) |
|---|---|
| Confusion matrix layout; precision, recall, F1, accuracy, balanced accuracy, macro / micro, top-k definitions | Edge cases (`—`, 0, classes left out of means), console accuracy ≠ test accuracy, top-5 clipped and useless at ≤ 5 classes, no micro in multi-class |
| ROC curve, ROC AUC, scikit-learn AP formula | One-against-the-rest on the class probability, ties accepted together, no interpolation (unlike detection AP) |
| Clopper-Pearson formula and 1 / √n narrowing | Which tasks use Clopper-Pearson versus Korn-Graubard, the bootstrap procedure and its degenerate cases, columns, no TensorBoard |
| IoU formula, PR curve points, VOC area / 11-point / COCO 101-point, COCO size ranges, mAR formula, mask IoU formula | No "+ 1", clamped union, float64; **VOC matching, not COCO numbers**; score threshold 0.05 versus deployment 0.5; size unit per case; COCO ignore rule; ≈ AR@100 |
| — | Threshold selection rule and `class_thresholds.json`; temperature invariances; multi-label 0.5 decisions; detection confusion matrix two-pass matching; proposal and anchor recall; mask pasting; original resolution |

Real-run numbers (35-class README classifier, tutorial DET and INS runs) are kept only where they illustrate a
diagnosis (one per metric at most); scores that only state a level (README.md:1785-1786, the block values of
README.md:1957) go, since they go stale on every retraining. Toy numbers live in the figures; the prose keeps a toy
example only where no figure draws it.

### Open points for the decision
- The scripts that drew the 11 `metrics_classification_*` / `metrics_detection_*` figures are not in the repository
  (added in c8486735 with the figures only; no tracked file names `metrics_detection_box_iou`). Extending a figure means
  re-writing its script first; new figures need a home such as `scripts/documentation/metrics_figures/`.
- Whether the pointer table stays at the end of the README or goes into "Evaluating Checkpoints" (README.md:766):
  recommended to keep it at the end (the user asked for the section there).

### Summary of the part

| Section | README lines now | README lines after | Lines in metrics_reference.md after | Main action |
|---|---|---|---|---|
| Metrics Explained (intro) | 19 | 14 | 6 | Pointer table task → headline → other metrics → link (covers all ten tasks); rest moves |
| Classification (intro) | 6 | 0 | 4 | New overview flow figure replaces the dense paragraph; one "Where" table for the family |
| Confusion Matrix | 13 | 0 | 5 | Textbook cut; figure carries P/R/accuracy/balanced/macro F1 |
| Precision, Recall And F1 | 15 | 0 | 3 | Formulas and toy cut; "Where" bullets into the shared table |
| Accuracy | 8 | 0 | 2 | Merged with balanced accuracy and macro F1 |
| Balanced Accuracy | 13 | 0 | 3 | Toy duplicated in caption cut |
| Macro And Micro Averages | 9 | 0 | 2 | Keep only the repository rules |
| Top-k Accuracy | 8 | 0 | 2 | Keep k = 5 clipped and the ≤ 5 classes trap |
| ROC Curve And ROC AUC | 9 | 0 | 3 | Keep rarity trap and two-class note |
| Average Precision Of A Class | 20 | 0 | 7 | Formula cut; GIF kept |
| Threshold At A Target Precision | 18 | 0 | 10 | Kept (deployment contract); rule shown in GIF last frame |
| Multi-Label Metrics | 21 | 0 | 8 | New decision-grid figure replaces the table |
| Logit Temperature | 15 | 0 | 7 | Invariance strip on the figure; tools_reference.md:45 links here |
| Recall Intervals | 27 | 0 | 15 | Own section for all tasks; procedure strip replaces the 4 steps |
| Sequences, Pairs And Frames | 8 | 0 | 5 | Small table; stale scores cut |
| DET / INS (intro) | 3 | 0 | 4 | New pipeline figure: which detections each metric sees |
| Box IoU | 23 | 0 | 10 | Formula cut; fifth panel for small objects |
| Matching | 14 | 0 | 9 | New VOC-versus-COCO figure |
| PR curve and AP | 31 | 0 | 13 | Walk-through and arithmetic cut (GIF has them); one detection "Where" table |
| mAP at thresholds | 21 | 0 | 8 | Toy cut; figure report panel gains mAR and per-class recall |
| By object size | 18 | 0 | 6 | Size labels and table in the GIF last frame |
| Mean Average Recall | 16 | 0 | 4 | Formula and toy cut |
| Per-class recall | 11 | 0 | 4 | Toy into figure; columns into table |
| Detection confusion matrix | 18 | 0 | 10 | Toy paragraph to one sentence |
| Proposal recall | 15 | 0 | 6 | Formula and half the run numbers cut |
| Anchor recall | 20 | 0 | 10 | Toy paragraph cut (figure has it) |
| Mask IoU and mask mAP | 20 | 0 | 10 | Toy and table-name list cut |
| Original resolution | 12 | 0 | 8 | New letterbox figure |
| **Total** | **431 (11,070 words)** | **14** | **about 190 (about 4,500 words, figures included)** | Move to documentation/reference/metrics_reference.md and shrink by ~60 % |

Line counts of the moved document include figure HTML (4 lines each); its paragraphs become shorter, so the word count
falls more than the line count (from 10,750 words to about 4,500).

## Part 7: Metrics Explained, KPT To SSL (README.md:2011-2643)

The range holds 633 lines and 14,896 words, which is 31 % of the README's words in 24 % of its lines, because the paragraphs are long. These ideas assume what part 6 is expected to propose: the whole "Metrics Explained" section moves to `documentation/reference/metrics_reference.md`, and the README keeps one pointer row per task in a short table. The README would then keep **0 lines** of this range. The "Estimated length after" values below are for the moved text in the new document.

### Cross-cutting ideas (apply to every block below)

1. **One names table per task instead of the "Its name is ..." bullets.** Every subsection repeats where its value is written (report, `class_metrics.csv`, TensorBoard, MLflow): README.md:2069, 2088, 2100, 2112, 2122, 2135-2136, 2163, 2287, 2399, 2444, 2457, 2476 (second half), 2520, 2541, 2555, 2583, 2589. The SEG section already has the right format (README.md:2155-2163). Each task gets one table with the columns `Metric | Results.txt | class_metrics.csv | TensorBoard | MLflow`, so every other subsection loses its names sentence. The same facts are also split across `tasks_reference.md:279-280, 317, 391, 437` and `configuration_reference.md:141-143, 287-291`. They should live in this table only, and those lines become links to it.
2. **Cut every worked-example paragraph that repeats its figure caption.** In each of the following, the caption under the figure already gives the same numbers: README.md:2056 (stick figure), 2071 and 2090 (toy matching), 2171 (toy confusion matrix), 2190, 2220 (boundary toy), 2335 (three predictions), 2357 (segment matching), 2377 (decodings), 2401, 2430, 2446, 2478, 2507 and 2514 (MD5 listing). Keep the figure and its caption, and drop the paragraph. This saves about 2,000 words.
3. **Keep at most one tutorial-run number per task.** Sentences like "in the tutorial run at iteration 6000 ..." go stale with every rerun, and the tutorials already show the real reports: README.md:2078, 2092, 2104, 2114, 2149, 2184, 2197, 2209, 2227, 2275, 2281, 2308, 2318, 2342, 2364, 2384, 2416, 2432, 2453, 2461, 2485, 2535, 2543, 2563, 2585, 2611, 2641. Two of them already confuse a newcomer:
   - "the README run" (README.md:2368-2485) is never defined. It is `tutorials/run_snapshots/readme_screenshots/SEL_..._FPC1-SLW1`.
   - `readme_screenshots/SEG_..._MC9` (README.md:2184) is a path without its `tutorials/run_snapshots/` prefix.
4. **Keep repository behaviour, cut textbook definitions.** IoU, Dice, pixel accuracy, frequency-weighted IoU, PCK, the COCO OKS formula, the k-NN vote, the InfoNCE and Barlow Twins losses and the action-segmentation edit score and F1@k are known to an engineer with 4+ years of CV/ML. Each gets one line, plus a paper link only where the name alone is ambiguous. The text keeps what differs from the textbook or from a library: which pixels, objects, frames and classes are counted or left out; ties; NaN and `—`; pooling over the split; and choices that make values not comparable with COCO, mmseg or papers.
5. **A "which number answers which question" matrix per task** replaces the "How to read it" caveat prose. It is a small markdown table of checkmarks, scannable at a glance. Example for KPT:

   | Metric | Uses scores | Counts false positives | Missed object counts as | Unit | Headline |
   |---|---|---|---|---|---|
   | OKS mAP@0.5/area | yes | yes | recall loss | % | yes |
   | Matched Objects | no | no | unmatched | % | |
   | PCK@0.05 | no | no | all keypoints wrong | % | |
   | Mean OKS | no | no | 0 | % | |
   | Mean Keypoint Error | no | no | left out | pixels | |

   This one table replaces README.md:2078, 2104 (last part), 2114 (first part) and 2124 (middle).
6. **Each fact in one place, shared with `tasks_reference.md`.** The task reference's "Evaluation" bullets (`tasks_reference.md:269-280` KPT, `309-317` SEG, `349-352` M2M, `383-391` SEL, `426-437` SSL) repeat short versions of the same definitions and the same toy examples. The OKS formula appears in both places (`tasks_reference.md:271` and README.md:2028), and so does the edit score example `A A A A A B B B B B` (`tasks_reference.md:352` and README.md:2335).
   - Recommendation: the task reference lists only the metric names with one phrase saying what each answers, plus the headline, each name linking to the metrics document. Formulas, examples and figures live only in the metrics document.
   - The exception is the SEL decodings (see that block), which are model behaviour: they stay in the task reference, and the metrics document links there.

---

#### Keypoint Detection (KPT) (README.md:2011-2150, 140 lines, 3,579 words)

**Intro (README.md:2011-2019)**
- **Keep**: the headline (`OKS mAP@0.5/area`) and three repository facts. Every grouped object is scored with an object score threshold of 0. The values are in network input pixels. Objects without any labelled keypoint are dropped by the loader (README.md:2013). About 3 lines.
- **Cut**: the 4-line report grid (README.md:2015-2019). The names table carries the names and the tutorials show the real report.

**Object Keypoint Similarity (OKS) (README.md:2021-2061)**
- **Keep**:
  - The formula, once (move it here from `tasks_reference.md:271`).
  - The repository-specific points:
    - Box area instead of segment area, so the values are not comparable with COCO (README.md:2032). This is the most important caveat.
    - A keypoint that was not found counts in the denominator.
    - Visibility 0 and padding positions are left out.
    - The OKS is 0 across classes.
    - The `ValueError` when merged categories give different sigmas.
    - Sigmas affect the evaluation only, not `heatmap_sigma` (README.md:2047).
  - The sigma configuration as a single line linking to `configuration_reference.md`. About 10 lines.
- **Cut**:
  - The YAML block (README.md:2039-2045), which belongs in the configuration reference.
  - The prose on tolerance against box size (README.md:2049), which `metrics_keypoints_oks_falloff.png` and its caption already show.
  - The stick figure paragraph (README.md:2056), which repeats the caption.
- **Illustrate**: keep `metrics_keypoints_oks_falloff.png` and `metrics_keypoints_stick_figure_oks.png` unchanged.
- **Estimated length after**: 18 lines (including the two figure blocks).

**Matched Objects (README.md:2063-2078)**
- **Keep**:
  - The definition, which is not COCO: a Hungarian assignment on the OKS matrix, pairs below 0.5 zeroed, scores ignored.
  - What a missed ground truth contributes to each accuracy metric (README.md:2067-2068).
  - One sentence saying it can disagree with the mAP matching. The figure shows how.
- **Cut**:
  - The toy paragraph (README.md:2071), which repeats the caption of `metrics_keypoints_matching.png`.
  - The caveat paragraph (README.md:2078), replaced by matrix row 2 (cross-cutting idea 5).
- **Illustrate**: keep `metrics_keypoints_matching.png`.
- **Estimated length after**: 8 lines.

**OKS mAP And mAR (README.md:2080-2092)**
- **Keep**: the headline sentence and "same as DET except" with three short bullets:
  - Ignored objects as in pycocotools, with the extended-box OKS (README.md:2085, compressed to 2 lines).
  - The object score is the mean keypoint score, a missing keypoint counting 0.
  - Small objects are kept, unlike COCO, and an unmatched prediction is sized by its keypoint box (README.md:2087).
  - Keep half a sentence of README.md:2092: 4,652 of the 11,004 COCO validation persons have no labelled keypoint. It justifies the pycocotools rule.
- **Cut**: the names bullet (README.md:2088), which goes to the names table, the toy paragraph (README.md:2090) and the tutorial numbers (README.md:2092).
- **Estimated length after**: 6 lines.

**PCK@0.05 (README.md:2094-2104)**
- **Keep**: three facts in one bullet list:
  - The tolerance is 0.05 × the longest side of the ground truth box, the same for every keypoint whatever its sigma.
  - The value is pooled over keypoints, so classes with many keypoints weigh more.
  - A keypoint that was not found, and every keypoint of a missed object, counts as wrong.
- **Cut**: the comparison with OKS and elongated boxes (README.md:2102), which the dotted line of the falloff figure already shows, and the rhombus example (README.md:2104). The `Labelled`/`Found` reading rule moves to the per-keypoint table block.
- **Estimated length after**: 4 lines.

**Mean OKS (README.md:2106-2114)**
- **Keep**: the identity `Mean OKS = Matched Objects × mean OKS of the matched objects` (README.md:2111). It says everything: a missed object scores 0, and each matched one lands between 0.5 and 1.
- **Cut**: README.md:2112, which goes to the names table, and the toy and tutorial examples (README.md:2114).
- **Estimated length after**: 2 lines.

**Mean Keypoint Error (README.md:2116-2129)**
- **Keep**:
  - Only keypoints that were found, in matched objects, are averaged.
  - The distances are in network input pixels.
  - There is no value when nothing is found (`—`, empty cell, no scalar).
  - One sentence on the trap: the error can fall when the model gets worse, because a drifting object leaves the mean. The GIF shows it.
- **Cut**: README.md:2122 (names) and README.md:2124, which repeats the GIF caption.
- **Illustrate**: keep `metrics_keypoints_drifting_prediction.gif`. It is the best figure of the block and the reason the paragraph can go.
- **Estimated length after**: 7 lines.

**Keypoint Accuracy Per Class And Per Keypoint (README.md:2131-2149)**
- **Keep**:
  - The two table names.
  - `class_metrics.csv` and `keypoint_metrics.csv`, which are written whatever `per_class_report` is.
  - What `Labelled` means: fewer than the number of objects when a keypoint falls outside the image.
  - The reading rule (README.md:2149, first sentence). `Found` well below `Labelled` with a small error means misses. `Found` near `Labelled` with a large error means misplacement or keypoints given to the wrong neighbouring object.
- **Cut**: the 3-row text block (README.md:2139-2147) once the figure below replaces it, and the tutorial numbers and the no-interval remark (README.md:2149, rest).
- **Illustrate**: **new figure** `metrics_keypoints_misses_versus_misplacements.png`.
  - Data: `tutorials/run_snapshots/keypoint_focal_synthetic/.../Checkpoint_Evaluation/Iteration_6000/keypoint_metrics.csv` (already tracked).
  - It is a scatter with one point per keypoint of every class. The x axis is `Found / Labelled` and the y axis is the mean error in pixels.
  - Two shaded quadrants are labelled "missed" (left, low error) and "misplaced or swapped" (right, high error). The stick figure elbows and hands are annotated at 13-19 pixels.
  - The reader sees at once which keypoints fail and how. It replaces the table excerpt and README.md:2149.
- **Estimated length after**: 8 lines.

**Estimated length after (whole KPT block)**: about 56 lines and 1,300 words, down from 140 lines and 3,579 words.

---

#### Semantic Segmentation (SEG) (README.md:2151-2284, 134 lines, 3,432 words)

**Intro and names table (README.md:2151-2165)**
- **Keep**:
  - The key repository fact: every metric is a ratio of counts pooled over the whole test split, never a mean over images (README.md:2153).
  - The headline, `Mean IoU`.
  - The names table (README.md:2155-2163). It is the template for cross-cutting idea 1.
  - One line saying that the training and validation `Pixel Accuracy` and `Mean IoU` are means of batch values and differ from the test number (README.md:2165, last sentence).
- **Cut**: the percentages-versus-0-to-1 sentence (README.md:2165, first part). Say it once in the intro of the metrics document (part 6's range).
- **Estimated length after**: 13 lines.

**Pixel Confusion Matrix And Unlabelled Pixels (README.md:2167-2178)**
- **Keep**:
  - 255 becomes `IGNORED_LABEL_INDEX`, which the letterbox padding also receives. A prediction on an ignored pixel counts nowhere.
  - The `Labelled Pixels` note and `confusion_matrix.csv`.
  - The warning that a wide 255 area hides the model's errors there (README.md:2178).
- **Cut**: the toy paragraph (README.md:2171), which repeats the caption of `metrics_segmentation_region_overlap.png`.
- **Illustrate**: keep `metrics_segmentation_region_overlap.png`.
- **Estimated length after**: 8 lines.

**Pixel Accuracy, IoU and Mean IoU, Dice, Frequency Weighted IoU (README.md:2180-2209, 30 lines)**
- **Keep**: one compact list of the four textbook metrics, one line each with the formula. The repository-specific rules stay:
  - **Mean IoU class set**: the classes with a non-empty union, background included. A class painted without any true pixel counts 0, and a class neither present nor predicted is left out, as in mmseg and Cityscapes (README.md:2188). It is the key fact of this block.
  - **The `—` trap**: such a class lowers the mean but shows `—` in the per-class table. Check its column in `confusion_matrix.csv` (README.md:2197).
  - **Mean Dice** uses the same class set as the mean IoU. The soft Dice loss is not this metric (README.md:2201-2203).
  - **Frequency weighted IoU** never sees a class predicted without true pixels (README.md:2209).
- **Cut**:
  - The arithmetic on the toy (README.md:2182, 2190, 2201, 2207).
  - The "background dominates" prose (README.md:2184, 2209), replaced by the figure below.
  - The tutorial numbers (README.md:2184, 2197, 2209).
- **Illustrate**: **extend** `metrics_segmentation_class_averaging.png`. Add, under each of its two toy cases, a small horizontal bar strip of the six headline values: Pixel Accuracy, Frequency Weighted IoU, Mean Dice, Mean IoU, Mean Boundary IoU and Mean Boundary F1. These come from the same `compute_segmentation_metrics` call. The reader then sees in one glance that pixel accuracy (97.36 %) and FWIoU (94.81 %) stay high while the mean IoU drops to 64.46 %. That makes "read pixel accuracy as a sanity check" visible without prose.
- **Estimated length after**: 12 lines (including the figure block).

**Boundary IoU And Boundary F1 (README.md:2211-2227)**
- **Keep**:
  - The repository-specific definitions:
    - The band uses 8-neighbour distance.
    - Ignored pixels and the image border draw no contour.
    - The tolerance is a fixed number of pixels (`boundary_tolerance`) instead of the Boundary IoU paper's share of the diagonal.
  - The two formulas, one line each.
  - The class-set rule: classes without a true edge are left out of the means even when predicted, unlike the mean IoU (README.md:2218).
  - One sentence on the difference. Boundary F1 saturates at 100 % once every contour pixel is within tolerance, while Boundary IoU keeps rewarding a closer contour (README.md:2227, first sentence).
- **Cut**: the toy paragraph (README.md:2220), which repeats the GIF caption, and the tutorial numbers (README.md:2227).
- **Illustrate**: keep `metrics_segmentation_boundary_tolerance.gif`.
- **Estimated length after**: 11 lines.

**Evaluated Resolution (README.md:2229-2236)**
- **Keep**: the three modes in three short bullets (network, tiled, `original_resolution`), all repository behaviour. Add one sentence: a uniform scaling leaves the region metrics as they are, but `boundary_tolerance` counts pixels of the evaluated resolution, so scale it with the resolution.
- **Cut**: the 4× numbers (README.md:2236), which go into the figure below. Make `tasks_reference.md:315` (`original_resolution`) a link here.
- **Illustrate**: add a **last frame** to `metrics_segmentation_boundary_tolerance.gif` (or a second static panel): the same square scaled 4×, with boundary F1 at 2 pixels falling from 75.63 % to 24.63 % while the region IoU stays at 81.40 %. The numbers already exist at README.md:2236.
- **Estimated length after**: 5 lines.

**Per Class Table (README.md:2238-2275)**
- **Keep**:
  - Worst IoU first.
  - `—` for a class without true pixels, or without true edges in the boundary columns.
  - ROC AUC and average precision come from histograms of 10,000 bins of pixel probabilities, only with `per_class_report`.
  - `class_metrics.csv` is written always.
  - The precision-versus-recall reading rule, in one sentence: high precision with low recall means missed; the reverse means painted over others (README.md:2275).
- **Cut**: the 30-line toy report block (README.md:2244-2273). The reader sees the real one at their first evaluation, and its column names are in the names table. Also cut the toy description (README.md:2242) and the tutorial example (README.md:2275, second sentence).
- **Estimated length after**: 5 lines.

**Pixel Recall Interval (README.md:2277-2283)**
- **Keep**:
  - The interval is taken over test images, not pixels, because the pixels of an image are correlated. This is repository behaviour and the reason the table exists.
  - Its width depends on the number of images holding the class, not on the number of pixels, so a higher resolution does not narrow it.
  - A change smaller than the half-width is noise.
  - The method itself (Korn-Graubard via bootstrap) is a link to the CLS recall intervals section (part 6's range).
- **Cut**: the toy counts and tutorial numbers (README.md:2281), replaced by the figure below.
- **Illustrate**: **new figure** `metrics_segmentation_pixel_versus_image_interval.png`. It is a two-row forest plot for the circle of the toy split, recall 76.24 %:
  - "over images" (33.75 % to 97.70 %) as a wide bar;
  - "over pixels, as if independent" (72.87 % to 79.38 %) as a narrow bar;
  - six dots for the per-image recalls (106/149, 113/113, 74/113, 0/81, 81/81, 149/149).

  The reader sees why pixel-level confidence is false precision. The data is at README.md:2281, computed by `compute_bootstrap_recall_intervals` and `compute_clopper_pearson_intervals`.
- **Estimated length after**: 7 lines (including the figure block).

**Estimated length after (whole SEG block)**: about 61 lines and 1,500 words, down from 134 lines and 3,432 words.

---

#### Sequence Labelling (M2M) (README.md:2285-2365, 81 lines, 1,700 words)

**Intro (README.md:2287)**
- **Keep**: the headline is `Accuracy` (frame accuracy), and `evaluation_summary.csv` keeps every metric for comparing on another one.
- **Cut**: the "every metric keeps its report name ... `test/` prefix" sentence, which goes to the metrics intro once for all tasks.
- **Estimated length after**: 2 lines.

**Frame Accuracy And The Frame Metrics (README.md:2289-2308)**
- **Keep**:
  - The frame metrics are the CLS metrics computed on labelled frames (link).
  - Padding frames and frames of unmapped categories (`-100`) never enter a metric.
  - Recall intervals are bootstrapped over sequences, not frames.
  - One warning sentence: the majority class makes accuracy misleading, so read balanced accuracy and macro F1 first.
- **Cut**: the formula block and the 3-sequence toy batch (README.md:2295-2306). They are trivial for the audience, and the "unlabelled frame predicted B" point is covered by the exact-match line below. Also cut the tutorial numbers (README.md:2308).
- **Estimated length after**: 4 lines.

**Sequence Exact Match (README.md:2310-2318)**
- **Keep**: the definition. Unlabelled frames always count as right, and sequences with no labelled frame are left out of both numerator and denominator and out of the `Test Sequences` count. Add one sentence: one wrong frame fails the whole sequence, so use the edit score and segment F1 to tell a shifted boundary from a missed event.
- **Cut**: the toy result (README.md:2316) and the tutorial numbers (README.md:2318).
- **Estimated length after**: 3 lines.

**Edit Score (README.md:2320-2342)**
- **Keep**:
  - The edit score is the standard action-segmentation edit score; cite Lea et al. 2017 in one line.
  - Repository-specific: a gap of unlabelled frames ends a segment in both the true and the predicted sequence, and each sequence weighs the same.
  - The formula block (README.md:2326-2331), 3 lines.
  - The `Segments` report note.
  - One reading sentence: it ignores positions, so a shifted boundary scores 100 %; read it with F1@50.
- **Cut**: the worked examples (README.md:2335), which repeat the caption of `metrics_sequences_frames_versus_segments.png`, and the tutorial numbers (README.md:2342). The same example is also in `tasks_reference.md:352`; keep it in one place only, as the figure in the metrics document, and make the task reference link to it.
- **Illustrate**: keep `metrics_sequences_frames_versus_segments.png`. It carries the whole argument "same frame accuracy, different edit score".
- **Estimated length after**: 11 lines.

**Segment F1@10, @25 And @50 (README.md:2344-2364)**
- **Keep**: the matching rule, 2 lines:
  - Predicted segments are taken in order.
  - Each is compared only with the true segment of its class that it overlaps most, using frame IoU.
  - It is a false positive if that true segment is already taken, even when another segment overlaps enough.
  - Counts are summed over the split.

  Also keep the thresholds `[0.10, 0.25, 0.50]` and one reading sentence: a gap between @10 and @50 means shifted or wrong-length events, and a low @10 means fragmentation or missed short events. Add half a sentence saying that with 1-2-frame segments the three thresholds agree.
- **Cut**: the precision/recall/F1 block (README.md:2350-2353, textbook), the toy paragraph (README.md:2357, repeats the GIF caption) and the tutorial numbers.
- **Illustrate**: keep `metrics_sequences_segment_matching.gif`.
- **Estimated length after**: 8 lines.

**Estimated length after (whole M2M block)**: about 28 lines and 650 words, down from 81 lines and 1,700 words.

---

#### Frame Selection (SEL) (README.md:2366-2486, 121 lines, 2,610 words)

Almost everything here is repository-specific (there is no textbook "selection accuracy"), so the savings come from captions, duplicated examples and names, not from definitions.

**Intro (README.md:2368)**
- **Keep**: the headline is the selection accuracy of `Selection.decoding`, plus the toy setup in one sentence: 4 sequences, triangle and star, one frame per class, `top_k` 2, target precision 99 %. Name "the README run" explicitly as `tutorials/run_snapshots/readme_screenshots/SEL_...`, or drop all its numbers (cross-cutting idea 3).
- **Estimated length after**: 3 lines.

**Hungarian And Greedy Decodings (README.md:2370-2384)**
- **Move**: the two decoding definitions (README.md:2374-2377) and `metrics_sequences_selection_decodings.gif` to the SEL section of `tasks_reference.md`. The task reference already explains the decodings with the same example (`tasks_reference.md:375-378`), and the decoding is model and deployment behaviour, not a metric. There, the GIF replaces the prose example at `tasks_reference.md:378`.
- **Keep**: one sentence, with a link: both decodings run at every evaluation, and the one in `Selection.decoding` is drawn, thresholded and deployed. Also keep the non-obvious insight (README.md:2384): the Hungarian decoding maximises the probability of the labelling, not the number of right picks, so it can lose to greedy (sequence 3 of the toy split).
- **Estimated length after**: 3 lines.

**Selection Accuracy (README.md:2386-2408)**
- **Keep**:
  - The formula block (README.md:2392-2397), 4 lines.
  - The order of picks inside a class does not matter.
  - Since exactly `frames_per_class` true frames and picks exist per class, the selection accuracy is at once recall and precision.
- **Cut**: the names (README.md:2399, which go to the names table), the toy result (README.md:2401) and README.md:2408. The background/swapped block covers the latter.
- **Illustrate**: keep `metrics_sequences_selection_test_split.png` as the anchor figure of the whole block. Every later subsection refers to it.
- **Estimated length after**: 10 lines.

**Fully Right Sequences (README.md:2410-2416)**
- **Keep**: the definition and the reading in one sentence: it tells whether wrong picks gather in a few sequences.
- **Cut**: the toy result and the run numbers.
- **Estimated length after**: 2 lines.

**Background And Swapped Picks (README.md:2418-2432)**
- **Keep**: the 3-line formula block, the `ValueError` for a pick of a class that is neither wanted nor unselected, and the identity right + background + swapped = 100 %.
- **Cut**: the toy paragraph (README.md:2430) and the run paragraph (README.md:2432), replaced by the figure below.
- **Illustrate**: **new figure** `metrics_sequences_selection_pick_outcomes.png`. It shows 100 % stacked horizontal bars, one per decoding, for the toy split:
  - Hungarian: 62.5 % right, 12.5 % background, 25 % swapped;
  - greedy: 50 / 12.5 / 37.5.

  Optionally add a third bar for the SEL snapshot run (95.83 / 3.50 / 0.67) from its `Results.txt`. The reader sees the identity and the kind of error at a glance.
- **Estimated length after**: 8 lines (including the figure block).

**Top-k Selection Accuracy And Mean True Frame Rank (README.md:2434-2453)**
- **Keep**:
  - The 3-line formula block.
  - `k` is also the number of candidates the deployed decoder returns.
  - The tie gotcha: ties take the best rank, so uniform probabilities give top-2 100 % and rank 1.0000 (README.md:2453, last sentence). It is repository behaviour, and a dangerous one.
  - The rank reads the class probability, not the Hungarian gain, so rank 1 does not guarantee the pick.
- **Cut**: the names (README.md:2444, rest), the toy ranks (README.md:2446, which repeat the caption) and the run numbers.
- **Illustrate**: keep `metrics_sequences_true_frame_ranks.png`.
- **Estimated length after**: 10 lines.

**Frame Accuracy, Frame Balanced Accuracy And Frame Macro F1 (README.md:2455-2461)**
- **Keep**:
  - These are the CLS metrics of the most probable class before decoding, with the unselected class included.
  - The gotcha: the confusion matrix of the report holds the *decoded* classes, not these.
  - Frame accuracy can sit below selection accuracy, because the decoding forces picks.
- **Cut**: the toy numbers (README.md:2459) and the run numbers.
- **Estimated length after**: 3 lines.

**Per Class Selection Table (README.md:2463-2472)**
- **Keep**: one sentence: one row per wanted class, worst selection accuracy first, with the columns listed in the names table; the `Per Class Recall` table follows.
- **Cut**: the toy table (README.md:2467-2472).
- **Estimated length after**: 1 line.

**Low-Confidence Thresholds (README.md:2474-2485)**
- **Keep**:
  - Split the 9-line single paragraph (README.md:2476) into 4 bullets:
    - **Rule**: link to the CLS threshold rule, applied to the picks of the decoding of the run.
    - **Recall**: the share of all true frames picked at or above the threshold. A null threshold is given when the target is never reached.
    - **Outputs**: `class_thresholds.json`, the two columns, `selections.csv`.
    - **Deployment flagging** (`flag_low_confidence`): a pick below the threshold, every pick of a class with a null threshold, a missing frame. Only one target precision is accepted.
  - The caveat in one sentence: a confident wrong pick passes, and thresholds found on a few hundred picks move between checkpoints.
- **Cut**: the toy paragraph (README.md:2478), which repeats the caption of `metrics_sequences_selection_thresholds.png`, and the run numbers (README.md:2485).
- **Illustrate**: keep `metrics_sequences_selection_thresholds.png`.
- **Estimated length after**: 10 lines.

**Estimated length after (whole SEL block)**: about 50 lines and 1,100 words, down from 121 lines and 2,610 words. Another 10 lines move to `tasks_reference.md`.

---

#### Self-Supervised Evaluation (SSL) (README.md:2487-2586, 100 lines, 2,452 words)

**Intro (README.md:2489)**
- **Keep**:
  - The evaluated object is the pooled backbone representation before the projection head, which is what the export serves and what other tasks start from.
  - Two questions are asked: are classes close (k-NN), and is the representation spread (rank and standard deviation).
  - Both the trained and the averaged weights are evaluated.
- **Estimated length after**: 3 lines.

**The k-NN Evaluation (README.md:2491-2514)**
- **Keep**: the repository-specific points:
  - The reference images are the `number_reference_images_per_class` training images per class with the smallest MD5 hashes of their file names, so the same images are used at every evaluation, resume and standalone run.
  - Every training class stays a possible prediction, and a test class without training images is announced in yellow.
  - Test images are compared only with reference images, never with each other, so one test image per class works (ImageNet-22k held-out images).
  - The `ValueError` when there are no reference images.
  - The report note line.
  - The vote formula, 4 lines (README.md:2499-2502). It is the DINO weighted k-NN with T = 0.07; cite it once.
- **Cut**:
  - The temperature explanation and the `torch.topk` tie remark (README.md:2505).
  - The worked example (README.md:2507), which repeats the caption of `metrics_self_supervised_knn_vote.png`.
  - The MD5 toy listing (README.md:2514), which is replaced by the one-line rule. Keep only the report note example.
- **Illustrate**: keep `metrics_self_supervised_knn_vote.png`.
- **Estimated length after**: 14 lines (including the figure block).

**k-NN Accuracy, Balanced Accuracy And Macro F1 (README.md:2516-2537)**
- **Keep**:
  - The headline is the k-NN accuracy when labelled test images exist; otherwise the effective rank is the headline, with the report line `k-NN : skipped`.
  - The three metrics are the CLS definitions applied to the vote (link to the CLS anchors).
  - `Labelled Test Images : <l> of <N>`.
  - The sparse confusion matrix above 1,000 classes.
  - Two reading bullets, condensed: the chance level is 1 / classes; compare only runs with the same `number_neighbours` and reference count. It measures the representation as is, not a fine-tuned classifier.
- **Cut**: the worked example table and its arithmetic (README.md:2522-2532, 11 lines). It illustrates the CLS macro-F1 class-set rule ("predicted but absent counts 0"), which belongs to the CLS "Macro And Micro Averages" section (part 6). Link to it instead. Also cut the names sentence (README.md:2520, last) and the tutorial numbers (README.md:2535).
- **Estimated length after**: 6 lines.

**Per Class k-NN Metrics (README.md:2539-2543)**
- **Keep**:
  - Clopper-Pearson intervals here, because the test images are independent (unlike the SEG and M2M bootstrap). It is worth one clause, since it explains why the interval methods differ between tasks.
  - A class without test images shows `—` in the tables but 0 in `class_metrics.csv`, with `samples` 0.
- **Cut**: the table and file names (they go to the names table) and the tutorial example (README.md:2543).
- **Estimated length after**: 3 lines.

**Effective Rank (README.md:2545-2571)**
- **Keep**:
  - The formula block (README.md:2549-2553, RankMe; cite it).
  - The maximum is min(N, F).
  - Computed in float64; a non-finite value gives `NaN`, which is never the best checkpoint.
  - Two repository gotchas:
    - The representations are not centred and not negative after ReLU and pooling, so the mean takes the first singular value (README.md:2564).
    - The first checkpoints can read about 1 without collapse, because batch normalization in evaluation mode has unsettled running statistics (README.md:2563, second half).
  - One sentence: it counts used directions, not class separation.
- **Cut**: README.md:2555 (the explanation of the exponential of the entropy, textbook for this audience), README.md:2565 (repeats the caption numbers) and the tutorial numbers.
- **Illustrate**: keep `metrics_self_supervised_effective_rank.png` (its non-negative case supports the not-centred gotcha) and `metrics_self_supervised_collapse.gif`.
- **Estimated length after**: 16 lines (including two figure blocks).

**Representation Standard Deviation (README.md:2573-2585)**
- **Merge** with "Embedding Standard Deviation" (README.md:2639-2643) into one subsection, "Representation And Embedding Standard Deviation". Both are the same function (`compute_embedding_standard_deviation`) on two different inputs. Keep the 3-line formula once (README.md:2578-2580) and a 2-row table:

  | Measured on | When | Healthy | Collapsed |
  |---|---|---|---|
  | test representations (backbone, not negative) | every evaluation | 0.5-0.7 | towards 0 |
  | embeddings of the first views (projection head) | every training and validation batch | about 1 | towards 0 |

  It also needs one sentence on the early-validation false alarm in evaluation mode (README.md:2643).
- **Cut**: the prose repeating the caption values (README.md:2585, 2641) and the tutorial numbers.
- **Estimated length after**: 9 lines for both merged subsections (counted here; the SSL training block counts 0 for "Embedding Standard Deviation").

**Estimated length after (whole SSL evaluation block)**: about 51 lines and 1,150 words, down from 100 lines and 2,452 words.

---

#### Self-Supervised Training Metrics (SSL) (README.md:2587-2643, 57 lines, 1,123 words)

**Intro (README.md:2589)**
- **Keep**: these are batch metrics on the projection-head embeddings, defined in `METHOD_TRACKED_METRICS`. They watch the training, while the evaluation judges the representation.
- **Cut**: the names and logging sentence, which goes to the names table.
- **Estimated length after**: 2 lines.

**Positive Pair Accuracy (SimCLR) (README.md:2591-2612)**
- **Keep**:
  - The definition in one line: the share of the 2N views whose most similar other view is their partner. Drop the 4-line block (README.md:2596-2599), which is standard InfoNCE.
  - The chance level is 1/(2N−1).
  - The repository tie rule: on a tie the first view wins, so a collapse gives 1/(2N) with the loss stuck at log(2N−1) (README.md:2602).
  - The shortcut reading: near 100 % long before the k-NN rises means the views share a cue (README.md:2610).
  - Compare only runs with the same batch size and augmentations.
- **Cut**: the tutorial numbers (README.md:2611; or keep the single "3.91 % before any update because views share colour" as the one tutorial number of the task).
- **Illustrate**: keep `metrics_self_supervised_positive_pair_accuracy.png`. It shows the shortcut better than the prose.
- **Estimated length after**: 9 lines.

**Diagonal And Off-Diagonal Correlation (Barlow Twins) (README.md:2614-2637)**
- **Keep**:
  - Two definitions, one line each: the diagonal is the mean of C_ii (invariance, target 1) and the off-diagonal is the mean of |C_ij| over D(D−1) (redundancy, target 0).
  - The standardisation epsilon 1e-5.
  - Three gotchas, condensed:
    - A constant feature standardises to 0, so a collapse gives a diagonal of 0 (not 1) and a loss of D.
    - The off-diagonal never reaches 0 on a finite batch (floor from sampling).
    - Being a mean, the off-diagonal compares across embedding sizes; the loss does not.
- **Cut**: the loss line (README.md:2624, Barlow Twins paper, textbook), the console `Best` marker remark (README.md:2627, last) and README.md:2636 (repeats the caption of the middle matrix).
- **Illustrate**: keep `metrics_self_supervised_barlow_twins_correlation.png`.
- **Estimated length after**: 10 lines.

**Embedding Standard Deviation (README.md:2639-2643)**
- **Move**: merged into the shared standard deviation subsection above.
- **Estimated length after**: 0 lines.

**Illustrate (whole SSL, both blocks)**: a **"collapse checklist" table** at the top of the SSL evaluation block replaces the scattered "How to read it" bullets about collapse (README.md:2563, 2585, 2602, 2635, 2643). It has 5 rows and 5 columns: signal | where it is logged | healthy | collapsed | false alarm early in a run.
- Effective Rank (evaluation; tens; about 1; first checkpoints, BN statistics).
- Representation SD (evaluation; 0.5-0.7; towards 0).
- Embedding SD (training; about 1; towards 0; first validation batches).
- Positive Pair Accuracy (training; rising; 1/(2N)).
- Diagonal Correlation (training; towards 1; 0).

This is what someone debugging a self-supervised run needs in one glance.

**Estimated length after (whole SSL training block)**: about 21 lines and 500 words, down from 57 lines and 1,123 words. Add 7 lines for the checklist table.

---

### Summary of the part

The README keeps 0 lines of this range: part 6's pointer table covers it. The "lines after" column is the length in the moved metrics document.

| Section | Lines now | Lines after | Main action |
|---|---|---|---|
| Keypoint Detection (KPT) | 140 | 56 | Cut caption-repeating examples and names bullets. Keep box-area (not comparable with COCO), matching and pycocotools differences. Add the KPT matrix and a misses-versus-misplacements scatter. |
| Semantic Segmentation (SEG) | 134 | 61 | Four textbook metrics in one list. Keep the pooled-counts, mean-IoU class-set and boundary class-set rules. Cut the 30-line toy report. Extend the class-averaging figure and add a pixel-versus-image interval plot. |
| Sequence Labelling (M2M) | 81 | 28 | Cite the edit score and F1@k papers. Keep the gap-splits-segments and matching rules. Cut the toy batch and examples covered by the two figures. |
| Frame Selection (SEL) | 121 | 50 | Move the decodings and their GIF to `tasks_reference.md`. Keep the formulas, tie and flag rules. Cut toy and run paragraphs. Add a stacked bar of pick outcomes. |
| Self-Supervised Evaluation (SSL) | 100 | 51 | Keep the MD5 references, the test-versus-reference-only rule and the not-centred rank gotcha. Cut the k-NN worked table (a CLS rule). Merge the two standard deviation subsections. |
| Self-Supervised Training Metrics (SSL) | 57 | 28 | Keep tie and collapse gotchas only. Add the collapse checklist table. |
| **Total** | **633** | **about 274** (0 in the README) | Names tables per task, no caption repeats, at most one run number per task, one place per fact shared with `tasks_reference.md`. |

Words go from 14,896 to about 6,200 in the moved document, and from 14,896 to 0 in the README.

Side findings, outside the shrinking itself:
- "README run" (README.md:2368-2485) is undefined for a reader.
- `readme_screenshots/SEG_..._MC9` (README.md:2184) lacks its `tutorials/run_snapshots/` prefix.
- None of the subsection anchors in this range (`#object-keypoint-similarity-oks`, `#effective-rank`, `#edit-score`, ...) is linked from outside README.md, so moving them breaks no external link. Only the internal links (e.g. `#recall-intervals`, used 10 times) need re-pointing.

## Part 8: Tasks Reference (tasks_reference.md, 441 lines)

Ideas only, nothing modified. Line numbers are those of `tasks_reference.md` on main (1c4d3d63) unless written `README.md:<n>`.

### Diagnosis in four points
- **Lines are not the problem, words are.** 441 lines but 8,843 words: many bullets are 100 to 250 words long (tasks_reference.md:148 is 330 words, :150 is 290). DET (1,480 words), KPT (1,294), CLS (1,033) and Synthetic Datasets (982) are 51 % of the file.
- **The Evaluation subsections repeat the README.** Every task has an Evaluation block (tasks_reference.md:114-122, 165-176, 205-208, 269-282, 310-317, 337, 348-352, 367, 384-394, 427-439), about 2,900 words in all. The README "Metrics Explained" (README.md:1580-2643) and the configuration reference "Evaluation Outputs" (configuration_reference.md:108-158) already explain the same metrics and list the same files. One sentence about percentages is repeated 8 times (tasks_reference.md:122, 175, 208, 280, 317, 352, 392, 437), and README.md:1582 already says it once.
- **The Configuration subsections repeat the YAML comments and the workflows.** The tracked configurations explain every key in their comments (for example `object_detection/configuration_example.yaml:294`, `:362-365`, `classification/configuration_example.yaml:263`). Every workflow also has a "Configuration Keys To Know First" table. The reference adds benchmark history on top ("since 2026-11-10, kept by stage 5b", "tied in stage 8 … removed", tasks_reference.md:146-150). That history belongs in the experiment plans.
- **The README and the reference overlap on labels.** README "Label Files" (README.md:352-729, 378 lines of annotated `jsonc`) and the short JSON excerpt of every task in the reference show the same files twice. The label-mapping rules appear in both places (README.md:355-376 and tasks_reference.md:64-86), and so do the sequence pitfalls (README.md:328-331 and tasks_reference.md:248, 321-322, 336).

### Ownership split proposed with the README (for the agent working on README.md:308-730)
| Fact | Owner | The other one keeps |
|---|---|---|
| Folder per split, `label_mapping.json`, classes from 0 or from 1 | README (`README/tasks_dataset_layout.png` + 3 sentences) | the reference shows the same figure once, at the top of Common Dataset Layout, without re-explaining it |
| Effect of a category left out of the mapping, merges, `binary` mode | **reference** (Common Dataset Layout) | README: one link (drop README.md:376) |
| Label file of every task (annotated `jsonc`) | **reference**, inside each task section, compacted | README: the "Label Files" heading becomes one sentence and a link (README.md:352-729, from 378 lines to about 3) |
| Per-task pitfalls (KPT flips, sequences fed whole, `flatten` for motions) and the pooling explanation with its 3 figures | **reference** (KPT, Sequence Tasks) | README: drop README.md:328-350, the Tasks table links the sections |
| One row per task: what it predicts, headline metric, link | README Tasks table, cut to 4 columns (Task, Predicts, Headline metric, Details) | the reference holds the full fact row of the task (labels, sections, converter, workflow, tutorial) |
| Metric definitions, formulas, worked examples | README Metrics Explained, or the metrics document if the README agent moves it out | the reference gives one Evaluation line per task with a link |
| Evaluation files per task | configuration_reference.md "Evaluation Outputs" (already there) | the reference links it |
| How the synthetic shapes are drawn | **reference** (Synthetic Datasets) | README: drop the long sentence at README.md:158, keep `README/synthetic_datasets.gif` and the Quick Start table |

Result: README.md:308-730 (423 lines) falls to about 25 lines. The reference takes the compacted label files and still drops from 8,843 to about 4,000 words.

### Uniform per-task template (every task section, 25 to 40 lines)
Every task section gets the same shape, so a reader always finds the same information in the same place:

```markdown
## Object Detection (DET)
Should you need to find every object of an image with its box, class and score, DET trains a two-stage
detector (Faster R-CNN style) whose detection levels read backbone blocks, through a neck or directly.

<p align="center"><img src="../../README/task_card_object_detection.png" width="100%" /></p>
<p align="center"><em>One synthetic sample, its labels.json record, the tensor shapes from image to detections, and the headline metric.</em></p>

| Labels | Own sections | Headline metric | Converter | Go further |
|---|---|---|---|---|
| COCO `bbox` | `DetectionLevels`, `Neck`, `RegionProposal`, `Detector` | `mAP@0.5/area` | `data_retriever.py` (COCO 2017) | [workflow](…) · [tutorial series 03](…) · [metrics](…) |

**Label file**: a compact annotated jsonc of 8 to 15 lines (from README Label Files).
**Rules that change results**: at most 5 bullets about what is refused, removed or silently changed.
**Concepts**: at most 4 bullets, each about something the YAML comments cannot show, ideally with its own figure.
```

- **Task card figure** (new, one per task, `README/task_card_<task>.png`, one script `documentation/presentation/figures/draw_task_cards.py`). It has 4 panels from left to right:
  1. **Sample**: one image (or a frame strip, or a before/after pair) written by `create_synthetic_dataset.py --task=<task>` into a temporary folder with tiny split sizes, drawn with its labels: boxes, masks, keypoints with the v flags, the class mask, the frame labels.
  2. **Record**: the actual `labels.json` record of that sample (5 to 8 lines, keys highlighted) and its `label_mapping.json` line, with an arrow from category to class index.
  3. **Model**: a vertical block diagram with the real tensor shapes, from `[B, 3, H, W]` to the outputs. The shapes are produced by running the task's synthetic configuration on that batch with random weights and forward hooks, so they are measured, not typed. The workflows' "The Task At A Glance" text diagrams (for example object_detection_workflow.md:29-55) list the stages to draw.
  4. **Outputs**: the deployed outputs with their export names (for example `detection_boxes [B, 100, 4]`), the headline metric and the evaluation folder.
  - The reader sees in one glance what goes in, what the network makes of it and what comes out. Reuse the panels of `README/tasks_overview.png` for the prediction drawing, and `tutorials/figures/<task>/dataset_figures.py` (`draw_labelled_image`, `draw_sequence_strip`, `draw_mask_figure`) for the sample panel.

---

#### Header and Table Of Content (tasks_reference.md:1-19, 19 lines)
- **Keep**: the title, the TOC, and one sentence on how the reference relates to the README and the workflows (tasks_reference.md:3, shortened to its first sentence).
- **Cut**: the second sentence of tasks_reference.md:3 ("the Details column of the README Tasks table links the workflow of every task"). The new per-task fact row links the workflow itself.
- **Move**: nothing.
- **Illustrate**: under the TOC, add `README/tasks_overview.png` (already drawn, used only at README.md:4) as a clickable map. Each tile carries the task code and anchors to its section. Alternative: a small image map of 10 tiles. Plain HTML has no per-tile links on GitHub, so the TOC stays as text and the figure sits beside it.
- **Estimated length after**: 22 lines.

#### Synthetic Datasets (tasks_reference.md:20-46, 27 lines, 982 words)
- **Keep**:
  - One opening sentence: the generator, `--task`, `--output_folder` (tasks_reference.md:21).
  - A compact table `--task` | images per split | image size | classes, one row per `--task` value (11 rows). It lists only what the gif cannot show; each table cell holds a few words.
  - The constant names as one line of anchors for readers who want to change them (`OBJECT_MINIMUM_SIZE`, `OBJECT_MAXIMUM_SIZE_FRACTION`, `SHAPE_MINIMUM_SIZES`, `HOLLOW_*`, `LOW_CONTRAST_DIFFERENCE_RANGE`, `OBJECT_MAXIMUM_OUTSIDE_FRACTION`, `MAXIMUM_OBJECT_INTERSECTION_OVER_UNION`, `keep_visible_objects`, `SHAPE_MINIMUM_VISIBLE_FRACTION`). These names appear in no other document (checked with grep), so the reference must keep them.
- **Cut**:
  - The benchmark history in the DET row (tasks_reference.md:28: "the instance segmentation benchmark used the same kind of images … set to its values"). The benchmark is over (commit 1c4d3d63), and the instance segmentation workflow already explains how to draw such a dataset.
  - The SSL row's reference to `sequence_labelling_experiment_plan.md` (tasks_reference.md:37) and the long multi-label row (tasks_reference.md:26). Each becomes one short cell: "5 to 10 shapes, distinct classes; initialises the M2M backbone".
- **Move**:
  - The console panel description (tasks_reference.md:46, 95 words) → one sentence here plus a screenshot (below).
  - The README sentence on how shapes are drawn (README.md:158) is deleted on the README side and replaced by a link to this section.
- **Illustrate**:
  - Reuse `README/synthetic_datasets.gif` (one tile per `--task`, already labelled) right under the opening sentence. It replaces the "Content" column of the table.
  - **New `README/synthetic_shape_rules.png`** replaces tasks_reference.md:39-45 (about 600 words). Its panels:
    - (a) The 35 shapes plus `stick_figure` and `house` as a 6 × 7 catalogue, each with its category identifier (1-37), the hollow ones on a shaded row. This replaces the shape list (:39) and the identifiers (:45).
    - (b) A size ruler from 16 pixels to 80 % of the side on a log axis, with the pentagon-to-nonagon minimum sizes (21-67 pixels) and the KPT minimum of 32 marked (:40).
    - (c) A hollow shape with hole ratio 0.4-0.6 and wall ≥ 2 pixels dimensioned (:40).
    - (d) Flat versus textured backgrounds, with a low-contrast shape (20-40 levels) next to a normal one (≥ 60) (:41).
    - (e) A shape 30 % outside the border, plus two boxes at IoU 0.3, the maximum overlap (:42).
    - (f) The "distinct classes" ranges: rhombus angle 44-70°, parallelogram ratio 1.5-2.5, ellipse and capsule elongation 1.5-3 (:43).
    - (g) Keypoints on a polygon, the star, the capsule, the stick figure (11 joints) and a hollow shape (outer corners then hole corners), with circle/ellipse/annulus crossed out as "no keypoints" (:44).
    - Draw it with the generator's own drawing functions so the figure cannot drift from the code.
  - **New `README/synthetic_dataset_console.png`**: a screenshot of the split panel and the closing "Synthetic Dataset" panel of a small run (`print_split_summary`, `print_dataset_summary`), replacing the prose of :46.
- **Estimated length after**: 22 lines (about 350 words).

#### Common Dataset Layout (tasks_reference.md:48-86, 39 lines)
- **Keep**:
  - The split folder keys and the validation/test behaviour (tasks_reference.md:64, shortened).
  - The class-index rule: 0 to K-1, or 1 to K plus background, no gaps (:72).
  - Merging (:73), mapping a category to 0 (:74), the `binary` mode (:75).
  - The "category missing from the mapping" table (:77-86). It is compact and it is the single home of this rule, so README.md:376 goes.
- **Cut**:
  - The text tree (:50-62). It is the same as `README/tasks_dataset_layout.png` and `README/data_folder_layout.png`.
  - The JSON `{"1": 0, …}` (:68-70). It is printed inside `tasks_dataset_layout.png`.
  - :65-66 become one sentence.
- **Move**: the long note on the validation loader at ratio 0 (:64, in parentheses) → training_reference.md "Run Folder" or configuration_reference.md, where `validation_iteration_ratio` is described.
- **Illustrate**:
  - Reuse `README/tasks_dataset_layout.png` in place of the tree and the JSON. It already shows the folder, the merge, the left-out category and both class ranges.
  - Optional new `README/left_out_category_effects.png`: one small panel per row of the :79-86 table. It would show the multi-class sample crossed out, the multi-label class struck from its list, the M2M frame greyed "ignored", the SEL frame turned "unselected", the DET box removed, and the SEG pixels turned background. Worth it only if the table is turned into a picture; recommendation: keep the table, it reads in 8 lines.
- **Estimated length after**: 22 lines.

#### Classification (CLS) (tasks_reference.md:88-122, 35 lines, 1,033 words)
- **Keep**:
  - The model sentence with the two output modes (:89).
  - The compact label file, single and multi-label (moved from README.md:378-427, about 12 lines with `//` comments).
  - The `class_folder_manifests.py` converter, 2 lines (:101), and the several-sources variant, 1 line (:105).
  - `decorrelation` (DeCov), 2 lines (:112). It is documented nowhere else ("Hidden Feature Correlation" appears in no other document).
  - The rule "normalized hidden layers need `batch_size` ≥ 2 and drop the last incomplete batch" (:109).
- **Cut**:
  - :108 and :110. They are YAML comments, and the configuration reference "Values Per Task" lists them.
  - The `pretrained.class_layer` paragraph (:111). training_reference.md:245, 270 covers it; replace it with the link "Transfer: class layer, see training reference".
- **Move**:
  - The ImageNet-22k split counts, MD5 rule and download retry behaviour (:102-104, about 230 words) → pretraining_reference.md "Downloading ImageNet-22k" / "Validation And Test Splits". The split sizes (14,127,545 / 18,846), the "re-split without downloading" behaviour and the Xet/timeout retry appear in no other document, so they must be moved, not deleted. This section keeps one line: "ImageNet-22k: see pretraining reference".
  - Evaluation (:114-122, about 600 words) → one row of the fact table: "`Accuracy` (multi-label `Subset Accuracy`); metrics: Metrics Explained Classification; files: configuration reference Evaluation Outputs; temperature explorer: tools reference Logit Temperature".
  - Before deleting, check that the `precision_recall_thresholds.png` rounding rule (:118, "fewest decimals … selecting the same samples") and the 9 significant digits of `predictions.csv` (:119) are in Metrics Explained or the classification workflow. The workflow has 10 hits on `precision_recall_thresholds.png`; move whatever is missing there.
- **Illustrate**:
  - **New `README/task_card_classification.png`** (template). Sample `hollow_trapezoid` → record `{"file_name": "train_000000.png", "category_id": 29}` → `"29": 28` → `[B, 3, 256, 256]` → backbone blocks → pooled `[B, C]` → hidden layers → logits `[B, 35]` → softmax → "hollow_trapezoid 0.97".
  - A second strip under it for `multi_label`: the counted-shapes image, `category_ids: [3, 8, 21]`, sigmoid bars per class with the 0.5 line, the present classes lit. It shows softmax against sigmoid in one glance and replaces the prose of :89 and :91.
  - Reuse `README/data_folder_layout.png`? No: it shows the same thing as the card. Use the card alone.
- **Estimated length after**: 30 lines (about 330 words).

#### Object Detection (DET) (tasks_reference.md:124-176, 53 lines, 1,480 words)
- **Keep**:
  - The model sentence and "ground truth boxes are added to the proposals" (:125).
  - The compact COCO label file (moved from README.md:455-500, from 44 lines to about 12).
  - "Other COCO keys are not read" (:127).
  - The 25 % visible-area removal rule (:138) and the translation border rule (:140).
  - The `data_retriever.py` converter, 1 line (:142).
  - `Optimization.loss_weights`: the losses are tracked unweighted, the total weighted (:151, 1 line).
- **Cut**:
  - All benchmark history in :146-150. Examples: "kept by stage 3 of the detection benchmark", "since stage 7 (+0.0104)", "tied with them in stage 8 … and were removed", "trained by stage 5 … below the smooth L1 and off in every configuration", "since 2026-11-10, kept by stage 5b". It is in `documentation/architecture_benchmarking/object_detection_experiment_plan.md` and in the YAML comments.
  - Likewise the lists of which configuration holds which value (`run-1.yaml`, task workflow run configurations…). Nobody looks for that in a task reference.
  - The YAML block (:153-163). The configuration files and the workflow table "Configuration Keys To Know First" (object_detection_workflow.md:86-102) hold it.
  - `random_zoom_out` (:139). training_reference.md:154 has it in the augmentation table.
- **Move**:
  - The export NMS implementation (second half of :150: `suppress_every_pair_by_distance`, `suppress_stacked_groups`, TensorRT FP16 65,504, `batched_nms` loop, about 150 words) → object_detection_workflow.md, export stage, or tools_reference.md "Export". It is a deployment detail, not a task fact. Keep here one line: "the exports run the same NMS as the evaluation".
  - Evaluation (:165-176, about 550 words) → one fact-table cell (`mAP@0.5/area`) plus links to Metrics Explained "Object Detection And Instance Segmentation" (it holds matching, size, mAR, per-class recall: README.md:1789-2010) and to configuration_reference.md "Evaluation Outputs".
  - The TensorBoard tag list (:176) → configuration_reference.md Evaluation Outputs, as one more column or table, since it is a lookup list.
  - Anchor recall and the `compute_anchor_coverage` mode (:170) → training_reference.md "Checking The Anchor Coverage" (already there, :194); keep a link.
- **Illustrate**:
  - **New `README/task_card_object_detection.png`** (template). Sample with 6 boxes → COCO record → `[64, 3, 512, 512]` → blocks 5/6/7 at strides 32/64/64 → neck → 2,304 anchors → objectness `[64, 2304]` → 500 proposals → RoIAlign 7×7 → logits `[64, 128, 36]` → `detection_boxes [B, 100, 4]`, `_scores`, `_labels`. Shapes are measured as in the template; object_detection_workflow.md:31-55 lists them.
  - **New `README/detection_levels_and_neck.png`** replaces :145-146 (about 300 words). It draws the backbone blocks 5, 6 and 7 as stacked maps with their strides, and the anchors of one cell drawn to scale on a 512-pixel sample at each level (scales × aspect ratios). Beside it, three small columns for `none` / `feature_pyramid` / `path_aggregation`, with arrows for the laterals, the top-down path and the bottom-up path. The reader learns in one glance which head reads what and why the anchors differ per level.
  - **Reuse** `tutorials/figures/object_detection/proposal_selection.py` (`make_proposal_selection_facts`) to draw **`README/proposal_selection_joint_per_level.png`**. On one image's proposals per level, `joint` lets the coarse levels take most of the 250 slots, while `per_level` keeps each level's best before a common cut. It replaces the 150-word explanation in :148.
  - **New small `README/nms_iou_versus_distance_iou.png`**: two pairs of boxes at the same IoU, one pair concentric (suppressed by both), one pair offset side by side (kept by DIoU-NMS), with the IoU and the IoU minus the distance term written. It replaces the definition in :150.
  - Reuse `README/anchor_coverage.png` next to the anchor recall link.
- **Estimated length after**: 34 lines (about 400 words).

#### Instance Segmentation (INS) (tasks_reference.md:178-208, 31 lines, 758 words)
- **Keep**:
  - The model sentence (:179).
  - The mask head in 2 lines: RoI Align 14 → convolutions → doublings to 28 → one mask per class (:180-182).
  - `shared` in 1 line (:183).
  - The finer inputs in 1 line with a link (:184).
  - The data rules: segmentation required and box from the mask (:186, :192, :194).
  - `instance_index_maps` in 1 line with its condition: run length encodings that never overlap, `False` for COCO (:193).
  - The 25 % mask pixels rule (:195).
  - The compact label file with a polygon and an RLE (moved from README.md:502-524).
- **Cut**:
  - The configuration list (:197-203). It repeats `configuration_reference.md` "Keys Per Task" and the YAML; keep only the hint "raise the `mask_loss` weight when the masks lag behind the boxes" (:201).
  - :184 shrinks from 190 words to 2 lines. The check rules on skip blocks are refusals printed by the configuration check (configuration_reference.md "Values Per Task / Task rules" holds the INS rules) and the stage 3b details belong in instance_segmentation_experiment_plan.md.
- **Move**: Evaluation (:205-208) → the fact table (`Mask mAP@0.5/area`; box metrics keep the DET names; `mask_` columns) plus links to Metrics Explained DET/INS (README/metrics_detection_mask_iou.png) and to Evaluation Outputs.
- **Illustrate**:
  - **New `README/task_card_instance_segmentation.png`** (template). Sample with masks → record with polygon or RLE → the DET path → per-box mask head `[N, 64, 14, 14]` → `[N, 35, 28, 28]` → pasted masks; outputs `detection_masks`.
  - **Reuse** `documentation/architecture_benchmarking/figures/instance_segmentation/mask_head_designs.png` (drawn by `scripts/models/instance_segmentation/draw_mask_head_designs.py`), or redraw it from `configuration_synthetic_shapes.yaml`. It shows the per-level mask head, `shared: True`, and the optional image branch and skips in one picture, replacing the 300 words of :180-184.
  - **New small `README/instance_index_map.png`**: the same image as N binary masks (`N × H × W` bits) and as one uint8 map numbering the objects, with the byte counts. It shows why it is about a ninth of the bytes and why overlapping COCO masks cannot use it.
- **Estimated length after**: 28 lines (about 330 words).

#### Keypoint Detection (KPT) (tasks_reference.md:210-282, 73 lines, 1,294 words)
- **Keep**:
  - The bottom-up sentence (:211).
  - Training: heatmap targets, focal loss, `ignore_unlabelled_objects`, pull and push, total loss, as 3 lines (:221-224).
  - Data: the v flags, categories with `keypoints` and `skeleton`, the merge rule (same names), the unlabelled test objects kept apart (:231, :241-242).
  - COCO 2017 person keypoints and evaluation on the validation split (:243).
  - Augmentation: the name-keeping rule (:248, the most important pitfall), the refused types and keypoints leaving the image (:246-247).
  - The input size rule, 1 line (:250).
  - The compact label file (moved from README.md:526-592, from 67 lines to about 18, keeping the three flag meanings as `//` comments).
- **Cut**:
  - The YAML block (:252-267). It sits in `keypoint_detection/configuration_example.yaml`, and the workflow lists the keys.
  - The pretrained paragraph (:250 after the first two sentences, about 120 words). It goes into training_reference.md transfer (:251 and :264 already describe the keypoint head and its class layers); keep one link.
  - The sample-image colours (:282). They go into configuration_reference.md's sample image table (KPT row, configuration_reference.md around :150).
- **Move**: Evaluation (:269-280, about 450 words) → the fact table (`OKS mAP@0.5/area`) plus a link to Metrics Explained "Keypoint Detection" (README.md:2011-2150). That section already has the OKS formula (README.md:2028), the PCK and `metrics_keypoints_oks_falloff.png`. The pycocotools unlabelled-object rule (:274) is a metrics fact; check it is in README.md:2011-2150 (memory says it was added on main ea8cfc6b) before deleting.
- **Illustrate**:
  - **New `README/task_card_keypoint_detection.png`** (template). Sample with keypoints, where visible dots are filled, hidden (v = 1) are rings and v = 0 are absent → COCO record with the triples highlighted in the same colours → stem `[B, C, 128, 128]` → HRNet → heatmaps `[B, 243, 128, 128]` plus tags, higher resolution `[B, 243, 256, 256]` → candidates `[B, 243, 30, 2]` → grouped objects. Shapes come from keypoint_detection_workflow.md:29-35.
  - **New `README/keypoint_bottom_up_pipeline.png`** replaces Network (:213-218) and Grouping (:226-229), about 400 words. It has three bands:
    - (a) Stem then the HRNet branches at 1/4, 1/8 and 1/16, drawn as parallel lanes with the exchange units between modules (the classic HRNet ladder).
    - (b) For one keypoint channel, the heatmap with its local maxima as candidates and the tag map with the tag values written.
    - (c) Grouping: candidates on a 1-D tag axis, two objects forming clusters, one candidate beyond `tag_threshold` starting a new object; Hungarian matching keypoint by keypoint.
    - **Reuse** `tutorials/figures/keypoint_detection/dataset_figures.py::draw_heatmap_panel` and `keypoint_inference.py`, which already compute heatmaps and groups on the tutorial run.
  - **New small `README/keypoint_flip_names.png`**: a stick figure and its horizontal flip, with `left_wrist` still labelled `left_wrist` but now on the right side of the image. One look explains why both configurations turn the flips off (:248, README.md:329).
- **Estimated length after**: 40 lines (about 450 words).

#### Semantic Segmentation (SEG) (tasks_reference.md:284-317, 34 lines, 644 words)
- **Keep**:
  - The U-Net sentence (:285).
  - The data rules: `masks/`, uint8, 0 background, 255 ignored, optional `category_ids` (:287).
  - The losses as a 4-row table, loss | what it optimizes | when to use it (:298-302, the same facts in fewer words), with "at least one of cross entropy, Dice, Lovász on".
  - Deep supervision, 1 line (:304).
  - Tiling, 2 lines (:306-308).
  - The compact label file (moved from README.md:429-453).
- **Cut**:
  - :296. It lists keys the YAML and configuration_reference.md "Keys Per Task" already give.
  - The Korn-Graubard details (:312). They are in Metrics Explained "Recall Intervals" (README.md:1767-1779).
- **Move**: Evaluation (:310-317, about 330 words) → the fact table (`Mean IoU`) plus links to Metrics Explained SEG (README.md:2151-2284, with `metrics_segmentation_boundary_tolerance.gif`) and Evaluation Outputs. Keep `original_resolution` as one line here, since it changes what is measured (:315).
- **Illustrate**:
  - **New `README/task_card_semantic_segmentation.png`** (template). Image → mask PNG shown with a palette plus the raw ids of a 6×6 crop (0, 3, 255 visible) → record → encoder blocks with the decoder levels merging the skips → logits `[B, 36, H, W]` → predicted mask; outputs `class_mask`.
  - **New `README/segmentation_tiling.png`** replaces :306-308. A large image scaled by `Data.tiling.scale`, the random training tiles redrawn per epoch (dashed, two colours for two epochs), the fixed validation tile, then the test image covered by overlapping tiles with the overlap band hatched and labelled "probabilities averaged".
  - Optional: reuse `tutorials/figures/semantic_segmentation/model_figures.py::draw_level_figure` for the decoder levels with deep supervision heads marked.
- **Estimated length after**: 30 lines (about 330 words).

#### Sequence Tasks (M2O, M2M, CMP, SEL) (tasks_reference.md:319-324, 6 lines)
- **Keep**: all five bullets (:320-324). They become the single home of the sequence rules.
- **Cut**: nothing.
- **Move here from README**:
  - README.md:330 (fed whole, padding, order-aware versus order-free temporal modules). It duplicates :321-322; drop it from the README.
  - The `flatten` explanation (README.md:338-350) with `README/sequence_pooling_mechanism.gif` and `README/sequence_pooling_similarity.png`. It is the "why" of the M2O rule at :336, and it serves every sequence task, so it goes here. Shorten the two captions to one line each.
  - `README/sequence_padded_batch.gif` (README.md:333-336), under the "fed whole, padded, masked" bullet.
- **Illustrate**: **new `README/sequence_task_family.png`**, the map of the four sequence tasks (replacing the opening sentences of the four sections below). A shared trunk: frames → the same backbone per frame → `global_pooling` → frame vectors `[B, T, D]` → temporal module, with the 7 types in two boxes, "ignore order" and "follow order". It then branches to four outputs:
  - M2O: one label per sequence.
  - M2M: one label per frame, with a causal mask drawing showing frame t seeing only ≤ t.
  - CMP: two members through the same trunk → `Fusion` (concatenation / difference / cross-attention) → one label.
  - SEL: per-frame probabilities → decoding → picks.
  - The reader sees that the four tasks share a trunk and differ by their head.
- **Estimated length after**: 22 lines (5 bullets, 3 figures with one-line captions, a 3-line pooling explanation).

#### Sequence Classification (M2O) (tasks_reference.md:326-337, 12 lines)
- **Keep**:
  - The one-sentence definition.
  - The `flatten` rule (:336), shortened to "needs `flatten`, see Sequence Tasks".
  - The compact label file (moved from README.md:594-624; the reference's own JSON at :329-334 is replaced by it).
- **Cut**: :337 becomes a cell of the fact table (CLS metrics on the sequences, no temperature explorer).
- **Move**: nothing else.
- **Illustrate**: **`README/task_card_sequence_classification.png`** (template): a frame strip of a moving shape, the record, `[B, T, 3, H, W]` → `[B, T, D]` → temporal `[B, D']` → logits `[B, 4]`. Reuse `tutorials/figures/sequence_classification/dataset_figures.py`.
- **Estimated length after**: 16 lines.

#### Sequence Labelling (M2M) (tasks_reference.md:339-352, 14 lines, 358 words)
- **Keep**:
  - The one-sentence definition.
  - `causal`, 1 line, with its refusal of sequence normalization (:340).
  - The data rule that unequal `frame_category_ids` and `frames` lengths stop the launch (:346).
  - The temporal types restriction, 1 line (:347).
  - The compact label file (moved from README.md:626-657).
- **Cut**: the Korn-Graubard sentence in :348. It is covered by Metrics Explained "Recall Intervals".
- **Move**: the edit score and segment F1 definitions plus the worked example (:349-352) → Metrics Explained "Sequence Labelling" (README.md:2320-2362). That section already redraws the same example in `metrics_sequences_segment_matching.gif` and even cites "the true sequence … of the tasks reference" (README.md:2335, 2362). When the example moves, the README text must stop pointing back here (coordination item for the README agent).
- **Illustrate**: **`README/task_card_sequence_labelling.png`** (template). A frame strip with per-frame labels (`no_change`, `appearing_star` …) → record → per-frame logits `[B, T, 71]`. Optionally, one inset of a causal attention mask (lower-triangular grid) for `causal: True`.
- **Estimated length after**: 16 lines.

#### Sequence Comparison (CMP) (tasks_reference.md:354-367, 14 lines)
- **Keep**:
  - The definition and the `Fusion.type` values (:355).
  - Members of different lengths (:365, first sentence).
  - `pair_list_manifests.py` with its input format, 2 lines (:365).
  - `sequence_normalization_scope` (:366).
  - The compact label file (moved from README.md:694-729, replacing :357-363).
- **Cut**: :367 becomes a cell of the fact table.
- **Move**: nothing.
- **Illustrate**: **`README/task_card_sequence_comparison.png`** (template). The before and after strips (one shape lost) → record → two `[B, T, D]` streams through the shared trunk → fusion → logits `[B, 2]`. The fusion modes are already in `sequence_task_family.png`, so no separate figure is needed.
- **Estimated length after**: 18 lines.

#### Frame Selection (SEL) (tasks_reference.md:369-394, 26 lines, 597 words)
- **Keep**:
  - The definition and the role of `unselected_category` (:370).
  - The model, 1 line (:372).
  - Both decodings, 2 lines (:375-376), plus "both scored, `Selection.decoding` drawn and deployed".
  - The data rules: the mapping must map the unselected category; sequences without exactly `frames_per_class` frames of every wanted class stop the training (:380).
  - The bidirectional/non-causal note (:382).
  - The compact label file (moved from README.md:659-692).
- **Cut**:
  - The numerical example (:378). `README/metrics_sequences_selection_decodings.gif` (README.md:2380) shows greedy against Hungarian on the same three frames. Show that gif here, where decoding is defined (it is a model behaviour that is deployed, not only a metric), and let Metrics Explained SEL link back. If the README agent keeps the gif in Metrics Explained, this section links it instead of repeating the example.
  - The sample-image description (:394). It is in configuration_reference.md's sample image table.
- **Move**: Evaluation (:384-392, about 330 words) → the fact table (`<Decoding> Selection Accuracy`) plus links to Metrics Explained "Frame Selection" (README.md:2366-2486, which has background/swapped picks, ranks and thresholds figures) and Evaluation Outputs.
- **Illustrate**:
  - Reuse `README/metrics_sequences_selection_decodings.gif` (see Cut).
  - **`README/task_card_frame_selection.png`** (template): a 12-frame strip, wanted frames outlined, record with `frame_category_ids` → per-frame probabilities `[B, T, 4]` → decoding → picks. Reuse `tutorials/figures/frame_selection/dataset_figures.py::draw_sequence_strip` and `selection_figures.py`.
- **Estimated length after**: 24 lines.

#### Self-Supervised Learning (SSL) (tasks_reference.md:396-441, 46 lines, 889 words)
- **Keep**:
  - The two methods, 1 line each (:398-399).
  - "Projection head only for training, the exported representation is the pooled map" (:401).
  - The data rules: the CLS layout, categories only for k-NN, `{}` mapping without labels (:403).
  - Grouped batches: the `groups` record, the three keys as a mini table, and the sampler fallback rules, 3 lines (:405-415).
  - Views: `random_crop` at probability 1.0 and off for counted shapes, second-view-only types, the shortcut warning (:417-421).
  - The batch size rule (:425).
  - "Using the backbone", 2 lines (:441).
- **Cut**:
  - The configuration file names and head sizes (:423-424). They are in the YAML and the self-supervised workflow; keep "SimCLR: small embedding, Barlow Twins: wide layers" in half a line.
  - The sample-image sentence (:439). It is in configuration_reference.md's sample image table.
- **Move**:
  - Training metrics (:427-430) and Evaluation (:432-437), about 450 words → Metrics Explained "Self-Supervised Evaluation" and "Self-Supervised Training Metrics" (README.md:2487-2643, already with the k-NN vote, effective rank, collapse and positive pair figures) plus a fact-table cell (`k-NN Accuracy`, or `Effective Rank` without labels).
  - Before deleting, check that the k-NN reference bank rule (at most `number_reference_images_per_class`, smallest MD5, the same at every evaluation, :433) is in Metrics Explained; if it is not, move it there.
- **Illustrate**:
  - **New `README/task_card_self_supervised.png`** (template, method variant). One image → two augmented views (reuse `tutorials/figures/self_supervised/view_figures.py::make_view_figures`) → shared backbone → pooled `[B, D]`, boxed as "exported" → projection head, boxed as "training only" → embedding `[B, 128]`. Two loss insets:
    - SimCLR: a 2B × 2B similarity matrix with the partner cells marked.
    - Barlow Twins: a D × D cross-correlation pushed to the identity.
    - The figure replaces :397-401.
  - **New `README/grouped_batches.png`** replaces :405-415. A batch of 16 thumbnails arranged in 4 blocks of 4 (`images_per_block`), each block coloured by its `camera` (`block_key`), the whole batch sharing one `site` (`batch_key`). Next to it, a random batch for contrast, with the caption "the negatives of an image now come from the same site".
- **Estimated length after**: 32 lines (about 380 words).

---

### New figures, in order of value
1. **Ten task cards** `README/task_card_<task>.png` (one script, `documentation/presentation/figures/draw_task_cards.py`): sample → record → measured shapes → outputs and headline metric. They are the backbone of the template and replace most of the "Data" and "Model" prose.
2. **`sequence_task_family.png`**: the shared trunk and four heads of M2O/M2M/CMP/SEL.
3. **`keypoint_bottom_up_pipeline.png`**: HRNet lanes, heatmap + tags, grouping on a tag axis.
4. **`detection_levels_and_neck.png`**: levels, strides, anchors to scale, none/FPN/PANet.
5. **`synthetic_shape_rules.png`**: catalogue with identifiers and annotated drawing rules.
6. Smaller ones: `proposal_selection_joint_per_level.png` (reuses a tutorial script), `nms_iou_versus_distance_iou.png`, `instance_index_map.png`, `segmentation_tiling.png`, `grouped_batches.png`, `keypoint_flip_names.png`, `synthetic_dataset_console.png`.
7. Existing images reused in the reference:
   - `tasks_overview.png`, `tasks_dataset_layout.png`, `synthetic_datasets.gif`.
   - Moved from the README: `sequence_padded_batch.gif`, `sequence_pooling_mechanism.gif`, `sequence_pooling_similarity.png`.
   - `metrics_sequences_selection_decodings.gif`, `anchor_coverage.png`, and `mask_head_designs.png` from the benchmark figures.

### Content that exists only in tasks_reference.md (must move, not vanish)
Checked with grep over README, the other references and the workflows:
- The DeCov `decorrelation` explanation and "Hidden Feature Correlation".
- The ImageNet-22k split counts and download retry details.
- The synthetic constant names (`HOLLOW_MINIMUM_SIZE`, `print_split_summary` …).
- `data_retriever.py` (only one README mention).
- The Lovász/Dice/boundary loss descriptions (the SEG workflow has a few).

### Summary of the part
| Section | Lines now (words) | Lines after (words) | Main action |
|---|---|---|---|
| Header and TOC (1-19) | 19 (105) | 22 (110) | add `tasks_overview.png` as the task map |
| Synthetic Datasets (20-46) | 27 (982) | 22 (350) | `synthetic_datasets.gif` + new `synthetic_shape_rules.png`; cut the benchmark history |
| Common Dataset Layout (48-86) | 39 (325) | 22 (220) | the tree replaced by `tasks_dataset_layout.png`; becomes the single home of the mapping rules |
| CLS (88-122) | 35 (1,033) | 30 (330) | task card; ImageNet-22k → pretraining reference; Evaluation → links |
| DET (124-176) | 53 (1,480) | 34 (400) | cut the benchmark history and the YAML; export NMS → workflow/tools; levels/neck and proposal figures |
| INS (178-208) | 31 (758) | 28 (330) | task card + mask head designs figure; config list cut |
| KPT (210-282) | 73 (1,294) | 40 (450) | bottom-up pipeline figure; YAML and OKS → links |
| SEG (284-317) | 34 (644) | 30 (330) | loss table, tiling figure, Evaluation → links |
| Sequence Tasks (319-324) | 6 (140) | 22 (250) | receives README.md:330-350 with 3 figures + new family figure |
| M2O (326-337) | 12 (99) | 16 (120) | task card + label file from the README |
| M2M (339-352) | 14 (358) | 16 (170) | worked example → Metrics Explained |
| CMP (354-367) | 14 (139) | 18 (170) | task card + label file from the README |
| SEL (369-394) | 26 (597) | 24 (280) | decoding gif here, Evaluation → links |
| SSL (396-441) | 46 (889) | 32 (380) | two-views card + grouped batches figure; metrics → Metrics Explained |
| **Total** | **441 (8,843)** | **≈ 356 (≈ 3,900)** | words −56 %; the reference absorbs the README label files (README.md:308-730, 423 lines → about 25) |
