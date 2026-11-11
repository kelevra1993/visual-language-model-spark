#!/usr/bin/env bash
# Launcher of the tutorial build of this repository: runs build-tutorials of tutorial-video-builder with the environment
# of this folder (its .venv made by `uv sync`, or `uv run` when it does not exist) on the project of this folder
# (tutorial_project.yaml), at the lowest CPU priority, with the hints of transparent huge pages of numpy and PyTorch
# turned off (with the memory fragmented by the trainings, they made every large allocation stall in the kernel), every
# argument passed on: see `tutorials/build_tutorials.sh --help`.
set -euo pipefail

tutorials_folder="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NUMPY_MADVISE_HUGEPAGE=0 THP_MEM_ALLOC_ENABLE=0 PYTHONUNBUFFERED=1

# The Python of the environment of this folder, created on first use
if [[ -x "${tutorials_folder}/.venv/bin/python" ]]; then
    exec nice -n 19 "${tutorials_folder}/.venv/bin/python" -m tutorial_builder.build_tutorials \
        --project "${tutorials_folder}" "$@"
fi
exec nice -n 19 uv run --project "${tutorials_folder}" python -m tutorial_builder.build_tutorials \
    --project "${tutorials_folder}" "$@"
