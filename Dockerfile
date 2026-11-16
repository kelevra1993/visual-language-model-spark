# syntax=docker/dockerfile:1
# Image of the visual-language-model-spark trainer, pushed to an artifact registry for the GPU pods of a cloud
# Kubernetes cluster (linux/amd64 Ubuntu nodes) and built for linux/arm64 to train and test on the DGX Spark. It runs
# vision_training.main and the tools of scripts/ with the environment of uv.lock: the CUDA 13 libraries come with the
# torch wheels, the GPU driver comes from the node. The files sent to the build are listed in .dockerignore.
#
# Multi-stage build: every FROM opens a named stage, a group of steps that "--target <stage>" builds alone.
#   uv       the uv binary, used by the builder only
#   base     Ubuntu 24.04, the architecture guard, the system libraries OpenCV loads at import, tini, the trainer user
#   builder  Python 3.13 managed by uv in /opt/python, the locked environment in /opt/venv, the code in /app
#   test     the builder with the dev group (pytest) and tests/, to run the suite on a GPU machine
#   runtime  base with the interpreter, the environment and the code of the builder only (default target, last stage)
#
# Each machine builds its own architecture natively, e.g. on the Spark for local trainings:
#   docker buildx build --platform linux/arm64 --tag vision-trainer:local --load .
# One tag holding both architectures, where QEMU emulates the other one or from two machines (README, Docker Image):
#   docker buildx build --platform linux/amd64,linux/arm64 --build-arg GIT_COMMIT=<commit> \
#       --tag <region>-docker.pkg.dev/<project>/<repository>/vision-trainer:<tag> --push .
# Start a training with the "args" of the container, not "command", so that tini stays the first process:
#   args: ["python", "-m", "vision_training.main", "--configuration_path=/configurations/<run>.yaml"]

# Versions of the base image, of uv and of Python, those of the development machine
ARG UBUNTU_VERSION=24.04
ARG UV_VERSION=0.12.21
ARG PYTHON_VERSION=3.13.13

# ----------------------------------------------------------------------------------------------------------------------
# Stage uv: the static uv binary of the official image, which holds no shell
# ----------------------------------------------------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ----------------------------------------------------------------------------------------------------------------------
# Stage base: operating system shared by the builder and the runtime, so that the environment built by the one runs on
# the system libraries of the other
# ----------------------------------------------------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS base

# Stop a build for another architecture than amd64 (cloud nodes) and arm64 (DGX Spark), the two of the wheels of uv.lock
# checked; TARGETARCH is set by BuildKit from --platform, empty with the legacy builder
ARG TARGETARCH
RUN case "${TARGETARCH}" in \
        amd64|arm64) ;; \
        *) echo "Image built for '${TARGETARCH}': pass --platform linux/amd64 or linux/arm64 to docker buildx." >&2; exit 1 ;; \
    esac

# Certificates for the downloads of uv, MLflow and Hugging Face; the GL, GLib and X11 libraries the Qt bundled with
# opencv-python loads at import, even without a display; tini to forward the stop signal and reap the loader workers
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgl1 libglib2.0-0t64 libice6 libsm6 libx11-6 \
        libxcb1 libxext6 tini \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged user of the trainings, numeric so that runAsNonRoot can check it (uid 1000 is the ubuntu user)
RUN groupadd --gid 10001 trainer && useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash trainer

# Python of the environment first on the path, logs sent to "kubectl logs" as soon as printed, and the folders where
# GKE mounts the driver libraries and nvidia-smi of the node (absent on nodes using the NVIDIA container toolkit)
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# ----------------------------------------------------------------------------------------------------------------------
# Stage builder: Python and the locked environment, rebuilt only when pyproject.toml or uv.lock change, then the code
# ----------------------------------------------------------------------------------------------------------------------
FROM base AS builder
ARG PYTHON_VERSION

# uv installs Python in /opt/python and the environment in /opt/venv, the paths the runtime copies; the packages are
# copied from the cache mount (another file system) and compiled to bytecode, so that every pod imports torch faster
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN uv python install ${PYTHON_VERSION}

# Locked dependencies without the default groups (dev, and datasets whose FiftyOne serves data_retriever.py only);
# --locked stops the build when uv.lock no longer matches pyproject.toml, and the repository itself is no package (no
# build-system): it runs from /app
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-install-project

# Code of the trainings and of the evaluation and export tools; .dockerignore leaves out the bytecode caches and the
# exported models, and the experiments folder is not part of the image
COPY vision_training ./vision_training
COPY scripts ./scripts

# ----------------------------------------------------------------------------------------------------------------------
# Stage test: the environment of the builder with pytest and the tests, built only by "--target test"
# ----------------------------------------------------------------------------------------------------------------------
FROM builder AS test

# Dev group added to the same environment, then the tests and their reference values
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group dev --no-install-project
COPY tests ./tests

# Whole suite by default, to run on a machine or in a pod holding a GPU (the build machines have none)
CMD ["python", "-m", "pytest", "tests"]

# ----------------------------------------------------------------------------------------------------------------------
# Stage runtime: the image pushed to the registry, without uv, the uv cache nor the tests
# ----------------------------------------------------------------------------------------------------------------------
FROM base AS runtime

# Interpreter, environment and code in separate layers, so that a change of the code pushes the code layer only
COPY --link --from=builder /opt/python /opt/python
COPY --link --from=builder /opt/venv /opt/venv
COPY --link --from=builder /app /app

# Packages importable from any working directory; no bytecode written next to the code, owned by root
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# The pre-trained ImageNet backbones (run folders given to pretrained.experiment_folder) are not in the image yet; the
# datasets, the run folders and the configurations come from the volumes mounted by the pod
USER 10001:10001

# A stop of the pod sends SIGINT, which the trainer handles as Ctrl+C: it saves the checkpoint of its last steps and
# closes its MLflow run before exiting (terminationGracePeriodSeconds must leave it the time to)
STOPSIGNAL SIGINT
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "vision_training.main", "--help"]

# Commit of the build, given by --build-arg GIT_COMMIT, read by the registry: the image holds no git history. It is
# also the version, replacing the "24.04" the Ubuntu image would pass on
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.title="visual-language-model-spark trainer" \
      org.opencontainers.image.description="Training, evaluation and export of the vision models of visual-language-model-spark." \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="${GIT_COMMIT}"
