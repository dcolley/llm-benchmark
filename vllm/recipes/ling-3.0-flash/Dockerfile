# Ling-3.0 (inclusionAI/vllm-ling-v3) for DGX Spark / aarch64.
# Host install via start.sh is usually faster; this image is for compose.
#
# Build (from this directory):
#   docker compose build
#
# Prefers VLLM_USE_PRECOMPILED=1 (model-card path). Full source compile on
# GB10 can take a long time if precompiled wheels are unavailable.

ARG CUDA_VERSION=13.0.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VLLM_USE_PRECOMPILED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /opt
RUN git clone -b ling_3_0 --depth 1 https://github.com/inclusionAI/vllm-ling-v3.git
WORKDIR /opt/vllm-ling-v3

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

# Matches https://huggingface.co/inclusionAI/Ling-3.0-flash-fp4 quickstart.
RUN uv pip install --upgrade pip \
    && uv pip install --editable . --torch-backend=auto

EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]
