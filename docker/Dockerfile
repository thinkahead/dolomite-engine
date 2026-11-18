# base image
FROM ubuntu:22.04

WORKDIR /app

# Add Google Cloud CLI and gcsfuse (EL10)
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates gnupg curl lsb-release git wget && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y google-cloud-cli && \
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.asc && \
    apt-get update && \
    apt-get install -y gcsfuse && \
    apt clean

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG EXTRA=tpu

COPY pyproject.toml envs/
COPY .python-version envs/

# Convert "data dev cuda" → "--extra data --extra dev --extra cuda"
RUN cd envs && \
    UV_CACHE_DIR=./tmp uv sync --extra ${EXTRA} --no-install-project && \
    rm -rf ./tmp

ENV PATH="/app/envs/.venv/bin:$PATH"

# FlashAttention-2 (prebuilt wheel: torch2.10 cu12 cp312 cxx11abiTRUE; A100 sm_80). No nvcc in image.
RUN /bin/uv pip install --python /app/envs/.venv/bin/python einops \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# C toolchain for Triton JIT: lm-engine's Triton kernels (e.g. fused_linear_cross_entropy)
# compile a launcher at runtime and need a host C compiler + python headers.
RUN apt-get update && apt-get install -y build-essential python3-dev && apt clean

# ── Vela: bake the (modified, FSDPDataset-restored) lm_engine source ──
# deps resolve from the venv on PATH; lm_engine resolves from PYTHONPATH.
COPY . /app/src
ENV PYTHONPATH=/app/src
WORKDIR /app/src
