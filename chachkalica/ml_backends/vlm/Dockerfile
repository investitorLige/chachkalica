# Vision-language model backend.
#
# Same shape as ml_backends/sam/Dockerfile, with one deliberate difference: SAM
# installs the CPU torch wheel even though its compose service reserves a GPU.
# A VLM on CPU is far too slow to sample video with, so this image pulls the
# CUDA build.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deb.debian.org is TLS-intercepted on this network: plain HTTP comes back with
# no status line (apt reports "Bad header line Bad header data") and HTTPS is
# MITM'd with a self-signed "atlas-MASTER" cert, so apt can use neither.
# cloudfront.debian.net is an official Debian CDN the proxy leaves alone.
# Override with --build-arg DEBIAN_MIRROR=https://deb.debian.org off this network.
ARG DEBIAN_MIRROR=https://cloudfront.debian.net
RUN sed -i -E "s|https?://deb\\.debian\\.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY ml_backends/vlm/requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -r requirements.txt

COPY ml_backends ./ml_backends

EXPOSE 9091

# HF_HOME/HF_HUB_OFFLINE come from compose: huggingface.co is unreachable on
# this network, so every model must already be in data/hf_cache/hub.
CMD ["uvicorn", "ml_backends.vlm.service:app", "--host", "0.0.0.0", "--port", "9091"]
