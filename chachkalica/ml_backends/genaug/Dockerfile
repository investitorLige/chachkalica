# Generative augmentation backend (FireRed-Image-Edit).
#
# Same shape as ml_backends/vlm/Dockerfile: its own torch/CUDA environment, the
# CUDA torch wheel, and weights from the mounted offline HF cache.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# deb.debian.org is TLS-intercepted on this network; cloudfront.debian.net is an
# official Debian CDN the proxy leaves alone. See ml_backends/vlm/Dockerfile.
ARG DEBIAN_MIRROR=https://cloudfront.debian.net
RUN sed -i -E "s|https?://deb\\.debian\\.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY ml_backends/genaug/requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -r requirements.txt

COPY ml_backends ./ml_backends

EXPOSE 9092

CMD ["uvicorn", "ml_backends.genaug.service:app", "--host", "0.0.0.0", "--port", "9092"]
