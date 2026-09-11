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

COPY ml_backends/sam/requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY ml_backends ./ml_backends
COPY ml_backends/sam/entrypoint.sh /usr/local/bin/sam-backend-entrypoint
RUN chmod +x /usr/local/bin/sam-backend-entrypoint

EXPOSE 9090

CMD ["sam-backend-entrypoint"]
