# Image for both the Django web process and the rq worker.
# The worker shells out to `docker` to provision Label Studio containers against
# the host's docker daemon (socket mounted at run time), so it needs only the
# docker *client* — copied from the official docker cli image.
FROM docker:27-cli AS dockercli

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker

# deb.debian.org is TLS-intercepted on this network: plain HTTP comes back with
# no status line (apt reports "Bad header line Bad header data") and HTTPS is
# MITM'd with a self-signed "atlas-MASTER" cert, so apt can use neither.
# cloudfront.debian.net is an official Debian CDN the proxy leaves alone.
# Override with --build-arg DEBIAN_MIRROR=https://deb.debian.org off this network.
ARG DEBIAN_MIRROR=https://cloudfront.debian.net
RUN sed -i -E "s|https?://deb\\.debian\\.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources

# ffmpeg: merges/normalizes downloaded video+audio into mp4 (Videos admin tab).
# nodejs: kept as a fallback JS runtime, but Debian's apt build is v20, below
# yt-dlp's Node>=22 floor (yt_dlp/utils/_jsruntime.py) — yt-dlp rejects it as
# "unsupported" and silently falls back to a no-JS player client whose stream
# URLs YouTube now 403s. deno (below) is what yt-dlp-ejs actually uses; it's
# tried first by downloader.py's JS_RUNTIMES order.
# libglib2.0-0: opencv-python-headless (frame extraction) still links against it
# on slim images even though it skips libGL.
ARG TARGETARCH
ARG DENO_VERSION=v2.9.5
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg nodejs libglib2.0-0 curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && case "${TARGETARCH}" in \
         amd64) DENO_ARCH=x86_64; DENO_SHA=8b010a3b1a4a0188a67cdb8a7a27348b2a501af78aec7fc74f2ace167368d530 ;; \
         arm64) DENO_ARCH=aarch64; DENO_SHA=6b7cae3a8fc4385a59dea3146fcb8bad7fea4230e0ad36a8c692afacbc254be0 ;; \
         *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
       esac \
    && curl -fsSL -o /tmp/deno.zip "https://github.com/denoland/deno/releases/download/${DENO_VERSION}/deno-${DENO_ARCH}-unknown-linux-gnu.zip" \
    && echo "${DENO_SHA}  /tmp/deno.zip" | sha256sum -c - \
    && unzip -q /tmp/deno.zip -d /usr/local/bin \
    && rm /tmp/deno.zip \
    && chmod +x /usr/local/bin/deno

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["gunicorn", "fleetsite.wsgi:application", "--bind", "0.0.0.0:8000"]
