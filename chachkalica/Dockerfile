# Image for both the Django web process and the rq worker.
# The worker shells out to `docker` to provision Label Studio containers against
# the host's docker daemon (socket mounted at run time), so it needs only the
# docker *client* — copied from the official docker cli image.
FROM docker:27-cli AS dockercli

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker

# ffmpeg: merges/normalizes downloaded video+audio into mp4 (Videos admin tab).
# nodejs: a JS runtime lets yt-dlp-ejs unlock >360p YouTube streams.
# libglib2.0-0: opencv-python-headless (frame extraction) still links against it
# on slim images even though it skips libGL.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg nodejs libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["gunicorn", "fleetsite.wsgi:application", "--bind", "0.0.0.0:8000"]
