# Image for both the Django web process and the rq worker.
# The worker shells out to `docker` to provision Label Studio containers against
# the host's docker daemon (socket mounted at run time), so it needs only the
# docker *client* — copied from the official docker cli image.
FROM docker:27-cli AS dockercli

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["gunicorn", "fleetsite.wsgi:application", "--bind", "0.0.0.0:8000"]
