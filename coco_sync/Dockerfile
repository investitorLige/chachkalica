FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The package is imported as `coco_sync` (matches `from . import ...`).
COPY . /app/coco_sync

EXPOSE 9000

# Single worker so the in-process per-file write locks actually serialize;
# threads give enough concurrency for human-paced annotation webhooks.
CMD ["gunicorn", "--bind", "0.0.0.0:9000", "--workers", "1", "--threads", "4", "--access-logfile", "-", "coco_sync.app:app"]
