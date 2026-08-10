# Trainer service image: the friendy_chachkalica torch pipeline behind its FastAPI
# wrapper (service.py). Django talks to it over HTTP and shares files on disk
# (see docker-compose.yml for how the shared filesystem is wired).
#
# Built from the REPO ROOT context (see docker-compose.yml `trainer.build`) so the
# image holds a repo-root layout: /app/friendy_chachkalica (this package, run
# flat from its own dir) and /app/chachak (the pipeline package, which imports
# friendy_chachkalica as a package via /app on sys.path). The /pipeline endpoint
# spawns /app/chachak/run.py; /train and /eval keep running friendy flat.
FROM python:3.12-slim

# opencv-python needs these shared libs present at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install every architecture's deps. rtdetr and rfdetr each pull in the base
# requirements (-r requirements.txt) plus their own extras — note they are
# DIFFERENT models: rtdetr (RT-DETR, via transformers/scipy) and rfdetr (RF-DETR,
# the Apache-2.0 `rfdetr` package). retinanet is plain torchvision (base), and
# yolox and ecdet are both vendored under vendor/ (no package) — ecdet adds only
# scipy for its Hungarian matcher. The default torch wheels are CUDA-enabled on
# linux/x86_64 and fall back to CPU when no GPU is visible.
# chachak adds no deps of its own — it reuses friendy_chachkalica + torch.
# Copied into the package subdir so the `-r requirements.txt` refs still resolve.
COPY friendy_chachkalica/requirements.txt friendy_chachkalica/requirements-rtdetr.txt \
     friendy_chachkalica/requirements-rfdetr.txt friendy_chachkalica/requirements-yolox.txt \
     friendy_chachkalica/requirements-ecdet.txt \
     friendy_chachkalica/requirements-trt.txt \
     /app/friendy_chachkalica/
RUN pip install --no-cache-dir \
        -r /app/friendy_chachkalica/requirements-rtdetr.txt \
        -r /app/friendy_chachkalica/requirements-rfdetr.txt \
        -r /app/friendy_chachkalica/requirements-ecdet.txt

# TensorRT engine export (/export_trt). GPU-only; the wheel matching the torch
# CUDA major is resolved from the bare `tensorrt` package. Kept as its own layer
# so a CPU-only or non-TRT build can drop this line without touching the rest.
RUN pip install --no-cache-dir -r /app/friendy_chachkalica/requirements-trt.txt

# Runtime packages, preserving the repo-root layout. friendy_chachkalica runs flat
# (its modules import each other bare, with cwd on sys.path — WORKDIR below);
# chachak imports `friendy_chachkalica.*` with /app on sys.path.
COPY friendy_chachkalica/ /app/friendy_chachkalica/
COPY chachak/ /app/chachak/
COPY onnx_infer/ /app/onnx_infer/
COPY trt_infer/ /app/trt_infer/

# Listen on the port Django targets (TrainingSettings.service_base_url). The
# value is overridable at runtime via the FC_PORT env var; EXPOSE is only docs.
ENV FC_PORT=8200
EXPOSE 8200

# Run from the friendy package dir so its flat imports resolve exactly as before;
# service.py's HERE is /app/friendy_chachkalica and HERE.parent is /app (repo
# root), where chachak/ lives. Shell form so $FC_PORT expands at container start.
WORKDIR /app/friendy_chachkalica
CMD uvicorn service:app --host 0.0.0.0 --port ${FC_PORT}
