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

# Install every architecture's deps. rtdetr, dfine, and rfdetr each pull in the
# base requirements (-r requirements.txt) plus their own extras — note they are
# DIFFERENT models: rtdetr (RT-DETR, via transformers/scipy), dfine (D-FINE, also
# via transformers/scipy — see adapters/dfine.py for why it isn't vendored like
# ecdet), and rfdetr (RF-DETR, the Apache-2.0 `rfdetr` package). retinanet is
# plain torchvision (base), and yolox and ecdet are both vendored under vendor/
# (no package) — ecdet adds only scipy for its Hungarian matcher. The default
# torch wheels are CUDA-enabled on linux/x86_64 and fall back to CPU when no GPU
# is visible.
# chachak adds no deps of its own — it reuses friendy_chachkalica + torch.
# Copied into the package subdir so the `-r requirements.txt` refs still resolve.
COPY friendy_chachkalica/requirements.txt friendy_chachkalica/requirements-rtdetr.txt \
     friendy_chachkalica/requirements-dfine.txt \
     friendy_chachkalica/requirements-rfdetr.txt friendy_chachkalica/requirements-yolox.txt \
     friendy_chachkalica/requirements-ecdet.txt \
     friendy_chachkalica/requirements-trt.txt \
     /app/friendy_chachkalica/
RUN pip install --no-cache-dir \
        -r /app/friendy_chachkalica/requirements-rtdetr.txt \
        -r /app/friendy_chachkalica/requirements-dfine.txt \
        -r /app/friendy_chachkalica/requirements-rfdetr.txt \
        -r /app/friendy_chachkalica/requirements-ecdet.txt

# TensorRT engine export (/export_trt). GPU-only; the wheel matching the torch
# CUDA major is resolved from the bare `tensorrt` package. Kept as its own layer
# so a CPU-only or non-TRT build can drop this line without touching the rest.
#
# The three steps after the install exist because `onnxruntime` and
# `onnxruntime-gpu` are two distributions that own the SAME `onnxruntime` package
# directory, and this image installs both: requirements-trt.txt opens with
# `-r requirements.txt`, whose CPU `onnxruntime` line cannot be retracted from the
# file that includes it, while this layer needs the GPU build (for modelopt's
# AutoCast and for onnx_infer/benchmark to reach CUDAExecutionProvider). Whichever
# wheel pip happens to unpack LAST owns the files, and that order is a resolver
# detail, not something written down — so a future rebuild could silently produce
# an image whose onnxruntime has no CUDA provider.
#
# Made deterministic instead of hoped for: drop the CPU distribution, then
# force-reinstall the GPU wheel at the version the resolver already picked (its
# ceiling comes from nvidia-modelopt) so every shared file is written by it last.
# The uninstall alone is NOT enough — it deletes the paths in the CPU wheel's
# RECORD, which by then hold the GPU wheel's files, leaving an `onnxruntime` that
# imports but has no `__version__` and no providers at all (verified).
#
# The final python line is a build-time assertion, not a printout: if this image
# ever comes out without a CUDA provider, the build fails here rather than a
# dfine fp16 export failing months later on a machine nobody is watching.
RUN pip install --no-cache-dir -r /app/friendy_chachkalica/requirements-trt.txt \
    && ORT_GPU_VERSION="$(python3 -c 'import importlib.metadata as m; print(m.version("onnxruntime-gpu"))')" \
    && pip uninstall -y onnxruntime \
    && pip install --no-cache-dir --force-reinstall --no-deps "onnxruntime-gpu==${ORT_GPU_VERSION}" \
    && python3 -c "import onnxruntime as ort; ps = ort.get_available_providers(); \
print('onnxruntime', ort.__version__, ps); \
assert 'CUDAExecutionProvider' in ps, f'no CUDA execution provider: {ps}'" \
    && python3 -c "from modelopt.onnx.autocast import convert_to_mixed_precision; print('modelopt AutoCast importable')"

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
