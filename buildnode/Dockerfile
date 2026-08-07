# Build node: compiles TensorRT engines and assembles infer bundles on THIS
# machine's GPU, so a bundle can be produced for the GPU it will actually run on.
#
# Built from the REPO ROOT context (like friendy_chachkalica/Dockerfile) because
# it needs the same repo-root layout at runtime: chachak/bundle_export/vendor.py
# resolves its sources as `Path(__file__).parents[2]`, i.e. /app, and copies
# friendy_chachkalica/*.py, onnx_infer/ and trt_infer/ from there into every
# bundle's runtime/.
#
#   docker build -f buildnode/Dockerfile -t chachkalica-buildnode .
#
# python:3.12-slim, NOT an nvidia/cuda base: the tensorrt wheel brings its own
# nvidia-* CUDA libraries, which is the job the CUDA-enabled torch wheels do on
# the trainer image. Verified — a torch-free container on this base compiles and
# deserializes a real engine. The host still needs the NVIDIA Container Toolkit.
FROM python:3.12-slim

# libgl/libglib: chachak's inference modules are copied in whole and some import
# cv2 transitively at module scope. Cheap, and avoids a surprise at bundle time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY buildnode/requirements.txt /app/buildnode/requirements.txt
RUN pip install --no-cache-dir -r /app/buildnode/requirements.txt

# ── the repo subset this node actually needs ────────────────────────────────
# chachak      — bundle assembly (bundle_export) + the modules vendored into a bundle
# onnx_infer   — vendored into every bundle's runtime/
# trt_infer    — vendored into engine bundles' runtime/
# friendy_*    — trt_export (the compile) + the four modules vendor.py copies
COPY chachak/ /app/chachak/
COPY onnx_infer/ /app/onnx_infer/
COPY trt_infer/ /app/trt_infer/
COPY friendy_chachkalica/__init__.py friendy_chachkalica/formats.py \
     friendy_chachkalica/metrics.py friendy_chachkalica/postprocess.py \
     friendy_chachkalica/device.py friendy_chachkalica/config.py \
     /app/friendy_chachkalica/
COPY friendy_chachkalica/ml/__init__.py /app/friendy_chachkalica/ml/
COPY friendy_chachkalica/ml/trt_export/ /app/friendy_chachkalica/ml/trt_export/
COPY buildnode/ /app/buildnode/

# ── the baked person detector ───────────────────────────────────────────────
# Both graphs: the standard one for ONNX bundles, and the EfficientNMS-prepared
# one that is the only form TensorRT can compile for a YOLOX. Generate them with
# buildnode/scripts/make_person_graphs.py (runs in the trainer container, needs
# torch and the .pt) BEFORE building this image. models/ is gitignored, so these
# are out-of-band artifacts like models/people/best_ckpt.engine already is.
COPY models/people/detector.onnx models/people/detector.meta.json \
     models/people/detector.trt.onnx models/people/detector.trt.meta.json \
     /opt/buildnode/person/

# Fail the BUILD, not the first person-crop bundle at runtime, if the graphs are
# missing or a placeholder slipped through.
RUN python -c "\
from pathlib import Path; import sys; \
missing = [p for p in ('detector.onnx','detector.meta.json','detector.trt.onnx','detector.trt.meta.json') \
           if not (Path('/opt/buildnode/person')/p).is_file() or (Path('/opt/buildnode/person')/p).stat().st_size == 0]; \
sys.exit(f'baked person detector incomplete: {missing}') if missing else print('person detector baked in')"

# Import-time smoke test: the whole point of this image is that bundle assembly
# and the TRT compile path load without torch. If a future refactor puts a
# module-scope torch import back on either path, break here rather than in
# production. (`import torch` must fail — nothing installs it.)
RUN python -c "\
import chachak.bundle_export.cli, buildnode.compile, buildnode.builds, buildnode.service; \
import sys; assert 'torch' not in sys.modules, 'torch got imported'; \
print('torch-free import path OK')"

ENV BUILDNODE_PORT=8300 \
    BUILDNODE_WORK_DIR=/var/lib/buildnode/builds \
    BUILDNODE_CACHE_DIR=/var/lib/buildnode/person-cache \
    BUILDNODE_PERSON_DIR=/opt/buildnode/person \
    PYTHONPATH=/app
EXPOSE 8300

# Shell form so $BUILDNODE_PORT expands at container start. One worker on
# purpose: a build takes the GPU, and the in-process lock that serializes them
# does not survive a second worker process.
CMD uvicorn buildnode.service:app --host 0.0.0.0 --port ${BUILDNODE_PORT} --workers 1
