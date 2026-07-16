"""Core inference + metrics loop, shared by the CLI.

Loads a model (any of .pt/.onnx/.engine), runs it over a YOLO-style dataset,
and returns a metrics dict: detection quality (map50, map50_95, precision,
recall, f1, ...) from ``friendy_chachkalica.metrics.evaluate_detection`` plus
timing (eval_seconds, inference_seconds, fps).
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import build_dataloader, resolve_dataset
from model_loader import load_adapter


def run_eval(
    dataset_path: str | Path,
    model_path: str | Path,
    threshold: float,
    nms: Optional[float] = None,
    device: str = "auto",
    batch_size: int = 8,
    num_workers: int = 4,
    warmup: int = 1,
) -> Dict[str, Any]:
    from chachak.infer import predict_adapter
    from friendy_chachkalica.device import resolve_device
    from friendy_chachkalica.metrics import evaluate_detection

    resolved_device = resolve_device(device)
    images_dir, labels_dir, classes = resolve_dataset(dataset_path)
    loader = build_dataloader(images_dir, labels_dir, classes, batch_size, num_workers)

    adapter, info = load_adapter(model_path, resolved_device)
    adapter.eval()
    prediction_classes = info.get("train_classes") or classes

    # Warm-up batches are still evaluated for real afterwards (via itertools.chain) —
    # only their timing is thrown away, so onnxruntime/TensorRT's first-call cost
    # doesn't skew fps without silently shrinking the eval set.
    batches = iter(loader)
    warmup_batches = list(itertools.islice(batches, max(0, warmup)))
    _warm_up(adapter, warmup_batches, resolved_device, threshold, predict_adapter)

    all_predictions = []
    all_targets = []
    inference_seconds = 0.0
    num_images = 0

    eval_start = time.perf_counter()
    for images, targets in itertools.chain(warmup_batches, batches):
        images = [image.to(resolved_device) for image in images]

        predict_start = time.perf_counter()
        predictions = predict_adapter(adapter, images, threshold)
        inference_seconds += time.perf_counter() - predict_start

        num_images += len(images)
        all_predictions.extend(prediction.detach().cpu() for prediction in predictions)
        all_targets.extend(targets)

    metrics = evaluate_detection(
        all_predictions,
        all_targets,
        score_threshold=threshold,
        operating_nms_threshold=nms,
        prediction_classes=prediction_classes,
        target_classes=classes,
        eval_classes=classes,
    )
    eval_seconds = time.perf_counter() - eval_start

    metrics.update(
        {
            "dataset_path": str(Path(dataset_path).resolve()),
            "model_path": str(Path(model_path).resolve()),
            "threshold": threshold,
            "nms": nms,
            "device": str(resolved_device),
            "num_images": num_images,
            "eval_seconds": round(eval_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "fps": round(num_images / inference_seconds, 3) if inference_seconds > 0 else 0.0,
        }
    )
    return metrics


def _warm_up(adapter, warmup_batches, device, threshold, predict_adapter):
    for images, _ in warmup_batches:
        images = [image.to(device) for image in images]
        predict_adapter(adapter, images, threshold)
