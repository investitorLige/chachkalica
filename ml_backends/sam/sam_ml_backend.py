import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import requests


try:
    from label_studio_ml.model import LabelStudioMLBase
except ImportError:  # pragma: no cover - optional runtime dependency
    LabelStudioMLBase = object


class SAMBackend(LabelStudioMLBase):
    """Interactive Label Studio ML backend for SAM point-prompt segmentation.

    Expected Label Studio config:

    <View>
      <Image name="image" value="$image"/>
      <BrushLabels name="brush" toName="image">
        <Label value="Object"/>
      </BrushLabels>
    </View>

    Configure this backend in Project Settings -> Model and enable interactive
    preannotations. Label Studio will call predict() with click context.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sam_ready = False
        self._grounding_sam_model = None
        self._configure()

    def _ensure_setup(self):
        if getattr(self, "_sam_ready", False):
            return
        self.setup()
        self._sam_ready = True

    def _configure(self):
        # Default to polygon output on the polygon control: the export pipeline
        # only handles bbox + polygon, so SAM vectorizes its mask to a polygon
        # rather than emitting a brush mask. Override via env for mask output.
        self.from_name = os.getenv("SAM_FROM_NAME", "segmentation")
        self.to_name = os.getenv("SAM_TO_NAME", "image")
        self.label = os.getenv("SAM_LABEL", "Object")
        self.output = os.getenv("SAM_OUTPUT", "polygon")
        self.data_root = Path(os.getenv("SAM_DATA_ROOT", "label_data")).resolve()
        self.ls_url = os.getenv("LABEL_STUDIO_URL", "").rstrip("/")
        self.api_token = os.getenv("LABEL_STUDIO_API_TOKEN", "")
        self.model_type = os.getenv("SAM_MODEL_TYPE", "vit_b")
        self.model_version = f"sam-{self.model_type}"
        self.grounding_sam_model_version = os.getenv("GROUNDING_SAM_MODEL_VERSION", "grounding-sam")
        self.grounding_sam_weights = os.getenv("GROUNDING_SAM_WEIGHTS", "")
        self.grounding_sam_confidence = float(os.getenv("GROUNDING_SAM_CONFIDENCE", "0.25"))
        self.min_available_ram_gb = min_available_ram_gb(self.model_type)
        self._image_cache: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}

    def setup(self):
        checkpoint = os.getenv("SAM_CHECKPOINT")
        device = os.getenv("SAM_DEVICE", "cuda")

        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "SAMBackend requires label-studio-ml-backend, torch, "
                "segment-anything, pillow, and opencv-python."
            ) from exc

        if not checkpoint:
            raise RuntimeError("Set SAM_CHECKPOINT to a Segment Anything checkpoint path")

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        if device == "cpu":
            require_available_ram(min_gb=self.min_available_ram_gb, model_type=self.model_type)

        sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def predict(self, tasks, context=None, **kwargs):
        if not tasks:
            return []

        params = kwargs.get("params") or kwargs
        print(f"SAMBackend predict params={safe_log_params(params)}", flush=True)
        if context:
            print("SAMBackend dispatch=sam_point", flush=True)
            return self.predict_sam_point(tasks, context=context)

        grounding_sam_prompt = params.get("grounding_sam")
        if grounding_sam_prompt:
            print("SAMBackend dispatch=grounding_sam", flush=True)
            return self.predict_grounding_sam(tasks, prompt=grounding_sam_prompt)

        print("SAMBackend dispatch=empty", flush=True)
        return [empty_prediction(self.model_version)]

    def predict_sam_point(self, tasks, *, context):
        self._ensure_setup()

        task = tasks[0]
        image = self._load_task_image(task)
        image_height, image_width = image.shape[:2]
        prompt = extract_sam_prompt(
            context,
            image_width=image_width,
            image_height=image_height,
            default_label=self.label,
        )
        if prompt is None:
            return [empty_prediction(self.model_version)]

        cache_key = image_cache_key(task)
        cached = self._image_cache.get(cache_key)
        if cached is None or cached[1] != (image_width, image_height):
            self.predictor.set_image(image)
            self._image_cache[cache_key] = (image, (image_width, image_height))
        else:
            self.predictor.set_image(cached[0])

        masks, scores, _ = self.predictor.predict(
            point_coords=prompt["point_coords"],
            point_labels=prompt["point_labels"],
            box=prompt["box"],
            multimask_output=True,
        )
        if len(masks) == 0:
            return [empty_prediction(self.model_version)]

        best_index = int(np.argmax(scores))
        mask = masks[best_index]
        score = float(scores[best_index])
        result = mask_to_result(
            mask=mask,
            image_width=image_width,
            image_height=image_height,
            from_name=self.from_name,
            to_name=self.to_name,
            label=prompt["label"],
            score=score,
            output=self.output,
        )

        return [{
            "result": result,
            "score": score,
            "model_version": self.model_version,
        }]

    def predict_grounding_sam(self, tasks, *, prompt: str):
        class_names = parse_grounding_sam_prompt(prompt)
        if not class_names:
            print("SAMBackend grounding_sam empty_prompt", flush=True)
            return [empty_prediction(self.grounding_sam_model_version) for _ in tasks]

        print(
            "SAMBackend grounding_sam start "
            f"tasks={len(tasks)} classes={class_names} confidence={self.grounding_sam_confidence}",
            flush=True,
        )
        model = self._ensure_grounding_sam_model()
        predictions = []
        for task_index, task in enumerate(tasks, start=1):
            task_id = task.get("id", "unknown")
            print(
                f"SAMBackend grounding_sam task_start {task_index}/{len(tasks)} task_id={task_id}",
                flush=True,
            )
            image = self._load_task_image(task)
            image_height, image_width = image.shape[:2]
            results = []
            scores = []

            for class_name in class_names:
                class_results, class_scores = self._predict_grounding_sam_class(
                    model=model,
                    image=image,
                    image_width=image_width,
                    image_height=image_height,
                    class_name=class_name,
                )
                results.extend(class_results)
                scores.extend(class_scores)
                print(
                    "SAMBackend grounding_sam class_done "
                    f"task_id={task_id} class={class_name!r} results={len(class_results)}",
                    flush=True,
                )

            prediction_score = sum(scores) / len(scores) if scores else 0.0
            predictions.append({
                "result": results,
                "score": prediction_score,
                "model_version": self.grounding_sam_model_version,
            })
            print(
                "SAMBackend grounding_sam task_done "
                f"{task_index}/{len(tasks)} task_id={task_id} results={len(results)} score={prediction_score:.4f}",
                flush=True,
            )

        print(
            "SAMBackend grounding_sam done "
            f"tasks={len(tasks)} total_results={sum(len(item['result']) for item in predictions)}",
            flush=True,
        )
        return predictions

    def _ensure_grounding_sam_model(self):
        if self._grounding_sam_model is None:
            print("SAMBackend grounding_sam model_load_start", flush=True)
            from grounding import load_model

            self._grounding_sam_model = load_model(model_weights=self.grounding_sam_weights)
            print("SAMBackend grounding_sam model_load_done", flush=True)
        return self._grounding_sam_model

    def _predict_grounding_sam_class(
        self,
        *,
        model,
        image: np.ndarray,
        image_width: int,
        image_height: int,
        class_name: str,
    ) -> tuple[list[dict], list[float]]:
        from PIL import Image

        from grounding import extract_prediction_parts, predict_for_prompt

        image_pil = Image.fromarray(image)
        prediction = predict_for_prompt(
            model=model,
            image_pil=image_pil,
            text_prompt=f"{class_name}.",
            confidence=self.grounding_sam_confidence,
        )
        _, mask_scores, masks = extract_prediction_parts(prediction)
        if not masks:
            return [], []

        results = []
        scores = []
        for index, mask in enumerate(masks):
            score = scalar_value(mask_scores[index]) if index < len(mask_scores) else None
            if score is not None and score < self.grounding_sam_confidence:
                continue
            result_score = score if score is not None else 1.0
            results.extend(mask_to_result(
                mask=np.asarray(mask),
                image_width=image_width,
                image_height=image_height,
                from_name=self.from_name,
                to_name=self.to_name,
                label=class_name,
                score=result_score,
                output=self.output,
            ))
            scores.append(result_score)

        return results, scores

    def _load_task_image(self, task: dict) -> np.ndarray:
        from PIL import Image

        path = self._resolve_local_task_image_path(task)
        if path is not None:
            return np.asarray(Image.open(path).convert("RGB"))

        image_value = task.get("data", {}).get("image")
        parsed = urlparse(image_value)
        if parsed.scheme in {"http", "https"}:
            response = requests.get(
                image_value,
                headers=self._auth_headers(),
                timeout=30,
            )
            response.raise_for_status()
            from io import BytesIO

            return np.asarray(Image.open(BytesIO(response.content)).convert("RGB"))

        if image_value.startswith("/") and self.ls_url:
            response = requests.get(
                f"{self.ls_url}{image_value}",
                headers=self._auth_headers(),
                timeout=30,
            )
            response.raise_for_status()
            from io import BytesIO

            return np.asarray(Image.open(BytesIO(response.content)).convert("RGB"))

        path = self._data_root_path(image_value.lstrip("/"))
        return np.asarray(Image.open(path).convert("RGB"))

    def _resolve_local_task_image_path(self, task: dict) -> Path | None:
        image_value = task.get("data", {}).get("image")
        if not image_value:
            raise RuntimeError(f"Task {task.get('id')} does not contain data.image")

        local_path = task.get("meta", {}).get("local_path")
        if local_path:
            return self._data_root_path(local_path)

        parsed = urlparse(image_value)
        query_path = parse_qs(parsed.query).get("d", [None])[0]
        if query_path:
            return self._data_root_path(query_path)

        return None

    def _data_root_path(self, task_path: str) -> Path:
        relative_path = unquote(task_path).lstrip("/")
        path = (self.data_root / relative_path).resolve()
        try:
            path.relative_to(self.data_root)
        except ValueError as exc:
            raise RuntimeError(f"Task image path escapes SAM_DATA_ROOT: {task_path}") from exc
        if not path.exists():
            raise FileNotFoundError(f"Task image not found: {path}")
        return path

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_token:
            return {}
        return {"Authorization": f"Token {self.api_token}"}


DEFAULT_MIN_AVAILABLE_RAM_GB = {
    "vit_b": 6.0,
    "vit_l": 12.0,
    "vit_h": 20.0,
}


def min_available_ram_gb(model_type: str) -> float:
    configured = os.getenv("SAM_MIN_AVAILABLE_RAM_GB")
    if configured is not None:
        try:
            min_gb = float(configured)
        except ValueError as exc:
            raise RuntimeError("SAM_MIN_AVAILABLE_RAM_GB must be a number") from exc
        if min_gb < 0:
            raise RuntimeError("SAM_MIN_AVAILABLE_RAM_GB must be greater than or equal to 0")
        return min_gb
    return DEFAULT_MIN_AVAILABLE_RAM_GB.get(model_type, DEFAULT_MIN_AVAILABLE_RAM_GB["vit_b"])


def available_ram_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None

    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        return int(parts[1]) / 1024 / 1024
    return None


def require_available_ram(*, min_gb: float, model_type: str):
    if min_gb <= 0:
        return

    available_gb = available_ram_gb()
    if available_gb is None:
        return
    if available_gb >= min_gb:
        return

    raise RuntimeError(
        f"Refusing to load SAM {model_type} on CPU: "
        f"available RAM is {available_gb:.1f} GB, minimum is {min_gb:.1f} GB. "
        "Close other applications, choose a smaller SAM_MODEL_TYPE, use SAM_DEVICE=cuda, "
        "or lower/disable this guard with SAM_MIN_AVAILABLE_RAM_GB."
    )


def extract_sam_prompt(
    context: dict | None,
    *,
    image_width: int,
    image_height: int,
    default_label: str,
) -> dict | None:
    if not context:
        return None

    point_coords = []
    point_labels = []
    input_box = None
    selected_label = default_label

    for result in iter_context_results(context):
        value = result.get("value")
        if not isinstance(value, dict):
            continue

        result_type = result.get("type")
        labels = value.get(result_type) if isinstance(result_type, str) else None
        if isinstance(labels, list) and labels:
            selected_label = str(labels[0])

        if result_type == "keypointlabels" and "x" in value and "y" in value:
            point_coords.append([
                percent_to_px(value["x"], image_width),
                percent_to_px(value["y"], image_height),
            ])
            is_positive = result.get("is_positive", True)
            point_labels.append(1 if is_positive else 0)
        elif result_type == "rectanglelabels" and all(key in value for key in ("x", "y", "width", "height")):
            x1 = percent_to_px(value["x"], image_width)
            y1 = percent_to_px(value["y"], image_height)
            x2 = percent_to_px(float(value["x"]) + float(value["width"]), image_width)
            y2 = percent_to_px(float(value["y"]) + float(value["height"]), image_height)
            input_box = np.array([x1, y1, x2, y2])

    if not point_coords and input_box is None:
        value = find_point_value(context)
        if value is None:
            return None
        point_coords.append([
            percent_to_px(value["x"], image_width),
            percent_to_px(value["y"], image_height),
        ])
        point_labels.append(1)

    return {
        "point_coords": np.array(point_coords) if point_coords else None,
        "point_labels": np.array(point_labels) if point_labels else None,
        "box": input_box,
        "label": selected_label,
    }


def iter_context_results(value: Any):
    if isinstance(value, dict):
        if isinstance(value.get("value"), dict) and isinstance(value.get("type"), str):
            yield value
        for child in value.values():
            yield from iter_context_results(child)
    elif isinstance(value, list):
        for item in value:
            yield from iter_context_results(item)


def find_point_value(value: Any) -> dict | None:
    if isinstance(value, dict):
        nested_value = value.get("value")
        if is_point_value(nested_value):
            return nested_value
        if is_point_value(value):
            return value
        for child in value.values():
            result = find_point_value(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        for item in value:
            result = find_point_value(item)
            if result is not None:
                return result
    return None


def is_point_value(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "x" in value
        and "y" in value
        and "width" not in value
        and "height" not in value
    )


def percent_to_px(value: float, size: int) -> int:
    return int(np.clip(float(value), 0, 100) * size / 100.0)


def image_cache_key(task: dict) -> str:
    task_id = task.get("id")
    image_value = task.get("data", {}).get("image", "")
    return f"{task_id}:{image_value}"


def empty_prediction(model_version: str) -> dict:
    return {
        "result": [],
        "score": 0.0,
        "model_version": model_version,
    }


def safe_log_params(params: dict) -> dict:
    redacted_keys = {"token", "password", "api_key", "access_token", "authorization"}
    safe_params = {}
    for key, value in params.items():
        key_text = str(key)
        if any(secret in key_text.lower() for secret in redacted_keys):
            safe_params[key_text] = "<redacted>"
        else:
            safe_params[key_text] = value
    return safe_params


def parse_grounding_sam_prompt(prompt: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"[,\n]+", str(prompt))
        if item.strip()
    ]


def scalar_value(value) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def mask_to_result(
    *,
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    from_name: str,
    to_name: str,
    label: str,
    score: float,
    output: str,
) -> list[dict]:
    if output == "brush":
        return [mask_to_brush_result(
            mask=mask,
            image_width=image_width,
            image_height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=label,
            score=score,
        )]

    return [
        polygon_to_result(
            points=points,
            image_width=image_width,
            image_height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=label,
            score=score,
        )
        for points in mask_to_polygons(mask, image_width=image_width, image_height=image_height)
    ]


def mask_to_brush_result(
    *,
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    from_name: str,
    to_name: str,
    label: str,
    score: float,
) -> dict:
    try:
        from label_studio_converter.brush import mask2rle
    except ImportError as exc:
        raise RuntimeError(
            "Brush output requires label-studio-converter. "
            "Set SAM_OUTPUT=polygon or install label-studio-converter."
        ) from exc

    rle = mask2rle((mask > 0).astype(np.uint8) * 255)
    return {
        "original_width": image_width,
        "original_height": image_height,
        "image_rotation": 0,
        "from_name": from_name,
        "to_name": to_name,
        "type": "brushlabels",
        "value": {
            "format": "rle",
            "rle": rle,
            "brushlabels": [label],
        },
        "score": score,
    }


def mask_to_polygons(
    mask: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> list[list[list[float]]]:
    import cv2

    mask_array = np.asarray(mask)
    if mask_array.ndim > 2:
        mask_array = np.squeeze(mask_array)
    mask_array = (mask_array > 0).astype("uint8") * 255

    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        points = []
        for point in approx.reshape(-1, 2):
            x, y = [float(item) for item in point]
            points.append([
                max(0.0, min(100.0, 100.0 * x / image_width)),
                max(0.0, min(100.0, 100.0 * y / image_height)),
            ])
        polygons.append(points)
    return polygons


def polygon_to_result(
    *,
    points: list[list[float]],
    image_width: int,
    image_height: int,
    from_name: str,
    to_name: str,
    label: str,
    score: float,
) -> dict:
    return {
        "original_width": image_width,
        "original_height": image_height,
        "image_rotation": 0,
        "from_name": from_name,
        "to_name": to_name,
        "type": "polygonlabels",
        "value": {
            "points": points,
            "polygonlabels": [label],
        },
        "score": score,
    }
