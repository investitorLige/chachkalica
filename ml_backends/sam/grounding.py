"""Grounding-SAM (text-prompt) helpers used by the SAM backend.

These were previously imported from the project-level `preannotation` package;
they now live alongside the backend so `ml_backends/sam` is self-contained.
Only the pieces the backend actually calls are kept: loading the Lang-SAM model
and running a single text-prompt prediction.
"""

import numpy as np


def tensor_to_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def load_model(*, model_weights: str):
    try:
        from lang_sam import LangSAM
    except ImportError as exc:
        raise RuntimeError(
            "Lang-SAM is required for text-prompt (grounding) segmentation. "
            "Install lang-segment-anything/lang-sam in the backend image."
        ) from exc

    if model_weights:
        try:
            return LangSAM(sam_ckpt_path=model_weights)
        except TypeError:
            return LangSAM(model_weights)
    return LangSAM()


def predict_for_prompt(*, model, image_pil, text_prompt: str, confidence: float):
    if hasattr(model, "gdino") and hasattr(model, "sam"):
        gdino_results = model.gdino.predict(
            [image_pil],
            [text_prompt],
            confidence,
            confidence,
        )
        if not gdino_results:
            return {"boxes": [], "scores": []}

        result = gdino_results[0]
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels") or []

        masks = []
        mask_scores = []
        if labels and boxes is not None:
            boxes_array = np.asarray(tensor_to_list(boxes), dtype=float)
            if len(boxes_array):
                masks, mask_scores, _ = model.sam.predict(np.asarray(image_pil), boxes_array)

        return {"boxes": boxes, "scores": scores, "masks": masks, "mask_scores": mask_scores}

    return model.predict(
        [image_pil],
        [text_prompt],
        box_threshold=confidence,
        text_threshold=confidence,
    )


def extract_prediction_parts(prediction):
    if isinstance(prediction, list):
        if not prediction:
            return [], [], []
        prediction = prediction[0]

    if isinstance(prediction, dict):
        boxes = prediction.get("boxes")
        scores = prediction.get("scores")
        if scores is None:
            scores = prediction.get("logits")
        masks = prediction.get("masks")
        return tensor_to_list(boxes), tensor_to_list(scores), tensor_to_list(masks)

    if isinstance(prediction, tuple) and len(prediction) >= 4:
        _, boxes, _, logits = prediction[:4]
        return tensor_to_list(boxes), tensor_to_list(logits), []

    return [], [], []
