"""Decide whether a generated variant still matches its source's labels.

A generative edit is asked to change only the look of a scene, but nothing
guarantees it did: a diffusion model can nudge a person, redraw a hand, drop a
helmet, or reframe the whole shot. The source's boxes are copied onto the
variant unchanged, so any of those makes the copied label wrong. This module
is what stands between "the prompt said preserve geometry" and "the labels are
trusted".

The contract is :class:`AugmentationValidator` — one ``validate(original,
generated, boxes)`` call returning a :class:`ValidationResult` — so a second
validator (running the existing detector, say) plugs in without touching the
builder or the studio.

:class:`GeometryValidator`, the default, compares *structure*, not pixels.
Every prompt this feature exists for (night, low light, compression, blur,
colour cast) changes pixels everywhere by design, so a raw pixel or SSIM
comparison would reject exactly the outputs we want. Instead both images go
through the same photometric-invariant transform — grey, blurred, then locally
normalized to zero mean / unit variance — which keeps edges and shapes and
throws away brightness, contrast and most noise. On those maps:

* **Global shift** — a masked correlation search over the frame's
  *background* (boxes masked out). A camera that moved or a reframed shot shifts every box at once.
* **Per annotated box** — each box (plus a margin) is resampled to a fixed
  grid, so a 20 px helmet and a 600 px person are judged at the same relative
  scale. The source box is searched for in the generated window by normalized
  cross-correlation: the score where the label says the object is gives
  ``similarity`` (is it still the same shape at all?), and the offset of the
  best score gives ``bbox_drift`` (how far it moved, as a fraction of its own
  size). An object that vanished, changed shape, or was replaced scores low
  similarity even when it did not "move".

Background changes are not checked beyond the global shift: they are allowed.
Boxes too small to judge (a few pixels) are skipped and counted, never used to
reject — a 6 px box has no structure to compare.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

#: Cells each box spans, per side, once resampled for comparison.
_BOX_GRID = 48
#: Margin around each box, as a fraction of its size, so a small move still
#: overlaps the compared window instead of sliding out of it.
_BOX_MARGIN = 0.15
#: Boxes whose shorter side is under this many pixels are skipped.
_MIN_CHECKABLE_SIDE = 12
#: Long side the global check works at; the frame-level question is coarse.
_GLOBAL_LONG_SIDE = 256
#: Largest global shift searched for, as a fraction of the frame's short side.
_GLOBAL_SEARCH = 0.1
#: How much better the best offset must score than the labelled position
#: before it counts as a move (see _check_box).
_MIN_MOVE_GAIN = 0.1
#: A generated image whose grey std is below this came back blank/flat.
_MIN_IMAGE_STD = 2.0


@dataclass
class BoxCheck:
    index: int
    similarity: float | None
    drift: float | None
    checked: bool
    reason: str | None = None


@dataclass
class ValidationResult:
    valid: bool
    bbox_drift: float
    reason: str | None
    global_shift: float = 0.0
    min_similarity: float | None = None
    boxes_checked: int = 0
    boxes_skipped: int = 0
    boxes: list[BoxCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("bbox_drift", "global_shift", "min_similarity"):
            if data[key] is not None:
                data[key] = round(float(data[key]), 4)
        for box in data["boxes"]:
            for key in ("similarity", "drift"):
                if box[key] is not None:
                    box[key] = round(float(box[key]), 4)
        return data


class AugmentationValidator:
    """Interface: judge one generated variant against its source.

    ``original`` and ``generated`` are HxWx3 uint8 RGB arrays of the same size;
    ``boxes`` are absolute pixel ``(x1, y1, x2, y2)`` in that frame.
    """

    name = "base"

    def validate(self, original, generated, boxes) -> ValidationResult:
        raise NotImplementedError


@dataclass
class GeometryThresholds:
    #: Largest allowed per-box move, as a fraction of the box's own size.
    max_bbox_drift: float = 0.05
    #: Smallest allowed structural similarity inside any annotated box.
    min_box_similarity: float = 0.45
    #: Largest allowed whole-frame shift, as a fraction of the frame diagonal.
    max_global_shift: float = 0.02


class GeometryValidator(AugmentationValidator):
    name = "geometry"

    def __init__(self, thresholds: GeometryThresholds | None = None):
        self.thresholds = thresholds or GeometryThresholds()

    def validate(self, original, generated, boxes) -> ValidationResult:
        import cv2
        import numpy as np

        t = self.thresholds
        boxes = [tuple(float(v) for v in box) for box in boxes]
        if generated.shape[:2] != original.shape[:2]:
            # The backend already resizes back to the source size; this only
            # guards a caller that did not, so the boxes still line up.
            generated = cv2.resize(generated, (original.shape[1], original.shape[0]),
                                   interpolation=cv2.INTER_AREA)

        grey_gen = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
        if float(grey_gen.std()) < _MIN_IMAGE_STD:
            return ValidationResult(False, 0.0, "generated image is blank")
        grey_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

        global_shift = _global_shift(grey_orig, grey_gen, boxes)
        if global_shift > t.max_global_shift:
            return ValidationResult(
                False, 0.0,
                f"whole scene shifted {global_shift:.1%} of the frame diagonal "
                f"(max {t.max_global_shift:.1%}) — camera moved or image reframed",
                global_shift=global_shift,
            )

        height, width = grey_orig.shape
        checks: list[BoxCheck] = []
        for index, box in enumerate(boxes):
            checks.append(_check_box(index, box, grey_orig, grey_gen, width, height))

        checked = [c for c in checks if c.checked]
        drift = max((c.drift for c in checked if c.drift is not None), default=0.0)
        sims = [c.similarity for c in checked if c.similarity is not None]
        min_sim = min(sims) if sims else None

        reason = None
        for check in checked:
            if check.similarity is not None and check.similarity < t.min_box_similarity:
                check.reason = "changed"
                reason = reason or (
                    f"box {check.index} changed shape or disappeared "
                    f"(similarity {check.similarity:.2f} < {t.min_box_similarity:.2f})"
                )
            elif check.drift is not None and check.drift > t.max_bbox_drift:
                check.reason = "moved"
                reason = reason or (
                    f"box {check.index} moved {check.drift:.1%} of its size "
                    f"(max {t.max_bbox_drift:.1%})"
                )

        return ValidationResult(
            valid=reason is None,
            bbox_drift=drift,
            reason=reason,
            global_shift=global_shift,
            min_similarity=min_sim,
            boxes_checked=len(checked),
            boxes_skipped=len(checks) - len(checked),
            boxes=checks,
        )


def _structure(grey, blur_sigma: float, window: int):
    """Photometric-invariant structure map: blurred, locally zero-mean/unit-var."""
    import cv2
    import numpy as np

    image = grey.astype(np.float32)
    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    mean = cv2.blur(image, (window, window))
    sq_mean = cv2.blur(image * image, (window, window))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    # The floor keeps flat regions (sky, a dark wall) from being amplified into
    # noise; it is in grey levels, so it is also what stops a night edit's
    # crushed shadows from reading as "changed".
    return (image - mean) / (std + 4.0)


def _global_shift(grey_orig, grey_gen, boxes) -> float:
    """How far the *background* moved — annotated boxes are masked out.

    Otherwise one large moved object dominates the correlation and reads as
    "the camera moved", which is the wrong diagnosis: the per-box check is the
    one that should report it, with the box's index.

    A masked correlation search, not phase correlation: zeroing the boxes out
    of both maps would leave identical hard mask edges in each, which phase
    correlation happily locks onto at zero shift. ``matchTemplate`` with a
    mask ignores those pixels outright. Shifts up to ``_GLOBAL_SEARCH`` of the
    frame are measured; anything larger fails every box check anyway.
    """
    import cv2
    import math
    import numpy as np

    height, width = grey_orig.shape
    scale = min(1.0, _GLOBAL_LONG_SIDE / max(height, width))
    size = (max(32, round(width * scale)), max(32, round(height * scale)))
    a = _structure(cv2.resize(grey_orig, size, interpolation=cv2.INTER_AREA), 1.0, 11)
    b = _structure(cv2.resize(grey_gen, size, interpolation=cv2.INTER_AREA), 1.0, 11)

    mask = np.ones(a.shape, np.float32)
    sx, sy = size[0] / width, size[1] / height
    for x1, y1, x2, y2 in boxes:
        # Padded a little, so an object's own edges do not leak in either.
        pad_x, pad_y = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        c1, c2 = max(0, int((x1 - pad_x) * sx)), min(size[0], int((x2 + pad_x) * sx) + 1)
        r1, r2 = max(0, int((y1 - pad_y) * sy)), min(size[1], int((y2 + pad_y) * sy) + 1)
        mask[r1:r2, c1:c2] = 0.0

    m = max(2, round(_GLOBAL_SEARCH * min(size)))
    template = np.ascontiguousarray(a[m:-m, m:-m])
    template_mask = np.ascontiguousarray(mask[m:-m, m:-m])
    if template_mask.mean() < 0.15:
        return 0.0  # boxes cover nearly the whole frame: no background to judge
    scores = cv2.matchTemplate(np.ascontiguousarray(b), template, cv2.TM_CCORR_NORMED,
                               mask=template_mask)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    at_zero = float(scores[m, m])
    _lo, best, _lo_loc, (bx, by) = cv2.minMaxLoc(scores)
    if best - at_zero <= _MIN_MOVE_GAIN:
        return 0.0
    return math.hypot(bx - m, by - m) / math.hypot(*size)


def _check_box(index, box, grey_orig, grey_gen, width, height) -> BoxCheck:
    """Similarity at the labelled position, and where the object best matches.

    The box's structure in the source is slid over the generated image within
    the margin (normalized cross-correlation at every offset — the
    ``matchTemplate`` search). The score at zero offset is ``similarity``; the
    offset of the best score is the drift. A symmetric degradation (blur,
    noise, compression) lowers the whole correlation surface but leaves its
    peak at zero, so it reads as "less similar", never as "moved".
    """
    import cv2
    import math
    import numpy as np

    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
    box_w, box_h = x2 - x1, y2 - y1
    if min(box_w, box_h) < _MIN_CHECKABLE_SIDE:
        return BoxCheck(index, None, None, checked=False)

    # Resample so the box itself spans _BOX_GRID cells on each side, then
    # carve the source template (the box) and the search window (box + margin)
    # out of the same resampled frame, so both share one scale.
    sx, sy = _BOX_GRID / box_w, _BOX_GRID / box_h
    pad = math.ceil(_BOX_GRID * _BOX_MARGIN)
    wx1 = max(0, math.floor(x1 - pad / sx))
    wy1 = max(0, math.floor(y1 - pad / sy))
    wx2 = min(width, math.ceil(x2 + pad / sx))
    wy2 = min(height, math.ceil(y2 + pad / sy))
    size = (max(1, round((wx2 - wx1) * sx)), max(1, round((wy2 - wy1) * sy)))

    def window(grey):
        crop = grey[wy1:wy2, wx1:wx2]
        interp = cv2.INTER_AREA if size[0] < crop.shape[1] else cv2.INTER_LINEAR
        return _structure(cv2.resize(crop, size, interpolation=interp), 1.5, 9)

    a, b = window(grey_orig), window(grey_gen)
    tx1, ty1 = round((x1 - wx1) * sx), round((y1 - wy1) * sy)
    tx2 = min(size[0], max(tx1 + 4, round((x2 - wx1) * sx)))
    ty2 = min(size[1], max(ty1 + 4, round((y2 - wy1) * sy)))
    template = np.ascontiguousarray(a[ty1:ty2, tx1:tx2])
    if float(template.std()) < 1e-3:
        # A featureless box (flat colour) has nothing to correlate against.
        return BoxCheck(index, None, None, checked=False)

    scores = cv2.matchTemplate(np.ascontiguousarray(b), template, cv2.TM_CCOEFF_NORMED)
    scores = np.nan_to_num(scores, nan=-1.0)
    similarity = float(scores[ty1, tx1])
    _min_v, best, _min_loc, (bx, by) = cv2.minMaxLoc(scores)
    drift = 0.0
    # Only a clearly better match elsewhere counts as a move. Blur and noise
    # flatten the correlation surface, and on a flat surface the argmax is
    # noise — measured on real frames, a 0.05 margin read an 11 px motion blur
    # on 25-35 px boxes as a 10% move; 0.1 does not, and still catches a real
    # 15% move (whose best score beats the labelled position by far more).
    if best - similarity > _MIN_MOVE_GAIN:
        drift = math.hypot((bx - tx1) / (tx2 - tx1), (by - ty1) / (ty2 - ty1))
    return BoxCheck(index, similarity, drift, checked=True)


def build_validator(settings: dict | None) -> AugmentationValidator | None:
    """The validator a build or preview uses, from its stored settings.

    ``{"enabled": False}`` means "trust the prompt" — every variant is accepted,
    and the manifest says it was never checked.
    """
    settings = settings or {}
    if not settings.get("enabled", True):
        return None
    defaults = GeometryThresholds()
    return GeometryValidator(GeometryThresholds(
        max_bbox_drift=float(settings.get("max_bbox_drift", defaults.max_bbox_drift)),
        min_box_similarity=float(settings.get("min_box_similarity", defaults.min_box_similarity)),
        max_global_shift=float(settings.get("max_global_shift", defaults.max_global_shift)),
    ))
