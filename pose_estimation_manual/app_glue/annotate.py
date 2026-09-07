"""Turn model output into boxes-as-dicts and a picture with those boxes drawn on.

Two steps that have to agree on one coordinate convention, so they live together:

- :func:`friendy_to_dicts` — the pipelines return Friendy ``(N, 6)`` tensors of
  ``[cx, cy, w, h, confidence, class_id]``, normalized to the frame. That becomes
  the dict shape used everywhere downstream (database, SSE payload, admin readout).
- :func:`draw_boxes` — burns those dicts onto a BGR frame, coloured red inside a
  danger zone (see :func:`detections.services.zones.evaluate`) and by
  :func:`class_color_hex` otherwise.
- :func:`draw_zones` — outlines the camera's danger zones underneath them.

Normalized coordinates are what make the resize in :func:`annotate` free of
consequence: the same boxes and polygons are correct against the full-resolution
frame and against the downscaled copy that actually ships.
"""

from __future__ import annotations

from typing import Any, Optional

# A box inside a danger zone, in red — overrides the class colour, because the
# one thing on this frame that needs to be obvious at a glance from across a
# room is whether an alert was real, not which class triggered it.
_BREACH_COLOR = (0, 0, 255)
# Zone outlines: amber, so a drawn boundary reads as scenery next to the boxes
# rather than competing with them.
_ZONE_COLOR = (0, 170, 255)
# Fallback for a box with no class id at all (malformed input) — the old flat box
# colour, kept only for that edge case now that a known class gets its own colour.
_BOX_COLOR = (0, 220, 0)

# Per-class colours, RGB hex. Cycled by class id rather than assigned in the order
# classes happen to appear, so a class draws the same colour across frames, cameras
# and worker restarts without remembering anything. Red and orange are deliberately
# absent: this frame already spends them as signals — a breach (:data:`_BREACH_COLOR`)
# and a zone outline (:data:`_ZONE_COLOR`) — and a class box in either colour would
# read as one of those by coincidence. What's left is Matplotlib's "tab10"
# categorical palette with its red and orange entries dropped.
#
# Hex rather than BGR because this list is also handed to the frontend (see
# ``stream.serializers.model_payload``), which draws its own overlay on the live HLS
# video and needs to agree with the colour already burned into this frame's boxes.
CLASS_PALETTE_HEX = [
    "#1f77b4",  # blue
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#bcbd22",  # olive
    "#7f7f7f",  # gray
]
_ZONE_THICKNESS = 2
_BOX_THICKNESS = 2
_FONT_SCALE = 0.5
# Line widths and text are scaled relative to this height, so a box on a 4K frame
# isn't outlined in a hairline that vanishes when the image is viewed scaled down.
_REFERENCE_HEIGHT = 720.0


def class_color_hex(class_id: Optional[int]) -> str:
    """The stable hex colour for ``class_id``, cycling through :data:`CLASS_PALETTE_HEX`.

    Public (unlike the rest of this module's colour constants) because
    ``stream.serializers.model_payload`` calls this too, so a frontend colouring its
    own overlay lands on the exact colour already burned into the frame here.
    """
    if class_id is None:
        return "#808080"
    return CLASS_PALETTE_HEX[int(class_id) % len(CLASS_PALETTE_HEX)]


def _class_color_bgr(class_id: Optional[int]) -> tuple[int, int, int]:
    """:func:`class_color_hex`, converted to the BGR tuple ``cv2.rectangle`` wants."""
    hex_color = class_color_hex(class_id)
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return (b, g, r)


def friendy_to_dicts(preds, engine=None, score_threshold: float = 0.0) -> list[dict[str, Any]]:
    """Friendy ``(N, 6)`` -> the canonical list of box dicts.

    A seventh column, if the pipeline was asked for one, is the index of the
    person this detection was found on and lands as ``person_idx`` (``None`` where
    the row belongs to no person). It is a position in *this frame's* person list
    and means nothing outside it, which is why the worker reads it for zone
    anchoring and drops it before anything is stored or published.

    ``engine`` supplies class names via ``Engine.class_name``; without one the
    boxes still carry their numeric ``class_id`` and get a ``class_<id>`` label.

    ``score_threshold`` is applied again here even though the pipelines already
    filter: a chained or tiled pipeline merges predictions from several passes, and
    this is the last point where one number governs everything that gets stored.
    """
    if preds is None:
        return []

    # Accept a torch tensor or a numpy array without importing torch — this is
    # called from the worker, but also from tests with plain arrays.
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    else:
        import numpy as np

        preds = np.asarray(preds)

    if preds.size == 0:
        return []
    width = preds.shape[-1] if preds.ndim >= 2 and preds.shape[-1] > 6 else 6
    preds = preds.reshape(-1, width)

    boxes = []
    for row in preds:
        confidence = float(row[4])
        if confidence < score_threshold:
            continue
        class_id = int(row[5])
        box = {
            "cx": round(float(row[0]), 6),
            "cy": round(float(row[1]), 6),
            "w": round(float(row[2]), 6),
            "h": round(float(row[3]), 6),
            "class_id": class_id,
            "class_name": engine.class_name(class_id) if engine is not None
                          else f"class_{class_id}",
            "confidence": round(confidence, 4),
        }
        if width > 6:
            person_idx = int(row[6])
            box["person_idx"] = person_idx if person_idx >= 0 else None
        boxes.append(box)
    # Most confident first: a frontend showing "the top few" then gets the right few,
    # and the admin readout reads sensibly without sorting again.
    boxes.sort(key=lambda box: box["confidence"], reverse=True)
    return boxes


def draw_boxes(frame, boxes: list[dict[str, Any]]) -> None:
    """Burn normalized centre-based boxes onto ``frame`` (a BGR ndarray), in place.

    Line widths and label text scale with the frame height (see
    :data:`_REFERENCE_HEIGHT`), and both the box and its label are clamped into the
    frame — a detection whose box extends past the top edge would otherwise have its
    label drawn at a negative y and simply vanish.
    """
    import cv2

    height, width = frame.shape[:2]
    scale = max(1.0, height / _REFERENCE_HEIGHT)
    thickness = max(1, round(_BOX_THICKNESS * scale))
    font_scale = _FONT_SCALE * scale
    text_thickness = max(1, round(scale))
    gap = round(6 * scale)

    for box in boxes:
        cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
        x1 = min(max(int((cx - w / 2) * width), 0), width - 1)
        y1 = min(max(int((cy - h / 2) * height), 0), height - 1)
        x2 = min(max(int((cx + w / 2) * width), 0), width - 1)
        y2 = min(max(int((cy + h / 2) * height), 0), height - 1)
        # Red beats class colour: whether a box is inside a danger zone is the one
        # distinction that must be unmissable, since the stored frame is what
        # somebody looks at to decide whether an alert was real. Outside a zone, the
        # box takes its class's own colour so classes are still tellable apart.
        color = _BREACH_COLOR if box.get("zone_ids") else _class_color_bgr(box.get("class_id"))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # The track id goes first so it reads at a glance on the live view: an id
        # that holds steady while someone walks across frame is the tracker working,
        # and one that climbs every frame is it thrashing.
        track_id = box.get("track_id")
        prefix = f"#{track_id} " if track_id is not None else ""
        # Confidence is absent on the anchor-person boxes `zones.breached_anchors`
        # synthesizes for drawing — they were never scored, so "0.00" would be a
        # lie rather than a missing value.
        confidence = box.get("confidence")
        suffix = f" {confidence:.2f}" if confidence is not None else ""
        label = f"{prefix}{box.get('class_name', '?')}{suffix}"
        (_, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        # Above the box when it fits, otherwise just inside its top edge.
        text_y = y1 - gap if y1 - gap - text_h > 0 else y1 + text_h + gap
        text_y = min(text_y, height - 1)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, text_thickness, cv2.LINE_AA)


def draw_zones(frame, zone_specs: list[dict[str, Any]]) -> None:
    """Outline each danger zone on ``frame`` (BGR ndarray), in place.

    Drawn before the boxes so an object standing in a zone is never hidden behind
    the boundary it crossed. This is what makes a zone checkable at all: a polygon
    is a list of numbers in a database until somebody sees it lying over the actual
    scene, at which point "that corner is a metre too far left" is obvious.
    """
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    scale = max(1.0, height / _REFERENCE_HEIGHT)
    thickness = max(1, round(_ZONE_THICKNESS * scale))
    font_scale = _FONT_SCALE * scale
    text_thickness = max(1, round(scale))

    for spec in zone_specs:
        polygon = spec.get("polygon") or []
        if len(polygon) < 3:
            continue
        points = np.array(
            [[min(max(int(p["x"] * width), 0), width - 1),
              min(max(int(p["y"] * height), 0), height - 1)] for p in polygon],
            dtype=np.int32,
        )
        cv2.polylines(frame, [points], isClosed=True, color=_ZONE_COLOR,
                      thickness=thickness, lineType=cv2.LINE_AA)

        name = spec.get("name")
        if not name:
            continue
        # At the topmost vertex, nudged inside the frame so a zone drawn against the
        # top edge doesn't lose its label off-screen.
        anchor = min(points.tolist(), key=lambda p: p[1])
        (_, text_h), _ = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        text_y = max(anchor[1] - round(4 * scale), text_h + 1)
        cv2.putText(frame, str(name), (anchor[0], text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, _ZONE_COLOR, text_thickness, cv2.LINE_AA)


def annotate(frame, boxes: list[dict[str, Any]],
             max_width: Optional[int] = None,
             zone_specs: Optional[list[dict[str, Any]]] = None) -> tuple[bytes, int, int]:
    """Downscale ``frame``, draw ``zone_specs`` and ``boxes`` on the copy, encode it.

    Returns ``(jpeg_bytes, width, height)``.

    **Resize first, draw second.** The annotated frame's only consumer is a browser,
    so unlike the raw frame there is no reason a full-resolution copy should exist
    at all — and drawing afterwards keeps the outlines and labels crisp, where
    drawing first would shrink them along with the picture until thin boxes
    disappeared.

    The source ``frame`` is never modified: ``resize`` returns a new array when it
    actually scales, and a copy is taken when it doesn't. The caller's frame is the
    full-resolution one the model saw, and something else may still want it.
    """
    from cameras.services import preview

    if max_width is None:
        max_width = preview.detection_frame_max_width()

    scaled = preview.resize(frame, max_width=max_width)
    if scaled is frame:
        # Already narrow enough — `resize` handed back the same object, and drawing
        # on it would scribble on the caller's frame.
        scaled = frame.copy()

    if zone_specs:
        draw_zones(scaled, zone_specs)
    draw_boxes(scaled, boxes)
    height, width = scaled.shape[:2]
    return preview.encode(scaled), int(width), int(height)
