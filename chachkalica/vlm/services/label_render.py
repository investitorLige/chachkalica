"""Burn a dataset image's ground-truth annotation onto the image itself.

The VLM dataset report shows one row per image, and the question you ask of a
mismatch is always the same: *what was actually labeled here?* Clicking the
thumbnail answers it by drawing the label file's boxes and polygons onto the
full-size image, server-side.

Two things it deliberately does not do:

* **It is not** ``videos.services.inference.draw_boxes``. That draws model
  output, and bakes a confidence into every label — ground truth has none, and
  a box captioned "helmet 0.00" would be worse than no caption. It also cannot
  draw polygons, which label files can carry.
* **It is not used for the thumbnails.** A page of 300 rows would mean 300
  decode-draw-encode round trips; the thumbnails stay raw and only the click
  pays for this.

Colours are the palette ``templates/admin/fleet/dataset_preview_viewer.html``
uses, indexed the same way, so a class is the same colour in the label preview
and here.
"""

#: Same list, same order, as the preview viewer's PALETTE — as #rrggbb.
_PALETTE_HEX = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#800000", "#aaffc3", "#808000", "#000075", "#ff8c00", "#00c853",
]

#: Geometry is scaled against this height so a 4K image does not get hairline
#: boxes and 3-pixel text — the same reference ``draw_boxes`` uses.
_REFERENCE_HEIGHT = 720
_LINE_THICKNESS = 2
_FONT_SCALE = 0.5
#: JPEG quality for the rendered copy. High enough that the annotation is not
#: being judged through compression artifacts.
_JPEG_QUALITY = 92


def _bgr(index: int) -> tuple[int, int, int]:
    """The palette colour for a class id, as cv2's BGR."""
    hex_color = _PALETTE_HEX[index % len(_PALETTE_HEX)]
    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return (blue, green, red)


def render(image_path, shapes) -> bytes:
    """Return ``image_path`` as JPEG bytes with ``shapes`` drawn on it.

    ``shapes`` is what :func:`fleet.services.datasets.label_shapes` returns:
    normalized centre-xywh boxes and flat normalized polygons, each with the
    class name to caption it with. Raises ``RuntimeError`` if the image cannot
    be decoded, which the caller turns into a 404 — a file that is not really an
    image is not a server error.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"could not decode {image_path}")

    height, width = image.shape[:2]
    scale = max(1.0, height / _REFERENCE_HEIGHT)
    thickness = max(1, round(_LINE_THICKNESS * scale))
    font_scale = _FONT_SCALE * scale
    text_thickness = max(1, round(scale))

    for shape in shapes:
        color = _bgr(shape.get("class_id", 0))
        polygon = shape.get("polygon") or []

        if shape.get("kind") == "polygon" and len(polygon) >= 6:
            points = np.array(
                [[int(polygon[i] * width), int(polygon[i + 1] * height)]
                 for i in range(0, len(polygon) - 1, 2)],
                dtype=np.int32,
            )
            cv2.polylines(image, [points], True, color, thickness, cv2.LINE_AA)
            anchor = (int(points[:, 0].min()), int(points[:, 1].min()))
        else:
            box = shape.get("bbox") or {}
            cx, cy = box.get("cx", 0.0), box.get("cy", 0.0)
            w, h = box.get("w", 0.0), box.get("h", 0.0)
            # Clamped into the frame: a label file can carry a box that runs off
            # the edge, and cv2 would happily draw it out of view.
            x1 = min(max(int((cx - w / 2) * width), 0), width - 1)
            y1 = min(max(int((cy - h / 2) * height), 0), height - 1)
            x2 = min(max(int((cx + w / 2) * width), 0), width - 1)
            y2 = min(max(int((cy + h / 2) * height), 0), height - 1)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            anchor = (x1, y1)

        _caption(image, shape.get("class_name") or "?", anchor, color,
                 font_scale, text_thickness, scale)

    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError(f"could not encode {image_path}")
    return encoded.tobytes()


def _caption(image, text: str, anchor, color, font_scale: float,
             text_thickness: int, scale: float) -> None:
    """Draw a filled label chip at ``anchor``, kept inside the image.

    Filled rather than outlined text, because ground-truth boxes are often
    drawn over exactly the busy region that makes bare text unreadable.
    """
    import cv2

    height, width = image.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    pad = max(2, round(3 * scale))
    chip_h = text_h + baseline + 2 * pad

    x = min(anchor[0], max(width - text_w - 2 * pad, 0))
    # Above the box when it fits, otherwise just inside it.
    y = anchor[1] - chip_h if anchor[1] - chip_h >= 0 else anchor[1]
    y = min(y, max(height - chip_h, 0))

    cv2.rectangle(image, (x, y), (x + text_w + 2 * pad, y + chip_h), color, -1)
    cv2.putText(image, text, (x + pad, y + chip_h - baseline - pad),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                text_thickness, cv2.LINE_AA)
