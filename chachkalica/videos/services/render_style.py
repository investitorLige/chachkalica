"""The look of a marketing render: one validated style dict, one renderer.

``videos.services.inference.draw_boxes`` is the *utilitarian* overlay — one line
width, one label format, colors picked by hashing the class name. It exists to
let an operator see what a model did, and the camera live path shares it, so its
appearance is not something to keep re-tuning.

This module is the other half: everything needed to make a clip presentable —
rounded or bracketed boxes, translucent fills, pill labels, a dimmed background
that spotlights the detections, a per-class counter, a watermark — plus the
temporal easing (:class:`_Tracker`) that turns per-frame model output into
boxes that glide instead of flickering. The "Run model inference for marketing…"
action writes one of these style dicts onto ``InferenceJob.render_style``; a job
with an empty ``render_style`` renders the utilitarian way, unchanged.

The knobs live in the ``_*_FIELDS`` tables below rather than as scattered
literals, so the admin form parser (:func:`parse_form`), the stored-JSON
validator (:func:`normalize`) and the renderer all agree on names, ranges and
defaults by construction — adding a knob means adding one table row plus the
drawing code that reads it.

Two conventions worth stating, because both differ from ``draw_boxes``:

* Sizes (thickness, corner radius, label size, margins) are in **pixels at
  1080p** and scale with the frame, same reference as ``draw_boxes``, but they
  scale *both ways* here: a 720p clip cut for social gets proportionally
  slimmer boxes rather than 1080p-sized ones, because a marketing render is
  watched at its own size rather than inspected.
* Colors are ``#rrggbb`` in the style dict (that's what an operator types and
  what an ``<input type=color>`` posts) and BGR tuples inside the renderer,
  which is what cv2 wants. :func:`hex_to_bgr` is the only crossing point.
"""

from __future__ import annotations

import colorsys
import contextlib
import hashlib
import math
import re

# --------------------------------------------------------------------- palettes

#: Curated per-class palettes, as ``#rrggbb``. A class name picks its color by
#: hashing into the chosen palette (see :func:`_class_color`), so the same label
#: keeps the same color across runs, machines and processes — md5 rather than the
#: builtin ``hash()``, which is salted per process.
PALETTES = {
    "vivid": ["#22c55e", "#f59e0b", "#3b82f6", "#ef4444", "#eab308",
              "#ec4899", "#06b6d4", "#8b5cf6", "#f97316", "#14b8a6"],
    "neon": ["#39ff14", "#ff00ff", "#00ffff", "#ffff00", "#ff3131",
             "#ff9900", "#7df9ff", "#bf00ff", "#00ff7f", "#ff1493"],
    "pastel": ["#a7f3d0", "#fde68a", "#bfdbfe", "#fecaca", "#ddd6fe",
               "#fbcfe8", "#c7f9cc", "#ffd6a5", "#b5ead7", "#e2d1f9"],
    "sunset": ["#ff6b6b", "#f9844a", "#fee440", "#ff8fab", "#c1121f",
               "#ffb703", "#fb8500", "#e76f51", "#f4a261", "#e63946"],
    "ocean": ["#00b4d8", "#0077b6", "#48cae4", "#90e0ef", "#023e8a",
              "#2a9d8f", "#00c2a8", "#5390d9", "#4cc9f0", "#118ab2"],
    "mono": ["#ffffff", "#e5e5e5", "#cccccc", "#b3b3b3", "#999999",
             "#ffffff", "#e5e5e5", "#cccccc", "#b3b3b3", "#999999"],
}

PALETTE_CHOICES = [
    ("vivid", "vivid — saturated, high contrast"),
    ("neon", "neon — glowing, dark footage"),
    ("pastel", "pastel — soft, light footage"),
    ("sunset", "sunset — warm reds and oranges"),
    ("ocean", "ocean — cool blues and teals"),
    ("mono", "mono — white / grey only"),
]

COLOR_MODE_CHOICES = [
    ("class", "one color per class (from the palette)"),
    ("fixed", "one color for every box"),
    ("confidence", "gradient by confidence (low → high)"),
]

BOX_STYLE_CHOICES = [
    ("solid", "solid — plain rectangle"),
    ("rounded", "rounded — soft corners"),
    ("corners", "brackets — four corner marks"),
    ("dashed", "dashed — broken rectangle"),
    ("dotted", "dotted — round beads"),
    ("double", "double — two concentric lines"),
    ("chamfer", "chamfer — cut corners, HUD panel"),
    ("crosshair", "crosshair — brackets, edge ticks and a centre mark"),
    ("glow", "glow — neon bloom"),
    ("sketch", "sketch — hand-drawn, doubled strokes"),
    ("underline", "underline — a bar under the subject only"),
    ("none", "none — fill / label only"),
]

ANIMATION_CHOICES = [
    ("none", "still"),
    ("march", "marching — dashes and dots crawl"),
    ("pulse", "pulse — the outline breathes"),
]

LABEL_TEXT_CHOICES = [
    ("class_conf", "class + confidence"),
    ("class", "class only"),
    ("conf", "confidence only"),
]

LABEL_STYLE_CHOICES = [
    ("pill", "pill — rounded filled chip"),
    ("bar", "bar — filled rectangle"),
    ("plain", "plain text"),
    ("outline", "outlined text (no background)"),
]

LABEL_POSITION_CHOICES = [
    ("above", "above the box"),
    ("inside", "inside the box, top"),
    ("below", "below the box"),
    ("inside_bottom", "inside the box, bottom"),
]

LABEL_COLOR_CHOICES = [
    ("auto", "auto — black or white, whichever reads"),
    ("white", "always white"),
    ("black", "always black"),
    ("box", "the box color"),
]

FONT_CHOICES = [
    ("duplex", "duplex — clean sans"),
    ("simplex", "simplex — thin sans"),
    ("triplex", "triplex — heavy serif"),
    ("plain", "plain — small bitmap"),
]

CORNER_CHOICES = [
    ("none", "hidden"),
    ("top_left", "top left"),
    ("top_right", "top right"),
    ("bottom_left", "bottom left"),
    ("bottom_right", "bottom right"),
]

#: cv2 font constants, resolved lazily — this module is imported by the admin,
#: which must not pull in cv2 just to render a form.
_FONTS = {
    "simplex": "FONT_HERSHEY_SIMPLEX",
    "duplex": "FONT_HERSHEY_DUPLEX",
    "triplex": "FONT_HERSHEY_TRIPLEX",
    "plain": "FONT_HERSHEY_PLAIN",
}

# ----------------------------------------------------------------- field tables
# (default, low, high) for numbers; (default, choices) for enumerations. Every
# knob the style dict can hold appears in exactly one table.

_FLOAT_FIELDS = {
    "box_opacity": (1.0, 0.0, 1.0),
    "fill_opacity": (0.12, 0.0, 1.0),
    "label_opacity": (0.85, 0.0, 1.0),
    "label_scale": (1.0, 0.2, 5.0),
    "dim_background": (0.0, 0.0, 0.95),
    "smoothing": (0.35, 0.0, 0.95),
    "min_box_area": (0.0, 0.0, 100.0),
    "watermark_opacity": (0.7, 0.0, 1.0),
    "watermark_scale": (1.0, 0.2, 5.0),
    "counter_scale": (1.0, 0.2, 5.0),
}

_INT_FIELDS = {
    "thickness": (3, 1, 40),
    "corner_radius": (8, 0, 200),
    "corner_length": (22, 2, 500),
    "dash_length": (14, 2, 200),
    "dot_spacing": (14, 2, 200),
    "double_gap": (5, 1, 100),
    "chamfer_size": (16, 2, 200),
    "glow_spread": (7, 1, 60),
    "sketch_jitter": (6, 0, 40),
    "animation_period": (24, 2, 600),
    "confidence_decimals": (2, 0, 3),
    "fade_frames": (3, 0, 120),
    "max_boxes": (0, 0, 500),          # 0 = no cap
    "crf": (20, 0, 51),
    "output_height": (0, 0, 4320),     # 0 = keep the source height
}

_CHOICE_FIELDS = {
    "palette": ("vivid", [v for v, _ in PALETTE_CHOICES]),
    "color_mode": ("class", [v for v, _ in COLOR_MODE_CHOICES]),
    "box_style": ("rounded", [v for v, _ in BOX_STYLE_CHOICES]),
    "animation": ("none", [v for v, _ in ANIMATION_CHOICES]),
    "label_text": ("class_conf", [v for v, _ in LABEL_TEXT_CHOICES]),
    "label_style": ("pill", [v for v, _ in LABEL_STYLE_CHOICES]),
    "label_position": ("above", [v for v, _ in LABEL_POSITION_CHOICES]),
    "label_color": ("auto", [v for v, _ in LABEL_COLOR_CHOICES]),
    "font": ("duplex", [v for v, _ in FONT_CHOICES]),
    "counter": ("none", [v for v, _ in CORNER_CHOICES]),
    "watermark_position": ("bottom_right", [v for v, _ in CORNER_CHOICES if v != "none"]),
}

_BOOL_FIELDS = {
    "show_label": True,
    "label_uppercase": False,
    "shadow": True,
}

_COLOR_FIELDS = {
    "fixed_color": "#22c55e",
    "conf_low_color": "#ef4444",
    "conf_high_color": "#22c55e",
    "watermark_color": "#ffffff",
}

_TEXT_FIELDS = {
    "watermark_text": "",
    #: Non-empty relabels every drawn box to this class name — see
    #: :meth:`MarketingRenderer._select`. Free text rather than a validated
    #: enumeration: it is only ever drawn, never matched against a class space,
    #: and the form fills its dropdown from the selected model's classes.
    "force_class": "",
}

#: Names shown in validation messages — only where the field name alone would be
#: cryptic in a red banner.
_LABELS = {
    "min_box_area": "Minimum box area (% of frame)",
    "animation_period": "Animation period (frames)",
    "glow_spread": "Glow spread",
    "double_gap": "Double-line gap",
    "chamfer_size": "Chamfer size",
    "dot_spacing": "Dot spacing",
    "sketch_jitter": "Sketch jitter",
    "crf": "Encode quality (CRF)",
    "output_height": "Output height",
    "fade_frames": "Fade in/out frames",
    "counter": "Detection counter",
    "force_class": "Label every box as",
}

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def _label(name: str) -> str:
    return _LABELS.get(name) or name.replace("_", " ").capitalize()


def defaults() -> dict:
    """A fresh style dict with every knob at its default — the look the
    marketing form arrives at before the operator touches anything."""
    style = {}
    for name, (default, _low, _high) in _FLOAT_FIELDS.items():
        style[name] = default
    for name, (default, _low, _high) in _INT_FIELDS.items():
        style[name] = default
    for name, (default, _choices) in _CHOICE_FIELDS.items():
        style[name] = default
    style.update(_BOOL_FIELDS)
    style.update(_COLOR_FIELDS)
    style.update(_TEXT_FIELDS)
    style["class_colors"] = {}
    style["class_filter"] = []
    return style


def normalize(raw) -> dict:
    """Coerce a stored/posted style dict into a complete, in-range one.

    Total by construction: anything missing, mistyped or out of range falls back
    to its default rather than raising, because this runs on the *render* side,
    where the job is already queued and a style written by an older version of
    this file (or hand-edited in the admin) must still produce a video. The
    place that rejects bad input is :func:`parse_form`, at the form, where there
    is still someone to tell.
    """
    raw = raw if isinstance(raw, dict) else {}
    style = defaults()

    for name, (default, low, high) in _FLOAT_FIELDS.items():
        try:
            style[name] = min(max(float(raw[name]), low), high)
        except (KeyError, TypeError, ValueError):
            style[name] = default
    for name, (default, low, high) in _INT_FIELDS.items():
        try:
            style[name] = min(max(int(raw[name]), low), high)
        except (KeyError, TypeError, ValueError):
            style[name] = default
    for name, (default, choices) in _CHOICE_FIELDS.items():
        value = raw.get(name)
        style[name] = value if value in choices else default
    for name, default in _BOOL_FIELDS.items():
        style[name] = bool(raw.get(name, default))
    for name, default in _COLOR_FIELDS.items():
        value = raw.get(name)
        style[name] = value if isinstance(value, str) and _HEX_RE.match(value) else default
    for name, default in _TEXT_FIELDS.items():
        value = raw.get(name)
        style[name] = value.strip() if isinstance(value, str) else default

    colors = raw.get("class_colors")
    style["class_colors"] = {
        str(k): v for k, v in colors.items()
        if isinstance(v, str) and _HEX_RE.match(v)
    } if isinstance(colors, dict) else {}

    names = raw.get("class_filter")
    style["class_filter"] = [str(n) for n in names if str(n).strip()] \
        if isinstance(names, list) else []
    return style


def parse_form(post) -> tuple[dict, str | None]:
    """Read a style dict out of the marketing form's POST data.

    Returns ``(style, None)``, or ``(style, message)`` naming the first bad
    field. Parsing does not stop at that first problem: the caller re-renders the
    form from the returned dict, and losing the other twenty-nine knobs someone
    just dialled in because one of them is out of range would be its own bug.
    Only the offending fields fall back to their defaults.

    Blank means "leave it at the default" for every knob — the form posts blanks
    for the rows it is hiding.
    """
    style = defaults()
    errors = []

    def read(name, coerce, kind):
        raw = (post.get(f"style_{name}") or "").strip()
        if not raw:
            return None
        try:
            return coerce(raw)
        except ValueError:
            errors.append(f"{_label(name)} must be {kind}.")
            return None

    for name, (default, low, high) in _FLOAT_FIELDS.items():
        value = read(name, float, "a number")
        if value is None:
            continue
        if not low <= value <= high:
            errors.append(f"{_label(name)} must be between {low} and {high}.")
            continue
        style[name] = value

    for name, (default, low, high) in _INT_FIELDS.items():
        value = read(name, int, "a whole number")
        if value is None:
            continue
        if not low <= value <= high:
            errors.append(f"{_label(name)} must be between {low} and {high}.")
            continue
        style[name] = value

    for name, (default, choices) in _CHOICE_FIELDS.items():
        raw = (post.get(f"style_{name}") or "").strip()
        if not raw:
            continue
        if raw not in choices:
            errors.append(f"Unknown {_label(name).lower()}: {raw!r}.")
            continue
        style[name] = raw

    # A checkbox posts nothing when unticked, so its absence *is* the value —
    # unlike every other field, where absence means "default". The form marks the
    # ones it rendered with a companion hidden input so this can tell "unticked"
    # from "this form never had the field".
    for name in _BOOL_FIELDS:
        if post.get(f"style_has_{name}"):
            style[name] = bool(post.get(f"style_{name}"))

    for name in _COLOR_FIELDS:
        raw = (post.get(f"style_{name}") or "").strip()
        if not raw:
            continue
        if not _HEX_RE.match(raw):
            errors.append(f"{_label(name)} must be a hex color like #22c55e.")
            continue
        style[name] = raw.lower()

    for name in _TEXT_FIELDS:
        style[name] = (post.get(f"style_{name}") or "").strip()[:200]

    class_colors, error = _parse_class_colors(post.get("style_class_colors"))
    if error:
        errors.append(error)
    else:
        style["class_colors"] = class_colors

    style["class_filter"] = [
        part.strip() for part in (post.get("style_class_filter") or "").split(",")
        if part.strip()
    ]
    return style, (errors[0] if errors else None)


def _parse_class_colors(raw: str | None) -> tuple[dict, str | None]:
    """``"helmet=#ff0000, vest=#0f0"`` → ``{"helmet": "#ff0000", "vest": "#0f0"}``.

    Per-class overrides beat the palette, which is what an operator reaches for
    when one class has to match a brand color.
    """
    colors = {}
    for part in (raw or "").replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not name:
            return {}, (f"Per-class colors look like 'helmet=#ff0000, vest=#3b82f6'; "
                        f"could not read {part!r}.")
        if not _HEX_RE.match(value):
            return {}, f"{name}: {value!r} is not a hex color like #ff0000."
        colors[name] = value.lower()
    return colors, None


def format_class_colors(colors: dict) -> str:
    """The inverse of :func:`_parse_class_colors`, for re-rendering the form."""
    return ", ".join(f"{name}={value}" for name, value in sorted((colors or {}).items()))


def form_values(style: dict) -> dict:
    """``style`` flattened into the ``style_<name>`` keys the template renders,
    with the two structured fields turned back into their text form."""
    values = {f"style_{name}": value for name, value in style.items()
              if name not in ("class_colors", "class_filter")}
    values["style_class_colors"] = format_class_colors(style.get("class_colors"))
    values["style_class_filter"] = ", ".join(style.get("class_filter") or [])
    return values


# ----------------------------------------------------------------------- colors

def hex_to_bgr(value: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` (or the 3-digit short form) → a cv2 BGR tuple."""
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return (b, g, r)


#: Brightness multipliers applied on top of the palette entry (see
#: :func:`_class_color`). Ten palette colors alone collide surprisingly often —
#: three classes have a ~1-in-4 chance of two of them sharing a color, which on a
#: marketing clip looks like a bug — and one hue at three brightnesses is still
#: three tellable-apart colors. Widening the palettes instead would mean
#: inventing thirty colors per palette that all still work together.
_CLASS_SHADES = (1.0, 0.72, 1.25)


def _class_color(name: str, style: dict) -> tuple[int, int, int]:
    """The color for one class label: stable for a given name, and independent of
    what else is in the frame — so a preview and the video it previews agree, and
    two runs of the same clip match."""
    override = (style.get("class_colors") or {}).get(name)
    if override:
        return hex_to_bgr(override)
    palette = PALETTES.get(style["palette"], PALETTES["vivid"])
    if not name:
        return hex_to_bgr(palette[0])
    digest = hashlib.md5(name.encode()).digest()
    base = hex_to_bgr(palette[digest[0] % len(palette)])
    shade = _CLASS_SHADES[digest[1] % len(_CLASS_SHADES)]
    return tuple(int(min(255, max(0, round(channel * shade)))) for channel in base)


def _confidence_color(confidence: float, style: dict) -> tuple[int, int, int]:
    """Interpolate low→high in HSV rather than RGB: a straight RGB lerp between
    two saturated hues runs through a muddy grey in the middle, which reads as
    "broken gradient" on a video."""
    low = colorsys.rgb_to_hsv(*[c / 255 for c in reversed(hex_to_bgr(style["conf_low_color"]))])
    high = colorsys.rgb_to_hsv(*[c / 255 for c in reversed(hex_to_bgr(style["conf_high_color"]))])
    t = min(max((float(confidence) - 0.3) / 0.7, 0.0), 1.0)
    h = low[0] + (high[0] - low[0]) * t
    s = low[1] + (high[1] - low[1]) * t
    v = low[2] + (high[2] - low[2]) * t
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))


def box_color(box: dict, style: dict) -> tuple[int, int, int]:
    mode = style["color_mode"]
    if mode == "fixed":
        return hex_to_bgr(style["fixed_color"])
    if mode == "confidence":
        return _confidence_color(box.get("confidence") or 0.0, style)
    return _class_color(box.get("class_name") or "", style)


def _text_color(background: tuple[int, int, int], style: dict) -> tuple[int, int, int]:
    mode = style["label_color"]
    if mode == "white":
        return (255, 255, 255)
    if mode == "black":
        return (0, 0, 0)
    if mode == "box":
        return background
    # Rec. 601 luma of the chip color decides black-or-white text on it.
    b, g, r = background
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else (255, 255, 255)


# ------------------------------------------------------------------- smoothing

#: Where :meth:`MarketingRenderer._select` parks a box's real class when
#: ``force_class`` has overwritten ``class_name``. Private to this module, and
#: read only through :func:`true_class`.
_TRUE_CLASS = "_true_class"


def true_class(box: dict) -> str:
    """The class the *model* gave ``box`` — its ``class_name``, unless a
    ``force_class`` relabel has moved it aside."""
    return str(box.get(_TRUE_CLASS, box.get("class_name") or "") or "")


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class _Tracker:
    """Frame-to-frame identity for detections, so boxes can ease instead of snap.

    Raw per-frame model output is jittery — a box wobbles by a few pixels every
    frame and drops out entirely for the odd frame — which is invisible in a
    debugging overlay and glaring in a clip someone is meant to watch. Two
    treatments, both keyed on matching this frame's boxes to last frame's by
    class and IoU (greedy, best pair first — good enough at these box counts,
    and it degrades to "no match" rather than to a wrong match). "Class" here is
    the *model's* class (:func:`true_class`), not a ``force_class`` relabel,
    which would otherwise make every box on the frame a match for every other:

    * ``smoothing`` exponentially eases each track's geometry toward its new
      position. With ``frame_stride > 1`` the *same* boxes are handed to the
      renderer for several frames running, so this also reads as the box gliding
      to its next inferred position rather than teleporting there.
    * ``fade_frames`` ramps a new track's opacity up from zero, and keeps a
      vanished one on screen fading out. Blinking boxes are the single most
      amateur-looking thing in a detection reel.

    A faded-out track is model output that has *stopped*, still drawn — which is
    a presentation choice, not a claim about the model. It is bounded by
    ``fade_frames`` and turned off by setting it to 0.
    """

    #: Above this overlap, two boxes of the same class are the same object.
    MATCH_IOU = 0.3

    #: …and so are two that barely overlap but sit within this many box-widths of
    #: each other. Overlap alone is not enough: a subject crossing frame, or any
    #: subject at all once ``frame_stride`` puts several frames between one set of
    #: detections and the next, can move most of its own width between updates —
    #: and a track that fails to match is a box that fades out while its
    #: replacement fades in, i.e. exactly the flicker the fading exists to remove.
    MATCH_DISTANCE = 0.75

    def __init__(self, style: dict):
        self.smoothing = float(style["smoothing"])
        self.fade_frames = int(style["fade_frames"])
        self._tracks: list[dict] = []

    def update(self, boxes: list[dict]) -> list[tuple[dict, float]]:
        """Advance one *rendered* frame and return ``(box, alpha)`` to draw.

        ``box`` is a copy of the model's dict with eased ``cx/cy/w/h``; ``alpha``
        is the fade multiplier in ``[0, 1]``.
        """
        pairs = []
        for i, track in enumerate(self._tracks):
            for j, box in enumerate(boxes):
                if true_class(track["box"]) != true_class(box):
                    continue
                a, b = _geom(track["box"]), _geom(box)
                overlap = _iou(a, b)
                if overlap >= self.MATCH_IOU or _near(a, b, self.MATCH_DISTANCE):
                    # Sorted by overlap first and proximity second, so genuinely
                    # overlapping pairs claim each other before the merely-close
                    # ones get a look in.
                    pairs.append((overlap, -_distance(a, b), i, j))
        pairs.sort(reverse=True)

        used_tracks, used_boxes, matched = set(), set(), {}
        for _overlap, _proximity, i, j in pairs:
            if i in used_tracks or j in used_boxes:
                continue
            used_tracks.add(i)
            used_boxes.add(j)
            matched[i] = j

        alive = []
        for i, track in enumerate(self._tracks):
            if i in matched:
                target = boxes[matched[i]]
                track["box"] = {**target, **_eased(track["box"], target, self.smoothing)}
                track["age"] = min(track["age"] + 1, self.fade_frames)
                track["missing"] = 0
                alive.append(track)
            else:
                track["missing"] += 1
                if track["missing"] <= self.fade_frames:
                    alive.append(track)

        for j, box in enumerate(boxes):
            if j not in used_boxes:
                alive.append({"box": dict(box), "age": 0, "missing": 0})

        self._tracks = alive
        return [(track["box"], self._alpha(track)) for track in alive]

    def _alpha(self, track: dict) -> float:
        if self.fade_frames <= 0:
            return 1.0
        if track["missing"]:
            return max(0.0, 1.0 - track["missing"] / (self.fade_frames + 1))
        return min(1.0, (track["age"] + 1) / (self.fade_frames + 1))


def _distance(a: tuple, b: tuple) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _near(a: tuple, b: tuple, factor: float) -> bool:
    """Whether two boxes are close enough, relative to their own size, to be the
    same object one update apart."""
    size = (a[2] + a[3] + b[2] + b[3]) / 4
    return size > 0 and _distance(a, b) <= factor * size


def _geom(box: dict) -> tuple[float, float, float, float]:
    return (float(box["cx"]), float(box["cy"]), float(box["w"]), float(box["h"]))


def _eased(previous: dict, target: dict, smoothing: float) -> dict:
    if smoothing <= 0:
        return {k: float(target[k]) for k in ("cx", "cy", "w", "h")}
    return {
        k: float(previous[k]) * smoothing + float(target[k]) * (1.0 - smoothing)
        for k in ("cx", "cy", "w", "h")
    }


# ------------------------------------------------------------------- rendering

#: Sizes in the style dict are pixels on a 1080p frame — same reference as
#: ``inference.draw_boxes``, but scaled in both directions here (see module
#: docstring), within these bounds so a thumbnail still gets a visible stroke and
#: an 8K frame doesn't get a 100px one.
_REFERENCE_HEIGHT = 1080
_MIN_SCALE = 0.4
_MAX_SCALE = 4.0

#: cv2's base text height at ``fontScale=1`` for the Hershey fonts, near enough
#: to turn a "label_scale" multiplier into a font scale that looks the same size
#: across fonts.
_BASE_FONT_SCALE = 0.62


@contextlib.contextmanager
def _blend(roi, alpha):
    """Draw into ``roi`` at ``alpha``, compositing on exit.

    Yields the canvas to draw on: ``roi`` itself when the draw is opaque (the
    common case — no copy, no blend), otherwise a scratch copy blended back when
    the ``with`` block ends. Coordinates are always local to ``roi``.
    """
    import cv2

    if alpha >= 0.999:
        yield roi
        return
    layer = roi.copy()
    yield layer
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, roi)


def _rounded_rect(img, p1, p2, color, thickness, radius, filled=False):
    """A rectangle with quarter-circle corners; ``radius=0`` is a plain one."""
    import cv2

    x1, y1 = p1
    x2, y2 = p2
    radius = int(min(radius, abs(x2 - x1) / 2, abs(y2 - y1) / 2))
    if radius <= 0:
        cv2.rectangle(img, p1, p2, color, -1 if filled else thickness, cv2.LINE_AA)
        return

    if filled:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
        return

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    for (cx, cy), start in (((x1 + radius, y1 + radius), 180),
                            ((x2 - radius, y1 + radius), 270),
                            ((x2 - radius, y2 - radius), 0),
                            ((x1 + radius, y2 - radius), 90)):
        cv2.ellipse(img, (cx, cy), (radius, radius), start, 0, 90, color,
                    thickness, cv2.LINE_AA)


def _corner_brackets(img, p1, p2, color, thickness, length):
    """Four L-shaped corner marks — the "viewfinder" look, and the least
    obtrusive way to point at something without boxing it in."""
    import cv2

    x1, y1 = p1
    x2, y2 = p2
    length = int(min(length, abs(x2 - x1) / 2, abs(y2 - y1) / 2))
    if length <= 0:
        return
    for (cx, cy), (dx, dy) in (((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                               ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness, cv2.LINE_AA)


def _perimeter(p1, p2):
    """The four corners as a closed loop of segments, walked clockwise from the
    top-left. Shared by every outline that steps along the box's edge."""
    x1, y1 = p1
    x2, y2 = p2
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return list(zip(corners, corners[1:] + corners[:1]))


def _walk(p1, p2, step, phase=0.0):
    """Yield ``(point, along)`` every ``step`` pixels around the box's perimeter,
    offset by ``phase`` (0–1) of one step.

    One traversal shared by the dashed and dotted outlines, so their marching
    animation is the same offset applied to the same walk rather than two
    hand-rolled loops that drift apart.
    """
    step = max(2, int(step))
    offset = (phase % 1.0) * step
    for (ax, ay), (bx, by) in _perimeter(p1, p2):
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if length < 1:
            continue
        dx, dy = (bx - ax) / length, (by - ay) / length
        along = offset
        while along < length:
            yield (int(round(ax + dx * along)), int(round(ay + dy * along))), along
            along += step


def _dashed_rect(img, p1, p2, color, thickness, dash, phase=0.0):
    """Dashes of ``dash`` pixels with ``dash`` pixels of gap, walked around the
    perimeter so the corners stay in step (and so ``phase`` makes them march)."""
    import cv2

    dash = max(2, int(dash))
    x1, y1 = p1
    x2, y2 = p2
    for (px, py), along in _walk(p1, p2, dash * 2, phase):
        # Each mark runs along the edge it started on; clamped to the box so a
        # dash near a corner stops there instead of overshooting.
        if py in (y1, y2):
            end = (min(px + dash, x2) if py == y1 else max(px - dash, x1), py)
        else:
            end = (px, max(py - dash, y1) if px == x1 else min(py + dash, y2))
        cv2.line(img, (px, py), end, color, thickness, cv2.LINE_AA)


def _dotted_rect(img, p1, p2, color, thickness, spacing, phase=0.0):
    """Round beads around the perimeter — a softer dashed line, and the outline
    that reads best over busy footage at small sizes."""
    import cv2

    radius = max(1, round(thickness * 0.9))
    for point, _along in _walk(p1, p2, spacing, phase):
        cv2.circle(img, point, radius, color, -1, cv2.LINE_AA)


def _double_rect(img, p1, p2, color, thickness, gap, radius=0):
    """Two concentric outlines. Reads as deliberate framing rather than a
    machine's bounding box — the classic "featured" treatment."""
    x1, y1 = p1
    x2, y2 = p2
    gap = max(1, int(gap))
    _rounded_rect(img, p1, p2, color, thickness, radius)
    inner = (x1 + gap, y1 + gap), (x2 - gap, y2 - gap)
    if inner[1][0] - inner[0][0] > 2 and inner[1][1] - inner[0][1] > 2:
        _rounded_rect(img, inner[0], inner[1], color, max(1, thickness // 2),
                      max(0, radius - gap))


def _chamfer_rect(img, p1, p2, color, thickness, cut):
    """An octagon: the rectangle with its corners cut off. The HUD-panel look,
    and the one outline that never collides with a corner-anchored label."""
    import cv2
    import numpy as np

    x1, y1 = p1
    x2, y2 = p2
    cut = int(min(cut, (x2 - x1) / 2, (y2 - y1) / 2))
    if cut <= 0:
        cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)
        return
    points = np.array([
        (x1 + cut, y1), (x2 - cut, y1), (x2, y1 + cut), (x2, y2 - cut),
        (x2 - cut, y2), (x1 + cut, y2), (x1, y2 - cut), (x1, y1 + cut),
    ], dtype=np.int32)
    cv2.polylines(img, [points], True, color, thickness, cv2.LINE_AA)


def _crosshair(img, p1, p2, color, thickness, length):
    """Corner brackets plus a tick at the middle of each edge and a small cross
    at the centre — the "acquiring target" look, without boxing the subject in."""
    import cv2

    x1, y1 = p1
    x2, y2 = p2
    _corner_brackets(img, p1, p2, color, thickness, length)
    tick = max(2, int(length // 2))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.line(img, (cx, y1), (cx, y1 + tick), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, y2), (cx, y2 - tick), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, cy), (x1 + tick, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, cy), (x2 - tick, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - tick, cy), (cx + tick, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy - tick), (cx, cy + tick), color, thickness, cv2.LINE_AA)


def _underline(img, p1, p2, color, thickness):
    """A bar under the subject and nothing else. The least "surveillance" of the
    outlines — an editorial underline rather than a box."""
    import cv2

    x1, y1 = p1
    x2, y2 = p2
    bar = max(2, thickness * 2)
    cv2.line(img, (x1, y2), (x2, y2), color, bar, cv2.LINE_AA)
    lift = max(3, bar * 2)
    cv2.line(img, (x1, y2), (x1, y2 - lift), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - lift), color, thickness, cv2.LINE_AA)


def _sketch_rect(img, p1, p2, color, thickness, jitter, seed):
    """A doubled, wobbling rectangle — the hand-drawn annotation look.

    ``seed`` comes from the class name rather than the frame or the box's
    position, so the wobble is *the same wobble* on every frame: a jitter reseeded
    per frame boils, which looks like video noise rather than a drawn line.
    """
    import cv2
    import numpy as np

    if jitter <= 0:
        cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)
        return

    def wobble(index):
        # A tiny deterministic PRNG: md5 of (seed, index) read as two signed
        # offsets. Avoids seeding numpy/random globally from inside a draw call.
        digest = hashlib.md5(f"{seed}:{index}".encode()).digest()
        return (digest[0] / 255 - 0.5) * 2 * jitter, (digest[1] / 255 - 0.5) * 2 * jitter

    x1, y1 = p1
    x2, y2 = p2
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for stroke_index in (0, 1):
        points = []
        for corner_index, (cx, cy) in enumerate(corners):
            ox, oy = wobble(stroke_index * 10 + corner_index)
            points.append((cx + ox, cy + oy))
            # A midpoint per edge, so the line bows rather than staying straight.
            nx, ny = corners[(corner_index + 1) % 4]
            mx, my = wobble(stroke_index * 10 + corner_index + 100)
            points.append(((cx + nx) / 2 + mx, (cy + ny) / 2 + my))
        cv2.polylines(img, [np.array(points, dtype=np.int32)], True, color,
                      max(1, thickness - stroke_index), cv2.LINE_AA)


#: The dark pass drawn under a stroke so any box color survives any footage — a
#: lime box on a hi-vis vest disappears without it. Half-transparent, and only a
#: couple of pixels wider than the stroke, because cv2 centres its strokes: an
#: opaque halo two pixels wider than a one-pixel line *is* the line.
_HALO_ALPHA = 0.5


class MarketingRenderer:
    """Stateful per-job renderer: :meth:`draw` burns one frame's boxes on.

    Stateful because of :class:`_Tracker` — smoothing and fading only mean
    anything across a sequence — so one renderer belongs to one video pass, and a
    still (a form preview) must construct it with ``still=True``, which drops the
    temporal treatments rather than showing every box mid-fade-in.
    """

    def __init__(self, style: dict, *, still: bool = False):
        self.style = normalize(style)
        self.still = still
        if still:
            self.style = {**self.style, "smoothing": 0.0, "fade_frames": 0,
                          "animation": "none"}
        self._tracker = _Tracker(self.style)
        self._frame = 0

    # -------------------------------------------------------------- public API
    def draw(self, frame, boxes: list[dict]) -> int:
        """Burn ``boxes`` (normalized center-xywh dicts, as ``/predict_image``
        returns them) onto ``frame`` in place. Returns how many were drawn."""
        import cv2

        style = self.style
        height, width = frame.shape[:2]
        scale = min(max(height / _REFERENCE_HEIGHT, _MIN_SCALE), _MAX_SCALE)
        if not self.still:
            self._frame += 1

        tracked = self._tracker.update(self._select(boxes))
        placed = [(box, alpha, self._rect(box, width, height))
                  for box, alpha in tracked]
        placed = [p for p in placed if p[2] is not None]

        if style["dim_background"] > 0 and placed:
            self._dim(frame, [rect for _b, _a, rect in placed])

        font = getattr(cv2, _FONTS[style["font"]])
        for box, alpha, rect in placed:
            self._draw_one(frame, box, alpha, rect, scale, font)

        if style["counter"] != "none":
            self._draw_counter(frame, [b for b, _a, _r in placed], scale, font)
        if style["watermark_text"]:
            self._draw_watermark(frame, scale, font)
        return len(placed)

    # ---------------------------------------------------------------- internals
    def _select(self, boxes: list[dict]) -> list[dict]:
        """Apply the "what is worth showing" knobs: class filter, minimum size,
        and a cap on how many boxes may share the frame — then ``force_class``.

        The first three exist because a clean shot sells better than a complete
        one — a swarm of 3-pixel background detections is honest and unwatchable.

        ``force_class`` is the one knob here that changes what a box *says*
        rather than whether it is drawn: every surviving box is relabelled to
        that class, so the whole frame comes out one name and one colour
        (``box_color`` and the label both read ``class_name``) with each box's
        own geometry and confidence untouched. It is a presentation override for
        a clip whose class names are not the ones to show an audience — a model
        that says ``person_no_helmet`` in a deck about ``violation`` — and it is
        applied last, so "only these classes" still selects on what the model
        actually returned, and so does the tracker (see :func:`true_class`).
        Nothing outside the drawing reads it: the job's own detection records,
        counts and thresholds are the model's, unrelabelled.
        """
        style = self.style
        wanted = {name.lower() for name in style["class_filter"]}
        minimum = style["min_box_area"] / 100.0
        kept = []
        for box in boxes:
            try:
                geometry = _geom(box)
            except (KeyError, TypeError, ValueError):
                continue
            if wanted and (box.get("class_name") or "").lower() not in wanted:
                continue
            if minimum and geometry[2] * geometry[3] < minimum:
                continue
            kept.append(box)
        if style["max_boxes"]:
            kept.sort(key=lambda b: b.get("confidence") or 0.0, reverse=True)
            kept = kept[:style["max_boxes"]]
        if style["force_class"]:
            kept = [{**box, _TRUE_CLASS: box.get("class_name") or "",
                     "class_name": style["force_class"]} for box in kept]
        return kept

    @staticmethod
    def _rect(box, width, height):
        """Pixel corners of ``box``, clamped to the frame, or ``None`` if it lands
        outside it entirely (a smoothed track can drift off-screen)."""
        cx, cy, w, h = _geom(box)
        x1 = int(round((cx - w / 2) * width))
        y1 = int(round((cy - h / 2) * height))
        x2 = int(round((cx + w / 2) * width))
        y2 = int(round((cy + h / 2) * height))
        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            return None
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width - 1), min(y2, height - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _dim(self, frame, rects):
        """Darken everything except the detections — the "spotlight" look.

        Done as one in-place scale of the whole frame with the box interiors
        pasted back from a copy, rather than per-region masking: at 4K the copy
        is the cheaper of the two.
        """
        import cv2

        original = frame.copy()
        cv2.convertScaleAbs(frame, dst=frame, alpha=1.0 - self.style["dim_background"], beta=0)
        for x1, y1, x2, y2 in rects:
            frame[y1:y2, x1:x2] = original[y1:y2, x1:x2]

    def _draw_one(self, frame, box, fade, rect, scale, font):
        import cv2

        style = self.style
        height, width = frame.shape[:2]
        color = box_color(box, style)
        thickness = max(1, round(style["thickness"] * scale * self._pulse()))
        x1, y1, x2, y2 = rect

        label = self._label_text(box)
        layout = self._label_layout(label, rect, scale, font, thickness, width, height) \
            if label else None

        # One ROI for the box, its widest stroke pass and its label, so the alpha
        # stages below copy a box-sized patch rather than the whole frame. Derived
        # from the passes rather than guessed at: a glow fans much further out
        # than its nominal thickness, and a too-small ROI clips it square.
        widest = max(width_px for width_px, _a in
                     self._stroke_passes(thickness, 1.0, scale))
        if style["shadow"]:
            widest = max(widest, thickness + max(2, round(2 * scale)))
        pad = widest + 2
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(width, x2 + pad + 1), min(height, y2 + pad + 1)
        if layout:
            (lx1, ly1, lx2, ly2) = layout["chip"]
            rx1, ry1 = max(0, min(rx1, lx1 - pad)), max(0, min(ry1, ly1 - pad))
            rx2, ry2 = min(width, max(rx2, lx2 + pad)), min(height, max(ry2, ly2 + pad))
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return

        def local(point):
            return (point[0] - rx1, point[1] - ry1)

        p1, p2 = local((x1, y1)), local((x2, y2))
        radius = round(style["corner_radius"] * scale)

        fill_alpha = style["fill_opacity"] * fade
        if fill_alpha > 0.01 and style["box_style"] != "none":
            with _blend(roi, fill_alpha) as canvas:
                _rounded_rect(canvas, p1, p2, color, -1,
                              radius if style["box_style"] == "rounded" else 0, filled=True)

        stroke_alpha = style["box_opacity"] * fade
        if stroke_alpha > 0.01 and style["box_style"] != "none":
            outline = style["box_style"]
            phase = self._phase()

            def stroke(canvas, shade, width_px):
                width_px = max(1, width_px)
                if outline == "corners":
                    _corner_brackets(canvas, p1, p2, shade, width_px,
                                     round(style["corner_length"] * scale))
                elif outline == "crosshair":
                    _crosshair(canvas, p1, p2, shade, width_px,
                               round(style["corner_length"] * scale))
                elif outline == "dashed":
                    _dashed_rect(canvas, p1, p2, shade, width_px,
                                 round(style["dash_length"] * scale), phase)
                elif outline == "dotted":
                    _dotted_rect(canvas, p1, p2, shade, width_px,
                                 round(style["dot_spacing"] * scale), phase)
                elif outline == "double":
                    _double_rect(canvas, p1, p2, shade, width_px,
                                 round(style["double_gap"] * scale), radius)
                elif outline == "chamfer":
                    _chamfer_rect(canvas, p1, p2, shade, width_px,
                                  round(style["chamfer_size"] * scale))
                elif outline == "underline":
                    _underline(canvas, p1, p2, shade, width_px)
                elif outline == "sketch":
                    # Floored at 2px: a one-pixel wobble is a straight line with
                    # extra steps, which is not what anyone picks "sketch" for.
                    jitter = style["sketch_jitter"]
                    _sketch_rect(canvas, p1, p2, shade, width_px,
                                 max(2, round(jitter * scale)) if jitter else 0,
                                 box.get("class_name") or "")
                else:
                    _rounded_rect(canvas, p1, p2, shade, width_px,
                                  radius if outline in ("rounded", "glow") else 0)

            if style["shadow"] and outline != "glow":
                # A glow is its own halo — putting a black one under it just
                # muddies the bloom.
                with _blend(roi, stroke_alpha * _HALO_ALPHA) as canvas:
                    stroke(canvas, (0, 0, 0), thickness + max(2, round(2 * scale)))

            for width_px, alpha in self._stroke_passes(thickness, stroke_alpha, scale):
                with _blend(roi, alpha) as canvas:
                    stroke(canvas, color, width_px)

        if layout:
            self._draw_label(roi, layout, color, fade, local, font)

    def _phase(self) -> float:
        """Where the marching outlines are in their cycle, 0–1.

        Driven by the rendered-frame counter rather than a timestamp, so a clip
        re-rendered later animates identically, and a preview (which never
        advances the counter) shows the outline at rest.
        """
        if self.style["animation"] != "march":
            return 0.0
        return (self._frame % self.style["animation_period"]) / self.style["animation_period"]

    def _stroke_passes(self, thickness, alpha, scale):
        """The (width, alpha) passes that make up one outline, widest first.

        One pass for every style but ``glow``, which is three translucent
        passes fanning outwards plus a bright core — cv2 has no additive
        blending, and stacking blended passes is how you fake a bloom without
        one.
        """
        if self.style["box_style"] != "glow":
            return [(thickness, alpha)]
        spread = max(1, round(self.style["glow_spread"] * scale))
        pulse = self._pulse()
        return [
            (thickness + spread * 2, alpha * 0.12 * pulse),
            (thickness + spread, alpha * 0.22 * pulse),
            (thickness, alpha),
            (max(1, thickness // 2), alpha),
        ]

    def _pulse(self) -> float:
        """A 0.6–1.4 breath on the ``pulse`` animation, 1.0 otherwise."""
        if self.style["animation"] != "pulse":
            return 1.0
        period = self.style["animation_period"]
        return 1.0 + 0.4 * math.sin(2 * math.pi * (self._frame % period) / period)

    def _label_text(self, box) -> str:
        style = self.style
        if not style["show_label"]:
            return ""
        name = str(box.get("class_name") or "?")
        confidence = f"{float(box.get('confidence') or 0.0):.{style['confidence_decimals']}f}"
        text = {"class": name, "conf": confidence}.get(
            style["label_text"], f"{name} {confidence}")
        return text.upper() if style["label_uppercase"] else text

    def _label_layout(self, text, rect, scale, font, thickness, width, height):
        """Where the label chip and its baseline go, in absolute pixels.

        Every position falls back to *inside* the box when it would leave the
        frame, which is the bug ``draw_boxes`` had to fix too: a detection at the
        top edge with an above-the-box label draws at a negative y and silently
        vanishes.
        """
        import cv2

        style = self.style
        x1, y1, x2, y2 = rect
        font_scale = _BASE_FONT_SCALE * style["label_scale"] * scale
        text_thickness = max(1, round(1.4 * style["label_scale"] * scale))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        pad_x = max(3, round(7 * scale * style["label_scale"]))
        pad_y = max(2, round(5 * scale * style["label_scale"]))
        chip_w = text_w + pad_x * 2
        chip_h = text_h + baseline + pad_y * 2
        gap = max(2, round(4 * scale)) + thickness

        position = style["label_position"]
        if position == "below":
            top = y2 + gap
        elif position == "inside":
            top = y1 + gap
        elif position == "inside_bottom":
            top = y2 - chip_h - gap
        else:
            top = y1 - chip_h - gap
        if top < 0 or top + chip_h > height:
            top = min(max(y1 + gap, 0), max(0, height - chip_h))

        left = min(max(x1, 0), max(0, width - chip_w))
        chip = (left, top, left + chip_w, top + chip_h)
        origin = (left + pad_x, top + pad_y + text_h)
        return {"chip": chip, "origin": origin, "text": text,
                "font_scale": font_scale, "thickness": text_thickness,
                "radius": chip_h // 2}

    def _draw_label(self, roi, layout, color, fade, local, font):
        import cv2

        style = self.style
        chip = layout["chip"]
        p1, p2 = local(chip[:2]), local(chip[2:])
        styled = style["label_style"]

        background = color
        if styled in ("pill", "bar"):
            alpha = style["label_opacity"] * fade
            if alpha > 0.01:
                with _blend(roi, alpha) as canvas:
                    _rounded_rect(canvas, p1, p2, color, -1,
                                  layout["radius"] if styled == "pill" else 0, filled=True)
        else:
            # No chip behind the text, so "auto" has the video to contrast with,
            # not a known color — the halo below is what keeps it readable.
            background = (0, 0, 0)

        text_color = _text_color(background, style)
        with _blend(roi, max(0.0, min(1.0, fade))) as canvas:
            origin = local(layout["origin"])
            if styled in ("plain", "outline") and style["shadow"]:
                cv2.putText(canvas, layout["text"], origin, font, layout["font_scale"],
                            (0, 0, 0), layout["thickness"] + 2, cv2.LINE_AA)
            cv2.putText(canvas, layout["text"], origin, font, layout["font_scale"],
                        color if styled == "outline" else text_color,
                        layout["thickness"], cv2.LINE_AA)

    def _draw_counter(self, frame, boxes, scale, font):
        """A stack of "class × N" chips in one corner — the overlay that makes a
        clip read as a *product* rather than a debug dump."""
        import cv2

        style = self.style
        counts: dict[str, int] = {}
        for box in boxes:
            name = str(box.get("class_name") or "?")
            counts[name] = counts.get(name, 0) + 1
        if not counts:
            return

        font_scale = _BASE_FONT_SCALE * style["counter_scale"] * scale
        thickness = max(1, round(1.4 * style["counter_scale"] * scale))
        pad = max(4, round(8 * scale * style["counter_scale"]))
        margin = max(6, round(18 * scale))
        rows = [(f"{name} × {count}", _class_color(name, style))
                for name, count in sorted(counts.items())]

        sizes = [cv2.getTextSize(text, font, font_scale, thickness)
                 for text, _c in rows]
        chip_w = max(w for (w, _h), _b in sizes) + pad * 3 + round(14 * scale)
        row_h = max(h + b for (_w, h), b in sizes) + pad * 2
        block_h = row_h * len(rows)

        height, width = frame.shape[:2]
        left = margin if style["counter"].endswith("left") else width - chip_w - margin
        top = margin if style["counter"].startswith("top") else height - block_h - margin
        left, top = max(0, left), max(0, top)

        for index, ((text, color), ((text_w, text_h), baseline)) in enumerate(zip(rows, sizes)):
            y = top + index * row_h
            roi = frame[y:min(y + row_h - 2, height), left:min(left + chip_w, width)]
            if roi.size == 0:
                continue
            with _blend(roi, 0.55) as canvas:
                _rounded_rect(canvas, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1),
                              (20, 20, 20), -1, round(6 * scale), filled=True)
            dot = round(5 * scale * style["counter_scale"])
            cv2.circle(roi, (pad + dot, roi.shape[0] // 2), dot, color, -1, cv2.LINE_AA)
            cv2.putText(roi, text, (pad * 2 + dot * 2, pad + text_h), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_watermark(self, frame, scale, font):
        import cv2

        style = self.style
        text = style["watermark_text"]
        font_scale = _BASE_FONT_SCALE * 1.2 * style["watermark_scale"] * scale
        thickness = max(1, round(1.6 * style["watermark_scale"] * scale))
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        margin = max(8, round(24 * scale))
        height, width = frame.shape[:2]

        x = margin if style["watermark_position"].endswith("left") else width - text_w - margin
        y = (margin + text_h if style["watermark_position"].startswith("top")
             else height - margin - baseline)
        x, y = max(0, x), max(text_h, min(y, height - 1))

        color = hex_to_bgr(style["watermark_color"])
        with _blend(frame, style["watermark_opacity"]) as canvas:
            cv2.putText(canvas, text, (x, y), font, font_scale, (0, 0, 0),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (x, y), font, font_scale, color,
                        thickness, cv2.LINE_AA)


def encode_options(style: dict) -> dict:
    """The ffmpeg-side half of a style: output height (0 = source) and x264 CRF.

    Not a drawing knob, but the same operator decision — a marketing clip is
    usually a 1080p, visually-lossless cut of a 4K source, and asking for that
    anywhere other than beside the look would be strange.
    """
    style = normalize(style)
    return {"crf": style["crf"], "output_height": style["output_height"]}
