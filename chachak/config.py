"""Config for chachak pipelines: frozen dataclasses + a YAML loader.

Mirrors ``friendy_chachkalica/config.py`` in spirit (frozen dataclasses, a
``load_*`` that reads YAML, validates, and resolves paths relative to the config
file). A request describes one pipeline run against one dataset.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

PIPELINE_NAMES = ("batch_detect", "people_detect_first", "batch_people", "chain")
_DEFAULT_IOU = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


@dataclass(frozen=True)
class TilingConfig:
    tile_size_px: Optional[int] = None
    tile_width_pct: float = 50.0
    tile_height_pct: float = 50.0
    overlap: float = 0.2
    nms_iou: float = 0.5


@dataclass(frozen=True)
class DetectorConfig:
    checkpoint: Optional[Path] = None
    score_threshold: float = 0.5
    person_class_name: Optional[str] = "person"
    person_class_id: Optional[int] = None
    expand_ratio: float = 0.0
    nms_iou: float = 0.5
    min_box_size: float = 0.0


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    pipeline: str
    model_checkpoint: Path
    images: Path
    labels: Optional[Path]
    classes: Dict[int, str]
    output_dir: Path
    device: str = "auto"
    infer_batch_size: int = 4
    num_workers: int = 4
    score_threshold: float = 0.001
    map_score_threshold: Optional[float] = None
    iou_thresholds: List[float] = field(default_factory=lambda: list(_DEFAULT_IOU))
    merge_nms_iou: float = 0.5
    tiling: TilingConfig = field(default_factory=TilingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    chain: List[str] = field(default_factory=list)
    # Extra model checkpoints combined with `model_checkpoint`, run through the
    # same pipeline; each model's predictions are remapped onto `classes` by
    # name and merged per image before scoring (see run.py:run_pipeline). Empty
    # for an ordinary single-model pipeline eval.
    extra_checkpoints: List[Path] = field(default_factory=list)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _parse_classes(value: Any) -> Dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        return {int(cid): str(name) for cid, name in value.items()}
    raise ValueError("classes must be a list or mapping")


def _require(raw: Dict[str, Any], key: str) -> Any:
    if key not in raw or raw[key] is None:
        raise ValueError(f"Missing required field: {key}")
    return raw[key]


def _parse_tiling(raw: Any) -> TilingConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("tiling must be a mapping")
    defaults = TilingConfig()
    tile_size_px = raw.get("tile_size_px", defaults.tile_size_px)
    if tile_size_px is not None:
        tile_size_px = int(tile_size_px)
        if tile_size_px <= 0:
            raise ValueError("tiling.tile_size_px must be greater than 0")
    overlap = float(raw.get("overlap", defaults.overlap))
    if not 0.0 <= overlap < 1.0:
        raise ValueError("tiling.overlap must be in [0, 1)")
    tile_width_pct = float(raw.get("tile_width_pct", defaults.tile_width_pct))
    tile_height_pct = float(raw.get("tile_height_pct", defaults.tile_height_pct))
    for label, value in (
        ("tile_width_pct", tile_width_pct),
        ("tile_height_pct", tile_height_pct),
    ):
        if not 0.0 < value <= 100.0:
            raise ValueError(f"tiling.{label} must be in (0, 100]")
    return TilingConfig(
        tile_size_px=tile_size_px,
        tile_width_pct=tile_width_pct,
        tile_height_pct=tile_height_pct,
        overlap=overlap,
        nms_iou=float(raw.get("nms_iou", defaults.nms_iou)),
    )


def _parse_detector(raw: Any, base_dir: Path) -> DetectorConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("detector must be a mapping")
    defaults = DetectorConfig()
    checkpoint = raw.get("checkpoint")
    person_class_id = raw.get("person_class_id")
    return DetectorConfig(
        checkpoint=_resolve_path(checkpoint, base_dir) if checkpoint else None,
        score_threshold=float(raw.get("score_threshold", defaults.score_threshold)),
        person_class_name=raw.get("person_class_name", defaults.person_class_name),
        person_class_id=None if person_class_id is None else int(person_class_id),
        expand_ratio=float(raw.get("expand_ratio", defaults.expand_ratio)),
        nms_iou=float(raw.get("nms_iou", defaults.nms_iou)),
        min_box_size=float(raw.get("min_box_size", defaults.min_box_size)),
    )


def load_pipeline_config(config_path: Union[str, Path]) -> PipelineConfig:
    config_path = Path(config_path).resolve()
    print(f"[chachak] Loading pipeline config: {config_path}")
    with open(config_path) as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError("Pipeline config must be a YAML mapping")
    return pipeline_config_from_dict(raw, config_path.parent)


def pipeline_config_from_dict(raw: Dict[str, Any], base_dir: Path) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from an already-parsed mapping.

    Shared by :func:`load_pipeline_config` (which reads YAML first) and callers
    that already hold a request dict (e.g. the trainer service's synchronous
    predict endpoint). ``base_dir`` anchors relative paths.
    """
    if not isinstance(raw, dict):
        raise ValueError("Pipeline config must be a mapping")

    pipeline = str(_require(raw, "pipeline"))
    if pipeline not in PIPELINE_NAMES:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'. Available: {', '.join(PIPELINE_NAMES)}"
        )

    name = str(raw.get("name", pipeline))
    classes = _parse_classes(_require(raw, "classes"))
    if not classes:
        raise ValueError("classes must contain at least one class")

    iou_thresholds = raw.get("iou_thresholds", list(_DEFAULT_IOU))
    if not isinstance(iou_thresholds, list) or not iou_thresholds:
        raise ValueError("iou_thresholds must be a non-empty list")
    iou_thresholds = [float(value) for value in iou_thresholds]

    chain = raw.get("chain", []) or []
    if pipeline == "chain":
        if not isinstance(chain, list) or not chain:
            raise ValueError("chain pipeline requires a non-empty 'chain' list")
        for child in chain:
            if child not in PIPELINE_NAMES or child == "chain":
                raise ValueError(f"Invalid chain member: {child}")

    detector = _parse_detector(raw.get("detector"), base_dir)
    needs_detector = pipeline in {"people_detect_first", "batch_people"} or (
        pipeline == "chain"
        and any(child in {"people_detect_first", "batch_people"} for child in chain)
    )
    if needs_detector and detector.checkpoint is None:
        raise ValueError(f"pipeline '{pipeline}' requires detector.checkpoint")

    extra_checkpoints = [
        _resolve_path(checkpoint, base_dir) for checkpoint in (raw.get("extra_checkpoints") or [])
    ]

    config = PipelineConfig(
        name=name,
        pipeline=pipeline,
        model_checkpoint=_resolve_path(_require(raw, "model_checkpoint"), base_dir),
        images=_resolve_path(_require(raw, "images"), base_dir),
        labels=(_resolve_path(raw["labels"], base_dir) if raw.get("labels") else None),
        classes=classes,
        output_dir=_resolve_path(raw.get("output_dir", f"runs/{name}"), base_dir),
        device=str(raw.get("device", "auto")),
        infer_batch_size=int(raw.get("infer_batch_size", 4)),
        num_workers=int(raw.get("num_workers", 4)),
        score_threshold=float(raw.get("score_threshold", 0.001)),
        map_score_threshold=(
            None if raw.get("map_score_threshold") is None
            else float(raw.get("map_score_threshold"))
        ),
        iou_thresholds=iou_thresholds,
        merge_nms_iou=float(raw.get("merge_nms_iou", 0.5)),
        tiling=_parse_tiling(raw.get("tiling")),
        detector=detector,
        chain=list(chain),
        extra_checkpoints=extra_checkpoints,
    )
    print(
        f"[chachak] Config: pipeline={config.pipeline} name={config.name} "
        f"classes={len(config.classes)} output_dir={config.output_dir}"
    )
    return config
