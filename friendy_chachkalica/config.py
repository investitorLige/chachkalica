from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    images: Path
    labels: Path
    classes: Dict[int, str]
    role: str
    weight: float = 1.0
    # Per-augmentation fraction of samples to augment each epoch, e.g.
    # {"hflip": 0.5, "scale_crop": 0.3}. Only honored for role="train".
    augmentation: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int | str
    params: Dict[str, Any] = field(default_factory=dict)
    init_checkpoint: Optional[Path] = None


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 0.0001
    weight_decay: float = 0.0001
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulerConfig:
    name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    device: str = "auto"
    seed: Optional[int] = None
    amp: bool = False
    gradient_clip_norm: Optional[float] = None
    early_stopping_patience: Optional[int] = None
    # Val metric(s) that select best.pt (see BEST_METRIC_CHOICES). More than
    # one entry means their average is tracked, e.g. ("f1", "map50").
    best_metric: tuple = ("map50",)
    # Run validation every N epochs (the final epoch always validates).
    val_interval: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    score_threshold: float = 0.001
    map_score_threshold: Optional[float] = None
    # Legacy global NMS applied to ALL predictions before ANY metric — this
    # prunes the low-confidence tail mAP integrates over, so prefer
    # operating_nms_threshold below and leave this null.
    nms_threshold: Optional[float] = None
    # Class-aware NMS IoU applied ONLY to the operating-point metrics
    # (precision/recall/F1, per-class, confusion matrix). mAP stays NMS-free.
    # A model entry's own nms_threshold param overrides this per run.
    operating_nms_threshold: Optional[float] = None
    iou_thresholds: List[float] = field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )


@dataclass(frozen=True)
class TilingSpec:
    # Blank knobs are None so chachak's own defaults apply downstream.
    tile_width_pct: Optional[float] = None
    tile_height_pct: Optional[float] = None
    overlap: Optional[float] = None


@dataclass(frozen=True)
class PipelineSpec:
    """A chachak pipeline attached to the experiment (train/val/test).

    Present only when the experiment YAML has a top-level ``pipeline`` block.
    ``name`` is one of chachak's pipeline names (batch_detect, ...). The knobs
    mirror the chachak request; anything left None falls back to chachak's
    defaults when the pipeline is built.
    """

    name: str
    detector_checkpoint: Optional[Path] = None
    tiling: TilingSpec = field(default_factory=TilingSpec)
    chain: List[str] = field(default_factory=list)
    merge_nms_iou: Optional[float] = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    train_datasets: List[DatasetConfig]
    models: List[ModelConfig]
    output_dir: Path
    val_dataset: Optional[DatasetConfig] = None
    test_dataset: Optional[DatasetConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    pipeline: Optional[PipelineSpec] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentRun:
    index: int
    train_dataset: DatasetConfig
    model: ModelConfig
    model_index: int
    val_dataset: Optional[DatasetConfig] = None
    test_dataset: Optional[DatasetConfig] = None
    pipeline: Optional["PipelineSpec"] = None

    @property
    def name(self) -> str:
        parts = [
            f"{self.index:02d}",
            _slug(self.train_dataset.name),
            f"{self.model_index:02d}",
            self.model.name,
        ]
        variant = self.model.params.get("variant")
        if variant:
            parts.append(str(variant).replace("/", "-"))
        return "-".join(parts)


def load_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).resolve()
    print(f"[config] Loading experiment config: {config_path}")
    with open(config_path) as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a YAML mapping")

    base_dir = config_path.parent
    name = _require_str(raw, "name")
    output_dir = _resolve_path(raw.get("output_dir", f"runs/{name}"), base_dir)

    datasets_raw = _require_mapping(raw, "datasets")
    train_datasets = _parse_dataset_list(datasets_raw.get("train"), base_dir, "datasets.train", role="train")
    val_dataset = _parse_optional_dataset(datasets_raw.get("val"), base_dir, "datasets.val", role="val")
    test_dataset = _parse_optional_dataset(datasets_raw.get("test"), base_dir, "datasets.test", role="test")

    models = _parse_models(_require_list(raw, "models"), base_dir)
    training = _parse_training(raw.get("training", {}))
    evaluation = _parse_evaluation(raw.get("evaluation", {}))
    pipeline = _parse_pipeline(raw.get("pipeline"), base_dir)

    config = ExperimentConfig(
        name=name,
        train_datasets=train_datasets,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        models=models,
        output_dir=output_dir,
        training=training,
        evaluation=evaluation,
        pipeline=pipeline,
        raw=raw,
    )
    print(
        "[config] Loaded "
        f"name={config.name} train_datasets={len(config.train_datasets)} "
        f"models={len(config.models)} val={config.val_dataset is not None} "
        f"test={config.test_dataset is not None} "
        f"pipeline={config.pipeline.name if config.pipeline else None} "
        f"output_dir={config.output_dir}"
    )
    return config


def build_experiment_runs(config: ExperimentConfig) -> List[ExperimentRun]:
    runs = []
    for train_dataset_config in config.train_datasets:
        for model_index, model_config in enumerate(config.models):
            runs.append(
                ExperimentRun(
                    index=len(runs),
                    train_dataset=train_dataset_config,
                    model=_resolve_model_for_dataset(model_config, train_dataset_config),
                    model_index=model_index,
                    val_dataset=config.val_dataset,
                    test_dataset=config.test_dataset,
                    pipeline=config.pipeline,
                )
            )
    print(f"[config] Expanded experiment matrix to {len(runs)} run(s)")
    for run in runs:
        print(
            f"[config] Run {run.index}: train={run.train_dataset.name} "
            f"model={run.model.name} num_classes={run.model.num_classes}"
        )
    return runs


def _resolve_model_for_dataset(model_config: ModelConfig, dataset_config: DatasetConfig) -> ModelConfig:
    if model_config.num_classes != "auto":
        return model_config
    return ModelConfig(
        name=model_config.name,
        num_classes=len(dataset_config.classes),
        params=dict(model_config.params),
        init_checkpoint=model_config.init_checkpoint,
    )


def _parse_dataset_list(value: Any, base_dir: Path, field_name: str, role: str) -> List[DatasetConfig]:
    if value is None:
        raise ValueError(f"Missing required field: {field_name}")

    entries = value if isinstance(value, list) else [value]
    if not entries:
        raise ValueError(f"{field_name} must contain at least one dataset")

    return [
        _parse_dataset(entry, base_dir, f"{field_name}[{index}]", role=role)
        for index, entry in enumerate(entries)
    ]


def _parse_optional_dataset(value: Any, base_dir: Path, field_name: str, role: str) -> Optional[DatasetConfig]:
    if value is None:
        return None
    return _parse_dataset(value, base_dir, field_name, role=role)


def _parse_dataset(value: Any, base_dir: Path, field_name: str, role: str) -> DatasetConfig:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")

    images_path = _resolve_path(_require_value(value, "images", field_name), base_dir)
    labels_path = _resolve_path(_require_value(value, "labels", field_name), base_dir)
    classes = _parse_classes(_require_value(value, "classes", field_name), field_name)
    if not classes:
        raise ValueError(f"{field_name}.classes must contain at least one class")

    weight = float(value.get("weight", 1.0))
    if weight <= 0:
        raise ValueError(f"{field_name}.weight must be greater than 0")

    augmentation = _parse_augmentation(value.get("augmentation"), field_name, role)

    name = str(value.get("name") or images_path.stem)
    print(
        f"[config] Dataset {field_name}: name={name} role={role} "
        f"images={images_path} labels={labels_path} classes={len(classes)} "
        f"augmentation={augmentation or '{}'}"
    )
    return DatasetConfig(
        name=name,
        images=images_path,
        labels=labels_path,
        classes=classes,
        role=role,
        weight=weight,
        augmentation=augmentation,
    )


AUGMENTATION_KEYS = ("hflip", "scale_crop")

# Keys of the per-epoch val mAP summary that training.best_metric may name.
# All are higher-is-better; precision/recall/f1 are computed at the operating
# score threshold, not at each class's best confidence.
BEST_METRIC_CHOICES = ("map50", "map50_95", "precision", "recall", "f1")


def _parse_augmentation(value: Any, field_name: str, role: str) -> Dict[str, float]:
    """Validate a dataset's optional ``augmentation`` mapping.

    Each entry is ``<name>: <fraction>`` — the fraction of samples the
    augmentation is applied to each epoch. Only train datasets are augmented,
    so declaring it elsewhere is rejected rather than silently ignored.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name}.augmentation must be a mapping")
    if role != "train":
        raise ValueError(
            f"{field_name}.augmentation is only supported for train datasets (role={role})"
        )

    augmentation: Dict[str, float] = {}
    for key, fraction in value.items():
        if key not in AUGMENTATION_KEYS:
            raise ValueError(
                f"{field_name}.augmentation.{key} is not a known augmentation "
                f"(supported: {', '.join(AUGMENTATION_KEYS)})"
            )
        try:
            fraction = float(fraction)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}.augmentation.{key} must be a number") from exc
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{field_name}.augmentation.{key} must be between 0 and 1")
        if fraction > 0:
            augmentation[key] = fraction
    return augmentation


def _parse_classes(value: Any, field_name: str) -> Dict[int, str]:
    if isinstance(value, list):
        return {index: str(class_name) for index, class_name in enumerate(value)}
    if isinstance(value, dict):
        return {int(class_id): str(class_name) for class_id, class_name in value.items()}
    raise ValueError(f"{field_name}.classes must be a list or mapping")


def _parse_num_classes(value: Any, field_name: str) -> int | str:
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"

    try:
        num_classes = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}.num_classes must be a positive integer or 'auto'") from exc

    if num_classes <= 0:
        raise ValueError(f"{field_name}.num_classes must be greater than 0")
    return num_classes


def _parse_models(value: List[Any], base_dir: Path) -> List[ModelConfig]:
    if not value:
        raise ValueError("models must contain at least one model")

    models = []
    for index, entry in enumerate(value):
        field_name = f"models[{index}]"
        if isinstance(entry, str):
            raise ValueError(f"{field_name} must include at least name and num_classes")
        if not isinstance(entry, dict):
            raise ValueError(f"{field_name} must be a mapping")

        params = dict(entry)
        name = str(params.pop("name", ""))
        if not name:
            raise ValueError(f"Missing required field: {field_name}.name")

        if "num_classes" not in params:
            raise ValueError(f"Missing required field: {field_name}.num_classes")
        num_classes = _parse_num_classes(params.pop("num_classes"), field_name)

        model_params = params.pop("params", {})
        if model_params is None:
            model_params = {}
        if not isinstance(model_params, dict):
            raise ValueError(f"{field_name}.params must be a mapping")
        model_params = {**params, **model_params}

        init_checkpoint = model_params.pop("init_checkpoint", None)
        if init_checkpoint is not None:
            init_checkpoint = _resolve_path(init_checkpoint, base_dir)

        models.append(
            ModelConfig(
                name=name,
                num_classes=num_classes,
                params=model_params,
                init_checkpoint=init_checkpoint,
            )
        )

    return models


def _parse_training(value: Any) -> TrainingConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("training must be a mapping")

    optimizer = _parse_optimizer(value.get("optimizer", {}))
    scheduler = _parse_scheduler(value.get("scheduler", {}))

    epochs = int(value.get("epochs", 100))
    batch_size = int(value.get("batch_size", 4))
    num_workers = int(value.get("num_workers", 4))
    if epochs <= 0:
        raise ValueError("training.epochs must be greater than 0")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be greater than 0")
    if num_workers < 0:
        raise ValueError("training.num_workers cannot be negative")

    gradient_clip_norm = value.get("gradient_clip_norm")
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
        if gradient_clip_norm <= 0:
            raise ValueError("training.gradient_clip_norm must be greater than 0")

    early_stopping_patience = value.get("early_stopping_patience")
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        if early_stopping_patience <= 0:
            raise ValueError("training.early_stopping_patience must be greater than 0")

    seed = value.get("seed")
    if seed is not None:
        seed = int(seed)

    best_metric = _parse_best_metric(value.get("best_metric"))

    val_interval = int(value.get("val_interval", 1))
    if val_interval <= 0:
        raise ValueError("training.val_interval must be greater than 0")

    return TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        device=str(value.get("device", "auto")),
        seed=seed,
        amp=bool(value.get("amp", False)),
        gradient_clip_norm=gradient_clip_norm,
        early_stopping_patience=early_stopping_patience,
        best_metric=best_metric,
        val_interval=val_interval,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _parse_best_metric(value: Any) -> tuple:
    """training.best_metric: one metric name or a list whose average is tracked."""
    if value is None:
        return ("map50",)

    entries = value if isinstance(value, list) else [value]
    if not entries:
        raise ValueError("training.best_metric must name at least one metric")

    metrics: List[str] = []
    for entry in entries:
        metric = str(entry).lower()
        if metric not in BEST_METRIC_CHOICES:
            raise ValueError(
                f"training.best_metric {entry!r} is not a known val metric "
                f"(supported: {', '.join(BEST_METRIC_CHOICES)})"
            )
        if metric not in metrics:
            metrics.append(metric)
    return tuple(metrics)


def _parse_optimizer(value: Any) -> OptimizerConfig:
    if value is None:
        value = {}
    if isinstance(value, str):
        return OptimizerConfig(name=value)
    if not isinstance(value, dict):
        raise ValueError("training.optimizer must be a string or mapping")

    params = dict(value)
    name = str(params.pop("name", "adamw"))
    lr = float(params.pop("lr", 0.0001))
    weight_decay = float(params.pop("weight_decay", 0.0001))
    if lr <= 0:
        raise ValueError("training.optimizer.lr must be greater than 0")
    if weight_decay < 0:
        raise ValueError("training.optimizer.weight_decay cannot be negative")

    extra_params = params.pop("params", {})
    if extra_params is None:
        extra_params = {}
    if not isinstance(extra_params, dict):
        raise ValueError("training.optimizer.params must be a mapping")

    return OptimizerConfig(
        name=name,
        lr=lr,
        weight_decay=weight_decay,
        params={**params, **extra_params},
    )


def _parse_scheduler(value: Any) -> SchedulerConfig:
    if value is None or value is False:
        return SchedulerConfig()
    if isinstance(value, str):
        return SchedulerConfig(name=value)
    if not isinstance(value, dict):
        raise ValueError("training.scheduler must be a string, mapping, false, or null")

    params = dict(value)
    name = params.pop("name", None)
    extra_params = params.pop("params", {})
    if extra_params is None:
        extra_params = {}
    if not isinstance(extra_params, dict):
        raise ValueError("training.scheduler.params must be a mapping")

    return SchedulerConfig(
        name=None if name is None else str(name),
        params={**params, **extra_params},
    )


def _parse_evaluation(value: Any) -> EvaluationConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("evaluation must be a mapping")

    batch_size = value.get("batch_size")
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("evaluation.batch_size must be greater than 0")

    num_workers = value.get("num_workers")
    if num_workers is not None:
        num_workers = int(num_workers)
        if num_workers < 0:
            raise ValueError("evaluation.num_workers cannot be negative")

    score_threshold = float(value.get("score_threshold", 0.001))
    if score_threshold < 0 or score_threshold > 1:
        raise ValueError("evaluation.score_threshold must be between 0 and 1")

    raw_map_score_threshold = value.get("map_score_threshold")
    map_score_threshold = None if raw_map_score_threshold is None else float(raw_map_score_threshold)
    if map_score_threshold is not None and (map_score_threshold < 0 or map_score_threshold > 1):
        raise ValueError("evaluation.map_score_threshold must be between 0 and 1")

    raw_nms_threshold = value.get("nms_threshold")
    nms_threshold = None if raw_nms_threshold is None else float(raw_nms_threshold)
    if nms_threshold is not None and (nms_threshold <= 0 or nms_threshold > 1):
        raise ValueError("evaluation.nms_threshold must be in (0, 1] or null")

    raw_operating_nms = value.get("operating_nms_threshold")
    operating_nms_threshold = None if raw_operating_nms is None else float(raw_operating_nms)
    if operating_nms_threshold is not None and (
        operating_nms_threshold <= 0 or operating_nms_threshold > 1
    ):
        raise ValueError("evaluation.operating_nms_threshold must be in (0, 1] or null")

    iou_thresholds = value.get(
        "iou_thresholds",
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    if not isinstance(iou_thresholds, list) or not iou_thresholds:
        raise ValueError("evaluation.iou_thresholds must be a non-empty list")
    iou_thresholds = [float(threshold) for threshold in iou_thresholds]
    for threshold in iou_thresholds:
        if threshold <= 0 or threshold > 1:
            raise ValueError("evaluation.iou_thresholds values must be in (0, 1]")

    return EvaluationConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        score_threshold=score_threshold,
        map_score_threshold=map_score_threshold,
        nms_threshold=nms_threshold,
        operating_nms_threshold=operating_nms_threshold,
        iou_thresholds=iou_thresholds,
    )


# Mirror of chachak.config.PIPELINE_NAMES; kept local so parsing the config does
# not import chachak (which imports friendy back).
PIPELINE_NAMES = ("batch_detect", "people_detect_first", "batch_people", "chain")


def _parse_pipeline(value: Any, base_dir: Path) -> Optional[PipelineSpec]:
    """Parse the optional top-level ``pipeline`` block.

    Returns ``None`` when absent (plain full-frame training/eval). Only the knobs
    present in the YAML are set; the rest stay ``None`` so chachak's defaults
    apply when the pipeline is built.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("pipeline must be a mapping")

    name = _require_str(value, "name")
    if name not in PIPELINE_NAMES:
        raise ValueError(f"pipeline.name must be one of {PIPELINE_NAMES}")

    detector_raw = value.get("detector") or {}
    if not isinstance(detector_raw, dict):
        raise ValueError("pipeline.detector must be a mapping")
    detector_checkpoint = detector_raw.get("checkpoint")
    if detector_checkpoint is not None:
        detector_checkpoint = _resolve_path(detector_checkpoint, base_dir)

    tiling_raw = value.get("tiling") or {}
    if not isinstance(tiling_raw, dict):
        raise ValueError("pipeline.tiling must be a mapping")

    def _opt_float(raw: Any, field_name: str) -> Optional[float]:
        if raw is None:
            return None
        return float(raw)

    tiling = TilingSpec(
        tile_width_pct=_opt_float(tiling_raw.get("tile_width_pct"), "tile_width_pct"),
        tile_height_pct=_opt_float(tiling_raw.get("tile_height_pct"), "tile_height_pct"),
        overlap=_opt_float(tiling_raw.get("overlap"), "overlap"),
    )

    chain = value.get("chain") or []
    if not isinstance(chain, list):
        raise ValueError("pipeline.chain must be a list")
    chain = [str(entry) for entry in chain]

    merge_nms_iou = value.get("merge_nms_iou")
    if merge_nms_iou is not None:
        merge_nms_iou = float(merge_nms_iou)

    return PipelineSpec(
        name=name,
        detector_checkpoint=detector_checkpoint,
        tiling=tiling,
        chain=chain,
        merge_nms_iou=merge_nms_iou,
    )


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _require_mapping(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = _require_value(raw, key, key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_list(raw: Dict[str, Any], key: str) -> List[Any]:
    value = _require_value(raw, key, key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _require_str(raw: Dict[str, Any], key: str) -> str:
    value = _require_value(raw, key, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_value(raw: Dict[str, Any], key: str, field_name: str) -> Any:
    if key not in raw or raw[key] is None:
        raise ValueError(f"Missing required field: {field_name}.{key}")
    return raw[key]


def _slug(value: str) -> str:
    return str(value).strip().replace("/", "-").replace(" ", "-")
