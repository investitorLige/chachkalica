try:
    from .ml.adapters.dfine import build_dfine
    from .ml.adapters.ecdet import build_ecdet
    from .ml.adapters.fasterrcnn import build_fasterrcnn
    from .ml.adapters.retinanet import build_retinanet
    from .ml.adapters.rfdetr import build_rfdetr
    from .ml.adapters.rtdetr import build_rtdetr
    from .ml.adapters.yolox import build_yolox
except ImportError:
    from ml.adapters.dfine import build_dfine
    from ml.adapters.ecdet import build_ecdet
    from ml.adapters.fasterrcnn import build_fasterrcnn
    from ml.adapters.retinanet import build_retinanet
    from ml.adapters.rfdetr import build_rfdetr
    from ml.adapters.rtdetr import build_rtdetr
    from ml.adapters.yolox import build_yolox


MODEL_REGISTRY = {
    "dfine": build_dfine,
    "ecdet": build_ecdet,
    "fasterrcnn": build_fasterrcnn,
    "retinanet": build_retinanet,
    "rfdetr": build_rfdetr,
    "rtdetr": build_rtdetr,
    "yolox": build_yolox,
}


def build_model(name, **kwargs):
    """Build a registered detector adapter.

    Args:
        name: Registered model name, for example "retinanet".
        **kwargs: Model-specific builder options such as num_classes, weights,
            score_threshold, weights_backbone, trainable_backbone_layers, and
            variant.

    Weight examples:
        build_model("fasterrcnn", num_classes=3)
        build_model("fasterrcnn", num_classes=3, variant="mobilenet_v3_large_fpn")
        build_model("fasterrcnn", num_classes=91, weights="DEFAULT")
        build_model("retinanet", num_classes=3)
        build_model("retinanet", num_classes=3, weights_backbone="DEFAULT")
        build_model("retinanet", num_classes=91, weights="DEFAULT")
        build_model("rtdetr", num_classes=3)
        build_model("rtdetr", num_classes=3, weights="PekingU/rtdetr_r50vd")
        build_model("rfdetr", num_classes=3)
        build_model("rfdetr", num_classes=3, variant="base")
        build_model("yolox", num_classes=3)
        build_model("yolox", num_classes=3, variant="yolox-s")
        build_model("ecdet", num_classes=3)
        build_model("ecdet", num_classes=3, variant="ecdet-s")
        build_model("ecdet", num_classes=3, weights="backbone")
        build_model("dfine", num_classes=3)
        build_model("dfine", num_classes=3, weights="ustc-community/dfine-small-coco")
    """
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return builder(**kwargs)
