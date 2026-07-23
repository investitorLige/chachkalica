"""Pipeline registry and builder, mirroring ``friendy_chachkalica/registry.py``."""

try:
    from .pipeline import (
        BatchDetectPipeline,
        BatchPeoplePipeline,
        ChainedPipeline,
        PeopleDetectFirstPipeline,
    )
except ImportError:  # run as a flat script
    from pipeline import (
        BatchDetectPipeline,
        BatchPeoplePipeline,
        ChainedPipeline,
        PeopleDetectFirstPipeline,
    )


PIPELINE_REGISTRY = {
    "batch_detect": BatchDetectPipeline,
    "people_detect_first": PeopleDetectFirstPipeline,
    "batch_people": BatchPeoplePipeline,
    "chain": ChainedPipeline,
}


def _build_named(name, config, model_adapter, device, detector, box_cache):
    try:
        cls = PIPELINE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(PIPELINE_REGISTRY))
        raise ValueError(
            f"Unknown pipeline '{name}'. Available pipelines: {available}"
        ) from exc
    return cls(model_adapter, device, config, detector=detector, box_cache=box_cache)


def build_pipeline(config, model_adapter, device, detector=None, box_cache=None):
    """Build the pipeline described by ``config`` (recursively for 'chain').

    ``box_cache``, when given, is a plain dict shared across every pipeline
    instance built from it: a per-image-id cache of detected person boxes
    (see ``Pipeline._cached_person_boxes``), letting repeated calls against the
    same frozen detector skip re-running it. None (the default) disables
    caching, preserving the original behavior exactly.
    """
    if config.pipeline == "chain":
        children = [
            _build_named(child, config, model_adapter, device, detector, box_cache)
            for child in config.chain
        ]
        return ChainedPipeline(
            model_adapter, device, config, detector=detector, pipelines=children, box_cache=box_cache
        )
    return _build_named(config.pipeline, config, model_adapter, device, detector, box_cache)
