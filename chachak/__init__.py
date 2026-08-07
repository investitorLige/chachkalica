"""chachak — stackable inference/eval pipelines over the trained detector.

Public API:
    load_pipeline_config, build_pipeline, run_pipeline, and the pipeline classes.

Attribute access is lazy (PEP 562). Importing them eagerly here would mean that
``import chachak.bundle_export.cli`` — pure file work that needs nothing but a
config parser — pulled in ``pipeline``/``run`` and therefore torch, which is what
the slim build node (``buildnode/``) exists to avoid. The names below still
resolve exactly as before on first access; only the timing changed.
"""

import importlib

# name -> submodule it lives in. Relative form; the flat-script fallback below
# strips the leading dot.
_LAZY = {
    "PipelineConfig": ".config",
    "load_pipeline_config": ".config",
    "PIPELINE_REGISTRY": ".registry",
    "build_pipeline": ".registry",
    "Pipeline": ".pipeline",
    "BatchDetectPipeline": ".pipeline",
    "BatchPeoplePipeline": ".pipeline",
    "ChainedPipeline": ".pipeline",
    "PeopleDetectFirstPipeline": ".pipeline",
    "run_pipeline": ".run",
}


def _import(module: str):
    """Import ``module`` relative to this package, falling back to flat.

    A flat import is how these modules are loaded when ``chachak/`` itself is on
    ``sys.path`` (running ``run.py`` as a script). If neither works the ORIGINAL
    error is raised rather than the flat one, so a genuinely missing dependency
    reports itself instead of "No module named 'pipeline'". Note the submodules
    carry the same try/except pattern internally, so an error raised from deeper
    than this call can still arrive pre-mangled — that predates this function.
    """
    try:
        return importlib.import_module(module, __name__)
    except ImportError as relative_error:
        try:
            return importlib.import_module(module.lstrip("."))
        except ImportError:
            raise relative_error from None


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import(module), name)
    globals()[name] = value  # resolve once; subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "PipelineConfig",
    "load_pipeline_config",
    "PIPELINE_REGISTRY",
    "build_pipeline",
    "run_pipeline",
    "Pipeline",
    "BatchDetectPipeline",
    "PeopleDetectFirstPipeline",
    "BatchPeoplePipeline",
    "ChainedPipeline",
]
