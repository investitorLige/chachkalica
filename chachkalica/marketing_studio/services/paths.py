"""Marketing Studio's two roots, and the bundle-settings shim.

The studio owns a video library and a bundle set of its own — deliberately
separate from ``FleetSettings.videos_dir`` and ``TrainingSettings.bundles_root``
(see the help text on the two fields). The video root matters most: the whole
render loop is reused from ``videos.services.inference``, which names its
scratch frame after the job's pk, and two tables with overlapping pks writing
into one directory would corrupt each other's frames mid-encode.
"""

from pathlib import Path

from fleet.services.paths import marketing_bundles_root, marketing_videos_root
from training.models import TrainingSettings

__all__ = [
    "marketing_videos_root",
    "marketing_bundles_root",
    "output_root",
    "bundle_settings",
    "bundles_root",
]


def output_root() -> Path:
    """What ``videos.services.inference``'s ``root=`` is handed; it appends
    ``inferred/`` itself, same layout as the Videos tab has under its own root."""
    return marketing_videos_root()


class _BundleSettings:
    """A ``TrainingSettings`` stand-in whose only difference is where bundles live.

    Every path in :mod:`training.services.bundles` is built from one attribute,
    ``bundles_root`` — but ``validate(load_test=True)`` hands this same object on
    to ``config_gen.build_predict_request`` and ``runner.predict_image``, which
    read ``service_base_url`` and friends and must keep getting them from the
    real singleton: it is the same trainer either way. Hence delegation rather
    than a bare namespace object, which would ``AttributeError`` inside the load
    test.

    Deliberately not a mutated copy of the row either. Nothing downstream calls
    ``ts.save()`` today, but if anything ever did, a mutated copy would write the
    marketing root back into TrainingSettings. This cannot.
    """

    def __init__(self, bundles_root: str, ts: TrainingSettings | None = None):
        # A str, matching what the model field holds — bundles.bundles_root()
        # resolves it against the project root itself.
        self.bundles_root = bundles_root
        self._ts = ts or TrainingSettings.load()

    def __getattr__(self, name):  # only fires for attributes not set above
        return getattr(self._ts, name)


def bundle_settings() -> _BundleSettings:
    """Pass this as ``ts=`` to any ``training.services.bundles`` function to make
    it resolve against the studio's bundle root instead of training's."""
    from fleet.models import FleetSettings

    return _BundleSettings(FleetSettings.load().marketing_bundles_dir)


def bundles_root() -> Path:
    from training.services import bundles

    return bundles.bundles_root(bundle_settings())
