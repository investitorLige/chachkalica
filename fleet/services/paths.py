"""Resolve the shared source/target roots from FleetSettings.

Paths in FleetSettings may be absolute or relative to the project root.
"""

from pathlib import Path

from django.conf import settings

from fleet.models import FleetSettings


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(settings.BASE_DIR) / p


def source_root(fleet_settings: FleetSettings | None = None) -> Path:
    fs = fleet_settings or FleetSettings.load()
    return _resolve(fs.source_dir)


def target_root(fleet_settings: FleetSettings | None = None) -> Path:
    fs = fleet_settings or FleetSettings.load()
    return _resolve(fs.target_dir)


def annotator_base_url(annotator) -> str:
    """Base URL to reach an annotator's Label Studio from server-side code.

    The stored ``annotator.ls_url`` is ``http://localhost:<port>`` — right for
    the operator's browser, but wrong from inside the web/worker container, which
    reaches the published instances via the docker host gateway. Rewrite the host
    to ``settings.LS_HOST`` (``host.docker.internal`` under compose).
    """
    return f"http://{settings.LS_HOST}:{annotator.port}"
