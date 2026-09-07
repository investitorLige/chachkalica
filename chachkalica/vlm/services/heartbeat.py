"""Redis heartbeat telling the worker whether anyone is watching a run.

A direct port of the ``:viewed`` key in ``cameras.services.frame_cache``, for
the same reason it exists there: a VLM frame costs real GPU time, and a closed
browser tab should stop paying for it. The live page's alert poll doubles as the
heartbeat — no separate endpoint — and the worker pauses (rather than aborting)
between frames while nobody is looking, so reopening the page resumes the run.

Reuses the connection django-rq already holds rather than opening a second one.
"""

import django_rq

_KEY_PREFIX = "vlm:run:"
_VIEWED_SUFFIX = ":viewed"

# The live page polls about once a second, so this is refreshed long before it
# could expire. Long enough that a momentary network hiccup or a slow reload
# doesn't look like an abandoned tab, short enough that a genuinely closed one
# stops the GPU within a few seconds.
VIEWED_TTL_SECONDS = 30


def _key(run_id) -> str:
    return f"{_KEY_PREFIX}{run_id}{_VIEWED_SUFFIX}"


def _redis():
    return django_rq.get_connection("default")


def mark_viewed(run_id) -> None:
    """Record that someone has this run's live page open right now."""
    _redis().setex(_key(run_id), VIEWED_TTL_SECONDS, b"1")


def recently_viewed(run_id) -> bool:
    """Whether the live page has polled recently for this run."""
    return bool(_redis().exists(_key(run_id)))


def clear_viewed(run_id) -> None:
    """Forget that this run was being watched — mainly for tests.

    Production lets the TTL do this, so a brief disconnect doesn't flap the
    worker off and on.
    """
    _redis().delete(_key(run_id))
