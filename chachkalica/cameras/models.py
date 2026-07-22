from django.core.exceptions import ValidationError
from django.db import models

_RTSP_SCHEMES = ("rtsp://", "rtsps://")


def validate_rtsp_url(value: str) -> None:
    if not value.lower().startswith(_RTSP_SCHEMES):
        raise ValidationError("Must be an rtsp:// or rtsps:// URL.")


class Camera(models.Model):
    """An IP camera reachable over RTSP.

    Only the connection details live in the database — frames are pulled live
    from the camera on demand (via ``cameras.services.rtsp``), nothing is
    stored on disk yet. ``status`` reflects the last time the stream was
    opened successfully, refreshed on every save.
    """

    UNKNOWN = "unknown"
    ONLINE = "online"
    OFFLINE = "offline"
    STATUS_CHOICES = [
        (UNKNOWN, "unknown"),
        (ONLINE, "online"),
        (OFFLINE, "offline"),
    ]

    name = models.CharField(max_length=255, unique=True, help_text="Display name.")
    rtsp_url = models.CharField(
        max_length=1024,
        validators=[validate_rtsp_url],
        help_text="Full stream URL, e.g. rtsp://user:pass@10.10.10.24:554/Streaming/Channels/101",
    )
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=UNKNOWN)
    last_error = models.TextField(blank=True)
    last_checked_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name
