"""Downscale camera frames to something worth sending to a browser.

A 4K camera frame is ~1.6 MB as JPEG, and the admin's MJPEG endpoint sends one
every time it polls — ~16 MB/s *per viewer*, for pixels an ``<img>`` a few
hundred CSS pixels wide never resolves. So the frame producers publish a
preview-sized copy and viewers read that instead: the resize is paid once per
frame rather than once per frame per viewer, and it happens in the worker
processes that were already decoding and encoding, not in a web worker shared
with the rest of the admin.

Full resolution is still what the *model* sees. ``camera:<id>:frame`` keeps the
untouched bytes off the RTSP stream, because ``live_inference.predict_frame``
hands exactly those to the trainer — downscaling before inference would be
throwing away the detail small/distant PPE detections depend on.
"""

# Comfortably above the width the admin's preview and live pages actually
# render at, so there's headroom for a full-screen viewer without the streams
# going soft. Downscaling 4K to this cuts a frame to roughly a tenth.
PREVIEW_MAX_WIDTH = 1280

# The preview is a lossy view of a lossy JPEG; 85 is the usual "no visible
# artefacts at a glance" point and saves a third over cv2's default 95.
JPEG_QUALITY = 85


def resize(frame):
    """Downscale ``frame`` (a BGR ndarray) to :data:`PREVIEW_MAX_WIDTH`.

    Aspect ratio is preserved. A frame already that narrow is returned *as-is*
    — the same object, which callers use to skip a redundant re-encode — so a
    720p camera costs nothing here.
    """
    import cv2

    height, width = frame.shape[:2]
    if width <= PREVIEW_MAX_WIDTH:
        return frame
    scaled_height = max(1, round(height * PREVIEW_MAX_WIDTH / width))
    # INTER_AREA is the filter meant for shrinking: it averages over the source
    # pixels that fall into each destination pixel, where INTER_LINEAR samples
    # only a 2x2 neighbourhood and aliases thin structures (railings, cables,
    # the edge of a helmet) into speckle at this kind of ratio.
    return cv2.resize(frame, (PREVIEW_MAX_WIDTH, scaled_height),
                      interpolation=cv2.INTER_AREA)


def encode(frame) -> bytes:
    """JPEG-encode a preview frame at :data:`JPEG_QUALITY`."""
    import cv2

    encoded, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not encoded:
        raise RuntimeError("Preview frame could not be JPEG-encoded.")
    return buf.tobytes()
