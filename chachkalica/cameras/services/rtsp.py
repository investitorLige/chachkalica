"""RTSP connection check — opens the stream just long enough to read one frame."""

DEFAULT_TIMEOUT_MS = 5000


def test_connection(rtsp_url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> tuple[bool, str]:
    """Try to open ``rtsp_url`` and read a single frame.

    Returns ``(True, "")`` on success, or ``(False, error_message)``. The
    open/read timeouts are FFMPEG-backend properties, so a hung/unreachable
    camera fails fast instead of blocking the caller indefinitely.
    """
    import cv2

    capture = cv2.VideoCapture()
    capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    try:
        if not capture.open(rtsp_url, cv2.CAP_FFMPEG):
            return False, "Could not open stream (check URL, credentials, and network)."
        ok, _frame = capture.read()
        if not ok:
            return False, "Stream opened but no frame could be read."
        return True, ""
    finally:
        capture.release()
