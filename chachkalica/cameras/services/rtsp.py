"""RTSP connection check — opens the stream just long enough to read one frame."""

import multiprocessing

DEFAULT_TIMEOUT_MS = 5000

# FFmpeg's OPEN/READ timeout properties bound the RTSP handshake but not
# reliably the initial TCP connect to an unreachable host, so an in-process
# call can hang far longer than ``timeout_ms``. The probe runs in its own
# subprocess so a hung attempt can be killed outright instead of blocking
# the caller (e.g. the single-worker gunicorn request thread) indefinitely.
_TERMINATE_GRACE_SECONDS = 2


def _probe(rtsp_url: str, timeout_ms: int, result_queue) -> None:
    import cv2

    capture = cv2.VideoCapture()
    capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
    try:
        if not capture.open(rtsp_url, cv2.CAP_FFMPEG):
            result_queue.put((False, "Could not open stream (check URL, credentials, and network)."))
            return
        ok, _frame = capture.read()
        if not ok:
            result_queue.put((False, "Stream opened but no frame could be read."))
            return
        result_queue.put((True, ""))
    finally:
        capture.release()


def test_connection(rtsp_url: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> tuple[bool, str]:
    """Try to open ``rtsp_url`` and read a single frame.

    Returns ``(True, "")`` on success, or ``(False, error_message)``. Always
    returns within roughly ``timeout_ms`` plus a couple of seconds, even if
    the underlying cv2/FFmpeg call itself would hang indefinitely.
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_probe, args=(rtsp_url, timeout_ms, result_queue))
    process.start()
    process.join(timeout_ms / 1000 + _TERMINATE_GRACE_SECONDS)

    if process.is_alive():
        process.terminate()
        process.join(_TERMINATE_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
            process.join()
        return False, "Connection attempt timed out."

    if not result_queue.empty():
        return result_queue.get()
    return False, "Connection probe exited unexpectedly."
