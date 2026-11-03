"""Read frames from a video file, an IP Webcam stream, or a local camera.

Files end when the last frame is read. Streams and cameras reconnect after a drop,
waiting 1, 2, 4, ... seconds (capped) between attempts, until close() is called.
"""

import logging
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from rsafe.types import SourceInfo

logger = logging.getLogger("rsafe.video")

STREAM_PREFIXES = ("http://", "https://", "rtsp://")


def frame_timestamp(start: datetime, index: int, fps: float) -> datetime:
    """When frame `index` of a recorded video was filmed, given when the video started."""
    return start + timedelta(seconds=index / fps) if fps > 0 else start


def video_fps(path: str) -> float:
    capture = cv2.VideoCapture(path)
    try:
        return float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()


class VideoSource:
    def __init__(
        self,
        source: str,
        camera_id: str,
        max_backoff_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        source = str(source).strip()
        if source.isdigit():
            kind, self._target = "device", int(source)
        elif source.lower().startswith(STREAM_PREFIXES):
            kind, self._target = "ip_webcam", source
        else:
            if not Path(source).is_file():
                raise FileNotFoundError(f"Video file not found: {source}")
            kind, self._target = "file", source

        self.info = SourceInfo(kind=kind, uri=source, camera_id=camera_id)
        self.max_backoff_s = max_backoff_s
        self._sleep = sleep
        self._closed = False

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        if self.info.kind == "file":
            yield from self._file_frames()
        else:
            yield from self._stream_frames()

    def close(self) -> None:
        self._closed = True

    def _file_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        capture = cv2.VideoCapture(self._target)
        try:
            index = 0
            while not self._closed:
                ok, frame = capture.read()
                if not ok:
                    return
                yield index, frame
                index += 1
        finally:
            capture.release()

    def _stream_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        index = 0
        backoff = 1.0
        while not self._closed:
            capture = cv2.VideoCapture(self._target)
            try:
                if not capture.isOpened():
                    logger.warning(
                        "[R-SAFE] Cannot open %s; retrying in %.0f s", self.info.uri, backoff
                    )
                else:
                    while not self._closed:
                        ok, frame = capture.read()
                        if not ok:
                            logger.warning(
                                "[R-SAFE] Stream dropped: %s; reconnecting in %.0f s",
                                self.info.uri,
                                backoff,
                            )
                            break
                        backoff = 1.0
                        yield index, frame
                        index += 1
            finally:
                capture.release()
            if self._closed:
                return
            self._sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff_s)
