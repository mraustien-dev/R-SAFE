"""The parts of the R-SAFE window app that don't need a window (so they can be tested).

AnalysisSession feeds one video source through the pipeline:
- video files: every `stride`-th frame, in order, timed by when the video was filmed;
- cameras and phone streams: always the newest frame (the engine is slower than the
  camera, so older frames are skipped instead of falling further and further behind).
"""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone

import cv2
import numpy as np

from rsafe.video.source import frame_timestamp

logger = logging.getLogger("rsafe.app")

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Turn the frame clockwise by 0, 90, 180 or 270 degrees."""
    if degrees not in ROTATIONS:
        raise ValueError(f"Rotation must be 0, 90, 180 or 270, not {degrees}")
    code = ROTATIONS[degrees]
    return frame if code is None else cv2.rotate(frame, code)


def normalize_stream_url(text: str) -> str:
    """'192.168.1.5:8080' -> 'http://192.168.1.5:8080/video' (IP Webcam app's address)."""
    url = text.strip()
    if not url:
        raise ValueError("Type the phone's address, e.g. 192.168.1.5:8080")
    if "://" not in url:
        url = "http://" + url
    scheme, rest = url.split("://", 1)
    if scheme.lower() in ("http", "https") and "/" not in rest.rstrip("/"):
        url = f"{scheme}://{rest.rstrip('/')}/video"
    return url


def _open_camera(index: int):
    # DirectShow opens Windows cameras much faster than the default backend
    return cv2.VideoCapture(index, cv2.CAP_DSHOW)


def find_cameras(max_index: int = 4, opener: Callable = _open_camera) -> list[int]:
    """Indexes of the cameras (built-in, USB) that open and give a picture."""
    found = []
    for index in range(max_index):
        capture = opener(index)
        try:
            if capture.isOpened() and capture.read()[0]:
                found.append(index)
        finally:
            capture.release()
    return found


class LatestFrameReader:
    """Reads a live source in its own thread and keeps only the newest frame."""

    def __init__(self, frames: Iterator[tuple[int, np.ndarray]]):
        self._frames = frames
        self._latest: tuple[int, np.ndarray] | None = None
        self._condition = threading.Condition()
        self.ended = False

    def start(self) -> None:
        threading.Thread(target=self._read, daemon=True).start()

    def latest(self, timeout: float) -> tuple[int, np.ndarray] | None:
        """The newest frame not returned before, or None if none came within `timeout`."""
        with self._condition:
            if self._latest is None and not self.ended:
                self._condition.wait(timeout)
            latest, self._latest = self._latest, None
            return latest

    def _read(self) -> None:
        try:
            for item in self._frames:
                with self._condition:
                    self._latest = item
                    self._condition.notify_all()
        except Exception:
            logger.exception("[R-SAFE] Camera reading failed")
        finally:
            with self._condition:
                self.ended = True
                self._condition.notify_all()


class AnalysisSession:
    def __init__(
        self,
        pipeline,
        source,
        *,
        stride: int,
        rotation: Callable[[], int],
        start_time: datetime,
        fps: float,
        on_frame: Callable[[np.ndarray, dict], None],
        on_saved: Callable[[object], None],
    ):
        self.pipeline, self.source = pipeline, source
        self.stride, self.rotation = stride, rotation
        self.start_time, self.fps = start_time, fps
        self.on_frame, self.on_saved = on_frame, on_saved
        self.processed = 0
        self.saved = 0
        self._stop = threading.Event()
        self._running = threading.Event()
        self._running.set()

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def stop(self) -> None:
        self._stop.set()
        self._running.set()  # wake a paused session so it can end
        self.source.close()

    def run(self) -> None:
        """Process the source until it ends or stop() is called, then save open tracks."""
        started = time.perf_counter()
        try:
            for index, frame, now in self._frames():
                frame = rotate_frame(frame, self.rotation())
                self._save(self.pipeline.process_frame(index, frame, self.source.info, now))
                self.processed += 1
                elapsed = time.perf_counter() - started
                self.on_frame(self.pipeline.annotate(frame), {
                    "processed": self.processed,
                    "speed": self.processed / elapsed if elapsed else 0.0,
                    "saved": self.saved,
                })
        finally:
            self.source.close()
            self._save(self.pipeline.finish(self.source.info))

    def _save(self, folders: list) -> None:
        for folder in folders:
            self.saved += 1
            self.on_saved(folder)

    def _wait_if_paused(self) -> bool:
        """Block while paused; True if the session should keep going."""
        self._running.wait()
        return not self._stop.is_set()

    def _frames(self) -> Iterator[tuple[int, np.ndarray, datetime]]:
        if self.source.info.kind == "file":
            for index, frame in self.source.frames():
                if not self._wait_if_paused():
                    return
                if index % self.stride == 0:
                    yield index, frame, frame_timestamp(self.start_time, index, self.fps)
            return

        reader = LatestFrameReader(self.source.frames())
        reader.start()
        count = 0
        while self._wait_if_paused():
            item = reader.latest(timeout=0.5)
            if item is None:
                if reader.ended:
                    return
                continue
            # number frames as if every `stride`-th frame was kept, so the rules'
            # "track lost after N frames" works the same as for video files
            yield count * self.stride, item[1], datetime.now(timezone.utc)
            count += 1
