"""The parts of the R-SAFE window app that don't need a window (so they can be tested).

AnalysisSession feeds one video source through the pipeline:
- video files: every `stride`-th frame, in order, timed by when the video was filmed
  (slower than the video itself on a CPU);
- video files with realtime=True: the video plays at its normal speed and the engine
  analyses the newest frame it can keep up with (fewer looks at each rider);
- cameras and phone streams: like realtime, since the camera never waits.
In the "newest frame" modes every frame is shown, with the latest boxes drawn on it.
"""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path

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


class VideoRecorder:
    """Writes the analysed frames (boxes drawn) to an .mp4 that plays at normal speed."""

    def __init__(self, path: Path, fps: float):
        self._target, self.fps = Path(path), fps
        self.path: Path | None = None  # set once the first frame is written
        self._writer = None
        self._size: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._closed:
                return
            if self._writer is None:
                self._target.parent.mkdir(parents=True, exist_ok=True)
                self._size = (frame.shape[1], frame.shape[0])
                self._writer = cv2.VideoWriter(str(self._target), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)
                self.path = self._target
            if (frame.shape[1], frame.shape[0]) != self._size:  # e.g. rotation changed mid-video
                frame = cv2.resize(frame, self._size)
            self._writer.write(frame)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._writer is not None:
                self._writer.release()


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
        realtime: bool = False,
        recorder: VideoRecorder | None = None,
    ):
        self.pipeline, self.source = pipeline, source
        self.realtime, self.recorder = realtime, recorder
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
        self._started = time.perf_counter()
        try:
            if self.source.info.kind == "file" and not self.realtime:
                self._run_every_frame()
            else:
                self._run_newest_frame()
        finally:
            self.source.close()
            self._save(self.pipeline.finish(self.source.info))
            if self.recorder:
                self.recorder.close()

    def _stats(self) -> dict:
        elapsed = time.perf_counter() - self._started
        return {"processed": self.processed, "speed": self.processed / elapsed if elapsed else 0.0,
                "saved": self.saved}

    def _show(self, frame: np.ndarray) -> None:
        annotated = self.pipeline.annotate(frame)
        if self.recorder:
            self.recorder.write(annotated)
        self.on_frame(annotated, self._stats())

    def _save(self, folders: list) -> None:
        for folder in folders:
            self.saved += 1
            self.on_saved(folder)

    def _wait_if_paused(self) -> bool:
        """Block while paused; True if the session should keep going."""
        self._running.wait()
        return not self._stop.is_set()

    def _run_every_frame(self) -> None:
        for index, frame in self.source.frames():
            if not self._wait_if_paused():
                return
            if index % self.stride:
                continue
            frame = rotate_frame(frame, self.rotation())
            now = frame_timestamp(self.start_time, index, self.fps)
            self._save(self.pipeline.process_frame(index, frame, self.source.info, now))
            self.processed += 1
            self._show(frame)

    def _played_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Every frame, rotated and shown; files are paced to their real speed."""
        paced = self.source.info.kind == "file" and self.fps > 0
        clock_start = time.perf_counter()
        for index, frame in self.source.frames():
            if self._stop.is_set():
                return
            if paced:
                if self.paused:
                    waited = time.perf_counter()
                    self._running.wait()
                    clock_start += time.perf_counter() - waited  # don't rush to catch up
                delay = clock_start + index / self.fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            frame = rotate_frame(frame, self.rotation())
            self._show(frame)
            yield index, frame

    def _run_newest_frame(self) -> None:
        reader = LatestFrameReader(self._played_frames())
        reader.start()
        is_file = self.source.info.kind == "file"
        count = 0
        while self._wait_if_paused():
            item = reader.latest(timeout=0.5)
            if item is None:
                if reader.ended:
                    return
                continue
            index, frame = item
            now = frame_timestamp(self.start_time, index, self.fps) if is_file else datetime.now(timezone.utc)
            # number frames as if every `stride`-th frame was kept, so the rules'
            # "track lost after N frames" works the same as for video files
            self._save(self.pipeline.process_frame(count * self.stride, frame, self.source.info, now))
            self.processed += 1
            count += 1
