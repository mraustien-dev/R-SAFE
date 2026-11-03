import cv2
import numpy as np
import pytest

import rsafe.video.source as source_module
from rsafe.video.source import VideoSource


def write_video(path, frame_count):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for index in range(frame_count):
        writer.write(np.full((48, 64, 3), index * 10, np.uint8))
    writer.release()


def test_file_source_yields_all_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    write_video(video, 12)

    source = VideoSource(str(video), "cam-01")
    frames = list(source.frames())

    assert [index for index, _ in frames] == list(range(12))
    assert frames[0][1].shape == (48, 64, 3)
    assert source.info.kind == "file" and source.info.camera_id == "cam-01"


def test_missing_file_raises_with_path(tmp_path):
    missing = tmp_path / "nope.mp4"

    with pytest.raises(FileNotFoundError, match="nope.mp4"):
        VideoSource(str(missing), "cam-01")


def test_kind_detection(tmp_path):
    video = tmp_path / "clip.mp4"
    write_video(video, 1)

    assert VideoSource("0", "cam").info.kind == "device"
    assert VideoSource("http://1.2.3.4:8080/video", "cam").info.kind == "ip_webcam"
    assert VideoSource("rtsp://1.2.3.4/stream", "cam").info.kind == "ip_webcam"
    assert VideoSource(str(video), "cam").info.kind == "file"


class FakeCapture:
    """Fails to open the first `failures` times, then serves frames forever."""

    opened = 0
    failures = 0

    def __init__(self, target):
        FakeCapture.opened += 1
        self._ok = FakeCapture.opened > FakeCapture.failures

    def isOpened(self):
        return self._ok

    def read(self):
        return (True, np.zeros((4, 4, 3), np.uint8)) if self._ok else (False, None)

    def release(self):
        pass


def test_stream_reconnect_backoff_capped(monkeypatch):
    FakeCapture.opened, FakeCapture.failures = 0, 7
    monkeypatch.setattr(source_module.cv2, "VideoCapture", FakeCapture)
    sleeps = []

    source = VideoSource("http://phone:8080/video", "cam", sleep=sleeps.append)
    index, frame = next(source.frames())

    assert sleeps == [1, 2, 4, 8, 16, 30, 30]
    assert index == 0 and frame.shape == (4, 4, 3)


class DroppingCapture:
    """Serves 2 frames per connection, then the read fails (stream dropped)."""

    def __init__(self, target):
        self._reads = 0

    def isOpened(self):
        return True

    def read(self):
        self._reads += 1
        return (True, np.zeros((4, 4, 3), np.uint8)) if self._reads <= 2 else (False, None)

    def release(self):
        pass


def test_stream_drop_reconnects_and_keeps_counting(monkeypatch):
    monkeypatch.setattr(source_module.cv2, "VideoCapture", DroppingCapture)
    sleeps = []

    source = VideoSource("http://phone:8080/video", "cam", sleep=sleeps.append)
    frames = source.frames()
    indices = [next(frames)[0] for _ in range(5)]

    assert indices == [0, 1, 2, 3, 4]
    assert sleeps == [1, 1]  # backoff resets after a good frame


def test_close_stops_stream(monkeypatch):
    monkeypatch.setattr(source_module.cv2, "VideoCapture", DroppingCapture)

    source = VideoSource("http://phone:8080/video", "cam", sleep=lambda s: None)
    frames = source.frames()
    next(frames)
    source.close()

    assert list(frames) == []
