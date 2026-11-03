import threading
import time
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest

from rsafe.app.session import (
    AnalysisSession,
    LatestFrameReader,
    VideoRecorder,
    find_cameras,
    normalize_stream_url,
    rotate_frame,
)
from rsafe.types import SourceInfo

TALL = np.zeros((4, 2, 3), np.uint8)  # 4 rows, 2 columns
TALL[0, 0] = 255  # mark the top-left pixel


@pytest.mark.parametrize(
    ("degrees", "shape", "marked"),
    [(0, (4, 2, 3), (0, 0)), (90, (2, 4, 3), (0, 3)), (180, (4, 2, 3), (3, 1)), (270, (2, 4, 3), (1, 0))],
)
def test_rotate_frame_clockwise(degrees, shape, marked):
    rotated = rotate_frame(TALL, degrees)

    assert rotated.shape == shape
    assert rotated[marked].all()


def test_rotate_frame_rejects_other_angles():
    with pytest.raises(ValueError, match="45"):
        rotate_frame(TALL, 45)


@pytest.mark.parametrize(
    ("typed", "url"),
    [
        ("192.168.1.5:8080", "http://192.168.1.5:8080/video"),
        ("  192.168.1.5:8080/  ", "http://192.168.1.5:8080/video"),
        ("http://192.168.1.5:8080", "http://192.168.1.5:8080/video"),
        ("http://192.168.1.5:8080/video", "http://192.168.1.5:8080/video"),
        ("rtsp://10.0.0.2:554/live", "rtsp://10.0.0.2:554/live"),
    ],
)
def test_normalize_stream_url(typed, url):
    assert normalize_stream_url(typed) == url


def test_normalize_stream_url_rejects_empty():
    with pytest.raises(ValueError):
        normalize_stream_url("   ")


class FakeCapture:
    def __init__(self, works):
        self.works, self.released = works, False

    def isOpened(self):
        return self.works

    def read(self):
        return self.works, (np.zeros((2, 2, 3), np.uint8) if self.works else None)

    def release(self):
        self.released = True


def test_find_cameras_returns_working_indices():
    opened = []

    def opener(index):
        opened.append(FakeCapture(index in (0, 2)))
        return opened[-1]

    assert find_cameras(max_index=4, opener=opener) == [0, 2]
    assert len(opened) == 4 and all(c.released for c in opened)


def test_latest_frame_reader_skips_to_newest():
    release = threading.Event()

    def frames():
        for index in range(5):
            yield index, np.full((2, 2, 3), index, np.uint8)
        release.wait(2)

    reader = LatestFrameReader(frames())
    reader.start()
    try:
        # wait until all 5 frames were read, then only the newest is returned
        for _ in range(100):
            got = reader.latest(timeout=0.05)
            if got and got[0] == 4:
                break
        assert got[0] == 4
        assert reader.latest(timeout=0.05) is None  # nothing new since
    finally:
        release.set()


def test_latest_frame_reader_reports_end():
    reader = LatestFrameReader(iter([(0, np.zeros((2, 2, 3), np.uint8))]))
    reader.start()

    assert reader.latest(timeout=1)[0] == 0
    assert reader.latest(timeout=1) is None and reader.ended


class FakeSource:
    def __init__(self, kind, count, shape=(4, 2, 3)):
        self.info = SourceInfo(kind=kind, uri="x", camera_id="cam-01")
        self.count, self.shape, self.closed = count, shape, False

    def frames(self):
        for index in range(self.count):
            if self.closed:
                return
            yield index, np.full(self.shape, index, np.uint8)

    def close(self):
        self.closed = True


class FakePipeline:
    def __init__(self, on_frame=None):
        self.calls, self.finished, self.on_frame = [], 0, on_frame

    def process_frame(self, index, frame, source, now):
        self.calls.append((index, frame.shape, now))
        if self.on_frame:
            self.on_frame(len(self.calls))
        return [f"folder-{index}"] if index == 6 else []

    def annotate(self, frame):
        return frame

    def finish(self, source):
        self.finished += 1
        return ["folder-last"]


START = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)


def test_file_session_uses_stride_rotation_and_film_time():
    pipeline, shown, saved = FakePipeline(), [], []
    session = AnalysisSession(
        pipeline, FakeSource("file", 10), stride=3, rotation=lambda: 90,
        start_time=START, fps=10.0, on_frame=lambda image, stats: shown.append(image.shape),
        on_saved=saved.append,
    )

    session.run()

    assert [c[0] for c in pipeline.calls] == [0, 3, 6, 9]
    assert all(c[1] == (2, 4, 3) for c in pipeline.calls)  # rotated before analysis
    assert pipeline.calls[1][2] == START + timedelta(seconds=0.3)
    assert shown == [(2, 4, 3)] * 4
    assert saved == ["folder-6", "folder-last"] and pipeline.finished == 1
    assert session.processed == 4 and session.saved == 2


def test_live_session_numbers_frames_as_if_contiguous_and_uses_now():
    source = FakeSource("ip_webcam", 5)
    pipeline = FakePipeline()
    session = AnalysisSession(
        pipeline, source, stride=3, rotation=lambda: 0, start_time=START, fps=0.0,
        on_frame=lambda image, stats: None, on_saved=lambda folder: None,
    )

    session.run()

    indices = [c[0] for c in pipeline.calls]
    assert indices == [3 * n for n in range(len(indices))] and indices
    assert all(c[2] > START for c in pipeline.calls)  # live: real clock time
    assert pipeline.finished == 1


def test_stop_ends_session_and_still_saves_open_tracks():
    holder = {}
    pipeline = FakePipeline(on_frame=lambda calls: holder["session"].stop() if calls == 2 else None)
    source = FakeSource("file", 100)
    session = AnalysisSession(
        pipeline, source, stride=1, rotation=lambda: 0, start_time=START, fps=10.0,
        on_frame=lambda image, stats: None, on_saved=lambda folder: None,
    )
    holder["session"] = session

    session.run()

    assert len(pipeline.calls) == 2 and source.closed and pipeline.finished == 1


def test_pause_blocks_until_resumed():
    pipeline = FakePipeline()
    session = AnalysisSession(
        pipeline, FakeSource("file", 3), stride=1, rotation=lambda: 0, start_time=START, fps=10.0,
        on_frame=lambda image, stats: None, on_saved=lambda folder: None,
    )
    session.pause()
    worker = threading.Thread(target=session.run)
    worker.start()

    worker.join(0.3)
    assert worker.is_alive() and pipeline.calls == []

    session.resume()
    worker.join(2)
    assert not worker.is_alive() and len(pipeline.calls) == 3


class FakeRecorder:
    def __init__(self):
        self.frames, self.closed = [], False

    def write(self, frame):
        self.frames.append(frame.shape)

    def close(self):
        self.closed = True


def make_session(pipeline, source, **options):
    defaults = dict(stride=3, rotation=lambda: 0, start_time=START, fps=10.0,
                    on_frame=lambda image, stats: None, on_saved=lambda folder: None)
    return AnalysisSession(pipeline, source, **{**defaults, **options})


def test_realtime_file_shows_every_frame_at_video_speed():
    pipeline, shown = FakePipeline(), []
    session = make_session(pipeline, FakeSource("file", 10), fps=20.0, realtime=True,
                           on_frame=lambda image, stats: shown.append(image.shape))
    started = time.perf_counter()

    session.run()

    assert time.perf_counter() - started >= 0.4  # 10 frames at 20 fps = 0.5 s
    assert len(shown) == 10  # every frame displayed
    indices = [c[0] for c in pipeline.calls]
    assert indices and indices == [3 * n for n in range(len(indices))]  # analysed: newest only
    assert pipeline.finished == 1


def test_realtime_file_records_film_time_of_the_analysed_frame():
    pipeline = FakePipeline()
    session = make_session(pipeline, FakeSource("file", 3), fps=20.0, realtime=True)

    session.run()

    for _, _, when in pipeline.calls:
        offset = (when - START).total_seconds()
        assert offset in (0.0, 0.05, 0.1)


def test_live_session_displays_every_frame_read():
    shown = []
    session = make_session(FakePipeline(), FakeSource("ip_webcam", 6), fps=0.0,
                           on_frame=lambda image, stats: shown.append(1))

    session.run()

    assert len(shown) == 6


def test_recorder_gets_analysed_frames_and_is_closed():
    recorder = FakeRecorder()
    session = make_session(FakePipeline(), FakeSource("file", 10), rotation=lambda: 90, recorder=recorder)

    session.run()

    assert recorder.frames == [(2, 4, 3)] * 4 and recorder.closed


def test_recorder_in_realtime_mode_gets_every_frame():
    recorder = FakeRecorder()
    session = make_session(FakePipeline(), FakeSource("file", 8), fps=50.0, realtime=True, recorder=recorder)

    session.run()

    assert len(recorder.frames) == 8 and recorder.closed


def test_video_recorder_writes_playable_file(tmp_path):
    path = tmp_path / "out" / "analysed.mp4"
    recorder = VideoRecorder(path, fps=10.0)
    for index in range(5):
        frame = np.full((48, 64, 3), index * 40, np.uint8)
        recorder.write(frame if index != 3 else rotate_frame(frame, 90))  # size change is resized
    recorder.close()

    capture = cv2.VideoCapture(str(path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (64, 48)
    capture.release()


def test_video_recorder_without_frames_writes_nothing(tmp_path):
    recorder = VideoRecorder(tmp_path / "none.mp4", fps=10.0)
    recorder.close()

    assert not (tmp_path / "none.mp4").exists() and recorder.path is None
