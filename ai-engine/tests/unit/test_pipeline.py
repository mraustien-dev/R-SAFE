import json
from datetime import datetime
from pathlib import Path

import numpy as np

from rsafe.config import load_config
from rsafe.evidence.writer import EvidenceWriter
from rsafe.pipeline import Pipeline
from rsafe.types import Detection, HelmetResult, PlateResult, SourceInfo

AI_ENGINE_DIR = Path(__file__).resolve().parents[2]
CFG = load_config(AI_ENGINE_DIR / "configs" / "default.yaml")
SOURCE = SourceInfo(kind="file", uri="clip.mp4", camera_id="cam-01")
NOW = datetime(2026, 11, 6, 14, 32, 12)
FRAME = np.zeros((480, 640, 3), np.uint8)
MOTO_BOX = (100.0, 100.0, 300.0, 300.0)
RIDER_BOX = (150.0, 60.0, 250.0, 290.0)


class FakeVehicle:
    name = "fake-vehicle"

    def __init__(self, track_id=1, rider_box=RIDER_BOX, fail_on=(), empty_from=None):
        self.track_id, self.rider_box = track_id, rider_box
        self.fail_on, self.empty_from = set(fail_on), empty_from
        self.calls = 0

    def track(self, frame):
        call, self.calls = self.calls, self.calls + 1
        if call in self.fail_on:
            raise RuntimeError("boom")
        if self.empty_from is not None and call >= self.empty_from:
            return [], []
        moto = Detection(MOTO_BOX, 0.9, "motorcycle", self.track_id)
        return [moto], [Detection(self.rider_box, 0.9, "person")]


class FakeHelmet:
    name = "fake-helmet"

    def __init__(self, label):
        self.label = label
        self.calls = 0

    def classify(self, crop):
        self.calls += 1
        return HelmetResult(self.label, 0.9)


class FakePlateReader:
    detector_name, ocr_name = "fake-plate", "fake-ocr"

    def __init__(self):
        self.calls = 0

    def read(self, frame, moto):
        self.calls += 1
        return PlateResult("read", text="ABC1234", ocr_confidence=0.95, crop=np.zeros((10, 30, 3), np.uint8))


class FakeSink:
    def __init__(self):
        self.sent = []

    def send(self, folder):
        self.sent.append(folder)


def make_pipeline(tmp_path, vehicle, helmet, plate_reader=None):
    sink = FakeSink()
    writer = EvidenceWriter(tmp_path / "violations", CFG.evidence.timezone_offset_hours)
    return Pipeline(CFG, vehicle, helmet, plate_reader, writer, sink), sink


def run(pipeline, frames):
    written = []
    for index in range(frames):
        written += pipeline.process_frame(index, FRAME, SOURCE, NOW)
    return written


def records(tmp_path):
    return [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "violations").rglob("violation.json")]


def test_no_helmet_rider_produces_one_violation_folder(tmp_path):
    pipeline, sink = make_pipeline(tmp_path, FakeVehicle(), FakeHelmet("no_helmet"))

    written = run(pipeline, 12) + pipeline.finish(SOURCE)

    assert len(written) == 1 and sink.sent == written
    (record,) = records(tmp_path)
    assert record["riders"][0]["helmet"] == "no_helmet"
    assert record["plate"]["status"] == "not_visible"
    assert record["status"] == "pending_review"
    assert record["models"] == {"vehicle": "fake-vehicle", "helmet": "fake-helmet", "plate": None, "ocr": None}


def test_helmet_rider_produces_nothing(tmp_path):
    pipeline, sink = make_pipeline(tmp_path, FakeVehicle(), FakeHelmet("helmet"))

    assert run(pipeline, 12) + pipeline.finish(SOURCE) == [] and sink.sent == []


def test_untracked_motorcycle_ignored(tmp_path):
    helmet = FakeHelmet("no_helmet")
    pipeline, _ = make_pipeline(tmp_path, FakeVehicle(track_id=None), helmet)

    assert run(pipeline, 12) + pipeline.finish(SOURCE) == []
    assert helmet.calls == 0


def test_small_rider_not_classified(tmp_path):
    helmet = FakeHelmet("no_helmet")
    tiny_rider = (190.0, 260.0, 210.0, 290.0)  # 30 px tall < min_rider_height_px 48
    pipeline, _ = make_pipeline(tmp_path, FakeVehicle(rider_box=tiny_rider), helmet)

    assert run(pipeline, 12) + pipeline.finish(SOURCE) == []
    assert helmet.calls == 0


def test_frame_exception_is_skipped(tmp_path):
    vehicle = FakeVehicle(fail_on={3})
    pipeline, _ = make_pipeline(tmp_path, vehicle, FakeHelmet("no_helmet"))

    written = run(pipeline, 12) + pipeline.finish(SOURCE)

    assert vehicle.calls == 12 and len(written) == 1


def test_lost_track_written_during_processing(tmp_path):
    # motorcycle seen for 12 frames, then gone; closed once > track_lost_frames (30) later
    pipeline, _ = make_pipeline(tmp_path, FakeVehicle(empty_from=12), FakeHelmet("no_helmet"))

    assert len(run(pipeline, 12 + 32)) == 1
    assert pipeline.finish(SOURCE) == []


def test_finish_flushes_active_track(tmp_path):
    pipeline, _ = make_pipeline(tmp_path, FakeVehicle(), FakeHelmet("no_helmet"))

    assert run(pipeline, 12) == []
    assert len(pipeline.finish(SOURCE)) == 1


def test_plate_read_only_for_no_helmet_and_capped(tmp_path):
    plates = FakePlateReader()
    pipeline, _ = make_pipeline(tmp_path, FakeVehicle(), FakeHelmet("no_helmet"), plates)

    run(pipeline, 12)
    pipeline.finish(SOURCE)

    assert plates.calls == CFG.plate.max_attempts
    (record,) = records(tmp_path)
    assert record["plate"] == {"text": "ABC1234", "ocr_confidence": 0.95, "status": "read"}
    assert record["models"]["plate"] == "fake-plate" and record["models"]["ocr"] == "fake-ocr"


def test_plate_not_read_when_riders_wear_helmets(tmp_path):
    plates = FakePlateReader()
    pipeline, _ = make_pipeline(tmp_path, FakeVehicle(), FakeHelmet("helmet"), plates)

    run(pipeline, 12)

    assert plates.calls == 0


def test_annotate_draws_on_a_copy(tmp_path):
    pipeline, _ = make_pipeline(tmp_path, FakeVehicle(), FakeHelmet("no_helmet"))
    run(pipeline, 1)

    shown = pipeline.annotate(FRAME)

    assert shown is not FRAME and shown.any() and not FRAME.any()


class FailingOnceSink(FakeSink):
    def send(self, folder):
        if not self.sent:
            self.sent.append(None)
            raise OSError("disk full")
        self.sent.append(folder)


class TwoMotorcycles(FakeVehicle):
    def track(self, frame):
        self.calls += 1
        motos = [Detection(MOTO_BOX, 0.9, "motorcycle", 1), Detection((400.0, 100.0, 600.0, 300.0), 0.9, "motorcycle", 2)]
        riders = [Detection(RIDER_BOX, 0.9, "person"), Detection((450.0, 60.0, 550.0, 290.0), 0.9, "person")]
        return motos, riders


def test_saving_error_does_not_stop_other_reports(tmp_path):
    writer = EvidenceWriter(tmp_path / "violations", CFG.evidence.timezone_offset_hours)
    pipeline = Pipeline(CFG, TwoMotorcycles(), FakeHelmet("no_helmet"), None, writer, FailingOnceSink())

    run(pipeline, 12)
    written = pipeline.finish(SOURCE)

    assert len(written) == 1 and len(records(tmp_path)) == 2
