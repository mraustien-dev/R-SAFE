import json
from datetime import datetime

import numpy as np

from rsafe.evidence.writer import EvidenceWriter, build_record, make_violation_id
from rsafe.types import Detection, PlateResult, RiderVerdict, SourceInfo, TrackReport

SEEN_AT = datetime(2026, 11, 6, 14, 32, 12)
SOURCE = SourceInfo("ip_webcam", "http://192.168.1.5:8080/video", "cam-01")
MODELS = {"vehicle": "yolo11s-coco", "helmet": "helmet-best.onnx", "plate": None, "ocr": None}

SPEC_KEYS = {
    "schema_version", "violation_id", "type", "status", "detected_at", "source",
    "track_id", "riders", "motorcycle", "plate", "evidence", "models",
}


def make_report(plate=PlateResult("not_visible")):
    return TrackReport(
        track_id=7,
        riders=[RiderVerdict(1, "no_helmet", 0.81, 9), RiderVerdict(2, "helmet", 0.77, 8)],
        motorcycle=Detection((412, 220, 610, 540), 0.88, "motorcycle", 7),
        frame=np.zeros((720, 1280, 3), np.uint8),
        rider_crop=np.zeros((180, 100, 3), np.uint8),
        plate=plate,
        seen_at=SEEN_AT,
        rider_boxes=[(430, 100, 520, 400), (500, 120, 580, 390)],
    )


def test_violation_id_format():
    assert make_violation_id(SEEN_AT, 7) == "V-20261106-143212-0007"


def test_record_matches_schema():
    record = build_record(make_report(), SOURCE, MODELS, tz_offset_hours=8)

    assert set(record) == SPEC_KEYS
    assert record["schema_version"] == "1.0"
    assert record["type"] == "no_helmet"
    assert record["status"] == "pending_review"
    assert record["violation_id"] == "V-20261106-143212-0007"
    assert record["detected_at"] == "2026-11-06T14:32:12+08:00"
    assert record["source"] == {
        "kind": "ip_webcam", "uri": "http://192.168.1.5:8080/video", "camera_id": "cam-01"
    }
    assert record["track_id"] == 7
    assert record["riders"] == [
        {"position": 1, "helmet": "no_helmet", "confidence": 0.81, "frames_observed": 9},
        {"position": 2, "helmet": "helmet", "confidence": 0.77, "frames_observed": 8},
    ]
    assert record["motorcycle"] == {"confidence": 0.88, "bbox": [412.0, 220.0, 610.0, 540.0]}
    assert record["plate"] == {"text": None, "ocr_confidence": None, "status": "not_visible"}
    assert record["evidence"] == {"frame": "frame.jpg", "rider": "rider.jpg", "plate": None}
    assert record["models"] == MODELS
    json.dumps(record)  # must be serialisable


def test_record_plate_text_only_when_read():
    unreadable = PlateResult("unreadable", "AB", 0.1, np.zeros((20, 60, 3), np.uint8))

    record = build_record(make_report(unreadable), SOURCE, MODELS, tz_offset_hours=8)

    assert record["plate"]["text"] is None
    assert record["plate"]["status"] == "unreadable"
    assert record["evidence"]["plate"] == "plate.jpg"


def test_writer_creates_dated_folder_and_files(tmp_path):
    folder = EvidenceWriter(tmp_path, tz_offset_hours=8).write(make_report(), SOURCE, MODELS)

    assert folder == tmp_path / "2026-11-06" / "V-20261106-143212-0007"
    assert sorted(p.name for p in folder.iterdir()) == ["frame.jpg", "rider.jpg", "violation.json"]
    record = json.loads((folder / "violation.json").read_text(encoding="utf-8"))
    assert record["status"] == "pending_review"


def test_writer_draws_boxes_on_frame_copy(tmp_path):
    report = make_report()

    folder = EvidenceWriter(tmp_path, tz_offset_hours=8).write(report, SOURCE, MODELS)

    assert report.frame.max() == 0  # original untouched
    import cv2
    assert cv2.imread(str(folder / "frame.jpg")).max() > 0  # boxes drawn


def test_writer_writes_plate_jpg_when_crop_present(tmp_path):
    plate = PlateResult("read", "ABC1234", 0.64, np.zeros((20, 60, 3), np.uint8))

    folder = EvidenceWriter(tmp_path, tz_offset_hours=8).write(make_report(plate), SOURCE, MODELS)

    assert (folder / "plate.jpg").exists()
    record = json.loads((folder / "violation.json").read_text(encoding="utf-8"))
    assert record["plate"] == {"text": "ABC1234", "ocr_confidence": 0.64, "status": "read"}


def test_utc_time_gives_local_id_date_folder_and_detected_at(tmp_path):
    from dataclasses import replace
    from datetime import timezone

    report = replace(make_report(), seen_at=datetime(2026, 11, 6, 20, 0, 5, tzinfo=timezone.utc))

    folder = EvidenceWriter(tmp_path, tz_offset_hours=8).write(report, SOURCE, MODELS)
    record = json.loads((folder / "violation.json").read_text(encoding="utf-8"))

    assert record["violation_id"] == "V-20261107-040005-0007"
    assert record["detected_at"] == "2026-11-07T04:00:05+08:00"
    assert folder == tmp_path / "2026-11-07" / "V-20261107-040005-0007"


def test_writes_images_under_non_ascii_folder(tmp_path):
    folder = EvidenceWriter(tmp_path / "Peña", tz_offset_hours=8).write(make_report(), SOURCE, MODELS)

    assert (folder / "frame.jpg").stat().st_size > 0 and (folder / "rider.jpg").stat().st_size > 0


def test_failed_image_save_raises_instead_of_writing_json(tmp_path, monkeypatch):
    import pytest

    import rsafe.evidence.writer as writer_module

    monkeypatch.setattr(writer_module.cv2, "imencode", lambda *a, **k: (False, None))

    with pytest.raises(OSError, match="frame.jpg"):
        EvidenceWriter(tmp_path, tz_offset_hours=8).write(make_report(), SOURCE, MODELS)
    assert not list(tmp_path.rglob("violation.json"))
