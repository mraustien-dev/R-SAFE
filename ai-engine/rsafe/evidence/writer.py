"""Write one evidence folder (images + violation.json) per violation.

The JSON shape is the integration contract documented in docs/violation_schema.md.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from rsafe.types import SourceInfo, TrackReport

logger = logging.getLogger("rsafe.evidence")

SCHEMA_VERSION = "1.0"
MOTORCYCLE_COLOR = (255, 160, 0)
RIDER_COLOR = (0, 0, 255)


def make_violation_id(seen_at: datetime, track_id: int) -> str:
    return f"V-{seen_at:%Y%m%d-%H%M%S}-{track_id:04d}"


def _local_time(seen_at: datetime, tz_offset_hours: int) -> datetime:
    zone = timezone(timedelta(hours=tz_offset_hours))
    return seen_at.replace(tzinfo=zone) if seen_at.tzinfo is None else seen_at.astimezone(zone)


def build_record(
    report: TrackReport, source: SourceInfo, models: dict[str, str], tz_offset_hours: int
) -> dict:
    plate = report.plate
    return {
        "schema_version": SCHEMA_VERSION,
        "violation_id": make_violation_id(report.seen_at, report.track_id),
        "type": "no_helmet",
        "status": "pending_review",
        "detected_at": _local_time(report.seen_at, tz_offset_hours).isoformat(timespec="seconds"),
        "source": {"kind": source.kind, "uri": source.uri, "camera_id": source.camera_id},
        "track_id": report.track_id,
        "riders": [
            {
                "position": rider.position,
                "helmet": rider.helmet,
                "confidence": rider.confidence,
                "frames_observed": rider.frames_observed,
            }
            for rider in report.riders
        ],
        "motorcycle": {
            "confidence": round(report.motorcycle.confidence, 4),
            "bbox": [round(float(v), 1) for v in report.motorcycle.bbox],
        },
        "plate": {
            "text": plate.text if plate.status == "read" else None,
            "ocr_confidence": plate.ocr_confidence if plate.status == "read" else None,
            "status": plate.status,
        },
        "evidence": {
            "frame": "frame.jpg",
            "rider": "rider.jpg",
            "plate": "plate.jpg" if plate.crop is not None else None,
        },
        "models": dict(models),
    }


def _draw_boxes(report: TrackReport) -> np.ndarray:
    image = report.frame.copy()
    x1, y1, x2, y2 = (int(v) for v in report.motorcycle.bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), MOTORCYCLE_COLOR, 2)
    cv2.putText(
        image, f"track {report.track_id}", (x1, max(y1 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, MOTORCYCLE_COLOR, 2, cv2.LINE_AA,
    )
    for box in report.rider_boxes:
        bx1, by1, bx2, by2 = (int(v) for v in box)
        cv2.rectangle(image, (bx1, by1), (bx2, by2), RIDER_COLOR, 2)
    return image


class EvidenceWriter:
    def __init__(self, output_dir: Path, tz_offset_hours: int):
        self.output_dir = Path(output_dir)
        self.tz_offset_hours = tz_offset_hours

    def write(self, report: TrackReport, source: SourceInfo, models: dict[str, str]) -> Path:
        record = build_record(report, source, models, self.tz_offset_hours)
        local = _local_time(report.seen_at, self.tz_offset_hours)
        folder = self.output_dir / f"{local:%Y-%m-%d}" / record["violation_id"]
        folder.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(folder / "frame.jpg"), _draw_boxes(report))
        cv2.imwrite(str(folder / "rider.jpg"), report.rider_crop)
        if report.plate.crop is not None:
            cv2.imwrite(str(folder / "plate.jpg"), report.plate.crop)
        (folder / "violation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

        logger.info("[R-SAFE] Violation saved: %s", folder)
        return folder
