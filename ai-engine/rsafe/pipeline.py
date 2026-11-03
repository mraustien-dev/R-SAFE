"""Run every stage on each frame: track -> riders -> helmet -> votes -> plate -> evidence -> sink.

Components are passed in (not built here) so tests can use fakes; build_pipeline()
makes the real ones from the config.
"""

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from rsafe.association.riders import associate_riders, head_crop
from rsafe.config import Config
from rsafe.evidence.writer import EvidenceWriter
from rsafe.sinks.base import Sink
from rsafe.types import Detection, HelmetResult, SourceInfo, TrackReport
from rsafe.violations.rules import ViolationTracker

logger = logging.getLogger("rsafe.pipeline")

MOTO_COLOR = (255, 160, 0)
LABEL_COLORS = {"helmet": (0, 200, 0), "no_helmet": (0, 0, 255), None: (160, 160, 160)}


class Pipeline:
    def __init__(self, cfg: Config, vehicle, helmet, plate_reader, writer: EvidenceWriter, sink: Sink):
        self.cfg = cfg
        self.vehicle = vehicle
        self.helmet = helmet
        self.plate_reader = plate_reader
        self.writer = writer
        self.sink = sink
        self.tracker = ViolationTracker(cfg.rules, cfg.plate.max_attempts)
        self.models = {
            "vehicle": vehicle.name,
            "helmet": helmet.name,
            "plate": plate_reader.detector_name if plate_reader else None,
            "ocr": plate_reader.ocr_name if plate_reader else None,
        }
        self._last_drawn: list[tuple[Detection, list[tuple[Detection, HelmetResult | None]]]] = []

    def process_frame(self, frame_index: int, frame: np.ndarray, source: SourceInfo, now: datetime) -> list[Path]:
        try:
            self._analyse(frame_index, frame, now)
        except Exception:
            logger.exception("[R-SAFE] Frame %d skipped after an error", frame_index)
        return self._emit(self.tracker.collect_finished(frame_index), source)

    def finish(self, source: SourceInfo) -> list[Path]:
        return self._emit(self.tracker.flush(), source)

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Copy of the frame with the last processed frame's motorcycles and riders drawn."""
        shown = frame.copy()
        for moto, riders in self._last_drawn:
            x1, y1, x2, y2 = (int(v) for v in moto.bbox)
            cv2.rectangle(shown, (x1, y1), (x2, y2), MOTO_COLOR, 2)
            cv2.putText(shown, f"track {moto.track_id}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, MOTO_COLOR, 2)
            for rider, result in riders:
                label = result.label if result else None
                rx1, ry1, rx2, ry2 = (int(v) for v in rider.bbox)
                color = LABEL_COLORS[label]
                cv2.rectangle(shown, (rx1, ry1), (rx2, ry2), color, 2)
                text = f"{label} {result.confidence:.2f}" if result else "?"
                cv2.putText(shown, text, (rx1, max(15, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return shown

    def _analyse(self, frame_index: int, frame: np.ndarray, now: datetime) -> None:
        self._last_drawn = []
        motorcycles, persons = self.vehicle.track(frame)
        motorcycles = [moto for moto in motorcycles if moto.track_id is not None]
        assoc, helmet_cfg = self.cfg.association, self.cfg.helmet
        riders_by_moto = associate_riders(motorcycles, persons, assoc.expand_ratio, assoc.max_riders)

        for index, moto in enumerate(motorcycles):
            riders = []
            for position, rider in enumerate(riders_by_moto[index], start=1):
                result = None
                if rider.bbox[3] - rider.bbox[1] >= helmet_cfg.min_rider_height_px:
                    crop = head_crop(frame, rider.bbox, helmet_cfg.head_ratio, helmet_cfg.pad_ratio)
                    result = self.helmet.classify(crop) if crop is not None else None
                riders.append((position, rider, result))
            self._last_drawn.append((moto, [(rider, result) for _, rider, result in riders]))
            self.tracker.update(frame_index, frame, moto, riders, now)

            # OCR is slow on CPU: read plates only while a confident no-helmet rider is in view
            violator_seen = any(
                result is not None
                and result.label == "no_helmet"
                and result.confidence >= self.cfg.rules.min_confidence
                for _, _, result in riders
            )
            if self.plate_reader and violator_seen and self.tracker.wants_plate(moto.track_id):
                self.tracker.add_plate(moto.track_id, self.plate_reader.read(frame, moto.bbox))

    def _emit(self, reports: list[TrackReport], source: SourceInfo) -> list[Path]:
        written = []
        for report in reports:
            try:
                folder = self.writer.write(report, source, self.models)
                logger.info("[R-SAFE] Potential violation saved: %s", folder)
                self.sink.send(folder)
            except Exception:
                logger.exception("[R-SAFE] Could not save or send the record for track %s", report.track_id)
                continue
            written.append(folder)
        return written


def build_pipeline(cfg: Config) -> Pipeline:
    """Load the real models named in the config (slow: a few seconds)."""
    from rsafe.detectors.helmet import HelmetClassifier
    from rsafe.detectors.vehicle import VehicleDetector
    from rsafe.sinks.http_sink import make_sink

    plate_reader = None
    if cfg.plate.enabled:
        from rsafe.ocr.plate_reader import PlateReader

        plate_reader = PlateReader(cfg.plate)
    return Pipeline(
        cfg,
        VehicleDetector(cfg.vehicle),
        HelmetClassifier(cfg.helmet.model, cfg.helmet.confidence),
        plate_reader,
        EvidenceWriter(cfg.evidence.output_dir, cfg.evidence.timezone_offset_hours),
        make_sink(cfg.sink, cfg.evidence.output_dir),
    )
