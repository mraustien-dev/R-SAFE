"""Detect and track motorcycles and people with a COCO-pretrained YOLO model."""

import logging

import numpy as np
from ultralytics import YOLO

from rsafe.config import VehicleConfig
from rsafe.types import Detection

logger = logging.getLogger("rsafe.detectors.vehicle")

PERSON_CLASS_ID = 0
MOTORCYCLE_CLASS_ID = 3


def parse_results(
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
    track_ids: np.ndarray | None,
    names: dict[int, str],
) -> tuple[list[Detection], list[Detection]]:
    """Split raw YOLO arrays into (motorcycles, persons)."""
    motorcycles: list[Detection] = []
    persons: list[Detection] = []
    for index, class_id in enumerate(int(c) for c in class_ids):
        if class_id not in (PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID):
            continue
        detection = Detection(
            bbox=tuple(float(v) for v in boxes_xyxy[index]),
            confidence=float(confs[index]),
            class_name=names[class_id],
            track_id=None if track_ids is None else int(track_ids[index]),
        )
        (motorcycles if class_id == MOTORCYCLE_CLASS_ID else persons).append(detection)
    return motorcycles, persons


class VehicleDetector:
    name = "yolo11s-coco"

    def __init__(self, cfg: VehicleConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model)
        logger.info("[R-SAFE] Vehicle model loaded: %s", cfg.model)

    def track(self, frame: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        result = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID, MOTORCYCLE_CLASS_ID],
            conf=self.cfg.confidence,
            imgsz=self.cfg.imgsz,
            tracker=self.cfg.tracker,
            verbose=False,
        )[0]
        boxes = result.boxes
        track_ids = None if boxes.id is None else boxes.id.cpu().numpy()
        return parse_results(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            track_ids,
            result.names,
        )
