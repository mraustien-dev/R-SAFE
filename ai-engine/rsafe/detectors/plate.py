"""Find a license plate inside a crop with the morsetechlab YOLO11s plate model (ONNX)."""

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from rsafe.types import Detection

logger = logging.getLogger("rsafe.detectors.plate")


class PlateDetector:
    name = "plate-yolo11s.onnx"

    def __init__(self, model_path: Path, confidence: float = 0.25, device: str = "cpu"):
        self.confidence = confidence
        self.device = 0 if device == "cuda" else "cpu"
        self.model = YOLO(str(model_path), task="detect")

    def detect(self, crop: np.ndarray) -> Detection | None:
        """Highest-confidence plate box in crop coordinates, or None."""
        if crop is None or crop.size == 0:
            return None
        result = self.model.predict(crop, conf=self.confidence, device=self.device, verbose=False)[0]
        if len(result.boxes) == 0:
            return None
        confs = result.boxes.conf.cpu().numpy()
        best = int(confs.argmax())
        x1, y1, x2, y2 = (float(v) for v in result.boxes.xyxy.cpu().numpy()[best])
        return Detection((x1, y1, x2, y2), float(confs[best]), "license_plate")
