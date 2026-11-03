from pathlib import Path

import numpy as np
import pytest

from rsafe.detectors.plate import PlateDetector

AI_ENGINE_DIR = Path(__file__).resolve().parents[2]
PLATE_MODEL = AI_ENGINE_DIR / "models" / "plate" / "plate.onnx"

pytestmark = pytest.mark.model


def test_plate_detector_loads_and_handles_blank_crop():
    detector = PlateDetector(PLATE_MODEL)

    assert detector.detect(np.zeros((120, 160, 3), np.uint8)) is None
    assert detector.detect(np.zeros((0, 0, 3), np.uint8)) is None
    assert detector.name == "plate-yolo11s.onnx"
