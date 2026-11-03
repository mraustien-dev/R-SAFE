from pathlib import Path

import cv2
import pytest

from rsafe.config import load_config
from rsafe.detectors.vehicle import VehicleDetector

AI_ENGINE_DIR = Path(__file__).resolve().parents[2]
SAMPLES = AI_ENGINE_DIR / "tests" / "samples"

pytestmark = pytest.mark.model


def test_detects_motorcycle_and_rider_in_sample():
    detector = VehicleDetector(load_config(AI_ENGINE_DIR / "configs" / "default.yaml").vehicle)

    motorcycles, persons = detector.track(cv2.imread(str(SAMPLES / "rider_with_helmet.jpg")))

    assert len(motorcycles) >= 1 and len(persons) >= 1
    assert detector.name == "yolo11s-coco"
