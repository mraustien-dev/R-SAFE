from pathlib import Path

import cv2
import numpy as np
import pytest

from rsafe.detectors.helmet import HelmetClassifier

AI_ENGINE_DIR = Path(__file__).resolve().parents[2]
MODEL = AI_ENGINE_DIR / "models" / "helmet" / "best.onnx"
SAMPLES = AI_ENGINE_DIR / "tests" / "samples"

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def classifier():
    return HelmetClassifier(MODEL, confidence=0.25)


def test_rider_with_helmet_sample(classifier):
    result = classifier.classify(cv2.imread(str(SAMPLES / "rider_with_helmet.jpg")))

    assert result is not None and result.label == "helmet"


def test_rider_without_helmet_sample(classifier):
    result = classifier.classify(cv2.imread(str(SAMPLES / "rider_without_helmet.jpg")))

    assert result is not None and result.label == "no_helmet"


def test_blank_crop_returns_none(classifier):
    assert classifier.classify(np.full((120, 120, 3), 114, np.uint8)) is None


def test_empty_array_returns_none(classifier):
    assert classifier.classify(np.zeros((0, 0, 3), np.uint8)) is None


def test_classifier_name(classifier):
    assert classifier.name == "helmet-best.onnx"


@pytest.mark.parametrize("imgsz", [416, 320])
def test_smaller_input_size_still_judges_samples(imgsz):
    small = HelmetClassifier(MODEL, confidence=0.25, imgsz=imgsz)

    assert small.detector.image_size == imgsz
    assert small.classify(cv2.imread(str(SAMPLES / "rider_with_helmet.jpg"))).label == "helmet"
    assert small.classify(cv2.imread(str(SAMPLES / "rider_without_helmet.jpg"))).label == "no_helmet"
