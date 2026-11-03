from pathlib import Path

import numpy as np
import pytest

from rsafe.config import PlateConfig
from rsafe.ocr.plate_reader import (
    PlateReader,
    join_fragments,
    normalize_plate_text,
    plate_search_region,
    upscale_for_ocr,
)
from rsafe.types import Detection

CFG = PlateConfig(
    enabled=True,
    detector_model=Path("unused.onnx"),
    ocr="easyocr",
    max_attempts=5,
    min_ocr_confidence=0.30,
)
FRAME = np.zeros((200, 300, 3), np.uint8)
MOTO = (100.0, 50.0, 200.0, 150.0)


class FakeDetector:
    name = "fake-plate"

    def __init__(self, box):
        self.box = box
        self.seen_shape = None

    def detect(self, crop):
        self.seen_shape = crop.shape
        return None if self.box is None else Detection(self.box, 0.8, "License_Plate")


class FakeOcr:
    name = "fake-ocr"

    def __init__(self, text, confidence):
        self.result = (text, confidence)

    def read(self, crop):
        return self.result


@pytest.mark.parametrize(
    "raw,expected",
    [("abc 1234", "ABC1234"), ("NAB-123", "NAB123"), (" 12 3a.b ", "123AB"), ("--", "")],
)
def test_normalize_plate_text(raw, expected):
    assert normalize_plate_text(raw) == expected


def test_search_region_expands_sideways_and_down():
    assert plate_search_region((200, 300), MOTO) == (90.0, 50.0, 210.0, 175.0)


def test_search_region_clipped_to_frame():
    x1, y1, x2, y2 = plate_search_region((200, 300), (0.0, 100.0, 120.0, 190.0))

    assert (x1, y2) == (0.0, 200.0)


def test_join_fragments_left_to_right_min_confidence():
    fragments = [
        ([[60, 0], [90, 0], [90, 10], [60, 10]], "1234", 0.9),
        ([[0, 0], [50, 0], [50, 10], [0, 10]], "ABC", 0.6),
    ]

    assert join_fragments(fragments) == ("ABC1234", 0.6)
    assert join_fragments([]) == ("", 0.0)


def test_join_fragments_two_line_plate_reads_top_line_first():
    # bottom line starts further left than the top line, as on many motorcycle plates
    fragments = [
        ([[2, 22], [40, 22], [40, 38], [2, 38]], "EJ2847", 0.7),
        ([[10, 2], [30, 2], [30, 18], [10, 18]], "TS07", 0.8),
    ]

    assert join_fragments(fragments) == ("TS07EJ2847", 0.7)


def test_reader_reads_plate_text():
    detector = FakeDetector((10.0, 100.0, 50.0, 120.0))
    reader = PlateReader(CFG, detector=detector, ocr=FakeOcr("abc 123", 0.7))

    result = reader.read(FRAME, MOTO)

    assert (result.status, result.text, result.ocr_confidence) == ("read", "ABC123", 0.7)
    assert result.crop.shape == (20, 40, 3)
    assert detector.seen_shape == (125, 120, 3)  # the search region, not the whole frame
    assert (reader.detector_name, reader.ocr_name) == ("fake-plate", "fake-ocr")


def test_reader_unreadable_when_text_empty():
    reader = PlateReader(CFG, detector=FakeDetector((10.0, 100.0, 50.0, 120.0)), ocr=FakeOcr("--", 0.9))

    result = reader.read(FRAME, MOTO)

    assert result.status == "unreadable" and result.text is None and result.crop is not None


def test_reader_not_visible_when_no_plate():
    reader = PlateReader(CFG, detector=FakeDetector(None), ocr=FakeOcr("ABC123", 0.9))

    result = reader.read(FRAME, MOTO)

    assert result.status == "not_visible" and result.crop is None


def test_reader_low_confidence_unreadable():
    reader = PlateReader(CFG, detector=FakeDetector((10.0, 100.0, 50.0, 120.0)), ocr=FakeOcr("ABC123", 0.1))

    assert reader.read(FRAME, MOTO).status == "unreadable"


def test_small_crop_upscaled_for_ocr_capped_at_4x():
    assert upscale_for_ocr(np.zeros((32, 58, 3), np.uint8)).shape == (96, 174, 3)
    assert upscale_for_ocr(np.zeros((10, 20, 3), np.uint8)).shape == (40, 80, 3)
    assert upscale_for_ocr(np.zeros((120, 200, 3), np.uint8)).shape == (120, 200, 3)
