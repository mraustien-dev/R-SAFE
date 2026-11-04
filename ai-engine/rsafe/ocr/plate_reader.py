"""Read a motorcycle's plate: find the plate near the motorcycle, then OCR it.

The text is a reading aid for the human reviewer, never proof: status "read" only
means OCR was confident enough to show its guess.
"""

import logging
import re

import cv2
import numpy as np

from rsafe.config import PlateConfig
from rsafe.types import BBox, PlateResult

logger = logging.getLogger("rsafe.ocr")

PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OCR_MIN_HEIGHT_PX = 96  # small plates read slightly better enlarged (measured on RideSafe-400)
OCR_MAX_UPSCALE = 4


def normalize_plate_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def plate_search_region(frame_shape: tuple[int, int], moto: BBox) -> BBox:
    """Motorcycle box widened 10% each side and extended 25% downward, clipped to the frame."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = moto
    pad_x, pad_down = (x2 - x1) * 0.10, (y2 - y1) * 0.25
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1),
        min(float(width), x2 + pad_x),
        min(float(height), y2 + pad_down),
    )


def join_fragments(fragments) -> tuple[str, float]:
    """EasyOCR results [(points, text, conf)] -> (text read line by line, lowest confidence).

    Two-line plates: a fragment whose vertical centre falls inside the current line's
    span joins that line; lines are read top to bottom, each left to right.
    """
    if not fragments:
        return "", 0.0

    def span(fragment):
        ys = [point[1] for point in fragment[0]]
        return min(ys), max(ys)

    lines: list[list] = []
    for fragment in sorted(fragments, key=lambda f: sum(span(f)) / 2):
        centre = sum(span(fragment)) / 2
        top, bottom = span(lines[-1][0]) if lines else (0, -1)
        if lines and top <= centre <= bottom:
            lines[-1].append(fragment)
        else:
            lines.append([fragment])
    ordered = [f for line in lines for f in sorted(line, key=lambda f: min(p[0] for p in f[0]))]
    return "".join(text for _, text, _ in ordered), min(float(conf) for _, _, conf in ordered)


def upscale_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Enlarge a crop shorter than OCR_MIN_HEIGHT_PX (at most OCR_MAX_UPSCALE times)."""
    height = crop.shape[0]
    if height >= OCR_MIN_HEIGHT_PX:
        return crop
    scale = min(OCR_MIN_HEIGHT_PX / height, OCR_MAX_UPSCALE)
    size = (int(round(crop.shape[1] * scale)), int(round(height * scale)))
    return cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)


class EasyOcr:
    name = "easyocr"

    def __init__(self, gpu: bool = False):
        import easyocr  # slow import; only loaded when plates are enabled

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        return join_fragments(self.reader.readtext(upscale_for_ocr(crop), allowlist=PLATE_CHARS))


class PlateReader:
    def __init__(self, cfg: PlateConfig, detector=None, ocr=None, device: str = "cpu"):
        if detector is None:
            from rsafe.detectors.plate import PlateDetector

            detector = PlateDetector(cfg.detector_model, device=device)
        self.cfg = cfg
        self.detector = detector
        self.ocr = ocr if ocr is not None else EasyOcr(gpu=device == "cuda")
        self.detector_name = detector.name
        self.ocr_name = self.ocr.name

    def read(self, frame: np.ndarray, moto: BBox) -> PlateResult:
        rx1, ry1, rx2, ry2 = (int(round(v)) for v in plate_search_region(frame.shape, moto))
        region = frame[ry1:ry2, rx1:rx2]
        plate = self.detector.detect(region)
        if plate is None:
            return PlateResult("not_visible")
        px1, py1, px2, py2 = (int(round(v)) for v in plate.bbox)
        crop = region[max(0, py1):py2, max(0, px1):px2].copy()
        if crop.size == 0:
            return PlateResult("not_visible")
        raw, confidence = self.ocr.read(crop)
        text = normalize_plate_text(raw)
        if not text or confidence < self.cfg.min_ocr_confidence:
            logger.debug("[R-SAFE] Plate unreadable: %r (%.2f)", raw, confidence)
            return PlateResult("unreadable", crop=crop)
        return PlateResult("read", text=text, ocr_confidence=round(confidence, 2), crop=crop)
