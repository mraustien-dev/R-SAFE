r"""Measure the plate detector and OCR on frames made by sample_ridesafe_plates.py.

Detection hit: detected plate box overlaps the labelled box with IoU >= 0.3.
OCR exact match: normalised OCR text equals the label (labels with "." - partly
unreadable to the annotator - are left out of the OCR numbers).

Usage:
  .venv\Scripts\python scripts\measure_plates.py --eval-dir <plate_eval dir>
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsafe.config import load_config  # noqa: E402
from rsafe.detectors.plate import PlateDetector  # noqa: E402
from rsafe.ocr.plate_reader import EasyOcr, normalize_plate_text, plate_search_region  # noqa: E402

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=AI_ENGINE_DIR / "configs" / "default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config).plate
    detector = PlateDetector(cfg.detector_model)
    ocr = EasyOcr()

    rows = list(csv.DictReader(open(args.eval_dir / "plates.csv", encoding="utf-8")))
    hits = ocr_total = exact = shown = shown_exact = 0
    readings = []  # (confidence, correct) per OCR'd plate
    for row in rows:
        frame = cv2.imread(str(args.eval_dir / "images" / row["image"]))
        label_box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
        if row["mx1"]:
            region = plate_search_region(frame.shape, tuple(float(row[k]) for k in ("mx1", "my1", "mx2", "my2")))
        else:
            region = (0.0, 0.0, float(frame.shape[1]), float(frame.shape[0]))
        rx1, ry1, rx2, ry2 = (int(round(v)) for v in region)
        plate = detector.detect(frame[ry1:ry2, rx1:rx2])
        if plate is None:
            continue
        x1, y1, x2, y2 = plate.bbox
        found = (x1 + rx1, y1 + ry1, x2 + rx1, y2 + ry1)
        if iou(found, label_box) < 0.3:
            continue
        hits += 1
        if "." in row["lp_number"]:
            continue
        crop = frame[int(found[1]):int(found[3]), int(found[0]):int(found[2])]
        raw, confidence = ocr.read(crop)
        text = normalize_plate_text(raw)
        ocr_total += 1
        exact += text == row["lp_number"]
        readings.append((confidence if text else 0.0, text == row["lp_number"]))
        if text and confidence >= cfg.min_ocr_confidence:
            shown += 1
            shown_exact += text == row["lp_number"]

    total = len(rows)
    print(f"frames: {total}")
    print(f"detection recall (IoU>=0.3): {hits}/{total} = {hits / total:.1%}")
    if ocr_total:
        print(f"OCR exact match on detected plates: {exact}/{ocr_total} = {exact / ocr_total:.1%}")
    if shown:
        print(
            f"shown as 'read' (conf >= {cfg.min_ocr_confidence}): {shown}/{ocr_total}; "
            f"exact among shown: {shown_exact}/{shown} = {shown_exact / shown:.1%}"
        )
    print("min_ocr_confidence -> shown as 'read' / correct among shown")
    for cutoff in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        kept = [correct for confidence, correct in readings if confidence >= cutoff]
        print(f"  {cutoff:.1f}: {len(kept):3d} shown, {sum(kept)} correct")


if __name__ == "__main__":
    main()
