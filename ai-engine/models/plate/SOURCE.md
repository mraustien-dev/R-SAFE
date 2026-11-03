# Plate detector source

- **Model:** `license-plate-finetune-v1s.onnx` (YOLO11s, 1 class `License_Plate`), saved here as `plate.onnx`
- **From:** https://huggingface.co/morsetechlab/yolov11-license-plate-detection
- **License:** AGPL-3.0 (same as Ultralytics). Original model card kept in `MODEL_CARD.md`.
- **Downloaded:** 2026-11-03
- **Why ONNX, not .pt:** a .pt file is a Python pickle and can run code when loaded; ONNX holds only weights.
- **Caution:** the publisher notes train/test overlap in its source dataset, so its published
  mAP is not trusted. R-SAFE measures it on RideSafe-400 (Task 9); results recorded below.

## R-SAFE measurement

Measured 2026-11-03 with `scripts/measure_plates.py` on 200 frames sampled by
`scripts/sample_ridesafe_plates.py` (seed 0) from RideSafe-400 videoset 4 part 2: one
random frame from each of 200 randomly chosen plate tracks whose text is labelled.
Indian dashcam footage, 2560x1440; plates are small (median 58x34 px) and mostly two-line.
The detector runs on the motorcycle search region, as in the engine.

| Measure | Result |
|---|---|
| Plate detection recall (IoU >= 0.3) | **162 / 200 = 81.0 %** |
| OCR exact match on detected plates (labels without ".") | 4 / 135 = 3.0 % |

OCR confidence cutoff (`plate.min_ocr_confidence`) vs. plate text shown as `read`:

| Cutoff | Shown as `read` | Correct |
|---|---|---|
| 0.30 | 42 | 4 |
| 0.50 | 23 | 3 |
| 0.70 | 10 | 3 |
| 0.90 | 4 | 3 |

**Decision:** plates enabled (recall >= 50 %), cutoff set to **0.90** so the reviewer
rarely sees a wrong plate number; every other detected plate is `unreadable` but its
`plate.jpg` crop is still saved for the reviewer to read. The cutoff was chosen on these
same 200 frames (4 readings above it), so it is a cautious setting, not a validated
accuracy figure. OCR should be re-measured on R-SAFE's own roadside footage, where
plates are expected to be larger.

