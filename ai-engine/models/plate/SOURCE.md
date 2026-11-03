# Plate detector source

- **Model:** `license-plate-finetune-v1s.onnx` (YOLO11s, 1 class `License_Plate`), saved here as `plate.onnx`
- **From:** https://huggingface.co/morsetechlab/yolov11-license-plate-detection
- **License:** AGPL-3.0 (same as Ultralytics). Original model card kept in `MODEL_CARD.md`.
- **Downloaded:** 2026-11-03
- **Why ONNX, not .pt:** a .pt file is a Python pickle and can run code when loaded; ONNX holds only weights.
- **Caution:** the publisher notes train/test overlap in its source dataset, so its published
  mAP is not trusted. R-SAFE measures it on RideSafe-400 (Task 9); results recorded below.

## R-SAFE measurement

(filled in by Task 9)
