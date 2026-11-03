# R-SAFE AI Engine — Design Spec

- **Date:** 2026-11-03
- **Scope owner:** AI engine (this spec). Backend, database, and frontend are built by teammates and integrated later.
- **Deadline:** Friday, 2026-11-06
- **Status:** Approved in conversation; pending written-spec review

## 1. Goal

Turn the existing `ai-engine/` folder into a clean, modular Python engine that:

1. Reads video from a file or an Android IP Webcam stream.
2. Detects motorcycles and the people riding them (driver and back riders).
3. Classifies each rider as `helmet` or `no_helmet`.
4. Reads the motorcycle's license plate with OCR when visible.
5. For every rider without a helmet, saves evidence (images + a JSON record) so the
   teammates' backend can ingest it later without changes to the engine.

Every detection is a **potential violation requiring human review**. The engine never
marks a violation as confirmed.

### Out of scope (this week)

- Backend API, PostgreSQL schema, React dashboard (teammates).
- User login, roles, live video streaming to the dashboard.
- Training new models on Kaggle (VisDrone motorcycle training and a custom plate detector
  are optional follow-ups; the design lets them be swapped in via config).

## 2. Current state (as found on 2026-11-03)

- Helmet model trained and exported: `ai-engine/models/helmet/best.onnx` (classes
  `helmet`, `no_helmet`). Good dataset metrics; weak on small/distant objects; sometimes
  labels cars as helmets when run on the full frame.
- `ai-engine/detection/helmet_detector.py` wraps the ONNX model (preprocess, NMS, draw).
- Diagnostic scripts exist: `rsafe_input_visual_inspector.py`,
  `rsafe_raw_prediction_diagnostic.py`, `inspect_rsafe_onnx.py`, `inspect_visdrone_motor.py`,
  `live_detection_gui.py`, `live_helmet_detection.py`, `test_hotspot_camera.py`.
- VisDrone-Motorcycle dataset prepared (`assets/datasets/VisDrone-Motorcycle`, 8,629
  images, 34,863 boxes). Local CPU training was attempted (1 epoch ≈ 3 h) and abandoned.
  No motorcycle model exists yet.
- No plate detector or OCR exists.
- `association/rider_association.py` is empty. `config/`, `evidence/`, `processing/`,
  `tracking/` are empty folders.
- Not a git repository. Root `.venv` is a stray Python 3.14 env; the real env is
  `ai-engine/.venv` (Python 3.12.10). `requirements.txt` is UTF-16 encoded.

## 3. Model strategy (Approach A: pretrained-first, modular)

| Stage | Model | Notes |
|---|---|---|
| Motorcycle + person | Ultralytics YOLO11s pretrained on COCO, classes `person` (0) and `motorcycle` (3) only | Street-level viewpoint matches phone footage better than aerial VisDrone. No training needed. Built-in ByteTrack provides track IDs. |
| Helmet | Existing `models/helmet/best.onnx` | Run on an enlarged **head crop** of each rider, never on the full frame. Fixes the small-object and car-as-helmet problems. |
| Plate detection | Pretrained license-plate YOLO model (candidate chosen and verified Thursday) | Run on the motorcycle crop (expanded downward/backward). |
| Plate OCR | OCR library behind a `PlateReader` interface (candidate chosen Thursday; EasyOCR is the default fallback since `torch` is already installed) | Text normalised to uppercase alphanumerics. |

Each model lives behind its own class with one public method. Replacing a model (e.g. a
future VisDrone-trained motorcycle model) means changing a path in
`configs/default.yaml`, not engine code.

## 4. Architecture

### 4.1 Per-frame flow

```
VideoSource (file | ip_webcam | device index)
  → VehicleDetector.track(frame)        motorcycles + persons, with track IDs
  → associate_riders(motorcycles, persons)
  → HelmetClassifier.classify(head_crop) for each rider
  → ViolationTracker.update(...)        per-track voting; emits a violation once
  → PlateReader.read(motorcycle_crop)   best reading kept across the track's frames
  → EvidenceWriter.write(violation)     images + violation.json
  → Sink.send(record)                   FileSink now, HttpSink later
```

Frames are processed every `frame_stride` frames (config) because the target laptop has
no GPU.

### 4.2 Package layout

```
ai-engine/
  rsafe/
    __init__.py
    config.py               load + validate YAML into dataclasses; fail fast on bad paths
    types.py                BBox, Detection, Rider, TrackState, Violation dataclasses
    pipeline.py             wires the stages; no model logic of its own
    video/source.py         open file / URL / device; reconnect with backoff
    detectors/vehicle.py    VehicleDetector (Ultralytics YOLO + ByteTrack)
    detectors/helmet.py     HelmetClassifier (refactored from detection/helmet_detector.py)
    detectors/plate.py      PlateDetector
    ocr/plate_reader.py     PlateReader (detector + OCR + text normalisation)
    association/riders.py   associate_riders(), head_crop()
    violations/rules.py     ViolationTracker (voting, one report per track, best frame)
    evidence/writer.py      EvidenceWriter (folder layout, images, JSON)
    sinks/base.py           Sink protocol
    sinks/file_sink.py      FileSink (default)
    sinks/http_sink.py      HttpSink (POST JSON + files; on failure keep file and retry)
  configs/default.yaml
  models/helmet/best.onnx   (existing)
  models/vehicle/           yolo11s.pt (downloaded)
  models/plate/             plate detector weights
  scripts/run_pipeline.py   CLI entry point: --source, --config, --show
  scripts/evaluate.py       compare engine output with hand-labelled frames
  tools/                    existing diagnostic scripts, moved unchanged in behaviour
  training/                 prepare_motorcycle_dataset.py, inspect_visdrone_motor.py
  tests/                    pytest unit + smoke tests
  docs/violation_schema.md  integration contract for teammates
  requirements.txt          UTF-8, pinned
```

### 4.3 Cleanup

- `git init` with a `.gitignore` excluding `.venv/`, `runs/`, `output/`, `assets/datasets/`,
  `*.zip`, `__pycache__/`, `*.pt` (vehicle weights are downloaded, not committed).
- Move `detection/runs/`, `detection/yolo26n.pt`, `detection/visdrone_motor_samples/` out
  of the source tree (into `ai-engine/_archive/`, gitignored). Nothing is deleted without
  the owner seeing it first.
- Root `R-SAFE/.venv` (Python 3.14, pip only) is flagged for the owner to delete.
- Re-save `requirements.txt` as UTF-8 and add the new dependencies.
- Existing scripts move to `tools/` / `training/` and must still run.

## 5. Rules

### 5.1 Rider association
A person is a rider of a motorcycle if the bottom-centre point of the person box lies
inside the motorcycle box expanded by `association.expand_ratio` (default 0.15). A person
matching several motorcycles goes to the one with the largest box overlap. At most
`association.max_riders` (default 3) riders per motorcycle, ordered front to back
(`position` 1 = driver guess). **Back riders are checked and flagged too** (RA 10054
covers passengers).

### 5.2 Helmet check
Head crop = top `helmet.head_ratio` (default 0.40) of the rider box, padded by
`helmet.pad_ratio` (default 0.20), clipped to the frame, letterboxed to the model's input
size. Skip riders whose box is shorter than `helmet.min_rider_height_px` (default 48);
too small to judge.

### 5.3 Voting and reporting
Per (track ID, rider position), keep the last `rules.window` (default 10) helmet results
with confidence ≥ `rules.min_confidence` (default 0.50). A rider is `no_helmet` when
no-helmet votes ≥ `rules.min_votes` (default 5) **and** no-helmet votes > helmet votes.
A motorcycle track produces **at most one** violation record, listing every rider judged
`no_helmet`. Evidence frame = the frame where the violating rider's box was largest.
The record is emitted when the track ends (not seen for `rules.track_lost_frames`,
default 30) or at end of video, so the best frame and plate reading can be chosen.

### 5.4 Plate
Plate reading is attempted on up to `plate.max_attempts` (default 5) frames of the track;
the highest-confidence reading wins. `plate.status`: `read`, `unreadable` (plate found,
OCR failed or low confidence), `not_visible` (no plate found). A violation is saved
regardless of plate status.

## 6. Output contract

### 6.1 Folder layout
```
output/violations/<YYYY-MM-DD>/<violation_id>/
  violation.json
  frame.jpg     full frame, boxes drawn
  rider.jpg     enlarged crop of the violating rider(s)
  plate.jpg     only when a plate was detected
```
`violation_id` = `V-<YYYYMMDD>-<HHMMSS>-<track_id zero-padded to 4>`.

### 6.2 violation.json (schema_version 1.0)
```json
{
  "schema_version": "1.0",
  "violation_id": "V-20261106-143212-0007",
  "type": "no_helmet",
  "status": "pending_review",
  "detected_at": "2026-11-06T14:32:12+08:00",
  "source": { "kind": "ip_webcam", "uri": "http://192.168.1.5:8080/video", "camera_id": "cam-01" },
  "track_id": 7,
  "riders": [
    { "position": 1, "helmet": "no_helmet", "confidence": 0.81, "frames_observed": 9 },
    { "position": 2, "helmet": "helmet", "confidence": 0.77, "frames_observed": 8 }
  ],
  "motorcycle": { "confidence": 0.88, "bbox": [412, 220, 610, 540] },
  "plate": { "text": "123ABC", "ocr_confidence": 0.64, "status": "read" },
  "evidence": { "frame": "frame.jpg", "rider": "rider.jpg", "plate": "plate.jpg" },
  "models": { "vehicle": "yolo11s-coco", "helmet": "helmet-best.onnx", "plate": "<name>", "ocr": "<name>" }
}
```
- `status` is always `pending_review` from the engine.
- `riders` lists all riders on the motorcycle, so the reviewer sees context.
- `bbox` is `[x1, y1, x2, y2]` in pixels of `frame.jpg`.
- `plate.text` is `null` unless `status` is `read`; `evidence.plate` is `null` when no plate crop exists.
- `detected_at` uses Philippine time (+08:00).
- Breaking changes to this file require bumping `schema_version`.

### 6.3 Sinks
`FileSink` (default) only writes the folder. `HttpSink` writes the folder, then POSTs
`violation.json` and the images as multipart to `sink.http.url`; on failure the folder is
kept and listed in `output/pending_upload.txt` for retry. Selected by `sink.kind` in config.

## 7. Error handling

| Situation | Behaviour |
|---|---|
| Missing model file / invalid config | Exit at startup with a message naming the path or key |
| Video source unavailable or drops | Log, reconnect with exponential backoff (cap 30 s); files end normally |
| Exception while processing one frame | Log with frame number, skip frame, continue |
| OCR failure | `plate.status = "unreadable"`, violation still saved |
| HTTP sink failure | Keep evidence on disk, queue for retry |

Logging uses Python `logging` with an `[R-SAFE]` prefix; level set in config.

## 8. Testing

1. **Unit tests (pytest, no models needed):** association, head crop maths, voting rule,
   one-report-per-track, violation ID format, JSON serialisation against the schema,
   config validation.
2. **Model wrapper tests:** helmet classifier loads `best.onnx` and returns the expected
   structure on `tests/samples/*.jpg` (existing images).
3. **Smoke test:** run `scripts/run_pipeline.py` on a short clip; assert output folders
   contain valid `violation.json`.
4. **Real-world evaluation:** ~150 frames sampled from the owner's traffic footage,
   hand-labelled (motorcycles, riders, helmet / no-helmet). `scripts/evaluate.py` reports
   precision and recall for motorcycle detection and no-helmet flagging. These numbers,
   not dataset mAP, are reported in the project documentation.
5. Measured FPS on the owner's laptop is recorded in the README.

## 9. Known limitations

- Motorcycle plates are on the rear; cameras facing oncoming traffic will mostly yield
  `not_visible`. Recommend filming departing traffic.
- Night, rain, heavy motion blur, and heavy occlusion reduce accuracy.
- COCO-pretrained detector is not tuned for Philippine traffic (tricycles, habal-habal
  with many riders); evaluation will show how much this matters.
- CPU-only inference limits throughput; frame stride trades recall for speed.

## 10. Schedule

| Day | Deliverable | Done when |
|---|---|---|
| Tue 11-03 | Spec + plan approved; git init; cleanup; package skeleton; config; types | Tests pass; moved tools still run |
| Wed 11-04 | Vehicle detector + tracking; association; helmet on head crops; voting | Clip shows per-rider helmet labels with track IDs |
| Thu 11-05 | Plate detector + OCR; evidence writer; file + HTTP sinks; schema doc | End-to-end run produces violation folders |
| Fri 11-06 | Real-world evaluation; threshold tuning; README + handoff guide; demo recording | Ready to hand to teammates |

**Main risk:** no verified pretrained plate detector yet. Fallback: plate stage reports
`not_visible` / `unreadable`; all other stages unaffected; a trained model can be added
later through config.
