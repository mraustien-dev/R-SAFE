# R-SAFE AI Engine

Road Safety Smart Automated Field Enforcement — the AI part. It watches video of a
road, finds motorcycles, checks every rider (driver **and** back riders, per RA 10054)
for a helmet, tries to read the plate, and saves evidence of each **potential**
no-helmet violation for a human to review.

> The engine never confirms a violation. Every record has `status: "pending_review"`.
> A reviewer on the dashboard decides.

```
video ─► motorcycles + people (YOLO11s COCO, ByteTrack) ─► riders per motorcycle
      ─► head crop ─► helmet / no_helmet (helmet model) ─► vote over frames
      ─► plate box (plate model) ─► OCR (EasyOCR) ─► evidence folder ─► backend
```

## 1. Setup (Windows PowerShell)

Python 3.12, run from the `ai-engine` folder:

```powershell
cd C:\Users\agang\Desktop\R-SAFE\ai-engine
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# EasyOCR installs a "headless" OpenCV that removes the video window; put the normal one back:
.venv\Scripts\python -m pip uninstall -y opencv-python-headless
.venv\Scripts\python -m pip install --force-reinstall --no-deps opencv-python==5.0.0.93
```

Models:

| File | What | Notes |
|---|---|---|
| `models/vehicle/yolo11s.pt` | motorcycles + people (COCO) | Downloaded automatically on first run (internet needed once). If it lands in the current folder, move it here. |
| `models/helmet/best.onnx` | helmet / no-helmet | Owner-trained model, run on enlarged head crops. |
| `models/plate/plate.onnx` | license plate box | morsetechlab YOLO11s, AGPL-3.0, see `models/plate/SOURCE.md`. |
| EasyOCR English model | plate text | Downloaded by EasyOCR on first use. |

Check everything works:

```powershell
.venv\Scripts\python -m pytest -q              # all tests (about 1 minute)
.venv\Scripts\python -m pytest -q -m "not model"   # fast tests only (no models are run)
```

## 2. Run

### The window app (easiest)

Double-click **`ai-engine\R-SAFE.bat`** (or run `.venv\Scripts\python scripts\rsafe_app.py`).
Wait for "Ready" (models load in about 20 s), then:

- **Choose video file**: pick any video on the laptop. Fill in *Video filmed at*
  first (e.g. `2026-11-04 08:00`) so records get the real filming time.
- **Connect camera**: the list shows the cameras found (built-in, USB).
- **Connect phone**: type the address shown by the Android **IP Webcam** app
  (e.g. `192.168.1.5:8080`; phone and laptop on the same Wi-Fi).
- **Rotate**: turns a sideways video/camera upright (can be changed while running).
- **Real-time playback** (video files): the video plays at normal speed and the engine
  analyses the newest frame it can keep up with. On a slow laptop it sees too few frames
  to flag riders (0 records on the 10 s demo clip at 0.8 frames/s), so use the normal
  mode for evidence; on a GPU laptop it analyses almost every frame.
- **Save analysed video**: writes the video with boxes to `outputideos\`, playing at
  normal speed (good for presentations).
- **Performance**: the level picked for this computer (see "Fits the computer" below).
- **Pause / Stop**, **Open evidence folder**, and a list of saved violations
  (double-click one to open its folder). Stopping or closing the window still saves
  the motorcycles that were being watched.

Cameras and phones are always analysed on the newest picture (older pictures are
skipped because the engine is slower than the camera). For short 10 fps clips use
`R-SAFE.bat --config configs\demo.yaml` (every frame, output in `output\demo`).

### Command line

```powershell
# a video file, with a live window (press q to stop); --start-time = when it was filmed
.venv\Scripts\python scripts\run_pipeline.py --source C:\path\to\clip.mp4 --show --start-time 2026-11-06T07:30:00

# phone camera with the IP Webcam app (phone and laptop on the same Wi-Fi)
.venv\Scripts\python scripts\run_pipeline.py --source http://192.168.1.5:8080/video --show

# laptop/USB camera 0, stop after 900 frames
.venv\Scripts\python scripts\run_pipeline.py --source 0 --max-frames 900
```

The window shows motorcycles in blue with their track number, riders in green
(helmet), red (no helmet) or grey (`?`, too small or not judged yet). At the end it
prints frames processed, speed, violations saved and the output folder. A dropped
phone stream reconnects by itself (waits 1, 2, 4 … up to 30 s).

Camera tips: film **departing** traffic (plates are on the back), from the roadside,
in daylight, with riders at least ~150 px tall in the picture.

### Fits the computer automatically

At start-up the engine uses an NVIDIA GPU if there is one, and on the first start it
runs a ~10 s speed test and picks a level (saved in `output\hardware_profile.json`,
measured again when the hardware changes):

| Level | Picked when | Every Nth frame | Motorcycle / helmet input size |
|---|---|---|---|
| low | below 3 frames/s (this laptop: Ryzen 3 3250U, 0.8 frames/s) | 3 | 640 / 640 |
| balanced | 3 to 15 frames/s | 2 | 640 / 640 |
| high | 15 frames/s or more (e.g. NVIDIA GPU) | 1 | 960 / 640 |

All levels keep the 640 helmet size: smaller sizes were faster but raised more false
alarms on HELMET (training clips: 8 → 11-12). The 960 motorcycle size of "high" has
**not been measured for accuracy yet**: run `evaluate_helmet.py --vehicle-imgsz 960` on the
GPU machine before relying on it. To use a GPU, install the CUDA builds of PyTorch
(CUDA 12.8+ for RTX 50 cards) and `onnxruntime-gpu`; nothing else changes.

## 3. Output (what teammates receive)

```
output/violations/<YYYY-MM-DD>/<violation_id>/
  violation.json   frame.jpg   rider.jpg   plate.jpg (when a plate was found)
```

The full field list, HTTP upload format and suggested database tables are in
**[docs/violation_schema.md](docs/violation_schema.md)** — that is the integration
contract. To send records to the backend instead of only saving them, set in
`configs/default.yaml`:

```yaml
sink:
  kind: http
  http: { url: "http://<backend>/api/violations", timeout_s: 10 }
```

Failed uploads are listed in `output/pending_upload.txt` and re-sent automatically the
next time the engine starts with `sink.kind: http`.

**Times in records:** for live cameras `detected_at` is the moment the frame was
processed. For video files it is `--start-time` plus the frame's position in the video;
without `--start-time` the processing start time is used, so always give it for
recorded evidence.

## 4. Settings (`configs/default.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `video.camera_id` | `cam-01` | Name written into every record. |
| `video.frame_stride` | `3` | Process every 3rd frame (speed vs. recall); set by the performance level unless `custom`. |
| `vehicle.confidence` | `0.35` | Minimum motorcycle/person detection score. |
| `association.max_riders` | `3` | Riders kept per motorcycle. |
| `helmet.confidence` | `0.25` | Minimum helmet-model score. |
| `helmet.min_rider_height_px` | `48` | Smaller riders are not judged. |
| `rules.window` | `10` | Last N confident helmet checks per rider. |
| `rules.min_votes` | `3` | No-helmet checks needed (and more than helmet checks) to flag. Tuned on HELMET validation clips. |
| `rules.track_lost_frames` | `30` | Frames without the motorcycle before its record is written. |
| `plate.enabled` | `true` | Plate box + OCR on. |
| `plate.min_ocr_confidence` | `0.90` | Below this the plate is `unreadable` (no text shown). Kept high because lower values mostly showed wrong text. |
| `helmet.imgsz` | `640` | Helmet model input size (set by the performance level). |
| `performance.device` | `auto` | `auto` (GPU if available), `cpu` or `cuda`. |
| `performance.level` | `auto` | `auto`, `low`, `balanced`, `high`, or `custom` (use `frame_stride` / `imgsz` exactly as written). |
| `evidence.output_dir` | `output/violations` | Where folders go. |
| `sink.kind` | `file` | `file` or `http`. |

**Swapping in a better model later:** put the new file in `models/...` and change the
path in `configs/default.yaml` (`vehicle.model`, `helmet.model` or
`plate.detector_model`). A helmet model must keep the classes `Helmet` / `No-Helmet`
and ONNX format; vehicle and plate models are any Ultralytics detection model.

## 5. Measured results

All measurements were made on data none of the models was trained on.

### Helmet violations — HELMET dataset (Myanmar, 1920×1080, 10 fps)

`scripts/evaluate_helmet.py`, one record per motorcycle compared with the annotated
motorcycle tracks (a true violation = at least one rider without a helmet). Settings
were tuned only on the 2 validation clips; the numbers below come from the other clips.

Settings: `vehicle.confidence 0.35`, `helmet.confidence 0.25`, `rules.min_votes 3`
(on validation, `min_votes` 5 → 3 raised caught violations from 2 to 3 of 7 with no false
alarms; lowering the two confidences changed nothing).

| Clips (HELMET `part_7`) | No-helmet motorcycles | Caught (tp) | False alarms (fp) | Missed (fn) | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **test** (7, official test split) | 36 | 19 | 4 | 17 | **0.83** | **0.53** | **0.64** |
| training split (19, unseen by our models, not used for tuning) | 95 | 38 | 8 | 57 | 0.83 | 0.40 | 0.54 |
| all 26 untuned clips | 131 | 57 | 12 | 74 | 0.83 | 0.44 | 0.57 |

Precision = of the motorcycles flagged, how many really had a rider without a helmet.
Recall = of the motorcycles with a rider without a helmet, how many were flagged.
Matching rule: a record matches a labelled motorcycle on its evidence frame when the
boxes overlap with IoU ≥ 0.3 or the bike box centre lies inside the labelled box (labelled
boxes include the riders). Per-clip numbers: `output/eval_helmet/<set>/summary.json`.

### Plates — RideSafe-400 (India dashcam, 2560×1440)

`scripts/measure_plates.py` on 200 sampled frames with labelled plate text:

| Measure | Result |
|---|---|
| Plate found (IoU ≥ 0.3) | 162 / 200 = **81 %** |
| OCR text exactly right (all detected plates) | 4 / 135 = 3 % |
| Plate text shown at cutoff 0.90 | 4, of which 3 correct |

These plates are small (median 58×34 px) and two-line, so OCR mostly fails; the
`plate.jpg` crop is always saved for the reviewer. Re-measure on R-SAFE's own footage.

### Speed (CPU only, this laptop)

| Run | Speed |
|---|---|
| RideSafe video 2560×1440, plates on, every 3rd frame | 1.4 processed frames/s (≈ 4 video frames/s covered) |
| HELMET clips 1920×1080, plates off, every frame | 0.61–0.76 frames/s |

## 6. Known limitations

- **Not live-speed on CPU.** A 30 fps camera is processed slower than real time; use
  recorded clips or raise `frame_stride`. A GPU would help most.
- **Misses more than it invents.** On HELMET the engine rarely flags helmeted riders
  but misses many no-helmet riders (see recall above). Reviewers see few false alarms;
  the count of violations is an undercount.
- **Plates:** facing oncoming traffic gives mostly `not_visible`; small or blurry plates
  are `unreadable`. Plate text, when shown, must still be checked by the reviewer.
- Night, rain, motion blur and crowded scenes reduce accuracy.
- The motorcycle detector is a general COCO model, not tuned for Philippine traffic
  (tricycles, habal-habal with many riders). No labelled Philippine dataset was found,
  so all accuracy numbers come from Myanmar and India footage.

## 7. Project layout

```
configs/default.yaml        all settings
rsafe/                      the engine (video, detectors, association, rules, OCR, evidence, sinks, pipeline)
R-SAFE.bat                  double-click: window app (scripts/rsafe_app.py, rsafe/app/)
scripts/run_pipeline.py     run the engine from the command line
scripts/evaluate_helmet.py  HELMET evaluation
scripts/sample_ridesafe_plates.py, scripts/measure_plates.py   plate measurement
docs/violation_schema.md    integration contract for the backend
tests/unit, tests/models, tests/smoke   pytest suites
tools/                      older camera-test and model-inspection tools
training/                   older dataset preparation scripts (VisDrone)
```

## 8. Data and model credits

- **HELMET dataset** (CC BY 4.0): H. Lin, J. D. Deng, D. Albers, F. W. Siebert,
  "Helmet Use Detection of Tracked Motorcycles using CNN-based Multi-task Learning,"
  *IEEE Access*, 2020, doi:10.1109/ACCESS.2020.3021357.
- **RideSafe-400** (CC BY-NC-SA, non-commercial): DashCop, arXiv:2503.00428.
- **YOLO11s** (Ultralytics, AGPL-3.0), COCO-pretrained.
- **Plate model**: morsetechlab/yolov11-license-plate-detection (AGPL-3.0).
- **EasyOCR** (Apache-2.0).

Datasets live in `assets/datasets/` (not in git).
