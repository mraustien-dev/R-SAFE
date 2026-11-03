# R-SAFE Violation Record — Integration Contract (schema 1.0)

This is what the AI engine hands to the backend for every **potential** helmet
violation. Build the backend and database against this document.

> Every record is a *potential* violation. The engine always sets
> `status: "pending_review"`. Only a human reviewer on the dashboard may confirm or
> reject it. The backend must never treat an engine record as proof of a violation.

## 1. One violation = one folder

```
ai-engine/output/violations/<YYYY-MM-DD>/<violation_id>/
  violation.json   the record (UTF-8, described below)
  frame.jpg        full frame; motorcycle box (blue) and rider boxes (red) drawn
  rider.jpg        crop of the violating rider (largest view seen)
  plate.jpg        plate crop (only when a plate was detected)
```

The date folder uses Philippine time (UTC+08:00).

One motorcycle produces **at most one** record, listing all its riders. A record is
written when the motorcycle leaves the camera view (about 1 s later) or when the video
ends.

## 2. `violation.json`

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
  "motorcycle": { "confidence": 0.88, "bbox": [412.0, 220.0, 610.0, 540.0] },
  "plate": { "text": "ABC1234", "ocr_confidence": 0.64, "status": "read" },
  "evidence": { "frame": "frame.jpg", "rider": "rider.jpg", "plate": "plate.jpg" },
  "models": { "vehicle": "yolo11s-coco", "helmet": "helmet-best.onnx", "plate": "plate-yolo11s.onnx", "ocr": "easyocr" }
}
```

| Field | Type | Values / meaning |
|---|---|---|
| `schema_version` | string | `"1.0"`. Changes only when a field is renamed, removed, or changes meaning. |
| `violation_id` | string | `V-<YYYYMMDD>-<HHMMSS>-<track_id, 4 digits>`. Unique per camera; combine with `source.camera_id` when several cameras run at once. |
| `type` | string | Always `"no_helmet"` in 1.0. |
| `status` | string | Always `"pending_review"` from the engine. Reviewer states (`confirmed`, `rejected`) belong to the backend. |
| `detected_at` | string | ISO 8601 with `+08:00`; when the evidence frame was captured. |
| `source.kind` | string | `"file"`, `"ip_webcam"` or `"device"`. |
| `source.uri` | string | Video file path, stream URL, or camera index. |
| `source.camera_id` | string | Name set in the engine config (e.g. `"cam-01"`). |
| `track_id` | integer | Tracker ID of the motorcycle within one engine run. Restarts each run. |
| `riders[]` | array | Every rider seen on the motorcycle, ordered front to back. At least one has `helmet: "no_helmet"`. |
| `riders[].position` | integer | 1 = front (driver guess), 2+ = back riders. |
| `riders[].helmet` | string | `"helmet"`, `"no_helmet"` or `"unknown"` (not enough confident checks). |
| `riders[].confidence` | number 0–1 | Average model confidence of the winning label (0 for unknown). |
| `riders[].frames_observed` | integer | Confident helmet checks used for the decision (max 10). |
| `motorcycle.confidence` | number 0–1 | Detector confidence in the evidence frame. |
| `motorcycle.bbox` | number[4] | `[x1, y1, x2, y2]` in pixels of `frame.jpg`. |
| `plate.status` | string | `"read"`, `"unreadable"` (plate found, text not reliable) or `"not_visible"` (no plate found). |
| `plate.text` | string or null | Uppercase letters/digits, no spaces. `null` unless status is `"read"`. **Must be verified by the reviewer.** |
| `plate.ocr_confidence` | number or null | OCR confidence; `null` unless status is `"read"`. |
| `evidence.frame` / `evidence.rider` | string | File names inside the folder; always present. |
| `evidence.plate` | string or null | `"plate.jpg"` or `null` when no plate crop exists. |
| `models` | object | Names of the models that produced the record, for audit. Values may be `null` when a stage is disabled. |

## 3. HTTP delivery (when `sink.kind: http`)

The engine sends one `POST` per violation to `sink.http.url` as
`multipart/form-data`:

| Part name | Content |
|---|---|
| `record` | `violation.json` text (`application/json`) |
| `frame` | `frame.jpg` |
| `rider` | `rider.jpg` |
| `plate` | `plate.jpg` (only when present) |

The backend should answer with any `2xx` status once stored. On a network error or
non-2xx answer, the engine keeps the folder on disk and appends its path to
`ai-engine/output/pending_upload.txt` for a later retry, so the backend should treat
`violation_id` + `camera_id` as an idempotency key (a retry may resend a record).

## 4. Suggested database mapping (for teammates)

One `violations` row per record (`violation_id`, `camera_id`, `detected_at`,
`plate_text`, `plate_status`, `status`, image paths), and one `violation_riders` row
per entry in `riders[]`. Keep the original `violation.json` for audit.
