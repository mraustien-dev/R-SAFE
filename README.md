# R-SAFE

AI system that detects motorcycle riders without helmets and turns each potential
violation into an evidence record for human review. BiPSU BSCS capstone project.

## Parts

| Folder | What it is | Stack |
|---|---|---|
| [`ai-engine/`](ai-engine/README.md) | Laptop detection engine: video file, webcam or phone camera stream | Python, YOLO11s + ByteTrack, ONNX Runtime, EasyOCR |
| [`mobile/`](mobile/README.md) | Android app: detects offline on the phone and uploads automatically | React Native (Expo), Kotlin module, ONNX Runtime Android, ML Kit |
| `backend/supabase/functions/device-api/` | Edge Function the phone uploads to: checks signatures, photo hashes, officer and duty | Supabase Edge Functions (TypeScript) |
| `database/migrations/` | Tables, columns and policies the phone sync needs | PostgreSQL (Supabase) |
| `documentation/` | Design spec and implementation plan of the engine | Markdown |

## How a violation is processed

1. **Detect** motorcycles and persons in each frame, and track each motorcycle.
2. **Match riders** to motorcycles and crop each rider's head.
3. **Classify** the head crop as helmet / no helmet, and collect votes over several frames.
4. **Decide** when the motorcycle leaves the view: "no helmet" needs at least 3 votes,
   and they must outnumber the "helmet" votes.
5. **Read the plate** (a reading aid, never proof) and save the evidence photos.
6. **Phone only:** hash the photos, sign the record with the phone's key, queue it, and
   upload it automatically when there is internet.
7. **Server:** verify the record and save the case as `verified` or `flagged`.
8. **Human review:** every case starts as `pending_review`.

## Models

Model weights are not stored in git. `ai-engine/models/` lists their sources, and
`ai-engine/tools/export_mobile_models.py` creates the 8-bit phone models in
`mobile/modules/rsafe-detector/android/src/main/assets/models/` (their hashes are in
`models.json` there).
