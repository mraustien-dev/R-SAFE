# R-SAFE Mobile (Android)

Phone app that detects riders without helmets **offline** and uploads each potential
violation automatically to the R-SAFE Supabase project, where it shows on the dashboard.

- React Native (Expo SDK 58, JavaScript) for screens, sign-in, queue and upload.
- `modules/rsafe-detector`: native Kotlin module, runs the same three ONNX models as the
  laptop engine (ONNX Runtime Android) and Google ML Kit for plate text, on camera frames.
- Server side: `../database/migrations/20261108_mobile_sync.sql` and
  `../backend/supabase/functions/device-api`.

## How it works

1. **Register the phone (once, online).** An admin adds a row to `cameras`, then runs
   `select create_device_provision_code('<camera id>');` and gives the 8-character code
   (valid 24 h) to the officer. The phone makes its own P-256 key; the private key never
   leaves the phone.
2. **Add an officer (once per phone, online).** Email + password of the dashboard account
   (`officer` or `admin` role) and a 4–6 digit PIN. The server links officer and phone.
3. **Sign in (offline OK).** Pick your name, enter your PIN. 5 wrong PINs → sign in online
   again. If the phone has not reached the server for 72 h, offline sign-in locks until it does.
4. **On duty.** Camera detection runs on the phone. A motorcycle is reported once, when it
   leaves the view, if a rider got at least 3 confident "no helmet" votes (same rules as the
   laptop engine).
5. **Automatic upload.** Each violation is numbered, signed with the phone's key at capture
   time (photo SHA-256 hashes included) and queued. It uploads immediately when online,
   otherwise as soon as the phone has internet (checked on every network change and every
   30 s). There is no upload, edit or delete button. Photos are deleted from the phone only
   after the server accepted them.

The server rejects unsigned/forged uploads, and **flags** (never drops) anything that does
not add up: changed photos, officer not enrolled on that phone, unknown duty session, missing
photo, time in the future. Missing violation numbers per phone appear in
`device_sequence_gaps`.

## Build and run on a phone

Requirements: Node 20+, JDK 17 (`C:\Users\agang\.jdks\jdk-17.0.21+9`), Android SDK
(`%LOCALAPPDATA%\Android\Sdk`), a phone with USB debugging on.

```bash
# models (not in git): 8-bit phone versions of the laptop engine's models
# vehicle source, once in ai-engine:  yolo export model=models/vehicle/yolo11s.pt format=onnx dynamic=True simplify=True opset=17
cd ../ai-engine && .venv/Scripts/python -I tools/export_mobile_models.py && cd ../mobile
# -> vehicle.onnx (480), helmet.onnx (320), plate.onnx (640) + models.json (hashes, saved in every violation)

npm install
export JAVA_HOME="/c/Users/agang/.jdks/jdk-17.0.21+9" ANDROID_HOME="$LOCALAPPDATA/Android/Sdk"
npx expo run:android      # builds, installs on the connected phone, starts Metro
```

Tests: `npm test` (signing, PIN, record format) and
`cd android && ./gradlew :rsafe-detector:testDebugUnitTest` (tracker, helmet queue, rules,
exposure). The EAS cloud build (`npx eas build -p android --profile preview`) runs these first;
a failing test fails the build.

## Detection pipeline (Pipeline.kt)

Two speeds, so live tracking never waits for helmet checks:
- **Tracker loop** (camera thread): 8-bit vehicle model at 480 on a 1080p frame, motion-predicting
  tracker (fast motorcycles keep their id), live boxes moved to "now" between detections.
- **Helmet worker**: freshest head crop per rider (cut from the full 1080p frame), votes, keeps
  the largest + sharpest no-helmet moment as the evidence photo.
- **Evidence worker**: reads the plate from the 3 largest views of the motorcycle, saves, hashes.
- **Exposure**: short shutter first (1/500 s), ISO rises before the shutter gets longer, so
  fast riders are not blurred. Falls back to auto exposure on phones without manual control.

## Known limits

- Uploads run while the app is open (it stays open on duty with the screen on). If the app
  is closed, queued violations upload the next time it opens.
- Speed depends on the phone (all CPU). The live frames/s are shown on the duty screen.
- ML Kit's confidence scale differs from EasyOCR: `minOcrConfidence` (0.85) in
  `Engine.kt` should be tuned on real plates.
