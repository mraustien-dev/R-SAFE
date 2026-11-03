from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "helmet"
    / "best.onnx"
)
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.20

def rotate_frame(frame, rotation):
    if rotation == "90° Clockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "90° Counterclockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180°":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def preprocess(frame):
    h, w = frame.shape[:2]
    scale = min(IMAGE_SIZE / w, IMAGE_SIZE / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh))

    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8)
    pad_x = (IMAGE_SIZE - nw) // 2
    pad_y = (IMAGE_SIZE - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    return np.ascontiguousarray(tensor), {
        "scale": scale, "pad_x": pad_x, "pad_y": pad_y,
        "width": w, "height": h
    }

def main():
    print("=" * 60)
    print("R-SAFE RAW ONNX PREDICTION DIAGNOSTIC")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Choose your traffic video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                   ("All files", "*.*")]
    )
    root.destroy()

    if not video_path:
        print("No video selected.")
        return

    rotation = input(
        "Rotation (0, CW, CCW, 180) [CCW]: "
    ).strip().upper()
    if not rotation:
        rotation = "CCW"

    rotation_map = {
        "0": "0°",
        "CW": "90° Clockwise",
        "CCW": "90° Counterclockwise",
        "180": "180°"
    }
    rotation = rotation_map.get(rotation, "90° Counterclockwise")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {Path(video_path).name}")
    print(f"Frames: {total_frames}, FPS: {fps:.2f}")
    print(f"Rotation: {rotation}")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    # Probe a few positions rather than processing the full hour.
    positions = [0.10, 0.25, 0.50, 0.75, 0.90]
    for fraction in positions:
        frame_index = min(max(int(total_frames * fraction), 0), max(total_frames - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            print(f"\nCould not read frame {frame_index}")
            continue

        frame = rotate_frame(frame, rotation)
        tensor, meta = preprocess(frame)
        raw = session.run(None, {input_name: tensor})[0]

        # Metadata states output is [1, 6, anchors].
        predictions = np.asarray(raw)[0].T
        scores = predictions[:, 4:6]
        helmet_scores = scores[:, 0]
        nohelmet_scores = scores[:, 1]
        best_scores = np.max(scores, axis=1)
        best_classes = np.argmax(scores, axis=1)

        print("\n" + "-" * 60)
        print(f"FRAME {frame_index} ({fraction:.0%} of video)")
        print(f"Raw output shape: {np.asarray(raw).shape}")
        print(f"Prediction rows: {predictions.shape}")
        print(f"Maximum Helmet score: {helmet_scores.max():.6f}")
        print(f"Maximum No-Helmet score: {nohelmet_scores.max():.6f}")
        print(f"Candidate count >= {CONFIDENCE_THRESHOLD}: "
              f"{int(np.count_nonzero(best_scores >= CONFIDENCE_THRESHOLD))}")

        top_indices = np.argsort(best_scores)[-10:][::-1]
        print("Top 10 raw candidates (class, score, cx, cy, w, h):")
        for idx in top_indices:
            p = predictions[idx]
            class_id = int(best_classes[idx])
            class_name = "Helmet" if class_id == 0 else "No-Helmet"
            print(
                f"  {class_name:9s} {best_scores[idx]:.5f} | "
                f"{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}, {p[3]:.1f}"
            )

    cap.release()
    print("\nDiagnostic completed. Share the full terminal output.")

if __name__ == "__main__":
    main()
