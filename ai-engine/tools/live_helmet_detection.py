
import cv2
import sys
import time
from pathlib import Path

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ENGINE_DIR))

from rsafe.detectors.helmet import HelmetDetector

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("rsafe").setLevel(logging.DEBUG)

CAMERA_URL = "http://10.242.108.138:8080/video"

WINDOW_NAME = "RSAFE - Live Helmet Detection"

# Performance settings
INFERENCE_INTERVAL = 3
INFERENCE_WIDTH = 1024
DISPLAY_WIDTH = 800

detector = HelmetDetector(
    confidence_threshold=0.35,
    iou_threshold=0.45,
    image_size=1024
)

cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Cannot connect to Android camera.")
    sys.exit(1)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)

frame_count = 0
cached_detections = []
start_time = time.time()
fps = 0

print("RSAFE optimized live detection started.")
print("Press Q to exit.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame unavailable.")
            break

        frame_count += 1

        # Run AI inference every third frame
        if frame_count % INFERENCE_INTERVAL == 0:

            height, width = frame.shape[:2]

            scale = INFERENCE_WIDTH / width
            resized_height = int(height * scale)

            inference_frame = cv2.resize(
                frame,
                (INFERENCE_WIDTH, resized_height)
            )

            detections = detector.detect(inference_frame)

            # Convert bounding boxes back to original coordinates
            cached_detections = []

            for detection in detections:
                x, y, w, h = detection["bbox"]

                detection["bbox"] = [
                    int(x / scale),
                    int(y / scale),
                    int(w / scale),
                    int(h / scale)
                ]

                cached_detections.append(detection)

        # Display latest camera frame with cached detections
        output = detector.draw_detections(
            frame,
            cached_detections
        )

        elapsed = time.time() - start_time

        if elapsed >= 1:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        helmet_count = sum(
            d["class_name"] == "Helmet"
            for d in cached_detections
        )

        no_helmet_count = sum(
            d["class_name"] == "No-Helmet"
            for d in cached_detections
        )

        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Helmet: {helmet_count} | No-Helmet: {no_helmet_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        display_height = int(
            output.shape[0] *
            DISPLAY_WIDTH / output.shape[1]
        )

        display = cv2.resize(
            output,
            (DISPLAY_WIDTH, display_height)
        )

        cv2.imshow(WINDOW_NAME, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("RSAFE detection stopped.")
