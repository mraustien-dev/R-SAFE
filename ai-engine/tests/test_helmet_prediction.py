
import sys
from pathlib import Path

import cv2

# Add the ai-engine directory to Python's import path
AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ENGINE_DIR))

from detection.helmet_detector import HelmetDetector


BASE_DIR = Path(__file__).resolve().parent

SAMPLE_DIR = BASE_DIR / "samples"
OUTPUT_DIR = BASE_DIR / "prediction_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    detector = HelmetDetector()

    if not SAMPLE_DIR.exists():
        print(f"Sample folder not found: {SAMPLE_DIR}")
        return

    image_files = [
        path for path in SAMPLE_DIR.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not image_files:
        print("No sample images found.")
        return

    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")

        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not read image.")
            continue

        detections = detector.detect(image)

        if not detections:
            print("No detections found.")

        for detection in detections:
            print(
                f'{detection["class_name"]}: '
                f'{detection["confidence"]:.3f}'
            )

        annotated = detector.draw_detections(
            image,
            detections
        )

        output_path = OUTPUT_DIR / f"result_{image_path.name}"

        cv2.imwrite(str(output_path), annotated)

        print(f"Saved: {output_path}")

    print("\nPrediction testing completed.")


if __name__ == "__main__":
    main()
