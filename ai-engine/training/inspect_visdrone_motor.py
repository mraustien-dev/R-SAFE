from pathlib import Path
import cv2
import random

DATASET = Path(r"C:\Users\agang\Desktop\R-SAFE\assets\datasets\VisDrone")
IMAGE_DIR = DATASET / "images" / "train"
LABEL_DIR = DATASET / "labels" / "train"

MOTOR_CLASS_ID = 9
OUTPUT_DIR = Path("visdrone_motor_samples")
OUTPUT_DIR.mkdir(exist_ok=True)

samples = []

for label_file in LABEL_DIR.glob("*.txt"):
    image_file = IMAGE_DIR / f"{label_file.stem}.jpg"

    if not image_file.exists():
        continue

    image = cv2.imread(str(image_file))
    if image is None:
        continue

    height, width = image.shape[:2]

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts)

        if int(cls) != MOTOR_CLASS_ID:
            continue

        box_width = int(bw * width)
        box_height = int(bh * height)

        samples.append({
            "image": image_file,
            "label": label_file,
            "width": box_width,
            "height": box_height,
        })

print(f"Total motorcycle bounding boxes: {len(samples)}")

if not samples:
    print("No motorcycle boxes found. Check the dataset paths.")
    exit()

print("\nBounding box dimensions (pixels):")

for sample in samples[:20]:
    print(
        f"{sample['image'].name}: "
        f"{sample['width']} x {sample['height']} px"
    )

selected = random.sample(samples, min(20, len(samples)))

for index, sample in enumerate(selected, start=1):
    image = cv2.imread(str(sample["image"]))
    label_file = sample["label"]

    height, width = image.shape[:2]

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts)

        if int(cls) != MOTOR_CLASS_ID:
            continue

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            "MOTOR",
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_file = OUTPUT_DIR / f"motor_sample_{index}.jpg"
    cv2.imwrite(str(output_file), image)

print(f"\nSaved visual samples to: {OUTPUT_DIR.resolve()}")
