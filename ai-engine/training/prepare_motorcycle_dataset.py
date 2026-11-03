from pathlib import Path
import shutil

# Original converted VisDrone dataset
SOURCE = Path(
    r"C:\Users\agang\Desktop\R-SAFE\assets\datasets\VisDrone"
)

# New motorcycle-only dataset
OUTPUT = Path(
    r"C:\Users\agang\Desktop\R-SAFE\assets\datasets\VisDrone-Motorcycle"
)

MOTOR_CLASS_ID = 9
NEW_CLASS_ID = 0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

splits = ["train", "val", "test"]

total_images = 0
total_motorcycle_boxes = 0

for split in splits:
    source_images = SOURCE / "images" / split
    source_labels = SOURCE / "labels" / split

    output_images = OUTPUT / "images" / split
    output_labels = OUTPUT / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    if not source_images.exists():
        print(f"Skipping missing image split: {split}")
        continue

    image_count = 0
    motorcycle_count = 0

    for image_file in source_images.iterdir():
        if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_file = source_labels / f"{image_file.stem}.txt"

        filtered_labels = []

        if label_file.exists():
            for line in label_file.read_text(
                encoding="utf-8"
            ).splitlines():

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))

                if class_id == MOTOR_CLASS_ID:
                    parts[0] = str(NEW_CLASS_ID)
                    filtered_labels.append(" ".join(parts))
                    motorcycle_count += 1

        # Copy image
        shutil.copy2(
            image_file,
            output_images / image_file.name
        )

        # Write motorcycle-only label file.
        # Empty files are retained for negative images.
        output_label_file = output_labels / f"{image_file.stem}.txt"

        output_label_file.write_text(
            "\n".join(filtered_labels),
            encoding="utf-8"
        )

        image_count += 1

    total_images += image_count
    total_motorcycle_boxes += motorcycle_count

    print(f"\nSplit: {split}")
    print(f"Images copied: {image_count}")
    print(f"Motorcycle boxes: {motorcycle_count}")

# Create custom dataset YAML
yaml_content = f"""# R-SAFE Motorcycle Detection Dataset

path: {OUTPUT.as_posix()}

train: images/train
val: images/val
test: images/test

names:
  0: motorcycle
"""

yaml_file = OUTPUT / "VisDrone-Motorcycle.yaml"

yaml_file.write_text(
    yaml_content,
    encoding="utf-8"
)

print("\n===================================")
print("DATASET PREPARATION COMPLETED")
print("===================================")
print(f"Dataset location: {OUTPUT}")
print(f"Total images: {total_images}")
print(f"Total motorcycle boxes: {total_motorcycle_boxes}")
print(f"Dataset YAML: {yaml_file}")