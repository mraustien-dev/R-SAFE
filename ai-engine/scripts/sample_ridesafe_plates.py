r"""Sample labelled plate frames from RideSafe-400 for the plate-reader measurement.

Picks plate tracks whose text is labelled (lp_number not "#"), one random frame per
track, and saves each frame as a JPG plus a CSV row with the plate box, the plate text
and the motorcycle box around the plate (empty when none). Videos are unzipped one at
a time and deleted after their frames are saved, so the 5 GB archive is never fully
extracted.

Usage:
  .venv\Scripts\python scripts\sample_ridesafe_plates.py --zip <videos.zip> --annotations <xml dir> --out <dir>
"""

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2


def parse_annotation(xml_path: Path):
    """Return (plates, motos): plates = [(track_id, frame, box, text)], motos = {frame: [box]}."""
    root = ET.parse(xml_path).getroot()
    plates, motos = [], {}
    for track in root.iter("track"):
        label = track.get("label")
        for box in track.iter("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame"))
            xyxy = tuple(float(box.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            if label == "motorcycle":
                motos.setdefault(frame, []).append(xyxy)
            elif label == "license_plate":
                text = ""
                for attribute in box.iter("attribute"):
                    if attribute.get("name") == "lp_number":
                        text = (attribute.text or "").strip()
                if text and text != "#":
                    plates.append((int(track.get("id")), frame, xyxy, text))
    return plates, motos


def moto_for_plate(plate, motos):
    """Smallest motorcycle box whose search region (10% sideways, 25% down) holds the plate centre."""
    cx, cy = (plate[0] + plate[2]) / 2, (plate[1] + plate[3]) / 2
    best = None
    for x1, y1, x2, y2 in motos:
        w, h = x2 - x1, y2 - y1
        if x1 - 0.1 * w <= cx <= x2 + 0.1 * w and y1 <= cy <= y2 + 0.25 * h:
            if best is None or w * h < (best[2] - best[0]) * (best[3] - best[1]):
                best = (x1, y1, x2, y2)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    archive = zipfile.ZipFile(args.zip)
    videos = {Path(name).stem: name for name in archive.namelist() if name.endswith(".mp4")}

    candidates = {}  # (video, track_id) -> [(frame, box, text, moto)]
    for stem, member in sorted(videos.items()):
        xml_path = args.annotations / f"{stem}.xml"
        if not xml_path.is_file():
            continue
        plates, motos = parse_annotation(xml_path)
        for track_id, frame, box, text in plates:
            moto = moto_for_plate(box, motos.get(frame, []))
            candidates.setdefault((stem, track_id), []).append((frame, box, text, moto))

    rng = random.Random(args.seed)
    keys = sorted(candidates)
    chosen = rng.sample(keys, min(args.count, len(keys)))
    picks = {}
    for stem, track_id in chosen:
        picks.setdefault(stem, []).append(rng.choice(candidates[(stem, track_id)]))
    print(f"{len(keys)} labelled plate tracks in {len(videos)} videos; sampling {len(chosen)}")

    images_dir = args.out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.out / "_video_tmp"
    rows = []
    for stem, items in sorted(picks.items()):
        video_path = Path(archive.extract(videos[stem], temp_dir))
        capture = cv2.VideoCapture(str(video_path))
        for frame_index, box, text, moto in sorted(items, key=lambda item: item[0]):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                print(f"skip {stem} frame {frame_index}: cannot read")
                continue
            name = f"{stem}_{frame_index:05d}.jpg"
            cv2.imwrite(str(images_dir / name), frame)
            rows.append([name, *(f"{v:.1f}" for v in box), text, *([f"{v:.1f}" for v in moto] if moto else ["", "", "", ""])])
        capture.release()
        shutil.rmtree(temp_dir)

    with open(args.out / "plates.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "x1", "y1", "x2", "y2", "lp_number", "mx1", "my1", "mx2", "my2"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} frames to {args.out}")


if __name__ == "__main__":
    main()
