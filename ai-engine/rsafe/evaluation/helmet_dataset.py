"""Ground truth from the HELMET dataset (Myanmar, CC BY 4.0) and matching against engine output.

Annotation CSV columns: track_id, frame_id (1-100), x, y, w, h, label. One track is one
motorcycle; its box covers the bike and its riders. Labels list each position:
D = driver, P0 = passenger in front of the driver, P1-P3 = passengers behind,
e.g. "DHelmetP1NoHelmet".
"""

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rsafe.types import BBox

POSITION_PATTERN = re.compile(r"(D|P\d)(NoHelmet|Helmet)")
MIN_IOU = 0.3


@dataclass(frozen=True)
class GroundTruthTrack:
    track_id: str
    violation: bool  # any rider without a helmet, by the track's most frequent label
    boxes: dict[int, BBox]  # frame_id -> (x1, y1, x2, y2)


def parse_label(label: str) -> dict[str, str]:
    return {
        position: "no_helmet" if state == "NoHelmet" else "helmet"
        for position, state in POSITION_PATTERN.findall(label)
    }


def load_ground_truth(csv_path: Path) -> list[GroundTruthTrack]:
    labels: dict[str, Counter] = {}
    boxes: dict[str, dict[int, BBox]] = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            track_id = row["track_id"]
            x, y, w, h = (float(row[key]) for key in ("x", "y", "w", "h"))
            boxes.setdefault(track_id, {})[int(row["frame_id"])] = (x, y, x + w, y + h)
            labels.setdefault(track_id, Counter())[row["label"]] += 1
    return [
        GroundTruthTrack(
            track_id,
            "no_helmet" in parse_label(labels[track_id].most_common(1)[0][0]).values(),
            boxes[track_id],
        )
        for track_id in boxes
    ]


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _same_motorcycle(predicted: BBox, truth: BBox) -> bool:
    """IoU >= 0.3, or the bike box centre lies inside the truth (bike + riders) box."""
    cx, cy = (predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2
    inside = truth[0] <= cx <= truth[2] and truth[1] <= cy <= truth[3]
    return inside or _iou(predicted, truth) >= MIN_IOU


def match_violations(truth: list[GroundTruthTrack], predicted: list[dict]) -> dict[str, int]:
    """Count tp / fp / fn. Each true violation track can be matched once; a prediction
    on a helmeted track, on no track, or repeating a matched track is a false alarm."""
    matched: set[str] = set()
    tp = fp = 0
    for record in predicted:
        frame, box = record["frame_index"], tuple(record["motorcycle"]["bbox"])
        candidates = [
            track
            for track in truth
            if track.violation
            and track.track_id not in matched
            and frame in track.boxes
            and _same_motorcycle(box, track.boxes[frame])
        ]
        if candidates:
            best = max(candidates, key=lambda track: _iou(box, track.boxes[frame]))
            matched.add(best.track_id)
            tp += 1
        else:
            fp += 1
    fn = sum(1 for track in truth if track.violation and track.track_id not in matched)
    return {"tp": tp, "fp": fp, "fn": fn}
