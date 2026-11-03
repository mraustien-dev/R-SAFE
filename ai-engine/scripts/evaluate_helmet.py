r"""Measure no-helmet violation detection on HELMET dataset clips (unseen by every model).

Runs the full engine (plates off) on each clip's 100 frames, then compares the
violations it writes with the annotated motorcycle tracks: a true violation is a
track whose riders include one without a helmet. Prints per-clip and per-split
precision / recall / F1.

Usage (from ai-engine\):
  .venv\Scripts\python scripts\evaluate_helmet.py --clips <part_7 clip folders> ^
      --annotations <annotation csv folder> --split <data_split.csv> --set validation
Tuning flags (--vehicle-conf, --helmet-conf, --min-votes) are for validation clips only.
"""

import argparse
import csv
import dataclasses
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsafe.config import load_config  # noqa: E402
from rsafe.detectors.helmet import HelmetClassifier  # noqa: E402
from rsafe.detectors.vehicle import VehicleDetector  # noqa: E402
from rsafe.evaluation.helmet_dataset import load_ground_truth, match_violations  # noqa: E402
from rsafe.evidence.writer import EvidenceWriter  # noqa: E402
from rsafe.pipeline import Pipeline  # noqa: E402
from rsafe.sinks.file_sink import FileSink  # noqa: E402
from rsafe.types import SourceInfo  # noqa: E402

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
CLIP_FPS = 10


class RecordingWriter(EvidenceWriter):
    """Writes evidence as usual and remembers each report's frame and motorcycle box."""

    def __init__(self, output_dir: Path, tz_offset_hours: int):
        super().__init__(output_dir, tz_offset_hours)
        self.predicted: list[dict] = []

    def write(self, report, source, models):
        self.predicted.append({"frame_index": report.frame_index, "motorcycle": {"bbox": list(report.motorcycle.bbox)}})
        return super().write(report, source, models)


def scores(counts: dict[str, int]) -> str:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f"tp {tp:3d}  fp {fp:3d}  fn {fn:3d}  precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--set", choices=["training", "validation", "test", "all"], default="test")
    parser.add_argument("--config", type=Path, default=AI_ENGINE_DIR / "configs" / "default.yaml")
    parser.add_argument("--out", type=Path, default=AI_ENGINE_DIR / "output" / "eval_helmet")
    parser.add_argument("--vehicle-conf", type=float)
    parser.add_argument("--helmet-conf", type=float)
    parser.add_argument("--min-votes", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = dataclasses.replace(cfg, plate=dataclasses.replace(cfg.plate, enabled=False))
    if args.vehicle_conf is not None:
        cfg = dataclasses.replace(cfg, vehicle=dataclasses.replace(cfg.vehicle, confidence=args.vehicle_conf))
    if args.helmet_conf is not None:
        cfg = dataclasses.replace(cfg, helmet=dataclasses.replace(cfg.helmet, confidence=args.helmet_conf))
    if args.min_votes is not None:
        cfg = dataclasses.replace(cfg, rules=dataclasses.replace(cfg.rules, min_votes=args.min_votes))

    with open(args.split, newline="", encoding="utf-8") as handle:
        split = {row["video_id"]: row["Set"] for row in csv.DictReader(handle)}
    clips = sorted(
        path for path in args.clips.iterdir()
        if path.is_dir() and (args.set == "all" or split.get(path.name) == args.set)
    )
    print(f"{len(clips)} clips ({args.set}); vehicle.confidence {cfg.vehicle.confidence}, "
          f"helmet.confidence {cfg.helmet.confidence}, rules.min_votes {cfg.rules.min_votes}")

    if args.out.exists():
        shutil.rmtree(args.out)
    helmet = HelmetClassifier(cfg.helmet.model, cfg.helmet.confidence)
    totals: dict[str, dict[str, int]] = {}
    frames_done, started = 0, time.perf_counter()
    for clip in clips:
        writer = RecordingWriter(args.out / clip.name, cfg.evidence.timezone_offset_hours)
        # a fresh detector per clip so tracker IDs do not carry over between clips
        pipeline = Pipeline(cfg, VehicleDetector(cfg.vehicle), helmet, None, writer, FileSink())
        source = SourceInfo(kind="file", uri=str(clip), camera_id="helmet-eval")
        clip_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for image in sorted(clip.glob("*.jpg"), key=lambda path: int(path.stem)):
            frame_id = int(image.stem)
            now = clip_start + timedelta(seconds=frame_id / CLIP_FPS)
            pipeline.process_frame(frame_id, cv2.imread(str(image)), source, now)
            frames_done += 1
        pipeline.finish(source)

        counts = match_violations(load_ground_truth(args.annotations / f"{clip.name}.csv"), writer.predicted)
        clip_set = split.get(clip.name, "?")
        bucket = totals.setdefault(clip_set, {"tp": 0, "fp": 0, "fn": 0})
        for key in bucket:
            bucket[key] += counts[key]
        print(f"{clip.name:28s} {clip_set:10s} {scores(counts)}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\nspeed: {frames_done / elapsed:.2f} frames/s on 1920x1080 ({frames_done} frames)")
    for clip_set, counts in sorted(totals.items()):
        print(f"TOTAL {clip_set:10s} {scores(counts)}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps({
        "set": args.set,
        "clips": [clip.name for clip in clips],
        "settings": {"vehicle.confidence": cfg.vehicle.confidence, "helmet.confidence": cfg.helmet.confidence,
                     "rules.min_votes": cfg.rules.min_votes},
        "totals": totals,
        "frames_per_second": round(frames_done / elapsed, 2),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
