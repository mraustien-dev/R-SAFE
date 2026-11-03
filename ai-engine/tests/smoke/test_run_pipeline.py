import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest
import yaml

AI_ENGINE_DIR = Path(__file__).resolve().parents[2]
SAMPLE = AI_ENGINE_DIR / "tests" / "samples" / "rider_without_helmet.jpg"

pytestmark = pytest.mark.model


def test_cli_smoke(tmp_path):
    image = cv2.imread(str(SAMPLE))
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (image.shape[1], image.shape[0]))
    for _ in range(30):
        writer.write(image)
    writer.release()

    settings = yaml.safe_load((AI_ENGINE_DIR / "configs" / "default.yaml").read_text(encoding="utf-8"))
    settings["evidence"]["output_dir"] = str(tmp_path / "violations")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")

    run = subprocess.run(
        [sys.executable, str(AI_ENGINE_DIR / "scripts" / "run_pipeline.py"),
         "--source", str(video), "--config", str(config), "--max-frames", "30"],
        capture_output=True, text=True, timeout=600,
    )

    assert run.returncode == 0, run.stderr
    assert "frames processed : 10" in run.stdout
    for record in (tmp_path / "violations").rglob("violation.json"):
        assert json.loads(record.read_text(encoding="utf-8"))["status"] == "pending_review"


def test_cli_missing_source_file_exits_2(tmp_path):
    run = subprocess.run(
        [sys.executable, str(AI_ENGINE_DIR / "scripts" / "run_pipeline.py"), "--source", str(tmp_path / "nope.mp4")],
        capture_output=True, text=True, timeout=120,
    )

    assert run.returncode == 2 and "nope.mp4" in run.stderr
