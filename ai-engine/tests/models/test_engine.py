import dataclasses
import json
from pathlib import Path

import pytest

from rsafe.config import load_config
from rsafe.hardware import LEVELS
from rsafe.pipeline import Engine

AI_ENGINE_DIR = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.model


def test_engine_measures_this_computer_and_applies_the_level(tmp_path):
    cfg = load_config(AI_ENGINE_DIR / "configs" / "default.yaml")
    cfg = dataclasses.replace(
        cfg,
        plate=dataclasses.replace(cfg.plate, enabled=False),
        evidence=dataclasses.replace(cfg.evidence, output_dir=tmp_path / "violations"),
    )

    engine = Engine(cfg)

    info = engine.performance
    assert info.device == "cpu" and info.level in LEVELS and info.measured_fps > 0
    assert json.loads((tmp_path / "hardware_profile.json").read_text(encoding="utf-8"))["level"] == info.level
    assert engine.cfg.video.frame_stride == LEVELS[info.level][0]

    engine.set_level("high")
    assert engine.cfg.video.frame_stride == 1 and engine.cfg.vehicle.imgsz == 960
    assert engine.new_pipeline().vehicle.cfg.imgsz == 960
