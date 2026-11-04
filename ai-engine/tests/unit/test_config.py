from pathlib import Path

import pytest
import yaml

from rsafe.config import ConfigError, load_config

AI_ENGINE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = AI_ENGINE_DIR / "configs" / "default.yaml"


def write_config(tmp_path, **overrides):
    """Copy default.yaml with section-level overrides, e.g. helmet={"model": "x"}."""
    data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for section, values in overrides.items():
        data[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_default_config_loads_spec_defaults():
    cfg = load_config(DEFAULT_CONFIG)

    assert cfg.rules.window == 10 and cfg.rules.min_votes == 3  # tuned (Task 11)
    assert cfg.rules.min_confidence == 0.50 and cfg.rules.track_lost_frames == 30
    assert cfg.helmet.head_ratio == 0.40 and cfg.helmet.pad_ratio == 0.20
    assert cfg.helmet.min_rider_height_px == 48 and cfg.helmet.imgsz == 640
    assert cfg.association.expand_ratio == 0.15 and cfg.association.max_riders == 3
    assert cfg.plate.max_attempts == 5
    assert cfg.video.frame_stride == 3
    assert cfg.evidence.timezone_offset_hours == 8
    assert cfg.helmet.model.is_absolute() and cfg.helmet.model.exists()
    assert cfg.evidence.output_dir.is_absolute()
    assert cfg.performance.device == "auto" and cfg.performance.level == "auto"


def test_missing_helmet_model_names_path(tmp_path):
    path = write_config(tmp_path, helmet={"model": "models/nope.onnx"})

    with pytest.raises(ConfigError, match="nope.onnx"):
        load_config(path)


def test_unknown_sink_kind_rejected(tmp_path):
    path = write_config(tmp_path, sink={"kind": "ftp"})

    with pytest.raises(ConfigError, match="sink.kind"):
        load_config(path)


def test_http_sink_requires_url(tmp_path):
    path = write_config(tmp_path, sink={"kind": "http", "http": {"url": "", "timeout_s": 10}})

    with pytest.raises(ConfigError, match="sink.http.url"):
        load_config(path)


def test_plate_disabled_skips_plate_model_check(tmp_path):
    path = write_config(
        tmp_path, plate={"enabled": False, "detector_model": "models/plate/missing.pt"}
    )

    cfg = load_config(path)

    assert cfg.plate.enabled is False


def test_plate_enabled_requires_plate_model(tmp_path):
    path = write_config(
        tmp_path, plate={"enabled": True, "detector_model": "models/plate/missing.pt"}
    )

    with pytest.raises(ConfigError, match="missing.pt"):
        load_config(path)


def test_missing_vehicle_weights_fall_back_to_download_name(tmp_path):
    path = write_config(tmp_path, vehicle={"model": "models/vehicle/absent/yolo11s.pt"})

    cfg = load_config(path)

    assert cfg.vehicle.model == "yolo11s.pt"


def test_missing_section_named_in_error(tmp_path):
    data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    del data["rules"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="rules"):
        load_config(path)
