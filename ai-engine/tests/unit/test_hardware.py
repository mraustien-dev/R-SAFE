import json
from pathlib import Path

import pytest

from rsafe.config import load_config
from rsafe.hardware import (
    LEVELS,
    apply_level,
    choose_level,
    onnx_providers,
    resolve_device,
    tune_for_hardware,
)

AI_ENGINE_DIR = Path(__file__).resolve().parents[2]
CFG = load_config(AI_ENGINE_DIR / "configs" / "default.yaml")
CUDA_AND_CPU = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_device():
    assert resolve_device("auto", cuda_available=True) == "cuda"
    assert resolve_device("auto", cuda_available=False) == "cpu"
    assert resolve_device("cpu", cuda_available=True) == "cpu"
    assert resolve_device("cuda", cuda_available=True) == "cuda"


def test_forced_cuda_without_gpu_explains():
    with pytest.raises(ValueError, match="No NVIDIA GPU"):
        resolve_device("cuda", cuda_available=False)


def test_unknown_device_rejected():
    with pytest.raises(ValueError, match="gpu"):
        resolve_device("gpu", cuda_available=True)


def test_onnx_providers():
    assert onnx_providers("cuda", CUDA_AND_CPU) == CUDA_AND_CPU
    assert onnx_providers("cuda", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]  # GPU package missing
    assert onnx_providers("cpu", CUDA_AND_CPU) == ["CPUExecutionProvider"]


@pytest.mark.parametrize(("fps", "level"), [(40, "high"), (15, "high"), (14.9, "balanced"), (3, "balanced"),
                                            (2.9, "low"), (0.7, "low")])
def test_choose_level(fps, level):
    assert choose_level(fps) == level


def test_levels_keep_accuracy_settings_on_slow_machines():
    low = apply_level(CFG, "low")
    assert (low.video.frame_stride, low.vehicle.imgsz, low.helmet.imgsz) == (3, 640, 640)
    balanced = apply_level(CFG, "balanced")
    assert (balanced.video.frame_stride, balanced.vehicle.imgsz, balanced.helmet.imgsz) == (2, 640, 640)
    high = apply_level(CFG, "high")
    assert (high.video.frame_stride, high.vehicle.imgsz, high.helmet.imgsz) == (1, 960, 640)


def test_custom_level_keeps_config_and_bad_level_rejected():
    assert apply_level(CFG, "custom") is CFG
    with pytest.raises(ValueError, match="turbo"):
        apply_level(CFG, "turbo")


class Counter:
    def __init__(self, fps):
        self.fps, self.calls = fps, 0

    def __call__(self):
        self.calls += 1
        return self.fps


def test_auto_level_benchmarks_once_and_reuses_result(tmp_path):
    cache = tmp_path / "hardware_profile.json"
    bench = Counter(0.8)

    first = tune_for_hardware("auto", "cpu", "Ryzen 3", cache, bench)
    second = tune_for_hardware("auto", "cpu", "Ryzen 3", cache, bench)

    assert first.level == "low" and first.measured_fps == 0.8 and first.device == "cpu"
    assert second.level == "low" and bench.calls == 1
    assert json.loads(cache.read_text(encoding="utf-8"))["hardware"] == "cpu | Ryzen 3"


def test_new_hardware_is_measured_again(tmp_path):
    cache = tmp_path / "hardware_profile.json"
    tune_for_hardware("auto", "cpu", "Ryzen 3", cache, Counter(0.8))

    upgraded = tune_for_hardware("auto", "cuda", "RTX 5060", cache, Counter(30.0))

    assert upgraded.level == "high" and upgraded.measured_fps == 30.0


def test_chosen_level_skips_benchmark(tmp_path):
    bench = Counter(30.0)

    info = tune_for_hardware("balanced", "cpu", "Ryzen 3", tmp_path / "p.json", bench)

    assert info.level == "balanced" and info.measured_fps is None and bench.calls == 0


def test_broken_cache_is_ignored(tmp_path):
    cache = tmp_path / "hardware_profile.json"
    cache.write_text("not json", encoding="utf-8")

    assert tune_for_hardware("auto", "cpu", "Ryzen 3", cache, Counter(5.0)).level == "balanced"


def test_level_names_listed_for_the_app():
    assert list(LEVELS) == ["low", "balanced", "high"]
