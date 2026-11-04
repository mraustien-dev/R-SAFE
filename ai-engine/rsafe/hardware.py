"""Fit the engine to the computer it runs on.

1. Device: an NVIDIA GPU is used when one is available (performance.device: auto).
2. Level: on the first start a ~10 s speed test picks low / balanced / high
   (performance.level: auto). The result is saved and reused until the hardware changes.

Every level keeps the accuracy settings that were measured on HELMET (640 / 640);
faster machines analyse more frames, and "high" also looks at the video in more detail.
"""

import dataclasses
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rsafe.config import Config

logger = logging.getLogger("rsafe.hardware")

DEVICES = ("auto", "cpu", "cuda")
# frame_stride, vehicle input size, helmet input size
LEVELS = {
    "low": (3, 640, 640),       # e.g. 2-core laptop CPU, no GPU (~1 frame/s)
    "balanced": (2, 640, 640),  # fast CPU
    "high": (1, 960, 640),      # NVIDIA GPU: every frame, more detail for far riders
}
HIGH_FPS, BALANCED_FPS = 15.0, 3.0
SAMPLE = Path(__file__).resolve().parents[1] / "tests" / "samples" / "rider_without_helmet.jpg"


@dataclass(frozen=True)
class PerformanceInfo:
    device: str
    hardware: str
    level: str
    measured_fps: float | None  # None when the level was chosen in the config

    def describe(self) -> str:
        where = "GPU" if self.device == "cuda" else "CPU only"
        speed = f", measured {self.measured_fps:.1f} frames/s" if self.measured_fps is not None else ""
        return f"{where}: {self.hardware} | level {self.level}{speed}"


def resolve_device(requested: str, cuda_available: bool) -> str:
    if requested not in DEVICES:
        raise ValueError(f"performance.device must be one of {DEVICES}, not {requested!r}")
    if requested == "cuda" and not cuda_available:
        raise ValueError("performance.device is 'cuda' but No NVIDIA GPU is usable "
                         "(or PyTorch was installed without CUDA). Use 'auto' or 'cpu'.")
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested


def onnx_providers(device: str, available: list[str]) -> list[str]:
    """ONNX Runtime providers for the helmet model (GPU only if onnxruntime-gpu is installed)."""
    if device == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        logger.warning("[R-SAFE] GPU found but onnxruntime-gpu is not installed; helmet model stays on CPU")
    return ["CPUExecutionProvider"]


def choose_level(fps: float) -> str:
    if fps >= HIGH_FPS:
        return "high"
    if fps >= BALANCED_FPS:
        return "balanced"
    return "low"


def apply_level(cfg: Config, level: str) -> Config:
    if level == "custom":
        return cfg
    if level not in LEVELS:
        raise ValueError(f"Unknown performance level {level!r} (expected {list(LEVELS)} or 'custom')")
    stride, vehicle_size, helmet_size = LEVELS[level]
    return dataclasses.replace(
        cfg,
        video=dataclasses.replace(cfg.video, frame_stride=stride),
        vehicle=dataclasses.replace(cfg.vehicle, imgsz=vehicle_size),
        helmet=dataclasses.replace(cfg.helmet, imgsz=helmet_size),
    )


def tune_for_hardware(
    level: str, device: str, hardware: str, cache: Path, benchmark: Callable[[], float]
) -> PerformanceInfo:
    """The level to use: the configured one, or (auto) measured once per hardware."""
    if level != "auto":
        return PerformanceInfo(device, hardware, level, None)
    signature = f"{device} | {hardware}"
    try:
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["hardware"] == signature:
            return PerformanceInfo(device, hardware, saved["level"], saved["fps"])
    except (OSError, ValueError, KeyError, TypeError):
        pass  # no (usable) saved result: measure
    logger.info("[R-SAFE] Measuring this computer's speed (first start only, about 10 s)...")
    fps = round(benchmark(), 2)
    info = PerformanceInfo(device, hardware, choose_level(fps), fps)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"hardware": signature, "fps": fps, "level": info.level}, indent=2),
                         encoding="utf-8")
    except OSError:
        logger.warning("[R-SAFE] Could not save the speed test result to %s", cache)
    return info


def measure_fps(vehicle, helmet, runs: int = 3) -> float:
    """Frames per second for one typical frame: motorcycle search + two helmet checks."""
    import cv2

    sample = cv2.imread(str(SAMPLE))
    frame = np.full((1080, 1920, 3), 114, np.uint8)
    frame[: sample.shape[0], : sample.shape[1]] = sample
    head = sample[: sample.shape[0] // 2]
    vehicle.track(frame)  # warm-up (first call is always slow)
    helmet.classify(head)
    started = time.perf_counter()
    for _ in range(runs):
        vehicle.track(frame)
        helmet.classify(head)
        helmet.classify(head)
    return runs / (time.perf_counter() - started)


def hardware_name(device: str) -> str:
    if device == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        import platform

        return platform.processor() or "unknown CPU"
