"""Load and validate the engine's YAML settings.

Every problem is reported at startup as a ConfigError naming the bad key or path,
so the engine never fails halfway through a video because of a typo.
"""

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

logger = logging.getLogger("rsafe.config")

AI_ENGINE_DIR = Path(__file__).resolve().parents[1]
PERFORMANCE_DEVICES = ("auto", "cpu", "cuda")
PERFORMANCE_LEVELS = ("auto", "low", "balanced", "high", "custom")
SINK_KINDS = {"file", "http"}


class ConfigError(Exception):
    """Raised when the configuration file is missing, incomplete, or invalid."""


@dataclass(frozen=True)
class VideoConfig:
    source: str
    camera_id: str
    frame_stride: int


@dataclass(frozen=True)
class VehicleConfig:
    model: str  # absolute path, or a bare Ultralytics name to download
    confidence: float
    imgsz: int
    tracker: str


@dataclass(frozen=True)
class AssociationConfig:
    expand_ratio: float
    max_riders: int


@dataclass(frozen=True)
class HelmetConfig:
    model: Path
    confidence: float
    head_ratio: float
    pad_ratio: float
    min_rider_height_px: int
    imgsz: int  # helmet model input size (smaller = faster)


@dataclass(frozen=True)
class RulesConfig:
    window: int
    min_confidence: float
    min_votes: int
    track_lost_frames: int


@dataclass(frozen=True)
class PlateConfig:
    enabled: bool
    detector_model: Path
    ocr: str
    max_attempts: int
    min_ocr_confidence: float


@dataclass(frozen=True)
class EvidenceConfig:
    output_dir: Path
    timezone_offset_hours: int


@dataclass(frozen=True)
class HttpSinkConfig:
    url: str
    timeout_s: float


@dataclass(frozen=True)
class SinkConfig:
    kind: str
    http: HttpSinkConfig


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class PerformanceConfig:
    device: str  # auto | cpu | cuda
    level: str  # auto | low | balanced | high | custom (custom = use the values above as written)


@dataclass(frozen=True)
class Config:
    video: VideoConfig
    vehicle: VehicleConfig
    association: AssociationConfig
    helmet: HelmetConfig
    rules: RulesConfig
    plate: PlateConfig
    evidence: EvidenceConfig
    sink: SinkConfig
    logging: LoggingConfig
    performance: PerformanceConfig


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (AI_ENGINE_DIR / path).resolve()


def _section(data: dict, name: str, cls):
    raw = data.get(name)
    if not isinstance(raw, dict):
        raise ConfigError(f"Missing config section: {name}")
    values = {}
    for f in fields(cls):
        if f.name not in raw:
            raise ConfigError(f"Missing config key: {name}.{f.name}")
        values[f.name] = raw[f.name]
    return values


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise ConfigError(f"Config file is not valid YAML: {path}: {error}") from error

    video = VideoConfig(**_section(data, "video", VideoConfig))

    vehicle_values = _section(data, "vehicle", VehicleConfig)
    vehicle_path = _resolve(vehicle_values["model"])
    if vehicle_path.exists():
        vehicle_values["model"] = str(vehicle_path)
    else:
        vehicle_values["model"] = vehicle_path.name
        logger.info(
            "[R-SAFE] Vehicle weights not found at %s; Ultralytics will download %s",
            vehicle_path,
            vehicle_path.name,
        )
    vehicle = VehicleConfig(**vehicle_values)

    association = AssociationConfig(**_section(data, "association", AssociationConfig))

    helmet_values = _section(data, "helmet", HelmetConfig)
    helmet_values["model"] = _resolve(helmet_values["model"])
    if not helmet_values["model"].exists():
        raise ConfigError(f"Helmet model not found: {helmet_values['model']}")
    helmet = HelmetConfig(**helmet_values)

    rules = RulesConfig(**_section(data, "rules", RulesConfig))

    plate_values = _section(data, "plate", PlateConfig)
    plate_values["enabled"] = bool(plate_values["enabled"])
    plate_values["detector_model"] = _resolve(plate_values["detector_model"])
    if plate_values["enabled"] and not plate_values["detector_model"].exists():
        raise ConfigError(f"Plate model not found: {plate_values['detector_model']}")
    plate = PlateConfig(**plate_values)

    evidence_values = _section(data, "evidence", EvidenceConfig)
    evidence_values["output_dir"] = _resolve(evidence_values["output_dir"])
    evidence = EvidenceConfig(**evidence_values)

    sink_values = _section(data, "sink", SinkConfig)
    if sink_values["kind"] not in SINK_KINDS:
        raise ConfigError(
            f"Invalid sink.kind: {sink_values['kind']!r} (expected one of {sorted(SINK_KINDS)})"
        )
    http = HttpSinkConfig(**_section(sink_values, "http", HttpSinkConfig))
    if sink_values["kind"] == "http" and not http.url:
        raise ConfigError("sink.http.url is required when sink.kind is 'http'")
    sink = SinkConfig(kind=sink_values["kind"], http=http)

    logging_config = LoggingConfig(**_section(data, "logging", LoggingConfig))

    performance = PerformanceConfig(**_section(data, "performance", PerformanceConfig))
    if performance.device not in PERFORMANCE_DEVICES:
        raise ConfigError(f"Invalid performance.device: {performance.device!r} (expected one of {PERFORMANCE_DEVICES})")
    if performance.level not in PERFORMANCE_LEVELS:
        raise ConfigError(f"Invalid performance.level: {performance.level!r} (expected one of {PERFORMANCE_LEVELS})")

    return Config(
        video=video,
        vehicle=vehicle,
        association=association,
        helmet=helmet,
        rules=rules,
        plate=plate,
        evidence=evidence,
        sink=sink,
        logging=logging_config,
        performance=performance,
    )
