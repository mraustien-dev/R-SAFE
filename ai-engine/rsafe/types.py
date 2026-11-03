"""Data shapes shared by every stage of the R-SAFE pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

# Pixel box in the original frame: (x1, y1, x2, y2).
BBox = tuple[float, float, float, float]

HelmetLabel = Literal["helmet", "no_helmet"]
PlateStatus = Literal["read", "unreadable", "not_visible"]
SourceKind = Literal["file", "ip_webcam", "device"]


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    confidence: float
    class_name: str
    track_id: int | None = None


@dataclass(frozen=True)
class HelmetResult:
    label: HelmetLabel
    confidence: float


@dataclass(frozen=True)
class RiderVerdict:
    position: int
    helmet: Literal["helmet", "no_helmet", "unknown"]
    confidence: float
    frames_observed: int


@dataclass(frozen=True)
class PlateResult:
    status: PlateStatus
    text: str | None = None
    ocr_confidence: float | None = None
    crop: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class SourceInfo:
    kind: SourceKind
    uri: str
    camera_id: str


@dataclass
class TrackReport:
    """Everything needed to write one violation for one motorcycle track."""

    track_id: int
    riders: list[RiderVerdict]
    motorcycle: Detection
    frame: np.ndarray = field(repr=False)
    rider_crop: np.ndarray = field(repr=False)
    plate: PlateResult
    seen_at: datetime
    rider_boxes: list[BBox] = field(default_factory=list)
    frame_index: int = -1  # source frame of the evidence (used by evaluation)
