"""Decide which motorcycle tracks are helmet violations.

A single frame is never enough: each rider collects helmet votes over the last
few confident checks, and each motorcycle track is reported at most once, when
it leaves the scene or the video ends.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from rsafe.config import RulesConfig
from rsafe.types import BBox, Detection, HelmetResult, PlateResult, RiderVerdict, TrackReport

PLATE_RANK = {"not_visible": 0, "unreadable": 1, "read": 2}


@dataclass
class _Evidence:
    area: float
    frame: np.ndarray
    motorcycle: Detection
    rider_crop: np.ndarray
    rider_boxes: list[BBox]
    seen_at: datetime
    frame_index: int


@dataclass
class _TrackState:
    last_seen: int
    votes: dict[int, deque] = field(default_factory=dict)
    evidence: _Evidence | None = None
    plate: PlateResult = field(default_factory=lambda: PlateResult("not_visible"))
    plate_attempts: int = 0


def _crop(frame: np.ndarray, box: BBox) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(width, int(round(box[2]))), min(height, int(round(box[3])))
    return frame[y1:y2, x1:x2].copy()


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _plate_better(new: PlateResult, old: PlateResult) -> bool:
    if PLATE_RANK[new.status] != PLATE_RANK[old.status]:
        return PLATE_RANK[new.status] > PLATE_RANK[old.status]
    return new.status == "read" and (new.ocr_confidence or 0.0) > (old.ocr_confidence or 0.0)


class ViolationTracker:
    def __init__(self, rules: RulesConfig, max_plate_attempts: int):
        self.rules = rules
        self.max_plate_attempts = max_plate_attempts
        self._tracks: dict[int, _TrackState] = {}
        # ByteTrack can re-find a motorcycle after we closed it; never report the same ID twice
        self._closed: set[int] = set()

    def update(
        self,
        frame_index: int,
        frame: np.ndarray,
        motorcycle: Detection,
        riders: list[tuple[int, Detection, HelmetResult | None]],
        now: datetime,
    ) -> None:
        if motorcycle.track_id in self._closed:
            return
        track = self._tracks.setdefault(motorcycle.track_id, _TrackState(last_seen=frame_index))
        track.last_seen = frame_index

        largest_violator: Detection | None = None
        for position, rider, result in riders:
            if result is None or result.confidence < self.rules.min_confidence:
                continue
            track.votes.setdefault(position, deque(maxlen=self.rules.window)).append(result)
            if result.label == "no_helmet" and (
                largest_violator is None or _area(rider.bbox) > _area(largest_violator.bbox)
            ):
                largest_violator = rider

        if largest_violator is None:
            return
        area = _area(largest_violator.bbox)
        if track.evidence is None or area > track.evidence.area:
            track.evidence = _Evidence(
                area=area,
                frame=frame.copy(),
                motorcycle=motorcycle,
                rider_crop=_crop(frame, largest_violator.bbox),
                rider_boxes=[rider.bbox for _, rider, _ in riders],
                seen_at=now,
                frame_index=frame_index,
            )

    def wants_plate(self, track_id: int) -> bool:
        track = self._tracks.get(track_id)
        return track is None or track.plate_attempts < self.max_plate_attempts

    def add_plate(self, track_id: int, result: PlateResult) -> None:
        track = self._tracks.get(track_id)
        if track is None:
            return
        track.plate_attempts += 1
        if _plate_better(result, track.plate):
            track.plate = result

    def collect_finished(self, frame_index: int) -> list[TrackReport]:
        finished = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_seen > self.rules.track_lost_frames
        ]
        return self._close(finished)

    def flush(self) -> list[TrackReport]:
        return self._close(list(self._tracks))

    def _close(self, track_ids: list[int]) -> list[TrackReport]:
        reports = []
        for track_id in track_ids:
            track = self._tracks.pop(track_id)
            self._closed.add(track_id)
            report = self._report(track_id, track)
            if report is not None:
                reports.append(report)
        return reports

    def _verdict(self, position: int, votes: deque) -> RiderVerdict:
        no_helmet = [v.confidence for v in votes if v.label == "no_helmet"]
        helmet = [v.confidence for v in votes if v.label == "helmet"]
        if len(no_helmet) >= self.rules.min_votes and len(no_helmet) > len(helmet):
            label, winning = "no_helmet", no_helmet
        elif len(helmet) > len(no_helmet):
            label, winning = "helmet", helmet
        else:
            label, winning = "unknown", []
        confidence = round(sum(winning) / len(winning), 4) if winning else 0.0
        return RiderVerdict(position, label, confidence, len(votes))

    def _report(self, track_id: int, track: _TrackState) -> TrackReport | None:
        riders = [self._verdict(pos, votes) for pos, votes in sorted(track.votes.items())]
        if track.evidence is None or not any(r.helmet == "no_helmet" for r in riders):
            return None
        evidence = track.evidence
        return TrackReport(
            track_id=track_id,
            riders=riders,
            motorcycle=evidence.motorcycle,
            frame=evidence.frame,
            rider_crop=evidence.rider_crop,
            plate=track.plate,
            seen_at=evidence.seen_at,
            rider_boxes=evidence.rider_boxes,
            frame_index=evidence.frame_index,
        )
