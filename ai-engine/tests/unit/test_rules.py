from datetime import datetime

import numpy as np

from rsafe.config import RulesConfig
from rsafe.types import Detection, HelmetResult, PlateResult
from rsafe.violations.rules import ViolationTracker

RULES = RulesConfig(window=10, min_confidence=0.50, min_votes=5, track_lost_frames=30)
NOW = datetime(2026, 11, 6, 14, 32, 12)
NO = HelmetResult("no_helmet", 0.8)
YES = HelmetResult("helmet", 0.8)


def make_tracker(max_plate_attempts=5):
    return ViolationTracker(RULES, max_plate_attempts)


def moto(track_id=1):
    return Detection((100, 100, 300, 300), 0.9, "motorcycle", track_id)


def rider(height=150):
    return Detection((150, 300 - height, 250, 300), 0.9, "person", None)


def frame(value=0):
    return np.full((480, 640, 3), value, dtype=np.uint8)


def feed(tracker, results, position=1, track_id=1, start=0):
    for offset, result in enumerate(results):
        tracker.update(start + offset, frame(), moto(track_id), [(position, rider(), result)], NOW)


def test_five_of_ten_no_helmet_votes_flags():
    tracker = make_tracker()
    feed(tracker, [NO] * 5 + [YES] * 4)

    reports = tracker.flush()

    assert len(reports) == 1
    assert reports[0].riders[0].helmet == "no_helmet"
    assert reports[0].riders[0].frames_observed == 9


def test_four_votes_not_enough():
    tracker = make_tracker()
    feed(tracker, [NO] * 4)

    assert tracker.flush() == []


def test_no_helmet_must_outnumber_helmet():
    tracker = make_tracker()
    feed(tracker, [NO] * 5 + [YES] * 5)

    assert tracker.flush() == []


def test_low_confidence_votes_ignored():
    tracker = make_tracker()
    feed(tracker, [HelmetResult("no_helmet", 0.4)] * 6)

    assert tracker.flush() == []


def test_window_keeps_only_last_ten_votes():
    tracker = make_tracker()
    feed(tracker, [NO] * 6 + [YES] * 10)

    assert tracker.flush() == []


def test_none_results_are_not_votes():
    tracker = make_tracker()
    feed(tracker, [NO] * 5 + [None] * 20)

    reports = tracker.flush()

    assert len(reports) == 1
    assert reports[0].riders[0].frames_observed == 5


def test_back_rider_flagged_and_driver_listed():
    tracker = make_tracker()
    for index in range(6):
        tracker.update(
            index, frame(), moto(), [(1, rider(160), YES), (2, rider(140), NO)], NOW
        )

    reports = tracker.flush()

    assert len(reports) == 1
    verdicts = {r.position: r.helmet for r in reports[0].riders}
    assert verdicts == {1: "helmet", 2: "no_helmet"}
    assert reports[0].riders[1].confidence == 0.8


def test_one_report_per_track():
    tracker = make_tracker()
    feed(tracker, [NO] * 30)

    assert len(tracker.flush()) == 1
    assert tracker.flush() == []


def test_lost_track_emitted_after_track_lost_frames():
    tracker = make_tracker()
    feed(tracker, [NO] * 11)  # frames 0..10

    assert tracker.collect_finished(40) == []
    reports = tracker.collect_finished(41)
    assert [r.track_id for r in reports] == [1]
    assert tracker.flush() == []


def test_lost_track_with_helmet_is_dropped_silently():
    tracker = make_tracker()
    feed(tracker, [YES] * 10)

    assert tracker.collect_finished(100) == []
    assert tracker.flush() == []


def test_flush_emits_active_track():
    tracker = make_tracker()
    feed(tracker, [NO] * 5, start=100)

    reports = tracker.flush()

    assert len(reports) == 1 and reports[0].track_id == 1


def test_separate_tracks_reported_separately():
    tracker = make_tracker()
    feed(tracker, [NO] * 5, track_id=1)
    feed(tracker, [NO] * 5, track_id=2)

    assert sorted(r.track_id for r in tracker.flush()) == [1, 2]


def test_best_frame_is_largest_rider_box():
    tracker = make_tracker()
    heights = [100, 180, 120, 140, 110]
    for index, height in enumerate(heights):
        seen = datetime(2026, 11, 6, 14, 32, index)
        tracker.update(index, frame(index * 10), moto(), [(1, rider(height), NO)], seen)

    report = tracker.flush()[0]

    assert report.frame[0, 0, 0] == 10  # frame from the 180 px update
    assert report.seen_at == datetime(2026, 11, 6, 14, 32, 1)
    assert report.rider_crop.shape[:2] == (180, 100)
    assert report.rider_boxes == [rider(180).bbox]


def test_evidence_frame_is_a_copy():
    tracker = make_tracker()
    shared = frame(7)
    for index in range(5):
        tracker.update(index, shared, moto(), [(1, rider(), NO)], NOW)
    shared[:] = 99

    assert tracker.flush()[0].frame[0, 0, 0] == 7


def test_plate_best_result_kept_and_attempts_capped():
    tracker = make_tracker(max_plate_attempts=5)
    feed(tracker, [NO] * 5)
    results = [
        PlateResult("not_visible"),
        PlateResult("read", "ABC123", 0.6),
        PlateResult("unreadable"),
        PlateResult("read", "ABC12", 0.4),
        PlateResult("not_visible"),
    ]

    for result in results:
        assert tracker.wants_plate(1)
        tracker.add_plate(1, result)

    assert not tracker.wants_plate(1)
    plate = tracker.flush()[0].plate
    assert (plate.status, plate.text, plate.ocr_confidence) == ("read", "ABC123", 0.6)


def test_unreadable_beats_not_visible():
    tracker = make_tracker()
    feed(tracker, [NO] * 5)
    tracker.add_plate(1, PlateResult("unreadable"))
    tracker.add_plate(1, PlateResult("not_visible"))

    assert tracker.flush()[0].plate.status == "unreadable"


def test_track_without_plate_attempts_is_not_visible():
    tracker = make_tracker()
    feed(tracker, [NO] * 5)

    assert tracker.flush()[0].plate.status == "not_visible"


def test_wants_plate_for_unknown_track_is_true():
    assert make_tracker().wants_plate(99)


def test_report_remembers_evidence_frame_index():
    tracker = make_tracker()
    feed(tracker, [NO] * 3, start=10)
    tracker.update(13, frame(), moto(), [(1, rider(height=190), NO)], NOW)  # larger view -> new evidence
    feed(tracker, [NO] * 3, start=14)

    (report,) = tracker.flush()

    assert report.frame_index == 13
