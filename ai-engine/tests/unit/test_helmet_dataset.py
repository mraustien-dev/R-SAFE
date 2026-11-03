from rsafe.evaluation.helmet_dataset import (
    GroundTruthTrack,
    load_ground_truth,
    match_violations,
    parse_label,
)


def test_parse_label():
    assert parse_label("DNoHelmetP0NoHelmetP1Helmet") == {"D": "no_helmet", "P0": "no_helmet", "P1": "helmet"}
    assert parse_label("DHelmet") == {"D": "helmet"}


def test_ground_truth_violation_flag(tmp_path):
    csv_path = tmp_path / "clip.csv"
    csv_path.write_text(
        "track_id,frame_id,x,y,w,h,label\n"
        "a,1,10,20,100,200,DHelmet\n"
        "a,2,12,20,100,200,DHelmet\n"
        "b,1,300,20,50,80,DHelmetP1NoHelmet\n"
        "b,2,302,20,50,80,DHelmetP1NoHelmet\n"
        "b,3,304,20,50,80,DHelmet\n",  # one odd frame; the most frequent label wins
        encoding="utf-8",
    )

    tracks = {track.track_id: track for track in load_ground_truth(csv_path)}

    assert tracks["a"].violation is False and tracks["b"].violation is True
    assert tracks["a"].boxes[2] == (12.0, 20.0, 112.0, 220.0)


def predicted(frame_index, bbox):
    return {"frame_index": frame_index, "motorcycle": {"bbox": list(bbox)}}


def test_match_counts():
    truth = [
        GroundTruthTrack("v1", True, {5: (0.0, 0.0, 100.0, 100.0)}),
        GroundTruthTrack("v2", True, {5: (500.0, 0.0, 600.0, 100.0)}),  # never predicted -> fn
        GroundTruthTrack("ok", False, {5: (200.0, 0.0, 300.0, 100.0)}),
    ]
    predictions = [
        predicted(5, (0.0, 10.0, 100.0, 100.0)),  # matches v1 -> tp
        predicted(5, (200.0, 0.0, 300.0, 100.0)),  # rider wore a helmet -> fp
    ]

    assert match_violations(truth, predictions) == {"tp": 1, "fp": 1, "fn": 1}


def test_bike_box_inside_rider_and_bike_box_matches():
    # HELMET boxes cover bike + riders; the COCO box covers only the bike (lower part)
    truth = [GroundTruthTrack("v1", True, {7: (100.0, 0.0, 200.0, 300.0)})]

    assert match_violations(truth, [predicted(7, (120.0, 200.0, 180.0, 300.0))]) == {"tp": 1, "fp": 0, "fn": 0}


def test_duplicate_report_for_same_track_is_false_alarm():
    truth = [GroundTruthTrack("v1", True, {5: (0.0, 0.0, 100.0, 100.0)})]
    box = (0.0, 0.0, 100.0, 100.0)

    assert match_violations(truth, [predicted(5, box), predicted(5, box)]) == {"tp": 1, "fp": 1, "fn": 0}


def test_prediction_on_wrong_frame_does_not_match():
    truth = [GroundTruthTrack("v1", True, {5: (0.0, 0.0, 100.0, 100.0)})]

    assert match_violations(truth, [predicted(9, (0.0, 0.0, 100.0, 100.0))]) == {"tp": 0, "fp": 1, "fn": 1}
