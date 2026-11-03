import numpy as np

from rsafe.detectors.vehicle import parse_results

NAMES = {0: "person", 3: "motorcycle"}


def test_parse_splits_motorcycles_and_persons():
    boxes = np.array([[10, 20, 110, 220], [30, 0, 80, 150], [200, 0, 260, 160]], dtype=float)

    motorcycles, persons = parse_results(
        boxes, np.array([0.9, 0.8, 0.7]), np.array([3, 0, 0]), np.array([5, 6, 7]), NAMES
    )

    assert len(motorcycles) == 1 and len(persons) == 2
    assert motorcycles[0].bbox == (10.0, 20.0, 110.0, 220.0)
    assert motorcycles[0].class_name == "motorcycle"
    assert motorcycles[0].track_id == 5
    assert [p.track_id for p in persons] == [6, 7]
    assert isinstance(motorcycles[0].track_id, int)
    assert motorcycles[0].confidence == 0.9


def test_parse_without_track_ids_gives_none():
    motorcycles, persons = parse_results(
        np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=float),
        np.array([0.9, 0.9]),
        np.array([3, 0]),
        None,
        NAMES,
    )

    assert motorcycles[0].track_id is None and persons[0].track_id is None


def test_parse_ignores_other_classes():
    motorcycles, persons = parse_results(
        np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9]), np.array([2]), None,
        {**NAMES, 2: "car"},
    )

    assert motorcycles == [] and persons == []


def test_parse_empty_arrays():
    empty = np.zeros((0, 4))

    assert parse_results(empty, np.zeros(0), np.zeros(0), None, NAMES) == ([], [])
