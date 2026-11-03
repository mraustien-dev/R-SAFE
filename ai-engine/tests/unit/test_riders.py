import numpy as np

from rsafe.association.riders import associate_riders, head_crop
from rsafe.types import Detection


def moto(bbox):
    return Detection(bbox=bbox, confidence=0.9, class_name="motorcycle")


def person(bbox):
    return Detection(bbox=bbox, confidence=0.9, class_name="person")


def test_person_on_motorcycle_is_rider():
    rider = person((120, 100, 180, 290))  # bottom-centre (150, 290)

    result = associate_riders([moto((100, 200, 200, 300))], [rider], 0.15, 3)

    assert result == {0: [rider]}


def test_pedestrian_beside_motorcycle_is_not_rider():
    pedestrian = person((290, 100, 350, 290))  # bottom-centre x=320, beyond 100..200 expanded by 15

    result = associate_riders([moto((100, 200, 200, 300))], [pedestrian], 0.15, 3)

    assert result == {0: []}


def test_expanded_box_catches_rider_just_outside():
    edge_rider = person((180, 100, 240, 290))  # bottom-centre x=210: outside 200, inside 200 + 0.15*100

    result = associate_riders([moto((100, 200, 200, 300))], [edge_rider], 0.15, 3)

    assert result == {0: [edge_rider]}


def test_person_matching_two_motorcycles_goes_to_larger_overlap():
    left = moto((0, 200, 110, 300))
    right = moto((90, 200, 300, 300))
    rider = person((95, 120, 125, 290))  # bottom-centre (110, 290) inside both expanded boxes

    result = associate_riders([left, right], [rider], 0.15, 3)

    assert result == {0: [], 1: [rider]}


def test_max_riders_caps_list_at_3():
    riders = [person((110 + i * 10, 100 + i, 150 + i * 10, 290)) for i in range(4)]

    result = associate_riders([moto((100, 200, 220, 300))], riders, 0.15, 3)

    assert len(result[0]) == 3


def test_back_rider_kept_as_position_2():
    driver = person((120, 80, 170, 290))  # taller: nearest / front
    back_rider = person((140, 110, 185, 285))

    result = associate_riders([moto((100, 200, 200, 300))], [back_rider, driver], 0.15, 3)

    assert result[0] == [driver, back_rider]


def test_no_motorcycles_gives_empty_mapping():
    assert associate_riders([], [person((0, 0, 10, 10))], 0.15, 3) == {}


def test_head_crop_is_top_40_percent_with_padding():
    frame = np.zeros((1000, 1000, 3), dtype=np.uint8)

    crop = head_crop(frame, (100, 100, 200, 500), head_ratio=0.40, pad_ratio=0.20)

    # head = 100x160 at (100,100); padding 20 px sideways, 32 px vertically
    assert crop.shape[:2] == (224, 140)


def test_head_crop_clips_at_frame_edge():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    crop = head_crop(frame, (-50, -50, 40, 300), head_ratio=0.40, pad_ratio=0.20)

    assert crop is not None and crop.size > 0
    assert crop.shape[0] <= 480 and crop.shape[1] <= 640


def test_head_crop_outside_frame_returns_none():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    assert head_crop(frame, (700, 10, 800, 200), head_ratio=0.40, pad_ratio=0.20) is None
