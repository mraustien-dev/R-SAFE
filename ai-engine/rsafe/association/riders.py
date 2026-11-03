"""Link detected persons to the motorcycle they are riding, and crop their heads."""

import numpy as np

from rsafe.types import BBox, Detection


def _bottom_centre(box: BBox) -> tuple[float, float]:
    x1, _, x2, y2 = box
    return (x1 + x2) / 2, y2


def _expand(box: BBox, ratio: float) -> BBox:
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * ratio
    dy = (y2 - y1) * ratio
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def _contains(box: BBox, point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2


def _intersection_area(a: BBox, b: BBox) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, width) * max(0.0, height)


def associate_riders(
    motorcycles: list[Detection],
    persons: list[Detection],
    expand_ratio: float,
    max_riders: int,
) -> dict[int, list[Detection]]:
    """Return {motorcycle index: riders}, riders ordered front (position 1) to back.

    A person rides a motorcycle when the bottom-centre of their box lies inside the
    motorcycle box enlarged by expand_ratio per side. A person matching several
    motorcycles goes to the one their box overlaps most. The nearest rider looks
    tallest in the image, so taller boxes are treated as further forward.
    """
    riders: dict[int, list[Detection]] = {index: [] for index in range(len(motorcycles))}
    expanded = [_expand(m.bbox, expand_ratio) for m in motorcycles]

    for person in persons:
        point = _bottom_centre(person.bbox)
        candidates = [i for i, box in enumerate(expanded) if _contains(box, point)]
        if not candidates:
            continue
        best = max(candidates, key=lambda i: _intersection_area(person.bbox, motorcycles[i].bbox))
        riders[best].append(person)

    for index, matched in riders.items():
        matched.sort(key=lambda p: (-(p.bbox[3] - p.bbox[1]), (p.bbox[0] + p.bbox[2]) / 2))
        riders[index] = matched[:max_riders]

    return riders


def head_crop(
    frame: np.ndarray, rider: BBox, head_ratio: float, pad_ratio: float
) -> np.ndarray | None:
    """Crop the top head_ratio of the rider box, padded by pad_ratio, clipped to the frame."""
    x1, y1, x2, _ = rider
    head_height = (rider[3] - y1) * head_ratio
    pad_x = (x2 - x1) * pad_ratio
    pad_y = head_height * pad_ratio

    frame_height, frame_width = frame.shape[:2]
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(frame_width, int(round(x2 + pad_x)))
    bottom = min(frame_height, int(round(y1 + head_height + pad_y)))

    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]
