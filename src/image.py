from typing import Sequence

import cv2
import numpy as np

__all__ = ["align_bbox", "crop_image", "expand_bbox", "resize_image"]


def align_bbox(bbox: np.ndarray, new_center: tuple[float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    center_x, center_y = new_center
    new_x = center_x - w / 2
    new_y = center_y - h / 2
    return np.array([new_x, new_y, new_x + w, new_y + h])


def crop_image(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    pad_left = pad_top = pad_right = pad_bottom = 0
    if x1 < 0:
        pad_left = -x1
        w -= x1
        x2 -= x1
        x1 = 0
    if y1 < 0:
        pad_top = -y1
        h -= y1
        y2 -= y1
        y1 = 0
    if x2 > w:
        pad_right = x2 - w
    if y2 > h:
        pad_bottom = y2 - h
    image = np.pad(
        image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    )
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def expand_bbox(bbox: np.ndarray, scale: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox.T
    w = x2 - x1
    h = y2 - y1
    center_x = x1 + w / 2
    center_y = y1 + h / 2
    side = np.maximum(w, h) * scale
    half_side = side / 2

    new_x1 = center_x - half_side
    new_y1 = center_y - half_side
    new_x2 = center_x + half_side
    new_y2 = center_y + half_side

    new_bbox = np.stack((new_x1, new_y1, new_x2, new_y2))
    return new_bbox.T


def resize_image(
    image: np.ndarray,
    size: int | float | Sequence[int | float],
    interpolation: int | None = None,
) -> np.ndarray:
    if isinstance(size, (int, float)):
        size = (size, size)
    is_scale = isinstance(size[0], float)

    if interpolation is None:
        scale = max(size) if is_scale else max(size) / max(image.shape[:2])
        interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA

    if is_scale:
        resized_img = cv2.resize(
            image, None, fx=size[0], fy=size[1], interpolation=interpolation
        )
    else:
        resized_img = cv2.resize(image, size, interpolation=interpolation)
    return resized_img
