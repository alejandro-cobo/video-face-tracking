from typing import Any

import cv2
import numpy as np

__all__ = ['draw_face_anns']


def draw_face_anns(image: np.ndarray, face_ann: dict[str, Any], face_idx: str) -> np.ndarray:
    canvas = image.copy()

    area = canvas.shape[0] * canvas.shape[1]
    thickness = int(pow(area, 0.125))

    x1, y1, x2, y2 = [int(x) for x in face_ann['bbox']]
    canvas = cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), thickness, cv2.LINE_AA)

    prob = face_ann['prob']
    label = f'Face {face_idx}: {prob:.2f}'
    canvas = put_text_with_background(canvas, label, [x1, y1, x2, y2], thickness/2, thickness//2)

    lnd = np.array(face_ann['landmarks']).astype(int).reshape(-1, 2)
    for x, y in lnd:
        canvas = cv2.circle(canvas, (x, y), thickness, (0, 0, 255), -1, cv2.LINE_AA)

    return canvas


def put_text_with_background(
    image: np.ndarray,
    label: str,
    bbox: list[int],
    font_size: float,
    thickness: int
) -> np.ndarray:
    canvas = image.copy()

    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_size, 2)
    x1, y1, _, y2 = bbox
    x = max(0, min(canvas.shape[1] - w, x1))
    y = y1 - h - 10
    if y < 0:
        y = y2

    sub_image = canvas[y:y+h+10, x:x+w]
    black_rect = np.zeros_like(sub_image)
    canvas[y:y+h+10, x:x+w] = cv2.addWeighted(sub_image, 0.3, black_rect, 0.7, 1.0)
    canvas = cv2.putText(
        canvas,
        label,
        (x, y + h + 5),
        cv2.FONT_HERSHEY_PLAIN,
        font_size,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
    return canvas

