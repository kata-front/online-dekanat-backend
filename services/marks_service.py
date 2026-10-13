"""Small image heuristics for the starter layouts, not authenticity checks.

Look for a round outline near М.П. and connected strokes beside Подпись.
Words alone never set a flag. Other layouts/shapes need testing and tuning.
"""

import re
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def _bounds(block: dict) -> tuple[float, float, float, float]:
    points = block['box']
    return (
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    )


def detect_marks(content: bytes, blocks: list[dict], needs_seal: bool) -> dict[str, bool]:
    with Image.open(BytesIO(content)) as image:
        gray = np.array(image.convert('L'))
    height, width = gray.shape
    ink = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)[1]

    signature_label = next(
        (b for b in blocks if b['text'].strip().lower().startswith('подпись')), None
    )
    seal_label = next(
        (
            b
            for b in blocks
            if re.sub(r'[^а-я]', '', b['text'].lower().translate(str.maketrans('mp', 'мп'))) == 'мп'
        ),
        None,
    )
    seal_box = None
    if needs_seal and seal_label:
        x1, y1, x2, y2 = _bounds(seal_label)
        label_height = max(1, y2 - y1)
        contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            if not perimeter:
                continue
            roundness = 4 * np.pi * cv2.contourArea(contour) / perimeter**2
            if (
                width * 0.04 < w < width * 0.35
                and 0.75 < w / h < 1.3
                and roundness > 0.65
                and x <= (x1 + x2) / 2 <= x + w
                and y1 - label_height * 15 < y < y1
                and y + h <= y2 + label_height
            ):
                seal_box = (x, y, w, h)
                break

    signature_present = False
    if signature_label:
        _, top, right, bottom = _bounds(signature_label)
        line_height = max(1, bottom - top)
        left = int(right + line_height * 0.4)
        limit = int(seal_box[0] if seal_box else width * 0.75)
        upper = max(0, int(top - line_height * 2))
        lower = min(height, int(bottom + line_height * 2))
        region = ink[upper:lower, left:limit].copy()
        if region.size:
            # Remove the printed horizontal line; keep curved connected strokes.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(line_height * 3)), 1))
            lines = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
            region = cv2.subtract(region, lines)
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            signature_present = any(
                w > line_height * 1.5
                and h > line_height * 0.35
                and cv2.contourArea(contour) > line_height * 0.5
                for contour in contours
                for _, _, w, h in [cv2.boundingRect(contour)]
            )

    marks = {'signature_present': bool(signature_present)}
    if needs_seal:
        marks['seal_present'] = seal_box is not None
    return marks
