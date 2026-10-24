from PIL import Image, ImageDraw
import re

def draw_bboxes(image: Image.Image, ocr_blocks: list, extracted_fields: dict) -> Image.Image:
    draw = ImageDraw.Draw(image)

    value_to_find = []
    for k, v in extracted_fields.items():
        if k not in ['Подпись', 'Печать', 'Ошибки'] and v != 'не найдено':
            str_v = str(v)

            if len(str_v) >= 1:
                value_to_find.append(str_v)
                
            if re.match(r'^\d{4}-\d{2}-\d{2}$', str_v):
                y, m, d = str_v.split('-')
                value_to_find.append(f"{d}.{m}.{y}")

    for bbox, text, conf in ocr_blocks:
        is_match = any(
            val in text or text in val
            for val in value_to_find
            if len(text) > 2
        )

        if is_match:
            x0, y0 = bbox[0]
            x1, y1 = bbox[2]
            draw.rectangle([(x0, y0), (x1, y1)], outline="red", width=2) 

    return image