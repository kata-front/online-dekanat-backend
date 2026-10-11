from typing import cast

import easyocr

reader = easyocr.Reader(['ru', 'en'], gpu=True)

def ocr(content: bytes) -> str:
    lines = cast(list[str],reader.readtext(content, detail=0, paragraph=True))
    return '\n'.join(lines)