import os
from io import BytesIO
from typing import Protocol

from PIL import Image, UnidentifiedImageError
from collections.abc import Sequence

MAX_IMAGE_PIXELS = 20_000_000


class Reader(Protocol):
    def readtext(self, image: bytes, *, detail: int, paragraph: bool) -> Sequence[object]: ...


class InvalidImage(ValueError):
    pass


def create_reader() -> Reader:
    gpu = os.getenv('OCR_GPU', 'false').lower()
    if gpu not in ('true', 'false', '1', '0'):
        raise ValueError('OCR_GPU must be true, false, 1 or 0')
    # Importing the API must not load models or require EasyOCR in unit tests.
    import easyocr

    return easyocr.Reader(['ru', 'en'], gpu=gpu in ('true', '1'))


def ocr(reader: Reader, content: bytes, content_type: str) -> str:
    try:
        with Image.open(BytesIO(content)) as image:
            expected_format = {'image/jpeg': 'JPEG', 'image/png': 'PNG'}[content_type]
            if image.format != expected_format:
                raise InvalidImage('Содержимое файла не соответствует формату JPEG или PNG')
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidImage('Разрешение фотографии не должно превышать 20 мегапикселей')
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImage('Файл не является корректным изображением') from exc

    result = reader.readtext(content, detail=0, paragraph=True)

    lines: list[str] = []
    for item in result:
        if not isinstance(item, str):
            raise TypeError('EasyOCR returned a non-string text item')
        lines.append(item)

    return '\n'.join(lines)
