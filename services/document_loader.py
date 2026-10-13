"""Convert a single-page PDF to an image; validate images before OCR."""

import math
from io import BytesIO

import pymupdf
from PIL import Image, UnidentifiedImageError

from services import ocr_service
from services.ocr_service import InvalidImage


def load_image(content: bytes, content_type: str) -> bytes:
    if content_type == 'application/pdf':
        content = _render_pdf(content)
        content_type = 'image/png'

    expected_format = {'image/jpeg': 'JPEG', 'image/png': 'PNG'}.get(content_type)
    if expected_format is None:
        raise InvalidImage('Поддерживаются JPEG, PNG и одностраничный PDF')

    try:
        with Image.open(BytesIO(content)) as image:
            if image.format != expected_format:
                raise InvalidImage('Содержимое файла не соответствует указанному формату')
            if image.width * image.height > ocr_service.MAX_IMAGE_PIXELS:
                raise InvalidImage('Разрешение изображения не должно превышать 20 мегапикселей')
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImage('Файл не является корректным изображением') from exc

    return content


def _render_pdf(content: bytes) -> bytes:
    try:
        with pymupdf.open(stream=content, filetype='pdf') as document:
            if not document.is_pdf:
                raise InvalidImage('Содержимое файла не соответствует PDF')
            if document.needs_pass:
                raise InvalidImage('PDF с паролем не поддерживается')
            if document.page_count != 1:
                raise InvalidImage('Прикрепите одностраничный PDF')
            page = document[0]
            zoom = 2  # 144 dpi: enough for the readable starter documents.
            pixels = math.ceil(page.rect.width * zoom) * math.ceil(page.rect.height * zoom)
            if pixels > ocr_service.MAX_IMAGE_PIXELS:
                raise InvalidImage('Страница PDF слишком большая')
            image = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False
            )
            if image.width * image.height > ocr_service.MAX_IMAGE_PIXELS:
                raise InvalidImage('Страница PDF слишком большая')
            return image.tobytes('png')
    except InvalidImage:
        raise
    except (RuntimeError, ValueError) as exc:
        raise InvalidImage('Не удалось прочитать PDF') from exc
