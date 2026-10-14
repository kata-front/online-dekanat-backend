import math
from io import BytesIO

import pymupdf
from PIL import Image

from services.ocr_service import InvalidImage, MAX_IMAGE_PIXELS

def load_image(content: bytes, content_type: str) -> bytes:
    try:
        if content_type == 'application/pdf':
            with pymupdf.open(stream=content, filetype='pdf') as document:
                if not document.is_pdf:
                    raise InvalidImage('Содержимое файла не является PDF')
                if document.needs_pass:
                    raise InvalidImage('PDF с паролем не поддерживается')
                if document.page_count != 1:
                    raise InvalidImage('PDF должен содержать одну страницу')

                page = document[0]
                width = math.ceil(page.rect.width * 2)
                height = math.ceil(page.rect.height * 2)

                if width * height > MAX_IMAGE_PIXELS:
                    raise InvalidImage('слишком большая картинка')

                content = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    colorspace=pymupdf.csRGB,
                    alpha=False
                ).tobytes('png')

            content_type = 'image/png'

        expected_format = {
            'image/jpeg': 'JPEG',
            'image/png': 'PNG',
        }.get(content_type)

        if expected_format is None:
            raise InvalidImage("Поддерживаются JPEG, PNG и PDF")

        with Image.open(BytesIO(content)) as image:
            if image.format != expected_format:
                raise InvalidImage("Содержимое не соответствует формату")

            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise InvalidImage("Изображение слишком большое")

            image.load()

        return content
    
    except InvalidImage:
        raise
    except (OSError, ValueError, RuntimeError, Image.DecompressionBombError) as exc:
        raise InvalidImage("Не удалось прочитать файл") from exc 