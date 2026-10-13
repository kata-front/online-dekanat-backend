"""Read text locally with Tesseract; keep EasyOCR for mark coordinates."""

import hashlib
import os
from collections.abc import Sequence
from functools import lru_cache
from importlib import import_module
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

from services.ocr_service import Reader

MODEL_DIR = Path.home() / '.cache' / 'online-dekanat' / 'tessdata'
MODEL_URL = 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/rus.traineddata'
MODEL_SHA256 = 'e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225'


@lru_cache(maxsize=1)
def _model_directory() -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = MODEL_DIR / 'rus.traineddata'
    if model.exists() and hashlib.sha256(model.read_bytes()).hexdigest() == MODEL_SHA256:
        return str(MODEL_DIR)

    # Download weights, never send documents to a remote service.
    temporary = MODEL_DIR / f'rus.{os.getpid()}.tmp'
    try:
        with urlopen(MODEL_URL, timeout=60) as response:
            data = response.read(10 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise RuntimeError('Неверная контрольная сумма русских OCR-весов')
        temporary.write_bytes(data)
        temporary.replace(model)
    finally:
        temporary.unlink(missing_ok=True)
    return str(MODEL_DIR)


def initialize_text_ocr() -> None:
    # tesserocr installs signal handlers: import it on the main thread at startup.
    import_module('tesserocr')


class DocumentReader:
    def __init__(self, layout_reader: Reader):
        self.layout_reader = layout_reader

    def readtext(self, image: bytes, *, detail: int, paragraph: bool) -> Sequence[object]:
        return self.layout_reader.readtext(image, detail=detail, paragraph=paragraph)

    def read_document_text(self, content: bytes) -> str:
        # Wheels include the native engine; unit tests do not need it or weights.
        from tesserocr import PSM, PyTessBaseAPI

        with PyTessBaseAPI(path=_model_directory(), lang='rus', psm=PSM.AUTO) as engine:
            with Image.open(BytesIO(content)) as image:
                engine.SetImage(image)
                return engine.GetUTF8Text()


if __name__ == '__main__':
    print(_model_directory())
