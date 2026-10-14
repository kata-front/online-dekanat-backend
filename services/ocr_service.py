from io import BytesIO

from PIL import Image
from tesserocr import PSM, RIL, PyTessBaseAPI

MAX_IMAGE_PIXELS = 20_000_000


class InvalidImage(ValueError):
    pass


def get_word_blocks(engine) -> list[dict]:
    blocks = []
    iterator = engine.GetIterator()

    while iterator is not None:
        word = iterator.GetUTF8Text(RIL.WORD)
        bounds = iterator.BoundingBox(RIL.WORD)

        if word and bounds:
            left, top, right, bottom = bounds

            blocks.append(
                {'text': word, 'box': [(left, top), (right, top), (right, bottom), (left, bottom)]}
            )

        if not iterator.Next(RIL.WORD):
            break

    return blocks


def read_ocr(content: bytes) -> dict:
    with Image.open(BytesIO(content)) as image:
        with PyTessBaseAPI(path='models/tessdata', lang='rus', psm=PSM.AUTO) as engine:
            engine.SetImage(image)

            text = engine.GetUTF8Text()
            blocks = get_word_blocks(engine)

            return {'text': text, 'blocks': blocks}
