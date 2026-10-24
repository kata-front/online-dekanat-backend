import easyocr
from PIL import Image
import io
import numpy as np

MAX_FILE_SIZE = 10 * 1024 * 1024

def run_easyocr(image_bytes: bytes, reader: easyocr.Reader):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)

    results = reader.readtext(image_np, detail=1)

    return {
        'text': '\n'.join([item[1] for item in results]), # type: ignore
        'blocks': results,
        'image': image
    }