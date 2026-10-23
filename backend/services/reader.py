import easyocr
from PIL import Image
import io
import numpy as np

MAX_FILE_SIZE = 10 * 1024 * 1024

def run_easyocr(image_bytes: bytes, reader: easyocr.Reader) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image)

    results = reader.readtext(image_np, detail=0)

    return "\n".join(results) # type: ignore