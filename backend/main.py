import asyncio

import easyocr
from fastapi import FastAPI, HTTPException, UploadFile

from services.process_pdf import process_pdf
from services.reader import MAX_FILE_SIZE, run_easyocr
from services.text_normalizer import normalize_ocr

app = FastAPI()
reader = easyocr.Reader(['en', 'ru'], gpu=True)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post('/analyze')
async def analyze(
    text: str,
    file: UploadFile
):
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    is_pdf = file.content_type == 'application/pdf' or (file.filename or "").lower().endswith('.pdf')

    try:
        if is_pdf:
            ocr_text = await asyncio.to_thread(process_pdf, contents, reader)
        else:
            ocr_text = await asyncio.to_thread(run_easyocr, contents, reader)

    except Exception as e:
        raise HTTPException(status_code=422, detail=f'Не удалось обработать файл: {e}')

    ocr_text = normalize_ocr(ocr_text)
    
    return {
        'filename': file.filename,
        'content_type': file.content_type,
        'size': len(contents),
        'ocr_text': ocr_text
    }
