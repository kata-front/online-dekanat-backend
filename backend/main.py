import asyncio
import base64
from enum import Enum
from io import BytesIO

import easyocr
from fastapi import FastAPI, HTTPException, UploadFile

from services.stat_creator import check_edu_sertificate, check_academic_leave, check_practice_certificate
from services.process_pdf import process_pdf
from services.reader import MAX_FILE_SIZE, run_easyocr
from services.text_normalizer import normalize_ocr
from services.bbox_drawer import draw_bboxes

class DocumentType(str, Enum):
    EDU_CERTIFICATE = 'edu_sertificate'
    ACADEMIC_LEAVE = 'academic_leave'
    PRACTICE_CERTIFICATE = 'practice_certificate'

app = FastAPI()
reader = easyocr.Reader(['en', 'ru'], gpu=True)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post('/analyze')
async def analyze(
    text: str,
    file: UploadFile,
    document_type: DocumentType
):
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    is_pdf = file.content_type == 'application/pdf' or (file.filename or "").lower().endswith('.pdf')

    try:
        if is_pdf:
            ocr_data = await asyncio.to_thread(process_pdf, contents, reader)
        else:
            ocr_data = await asyncio.to_thread(run_easyocr, contents, reader)

    except Exception as e:
        raise HTTPException(status_code=422, detail=f'Не удалось обработать файл: {e}')

    ocr_text = normalize_ocr(ocr_data['text']) # type: ignore

    statistic = None

    if document_type == 'edu_sertificate':
        statistic = check_edu_sertificate(ocr_text, False, False)
    elif document_type == 'academic_leave':
        statistic = check_academic_leave(ocr_text, False)
    elif document_type == 'practice_certificate':
        statistic = check_practice_certificate(ocr_text, False, False)

    if not statistic:
        raise HTTPException(status_code=422, detail='Не удалось распознать документ')


    annotated_image = draw_bboxes(
        ocr_data['image'], # type: ignore
        ocr_data['blocks'], # type: ignore
        statistic['fields']
    )

    buffered = BytesIO()
    annotated_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        'filename': file.filename,
        'content_type': file.content_type,
        'size': len(contents),
        'ocr_text': ocr_text,
        'statistic': statistic,
        'annotated_image': f'data:image/png;base64,{img_base64}'
    }
