from typing import Literal

from anyio import to_thread
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from services.document_fields import extract_fields
from services.document_loader import load_image
from services.marks_service import detect_marks, detect_round_seal
from services.ocr_service import InvalidImage, read_ocr
from services.question_model import classify_question_for_api
from services.validation.service import validate_fileds

app = FastAPI()

DocumentType = Literal[
    'study_certificate',
    'academic_leave_application',
    'internship_certificate',
]

MAX_FILE_BYTES = 10 * 1024 * 1024


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/analyze')
async def analyze(
    photos: list[UploadFile] = File(...),
    prompt: str = Form(..., max_length=2000),
    document_type: DocumentType = Form(...),
):
    try:
        if len(photos) != 1:
            raise HTTPException(422, 'Прикрепите один файл')

        if not prompt.strip():
            raise HTTPException(422, 'Введите вопрос')

        photo = photos[0]

        if photo.content_type not in ('image/jpeg', 'image/png', 'application/pdf'):
            raise HTTPException(422, 'Поддерживаются JPEG, PNG и одностраничные PDF')

        content = await photo.read(MAX_FILE_BYTES + 1)

        if not content:
            raise HTTPException(422, 'Файл пустой')

        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(413, 'Размер файла не должен превышать 10 МиБ')

        try:
            image = await to_thread.run_sync(
                load_image,
                content,
                photo.content_type,
            )

            ocr_result = await to_thread.run_sync(read_ocr, image)

            text = ocr_result['text']
            blocks = ocr_result['blocks']

        except InvalidImage as exc:
            raise HTTPException(422, str(exc)) from exc

        except Exception as exc:
            raise HTTPException(
                503,
                'Не удалось распознать документ',
            ) from exc

        fields = extract_fields(text, document_type)

        sugnature_result = await to_thread.run_sync(detect_marks, image, blocks, False)

        fields.update(sugnature_result)

        if 'seal_present' in fields:
            fields['seal_present'] = await to_thread.run_sync(
                detect_round_seal,
                image,
                blocks,
            )

        validation = validate_fileds(fields, document_type)
        classification = await classify_question_for_api(prompt, fields)

        return {
            "document_type": document_type,
            "ocr_text": text,
            "fields": fields,
            "validation": validation,
            "classification": classification,
        }
    finally:
        for photo in photos:
            await photo.close()
