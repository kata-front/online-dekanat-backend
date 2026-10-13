import logging
from contextlib import asynccontextmanager
from enum import Enum

from anyio import CapacityLimiter, to_thread
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from services.ocr_service import InvalidImage, create_reader
from services.recognition_service import recognize_document
from services.text_ocr_service import initialize_text_ocr

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 42 * 1024 * 1024
logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    STUDY = 'study_certificate'
    LEAVE = 'academic_leave_application'
    INTERNSHIP = 'internship_certificate'


class RequestSizeLimit:
    """Limit the body before multipart parsing, including chunked uploads."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        headers = dict(scope['headers'])
        try:
            declared_size = int(headers.get(b'content-length', b'0'))
        except ValueError:
            await JSONResponse({'detail': 'Некорректный Content-Length'}, 400)(scope, receive, send)
            return
        if declared_size > MAX_REQUEST_BYTES:
            await JSONResponse({'detail': 'Запрос слишком большой'}, 413)(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > MAX_REQUEST_BYTES:
                    raise HTTPException(413, 'Запрос слишком большой')
            return message

        await self.app(scope, limited_receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_text_ocr()
    app.state.reader = await to_thread.run_sync(create_reader)
    app.state.ocr_limiter = CapacityLimiter(1)
    try:
        yield
    finally:
        del app.state.reader


class AnalyzeResponse(BaseModel):
    status: str
    ocr_text: str
    document_type: DocumentType
    fields: dict[str, str | bool]


app = FastAPI(title='Online Dekanat API', version='0.0.1', lifespan=lifespan)
app.add_middleware(RequestSizeLimit)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/analyze', response_model=AnalyzeResponse)
async def analyze(
    photos: list[UploadFile] = File(...),
    prompt: str = Form(..., max_length=2000),
    document_type: DocumentType = Form(...),
) -> AnalyzeResponse:
    try:
        prompt = prompt.strip()
        if not prompt:
            raise HTTPException(422, 'Введите вопрос')
        if len(photos) != 1:
            raise HTTPException(422, 'Прикрепите один документ')

        for photo in photos:
            if photo.content_type not in ('image/jpeg', 'image/png', 'application/pdf'):
                raise HTTPException(422, 'Прикрепите документ в формате JPEG, PNG или PDF')

            content = await photo.read(MAX_FILE_BYTES + 1)
            if len(content) > MAX_FILE_BYTES:
                raise HTTPException(413, 'Размер фотографии не должен превышать 10 МиБ')
            if not content:
                raise HTTPException(422, 'Файл не может быть пустым')

            try:
                result = await to_thread.run_sync(
                    recognize_document,
                    app.state.reader,
                    content,
                    photo.content_type,
                    document_type.value,
                    limiter=app.state.ocr_limiter,
                )
            except InvalidImage as exc:
                raise HTTPException(422, str(exc)) from exc
            except Exception as exc:
                logger.error('OCR failed (%s)', type(exc).__name__)
                raise HTTPException(503, 'Не удалось распознать текст. Попробуйте позже') from exc

        return AnalyzeResponse(status='received', document_type=document_type, **result)
    finally:
        for photo in photos:
            await photo.close()
