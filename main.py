from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from services.ocr_service import ocr

app = FastAPI(title='Online Dekanat API', version='0.0.1')

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/analyze')
async def analyze(
    photos: list[UploadFile] = File(...),
    prompt: str = Form(...)
) -> dict[str, str | int | list[str]]:
    try:
        texts = []
        prompt = prompt.strip()

        if not prompt:
            raise HTTPException(422, 'Введите вопрос')

        if not 1 <= len(photos) <= 4:
            raise HTTPException(422, 'Прикрепите от 1 до 4 фотографий')

        for photo in photos:
            if photo.content_type not in ['image/jpeg', 'image/png']:
                raise HTTPException(422, 'Прикрепите фотографии в формате JPEG или PNG')

            content = await photo.read()

            if not content:
                raise HTTPException(422, 'Файл не может быть пустым')

            text = ocr(content)
            texts.append(text)

        return {
            'status': 'received',
            'promt': prompt,
            'photos': len(photos),
            'answers': texts
        }
    finally:
        for photo in photos:
                await photo.close()