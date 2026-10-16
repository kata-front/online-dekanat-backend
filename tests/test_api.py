import json
import threading
from io import BytesIO

import anyio
import httpx
import pytest  # type: ignore
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import Headers

import main
from services import document_loader


def image_bytes(image_format='PNG'):
    output = BytesIO()
    Image.new('RGB', (20, 20), 'white').save(output, format=image_format)
    return output.getvalue()


class FakeOCR:
    def __init__(self):
        self.calls = 0
        self.text = 'ФИО студента: Иванов Иван Иванович'
        self.error = None

    def __call__(self, image):
        self.calls += 1
        if self.error:
            raise self.error
        return {'text': self.text, 'blocks': []}


@pytest.fixture
def reader(monkeypatch):
    instance = FakeOCR()
    monkeypatch.setattr(main, 'read_ocr', instance)
    return instance


@pytest.fixture
def client(reader):
    with TestClient(main.app) as test_client:
        yield test_client


def upload(client, content=None, mime='image/png', prompt='Вопрос', count=1):
    if content is None:
        content = image_bytes()
    return client.post(
        '/analyze',
        data={'prompt': prompt, 'document_type': 'study_certificate'},
        files=[('photos', ('photo', content, mime)) for _ in range(count)],
    )


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


@pytest.mark.parametrize(('image_format', 'mime'), [('PNG', 'image/png'), ('JPEG', 'image/jpeg')])
def test_valid_image_preserves_contract(client, reader, image_format, mime):
    response = upload(client, image_bytes(image_format), mime, '  Вопрос  ')
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {
        'answerable',
        'value',
        'answer',
        'sources',
        'document_type',
        'ocr_text',
        'fields',
        'validation',
        'classification',
        'manual_fields',
    }
    assert result['document_type'] == 'study_certificate'
    assert result['fields']['student_name'] == 'Иванов Иван Иванович'
    assert result['fields']['signature_present'] is False
    assert result['fields']['seal_present'] is False
    assert reader.calls == 1


def test_four_photos(client, reader):
    response = upload(client, count=4)
    assert response.status_code == 422
    assert reader.calls == 0


def test_no_text(client, reader):
    reader.text = ''
    fields = upload(client).json()['fields']
    assert fields['student_name'] == 'не найдено'


def test_manual_date_is_used_for_answer_and_validation(client, reader, monkeypatch):
    async def classify_question(prompt, fields):
        return {'intent': 'field', 'field': 'issue_date'}

    monkeypatch.setattr(main, 'classify_question_for_api', classify_question)
    response = client.post(
        '/analyze',
        data={
            'prompt': 'Какая дата выдачи?',
            'document_type': 'study_certificate',
            'manual_fields': json.dumps(
                [
                    {
                        'id': 'issue_date',
                        'label': 'Дата выдачи',
                        'value': '2026-10-15',
                        'status': 'manual',
                        'source': 'manual',
                    }
                ]
            ),
        },
        files={'photos': ('photo.png', image_bytes(), 'image/png')},
    )

    assert response.status_code == 200
    result = response.json()
    assert result['fields']['issue_date'] == '2026-10-15'
    assert 'issue_date' not in result['validation']['missing_fields']
    assert result['value'] == '2026-10-15'
    assert result['manual_fields'] == ['issue_date']


@pytest.mark.parametrize(
    'manual_fields',
    [
        '[{"id":"unknown_field","value":"value"}]',
        '[{"id":"signature_present","value":"maybe"}]',
        '[{"id":"issue_date","value":""}]',
        '[{"id":"issue_date","value":"2026-10-15"},{"id":"issue_date","value":"2026-10-16"}]',
        '{"issue_date":"2026-10-15"}',
        'not-json',
    ],
)
def test_invalid_manual_fields_are_rejected(client, manual_fields):
    response = client.post(
        '/analyze',
        data={
            'prompt': 'Вопрос',
            'document_type': 'study_certificate',
            'manual_fields': manual_fields,
        },
        files={'photos': ('photo.png', image_bytes(), 'image/png')},
    )

    assert response.status_code == 422


@pytest.mark.parametrize('prompt', ['', '   ', 'x' * 2001])
def test_invalid_question(client, reader, prompt):
    assert upload(client, prompt=prompt).status_code == 422
    assert reader.calls == 0


def test_missing_question(client, reader):
    response = client.post('/analyze', files={'photos': ('photo.png', image_bytes(), 'image/png')})
    assert response.status_code == 422
    assert reader.calls == 0


def test_missing_photos(client):
    assert client.post('/analyze', data={'prompt': 'Вопрос'}).status_code == 422


def test_too_many_photos(client, reader):
    assert upload(client, count=5).status_code == 422
    assert reader.calls == 0


@pytest.mark.parametrize(
    ('content', 'mime'),
    [
        (b'', 'image/png'),
        (b'not an image', 'image/png'),
        (image_bytes(), 'application/octet-stream'),
        (image_bytes(), 'image/jpeg'),
        (image_bytes('GIF'), 'image/png'),
        (image_bytes()[:40], 'image/png'),
        (image_bytes('JPEG')[:-50], 'image/jpeg'),
    ],
)
def test_invalid_image(client, reader, content, mime):
    assert upload(client, content, mime).status_code == 422
    assert reader.calls == 0


def test_file_byte_limit(client, reader, monkeypatch):
    content = image_bytes()
    monkeypatch.setattr(main, 'MAX_FILE_BYTES', len(content) - 1)
    assert upload(client, content).status_code == 413
    assert reader.calls == 0


def test_exact_file_byte_limit(client, monkeypatch):
    content = image_bytes()
    monkeypatch.setattr(main, 'MAX_FILE_BYTES', len(content))
    assert upload(client, content).status_code == 200


def test_image_pixel_limit(client, reader, monkeypatch):
    monkeypatch.setattr(document_loader, 'MAX_IMAGE_PIXELS', 399)
    assert upload(client).status_code == 422
    assert reader.calls == 0


def test_ocr_error_is_private(client, reader, caplog):
    reader.error = RuntimeError('secret document content')
    response = upload(client)
    assert response.status_code == 503
    assert 'secret document content' not in response.text
    assert 'secret document content' not in caplog.text


@pytest.mark.parametrize('fails', [False, True])
def test_files_closed_after_handler(client, monkeypatch, fails):
    photo = UploadFile(
        BytesIO(b'' if fails else image_bytes()),
        headers=Headers({'content-type': 'image/png'}),
    )

    async def check():
        if fails:
            with pytest.raises(HTTPException):
                await main.analyze([photo], 'Вопрос', 'study_certificate')
        else:
            await main.analyze([photo], 'Вопрос', 'study_certificate')
        assert photo.file.closed

    anyio.run(check)


def test_health_remains_available_during_ocr(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_ocr(image):
        started.set()
        assert release.wait(5), 'OCR was not released'
        return {'text': '', 'blocks': []}

    monkeypatch.setattr(main, 'read_ocr', slow_ocr)

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url='http://test'
        ) as client:
            responses = []

            async def request_ocr():
                responses.append(
                    await client.post(
                        '/analyze',
                        data={'prompt': 'Вопрос', 'document_type': 'study_certificate'},
                        files={'photos': ('photo.png', image_bytes(), 'image/png')},
                    )
                )

            try:
                async with anyio.create_task_group() as group:
                    group.start_soon(request_ocr)
                    assert await anyio.to_thread.run_sync(started.wait, 2)
                    with anyio.fail_after(1):
                        response = await client.get('/health')
                    assert response.status_code == 200
                    assert not release.is_set()
                    release.set()
            finally:
                release.set()
            assert responses[0].status_code == 200

    anyio.run(check)


def test_exact_image_pixel_limit(client, monkeypatch):
    monkeypatch.setattr(document_loader, 'MAX_IMAGE_PIXELS', 400)
    assert upload(client).status_code == 200
