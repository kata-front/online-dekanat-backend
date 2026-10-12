import threading
from io import BytesIO

import anyio
import httpx
import pytest # type: ignore
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import Headers

import main
from services import ocr_service


def image_bytes(image_format='PNG'):
    output = BytesIO()
    Image.new('RGB', (20, 20), 'white').save(output, format=image_format)
    return output.getvalue()


class FakeReader:
    def __init__(self):
        self.calls = 0
        self.lines = ['Первая строка', 'Second line']
        self.error = None

    def readtext(self, image, *, detail, paragraph):
        assert detail == 0
        assert paragraph is True
        self.calls += 1
        if self.error:
            raise self.error
        return self.lines


@pytest.fixture
def reader(monkeypatch):
    instance = FakeReader()
    monkeypatch.setattr(main, 'create_reader', lambda: instance)
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
        data={'prompt': prompt},
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
    assert response.json() == {
        'status': 'received',
        'promt': 'Вопрос',
        'photos': 1,
        'answers': ['Первая строка\nSecond line'],
    }
    assert reader.calls == 1


def test_four_photos(client, reader):
    response = upload(client, count=4)
    assert response.status_code == 200
    assert response.json()['photos'] == 4
    assert len(response.json()['answers']) == 4
    assert reader.calls == 4


def test_no_text(client, reader):
    reader.lines = []
    assert upload(client).json()['answers'] == ['']


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
    monkeypatch.setattr(ocr_service, 'MAX_IMAGE_PIXELS', 399)
    assert upload(client).status_code == 422
    assert reader.calls == 0


def test_declared_request_limit(client, reader, monkeypatch):
    monkeypatch.setattr(main, 'MAX_REQUEST_BYTES', 1)
    assert upload(client).status_code == 413
    assert reader.calls == 0


@pytest.mark.parametrize('declared_length', [None, '1'])
def test_streamed_request_limit(client, reader, monkeypatch, declared_length):
    monkeypatch.setattr(main, 'MAX_REQUEST_BYTES', 30)
    body = (
        b'--boundary\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
        b'question\r\n--boundary--\r\n'
    )
    headers = {'Content-Type': 'multipart/form-data; boundary=boundary'}
    if declared_length is not None:
        headers['Content-Length'] = declared_length
    response = client.post('/analyze', content=iter([body[:20], body[20:]]), headers=headers)
    assert response.status_code == 413
    assert reader.calls == 0


def test_invalid_content_length(client):
    response = client.post('/analyze', content=b'x', headers={'Content-Length': 'invalid'})
    assert response.status_code == 400


def test_ocr_error_is_private(client, reader, caplog):
    reader.error = RuntimeError('secret document content')
    response = upload(client)
    assert response.status_code == 503
    assert 'secret document content' not in response.text
    assert 'secret document content' not in caplog.text
    assert 'RuntimeError' in caplog.text


@pytest.mark.parametrize('fails', [False, True])
def test_files_closed_after_handler(client, monkeypatch, fails):
    photo = UploadFile(
        BytesIO(b'' if fails else image_bytes()),
        headers=Headers({'content-type': 'image/png'}),
    )

    async def check():
        if fails:
            with pytest.raises(HTTPException):
                await main.analyze([photo], 'Вопрос')
        else:
            await main.analyze([photo], 'Вопрос')
        assert photo.file.closed

    anyio.run(check)


def test_health_remains_available_and_ocr_is_serial(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0

    class SlowReader(FakeReader):
        def readtext(self, image, *, detail, paragraph):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            started.set()
            try:
                assert release.wait(5), 'OCR was not released'
                return super().readtext(image, detail=detail, paragraph=paragraph)
            finally:
                with lock:
                    active -= 1

    reader = SlowReader()
    monkeypatch.setattr(main, 'create_reader', lambda: reader)

    async def check():
        async with main.lifespan(main.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url='http://test'
            ) as client:
                responses = []

                async def request_ocr():
                    responses.append(
                        await client.post(
                            '/analyze',
                            data={'prompt': 'Вопрос'},
                            files={'photos': ('photo.png', image_bytes(), 'image/png')},
                        )
                    )

                try:
                    async with anyio.create_task_group() as group:
                        group.start_soon(request_ocr)
                        assert await anyio.to_thread.run_sync(started.wait, 2)
                        group.start_soon(request_ocr)
                        await anyio.sleep(0.05)
                        with anyio.fail_after(1):
                            response = await client.get('/health')
                        assert response.status_code == 200
                        assert not release.is_set()
                        release.set()
                finally:
                    release.set()
                assert len(responses) == 2
                assert all(response.status_code == 200 for response in responses)
                assert reader.calls == 2
                assert peak == 1

    anyio.run(check)


def test_reader_is_removed_after_shutdown(reader):
    with TestClient(main.app):
        assert main.app.state.reader is reader
    assert not hasattr(main.app.state, 'reader')


def test_startup_failure_is_not_hidden(monkeypatch):
    def fail():
        raise RuntimeError('model unavailable')

    monkeypatch.setattr(main, 'create_reader', fail)
    with pytest.raises(RuntimeError, match='model unavailable'), TestClient(main.app):
        pass


@pytest.mark.parametrize(
    ('setting', 'expected'),
    [(None, False), ('false', False), ('0', False), ('true', True), ('1', True), ('TRUE', True)],
)
def test_reader_gpu_setting(monkeypatch, setting, expected):
    import sys
    from types import SimpleNamespace

    if setting is None:
        monkeypatch.delenv('OCR_GPU', raising=False)
    else:
        monkeypatch.setenv('OCR_GPU', setting)
    calls = []
    sentinel = object()

    def factory(languages, *, gpu):
        calls.append((languages, gpu))
        return sentinel

    monkeypatch.setitem(sys.modules, 'easyocr', SimpleNamespace(Reader=factory))
    assert ocr_service.create_reader() is sentinel
    assert calls == [(['ru', 'en'], expected)]


def test_invalid_gpu_setting(monkeypatch):
    monkeypatch.setenv('OCR_GPU', 'invalid')
    with pytest.raises(ValueError, match='OCR_GPU'):
        ocr_service.create_reader()


def test_exact_image_pixel_limit(client, monkeypatch):
    monkeypatch.setattr(ocr_service, 'MAX_IMAGE_PIXELS', 400)
    assert upload(client).status_code == 200


def test_multipart_temporary_files_closed_on_stream_limit(monkeypatch):
    from starlette import formparsers

    files = []
    original = formparsers.SpooledTemporaryFile # type: ignore

    def temporary_file(*args, **kwargs):
        file = original(*args, **kwargs)
        files.append(file)
        return file

    monkeypatch.setattr(formparsers, 'SpooledTemporaryFile', temporary_file)
    prefix = (
        b'--boundary\r\nContent-Disposition: form-data; name="photos"; filename="p.png"\r\n'
        b'Content-Type: image/png\r\n\r\nx'
    )
    monkeypatch.setattr(main, 'MAX_REQUEST_BYTES', len(prefix) + 10)

    async def check():
        async def body():
            yield prefix
            yield b'x' * 100

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url='http://test'
        ) as client:
            response = await client.post(
                '/analyze',
                content=body(),
                headers={'Content-Type': 'multipart/form-data; boundary=boundary'},
            )
        assert response.status_code == 413
        assert files
        assert all(file.closed for file in files)

    anyio.run(check)
