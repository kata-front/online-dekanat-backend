"""Regression checks without native OCR models or network requests."""

import hashlib
import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from services import text_ocr_service
from services.field_extractor import extract_study_fields
from services.recognition_service import recognize_document
from services.text_ocr_service import DocumentReader


def image_bytes():
    out = BytesIO()
    Image.new('RGB', (100, 100), 'white').save(out, format='PNG')
    return out.getvalue()


@pytest.mark.parametrize('placeholder', ['___', '———', '---', '   '])
def test_printed_blank_is_missing(placeholder):
    fields = extract_study_fields(
        f'ФИО студента: {placeholder}\nНомер справки: {placeholder}\n'
        'Название вуза: Новый университет «Пример»'
    )
    assert fields['student_name'] == 'не найдено'
    assert fields['document_number'] == 'не найдено'
    assert fields['university'] == 'Новый университет «Пример»'


def test_reader_keeps_layout_and_reads_real_image(monkeypatch):
    calls = []

    class LayoutReader:
        def readtext(self, image, *, detail, paragraph):
            calls.append((image, detail, paragraph))
            return []

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs == {'path': '/model-dir', 'lang': 'rus', 'psm': 3}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def SetImage(self, image):
            assert image.size == (100, 100)
            assert image.getpixel((0, 0)) == (255, 255, 255)

        def GetUTF8Text(self):
            return 'Название вуза: Новый вуз «Пример»\nДата выдачи: 1 июня 2026 года'

    monkeypatch.setitem(
        sys.modules, 'tesserocr', SimpleNamespace(PSM=SimpleNamespace(AUTO=3), PyTessBaseAPI=Engine)
    )
    monkeypatch.setattr(text_ocr_service, '_model_directory', lambda: '/model-dir')
    reader = DocumentReader(LayoutReader())
    content = image_bytes()
    result = recognize_document(reader, content, 'image/png', 'study_certificate')
    assert result['fields']['university'] == 'Новый вуз «Пример»'
    assert result['fields']['issue_date'] == '2026-06-01'
    assert calls == [(content, 1, False)]


@pytest.fixture
def model_cache(tmp_path, monkeypatch):
    text_ocr_service._model_directory.cache_clear()
    monkeypatch.setattr(text_ocr_service, 'MODEL_DIR', tmp_path)
    yield tmp_path
    text_ocr_service._model_directory.cache_clear()


def test_valid_weights_do_not_need_network(model_cache, monkeypatch):
    data = b'test weights'
    (model_cache / 'rus.traineddata').write_bytes(data)
    monkeypatch.setattr(text_ocr_service, 'MODEL_SHA256', hashlib.sha256(data).hexdigest())

    def no_network(*args, **kwargs):
        pytest.fail('Cached weights must not be downloaded')

    monkeypatch.setattr(text_ocr_service, 'urlopen', no_network)
    assert text_ocr_service._model_directory() == str(model_cache)


def test_downloaded_weights_are_checked(model_cache, monkeypatch):
    data = b'test weights'
    monkeypatch.setattr(text_ocr_service, 'MODEL_SHA256', hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(text_ocr_service, 'urlopen', lambda *a, **kw: BytesIO(data))
    assert text_ocr_service._model_directory() == str(model_cache)
    assert (model_cache / 'rus.traineddata').read_bytes() == data


def test_corrupt_weights_are_not_used(model_cache, monkeypatch):
    monkeypatch.setattr(text_ocr_service, 'urlopen', lambda *a, **kw: BytesIO(b'corrupt'))
    with pytest.raises(RuntimeError, match='контрольная сумма'):
        text_ocr_service._model_directory()
    assert not (model_cache / 'rus.traineddata').exists()
    assert not list(model_cache.glob('*.tmp'))


def test_native_engine_is_initialized_explicitly(monkeypatch):
    imported = []
    monkeypatch.setattr(text_ocr_service, 'import_module', imported.append)
    text_ocr_service.initialize_text_ocr()
    assert imported == ['tesserocr']
