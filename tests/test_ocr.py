"""Test the current OCR contract without native libraries or model files."""

from io import BytesIO

from PIL import Image

from services import ocr_service


def test_read_ocr_returns_text_and_word_boxes(monkeypatch):
    calls = []

    class Iterator:
        def GetUTF8Text(self, level):
            return 'Слово'

        def BoundingBox(self, level):
            return (1, 2, 10, 20)

        def Next(self, level):
            return False

    class Engine:
        def __init__(self, **kwargs):
            assert kwargs['path'] == 'models/tessdata'
            assert kwargs['lang'] == 'rus'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def SetImage(self, image):
            assert image.size == (20, 20)

        def GetUTF8Text(self):
            calls.append('recognize')
            return 'Слово'

        def GetIterator(self):
            return Iterator()

    monkeypatch.setattr(ocr_service, 'PyTessBaseAPI', Engine)
    content = BytesIO()
    Image.new('RGB', (20, 20), 'white').save(content, format='PNG')
    result = ocr_service.read_ocr(content.getvalue())
    assert result == {
        'text': 'Слово',
        'blocks': [{'text': 'Слово', 'box': [(1, 2), (10, 2), (10, 20), (1, 20)]}],
    }
    assert calls == ['recognize']
