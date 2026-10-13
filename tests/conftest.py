import pytest

import main


@pytest.fixture(autouse=True)
def no_native_ocr_startup(monkeypatch):
    # API unit tests use fake readers, not native OCR engines or model weights.
    monkeypatch.setattr(main, 'initialize_text_ocr', lambda: None)
