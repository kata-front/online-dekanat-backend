"""Unit tests substitute native OCR; real document checks run outside pytest."""

import sys
from types import SimpleNamespace


def unmocked_engine(*args, **kwargs):
    raise AssertionError('Подмените OCR в unit-тесте; настоящий OCR проверяется отдельно')


sys.modules['tesserocr'] = SimpleNamespace(
    PSM=SimpleNamespace(AUTO=3),
    RIL=SimpleNamespace(WORD=3),
    PyTessBaseAPI=unmocked_engine,
)
