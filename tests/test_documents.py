"""Run without OCR weights; images and PDF are generated in the tests."""

from io import BytesIO

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import main
from services.document_fields import extract_fields
from services.document_loader import load_image
from services.field_extractor import _normalize_date, extract_study_fields
from services.marks_service import detect_marks
from services.ocr_service import InvalidImage
from services.recognition_service import blocks_to_text


def block(text, x=40, y=480, width=120, height=20):
    return {
        'text': text,
        'confidence': 0.99,
        'box': [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
    }


def make_pdf(pages=1, password=None):
    with pymupdf.open() as document:
        for _ in range(pages):
            page = document.new_page(width=300, height=400)
            page.insert_text((30, 40), 'Test document')
        options = {}
        if password:
            options = {
                'encryption': pymupdf.PDF_ENCRYPT_AES_256,
                'owner_pw': password,
                'user_pw': password,
            }
        return document.tobytes(**options)


def test_pdf_becomes_png():
    image = load_image(make_pdf(), 'application/pdf')
    with Image.open(BytesIO(image)) as picture:
        assert picture.format == 'PNG'
        assert picture.size == (600, 800)


@pytest.mark.parametrize('content', [b'', b'not a PDF', make_pdf(2), make_pdf(password='secret')])
def test_invalid_pdf(content):
    with pytest.raises(InvalidImage):
        load_image(content, 'application/pdf')


def test_pdf_pixel_limit(monkeypatch):
    monkeypatch.setattr('services.ocr_service.MAX_IMAGE_PIXELS', 100)
    with pytest.raises(InvalidImage, match='слишком большая'):
        load_image(make_pdf(), 'application/pdf')


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('01.09.2026', '2026-09-01'),
        (' 1 СЕНТЯБРЯ 2026 года ', '2026-09-01'),
        ('1 сентября 2026', '2026-09-01'),
        ('29.02.2024', '2024-02-29'),
        ('29.02.2026', 'не найдено'),
        ('31 февраля 2026 года', 'не найдено'),
        ('не найдено', 'не найдено'),
    ],
)
def test_dates(raw, expected):
    assert _normalize_date(raw) == expected


def test_empty_field_does_not_take_next_line():
    fields = extract_study_fields('ФИО студента:   \nНазвание вуза: Университет')
    assert fields['student_name'] == 'не найдено'
    assert fields['university'] == 'Университет'
    assert fields['issue_date'] == 'не найдено'
    assert len(fields) == 10


def test_academic_leave_fields():
    fields = extract_fields(
        'ФИО студента: Петров Иван Сергеевич\n'
        'Группа: ЭК-21\nКурс: 02\nПричина отпуска: По семейным обстоятельствам\n'
        'Дата начала отпуска: 1 сентября 2026 года\n'
        'Дата окончания отпуска: 31.08.2027\nДата заявления: 20.08.2026',
        'academic_leave_application',
    )
    assert fields == {
        'student_name': 'Петров Иван Сергеевич',
        'group': 'ЭК-21',
        'course': '2',
        'leave_reason': 'По семейным обстоятельствам',
        'leave_start': '2026-09-01',
        'leave_end': '2027-08-31',
        'application_date': '2026-08-20',
        'signature_present': False,
    }


def test_internship_fields():
    fields = extract_fields(
        'ФИО студента: Иванова Анна Петровна\nНазвание организации: ООО «Пример»\n'
        'Начало практики: 01.06.2026\nОкончание практики: 30.06.2026\n'
        'Руководитель практики: Сидоров Пётр Иванович\nОценка: отлично\n'
        'Номер справки: ПР-1\nДата выдачи: 1 июля 2026 года',
        'internship_certificate',
    )
    assert fields == {
        'student_name': 'Иванова Анна Петровна',
        'organization': 'ООО «Пример»',
        'internship_start': '2026-06-01',
        'internship_end': '2026-06-30',
        'supervisor_name': 'Сидоров Пётр Иванович',
        'grade': 'отлично',
        'document_number': 'ПР-1',
        'issue_date': '2026-07-01',
        'signature_present': False,
        'seal_present': False,
    }


def test_split_label_and_value_are_joined():
    blocks = [
        block('01.09.2026', x=250, y=80),
        block('Дата выдачи:', x=40, y=82),
        block('ФИО студента:', x=40, y=40),
        block('Иванов Иван Иванович', x=220, y=40),
    ]
    text = blocks_to_text(blocks)
    assert extract_study_fields(text)['issue_date'] == '2026-09-01'
    assert extract_study_fields(text)['student_name'] == 'Иванов Иван Иванович'


def marks_image(signature=False, seal=False, printed_only=False):
    image = Image.new('RGB', (600, 800), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((40, 480), 'Signature:', fill='black')
    draw.line((220, 500, 400, 500), fill='gray', width=1)
    draw.text((450, 550), 'M.P.', fill='black')
    if printed_only:
        draw.text((250, 480), 'Signature', fill='black')
        draw.text((455, 490), 'STAMP', fill='black')
    if signature:
        draw.line(
            [(240, 495), (250, 465), (265, 498), (280, 470), (295, 496), (330, 480)],
            fill='#245a9a',
            width=3,
        )
    if seal:
        draw.ellipse((430, 440, 530, 540), outline='#245a9a', width=3)
    output = BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


@pytest.mark.parametrize(
    ('signature', 'seal'), [(False, False), (True, False), (False, True), (True, True)]
)
def test_graphic_marks(signature, seal):
    blocks = [block('Подпись сотрудника:'), block('М.П.', x=450, y=550, width=40)]
    assert detect_marks(marks_image(signature, seal), blocks, True) == {
        'signature_present': signature,
        'seal_present': seal,
    }


def test_words_and_line_are_not_marks():
    blocks = [block('Подпись сотрудника:'), block('М.П.', x=450, y=550, width=40)]
    assert detect_marks(marks_image(printed_only=True), blocks, True) == {
        'signature_present': False,
        'seal_present': False,
    }


def test_leave_has_no_seal_field():
    result = detect_marks(marks_image(True, True), [block('Подпись студента:')], False)
    assert result == {'signature_present': True}


def test_missing_anchors_do_not_confirm_marks():
    assert detect_marks(marks_image(True, True), [], True) == {
        'signature_present': False,
        'seal_present': False,
    }


@pytest.mark.parametrize(
    'document_type',
    [
        'study_certificate',
        'academic_leave_application',
        'internship_certificate',
    ],
)
def test_pdf_api_for_each_type(monkeypatch, document_type):
    class Reader:
        calls = 0

        def readtext(self, image, *, detail, paragraph):
            assert detail == 1 and paragraph is False
            with Image.open(BytesIO(image)) as picture:
                assert picture.format == 'PNG'
            self.calls += 1
            return [
                (
                    block('ФИО студента: Иванов Иван Иванович', y=40)['box'],
                    'ФИО студента: Иванов Иван Иванович',
                    0.99,
                )
            ]

    reader = Reader()
    monkeypatch.setattr(main, 'create_reader', lambda: reader)
    with TestClient(main.app) as client:
        response = client.post(
            '/analyze',
            data={'prompt': 'Как зовут студента?', 'document_type': document_type},
            files={'photos': ('document.pdf', make_pdf(), 'application/pdf')},
        )
    assert response.status_code == 200
    result = response.json()
    assert result['document_type'] == document_type
    assert result['fields']['student_name'] == 'Иванов Иван Иванович'
    assert result['ocr_text']
    assert reader.calls == 1


def test_pdf_error_is_422(monkeypatch):
    monkeypatch.setattr(main, 'create_reader', object)
    with TestClient(main.app) as client:
        response = client.post(
            '/analyze',
            data={'prompt': 'Вопрос', 'document_type': 'study_certificate'},
            files={'photos': ('document.pdf', make_pdf(2), 'application/pdf')},
        )
    assert response.status_code == 422


def test_unknown_type_is_422(monkeypatch):
    monkeypatch.setattr(main, 'create_reader', object)
    with TestClient(main.app) as client:
        response = client.post(
            '/analyze',
            data={'prompt': 'Вопрос', 'document_type': 'study_sertificate'},
            files={'photos': ('document.pdf', make_pdf(), 'application/pdf')},
        )
    assert response.status_code == 422


def test_image_labeled_as_pdf_is_rejected():
    with pytest.raises(InvalidImage):
        load_image(marks_image(), 'application/pdf')


def test_latin_stamp_label():
    blocks = [block('Подпись сотрудника:'), block('M.P.', x=450, y=550, width=40)]
    assert detect_marks(marks_image(seal=True), blocks, True)['seal_present'] is True
