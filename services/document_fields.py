"""The exact field keys from the starter README, with visible document labels."""

import re

from services.field_extractor import _extract_labeled_value, _normalize_date, extract_study_fields

SCHEMAS = {
    'academic_leave_application': {
        'student_name': ('ФИО студента',),
        'group': ('Учебная группа', 'Группа'),
        'course': ('Курс',),
        'leave_reason': ('Причина отпуска',),
        'leave_start': ('Дата начала отпуска', 'Начало отпуска'),
        'leave_end': ('Дата окончания отпуска', 'Окончание отпуска'),
        'application_date': ('Дата заявления',),
    },
    'internship_certificate': {
        'student_name': ('ФИО студента',),
        'organization': ('Название организации', 'Организация'),
        'internship_start': ('Дата начала практики', 'Начало практики'),
        'internship_end': ('Дата окончания практики', 'Окончание практики'),
        'supervisor_name': ('ФИО руководителя практики', 'Руководитель практики'),
        'grade': ('Оценка',),
        'document_number': ('Номер справки',),
        'issue_date': ('Дата выдачи',),
    },
}


def extract_fields(text: str, document_type: str) -> dict[str, str | bool]:
    if document_type == 'study_certificate':
        return extract_study_fields(text)

    fields: dict[str, str | bool] = {}
    for key, labels in SCHEMAS[document_type].items():
        value = 'не найдено'
        for label in labels:
            value = _extract_labeled_value(text, label)
            if value != 'не найдено':
                break
        if key.endswith(('_start', '_end', '_date')):
            value = _normalize_date(value)
        elif key == 'course':
            match = re.fullmatch(r'(\d+)(?:\s*(?:-?й|курс))?', value, re.IGNORECASE)
            value = str(int(match[1])) if match else 'не найдено'
        fields[key] = value

    fields['signature_present'] = False
    if document_type == 'internship_certificate':
        fields['seal_present'] = False
    return fields
