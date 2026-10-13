import re
import unicodedata
from datetime import datetime

STUDY_FIELDS = (
    'student_name',
    'university',
    'program',
    'study_form',
    'study_start',
    'study_end',
    'document_number',
    'issue_date',
)


def _create_empty_fields() -> dict[str, str | bool]:
    fields: dict[str, str | bool] = {}

    for field in STUDY_FIELDS:
        fields[field] = 'не найдено'

    fields['signature_present'] = False
    fields['seal_present'] = False

    return fields


def _extract_labeled_value(text: str, label: str) -> str:
    pattern = rf'^[ \t]*{re.escape(label)}[ \t]*:[ \t]*([^\r\n]*)'

    match = re.search(
        pattern,
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    if not match:
        return 'не найдено'

    value = match.group(1).strip()

    if not value or re.fullmatch(r'[_–—\-\s]+', value):
        return 'не найдено'
    return value


def _normalize_date(value: str) -> str:
    value = unicodedata.normalize('NFC', value).strip().lower()
    try:
        date = datetime.strptime(value, '%d.%m.%Y')
    except ValueError:
        pass
    else:
        return date.strftime('%Y-%m-%d')

    months = {
        'января': 1,
        'февраля': 2,
        'марта': 3,
        'апреля': 4,
        'мая': 5,
        'июня': 6,
        'июля': 7,
        'августа': 8,
        'сентября': 9,
        'октября': 10,
        'ноября': 11,
        'декабря': 12,
    }

    match = re.fullmatch(
        r'(\d{1,2})\s+([а-яё]+)\s+(\d{4})(?:\s+года)?',
        value,
    )

    if not match:
        return 'не найдено'

    day_text, month_text, year_text = match.groups()
    month = months.get(month_text)

    if not month:
        return 'не найдено'

    try:
        parsed = datetime(
            year=int(year_text),
            month=month,
            day=int(day_text),
        )
    except ValueError:
        return 'не найдено'

    return parsed.strftime('%Y-%m-%d')


def extract_study_fields(text: str) -> dict[str, str | bool]:
    fields = _create_empty_fields()

    text_labels = {
        'student_name': 'ФИО студента',
        'university': 'Название вуза',
        'program': 'Направление подготовки',
        'study_form': 'Форма обучения',
        'document_number': 'Номер справки',
    }

    date_labels = {
        'study_start': 'Начало периода обучения',
        'study_end': 'Окончание периода обучения',
        'issue_date': 'Дата выдачи',
    }

    for field, label in text_labels.items():
        value = _extract_labeled_value(text, label)
        fields[field] = value

    for field, label in date_labels.items():
        value = _extract_labeled_value(text, label)
        normalized_value = _normalize_date(value)
        fields[field] = normalized_value

    return fields
