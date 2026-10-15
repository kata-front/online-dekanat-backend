import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from services.field_extractor import _normalize_date

MAX_MANUAL_FIELD_VALUE_LENGTH = 2000
MAX_MANUAL_FIELDS_PAYLOAD_LENGTH = 32_000
MAX_MANUAL_FIELDS_COUNT = 32


class ManualFieldCorrection(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True, frozen=True)

    id: Annotated[str, Field(min_length=1, max_length=100)]
    value: Annotated[str, Field(max_length=MAX_MANUAL_FIELD_VALUE_LENGTH)]


_MANUAL_FIELDS_ADAPTER = TypeAdapter(
    Annotated[list[ManualFieldCorrection], Field(max_length=MAX_MANUAL_FIELDS_COUNT)]
)

_TRUE_MARK_VALUES = {
    'да',
    'есть',
    'имеется',
    'присутствует',
    'подписан',
    'подписана',
    'подписано',
    'поставлен',
    'поставлена',
}
_FALSE_MARK_VALUES = {
    'нет',
    'отсутствует',
    'не имеется',
    'не присутствует',
    'не подписан',
    'не подписана',
    'не подписано',
    'не поставлен',
    'не поставлена',
}
_DATE_FIELDS = {
    'study_start',
    'study_end',
    'issue_date',
    'leave_start',
    'leave_end',
    'application_date',
    'internship_start',
    'internship_end',
}


def _normalize_mark(value: str) -> bool:
    normalized = ' '.join(value.casefold().replace('ё', 'е').split()).strip(' .,!?:;')
    if normalized in _TRUE_MARK_VALUES:
        return True
    if normalized in _FALSE_MARK_VALUES:
        return False
    raise ValueError('Для подписи или печати укажите «Есть» или «Нет»')


def _normalize_manual_value(field_id: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError('Ручное значение не должно быть пустым')
    if len(normalized) > MAX_MANUAL_FIELD_VALUE_LENGTH:
        raise ValueError('Ручное значение слишком длинное')
    if normalized.casefold().replace('ё', 'е') == 'не найдено':
        raise ValueError('Для отсутствующего значения оставьте поле пустым')

    if field_id in _DATE_FIELDS:
        if _normalize_date(normalized) == 'не найдено':
            try:
                normalized = datetime.strptime(normalized, '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError as exc:
                raise ValueError('Введите корректную дату, например 2026-10-15') from exc
        else:
            normalized = _normalize_date(normalized)

    if field_id == 'course':
        match = re.fullmatch(r'(\d+)(?:\s*-?й)?(?:\s*курс)?', normalized, re.IGNORECASE)
        if not match:
            raise ValueError('Введите номер курса, например 2')
        normalized = str(int(match[1]))

    return normalized


def apply_manual_fields(
    fields: Mapping[str, str | bool],
    raw_manual_fields: str | None,
) -> tuple[dict[str, str | bool], list[str]]:
    """Validate and apply user corrections without mutating OCR output fields."""
    if raw_manual_fields is None:
        return dict(fields), []
    if len(raw_manual_fields) > MAX_MANUAL_FIELDS_PAYLOAD_LENGTH:
        raise ValueError('Слишком большой список ручных уточнений')

    try:
        parsed = json.loads(raw_manual_fields)
        corrections = _MANUAL_FIELDS_ADAPTER.validate_python(parsed)
    except json.JSONDecodeError as exc:
        raise ValueError('Некорректный формат ручных уточнений') from exc
    except ValidationError as exc:
        raise ValueError('Некорректный формат ручных уточнений') from exc

    updated_fields = dict(fields)
    seen_ids: set[str] = set()
    manual_ids: set[str] = set()

    for correction in corrections:
        field_id = correction.id
        value = correction.value
        if field_id not in updated_fields:
            raise ValueError('Ручное уточнение содержит неизвестное поле')
        if field_id in seen_ids:
            raise ValueError('Одно поле нельзя уточнять несколько раз')
        seen_ids.add(field_id)

        original_value = updated_fields[field_id]
        if isinstance(original_value, bool):
            updated_value = _normalize_mark(value)
            updated_fields[field_id] = updated_value
            if updated_value:
                manual_ids.add(field_id)
        else:
            updated_fields[field_id] = _normalize_manual_value(field_id, value)
            manual_ids.add(field_id)

    return updated_fields, sorted(manual_ids)
