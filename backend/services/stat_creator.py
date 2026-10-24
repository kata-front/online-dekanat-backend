from datetime import datetime
import re

def _parse_date(date_str: str | None) -> str | None:
    if not date_str:
        return None
    s = date_str.strip()
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def _extract_value(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group('value').strip() if match else None

def _extract_first(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        value = _extract_value(text, pattern)
        if value is not None:
            return value
    return None

def _build_result(fields: dict, rules_map: dict) -> dict:
    """Единая функция сборки ответа для всех типов документов (DRY)"""
    clean_fields = {}
    missing = []

    for field, value in fields.items():
        if not value or value == "не найдено":
            clean_fields[field] = "не найдено"
            missing.append({"field": field, "rule": rules_map.get(field, "—")})
        else:
            clean_fields[field] = value

    return {
        "fields": clean_fields,
        "missing": missing,
        "ok": len(missing) == 0
    }

def check_edu_sertificate(text: str, has_signature: bool, has_stamp: bool) -> dict:
    sep = r'\s*[:\-–]?\s*(?P<value>[^\n]+)'
    
    fields = {
        'ФИО студента': _extract_value(text, r'ФИО\s+студента' + sep),
        'Название вуза': _extract_value(text, r'Название\s+вуза' + sep),
        'Направление подготовки': _extract_value(text, r'Направление\s+подготовки' + sep),
        'Форма обучения': _extract_value(text, r'Форма\s+обучения' + sep),
        'Номер справки': _extract_value(text, r'Номер\s+справки' + sep),
        'Начало периода обучения': _parse_date(_extract_value(text, r'Начало\s+периода\s+обучения' + sep)),
        'Окончание периода обучения': _parse_date(_extract_value(text, r'Окончание\s+периода\s+обучения' + sep)),
        'Дата выдачи': _parse_date(_extract_value(text, r'Дата\s+выдачи' + sep)),
        'Подпись': "есть" if has_signature else None,
        'Печать': "есть" if has_stamp else None,
    }

    rules = {
        'Подпись': '3.2', 'Печать': '3.3'
    }
    for k in fields.keys():
        rules.setdefault(k, '3.1')

    return _build_result(fields, rules)

def check_academic_leave(text: str, has_signature: bool = False, **kwargs) -> dict:
    sep = r'\s*[:\-–]?\s*(?P<value>[^\n]+)'
    
    fields = {
        'ФИО студента': _extract_first(text, [r'ФИО\s+студента' + sep]),
        'Учебная группа': _extract_first(text, [r'Учебная\s+группа' + sep, r'Группа' + sep]),
        'Курс': _extract_first(text, [r'Курс' + sep]),
        'Причина отпуска': _extract_first(text, [r'Причина\s+отпуска' + sep, r'Причина' + sep]),
        'Дата начала отпуска': _parse_date(_extract_first(text, [r'Дата\s+начала\s+отпуска' + sep, r'Начало\s+отпуска' + sep])),
        'Дата окончания отпуска': _parse_date(_extract_first(text, [r'Дата\s+окончания\s+отпуска' + sep, r'Окончание\s+отпуска' + sep])),
        'Дата заявления': _parse_date(_extract_first(text, [r'Дата\s+заявления' + sep, r'Дата\s+подачи' + sep])),
        'Подпись': "есть" if has_signature else None,
    }

    rules = {
        'Подпись': '4.2'
    }
    for k in fields.keys():
        rules.setdefault(k, '4.1')

    return _build_result(fields, rules)

def check_practice_certificate(text: str, has_signature: bool = False, has_stamp: bool = False, **kwargs) -> dict:
    sep = r'\s*[:\-–]?\s*(?P<value>[^\n]+)'
    
    fields = {
        'ФИО студента': _extract_first(text, [r'ФИО\s+студента' + sep]),
        'Название организации': _extract_first(text, [r'Название\s+организации' + sep, r'Организация' + sep]),
        'Дата начала практики': _parse_date(_extract_first(text, [r'Дата\s+начала\s+практики' + sep, r'Начало\s+практики' + sep])),
        'Дата окончания практики': _parse_date(_extract_first(text, [r'Дата\s+окончания\s+практики' + sep, r'Окончание\s+практики' + sep])),
        'ФИО руководителя практики': _extract_first(text, [r'ФИО\s+руководителя\s+практики' + sep, r'Руководитель\s+практики' + sep]),
        'Оценка': _extract_first(text, [r'Оценка' + sep]),
        'Номер справки': _extract_first(text, [r'Номер\s+справки' + sep]),
        'Дата выдачи': _parse_date(_extract_first(text, [r'Дата\s+выдачи' + sep])),
        'Подпись': "есть" if has_signature else None,
        'Печать': "есть" if has_stamp else None,
    }

    rules = {
        'Подпись': '5.2', 'Печать': '5.3'
    }
    for k in fields.keys():
        rules.setdefault(k, '5.1')

    return _build_result(fields, rules)