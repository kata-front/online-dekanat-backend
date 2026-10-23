from datetime import datetime
import re

def _parse_date(date_str: str | None):
    if not date_str:
        return None

    try:
        dt = datetime.strptime(date_str.strip(), "%d.%m.%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None

def _extract_value(text: str, pattern):
    
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group('value').strip() if match else None

def check_edu_sertificate(text, has_signature, has_stamp):
    fio = _extract_value(text, r'ФИО\s+студента\s*:\s*(?P<value>[^\n]+)')
    vuz = _extract_value(text, r'Название\s+вуза\s*:\s*(?P<value>[^\n]+)')
    direction = _extract_value(text, r'Направление\s+подготовки\s*:\s*(?P<value>[^\n]+)')
    form = _extract_value(text, r'Форма\s+обучения\s*:\s*(?P<value>[^\n]+)')
    cert_num = _extract_value(text, r'Номер\s+справки\s*:\s*(?P<value>[^\n]+)')
    
    start_date_raw = _extract_value(text, r'Начало\s+периода\s+обучения\s*:\s*(?P<value>[^\n]+)')
    end_date_raw = _extract_value(text, r'Окончание\s+периода\s+обучения\s*:\s*(?P<value>[^\n]+)')
    issue_date_raw = _extract_value(text, r'Дата\s+выдачи\s*:\s*(?P<value>[^\n]+)')

    results = {
        'ФИО студента': fio or "не найдено",
        'Название вуза': vuz or "не найдено",
        'Направление': direction or "не найдено",
        'Форма обучения': form or "не найдено",
        'Номер справки': cert_num or "не найдено",
        'Начало периода': _parse_date(start_date_raw) or "не найдено",
        'Окончание периода': _parse_date(end_date_raw) or "не найдено",
        'Дата выдачи': _parse_date(issue_date_raw) or "не найдено",
        
        'Подпись': "есть" if has_signature else "не найдено",
        'Печать': "есть" if has_stamp else "не найдено"
    }

    missing_fields = []

    for field, value in results.items():
        if value == "не найдено":
            missing_fields.append(field)

            match field:
                case "Подпись":
                    rule = '3.2'
                case "Печать":
                    rule = '3.3'
                case _:
                    rule = '3.1'

            missing_fields.append(f'{field}: {rule}')

    if not missing_fields:
        results['Ошибки'] = "нет"
    else:
        results['Ошибки'] = ', '.join(missing_fields)

    return results