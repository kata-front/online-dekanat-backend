RULES = {
    'study_certificate': {
        'text': '3.1',
        'signature_present': '3.2',
        'seal_present': '3.3',
    },
    'academic_leave_application': {
        'text': '4.1',
        'signature_present': '4.2',
    },
    'internship_certificate': {
        'text': '5.1',
        'signature_present': '5.2',
        'seal_present': '5.3',
    },
}


def validate_fileds(fields: dict, document_type: str) -> dict:
    rules = RULES[document_type]

    missing = [key for key, value in fields.items() if value == 'не найдено' or value is False]

    if missing:
        required_rules = [rules.get(key, rules['text']) for key in missing]
    else:
        required_rules = list(rules.values())

    return {
        'status': 'missing_fields' if missing else 'document_ok',
        'missing_fields': missing,
        'sources': [f'rule:{number}' for number in sorted(set(required_rules))],
    }
