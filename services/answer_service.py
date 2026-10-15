"""Deterministic answers grounded in extracted fields and the case rules."""

from collections.abc import Mapping
from typing import Any

from services.question_model import FIELD_NAMES, QuestionClassification
from services.validation.service import RULES, validate_fileds

_TEXT_FIELDS = {
    'study_certificate': (
        'student_name',
        'university',
        'program',
        'study_form',
        'study_start',
        'study_end',
        'document_number',
        'issue_date',
    ),
    'academic_leave_application': (
        'student_name',
        'group',
        'course',
        'leave_reason',
        'leave_start',
        'leave_end',
        'application_date',
    ),
    'internship_certificate': (
        'student_name',
        'organization',
        'internship_start',
        'internship_end',
        'supervisor_name',
        'grade',
        'document_number',
        'issue_date',
    ),
}
_NO_INFORMATION = (
    'В документе и в Положении нет информации для ответа на этот вопрос. Обратитесь в деканат.'
)
_DECISION = 'Окончательное решение примет сотрудник деканата.'


def _answer(
    answerable: bool,
    value: str | None,
    answer: str,
    sources: list[str],
) -> dict[str, Any]:
    return {
        'answerable': answerable,
        'value': value,
        'answer': answer,
        'sources': sources,
    }


def _check_card(
    document_type: str,
    fields: Mapping[str, str | bool],
    validation: Mapping[str, Any],
) -> None:
    """Reject broken internal contracts rather than invent a successful answer."""
    if document_type not in _TEXT_FIELDS:
        raise ValueError('Unknown document type')
    text_keys = set(_TEXT_FIELDS[document_type])
    mark_keys = set(RULES[document_type]) - {'text'}
    if set(fields) != text_keys | mark_keys:
        raise ValueError('Incomplete or unexpected document fields')
    if any(type(fields[key]) is not str or not fields[key].strip() for key in text_keys):
        raise ValueError('Text fields must contain text or the missing-value marker')
    if any(type(fields[key]) is not bool for key in mark_keys):
        raise ValueError('Document marks must be boolean')
    if set(validation) != {'status', 'missing_fields', 'sources'}:
        raise ValueError('Invalid validation structure')
    for key in ('missing_fields', 'sources'):
        items = validation[key]
        if (
            not isinstance(items, list)
            or any(type(item) is not str for item in items)
            or len(items) != len(set(items))
        ):
            raise ValueError('Validation lists must contain unique strings')
    expected = validate_fileds(dict(fields), document_type)
    if (
        validation['status'] != expected['status']
        or set(validation['missing_fields']) != set(expected['missing_fields'])
        or set(validation['sources']) != set(expected['sources'])
    ):
        raise ValueError('Validation contradicts the extracted fields')


def build_answer(
    document_type: str,
    fields: Mapping[str, str | bool],
    validation: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> dict[str, Any]:
    """Return answerable/value/answer/sources; never call a language model.

    Classification errors cannot be repaired here. The caller is responsible
    for supplying fields extracted from the actual uploaded document.
    """
    _check_card(document_type, fields, validation)
    route = QuestionClassification.model_validate(dict(classification))
    if route.intent != 'field' and route.field is not None:
        raise ValueError('Non-field intent must not specify a field')

    if route.intent == 'unsupported':
        return _answer(False, None, _NO_INFORMATION, [])

    if route.intent == 'decision':
        return _answer(
            True,
            'сотрудник деканата',
            'Окончательное решение о принятии документа и удовлетворении '
            'заявления принимает сотрудник деканата, пункт 1.3 Положения.',
            ['rule:1.3'],
        )

    if route.intent == 'field':
        key = route.field
        if key is None or key not in fields:
            raise ValueError('Field intent requires an available field')
        value = fields[key]
        label = FIELD_NAMES[key]
        if type(value) is bool:
            # False means an absent graphic mark, not an unreadable text field.
            name = 'Подпись' if key == 'signature_present' else 'Печать'
            text = 'присутствует' if value else 'отсутствует'
            return _answer(
                True,
                'да' if value else 'нет',
                f'{name} {text}. Проверено наличие, а не подлинность.',
                [f'field:{key}'],
            )
        if value.strip().casefold().replace('ё', 'е') == 'не найдено':
            return _answer(
                False,
                None,
                f'{label}: не найдено. Значение отсутствует или не читается '
                'в загруженном документе. Обратитесь в деканат.',
                [],
            )
        return _answer(True, value, f'{label}: {value}.', [f'field:{key}'])

    # The only remaining validated intent is validation.
    sources = list(validation['sources'])
    numbers = ', '.join(source.removeprefix('rule:') for source in sources)
    if validation['status'] == 'document_ok':
        text = (
            'Документ содержит все обязательные сведения по пунктам '
            f'{numbers} Положения. {_DECISION}'
        )
    else:
        missing = '; '.join(FIELD_NAMES[key] for key in validation['missing_fields'])
        text = (
            f'Не хватает обязательных сведений: {missing}. '
            f'Требования установлены пунктами {numbers} Положения. {_DECISION}'
        )
    return _answer(True, validation['status'], text, sources)


def export_results(
    documents_path: str,
    questions_path: str,
    output_path: str,
    base_url: str = 'http://127.0.0.1:8000',
) -> None:
    """Build results.json through the running API, never through reference data.

    Input paths in documents.json are relative to that manifest. Prefer its
    image when present; otherwise use the PDF. Fail before requests if any
    document is missing. Only replace the output after the entire run succeeds.
    """
    import json
    import os
    import tempfile
    import uuid
    from pathlib import Path
    from urllib.parse import urlsplit
    from urllib.request import Request, urlopen

    url = urlsplit(base_url)
    if url.scheme not in {'http', 'https'} or not url.netloc or url.query or url.fragment:
        raise ValueError('base_url must be an HTTP(S) server URL')
    manifest = Path(documents_path)
    documents = json.loads(manifest.read_text(encoding='utf-8'))
    questions = json.loads(Path(questions_path).read_text(encoding='utf-8'))
    if not isinstance(documents, list) or not documents:
        raise ValueError('documents.json must be a nonempty array')
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions.json must be a nonempty array')
    by_id: dict[str, dict[str, Any]] = {}
    file_paths: dict[str, Any] = {}
    for doc in documents:
        did = doc['document_id']
        if not isinstance(did, str) or not did or did in by_id:
            raise ValueError('Document IDs must be nonempty and unique')
        if doc['document_type'] not in _TEXT_FIELDS:
            raise ValueError('Unknown document type in manifest')
        candidates = [manifest.parent / doc[key] for key in ('image', 'pdf') if doc.get(key)]
        chosen = next((path for path in candidates if path.is_file()), None)
        if chosen is None:
            raise ValueError(f'Missing input file: {did}')
        by_id[did] = doc
        file_paths[did] = chosen
    seen: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {did: [] for did in by_id}
    for question in questions:
        qid = question['question_id']
        did = question['document_id']
        text = question['question']
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError('Question IDs must be nonempty and unique')
        if did not in grouped or not isinstance(text, str) or not text.strip():
            raise ValueError('Question must reference a document and contain text')
        seen.add(qid)
        grouped[did].append(question)
    if any(not items for items in grouped.values()):
        raise ValueError('Each document must have at least one question')

    result: dict[str, Any] = {'version': '1.0', 'documents': []}
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.pdf': 'application/pdf',
    }
    for did, doc in by_id.items():
        path = file_paths[did]
        mime = mime_types.get(path.suffix.lower())
        if mime is None:
            raise ValueError(f'Unsupported file extension: {path.suffix}')
        content = path.read_bytes()
        if not content or len(content) > 10 * 1024 * 1024:
            raise ValueError(f'Invalid file size: {did}')
        card = None
        answers = []
        for question in grouped[did]:
            boundary = uuid.uuid4().hex
            parts = []
            for key, value in (
                ('prompt', question['question']),
                ('document_type', doc['document_type']),
            ):
                parts.append(
                    (
                        f'--{boundary}\r\n'
                        f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
                        f'{value}\r\n'
                    ).encode()
                )
            parts.append(
                (
                    f'--{boundary}\r\n'
                    'Content-Disposition: form-data; name="photos"; '
                    f'filename="document{path.suffix.lower()}"\r\n'
                    f'Content-Type: {mime}\r\n\r\n'
                ).encode()
            )
            parts.extend([content, f'\r\n--{boundary}--\r\n'.encode()])
            request = Request(
                base_url.rstrip('/') + '/analyze',
                data=b''.join(parts),
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
                method='POST',
            )
            with urlopen(request, timeout=180) as response:
                body = json.load(response)
            if body['document_type'] != doc['document_type']:
                raise ValueError('API returned a different document type')
            current = {key: body[key] for key in ('document_type', 'fields', 'validation')}
            if card is not None and current != card:
                raise ValueError(f'Inconsistent document extraction: {did}')
            card = current
            expected = build_answer(
                doc['document_type'],
                body['fields'],
                body['validation'],
                body['classification'],
            )
            if any(body[key] != value for key, value in expected.items()):
                raise ValueError('API answer does not match its declared evidence')
            answers.append({'question_id': question['question_id'], **expected})
        result['documents'].append({'document_id': did, **card, 'answers': answers})

    target = Path(output_path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            'w', encoding='utf-8', dir=target.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write('\n')
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description='Export results.json using the running /analyze API'
    )
    parser.add_argument('--documents', required=True)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    export_results(args.documents, args.questions, args.output, args.base_url)


if __name__ == '__main__':
    _main()
