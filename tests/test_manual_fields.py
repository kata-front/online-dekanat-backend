import json

import pytest

from services.manual_fields import apply_manual_fields


def test_manual_date_normalizes_to_api_date_format():
    updated, applied = apply_manual_fields(
        {'issue_date': 'не найдено'},
        json.dumps([{'id': 'issue_date', 'value': '15.10.2026'}]),
    )

    assert updated['issue_date'] == '2026-10-15'
    assert applied == ['issue_date']


def test_manual_fields_accepts_frontend_metadata_but_uses_only_id_and_value():
    fields = {'issue_date': 'не найдено'}
    updated, applied = apply_manual_fields(
        fields,
        json.dumps(
            [
                {
                    'id': 'issue_date',
                    'label': 'Дата выдачи',
                    'value': '2026-10-15',
                    'status': 'manual',
                    'source': 'manual',
                }
            ]
        ),
    )

    assert updated['issue_date'] == '2026-10-15'
    assert fields['issue_date'] == 'не найдено'
    assert applied == ['issue_date']


def test_manual_positive_mark_is_applied_and_reported():
    updated, applied = apply_manual_fields(
        {'signature_present': False},
        json.dumps([{'id': 'signature_present', 'value': 'Есть'}]),
    )

    assert updated['signature_present'] is True
    assert applied == ['signature_present']


def test_manual_negative_mark_does_not_count_as_fulfilled():
    updated, applied = apply_manual_fields(
        {'signature_present': False},
        json.dumps([{'id': 'signature_present', 'value': 'Нет'}]),
    )

    assert updated['signature_present'] is False
    assert applied == []


@pytest.mark.parametrize('value', ['not a date', '31.02.2026'])
def test_invalid_manual_date_is_rejected(value):
    with pytest.raises(ValueError, match='дату'):
        apply_manual_fields(
            {'issue_date': 'не найдено'}, json.dumps([{'id': 'issue_date', 'value': value}])
        )


@pytest.mark.parametrize(
    'payload',
    [
        '[{"id": 1, "value": "2026-10-15"}]',
        '[{"id": "issue_date", "value": 7}]',
        '[{"id": "issue_date"}]',
        '[null]',
    ],
)
def test_manual_fields_reject_invalid_payload_shapes(payload):
    with pytest.raises(ValueError, match='формат'):
        apply_manual_fields({'issue_date': 'не найдено'}, payload)


def test_manual_course_is_normalized():
    updated, applied = apply_manual_fields(
        {'course': 'не найдено'}, json.dumps([{'id': 'course', 'value': '2-й курс'}])
    )

    assert updated['course'] == '2'
    assert applied == ['course']
