

from services.field_extractor import normalize_date


assert normalize_date("01.09.2026") == "2026-09-01"
assert normalize_date("1 сентября 2026 года") == "2026-09-01"
assert normalize_date("1 сентября 2026") == "2026-09-01"
assert normalize_date("31 февраля 2026 года") == "не найдено"
assert normalize_date("не найдено") == "не найдено"

print("Проверки дат прошли")