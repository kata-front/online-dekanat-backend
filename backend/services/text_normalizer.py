"""Normalization helpers for OCR text produced from Russian documents."""

import re
from collections.abc import Callable

# Visually similar Latin -> Cyrillic confusables.
CONFUSABLES: dict[str, str] = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
    "X": "Х", "a": "а", "c": "с", "e": "е", "o": "о",
    "p": "р", "x": "х", "y": "у",
}

# Phrases that are frequently misread as a whole word.
PHRASES: dict[str, str] = {"HE": "НЕ", "Bce": "Все"}

Replacement = str | Callable[[re.Match[str]], str]

PATTERNS: list[tuple[str, re.Pattern[str], Replacement]] = [
    ("known_phrase", re.compile(r"\b(?:HE|Bce)\b"), lambda m: PHRASES[m.group(0)]),
    ("single_latin_in_cyrillic", re.compile(r"(?<=[А-Яа-яЁё])[A-Za-z](?=[А-Яа-яЁё])"), lambda m: CONFUSABLES.get(m.group(0), m.group(0))),
    ("quote_heuristic", re.compile(r"\bк([А-ЯЁ][А-Яа-яЁё]*)х\b"), lambda m: "«" + m.group(1) + "»"),
    ("multiple_spaces", re.compile(r"[ \t]{2,}"), " "),
    ("line_edge_spaces", re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE), ""),
    ("excessive_newlines", re.compile(r"\n{3,}"), "\n\n"),
    ("hyphenated_line_break", re.compile(r"(\w)-\n(\w)"), r"\1\2"),
]


def fix_homoglyphs(text: str) -> str:
    """Replace Latin confusables that appear inside Russian words.

    Single Latin letters are replaced only when they are surrounded by
    Cyrillic letters, so ordinary Latin words are left untouched. Known
    misread phrases are replaced as whole words via ``PHRASES``.
    """
    for source, target in PHRASES.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    text = re.sub(
        r"(?<=[А-Яа-яЁё])[A-Za-z](?=[А-Яа-яЁё])",
        lambda m: CONFUSABLES.get(m.group(0), m.group(0)),
        text,
    )
    return text


def fix_quotes(text: str) -> str:
    """Heuristic: a capitalized word framed by Cyrillic ``к`` and ``х`` is
    likely a word wrapped in guillemets misread by OCR.

    This is NOT a 100% rule and may produce false positives.
    """
    return re.sub(
        r"\bк([А-ЯЁ][А-Яа-яЁё]*)х\b",
        lambda m: "«" + m.group(1) + "»",
        text,
    )


def normalize_ocr(text: str) -> str:
    """Apply all OCR normalization rules and return cleaned text."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = fix_homoglyphs(text)
    text = fix_quotes(text)
    for name, pattern, replacement in PATTERNS:
        if name in {"multiple_spaces", "line_edge_spaces", "excessive_newlines", "hyphenated_line_break"}:
            text = pattern.sub(replacement, text)
    return text


def find_issues(text: str) -> list[dict[str, str]]:
    """Return a list of detected issues as ``{"type": ..., "match": ...}``."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    issues: list[dict[str, str]] = []
    for name, pattern, _ in PATTERNS:
        for match in pattern.finditer(text):
            issues.append({"type": name, "match": match.group(0)})
    return issues
