"""One local OCR pass: file -> image -> blocks -> text -> document card."""

from services.document_fields import extract_fields
from services.document_loader import load_image
from services.marks_service import detect_marks
from services.ocr_service import Reader, read_ocr_blocks


def blocks_to_text(blocks: list[dict]) -> str:
    # EasyOCR can split a label and its value; join blocks on the same row.
    rows: list[list[dict]] = []
    for block in sorted(blocks, key=lambda b: min(p[1] for p in b['box'])):
        top = min(p[1] for p in block['box'])
        bottom = max(p[1] for p in block['box'])
        center = (top + bottom) / 2
        if rows:
            previous = rows[-1][0]['box']
            prev_top = min(p[1] for p in previous)
            prev_bottom = max(p[1] for p in previous)
            if (
                abs(center - (prev_top + prev_bottom) / 2)
                < min(bottom - top, prev_bottom - prev_top) * 0.5
            ):
                rows[-1].append(block)
                continue
        rows.append([block])
    return '\n'.join(
        ' '.join(b['text'] for b in sorted(row, key=lambda b: min(p[0] for p in b['box'])))
        for row in rows
    )


def recognize_document(
    reader: Reader, content: bytes, content_type: str, document_type: str
) -> dict:
    image = load_image(content, content_type)
    blocks = read_ocr_blocks(reader, image)
    text = blocks_to_text(blocks)
    fields = extract_fields(text, document_type)
    fields.update(detect_marks(image, blocks, needs_seal='seal_present' in fields))
    return {'ocr_text': text, 'blocks': blocks, 'fields': fields}
