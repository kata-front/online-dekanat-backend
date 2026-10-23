import easyocr
import fitz

from services.reader import run_easyocr

def process_pdf(pdf_bytes: bytes, reader: easyocr.Reader) -> str:
    text_parts = []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        if text.strip(): # type: ignore
            text_parts.append(text)
        else:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")

            ocr_text = run_easyocr(img_bytes, reader)     
            text_parts.append(ocr_text)

    doc.close()

    return "\n\n--- Страница ---\n\n".join(text_parts)
