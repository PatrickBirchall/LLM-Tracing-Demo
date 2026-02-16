"""Extract text from PDF bytes for use as document context in LLM requests."""

from pypdf import PdfReader
from io import BytesIO


class PDFServiceError(Exception):
    """Raised when PDF parsing or text extraction fails."""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract plain text from PDF bytes.

    :param pdf_bytes: Raw bytes of the PDF file.
    :return: Extracted text, with pages separated by newlines.
    :raises PDFServiceError: If the PDF cannot be read or has no extractable text.
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:
        raise PDFServiceError("Invalid or corrupted PDF") from exc

    if len(reader.pages) == 0:
        raise PDFServiceError("PDF has no pages")

    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            parts.append(text.strip())

    if not parts:
        raise PDFServiceError("PDF contains no extractable text")

    return "\n\n".join(parts)
