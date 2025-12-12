from __future__ import annotations

import io
import re
from typing import List

from PyPDF2 import PdfReader


def extract_text_from_pdf_bytes(pdf_bytes: bytes, *, max_chars: int = 100_000) -> str:
    """
    Deterministic PDF text extraction.
    Keeps it simple and bounded to protect latency and prompt length.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks: List[str] = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t:
            chunks.append(t)
        if sum(len(c) for c in chunks) >= max_chars:
            break
    text = "\n".join(chunks)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


