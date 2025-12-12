from __future__ import annotations

import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

from utils.pdf import extract_text_from_pdf_bytes
from utils.security import sanitize_user_text


TRANSCRIPT_ALLOWED_DOMAINS = {"supremecourt.gov", "oyez.org"}


def _domain_allowed(url: str) -> bool:
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        return any(host == d or host.endswith("." + d) for d in TRANSCRIPT_ALLOWED_DOMAINS)
    except Exception:
        return False


async def fetch_bytes(session: aiohttp.ClientSession, *, url: str, max_bytes: int = 14_000_000) -> bytes:
    async with session.get(url) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Fetch error {resp.status}: {body[:300]}")
        data = await resp.read()
        if len(data) > max_bytes:
            raise RuntimeError("Response too large.")
        return data


def extract_text_from_html_bytes(html_bytes: bytes, *, max_chars: int = 350_000) -> str:
    try:
        html_text = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        html_text = str(html_bytes)
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


async def fetch_transcript_text(
    session: aiohttp.ClientSession,
    *,
    transcript_url: str,
) -> Dict[str, Any]:
    """
    Deterministic transcript fetcher used for backtesting questions.
    Requires the user to provide a transcript URL on an allowlisted domain.
    """
    url = sanitize_user_text(transcript_url, max_len=2048)
    if not url:
        return {"transcript_url": "", "transcript_text": "", "transcript_found": False}
    if not _domain_allowed(url):
        return {"transcript_url": url, "transcript_text": "", "transcript_found": False, "error": "Domain not allowed."}

    data = await fetch_bytes(session, url=url)
    if url.lower().endswith(".pdf"):
        text = extract_text_from_pdf_bytes(data, max_chars=300_000)
        return {"transcript_url": url, "transcript_text": text, "transcript_found": bool(text)}

    # HTML transcript page (e.g., Oyez)
    text = extract_text_from_html_bytes(data, max_chars=300_000)
    return {"transcript_url": url, "transcript_text": text, "transcript_found": bool(text)}


