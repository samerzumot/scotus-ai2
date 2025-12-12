from __future__ import annotations

import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

from utils.pdf import extract_text_from_pdf_bytes
from utils.security import sanitize_user_text
from utils.transcript_cache import get_cached_transcript, cache_transcript


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


async def fetch_bytes(session: aiohttp.ClientSession, *, url: str, max_bytes: int = 1_500_000) -> bytes:
    """
    Fetch bytes with streaming and early termination for faster downloads.
    Reduced max_bytes to 1.5MB - questions are usually in first 20-30% of transcript.
    Uses larger chunk size for faster streaming.
    """
    async with session.get(
        url,
        headers={
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (compatible; SCOTUS-AI/1.0)",
        },
        timeout=aiohttp.ClientTimeout(total=10, connect=5),  # Aggressive timeouts
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Fetch error {resp.status}: {body[:300]}")
        
        # Stream read with early termination - use larger chunks for speed
        data = bytearray()
        async for chunk in resp.content.iter_chunked(32768):  # 32KB chunks (faster)
            data.extend(chunk)
            if len(data) > max_bytes:
                # Stop reading once we hit the limit
                break
        
        return bytes(data)


def extract_text_from_html_bytes(html_bytes: bytes, *, max_chars: int = 100_000) -> str:
    """
    Optimized HTML extraction - stops early if we have enough text.
    Questions are usually in first 30-40% of transcript, so 100k chars is plenty.
    """
    try:
        # Only decode what we need
        html_text = html_bytes[:max_chars * 2].decode("utf-8", errors="ignore")
    except Exception:
        html_text = str(html_bytes[:max_chars * 2])
    
    # Use faster lxml parser if available, fallback to html.parser
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:
        soup = BeautifulSoup(html_text, "html.parser")
    
    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer"]):
        tag.decompose()
    
    # Extract text with minimal processing
    text = soup.get_text(separator="\n")
    # Quick cleanup
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    
    # Truncate if needed
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    
    return text


async def fetch_transcript_text(
    session: aiohttp.ClientSession,
    *,
    transcript_url: str,
    max_chars: int = 100_000,  # Reduced to 100k - questions are usually in first 30-40%
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Optimized transcript fetcher with caching and aggressive size limits.
    Most important questions appear early in transcripts (first 30-40%).
    """
    url = sanitize_user_text(transcript_url, max_len=2048)
    if not url:
        return {"transcript_url": "", "transcript_text": "", "transcript_found": False}
    if not _domain_allowed(url):
        return {"transcript_url": url, "transcript_text": "", "transcript_found": False, "error": "Domain not allowed."}

    # Check cache first
    if use_cache:
        cached = get_cached_transcript(url)
        if cached:
            return cached

    # Reduced max_bytes to 1.5MB for faster downloads (plenty for 100k chars)
    data = await fetch_bytes(session, url=url, max_bytes=1_500_000)
    
    if url.lower().endswith(".pdf"):
        # For PDFs, extract first portion (most questions are early)
        text = extract_text_from_pdf_bytes(data, max_chars=max_chars)
        result = {"transcript_url": url, "transcript_text": text, "transcript_found": bool(text)}
    else:
        # HTML transcript page (e.g., Oyez) - extract first portion
        text = extract_text_from_html_bytes(data, max_chars=max_chars)
        result = {"transcript_url": url, "transcript_text": text, "transcript_found": bool(text)}
    
    # Cache the result
    if use_cache and result.get("transcript_found"):
        cache_transcript(url, result)
    
    return result


