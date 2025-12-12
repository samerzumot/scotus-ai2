from __future__ import annotations

import html
import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse, urlunparse


def sanitize_user_text(text: str, *, max_len: int) -> str:
    """
    Minimal, deterministic sanitation for user-controlled text:
    - strip NULs
    - normalize newlines
    - cap length
    """
    if text is None:
        return ""
    text = str(text).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"
    return text


def xml_escape(text: str) -> str:
    if text is None:
        return ""
    # Escape &, <, >, and quotes to prevent tag-breaking.
    return html.escape(str(text), quote=True)


def wrap_xml_tag(tag: str, text: str) -> str:
    safe = xml_escape(text)
    return f"<{tag}>{safe}</{tag}>"


def normalize_url(url: str) -> str:
    """
    Normalize a URL for comparison:
    - drop fragment
    - lowercase scheme/netloc
    - strip trailing slash (except root)
    - coerce http -> https for matching
    """
    if not url:
        return ""
    url = str(url).strip()
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            parsed = urlparse("https://" + url)
        scheme = parsed.scheme.lower()
        if scheme == "http":
            scheme = "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path or "/"
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        parsed = parsed._replace(scheme=scheme, netloc=netloc, path=path, fragment="")
        return urlunparse(parsed)
    except Exception:
        return url


def url_key(url: str) -> str:
    """
    Canonical key for set membership checks. We intentionally ignore scheme differences.
    """
    n = normalize_url(url)
    if not n:
        return ""
    p = urlparse(n)
    key = f"{p.netloc}{p.path}"
    if p.query:
        key += f"?{p.query}"
    return key


MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def extract_markdown_links(text: str) -> List[Tuple[int, int, str, str]]:
    """Return (start, end, label, url) for markdown links in text."""
    if not text:
        return []
    out: List[Tuple[int, int, str, str]] = []
    for m in MARKDOWN_LINK_RE.finditer(text):
        out.append((m.start(), m.end(), m.group(1), m.group(2)))
    return out


def _sentence_bounds(text: str, idx: int) -> Tuple[int, int]:
    """
    Naive sentence boundary finder used to redact a claim that relies on a hallucinated URL.
    It prefers punctuation boundaries but also respects newlines.
    """
    if not text:
        return (0, 0)
    idx = max(0, min(idx, len(text)))

    # Start: last boundary before idx
    candidates = [
        text.rfind("\n", 0, idx),
        text.rfind(". ", 0, idx),
        text.rfind("? ", 0, idx),
        text.rfind("! ", 0, idx),
    ]
    start = max(candidates)
    start = 0 if start < 0 else start + (2 if text[start : start + 2] in [". ", "? ", "! "] else 1)

    # End: next boundary after idx
    next_candidates = []
    for token in ["\n", ". ", "? ", "! "]:
        j = text.find(token, idx)
        if j != -1:
            next_candidates.append((j, token))
    if not next_candidates:
        end = len(text)
    else:
        j, token = min(next_candidates, key=lambda t: t[0])
        end = j + (2 if token in [". ", "? ", "! "] else 1)

    return (start, end)


def redact_hallucinated_citations_in_text(text: str, *, allowed_url_keys: Sequence[str]) -> str:
    """
    Replace any sentence containing a markdown link whose URL is not in allowed_url_keys.
    This is intentionally conservative: better to redact than to leak uncited claims.
    """
    if not text:
        return text

    allowed = set(k for k in allowed_url_keys if k)
    links = extract_markdown_links(text)
    if not links:
        return text

    # Work in reverse so indexes stay stable.
    redactions: List[Tuple[int, int]] = []
    for start, _end, _label, url in links:
        if url_key(url) not in allowed:
            s0, s1 = _sentence_bounds(text, start)
            redactions.append((s0, s1))

    if not redactions:
        return text

    # Merge overlaps
    redactions.sort()
    merged: List[Tuple[int, int]] = []
    for a, b in redactions:
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))

    out = text
    for a, b in reversed(merged):
        out = out[:a] + "[REDACTED: citation not found in <search_results>]" + out[b:]
    return out


def redact_hallucinated_citations(obj: Any, *, allowed_url_keys: Sequence[str]) -> Any:
    """
    Walk a JSON-like structure and redact any string fields containing hallucinated citations.
    """
    if isinstance(obj, str):
        return redact_hallucinated_citations_in_text(obj, allowed_url_keys=allowed_url_keys)
    if isinstance(obj, list):
        return [redact_hallucinated_citations(v, allowed_url_keys=allowed_url_keys) for v in obj]
    if isinstance(obj, dict):
        return {k: redact_hallucinated_citations(v, allowed_url_keys=allowed_url_keys) for k, v in obj.items()}
    return obj


def safe_json_dumps(obj: Any) -> str:
    """Stable JSON for embedding inside XML tags."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


