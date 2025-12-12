from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from utils.google_inference import GoogleInferenceClient, GoogleInferenceError
from utils.security import sanitize_user_text


class HistoricalCase(BaseModel):
    case_id: str
    case_name: str
    term: Optional[int] = None
    tags: List[str] = []
    outcome: Optional[str] = None
    # Free text used for retrieval (e.g., summary, issue framing)
    summary: str = ""
    # Optional precomputed embedding vector
    embedding: Optional[List[float]] = None
    # Optional transcript URL for backtesting
    transcript_url: Optional[str] = None
    # Optional docket number (e.g., "21-1234") for SCOTUS.gov transcript lookup
    docket: Optional[str] = None


def _tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9']+", (s or "").lower()) if len(t) > 2]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def load_cases_jsonl(path: str) -> List[HistoricalCase]:
    if not path or not os.path.exists(path):
        return []
    cases: List[HistoricalCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cases.append(HistoricalCase.model_validate(obj))
            except Exception:
                continue
    return cases


def case_to_retrieval_text(case: HistoricalCase) -> str:
    parts = [case.case_name, case.summary, " ".join(case.tags or [])]
    return sanitize_user_text("\n".join([p for p in parts if p]), max_len=2400)


@dataclass
class RetrievalResult:
    case: HistoricalCase
    score: float


async def retrieve_similar_cases(
    *,
    brief_text: str,
    cases: List[HistoricalCase],
    top_k: int,
    google_client: Optional[GoogleInferenceClient],
    embed_model: Optional[str],
) -> List[RetrievalResult]:
    brief_text = sanitize_user_text(brief_text, max_len=12000)
    if not brief_text or not cases or top_k <= 0:
        return []

    top_k = max(1, min(int(top_k), 10))
    embed_model = (embed_model or "").strip() or None

    # Prefer embeddings if available and the corpus is already embedded.
    if embed_model and google_client:
        try:
            q_vec = await google_client.embed_text(model=embed_model, text=brief_text[:3500])
            scored: List[RetrievalResult] = []
            for c in cases:
                if c.embedding and len(c.embedding) == len(q_vec):
                    score = _cosine(q_vec, c.embedding)
                else:
                    # Avoid N embeddings calls per request. Fall back to lexical similarity.
                    score = _jaccard(_tokenize(brief_text), _tokenize(case_to_retrieval_text(c)))
                scored.append(RetrievalResult(case=c, score=float(score)))
            scored.sort(key=lambda r: r.score, reverse=True)
            return scored[:top_k]
        except GoogleInferenceError:
            # Fall through to lexical retrieval if embeddings fail.
            pass

    q_tok = _tokenize(brief_text)
    scored = [RetrievalResult(case=c, score=_jaccard(q_tok, _tokenize(case_to_retrieval_text(c)))) for c in cases]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]


async def ensure_embeddings(
    *,
    cases: List[HistoricalCase],
    google_client: Optional[GoogleInferenceClient],
    embed_model: Optional[str],
    max_cases: int = 200,
) -> None:
    """
    Optional helper to embed a small local corpus (best-effort).
    For larger corpora, precompute embeddings offline and store in the JSONL.
    """
    embed_model = (embed_model or "").strip() or None
    if not embed_model or not google_client:
        return
    if len(cases) > max_cases:
        return
    for c in cases:
        if c.embedding:
            continue
        text = case_to_retrieval_text(c)
        if not text:
            continue
        try:
            c.embedding = await google_client.embed_text(model=embed_model, text=text[:2000])
        except Exception:
            # Best-effort.
            c.embedding = None


