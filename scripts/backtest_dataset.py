#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from dotenv import load_dotenv

from utils.backtest import extract_questions_from_transcript, score_predicted_questions
from utils.pdf import extract_text_from_pdf_bytes
from utils.predictor import predict_votes_and_questions
from utils.security import sanitize_user_text
from utils.transcripts import fetch_transcript_text


@dataclass
class CaseRecord:
    case_id: str
    uploader_side: str
    case_hint: str
    brief_text: str
    transcript_url: str
    transcript_text: str


def load_env(repo_root: str) -> None:
    for p in [os.getenv("SCOTUS_AI_ENV_FILE") or "", os.path.join(repo_root, "env.local"), os.path.join(repo_root, ".env")]:
        if p and os.path.exists(p):
            load_dotenv(dotenv_path=p, override=False)
            return


def read_jsonl(path: str, *, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


def to_record(obj: Dict[str, Any], *, repo_root: str) -> CaseRecord:
    case_id = sanitize_user_text(obj.get("case_id") or "", max_len=80) or "unknown"
    uploader_side = sanitize_user_text(obj.get("uploader_side") or "UNKNOWN", max_len=16).upper()
    case_hint = sanitize_user_text(obj.get("case_hint") or obj.get("case_name") or "", max_len=240)

    brief_text = obj.get("brief_text") or ""
    brief_pdf_path = obj.get("brief_pdf_path") or ""
    if not brief_text and brief_pdf_path:
        pdf_path = brief_pdf_path
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(repo_root, pdf_path)
        with open(pdf_path, "rb") as f:
            brief_text = extract_text_from_pdf_bytes(f.read(), max_chars=220_000)

    transcript_url = sanitize_user_text(obj.get("transcript_url") or "", max_len=2048)
    transcript_text = sanitize_user_text(obj.get("transcript_text") or "", max_len=450_000)

    return CaseRecord(
        case_id=case_id,
        uploader_side=uploader_side,
        case_hint=case_hint,
        brief_text=sanitize_user_text(brief_text, max_len=220_000),
        transcript_url=transcript_url,
        transcript_text=transcript_text,
    )


async def main_async() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_env(repo_root)

    ap = argparse.ArgumentParser(description="Backtest SCOTUS Brief Predictor on a JSONL dataset.")
    ap.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    ap.add_argument("--max-cases", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--corpus", default=os.path.join(repo_root, "data", "historical_cases.sample.jsonl"))
    ap.add_argument("--top-k", type=int, default=int(os.getenv("RETRIEVAL_TOP_K") or "5"))
    args = ap.parse_args()

    raw = read_jsonl(args.dataset, limit=max(1, int(args.max_cases)))
    records = [to_record(r, repo_root=repo_root) for r in raw]

    timeout = aiohttp.ClientTimeout(total=60)
    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    scores: List[int] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def run_one(rec: CaseRecord):
            async with sem:
                pred = await predict_votes_and_questions(
                    session=session,
                    brief_text=rec.brief_text,
                    uploader_side=rec.uploader_side,
                    case_hint=rec.case_hint,
                    corpus_path=args.corpus,
                    retrieval_top_k=int(args.top_k),
                )

                transcript_text = rec.transcript_text
                transcript_url = rec.transcript_url
                if not transcript_text and transcript_url:
                    tr = await fetch_transcript_text(session, transcript_url=transcript_url)
                    transcript_text = tr.get("transcript_text") or ""

                actual_questions = extract_questions_from_transcript(transcript_text, limit=250)
                predicted_questions = [q.question for q in pred.questions]
                score, _matches, _explanation = score_predicted_questions(predicted_questions, actual_questions)
                scores.append(score)
                print(f"{rec.case_id}: questions_score_pct={score}")

        await asyncio.gather(*(run_one(r) for r in records))

    avg = int(round(sum(scores) / max(1, len(scores)))) if scores else 0
    print(f"\nAverage questions_score_pct across {len(scores)} cases: {avg}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())


