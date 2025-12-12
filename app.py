from __future__ import annotations

import asyncio
import inspect
import os
import time
from typing import Any, Dict, Optional

import aiohttp
from dotenv import load_dotenv
from quart import Quart, jsonify, render_template, request

from utils.backtest import extract_questions_from_transcript, score_predicted_questions, score_predicted_questions_semantic
from utils.pdf import extract_text_from_pdf_bytes
from utils.predictor import PredictionConfigError, predict_votes_and_questions
from utils.schemas import BacktestResult
from utils.security import sanitize_user_text
from utils.transcripts import fetch_transcript_text


# --- Env loading (supports env.local to keep secrets out of git) ---
_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ENV_CANDIDATES = [
    os.getenv("SCOTUS_AI_ENV_FILE") or "",
    os.path.join(_ROOT, "env.local"),
    os.path.join(_ROOT, ".env"),
]
for _p in _DEFAULT_ENV_CANDIDATES:
    if _p and os.path.exists(_p):
        load_dotenv(dotenv_path=_p, override=False)
        break


app = Quart(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB
if os.getenv("APP_ENV") == "development":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


_aiohttp_session: Optional[aiohttp.ClientSession] = None
_server_start_ms: int = int(time.time() * 1000)


@app.before_serving
async def _startup() -> None:
    global _aiohttp_session
    timeout = aiohttp.ClientTimeout(total=45)
    _aiohttp_session = aiohttp.ClientSession(timeout=timeout)


@app.after_serving
async def _shutdown() -> None:
    global _aiohttp_session
    if _aiohttp_session:
        await _aiohttp_session.close()
    _aiohttp_session = None


def _session() -> aiohttp.ClientSession:
    if _aiohttp_session is None:
        raise RuntimeError("HTTP session not initialized.")
    return _aiohttp_session


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _hotreload_token_path() -> str:
    return os.path.join(_repo_root(), "hotreload.token")


def _read_hotreload_token() -> int:
    try:
        with open(_hotreload_token_path(), "r", encoding="utf-8") as f:
            s = (f.read() or "").strip()
        return int(s) if s else 0
    except Exception:
        return 0


def _json_error(message: str, *, status: int = 400, code: str = "bad_request"):
    return jsonify({"ok": False, "error": message, "code": code}), status


async def _read_filestorage_bytes(file_storage) -> bytes:
    data = file_storage.read()
    if inspect.isawaitable(data):
        return await data
    return await asyncio.to_thread(lambda: data)


@app.get("/")
async def index():
    return await render_template("index.html")


@app.get("/health")
async def health():
    return jsonify({"ok": True})


@app.get("/__hotreload")
async def hotreload_token():
    if os.getenv("APP_ENV") != "development":
        return _json_error("Not found.", status=404, code="not_found")
    return jsonify({"ok": True, "token": _read_hotreload_token(), "server_start_ms": _server_start_ms})


@app.post("/predict")
async def predict():
    files = await request.files
    if not files or "brief" not in files:
        return _json_error("Upload a PDF as multipart field 'brief'.", status=400)

    fs = files["brief"]
    if not (fs.filename or "").lower().endswith(".pdf"):
        return _json_error("Only PDF uploads are supported.", status=400)

    form = await request.form
    uploader_side = sanitize_user_text(form.get("side") or "UNKNOWN", max_len=16).upper()
    case_hint = sanitize_user_text(form.get("case_hint") or "", max_len=240)
    transcript_url = sanitize_user_text(form.get("transcript_url") or "", max_len=2048)
    run_backtest = sanitize_user_text(form.get("run_backtest") or "false", max_len=10).lower() in {"1", "true", "yes", "on"}
    
    # Auto-detect transcript URL if not provided
    transcript_auto_detected = False
    if not transcript_url and case_hint:
        from utils.transcript_finder import find_transcript_urls, extract_case_name_from_hint
        case_name = extract_case_name_from_hint(case_hint)
        if case_name:
            candidates = find_transcript_urls(case_name)
            if candidates:
                transcript_url = candidates[0]  # Try first candidate
                transcript_auto_detected = True

    corpus_path = os.getenv("HISTORICAL_CASES_PATH") or os.path.join(_ROOT, "data", "historical_cases.sample.jsonl")
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K") or "5")

    try:
        pdf_bytes = await _read_filestorage_bytes(fs)
        brief_text = extract_text_from_pdf_bytes(pdf_bytes, max_chars=220_000)
        if not brief_text:
            return _json_error("Could not extract text from PDF.", status=400)

        # Auto-detect transcript URL if not provided but case_hint is available
        # Also check retrieved cases for transcript URLs
        if not transcript_url and case_hint:
            from utils.transcript_finder import find_transcript_urls, extract_case_name_from_hint
            case_name = extract_case_name_from_hint(case_hint)
            if case_name:
                # Extract term from hint if available (e.g., "Dobbs (2022)")
                term_match = re.search(r'\((\d{4})\)', case_hint)
                term = int(term_match.group(1)) if term_match else None
                candidates = find_transcript_urls(case_name, term=term)
                if candidates:
                    transcript_url = candidates[0]
                    transcript_auto_detected = True
        
        # Also try to get transcript from retrieved cases (after prediction)
        # We'll check this after prediction runs
        
        # If we have a transcript URL, fetch it in parallel while the model runs
        transcript_task = None
        if transcript_url:
            transcript_task = asyncio.create_task(fetch_transcript_text(_session(), transcript_url=transcript_url))

        prediction = await predict_votes_and_questions(
            session=_session(),
            brief_text=brief_text,
            uploader_side=uploader_side,
            case_hint=case_hint,
            corpus_path=corpus_path,
            retrieval_top_k=retrieval_top_k,
        )

        # If we still don't have a transcript URL, try to get it from retrieved cases
        if not transcript_url and prediction.retrieved_cases:
            from utils.transcript_finder import find_transcript_urls
            from utils.retrieval import load_cases_jsonl
            # Load the full case data to check for transcript_url field
            corpus_cases = load_cases_jsonl(corpus_path)
            case_map = {c.case_name.lower(): c for c in corpus_cases}
            
            # Check retrieved cases for transcript URLs
            for retrieved_case in prediction.retrieved_cases:
                case_key = retrieved_case.case_name.lower()
                if case_key in case_map:
                    full_case = case_map[case_key]
                    if full_case.transcript_url:
                        transcript_url = full_case.transcript_url
                        transcript_auto_detected = True
                        break
                    # Otherwise try to construct from case name and term
                    elif full_case.case_name:
                        candidates = find_transcript_urls(full_case.case_name, term=full_case.term, docket=full_case.docket)
                        if candidates:
                            transcript_url = candidates[0]
                            transcript_auto_detected = True
                            break
            
            # Start fetching if we just found it
            if transcript_url and transcript_task is None:
                transcript_task = asyncio.create_task(fetch_transcript_text(_session(), transcript_url=transcript_url))

        # Auto-run backtest if transcript URL is available (auto-detected or provided)
        backtest_obj: Optional[BacktestResult] = None
        if transcript_url:
            # If we haven't started fetching yet, do it now
            if transcript_task is None:
                transcript_task = asyncio.create_task(fetch_transcript_text(_session(), transcript_url=transcript_url))
            
            transcript = await transcript_task
            transcript_found = bool(transcript.get("transcript_found"))
            
            if transcript_found:
                actual_questions = extract_questions_from_transcript(transcript.get("transcript_text") or "", limit=200)
                predicted_questions = [q.question for q in prediction.questions]
                
                # Use semantic similarity if Google client is available (better for topic/gist matching)
                google_key = (os.getenv("GOOGLE_AI_KEY") or "").strip()
                embed_model = (os.getenv("GOOGLE_EMBED_MODEL") or "").strip() or "models/text-embedding-004"
                
                if google_key:
                    try:
                        from utils.google_inference import GoogleInferenceClient
                        google_client = GoogleInferenceClient(api_key=google_key)
                        score, matches, explanation = await score_predicted_questions_semantic(
                            predicted_questions,
                            actual_questions,
                            google_client=google_client,
                            embed_model=embed_model,
                        )
                    except Exception as e:
                        # Fallback to lexical if semantic fails - log warning
                        import sys
                        print(f"⚠️ WARNING: Semantic backtest scoring failed, using lexical similarity: {e}", file=sys.stderr)
                        score, matches, explanation = score_predicted_questions(predicted_questions, actual_questions)
                        # Add warning to explanation
                        explanation = f"⚠️ **Note:** Semantic similarity unavailable, using lexical matching.\n\n{explanation}"
                else:
                    # Use lexical similarity if no Google key - log warning
                    import sys
                    print("⚠️ WARNING: GOOGLE_AI_KEY not set, using lexical similarity for backtest (less accurate). Set GOOGLE_AI_KEY for semantic matching.", file=sys.stderr)
                    score, matches, explanation = score_predicted_questions(predicted_questions, actual_questions)
                    # Add warning to explanation
                    explanation = f"⚠️ **Note:** Using lexical similarity (word overlap) instead of semantic matching. Set GOOGLE_AI_KEY for better topic-based scoring.\n\n{explanation}"
                backtest_obj = BacktestResult(
                    transcript_url=transcript.get("transcript_url") or transcript_url,
                    transcript_found=True,
                    transcript_auto_detected=transcript_auto_detected,
                    questions_score_pct=score,
                    matches=[
                        {"predicted": p, "best_actual": a, "similarity": s}
                        for (p, a, s) in matches[:10]
                    ],
                    explanation=explanation,
                )
            else:
                # Transcript URL provided but not found
                backtest_obj = BacktestResult(
                    transcript_url=transcript_url,
                    transcript_found=False,
                    transcript_auto_detected=transcript_auto_detected,
                    questions_score_pct=0,
                    matches=[],
                    explanation=f"⚠️ Transcript URL provided but not found. The URL may be invalid or the transcript may not be available at: {transcript_url[:100]}",
                )

        # Collect warnings for user visibility
        warnings = []
        if prediction.model.provider == "fallback":
            warnings.append("⚠️ Using fallback/hardcoded predictions. Model not configured or failed.")
        if backtest_obj and not backtest_obj.transcript_found and transcript_url:
            warnings.append("⚠️ Transcript URL provided but transcript not found.")
        if backtest_obj and backtest_obj.explanation and "lexical" in backtest_obj.explanation.lower() and "semantic" in backtest_obj.explanation.lower():
            warnings.append("⚠️ Using lexical similarity for backtest (less accurate). Set GOOGLE_AI_KEY for semantic matching.")
        
        return jsonify(
            {
                "ok": True,
                "data": {
                    "prediction": prediction.model_dump(),
                    "backtest": backtest_obj.model_dump() if backtest_obj else None,
                    "brief_excerpt": brief_text[:7000],
                    "warnings": warnings,  # Always include warnings array for user visibility
                },
            }
        )
    except PredictionConfigError as e:
        return _json_error(str(e), status=500, code="predict_config")
    except Exception as e:
        return _json_error(f"Prediction failed: {e}", status=500, code="server_error")


if __name__ == "__main__":
    port = int(os.getenv("PORT") or "8000")
    app.run(host="0.0.0.0", port=port, debug=(os.getenv("APP_ENV") == "development"))


