"""
Streamlit version of SCOTUS AI Brief Predictor.

To run:
    streamlit run streamlit_app.py

To deploy:
    streamlit cloud deploy
    # Or use: streamlit run streamlit_app.py --server.port=8501
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import aiohttp
import streamlit as st
from dotenv import load_dotenv

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.backtest import extract_questions_from_transcript, score_predicted_questions, score_predicted_questions_semantic
from utils.pdf import extract_text_from_pdf_bytes
from utils.predictor import predict_votes_and_questions
from utils.schemas import BacktestResult
from utils.transcript_finder import find_transcript_urls, extract_case_name_from_hint

# Load environment
_ROOT = Path(__file__).parent
for env_file in [_ROOT / "env.local", _ROOT / ".env"]:
    if env_file.exists():
        load_dotenv(env_file)
        break

# Page config
st.set_page_config(
    page_title="SCOTUS AI: Legal Strategist",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "session" not in st.session_state:
    st.session_state.session = None


# Store session globally, but create it lazily in async context
_aiohttp_session: Optional[aiohttp.ClientSession] = None

async def get_session_async():
    """Get or create aiohttp session (must be called from async context)."""
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout = aiohttp.ClientTimeout(total=45)
        _aiohttp_session = aiohttp.ClientSession(timeout=timeout)
    return _aiohttp_session


def run_async(coro):
    """Run async function in Streamlit with proper event loop handling."""
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # We're in an async context, need to use nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
            # Now we can run the coro
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except ImportError:
            # nest_asyncio not available, use thread pool with new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)


def main():
    st.title("⚖️ SCOTUS AI: Legal Strategist")
    st.markdown("**AI-Powered Vote Predictions & Oral Argument Questions**")
    
    with st.sidebar:
        st.header("Configuration")
        google_key = st.text_input(
            "Google AI Key",
            value=os.getenv("GOOGLE_AI_KEY", ""),
            type="password",
            help="Get your key at https://aistudio.google.com/app/apikey"
        )
        if google_key:
            os.environ["GOOGLE_AI_KEY"] = google_key
        
        st.markdown("---")
        st.caption("Set environment variables or enter them above")
    
    # File upload
    st.header("📄 Upload Brief")
    uploaded_file = st.file_uploader(
        "Choose a PDF brief",
        type=["pdf"],
        help="Upload a PDF brief to get predictions"
    )
    
    if not uploaded_file:
        st.info("👆 Upload a PDF brief to get started")
        return
    
    # Form inputs
    col1, col2 = st.columns(2)
    with col1:
        uploader_side = st.selectbox(
            "Brief Side",
            ["UNKNOWN", "PETITIONER", "RESPONDENT", "AMICUS"],
            index=0
        )
    
    with col2:
        case_hint = st.text_input(
            "Case Hint (optional)",
            placeholder="e.g., Dobbs v. Jackson (2022)",
            help="Case name or docket number for transcript auto-detection"
        )
    
    transcript_url = st.text_input(
        "Transcript URL (optional)",
        placeholder="https://www.oyez.org/cases/...",
        help="For backtesting predicted questions against actual transcript"
    )
    
    # Auto-detect transcript if case hint provided
    if case_hint and not transcript_url:
        case_name = extract_case_name_from_hint(case_hint)
        if case_name:
            candidates = find_transcript_urls(case_name)
            if candidates:
                transcript_url = candidates[0]
                st.info(f"🔍 Auto-detected transcript: {transcript_url}")
    
    # Analyze button
    if st.button("🔍 Analyze Brief", type="primary", use_container_width=True):
        if not uploaded_file:
            st.error("Please upload a PDF brief first")
            return
        
        with st.spinner("📊 Analyzing brief and generating predictions..."):
            try:
                # Read PDF
                pdf_bytes = uploaded_file.read()
                brief_text = extract_text_from_pdf_bytes(pdf_bytes, max_chars=220_000)
                
                if not brief_text:
                    st.error("Could not extract text from PDF")
                    return
                
                # Get config
                corpus_path = os.getenv("HISTORICAL_CASES_PATH") or str(_ROOT / "data" / "historical_cases.jsonl")
                retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K") or "5")
                
                # Predict (session will be created inside async context)
                async def _predict():
                    session = await get_session_async()
                    return await predict_votes_and_questions(
                        session=session,
                        brief_text=brief_text,
                        uploader_side=uploader_side,
                        case_hint=case_hint,
                        corpus_path=corpus_path,
                        retrieval_top_k=retrieval_top_k,
                    )
                
                prediction = run_async(_predict())
                
                # Check for fallback
                is_fallback = prediction.model.provider == "fallback"
                if is_fallback:
                    st.warning(f"⚠️ **FALLBACK DATA**: {prediction.overall.why}")
                
                # Display results
                st.success("✅ Analysis complete!")
                
                # Overall prediction
                st.header("🎯 Overall Prediction")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Winner", prediction.overall.predicted_winner)
                with col2:
                    st.metric("Confidence", f"{prediction.overall.confidence * 100:.0f}%")
                with col3:
                    if prediction.overall.swing_justice:
                        st.metric("Swing Justice", prediction.overall.swing_justice)
                
                if prediction.overall.why:
                    st.info(prediction.overall.why)
                
                # Votes
                st.header("👨‍⚖️ Predicted Votes")
                vote_cols = st.columns(3)
                for idx, vote in enumerate(prediction.votes):
                    col_idx = idx % 3
                    with vote_cols[col_idx]:
                        vote_color = {
                            "PETITIONER": "🟦",
                            "RESPONDENT": "🟥",
                            "UNCERTAIN": "🟨"
                        }.get(vote.vote, "⚪")
                        st.markdown(f"**{vote.justice_name}**")
                        st.markdown(f"{vote_color} {vote.vote}")
                        st.caption(f"Confidence: {vote.confidence * 100:.0f}%")
                        if vote.rationale:
                            st.caption(vote.rationale)
                
                # Questions
                st.header("❓ Predicted Questions")
                for question in prediction.questions:
                    with st.expander(f"**{question.justice_name}**"):
                        st.markdown(f"**Question:** {question.question}")
                        if question.what_it_tests:
                            st.caption(f"*What it tests:* {question.what_it_tests}")
                
                # Backtest
                if transcript_url:
                    st.header("📊 Backtest Results")
                    with st.spinner("Fetching transcript and scoring questions..."):
                        from utils.transcripts import fetch_transcript_text
                        from utils.google_inference import GoogleInferenceClient
                        
                        async def _fetch_transcript():
                            session = await get_session_async()
                            return await fetch_transcript_text(session, transcript_url=transcript_url)
                        
                        transcript = run_async(_fetch_transcript())
                        
                        if transcript.get("transcript_found"):
                            actual_questions = extract_questions_from_transcript(
                                transcript.get("transcript_text") or "", limit=200
                            )
                            predicted_questions = [q.question for q in prediction.questions]
                            
                            # Try semantic scoring
                            google_key = os.getenv("GOOGLE_AI_KEY", "")
                            embed_model = os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
                            
                            if google_key:
                                try:
                                    google_client = GoogleInferenceClient(api_key=google_key)
                                    score, matches, explanation = run_async(
                                        score_predicted_questions_semantic(
                                            predicted_questions,
                                            actual_questions,
                                            google_client=google_client,
                                            embed_model=embed_model,
                                        )
                                    )
                                except Exception:
                                    from utils.backtest import score_predicted_questions
                                    score, matches, explanation = score_predicted_questions(
                                        predicted_questions, actual_questions
                                    )
                            else:
                                from utils.backtest import score_predicted_questions
                                score, matches, explanation = score_predicted_questions(
                                    predicted_questions, actual_questions
                                )
                            
                            st.metric("Question Match Score", f"{score}%")
                            st.markdown(explanation)
                            
                            # Show top matches
                            if matches:
                                with st.expander("Top Question Matches"):
                                    for pred_q, actual_q, sim in matches[:5]:
                                        st.markdown(f"**Predicted:** {pred_q}")
                                        st.markdown(f"**Actual:** {actual_q}")
                                        st.progress(sim, text=f"Similarity: {sim * 100:.0f}%")
                                        st.divider()
                        else:
                            st.warning(f"⚠️ Transcript not found at: {transcript_url}")
                
                # Retrieved cases
                if prediction.retrieved_cases:
                    st.header("📚 Similar Historical Cases")
                    for case in prediction.retrieved_cases:
                        st.markdown(f"**{case.case_name}**")
                        if case.term:
                            st.caption(f"Term {case.term}")
                        if case.tags:
                            st.caption(f"Tags: {', '.join(case.tags)}")
                        st.divider()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()

