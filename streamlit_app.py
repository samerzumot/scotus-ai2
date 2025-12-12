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
from utils.predictor import predict_votes_and_questions, _load_corpus_cached, _build_prompt, _coerce_prediction
from utils.schemas import BacktestResult
from utils.transcript_finder import find_transcript_urls, extract_case_name_from_hint
from utils.transcripts import fetch_transcript_text

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


# For Streamlit, we create a fresh session for each operation to avoid event loop issues
async def get_session_async():
    """Create a new aiohttp session (must be called from async context)."""
    # Reduced timeout for faster failure detection
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    return aiohttp.ClientSession(timeout=timeout)


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
    
    # Sample briefs section
    st.header("📄 Upload Brief")
    
    # Load sample briefs
    sample_briefs_path = _ROOT / "data" / "sample_briefs.json"
    sample_briefs = []
    if sample_briefs_path.exists():
        try:
            import json
            with open(sample_briefs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                sample_briefs = data.get("sample_briefs", [])
        except Exception:
            pass
    
    # Show sample briefs if available
    use_sample = False
    selected_sample = None
    
    if sample_briefs:
        st.subheader("📚 Quick Start: Sample Briefs")
        st.caption("Try these pre-loaded briefs with backtest data (no upload needed)")
        
        sample_options = ["-- Select a sample brief --"] + [f"{s['case_name']} ({s.get('term', 'N/A')})" for s in sample_briefs]
        selected_sample_name = st.selectbox(
            "Choose a sample brief",
            sample_options,
            key="sample_brief_selector"
        )
        
        if selected_sample_name != "-- Select a sample brief --":
            selected_sample = next((s for s in sample_briefs if f"{s['case_name']} ({s.get('term', 'N/A')})" == selected_sample_name), None)
            if selected_sample:
                use_sample = True
                st.success(f"✅ Selected: **{selected_sample['case_name']}**")
                if selected_sample.get('summary'):
                    st.info(f"📋 *{selected_sample['summary']}*")
    
    st.divider()
    
    # File upload
    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader(
            "Or upload your own PDF brief",
            type=["pdf"],
            help="Upload a PDF brief to get predictions"
        )
    
    if not use_sample and not uploaded_file:
        st.info("👆 Select a sample brief above or upload your own PDF to get started")
        return
    
    # Form inputs (only show if not using sample)
    if not use_sample:
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
    else:
        # Use sample values (will be set in analyze section)
        uploader_side = "UNKNOWN"
        case_hint = ""
        transcript_url = ""
    
    # Analyze button
    if st.button("🔍 Analyze Brief", type="primary", use_container_width=True):
        if not use_sample and not uploaded_file:
            st.error("Please select a sample brief or upload a PDF first")
            return
        
        # Get brief text and metadata
        if use_sample and selected_sample:
            brief_text = selected_sample.get("brief_text", "")
            uploader_side = selected_sample.get("uploader_side", "UNKNOWN")
            case_hint = selected_sample.get("case_hint", selected_sample.get("case_name", ""))
            transcript_url = selected_sample.get("transcript_url", "")
            
            if not brief_text:
                st.error("Sample brief text is missing")
                return
            
            st.info(f"📄 Using sample brief: **{selected_sample['case_name']}**")
        else:
            # Read PDF
            with st.spinner("Reading PDF..."):
                pdf_bytes = uploaded_file.read()
                brief_text = extract_text_from_pdf_bytes(pdf_bytes, max_chars=220_000)
                
                if not brief_text:
                    st.error("Could not extract text from PDF")
                    return
        
        # Get config
        corpus_path = os.getenv("HISTORICAL_CASES_PATH") or str(_ROOT / "data" / "historical_cases.jsonl")
        retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K") or "5")
        
        # Predict with detailed progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        detail_text = st.empty()
        
        async def _predict():
                    session = await get_session_async()
                    try:
                        import asyncio
                        from utils.google_inference import GoogleInferenceClient
                        from utils.retrieval import retrieve_similar_cases
                        from utils.schemas import ModelInfo, RetrievedCaseRef
                        
                        # Step 1: Initialize Google client
                        status_text.text("🔧 Initializing AI model...")
                        detail_text.text("Setting up Google Gemini API client")
                        progress_bar.progress(5)
                        await asyncio.sleep(0.1)  # Allow UI to update
                        
                        google_key = os.getenv("GOOGLE_AI_KEY", "").strip()
                        if not google_key:
                            raise RuntimeError("GOOGLE_AI_KEY not set")
                        
                        predict_model = os.getenv("GOOGLE_PREDICT_MODEL", "models/gemini-2.5-pro").strip() or "models/gemini-2.5-pro"
                        if not predict_model.startswith("models/"):
                            predict_model = f"models/{predict_model}"
                        embed_model = os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004").strip() or "models/text-embedding-004"
                        
                        google_client = GoogleInferenceClient(api_key=google_key)
                        
                        # Step 2: Load historical cases corpus
                        status_text.text("📚 Loading historical cases corpus...")
                        detail_text.text(f"Reading cases from {corpus_path}")
                        progress_bar.progress(10)
                        await asyncio.sleep(0.1)
                        
                        cases = await _load_corpus_cached(
                            corpus_path=corpus_path,
                            google_client=google_client,
                            embed_model=embed_model,
                        )
                        detail_text.text(f"Loaded {len(cases)} historical cases")
                        progress_bar.progress(20)
                        await asyncio.sleep(0.1)
                        
                        # Step 3: Generate embeddings and retrieve similar cases
                        status_text.text("🔍 Finding similar historical cases...")
                        detail_text.text("Generating embeddings and computing similarity")
                        progress_bar.progress(30)
                        await asyncio.sleep(0.1)
                        
                        retrieved_results = await retrieve_similar_cases(
                            brief_text=brief_text,
                            cases=cases,
                            top_k=retrieval_top_k,
                            google_client=google_client,
                            embed_model=embed_model,
                        )
                        detail_text.text(f"Found {len(retrieved_results)} similar cases")
                        progress_bar.progress(45)
                        await asyncio.sleep(0.1)
                        
                        retrieved_refs = [
                            RetrievedCaseRef(
                                case_id=r.case.case_id,
                                case_name=r.case.case_name,
                                term=r.case.term,
                                tags=r.case.tags,
                                outcome=r.case.outcome
                            )
                            for r in retrieved_results
                        ]
                        
                        # Step 4: Build prompt
                        status_text.text("📝 Preparing analysis prompt...")
                        detail_text.text("Combining brief text with historical context")
                        progress_bar.progress(55)
                        await asyncio.sleep(0.1)
                        
                        model_info = ModelInfo(
                            provider="google",
                            predict_model=predict_model,
                            embed_model=embed_model,
                            retrieval_top_k=retrieval_top_k
                        )
                        
                        prompt = _build_prompt(
                            brief_text=brief_text,
                            uploader_side=(uploader_side or "UNKNOWN").strip().upper(),
                            case_hint=case_hint,
                            retrieved=retrieved_results,
                        )
                        
                        # Step 5: Generate predictions with AI
                        status_text.text("🤖 Generating predictions with AI...")
                        detail_text.text(f"Using {predict_model} to analyze brief and predict votes/questions")
                        progress_bar.progress(65)
                        await asyncio.sleep(0.1)
                        
                        system_instruction = """You are SCOTUS AI, a legal prediction system. Analyze the brief and predict:
1. How each of the 9 Justices will vote (PETITIONER/RESPONDENT/UNCERTAIN)
2. One tough oral-argument question each Justice is likely to ask

CRITICAL: Confidence values must be decimals between 0.0 and 1.0 (NOT percentages).
- 0.85 = 85% confidence (CORRECT)
- 85 = 8500% confidence (WRONG - will cause validation error)

Return ONLY valid JSON matching the exact schema provided. No markdown, no explanations outside the JSON."""
                        
                        obj = await google_client.generate_json(
                            model=predict_model,
                            prompt=prompt,
                            system_instruction=system_instruction,
                            temperature=0.2,
                            max_output_tokens=8192,
                        )
                        
                        # Step 6: Validate and normalize results
                        status_text.text("✅ Validating and processing results...")
                        detail_text.text("Normalizing confidence values and validating predictions")
                        progress_bar.progress(85)
                        await asyncio.sleep(0.1)
                        
                        prediction = _coerce_prediction(
                            obj,
                            uploader_side=uploader_side,
                            model_info=model_info,
                            retrieved=retrieved_refs
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        detail_text.text(f"Generated predictions for {len(prediction.votes)} justices")
                        await asyncio.sleep(0.5)  # Show completion message briefly
                        
                        return prediction
                        
                    except asyncio.TimeoutError:
                        status_text.text("⏱️ Analysis timed out")
                        detail_text.text("The operation took longer than 60 seconds")
                        raise RuntimeError("Analysis timed out after 60 seconds. The brief may be too long or the API is slow.")
                    except Exception as e:
                        status_text.text("❌ Error during analysis")
                        detail_text.text(f"Error: {str(e)[:100]}")
                        raise
                    finally:
                        await session.close()
        
        try:
            prediction = run_async(_predict())
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            detail_text.empty()
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 **Tips:**\n- Try a shorter brief\n- Check your Google AI key\n- The API may be experiencing delays")
            return
        
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
        
        # Run backtest first if transcript is available (to get matches for questions)
        matches_dict = {}  # Map predicted question text to match info
        backtest_score = None
        backtest_explanation = None
        
        if transcript_url:
            st.header("📊 Backtest Results")
            backtest_progress = st.progress(0)
            backtest_status = st.empty()
            
            async def _fetch_transcript():
                session = await get_session_async()
                try:
                    backtest_status.text("📥 Fetching transcript (checking cache first)...")
                    backtest_progress.progress(10)
                    
                    import asyncio
                    # Aggressive optimizations: 100k chars, 8s timeout, caching enabled
                    fetch_task = fetch_transcript_text(
                        session,
                        transcript_url=transcript_url,
                        max_chars=100_000,  # Reduced to 100k - questions are in first 30-40%
                        use_cache=True,  # Use cache for instant results on repeat
                    )
                    transcript = await asyncio.wait_for(fetch_task, timeout=8.0)  # Reduced to 8s
                    
                    backtest_progress.progress(50)
                    return transcript
                except asyncio.TimeoutError:
                    backtest_status.text("⏱️ Transcript fetch timed out (>8s)")
                    raise RuntimeError("Transcript fetch timed out after 8 seconds")
                finally:
                    await session.close()
            
            try:
                transcript = run_async(_fetch_transcript())
            except Exception as e:
                backtest_progress.empty()
                backtest_status.empty()
                st.warning(f"⚠️ Could not fetch transcript: {str(e)}")
                transcript = None
                        
                        if transcript and transcript.get("transcript_found"):
                backtest_status.text("🔍 Extracting questions from transcript...")
                backtest_progress.progress(60)
                
                # Extract questions (limit to 100 for faster processing)
                actual_questions = extract_questions_from_transcript(
                    transcript.get("transcript_text") or "", limit=100
                )
                predicted_questions = [q.question for q in prediction.questions]
                
                backtest_status.text("📊 Scoring questions...")
                backtest_progress.progress(80)
                
                # Try semantic scoring
                google_key = os.getenv("GOOGLE_AI_KEY", "")
                embed_model = os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
                
                if google_key:
                    try:
                        google_client = GoogleInferenceClient(api_key=google_key)
                        
                        async def _score():
                            import asyncio
                            score_task = score_predicted_questions_semantic(
                                predicted_questions,
                                actual_questions,
                                google_client=google_client,
                                embed_model=embed_model,
                            )
                            return await asyncio.wait_for(score_task, timeout=30.0)
                        
                        score, matches, explanation = run_async(_score())
                        backtest_progress.progress(100)
                        backtest_status.empty()
                    except Exception:
                        from utils.backtest import score_predicted_questions
                        score, matches, explanation = score_predicted_questions(
                            predicted_questions, actual_questions
                        )
                        backtest_progress.progress(100)
                        backtest_status.empty()
                else:
                    from utils.backtest import score_predicted_questions
                    score, matches, explanation = score_predicted_questions(
                        predicted_questions, actual_questions
                    )
                    backtest_progress.progress(100)
                    backtest_status.empty()
                
                # Store backtest results
                backtest_score = score
                backtest_explanation = explanation
                
                # Create mapping from predicted question to match info
                for m in matches:
                    if isinstance(m, dict):
                        pred_q = m.get("predicted", "")
                        if pred_q:
                            matches_dict[pred_q] = m
                    else:
                        # Tuple format (legacy)
                        pred_q = m[0] if len(m) > 0 else ""
                        if pred_q:
                            matches_dict[pred_q] = {
                                "best_actual": m[1] if len(m) > 1 else "",
                                "similarity": m[2] if len(m) > 2 else 0.0,
                            }
                
                st.metric("Question Match Score", f"{score}%")
                st.markdown(explanation)
                
                # Show top matches
                if matches:
                    with st.expander("Top Question Matches"):
                        for m in matches[:5]:
                            # Handle both dict and tuple formats for backward compatibility
                            if isinstance(m, dict):
                                pred_q = m.get("predicted", "")
                                actual_q = m.get("best_actual", "")
                                sim = m.get("similarity", 0.0)
                                justice_name = m.get("justice_name", "")
                                pred_citations = m.get("predicted_citations", [])
                                actual_citations = m.get("actual_citations", [])
                            else:
                                # Tuple format (legacy)
                                pred_q, actual_q, sim = m if len(m) >= 3 else (m[0] if len(m) > 0 else "", m[1] if len(m) > 1 else "", 0.0)
                                justice_name = ""
                                pred_citations = []
                                actual_citations = []
                            
                            # Display justice name if available
                            if justice_name:
                                st.markdown(f"**Justice {justice_name}**")
                            st.markdown(f"**Predicted:** {pred_q}")
                            if pred_citations:
                                st.caption(f"📚 Citations: {', '.join(pred_citations)}")
                            st.markdown(f"**Best Actual Match:** {actual_q}")
                            if actual_citations:
                                st.caption(f"📚 Citations: {', '.join(actual_citations)}")
                            st.progress(sim, text=f"Similarity: {sim * 100:.0f}%")
                            st.divider()
            elif transcript:
                backtest_progress.empty()
                backtest_status.empty()
                st.warning(f"⚠️ Transcript not found at: {transcript_url}")
        
        # Questions with matches
        st.header("❓ Predicted Questions")
        for question in prediction.questions:
            match_info = matches_dict.get(question.question, None)
            
            # Build expander title
            title = f"**{question.justice_name}**"
            if match_info:
                sim = match_info.get("similarity", 0.0) if isinstance(match_info, dict) else 0.0
                title += f" ({sim * 100:.0f}% match)"
            
            with st.expander(title):
                st.markdown(f"**Predicted Question:** {question.question}")
                
                # Show matching actual question if available
                if match_info:
                    if isinstance(match_info, dict):
                        best_actual = match_info.get("best_actual", "")
                        similarity = match_info.get("similarity", 0.0)
                        actual_citations = match_info.get("actual_citations", [])
                        
                        if best_actual:
                            st.divider()
                            st.markdown("**📋 Best Matching Actual Question:**")
                            st.markdown(f"{best_actual}")
                            if actual_citations:
                                st.caption(f"📚 Citations: {', '.join(actual_citations)}")
                            st.progress(similarity, text=f"Similarity: {similarity * 100:.0f}%")
                
                if question.what_it_tests:
                    st.divider()
                    st.caption(f"*What it tests:* {question.what_it_tests}")
        
        # Backtest summary (if available)
        if backtest_score is not None:
            st.header("📊 Backtest Summary")
            st.metric("Question Match Score", f"{backtest_score}%")
            if backtest_explanation:
                st.markdown(backtest_explanation)
            
            # Show top matches in detail
            if matches_dict:
                with st.expander("📋 All Question Matches"):
                    for m in list(matches_dict.values())[:10]:
                        if isinstance(m, dict):
                            pred_q = m.get("predicted", "")
                            actual_q = m.get("best_actual", "")
                            sim = m.get("similarity", 0.0)
                            justice_name = m.get("justice_name", "")
                            
                            if pred_q and actual_q:
                                st.markdown(f"**{justice_name if justice_name else 'Question'}**")
                                st.markdown(f"*Predicted:* {pred_q}")
                                st.markdown(f"*Actual:* {actual_q}")
                                st.progress(sim, text=f"Similarity: {sim * 100:.0f}%")
                                st.divider()
        
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


if __name__ == "__main__":
    main()

