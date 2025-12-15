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

import re

from utils.backtest import extract_questions_from_transcript, score_predicted_questions, score_predicted_questions_semantic
from utils.google_inference import GoogleInferenceClient
from utils.pdf import extract_text_from_pdf_bytes
from utils.predictor import predict_votes_and_questions, _load_corpus_cached, _build_prompt, _coerce_prediction
from utils.schemas import BacktestResult
from utils.semantic_matcher import analyze_question_semantic_match
from utils.topic_extractor import extract_key_topics, find_topic_mentions_in_transcript
from utils.scotus import JUSTICE_NAMES
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

# Professional legal-themed styling
st.markdown("""
<style>
/* Global typography improvements */
@import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Source+Sans+Pro:wght@400;600&display=swap');

/* Headers with serif font for legal authority */
h1 {
    font-family: 'Merriweather', Georgia, serif !important;
    color: #1a1a2e !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

h2 {
    font-family: 'Merriweather', Georgia, serif !important;
    color: #2d3748 !important;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.5rem;
    margin-top: 2rem !important;
}

h3 {
    font-family: 'Source Sans Pro', sans-serif !important;
    color: #4a5568 !important;
    font-weight: 600 !important;
}

/* Better metric styling - professional solid colors */
[data-testid="stMetric"] {
    background: #1a365d;
    padding: 1rem;
    border-radius: 8px;
    color: white;
}

[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.85) !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: white !important;
    font-weight: 700;
}

/* Cleaner expanders */
.streamlit-expanderHeader {
    font-family: 'Source Sans Pro', sans-serif !important;
    font-weight: 600 !important;
    background-color: #f7fafc;
    border-radius: 8px;
}

/* Button styling - professional solid */
.stButton > button {
    background: #1a365d;
    color: white;
    font-weight: 600;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 8px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #2c5282;
    box-shadow: 0 2px 8px rgba(26, 54, 93, 0.2);
}

/* Info/Success/Warning boxes */
.stAlert {
    border-radius: 8px;
    border-left-width: 4px;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Spacing improvements */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Custom vote cards for bench layout */
.vote-card {
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s ease;
}

.vote-card:hover {
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session" not in st.session_state:
    st.session_state.session = None


# For Streamlit, we create a fresh session for each operation to avoid event loop issues
async def get_session_async():
    """Create a new aiohttp session (must be called from async context)."""
    # Reduced timeout for faster failure detection
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    return aiohttp.ClientSession(timeout=timeout)


def _extract_question_snippet(transcript_text: str, question_text: str, question_index: Optional[int] = None) -> Optional[str]:
    """
    Extract a simple snippet: 500 chars before and after the matched question text.
    Bold the matching question text in the snippet.
    If question_index is provided, uses that to find the question in extracted questions list.
    
    Returns the context snippet with bolded matching text, or None if not found.
    """
    if not transcript_text or not question_text:
        return None
    
    # Find the question in the transcript (case-insensitive)
    question_lower = question_text.lower().strip()
    transcript_lower = transcript_text.lower()
    
    # Try to find the question text
    question_pos = transcript_lower.find(question_lower)
    
    # If not found, try finding a shorter substring (first 50 chars)
    if question_pos == -1 and len(question_lower) > 50:
        question_pos = transcript_lower.find(question_lower[:50])
    
    if question_pos == -1:
        return None
    
    # Simple: extract 500 chars before and after the match
    start = max(0, question_pos - 500)
    end = min(len(transcript_text), question_pos + len(question_text) + 500)
    
    snippet = transcript_text[start:end].strip()
    
    # Find the actual position in the original (non-lowercase) text
    actual_start = start + question_pos - (transcript_lower[:question_pos].rfind(transcript_lower[start:question_pos]))
    # More reliable: find the match in the snippet itself
    snippet_lower = snippet.lower()
    match_in_snippet = snippet_lower.find(question_lower)
    
    if match_in_snippet != -1:
        # Bold the matching text
        match_start = match_in_snippet
        match_end = match_start + len(question_text)
        before_match = snippet[:match_start]
        matched_text = snippet[match_start:match_end]
        after_match = snippet[match_end:]
        snippet = f"{before_match}**{matched_text}**{after_match}"
    
    # Clean up whitespace
    snippet = re.sub(r'[ \t]+', ' ', snippet)
    snippet = re.sub(r'\n{3,}', '\n\n', snippet)
    
    return snippet


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
    # Clean header
    st.title("⚖️ SCOTUS AI")
    st.markdown("*Predict Supreme Court votes and oral argument questions from legal briefs*")
    
    # Sidebar for configuration (hidden by default)
    with st.sidebar:
        st.header("Settings")
        google_key = st.text_input(
            "Google AI Key",
            value=os.getenv("GOOGLE_AI_KEY", ""),
            type="password",
            help="Required for fresh predictions. Get key at aistudio.google.com"
        )
        if google_key:
            os.environ["GOOGLE_AI_KEY"] = google_key
    
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
    
    # Two-column layout for input selection
    use_sample = False
    selected_sample = None
    uploaded_file = None
    
    # Input section
    st.header("Select a Case")
    
    if sample_briefs:
        # Sample brief selection
        sample_options = ["Select a case..."] + [f"{s['case_name']} ({s.get('term', 'N/A')})" for s in sample_briefs]
        selected_sample_name = st.selectbox(
            "Pre-loaded cases with backtest data",
            sample_options,
            key="sample_brief_selector",
            label_visibility="collapsed"
        )
        
        if selected_sample_name != "Select a case...":
            selected_sample = next((s for s in sample_briefs if f"{s['case_name']} ({s.get('term', 'N/A')})" == selected_sample_name), None)
            if selected_sample:
                use_sample = True
                # Show case summary in a styled card
                st.markdown(f"""
                <div style="background: #f8fafc; border-left: 4px solid #3182ce; padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
                    <strong>{selected_sample['case_name']}</strong><br>
                    <span style="color: #718096;">{selected_sample.get('summary', '')}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # File upload (collapsible)
    if not use_sample:
        with st.expander("Or upload your own brief (PDF)", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                key="brief_uploader",
                label_visibility="collapsed"
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
                    st.info(f"Auto-detected transcript: {transcript_url}")
    else:
        # Use sample values (will be set in analyze section)
        uploader_side = "UNKNOWN"
        case_hint = ""
        # Allow user to enter/override transcript URL for sample briefs
        default_transcript_url = selected_sample.get("transcript_url", "") if selected_sample else ""
        transcript_url = st.text_input(
            "Transcript URL (optional)",
            value=default_transcript_url,
            placeholder="https://www.oyez.org/cases/...",
            help="For backtesting predicted questions against actual transcript. Pre-filled from sample if available.",
            key="sample_transcript_url"
        )
    
    # Analyze button
    if st.button("Analyze Brief", type="primary", use_container_width=True):
        if not use_sample and not uploaded_file:
            st.error("Please select a sample brief or upload a PDF first")
            return
        
        # Get brief text and metadata
        use_precomputed = False
        prediction = None
        precomputed_prediction = None
        
        if use_sample and selected_sample:
            brief_text = selected_sample.get("brief_text", "")
            uploader_side = selected_sample.get("uploader_side", "UNKNOWN")
            case_hint = selected_sample.get("case_hint", selected_sample.get("case_name", ""))
            transcript_url = selected_sample.get("transcript_url", "")
            precomputed_prediction = selected_sample.get("precomputed_prediction")
            
            # Set use_precomputed to True if we have cached data
            if precomputed_prediction:
                use_precomputed = True
            
            if not brief_text:
                st.error("Sample brief text is missing")
                return
            
            st.info(f"Using sample brief: **{selected_sample['case_name']}**")
            
            # Use precomputed prediction if available (instant results - skip all loading steps)
            if use_precomputed and precomputed_prediction:
                from utils.schemas import VoteQuestionPrediction
                try:
                    prediction = VoteQuestionPrediction.model_validate(precomputed_prediction)
                    # Skip directly to displaying results - no progress bars, no API calls
                except Exception as e:
                    st.warning(f"Could not load precomputed prediction: {e}. Generating fresh prediction...")
                    use_precomputed = False
                    precomputed_prediction = None  # Fall through to generate fresh
        else:
            # Read PDF from uploaded file
            with st.spinner("Reading PDF..."):
                pdf_bytes = uploaded_file.read()
                brief_text = extract_text_from_pdf_bytes(pdf_bytes, max_chars=220_000)
                
                if not brief_text:
                    st.error("Could not extract text from PDF")
                    return
        
        # If we have a precomputed prediction, skip ALL API calls and loading steps
        if use_precomputed and prediction is not None:
            # Skip directly to displaying results - no progress indicators, no corpus loading, no API calls
            # Do NOT create any progress bars or status messages - go straight to results
            pass
        else:
            # Generate fresh prediction with detailed progress indicators
            # Get config
            corpus_path = os.getenv("HISTORICAL_CASES_PATH") or str(_ROOT / "data" / "historical_cases.jsonl")
            retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K") or "5")
            
            # Only show progress indicators for fresh predictions (not precomputed)
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
                    status_text.text("Loading historical cases corpus...")
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
                    status_text.text("Finding similar historical cases...")
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
                            outcome=r.case.outcome,
                            transcript_url=r.case.transcript_url,
                            docket=r.case.docket
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
                    status_text.text("Generating predictions with AI...")
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
                    status_text.text("Validating and processing results...")
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
                    status_text.text("Analysis complete!")
                    detail_text.text(f"Generated predictions for {len(prediction.votes)} justices")
                    await asyncio.sleep(0.5)  # Show completion message briefly
                    
                    return prediction
                    
                except asyncio.TimeoutError:
                    status_text.text("Analysis timed out")
                    detail_text.text("The operation took longer than 60 seconds")
                    raise RuntimeError("Analysis timed out after 60 seconds. The brief may be too long or the API is slow.")
                except Exception as e:
                    status_text.text("Error during analysis")
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
                st.error(f"Analysis failed: {str(e)}")
                st.info("**Tips:**\n- Try a shorter brief\n- Check your Google AI key\n- The API may be experiencing delays")
                return
        
        # Check for fallback
        is_fallback = prediction.model.provider == "fallback"
        if is_fallback:
            st.warning(f"**FALLBACK DATA**: {prediction.overall.why}")
        
        # Display results
        st.success("Analysis complete!")
        
        # Check for fallback
        is_fallback = prediction.model.provider == "fallback"
        if is_fallback:
            st.warning(f"**FALLBACK DATA**: {prediction.overall.why}")
        
        # Overall prediction
        st.header("Overall Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Winner", prediction.overall.predicted_winner)
        with col2:
            st.metric("Confidence", f"{prediction.overall.confidence * 100:.0f}%")
        with col3:
            if prediction.overall.swing_justice:
                # Convert justice_id to full name with proper capitalization
                swing_id = prediction.overall.swing_justice.lower().strip()
                swing_name = JUSTICE_NAMES.get(swing_id, swing_id.title())
                st.metric("Swing Justice", swing_name)
        
        if prediction.overall.why:
            st.info(prediction.overall.why)
        
        # Votes - Visual Bench Layout
        st.header("Predicted Votes")
        
        # Create a visual bench layout (traditional SCOTUS seating)
        # Bench order from left to right as viewed from audience:
        # Row 1 (back): Thomas, Sotomayor, Alito, Kagan
        # Row 2 (front): Jackson, Kavanaugh, Roberts (center), Gorsuch, Barrett
        
        # Build vote lookup by justice_id
        votes_by_id = {v.justice_id.lower(): v for v in prediction.votes}
        
        def get_vote_style(vote_value, confidence):
            """Return HTML styling for vote display."""
            colors = {
                "PETITIONER": ("#1e40af", "🟦"),  # Blue
                "RESPONDENT": ("#dc2626", "🟥"),  # Red
                "UNCERTAIN": ("#ca8a04", "🟨"),   # Yellow
            }
            color, emoji = colors.get(vote_value, ("#6b7280", "⚪"))
            opacity = 0.4 + (confidence * 0.6)  # Scale opacity with confidence
            return color, emoji, opacity
        
        # Count votes for summary
        pet_votes = sum(1 for v in prediction.votes if v.vote == "PETITIONER")
        resp_votes = sum(1 for v in prediction.votes if v.vote == "RESPONDENT")
        uncertain_votes = sum(1 for v in prediction.votes if v.vote == "UNCERTAIN")
        
        # Display vote count summary
        vote_summary_cols = st.columns(3)
        with vote_summary_cols[0]:
            st.metric("🟦 Petitioner", pet_votes)
        with vote_summary_cols[1]:
            st.metric("🟥 Respondent", resp_votes)
        with vote_summary_cols[2]:
            st.metric("🟨 Uncertain", uncertain_votes)
        
        st.markdown("---")
        st.markdown("**Supreme Court Bench Layout** *(as viewed from audience)*")
        
        def render_justice_box(justice_id, display_name, is_chief=False):
            """Render a justice vote box using Streamlit native components."""
            vote = votes_by_id.get(justice_id.lower())
            if vote:
                color, emoji, _ = get_vote_style(vote.vote, vote.confidence)
                chief_marker = " 👑" if is_chief else ""
                vote_label = vote.vote[:3]  # PET/RES/UNC
                st.markdown(
                    f"""<div style="background-color: {color}; color: white; padding: 10px; 
                    border-radius: 8px; text-align: center; margin: 2px;">
                    <strong>{emoji} {display_name}{chief_marker}</strong><br>
                    <small>{vote_label} • {vote.confidence * 100:.0f}%</small>
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style="background-color: #6b7280; color: white; padding: 10px; 
                    border-radius: 8px; text-align: center; margin: 2px;">
                    <strong>{display_name}</strong><br><small>N/A</small>
                    </div>""",
                    unsafe_allow_html=True
                )
        
        # Back row (4 justices) - more senior by service
        back_cols = st.columns(4)
        with back_cols[0]:
            render_justice_box("thomas", "Thomas")
        with back_cols[1]:
            render_justice_box("sotomayor", "Sotomayor")
        with back_cols[2]:
            render_justice_box("alito", "Alito")
        with back_cols[3]:
            render_justice_box("kagan", "Kagan")
        
        # Front row (5 justices) - Roberts center
        front_cols = st.columns(5)
        with front_cols[0]:
            render_justice_box("jackson", "Jackson")
        with front_cols[1]:
            render_justice_box("kavanaugh", "Kavanaugh")
        with front_cols[2]:
            render_justice_box("roberts", "Roberts", is_chief=True)
        with front_cols[3]:
            render_justice_box("gorsuch", "Gorsuch")
        with front_cols[4]:
            render_justice_box("barrett", "Barrett")
        
        # Detailed vote cards below
        st.markdown("---")
        st.subheader("Detailed Vote Predictions")
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
        matches_by_justice = {}  # Map justice_id to match info (more reliable lookup)
        backtest_score = None
        backtest_explanation = None
        transcript = None  # Initialize to avoid NameError when using precomputed backtest
        
        # Check for precomputed backtest results (for sample briefs)
        precomputed_backtest = None
        if use_sample and selected_sample:
            precomputed_backtest = selected_sample.get("precomputed_backtest")
        
        if precomputed_backtest:
            # Use precomputed backtest results
            backtest_score = precomputed_backtest.get("questions_score_pct", 0)
            backtest_explanation = precomputed_backtest.get("explanation", "")
            matches = precomputed_backtest.get("matches", [])
            
            # Create mapping by justice_id for finding matches later
            matches_by_justice = {}
            for m in matches:
                if isinstance(m, dict):
                    pred_q = m.get("predicted", "")
                    justice_id = m.get("justice_id", "")
                    if pred_q:
                        matches_dict[pred_q] = m
                    if justice_id:
                        matches_by_justice[justice_id] = m
            
            # Display backtest results with matches
            st.header("📊 Backtest Results")
            st.metric("Question Match Score", f"{backtest_score}%")
            if backtest_explanation:
                st.markdown(backtest_explanation)
            

        elif transcript_url:
            st.header("📊 Backtest Results")
            backtest_progress = st.progress(0)
            backtest_status = st.empty()
            
            async def _fetch_transcript():
                session = await get_session_async()
                try:
                    backtest_status.text("Fetching transcript (checking cache first)...")
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
                    backtest_status.text("Transcript fetch timed out (>8s)")
                    raise RuntimeError("Transcript fetch timed out after 8 seconds")
                finally:
                    await session.close()
            
            try:
                transcript = run_async(_fetch_transcript())
            except Exception as e:
                backtest_progress.empty()
                backtest_status.empty()
                st.warning(f"Could not fetch transcript: {str(e)}")
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
                
                # Always use semantic scoring with Google embeddings
                google_key = (os.getenv("GOOGLE_AI_KEY") or "").strip()
                embed_model = (os.getenv("GOOGLE_EMBED_MODEL") or "").strip() or "models/text-embedding-004"
                
                if not google_key:
                    score, matches, explanation = 0, [], "Backtest requires GOOGLE_AI_KEY for semantic similarity. Please configure it in env.local or in the sidebar above."
                    backtest_progress.progress(100)
                    backtest_status.empty()
                elif not embed_model:
                    score, matches, explanation = 0, [], "Backtest requires GOOGLE_EMBED_MODEL. Defaulting to models/text-embedding-004."
                    embed_model = "models/text-embedding-004"
                    # Continue with default
                    try:
                        google_client = GoogleInferenceClient(api_key=google_key)
                        
                        async def _score():
                            import asyncio
                            score_task = score_predicted_questions_semantic(
                                predicted_questions,
                                actual_questions,
                                google_client=google_client,
                                embed_model=embed_model,
                                use_gemini_for_selection=True,  # Use Gemini for final selection
                            )
                            return await asyncio.wait_for(score_task, timeout=30.0)
                        
                        score, matches, explanation = run_async(_score())
                        backtest_progress.progress(100)
                        backtest_status.empty()
                    except Exception as e:
                        backtest_progress.progress(100)
                        backtest_status.empty()
                        score, matches, explanation = 0, [], f"Semantic backtest failed: {str(e)}. Please check GOOGLE_AI_KEY and GOOGLE_EMBED_MODEL configuration."
                else:
                    try:
                        google_client = GoogleInferenceClient(api_key=google_key)
                        
                        async def _score():
                            import asyncio
                            score_task = score_predicted_questions_semantic(
                                predicted_questions,
                                actual_questions,
                                google_client=google_client,
                                embed_model=embed_model,
                                use_gemini_for_selection=True,  # Use Gemini for final selection
                            )
                            return await asyncio.wait_for(score_task, timeout=30.0)
                        
                        score, matches, explanation = run_async(_score())
                        backtest_progress.progress(100)
                        backtest_status.empty()
                    except Exception as e:
                        backtest_progress.progress(100)
                        backtest_status.empty()
                        score, matches, explanation = 0, [], f"Could not complete backtest analysis. Please try again later."
                
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
            elif transcript:
                backtest_progress.empty()
                backtest_status.empty()
                st.warning(f"Transcript not found at: {transcript_url}")
        
        # Questions grouped by justice
        st.header("Predicted Questions")
        
        # Group questions by justice
        from utils.scotus import BENCH_ORDER
        questions_by_justice = {}
        for question in prediction.questions:
            justice_id = question.justice_id
            if justice_id not in questions_by_justice:
                questions_by_justice[justice_id] = []
            questions_by_justice[justice_id].append(question)
        
        # Display questions in bench order, grouped by justice
        for justice_id in BENCH_ORDER:
            if justice_id not in questions_by_justice:
                continue
            
            justice_name = JUSTICE_NAMES.get(justice_id, justice_id)
            questions = questions_by_justice[justice_id]
            
            # Use the first question's match info for the header
            first_question = questions[0]
            match_info = matches_dict.get(first_question.question, None)
            
            # Build expander title
            title = f"**{justice_name}**"
            if match_info:
                sim = match_info.get("similarity", 0.0) if isinstance(match_info, dict) else 0.0
                title += f" ({sim * 100:.0f}% match)"
            
            with st.expander(title):
                for question in questions:
                    # Look up match by justice_id first (more reliable), then by question text
                    match_info = matches_by_justice.get(justice_id)
                    if not match_info:
                        match_info = matches_dict.get(question.question, None)
                    
                    st.markdown(f"**Predicted Question:** {question.question}")
                    
                    # Extract key topics from predicted question
                    key_topics = extract_key_topics(question.question)
                    
                    # Show matching actual question if available
                    if match_info:
                        if isinstance(match_info, dict):
                            best_actual = match_info.get("best_actual", "")
                            similarity = match_info.get("similarity", 0.0)
                            actual_citations = match_info.get("actual_citations", [])
                            
                            if best_actual:
                                st.divider()
                                st.markdown("**Best Matching Actual Question:**")
                                st.markdown(f"*\"{best_actual}\"*")
                                if actual_citations:
                                    st.caption(f"Citations: {', '.join(actual_citations)}")
                                
                                # Show transcript context - check precomputed first, then extract live
                                precomputed_context = match_info.get("transcript_context", "") if isinstance(match_info, dict) else ""
                                precomputed_url = None
                                if precomputed_backtest:
                                    precomputed_url = precomputed_backtest.get("transcript_url", "")
                                
                                # Determine which transcript URL to use
                                display_url = transcript_url or precomputed_url or ""
                                
                                if precomputed_context:
                                    # Use precomputed context from sample brief
                                    with st.expander("View context from transcript", expanded=True):
                                        # Highlight the matched question in the context
                                        context_display = precomputed_context
                                        if best_actual.lower() in context_display.lower():
                                            # Bold the matched text
                                            idx = context_display.lower().find(best_actual.lower())
                                            if idx != -1:
                                                before = context_display[:idx]
                                                matched = context_display[idx:idx+len(best_actual)]
                                                after = context_display[idx+len(best_actual):]
                                                context_display = f"{before}**{matched}**{after}"
                                        st.markdown(f"...{context_display}...")
                                        if display_url:
                                            st.caption(f"[View full transcript]({display_url})")
                                elif transcript_url and transcript:
                                    # Live extraction fallback
                                    transcript_text = transcript.get("transcript_text", "")
                                    if transcript_text:
                                        question_snippet = _extract_question_snippet(transcript_text, best_actual)
                                        if question_snippet:
                                            with st.expander("View context from transcript", expanded=True):
                                                st.markdown(question_snippet)
                                                st.caption(f"[View full transcript]({transcript_url})")
                                elif display_url:
                                    # Just show transcript link if no context available
                                    st.caption(f"[View full transcript]({display_url})")
                                # Show match quality indicator based on similarity level
                                if similarity >= 0.7:
                                    st.progress(similarity, text=f"🎯 Strong Match: {similarity * 100:.0f}%")
                                elif similarity >= 0.5:
                                    st.progress(similarity, text=f"✅ Good Match: {similarity * 100:.0f}%")
                                elif similarity >= 0.3:
                                    st.progress(similarity, text=f"Partial Match: {similarity * 100:.0f}%")
                                else:
                                    st.progress(similarity, text=f"❌ Weak Match: {similarity * 100:.0f}%")
                                    st.caption("ℹ️ This match has low similarity - the actual question may address different aspects of the case.")
                                
                                # Use Gemini to analyze semantic match (skip if low similarity)
                                google_key = (os.getenv("GOOGLE_AI_KEY") or "").strip()
                                if google_key and similarity >= 0.3:  # Only analyze if similarity is reasonable
                                    with st.spinner("🤖 Analyzing semantic match with Gemini..."):
                                        try:
                                            google_client = GoogleInferenceClient(api_key=google_key)
                                            semantic_analysis = run_async(
                                                analyze_question_semantic_match(
                                                    question.question,
                                                    best_actual,
                                                    google_client,
                                                )
                                            )
                                            # Only show if NOT a fallback and we got a valid analysis
                                            is_fallback = semantic_analysis.get("is_fallback", False)
                                            has_explanation = semantic_analysis.get("explanation") and len(semantic_analysis.get("explanation", "")) > 0
                                            
                                            if not is_fallback and has_explanation:
                                                if semantic_analysis.get("semantic_match"):
                                                    st.success(f"✅ **Semantic Match**: {semantic_analysis.get('explanation', '')}")
                                                    if semantic_analysis.get("key_topics_aligned"):
                                                        st.caption(f"🔑 Aligned topics: {', '.join(semantic_analysis['key_topics_aligned'][:5])}")
                                                else:
                                                    st.info(f"ℹ️ **Semantic Analysis**: {semantic_analysis.get('explanation', '')}")
                                        except Exception as e:
                                            # Silently fail - don't show error to user
                                            pass
                    
                    # Show topic mentions in transcript if available (with progress indicator)
                    if transcript_url and transcript:
                        transcript_text = transcript.get("transcript_text", "")
                        if transcript_text and key_topics:
                            # Show progress indicator
                            topic_status = st.empty()
                            topic_status.info("🔍 Searching for key topics in justice questions...")
                            topic_mentions = find_topic_mentions_in_transcript(transcript_text, key_topics)
                            topic_status.empty()
                            
                            if topic_mentions:
                                st.divider()
                                st.markdown("**🔍 Key Topics Found in Transcript:**")
                                for mention in topic_mentions[:3]:  # Show top 3 mentions
                                    with st.expander(f"📌 '{mention['topic']}' mentioned"):
                                        st.markdown(f"*...{mention['snippet']}...*")
                                        if transcript_url:
                                            st.caption(f"[View in transcript]({transcript_url})")
                            else:
                                st.caption("ℹ️ No mentions of key topics found in justice questions")
                    
                    if question.what_it_tests:
                        st.divider()
                        st.caption(f"*What it tests:* {question.what_it_tests}")
                    
                    # Add divider between questions if multiple
                    if len(questions) > 1 and question != questions[-1]:
                        st.divider()
        
        # Backtest summary (if available and not already displayed via precomputed)
        if backtest_score is not None and not precomputed_backtest:
            st.header("📊 Backtest Summary")
            st.metric("Question Match Score", f"{backtest_score}%")
            if backtest_explanation:
                st.markdown(backtest_explanation)
        
        # Retrieved cases with links
        if prediction.retrieved_cases:
            st.header("Similar Historical Cases")
            for case in prediction.retrieved_cases:
                # Build case name with link - ONLY use Google Search API to find verified links
                # Never fall back to hallucinated/estimated links
                case_display = case.case_name
                case_link = None
                link_source = None
                
                # ONLY use Google Search API to find verified case links
                google_search_key = (os.getenv("GOOGLE_SEARCH_KEY") or "").strip()
                if google_search_key:
                    async def _find_case_link():
                        session = await get_session_async()
                        try:
                            from utils.case_search import find_case_link_via_search
                            return await find_case_link_via_search(
                                case.case_name,
                                term=case.term,
                                session=session,
                            )
                        finally:
                            await session.close()
                    
                    case_link = run_async(_find_case_link())
                    if case_link:
                        if "oyez.org" in case_link:
                            link_source = "oyez.org (verified via search)"
                        elif "supremecourt.gov" in case_link:
                            link_source = "scotus.gov (verified via search)"
                        else:
                            link_source = "verified via search"
                
                # Only show link if it was verified via Google Search API
                # Never show estimated, hallucinated, or unverified links
                if case_link:
                    case_display = f"[{case.case_name}]({case_link})"
                
                st.markdown(f"**{case_display}**")
                if case.term:
                    st.caption(f"Term {case.term}")
                if case.tags:
                    st.caption(f"Tags: {', '.join(case.tags)}")
                if case.outcome:
                    st.caption(f"Outcome: {case.outcome}")
                if case.docket:
                    st.caption(f"Docket: {case.docket}")
                st.divider()


if __name__ == "__main__":
    main()

