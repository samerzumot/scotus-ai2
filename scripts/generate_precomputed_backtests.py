#!/usr/bin/env python3
"""Generate precomputed backtest results for sample briefs."""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from utils.backtest import extract_questions_from_transcript, score_predicted_questions_semantic
from utils.google_inference import GoogleInferenceClient
from utils.transcripts import fetch_transcript_text
import aiohttp

# Load env
_ROOT = Path(__file__).parent.parent
for env_file in [_ROOT / "env.local", _ROOT / ".env"]:
    if env_file.exists():
        load_dotenv(env_file)

async def generate_backtest_for_sample(sample: dict, session: aiohttp.ClientSession) -> dict:
    """Generate backtest results for a sample brief."""
    case_name = sample.get("case_name", "Unknown")
    transcript_url = sample.get("transcript_url", "")
    precomputed_prediction = sample.get("precomputed_prediction")
    
    if not transcript_url:
        print(f"  ⏭️  Skipping {case_name}: No transcript URL")
        return None
    
    if not precomputed_prediction:
        print(f"  ⏭️  Skipping {case_name}: No precomputed prediction")
        return None
    
    print(f"  📊 Generating backtest for {case_name}...")
    
    try:
        # Fetch transcript
        print(f"    📥 Fetching transcript from {transcript_url[:60]}...")
        transcript = await fetch_transcript_text(
            session,
            transcript_url=transcript_url,
            max_chars=100_000,
            use_cache=True,
        )
        
        if not transcript or not transcript.get("transcript_found"):
            print(f"    ⚠️  Transcript not found for {case_name}")
            return None
        
        # Extract actual questions
        print(f"    🔍 Extracting questions from transcript...")
        actual_questions = extract_questions_from_transcript(
            transcript.get("transcript_text") or "", limit=100
        )
        
        if not actual_questions:
            print(f"    ⚠️  No questions found in transcript for {case_name}")
            return None
        
        # Get predicted questions from precomputed prediction
        predicted_questions = [q.get("question", "") for q in precomputed_prediction.get("questions", [])]
        predicted_questions = [q for q in predicted_questions if q]
        
        if not predicted_questions:
            print(f"    ⚠️  No predicted questions found for {case_name}")
            return None
        
        # Run backtest
        print(f"    📊 Running semantic backtest ({len(predicted_questions)} predicted vs {len(actual_questions)} actual)...")
        google_key = (os.getenv("GOOGLE_AI_KEY") or "").strip()
        embed_model = (os.getenv("GOOGLE_EMBED_MODEL") or "").strip() or "models/text-embedding-004"
        
        if not google_key:
            print(f"    ❌ GOOGLE_AI_KEY not set, skipping {case_name}")
            return None
        
        google_client = GoogleInferenceClient(api_key=google_key)
        
        # Build predicted_with_justice list
        predicted_with_justice = []
        for q in precomputed_prediction.get("questions", []):
            predicted_with_justice.append((
                q.get("question", ""),
                q.get("justice_id", ""),
                q.get("justice_name", ""),
            ))
        
        score, matches, explanation = await score_predicted_questions_semantic(
            predicted_questions,
            actual_questions,
            google_client=google_client,
            embed_model=embed_model,
            predicted_with_justice=predicted_with_justice if predicted_with_justice else None,
            use_gemini_for_selection=True,
        )
        
        # Get transcript text for context extraction
        transcript_text = transcript.get("transcript_text", "")
        
        # Helper function to extract context around a question
        def extract_context(question_text: str, max_context: int = 300) -> str:
            if not transcript_text or not question_text:
                return ""
            try:
                # Find question in transcript
                question_lower = question_text.lower().strip()
                idx = transcript_text.lower().find(question_lower)
                if idx == -1 and len(question_lower) > 40:
                    idx = transcript_text.lower().find(question_lower[:40])
                if idx == -1:
                    return ""
                # Extract context: 300 chars before and after
                start = max(0, idx - max_context)
                end = min(len(transcript_text), idx + len(question_text) + max_context)
                snippet = transcript_text[start:end].strip()
                # Clean whitespace
                import re
                snippet = re.sub(r'[ \t]+', ' ', snippet)
                snippet = re.sub(r'\n{3,}', '\n\n', snippet)
                return snippet
            except Exception:
                return ""
        
        # Convert matches to dict format with context snippets
        matches_list = []
        for m in matches:
            if isinstance(m, dict):
                best_actual = m.get("best_actual", "")
                context_snippet = extract_context(best_actual)
                matches_list.append({
                    "predicted": m.get("predicted", ""),
                    "best_actual": best_actual,
                    "similarity": float(m.get("similarity", 0.0)),
                    "justice_id": m.get("justice_id", ""),
                    "justice_name": m.get("justice_name", ""),
                    "predicted_citations": m.get("predicted_citations", []),
                    "actual_citations": m.get("actual_citations", []),
                    "transcript_context": context_snippet,  # NEW: Store context snippet
                })
            else:
                # Legacy tuple format
                best_actual = m[1] if len(m) > 1 else ""
                context_snippet = extract_context(best_actual)
                matches_list.append({
                    "predicted": m[0] if len(m) > 0 else "",
                    "best_actual": best_actual,
                    "similarity": float(m[2] if len(m) > 2 else 0.0),
                    "justice_id": "",
                    "justice_name": "",
                    "predicted_citations": [],
                    "actual_citations": [],
                    "transcript_context": context_snippet,  # NEW: Store context snippet
                })
        
        backtest_result = {
            "questions_score_pct": score,
            "explanation": explanation,
            "matches": matches_list,
            "transcript_url": transcript_url,  # NEW: Store transcript URL for linking
        }
        
        print(f"    ✅ Backtest complete: {score}% match")
        return backtest_result
        
    except Exception as e:
        print(f"    ❌ Error generating backtest for {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Generate precomputed backtests for all sample briefs."""
    sample_briefs_path = Path(__file__).parent.parent / "data" / "sample_briefs.json"
    
    if not sample_briefs_path.exists():
        print(f"❌ Sample briefs file not found: {sample_briefs_path}")
        return 1
    
    print(f"📚 Loading sample briefs from {sample_briefs_path}...")
    with open(sample_briefs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sample_briefs = data.get("sample_briefs", [])
    print(f"✅ Found {len(sample_briefs)} sample briefs\n")
    
    # Create aiohttp session
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        updated_count = 0
        for i, sample in enumerate(sample_briefs, 1):
            case_name = sample.get("case_name", f"Sample {i}")
            print(f"[{i}/{len(sample_briefs)}] Processing {case_name}...")
            
            backtest_result = await generate_backtest_for_sample(sample, session)
            
            if backtest_result:
                sample["precomputed_backtest"] = backtest_result
                updated_count += 1
                print(f"  ✅ Added precomputed backtest to {case_name}\n")
            else:
                print(f"  ⏭️  Skipped {case_name}\n")
    
    # Save updated sample briefs
    print(f"💾 Saving updated sample briefs...")
    with open(sample_briefs_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Complete! Updated {updated_count} sample briefs with precomputed backtest results.")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

