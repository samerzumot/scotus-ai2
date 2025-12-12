"""
Generate precomputed predictions for sample briefs.
Run this script to populate sample_briefs.json with precomputed predictions.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from utils.google_inference import GoogleInferenceClient
from utils.predictor import predict_votes_and_questions
import aiohttp

# Load environment
_ROOT = Path(__file__).parent.parent
for env_file in [_ROOT / "env.local", _ROOT / ".env"]:
    if env_file.exists():
        load_dotenv(env_file)

async def generate_predictions():
    """Generate predictions for all sample briefs."""
    sample_briefs_path = _ROOT / "data" / "sample_briefs.json"
    
    with open(sample_briefs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        sample_briefs = data.get("sample_briefs", [])
    
    google_key = os.getenv("GOOGLE_AI_KEY", "").strip()
    if not google_key:
        print("❌ ERROR: GOOGLE_AI_KEY not set. Cannot generate predictions.")
        return 1
    
    corpus_path = os.getenv("HISTORICAL_CASES_PATH") or str(_ROOT / "data" / "historical_cases.jsonl")
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K") or "5")
    
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, brief in enumerate(sample_briefs):
            case_name = brief.get("case_name", "Unknown")
            print(f"\n[{i+1}/{len(sample_briefs)}] Generating prediction for {case_name}...")
            
            try:
                prediction = await predict_votes_and_questions(
                    session=session,
                    brief_text=brief.get("brief_text", ""),
                    uploader_side=brief.get("uploader_side", "UNKNOWN"),
                    case_hint=brief.get("case_hint", ""),
                    corpus_path=corpus_path,
                    retrieval_top_k=retrieval_top_k,
                )
                
                # Convert to dict for JSON serialization
                brief["precomputed_prediction"] = prediction.model_dump()
                print(f"✅ Generated prediction for {case_name}")
                
            except Exception as e:
                print(f"❌ Failed to generate prediction for {case_name}: {e}")
                continue
    
    # Save updated sample briefs
    with open(sample_briefs_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Updated {sample_briefs_path} with precomputed predictions")
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(generate_predictions())
    sys.exit(exit_code)

