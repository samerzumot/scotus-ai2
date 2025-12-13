"""
Use Gemini to analyze semantic match between predicted and actual questions.
"""

from typing import Any, Dict, List, Optional

from utils.google_inference import GoogleInferenceClient


async def analyze_question_semantic_match(
    predicted_question: str,
    actual_question: str,
    google_client: GoogleInferenceClient,
    model: str = "models/gemini-2.5-pro",
) -> Dict[str, Any]:
    """
    Use Gemini to analyze the semantic match between a predicted question and an actual question.
    
    Returns a dict with:
    - similarity_score: float 0.0-1.0
    - semantic_match: bool (whether they address the same core issue)
    - explanation: str (why they match or don't match)
    - key_topics_aligned: List[str] (topics that appear in both)
    """
    prompt = f"""Analyze whether these two questions from a Supreme Court oral argument address the same core legal issue or topic.

PREDICTED QUESTION:
{predicted_question}

ACTUAL QUESTION:
{actual_question}

Analyze:
1. Do they address the same core legal issue or topic? (semantic_match: true/false)
2. What is the similarity score (0.0-1.0) where 1.0 = identical topic/issue, 0.0 = completely different?
3. What key topics or concepts appear in both questions? (key_topics_aligned: list)
4. Why do they match or not match? (explanation: brief explanation)

Return ONLY valid JSON with this structure:
{{
    "semantic_match": true/false,
    "similarity_score": 0.0-1.0,
    "key_topics_aligned": ["topic1", "topic2"],
    "explanation": "brief explanation"
}}"""

    system_instruction = """You are a legal analysis expert. Analyze whether two questions address the same core legal issue, even if worded differently. Focus on semantic meaning, not exact word matches. Return only valid JSON."""

    try:
        result = await google_client.generate_json(
            model=model,
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1,  # Low temperature for consistent analysis
            max_output_tokens=512,
        )
        
        # Validate and normalize result
        similarity_score = float(result.get("similarity_score", 0.0))
        similarity_score = max(0.0, min(1.0, similarity_score))  # Clamp to [0.0, 1.0]
        
        return {
            "semantic_match": bool(result.get("semantic_match", False)),
            "similarity_score": similarity_score,
            "key_topics_aligned": result.get("key_topics_aligned", []),
            "explanation": result.get("explanation", ""),
        }
    except Exception as e:
        # Fallback to basic analysis
        return {
            "semantic_match": False,
            "similarity_score": 0.0,
            "key_topics_aligned": [],
            "explanation": f"Analysis failed: {str(e)}",
        }

