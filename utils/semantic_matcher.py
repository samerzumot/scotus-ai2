"""
Use Gemini to analyze semantic match between predicted and actual questions.
"""

import re
from typing import Any, Dict, List, Optional

from utils.google_inference import GoogleInferenceClient


async def select_best_semantic_match(
    predicted_question: str,
    candidate_questions: List[str],
    google_client: GoogleInferenceClient,
    model: str = "models/gemini-2.5-pro",
) -> Dict[str, Any]:
    """
    Use Gemini to select the best semantic match from multiple candidate questions.
    
    Returns a dict with:
    - best_question: str (the best matching question)
    - similarity_score: float 0.0-1.0
    - explanation: str (why this is the best match)
    """
    if not candidate_questions:
        return {
            "best_question": "",
            "similarity_score": 0.0,
            "explanation": "No candidate questions provided",
        }
    
    if len(candidate_questions) == 1:
        # Only one candidate, return it
        return {
            "best_question": candidate_questions[0],
            "similarity_score": 0.5,  # Default score
            "explanation": "Only one candidate question matched",
        }
    
    prompt = f"""Given a predicted question from a Supreme Court oral argument, select the best matching actual question from the candidates below.

PREDICTED QUESTION:
{predicted_question}

CANDIDATE ACTUAL QUESTIONS:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(candidate_questions))}

Select the candidate question that best matches the predicted question in terms of:
1. Same core legal issue or topic
2. Similar question intent or purpose
3. Relevant legal concepts discussed

Return ONLY valid JSON with this structure:
{{
    "best_index": 1-based index of best match,
    "similarity_score": 0.0-1.0,
    "explanation": "brief explanation of why this is the best match"
}}"""

    system_instruction = """You are a legal analysis expert. Select the candidate question that best semantically matches the predicted question. Focus on core legal issues and intent, not exact wording. Return only valid JSON."""

    try:
        result = await google_client.generate_json(
            model=model,
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=0.1,
            max_output_tokens=512,
        )
        
        best_index = int(result.get("best_index", 1)) - 1  # Convert to 0-based
        best_index = max(0, min(len(candidate_questions) - 1, best_index))  # Clamp to valid range
        
        similarity_score = float(result.get("similarity_score", 0.5))
        similarity_score = max(0.0, min(1.0, similarity_score))
        
        return {
            "best_question": candidate_questions[best_index],
            "similarity_score": similarity_score,
            "explanation": result.get("explanation", ""),
        }
    except Exception as e:
        # Fallback: return first candidate
        return {
            "best_question": candidate_questions[0],
            "similarity_score": 0.5,
            "explanation": f"Fallback selection (Gemini unavailable): {str(e)[:100]}",
        }


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
        error_msg = str(e)
        # Check if it's a safety filter or blocked response
        if "finish_reason" in error_msg or "safety" in error_msg.lower() or "blocked" in error_msg.lower():
            # Fallback: use simple keyword matching as basic semantic analysis
            pred_lower = predicted_question.lower()
            actual_lower = actual_question.lower()
            
            # Extract key words from both
            pred_words = set(re.findall(r'\b[a-z]{4,}\b', pred_lower))
            actual_words = set(re.findall(r'\b[a-z]{4,}\b', actual_lower))
            
            # Find common significant words
            common_words = pred_words & actual_words
            # Filter out very common words
            common_filtered = {w for w in common_words if w not in {
                'what', 'this', 'that', 'would', 'could', 'should', 'does', 'mean',
                'question', 'principle', 'about', 'there', 'their', 'where', 'when'
            }}
            
            # Basic similarity based on word overlap
            if len(pred_words) > 0 and len(actual_words) > 0:
                word_similarity = len(common_filtered) / max(len(pred_words), len(actual_words))
            else:
                word_similarity = 0.0
            
            return {
                "semantic_match": len(common_filtered) >= 2 or word_similarity > 0.3,
                "similarity_score": min(1.0, word_similarity * 1.5),  # Scale up a bit
                "key_topics_aligned": list(common_filtered)[:5],
                "explanation": f"Found {len(common_filtered)} common terms.",
                "is_fallback": True,  # Flag to indicate this is a fallback analysis
            }
        else:
            # Other errors - return minimal info with fallback flag
            return {
                "semantic_match": False,
                "similarity_score": 0.0,
                "key_topics_aligned": [],
                "explanation": "",
                "is_fallback": True,  # Flag to indicate this is a fallback analysis
            }

