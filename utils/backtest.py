from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from utils.security import sanitize_user_text

if TYPE_CHECKING:
    from utils.google_inference import GoogleInferenceClient


def extract_questions_from_transcript(transcript_text: str, *, limit: int = 200) -> List[str]:
    """
    Deterministic heuristic: keep lines that look like questions.
    """
    transcript_text = sanitize_user_text(transcript_text, max_len=450_000)
    if not transcript_text:
        return []
    candidates: List[str] = []
    for line in transcript_text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if "?" in line and len(line) >= 18:
            candidates.append(line)
    # Dedup, keep order
    seen = set()
    out: List[str] = []
    for c in candidates:
        k = c.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def jaccard_similarity(a: str, b: str) -> float:
    """Lexical similarity based on word overlap (fallback method)."""
    tok = lambda s: {t for t in re.findall(r"[a-z0-9']+", (s or "").lower()) if len(t) > 2}
    A, B = tok(a), tok(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def score_predicted_questions_semantic(
    predicted: List[str],
    actual: List[str],
    google_client: Optional[Any] = None,  # GoogleInferenceClient, but avoid circular import
    embed_model: Optional[str] = None,
    predicted_with_justice: Optional[List[Tuple[str, str, str]]] = None,  # List of (question, justice_id, justice_name)
) -> Tuple[int, List[Dict[str, Any]], str]:
    """
    Score predicted questions using semantic similarity (embeddings).
    Falls back to lexical similarity if embeddings aren't available.
    
    Returns:
    - score_pct: average best-match similarity across predicted questions
    - matches: list of (predicted, best_actual, similarity) for inspection
    - explanation: human-readable explanation of the results
    """
    predicted = [sanitize_user_text(q, max_len=260) for q in (predicted or []) if isinstance(q, str)]
    predicted = [q for q in predicted if q]
    actual = [sanitize_user_text(q, max_len=340) for q in (actual or []) if isinstance(q, str)]
    actual = [q for q in actual if q]

    if not predicted or not actual:
        explanation = "⚠️ Cannot compute backtest: "
        if not predicted:
            explanation += "No predicted questions available."
        elif not actual:
            explanation += "No actual questions found in transcript."
        return 0, [], explanation

    # Always use semantic similarity with embeddings - require Google client
    if not google_client or not embed_model:
        explanation = "⚠️ Cannot compute semantic backtest: GOOGLE_AI_KEY or GOOGLE_EMBED_MODEL not configured. Semantic similarity requires Google embeddings."
        return 0, [], explanation
    
    try:
        # Embed all questions
        predicted_embeddings = []
        actual_embeddings = []
        
        for pq in predicted:
            try:
                emb = await google_client.embed_text(model=embed_model, text=pq[:500])
                predicted_embeddings.append(emb)
            except Exception:
                predicted_embeddings.append(None)
        
        for aq in actual:
            try:
                emb = await google_client.embed_text(model=embed_model, text=aq[:500])
                actual_embeddings.append(emb)
            except Exception:
                actual_embeddings.append(None)
        
        # Extract major topic words from predicted questions for filtering
        from utils.topic_extractor import extract_key_topics
        
        # Compute semantic similarities with two-stage matching
        matches: List[Dict[str, Any]] = []
        sims: List[float] = []
        
        for i, pq in enumerate(predicted):
            pred_emb = predicted_embeddings[i] if i < len(predicted_embeddings) else None
            
            # Get justice info if available
            justice_id = ""
            justice_name = ""
            if predicted_with_justice and i < len(predicted_with_justice):
                _, justice_id, justice_name = predicted_with_justice[i]
            
            # Stage 1: Filter actual questions by major topic word matches
            # Extract major topics from predicted question
            major_topics = extract_key_topics(pq)
            # Get significant words (4+ chars) from topics
            topic_words = set()
            for topic in major_topics[:5]:  # Top 5 topics
                words = re.findall(r'\b[a-z]{4,}\b', topic.lower())
                topic_words.update(words)
            
            # Filter actual questions that contain at least one major topic word
            candidate_actuals = []
            candidate_embeddings = []
            candidate_indices = []
            
            for j, aq in enumerate(actual):
                actual_emb = actual_embeddings[j] if j < len(actual_embeddings) else None
                aq_lower = aq.lower()
                
                # Check if actual question contains at least one major topic word
                has_topic_match = any(word in aq_lower for word in topic_words if len(word) >= 4)
                
                if has_topic_match and actual_emb:
                    candidate_actuals.append(aq)
                    candidate_embeddings.append(actual_emb)
                    candidate_indices.append(j)
            
            # Stage 2: If multiple candidates, use semantic similarity to rank
            # If only one candidate, use it
            # If no candidates from topic match, fall back to all questions
            if len(candidate_actuals) == 0:
                # Fallback: use all actual questions
                candidate_actuals = actual
                candidate_embeddings = actual_embeddings
                candidate_indices = list(range(len(actual)))
            
            best_a = ""
            best_s = 0.0
            best_j = -1
            
            if pred_emb:
                # Compute similarity with all candidates
                for idx, (aq, aq_emb) in enumerate(zip(candidate_actuals, candidate_embeddings)):
                    if aq_emb:
                        s = cosine_similarity(pred_emb, aq_emb)
                        if s > best_s:
                            best_s = s
                            best_a = aq
                            best_j = candidate_indices[idx] if idx < len(candidate_indices) else idx
            else:
                # If embedding failed, use first candidate
                if candidate_actuals:
                    best_a = candidate_actuals[0]
                    best_s = 0.0
            
            # Stage 3: If multiple candidates with similar scores, use Gemini to select best
            if len(candidate_actuals) > 1 and best_s > 0.3:
                # Get top candidates (within 0.1 of best score)
                top_candidates = []
                for idx, (aq, aq_emb) in enumerate(zip(candidate_actuals, candidate_embeddings)):
                    if aq_emb and pred_emb:
                        s = cosine_similarity(pred_emb, aq_emb)
                        if s >= best_s - 0.1:  # Within 0.1 of best
                            top_candidates.append((aq, s))
                
                # If multiple top candidates, use Gemini to select best
                if len(top_candidates) > 1:
                    try:
                        from utils.semantic_matcher import select_best_semantic_match
                        # Only use Gemini if we have the client (will be passed separately if needed)
                        # For now, just use the highest similarity score
                        top_candidates.sort(key=lambda x: x[1], reverse=True)
                        best_a = top_candidates[0][0]
                        best_s = top_candidates[0][1]
                    except Exception:
                        # Fallback: use highest similarity
                        pass
            
            sims.append(best_s)
            
            # Extract citations
            from utils.citations import extract_case_citations
            pred_citations = extract_case_citations(pq)
            actual_citations = extract_case_citations(best_a)
            
            matches.append({
                "predicted": pq,
                "best_actual": best_a,
                "similarity": float(best_s),
                "justice_id": justice_id,
                "justice_name": justice_name,
                "predicted_citations": pred_citations,
                "actual_citations": actual_citations,
            })
        
        score = int(round(100.0 * (sum(sims) / max(1, len(sims)))))
        matches.sort(key=lambda t: t.get("similarity", 0.0), reverse=True)
        
        explanation = _generate_backtest_explanation(score, len(predicted), len(actual), matches, semantic=True)
        return score, matches, explanation
        
    except Exception as e:
        # If semantic similarity fails, raise error rather than falling back to lexical
        import sys
        error_msg = f"Semantic similarity failed: {str(e)}"
        print(f"⚠️ ERROR: {error_msg}", file=sys.stderr)
        explanation = f"⚠️ Backtest failed: {error_msg}. Please check GOOGLE_AI_KEY and GOOGLE_EMBED_MODEL configuration."
        return 0, [], explanation


def score_predicted_questions(
    predicted: List[str],
    actual: List[str],
    predicted_with_justice: Optional[List[Tuple[str, str, str]]] = None,  # List of (question, justice_id, justice_name)
) -> Tuple[int, List[Dict[str, Any]], str]:
    """
    Synchronous version using lexical similarity (for backward compatibility).
    Use score_predicted_questions_semantic for better semantic matching.
    """
    predicted = [sanitize_user_text(q, max_len=260) for q in (predicted or []) if isinstance(q, str)]
    predicted = [q for q in predicted if q]
    actual = [sanitize_user_text(q, max_len=340) for q in (actual or []) if isinstance(q, str)]
    actual = [q for q in actual if q]

    if not predicted or not actual:
        explanation = "⚠️ Cannot compute backtest: "
        if not predicted:
            explanation += "No predicted questions available."
        elif not actual:
            explanation += "No actual questions found in transcript."
        return 0, [], explanation

    matches: List[Dict[str, Any]] = []
    sims: List[float] = []
    for i, pq in enumerate(predicted):
        best_a = ""
        best_s = 0.0
        
        # Get justice info if available
        justice_id = ""
        justice_name = ""
        if predicted_with_justice and i < len(predicted_with_justice):
            _, justice_id, justice_name = predicted_with_justice[i]
        
        for aq in actual:
            s = jaccard_similarity(pq, aq)
            if s > best_s:
                best_s = s
                best_a = aq
        sims.append(best_s)
        
        # Extract citations
        from utils.citations import extract_case_citations
        pred_citations = extract_case_citations(pq)
        actual_citations = extract_case_citations(best_a)
        
        matches.append({
            "predicted": pq,
            "best_actual": best_a,
            "similarity": float(best_s),
            "justice_id": justice_id,
            "justice_name": justice_name,
            "predicted_citations": pred_citations,
            "actual_citations": actual_citations,
        })

    score = int(round(100.0 * (sum(sims) / max(1, len(sims)))))
    matches.sort(key=lambda t: t.get("similarity", 0.0), reverse=True)
    
    explanation = _generate_backtest_explanation(score, len(predicted), len(actual), matches, semantic=False)
    
    return score, matches, explanation


def _generate_backtest_explanation(
    score_pct: int,
    num_predicted: int,
    num_actual: int,
    matches: List[Any],  # Can be List[Tuple] or List[Dict]
    semantic: bool = False,
) -> str:
    """Generate a human-readable explanation of backtest results."""
    if score_pct >= 70:
        quality = "excellent"
        emoji = "🎯"
    elif score_pct >= 50:
        quality = "good"
        emoji = "✅"
    elif score_pct >= 30:
        quality = "moderate"
        emoji = "⚠️"
    else:
        quality = "poor"
        emoji = "❌"
    
    method = "semantic similarity" if semantic else "lexical similarity"
    explanation = f"{emoji} **Backtest Score: {score_pct}%** ({quality})\n\n"
    
    explanation += f"**Summary:** Compared {num_predicted} predicted questions against {num_actual} actual questions from the transcript using {method}.\n\n"
    
    # Analyze top matches (handle both tuple and dict formats)
    def get_similarity(m):
        if isinstance(m, dict):
            return m.get("similarity", 0.0)
        elif isinstance(m, tuple) and len(m) >= 3:
            return m[2]
        return 0.0
    
    high_sim = [m for m in matches if get_similarity(m) >= 0.5]
    medium_sim = [m for m in matches if 0.3 <= get_similarity(m) < 0.5]
    low_sim = [m for m in matches if get_similarity(m) < 0.3]
    
    if high_sim:
        explanation += f"**Strong matches ({len(high_sim)}):** Questions with ≥50% similarity to actual transcript questions.\n"
    if medium_sim:
        explanation += f"**Partial matches ({len(medium_sim)}):** Questions with 30-50% similarity.\n"
    if low_sim:
        explanation += f"**Weak matches ({len(low_sim)}):** Questions with <30% similarity.\n"
    
    explanation += "\n**Interpretation:** "
    if score_pct >= 70:
        explanation += "The model's predicted questions closely match the actual questions asked during oral arguments. This suggests the model has a strong understanding of the legal issues and Justice questioning patterns."
    elif score_pct >= 50:
        explanation += "The model's predicted questions show moderate alignment with actual questions. Some predictions capture key themes, but there's room for improvement in specificity and Justice-specific patterns."
    elif score_pct >= 30:
        explanation += "The model's predicted questions show limited alignment with actual questions. The predictions may capture general themes but miss specific legal nuances and Justice-specific concerns."
    else:
        explanation += "The model's predicted questions show poor alignment with actual questions. This may indicate the model needs better grounding in historical case patterns or more specific legal context."
    
    # Add specific insights from top matches
    if matches:
        top_match = matches[0]
        pred_text = top_match.get("predicted", top_match[0] if isinstance(top_match, tuple) else "")[:80]
        actual_text = top_match.get("best_actual", top_match[1] if isinstance(top_match, tuple) else "")[:80]
        sim = get_similarity(top_match)
        justice_name = top_match.get("justice_name", "") if isinstance(top_match, dict) else ""
        justice_label = f" (Justice {justice_name})" if justice_name else ""
        
        if sim >= 0.6:
            explanation += f"\n\n**Best match{justice_label}:** The predicted question \"{pred_text}...\" closely matches the actual question \"{actual_text}...\" ({int(sim*100)}% similarity)."
        elif sim >= 0.3:
            explanation += f"\n\n**Closest match{justice_label}:** The predicted question \"{pred_text}...\" partially matches \"{actual_text}...\" ({int(sim*100)}% similarity)."
    
    return explanation


