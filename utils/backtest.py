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
) -> Tuple[int, List[Tuple[str, str, float]], str]:
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

    # Try semantic similarity with embeddings if available
    use_semantic = google_client and embed_model
    if use_semantic:
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
            
            # Compute semantic similarities
            matches: List[Tuple[str, str, float]] = []
            sims: List[float] = []
            
            for i, pq in enumerate(predicted):
                pred_emb = predicted_embeddings[i] if i < len(predicted_embeddings) else None
                best_a = ""
                best_s = 0.0
                
                for j, aq in enumerate(actual):
                    actual_emb = actual_embeddings[j] if j < len(actual_embeddings) else None
                    
                    if pred_emb and actual_emb:
                        # Use semantic similarity
                        s = cosine_similarity(pred_emb, actual_emb)
                    else:
                        # Fallback to lexical
                        s = jaccard_similarity(pq, aq)
                    
                    if s > best_s:
                        best_s = s
                        best_a = aq
                
                sims.append(best_s)
                matches.append((pq, best_a, float(best_s)))
            
            score = int(round(100.0 * (sum(sims) / max(1, len(sims)))))
            matches.sort(key=lambda t: t[2], reverse=True)
            
            explanation = _generate_backtest_explanation(score, len(predicted), len(actual), matches, semantic=True)
            return score, matches, explanation
            
        except Exception as e:
            # Fall through to lexical if semantic fails - log warning
            import sys
            print(f"⚠️ WARNING: Semantic similarity failed, falling back to lexical similarity: {e}", file=sys.stderr)
            use_semantic = False

    # Fallback to lexical similarity (Jaccard)
    matches: List[Tuple[str, str, float]] = []
    sims: List[float] = []
    for pq in predicted:
        best_a = ""
        best_s = 0.0
        for aq in actual:
            s = jaccard_similarity(pq, aq)
            if s > best_s:
                best_s = s
                best_a = aq
        sims.append(best_s)
        matches.append((pq, best_a, float(best_s)))

    score = int(round(100.0 * (sum(sims) / max(1, len(sims)))))
    matches.sort(key=lambda t: t[2], reverse=True)
    
    explanation = _generate_backtest_explanation(score, len(predicted), len(actual), matches, semantic=False)
    
    return score, matches, explanation


def score_predicted_questions(predicted: List[str], actual: List[str]) -> Tuple[int, List[Tuple[str, str, float]], str]:
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

    matches: List[Tuple[str, str, float]] = []
    sims: List[float] = []
    for pq in predicted:
        best_a = ""
        best_s = 0.0
        for aq in actual:
            s = jaccard_similarity(pq, aq)
            if s > best_s:
                best_s = s
                best_a = aq
        sims.append(best_s)
        matches.append((pq, best_a, float(best_s)))

    score = int(round(100.0 * (sum(sims) / max(1, len(sims)))))
    matches.sort(key=lambda t: t[2], reverse=True)
    
    explanation = _generate_backtest_explanation(score, len(predicted), len(actual), matches, semantic=False)
    
    return score, matches, explanation


def _generate_backtest_explanation(
    score_pct: int,
    num_predicted: int,
    num_actual: int,
    matches: List[Tuple[str, str, float]],
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
    
    # Analyze top matches
    high_sim = [m for m in matches if m[2] >= 0.5]
    medium_sim = [m for m in matches if 0.3 <= m[2] < 0.5]
    low_sim = [m for m in matches if m[2] < 0.3]
    
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
        if top_match[2] >= 0.6:
            explanation += f"\n\n**Best match:** The predicted question \"{top_match[0][:80]}...\" closely matches the actual question \"{top_match[1][:80]}...\" ({int(top_match[2]*100)}% similarity)."
        elif top_match[2] >= 0.3:
            explanation += f"\n\n**Closest match:** The predicted question \"{top_match[0][:80]}...\" partially matches \"{top_match[1][:80]}...\" ({int(top_match[2]*100)}% similarity)."
    
    return explanation


