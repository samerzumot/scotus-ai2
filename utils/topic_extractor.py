"""
Extract key topics and entities from predicted questions for better transcript matching.
"""

import re
from typing import List, Set


def extract_key_topics(question: str) -> List[str]:
    """
    Extract key topics, entities, and important terms from a question.
    
    Returns a list of significant terms that should be searched for in transcripts.
    """
    question = question.lower()
    topics: Set[str] = set()
    
    # Extract quoted phrases (often key legal concepts)
    quoted = re.findall(r'"([^"]+)"', question)
    topics.update(set(quoted))
    
    # Extract capitalized phrases (proper nouns, case names, institutions)
    # Look for patterns like "Federal Reserve", "Humphrey's Executor", etc.
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
    topics.update(set(c for c in capitalized if len(c) > 3))
    
    # Extract important legal/technical terms (common patterns)
    legal_terms = [
        r'\b(federal reserve|federal reserve board|chairman of the federal reserve)\b',
        r'\b(monetary policy|economic stability|interest rates)\b',
        r'\b(overrule|overturn|precedent|stare decisis)\b',
        r'\b(limiting principle|constitutional principle)\b',
        r'\b(executive power|presidential power|removal power)\b',
        r'\b(administrative law|agency|independent agency)\b',
        r'\b(first amendment|free speech|free expression)\b',
        r'\b(due process|equal protection|fourteenth amendment)\b',
        r'\b(standing|jurisdiction|mootness)\b',
        r'\b(strict scrutiny|intermediate scrutiny|rational basis)\b',
    ]
    
    for pattern in legal_terms:
        matches = re.findall(pattern, question, re.IGNORECASE)
        # Flatten list of tuples if needed
        flat_matches = []
        for m in matches:
            if isinstance(m, tuple):
                flat_matches.extend(m)
            else:
                flat_matches.append(m)
        topics.update(set(flat_matches))
    
    # Extract case names (patterns like "X v. Y" or "X's Executor")
    case_patterns = [
        r'\b([A-Z][a-z]+(?:\'s)?\s+(?:Executor|Case|Decision))\b',
        r'\b([A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+)\b',
    ]
    for pattern in case_patterns:
        matches = re.findall(pattern, question)
        topics.update(set(matches))
    
    # Extract important nouns (longer words that are likely significant)
    # Filter out common words
    common_words = {
        'what', 'this', 'that', 'would', 'could', 'should', 'does', 'mean',
        'question', 'principle', 'consequence', 'disagree', 'policy'
    }
    words = re.findall(r'\b[a-z]{5,}\b', question)
    significant_words = [w for w in words if w not in common_words and len(w) > 4]
    topics.update(set(significant_words[:5]))  # Limit to top 5
    
    # Clean and normalize
    cleaned = []
    for topic in topics:
        topic = topic.strip()
        if len(topic) > 2 and topic not in cleaned:
            cleaned.append(topic)
    
    # Sort by length (longer phrases are usually more specific)
    cleaned.sort(key=len, reverse=True)
    
    return cleaned[:10]  # Return top 10 most significant topics


def find_topic_mentions_in_transcript(transcript_text: str, topics: List[str], context_chars: int = 200) -> List[dict]:
    """
    Find mentions of key topics in the transcript and return context snippets.
    
    Returns a list of dicts with:
    - topic: the topic that was found
    - snippet: surrounding text from transcript
    - position: approximate position in transcript
    """
    if not transcript_text or not topics:
        return []
    
    transcript_lower = transcript_text.lower()
    mentions = []
    
    for topic in topics:
        topic_lower = topic.lower()
        # Find all occurrences
        start = 0
        while True:
            pos = transcript_lower.find(topic_lower, start)
            if pos == -1:
                break
            
            # Extract context around the mention
            context_start = max(0, pos - context_chars)
            context_end = min(len(transcript_text), pos + len(topic) + context_chars)
            snippet = transcript_text[context_start:context_end]
            
            # Clean up snippet (remove extra whitespace)
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            
            mentions.append({
                'topic': topic,
                'snippet': snippet,
                'position': pos,
                'relevance': len(topic)  # Longer topics are more specific
            })
            
            start = pos + 1
    
    # Sort by relevance (longer topics first) and position
    mentions.sort(key=lambda x: (-x['relevance'], x['position']))
    
    # Deduplicate similar snippets
    unique_mentions = []
    seen_snippets = set()
    for mention in mentions:
        snippet_key = mention['snippet'][:100].lower()
        if snippet_key not in seen_snippets:
            seen_snippets.add(snippet_key)
            unique_mentions.append(mention)
    
    return unique_mentions[:5]  # Return top 5 most relevant mentions

