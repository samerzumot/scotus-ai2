"""
Extract key topics and entities from predicted questions for better transcript matching.
"""

import re
from typing import List, Set


def extract_key_topics(question: str) -> List[str]:
    """
    Extract key topics, entities, and important terms from a question using general patterns.
    No hardcoded keywords - purely pattern-based extraction.
    
    Returns a list of significant terms that should be searched for in transcripts.
    """
    # Keep original for capitalized phrase extraction
    question_original = question
    question_lower = question.lower()
    topics: Set[str] = set()
    
    # Extract quoted phrases (often key legal concepts)
    quoted = re.findall(r'"([^"]+)"', question_original)
    topics.update(set(q.lower() for q in quoted))
    
    # Extract multi-word capitalized phrases (proper nouns, case names, institutions)
    # Pattern: One or more capitalized words in sequence
    # Examples: "Federal Reserve", "Humphrey's Executor", "Supreme Court"
    capitalized_phrases = re.findall(r'\b([A-Z][a-z]+(?:\'[a-z]+)?(?:\s+[A-Z][a-z]+)+)\b', question_original)
    for phrase in capitalized_phrases:
        if len(phrase) > 3:
            # Add lowercase version for searching
            topics.add(phrase.lower())
            # For multi-word phrases, also add individual significant words
            words = re.findall(r'\b([A-Z][a-z]{4,})\b', phrase)
            for word in words:
                topics.add(word.lower())
    
    # Extract single capitalized words that are likely significant (proper nouns)
    # Only if they're longer than 4 chars to avoid common words
    single_caps = re.findall(r'\b([A-Z][a-z]{4,})\b', question_original)
    topics.update(set(w.lower() for w in single_caps))
    
    # Extract case names (patterns like "X v. Y" or "X's Executor")
    case_patterns = [
        r'\b([A-Z][a-z]+(?:\'s)?\s+(?:Executor|Case|Decision|Doctrine))\b',
        r'\b([A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+)\b',
    ]
    for pattern in case_patterns:
        matches = re.findall(pattern, question_original)
        topics.update(set(m.lower() for m in matches))
    
    # Extract multi-word technical/legal phrases (general pattern)
    # Pattern: 2-4 word phrases where words are 4+ chars (likely significant terms)
    # This catches phrases like "monetary policy", "limiting principle", "economic stability"
    multi_word_phrases = re.findall(
        r'\b([a-z]{4,}(?:\s+[a-z]{4,}){1,3})\b',
        question_lower
    )
    # Filter out common phrases and keep only substantial ones
    common_phrases = {
        'what would', 'what does', 'does this', 'this mean', 'would be',
        'can fire', 'for disagreeing', 'and what', 'what is', 'is the'
    }
    for phrase in multi_word_phrases:
        if len(phrase) > 8 and phrase not in common_phrases:
            # Check if it's a meaningful phrase (not just filler)
            words = phrase.split()
            if len(words) >= 2 and all(len(w) >= 4 for w in words):
                topics.add(phrase)
    
    # Extract important single words (longer words that are likely significant)
    # Filter out common words
    common_words = {
        'what', 'this', 'that', 'would', 'could', 'should', 'does', 'mean',
        'question', 'principle', 'consequence', 'disagree', 'policy', 'fire',
        'mean', 'mean', 'president', 'chairman'
    }
    words = re.findall(r'\b[a-z]{5,}\b', question_lower)
    significant_words = [w for w in words if w not in common_words and len(w) > 4]
    topics.update(set(significant_words[:8]))  # Limit to top 8
    
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


def _extract_oral_arguments_content(transcript_text: str) -> str:
    """
    Extract only the oral arguments content, removing headers, footers, and metadata.
    
    Looks for common patterns that indicate the start/end of actual oral arguments.
    """
    if not transcript_text:
        return ""
    
    # Common patterns that indicate start of oral arguments
    start_patterns = [
        r'(?:^|\n)\s*(?:ORAL\s+ARGUMENT|ORAL\s+ARGUMENTS|ARGUMENT\s+BEGINS?|ARGUMENT\s+OF)',
        r'(?:^|\n)\s*(?:CHIEF\s+JUSTICE|JUSTICE)\s+[A-Z]',
        r'(?:^|\n)\s*(?:MR\.|MS\.|MRS\.)\s+[A-Z]',
        r'(?:^|\n)\s*(?:COUNSEL|ATTORNEY)',
    ]
    
    # Common patterns that indicate end of oral arguments
    end_patterns = [
        r'(?:^|\n)\s*(?:ARGUMENT\s+CONCLUDED|ARGUMENT\s+ENDS?|ADJOURNED)',
        r'(?:^|\n)\s*(?:CHIEF\s+JUSTICE.*?ADJOURNED)',
        r'(?:^|\n)\s*(?:THE\s+CASE\s+IS\s+SUBMITTED)',
    ]
    
    # Find start of oral arguments
    start_pos = 0
    for pattern in start_patterns:
        match = re.search(pattern, transcript_text, re.IGNORECASE | re.MULTILINE)
        if match:
            start_pos = match.start()
            break
    
    # Find end of oral arguments
    end_pos = len(transcript_text)
    for pattern in end_patterns:
        match = re.search(pattern, transcript_text[start_pos:], re.IGNORECASE | re.MULTILINE)
        if match:
            end_pos = start_pos + match.start()
            break
    
    # Extract the oral arguments section
    oral_args = transcript_text[start_pos:end_pos]
    
    # Remove common header/footer patterns
    # Remove page numbers and headers
    oral_args = re.sub(r'(?:^|\n)\s*\d+\s*(?:\n|$)', '\n', oral_args, flags=re.MULTILINE)
    oral_args = re.sub(r'(?:^|\n)\s*(?:Page\s+\d+|PAGE\s+\d+)\s*(?:\n|$)', '\n', oral_args, flags=re.MULTILINE)
    
    # Remove case name headers that repeat
    lines = oral_args.split('\n')
    filtered_lines = []
    seen_headers = set()
    for line in lines:
        line_stripped = line.strip()
        # Skip very short lines that are likely headers/footers
        if len(line_stripped) < 3:
            continue
        # Skip lines that look like headers (all caps, very short)
        if line_stripped.isupper() and len(line_stripped) < 50:
            if line_stripped in seen_headers:
                continue  # Skip duplicate headers
            seen_headers.add(line_stripped)
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def find_topic_mentions_in_transcript(transcript_text: str, topics: List[str], context_chars: int = 200) -> List[dict]:
    """
    Find mentions of key topics in the transcript and return context snippets.
    Only searches within the oral arguments content, excluding headers/footers.
    
    Returns a list of dicts with:
    - topic: the topic that was found
    - snippet: surrounding text from transcript
    - position: approximate position in transcript
    """
    if not transcript_text or not topics:
        return []
    
    # Extract only oral arguments content, excluding headers/footers
    oral_args_text = _extract_oral_arguments_content(transcript_text)
    if not oral_args_text:
        # Fallback to full text if extraction fails
        oral_args_text = transcript_text
    
    transcript_lower = oral_args_text.lower()
    mentions = []
    
    for topic in topics:
        topic_lower = topic.lower()
        # Find all occurrences
        start = 0
        while True:
            pos = transcript_lower.find(topic_lower, start)
            if pos == -1:
                break
            
            # Extract context around the mention (use oral_args_text, not full transcript)
            context_start = max(0, pos - context_chars)
            context_end = min(len(oral_args_text), pos + len(topic) + context_chars)
            snippet = oral_args_text[context_start:context_end]
            
            # Clean up snippet (remove extra whitespace)
            snippet = re.sub(r'\s+', ' ', snippet).strip()
            
            mentions.append({
                'topic': topic,
                'snippet': snippet,
                'position': pos,
                'relevance': len(topic)  # Longer topics are more specific
            })
            
            start = pos + 1
    
    # Group mentions by topic (case-insensitive) to ensure diversity
    mentions_by_topic = {}
    for mention in mentions:
        topic_key = mention['topic'].lower()
        if topic_key not in mentions_by_topic:
            mentions_by_topic[topic_key] = []
        mentions_by_topic[topic_key].append(mention)
    
    # Sort each topic's mentions by relevance, then take best from each topic
    diverse_mentions = []
    for topic_key, topic_mentions in mentions_by_topic.items():
        # Sort by relevance (longer topics first) and position
        topic_mentions.sort(key=lambda x: (-x['relevance'], x['position']))
        # Take the best mention from each topic
        diverse_mentions.append(topic_mentions[0])
    
    # Sort all diverse mentions by relevance
    diverse_mentions.sort(key=lambda x: (-x['relevance'], x['position']))
    
    # Deduplicate similar snippets (but keep different topics)
    unique_mentions = []
    seen_snippets = set()
    for mention in diverse_mentions:
        snippet_key = mention['snippet'][:100].lower()
        if snippet_key not in seen_snippets:
            seen_snippets.add(snippet_key)
            unique_mentions.append(mention)
    
    return unique_mentions[:5]  # Return top 5 most relevant mentions (one per topic)

