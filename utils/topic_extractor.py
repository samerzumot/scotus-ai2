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
    # Keep original for capitalized phrase extraction
    question_original = question
    question_lower = question.lower()
    topics: Set[str] = set()
    
    # Extract quoted phrases (often key legal concepts)
    quoted = re.findall(r'"([^"]+)"', question_original)
    topics.update(set(q.lower() for q in quoted))
    
    # Extract capitalized phrases (proper nouns, case names, institutions)
    # Look for patterns like "Federal Reserve", "Humphrey's Executor", etc.
    # Extract BEFORE lowercasing to catch proper nouns
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question_original)
    for cap_phrase in capitalized:
        if len(cap_phrase) > 3:
            # Add both original and lowercase versions
            topics.add(cap_phrase.lower())
            # Also extract individual capitalized words as potential topics
            words = cap_phrase.split()
            if len(words) >= 2:
                # For multi-word phrases, also add the key words separately
                for word in words:
                    if len(word) > 4 and word[0].isupper():
                        topics.add(word.lower())
    
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
        matches = re.findall(pattern, question_lower, re.IGNORECASE)
        # Flatten list of tuples if needed
        flat_matches = []
        for m in matches:
            if isinstance(m, tuple):
                flat_matches.extend(m)
            else:
                flat_matches.append(m)
        topics.update(set(flat_matches))
    
    # Also extract "Federal Reserve" as a standalone topic if it appears
    if re.search(r'\bfederal\s+reserve\b', question_lower, re.IGNORECASE):
        topics.add('federal reserve')
    
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

