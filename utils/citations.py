"""
Extract and format legal citations from text.
"""

from __future__ import annotations

import re
from typing import List, Set


def extract_case_citations(text: str) -> List[str]:
    """
    Extract case citations from text.
    Looks for patterns like:
    - "Roe v. Wade"
    - "505 U.S. 833"
    - "Roe v. Wade, 410 U.S. 113 (1973)"
    - "Chevron U.S.A. Inc. v. Natural Resources Defense Council"
    """
    if not text:
        return []
    
    citations: Set[str] = set()
    
    # Pattern 1: "Case Name v. Case Name" (common case name pattern)
    # Match case names like "Roe v. Wade" or "Chevron U.S.A. Inc. v. Natural Resources Defense Council"
    case_name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z.]+)*\s+v\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z.]+)*(?:\s+[A-Z][a-z.]+)*)'
    matches = re.findall(case_name_pattern, text)
    for match in matches:
        # Clean up the match - remove leading words like "In", "See", "Under"
        citation = match.strip()
        # Remove common leading words
        for prefix in ["In ", "See ", "Under ", "Following ", "After "]:
            if citation.startswith(prefix):
                citation = citation[len(prefix):].strip()
        if len(citation) > 5 and len(citation) < 150:  # Reasonable length
            citations.add(citation)
    
    # Pattern 2: "### U.S. ###" (Supreme Court citation format)
    scotus_cite_pattern = r'\b(\d{1,4}\s+U\.S\.\s+\d{1,4}(?:\s+\(\d{4}\))?)'
    matches = re.findall(scotus_cite_pattern, text)
    citations.update(matches)
    
    # Pattern 3: "### F.2d ###" or "### F.3d ###" (Federal Reporter)
    fed_cite_pattern = r'\b(\d{1,4}\s+F\.(?:2d|3d)\s+\d{1,4})'
    matches = re.findall(fed_cite_pattern, text)
    citations.update(matches)
    
    # Pattern 4: "### S.Ct. ###" (Supreme Court Reporter)
    sct_cite_pattern = r'\b(\d{1,4}\s+S\.Ct\.\s+\d{1,4})'
    matches = re.findall(sct_cite_pattern, text)
    citations.update(matches)
    
    # Pattern 5: Combined "Case Name, ### U.S. ###"
    combined_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s+(\d{1,4}\s+U\.S\.\s+\d{1,4})'
    matches = re.findall(combined_pattern, text)
    for case_name, cite in matches:
        citations.add(f"{case_name}, {cite}")
    
    # Remove duplicates and sort
    return sorted(list(citations), key=len, reverse=True)[:10]  # Limit to 10 citations


def format_citations_inline(citations: List[str]) -> str:
    """Format citations as inline text with proper formatting."""
    if not citations:
        return ""
    
    if len(citations) == 1:
        return f"<em>(citing {citations[0]})</em>"
    elif len(citations) == 2:
        return f"<em>(citing {citations[0]} and {citations[1]})</em>"
    else:
        all_but_last = ", ".join(citations[:-1])
        return f"<em>(citing {all_but_last}, and {citations[-1]})</em>"


def extract_and_format_citations(text: str) -> tuple[str, List[str]]:
    """
    Extract citations from text and return text with citations formatted inline.
    Returns: (formatted_text, citations_list)
    """
    citations = extract_case_citations(text)
    if not citations:
        return text, []
    
    # Add citations at the end of the text
    citation_text = format_citations_inline(citations)
    if citation_text:
        return f"{text} {citation_text}", citations
    
    return text, citations

