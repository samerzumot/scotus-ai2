"""
Auto-detect transcript URLs from case names and hints.
"""

from __future__ import annotations

import re
from typing import List, Optional

from utils.security import sanitize_user_text


def normalize_case_name(case_name: str) -> str:
    """Normalize case name for URL construction."""
    # Remove common suffixes
    case_name = re.sub(r'\s+v\.\s+', ' v ', case_name, flags=re.IGNORECASE)
    case_name = re.sub(r'\s+et\s+al\.?', '', case_name, flags=re.IGNORECASE)
    case_name = re.sub(r'\s+Inc\.?', '', case_name, flags=re.IGNORECASE)
    case_name = re.sub(r'\s+Corp\.?', '', case_name, flags=re.IGNORECASE)
    case_name = re.sub(r'\s+LLC', '', case_name, flags=re.IGNORECASE)
    # Remove special chars
    case_name = re.sub(r'[^\w\s&]', '', case_name)
    return case_name.strip()


def find_oyez_transcript_url(case_name: str, term: Optional[int] = None) -> Optional[str]:
    """
    Generate a potential Oyez transcript URL from case name.
    Oyez URLs are typically: https://www.oyez.org/cases/{term}/{slug}
    
    Note: This generates estimated URLs. Oyez uses specific slugs that may not match
    our generated slug. The URL should be verified before use.
    """
    if not case_name:
        return None
    
    # Normalize case name
    normalized = normalize_case_name(case_name)
    if not normalized:
        return None
    
    # Create slug (lowercase, replace spaces with hyphens, remove special chars)
    # Oyez slugs are typically: case-name-v-case-name (lowercase, hyphens)
    slug = normalized.lower()
    # Remove common legal suffixes
    slug = re.sub(r'\s+v\.?\s+', '-v-', slug)
    slug = re.sub(r'\s+', '-', slug)
    # Remove special characters but keep hyphens
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    
    if not slug or len(slug) < 3:
        return None
    
    # If we have a term, use it; otherwise try recent terms
    if term:
        return f"https://www.oyez.org/cases/{term}/{slug}"
    
    # Try recent terms (2020-2024) - return first attempt
    # Note: Without term, this is less reliable
    for year in range(2024, 2019, -1):
        url = f"https://www.oyez.org/cases/{year}/{slug}"
        return url
    
    return None


def find_scotus_transcript_url(case_name: str, term: Optional[int] = None, docket: Optional[str] = None) -> Optional[str]:
    """
    Generate a potential SCOTUS.gov transcript URL.
    SCOTUS.gov URLs are: https://www.supremecourt.gov/oral_arguments/argument_transcripts/{year}/{docket}_{suffix}.pdf
    
    Note: The actual URL format includes a suffix (e.g., "25-332_7lhn.pdf").
    We'll try the base format first, and the fetcher can handle variations.
    
    If docket is provided, use it directly. Otherwise, try to extract from case name or use common patterns.
    """
    if not case_name:
        return None
    
    # If we have a docket number, use it directly
    if docket:
        docket_clean = re.sub(r'[^0-9-]', '', docket)
        if docket_clean and term:
            # Try base format first (most common)
            base_url = f"https://www.supremecourt.gov/oral_arguments/argument_transcripts/{term}/{docket_clean}.pdf"
            # Also try with common suffixes (the fetcher will verify)
            return base_url
    
    # Try to extract docket from case name (e.g., "No. 21-1234" or "21-1234")
    docket_match = re.search(r'(?:no\.?\s*)?(\d{2}-\d{2,4})', case_name, re.IGNORECASE)
    if docket_match and term:
        docket_num = docket_match.group(1)
        return f"https://www.supremecourt.gov/oral_arguments/argument_transcripts/{term}/{docket_num}.pdf"
    
    # For well-known cases, we can try common docket patterns
    # But without a docket database, this is limited
    # We'll rely on Oyez for most cases, and SCOTUS.gov when docket is known
    return None


def find_transcript_urls(case_name: str, term: Optional[int] = None, docket: Optional[str] = None) -> List[str]:
    """
    Try to find transcript URLs from case name, term, and optionally docket number.
    Returns a list of candidate URLs (most likely first).
    
    Tries multiple sources:
    1. SCOTUS.gov (if docket is available) - most authoritative
    2. Oyez.org - most comprehensive for public transcripts
    """
    candidates: List[str] = []
    
    # Try SCOTUS.gov first if we have docket (most authoritative)
    if docket or (term and re.search(r'\d{2}-\d{2,4}', case_name)):
        scotus_url = find_scotus_transcript_url(case_name, term, docket)
        if scotus_url:
            candidates.append(scotus_url)
    
    # Try Oyez (most reliable for public transcripts, works without docket)
    oyez_url = find_oyez_transcript_url(case_name, term)
    if oyez_url:
        candidates.append(oyez_url)
    
    return candidates


def extract_case_name_from_hint(hint: str) -> Optional[str]:
    """Extract a case name from a case hint string."""
    if not hint:
        return None
    
    # Look for patterns like "Case Name v. Defendant" or "Case Name (2024)"
    # Remove common prefixes
    hint = re.sub(r'^(case|in|re|ex parte)\s+', '', hint, flags=re.IGNORECASE)
    
    # Extract before "v." or "vs." or "("
    match = re.search(r'^([^v(]+?)(?:\s+v\.?\s+|vs\.?\s+|\(|$)', hint, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # If no pattern, return first 100 chars as potential case name
    return hint[:100].strip() if hint else None

