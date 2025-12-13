"""
Use Google Custom Search API to find accurate case links (Oyez, SCOTUS.gov, etc.).
"""

import os
from typing import Optional

import aiohttp


async def find_case_link_via_search(
    case_name: str,
    term: Optional[int] = None,
    *,
    session: aiohttp.ClientSession,
) -> Optional[str]:
    """
    Use Google Custom Search API to find the correct case link.
    Searches for Oyez.org and SCOTUS.gov links.
    
    Returns the first matching case link, or None if not found.
    """
    google_search_key = (os.getenv("GOOGLE_SEARCH_KEY") or "").strip()
    search_engine_id = (os.getenv("SEARCH_ENGINE_ID") or "").strip()
    
    if not google_search_key or not search_engine_id:
        return None
    
    # Build search query: case name + term if available
    query_parts = [case_name]
    if term:
        query_parts.append(str(term))
    
    query = " ".join(query_parts)
    
    # Search for Oyez.org and SCOTUS.gov links
    # We'll search without site restriction first to get best results
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": google_search_key,
            "cx": search_engine_id,
            "q": query,
            "num": 10,  # Get top 10 results to find best match
        }
        
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            
            data = await resp.json()
            
            # Extract URLs from search results
            items = data.get("items", [])
            
            # Priority 1: Oyez.org case pages (most comprehensive)
            for item in items:
                link = item.get("link", "")
                # Look for Oyez case pages
                if "oyez.org" in link and ("cases" in link or "case" in link.lower()):
                    # Verify it's a case page, not a transcript or other page
                    if "/cases/" in link:
                        return link
            
            # Priority 2: SCOTUS.gov transcript or case pages
            for item in items:
                link = item.get("link", "")
                if "supremecourt.gov" in link:
                    # Prefer transcript URLs, but accept any SCOTUS.gov link
                    if "oral_arguments" in link or "cases" in link:
                        return link
            
            # Priority 3: Any Oyez.org link (fallback)
            for item in items:
                link = item.get("link", "")
                if "oyez.org" in link:
                    return link
            
            return None
    except Exception:
        # Silently fail - return None
        return None

