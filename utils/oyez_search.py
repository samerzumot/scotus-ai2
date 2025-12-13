"""
Use Google Custom Search API to find accurate Oyez transcript URLs.
"""

import os
from typing import List, Optional

import aiohttp


async def find_oyez_url_via_search(
    case_name: str,
    term: Optional[int] = None,
    *,
    session: aiohttp.ClientSession,
) -> Optional[str]:
    """
    Use Google Custom Search API restricted to oyez.org to find the correct transcript URL.
    
    Returns the first matching Oyez URL, or None if not found.
    """
    google_search_key = (os.getenv("GOOGLE_SEARCH_KEY") or "").strip()
    search_engine_id = (os.getenv("SEARCH_ENGINE_ID") or "").strip()
    
    if not google_search_key or not search_engine_id:
        return None
    
    # Build search query: case name + "oyez" + term if available
    query_parts = [case_name, "oyez"]
    if term:
        query_parts.append(str(term))
    
    query = " ".join(query_parts)
    
    # Restrict to oyez.org domain
    site_restrict = "site:oyez.org"
    
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": google_search_key,
            "cx": search_engine_id,
            "q": f"{query} {site_restrict}",
            "num": 5,  # Get top 5 results
        }
        
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            
            data = await resp.json()
            
            # Extract URLs from search results
            items = data.get("items", [])
            for item in items:
                link = item.get("link", "")
                # Verify it's an Oyez transcript URL
                if "oyez.org" in link and ("cases" in link or "transcript" in link.lower()):
                    return link
            
            return None
    except Exception:
        # Silently fail - return None
        return None

