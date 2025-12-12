"""
Simple file-based cache for transcripts to avoid re-fetching.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "transcripts"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(url: str) -> str:
    """Generate a cache key from URL."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def get_cached_transcript(url: str) -> Optional[Dict[str, Any]]:
    """Get cached transcript if available."""
    key = _cache_key(url)
    cache_file = _CACHE_DIR / f"{key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Verify it's not too old (30 days)
                import time
                if time.time() - data.get("cached_at", 0) < 30 * 24 * 60 * 60:
                    return data.get("transcript")
        except Exception:
            pass
    
    return None


def cache_transcript(url: str, transcript_data: Dict[str, Any]) -> None:
    """Cache transcript data."""
    key = _cache_key(url)
    cache_file = _CACHE_DIR / f"{key}.json"
    
    try:
        import time
        data = {
            "url": url,
            "cached_at": time.time(),
            "transcript": transcript_data,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass  # Cache failures shouldn't break the app


def clear_cache() -> None:
    """Clear all cached transcripts."""
    try:
        for cache_file in _CACHE_DIR.glob("*.json"):
            cache_file.unlink()
    except Exception:
        pass

