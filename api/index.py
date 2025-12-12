"""
Vercel serverless adapter for Quart app.
Note: Vercel has limitations with async/await and long-running requests.
For better performance, consider Railway, Render, or Fly.io instead.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app


def handler(request):
    """Vercel serverless handler."""
    # Convert Vercel request to Quart-compatible format
    # Note: This is a simplified adapter - full async support may be limited
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Use Quart's ASGI adapter
        from hypercorn.asyncio import serve
        # This won't work directly - Vercel needs a different approach
        # For now, return a simple response
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "text/html"},
            "body": "<html><body><h1>SCOTUS AI</h1><p>For full functionality, deploy to Railway, Render, or Fly.io instead of Vercel.</p><p>Vercel has limitations with async Python apps.</p></body></html>"
        }
    finally:
        loop.close()

