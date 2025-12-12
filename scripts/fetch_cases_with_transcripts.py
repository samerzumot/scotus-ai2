#!/usr/bin/env python3
"""
Fetch SCOTUS cases with transcript URLs from multiple sources.

This script builds a comprehensive corpus with transcript links for better backtesting.
Uses known case databases and attempts to find transcript URLs for each case.

Usage:
    python scripts/fetch_cases_with_transcripts.py --output data/historical_cases.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp
from bs4 import BeautifulSoup

from utils.transcript_finder import find_transcript_urls


# Extended list of cases with known docket numbers (for SCOTUS.gov transcripts)
CASES_WITH_DOCKETS = {
    "Dobbs v. Jackson Women's Health Organization": ("21-830", 2022),
    "New York State Rifle & Pistol Association v. Bruen": ("20-843", 2022),
    "West Virginia v. EPA": ("20-1530", 2022),
    "Students for Fair Admissions v. Harvard": ("20-1199", 2023),
    "303 Creative LLC v. Elenis": ("21-476", 2023),
    "United States v. Rahimi": ("22-915", 2024),
    "Loper Bright Enterprises v. Raimondo": ("22-451", 2024),
    "Trump v. United States": ("23-939", 2024),
    "Moore v. United States": ("22-800", 2024),
    "Grants Pass v. Johnson": ("23-175", 2024),
    "Biden v. Nebraska": ("22-506", 2023),
    "Counterman v. Colorado": ("22-138", 2023),
    "Groff v. DeJoy": ("22-174", 2023),
    "Fulton v. City of Philadelphia": ("19-123", 2021),
    "Brnovich v. Democratic National Committee": ("19-1257", 2021),
    "Mahanoy Area School District v. B.L.": ("20-255", 2021),
    "California v. Texas": ("19-840", 2021),
    "Bostock v. Clayton County": ("17-1618", 2020),
    "Department of Homeland Security v. Regents of the University of California": ("18-587", 2020),
    "Espinoza v. Montana Department of Revenue": ("18-1195", 2020),
    "June Medical Services v. Russo": ("18-1323", 2020),
    "Rucho v. Common Cause": ("18-422", 2019),
    "Department of Commerce v. New York": ("18-966", 2019),
    "Kisor v. Wilkie": ("18-15", 2019),
    "Janus v. AFSCME": ("16-1466", 2018),
    "Masterpiece Cakeshop v. Colorado Civil Rights Commission": ("16-111", 2018),
    "South Dakota v. Wayfair": ("17-494", 2018),
    "Trump v. Hawaii": ("17-965", 2018),
    "Sessions v. Dimaya": ("15-1498", 2018),
    "Whole Woman's Health v. Hellerstedt": ("14-1378", 2016),
    "Fisher v. University of Texas": ("14-981", 2016),
    "Utah v. Strieff": ("14-1373", 2016),
    "Obergefell v. Hodges": ("14-556", 2015),
    "King v. Burwell": ("14-114", 2015),
    "Arizona State Legislature v. Arizona Independent Redistricting Commission": ("13-1314", 2015),
}


async def verify_transcript_url(session: aiohttp.ClientSession, url: str) -> bool:
    """Check if a transcript URL is accessible."""
    try:
        async with session.head(url, timeout=aiohttp.ClientTimeout(total=5), allow_redirects=True) as resp:
            return resp.status == 200
    except Exception:
        return False


async def find_best_transcript_url(
    session: aiohttp.ClientSession,
    case_name: str,
    term: Optional[int] = None,
    docket: Optional[str] = None,
) -> Optional[str]:
    """Find the best available transcript URL for a case."""
    candidates = find_transcript_urls(case_name, term=term, docket=docket)
    
    # Verify URLs in order (SCOTUS.gov first, then Oyez)
    for url in candidates:
        if await verify_transcript_url(session, url):
            return url
    
    # If verification fails, return first candidate anyway (might work on full GET)
    return candidates[0] if candidates else None


async def load_existing_cases(path: str) -> Set[str]:
    """Load existing case names to avoid duplicates."""
    if not os.path.exists(path):
        return set()
    
    existing = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
                name = (case.get("case_name") or "").lower()
                if name:
                    existing.add(name)
            except Exception:
                continue
    return existing


async def main_async(output_path: str, limit: int = 200, add_to: bool = False) -> int:
    """Main async function."""
    existing = await load_existing_cases(output_path) if add_to else set()
    
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        cases: List[Dict[str, Any]] = []
        
        for case_name, (docket, term) in CASES_WITH_DOCKETS.items():
            if len(cases) >= limit:
                break
            
            name_key = case_name.lower()
            if name_key in existing:
                continue
            
            # Find transcript URL
            transcript_url = await find_best_transcript_url(session, case_name, term=term, docket=docket)
            
            # Extract tags from case name and known info
            tags = []
            if "amendment" in case_name.lower() or any(am in case_name for am in ["First", "Second", "Fourth", "Eighth", "Fourteenth"]):
                tags.append("constitutional law")
            if "administrative" in case_name.lower() or "EPA" in case_name or "agency" in case_name.lower():
                tags.append("administrative law")
            
            case_id = f"scotus-{case_name.lower().replace(' ', '-').replace('&', 'and').replace('.', '')[:60]}"
            case_id = re.sub(r'[^a-z0-9-]', '', case_id)
            
            cases.append({
                "case_id": case_id,
                "case_name": case_name,
                "term": term,
                "tags": tags,
                "outcome": None,  # Would need to fetch from database
                "summary": f"SCOTUS case from {term} term.",
                "transcript_url": transcript_url,
                "docket": docket,
            })
            
            # Rate limiting
            await asyncio.sleep(0.3)
        
        # Write to JSONL
        mode = "a" if add_to else "w"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, mode, encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        
        print(f"✅ Added {len(cases)} cases with transcript URLs to {output_path}", file=sys.stderr)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch SCOTUS cases with transcript URLs")
    parser.add_argument("--output", default="data/historical_cases.jsonl", help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of cases to fetch")
    parser.add_argument("--add-to", action="store_true", help="Append to existing file (skip duplicates)")
    args = parser.parse_args()
    
    return asyncio.run(main_async(args.output, limit=args.limit, add_to=args.add_to))


if __name__ == "__main__":
    raise SystemExit(main())

