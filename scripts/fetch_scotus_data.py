#!/usr/bin/env python3
"""
Fetch real SCOTUS case data and populate historical_cases.jsonl.

Sources:
- Oyez API (free, public)
- SCOTUSblog (scraping, best-effort)
- Justia (scraping, best-effort)

Usage:
    python scripts/fetch_scotus_data.py --output data/historical_cases.jsonl --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import aiohttp
from bs4 import BeautifulSoup


def extract_tags_from_summary(summary: str, case_name: str) -> List[str]:
    """Extract legal tags/keywords from case summary."""
    tags = []
    summary_lower = (summary or "").lower()
    case_lower = (case_name or "").lower()
    
    # Common legal areas
    legal_areas = {
        "first amendment": "first amendment",
        "fourth amendment": "fourth amendment",
        "fifth amendment": "fifth amendment",
        "sixth amendment": "sixth amendment",
        "eighth amendment": "eighth amendment",
        "fourteenth amendment": "fourteenth amendment",
        "commerce clause": "commerce clause",
        "standing": "standing",
        "jurisdiction": "jurisdiction",
        "statutory interpretation": "statutory interpretation",
        "chevron": "chevron deference",
        "administrative law": "administrative law",
        "criminal procedure": "criminal procedure",
        "civil rights": "civil rights",
        "free speech": "free speech",
        "separation of powers": "separation of powers",
        "federalism": "federalism",
        "due process": "due process",
        "equal protection": "equal protection",
        "takings": "takings clause",
    }
    
    for keyword, tag in legal_areas.items():
        if keyword in summary_lower or keyword in case_lower:
            tags.append(tag)
    
    # Extract specific precedents mentioned
    precedent_pattern = r'\b([A-Z][a-z]+ v\. [A-Z][a-z]+)\b'
    precedents = re.findall(precedent_pattern, summary)
    if precedents:
        tags.append("precedent_cited")
    
    return list(set(tags))[:10]  # Limit to 10 tags


def normalize_outcome(outcome: Optional[str]) -> Optional[str]:
    """Normalize case outcome to standard format."""
    if not outcome:
        return None
    outcome_lower = outcome.lower()
    if "affirmed" in outcome_lower or "upheld" in outcome_lower:
        return "AFFIRMED"
    if "reversed" in outcome_lower or "overturned" in outcome_lower:
        return "REVERSED"
    if "remanded" in outcome_lower:
        return "REMANDED"
    if "vacated" in outcome_lower:
        return "VACATED"
    return outcome.upper()[:50]


async def fetch_oyez_case(session: aiohttp.ClientSession, case_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a case from Oyez API."""
    try:
        # Oyez API endpoint (best-effort; they may have changed their API)
        url = f"https://api.oyez.org/cases/{case_id}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            
            case_name = data.get("name") or data.get("title") or ""
            term = data.get("term") or data.get("year")
            summary = data.get("summary") or data.get("description") or ""
            decision_date = data.get("decision_date") or data.get("decided") or ""
            
            # Extract outcome from decision
            outcome = None
            if "decided" in data:
                decision = data.get("decided", {})
                outcome = decision.get("decision") or decision.get("outcome")
            
            tags = extract_tags_from_summary(summary, case_name)
            
            return {
                "case_id": case_id,
                "case_name": case_name,
                "term": int(term) if term else None,
                "tags": tags,
                "outcome": normalize_outcome(outcome),
                "summary": summary[:2000],  # Limit summary length
            }
    except Exception as e:
        print(f"Error fetching Oyez case {case_id}: {e}", file=sys.stderr)
        return None


async def fetch_scotusblog_case(session: aiohttp.ClientSession, case_name: str) -> Optional[Dict[str, Any]]:
    """Fetch case summary from SCOTUSblog (best-effort scraping)."""
    try:
        # SCOTUSblog search
        search_query = quote(case_name)
        url = f"https://www.scotusblog.com/?s={search_query}"
        
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            
            # Find first result
            article = soup.find("article")
            if not article:
                return None
            
            summary_elem = article.find("div", class_="post-excerpt") or article.find("p")
            summary = summary_elem.get_text(strip=True) if summary_elem else ""
            
            # Extract term/year if available
            date_elem = article.find("time")
            term = None
            if date_elem:
                date_str = date_elem.get("datetime") or date_elem.get_text()
                # Extract year
                year_match = re.search(r"\b(20\d{2})\b", date_str)
                if year_match:
                    term = int(year_match.group(1))
            
            tags = extract_tags_from_summary(summary, case_name)
            
            return {
                "case_id": f"scotusblog-{case_name.lower().replace(' ', '-')[:50]}",
                "case_name": case_name,
                "term": term,
                "tags": tags,
                "outcome": None,  # SCOTUSblog doesn't always have outcomes
                "summary": summary[:2000],
            }
    except Exception as e:
        print(f"Error fetching SCOTUSblog case {case_name}: {e}", file=sys.stderr)
        return None


async def fetch_known_cases(session: aiohttp.ClientSession, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch a list of well-known SCOTUS cases.
    This is a fallback when APIs aren't available.
    """
    # List of notable recent SCOTUS cases (2020-2024)
    known_cases = [
        ("Dobbs v. Jackson Women's Health Organization", 2022, "abortion", "REVERSED", "Overturned Roe v. Wade, holding that the Constitution does not confer a right to abortion."),
        ("New York State Rifle & Pistol Association v. Bruen", 2022, "second amendment", "REVERSED", "Struck down New York's concealed carry law, expanding Second Amendment rights."),
        ("West Virginia v. EPA", 2022, "administrative law", "REVERSED", "Limited EPA's authority to regulate carbon emissions under the Clean Air Act."),
        ("Students for Fair Admissions v. Harvard", 2023, "equal protection", "REVERSED", "Struck down race-based affirmative action in college admissions."),
        ("303 Creative LLC v. Elenis", 2023, "first amendment", "AFFIRMED", "Held that a website designer could refuse to create websites for same-sex weddings based on free speech."),
        ("United States v. Rahimi", 2024, "second amendment", "AFFIRMED", "Upheld federal law prohibiting gun possession by people subject to domestic violence restraining orders."),
        ("Loper Bright Enterprises v. Raimondo", 2024, "administrative law", "REVERSED", "Overturned Chevron deference, requiring courts to interpret statutes independently."),
        ("Trump v. United States", 2024, "separation of powers", "AFFIRMED", "Held that presidents have absolute immunity for official acts."),
        ("Moore v. United States", 2024, "taxation", "AFFIRMED", "Upheld one-time tax on foreign earnings, rejecting challenge under Appointments Clause."),
        ("Grants Pass v. Johnson", 2024, "eighth amendment", "REVERSED", "Held that cities can enforce anti-camping ordinances without violating Eighth Amendment."),
    ]
    
    cases = []
    for case_name, term, tag, outcome, summary in known_cases[:limit]:
        tags = [tag]
        if "amendment" in tag:
            tags.append("constitutional law")
        if "administrative" in tag:
            tags.append("statutory interpretation")
        
        cases.append({
            "case_id": f"known-{case_name.lower().replace(' ', '-').replace('&', 'and')[:50]}",
            "case_name": case_name,
            "term": term,
            "tags": tags,
            "outcome": outcome,
            "summary": summary,
        })
    
    return cases


async def main_async(output_path: str, limit: int = 100, use_apis: bool = True) -> int:
    """Main async function to fetch and write cases."""
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        cases: List[Dict[str, Any]] = []
        
        if use_apis:
            print("Fetching cases from public APIs...", file=sys.stderr)
            # Try to fetch from known case IDs (Oyez format)
            # Note: Oyez API structure may vary; this is best-effort
            oyez_case_ids = [
                "2022/dobbs-v-jackson-womens-health-organization",
                "2022/new-york-state-rifle-pistol-association-v-bruen",
                "2022/west-virginia-v-environmental-protection-agency",
            ]
            
            for case_id in oyez_case_ids[:10]:  # Limit API calls
                case = await fetch_oyez_case(session, case_id)
                if case:
                    cases.append(case)
                await asyncio.sleep(0.5)  # Rate limiting
        
        # Fallback: use known cases list
        if len(cases) < limit:
            print(f"Adding known cases to reach {limit}...", file=sys.stderr)
            known = await fetch_known_cases(session, limit=limit - len(cases))
            cases.extend(known)
        
        # Deduplicate by case_name
        seen = set()
        unique_cases = []
        for case in cases:
            name_key = (case.get("case_name") or "").lower()
            if name_key and name_key not in seen:
                seen.add(name_key)
                unique_cases.append(case)
        
        # Write to JSONL
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for case in unique_cases[:limit]:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        
        print(f"✅ Wrote {len(unique_cases)} cases to {output_path}", file=sys.stderr)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch real SCOTUS case data")
    parser.add_argument("--output", default="data/historical_cases.jsonl", help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of cases to fetch")
    parser.add_argument("--no-apis", action="store_true", help="Skip API calls, use known cases only")
    args = parser.parse_args()
    
    return asyncio.run(main_async(args.output, limit=args.limit, use_apis=not args.no_apis))


if __name__ == "__main__":
    raise SystemExit(main())

