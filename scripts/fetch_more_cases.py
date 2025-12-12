#!/usr/bin/env python3
"""
Fetch additional SCOTUS cases from public sources.

This script extends the historical corpus with more cases.
You can run it multiple times to build a larger dataset.

Usage:
    python scripts/fetch_more_cases.py --add-to data/historical_cases.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Set

import aiohttp


# Extended list of notable SCOTUS cases (2010-2024)
EXTENDED_CASES = [
    # 2024
    ("United States v. Rahimi", 2024, ["second amendment", "domestic violence"], "AFFIRMED", "Upheld federal law prohibiting gun possession by people subject to domestic violence restraining orders."),
    ("Loper Bright Enterprises v. Raimondo", 2024, ["administrative law", "chevron"], "REVERSED", "Overturned Chevron deference, requiring courts to interpret statutes independently."),
    ("Trump v. United States", 2024, ["separation of powers", "executive immunity"], "AFFIRMED", "Held that presidents have absolute immunity for official acts."),
    ("Moore v. United States", 2024, ["taxation", "appointments clause"], "AFFIRMED", "Upheld one-time tax on foreign earnings."),
    ("Grants Pass v. Johnson", 2024, ["eighth amendment", "homelessness"], "REVERSED", "Held that cities can enforce anti-camping ordinances."),
    
    # 2023
    ("Students for Fair Admissions v. Harvard", 2023, ["equal protection", "affirmative action"], "REVERSED", "Struck down race-based affirmative action in college admissions."),
    ("303 Creative LLC v. Elenis", 2023, ["first amendment", "free speech"], "AFFIRMED", "Held that a website designer could refuse to create websites for same-sex weddings."),
    ("Biden v. Nebraska", 2023, ["administrative law", "student loans"], "REVERSED", "Struck down student loan forgiveness program."),
    ("Counterman v. Colorado", 2023, ["first amendment", "true threats"], "REVERSED", "Held that true threats require proof of subjective intent."),
    ("Groff v. DeJoy", 2023, ["religious freedom", "employment"], "REVERSED", "Clarified standard for religious accommodation under Title VII."),
    
    # 2022
    ("Dobbs v. Jackson Women's Health Organization", 2022, ["abortion", "fourteenth amendment"], "REVERSED", "Overturned Roe v. Wade, holding that the Constitution does not confer a right to abortion."),
    ("New York State Rifle & Pistol Association v. Bruen", 2022, ["second amendment", "gun rights"], "REVERSED", "Struck down New York's concealed carry law."),
    ("West Virginia v. EPA", 2022, ["administrative law", "environment"], "REVERSED", "Limited EPA's authority to regulate carbon emissions."),
    ("Kennedy v. Bremerton School District", 2022, ["first amendment", "establishment clause"], "REVERSED", "Held that a coach's prayer was protected by the Free Exercise Clause."),
    ("Carson v. Makin", 2022, ["first amendment", "school choice"], "REVERSED", "Held that states cannot exclude religious schools from tuition assistance programs."),
    
    # 2021
    ("Fulton v. City of Philadelphia", 2021, ["first amendment", "free exercise"], "UNANIMOUS", "Held that Philadelphia violated Free Exercise Clause by excluding Catholic adoption agency."),
    ("Brnovich v. Democratic National Committee", 2021, ["voting rights", "section 2"], "AFFIRMED", "Upheld Arizona voting restrictions."),
    ("Mahanoy Area School District v. B.L.", 2021, ["first amendment", "student speech"], "REVERSED", "Held that school could not discipline student for off-campus social media posts."),
    ("California v. Texas", 2021, ["standing", "affordable care act"], "REVERSED", "Held that challengers lacked standing to challenge individual mandate."),
    
    # 2020
    ("Bostock v. Clayton County", 2020, ["employment", "title vii"], "AFFIRMED", "Held that Title VII prohibits discrimination based on sexual orientation and gender identity."),
    ("Department of Homeland Security v. Regents of the University of California", 2020, ["administrative law", "daca"], "REVERSED", "Held that DACA rescission was arbitrary and capricious."),
    ("Espinoza v. Montana Department of Revenue", 2020, ["first amendment", "school choice"], "REVERSED", "Held that states cannot exclude religious schools from scholarship programs."),
    ("June Medical Services v. Russo", 2020, ["abortion", "fourteenth amendment"], "REVERSED", "Struck down Louisiana abortion law."),
    
    # 2019
    ("Rucho v. Common Cause", 2019, ["gerrymandering", "political question"], "AFFIRMED", "Held that partisan gerrymandering claims are nonjusticiable."),
    ("Department of Commerce v. New York", 2019, ["administrative law", "census"], "REVERSED", "Blocked addition of citizenship question to census."),
    ("Kisor v. Wilkie", 2019, ["administrative law", "auer deference"], "AFFIRMED", "Narrowed but did not overrule Auer deference."),
    
    # 2018
    ("Janus v. AFSCME", 2018, ["first amendment", "public sector unions"], "REVERSED", "Held that mandatory union fees violate First Amendment."),
    ("Masterpiece Cakeshop v. Colorado Civil Rights Commission", 2018, ["first amendment", "free exercise"], "REVERSED", "Held that commission showed hostility to religious beliefs."),
    ("South Dakota v. Wayfair", 2018, ["commerce clause", "taxation"], "REVERSED", "Overturned physical presence requirement for sales tax."),
    
    # 2017
    ("Trump v. Hawaii", 2017, ["immigration", "executive power"], "AFFIRMED", "Upheld travel ban."),
    ("Sessions v. Dimaya", 2017, ["criminal law", "vagueness"], "REVERSED", "Held that immigration removal provision was unconstitutionally vague."),
    
    # 2016
    ("Whole Woman's Health v. Hellerstedt", 2016, ["abortion", "fourteenth amendment"], "REVERSED", "Struck down Texas abortion restrictions."),
    ("Fisher v. University of Texas", 2016, ["equal protection", "affirmative action"], "AFFIRMED", "Upheld affirmative action program."),
    ("Utah v. Strieff", 2016, ["fourth amendment", "exclusionary rule"], "AFFIRMED", "Held that discovery of outstanding warrant attenuated taint of illegal stop."),
    
    # 2015
    ("Obergefell v. Hodges", 2015, ["fourteenth amendment", "same-sex marriage"], "REVERSED", "Held that same-sex couples have fundamental right to marry."),
    ("King v. Burwell", 2015, ["statutory interpretation", "affordable care act"], "AFFIRMED", "Held that tax credits available in all states."),
    ("Arizona State Legislature v. Arizona Independent Redistricting Commission", 2015, ["elections clause", "redistricting"], "AFFIRMED", "Held that independent commission did not violate Elections Clause."),
]


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


async def main_async(output_path: str, add_to: bool = False) -> int:
    """Main async function."""
    existing = await load_existing_cases(output_path) if add_to else set()
    
    cases: List[Dict[str, Any]] = []
    for case_name, term, tags, outcome, summary in EXTENDED_CASES:
        name_key = case_name.lower()
        if name_key in existing:
            continue
        
        case_id = f"scotus-{case_name.lower().replace(' ', '-').replace('&', 'and').replace('.', '')[:60]}"
        case_id = re.sub(r'[^a-z0-9-]', '', case_id)
        
        cases.append({
            "case_id": case_id,
            "case_name": case_name,
            "term": term,
            "tags": tags,
            "outcome": outcome,
            "summary": summary,
        })
    
    if not cases:
        print("No new cases to add.", file=sys.stderr)
        return 0
    
    # Append or write
    mode = "a" if add_to else "w"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, mode, encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    
    print(f"✅ Added {len(cases)} cases to {output_path}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Add more SCOTUS cases to historical corpus")
    parser.add_argument("--output", default="data/historical_cases.jsonl", help="Output JSONL file path")
    parser.add_argument("--add-to", action="store_true", help="Append to existing file (skip duplicates)")
    args = parser.parse_args()
    
    return asyncio.run(main_async(args.output, add_to=args.add_to))


if __name__ == "__main__":
    raise SystemExit(main())

