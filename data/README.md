## Historical corpus format

`utils/retrieval.py` expects a JSONL file (one JSON object per line). Minimal fields:

```json
{
  "case_id": "unique-id",
  "case_name": "Case Name",
  "term": 2024,
  "tags": ["admin law", "standing"],
  "outcome": "optional string",
  "summary": "short text used for retrieval",
  "embedding": [0.0123, -0.0456, "... optional precomputed vector ..."]
}
```

### Populating with real data

**Quick start** (uses known cases):
```bash
python scripts/fetch_scotus_data.py --output data/historical_cases.jsonl --limit 100
```

**Add more cases**:
```bash
python scripts/fetch_more_cases.py --add-to data/historical_cases.jsonl
```

**Fetch from APIs** (best-effort, may require API keys):
```bash
python scripts/fetch_scotus_data.py --output data/historical_cases.jsonl --limit 200
```

### Notes
- If `embedding` is missing and `GOOGLE_EMBED_MODEL` is configured, the app will **best-effort** embed small corpora automatically.
- For large corpora, precompute and store embeddings offline (to avoid N API calls at runtime).
- The current corpus includes notable SCOTUS cases from 2010-2024 with actual outcomes and legal tags.

### Data sources
- **Known cases**: Curated list of notable SCOTUS cases (2010-2024)
- **Oyez API**: Public API for case data (best-effort)
- **SCOTUSblog**: Case summaries and analysis (scraping, best-effort)

### Building a larger corpus

**Add cases with transcript URLs:**
```bash
python scripts/fetch_cases_with_transcripts.py --add-to data/historical_cases.jsonl --limit 200
```

This script:
- Adds cases with known docket numbers (for SCOTUS.gov transcripts)
- Verifies transcript URLs are accessible
- Includes both SCOTUS.gov and Oyez.org transcript links

**For production use, consider:**
1. **ConvoKit Supreme Court Corpus** - 1.8M utterances from 8,300+ transcripts (1955-2023)
2. **WalkerDB Supreme Court Transcripts** - Structured data with audio timestamps
3. **Scraping SCOTUSblog** systematically (with rate limiting)
4. **Using CourtListener API** (requires API key, has comprehensive docket data)
5. **Academic datasets** (e.g., Supreme Court Database, SCOTUSvotes)

**Note on poor alignment:** If backtest scores are low (<30%), the model likely needs:
- **More training data**: Expand corpus to 1000+ cases with actual votes and questions
- **Better grounding**: Include more similar historical cases in retrieval (increase `RETRIEVAL_TOP_K`)
- **Fine-tuning**: Train on SCOTUS-specific question patterns rather than general LLM knowledge
