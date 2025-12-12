## SCOTUS Brief Predictor (single feature)

Upload a PDF brief and get:
- **Predicted SCOTUS votes** (9-Justice vote map)
- **Predicted oral-argument questions** (one per Justice)
- **Optional backtest**: score predicted questions against a provided transcript URL (Oyez / SupremeCourt.gov)

This version intentionally focuses on **one product loop** and uses the **latest Google Gemini models** (deterministic tool layer) for LLM-based predictions.

**Note**: This is currently **LLM-based prediction** (not traditional ML trained on SCOTUS data). For production-grade ML-based prediction with systematic backtesting, see [`docs/ML_PREDICTION.md`](docs/ML_PREDICTION.md).

### Key guarantees
- **Async**: PDF parse + HF inference + (optional) transcript fetch run without blocking the server.
- **Deterministic tools**: the model never fetches data; Python code calls HF + transcript fetchers.
- **Prompt security**: brief text is wrapped in XML tags and treated as untrusted.
- **Structured output**: model output is validated into a single JSON object.

### Repo layout (current)
- `app.py`: Quart app with `POST /predict`
- `utils/predictor.py`: prompt + HF inference + strict JSON validation
- `utils/retrieval.py`: local historical corpus retrieval (embeddings if available; lexical fallback)
- `utils/transcripts.py`, `utils/backtest.py`: transcript fetch + scoring
- `data/historical_cases.sample.jsonl`: sample corpus format (replace with real data)
- `templates/index.html`, `static/styles.css`, `static/app.js`: single-page UI

### Setup
1. Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use Make:

```bash
make install
```

2. Create `.env`:
- This environment blocks committing/creating `.env*` via tools, so we ship `env.template`.
- Copy it to `.env` locally and fill (or use `env.local`):

```bash
cp env.template .env
```

Alternative (recommended in this repo): use `env.local`:

```bash
cp env.template env.local
```

Required:
- `GOOGLE_AI_KEY` (Google Gemini API key) — **⚠️ Without this, you'll get hardcoded fallback data instead of real predictions.** Get one at: https://aistudio.google.com/app/apikey

Optional:
- `GOOGLE_PREDICT_MODEL` (defaults to `models/gemini-2.5-pro`) — Latest models: `models/gemini-2.5-pro` (best quality), `models/gemini-2.5-flash` (fastest), `models/gemini-2.0-flash` (stable fast). **Note**: Model names MUST include the `models/` prefix.
- `GOOGLE_EMBED_MODEL` (defaults to `models/text-embedding-004`) — For retrieving similar historical cases.
- `HISTORICAL_CASES_PATH` (defaults to `data/historical_cases.jsonl`) — Path to your JSONL corpus. Run `python scripts/fetch_scotus_data.py` to populate with real SCOTUS cases.
- `RETRIEVAL_TOP_K` (defaults to `5`) — Number of similar cases to retrieve.

3. Run:

```bash
hypercorn app:app --bind 0.0.0.0:8000
```

### Hotrun (auto-reload dev runner)
Runs the server and automatically restarts on changes to `app.py`, `utils/`, `templates/`, or `static/`.

```bash
make hotrun
```

Then open `http://localhost:8000`.

### Endpoints
- `POST /predict` (multipart: `brief=<pdf>`, optional `side`, `case_hint`, `transcript_url`, `run_backtest`) → votes + questions (+ optional backtest)

### Dataset backtest (offline)
If you have a JSONL dataset with `brief_text` (or `brief_pdf_path`) and `transcript_url` (or `transcript_text`), you can backtest question predictions:

```bash
.venv/bin/python scripts/backtest_dataset.py --dataset /path/to/your_backtest.jsonl --max-cases 25
```

### Notes (legal)
This is an engineering tool and **not legal advice**. Predictions are probabilistic and should be treated as research support, not authority.


