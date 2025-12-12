# Deployment Guide

This app can be deployed to various platforms. Choose based on your needs:

## 🚀 Recommended: Railway / Render / Fly.io

These platforms are **best suited** for Quart async Python apps:

### Railway (Recommended)
1. Sign up at [railway.app](https://railway.app)
2. Create new project → Deploy from GitHub
3. Add environment variables:
   - `GOOGLE_AI_KEY`
   - `GOOGLE_PREDICT_MODEL` (optional)
   - `GOOGLE_EMBED_MODEL` (optional)
4. Railway auto-detects Python and runs `hypercorn app:app --bind 0.0.0.0:$PORT`

### Render
1. Sign up at [render.com](https://render.com)
2. Create new Web Service → Connect GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `hypercorn app:app --bind 0.0.0.0:$PORT`
5. Add environment variables in dashboard

### Fly.io
1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Run: `fly launch`
3. Add secrets: `fly secrets set GOOGLE_AI_KEY=your_key`
4. Deploy: `fly deploy`

---

## 📊 Streamlit Cloud (Alternative)

**Note**: Requires converting the app to Streamlit format (see `streamlit_app.py`).

### Deploy to Streamlit Cloud:
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Select `streamlit_app.py` as the main file
5. Add secrets:
   - `GOOGLE_AI_KEY`
   - `GOOGLE_PREDICT_MODEL` (optional)
   - `GOOGLE_EMBED_MODEL` (optional)

### Run locally:
```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

---

## ⚠️ Vercel (Not Recommended)

Vercel has limitations with async Python apps and long-running requests. The Quart app uses async/await extensively, which doesn't work well with Vercel's serverless model.

**Better alternatives**: Railway, Render, or Fly.io

If you must use Vercel, you'd need to:
1. Convert to serverless functions (major rewrite)
2. Handle async operations differently
3. Accept performance limitations

---

## 🔧 Environment Variables

Set these in your deployment platform:

**Required:**
- `GOOGLE_AI_KEY` - Get at https://aistudio.google.com/app/apikey

**Optional:**
- `GOOGLE_PREDICT_MODEL` - Default: `models/gemini-2.5-pro`
- `GOOGLE_EMBED_MODEL` - Default: `models/text-embedding-004`
- `HISTORICAL_CASES_PATH` - Default: `data/historical_cases.jsonl`
- `RETRIEVAL_TOP_K` - Default: `5`
- `PORT` - Usually auto-set by platform

---

## 📦 Build & Deploy Checklist

- [ ] Push code to GitHub
- [ ] Set environment variables in deployment platform
- [ ] Ensure `requirements.txt` is up to date
- [ ] Test locally: `hypercorn app:app --bind 0.0.0.0:8000`
- [ ] Deploy and test the live URL
- [ ] Verify file uploads work
- [ ] Check that predictions are generated (not fallback)

---

## 🐳 Docker (Alternative)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["hypercorn", "app:app", "--bind", "0.0.0.0:8000"]
```

Then deploy to any Docker-compatible platform (Railway, Render, Fly.io, etc.)

