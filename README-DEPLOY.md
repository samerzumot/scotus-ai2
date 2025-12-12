# Quick Deployment Guide

## 🚀 Streamlit Cloud (Easiest)

1. **Install Streamlit locally** (optional, for testing):
   ```bash
   pip install streamlit
   streamlit run streamlit_app.py
   ```

2. **Deploy to Streamlit Cloud**:
   - Push code to GitHub
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect GitHub repo
   - Main file: `streamlit_app.py`
   - Add secrets:
     - `GOOGLE_AI_KEY` = your API key
   - Deploy!

**That's it!** Your app will be live at `https://your-app.streamlit.app`

---

## 🚂 Railway (Best for Quart App)

1. Sign up at [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select this repo
4. Add environment variable: `GOOGLE_AI_KEY`
5. Railway auto-detects and deploys!

**Railway auto-detects** the Quart app and runs it correctly.

---

## 🎨 Render

1. Sign up at [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repo
4. Settings:
   - Build: `pip install -r requirements.txt`
   - Start: `hypercorn app:app --bind 0.0.0.0:$PORT`
5. Add `GOOGLE_AI_KEY` in Environment

---

## ⚠️ Vercel (Not Recommended)

Vercel doesn't work well with async Python apps. Use Railway or Render instead.

---

## 📝 Environment Variables

Set these in your deployment platform:

- `GOOGLE_AI_KEY` (required)
- `GOOGLE_PREDICT_MODEL` (optional, default: `models/gemini-2.5-pro`)
- `GOOGLE_EMBED_MODEL` (optional, default: `models/text-embedding-004`)

---

## 🧪 Test Locally First

```bash
# For Quart app
hypercorn app:app --bind 0.0.0.0:8000

# For Streamlit app
streamlit run streamlit_app.py
```

