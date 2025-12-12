# 🚀 Deploy Now - Step by Step

Your code is committed and ready to push! Follow these steps:

## Step 1: Push to GitHub

```bash
# If you haven't authenticated, use one of these:

# Option A: Use GitHub CLI (if installed)
gh auth login
git push -u origin main

# Option B: Use SSH (if you have SSH keys set up)
git remote set-url origin git@github.com:samerzumot/scotus-ai2.git
git push -u origin main

# Option C: Use Personal Access Token
# Go to GitHub.com → Settings → Developer settings → Personal access tokens
# Create token with 'repo' scope
# Then:
git push -u origin main
# When prompted, use your GitHub username and the token as password
```

## Step 2: Deploy to Streamlit Cloud (Easiest)

1. **Go to**: https://share.streamlit.io
2. **Sign in** with GitHub
3. **Click**: "New app"
4. **Select**:
   - Repository: `samerzumot/scotus-ai2`
   - Branch: `main`
   - Main file: `streamlit_app.py`
5. **Add Secret**:
   - Name: `GOOGLE_AI_KEY`
   - Value: `AIzaSyBoxCtwiQqwEyEpl0ww2VnqhH7asfCQDVs` (your key)
6. **Click**: "Deploy"

**Done!** Your app will be live at `https://your-app-name.streamlit.app`

---

## Step 3: Deploy to Railway (Best for Quart App)

1. **Go to**: https://railway.app
2. **Sign up** with GitHub
3. **New Project** → Deploy from GitHub repo
4. **Select**: `samerzumot/scotus-ai2`
5. **Add Variable**:
   - Name: `GOOGLE_AI_KEY`
   - Value: `AIzaSyBoxCtwiQqwEyEpl0ww2VnqhH7asfCQDVs`
6. Railway auto-detects and deploys!

**Done!** Railway gives you a live URL automatically.

---

## Step 4: Deploy to Render

1. **Go to**: https://render.com
2. **Sign up** with GitHub
3. **New** → Web Service
4. **Connect**: `samerzumot/scotus-ai2`
5. **Settings**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `hypercorn app:app --bind 0.0.0.0:$PORT`
6. **Environment**:
   - Add `GOOGLE_AI_KEY` = `AIzaSyBoxCtwiQqwEyEpl0ww2VnqhH7asfCQDVs`
7. **Deploy**

---

## Quick Commands

```bash
# Push to GitHub (run this first)
git push -u origin main

# Test Streamlit locally
pip install streamlit
streamlit run streamlit_app.py

# Test Quart locally
hypercorn app:app --bind 0.0.0.0:8000
```

---

## ⚠️ Important Notes

1. **Your API key is in env.local** - Make sure it's NOT committed (it's in .gitignore ✅)
2. **Set the API key as a secret/environment variable** in your deployment platform
3. **Don't commit env.local** - It's already ignored

---

## Which Platform to Choose?

- **Streamlit Cloud**: Easiest, free, great for demos
- **Railway**: Best for production, auto-detects everything
- **Render**: Good free tier, reliable
- **Vercel**: Not recommended (async Python limitations)

**Recommendation**: Start with **Streamlit Cloud** for quickest deployment!

