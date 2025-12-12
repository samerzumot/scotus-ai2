# 🚀 Deploy to Streamlit Cloud - Quick Guide

Your code is on GitHub: **https://github.com/samerzumot/scotus-ai2**

## Step 1: Go to Streamlit Cloud

1. **Visit**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click**: "New app"

## Step 2: Configure Your App

Fill in the form:

- **Repository**: `samerzumot/scotus-ai2`
- **Branch**: `main`
- **Main file path**: `streamlit_app.py`
- **App URL** (optional): Choose a custom subdomain like `scotus-ai`

## Step 3: Add Your Google AI Key

1. **Click**: "Advanced settings" (at the bottom)
2. **Click**: "Secrets" tab
3. **Add secret**:
   - **Name**: `GOOGLE_AI_KEY`
   - **Value**: `AIzaSyBoxCtwiQqwEyEpl0ww2VnqhH7asfCQDVs`

4. **Optional secrets** (if you want to customize):
   - `GOOGLE_PREDICT_MODEL` = `models/gemini-2.5-pro` (default)
   - `GOOGLE_EMBED_MODEL` = `models/text-embedding-004` (default)
   - `HISTORICAL_CASES_PATH` = `data/historical_cases.jsonl` (default)
   - `RETRIEVAL_TOP_K` = `5` (default)

## Step 4: Deploy!

1. **Click**: "Deploy"
2. **Wait**: ~30-60 seconds for the build
3. **Done!** Your app will be live at: `https://your-app-name.streamlit.app`

---

## 🎯 Quick Commands

```bash
# Auto-start hotrun locally
./auto-start.sh

# Or manually
make hotrun

# Push to GitHub (triggers Streamlit auto-redeploy)
git add -A && git commit -m "update" && git push
```

---

## 📝 Notes

- **Streamlit auto-redeploys** when you push to GitHub
- **Secrets are secure** - only visible in Streamlit dashboard
- **Free tier** includes unlimited apps
- **Custom domain** available on paid plans

---

## 🔧 Troubleshooting

**Build fails?**
- Check that `requirements-streamlit.txt` exists
- Verify `streamlit_app.py` is in the root directory

**App shows fallback data?**
- Check that `GOOGLE_AI_KEY` secret is set correctly
- Look at Streamlit logs for errors

**Need to update?**
- Just push to GitHub - Streamlit auto-redeploys!

