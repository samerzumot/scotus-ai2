# ⚡ Quick Deploy Instructions

## Your code is committed! Now push and deploy:

### 1️⃣ Push to GitHub

Run this command (you'll need to authenticate):

```bash
git push -u origin main
```

**If authentication fails**, use one of these:

**Option A: GitHub CLI** (if installed)
```bash
gh auth login
git push -u origin main
```

**Option B: SSH** (if you have SSH keys)
```bash
git remote set-url origin git@github.com:samerzumot/scotus-ai2.git
git push -u origin main
```

**Option C: Personal Access Token**
1. Go to: https://github.com/settings/tokens
2. Generate new token (classic) with `repo` scope
3. Copy the token
4. Run: `git push -u origin main`
5. When prompted:
   - Username: your GitHub username
   - Password: paste the token

---

### 2️⃣ Deploy to Streamlit Cloud (Easiest - 2 minutes)

1. **Go to**: https://share.streamlit.io
2. **Sign in** with GitHub
3. **Click**: "New app"
4. **Fill in**:
   - Repository: `samerzumot/scotus-ai2`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
5. **Click**: "Advanced settings"
6. **Add secret**:
   - Name: `GOOGLE_AI_KEY`
   - Value: `AIzaSyBoxCtwiQqwEyEpl0ww2VnqhH7asfCQDVs`
7. **Click**: "Deploy"

**Done!** Your app will be live in ~30 seconds at `https://your-app-name.streamlit.app`

---

### 3️⃣ Deploy to Railway (Best for Production)

1. **Go to**: https://railway.app
2. **Sign up** with GitHub
3. **Click**: "New Project"
4. **Select**: "Deploy from GitHub repo"
5. **Choose**: `samerzumot/scotus-ai2`
6. **Click**: "Add Variable"
   - Name: `GOOGLE_AI_KEY`
   - Value: `AIzaSyBoxCtwiQqwEyEpl0ww2VnqhH7asfCQDVs`
7. Railway auto-detects and deploys!

**Done!** Railway gives you a live URL automatically.

---

## 🎯 Recommended: Streamlit Cloud

**Fastest deployment** - Just connect GitHub and add the API key secret!

---

## 📝 Your Repository

**GitHub**: https://github.com/samerzumot/scotus-ai2

**Status**: ✅ Code is committed and ready to push!

