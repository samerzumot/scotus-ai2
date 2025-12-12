# 🚀 Auto Hot Run & Streamlit Deployment

## ✅ What's Set Up

1. **Auto Hot Run**: Server automatically restarts on file changes
2. **GitHub**: Code is pushed and ready
3. **Streamlit**: Ready to deploy

---

## 🔥 Auto Hot Run

### Start Server (Auto-Reload)
```bash
./auto-start.sh
```

Or manually:
```bash
make hotrun
```

The server will:
- ✅ Auto-reload on file changes
- ✅ Watch `.py`, `.html`, `.css`, `.js` files
- ✅ Restart when `env.local` changes
- ✅ Run on http://localhost:8000

### Stop Server
```bash
pkill -f hotrun.py
```

### View Logs
```bash
tail -f hotrun.log
```

---

## 📤 Push to GitHub

```bash
# Quick push
git add -A && git commit -m "update" && git push

# Or use the script
./push-and-deploy.sh
```

---

## 🌐 Deploy to Streamlit Cloud

### Quick Steps:

1. **Go to**: https://share.streamlit.io
2. **Sign in** with GitHub
3. **New app** → Select `samerzumot/scotus-ai2`
4. **Main file**: `streamlit_app.py`
5. **Add secret**: `GOOGLE_AI_KEY` = `AIzaSyBoxCtwiQqwEyEpl0ww2VnqhH7asfCQDVs`
6. **Deploy!**

**Full guide**: See `STREAMLIT-DEPLOY.md`

---

## 🎯 Workflow

1. **Edit code** → Auto-reloads locally
2. **Test locally** → http://localhost:8000
3. **Push to GitHub** → Streamlit auto-redeploys
4. **Done!** → Live at `https://your-app.streamlit.app`

---

## 📝 Files

- `auto-start.sh` - Auto-start hotrun script
- `hotrun.py` - Hot reload server
- `streamlit_app.py` - Streamlit version
- `STREAMLIT-DEPLOY.md` - Deployment guide

---

## ⚡ Quick Commands

```bash
# Start auto hot run
./auto-start.sh

# Push to GitHub
git push

# Check server status
curl http://localhost:8000/health

# View logs
tail -f hotrun.log
```

