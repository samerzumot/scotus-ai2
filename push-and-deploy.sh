#!/bin/bash
# Quick script to push and deploy

set -e

echo "🚀 SCOTUS AI - Push and Deploy"
echo "================================"
echo ""

# Check if we're in a git repo
if [ ! -d .git ]; then
    echo "❌ Not a git repository"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  You have uncommitted changes. Committing them now..."
    git add -A
    git commit -m "chore: update before deployment"
fi

# Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
echo "   Repository: $(git remote get-url origin)"
echo ""
echo "   If authentication fails, use one of these:"
echo "   1. GitHub CLI: gh auth login"
echo "   2. SSH: git remote set-url origin git@github.com:samerzumot/scotus-ai2.git"
echo "   3. Personal Access Token: https://github.com/settings/tokens"
echo ""

git push -u origin main || {
    echo ""
    echo "❌ Push failed. Please authenticate and try again."
    echo ""
    echo "Quick fix options:"
    echo "  Option 1: Use GitHub CLI"
    echo "    gh auth login"
    echo "    git push -u origin main"
    echo ""
    echo "  Option 2: Use SSH"
    echo "    git remote set-url origin git@github.com:samerzumot/scotus-ai2.git"
    echo "    git push -u origin main"
    echo ""
    echo "  Option 3: Use Personal Access Token"
    echo "    Create token at: https://github.com/settings/tokens"
    echo "    Use token as password when prompted"
    echo ""
    exit 1
}

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
echo "📊 Next Steps - Deploy to:"
echo ""
echo "1. Streamlit Cloud (Easiest):"
echo "   → https://share.streamlit.io"
echo "   → New app → Connect repo → streamlit_app.py"
echo "   → Add secret: GOOGLE_AI_KEY"
echo ""
echo "2. Railway (Recommended for Quart):"
echo "   → https://railway.app"
echo "   → New project → Deploy from GitHub"
echo "   → Add variable: GOOGLE_AI_KEY"
echo ""
echo "3. Render:"
echo "   → https://render.com"
echo "   → New Web Service → Connect GitHub"
echo "   → Add environment: GOOGLE_AI_KEY"
echo ""
echo "🎉 Your code is on GitHub: https://github.com/samerzumot/scotus-ai2"
echo ""

