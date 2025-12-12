#!/bin/bash
# Auto-start script that runs hotrun in the background and monitors it

set -e

cd "$(dirname "$0")"

echo "🚀 SCOTUS AI - Auto Hot Run"
echo "============================"
echo ""

# Check if already running
if pgrep -f "hotrun.py" > /dev/null; then
    echo "⚠️  Hotrun is already running (PID: $(pgrep -f 'hotrun.py'))"
    echo "   Kill it first with: pkill -f hotrun.py"
    exit 1
fi

# Start hotrun in background
echo "📡 Starting hotrun server..."
nohup .venv/bin/python hotrun.py > hotrun.log 2>&1 &
HOTRUN_PID=$!

echo "✅ Hotrun started (PID: $HOTRUN_PID)"
echo "📝 Logs: tail -f hotrun.log"
echo "🛑 Stop: kill $HOTRUN_PID"
echo ""
echo "Server will auto-reload on file changes."
echo "Access at: http://localhost:8000"
echo ""

# Wait a moment and check if it's still running
sleep 2
if ! kill -0 $HOTRUN_PID 2>/dev/null; then
    echo "❌ Hotrun failed to start. Check hotrun.log for errors."
    exit 1
fi

echo "✅ Server is running!"
echo ""
echo "To view logs: tail -f hotrun.log"
echo "To stop: kill $HOTRUN_PID"

