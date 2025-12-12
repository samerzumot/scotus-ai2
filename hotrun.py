#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable


WATCH_DIRS = ["utils", "templates", "static"]
WATCH_FILES = ["app.py", ".env", "env.local"]
WATCH_EXTS = {".py", ".html", ".css", ".js"}
TOKEN_FILE = "hotreload.token"


def iter_watch_paths(root: Path) -> Iterable[Path]:
    for f in WATCH_FILES:
        yield root / f
    for d in WATCH_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in WATCH_EXTS:
                yield p


def snapshot(paths: Iterable[Path]) -> Dict[str, float]:
    snap: Dict[str, float] = {}
    for p in paths:
        key = str(p)
        try:
            snap[key] = p.stat().st_mtime
        except FileNotFoundError:
            snap[key] = -1.0
    return snap


def has_changed(prev: Dict[str, float], curr: Dict[str, float]) -> bool:
    if prev.keys() != curr.keys():
        return True
    for k, v in curr.items():
        if prev.get(k) != v:
            return True
    return False


def changed_paths(prev: Dict[str, float], curr: Dict[str, float]) -> Iterable[str]:
    keys = set(prev.keys()) | set(curr.keys())
    for k in sorted(keys):
        if prev.get(k) != curr.get(k):
            yield k


def touch_token(root: Path) -> None:
    try:
        (root / TOKEN_FILE).write_text(str(int(time.time() * 1000)), encoding="utf-8")
    except Exception:
        pass


def start_server(*, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("APP_ENV", "development")
    env.setdefault("PORT", str(port))
    cmd = [sys.executable, "-m", "hypercorn", "app:app", "--bind", f"0.0.0.0:{port}"]
    return subprocess.Popen(cmd, env=env)


def terminate(proc: subprocess.Popen, *, timeout_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        return
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        proc.kill()
    except Exception:
        pass


def main() -> int:
    root = Path(__file__).resolve().parent
    port = int(os.getenv("PORT") or "8000")
    poll_s = float(os.getenv("HOTRUN_POLL_S") or "0.8")

    print(f"SCOTUS AI hotrun: starting dev server on :{port} (poll {poll_s:.1f}s). Ctrl-C to stop.")
    touch_token(root)
    proc = start_server(port=port)
    prev = snapshot(iter_watch_paths(root))

    try:
        while True:
            time.sleep(poll_s)

            # If the server died, restart it.
            if proc.poll() is not None:
                print("SCOTUS AI hotrun: server exited; restarting…")
                proc = start_server(port=port)
                prev = snapshot(iter_watch_paths(root))
                continue

            curr = snapshot(iter_watch_paths(root))
            if has_changed(prev, curr):
                files = list(changed_paths(prev, curr))
                touch_token(root)

                needs_restart = any(Path(f).suffix.lower() == ".py" for f in files) or any(
                    Path(f).name in {".env", "env.local"} for f in files
                )
                if needs_restart:
                    print("SCOTUS AI hotrun: backend/env change detected; restarting…")
                    terminate(proc)
                    proc = start_server(port=port)
                else:
                    print("SCOTUS AI hotrun: asset/template change detected; triggering browser reload…")

                prev = curr
    except KeyboardInterrupt:
        print("SCOTUS AI hotrun: stopping…")
        return 0
    finally:
        terminate(proc)


if __name__ == "__main__":
    raise SystemExit(main())


