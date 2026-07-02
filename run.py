#!/usr/bin/env python3
"""Start Savings Scout (FastAPI backend + Vite frontend) for development.

    python3 run.py

Backend:  http://127.0.0.1:8000   (API)
Frontend: http://127.0.0.1:5173   (open this in your browser)

The Vite dev server proxies /api -> :8000, so just use the 5173 URL.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def main():
    print("Starting FastAPI backend on :8000 ...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--port", "8000"],
        cwd=ROOT,
    )

    try:
        time.sleep(2)

        if not (FRONTEND / "node_modules").exists():
            print("Installing frontend dependencies (first run) ...")
            subprocess.run(["npm", "install"], cwd=FRONTEND, check=True)

        print("Starting Vite frontend on :5173 ...")
        print("==> Open http://localhost:5173 in your browser <==")
        subprocess.run(["npm", "run", "dev"], cwd=FRONTEND)
    finally:
        print("Shutting down backend ...")
        backend.terminate()
        try:
            backend.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend.kill()


if __name__ == "__main__":
    main()
