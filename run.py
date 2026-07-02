#!/usr/bin/env python3
"""Start Savings Scout and keep the frontend build up to date.

    python3 run.py     (run it with the venv active so uvicorn is importable)
    python3 run.py --prod

Backend:  http://127.0.0.1:8000   (API)
Frontend: http://127.0.0.1:5173   (open this in your browser)
Prod app: http://127.0.0.1:8000   (with --prod)

On every start, run.py checks frontend sources and rebuilds frontend/dist when
it is missing or stale. The Vite dev server still proxies /api -> :8000, so in
normal dev mode use the 5173 URL. In --prod mode, run.py builds then serves the
compiled React app through FastAPI only.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"

FRONTEND_BUILD_INPUTS = [
    FRONTEND / "index.html",
    FRONTEND / "package.json",
    FRONTEND / "package-lock.json",
    FRONTEND / "vite.config.js",
    FRONTEND / "tailwind.config.js",
    FRONTEND / "postcss.config.js",
    FRONTEND / "src",
    FRONTEND / "public",
]


def find_npm():
    """Locate the npm binary.

    `shutil.which` handles a normal install. nvm often exposes npm as a lazy
    shell *function* that isn't on PATH for subprocesses, so fall back to the
    standard nvm install location and pick the newest version.
    """
    npm = shutil.which("npm")
    if npm:
        return npm
    matches = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/*/bin/npm")))
    return matches[-1] if matches else None


def newest_mtime(paths):
    newest = 0.0
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            newest = max(newest, path.stat().st_mtime)
            continue
        for child in path.rglob("*"):
            if child.is_file():
                newest = max(newest, child.stat().st_mtime)
    return newest


def dist_is_stale():
    index = DIST / "index.html"
    if not index.exists():
        return True
    return newest_mtime(FRONTEND_BUILD_INPUTS) > index.stat().st_mtime


def npm_env(npm):
    # nvm's npm needs node (same bin dir) on PATH to run — inject it so the
    # subprocess can find both even when they aren't on the inherited PATH.
    npm_bin = str(Path(npm).parent)
    return {**os.environ, "PATH": npm_bin + os.pathsep + os.environ.get("PATH", "")}


def ensure_frontend_ready(npm, env, build=True):
    node_modules = FRONTEND / "node_modules"
    package_lock = FRONTEND / "package-lock.json"
    package_json = FRONTEND / "package.json"
    install_needed = not node_modules.exists()
    if node_modules.exists() and package_lock.exists():
        install_needed = package_lock.stat().st_mtime > node_modules.stat().st_mtime
    elif node_modules.exists() and package_json.exists():
        install_needed = package_json.stat().st_mtime > node_modules.stat().st_mtime

    if install_needed:
        print("Installing/updating frontend dependencies ...")
        subprocess.run([npm, "install"], cwd=FRONTEND, check=True, env=env)

    if build:
        if dist_is_stale():
            print("Building frontend/dist ...")
            subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True, env=env)
        else:
            print("frontend/dist is up to date.")


def start_backend(reload=False):
    cmd = [sys.executable, "-m", "uvicorn", "api:app", "--port", "8000"]
    if reload:
        cmd.append("--reload")
    print("Starting FastAPI backend on :8000 ...")
    return subprocess.Popen(cmd, cwd=ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Savings Scout.")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Build frontend and serve only FastAPI at http://localhost:8000.",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip the frontend/dist freshness check and build step.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Run uvicorn with --reload.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    npm = find_npm()
    env = npm_env(npm) if npm else None
    if npm:
        ensure_frontend_ready(npm, env, build=not args.no_build)
    elif not DIST.is_dir():
        print("\n⚠️  npm not found and frontend/dist does not exist.")
        print("    Install Node (e.g. `nvm install --lts`) or run from an environment with npm.")
        print("    Starting API only.\n")
    else:
        print("\n⚠️  npm not found — using existing frontend/dist.")

    backend = start_backend(reload=args.reload)

    try:
        time.sleep(2)

        if args.prod:
            print("==> Open http://localhost:8000 in your browser <==")
            backend.wait()
            return

        if not npm:
            print("    Backend is running; press Ctrl+C to stop.")
            if DIST.is_dir():
                print("==> Open http://localhost:8000 in your browser <==")
            backend.wait()
            return

        print("Starting Vite frontend on :5173 ...")
        print("==> Open http://localhost:5173 in your browser <==")
        subprocess.run([npm, "run", "dev"], cwd=FRONTEND, env=env)
    finally:
        print("Shutting down backend ...")
        backend.terminate()
        try:
            backend.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend.kill()


if __name__ == "__main__":
    main()
