# 🚀 Savings Scout — Free Deployment Plan

> Status: **planning notes** (not deployed yet). This file lives under `miscellaneous/`
> which is git-ignored, so it stays on your machine.

## TL;DR

- **Yes — once deployed, you can use it on your phone.** It's a web app, so any browser
  (phone, tablet, laptop) just opens the URL. No app-store install needed. You can even
  "Add to Home Screen" so it feels like an app.
- All free hosts are **Linux-based**. You deploy by connecting your **GitHub repo**
  (`https://github.com/Yang0427/stocks.git`) — the host pulls your code, builds it, and runs it.
- The one real gotcha: **your app writes data to files** (`purchase_log.json`,
  `sell_log.json`, `stocks.json`, `.cache/`). Most free tiers have an *ephemeral* disk that
  resets on every redeploy/restart. See [Data persistence](#-data-persistence--the-one-real-gotcha).

---

## 🏗️ How the app needs to run in production

Locally you run **two** processes (FastAPI on :8000, Vite dev server on :5173). In production
you run **one**: you build the React app to static files, and FastAPI serves them. `api.py`
already supports this — it serves `frontend/dist/` when that folder exists.

So the production sequence on any Linux host is:

```bash
# 1. Build step
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..

# 2. Run step  (host provides the $PORT env var)
python -m uvicorn api:app --host 0.0.0.0 --port $PORT
```

One process, one public URL, serving both the API (`/api/...`) and the web UI (`/`).

> ⚠️ **CORS note:** `api.py` currently allows only `localhost:5173`. In production the UI is
> served from the *same origin* as the API, so CORS isn't needed — but if you ever host the
> frontend on a different domain than the API, add that domain to `allow_origins` in `api.py`.

---

## 🆓 Free hosting options compared

| Platform | Truly free? | Persistent data (free)? | Sleeps when idle? | Setup effort |
|----------|-------------|--------------------------|-------------------|--------------|
| **Render** | ✅ Free web service | ❌ (disk is a paid add-on) | 💤 ~15 min idle → ~30 s cold start | ⭐ Easiest |
| **Fly.io** | ✅ Free allowance | ✅ Free small volume | Optional scale-to-zero | ⭐⭐ Medium (CLI + Dockerfile) |
| **Hugging Face Spaces** | ✅ Free | ✅ Persistent storage | No | ⭐⭐ Medium (Dockerfile) |
| **Railway** | ⏳ Trial credit then paid | ✅ Volumes | No | ⭐ Easy |
| **Vercel / Netlify** | ✅ | ❌ (no long-running Python server) | — | ❌ Not suitable (static/serverless only) |

**Recommendation for a personal, single-user app:**
- Want the **simplest** path and don't mind re-entering data occasionally → **Render**.
- Want **free persistent data** that survives redeploys → **Fly.io** (free volume) or
  **commit logs back to GitHub** (works on any host).

---

## 💾 Data persistence — the one real gotcha

Your "database" is just a few small JSON files. On a free ephemeral filesystem they reset
whenever the container restarts (which happens on redeploy, and on Render also after it sleeps).
Three ways to handle it:

### Option A — Ephemeral (simplest, accept data loss)
Do nothing. Deploy as-is. You can log buys/sells and they persist *until the next restart*.
Fine if you mostly run locally and the deployed copy is just for convenience/demo.
Your `stocks.json` watchlist resets to whatever is committed in the repo — usually fine, since
you keep your canonical watchlist in git anyway.

### Option B — Commit data back to GitHub (robust, host-agnostic) ⭐ best for keeping history
The app commits `purchase_log.json` / `sell_log.json` to your repo whenever they change, so
data survives any redeploy on any host.
- Needs a **GitHub Personal Access Token** (fine-grained, repo contents read/write) stored as a
  secret/env var on the host.
- Requires a small code addition: after a successful log write, commit+push (or use the GitHub
  "update file contents" REST API).
- Trade-off: every log write makes a network call to GitHub. For a once-a-month app that's nothing.

### Option C — Platform persistent volume (host-specific)
Mount a persistent disk and point the data files at it.
- **Fly.io**: free small volume — `fly volumes create data` and mount at e.g. `/data`.
- **Hugging Face Spaces**: persistent storage on the free tier.
- **Render**: disks are a **paid** add-on (not free).
- Requires making the data path configurable (e.g. read a `DATA_DIR` env var in `engine.py`
  instead of always using the repo folder).

> Recommendation: start with **A** to get it live, switch to **B** if you find you want your
> phone-logged buys to stick around.

---

## 📋 Step-by-step: Render (easiest)

1. **Push your code to GitHub** (you already have the remote `Yang0427/stocks`):
   ```bash
   git add -A && git commit -m "Prepare for deployment" && git push
   ```
   Make sure `frontend/node_modules/` and `.cache/` are git-ignored (they are).

2. **Add a `render.yaml`** to the repo root (a "blueprint" Render reads automatically):
   ```yaml
   services:
     - type: web
       name: savings-scout
       runtime: python
       plan: free
       buildCommand: |
         pip install -r requirements.txt
         cd frontend && npm install && npm run build
       startCommand: python -m uvicorn api:app --host 0.0.0.0 --port $PORT
       envVars:
         - key: PYTHON_VERSION
           value: "3.11"
   ```
   > Render's Python runtime includes Node, so `npm` is available during the build step.

3. **Create the service**: render.com → New → Blueprint → connect the GitHub repo → it reads
   `render.yaml` and deploys.

4. **Open the URL** Render gives you (e.g. `https://savings-scout.onrender.com`) — on your
   laptop *or your phone*. First load after idle takes ~30 s (free tier waking from sleep).

5. **Auto-deploy**: every `git push` to the main branch redeploys automatically.

---

## 📋 Step-by-step: Fly.io (free persistent data)

1. **Install the CLI** and log in:
   ```powershell
   # Windows (PowerShell):
   iwr https://fly.io/install.ps1 -useb | iex
   fly auth signup    # or: fly auth login
   ```

2. **Add a `Dockerfile`** to the repo root (multi-stage: build React, then run Python):
   ```dockerfile
   # ---- build frontend ----
   FROM node:20-alpine AS web
   WORKDIR /app/frontend
   COPY frontend/package*.json ./
   RUN npm install
   COPY frontend/ ./
   RUN npm run build

   # ---- python runtime ----
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt ./
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   COPY --from=web /app/frontend/dist ./frontend/dist
   ENV PORT=8080
   EXPOSE 8080
   CMD ["sh", "-c", "python -m uvicorn api:app --host 0.0.0.0 --port ${PORT}"]
   ```

3. **Launch** (creates `fly.toml`; say no to a managed DB, yes to deploy):
   ```bash
   fly launch
   ```

4. **(Optional) Free persistent volume** so your logs survive redeploys:
   ```bash
   fly volumes create data --size 1     # 1 GB, free tier
   ```
   Then add a mount in `fly.toml` and set `DATA_DIR=/data` (needs the small `engine.py` change
   from Option C). Copy the JSON files there on first boot.

5. **Deploy updates** anytime with `fly deploy`. URL looks like `https://savings-scout.fly.dev`.

---

## 📱 Using it on your phone

Once you have a public URL:
1. Open it in your phone's browser.
2. **iPhone (Safari):** Share → *Add to Home Screen*. **Android (Chrome):** ⋮ → *Add to Home screen*.
3. It now launches like an app. "Run analysis" works the same; the first run after the host has
   been idle may take ~30 s (Render) while it wakes + fetches data.

> The UI is already responsive (Tailwind), so it adapts to a phone screen.

---

## ✅ Pre-deploy checklist

- [ ] `requirements.txt` current (`fastapi`, `uvicorn[standard]`, `pydantic`, `yfinance`, `pandas`, `ta`, `curl_cffi`). ✔ done.
- [ ] `frontend/node_modules/`, `frontend/dist/`, `.cache/` git-ignored. ✔ done.
- [ ] `api.py` serves `frontend/dist` in production. ✔ implemented.
- [ ] Decide data persistence: A (ephemeral), B (commit to GitHub), or C (volume).
- [ ] Add the platform config file (`render.yaml` **or** `Dockerfile` + `fly.toml`).
- [ ] If hosting frontend separately later: add the prod domain to CORS `allow_origins` in `api.py`.
- [ ] Push to GitHub and connect the host.

---

## ⚠️ Things to know before going public

- **No auth.** Anyone with the URL can see your watchlist, holdings, and P/L, and could log fake
  trades. For a personal tool that's usually fine, but if you want privacy, add HTTP basic auth
  (a few lines in `api.py`) or pick a host that supports password protection.
- **yfinance rate limits.** Free Yahoo data is flaky/rate-limited. The 6-hour disk cache helps,
  but on an ephemeral host the cache resets on restart, so the first run re-fetches all tickers
  (~1–2 min). Fine for monthly use.
- **Cold starts (Render free tier).** ~30 s to wake after 15 min idle. Fly.io and HF Spaces
  don't sleep the same way.

---

## When you're ready

Tell me which platform + data option you picked and I'll generate the actual config files
(`render.yaml` or `Dockerfile`/`fly.toml`), make the small `engine.py` change if you want a
persistent volume or GitHub-backed data, add a "Deployment" section to the main `README.md`,
and walk you through the first deploy.
