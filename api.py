"""
api.py — FastAPI backend for Savings Scout.

Exposes engine.py over HTTP and serves the built React frontend. Run:

    uvicorn api:app --reload --port 8000

Endpoints
    GET  /api/health
    GET  /api/config            -> watchlist + cache freshness
    POST /api/analyze           -> start analysis job (background), returns job id
    GET  /api/job/{id}          -> poll job status/progress + result
    POST /api/budget            -> budget-fit pick from the latest analysis
    POST /api/refresh           -> start cache-refresh job (background)
    GET  /api/holdings          -> open positions + dividend income (uses cache)
    POST /api/log/buy           -> append a structured buy to purchase_log.json
    POST /api/log/sell          -> append a structured sell to sell_log.json
    GET  /api/log/buy           -> read purchase_log.json
    GET  /api/log/sell          -> read sell_log.json
"""

import os
import threading
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

import engine

app = FastAPI(title="Savings Scout API", version="1.0")

# Allow the Vite dev server (5173) to call us during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory job store (single-user app, so a dict + lock is plenty)
# ─────────────────────────────────────────────────────────────────────────────
_jobs = {}
_jobs_lock = threading.Lock()
_last_analysis = {"data": None}   # cache the most recent successful analysis


def _new_job(kind):
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id, "kind": kind, "status": "running",
            "done": 0, "total": 0, "message": "Starting...",
            "result": None, "error": None,
            "started_at": datetime.utcnow().isoformat(),
        }
    return job_id


def _update_job(job_id, **fields):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _run_analysis_job(job_id, use_cache, ttl_hours):
    def progress(done, total, msg):
        _update_job(job_id, done=done, total=total, message=msg)
    try:
        data = engine.analyze_all(use_cache=use_cache, ttl_hours=ttl_hours, progress=progress)
        _last_analysis["data"] = data
        _update_job(job_id, status="done", result=data, message="Analysis complete")
    except Exception as e:
        _update_job(job_id, status="error", error=str(e), message=f"Failed: {e}")


def _run_refresh_job(job_id):
    def progress(done, total, msg):
        _update_job(job_id, done=done, total=total, message=msg)
    try:
        summary = engine.refresh_cache(progress=progress)
        _update_job(job_id, status="done", result=summary, message="Cache refreshed")
    except Exception as e:
        _update_job(job_id, status="error", error=str(e), message=f"Failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    use_cache: bool = True
    ttl_hours: float = engine.DEFAULT_CACHE_TTL_HOURS


class BudgetRequest(BaseModel):
    budget: float = Field(gt=0)
    only_actionable: bool = True
    prefer: Optional[str] = None


class LogEntry(BaseModel):
    month: str                      # "YYYY-MM"
    ticker: str
    price: float = Field(gt=0)
    currency: str
    lots: int = Field(gt=0)


class TickerEntry(BaseModel):
    ticker: str
    sector: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


@app.get("/api/config")
def get_config():
    cfg = engine.load_config() or {"tickers": [], "period": "2y"}
    return {
        "period": cfg.get("period", "2y"),
        "tickers": cfg.get("tickers", []),
        "cache": engine.cache_status(cfg),
        "known_sectors": engine.KNOWN_SECTORS,
    }


@app.get("/api/watchlist")
def get_watchlist():
    cfg = engine.load_config() or {"tickers": [], "period": "2y"}
    return {"tickers": cfg.get("tickers", []), "known_sectors": engine.KNOWN_SECTORS}


@app.post("/api/watchlist")
def add_watchlist_ticker(entry: TickerEntry):
    try:
        tickers = engine.add_ticker(entry.ticker, entry.sector)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "tickers": tickers}


@app.delete("/api/watchlist/{ticker}")
def delete_watchlist_ticker(ticker: str):
    try:
        tickers = engine.remove_ticker(ticker)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True, "tickers": tickers}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    job_id = _new_job("analyze")
    threading.Thread(
        target=_run_analysis_job, args=(job_id, req.use_cache, req.ttl_hours), daemon=True
    ).start()
    return {"job_id": job_id}


@app.post("/api/refresh")
def refresh():
    job_id = _new_job("refresh")
    threading.Thread(target=_run_refresh_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return dict(job)


@app.post("/api/budget")
def budget(req: BudgetRequest):
    data = _last_analysis["data"]
    if not data:
        raise HTTPException(400, "No analysis available yet — run /api/analyze first")
    pick = engine.budget_fit(
        data["results"], req.budget,
        only_actionable=req.only_actionable, prefer=req.prefer,
    )
    if not pick:
        return {"pick": None, "message": "Nothing in the watchlist fits this budget."}
    return {"pick": pick}


@app.get("/api/holdings")
def holdings():
    cfg = engine.load_config() or {"tickers": [], "period": "2y"}
    purchase_log = engine.load_purchase_log()
    sell_log = engine.load_sell_log()
    tsm = engine.build_ticker_sector_map(cfg.get("tickers", []))
    positions, realized, errors = engine.build_open_positions(purchase_log, sell_log, tsm)
    period = cfg.get("period", "2y")

    # Use the most recent analysis as a price source when available (fast),
    # else read fresh from cache.
    by_ticker = {}
    if _last_analysis["data"]:
        by_ticker = {r["ticker"]: r for r in _last_analysis["data"]["results"]}

    enriched = []
    total_annual_div = 0.0
    portfolio_currency = "RM"

    # Portfolio-wide running totals
    total_cost = 0.0           # what you paid for the still-open units
    total_value = 0.0          # what those units are worth now
    total_unrealized = 0.0     # value - cost
    total_realized = 0.0       # locked-in P/L from sells
    best = None                # biggest % gainer
    worst = None               # biggest % loser

    for pos in positions:
        ticker = pos["ticker"]
        buy_price = pos["avg_buy_price"]
        open_units = pos["open_units"]
        portfolio_currency = pos["currency"]

        if ticker in by_ticker:
            r = by_ticker[ticker]
            current_price = r["price"]
            week52_high = r["week52_high"]
            div_yield = r["div_yield"]
            name = r["name"]
        else:
            name, div_yield, *_ = engine.get_stock_info(ticker)
            df = engine.fetch_data(ticker, period)
            stats = engine.analyze_stock(df) if not df.empty else None
            if not stats:
                continue
            current_price = stats["price"]
            week52_high = stats["week52_high"]

        pnl_pct = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else None
        cost = buy_price * open_units
        value = current_price * open_units
        unrealized = value - cost
        realized = pos["realized_pnl"]["value"]
        annual_div, monthly_div = engine.project_annual_dividend(open_units, div_yield, current_price)
        verdict, reason = engine.evaluate_accumulation_signal(current_price, week52_high, div_yield)

        total_cost += cost
        total_value += value
        total_unrealized += unrealized
        total_realized += realized
        total_annual_div += annual_div

        if pnl_pct is not None:
            if best is None or pnl_pct > best["pnl_pct"]:
                best = {"ticker": ticker, "pnl_pct": round(pnl_pct, 2)}
            if worst is None or pnl_pct < worst["pnl_pct"]:
                worst = {"ticker": ticker, "pnl_pct": round(pnl_pct, 2)}

        enriched.append({
            "ticker": ticker, "name": name, "sector": pos.get("sector", ""),
            "currency": pos["currency"], "open_lots": pos["open_lots"], "open_units": open_units,
            "avg_buy_price": round(buy_price, 4), "current_price": round(current_price, 4),
            "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
            "cost": round(cost, 2), "value": round(value, 2),
            "unrealized": round(unrealized, 2),
            "realized": round(realized, 2),
            "div_yield": round(div_yield, 2),
            "annual_div": round(annual_div, 2), "monthly_div": round(monthly_div, 2),
            "verdict": verdict, "reason": reason, "since": pos["first_month"],
        })

    # Total return blends unrealized + realized against money put in.
    total_pl = total_unrealized + total_realized
    total_return_pct = (total_unrealized / total_cost * 100) if total_cost > 0 else None
    winners = sum(1 for p in enriched if p["pnl_pct"] is not None and p["pnl_pct"] > 0)
    losers = sum(1 for p in enriched if p["pnl_pct"] is not None and p["pnl_pct"] < 0)

    return {
        "positions": enriched,
        "errors": errors,
        "total_annual_div": round(total_annual_div, 2),
        "monthly_avg_div": round(total_annual_div / 12, 2),
        "currency": portfolio_currency,
        "summary": {
            "total_cost": round(total_cost, 2),
            "total_value": round(total_value, 2),
            "total_unrealized": round(total_unrealized, 2),
            "total_realized": round(total_realized, 2),
            "total_pl": round(total_pl, 2),
            "total_return_pct": round(total_return_pct, 2) if total_return_pct is not None else None,
            "winners": winners,
            "losers": losers,
            "positions": len(enriched),
            "best": best,
            "worst": worst,
        },
    }


@app.get("/api/log/buy")
def get_buy_log():
    return engine.load_purchase_log()


@app.get("/api/log/sell")
def get_sell_log():
    return engine.load_sell_log()


@app.post("/api/log/buy")
def log_buy(entry: LogEntry):
    try:
        data = engine.append_log_entry("purchase_log.json", entry.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "log": data}


@app.post("/api/log/sell")
def log_sell(entry: LogEntry):
    try:
        data = engine.append_log_entry("sell_log.json", entry.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "log": data}


# ─────────────────────────────────────────────────────────────────────────────
# Serve the built React app (production). In dev you run Vite separately.
# ─────────────────────────────────────────────────────────────────────────────
_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_DIST, "assets")), name="assets")

    @app.get("/")
    def _index():
        return FileResponse(os.path.join(_DIST, "index.html"))

    @app.get("/{full_path:path}")
    def _spa(full_path: str):
        # SPA fallback — let React Router handle client routes.
        target = os.path.join(_DIST, full_path)
        if os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(_DIST, "index.html"))
