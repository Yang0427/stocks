# 🎯 Savings Scout

A dividend-accumulation stock picker for **Bursa Malaysia (.KL)**, **US**, and **Hong Kong (.HK)**.
Tell it your monthly budget; it tells you **exactly what to buy this month** — one stock, the exact
number of lots that fits your money, and a one-line reason. No more agonising before each purchase.

It is built for the "DCA my salary every payday" strategy: long-term accumulation of quality
dividend payers, bought when they're on sale.

## What's in the box

| File | Role |
|------|------|
| `engine.py` | **The brain.** All scoring, data fetching, P/L reconciliation, budget-fit and caching. Single source of truth. |
| `main.py` | CLI front-end. Prints the full monthly report to the terminal. Imports `engine.py`. |
| `api.py` | FastAPI backend. Exposes `engine.py` over HTTP and serves the web app. |
| `run.py` | Starts the backend and frontend dev servers together (`python3 run.py`). |
| `frontend/` | Vite + React + Tailwind web app (the appealing interface). |
| `stocks.json` | Your watchlist (ticker + sector). |
| `purchase_log.json` / `sell_log.json` | Your transaction history (drives sector-rotation, freshness, P/L). |

The CLI and the web app share **the same scoring engine** — there is no duplicated logic.

## The scoring engine (≈117 pts)

Each stock is scored on: trend (golden/death cross), pullback entry depth, sector-aware PE valuation,
dividend yield (capped for cyclicals), ROE, analyst target upside, EPS growth, historical-yield
mean-reversion, dividend track record, sector rotation vs your recent buys, freshness, a recent-buy
penalty (so it diversifies), and an **earnings-health penalty** that detects deteriorating
fundamentals (multi-quarter declines, payout-ratio stress, downward EPS revisions). It also runs
short-term **timing signals** (ex-dividend countdown, upcoming earnings, momentum, RSI) to decide
whether *now* is a clean entry.

**Budget-fit** then takes the ranked list and your budget and finds the highest-ranked stock whose
position actually fits — e.g. if the #1 pick costs RM2,098/lot and your budget is RM1,000, it
automatically drops to the best pick you can actually afford.

## Run the web app (recommended)

Prerequisites: **Python 3.11+** and **Node 18+**.

```bash
# 1. install python deps (once)
pip3 install -r requirements.txt

# 2. start everything
python3 run.py
```

Then open **http://localhost:5173**.

1. Enter your **monthly budget** (e.g. RM 1000).
2. Click **Run analysis** (first run fetches live data; later runs use the 6-hour disk cache and are instant).
3. Read **This month's buy** — the headline card tells you the stock, lots, total cost, and why.
4. Place your limit order in MooMoo. When it **fills**, click **Log this buy** and confirm the real price/lots.
5. Use **Log sell** when you sell. The Holdings tab shows live P/L and projected dividend income.

> Use the **↻ Refresh data** button to force-refresh market prices (ignores the cache), then re-run.

### Manual start (two terminals)

```bash
# terminal 1 — backend
python3 -m uvicorn api:app --reload --port 8000

# terminal 2 — frontend
cd frontend
npm install   # first time only
npm run dev
```

### Production build (single process)

```bash
cd frontend
npm run build          # outputs frontend/dist
cd ..
python3 -m uvicorn api:app --port 8000
```

FastAPI then serves the built React app at **http://localhost:8000** directly.

## Run the CLI (terminal report)

```bash
python3 main.py
```

Prints the full ranked report, holdings review, dividend income, and the final "best buy" verdict.

## Configure your watchlist — `stocks.json`

```json
{
  "period": "2y",
  "tickers": [
    { "ticker": "1155.KL", "sector": "bank" },
    { "ticker": "5176.KL", "sector": "reit" },
    { "ticker": "NVDA",    "sector": "tech" }
  ]
}
```

Valid sectors: `bank`, `reit`, `telco`, `utilities`, `energy`, `general`, `consumer`,
`tech`, `healthcare`, `consumer_staples`. Unknown tickers fall back to `general` (sector
is also auto-inferred from yfinance when omitted).

## Transaction logs

`purchase_log.json` and `sell_log.json` are arrays of:

```json
{ "month": "2026-06", "ticker": "5176.KL", "price": 2.30, "currency": "RM", "lots": 2 }
```

You don't need to hand-edit these — use the **Log buy / Log sell** forms in the app. Because MooMoo
limit orders fill later, only log a trade once it has actually executed at a known price.

## API reference

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/config`      | Watchlist + cache freshness |
| POST | `/api/analyze`     | Start an analysis job → `{job_id}` |
| GET  | `/api/job/{id}`    | Poll job progress/result |
| POST | `/api/budget`      | `{budget, only_actionable, prefer}` → the buy that fits |
| POST | `/api/refresh`     | Force re-fetch market data to cache |
| GET  | `/api/holdings`    | Open positions + dividend income |
| POST | `/api/log/buy`     | Append a buy to `purchase_log.json` |
| POST | `/api/log/sell`    | Append a sell to `sell_log.json` |

---

*Not financial advice. This is a personal decision-support tool — always do your own diligence.*
