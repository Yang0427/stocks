"""
engine.py — Savings Scout scoring engine (single source of truth).

This module holds ALL the data-fetching, scoring, P/L and budget logic.
Both the CLI (main.py) and the web API (api.py) import from here so there is
exactly one brain. No scoring logic should live anywhere else.

Public entry points:
    analyze_all(config, purchase_log, sell_log, ...) -> AnalysisResult (dataclass)
    budget_fit(results, budget, currency)            -> BudgetPick | None
    build_open_positions(...)                         -> holdings + realized P/L
    refresh_cache(tickers, ...)                       -> force re-fetch to disk
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf
import pandas as pd
import ta

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR-AWARE PE THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_PE_THRESHOLDS = {
    'bank':      {'great': 10, 'ok': 14},
    'exchange':  {'great': 18, 'ok': 25},
    'reit':      {'great': 14, 'ok': 18},
    'telco':     {'great': 18, 'ok': 25},
    'utilities': {'great': 18, 'ok': 25},
    'energy':    {'great': 15, 'ok': 22},
    'general':   {'great': 15, 'ok': 20},
    'consumer':  {'great': 18, 'ok': 25},
    'materials':        {'great': 18, 'ok': 26},
    'chemicals':        {'great': 16, 'ok': 24},
    'property':         {'great': 12, 'ok': 18},
    'industrial':       {'great': 16, 'ok': 24},
    'tech':             {'great': 20, 'ok': 30},
    'healthcare':       {'great': 22, 'ok': 32},
    'consumer_staples': {'great': 22, 'ok': 30},
}

# Cyclical sectors where trailing yield is unreliable — yield score is capped
# and "falling knife" pullback detection applies.
CYCLICAL_SECTORS = {'consumer', 'energy', 'general', 'tech', 'materials', 'chemicals', 'property'}

# Max theoretical raw score, used to normalise the score bar in any UI.
MAX_SCORE = 117
MAX_LONG_TERM_SCORE = 100

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')
DEFAULT_CACHE_TTL_HOURS = 6


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & LOG IO
# ─────────────────────────────────────────────────────────────────────────────
def _here(filename):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def load_config(path='stocks.json'):
    try:
        with open(_here(path), 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# Sectors the score engine understands; the UI offers these in a dropdown.
KNOWN_SECTORS = sorted(SECTOR_PE_THRESHOLDS.keys())


def _save_config(config, path='stocks.json'):
    with open(_here(path), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def add_ticker(ticker, sector, path='stocks.json'):
    """Add one ticker to the watchlist. Returns the updated tickers list.

    Validates the sector against KNOWN_SECTORS and rejects duplicates so the
    config never ends up with a sector the score engine can't band correctly.
    """
    ticker = (ticker or '').strip().upper()
    sector = (sector or '').strip().lower()
    if not ticker:
        raise ValueError("Ticker is required")
    if sector not in SECTOR_PE_THRESHOLDS:
        raise ValueError(f"Unknown sector '{sector}'. Valid: {', '.join(KNOWN_SECTORS)}")

    config = load_config(path) or {'period': '2y', 'tickers': []}
    tickers = config.setdefault('tickers', [])
    if any((t.get('ticker') or '').strip().upper() == ticker for t in tickers):
        raise ValueError(f"{ticker} is already in the watchlist")

    tickers.append({'ticker': ticker, 'sector': sector})
    _save_config(config, path)
    return tickers


def remove_ticker(ticker, path='stocks.json'):
    """Remove a ticker from the watchlist. Returns the updated tickers list."""
    ticker = (ticker or '').strip().upper()
    config = load_config(path) or {'period': '2y', 'tickers': []}
    tickers = config.get('tickers', [])
    new_tickers = [t for t in tickers if (t.get('ticker') or '').strip().upper() != ticker]
    if len(new_tickers) == len(tickers):
        raise ValueError(f"{ticker} not found in the watchlist")
    config['tickers'] = new_tickers
    _save_config(config, path)
    return new_tickers


def _load_json_list(path):
    """Load a JSON-array log file, always returning a list.

    Recovers gracefully from common hand-edit mistakes: a missing file, an empty
    file, or a file that holds something other than an array (e.g. an empty
    object `{}`). Anything that isn't a list is treated as an empty log so a
    malformed file can never crash a save or analysis.
    """
    try:
        with open(_here(path), 'r', encoding='utf-8-sig') as f:
            text = f.read().strip()
        if not text:
            return []
        data = json.loads(text)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def load_purchase_log():
    return _load_json_list('purchase_log.json')


def load_sell_log():
    return _load_json_list('sell_log.json')


def append_log_entry(path, entry):
    """Append one transaction dict to a JSON-array log file, creating it if needed.

    Returns the full updated list. Validates required fields before writing so
    the engine never corrupts a log with a half-formed entry.
    """
    required = ('month', 'ticker', 'price', 'currency', 'lots')
    missing = [k for k in required if entry.get(k) in (None, '')]
    if missing:
        raise ValueError(f"Log entry missing required fields: {missing}")

    try:
        entry = {
            'month':    str(entry['month']),
            'ticker':   str(entry['ticker']).strip().upper(),
            'price':    round(float(entry['price']), 4),
            'currency': str(entry['currency']).strip(),
            'lots':     int(entry['lots']),
        }
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid types in log entry: {e}")

    if entry['price'] <= 0 or entry['lots'] <= 0:
        raise ValueError("price and lots must be positive")

    data = _load_json_list(path)
    data.append(entry)
    with open(_here(path), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=3, ensure_ascii=False)
    return data


def build_ticker_sector_map(tickers_cfg):
    mapping = {}
    for entry in tickers_cfg:
        ticker = entry.get('ticker')
        sector = entry.get('sector')
        if ticker and sector:
            mapping[ticker] = sector
    return mapping


# yfinance returns sector strings like "Financial Services", "Real Estate", etc.
_YFINANCE_SECTOR_MAP = {
    'financial services': 'bank',
    'financials':         'bank',
    'banking':            'bank',
    'capital markets':    'exchange',
    'exchange':           'exchange',
    'real estate':        'reit',
    'communication services': 'telco',
    'telecommunications': 'telco',
    'utilities':          'utilities',
    'energy':             'energy',
    'basic materials':    'materials',
    'materials':          'materials',
    'chemicals':          'chemicals',
    'property':           'property',
    'industrials':        'industrial',
}


def infer_sector_from_yfinance(ticker):
    """Fetch sector from yfinance, map to internal name, fall back to 'general'."""
    try:
        info = yf.Ticker(ticker).info
        raw = (info.get('sector') or info.get('industry') or '').lower()
        for key, internal in _YFINANCE_SECTOR_MAP.items():
            if key in raw:
                return internal
    except Exception:
        pass
    return 'general'


def get_recent_sectors(log, ticker_sector_map, n=3):
    """Return the sectors of the last n purchases (most recent first)."""
    if not log:
        return []
    sorted_log = sorted(log, key=lambda e: e.get('month', ''), reverse=True)
    sectors = []
    for entry in sorted_log[:n]:
        ticker = entry.get('ticker')
        sector = entry.get('sector') or ticker_sector_map.get(ticker)
        if not sector:
            sector = infer_sector_from_yfinance(ticker)
        if sector:
            sectors.append(sector)
    return sectors


def get_last_sector(log, ticker_sector_map):
    if not log:
        return None
    latest = max(log, key=lambda e: e.get('month', ''))
    ticker = latest.get('ticker')
    if latest.get('sector'):
        return latest.get('sector')
    if ticker in ticker_sector_map:
        return ticker_sector_map[ticker]
    return infer_sector_from_yfinance(ticker)


def get_months_since_last_buy(log, ticker):
    """Return how many months ago this ticker was last purchased, or None."""
    entries = [e for e in log if e.get('ticker') == ticker and e.get('month')]
    if not entries:
        return None
    last_month_str = max(e['month'] for e in entries)
    try:
        last_dt = datetime.strptime(last_month_str, "%Y-%m")
        now = datetime.now()
        return (now.year - last_dt.year) * 12 + (now.month - last_dt.month)
    except ValueError:
        return None


def units_per_lot(ticker):
    return 100 if ticker.endswith('.KL') else 1


# ─────────────────────────────────────────────────────────────────────────────
# OPEN POSITIONS (FIFO P/L reconciliation)
# ─────────────────────────────────────────────────────────────────────────────
def build_open_positions(purchase_log, sell_log, ticker_sector_map):
    buy_queues = {}
    realized_by_ticker = {}
    errors = []

    for buy in purchase_log:
        ticker = buy.get('ticker')
        price = buy.get('price')
        lots = buy.get('lots')
        currency = buy.get('currency')
        month = buy.get('month', 'N/A')
        sector = buy.get('sector')

        if not ticker or price is None or lots is None or not currency:
            errors.append(f"Buy entry missing required fields: {buy}")
            continue

        if not sector:
            sector = ticker_sector_map.get(ticker)
            if not sector:
                sector = infer_sector_from_yfinance(ticker)

        try:
            price = float(price)
            lots = int(lots)
        except (TypeError, ValueError):
            errors.append(f"{ticker}: Invalid buy entry types for price/lots: {buy}")
            continue

        if price <= 0 or lots <= 0:
            errors.append(f"{ticker}: Buy entry must have positive price/lots: {buy}")
            continue

        buy_queues.setdefault(ticker, []).append({
            'lots_remaining': lots,
            'buy_price': price,
            'currency': currency,
            'month': month,
            'sector': sector,
        })

    for sell in sell_log:
        ticker = sell.get('ticker')
        price = sell.get('price')
        lots = sell.get('lots')
        currency = sell.get('currency')

        if not ticker or price is None or lots is None or not currency:
            errors.append(f"Sell entry missing required fields: {sell}")
            continue

        try:
            price = float(price)
            lots = int(lots)
        except (TypeError, ValueError):
            errors.append(f"{ticker}: Invalid sell entry types for price/lots: {sell}")
            continue

        if price <= 0 or lots <= 0:
            errors.append(f"{ticker}: Sell entry must have positive price/lots: {sell}")
            continue

        if ticker not in buy_queues or not buy_queues[ticker]:
            errors.append(
                f"{ticker}: Sell of {lots} lot(s) @ {currency} {price:.3f} has no matching buy record "
                f"— add the original buy to purchase_log.json if you want P/L tracking"
            )
            continue

        if buy_queues[ticker][0]['currency'] != currency:
            errors.append(f"{ticker}: Currency mismatch between buy log and sell log")
            continue

        lots_to_match = lots
        realized = realized_by_ticker.setdefault(ticker, {'value': 0.0, 'currency': currency})
        unit_size = units_per_lot(ticker)

        while lots_to_match > 0 and buy_queues[ticker]:
            oldest_buy = buy_queues[ticker][0]
            matched_lots = min(lots_to_match, oldest_buy['lots_remaining'])
            pnl_per_unit = price - oldest_buy['buy_price']
            realized['value'] += pnl_per_unit * matched_lots * unit_size

            oldest_buy['lots_remaining'] -= matched_lots
            lots_to_match -= matched_lots

            if oldest_buy['lots_remaining'] == 0:
                buy_queues[ticker].pop(0)

        if lots_to_match > 0:
            errors.append(f"{ticker}: Sell lots exceed available open lots by {lots_to_match}")

    open_positions = []
    for ticker, queue in buy_queues.items():
        remaining = [b for b in queue if b['lots_remaining'] > 0]
        if not remaining:
            continue

        currencies = {b['currency'] for b in remaining}
        if len(currencies) != 1:
            errors.append(f"{ticker}: Open position has mixed currencies; cannot evaluate")
            continue

        currency = remaining[0]['currency']
        unit_size = units_per_lot(ticker)
        total_lots = sum(b['lots_remaining'] for b in remaining)
        total_units = total_lots * unit_size
        total_cost = sum(b['buy_price'] * b['lots_remaining'] * unit_size for b in remaining)
        avg_buy = total_cost / total_units if total_units > 0 else 0.0
        first_month = remaining[0].get('month', 'N/A')
        sector = remaining[0].get('sector')

        open_positions.append({
            'ticker': ticker,
            'currency': currency,
            'sector': sector,
            'open_lots': total_lots,
            'open_units': total_units,
            'avg_buy_price': avg_buy,
            'first_month': first_month,
            'realized_pnl': realized_by_ticker.get(ticker, {'value': 0.0, 'currency': currency}),
        })

    return open_positions, realized_by_ticker, errors


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING (+ disk cache)
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(ticker, period):
    safe = ticker.replace('/', '_').replace('\\', '_')
    return os.path.join(CACHE_DIR, f"{safe}_{period}.json")


def _cache_age_hours(path):
    if not os.path.exists(path):
        return None
    age_s = time.time() - os.path.getmtime(path)
    return age_s / 3600.0


def _clean_price_history(df):
    """Drop unusable OHLC rows so bad quote ticks cannot poison the engine."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    required = [c for c in ('Open', 'High', 'Low', 'Close') if c in out.columns]
    if not required:
        return pd.DataFrame()
    for col in required + (['Volume'] if 'Volume' in out.columns else []):
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.replace([float('inf'), float('-inf')], pd.NA)
    out = out.dropna(subset=required)
    for col in required:
        out = out[out[col] > 0]
    return out


def _read_cached_history(ticker, period, ttl_hours):
    """Return a cached price DataFrame if fresh enough, else None."""
    path = _cache_path(ticker, period)
    age = _cache_age_hours(path)
    if age is None or (ttl_hours is not None and age > ttl_hours):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        df = pd.DataFrame(payload['data'])
        if df.empty:
            return df
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        return _clean_price_history(df)
    except Exception:
        return None


def _write_cached_history(ticker, period, df):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        out = _clean_price_history(df)
        if out.empty:
            return
        out = out.reset_index()
        # Normalise the date column name yfinance may emit as 'Date' or 'index'
        date_col = 'Date' if 'Date' in out.columns else out.columns[0]
        out = out.rename(columns={date_col: 'Date'})
        out['Date'] = out['Date'].astype(str)
        payload = {
            'ticker': ticker,
            'period': period,
            'fetched_at': datetime.now(timezone.utc).isoformat(),
            'data': out.to_dict(orient='records'),
        }
        with open(_cache_path(ticker, period), 'w', encoding='utf-8') as f:
            json.dump(payload, f, allow_nan=False)
    except Exception:
        pass  # caching is best-effort; never break analysis over a write failure


def fetch_data(ticker, period='2y', use_cache=True, ttl_hours=DEFAULT_CACHE_TTL_HOURS,
               log=None):
    """Fetch daily history, preferring disk cache when fresh.

    `log` is an optional callable(str) for progress messages (CLI uses print).
    """
    if use_cache:
        cached = _read_cached_history(ticker, period, ttl_hours)
        if cached is not None:
            if log:
                log(f"{ticker}: cache hit")
            return cached
    try:
        if log:
            log(f"{ticker}: fetching...")
        df = yf.download(ticker, period=period, interval='1d', progress=False)
        df = _clean_price_history(df)
        if df is not None and not df.empty:
            _write_cached_history(ticker, period, df)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        if log:
            log(f"{ticker}: fetch error {e}")
        return pd.DataFrame()


def check_dividend_consistency(ticker_obj):
    """Returns (years_paying: int, is_growing: bool) over last 5 years."""
    try:
        divs = ticker_obj.dividends
        if divs.empty:
            return 0, False
        if divs.index.tz is None:
            divs.index = divs.index.tz_localize('UTC')
        cutoff = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=5)
        recent = divs[divs.index >= cutoff]
        if recent.empty:
            return 0, False
        by_year = recent.groupby(recent.index.year).sum()
        years_paying = len(by_year)
        is_growing = False
        if len(by_year) >= 3:
            vals = by_year.values
            is_growing = float(vals[-1]) >= float(vals[-3])
        return years_paying, is_growing
    except Exception:
        return 0, False


def get_stock_info(ticker):
    """Fetches name, yield, currency, fundamentals, dividend history, debt, earnings health, analysts."""
    try:
        t = yf.Ticker(ticker)
        info = t.info

        long_name = info.get('longName')
        short_name = info.get('shortName')
        name = (
            f"{long_name} - {short_name}"
            if (long_name and short_name and long_name != short_name)
            else (long_name or short_name or ticker)
        )

        if   ticker.endswith('.KL'): currency = "RM"
        elif ticker.endswith('.HK'): currency = "HKD"
        else:                        currency = "USD"

        div_rate = info.get('dividendRate', 0)
        current_price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0) or info.get('previousClose', 0)
        if div_rate and current_price and current_price > 0:
            yield_pct = (div_rate / current_price) * 100
        else:
            raw = info.get('dividendYield', 0) or 0
            yield_pct = raw * 100 if raw < 1 else raw

        pe_ratio = info.get('trailingPE') or info.get('forwardPE') or 0.0
        roe = info.get('returnOnEquity', 0)
        roe = float(roe) * 100 if roe else 0.0
        revenue = info.get('totalRevenue', 0) or 0
        profit_margin = info.get('profitMargins', 0)
        profit_margin = float(profit_margin) * 100 if profit_margin else 0.0

        debt_to_equity = info.get('debtToEquity') or 0.0

        trailing_pe = info.get('trailingPE') or 0.0
        forward_pe = info.get('forwardPE') or 0.0
        trailing_eps = info.get('trailingEps') or 0.0
        forward_eps = info.get('forwardEps') or 0.0
        earnings_growth_qoq = info.get('earningsQuarterlyGrowth') or None
        earnings_growth_ttm = info.get('earningsGrowth') or None
        revenue_growth = info.get('revenueGrowth') or None
        payout_ratio = info.get('payoutRatio') or None

        quarterly_neg_count = 0
        try:
            qi = t.quarterly_income_stmt
            if qi is not None and not qi.empty:
                ni_row = None
                for label in ('Net Income', 'NetIncome', 'Net Income Common Stockholders'):
                    if label in qi.index:
                        ni_row = qi.loc[label]
                        break
                if ni_row is not None:
                    vals = ni_row.dropna().values
                    neg_count = 0
                    for i in range(min(4, len(vals) - 4)):
                        if vals[i] < vals[i + 4]:
                            neg_count += 1
                    quarterly_neg_count = neg_count
        except Exception:
            pass

        def fmt(v):
            if v >= 1e9: return f"{v/1e9:.2f}B"
            if v >= 1e6: return f"{v/1e6:.2f}M"
            return str(v)

        div_years, div_growing = check_dividend_consistency(t)
        five_yr_avg_div_yield = info.get('fiveYearAvgDividendYield') or 0.0

        earnings_health = {
            'trailing_pe':            float(trailing_pe),
            'forward_pe':             float(forward_pe),
            'trailing_eps':           float(trailing_eps),
            'forward_eps':            float(forward_eps),
            'earnings_growth_qoq':    earnings_growth_qoq,
            'earnings_growth_ttm':    earnings_growth_ttm,
            'revenue_growth':         revenue_growth,
            'payout_ratio':           payout_ratio,
            'quarterly_neg_count':    quarterly_neg_count,
            'five_yr_avg_div_yield':  float(five_yr_avg_div_yield),
        }

        target_mean_price = info.get('targetMeanPrice') or 0.0
        target_high_price = info.get('targetHighPrice') or 0.0
        target_low_price = info.get('targetLowPrice') or 0.0
        recommendation_key = info.get('recommendationKey') or ''
        num_analysts = info.get('numberOfAnalystOpinions') or 0

        def info_float(*keys):
            for key in keys:
                val = info.get(key)
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(val) and val > 0:
                    return val
            return 0.0

        analyst_data = {
            'target_mean':    float(target_mean_price),
            'target_high':    float(target_high_price),
            'target_low':     float(target_low_price),
            'recommendation': recommendation_key.lower(),
            'num_analysts':   int(num_analysts),
            'current_price':  float(current_price),
            'day_open':       info_float('open', 'regularMarketOpen'),
            'day_high':       info_float('dayHigh', 'regularMarketDayHigh'),
            'day_low':        info_float('dayLow', 'regularMarketDayLow'),
            'previous_close':  info_float('previousClose', 'regularMarketPreviousClose'),
        }

        return (name, yield_pct, currency, float(pe_ratio), roe,
                fmt(revenue), profit_margin, div_years, div_growing,
                t, info, debt_to_equity, earnings_health, analyst_data)

    except Exception:
        return (ticker, 0.0, "N/A", 0.0, 0.0, "N/A", 0.0, 0, False, None, {}, 0.0, {}, {})


# ─────────────────────────────────────────────────────────────────────────────
# TIMING SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
def get_timing_signals(ticker_obj, info, df):
    green = []
    yellow = []
    now = pd.Timestamp.now(tz='UTC')

    try:
        ex_ts = info.get('exDividendDate')
        if ex_ts:
            ex_date = pd.Timestamp(ex_ts, unit='s', tz='UTC')
            days_gap = (ex_date - now).days
            ex_str = ex_date.strftime('%d %b %Y')
            div_rate = info.get('dividendRate', 0) or 0
            per_pmt = div_rate / 4 if div_rate else 0
            if 0 < days_gap <= 30:
                amt = f" — collect ~RM {per_pmt:.4f}/share" if per_pmt > 0 else ""
                green.append(f"📅 Ex-dividend in {days_gap} days ({ex_str}){amt}")
            elif -5 <= days_gap <= 0:
                yellow.append(
                    f"📅 Ex-dividend just passed {abs(days_gap)} day(s) ago "
                    f"— missed this round, next in ~3 months"
                )
            elif days_gap > 30:
                green.append(f"📅 Next ex-dividend in {days_gap} days ({ex_str}) — no timing urgency")
    except Exception:
        pass

    try:
        cal = ticker_obj.calendar if ticker_obj is not None else None
        e_date = None
        if isinstance(cal, dict):
            raw = cal.get('Earnings Date') or cal.get('earningsDate')
            if raw:
                e_date = pd.Timestamp(raw[0] if isinstance(raw, list) else raw)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            e_date = pd.Timestamp(cal.columns[0])

        if e_date is not None:
            if e_date.tz is None:
                e_date = e_date.tz_localize('UTC')
            days_to_earn = (e_date - now).days
            if days_to_earn >= 0:
                e_str = e_date.strftime('%d %b %Y')
                if days_to_earn <= 7:
                    yellow.append(f"⚡ Earnings in {days_to_earn} day(s) ({e_str}) — HIGH volatility risk, price can swing ±10%")
                elif days_to_earn <= 14:
                    yellow.append(f"⚡ Earnings in {days_to_earn} days ({e_str}) — expect price movement after results")
                elif days_to_earn <= 45:
                    green.append(f"📊 Earnings in {days_to_earn} days ({e_str}) — results coming, watch for upgrades")
    except Exception:
        pass

    try:
        close = df['Close'].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df['Close']
        if len(close) >= 10:
            p_now = float(close.iloc[-1])
            p_5d = float(close.iloc[-5])
            p_10d = float(close.iloc[-10])
            mom_5d = ((p_now - p_5d) / p_5d) * 100
            mom_10d = ((p_now - p_10d) / p_10d) * 100
            if mom_5d <= -3:
                green.append(f"📉 Down {abs(mom_5d):.1f}% in 5 days — short-term dip, price is lower than last week")
            elif mom_5d >= 5:
                yellow.append(f"📈 Up {mom_5d:.1f}% in 5 days — short-term spike, may pull back slightly")
            else:
                green.append(f"➡️  Stable {mom_5d:+.1f}% over 5 days — no short-term spike")
            if mom_10d <= -5:
                green.append(f"📉 Down {abs(mom_10d):.1f}% over 10 days — extended pullback, entry looks timely")
            elif mom_10d >= 8:
                yellow.append(f"📈 Up {mom_10d:.1f}% over 10 days — check if you're buying into a recent run-up")
    except Exception:
        pass

    try:
        close_s = df['Close'].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df['Close']
        if len(close_s) >= 14:
            rsi_val = float(ta.momentum.rsi(close_s, window=14).iloc[-1])
            if rsi_val >= 70:
                yellow.append(f"🔴 RSI {rsi_val:.1f} — overbought territory, elevated pullback risk")
            elif rsi_val <= 30:
                green.append(f"🟢 RSI {rsi_val:.1f} — oversold, potential mean-reversion bounce")
    except Exception:
        pass

    n_yellow = len(yellow)
    if n_yellow >= 2:
        verdict = "⏳ WAIT"
    elif n_yellow == 1:
        verdict = "🟡 CAUTION"
    else:
        verdict = "🟢 GOOD"

    return green, yellow, verdict


# ─────────────────────────────────────────────────────────────────────────────
# DEBT / LEVERAGE FLAG
# ─────────────────────────────────────────────────────────────────────────────
def debt_flag(sector, debt_to_equity):
    if debt_to_equity <= 0:
        return "N/A", False
    ratio = debt_to_equity / 100
    if sector == 'bank':
        if ratio > 15:
            return f"⚠️  D/E {ratio:.1f}x — extremely high even for a bank", True
        elif ratio > 10:
            return f"🟡 D/E {ratio:.1f}x — high leverage, monitor capital ratios", False
        return f"✅ D/E {ratio:.1f}x — normal bank leverage", False
    elif sector == 'reit':
        if ratio > 1.5:
            return f"⚠️  D/E {ratio:.1f}x — high gearing for REIT, rate-sensitive", True
        elif ratio > 0.8:
            return f"🟡 D/E {ratio:.1f}x — moderate gearing, watch interest coverage", False
        return f"✅ D/E {ratio:.1f}x — conservative gearing", False
    else:
        if ratio > 2.0:
            return f"⚠️  D/E {ratio:.1f}x — high debt load", True
        elif ratio > 1.0:
            return f"🟡 D/E {ratio:.1f}x — moderate debt", False
        return f"✅ D/E {ratio:.1f}x — low debt", False


# ─────────────────────────────────────────────────────────────────────────────
# EARNINGS HEALTH
# ─────────────────────────────────────────────────────────────────────────────
def get_earnings_health_signals(earnings_health, sector='general'):
    warnings = []
    penalty = 0
    is_cyclical = sector in CYCLICAL_SECTORS

    eq = earnings_health.get('earnings_growth_qoq')
    et = earnings_health.get('earnings_growth_ttm')
    rv = earnings_health.get('revenue_growth')
    pr = earnings_health.get('payout_ratio')
    tpe = earnings_health.get('trailing_pe', 0.0)
    fpe = earnings_health.get('forward_pe', 0.0)
    teps = earnings_health.get('trailing_eps', 0.0)
    feps = earnings_health.get('forward_eps', 0.0)
    qneg = earnings_health.get('quarterly_neg_count', 0)

    if qneg >= 3:
        warnings.append(f"🔴 {qneg}/4 recent quarters below prior-year — sustained earnings decline")
        penalty += 20
    elif qneg == 2:
        warnings.append(f"🟠 {qneg}/4 recent quarters below prior-year — pattern of weakness")
        penalty += 10
    elif qneg == 1:
        warnings.append(f"🟡 1/4 recent quarters below prior-year — monitor next result")
        penalty += 3

    if eq is not None:
        pct = eq * 100
        if eq <= -0.40:
            warnings.append(f"🔴 Latest quarter earnings {pct:+.0f}% YoY — severe collapse")
            penalty += 25
        elif eq <= -0.20:
            warnings.append(f"🟠 Latest quarter earnings {pct:+.0f}% YoY — significant decline")
            penalty += 15
        elif eq <= -0.05:
            warnings.append(f"🟡 Latest quarter earnings {pct:+.0f}% YoY — softening")
            penalty += 5
        elif eq >= 0.15:
            if not is_cyclical or qneg == 0:
                warnings.append(f"🟢 Latest quarter earnings {pct:+.0f}% YoY — strong growth")
                penalty -= 5
            else:
                warnings.append(
                    f"🟡 Latest quarter earnings {pct:+.0f}% YoY — bounce after weakness "
                    f"({qneg} of last 4 quarters still below prior-year)"
                )

    if et is not None:
        pct = et * 100
        if et <= -0.30 and (eq is None or eq > -0.20):
            warnings.append(f"🟠 TTM earnings growth {pct:+.0f}% — full-year trend declining")
            penalty += 10
        elif et >= 0.10 and eq is None:
            warnings.append(f"🟢 TTM earnings growth {pct:+.0f}% — improving fundamentals")
            penalty -= 3

    if rv is not None:
        pct = rv * 100
        if rv <= -0.15:
            warnings.append(f"🟠 Revenue shrinking {pct:+.0f}% YoY — top-line under pressure")
            penalty += 8
        elif rv >= 0.10:
            warnings.append(f"🟢 Revenue growing {pct:+.0f}% YoY — business expanding")
            penalty -= 3

    if teps > 0 and feps > 0:
        eps_change = (feps - teps) / abs(teps)
        if eps_change <= -0.20:
            warnings.append(f"🔴 Forward EPS {feps:.3f} vs Trailing {teps:.3f} — analysts cut estimates {abs(eps_change)*100:.0f}%, earnings declining")
            penalty += 18
        elif eps_change <= -0.08:
            warnings.append(f"🟠 Forward EPS {feps:.3f} vs Trailing {teps:.3f} — mild downward revision, earnings under pressure")
            penalty += 8
        elif eps_change >= 0.10:
            warnings.append(f"🟢 Forward EPS {feps:.3f} vs Trailing {teps:.3f} — analysts raising estimates, earnings momentum positive")
            penalty -= 5

    if pr is not None and pr > 0:
        t1 = 0.60 if is_cyclical else 0.75
        t2 = 0.75 if is_cyclical else 0.90
        t3 = 0.90 if is_cyclical else 1.00
        if pr > t3:
            warnings.append(
                f"🔴 Payout ratio {pr*100:.0f}% — dividend exceeds safe limit"
                + (" for cyclical sector" if is_cyclical else ", paid from reserves")
            )
            penalty += 20
        elif pr > t2:
            warnings.append(f"🟠 Payout ratio {pr*100:.0f}% — nearly all earnings consumed, cut risk high")
            penalty += 12
        elif pr > t1:
            warnings.append(
                f"🟡 Payout ratio {pr*100:.0f}% — stretched"
                + (" (tighter limit for cyclical sector)" if is_cyclical else ", limited buffer")
            )
            penalty += 5
        else:
            warnings.append(f"🟢 Payout ratio {pr*100:.0f}% — dividend well covered by earnings")

    if tpe > 0 and fpe > 0:
        pe_expansion = (fpe - tpe) / tpe
        eps_declining = (teps > 0 and feps > 0 and feps < teps)
        if pe_expansion >= 0.25:
            warnings.append(f"🔴 Forward PE {fpe:.1f}x vs Trailing {tpe:.1f}x — analysts pricing in earnings decline")
            penalty += 15
        elif pe_expansion >= 0.10:
            warnings.append(f"🟡 Forward PE {fpe:.1f}x vs Trailing {tpe:.1f}x — mild earnings headwind expected")
            penalty += 5
        elif pe_expansion <= -0.15:
            if eps_declining:
                warnings.append(f"🟡 Forward PE {fpe:.1f}x vs Trailing {tpe:.1f}x — looks cheaper but EPS estimates were cut, not earnings growth")
                penalty += 5
            else:
                warnings.append(f"🟢 Forward PE {fpe:.1f}x vs Trailing {tpe:.1f}x — analysts expect {abs(pe_expansion)*100:.0f}% earnings growth")
                penalty -= 5

    if penalty >= 35:
        label = "🔴 DETERIORATING"
    elif penalty >= 15:
        label = "🟠 WEAKENING"
    elif penalty >= 5:
        label = "🟡 MIXED"
    elif penalty <= -5:
        label = "🟢 IMPROVING"
    else:
        label = "🟢 STABLE"

    return warnings, max(penalty, 0), label


# ─────────────────────────────────────────────────────────────────────────────
# FEE CALCULATOR (BURSA)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_min_lots(price_per_unit, ticker):
    if not ticker.endswith('.KL'):
        return None
    if not math.isfinite(price_per_unit) or price_per_unit <= 0:
        return None
    for lots in range(1, 100):
        total_value = price_per_unit * 100 * lots
        stamp_duty = max(math.ceil(total_value / 1000), 1)
        clearing_fee = total_value * 0.0003
        platform_fee = 3.00
        total_fee = platform_fee + stamp_duty + clearing_fee
        fee_pct = (total_fee / total_value) * 100
        if fee_pct < 1.0:
            return {'lots': lots, 'cost': total_value, 'fee_pct': fee_pct}
    return {'lots': 1, 'cost': price_per_unit * 100, 'fee_pct': 100.0}


def bursa_fees(total_value):
    """Return total transaction fee (RM) for a Bursa trade of given value."""
    if not math.isfinite(total_value) or total_value <= 0:
        return 0.0
    stamp_duty = max(math.ceil(total_value / 1000), 1)
    clearing_fee = total_value * 0.0003
    platform_fee = 3.00
    return platform_fee + stamp_duty + clearing_fee


def price_tick(ticker, price):
    """Approximate board-lot tick size for limit-order suggestions."""
    if not math.isfinite(price) or price <= 0:
        return 0.01
    if ticker.endswith('.KL'):
        if price < 1:
            return 0.005
        if price < 10:
            return 0.01
        if price < 100:
            return 0.02
        return 0.10
    return 0.01


def round_to_tick(price, ticker, mode='nearest'):
    """Round a price to a valid-looking tick. `mode` can be nearest/down/up."""
    if not math.isfinite(price) or price <= 0:
        return 0.0
    tick = price_tick(ticker, price)
    scaled = price / tick
    if mode == 'down':
        rounded = math.floor(scaled) * tick
    elif mode == 'up':
        rounded = math.ceil(scaled) * tick
    else:
        rounded = round(scaled) * tick
    return round(max(0.0, rounded), 4)


def limit_order_plan(item):
    """Suggest a practical buy-limit ladder from the latest quote and day range.

    This is an execution helper, not a prediction engine: it gives a reasonable
    bid inside today's range so the user can avoid chasing a live quote at the
    high of the session.
    """
    ticker = item.get('ticker', '')
    curr = item.get('currency', '')
    live = item.get('order_price', item.get('price', 0.0))
    close = item.get('price', live)
    if not math.isfinite(live) or live <= 0:
        return None

    day_low = item.get('day_low', 0.0)
    day_high = item.get('day_high', 0.0)
    day_open = item.get('day_open', 0.0)
    prev_close = item.get('prev_close', close)

    anchors = [v for v in (live, close, day_open, prev_close) if math.isfinite(v) and v > 0]
    if not (math.isfinite(day_low) and day_low > 0):
        day_low = min(anchors) if anchors else live
    if not (math.isfinite(day_high) and day_high > 0):
        day_high = max(anchors) if anchors else live
    if day_high < day_low:
        day_low, day_high = day_high, day_low

    day_range = day_high - day_low
    synthetic_range = max(live * 0.0125, price_tick(ticker, live) * 4)
    if day_range <= 0:
        day_low = max(price_tick(ticker, live), live - synthetic_range / 2)
        day_high = live + synthetic_range / 2
        day_range = day_high - day_low

    position_in_range = (live - day_low) / day_range if day_range > 0 else 0.5
    rsi = item.get('rsi', 50.0)

    if position_in_range >= 0.67:
        strategy = 'patient_pullback'
        label = 'Patient pullback bid'
        target_raw = live - (day_range * 0.30)
        patient_raw = live - (day_range * 0.50)
    elif rsi >= 65:
        strategy = 'cool_off'
        label = 'Let it cool off'
        target_raw = live - (day_range * 0.25)
        patient_raw = live - (day_range * 0.45)
    elif position_in_range <= 0.33:
        strategy = 'near_low'
        label = 'Near day low'
        target_raw = live
        patient_raw = max(day_low, live - (day_range * 0.20))
    else:
        strategy = 'mid_range'
        label = 'Mid-range limit'
        target_raw = live - (day_range * 0.18)
        patient_raw = live - (day_range * 0.38)

    floor_price = max(price_tick(ticker, live), day_low)
    suggested = round_to_tick(max(floor_price, min(live, target_raw)), ticker, 'nearest')
    patient = round_to_tick(max(floor_price, min(suggested, patient_raw)), ticker, 'down')
    max_entry = round_to_tick(live, ticker, 'up')

    # Keep the main suggestion no higher than the live quote and no lower than
    # the patient bid after tick rounding.
    suggested = min(suggested, max_entry)
    if suggested < patient:
        suggested = patient

    gap_pct = ((live - suggested) / live * 100) if live > 0 else 0.0
    day_low_s = f"{curr} {day_low:.3f}" if curr else f"{day_low:.3f}"
    day_high_s = f"{curr} {day_high:.3f}" if curr else f"{day_high:.3f}"
    live_s = f"{curr} {live:.3f}" if curr else f"{live:.3f}"
    suggested_s = f"{curr} {suggested:.3f}" if curr else f"{suggested:.3f}"

    if suggested < live:
        reason = (
            f"Live quote {live_s} sits at {position_in_range * 100:.0f}% of today's "
            f"{day_low_s}-{day_high_s} range, so bid around {suggested_s} instead of chasing."
        )
    else:
        reason = (
            f"Live quote {live_s} is already close to the lower part of today's "
            f"{day_low_s}-{day_high_s} range."
        )

    return {
        'strategy': strategy,
        'label': label,
        'suggested_limit_price': suggested,
        'patient_limit_price': patient,
        'max_entry_price': max_entry,
        'day_low': round(day_low, 4),
        'day_high': round(day_high, 4),
        'day_open': round(day_open, 4) if math.isfinite(day_open) and day_open > 0 else None,
        'gap_from_live_pct': round(gap_pct, 2),
        'reason': reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_stock(df):
    df = _clean_price_history(df)
    if len(df) < 200:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    lookback = min(252, len(df))
    high_window = df['High'].iloc[-lookback:]
    high_idx = high_window.idxmax()
    now_ts = pd.Timestamp.now(tz=high_idx.tz if high_idx.tz else None)
    months_since_high = (now_ts.year - high_idx.year) * 12 + (now_ts.month - high_idx.month)
    stats = {
        'price':             float(latest['Close']),
        'day_open':          float(latest['Open']),
        'day_high':          float(latest['High']),
        'day_low':           float(latest['Low']),
        'prev_close':        float(prev['Close']),
        'rsi':               float(latest['RSI']),
        'sma50':             float(latest['SMA_50']),
        'sma200':            float(latest['SMA_200']),
        'volume_strong':     float(latest['Volume']) > float(latest['Vol_Avg']) * 1.5,
        'week52_high':       float(high_window.max()),
        'week52_low':        float(df['Low'].iloc[-lookback:].min()),
        'months_since_high': months_since_high,
    }
    required = ('price', 'rsi', 'sma50', 'sma200', 'week52_high', 'week52_low')
    if any(not math.isfinite(stats[k]) or stats[k] <= 0 for k in required if k != 'rsi'):
        return None
    if not math.isfinite(stats['rsi']):
        stats['rsi'] = 50.0
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# SAVINGS SCORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def calculate_savings_score(stock_data, sector, recent_sectors, months_since_last_buy):
    price = stock_data['price']
    sma50 = stock_data['sma50']
    sma200 = stock_data['sma200']
    pe = stock_data.get('pe_ratio', 0)
    roe = stock_data.get('roe', 0)
    div_yield = stock_data.get('div_yield', 0)
    week52_high = stock_data.get('week52_high', price)
    div_years = stock_data.get('div_years', 0)
    div_growing = stock_data.get('div_growing', False)
    eh_penalty = stock_data.get('eh_penalty', 0)

    score = 0
    bd = {}

    rsi = stock_data.get('rsi', 50)
    months_since_high = stock_data.get('months_since_high', 0)
    is_cyclical = sector in CYCLICAL_SECTORS

    trend_pts = 15 if sma50 > sma200 else 0
    score += trend_pts
    bd['trend'] = trend_pts

    pullback = ((week52_high - price) / week52_high * 100) if week52_high > 0 else 0
    falling_knife = is_cyclical and months_since_high >= 9 and price < sma200

    if pullback >= 20:
        entry_pts = 30
    elif pullback >= 15:
        entry_pts = 28
    elif pullback >= 10:
        entry_pts = 22
    elif pullback >= 5:
        entry_pts = 10
    elif pullback >= 3:
        entry_pts = 6
    else:
        entry_pts = -10

    if falling_knife:
        entry_pts = min(entry_pts, 12)
    if rsi >= 65 and entry_pts > 0:
        entry_pts -= 5

    score += entry_pts
    bd['entry'] = entry_pts

    thresh = SECTOR_PE_THRESHOLDS.get(sector, SECTOR_PE_THRESHOLDS['general'])
    val_pts = 0 if (pe <= 0 or pe > 200) else 15 if pe < thresh['great'] else 8 if pe < thresh['ok'] else 2
    score += val_pts
    bd['valuation'] = val_pts

    if div_yield >= 8:
        yield_pts = 20
    elif div_yield >= 6:
        yield_pts = 16
    elif div_yield >= 4:
        yield_pts = 12
    elif div_yield >= 2:
        yield_pts = 7
    else:
        yield_pts = round(min(div_yield * 2, 6))

    if is_cyclical:
        yield_pts = min(yield_pts, 12)

    yield_trap = 0
    if div_yield >= 6 and pullback < 8:
        yield_trap = -5
        yield_pts = max(0, yield_pts + yield_trap)

    score += yield_pts
    bd['yield'] = yield_pts
    bd['yield_trap'] = yield_trap

    roe_pts = 10 if roe > 12 else 5 if roe > 8 else 2 if roe > 0 else 0
    score += roe_pts
    bd['roe'] = roe_pts

    analyst_data = stock_data.get('analyst_data', {})
    target_mean = analyst_data.get('target_mean', 0.0)
    curr_price_a = analyst_data.get('current_price', price)
    rec_key = analyst_data.get('recommendation', '')
    n_analysts = analyst_data.get('num_analysts', 0)

    analyst_pts = 0
    if n_analysts >= 5 and target_mean > 0 and curr_price_a > 0:
        upside = (target_mean - curr_price_a) / curr_price_a
        if upside >= 0.20:
            analyst_pts = 10
        elif upside >= 0.12:
            analyst_pts = 7
        elif upside >= 0.06:
            analyst_pts = 4
        elif upside < 0:
            analyst_pts = -5

    if analyst_pts > 0:
        if rec_key in ('strong_buy', 'buy'):
            analyst_pts = min(analyst_pts + 2, 10)
        elif rec_key in ('underperform', 'sell', 'strong_sell'):
            analyst_pts = max(analyst_pts - 4, 0)

    score += analyst_pts
    bd['analyst'] = analyst_pts

    earnings_health_data = stock_data.get('earnings_health', {})
    feps_g = earnings_health_data.get('forward_eps', 0.0)
    teps_g = earnings_health_data.get('trailing_eps', 0.0)
    is_tech = (sector == 'tech')

    growth_pts = 0
    if teps_g > 0 and feps_g > 0:
        eps_growth_rate = (feps_g - teps_g) / abs(teps_g)
        if eps_growth_rate >= 0.20:
            growth_pts = 10 if is_tech else 6
        elif eps_growth_rate >= 0.10:
            growth_pts = 7 if is_tech else 4
        elif eps_growth_rate >= 0.05:
            growth_pts = 4 if is_tech else 2
        elif eps_growth_rate <= -0.10:
            growth_pts = -3
    score += growth_pts
    bd['eps_growth'] = growth_pts

    five_yr_avg_dy = earnings_health_data.get('five_yr_avg_div_yield', 0.0)
    rerate_pts = 0
    if five_yr_avg_dy > 0 and div_yield > 0:
        yield_premium = (div_yield - five_yr_avg_dy) / five_yr_avg_dy
        if yield_premium >= 0.30:
            rerate_pts = 8
        elif yield_premium >= 0.15:
            rerate_pts = 4
        elif yield_premium <= -0.20:
            rerate_pts = -3
    score += rerate_pts
    bd['rerate'] = rerate_pts

    div_pts = 15 if div_years >= 5 else 10 if div_years >= 3 else 5 if div_years >= 1 else 0
    if div_growing and div_pts > 0:
        div_pts = min(div_pts + 3, 15)
    score += div_pts
    bd['dividend'] = div_pts

    sector_count = recent_sectors.count(sector) if recent_sectors else 0
    if not recent_sectors:
        rot_pts = 4
    elif sector_count == 0:
        rot_pts = 8
    elif sector_count == 1:
        rot_pts = 3
    else:
        rot_pts = 0
    score += rot_pts
    bd['rotation'] = rot_pts

    if months_since_last_buy is None:
        fresh_pts = 4
    elif months_since_last_buy >= 6:
        fresh_pts = 4
    elif months_since_last_buy >= 3:
        fresh_pts = 2
    else:
        fresh_pts = 0
    score += fresh_pts
    bd['freshness'] = fresh_pts

    if months_since_last_buy is not None and months_since_last_buy < 2:
        score -= 20
        bd['recent_buy'] = -20
    else:
        bd['recent_buy'] = 0

    score -= eh_penalty
    bd['eh_penalty'] = -eh_penalty

    return round(score), bd


def _points_from_roe(roe, sector):
    if roe <= 0:
        return 0
    if sector == 'bank':
        if roe >= 12: return 14
        if roe >= 10: return 11
        if roe >= 8:  return 8
        if roe >= 5:  return 4
        return 1
    if roe >= 22: return 14
    if roe >= 16: return 12
    if roe >= 12: return 10
    if roe >= 8:  return 6
    if roe >= 4:  return 3
    return 1


def calculate_long_term_score(stock_data, sector, recent_sectors, months_since_last_buy):
    """Rank for the default monthly pick: durable business first, entry second.

    This intentionally differs from the legacy savings score. The old score is
    useful for finding a technical/value setup; this one is for money intended
    to sit for years, so weak quality can cap an otherwise cheap stock.
    """
    price = stock_data['price']
    sma50 = stock_data['sma50']
    sma200 = stock_data['sma200']
    pe = stock_data.get('pe_ratio', 0)
    roe = stock_data.get('roe', 0)
    div_yield = stock_data.get('div_yield', 0)
    week52_high = stock_data.get('week52_high', price)
    profit_margin = stock_data.get('profit_margin', 0)
    div_years = stock_data.get('div_years', 0)
    div_growing = stock_data.get('div_growing', False)
    earnings_health = stock_data.get('earnings_health', {})
    eh_label = stock_data.get('eh_label', '')
    analyst_data = stock_data.get('analyst_data', {})
    pullback = ((week52_high - price) / week52_high * 100) if week52_high > 0 else 0
    rsi = stock_data.get('rsi', 50)

    bd = {}
    flags = []

    quality = _points_from_roe(roe, sector)
    if sector != 'bank':
        if profit_margin >= 20:
            quality += 6
        elif profit_margin >= 12:
            quality += 4
        elif profit_margin >= 6:
            quality += 2

    if eh_label == '🟢 IMPROVING':
        quality += 8
    elif eh_label == '🟢 STABLE':
        quality += 6
    elif eh_label == '🟡 MIXED':
        quality += 2
    elif eh_label == '🟠 WEAKENING':
        quality -= 4
    elif eh_label == '🔴 DETERIORATING':
        quality -= 12

    teps = earnings_health.get('trailing_eps', 0.0)
    feps = earnings_health.get('forward_eps', 0.0)
    if teps > 0 and feps > 0:
        eps_growth = (feps - teps) / abs(teps)
        if eps_growth >= 0.15:
            quality += 5
        elif eps_growth >= 0.05:
            quality += 3
        elif eps_growth <= -0.10:
            quality -= 4

    if div_years >= 5:
        quality += 5
    elif div_years >= 3:
        quality += 3
    elif div_years >= 1:
        quality += 1
    if div_growing and div_years >= 3:
        quality += 2

    if sector == 'bank' and 0 < roe < 6:
        quality = min(quality, 18)
        flags.append("Low bank ROE caps long-term conviction")

    quality = max(0, min(40, quality))
    bd['quality'] = round(quality)

    thresh = SECTOR_PE_THRESHOLDS.get(sector, SECTOR_PE_THRESHOLDS['general'])
    valuation = 0
    if pe > 0 and pe <= 200:
        if pe < thresh['great']:
            valuation += 12
        elif pe < thresh['ok']:
            valuation += 8
        elif pe < thresh['ok'] * 1.25:
            valuation += 4
        else:
            valuation += 1

    five_yr_avg_dy = earnings_health.get('five_yr_avg_div_yield', 0.0)
    if five_yr_avg_dy > 0 and div_yield > 0:
        yield_premium = (div_yield - five_yr_avg_dy) / five_yr_avg_dy
        if yield_premium >= 0.30:
            valuation += 6
        elif yield_premium >= 0.15:
            valuation += 3
        elif yield_premium <= -0.20:
            valuation -= 3

    target_mean = analyst_data.get('target_mean', 0.0)
    curr_price_a = analyst_data.get('current_price', price)
    n_analysts = analyst_data.get('num_analysts', 0)
    if n_analysts >= 5 and target_mean > 0 and curr_price_a > 0:
        upside = (target_mean - curr_price_a) / curr_price_a
        if upside >= 0.20:
            valuation += 4
        elif upside >= 0.10:
            valuation += 2
        elif upside < 0:
            valuation -= 3

    if sector == 'bank' and 0 < roe < 6:
        valuation = min(valuation, 12)
    if sector in {'materials', 'chemicals', 'energy'} and pe > thresh['ok'] and div_yield < 2:
        valuation = min(valuation, 8)
        flags.append("Cyclical low-yield stock needs a wider valuation margin")
    valuation = max(0, min(25, valuation))
    bd['valuation'] = round(valuation)

    if pullback >= 20:
        entry = 10
    elif pullback >= 15:
        entry = 9
    elif pullback >= 10:
        entry = 7
    elif pullback >= 5:
        entry = 5
    elif pullback >= 3:
        entry = 3
    else:
        entry = -2
    if sma50 > sma200:
        entry += 5
    if price > sma200:
        entry += 3
    if rsi <= 35:
        entry += 2
    elif rsi >= 70:
        entry -= 3
    if sector in CYCLICAL_SECTORS and stock_data.get('months_since_high', 0) >= 9 and price < sma200:
        entry = min(entry, 8)
        flags.append("Cyclical falling-knife risk limits entry score")
    if sector in {'materials', 'chemicals', 'energy'} and pe > thresh['ok'] and div_yield < 2:
        entry = min(entry, 14)
    entry = max(0, min(20, entry))
    bd['entry'] = round(entry)

    income = 0
    if 4 <= div_yield <= 7:
        income += 5
    elif 2 <= div_yield < 4:
        income += 3
    elif div_yield > 7:
        income += 3
        flags.append("Very high yield needs manual sustainability check")
    elif div_yield > 0:
        income += 1
    if div_years >= 5:
        income += 4
    elif div_years >= 3:
        income += 3
    elif div_years >= 1:
        income += 1
    if div_growing and div_years >= 3:
        income += 1
    income = max(0, min(10, income))
    bd['income'] = round(income)

    sector_count = recent_sectors.count(sector) if recent_sectors else 0
    if not recent_sectors:
        portfolio = 3
    elif sector_count == 0:
        portfolio = 5
    elif sector_count == 1:
        portfolio = 2
    else:
        portfolio = 0
        flags.append(f"Recent purchases already concentrated in {sector}")
    if months_since_last_buy is not None and months_since_last_buy < 3:
        portfolio = min(portfolio, 1)
        flags.append("Bought recently; avoid doubling up too soon")
    bd['portfolio'] = portfolio

    total = quality + valuation + entry + income + portfolio
    if stock_data.get('is_high_debt'):
        total -= 8
        flags.append("High leverage reduces long-term margin of safety")
    if stock_data.get('data_uncertain'):
        total -= 40
        flags.append("Price data uncertain")
    if sector in {'materials', 'chemicals', 'energy'} and pe > thresh['ok'] and div_yield < 2:
        total -= 4

    total = max(0, min(MAX_LONG_TERM_SCORE, round(total)))
    if total >= 80:
        label = "🟢 HIGH-CONVICTION"
    elif total >= 68:
        label = "🟢 BUY CANDIDATE"
    elif total >= 55:
        label = "🟡 WATCHLIST"
    else:
        label = "🔴 NOT INVESTMENT-GRADE"
    return total, bd, label, flags


def effective_score(item):
    """Score adjusted for actionability (used for sort order, not display)."""
    s = item['score']
    is_golden = item['sma50'] > item['sma200']
    tv = item['timing_verdict']
    if not is_golden:
        s -= 40
    elif tv == "⏳ WAIT":
        s -= 15
    elif tv == "🟡 CAUTION":
        s -= 5
    return s


def effective_long_term_score(item):
    """Long-term score adjusted for actionability; this drives default ranking."""
    s = item.get('long_term_score', item.get('score', 0))
    is_golden = item['sma50'] > item['sma200']
    tv = item['timing_verdict']
    if item.get('data_uncertain'):
        s -= 40
    if not is_golden:
        s -= 18
    elif tv == "⏳ WAIT":
        s -= 10
    elif tv == "🟡 CAUTION":
        s -= 4
    if item.get('eh_label') == '🔴 DETERIORATING':
        s -= 25
    elif item.get('eh_label') == '🟠 WEAKENING':
        s -= 8
    return s


def is_actionable(r):
    """Whether a stock is a clean buy *right now*.

    Single source of truth for the "should I actually pull the trigger" gate,
    used by pick_actionable, budget_fit and the UI badge so they can never drift
    apart. A stock is actionable only if it is in an uptrend, not flagged to
    WAIT, not showing deteriorating earnings, not bought in the last 2 months,
    and its price data looks trustworthy (see the stale-price guardrail).
    """
    return bool(
        r.get('is_golden')
        and r.get('timing_verdict') != "⏳ WAIT"
        and r.get('eh_label', '') != '🔴 DETERIORATING'
        and r.get('long_term_score', 100) >= 55
        and (r.get('months_since_buy') is None or r['months_since_buy'] >= 2)
        and not r.get('data_uncertain', False)
    )


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS (shared by CLI & API explanations)
# ─────────────────────────────────────────────────────────────────────────────
def trend_label(sma50, sma200, price):
    if sma50 > sma200 and price > sma200:
        return "🌟 GOLDEN + above SMA200"
    elif sma50 > sma200:
        return "🌤  GOLDEN (price still below SMA200)"
    return "💀 DEATH cross"


def pe_flag(pe):
    if pe <= 0:  return " ⚠️ N/A"
    if pe < 3:   return " ⚠️ Suspect data"
    if pe > 200: return " ⚠️ Extreme"
    return ""


def evaluate_accumulation_signal(current_price, week52_high, div_yield):
    pullback = ((week52_high - current_price) / week52_high * 100) if week52_high > 0 else 0
    if pullback >= 15:
        return ("🛒 ADD MORE", f"Price {pullback:.1f}% below 52-week high — great accumulation point")
    elif pullback >= 8:
        return ("💰 CONSIDER ADDING", f"Price {pullback:.1f}% below peak — decent entry for extra savings")
    elif pullback <= 3:
        return ("✅ HOLD & COLLECT", f"Near 52-week high ({pullback:.1f}% below) — sit tight, collect dividends")
    return ("✅ HOLD", f"Fair price zone ({pullback:.1f}% below 52-week high) — hold for income")


def project_annual_dividend(open_units, div_yield_pct, current_price):
    annual_per_share = (div_yield_pct / 100) * current_price
    annual_total = annual_per_share * open_units
    return annual_total, annual_total / 12


def one_line_reason(item):
    """The single sentence that justifies the buy — what you read instead of thinking."""
    bits = []
    if item.get('long_term_score', 0) >= 68:
        bits.append(f"long-term score {item['long_term_score']}")
    if item.get('roe', 0) >= 12:
        bits.append(f"{item['roe']:.0f}% ROE")
    pullback = (item['week52_high'] - item['price']) / item['week52_high'] * 100 if item['week52_high'] > 0 else 0
    if pullback >= 10:
        bits.append(f"{pullback:.0f}% below its 52-week high")
    if item.get('div_yield', 0) >= 4:
        bits.append(f"{item['div_yield']:.1f}% yield")
    if item['sma50'] > item['sma200']:
        bits.append("uptrend intact")
    if item.get('div_years', 0) >= 5:
        bits.append(f"{item['div_years']} yrs of dividends")
    ad = item.get('analyst_data', {})
    if ad.get('target_mean', 0) > 0 and ad.get('current_price', 0) > 0:
        up = (ad['target_mean'] - ad['current_price']) / ad['current_price'] * 100
        if up >= 8:
            bits.append(f"{up:.0f}% analyst upside")
    if not bits:
        bits.append("best available score this month")
    return ", ".join(bits[:3]) + "."


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL ANALYSIS  (returns structured data — no printing)
# ─────────────────────────────────────────────────────────────────────────────
def analyze_all(config=None, purchase_log=None, sell_log=None,
                use_cache=True, ttl_hours=DEFAULT_CACHE_TTL_HOURS,
                progress=None, polite_delay=0.0):
    """Score every ticker in the config and return ranked results + context.

    progress: optional callable(done:int, total:int, message:str) for UIs.
    Returns a plain dict (JSON-serialisable) so the API can return it directly.
    """
    if config is None:
        config = load_config() or {'tickers': [], 'period': '2y'}
    if purchase_log is None:
        purchase_log = load_purchase_log()
    if sell_log is None:
        sell_log = load_sell_log()

    tickers_cfg = config.get('tickers', [])
    period = config.get('period', '2y')
    ticker_sector_map = build_ticker_sector_map(tickers_cfg)
    recent_sectors = get_recent_sectors(purchase_log, ticker_sector_map, n=3)
    last_sector = get_last_sector(purchase_log, ticker_sector_map)

    results = []
    fetched_cache = {}
    total = len(tickers_cfg)

    for i, entry in enumerate(tickers_cfg):
        ticker = entry['ticker']
        sector = entry.get('sector', 'general')
        if progress:
            progress(i, total, f"Analyzing {ticker}")

        (name, div_yield, currency, pe_ratio, roe,
         revenue, profit_margin, div_years, div_growing,
         ticker_obj, info, debt_to_equity, earnings_health, analyst_data) = get_stock_info(ticker)

        df = fetch_data(ticker, period, use_cache=use_cache, ttl_hours=ttl_hours)
        fetched_cache[ticker] = (df, div_yield, name)

        if df.empty:
            continue
        stats = analyze_stock(df)
        if not stats:
            continue

        t_green, t_yellow, t_verdict = get_timing_signals(ticker_obj, info, df)
        months_since = get_months_since_last_buy(purchase_log, ticker)
        debt_str, is_high_debt = debt_flag(sector, debt_to_equity)
        eh_warnings, eh_penalty, eh_label = get_earnings_health_signals(earnings_health, sector)

        # The score is built on daily history from `df`, but order sizing should
        # use the best live quote yfinance exposes. Bursa history can lag intra-
        # day and sometimes leaves the open as the latest "Close"; do not size a
        # broker order from that if `info.currentPrice` is available.
        info_price = analyst_data.get('current_price', 0) or 0
        df_price = stats['price']
        data_uncertain = False
        data_warning = None
        order_price = df_price
        price_source = 'history_close'
        if info_price > 0 and df_price > 0:
            divergence = abs(info_price - df_price) / df_price
            if math.isfinite(info_price):
                order_price = info_price
                price_source = 'live_quote'
            if divergence > 0.05:
                data_uncertain = True
                data_warning = (
                    f"Live price {currency} {info_price:.3f} diverges {divergence * 100:.0f}% "
                    f"from last close {currency} {df_price:.3f} — price data may be stale"
                )
            elif divergence > 0.01:
                data_warning = (
                    f"Order price uses live quote {currency} {info_price:.3f}; "
                    f"history close is {currency} {df_price:.3f}"
                )

        # Prefer live quote-session OHLC for execution advice when available.
        for stat_key, info_key in (
            ('day_open', 'day_open'),
            ('day_high', 'day_high'),
            ('day_low', 'day_low'),
            ('prev_close', 'previous_close'),
        ):
            live_val = analyst_data.get(info_key, 0) or 0
            if math.isfinite(live_val) and live_val > 0:
                stats[stat_key] = live_val

        stats.update({
            'ticker': ticker, 'sector': sector, 'name': name, 'currency': currency,
            'div_yield': div_yield, 'pe_ratio': pe_ratio, 'roe': roe, 'revenue': revenue,
            'profit_margin': profit_margin, 'div_years': div_years, 'div_growing': div_growing,
            'debt_to_equity': debt_to_equity, 'debt_str': debt_str, 'is_high_debt': is_high_debt,
            'order_price': order_price, 'price_source': price_source,
            'smart_lots': calculate_min_lots(order_price, ticker),
            'timing_green': t_green, 'timing_yellow': t_yellow, 'timing_verdict': t_verdict,
            'months_since_buy': months_since,
            'eh_warnings': eh_warnings, 'eh_penalty': eh_penalty, 'eh_label': eh_label,
            'analyst_data': analyst_data, 'earnings_health': earnings_health,
            'data_uncertain': data_uncertain, 'data_warning': data_warning,
        })

        score, breakdown = calculate_savings_score(stats, sector, recent_sectors, months_since)
        stats['score'] = score
        stats['breakdown'] = breakdown
        stats['effective_score'] = effective_score(stats)
        stats['is_golden'] = stats['sma50'] > stats['sma200']
        long_score, long_bd, long_label, long_flags = calculate_long_term_score(
            stats, sector, recent_sectors, months_since
        )
        stats['long_term_score'] = long_score
        stats['long_term_breakdown'] = long_bd
        stats['long_term_label'] = long_label
        stats['long_term_flags'] = long_flags
        stats['effective_long_term_score'] = effective_long_term_score(stats)
        stats['pullback_pct'] = (
            (stats['week52_high'] - stats['price']) / stats['week52_high'] * 100
            if stats['week52_high'] > 0 else 0
        )
        stats['trend_label'] = trend_label(stats['sma50'], stats['sma200'], stats['price'])
        stats['reason'] = one_line_reason(stats)
        stats['entry_plan'] = limit_order_plan(stats)
        if stats['entry_plan']:
            stats['suggested_limit_price'] = stats['entry_plan']['suggested_limit_price']
            stats['patient_limit_price'] = stats['entry_plan']['patient_limit_price']
            stats['max_entry_price'] = stats['entry_plan']['max_entry_price']
        results.append(stats)

        if polite_delay:
            time.sleep(polite_delay)

    results.sort(key=lambda r: r['effective_long_term_score'], reverse=True)
    if progress:
        progress(total, total, "Done")

    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'currency_hint': results[0]['currency'] if results else 'RM',
        'recent_sectors': recent_sectors,
        'last_sector': last_sector,
        'sector_counts': dict(Counter(r['sector'] for r in results)),
        'results': results,
        'max_score': MAX_SCORE,
        'max_long_term_score': MAX_LONG_TERM_SCORE,
    }


def pick_actionable(results):
    """The best fully-actionable stock this month, or None if there isn't one.

    Returns None when nothing clears the actionability bar (see is_actionable) —
    "NO PURCHASE THIS MONTH" is a valid, first-class answer. We deliberately do
    NOT fall back to the least-bad name: a screener forced to name a pick every
    month hands you a loser dressed up as a green light. Carrying cash forward is
    the whole point of a discipline tool.
    """
    actionable = [r for r in results if is_actionable(r)]
    return actionable[0] if actionable else None


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET FIT  — the headline feature: "spend RM X on the best buy"
# ─────────────────────────────────────────────────────────────────────────────
def _build_pick(r, lots, units, spend, fee, budget):
    """Assemble the buy dict returned by budget_fit (shared by .KL and US/HK)."""
    order_price = r.get('order_price', r['price'])
    entry_plan = limit_order_plan(r)
    suggested_limit = entry_plan.get('suggested_limit_price') if entry_plan else None
    suggested_spend = (suggested_limit * units) if suggested_limit else None
    suggested_fee = (
        bursa_fees(suggested_spend)
        if suggested_spend and r['ticker'].endswith('.KL')
        else 0.0
    )
    return {
        'ticker': r['ticker'], 'name': r['name'], 'sector': r['sector'],
        'currency': r['currency'], 'price': round(order_price, 4),
        'analysis_price': round(r['price'], 4),
        'price_source': r.get('price_source', 'history_close'),
        'lots': lots, 'units': units,
        'estimated_cost': round(spend, 2), 'estimated_fee': round(fee, 2),
        'fee_pct': round((fee / spend) * 100, 3) if spend else 0.0,
        'total_outlay': round(spend + fee, 2),
        'budget': budget, 'leftover': round(budget - spend - fee, 2),
        'entry_plan': entry_plan,
        'suggested_limit_price': round(suggested_limit, 4) if suggested_limit else None,
        'patient_limit_price': entry_plan.get('patient_limit_price') if entry_plan else None,
        'max_entry_price': entry_plan.get('max_entry_price') if entry_plan else None,
        'suggested_cost': round(suggested_spend, 2) if suggested_spend else None,
        'suggested_fee': round(suggested_fee, 2) if suggested_spend else None,
        'suggested_total_outlay': round(suggested_spend + suggested_fee, 2) if suggested_spend else None,
        'suggested_leftover': round(budget - suggested_spend - suggested_fee, 2) if suggested_spend else None,
        'score': r['score'], 'effective_score': r['effective_score'],
        'long_term_score': r.get('long_term_score'),
        'effective_long_term_score': r.get('effective_long_term_score'),
        'long_term_label': r.get('long_term_label'),
        'long_term_breakdown': r.get('long_term_breakdown', {}),
        'long_term_flags': r.get('long_term_flags', []),
        'timing_verdict': r['timing_verdict'], 'eh_label': r.get('eh_label', ''),
        'div_yield': r['div_yield'], 'pullback_pct': round(r['pullback_pct'], 1),
        'reason': r['reason'], 'is_actionable': is_actionable(r),
        'data_uncertain': r.get('data_uncertain', False),
        'data_warning': r.get('data_warning'),
    }


def budget_fit(results, budget, only_actionable=True, prefer=None):
    """Given a monthly budget, return the best stock and exact lots that fit.

    - Picks the highest-ranked stock whose minimum sensible position fits the budget.
    - For Bursa (.KL) stocks: buys as many 100-unit lots as the budget allows,
      keeping fee% under control (fees are tiny above ~1 lot anyway).
    - For US/HK: buys as many whole shares as fit.
    - `prefer`: optional ticker to force (e.g. user clicked a specific row).

    Returns either a buy dict, or a {'no_purchase': True, 'reason': ...} dict when
    the honest answer is "buy nothing this month" — distinguishing a *quality*
    veto (nothing clears the bar) from an *affordability* miss (qualifying names
    exist but none fit the budget). Returns None only when there's nothing to
    evaluate at all. Budget never influences ranking — it only sizes the position
    of the already-ranked pick.
    """
    if not results or not budget or budget <= 0:
        return None

    if prefer:
        candidates = [r for r in results if r['ticker'] == prefer] or results
    elif only_actionable:
        candidates = [r for r in results if is_actionable(r)]
        if not candidates:
            # Quality veto — not an affordability problem. Honour NO PURCHASE.
            return {
                'no_purchase': True, 'kind': 'quality', 'budget': budget,
                'reason': (
                    "No stock clears the actionability bar this month — every name is in a "
                    "death cross, flagged WAIT, showing deteriorating earnings, was bought "
                    "in the last 2 months, or has uncertain price data. Carry this month's "
                    "cash forward rather than forcing a least-bad buy."
                ),
            }
    else:
        candidates = results

    for r in candidates:
        ticker = r['ticker']
        price = r.get('order_price', r['price'])
        if not math.isfinite(price) or price <= 0:
            continue
        unit = units_per_lot(ticker)

        if ticker.endswith('.KL'):
            # How many 100-unit lots fit the budget (leave room for fees)?
            cost_per_lot = price * unit
            if not math.isfinite(cost_per_lot) or cost_per_lot <= 0:
                continue
            max_lots = int(budget // cost_per_lot)
            if max_lots < 1:
                continue  # even one lot too expensive — try next candidate
            spend = cost_per_lot * max_lots
            fee = bursa_fees(spend)
            # Trim a lot if fees+cost overshoot budget
            while max_lots > 1 and (spend + fee) > budget:
                max_lots -= 1
                spend = cost_per_lot * max_lots
                fee = bursa_fees(spend)
            return _build_pick(r, max_lots, max_lots * unit, spend, fee, budget)
        else:
            max_shares = int(budget // price)
            if max_shares < 1:
                continue
            spend = price * max_shares
            return _build_pick(r, max_shares, max_shares, spend, 0.0, budget)

    # Qualifying candidates existed, but none fit the budget → affordability miss.
    priced_candidates = [
        r for r in candidates
        if math.isfinite(r.get('order_price', r.get('price', 0))) and r.get('order_price', r.get('price', 0)) > 0
    ]
    if not priced_candidates:
        return {
            'no_purchase': True, 'kind': 'data', 'budget': budget,
            'reason': "No qualifying pick has a valid price right now. Refresh data and verify quotes manually.",
        }
    cheapest = min(priced_candidates, key=lambda r: r.get('order_price', r['price']) * units_per_lot(r['ticker']))
    unit_word = 'lot' if cheapest['ticker'].endswith('.KL') else 'share'
    min_cost = cheapest.get('order_price', cheapest['price']) * units_per_lot(cheapest['ticker'])
    return {
        'no_purchase': True, 'kind': 'budget', 'budget': budget,
        'cheapest': {'ticker': cheapest['ticker'], 'min_cost': round(min_cost, 2),
                     'currency': cheapest['currency']},
        'reason': (
            f"The cheapest qualifying pick, {cheapest['ticker']}, needs about "
            f"{cheapest['currency']} {min_cost:,.2f} for one {unit_word} — "
            f"{min_cost / budget:.1f}× your budget. Increase the budget or wait."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CACHE MAINTENANCE
# ─────────────────────────────────────────────────────────────────────────────
def refresh_cache(config=None, progress=None):
    """Force re-fetch every ticker's price history to disk (ignores TTL)."""
    if config is None:
        config = load_config() or {'tickers': [], 'period': '2y'}
    tickers_cfg = config.get('tickers', [])
    period = config.get('period', '2y')
    total = len(tickers_cfg)
    refreshed = 0
    for i, entry in enumerate(tickers_cfg):
        ticker = entry['ticker']
        if progress:
            progress(i, total, f"Refreshing {ticker}")
        df = fetch_data(ticker, period, use_cache=False)
        if not df.empty:
            refreshed += 1
    if progress:
        progress(total, total, "Cache refreshed")
    return {'refreshed': refreshed, 'total': total}


def cache_status(config=None):
    """Report age of each cached ticker so the UI can show freshness."""
    if config is None:
        config = load_config() or {'tickers': [], 'period': '2y'}
    period = config.get('period', '2y')
    out = []
    oldest = None
    for entry in config.get('tickers', []):
        ticker = entry['ticker']
        age = _cache_age_hours(_cache_path(ticker, period))
        out.append({'ticker': ticker, 'age_hours': round(age, 2) if age is not None else None})
        if age is not None:
            oldest = age if oldest is None else max(oldest, age)
    return {'tickers': out, 'oldest_hours': round(oldest, 2) if oldest is not None else None}
