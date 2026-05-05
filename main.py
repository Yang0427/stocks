import json
import math
import time
from collections import Counter
from datetime import datetime

import yfinance as yf
import pandas as pd
import ta

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR-AWARE PE THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_PE_THRESHOLDS = {
    'bank':      {'great': 10, 'ok': 14},
    'reit':      {'great': 14, 'ok': 18},
    'telco':     {'great': 18, 'ok': 25},
    'utilities': {'great': 18, 'ok': 25},
    'energy':    {'great': 15, 'ok': 22},
    'general':   {'great': 15, 'ok': 20},
    'consumer':  {'great': 18, 'ok': 25},
}

# Cyclical sectors where trailing yield is unreliable — yield score is capped
# and "falling knife" pullback detection applies (fix #5 and fix #4 scope).
CYCLICAL_SECTORS = {'consumer', 'energy', 'general'}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & PURCHASE LOG
# ─────────────────────────────────────────────────────────────────────────────
def load_config():
    try:
        with open('stocks.json', 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Error: 'stocks.json' not found.")
        return None

def load_purchase_log():
    try:
        with open('purchase_log.json', 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def load_sell_log():
    try:
        with open('sell_log.json', 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def build_ticker_sector_map(tickers_cfg):
    mapping = {}
    for entry in tickers_cfg:
        ticker = entry.get('ticker')
        sector = entry.get('sector')
        if ticker and sector:
            mapping[ticker] = sector
    return mapping

# yfinance returns sector strings like "Financial Services", "Real Estate", etc.
# Map them to the internal keys used in SECTOR_PE_THRESHOLDS.
_YFINANCE_SECTOR_MAP = {
    'financial services': 'bank',
    'financials':         'bank',
    'banking':            'bank',
    'real estate':        'reit',
    'communication services': 'telco',
    'telecommunications': 'telco',
    'utilities':          'utilities',
    'energy':             'energy',
}

def infer_sector_from_yfinance(ticker):
    """
    Fetch sector from yfinance and map to an internal sector name.
    Falls back to 'general' if unrecognised.
    """
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
        months = (now.year - last_dt.year) * 12 + (now.month - last_dt.month)
        return months
    except ValueError:
        return None

def units_per_lot(ticker):
    return 100 if ticker.endswith('.KL') else 1

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
                print(f"   ℹ️  {ticker}: sector not in stocks.json — inferred '{sector}' from yfinance")

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
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def fetch_data(ticker, period='2y'):
    print(f"Fetching {ticker}...", end=" ", flush=True)
    try:
        df = yf.download(ticker, period=period, interval='1d', progress=False)
        return df
    except Exception as e:
        print(f"Error: {e}")
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
    """Fetches name, yield, currency, fundamentals, dividend history, debt ratio, and earnings health."""
    try:
        t = yf.Ticker(ticker)
        info = t.info

        long_name  = info.get('longName')
        short_name = info.get('shortName')
        name = (
            f"{long_name} - {short_name}"
            if (long_name and short_name and long_name != short_name)
            else (long_name or short_name or ticker)
        )

        if   ticker.endswith('.KL'): currency = "RM"
        elif ticker.endswith('.HK'): currency = "HKD"
        else:                        currency = "USD"

        div_rate      = info.get('dividendRate', 0)
        current_price = info.get('currentPrice', 0) or info.get('previousClose', 0)
        if div_rate and current_price and current_price > 0:
            yield_pct = (div_rate / current_price) * 100
        else:
            raw = info.get('dividendYield', 0) or 0
            yield_pct = raw * 100 if raw < 1 else raw

        pe_ratio      = info.get('trailingPE') or info.get('forwardPE') or 0.0
        roe           = info.get('returnOnEquity', 0)
        roe           = float(roe) * 100 if roe else 0.0
        revenue       = info.get('totalRevenue', 0) or 0
        profit_margin = info.get('profitMargins', 0)
        profit_margin = float(profit_margin) * 100 if profit_margin else 0.0

        # Debt-to-equity ratio (reported as %, so 150 = 1.5x)
        debt_to_equity = info.get('debtToEquity') or 0.0

        # ── Earnings health signals ──────────────────────────────────────────
        trailing_pe = info.get('trailingPE') or 0.0
        forward_pe  = info.get('forwardPE')  or 0.0
        trailing_eps = info.get('trailingEps') or 0.0
        forward_eps  = info.get('forwardEps')  or 0.0
        earnings_growth_qoq = info.get('earningsQuarterlyGrowth') or None  # latest Q vs same Q last yr
        earnings_growth_ttm = info.get('earningsGrowth')           or None  # TTM YoY
        revenue_growth      = info.get('revenueGrowth')            or None  # TTM YoY
        payout_ratio        = info.get('payoutRatio')              or None  # 0–1 float

        # Multi-quarter earnings trend — last 4 quarters of YoY growth
        # Uses quarterly income_stmt (Net Income row) — avoids deprecated Ticker.earnings
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
                    vals = ni_row.dropna().values  # newest quarter first
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

        earnings_health = {
            'trailing_pe':          float(trailing_pe),
            'forward_pe':           float(forward_pe),
            'trailing_eps':         float(trailing_eps),
            'forward_eps':          float(forward_eps),
            'earnings_growth_qoq':  earnings_growth_qoq,
            'earnings_growth_ttm':  earnings_growth_ttm,
            'revenue_growth':       revenue_growth,
            'payout_ratio':         payout_ratio,
            'quarterly_neg_count':  quarterly_neg_count,
        }

        # ── Analyst consensus ────────────────────────────────────────────────
        target_mean_price  = info.get('targetMeanPrice')  or 0.0
        target_high_price  = info.get('targetHighPrice')  or 0.0
        target_low_price   = info.get('targetLowPrice')   or 0.0
        recommendation_key = info.get('recommendationKey') or ''   # 'buy','hold','sell' etc.
        num_analysts       = info.get('numberOfAnalystOpinions') or 0

        analyst_data = {
            'target_mean':      float(target_mean_price),
            'target_high':      float(target_high_price),
            'target_low':       float(target_low_price),
            'recommendation':   recommendation_key.lower(),
            'num_analysts':     int(num_analysts),
            'current_price':    float(current_price),
        }

        return (name, yield_pct, currency, float(pe_ratio), roe,
                fmt(revenue), profit_margin, div_years, div_growing,
                t, info, debt_to_equity, earnings_health, analyst_data)

    except Exception:
        return (ticker, 0.0, "N/A", 0.0, 0.0, "N/A", 0.0, 0, False, None, {}, 0.0, {}, {})

# ─────────────────────────────────────────────────────────────────────────────
# TIMING SIGNALS  (short-term only — intentionally separate from score)
#
#  Four checks:
#  1. Ex-dividend countdown — buy before ex-date to collect dividend
#  2. Earnings date warning — future results only (past dates ignored)
#  3. Short-term momentum   — 5-day and 10-day price change
#  4. RSI extremes          — overbought >70 or oversold <30
#
#  Verdict:
#   🟢 GOOD    — clean entry, buy full amount
#   🟡 CAUTION — one risk, consider buying half now
#   ⏳ WAIT    — multiple risks, wait a week
# ─────────────────────────────────────────────────────────────────────────────
def get_timing_signals(ticker_obj, info, df):
    green  = []
    yellow = []
    now    = pd.Timestamp.now(tz='UTC')

    # ── 1. EX-DIVIDEND DATE ─────────────────────────────────────────────────
    try:
        ex_ts = info.get('exDividendDate')
        if ex_ts:
            ex_date  = pd.Timestamp(ex_ts, unit='s', tz='UTC')
            days_gap = (ex_date - now).days
            ex_str   = ex_date.strftime('%d %b %Y')
            div_rate = info.get('dividendRate', 0) or 0
            per_pmt  = div_rate / 4 if div_rate else 0

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

    # ── 2. EARNINGS DATE (future only) ──────────────────────────────────────
    try:
        cal = ticker_obj.calendar
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

            # Only act on future earnings dates — ignore stale/past data
            if days_to_earn >= 0:
                e_str = e_date.strftime('%d %b %Y')
                if days_to_earn <= 7:
                    yellow.append(
                        f"⚡ Earnings in {days_to_earn} day(s) ({e_str}) "
                        f"— HIGH volatility risk, price can swing ±10%"
                    )
                elif days_to_earn <= 14:
                    yellow.append(
                        f"⚡ Earnings in {days_to_earn} days ({e_str}) "
                        f"— expect price movement after results"
                    )
                elif days_to_earn <= 45:
                    green.append(
                        f"📊 Earnings in {days_to_earn} days ({e_str}) "
                        f"— results coming, watch for upgrades"
                    )
    except Exception:
        pass

    # ── 3. SHORT-TERM MOMENTUM ──────────────────────────────────────────────
    try:
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
        else:
            close = df['Close']

        if len(close) >= 10:
            p_now   = float(close.iloc[-1])
            p_5d    = float(close.iloc[-5])
            p_10d   = float(close.iloc[-10])
            mom_5d  = ((p_now - p_5d)  / p_5d)  * 100
            mom_10d = ((p_now - p_10d) / p_10d) * 100

            if mom_5d <= -3:
                green.append(
                    f"📉 Down {abs(mom_5d):.1f}% in 5 days "
                    f"— short-term dip, price is lower than last week"
                )
            elif mom_5d >= 5:
                yellow.append(
                    f"📈 Up {mom_5d:.1f}% in 5 days "
                    f"— short-term spike, may pull back slightly"
                )
            else:
                green.append(f"➡️  Stable {mom_5d:+.1f}% over 5 days — no short-term spike")

            if mom_10d <= -5:
                green.append(
                    f"📉 Down {abs(mom_10d):.1f}% over 10 days "
                    f"— extended pullback, entry looks timely"
                )
            elif mom_10d >= 8:
                yellow.append(
                    f"📈 Up {mom_10d:.1f}% over 10 days "
                    f"— check if you're buying into a recent run-up"
                )
    except Exception:
        pass

    # ── 4. RSI EXTREMES ─────────────────────────────────────────────────────
    try:
        if isinstance(df.columns, pd.MultiIndex):
            close_s = df['Close'].iloc[:, 0]
        else:
            close_s = df['Close']

        if len(close_s) >= 14:
            rsi_series = ta.momentum.rsi(close_s, window=14)
            rsi_val = float(rsi_series.iloc[-1])
            if rsi_val >= 70:
                yellow.append(
                    f"🔴 RSI {rsi_val:.1f} — overbought territory, elevated pullback risk"
                )
            elif rsi_val <= 30:
                green.append(
                    f"🟢 RSI {rsi_val:.1f} — oversold, potential mean-reversion bounce"
                )
    except Exception:
        pass

    # ── OVERALL TIMING VERDICT ───────────────────────────────────────────────
    n_yellow = len(yellow)
    if n_yellow >= 2:
        verdict = "⏳ WAIT"
    elif n_yellow == 1:
        verdict = "🟡 CAUTION"
    else:
        verdict = "🟢 GOOD"

    return green, yellow, verdict

# ─────────────────────────────────────────────────────────────────────────────
# DEBT / LEVERAGE FLAG  (for banks and REITs)
# ─────────────────────────────────────────────────────────────────────────────
def debt_flag(sector, debt_to_equity):
    """
    Returns (flag_str, is_high_debt).
    debt_to_equity from yfinance is reported as a percentage (e.g. 150 = 1.5x).
    Banks are inherently highly leveraged — threshold is much higher.
    REITs typically run 0.5x–1.5x (50–150%).
    """
    if debt_to_equity <= 0:
        return "N/A", False

    ratio = debt_to_equity / 100  # convert to multiplier

    if sector == 'bank':
        # Banks operate at 8x–12x leverage normally; flag only extreme cases
        if ratio > 15:
            return f"⚠️  D/E {ratio:.1f}x — extremely high even for a bank", True
        elif ratio > 10:
            return f"🟡 D/E {ratio:.1f}x — high leverage, monitor capital ratios", False
        else:
            return f"✅ D/E {ratio:.1f}x — normal bank leverage", False
    elif sector == 'reit':
        if ratio > 1.5:
            return f"⚠️  D/E {ratio:.1f}x — high gearing for REIT, rate-sensitive", True
        elif ratio > 0.8:
            return f"🟡 D/E {ratio:.1f}x — moderate gearing, watch interest coverage", False
        else:
            return f"✅ D/E {ratio:.1f}x — conservative gearing", False
    else:
        if ratio > 2.0:
            return f"⚠️  D/E {ratio:.1f}x — high debt load", True
        elif ratio > 1.0:
            return f"🟡 D/E {ratio:.1f}x — moderate debt", False
        else:
            return f"✅ D/E {ratio:.1f}x — low debt", False

# ─────────────────────────────────────────────────────────────────────────────
# EARNINGS HEALTH  — detects deteriorating fundamentals the score engine misses
# ─────────────────────────────────────────────────────────────────────────────
def get_earnings_health_signals(earnings_health, sector='general'):
    """
    Returns (warnings: list[str], score_penalty: int, health_label: str).

    Five checks:
      A. Multi-quarter trend  — how many of last 4 quarters were YoY negative
      B. Latest quarter & TTM — single-quarter and full-year direction
      C. EPS revision direction — forward EPS vs trailing EPS (fix #2)
      D. Dividend sustainability — payout ratio, with tighter thresholds for cyclicals
      E. Forward vs trailing PE — but only penalise when EPS is also being revised down
    """
    warnings = []
    penalty  = 0
    is_cyclical = sector in CYCLICAL_SECTORS

    eq   = earnings_health.get('earnings_growth_qoq')    # float or None
    et   = earnings_health.get('earnings_growth_ttm')    # float or None
    rv   = earnings_health.get('revenue_growth')         # float or None
    pr   = earnings_health.get('payout_ratio')           # float 0-1 or None
    tpe  = earnings_health.get('trailing_pe',  0.0)
    fpe  = earnings_health.get('forward_pe',   0.0)
    teps = earnings_health.get('trailing_eps', 0.0)
    feps = earnings_health.get('forward_eps',  0.0)
    qneg = earnings_health.get('quarterly_neg_count', 0)  # 0-4 quarters negative YoY

    # ── A. Multi-quarter trend (fix #3) ─────────────────────────────────────
    # 2+ of last 4 quarters negative YoY = structural decline, not a blip
    if qneg >= 3:
        warnings.append(f"🔴 {qneg}/4 recent quarters below prior-year — sustained earnings decline")
        penalty += 20
    elif qneg == 2:
        warnings.append(f"🟠 {qneg}/4 recent quarters below prior-year — pattern of weakness")
        penalty += 10
    elif qneg == 1:
        warnings.append(f"🟡 1/4 recent quarters below prior-year — monitor next result")
        penalty += 3

    # ── B. Latest quarter & TTM ──────────────────────────────────────────────
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
            # For cyclicals: a single good quarter after a collapse is a bounce, not a recovery.
            # Only give the reward if the multi-quarter trend is also clean (qneg == 0).
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

    # ── C. EPS revision direction (fix #2) ──────────────────────────────────
    # Forward EPS < trailing EPS means analysts are cutting estimates.
    # This catches downward EPS revision cycles even when forward PE looks "cheap".
    if teps > 0 and feps > 0:
        eps_change = (feps - teps) / abs(teps)
        if eps_change <= -0.20:
            warnings.append(
                f"🔴 Forward EPS {feps:.3f} vs Trailing {teps:.3f} "
                f"— analysts cut estimates {abs(eps_change)*100:.0f}%, earnings declining"
            )
            penalty += 18
        elif eps_change <= -0.08:
            warnings.append(
                f"🟠 Forward EPS {feps:.3f} vs Trailing {teps:.3f} "
                f"— mild downward revision, earnings under pressure"
            )
            penalty += 8
        elif eps_change >= 0.10:
            warnings.append(
                f"🟢 Forward EPS {feps:.3f} vs Trailing {teps:.3f} "
                f"— analysts raising estimates, earnings momentum positive"
            )
            penalty -= 5

    # ── D. Dividend sustainability (fix #5 — tighter for cyclicals) ─────────
    if pr is not None and pr > 0:
        # Cyclical sectors get tighter thresholds: 60%/75%/90% instead of 75%/90%/100%
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
            warnings.append(
                f"🟠 Payout ratio {pr*100:.0f}% — nearly all earnings consumed, cut risk high"
            )
            penalty += 12
        elif pr > t1:
            warnings.append(
                f"🟡 Payout ratio {pr*100:.0f}% — stretched"
                + (" (tighter limit for cyclical sector)" if is_cyclical else ", limited buffer")
            )
            penalty += 5
        else:
            warnings.append(f"🟢 Payout ratio {pr*100:.0f}% — dividend well covered by earnings")

    # ── E. Forward vs trailing PE — only penalise when EPS also revised down ─
    # This prevents "fake good news" when fPE < tPE is caused by the stock
    # price falling (not by earnings growing). (fix #2 companion)
    if tpe > 0 and fpe > 0:
        pe_expansion = (fpe - tpe) / tpe
        eps_declining = (teps > 0 and feps > 0 and feps < teps)

        if pe_expansion >= 0.25:
            warnings.append(
                f"🔴 Forward PE {fpe:.1f}x vs Trailing {tpe:.1f}x "
                f"— analysts pricing in earnings decline"
            )
            penalty += 15
        elif pe_expansion >= 0.10:
            warnings.append(
                f"🟡 Forward PE {fpe:.1f}x vs Trailing {tpe:.1f}x "
                f"— mild earnings headwind expected"
            )
            penalty += 5
        elif pe_expansion <= -0.15:
            if eps_declining:
                # fPE looks better only because stock price dropped — not genuine growth
                warnings.append(
                    f"🟡 Forward PE {fpe:.1f}x vs Trailing {tpe:.1f}x "
                    f"— looks cheaper but EPS estimates were cut, not earnings growth"
                )
                penalty += 5
            else:
                warnings.append(
                    f"🟢 Forward PE {fpe:.1f}x vs Trailing {tpe:.1f}x "
                    f"— analysts expect {abs(pe_expansion)*100:.0f}% earnings growth"
                )
                penalty -= 5

    # ── Health label ─────────────────────────────────────────────────────────
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
# FEE CALCULATOR (BURSA / MALAYSIA ONLY)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_min_lots(price_per_unit, ticker):
    if not ticker.endswith('.KL'):
        return None
    for lots in range(1, 100):
        total_value  = price_per_unit * 100 * lots
        stamp_duty   = max(math.ceil(total_value / 1000), 1)
        clearing_fee = total_value * 0.0003
        platform_fee = 3.00
        total_fee    = platform_fee + stamp_duty + clearing_fee
        fee_pct      = (total_fee / total_value) * 100
        if fee_pct < 1.0:
            return {'lots': lots, 'cost': total_value, 'fee_pct': fee_pct}
    return {'lots': 1, 'cost': price_per_unit * 100, 'fee_pct': 100.0}

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_stock(df):
    if len(df) < 200:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['RSI']     = ta.momentum.rsi(df['Close'], window=14)
    df['SMA_50']  = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    latest   = df.iloc[-1]
    lookback = min(252, len(df))
    high_window = df['High'].iloc[-lookback:]
    high_idx    = high_window.idxmax()
    now_ts      = pd.Timestamp.now(tz=high_idx.tz if high_idx.tz else None)
    months_since_high = (
        (now_ts.year - high_idx.year) * 12 + (now_ts.month - high_idx.month)
    )
    return {
        'price':              float(latest['Close']),
        'rsi':                float(latest['RSI']),
        'sma50':              float(latest['SMA_50']),
        'sma200':             float(latest['SMA_200']),
        'volume_strong':      float(latest['Volume']) > float(latest['Vol_Avg']) * 1.5,
        'week52_high':        float(high_window.max()),
        'week52_low':         float(df['Low'].iloc[-lookback:].min()),
        'months_since_high':  months_since_high,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SAVINGS SCORE ENGINE  (max ~117 pts — entry can go negative for peak buys)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_savings_score(stock_data, sector, recent_sectors, months_since_last_buy):
    price       = stock_data['price']
    sma50       = stock_data['sma50']
    sma200      = stock_data['sma200']
    pe          = stock_data.get('pe_ratio', 0)
    roe         = stock_data.get('roe', 0)
    div_yield   = stock_data.get('div_yield', 0)
    week52_high = stock_data.get('week52_high', price)
    div_years   = stock_data.get('div_years', 0)
    div_growing = stock_data.get('div_growing', False)
    eh_penalty  = stock_data.get('eh_penalty', 0)

    score = 0
    bd    = {}

    rsi              = stock_data.get('rsi', 50)
    months_since_high = stock_data.get('months_since_high', 0)
    is_cyclical       = sector in CYCLICAL_SECTORS

    # ── Trend (15 pts) ───────────────────────────────────────────────────────
    trend_pts   = 15 if sma50 > sma200 else 0
    score      += trend_pts
    bd['trend'] = trend_pts

    # ── Entry / Pullback (max 30 pts, min -10 pts penalty) ──────────────────
    # Fix #4: if the 52-week high was set 9+ months ago AND the sector is
    # cyclical AND price is below SMA200, treat as a structural downtrend
    # (falling knife) and cap entry points at 12 regardless of pullback depth.
    pullback = ((week52_high - price) / week52_high * 100) if week52_high > 0 else 0
    falling_knife = (
        is_cyclical
        and months_since_high >= 9
        and price < sma200
    )

    # Tweak #3: widen gap between decent and deep pullbacks to reward real value entries
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
        entry_pts = min(entry_pts, 12)   # cap — deep pullback on a downtrend is not a bargain

    if rsi >= 65 and entry_pts > 0:
        entry_pts -= 5

    score      += entry_pts
    bd['entry'] = entry_pts

    # ── Valuation / PE (15 pts) ──────────────────────────────────────────────
    thresh = SECTOR_PE_THRESHOLDS.get(sector, SECTOR_PE_THRESHOLDS['general'])
    val_pts = 0 if (pe <= 0 or pe > 200) else 15 if pe < thresh['great'] else 8 if pe < thresh['ok'] else 2
    score          += val_pts
    bd['valuation'] = val_pts

    # ── Yield (max 20 pts for defensive, max 12 pts for cyclicals) ───────────
    # Fix #1 & #5: cyclical sector yields are unreliable (paid from reserves
    # during earnings downturns). Cap their yield score so a 7%+ trailing
    # yield on an auto distributor doesn't outscore a 6% bank yield.
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
        yield_pts = min(yield_pts, 12)   # hard cap for cyclicals

    # Tweak #2: yield trap penalty — high yield near peak is a trap, not a gift.
    # A 6%+ yield with <8% pullback means you're paying near-peak for income.
    yield_trap = 0
    if div_yield >= 6 and pullback < 8:
        yield_trap  = -5
        yield_pts   = max(0, yield_pts + yield_trap)

    score              += yield_pts
    bd['yield']         = yield_pts
    bd['yield_trap']    = yield_trap

    # ── ROE (10 pts) ─────────────────────────────────────────────────────────
    roe_pts   = 10 if roe > 12 else 5 if roe > 8 else 2 if roe > 0 else 0
    score     += roe_pts
    bd['roe'] = roe_pts

    # ── Analyst consensus bonus (max 10 pts) — tweak #1 ─────────────────────
    # Rewards stocks where a meaningful number of analysts see real upside.
    # Only fires when ≥5 analysts cover the stock, to avoid noise.
    analyst_data  = stock_data.get('analyst_data', {})
    target_mean   = analyst_data.get('target_mean', 0.0)
    curr_price_a  = analyst_data.get('current_price', price)
    rec_key       = analyst_data.get('recommendation', '')
    n_analysts    = analyst_data.get('num_analysts', 0)

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
            analyst_pts = -5   # analysts see downside — penalise

    # Boost if strong buy consensus, trim if hold/sell
    if analyst_pts > 0:
        if rec_key in ('strong_buy', 'buy'):
            analyst_pts = min(analyst_pts + 2, 10)
        elif rec_key in ('underperform', 'sell', 'strong_sell'):
            analyst_pts = max(analyst_pts - 4, 0)

    score             += analyst_pts
    bd['analyst']      = analyst_pts

    # ── Dividend history (15 pts) ────────────────────────────────────────────
    div_pts = 15 if div_years >= 5 else 10 if div_years >= 3 else 5 if div_years >= 1 else 0
    if div_growing and div_pts > 0:
        div_pts = min(div_pts + 3, 15)
    score          += div_pts
    bd['dividend']  = div_pts

    # ── Sector rotation — checks last 3 purchases, not just 1 ────────────────
    sector_count = recent_sectors.count(sector) if recent_sectors else 0
    if not recent_sectors:
        rot_pts = 4
    elif sector_count == 0:
        rot_pts = 8      # fresh sector → full bonus
    elif sector_count == 1:
        rot_pts = 3      # bought once in last 3 months → small credit
    else:
        rot_pts = 0      # bought 2+ times recently → no bonus
    score          += rot_pts
    bd['rotation']  = rot_pts

    # ── Freshness bonus — reward stocks not bought in a long time ─────────────
    if months_since_last_buy is None:
        fresh_pts = 4    # never bought → fresh pick
    elif months_since_last_buy >= 6:
        fresh_pts = 4    # dormant for 6+ months → accumulate again
    elif months_since_last_buy >= 3:
        fresh_pts = 2
    else:
        fresh_pts = 0    # bought very recently
    score           += fresh_pts
    bd['freshness']  = fresh_pts

    # ── Recent-buy penalty (fix #6) — don't double-down within 2 months ───────
    # Even if freshness=0, a stock bought last month can still rank #1 on raw
    # fundamentals. This hard-demotes it so the script actively diversifies.
    if months_since_last_buy is not None and months_since_last_buy < 2:
        recent_buy_pen   = 20
        score           -= recent_buy_pen
        bd['recent_buy'] = -recent_buy_pen
    else:
        bd['recent_buy'] = 0

    # ── Earnings health penalty — deduct for deteriorating fundamentals ───────
    score              -= eh_penalty
    bd['eh_penalty']    = -eh_penalty   # negative so it reads as a deduction in breakdown

    return round(score), bd

# ─────────────────────────────────────────────────────────────────────────────
# EFFECTIVE RANK  (score adjusted for actionability)
# ─────────────────────────────────────────────────────────────────────────────
def effective_score(item):
    """
    Demote stocks that cannot be acted on this month so the displayed order
    reflects what you should actually buy, not just the best fundamentals.
      - Death cross (not golden)  → heavily demoted
      - ⏳ WAIT timing            → moderately demoted
      - 🟡 CAUTION timing         → lightly demoted
    Raw score is preserved for display; only sorting is affected.
    """
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

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def trend_label(sma50, sma200, price):
    if sma50 > sma200 and price > sma200:
        return "🌟 GOLDEN + above SMA200"
    elif sma50 > sma200:
        return "🌤  GOLDEN (price still below SMA200)"
    else:
        return "💀 DEATH cross"

def pe_flag(pe):
    if pe <= 0:  return " ⚠️ N/A"
    if pe < 3:   return " ⚠️ Suspect data"
    if pe > 200: return " ⚠️ Extreme"
    return ""

def evaluate_accumulation_signal(current_price, week52_high, div_yield):
    """
    Long-term savings approach: no sell signals.
    Tells you whether to add more savings to this position this month.
    """
    pullback = ((week52_high - current_price) / week52_high * 100) if week52_high > 0 else 0

    if pullback >= 15:
        return ("🛒 ADD MORE", f"Price {pullback:.1f}% below 52-week high — great accumulation point")
    elif pullback >= 8:
        return ("💰 CONSIDER ADDING", f"Price {pullback:.1f}% below peak — decent entry for extra savings")
    elif pullback <= 3:
        return ("✅ HOLD & COLLECT", f"Near 52-week high ({pullback:.1f}% below) — sit tight, collect dividends")
    else:
        return ("✅ HOLD", f"Fair price zone ({pullback:.1f}% below 52-week high) — hold for income")


def project_annual_dividend(open_units, div_yield_pct, current_price):
    """Estimate annual dividend income from a position based on current yield."""
    annual_per_share = (div_yield_pct / 100) * current_price
    annual_total = annual_per_share * open_units
    return annual_total, annual_total / 12

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    if not config:
        return

    tickers_cfg       = config.get('tickers', [])
    period            = config.get('period', '2y')
    ticker_sector_map = build_ticker_sector_map(tickers_cfg)
    purchase_log      = load_purchase_log()
    sell_log          = load_sell_log()
    last_sector       = get_last_sector(purchase_log, ticker_sector_map)
    recent_sectors    = get_recent_sectors(purchase_log, ticker_sector_map, n=3)
    results           = []

    # Cache fetched data to avoid double API calls for holdings that are also on watchlist
    fetched_cache = {}

    print(f"\n🚀 SAVINGS SCOUT — fetching {len(tickers_cfg)} stocks...\n")

    for entry in tickers_cfg:
        ticker = entry['ticker']
        sector = entry.get('sector', 'general')

        (name, div_yield, currency, pe_ratio, roe,
         revenue, profit_margin, div_years, div_growing,
         ticker_obj, info, debt_to_equity, earnings_health, analyst_data) = get_stock_info(ticker)

        df = fetch_data(ticker, period)
        fetched_cache[ticker] = (df, div_yield, name)

        if df.empty:
            print()
            continue

        stats = analyze_stock(df)
        if not stats:
            print(f"⚠️  {ticker}: not enough data\n")
            continue

        t_green, t_yellow, t_verdict = get_timing_signals(ticker_obj, info, df)
        months_since = get_months_since_last_buy(purchase_log, ticker)
        debt_str, is_high_debt = debt_flag(sector, debt_to_equity)
        eh_warnings, eh_penalty, eh_label = get_earnings_health_signals(earnings_health, sector)

        stats.update({
            'ticker':           ticker,
            'sector':           sector,
            'name':             name,
            'currency':         currency,
            'div_yield':        div_yield,
            'pe_ratio':         pe_ratio,
            'roe':              roe,
            'revenue':          revenue,
            'profit_margin':    profit_margin,
            'div_years':        div_years,
            'div_growing':      div_growing,
            'debt_to_equity':   debt_to_equity,
            'debt_str':         debt_str,
            'is_high_debt':     is_high_debt,
            'smart_lots':       calculate_min_lots(stats['price'], ticker),
            'timing_green':     t_green,
            'timing_yellow':    t_yellow,
            'timing_verdict':   t_verdict,
            'months_since_buy': months_since,
            'eh_warnings':      eh_warnings,
            'eh_penalty':       eh_penalty,
            'eh_label':         eh_label,
            'months_since_high': stats.get('months_since_high', 0),
            'analyst_data':     analyst_data,
        })

        score, breakdown = calculate_savings_score(stats, sector, recent_sectors, months_since)
        stats['score']     = score
        stats['breakdown'] = breakdown
        results.append(stats)
        print()
        time.sleep(0.5)

    # Sort by effective (actionability-adjusted) score, but keep raw score for display
    results.sort(key=effective_score, reverse=True)

    # ── REPORT ───────────────────────────────────────────────────────────────
    W = 80
    print("\n" + "=" * W)
    print("🏆  SAVINGS SCOUT — MONTHLY PICK REPORT")
    print("=" * W)

    if recent_sectors:
        print(f"\n📌 Recent sectors purchased : {' → '.join(s.upper() for s in recent_sectors)}")
    if purchase_log:
        last = max(purchase_log, key=lambda e: e.get('month', ''))
        print(f"   Last stock bought        : {last.get('ticker')} @ "
              f"{last.get('currency','RM')} {last.get('price','?')} ({last.get('month','?')})")
    print()

    medals = {0: "🥇 TOP PICK", 1: "🥈 RUNNER-UP", 2: "🥉 THIRD"}

    for rank, item in enumerate(results):
        price    = item['price']
        sma50    = item['sma50']
        sma200   = item['sma200']
        curr     = item['currency']
        score    = item['score']
        bd       = item['breakdown']
        pullback = (item['week52_high'] - price) / item['week52_high'] * 100 if item['week52_high'] > 0 else 0
        eff_s    = effective_score(item)
        is_golden = sma50 > sma200
        tv        = item['timing_verdict']

        if rank in medals:
            print("─" * W)
            actionable = is_golden and tv != "⏳ WAIT"
            tag = "" if actionable else "  ⚠️  NOT ACTIONABLE THIS MONTH"
            print(f"  {medals[rank]}  (Score: {score}  |  Effective: {eff_s}){tag}")
        elif rank == 3:
            print("─" * W)
            print("  📋  FULL WATCHLIST")

        bar_fill = max(0, min(20, round(score / 6)))   # max ~117 → 20 bars
        bar      = "█" * bar_fill + "░" * (20 - bar_fill)

        if   item['div_years'] >= 5: div_str = f"✅ {item['div_years']} yrs"
        elif item['div_years'] >= 1: div_str = f"⚠️  {item['div_years']} yr(s) only"
        else:                         div_str = "❌ No history"
        if item['div_growing'] and item['div_years'] >= 1:
            div_str += " 📈 growing"

        roe_display = f"{item['roe']:.2f}%" if item['roe'] > 0 else "N/A (missing)"

        months_tag = ""
        if item['months_since_buy'] is not None:
            months_tag = f"  |  Last bought: {item['months_since_buy']}m ago"
        else:
            months_tag = "  |  Never bought"

        print(f"\n  {item['name']} ({item['ticker']})  [{item['sector'].upper()}]{months_tag}")
        print(f"  [{bar}] Score: {score}  (effective {eff_s})")
        print(f"  Price:    {curr} {price:.3f}  |  52w High: {curr} {item['week52_high']:.3f}  |  Pullback: {pullback:.1f}%")
        print(f"  Yield:    {item['div_yield']:.2f}%  |  RSI: {item['rsi']:.1f} (info only)  |  Vol: {'🔥 HIGH' if item['volume_strong'] else 'Normal'}")
        print(f"  PE:       {item['pe_ratio']:.2f}{pe_flag(item['pe_ratio'])}  |  ROE: {roe_display}  |  Margin: {item['profit_margin']:.2f}%")
        print(f"  Revenue:  {item['revenue']} (TTM)  |  Sector PE bands: "
              f"Great <{SECTOR_PE_THRESHOLDS.get(item['sector'],{}).get('great','?')}  "
              f"OK <{SECTOR_PE_THRESHOLDS.get(item['sector'],{}).get('ok','?')}")
        print(f"  Debt:     {item['debt_str']}")
        print(f"  Trend:    {trend_label(sma50, sma200, price)}")
        print(f"            SMA50: {sma50:.3f}  |  SMA200: {sma200:.3f}")
        print(f"  Dividend: {div_str}")
        # ── Analyst line ──────────────────────────────────────────────────────
        ad           = item.get('analyst_data', {})
        tgt          = ad.get('target_mean', 0.0)
        tgt_hi       = ad.get('target_high', 0.0)
        n_an         = ad.get('num_analysts', 0)
        rec          = ad.get('recommendation', '')
        curr_price_a = ad.get('current_price', price)
        if tgt > 0 and curr_price_a > 0:
            upside_pct   = (tgt - curr_price_a) / curr_price_a * 100
            rec_str      = rec.replace('_', ' ').upper() if rec else 'N/A'
            analyst_line = (
                f"  Analysts: {n_an} analysts  |  Target {curr} {tgt:.3f}"
                f"  ({upside_pct:+.1f}%)  |  High {curr} {tgt_hi:.3f}"
                f"  |  Consensus: {rec_str}"
            )
        else:
            analyst_line = "  Analysts: No coverage data"
        print(analyst_line)

        falling_knife_tag = ""
        if (sector in CYCLICAL_SECTORS
                and item.get('months_since_high', 0) >= 9
                and price < item['sma200']):
            falling_knife_tag = "  ⚠️ FALLING KNIFE"
        print(f"  Score:    Trend={bd['trend']} | Entry={bd['entry']}{falling_knife_tag} | "
              f"PE={bd['valuation']} | Yield={bd['yield']}(trap={bd.get('yield_trap',0)}) | "
              f"ROE={bd['roe']} | Div={bd['dividend']} | Rotation={bd['rotation']} | "
              f"Fresh={bd['freshness']} | Analyst={bd.get('analyst',0)} | "
              f"RecentBuy={bd.get('recent_buy', 0)} | EarningsHealth={bd.get('eh_penalty', 0)}")

        # ── EARNINGS HEALTH BLOCK ─────────────────────────────────────────────
        eh_label = item.get('eh_label', '🟢 STABLE')
        eh_warn  = item.get('eh_warnings', [])
        print(f"  ┌─ EARNINGS HEALTH: {eh_label} " + "─" * max(0, 46 - len(eh_label)))
        if eh_warn:
            for w in eh_warn:
                print(f"  │  {w}")
        else:
            print(f"  │  (No earnings trend data available)")
        if item.get('eh_penalty', 0) > 0:
            print(f"  └─ ⚠️  Score penalised -{item['eh_penalty']} pts for fundamental deterioration")
        else:
            print(f"  └─ Fundamentals look clean")

        # ── TIMING BLOCK ─────────────────────────────────────────────────────
        print(f"  ┌─ TIMING: {tv} " + "─" * max(0, 56 - len(tv)))
        for g in item['timing_green']:
            print(f"  │  ✔ {g}")
        for y in item['timing_yellow']:
            print(f"  │  ✘ {y}")
        if not item['timing_green'] and not item['timing_yellow']:
            print(f"  │  (No timing data available)")

        lots = item['smart_lots']

        if not is_golden:
            print(f"  └─ ⛔ SKIP: Death cross — wait for trend reversal first")
        elif tv == "⏳ WAIT":
            print(f"  └─ ⏳ WAIT: Multiple short-term risks — revisit in 1–2 weeks")
        elif tv == "🟡 CAUTION":
            half = max(1, (lots['lots'] // 2)) if lots else 1
            half_cost = half * 100 * price
            print(f"  └─ 🟡 BUY HALF NOW: {half} lot(s) → {curr} {half_cost:.2f}  |  Hold remaining for after risk clears")
        else:
            if lots:
                print(f"  └─ 🟢 BUY FULL: {lots['lots']} lot(s) → {curr} {lots['cost']:.2f}  (fee: {lots['fee_pct']:.2f}%)")
            else:
                print(f"  └─ 🟢 BUY FULL: 1 unit → {curr} {price:.3f}")

        print()

    # ── SECTOR SUMMARY ───────────────────────────────────────────────────────
    print("─" * W)
    print("📊  YOUR WATCHLIST BY SECTOR")
    sector_map = Counter(r['sector'] for r in results)
    for sec, count in sorted(sector_map.items()):
        flags = []
        if sec == last_sector:
            flags.append("← bought last month")
        if recent_sectors.count(sec) >= 2:
            flags.append("⚠️ heavy recent concentration")
        flag_str = "  " + " | ".join(flags) if flags else ""
        print(f"   {sec.upper():15}  {count} stock(s){flag_str}")

    # ── HOLDINGS REVIEW (DIVIDEND INCOME & ACCUMULATION) ─────────────────────
    print("\n" + "─" * W)
    print("📊  HOLDINGS — DIVIDEND INCOME & ACCUMULATION")
    open_positions, realized_by_ticker, txn_errors = build_open_positions(
        purchase_log, sell_log, ticker_sector_map
    )
    if txn_errors:
        print("   ⚠️ Transaction log issues detected:")
        for err in txn_errors:
            print(f"   - {err}")

    if not purchase_log:
        print("   No holdings in purchase_log.json")
    elif not sell_log:
        print("   ℹ️ sell_log.json not found or empty — treating all buys as open positions")

    if not open_positions:
        if purchase_log:
            print("   No open positions after reconciling buys/sells")
    else:
        total_annual_div   = 0.0
        portfolio_currency = "RM"
        for pos in open_positions:
            ticker     = pos['ticker']
            buy_price  = pos['avg_buy_price']
            curr       = pos['currency']
            open_lots  = pos['open_lots']
            open_units = pos['open_units']
            month      = pos['first_month']
            portfolio_currency = curr

            # Reuse cached data if we already fetched this ticker in the main loop
            if ticker in fetched_cache:
                df_h, div_yield_h, stock_name = fetched_cache[ticker]
            else:
                stock_name, div_yield_h, *_ = get_stock_info(ticker)
                df_h = fetch_data(ticker, period)

            if df_h.empty:
                print(f"   ❌ {ticker}: No market data available")
                continue

            stats_h = analyze_stock(df_h)
            if not stats_h:
                print(f"   ❌ {ticker}: Not enough data")
                continue

            current_price    = stats_h['price']
            week52_high      = stats_h['week52_high']
            pnl_pct          = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else float('nan')
            unrealized_total = (current_price - buy_price) * open_units
            realized_total   = pos['realized_pnl']['value']

            annual_div, monthly_div = project_annual_dividend(open_units, div_yield_h, current_price)
            total_annual_div += annual_div
            verdict, reason = evaluate_accumulation_signal(current_price, week52_high, div_yield_h)
            sign = "+" if pnl_pct >= 0 else ""

            print(f"\n   {stock_name} ({ticker})  [{pos.get('sector', '').upper()}]  (since {month})")
            print(f"   Holding:  {open_lots} lot(s) / {open_units} units")
            print(f"   Avg Buy:  {curr} {buy_price:.3f}  |  Now: {curr} {current_price:.3f}  |  P/L: {sign}{pnl_pct:.2f}%")
            print(f"   Capital:  Unrealized {curr} {unrealized_total:+.2f}  |  Realized {curr} {realized_total:+.2f}")
            print(f"   Dividend: {div_yield_h:.2f}% yield  →  {curr} {annual_div:.2f}/yr  (~{curr} {monthly_div:.2f}/mth)")
            print(f"   Verdict:  {verdict} — {reason}")

        print(f"\n   {'─' * 60}")
        print(f"   💰 ESTIMATED ANNUAL DIVIDEND INCOME : {portfolio_currency} {total_annual_div:.2f}")
        print(f"      Monthly average                  : {portfolio_currency} {total_annual_div / 12:.2f}")
        print(f"      Reinvest dividends to compound your savings over time!")

    closed_realized = [
        (ticker, data['value'], data['currency'])
        for ticker, data in realized_by_ticker.items()
        if ticker not in {p['ticker'] for p in open_positions}
    ]
    if closed_realized:
        print("\n   Rebalanced / closed positions:")
        for ticker, value, currency in closed_realized:
            print(f"   - {ticker}: Realized P/L {currency} {value:+.2f}")

    # ── LOG HELPER ───────────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("📝  TO LOG THIS MONTH'S PURCHASE")
    print("   Copy the line below into purchase_log.json (inside the [ ] array):\n")

    # Suggest the first fully actionable stock:
    # golden cross + not WAIT + healthy earnings + not bought within 2 months
    actionable = [
        r for r in results
        if r['sma50'] > r['sma200']
        and r['timing_verdict'] != "⏳ WAIT"
        and r.get('eh_label', '') not in ('🔴 DETERIORATING',)
        and (r.get('months_since_buy') is None or r['months_since_buy'] >= 2)
    ]
    top = actionable[0] if actionable else (results[0] if results else None)

    if top:
        lots = top['smart_lots']
        log_entry = {
            "month":    datetime.now().strftime("%Y-%m"),
            "ticker":   top['ticker'],
            "price":    round(top['price'], 3),
            "currency": top['currency'],
            "lots":     lots['lots'] if lots else 1,
        }
        print(f"   {json.dumps(log_entry)}")
        if top != results[0]:
            print(f"   (Note: {results[0]['ticker']} ranks highest on score but skipped — "
                  f"earnings health: {results[0].get('eh_label','?')})")

    # ── FINAL VERDICT ────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("🎯  THIS MONTH'S BEST BUY — FINAL VERDICT")
    print("=" * W)
    if top:
        curr  = top['currency']
        lots  = top['smart_lots']
        tv    = top['timing_verdict']
        score = top['score']
        eff   = effective_score(top)
        eh    = top.get('eh_label', '🟢 STABLE')
        pr    = top.get('eh_warnings', [])

        print(f"\n  ✅ {top['name']} ({top['ticker']})  [{top['sector'].upper()}]")
        print(f"     Score {score}  |  Effective {eff}  |  Earnings: {eh}")
        print(f"     Price {curr} {top['price']:.3f}  |  Yield {top['div_yield']:.2f}%  |  PE {top['pe_ratio']:.2f}")
        print(f"     Timing: {tv}")

        if lots:
            if tv == "🟡 CAUTION":
                half      = max(1, lots['lots'] // 2)
                half_cost = half * 100 * top['price']
                print(f"\n  👉 BUY {half} lot(s) now → {curr} {half_cost:.2f}  (half position, caution flag active)")
            else:
                print(f"\n  👉 BUY {lots['lots']} lot(s) → {curr} {lots['cost']:.2f}  (fee: {lots['fee_pct']:.2f}%)")
        else:
            print(f"\n  👉 BUY 1 unit → {curr} {top['price']:.3f}")

        # Warn if this pick still has yellow earnings signals
        eh_yellow = [w for w in pr if w.startswith('🟡') or w.startswith('🟠')]
        if eh_yellow:
            print(f"\n  ⚠️  Watch points:")
            for w in eh_yellow:
                print(f"     {w}")
    else:
        print("\n  ❌ No fully actionable picks this month — all either death cross, ⏳ WAIT, or deteriorating earnings.")
        print("     Consider waiting or parking savings in a money market fund.")

    print("\n✅ Done. Run again next payday.\n")

if __name__ == "__main__":
    main()
