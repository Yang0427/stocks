import json
import math
import time
from datetime import datetime

import yfinance as yf
import pandas as pd
import ta

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR-AWARE PE THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_PE_THRESHOLDS = {
    'bank':         {'great': 10, 'ok': 14},
    'reit':         {'great': 14, 'ok': 18},
    'consumer':     {'great': 15, 'ok': 22},
    'telco':        {'great': 18, 'ok': 25},
    'utilities':    {'great': 18, 'ok': 25},
    'property':     {'great': 10, 'ok': 15},
    'industrial':   {'great': 15, 'ok': 22},
    'tech':         {'great': 25, 'ok': 35},
    'auto':         {'great': 10, 'ok': 15},
    'retail':       {'great': 20, 'ok': 30},
    'conglomerate': {'great': 12, 'ok': 18},
    'general':      {'great': 15, 'ok': 20},
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & PURCHASE LOG
# ─────────────────────────────────────────────────────────────────────────────
def load_config():
    try:
        with open('stocks.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Error: 'stocks.json' not found.")
        return None

def load_purchase_log():
    try:
        with open('purchase_log.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def load_sell_log():
    try:
        with open('sell_log.json', 'r') as f:
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

def get_last_sector(log, ticker_sector_map):
    if not log:
        return None
    for entry in reversed(log):
        ticker = entry.get('ticker')
        if entry.get('sector'):
            return entry.get('sector')
        if ticker in ticker_sector_map:
            return ticker_sector_map[ticker]
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
                errors.append(f"{ticker}: Sector missing in purchase_log and not found in stocks.json")
                continue

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
            errors.append(f"{ticker}: Sell exists but no matching buys available")
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
    """Fetches name, yield, currency, fundamentals, dividend history."""
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

        def fmt(v):
            if v >= 1e9: return f"{v/1e9:.2f}B"
            if v >= 1e6: return f"{v/1e6:.2f}M"
            return str(v)

        div_years, div_growing = check_dividend_consistency(t)

        # Return ticker_obj and info for timing signals
        return (name, yield_pct, currency, float(pe_ratio), roe,
                fmt(revenue), profit_margin, div_years, div_growing, t, info)

    except Exception:
        return (ticker, 0.0, "N/A", 0.0, 0.0, "N/A", 0.0, 0, False, None, {})

# ─────────────────────────────────────────────────────────────────────────────
# TIMING SIGNALS  (short-term only — intentionally separate from score)
#
#  Three checks:
#  1. Ex-dividend countdown — buy before ex-date to collect dividend
#  2. Earnings date warning — results within 14 days = volatility risk
#  3. Short-term momentum   — 5-day and 10-day price change
#
#  Verdict:
#   🟢 GOOD    — clean entry, buy full amount
#   🟡 CAUTION — one risk, consider buying half now
#   ⏳ WAIT    — multiple risks, wait a week
# ─────────────────────────────────────────────────────────────────────────────
def get_timing_signals(ticker_obj, info, df):
    green  = []   # tailwinds / positive timing
    yellow = []   # risks / cautions
    now    = pd.Timestamp.now(tz='UTC')

    # ── 1. EX-DIVIDEND DATE ─────────────────────────────────────────────────
    try:
        ex_ts = info.get('exDividendDate')
        if ex_ts:
            ex_date  = pd.Timestamp(ex_ts, unit='s', tz='UTC')
            days_gap = (ex_date - now).days
            ex_str   = ex_date.strftime('%d %b %Y')
            div_rate = info.get('dividendRate', 0) or 0
            per_pmt  = div_rate / 4 if div_rate else 0   # estimate per quarter

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

    # ── 2. EARNINGS DATE ────────────────────────────────────────────────────
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
            e_str = e_date.strftime('%d %b %Y')

            if 0 <= days_to_earn <= 7:
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
    return {
        'price':         float(latest['Close']),
        'rsi':           float(latest['RSI']),
        'sma50':         float(latest['SMA_50']),
        'sma200':        float(latest['SMA_200']),
        'volume_strong': float(latest['Volume']) > float(latest['Vol_Avg']) * 1.5,
        'week52_high':   float(df['High'].iloc[-lookback:].max()),
        'week52_low':    float(df['Low'].iloc[-lookback:].min()),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SAVINGS SCORE ENGINE  (max 100 pts — long-term fundamentals only)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_savings_score(stock_data, sector, last_sector):
    price       = stock_data['price']
    sma50       = stock_data['sma50']
    sma200      = stock_data['sma200']
    pe          = stock_data.get('pe_ratio', 0)
    roe         = stock_data.get('roe', 0)
    div_yield   = stock_data.get('div_yield', 0)
    week52_high = stock_data.get('week52_high', price)
    div_years   = stock_data.get('div_years', 0)
    div_growing = stock_data.get('div_growing', False)

    score = 0
    bd    = {}

    trend_pts   = 15 if sma50 > sma200 else 0
    score      += trend_pts
    bd['trend'] = trend_pts

    pullback    = ((week52_high - price) / week52_high * 100) if week52_high > 0 else 0
    entry_pts   = round(min(pullback * 0.8, 20))
    score      += entry_pts
    bd['entry'] = entry_pts

    thresh = SECTOR_PE_THRESHOLDS.get(sector, SECTOR_PE_THRESHOLDS['general'])
    val_pts = 0 if (pe <= 0 or pe > 200) else 15 if pe < thresh['great'] else 8 if pe < thresh['ok'] else 2
    score          += val_pts
    bd['valuation'] = val_pts

    yield_pts   = round(min(div_yield * 2, 15))
    score      += yield_pts
    bd['yield'] = yield_pts

    roe_pts   = 10 if roe > 12 else 5 if roe > 8 else 2 if roe > 0 else 0
    score     += roe_pts
    bd['roe'] = roe_pts

    div_pts = 15 if div_years >= 5 else 10 if div_years >= 3 else 5 if div_years >= 1 else 0
    if div_growing and div_pts > 0:
        div_pts = min(div_pts + 3, 15)
    score          += div_pts
    bd['dividend']  = div_pts

    rot_pts = 5 if last_sector is None else 10 if sector != last_sector else 0
    score          += rot_pts
    bd['rotation']  = rot_pts

    return round(score), bd

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

def evaluate_holding_signal(buy_price, current_price, rsi):
    if buy_price <= 0:
        return ("❌ ERROR", "Invalid buy price in purchase_log.json")
    if pd.isna(rsi):
        return ("❌ ERROR", "RSI unavailable; cannot produce sell verdict")

    gain_pct = ((current_price - buy_price) / buy_price) * 100
    take_profit_price = buy_price * 1.15
    trim_price = buy_price * 1.10
    cut_loss_price = buy_price * 0.92

    if current_price <= cut_loss_price:
        return ("🛑 CUT LOSS", f"Price is at/below 8% stop ({cut_loss_price:.3f})")
    if current_price >= take_profit_price and rsi > 70:
        return ("🔴 TAKE PROFIT", f"Gain {gain_pct:.1f}% and RSI {rsi:.1f} > 70")
    if current_price >= trim_price and 65 <= rsi <= 70:
        return ("🟡 TRIM", f"Gain {gain_pct:.1f}% with RSI {rsi:.1f} in 65–70 zone")
    if rsi > 70 and current_price < trim_price:
        return ("🟠 WATCH", f"RSI {rsi:.1f} is overbought; gain not yet 10%")
    return ("✅ HOLD", f"Trend condition healthy for now (RSI {rsi:.1f})")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    if not config:
        return

    tickers_cfg  = config.get('tickers', [])
    period       = config.get('period', '2y')
    ticker_sector_map = build_ticker_sector_map(tickers_cfg)
    purchase_log = load_purchase_log()
    sell_log     = load_sell_log()
    last_sector  = get_last_sector(purchase_log, ticker_sector_map)
    results      = []

    print(f"\n🚀 SAVINGS SCOUT — fetching {len(tickers_cfg)} stocks...\n")

    for entry in tickers_cfg:
        ticker = entry['ticker']
        sector = entry.get('sector', 'general')

        (name, div_yield, currency, pe_ratio, roe,
         revenue, profit_margin, div_years, div_growing,
         ticker_obj, info) = get_stock_info(ticker)

        df = fetch_data(ticker, period)
        if df.empty:
            print()
            continue

        stats = analyze_stock(df)
        if not stats:
            print(f"⚠️  {ticker}: not enough data\n")
            continue

        t_green, t_yellow, t_verdict = get_timing_signals(ticker_obj, info, df)

        stats.update({
            'ticker':         ticker,
            'sector':         sector,
            'name':           name,
            'currency':       currency,
            'div_yield':      div_yield,
            'pe_ratio':       pe_ratio,
            'roe':            roe,
            'revenue':        revenue,
            'profit_margin':  profit_margin,
            'div_years':      div_years,
            'div_growing':    div_growing,
            'smart_lots':     calculate_min_lots(stats['price'], ticker),
            'timing_green':   t_green,
            'timing_yellow':  t_yellow,
            'timing_verdict': t_verdict,
        })

        score, breakdown = calculate_savings_score(stats, sector, last_sector)
        stats['score']     = score
        stats['breakdown'] = breakdown
        results.append(stats)
        print()
        time.sleep(0.5)

    results.sort(key=lambda x: x['score'], reverse=True)

    # ── REPORT ───────────────────────────────────────────────────────────────
    W = 80
    print("\n" + "=" * W)
    print("🏆  SAVINGS SCOUT — MONTHLY PICK REPORT")
    print("=" * W)

    if last_sector:
        print(f"\n📌 Last purchase sector : {last_sector.upper()}")
    if purchase_log:
        last = purchase_log[-1]
        print(f"   Last stock bought    : {last.get('ticker')} @ "
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

        if rank in medals:
            print("─" * W)
            print(f"  {medals[rank]}  (Score: {score}/100)")
        elif rank == 3:
            print("─" * W)
            print("  📋  FULL WATCHLIST")

        filled = round(score / 5)
        bar    = "█" * filled + "░" * (20 - filled)

        if   item['div_years'] >= 5: div_str = f"✅ {item['div_years']} yrs"
        elif item['div_years'] >= 1: div_str = f"⚠️  {item['div_years']} yr(s) only"
        else:                         div_str = "❌ No history"
        if item['div_growing'] and item['div_years'] >= 1:
            div_str += " 📈 growing"

        roe_display = f"{item['roe']:.2f}%" if item['roe'] > 0 else "N/A (missing)"

        print(f"\n  {item['name']} ({item['ticker']})  [{item['sector'].upper()}]")
        print(f"  [{bar}] {score}/100")
        print(f"  Price:    {curr} {price:.3f}  |  52w High: {curr} {item['week52_high']:.3f}  |  Pullback: {pullback:.1f}%")
        print(f"  Yield:    {item['div_yield']:.2f}%  |  RSI: {item['rsi']:.1f} (info only)  |  Vol: {'🔥 HIGH' if item['volume_strong'] else 'Normal'}")
        print(f"  PE:       {item['pe_ratio']:.2f}{pe_flag(item['pe_ratio'])}  |  ROE: {roe_display}  |  Margin: {item['profit_margin']:.2f}%")
        print(f"  Revenue:  {item['revenue']} (TTM)  |  Sector PE bands: "
              f"Great <{SECTOR_PE_THRESHOLDS.get(item['sector'],{}).get('great','?')}  "
              f"OK <{SECTOR_PE_THRESHOLDS.get(item['sector'],{}).get('ok','?')}")
        print(f"  Trend:    {trend_label(sma50, sma200, price)}")
        print(f"            SMA50: {sma50:.3f}  |  SMA200: {sma200:.3f}")
        print(f"  Dividend: {div_str}")
        print(f"  Score:    Trend={bd['trend']} | Entry={bd['entry']} | PE={bd['valuation']} | "
              f"Yield={bd['yield']} | ROE={bd['roe']} | Div={bd['dividend']} | Rotation={bd['rotation']}")

        # ── TIMING BLOCK ─────────────────────────────────────────────────────
        tv = item['timing_verdict']
        print(f"  ┌─ TIMING: {tv} " + "─" * max(0, 56 - len(tv)))
        for g in item['timing_green']:
            print(f"  │  ✔ {g}")
        for y in item['timing_yellow']:
            print(f"  │  ✘ {y}")
        if not item['timing_green'] and not item['timing_yellow']:
            print(f"  │  (No timing data available)")

        is_golden = sma50 > sma200
        lots      = item['smart_lots']

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
    from collections import Counter
    sector_map = Counter(r['sector'] for r in results)
    for sec, count in sorted(sector_map.items()):
        flag = "  ← bought last month" if sec == last_sector else ""
        print(f"   {sec.upper():15}  {count} stock(s){flag}")

    # ── HOLDINGS REVIEW (SELL SIGNALS) ───────────────────────────────────────
    print("\n" + "─" * W)
    print("📈  HOLDINGS REVIEW — SELL SIGNALS")
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
        print("   ℹ️ sell_log.json not found or empty: treating all buys as open positions")

    if not open_positions:
        if purchase_log:
            print("   No open positions after reconciling buys/sells")
    else:
        for pos in open_positions:
            ticker = pos['ticker']
            buy_price = pos['avg_buy_price']
            curr = pos['currency']
            open_lots = pos['open_lots']
            open_units = pos['open_units']
            month = pos['first_month']
            (stock_name, _, _, _, _, _, _, _, _, _, _) = get_stock_info(ticker)

            df_h = fetch_data(ticker, period)
            if df_h.empty:
                print(f"   ❌ {ticker}: No market data, cannot evaluate sell signal")
                continue

            stats_h = analyze_stock(df_h)
            if not stats_h:
                print(f"   ❌ {ticker}: Not enough data to calculate RSI/SMA")
                continue

            current_price = stats_h['price']
            rsi = stats_h['rsi']
            pnl_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else float('nan')
            pnl_per_share = current_price - buy_price
            unrealized_total = pnl_per_share * open_units
            realized_total = pos['realized_pnl']['value']

            verdict, reason = evaluate_holding_signal(buy_price, current_price, rsi)
            sign = "+" if pnl_pct >= 0 else ""

            print(f"\n   {stock_name} ({ticker}) (first open lot from {month})")
            print(f"   Open: {open_lots} lot(s) / {open_units} unit(s)")
            print(f"   Avg Buy: {curr} {buy_price:.3f} | Now: {curr} {current_price:.3f} | RSI: {rsi:.1f}")
            print(f"   Unrealized P/L: {sign}{pnl_pct:.2f}% ({curr} {pnl_per_share:+.3f}/unit, total {curr} {unrealized_total:+.2f})")
            print(f"   Realized P/L: {curr} {realized_total:+.2f}")
            print(f"   Verdict: {verdict} — {reason}")

    closed_realized = [
        (ticker, data['value'], data['currency'])
        for ticker, data in realized_by_ticker.items()
        if ticker not in {p['ticker'] for p in open_positions}
    ]
    if closed_realized:
        print("\n   Closed positions (already fully sold):")
        for ticker, value, currency in closed_realized:
            print(f"   - {ticker}: Realized P/L {currency} {value:+.2f}")

    # ── LOG HELPER ───────────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("📝  TO LOG THIS MONTH'S PURCHASE")
    print("   Copy the line below into purchase_log.json (inside the [ ] array):\n")
    if results:
        top  = results[0]
        lots = top['smart_lots']
        log_entry = {
            "month":    datetime.now().strftime("%Y-%m"),
            "ticker":   top['ticker'],
            "price":    round(top['price'], 3),
            "currency": top['currency'],
            "lots":     lots['lots'] if lots else 1,
        }
        print(f"   {json.dumps(log_entry)}")

    print("\n✅ Done. Run again next payday.\n")

if __name__ == "__main__":
    main()