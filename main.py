import json
import math
import time
from datetime import datetime

import yfinance as yf
import pandas as pd
import ta

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR-AWARE PE THRESHOLDS
# Banks and REITs are valued differently from consumer/retail stocks.
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

def get_last_sector(log):
    return log[-1].get('sector') if log else None

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
    """
    Returns (years_paying: int, is_growing: bool)
    Looks at last 5 years of dividend history.
    """
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
    """Fetches name, yield, currency, fundamentals, and dividend history."""
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # Name
        long_name  = info.get('longName')
        short_name = info.get('shortName')
        name = (
            f"{long_name} - {short_name}"
            if (long_name and short_name and long_name != short_name)
            else (long_name or short_name or ticker)
        )

        # Currency
        if   ticker.endswith('.KL'): currency = "RM"
        elif ticker.endswith('.HK'): currency = "HKD"
        else:                        currency = "USD"

        # Dividend yield
        div_rate      = info.get('dividendRate', 0)
        current_price = info.get('currentPrice', 0) or info.get('previousClose', 0)
        if div_rate and current_price and current_price > 0:
            yield_pct = (div_rate / current_price) * 100
        else:
            raw = info.get('dividendYield', 0) or 0
            yield_pct = raw * 100 if raw < 1 else raw

        # Fundamentals
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

        return (name, yield_pct, currency, float(pe_ratio), roe,
                fmt(revenue), profit_margin, div_years, div_growing)

    except Exception:
        return (ticker, 0.0, "N/A", 0.0, 0.0, "N/A", 0.0, 0, False)

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

    latest = df.iloc[-1]

    # 52-week high/low from available data (up to last 252 trading days)
    lookback    = min(252, len(df))
    week52_high = float(df['High'].iloc[-lookback:].max())
    week52_low  = float(df['Low'].iloc[-lookback:].min())

    return {
        'price':         float(latest['Close']),
        'rsi':           float(latest['RSI']),
        'sma50':         float(latest['SMA_50']),
        'sma200':        float(latest['SMA_200']),
        'volume_strong': float(latest['Volume']) > float(latest['Vol_Avg']) * 1.5,
        'week52_high':   week52_high,
        'week52_low':    week52_low,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SAVINGS SCORE ENGINE  (max 100 pts)
#
#  Trend        0–15   Golden cross = baseline safety check
#  Entry        0–20   How deep is the pullback from 52w high?
#  Valuation    0–15   Sector-aware PE bands
#  Yield        0–15   Higher dividend yield = higher score
#  ROE          0–10   Return on equity efficiency
#  Dividend     0–15   Consistency + growth history
#  Rotation     0–10   Sector diversification from last purchase
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

    # 1. Trend: golden cross (0 or 15)
    trend_pts    = 15 if sma50 > sma200 else 0
    score       += trend_pts
    bd['trend']  = trend_pts

    # 2. Entry quality: pullback from 52w high (0–20)
    pullback      = ((week52_high - price) / week52_high * 100) if week52_high > 0 else 0
    entry_pts     = round(min(pullback * 0.8, 20))
    score        += entry_pts
    bd['entry']   = entry_pts

    # 3. Valuation: sector-aware PE (0–15)
    thresh = SECTOR_PE_THRESHOLDS.get(sector, SECTOR_PE_THRESHOLDS['general'])
    if pe <= 0 or pe > 200:
        val_pts = 0      # missing, negative, or extreme — can't score
    elif pe < thresh['great']:
        val_pts = 15
    elif pe < thresh['ok']:
        val_pts = 8
    else:
        val_pts = 2
    score          += val_pts
    bd['valuation'] = val_pts

    # 4. Yield (0–15): 7.5%+ yield = full marks
    yield_pts    = round(min(div_yield * 2, 15))
    score       += yield_pts
    bd['yield']  = yield_pts

    # 5. ROE / efficiency (0–10)
    roe_pts    = 10 if roe > 12 else 5 if roe > 8 else 2 if roe > 0 else 0
    score     += roe_pts
    bd['roe']  = roe_pts

    # 6. Dividend consistency (0–15)
    div_pts = 15 if div_years >= 5 else 10 if div_years >= 3 else 5 if div_years >= 1 else 0
    if div_growing and div_pts > 0:
        div_pts = min(div_pts + 3, 15)   # bonus for growing dividend
    score          += div_pts
    bd['dividend']  = div_pts

    # 7. Sector rotation (0–10)
    if last_sector is None:
        rot_pts = 5     # no history yet — neutral
    elif sector != last_sector:
        rot_pts = 10    # different sector = diversification bonus
    else:
        rot_pts = 0     # same sector as last month = no bonus
    score           += rot_pts
    bd['rotation']   = rot_pts

    return round(score), bd

# ─────────────────────────────────────────────────────────────────────────────
# TREND LABEL
# ─────────────────────────────────────────────────────────────────────────────
def trend_label(sma50, sma200, price):
    if sma50 > sma200 and price > sma200:
        return "🌟 GOLDEN + above SMA200"
    elif sma50 > sma200:
        return "🌤  GOLDEN (price still below SMA200)"
    else:
        return "💀 DEATH cross"

# ─────────────────────────────────────────────────────────────────────────────
# PE SANITY FLAG
# ─────────────────────────────────────────────────────────────────────────────
def pe_flag(pe):
    if pe <= 0:    return " ⚠️ N/A"
    if pe < 3:     return " ⚠️ Suspect data"
    if pe > 200:   return " ⚠️ Extreme"
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    if not config:
        return

    tickers_cfg  = config.get('tickers', [])
    period       = config.get('period', '2y')
    purchase_log = load_purchase_log()
    last_sector  = get_last_sector(purchase_log)

    results = []

    print(f"\n🚀 SAVINGS SCOUT — fetching {len(tickers_cfg)} stocks...\n")

    for entry in tickers_cfg:
        ticker = entry['ticker']
        sector = entry.get('sector', 'general')

        (name, div_yield, currency, pe_ratio, roe,
         revenue, profit_margin, div_years, div_growing) = get_stock_info(ticker)

        df = fetch_data(ticker, period)
        if df.empty:
            print()
            continue

        stats = analyze_stock(df)
        if not stats:
            print(f"⚠️  {ticker}: not enough data (need 200 days)\n")
            continue

        stats.update({
            'ticker':        ticker,
            'sector':        sector,
            'name':          name,
            'currency':      currency,
            'div_yield':     div_yield,
            'pe_ratio':      pe_ratio,
            'roe':           roe,
            'revenue':       revenue,
            'profit_margin': profit_margin,
            'div_years':     div_years,
            'div_growing':   div_growing,
            'smart_lots':    calculate_min_lots(stats['price'], ticker),
        })

        score, breakdown = calculate_savings_score(stats, sector, last_sector)
        stats['score']     = score
        stats['breakdown'] = breakdown

        results.append(stats)
        print()    # newline after each ticker fetch line
        time.sleep(0.5)

    # Sort by score — best pick is #1
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
              f"{last.get('currency','RM')} {last.get('price','?')} "
              f"({last.get('month','?')})")
    print()

    medals = {0: "🥇 TOP PICK", 1: "🥈 RUNNER-UP", 2: "🥉 THIRD"}

    for rank, item in enumerate(results):
        price    = item['price']
        sma50    = item['sma50']
        sma200   = item['sma200']
        curr     = item['currency']
        score    = item['score']
        bd       = item['breakdown']
        pullback = (
            (item['week52_high'] - price) / item['week52_high'] * 100
            if item['week52_high'] > 0 else 0
        )

        # Section headers
        if rank in medals:
            print("─" * W)
            print(f"  {medals[rank]}  (Score: {score}/100)")
        elif rank == 3:
            print("─" * W)
            print("  📋  FULL WATCHLIST")

        # Score bar
        filled = round(score / 5)
        bar    = "█" * filled + "░" * (20 - filled)

        # Dividend label
        if   item['div_years'] >= 5: div_str = f"✅ {item['div_years']} yrs"
        elif item['div_years'] >= 1: div_str = f"⚠️  {item['div_years']} yr(s) only"
        else:                         div_str = "❌ No history"
        if item['div_growing'] and item['div_years'] >= 1:
            div_str += " 📈 growing"

        # ROE display — flag if zero (likely missing data)
        roe_display = f"{item['roe']:.2f}%" if item['roe'] > 0 else "N/A (missing)"

        print(f"\n  {item['name']} ({item['ticker']})  [{item['sector'].upper()}]")
        print(f"  [{bar}] {score}/100")
        print(f"  Price:    {curr} {price:.3f}  |  52w High: {curr} {item['week52_high']:.3f}  |  "
              f"Pullback: {pullback:.1f}%")
        print(f"  Yield:    {item['div_yield']:.2f}%  |  RSI: {item['rsi']:.1f} (info only)  |  "
              f"Vol: {'🔥 HIGH' if item['volume_strong'] else 'Normal'}")
        print(f"  PE:       {item['pe_ratio']:.2f}{pe_flag(item['pe_ratio'])}  |  "
              f"ROE: {roe_display}  |  Margin: {item['profit_margin']:.2f}%")
        print(f"  Revenue:  {item['revenue']} (TTM)  |  Sector PE bands: "
              f"Great <{SECTOR_PE_THRESHOLDS.get(item['sector'],{}).get('great','?')}  "
              f"OK <{SECTOR_PE_THRESHOLDS.get(item['sector'],{}).get('ok','?')}")
        print(f"  Trend:    {trend_label(sma50, sma200, price)}")
        print(f"            SMA50: {sma50:.3f}  |  SMA200: {sma200:.3f}")
        print(f"  Dividend: {div_str}")
        print(f"  Score breakdown →  "
              f"Trend: {bd['trend']}  Entry: {bd['entry']}  PE: {bd['valuation']}  "
              f"Yield: {bd['yield']}  ROE: {bd['roe']}  Div: {bd['dividend']}  "
              f"Rotation: {bd['rotation']}")

        # Buy / Skip suggestion
        lots = item['smart_lots']
        if sma50 > sma200:
            if lots:
                print(f"  🛒 SUGGESTED BUY: {lots['lots']} lot(s)  →  "
                      f"{curr} {lots['cost']:.2f}  (fee: {lots['fee_pct']:.2f}%)")
            else:
                print(f"  🛒 SUGGESTED BUY: 1 unit  →  {curr} {price:.3f}")
        else:
            print(f"  ⛔ SKIP THIS MONTH: Death cross — wait for trend to reverse")

        print()

    # ── SECTOR SUMMARY ───────────────────────────────────────────────────────
    print("─" * W)
    print("📊  YOUR WATCHLIST BY SECTOR")
    from collections import Counter
    sector_map = Counter(r['sector'] for r in results)
    for sec, count in sorted(sector_map.items()):
        flag = "  ← bought last month" if sec == last_sector else ""
        print(f"   {sec.upper():15}  {count} stock(s){flag}")

    # ── LOG HELPER ───────────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("📝  TO LOG THIS MONTH'S PURCHASE")
    print("   Copy the line below into purchase_log.json (inside the [ ] array):\n")
    if results:
        top  = results[0]
        lots = top['smart_lots']
        entry = {
            "month":    datetime.now().strftime("%Y-%m"),
            "ticker":   top['ticker'],
            "sector":   top['sector'],
            "price":    round(top['price'], 3),
            "currency": top['currency'],
            "lots":     lots['lots'] if lots else 1,
        }
        print(f"   {json.dumps(entry)}")

    print("\n✅ Done. Run again next payday.\n")

if __name__ == "__main__":
    main()