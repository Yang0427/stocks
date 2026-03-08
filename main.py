import json
import yfinance as yf
import pandas as pd
import ta
import time
import math

# --- CONFIGURATION ---
def load_config():
    try:
        with open('stocks.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Error: 'stocks.json' file not found.")
        return None

# --- DATA FETCHING ---
def fetch_data(ticker, period='2y'):
    print(f"Fetching data for {ticker}...", end=" ")
    try:
        df = yf.download(ticker, period=period, interval='1d', progress=False)
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def get_stock_info(ticker):
    """Fetches Name, Yield, Currency, and FUNDAMENTALS"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # --- GET NAMES & CURRENCY ---
        long_name = info.get('longName')
        short_name = info.get('shortName')
        name = f"{long_name} - {short_name}" if (long_name and short_name and long_name != short_name) else (long_name or short_name or ticker)
        
        if ticker.endswith('.KL'):
            currency = "RM"
        elif ticker.endswith('.HK'):
            currency = "HKD"
        else:
            currency = "USD"

        # --- YIELD CALCULATION ---
        div_rate = info.get('dividendRate', 0)
        current_price = info.get('currentPrice', 0) or info.get('previousClose', 0)
        if div_rate and current_price and current_price > 0:
            yield_pct = (div_rate / current_price) * 100
        else:
            raw_yield = info.get('dividendYield', 0)
            if raw_yield is None:
                yield_pct = 0.0
            elif raw_yield < 1: 
                yield_pct = raw_yield * 100
            else:
                yield_pct = raw_yield

        # --- NEW: FUNDAMENTAL METRICS ---
        pe_ratio = info.get('trailingPE', 0) or info.get('forwardPE', 0) or 0.0 # Valuation
        roe = info.get('returnOnEquity', 0) # Profitability
        if roe: roe = roe * 100 # Convert to percentage
        else: roe = 0.0
        
        revenue = info.get('totalRevenue', 0) or 0
        profit_margin = info.get('profitMargins', 0)
        if profit_margin: profit_margin = profit_margin * 100
        else: profit_margin = 0.0

        # Format Revenue to be readable (e.g., 1.5B, 500M)
        def format_currency(value):
            if value >= 1e9: return f"{value/1e9:.2f}B"
            if value >= 1e6: return f"{value/1e6:.2f}M"
            return str(value)
            
        rev_formatted = format_currency(revenue)

        return name, yield_pct, currency, pe_ratio, roe, rev_formatted, profit_margin
    except Exception as e:
        return ticker, 0.0, "N/A", 0.0, 0.0, "N/A", 0.0

# --- FEE CALCULATOR (MALAYSIA ONLY) ---
def calculate_min_lots(price_per_unit, ticker):
    """
    Only calculates lots for .KL stocks. 
    Returns None for International stocks (fees vary too much).
    """
    if not ticker.endswith('.KL'):
        return None # Return None means "Don't calculate lots"

    for lots in range(1, 100):
        total_value = price_per_unit * 100 * lots
        
        # Estimate Fees (Rakuten/Broker standard)
        stamp_duty = math.ceil(total_value / 1000) 
        if stamp_duty < 1: stamp_duty = 1
        
        clearing_fee = total_value * 0.0003
        platform_fee = 3.00 
        
        total_fee = platform_fee + stamp_duty + clearing_fee
        fee_percentage = (total_fee / total_value) * 100
        
        if fee_percentage < 1.0: 
            return {'lots': lots, 'cost': total_value, 'fee_pct': fee_percentage}
            
    return {'lots': 1, 'cost': price_per_unit*100, 'fee_pct': 100}

# --- TECHNICAL ANALYSIS ---
def analyze_stock(df):
    if len(df) < 200: return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Price Indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    
    # 2. Volume Indicators
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    latest = df.iloc[-1]
    is_high_volume = latest['Volume'] > (latest['Vol_Avg'] * 1.2)
    
    return {
        'price': latest['Close'],
        'rsi': latest['RSI'],
        'sma50': latest['SMA_50'],
        'sma200': latest['SMA_200'],
        'volume_strong': is_high_volume
    }

# --- THE "BRAIN" ---
def generate_long_term_verdict(stock_data):
    price = stock_data['price']
    sma50 = stock_data['sma50']
    sma200 = stock_data['sma200']
    rsi = stock_data['rsi']
    vol_strong = stock_data['volume_strong']
    pe = stock_data.get('pe_ratio', 0)
    roe = stock_data.get('roe', 0)
    
    is_bull_market = sma50 > sma200
    is_price_healthy = price > sma200
    is_undervalued = (0 < pe < 20)
    is_efficient = roe > 10

    category = ""
    why = ""
    verdict = ""
    target_price = 0.0

    # 1. BULL MARKET
    if is_bull_market and is_price_healthy:
        if rsi < 55 and is_undervalued and is_efficient:
            category = "💎 STRONG BUY (Ultimate Value)"
            target_price = price 
        elif rsi < 55:
            category = "🟢 BUY (Technical Discount)"
            target_price = price
        elif rsi > 70:
            category = "✅ HOLD / WAIT (Overheated)"
            target_price = sma50
        else:
            if vol_strong:
                 category = "🚀 BUY (High Momentum)"
            else:
                 category = "📈 BUY / ACCUMULATE"
            target_price = price

    # 2. BEAR MARKET
    elif not is_bull_market:
        category = "⛔ AVOID / SELL"
        target_price = 0.0

    # 3. UNCERTAIN
    else:
        category = "⚠️ CAUTION (Trend Testing)"
        target_price = sma200

    return {
        "category": category,
        "verdict": verdict,
        "target_price": target_price
    }

# --- MAIN EXECUTION ---
def main():
    config = load_config()
    if not config: return

    tickers = config.get('tickers', [])
    period = '2y'
    portfolio_analysis = []

    print(f"\n🚀 STARTING UNIVERSAL ANALYSIS (MY/US/HK)...\n")

    for ticker in tickers:
        # Get Currency along with other info
        name, div_yield, currency, pe_ratio, roe, revenue, profit_margin = get_stock_info(ticker)
        df = fetch_data(ticker, period)
        
        if not df.empty:
            stats = analyze_stock(df)
            if stats:
                stats['ticker'] = ticker
                stats['name'] = name
                stats['currency'] = currency
                stats['div_yield'] = div_yield
                stats['pe_ratio'] = pe_ratio
                stats['roe'] = roe
                stats['revenue'] = revenue
                stats['profit_margin'] = profit_margin
                stats['analysis'] = generate_long_term_verdict(stats)
                
                # Calculate Smart Lots (Only returns data if .KL)
                stats['smart_lots'] = calculate_min_lots(stats['price'], ticker)
                
                portfolio_analysis.append(stats)
        time.sleep(0.5) 

    # --- FINAL REPORT ---
    print("\n" + "="*80)
    print(f"🎯  GLOBAL MARKET REPORT")
    print("="*80)

    for item in portfolio_analysis:
        a = item['analysis']
        price = item['price']
        target = a['target_price']
        sma50 = item['sma50']
        sma200 = item['sma200']
        curr = item['currency'] # Use correct currency
        
        vol_msg = "🔥 HIGH" if item['volume_strong'] else "Normal"
        
        # Calculate discount
        if target > 0 and price > target:
            discount = ((price - target) / price) * 100
            target_str = f"Wait for {curr} {target:.3f} (-{discount:.1f}%)"
        elif target > 0:
            target_str = "✅ BUY NOW"
        else:
            target_str = "❌ NO ENTRY"

        trend_icon = "🌟 GOLDEN" if sma50 > sma200 else "💀 DEATH"
        
        print(f"\n🔹 {item['name']} ({item['ticker']})")
        # FORMATTING CHANGE: .3f used below
        print(f"   Price:        {curr} {price:.3f}  |  Yield: {item['div_yield']:.2f}%")
        print(f"   Fundamentals: PE: {item['pe_ratio']:.2f} (Valuation)  |  ROE: {item['roe']:.2f}% (Efficiency)")
        print(f"                 Revenue: {item['revenue']} (Last 4 Quarters)  |  Margin: {item['profit_margin']:.2f}% (Profitability)")
        print(f"   Indicators:   RSI: {item['rsi']:.1f}  |  Vol: {vol_msg}")
        print(f"   Trend:        {trend_icon} (SMA50: {sma50:.3f} | SMA200: {sma200:.3f})")
        print(f"   Verdict:      {a['category']}")
        print(f"   Target:       {target_str}")
        
        # Suggestion Logic
        lots_data = item['smart_lots']
        if lots_data: # If it is a Malaysian stock (.KL)
            if target > 0:
                print(f"   🛒 SUGGESTION: Buy {lots_data['lots']} Lots ({curr} {lots_data['cost']:.2f})")
                print(f"      (Keeps fee impact low at {lots_data['fee_pct']:.2f}%)")
        else: # If it is US/International
            if target > 0:
                 print(f"   🛒 SUGGESTION: Buy 1 Unit ({curr} {price:.3f})")
                 print(f"      (US/HK Stocks do not use 100-unit lots)")
             
        print("-" * 80)

    print("\n✅ Analysis Complete.")

if __name__ == "__main__":
    main()