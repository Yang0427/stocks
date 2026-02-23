import streamlit as st
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
        return None

# --- DATA FETCHING ---
def fetch_data(ticker, period='2y'):
    try:
        df = yf.download(ticker, period=period, interval='1d', progress=False)
        return df
    except Exception as e:
        return pd.DataFrame()

def get_stock_info(ticker):
    """Fetches Name, Yield, and Currency"""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        name = info.get('shortName') or info.get('longName') or ticker
        
        if ticker.endswith('.KL'):
            currency = "RM"
        elif ticker.endswith('.HK'):
            currency = "HKD"
        else:
            currency = "USD"

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
            
        return name, yield_pct, currency
    except:
        return ticker, 0.0, "N/A"

# --- FEE CALCULATOR (MALAYSIA ONLY) ---
def calculate_min_lots(price_per_unit, ticker):
    if not ticker.endswith('.KL'):
        return None

    for lots in range(1, 100):
        total_value = price_per_unit * 100 * lots
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
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
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
    is_bull_market = sma50 > sma200
    is_price_healthy = price > sma200
    category = ""
    target_price = 0.0
    if is_bull_market and is_price_healthy:
        if rsi < 55:
            category = "💎 STRONG BUY (Discount)"
            target_price = price 
        elif rsi > 70:
            category = "✅ HOLD / WAIT (Overheated)"
            target_price = sma50
        else:
            category = "🚀 BUY (High Momentum)" if vol_strong else "📈 BUY / ACCUMULATE"
            target_price = price
    elif not is_bull_market:
        category = "⛔ AVOID / SELL"
        target_price = 0.0
    else:
        category = "⚠️ CAUTION (Trend Testing)"
        target_price = sma200
    return {"category": category, "target_price": target_price}

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Stock Teacher", page_icon="🎯", layout="wide")
    st.title("🎯 Global Market Report")
    st.markdown("### My personal stock teacher for MY/US/HK markets.")

    config = load_config()
    if not config:
        st.error("❌ Error: 'stocks.json' file not found.")
        return

    tickers = config.get('tickers', [])
    
    if st.button("🚀 Run Market Analysis"):
        with st.spinner('Fetching data and analyzing...'):
            for ticker in tickers:
                name, div_yield, currency = get_stock_info(ticker)
                df = fetch_data(ticker, '2y')
                
                if not df.empty:
                    stats = analyze_stock(df)
                    if stats:
                        a = generate_long_term_verdict(stats)
                        price = stats['price']
                        target = a['target_price']
                        sma50 = stats['sma50']
                        sma200 = stats['sma200']
                        
                        st.subheader(f"🔹 {name} ({ticker})")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Current Price", f"{currency} {price:.3f}")
                        col2.metric("Dividend Yield", f"{div_yield:.2f}%")
                        col3.metric("RSI (14)", f"{stats['rsi']:.1f}")
                        
                        vol_msg = "🔥 HIGH" if stats['volume_strong'] else "Normal"
                        trend_icon = "🌟 GOLDEN" if sma50 > sma200 else "💀 DEATH"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Volume:** {vol_msg}")
                            st.write(f"**Trend:** {trend_icon}")
                        with col2:
                            st.write(f"**SMA50:** {sma50:.3f}")
                            st.write(f"**SMA200:** {sma200:.3f}")
                        
                        st.info(f"**Verdict:** {a['category']}")
                        
                        # Target and Suggestion
                        if target > 0 and price > target:
                            discount = ((price - target) / price) * 100
                            target_str = f"Wait for {currency} {target:.3f} (-{discount:.1f}%)"
                        elif target > 0:
                            target_str = "✅ BUY NOW"
                        else:
                            target_str = "❌ NO ENTRY"
                        
                        st.write(f"**Target:** {target_str}")
                        
                        # Suggestion Logic
                        lots_data = calculate_min_lots(price, ticker)
                        if lots_data: # MY Stock
                            if target > 0:
                                st.success(f"🛒 **SUGGESTION:** Buy {lots_data['lots']} Lots ({currency} {lots_data['cost']:.2f})")
                                st.caption(f"(Keeps fee impact low at {lots_data['fee_pct']:.2f}%)")
                        else: # Int/US Stock
                            if target > 0:
                                st.success(f"🛒 **SUGGESTION:** Buy 1 Unit ({currency} {price:.3f})")
                                st.caption("(US/HK Stocks do not use 100-unit lots)")
                        
                        st.divider()
                        time.sleep(0.2)
            st.success("✅ Analysis Complete.")

if __name__ == "__main__":
    main()
