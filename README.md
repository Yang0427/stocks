# 📈 Smart Stock Analysis & Lot Optimizer

A comprehensive Python tool for analyzing stocks across **Bursa Malaysia (.KL)**, **US Markets (USD)**, and **Hong Kong (.HK)**. It combines technical analysis with a smart fee calculator to optimize entry points and transaction costs.

## 🚀 Features

### 1. 🌍 Multi-Market Support
* **Malaysia (.KL):** Auto-calculates fees in RM.
* **USA (No Suffix):** Auto-detects USD.
* **Hong Kong (.HK):** Auto-detects HKD.

### 2. 🧠 Smart Entry Logic
* **Trend Detection:** Monitors **SMA50 vs SMA200** for Golden Cross (Bullish) or Death Cross (Bearish).
* **Momentum Scanner:** Uses **RSI (14-day)** to detect Overbought (>70), Oversold (<30), or Healthy Momentum (45-65).
* **Real-Time Yield:** Manually calculates Dividend Yield based on *live* prices to fix data provider glitches.

### 3. 💰 "Smart Lot" Calculator (Malaysia Only)
Transaction fees can eat into profits for small trades. This script calculates the **Minimum Lot Size** required to keep total fees (Platform + Stamp Duty + Clearing) **below 1%** of your investment value.

* *Example:* If buying 1 lot costs too much in fees (2%+), the script will suggest buying 3 lots to lower the fee impact to ~0.7%.

## 🛠️ Installation

1.  **Clone the repository** (or download the files).
2.  **Install dependencies:**
    ```bash
    pip install yfinance pandas ta
    ```

## ⚙️ Configuration (`stocks.json`)

Create a `stocks.json` file in the root directory to define your watchlist:

```json
{
  "tickers": [
    "5212.KL",   
    "1155.KL",
    "NVDA",
    "TSLA",
    "9988.HK"
  ],
  "period": "2y"
}