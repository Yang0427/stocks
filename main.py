"""
main.py — Savings Scout CLI.

Thin command-line front-end over engine.py. All scoring/data/P&L logic lives in
engine.py so the CLI and the web API (api.py) share one brain. Run:

    python main.py
"""

import json
from collections import Counter
from datetime import datetime

import engine
from engine import (
    SECTOR_PE_THRESHOLDS, CYCLICAL_SECTORS,
    load_config, load_purchase_log, load_sell_log,
    build_ticker_sector_map, analyze_all, build_open_positions,
    project_annual_dividend, evaluate_accumulation_signal,
    trend_label, pe_flag, pick_actionable, fetch_data, get_stock_info,
    analyze_stock, effective_score,
)


def main():
    config = load_config()
    if not config:
        print("❌ Error: 'stocks.json' not found.")
        return

    tickers_cfg = config.get('tickers', [])
    period = config.get('period', '2y')
    ticker_sector_map = build_ticker_sector_map(tickers_cfg)
    purchase_log = load_purchase_log()
    sell_log = load_sell_log()

    print(f"\n🚀 SAVINGS SCOUT — fetching {len(tickers_cfg)} stocks...\n")

    def progress(done, total, msg):
        print(f"   [{done}/{total}] {msg}", flush=True)

    # CLI fetches live (small polite delay), matching previous behaviour.
    analysis = analyze_all(
        config=config, purchase_log=purchase_log, sell_log=sell_log,
        use_cache=True, progress=progress, polite_delay=0.3,
    )
    results = analysis['results']
    recent_sectors = analysis['recent_sectors']
    last_sector = analysis['last_sector']

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
        price = item['price']
        sma50 = item['sma50']
        sma200 = item['sma200']
        curr = item['currency']
        score = item['score']
        long_score = item.get('long_term_score', score)
        bd = item['breakdown']
        ltd = item.get('long_term_breakdown', {})
        pullback = item['pullback_pct']
        eff_s = item['effective_score']
        eff_lt = item.get('effective_long_term_score', long_score)
        is_golden = item['is_golden']
        tv = item['timing_verdict']

        if rank in medals:
            print("─" * W)
            actionable = is_golden and tv != "⏳ WAIT"
            tag = "" if actionable else "  ⚠️  NOT ACTIONABLE THIS MONTH"
            print(f"  {medals[rank]}  (Long-term: {long_score}  |  Effective LT: {eff_lt}  |  Setup: {score}){tag}")
        elif rank == 3:
            print("─" * W)
            print("  📋  FULL WATCHLIST")

        bar_fill = max(0, min(20, round(long_score / 5)))
        bar = "█" * bar_fill + "░" * (20 - bar_fill)

        if   item['div_years'] >= 5: div_str = f"✅ {item['div_years']} yrs"
        elif item['div_years'] >= 1: div_str = f"⚠️  {item['div_years']} yr(s) only"
        else:                        div_str = "❌ No history"
        if item['div_growing'] and item['div_years'] >= 1:
            div_str += " 📈 growing"

        roe_display = f"{item['roe']:.2f}%" if item['roe'] > 0 else "N/A (missing)"

        if item['months_since_buy'] is not None:
            months_tag = f"  |  Last bought: {item['months_since_buy']}m ago"
        else:
            months_tag = "  |  Never bought"

        print(f"\n  {item['name']} ({item['ticker']})  [{item['sector'].upper()}]{months_tag}")
        print(f"  [{bar}] Long-term score: {long_score}  (effective {eff_lt})  |  Legacy setup: {score} (effective {eff_s})")
        if item.get('long_term_label'):
            print(f"  Conviction: {item['long_term_label']}")
        if ltd:
            print("  Long-term: "
                  f"Quality={ltd.get('quality',0)} | Valuation={ltd.get('valuation',0)} | "
                  f"Entry={ltd.get('entry',0)} | Income={ltd.get('income',0)} | "
                  f"Portfolio={ltd.get('portfolio',0)}")
        for flag in item.get('long_term_flags', []):
            print(f"  ⚠️  {flag}")
        if item.get('data_uncertain'):
            print(f"  ⚠️  DATA UNCERTAIN: {item.get('data_warning', 'price data may be stale')}")
            print("       → excluded from the auto-pick until the two price feeds agree")
        quote_price = item.get('order_price', price)
        if abs(quote_price - price) > 0.001:
            print(
                f"  Quote:    {curr} {quote_price:.3f}  |  Analysis close: {curr} {price:.3f}  |  "
                f"52w High: {curr} {item['week52_high']:.3f}  |  Pullback: {pullback:.1f}%"
            )
        else:
            print(f"  Quote:    {curr} {quote_price:.3f}  |  52w High: {curr} {item['week52_high']:.3f}  |  Pullback: {pullback:.1f}%")
        entry_plan = engine.limit_order_plan(item)
        if entry_plan:
            print(
                f"  Entry:    Limit {curr} {entry_plan['suggested_limit_price']:.3f}  |  "
                f"Patient {curr} {entry_plan['patient_limit_price']:.3f}  |  "
                f"Max chase {curr} {entry_plan['max_entry_price']:.3f}"
            )
            print(f"            {entry_plan['reason']}")
        print(f"  Yield:    {item['div_yield']:.2f}%  |  RSI: {item['rsi']:.1f} (info only)  |  Vol: {'🔥 HIGH' if item['volume_strong'] else 'Normal'}")
        print(f"  PE:       {item['pe_ratio']:.2f}{pe_flag(item['pe_ratio'])}  |  ROE: {roe_display}  |  Margin: {item['profit_margin']:.2f}%")
        print(f"  Revenue:  {item['revenue']} (TTM)  |  Sector PE bands: "
              f"Great <{SECTOR_PE_THRESHOLDS.get(item['sector'],{}).get('great','?')}  "
              f"OK <{SECTOR_PE_THRESHOLDS.get(item['sector'],{}).get('ok','?')}")
        print(f"  Debt:     {item['debt_str']}")
        print(f"  Trend:    {trend_label(sma50, sma200, price)}")
        print(f"            SMA50: {sma50:.3f}  |  SMA200: {sma200:.3f}")
        print(f"  Dividend: {div_str}")

        ad = item.get('analyst_data', {})
        tgt = ad.get('target_mean', 0.0)
        tgt_hi = ad.get('target_high', 0.0)
        n_an = ad.get('num_analysts', 0)
        rec = ad.get('recommendation', '')
        curr_price_a = ad.get('current_price', price)
        if tgt > 0 and curr_price_a > 0:
            upside_pct = (tgt - curr_price_a) / curr_price_a * 100
            rec_str = rec.replace('_', ' ').upper() if rec else 'N/A'
            print(f"  Analysts: {n_an} analysts  |  Target {curr} {tgt:.3f}"
                  f"  ({upside_pct:+.1f}%)  |  High {curr} {tgt_hi:.3f}  |  Consensus: {rec_str}")
        else:
            print("  Analysts: No coverage data")

        falling_knife_tag = ""
        if (item['sector'] in CYCLICAL_SECTORS
                and item.get('months_since_high', 0) >= 9
                and price < item['sma200']):
            falling_knife_tag = "  ⚠️ FALLING KNIFE"
        print(f"  Score:    Trend={bd['trend']} | Entry={bd['entry']}{falling_knife_tag} | "
              f"PE={bd['valuation']} | Yield={bd['yield']}(trap={bd.get('yield_trap',0)}) | "
              f"ROE={bd['roe']} | Div={bd['dividend']} | Rotation={bd['rotation']} | "
              f"Fresh={bd['freshness']} | Analyst={bd.get('analyst',0)} | "
              f"EPSGrowth={bd.get('eps_growth',0)} | Rerate={bd.get('rerate',0)} | "
              f"RecentBuy={bd.get('recent_buy', 0)} | EarningsHealth={bd.get('eh_penalty', 0)}")

        eh_label = item.get('eh_label', '🟢 STABLE')
        eh_warn = item.get('eh_warnings', [])
        eh_data = item.get('earnings_health', {})
        print(f"  ┌─ EARNINGS HEALTH: {eh_label} " + "─" * max(0, 46 - len(eh_label)))
        if eh_warn:
            for w in eh_warn:
                print(f"  │  {w}")
        else:
            print("  │  (No earnings trend data available)")
        feps_d = eh_data.get('forward_eps', 0.0)
        teps_d = eh_data.get('trailing_eps', 0.0)
        if teps_d > 0 and feps_d > 0:
            eps_gr = (feps_d - teps_d) / abs(teps_d) * 100
            eps_icon = "🟢" if eps_gr >= 5 else ("🟡" if eps_gr >= 0 else "🔴")
            print(f"  │  {eps_icon} EPS growth (fwd vs trailing): {eps_gr:+.1f}%  "
                  f"(Trailing EPS {teps_d:.3f}  →  Forward EPS {feps_d:.3f})")
        five_dy = eh_data.get('five_yr_avg_div_yield', 0.0)
        curr_dy = item.get('div_yield', 0.0)
        if five_dy > 0 and curr_dy > 0:
            prem = (curr_dy - five_dy) / five_dy * 100
            if prem >= 15:
                rerate_icon, rerate_msg = "🟢", f"yield {prem:+.0f}% above 5-yr avg — historically cheap"
            elif prem <= -15:
                rerate_icon, rerate_msg = "🔴", f"yield {prem:+.0f}% below 5-yr avg — priced richly vs own history"
            else:
                rerate_icon, rerate_msg = "🟡", f"yield near 5-yr average ({five_dy:.2f}%) — fairly valued vs history"
            print(f"  │  {rerate_icon} Historical yield: now {curr_dy:.2f}%  |  5-yr avg {five_dy:.2f}%  — {rerate_msg}")
        if item.get('eh_penalty', 0) > 0:
            print(f"  └─ ⚠️  Score penalised -{item['eh_penalty']} pts for fundamental deterioration")
        else:
            print("  └─ Fundamentals look clean")

        print(f"  ┌─ TIMING: {tv} " + "─" * max(0, 56 - len(tv)))
        for g in item['timing_green']:
            print(f"  │  ✔ {g}")
        for y in item['timing_yellow']:
            print(f"  │  ✘ {y}")
        if not item['timing_green'] and not item['timing_yellow']:
            print("  │  (No timing data available)")

        lots = item['smart_lots']
        if not is_golden:
            print("  └─ ⛔ SKIP: Death cross — wait for trend reversal first")
        elif tv == "⏳ WAIT":
            print("  └─ ⏳ WAIT: Multiple short-term risks — revisit in 1–2 weeks")
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

    # ── HOLDINGS REVIEW ───────────────────────────────────────────────────────
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

    # Build a quick lookup of already-analysed tickers to avoid refetching
    by_ticker = {r['ticker']: r for r in results}

    if not open_positions:
        if purchase_log:
            print("   No open positions after reconciling buys/sells")
    else:
        total_annual_div = 0.0
        portfolio_currency = "RM"
        for pos in open_positions:
            ticker = pos['ticker']
            buy_price = pos['avg_buy_price']
            curr = pos['currency']
            open_lots = pos['open_lots']
            open_units = pos['open_units']
            month = pos['first_month']
            portfolio_currency = curr

            if ticker in by_ticker:
                r = by_ticker[ticker]
                current_price = r['price']
                week52_high = r['week52_high']
                div_yield_h = r['div_yield']
                stock_name = r['name']
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
                current_price = stats_h['price']
                week52_high = stats_h['week52_high']

            pnl_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else float('nan')
            unrealized_total = (current_price - buy_price) * open_units
            realized_total = pos['realized_pnl']['value']
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
        print("      Reinvest dividends to compound your savings over time!")

    closed_realized = [
        (ticker, data['value'], data['currency'])
        for ticker, data in realized_by_ticker.items()
        if ticker not in {p['ticker'] for p in open_positions}
    ]
    if closed_realized:
        print("\n   Rebalanced / closed positions:")
        for ticker, value, currency in closed_realized:
            print(f"   - {ticker}: Realized P/L {currency} {value:+.2f}")

    # ── LOG HELPER + FINAL VERDICT ────────────────────────────────────────────
    print("\n" + "─" * W)
    print("📝  TO LOG THIS MONTH'S PURCHASE")
    print("   Copy the line below into purchase_log.json (inside the [ ] array):\n")

    top = pick_actionable(results)
    if top:
        lots = top['smart_lots']
        order_price = top.get('order_price', top['price'])
        entry_plan = engine.limit_order_plan(top)
        target_price = entry_plan['suggested_limit_price'] if entry_plan else order_price
        log_entry = {
            "month": datetime.now().strftime("%Y-%m"),
            "ticker": top['ticker'],
            "price": round(target_price, 3),
            "currency": top['currency'],
            "lots": lots['lots'] if lots else 1,
        }
        print(f"   {json.dumps(log_entry)}")
        if entry_plan:
            print(
                f"   (Suggested limit: {top['currency']} {target_price:.3f}; "
                f"only log once your broker order actually fills.)"
            )
        if top != results[0]:
            print(f"   (Note: {results[0]['ticker']} ranks highest on long-term score but skipped — "
                  f"earnings health: {results[0].get('eh_label','?')})")
    else:
        print("   (Nothing actionable to log this month — carry your cash forward.)")

    print("\n" + "=" * W)
    print("🎯  THIS MONTH'S BEST BUY — FINAL VERDICT")
    print("=" * W)
    if top:
        curr = top['currency']
        lots = top['smart_lots']
        order_price = top.get('order_price', top['price'])
        entry_plan = engine.limit_order_plan(top)
        target_price = entry_plan['suggested_limit_price'] if entry_plan else order_price
        tv = top['timing_verdict']
        print(f"\n  ✅ {top['name']} ({top['ticker']})  [{top['sector'].upper()}]")
        print(
            f"     Long-term {top.get('long_term_score', top['score'])} "
            f"| Effective LT {top.get('effective_long_term_score', top.get('long_term_score', top['score']))} "
            f"| Legacy setup {top['score']}  |  Earnings: {top.get('eh_label','🟢 STABLE')}"
        )
        print(f"     Order price {curr} {order_price:.3f}  |  Yield {top['div_yield']:.2f}%  |  PE {top['pe_ratio']:.2f}")
        if entry_plan:
            print(
                f"     Suggested limit {curr} {target_price:.3f}  |  "
                f"Patient {curr} {entry_plan['patient_limit_price']:.3f}  |  "
                f"Don't chase above {curr} {entry_plan['max_entry_price']:.3f}"
            )
            print(f"     Entry note: {entry_plan['reason']}")
        if abs(order_price - top['price']) > 0.001:
            print(f"     Analysis close {curr} {top['price']:.3f}  |  Source: {top.get('price_source', 'history_close')}")
        print(f"     Timing: {tv}")
        print(f"     Why: {top['reason']}")
        if lots:
            if tv == "🟡 CAUTION":
                half = max(1, lots['lots'] // 2)
                half_cost = half * 100 * target_price
                print(f"\n  👉 BUY {half} lot(s) now → {curr} {half_cost:.2f}  (half position, caution flag active)")
            else:
                target_cost = lots['lots'] * 100 * target_price
                target_fee = engine.bursa_fees(target_cost) if top['ticker'].endswith('.KL') else 0.0
                print(
                    f"\n  👉 PLACE LIMIT BUY {lots['lots']} lot(s) @ {curr} {target_price:.3f} "
                    f"→ est. {curr} {target_cost + target_fee:.2f} incl. fee"
                )
        else:
            print(f"\n  👉 BUY 1 unit → {curr} {target_price:.3f}")
        eh_yellow = [w for w in top.get('eh_warnings', []) if w.startswith('🟡') or w.startswith('🟠')]
        if eh_yellow:
            print("\n  ⚠️  Watch points:")
            for w in eh_yellow:
                print(f"     {w}")
    else:
        print("\n  ❌ No fully actionable picks this month — all either death cross, ⏳ WAIT, or deteriorating earnings.")
        print("     Consider waiting or parking savings in a money market fund.")

    print("\n✅ Done. Run again next payday.\n")


if __name__ == "__main__":
    main()
