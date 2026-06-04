import { Card, Pill, Stat, money } from "./ui";

// Open positions with live P/L and projected dividend income (FIFO-reconciled
// by the engine from your buy + sell logs), plus a portfolio-wide performance
// summary at the top.
export default function Holdings({ data, loading }) {
  if (loading) {
    return <Card className="p-6 text-center text-muted animate-pulse">Loading holdings…</Card>;
  }
  if (!data) return null;
  const { positions, total_annual_div, monthly_avg_div, currency, errors, summary } = data;

  return (
    <div className="space-y-4">
      {/* ── Portfolio performance hero ─────────────────────────────────── */}
      {summary && positions.length > 0 && (
        <PortfolioSummary s={summary} currency={currency} />
      )}

      {/* ── Dividend income ────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <Stat label="Annual dividend (est.)" value={money(currency, total_annual_div)} tone="good" />
        <Stat label="Monthly average" value={money(currency, monthly_avg_div)} tone="good" />
        <Stat label="Open positions" value={positions.length} />
      </div>

      {errors?.length > 0 && (
        <Card className="border-warn/40 p-4">
          <div className="text-sm font-semibold text-warn">⚠️ Log issues</div>
          <ul className="mt-1 list-disc pl-5 text-xs text-muted">
            {errors.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </Card>
      )}

      {positions.length === 0 ? (
        <Card className="p-6 text-center text-muted">No open positions yet. Log a buy to start tracking.</Card>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {positions.map((p) => {
            const up = p.pnl_pct ?? 0;
            const tone = up > 0 ? "good" : up < 0 ? "bad" : "muted";
            return (
              <Card key={p.ticker} className="p-4">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="font-bold">{p.ticker}</div>
                    <div className="text-xs text-muted">{p.name}</div>
                  </div>
                  <Pill tone={tone}>{up >= 0 ? "+" : ""}{p.pnl_pct?.toFixed(2)}%</Pill>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                  <Line label="Holding" value={`${p.open_lots} lot(s) / ${p.open_units}u`} />
                  <Line label="Avg buy" value={money(p.currency, p.avg_buy_price, 3)} />
                  <Line label="Now" value={money(p.currency, p.current_price, 3)} />
                  <Line label="Unrealized" value={`${p.unrealized >= 0 ? "+" : ""}${money(p.currency, p.unrealized)}`} tone={tone} />
                  <Line label="Invested" value={money(p.currency, p.cost)} />
                  <Line label="Worth now" value={money(p.currency, p.value)} />
                  <Line label="Yield" value={`${p.div_yield}%`} />
                  <Line label="Div/yr" value={money(p.currency, p.annual_div)} tone="good" />
                </div>
                <div className="mt-3 rounded-lg border border-edge bg-panel2/50 px-3 py-2 text-xs">
                  <span className="font-semibold">{p.verdict}</span>{" "}
                  <span className="text-muted">— {p.reason}</span>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

function PortfolioSummary({ s, currency }) {
  const pl = s.total_pl ?? 0;
  const up = pl >= 0;
  const accentText = up ? "text-accent" : "text-bad";
  const ring = up ? "from-accent/20" : "from-bad/20";
  const arrow = up ? "▲" : "▼";

  return (
    <Card className="rise overflow-hidden">
      <div className={`bg-gradient-to-r ${ring} via-transparent to-transparent px-6 py-5 border-b border-edge`}>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="text-xs uppercase tracking-widest text-muted">Total profit / loss</div>
            <div className={`text-4xl font-extrabold ${accentText}`}>
              {up ? "+" : ""}{money(currency, pl)}
            </div>
            <div className={`text-sm ${accentText}`}>
              {arrow} {s.total_return_pct == null ? "—" : `${up ? "+" : ""}${s.total_return_pct}% on open positions`}
            </div>
          </div>
          <div className="text-right text-sm">
            <div className="text-muted">
              Unrealized{" "}
              <span className={s.total_unrealized >= 0 ? "text-accent" : "text-bad"}>
                {s.total_unrealized >= 0 ? "+" : ""}{money(currency, s.total_unrealized)}
              </span>
            </div>
            <div className="text-muted">
              Realized{" "}
              <span className={s.total_realized >= 0 ? "text-accent" : "text-bad"}>
                {s.total_realized >= 0 ? "+" : ""}{money(currency, s.total_realized)}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-px bg-edge sm:grid-cols-4">
        <Cell label="Invested" value={money(currency, s.total_cost)} />
        <Cell label="Current value" value={money(currency, s.total_value)} />
        <Cell
          label="Winners / losers"
          value={`${s.winners} 🟢 / ${s.losers} 🔴`}
        />
        <Cell
          label="Best / worst"
          value={
            s.best && s.worst ? (
              <span className="text-xs">
                <span className="text-accent">{s.best.ticker} {s.best.pnl_pct >= 0 ? "+" : ""}{s.best.pnl_pct}%</span>
                {" · "}
                <span className="text-bad">{s.worst.ticker} {s.worst.pnl_pct}%</span>
              </span>
            ) : "—"
          }
        />
      </div>
    </Card>
  );
}

function Cell({ label, value }) {
  return (
    <div className="bg-panel px-4 py-3">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-0.5 text-base font-bold text-slate-100">{value}</div>
    </div>
  );
}

function Line({ label, value, tone }) {
  const c = tone === "good" ? "text-accent" : tone === "bad" ? "text-bad" : "text-slate-100";
  return (
    <div className="flex justify-between">
      <span className="text-muted">{label}</span>
      <span className={`font-medium ${c}`}>{value}</span>
    </div>
  );
}
