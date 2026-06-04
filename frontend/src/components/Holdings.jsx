import { Card, Pill, Stat, money } from "./ui";

// Open positions with live P/L and projected dividend income (FIFO-reconciled
// by the engine from your buy + sell logs).
export default function Holdings({ data, loading }) {
  if (loading) {
    return (
      <Card className="p-6 text-center text-muted animate-pulse">Loading holdings…</Card>
    );
  }
  if (!data) return null;
  const { positions, total_annual_div, monthly_avg_div, currency, errors } = data;

  return (
    <div className="space-y-4">
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
                  <Line label="Unrealized" value={money(p.currency, p.unrealized)} tone={tone} />
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

function Line({ label, value, tone }) {
  const c = tone === "good" ? "text-accent" : tone === "bad" ? "text-bad" : "text-slate-100";
  return (
    <div className="flex justify-between">
      <span className="text-muted">{label}</span>
      <span className={`font-medium ${c}`}>{value}</span>
    </div>
  );
}
