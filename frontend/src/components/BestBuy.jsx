import { Card, Pill, Button, Stat, money, timingTone, healthTone } from "./ui";

// The headline answer: "Buy THIS, this many lots, for this much."
export default function BestBuy({ pick, onLog, loading }) {
  if (loading) {
    return (
      <Card className="rise p-8 text-center">
        <div className="animate-pulse text-muted">Crunching your watchlist…</div>
      </Card>
    );
  }
  if (!pick) {
    return (
      <Card className="rise p-8 text-center">
        <div className="text-2xl">🤔</div>
        <div className="mt-2 font-semibold">Nothing fits this budget yet</div>
        <div className="text-sm text-muted">
          Increase the budget, or run an analysis first. Cheapest entry on your list may exceed what you entered.
        </div>
      </Card>
    );
  }

  const c = pick.currency;
  return (
    <Card className="rise overflow-hidden">
      <div className="bg-gradient-to-r from-accent/20 via-accent2/10 to-transparent px-6 py-4 border-b border-edge">
        <div className="flex items-center justify-between gap-3">
          <div className="text-sm font-semibold uppercase tracking-widest text-accent">
            🎯 This month, buy
          </div>
          <div className="flex gap-2">
            <Pill tone={pick.is_actionable ? "good" : "warn"}>
              {pick.is_actionable ? "Actionable now" : "Best available"}
            </Pill>
            <Pill tone={timingTone(pick.timing_verdict)}>{pick.timing_verdict}</Pill>
            <Pill tone={healthTone(pick.eh_label)}>{pick.eh_label || "—"}</Pill>
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="text-3xl font-extrabold tracking-tight">{pick.ticker}</div>
            <div className="text-muted">{pick.name}</div>
            <div className="mt-1">
              <Pill tone="info">{pick.sector?.toUpperCase()}</Pill>
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs uppercase tracking-wide text-muted">Total outlay</div>
            <div className="text-4xl font-extrabold text-accent">{money(c, pick.total_outlay)}</div>
            <div className="text-xs text-muted">
              of {money(c, pick.budget)} budget · {money(c, pick.leftover)} left over
            </div>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Buy" value={`${pick.lots} ${pick.ticker.endsWith(".KL") ? "lot(s)" : "share(s)"}`} sub={`${pick.units} units`} />
          <Stat label="At price" value={money(c, pick.price, 3)} />
          <Stat label="Est. fee" value={money(c, pick.estimated_fee)} sub={`${pick.fee_pct}%`} tone={pick.fee_pct < 1 ? "good" : "warn"} />
          <Stat label="Dividend yield" value={`${pick.div_yield?.toFixed(2)}%`} tone="good" />
        </div>

        <div className="mt-5 rounded-xl border border-edge bg-panel2/50 px-4 py-3">
          <div className="text-[11px] uppercase tracking-wide text-muted">Why this one</div>
          <div className="mt-1 text-slate-100">{pick.reason}</div>
          <div className="mt-1 text-xs text-muted">
            Score {pick.score} (effective {pick.effective_score}) · {pick.pullback_pct}% below 52-week high
          </div>
        </div>

        <div className="mt-5 flex flex-wrap gap-3">
          <Button onClick={() => onLog(pick)}>📝 Log this buy</Button>
        </div>
        <p className="mt-3 text-xs text-muted">
          MooMoo tip: this is the target. Place your limit order — only log it here once it actually fills.
        </p>
      </div>
    </Card>
  );
}
