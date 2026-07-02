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
        <div className="mt-2 font-semibold">Nothing to show yet</div>
        <div className="text-sm text-muted">
          Run an analysis first — then set your budget to get this month's pick.
        </div>
      </Card>
    );
  }

  // First-class "buy nothing this month" answer. A quality veto (nothing clears
  // the bar) is the discipline win the whole tool exists for; an affordability
  // miss is a gentler "raise the budget or wait".
  if (pick.no_purchase) {
    const quality = pick.kind === "quality";
    return (
      <Card className="rise overflow-hidden">
        <div className={`px-6 py-4 border-b border-edge ${quality ? "bg-warn/10" : "bg-panel2/40"}`}>
          <div className="text-sm font-semibold uppercase tracking-widest text-warn">
            {quality ? "⏸️ No purchase this month" : "💸 Budget too tight"}
          </div>
        </div>
        <div className="p-6">
          <div className="text-lg font-bold">
            {quality ? "Carry your cash forward" : "Nothing fits this budget yet"}
          </div>
          <p className="mt-2 text-sm text-slate-200">{pick.reason}</p>
          {quality ? (
            <p className="mt-4 text-xs text-muted">
              Being told to wait is a feature, not a failure — a forced monthly pick is how you
              end up holding the least-bad name. Park the money and check again next payday, or
              untick <span className="text-slate-200">“Only actionable”</span> to see the full ranked list anyway.
            </p>
          ) : (
            <p className="mt-4 text-xs text-muted">
              Raise your monthly budget, or wait for a cheaper entry on a qualifying name.
            </p>
          )}
        </div>
      </Card>
    );
  }

  const c = pick.currency;
  const longScore = pick.long_term_score ?? pick.score;
  const entryPlan = pick.entry_plan;
  const targetOutlay = pick.suggested_total_outlay ?? pick.total_outlay;
  const targetLeftover = pick.suggested_leftover ?? pick.leftover;
  const targetFee = pick.suggested_fee ?? pick.estimated_fee;
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
            {pick.long_term_label && (
              <Pill tone={longScore >= 68 ? "good" : longScore >= 55 ? "warn" : "bad"}>
                {pick.long_term_label}
              </Pill>
            )}
            <Pill tone={timingTone(pick.timing_verdict)}>{pick.timing_verdict}</Pill>
            <Pill tone={healthTone(pick.eh_label)}>{pick.eh_label || "—"}</Pill>
          </div>
        </div>
      </div>

      <div className="p-6">
        {pick.data_uncertain && (
          <div className="mb-4 rounded-xl border border-warn/30 bg-warn/10 px-4 py-3 text-sm text-warn">
            ⚠️ {pick.data_warning || "Price data for this stock looks stale — verify before buying."}
          </div>
        )}
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="text-3xl font-extrabold tracking-tight">{pick.ticker}</div>
            <div className="text-muted">{pick.name}</div>
            <div className="mt-1">
              <Pill tone="info">{pick.sector?.toUpperCase()}</Pill>
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs uppercase tracking-wide text-muted">
              {pick.suggested_limit_price ? "Suggested limit" : "Total outlay"}
            </div>
            <div className="text-4xl font-extrabold text-accent">
              {pick.suggested_limit_price ? money(c, pick.suggested_limit_price, 3) : money(c, targetOutlay)}
            </div>
            <div className="text-xs text-muted">
              {pick.suggested_limit_price
                ? `${money(c, targetOutlay)} target outlay · ${money(c, targetLeftover)} left over`
                : `of ${money(c, pick.budget)} budget · ${money(c, targetLeftover)} left over`}
            </div>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Stat label="Buy" value={`${pick.lots} ${pick.ticker.endsWith(".KL") ? "lot(s)" : "share(s)"}`} sub={`${pick.units} units`} />
          <Stat
            label="Live quote"
            value={money(c, pick.price, 3)}
            sub={pick.price_source === "live_quote" ? "live quote" : "history close"}
          />
          <Stat
            label={pick.suggested_limit_price ? "Target outlay" : "Est. outlay"}
            value={money(c, targetOutlay)}
            sub={pick.suggested_limit_price && pick.total_outlay !== targetOutlay ? `max ${money(c, pick.total_outlay)} at quote` : "if order fills"}
            tone="good"
          />
          <Stat label="Est. fee" value={money(c, targetFee)} sub={pick.suggested_limit_price ? "if limit fills" : `${pick.fee_pct}%`} tone={pick.fee_pct < 1 ? "good" : "warn"} />
          <Stat label="Dividend yield" value={`${pick.div_yield?.toFixed(2)}%`} tone="good" />
        </div>

        {entryPlan && (
          <div className="mt-5 rounded-xl border border-accent/25 bg-accent/10 px-4 py-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="text-[11px] uppercase tracking-wide text-accent">Limit order plan</div>
                <div className="mt-1 text-slate-100">
                  Place around <span className="font-bold">{money(c, entryPlan.suggested_limit_price, 3)}</span>
                  {entryPlan.gap_from_live_pct > 0 ? ` (${entryPlan.gap_from_live_pct}% below live quote)` : ""}
                </div>
              </div>
              <div className="text-xs text-muted sm:text-right">
                Patient {money(c, entryPlan.patient_limit_price, 3)} · don't chase above {money(c, entryPlan.max_entry_price, 3)}
              </div>
            </div>
            <div className="mt-2 text-xs text-muted">{entryPlan.reason}</div>
          </div>
        )}

        <div className="mt-5 rounded-xl border border-edge bg-panel2/50 px-4 py-3">
          <div className="text-[11px] uppercase tracking-wide text-muted">Why this one</div>
          <div className="mt-1 text-slate-100">{pick.reason}</div>
          <div className="mt-1 text-xs text-muted">
            Long-term score {longScore} · Legacy setup score {pick.score} · {pick.pullback_pct}% below 52-week high
            {pick.analysis_price && Math.abs(pick.analysis_price - pick.price) > 0.001
              ? ` · analysis close ${money(c, pick.analysis_price, 3)}`
              : ""}
          </div>
        </div>

        {pick.long_term_breakdown && (
          <div className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-5">
            {Object.entries(pick.long_term_breakdown).map(([k, v]) => (
              <div key={k} className="rounded-lg border border-edge bg-ink/40 px-3 py-2">
                <div className="uppercase tracking-wide text-muted">{k}</div>
                <div className="font-bold text-accent">{v}</div>
              </div>
            ))}
          </div>
        )}

        {pick.long_term_flags?.length > 0 && (
          <div className="mt-3 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
            {pick.long_term_flags.join(" · ")}
          </div>
        )}

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
