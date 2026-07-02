import { useState } from "react";
import { Pill, ScoreBar, money, timingTone, healthTone } from "./ui";

// One stock in the ranked watchlist. Collapsed by default; expands to show the
// full score breakdown + timing/earnings signals (the detail the CLI prints).
export default function StockRow({ item, rank, maxScore, onPick }) {
  const [open, setOpen] = useState(false);
  const c = item.currency;
  const medal = ["🥇", "🥈", "🥉"][rank] || `#${rank + 1}`;
  const bd = item.breakdown || {};
  const ltd = item.long_term_breakdown || {};
  const longScore = item.long_term_score ?? item.score;
  const livePrice = item.order_price ?? item.price;
  const entryPlan = item.entry_plan;
  const targetPrice = item.suggested_limit_price ?? entryPlan?.suggested_limit_price;
  const analysisPriceDiffers = Math.abs((livePrice ?? 0) - (item.price ?? 0)) > 0.001;

  return (
    <div className="rounded-xl border border-edge bg-panel2/40">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-panel2/70 rounded-xl"
      >
        <div className="w-8 text-center text-lg">{medal}</div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-bold">{item.ticker}</span>
            <span className="truncate text-sm text-muted">{item.name}</span>
          </div>
          <div className="mt-1.5 max-w-md">
            <ScoreBar score={longScore} max={maxScore} />
          </div>
        </div>
        <div className="hidden min-w-[150px] sm:flex flex-col items-end gap-1">
          <div className="text-sm font-semibold">
            <span className="text-xs font-normal text-muted">Live </span>
            {money(c, livePrice, 3)}
          </div>
          <div className="text-xs font-semibold text-accent">
            <span className="font-normal text-muted">Target </span>
            {targetPrice ? money(c, targetPrice, 3) : "—"}
          </div>
          <div className="text-xs text-muted">{item.pullback_pct?.toFixed(1)}% off high</div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <div className="text-lg font-extrabold text-accent">{longScore}</div>
          <div className="flex gap-1">
            {item.data_uncertain && <Pill tone="warn">⚠️ STALE?</Pill>}
            {item.long_term_label && <Pill tone={item.long_term_score >= 68 ? "good" : item.long_term_score >= 55 ? "warn" : "bad"}>{item.long_term_label}</Pill>}
            <Pill tone={item.is_golden ? "good" : "bad"}>{item.is_golden ? "GOLDEN" : "DEATH"}</Pill>
          </div>
        </div>
      </button>

      {open && (
        <div className="rise border-t border-edge px-4 py-4 text-sm">
          <div className="flex flex-wrap gap-2">
            <Pill tone="info">{item.sector?.toUpperCase()}</Pill>
            <Pill tone={timingTone(item.timing_verdict)}>{item.timing_verdict}</Pill>
            <Pill tone={healthTone(item.eh_label)}>{item.eh_label}</Pill>
            <Pill tone="muted">Yield {item.div_yield?.toFixed(2)}%</Pill>
            <Pill tone="muted">PE {item.pe_ratio?.toFixed(1)}</Pill>
            <Pill tone="muted">RSI {item.rsi?.toFixed(0)}</Pill>
            <Pill tone="muted">
              {item.months_since_buy == null ? "Never bought" : `Bought ${item.months_since_buy}m ago`}
            </Pill>
          </div>

          <div className="mt-3 rounded-lg border border-edge bg-ink/40 px-3 py-2 text-slate-200">
            💡 {item.reason}
          </div>

          <div className="mt-3 rounded-lg border border-accent/25 bg-accent/10 px-3 py-3">
            <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
              <div>
                <div className="uppercase tracking-wide text-muted">Live price</div>
                <div className="text-base font-bold text-slate-100">{money(c, livePrice, 3)}</div>
                <div className="text-muted">{item.price_source === "live_quote" ? "live quote" : "history close"}</div>
              </div>
              <div>
                <div className="uppercase tracking-wide text-muted">Target limit</div>
                <div className="text-base font-bold text-accent">{targetPrice ? money(c, targetPrice, 3) : "—"}</div>
                <div className="text-muted">{entryPlan?.label || "limit order"}</div>
              </div>
              <div>
                <div className="uppercase tracking-wide text-muted">Patient bid</div>
                <div className="text-base font-bold text-slate-100">
                  {entryPlan?.patient_limit_price ? money(c, entryPlan.patient_limit_price, 3) : "—"}
                </div>
                <div className="text-muted">lower fill chance</div>
              </div>
              <div>
                <div className="uppercase tracking-wide text-muted">Max chase</div>
                <div className="text-base font-bold text-slate-100">
                  {entryPlan?.max_entry_price ? money(c, entryPlan.max_entry_price, 3) : "—"}
                </div>
                <div className="text-muted">above this, wait</div>
              </div>
            </div>
            {entryPlan?.reason && <div className="mt-2 text-xs text-muted">{entryPlan.reason}</div>}
            {analysisPriceDiffers && (
              <div className="mt-1 text-xs text-muted">
                Analysis close {money(c, item.price, 3)}; execution uses the live quote.
              </div>
            )}
          </div>

          <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3 font-mono text-xs text-muted">
            {Object.entries(ltd).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span>LT {k}</span>
                <span className={v < 0 ? "text-bad" : v > 0 ? "text-accent" : ""}>{v > 0 ? `+${v}` : v}</span>
              </div>
            ))}
          </div>

          {item.long_term_flags?.length > 0 && (
            <div className="mt-2 space-y-1">
              {item.long_term_flags.map((f, i) => (
                <div key={`lf${i}`} className="text-xs text-warn/90">⚠ {f}</div>
              ))}
            </div>
          )}

          {item.data_uncertain && (
            <div className="mt-2 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-xs text-warn">
              ⚠️ {item.data_warning || "Price data looks stale — barred from the auto-pick until it agrees."}
            </div>
          )}

          {/* Legacy setup-score breakdown */}
          <div className="mt-3 text-[11px] uppercase tracking-wide text-muted">
            Legacy setup score {item.score} (effective {item.effective_score})
          </div>
          <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3 font-mono text-xs text-muted">
            {Object.entries(bd).map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span>{k}</span>
                <span className={v < 0 ? "text-bad" : v > 0 ? "text-accent" : ""}>{v > 0 ? `+${v}` : v}</span>
              </div>
            ))}
          </div>

          {/* Signals */}
          {(item.timing_green?.length > 0 || item.timing_yellow?.length > 0) && (
            <div className="mt-3 space-y-1">
              {item.timing_green?.map((g, i) => (
                <div key={`g${i}`} className="text-xs text-accent/90">✔ {g}</div>
              ))}
              {item.timing_yellow?.map((y, i) => (
                <div key={`y${i}`} className="text-xs text-warn/90">✘ {y}</div>
              ))}
            </div>
          )}
          {item.eh_warnings?.length > 0 && (
            <div className="mt-2 space-y-1">
              {item.eh_warnings.map((w, i) => (
                <div key={`w${i}`} className="text-xs text-slate-300">{w}</div>
              ))}
            </div>
          )}

          <div className="mt-3">
            <button
              onClick={() => onPick(item.ticker)}
              className="text-xs font-semibold text-accent2 hover:underline"
            >
              → Fit my budget to {item.ticker} instead
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
