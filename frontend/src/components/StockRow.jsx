import { useState } from "react";
import { Pill, ScoreBar, money, timingTone, healthTone } from "./ui";

// One stock in the ranked watchlist. Collapsed by default; expands to show the
// full score breakdown + timing/earnings signals (the detail the CLI prints).
export default function StockRow({ item, rank, maxScore, onPick }) {
  const [open, setOpen] = useState(false);
  const c = item.currency;
  const medal = ["🥇", "🥈", "🥉"][rank] || `#${rank + 1}`;
  const bd = item.breakdown || {};

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
            <ScoreBar score={item.score} max={maxScore} />
          </div>
        </div>
        <div className="hidden sm:flex flex-col items-end gap-1">
          <div className="text-sm font-semibold">{money(c, item.price, 3)}</div>
          <div className="text-xs text-muted">{item.pullback_pct?.toFixed(1)}% off high</div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <div className="text-lg font-extrabold text-accent">{item.score}</div>
          <div className="flex gap-1">
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

          {/* Score breakdown */}
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
