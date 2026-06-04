// Small shared presentational helpers used across the app.

export function Card({ children, className = "" }) {
  return (
    <div
      className={`rounded-2xl border border-edge bg-panel/70 backdrop-blur-sm shadow-lg shadow-black/20 ${className}`}
    >
      {children}
    </div>
  );
}

export function Pill({ children, tone = "muted" }) {
  const tones = {
    good: "bg-accent/15 text-accent border-accent/30",
    warn: "bg-warn/15 text-warn border-warn/30",
    bad: "bg-bad/15 text-bad border-bad/30",
    muted: "bg-panel2 text-muted border-edge",
    info: "bg-accent2/15 text-accent2 border-accent2/30",
  };
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Button({ children, onClick, variant = "primary", disabled, className = "", type = "button" }) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed";
  const variants = {
    primary: "bg-accent text-ink hover:bg-accent/90 shadow-lg shadow-accent/20",
    ghost: "bg-panel2 text-slate-200 hover:bg-edge border border-edge",
    info: "bg-accent2 text-ink hover:bg-accent2/90",
    danger: "bg-bad text-ink hover:bg-bad/90",
  };
  return (
    <button type={type} onClick={onClick} disabled={disabled} className={`${base} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
}

export function ScoreBar({ score, max = 117 }) {
  const pct = Math.max(0, Math.min(100, (score / max) * 100));
  const color = pct >= 70 ? "bg-accent" : pct >= 45 ? "bg-accent2" : pct >= 25 ? "bg-warn" : "bg-bad";
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-panel2">
      <div className={`h-full rounded-full ${color} transition-all`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Stat({ label, value, sub, tone }) {
  const toneCls =
    tone === "good" ? "text-accent" : tone === "bad" ? "text-bad" : tone === "warn" ? "text-warn" : "text-slate-100";
  return (
    <div className="rounded-xl border border-edge bg-panel2/60 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className={`text-lg font-bold ${toneCls}`}>{value}</div>
      {sub && <div className="text-[11px] text-muted">{sub}</div>}
    </div>
  );
}

export function money(currency, n, dp = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return `${currency} ${Number(n).toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp })}`;
}

export function timingTone(verdict) {
  if (verdict?.includes("GOOD")) return "good";
  if (verdict?.includes("CAUTION")) return "warn";
  return "bad";
}

export function healthTone(label) {
  if (!label) return "muted";
  if (label.includes("IMPROVING") || label.includes("STABLE")) return "good";
  if (label.includes("MIXED")) return "warn";
  return "bad";
}
