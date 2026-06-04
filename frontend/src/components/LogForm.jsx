import { useEffect, useState } from "react";
import { Button, Pill } from "./ui";

// Modal form for logging a BUY or SELL. You fill in the actual fill price /
// lots / month (MooMoo limit orders fill later), and the app structures it into
// the exact JSON shape purchase_log.json / sell_log.json expect.
function currencyFor(ticker) {
  const t = (ticker || "").toUpperCase();
  if (t.endsWith(".KL")) return "RM";
  if (t.endsWith(".HK")) return "HKD";
  return "USD";
}

function thisMonth() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
}

export default function LogForm({ kind, prefill, onClose, onSubmit }) {
  const [ticker, setTicker] = useState(prefill?.ticker || "");
  const [price, setPrice] = useState(prefill?.price ?? "");
  const [lots, setLots] = useState(prefill?.lots ?? 1);
  const [month, setMonth] = useState(prefill?.month || thisMonth());
  const [currency, setCurrency] = useState(prefill?.currency || currencyFor(prefill?.ticker));
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  // Keep currency in sync with the ticker suffix unless user overrides.
  useEffect(() => {
    setCurrency(currencyFor(ticker));
  }, [ticker]);

  const isBuy = kind === "buy";
  const units = (ticker.toUpperCase().endsWith(".KL") ? 100 : 1) * (Number(lots) || 0);
  const estimate = (Number(price) || 0) * units;

  async function submit(e) {
    e.preventDefault();
    setErr("");
    if (!ticker || !price || !lots || !month) {
      setErr("Fill in ticker, price, lots and month.");
      return;
    }
    setBusy(true);
    try {
      await onSubmit({
        month,
        ticker: ticker.toUpperCase().trim(),
        price: Number(price),
        currency,
        lots: Number(lots),
      });
      onClose();
    } catch (e2) {
      setErr(e2.message || "Failed to save.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={onClose}>
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={submit}
        className="rise w-full max-w-md rounded-2xl border border-edge bg-panel p-6 shadow-2xl"
      >
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-bold">
            {isBuy ? "📝 Log a Buy" : "💸 Log a Sell"}
          </h3>
          <Pill tone={isBuy ? "good" : "warn"}>{isBuy ? "purchase_log.json" : "sell_log.json"}</Pill>
        </div>
        <p className="mt-1 text-xs text-muted">
          Enter the real fill once your MooMoo order executes. Units shown assume 100/lot for Bursa (.KL).
        </p>

        <div className="mt-4 grid grid-cols-2 gap-3">
          <Field label="Ticker" full>
            <input
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="5176.KL"
              className={inputCls}
            />
          </Field>
          <Field label={`Fill price (${currency}/share)`}>
            <input
              type="number"
              step="0.0001"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              placeholder="2.300"
              className={inputCls}
            />
          </Field>
          <Field label={ticker.toUpperCase().endsWith(".KL") ? "Lots (×100)" : "Shares"}>
            <input
              type="number"
              min="1"
              value={lots}
              onChange={(e) => setLots(e.target.value)}
              className={inputCls}
            />
          </Field>
          <Field label="Month">
            <input
              type="month"
              value={month}
              onChange={(e) => setMonth(e.target.value)}
              className={inputCls}
            />
          </Field>
          <Field label="Currency">
            <select value={currency} onChange={(e) => setCurrency(e.target.value)} className={inputCls}>
              <option>RM</option>
              <option>USD</option>
              <option>HKD</option>
            </select>
          </Field>
        </div>

        <div className="mt-4 rounded-xl border border-edge bg-panel2/50 px-4 py-3 text-sm">
          <div className="flex justify-between">
            <span className="text-muted">Units</span>
            <span className="font-mono">{units}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted">Gross value</span>
            <span className="font-mono">
              {currency} {estimate.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </span>
          </div>
        </div>

        {err && <div className="mt-3 rounded-lg bg-bad/15 px-3 py-2 text-sm text-bad">{err}</div>}

        <div className="mt-5 flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button type="submit" variant={isBuy ? "primary" : "danger"} disabled={busy}>
            {busy ? "Saving…" : isBuy ? "Save buy" : "Save sell"}
          </Button>
        </div>
      </form>
    </div>
  );
}

const inputCls =
  "w-full rounded-lg border border-edge bg-ink/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-accent2";

function Field({ label, children, full }) {
  return (
    <label className={`block ${full ? "col-span-2" : ""}`}>
      <span className="mb-1 block text-[11px] uppercase tracking-wide text-muted">{label}</span>
      {children}
    </label>
  );
}
