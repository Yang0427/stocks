import { useEffect, useState } from "react";
import { api } from "../api";
import { Card, Pill, Button } from "./ui";

// View / add / delete the tickers in stocks.json. After editing, the user
// should re-run analysis to pick up the change.
export default function Watchlist({ onChanged, flash }) {
  const [tickers, setTickers] = useState([]);
  const [sectors, setSectors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newTicker, setNewTicker] = useState("");
  const [newSector, setNewSector] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [confirmDel, setConfirmDel] = useState(null);

  async function load() {
    setLoading(true);
    try {
      const w = await api.getWatchlist();
      setTickers(w.tickers || []);
      setSectors(w.known_sectors || []);
      if (!newSector && w.known_sectors?.length) setNewSector(w.known_sectors[0]);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function add(e) {
    e.preventDefault();
    setErr("");
    if (!newTicker.trim()) {
      setErr("Enter a ticker symbol.");
      return;
    }
    setBusy(true);
    try {
      const r = await api.addTicker(newTicker.trim(), newSector);
      setTickers(r.tickers);
      setNewTicker("");
      flash?.(`Added ${newTicker.trim().toUpperCase()} ✅ — re-run analysis to include it`);
      onChanged?.();
    } catch (e2) {
      setErr(e2.message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(ticker) {
    setErr("");
    setBusy(true);
    try {
      const r = await api.removeTicker(ticker);
      setTickers(r.tickers);
      setConfirmDel(null);
      flash?.(`Removed ${ticker}`);
      onChanged?.();
    } catch (e2) {
      setErr(e2.message);
    } finally {
      setBusy(false);
    }
  }

  // Group by sector for a tidy view.
  const grouped = tickers.reduce((acc, t) => {
    const s = (t.sector || "general").toLowerCase();
    (acc[s] ||= []).push(t);
    return acc;
  }, {});

  return (
    <div className="space-y-4">
      {/* Add form */}
      <Card className="p-5">
        <div className="mb-3 text-sm font-semibold">➕ Add a stock to your watchlist</div>
        <form onSubmit={add} className="flex flex-wrap items-end gap-3">
          <label className="flex-1 min-w-[160px]">
            <span className="mb-1 block text-[11px] uppercase tracking-wide text-muted">
              Ticker (e.g. 5176.KL, NVDA, 9988.HK)
            </span>
            <input
              value={newTicker}
              onChange={(e) => setNewTicker(e.target.value)}
              placeholder="5176.KL"
              className="w-full rounded-lg border border-edge bg-ink/60 px-3 py-2 text-sm outline-none focus:border-accent2"
            />
          </label>
          <label className="min-w-[160px]">
            <span className="mb-1 block text-[11px] uppercase tracking-wide text-muted">Sector</span>
            <select
              value={newSector}
              onChange={(e) => setNewSector(e.target.value)}
              className="w-full rounded-lg border border-edge bg-ink/60 px-3 py-2 text-sm outline-none focus:border-accent2"
            >
              {sectors.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </label>
          <Button type="submit" disabled={busy}>{busy ? "Adding…" : "Add"}</Button>
        </form>
        {err && <div className="mt-3 rounded-lg bg-bad/15 px-3 py-2 text-sm text-bad">{err}</div>}
      </Card>

      {/* List grouped by sector */}
      {loading ? (
        <Card className="p-6 text-center text-muted animate-pulse">Loading watchlist…</Card>
      ) : tickers.length === 0 ? (
        <Card className="p-6 text-center text-muted">No stocks yet. Add one above.</Card>
      ) : (
        <Card className="p-5">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-sm font-semibold">📋 {tickers.length} stocks tracked</div>
            <span className="text-xs text-muted">Edit, then re-run analysis</span>
          </div>
          <div className="space-y-4">
            {Object.entries(grouped)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([sector, list]) => (
                <div key={sector}>
                  <div className="mb-2">
                    <Pill tone="info">{sector.toUpperCase()} · {list.length}</Pill>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {list.map((t) => {
                      const tk = t.ticker;
                      const isConfirm = confirmDel === tk;
                      return (
                        <div
                          key={tk}
                          className="flex items-center gap-2 rounded-lg border border-edge bg-panel2/50 py-1.5 pl-3 pr-1.5"
                        >
                          <span className="font-mono text-sm">{tk}</span>
                          {isConfirm ? (
                            <>
                              <button
                                onClick={() => remove(tk)}
                                disabled={busy}
                                className="rounded-md bg-bad px-2 py-0.5 text-xs font-semibold text-ink hover:bg-bad/90"
                              >
                                Delete
                              </button>
                              <button
                                onClick={() => setConfirmDel(null)}
                                className="rounded-md px-1.5 py-0.5 text-xs text-muted hover:text-slate-200"
                              >
                                ✕
                              </button>
                            </>
                          ) : (
                            <button
                              onClick={() => setConfirmDel(tk)}
                              title={`Remove ${tk}`}
                              className="rounded-md px-1.5 py-0.5 text-xs text-muted hover:bg-bad/20 hover:text-bad"
                            >
                              🗑
                            </button>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ))}
          </div>
        </Card>
      )}
    </div>
  );
}
