import { useEffect, useState, useCallback } from "react";
import { api, pollJob } from "./api";
import { Card, Button, Pill } from "./components/ui";
import BestBuy from "./components/BestBuy";
import StockRow from "./components/StockRow";
import Holdings from "./components/Holdings";
import LogForm from "./components/LogForm";
import Watchlist from "./components/Watchlist";

export default function App() {
  const [budget, setBudget] = useState(1000);
  const [onlyActionable, setOnlyActionable] = useState(true);

  const [analysis, setAnalysis] = useState(null);   // full /analyze result
  const [pick, setPick] = useState(null);           // budget_fit result
  const [holdings, setHoldings] = useState(null);

  const [tab, setTab] = useState("pick");           // pick | watchlist | holdings
  const [job, setJob] = useState(null);             // {message, done, total}
  const [running, setRunning] = useState(false);
  const [pickLoading, setPickLoading] = useState(false);
  const [cacheInfo, setCacheInfo] = useState(null);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [logModal, setLogModal] = useState(null);   // {kind, prefill}

  const flash = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(""), 3000);
  };

  // Load config / cache freshness on mount.
  useEffect(() => {
    api.config().then((c) => setCacheInfo(c.cache)).catch(() => {});
  }, []);

  const refreshBudget = useCallback(
    async (prefer = null) => {
      setPickLoading(true);
      try {
        const r = await api.budget(Number(budget), { onlyActionable, prefer });
        setPick(r.pick);
        if (prefer) setTab("pick");
      } catch (e) {
        setError(e.message);
      } finally {
        setPickLoading(false);
      }
    },
    [budget, onlyActionable]
  );

  async function runAnalysis(useCache = true) {
    setError("");
    setRunning(true);
    setJob({ message: "Starting…", done: 0, total: 0 });
    try {
      const { job_id } = await api.startAnalyze(useCache);
      const done = await pollJob(job_id, (j) => setJob(j));
      setAnalysis(done.result);
      // Immediately compute the budget pick from the fresh analysis.
      const r = await api.budget(Number(budget), { onlyActionable });
      setPick(r.pick);
      api.config().then((c) => setCacheInfo(c.cache)).catch(() => {});
      flash("Analysis ready ✅");
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
      setJob(null);
    }
  }

  async function refreshData() {
    setError("");
    setRunning(true);
    setJob({ message: "Refreshing market data…", done: 0, total: 0 });
    try {
      const { job_id } = await api.startRefresh();
      await pollJob(job_id, (j) => setJob(j));
      flash("Market data refreshed — now re-run analysis");
      api.config().then((c) => setCacheInfo(c.cache)).catch(() => {});
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
      setJob(null);
    }
  }

  async function loadHoldings() {
    setHoldings(null);
    try {
      setHoldings(await api.holdings());
    } catch (e) {
      setError(e.message);
    }
  }

  async function submitLog(entry) {
    if (logModal.kind === "buy") await api.logBuy(entry);
    else await api.logSell(entry);
    flash(`${logModal.kind === "buy" ? "Buy" : "Sell"} logged ✅`);
    if (tab === "holdings") loadHoldings();
  }

  const progressPct = job && job.total ? Math.round((job.done / job.total) * 100) : null;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      {/* Header */}
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight">🎯 Savings Scout</h1>
          <p className="text-sm text-muted">
            Tell it your budget. It tells you exactly what to buy this month — no overthinking.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {cacheInfo?.oldest_hours != null && (
            <Pill tone={cacheInfo.oldest_hours > 24 ? "warn" : "muted"}>
              data ~{cacheInfo.oldest_hours.toFixed(0)}h old
            </Pill>
          )}
          <Button variant="ghost" onClick={refreshData} disabled={running}>↻ Refresh data</Button>
        </div>
      </header>

      {/* Budget bar */}
      <Card className="mb-6 p-5">
        <div className="flex flex-wrap items-end gap-4">
          <label className="flex-1 min-w-[200px]">
            <span className="mb-1 block text-[11px] uppercase tracking-wide text-muted">Monthly budget</span>
            <div className="flex items-center gap-2 rounded-xl border border-edge bg-ink/60 px-3 py-2">
              <span className="text-muted">RM</span>
              <input
                type="number"
                min="0"
                step="50"
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                className="w-full bg-transparent text-xl font-bold outline-none"
              />
            </div>
          </label>
          <label className="flex items-center gap-2 pb-2 text-sm text-muted">
            <input
              type="checkbox"
              checked={onlyActionable}
              onChange={(e) => setOnlyActionable(e.target.checked)}
              className="h-4 w-4 accent-[#34d399]"
            />
            Only actionable (investment-grade, golden cross, not WAIT, healthy)
          </label>
          <div className="flex gap-2">
            <Button onClick={() => runAnalysis(true)} disabled={running}>
              {running ? "Working…" : analysis ? "Re-run analysis" : "🚀 Run analysis"}
            </Button>
            {analysis && (
              <Button variant="info" onClick={() => refreshBudget()} disabled={pickLoading}>
                Recompute pick
              </Button>
            )}
          </div>
        </div>

        {job && (
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-xs text-muted">
              <span>{job.message}</span>
              {progressPct != null && <span>{progressPct}%</span>}
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-panel2">
              <div className="h-full rounded-full bg-accent2 transition-all" style={{ width: `${progressPct ?? 15}%` }} />
            </div>
          </div>
        )}
        {error && <div className="mt-4 rounded-lg bg-bad/15 px-3 py-2 text-sm text-bad">{error}</div>}
      </Card>

      {/* Tabs */}
      <div className="mb-4 flex gap-2">
        <TabButton active={tab === "pick"} onClick={() => setTab("pick")}>This month's buy</TabButton>
        <TabButton active={tab === "watchlist"} onClick={() => setTab("watchlist")}>
          Watchlist {analysis ? `(${analysis.results.length})` : ""}
        </TabButton>
        <TabButton active={tab === "holdings"} onClick={() => { setTab("holdings"); if (!holdings) loadHoldings(); }}>
          Holdings
        </TabButton>
        <TabButton active={tab === "watchlist-edit"} onClick={() => setTab("watchlist-edit")}>
          Edit stocks
        </TabButton>
        <div className="ml-auto flex gap-2">
          <Button variant="ghost" onClick={() => setLogModal({ kind: "buy" })}>+ Log buy</Button>
          <Button variant="ghost" onClick={() => setLogModal({ kind: "sell" })}>− Log sell</Button>
        </div>
      </div>

      {/* Content */}
      {tab === "pick" && (
        <div className="space-y-4">
          {!analysis && !running && (
            <Card className="p-8 text-center text-muted">
              Set your budget and hit <span className="text-accent font-semibold">Run analysis</span> to get this month's pick.
            </Card>
          )}
          {(analysis || running) && (
            <BestBuy
              pick={pick}
              loading={running || pickLoading}
              onLog={(p) =>
                setLogModal({
                  kind: "buy",
                  prefill: {
                    ticker: p.ticker,
                    price: p.suggested_limit_price ?? p.price,
                    lots: p.lots,
                    currency: p.currency,
                  },
                })
              }
            />
          )}
        </div>
      )}

      {tab === "watchlist" && (
        <div className="space-y-2">
          {!analysis ? (
            <Card className="p-8 text-center text-muted">Run an analysis to populate the watchlist.</Card>
          ) : (
            analysis.results.map((item, i) => (
              <StockRow
                key={item.ticker}
                item={item}
                rank={i}
                maxScore={analysis.max_long_term_score || 100}
                onPick={(t) => refreshBudget(t)}
              />
            ))
          )}
        </div>
      )}

      {tab === "holdings" && <Holdings data={holdings} loading={holdings === null} />}

      {tab === "watchlist-edit" && (
        <Watchlist
          flash={flash}
          onChanged={() => api.config().then((c) => setCacheInfo(c.cache)).catch(() => {})}
        />
      )}

      {logModal && (
        <LogForm kind={logModal.kind} prefill={logModal.prefill} onClose={() => setLogModal(null)} onSubmit={submitLog} />
      )}

      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 rounded-xl border border-edge bg-panel px-4 py-2 text-sm shadow-xl rise">
          {toast}
        </div>
      )}

      <footer className="mt-10 text-center text-xs text-muted">
        Same scoring brain as <code className="rounded bg-panel2 px-1">main.py</code> · not financial advice.
      </footer>
    </div>
  );
}

function TabButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${
        active ? "bg-panel2 text-slate-100 border border-edge" : "text-muted hover:text-slate-200"
      }`}
    >
      {children}
    </button>
  );
}
