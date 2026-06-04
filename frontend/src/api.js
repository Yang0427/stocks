// Thin fetch wrapper around the FastAPI backend. All paths go through /api,
// which Vite proxies to http://127.0.0.1:8000 in dev and FastAPI serves itself
// in production.

async function jget(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

async function jpost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${path} → ${r.status}`);
  return data;
}

async function jdelete(path) {
  const r = await fetch(path, { method: "DELETE" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${path} → ${r.status}`);
  return data;
}

export const api = {
  config: () => jget("/api/config"),
  health: () => jget("/api/health"),

  startAnalyze: (useCache = true, ttlHours = 6) =>
    jpost("/api/analyze", { use_cache: useCache, ttl_hours: ttlHours }),
  startRefresh: () => jpost("/api/refresh", {}),
  job: (id) => jget(`/api/job/${id}`),

  budget: (budget, opts = {}) =>
    jpost("/api/budget", {
      budget,
      only_actionable: opts.onlyActionable ?? true,
      prefer: opts.prefer ?? null,
    }),

  holdings: () => jget("/api/holdings"),

  getBuyLog: () => jget("/api/log/buy"),
  getSellLog: () => jget("/api/log/sell"),
  logBuy: (entry) => jpost("/api/log/buy", entry),
  logSell: (entry) => jpost("/api/log/sell", entry),

  getWatchlist: () => jget("/api/watchlist"),
  addTicker: (ticker, sector) => jpost("/api/watchlist", { ticker, sector }),
  removeTicker: (ticker) => jdelete(`/api/watchlist/${encodeURIComponent(ticker)}`),
};

// Poll a job until it finishes; calls onTick(job) on each poll.
export async function pollJob(id, onTick, intervalMs = 700) {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const job = await api.job(id);
    if (onTick) onTick(job);
    if (job.status === "done") return job;
    if (job.status === "error") throw new Error(job.error || "Job failed");
    await new Promise((res) => setTimeout(res, intervalMs));
  }
}
