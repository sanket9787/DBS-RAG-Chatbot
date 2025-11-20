"use client";

import { useEffect, useState } from "react";
import type { HealthResponse, StatsResponse } from "@/lib/api";
import { fetchHealth, fetchStats } from "@/lib/api";

export function SystemStatusCard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const [healthResp, statsResp] = await Promise.all([
        fetchHealth(),
        fetchStats(),
      ]);
      setHealth(healthResp);
      setStats(statsResp);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to reach backend");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-lg shadow-sky-500/5">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-slate-400">System Status</div>
        <button
          type="button"
          onClick={load}
          className="text-xs text-slate-400 underline-offset-2 hover:text-slate-200"
        >
          Refresh
        </button>
      </div>
      <div className="mt-4 space-y-3 text-sm">
        <div className="flex items-center justify-between rounded-2xl bg-slate-900/60 px-4 py-3">
          <span className="text-slate-300">Backend</span>
          <span
            className={
              loading
                ? "text-amber-300"
                : health
                ? "text-emerald-300"
                : "text-rose-300"
            }
          >
            {loading ? "Checking…" : health ? health.status : "Offline"}
          </span>
        </div>
        <div className="flex items-center justify-between rounded-2xl bg-slate-900/60 px-4 py-3">
          <span className="text-slate-300">Knowledge Base</span>
          <span className="text-sky-300">
            {stats ? `${stats.total_documents} docs` : loading ? "…" : "—"}
          </span>
        </div>
        <div className="flex items-center justify-between rounded-2xl bg-slate-900/60 px-4 py-3">
          <span className="text-slate-300">LLM</span>
          <span className="text-slate-100">GPT-4 Turbo</span>
        </div>
      </div>
      {error && (
        <p className="mt-3 rounded-2xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-100">
          {error}
        </p>
      )}
    </div>
  );
}

