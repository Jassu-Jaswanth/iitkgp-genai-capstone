// Every answer with latency / tokens / tools; click one for its waterfall and full trace.
import { useEffect, useState } from "react";
import { api } from "../api";
import type { TurnRow } from "../types";
import TraceView from "../components/TraceView";

const quantile = (xs: number[], q: number) => {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};

export default function Traces() {
  const [rows, setRows] = useState<TurnRow[] | null>(null);
  const [sel, setSel] = useState<TurnRow | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { api.turns(200).then(setRows).catch((e: Error) => setErr(e.message)); }, []);

  if (err) return <div className="page"><div className="alert">{err}</div></div>;
  if (!rows) return <div className="page muted">Loading…</div>;
  if (!rows.length) return <div className="page"><p className="muted">No answers yet. Ask something on the Chat page.</p></div>;

  const lat = rows.map((r) => (r.latency_ms ?? 0) / 1000);
  const tokens = rows.map((r) => (r.usage?.prompt_tokens ?? 0) + (r.usage?.completion_tokens ?? 0));
  const fresh = rows.filter((r) => !r.cached_from);
  const kpis: [string, string][] = [
    ["Answers", String(rows.length)],
    ["Median latency", `${quantile(lat, 0.5).toFixed(1)} s`],
    ["p90 latency", `${quantile(lat, 0.9).toFixed(1)} s`],
    ["Avg tokens / fresh answer", fresh.length ? String(Math.round(fresh.reduce((a, r) => a + (r.usage?.prompt_tokens ?? 0) + (r.usage?.completion_tokens ?? 0), 0) / fresh.length)) : "-"],
    ["From cache", String(rows.length - fresh.length)],
    ["Not completed", String(rows.filter((r) => r.status !== "completed").length)],
  ];

  return (
    <div className="page">
      <h2>Traces</h2>
      <div className="kpis">
        {kpis.map(([k, v]) => <div className="kpi" key={k}><div className="muted">{k}</div><div className="kpi-v">{v}</div></div>)}
      </div>
      <p className="muted">Click a row to open its trace.</p>
      <div className="table-wrap">
        <table className="grid clickable">
          <thead><tr><th>time</th><th>question</th><th>route</th><th>latency</th><th>model calls</th><th>tokens</th><th>tools</th><th>status</th></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.turn_id} className={sel?.turn_id === r.turn_id ? "selected" : ""} onClick={() => setSel(r)}>
                <td className="nowrap">{new Date(r.created_at).toLocaleString()}</td>
                <td>{r.question}</td>
                <td>{r.route}</td>
                <td>{lat[i].toFixed(1)} s</td>
                <td>{r.llm_calls}</td>
                <td>{tokens[i]}</td>
                <td>{(r.tools ?? []).join(", ")}</td>
                <td>{r.cached_from ? "cached" : r.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sel && (
        <div className="panel">
          <h3>{sel.question}</h3>
          <TraceView turnId={sel.turn_id} waterfall />
        </div>
      )}
    </div>
  );
}
