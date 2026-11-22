// Retrieval experiments (rag/experiments.py): recall@k per chunking x retriever, measured apart from answer quality.
import { useEffect, useState } from "react";
import { api } from "../api";
import type { RetrievalRun } from "../types";

const METRICS = ["sec@1", "sec@5", "cl@5", "fact@5", "MRR", "sec@10", "ms/query"];

export default function Retrieval() {
  const [runs, setRuns] = useState<RetrievalRun[] | null>(null);
  const [pick, setPick] = useState(0);
  const [metric, setMetric] = useState("sec@5");

  useEffect(() => { api.retrieval().then(setRuns).catch(() => setRuns([])); }, []);
  if (!runs) return <div className="page muted">Loading…</div>;
  if (!runs.length) return <div className="page"><p className="muted">No retrieval experiments yet. Run <code>rag/experiments.py</code>.</p></div>;

  const rows = runs[pick].rows;
  const lowerIsBetter = metric === "ms/query";
  const num = (r: Record<string, string | number>) => Number(r[metric] ?? NaN);
  const sorted = [...rows].sort((a, b) => (lowerIsBetter ? num(a) - num(b) : num(b) - num(a)));
  const cols = Object.keys(rows[0] ?? {});

  // heatmap: chunking (rows) x retriever/query (columns), mean of the metric
  const ys = [...new Set(rows.map((r) => String(r.chunking)))];
  const xs = [...new Set(rows.map((r) => `${r.retriever} / ${r.query}`))];
  const cell = (y: string, x: string) => {
    const v = rows.filter((r) => String(r.chunking) === y && `${r.retriever} / ${r.query}` === x).map(num).filter((n) => !isNaN(n));
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
  };
  const vals = rows.map(num).filter((n) => !isNaN(n));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const shade = (v: number) => {
    if (isNaN(v)) return "transparent";
    let t = hi > lo ? (v - lo) / (hi - lo) : 1;
    if (lowerIsBetter) t = 1 - t;
    return `color-mix(in srgb, var(--accent) ${Math.round(10 + t * 70)}%, var(--bg))`;
  };

  return (
    <div className="page">
      <h2>Retrieval experiments</h2>
      <div className="row">
        <select value={pick} onChange={(e) => setPick(Number(e.target.value))}>
          {runs.map((r, i) => <option key={r.run} value={i}>{r.run}</option>)}
        </select>
        <select value={metric} onChange={(e) => setMetric(e.target.value)}>
          {METRICS.filter((m) => cols.includes(m)).map((m) => <option key={m}>{m}</option>)}
        </select>
      </div>
      <div className="panel">
        <h3>{metric} by chunking × retriever</h3>
        <div className="table-wrap">
          <table className="grid heat">
            <thead><tr><th />{xs.map((x) => <th key={x}>{x}</th>)}</tr></thead>
            <tbody>
              {ys.map((y) => (
                <tr key={y}><th>{y}</th>{xs.map((x) => { const v = cell(y, x); return <td key={x} style={{ background: shade(v) }}>{isNaN(v) ? "" : v.toFixed(2)}</td>; })}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="table-wrap">
        <table className="grid small">
          <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>{sorted.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c}>{String(r[c])}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </div>
  );
}
