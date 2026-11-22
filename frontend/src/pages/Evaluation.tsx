// Gold-set runs (eval/run_eval.py): summary per run, accuracy by category, failure taxonomy, per-question rows.
import { useEffect, useState } from "react";
import { api } from "../api";
import type { EvalRow, EvalSummary } from "../types";
import TraceView from "../components/TraceView";

const SUMMARY_COLS = ["n", "accuracy", "behavior_ok", "completion_rate", "cited_rate_policy", "retrieval_recall",
  "avg_tool_calls", "avg_llm_calls", "avg_prompt_tokens", "p50_latency_ms"];

function Bars({ data, fmt }: { data: [string, number][]; fmt: (v: number) => string }) {
  const max = Math.max(...data.map(([, v]) => v), 1e-9);
  return (
    <div className="bars">
      {data.map(([k, v]) => (
        <div className="bar-row" key={k}>
          <span className="bar-label">{k}</span>
          <span className="bar-track"><span className="bar" style={{ width: `${(v / max) * 100}%` }} /></span>
          <span className="bar-v">{fmt(v)}</span>
        </div>
      ))}
    </div>
  );
}

export default function Evaluation() {
  const [runs, setRuns] = useState<EvalSummary[] | null>(null);
  const [pick, setPick] = useState<string>("");
  const [rows, setRows] = useState<EvalRow[]>([]);
  const [onlyFail, setOnlyFail] = useState(false);
  const [trace, setTrace] = useState<string | null>(null);

  useEffect(() => {
    api.evalRuns().then((r) => { setRuns(r); if (r.length) setPick(r[0].run); }).catch(() => setRuns([]));
  }, []);
  useEffect(() => { if (pick) api.evalRun(pick).then(setRows).catch(() => setRows([])); setTrace(null); }, [pick]);

  if (!runs) return <div className="page muted">Loading…</div>;
  if (!runs.length) return <div className="page"><p className="muted">No runs yet. With the stack up: <code>python eval/run_eval.py --mode agent --gold data/queries/heldout_v1.jsonl</code></p></div>;

  const cats = new Map<string, { ok: number; n: number }>();
  rows.forEach((r) => { const c = cats.get(r.category) ?? { ok: 0, n: 0 }; c.n += 1; c.ok += r.correct ? 1 : 0; cats.set(r.category, c); });
  const fails = new Map<string, number>();
  rows.forEach((r) => { if (r.failure) fails.set(r.failure, (fails.get(r.failure) ?? 0) + 1); });
  const shown = onlyFail ? rows.filter((r) => !r.correct) : rows;

  return (
    <div className="page">
      <h2>Gold-set evaluation</h2>
      <div className="table-wrap">
        <table className="grid clickable small">
          <thead><tr><th>run</th>{SUMMARY_COLS.map((c) => <th key={c}>{c}</th>)}<th>failures</th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run} className={r.run === pick ? "selected" : ""} onClick={() => setPick(r.run)}>
                <td className="nowrap">{r.run}</td>
                {SUMMARY_COLS.map((c) => <td key={c}>{r[c] === undefined ? "" : String(r[c])}</td>)}
                <td>{Object.entries(r.failures ?? {}).map(([k, v]) => `${k}:${v}`).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="two-col">
        <div className="panel">
          <h3>Accuracy by category</h3>
          <Bars data={[...cats].map(([k, c]) => [k, c.ok / c.n])} fmt={(v) => `${Math.round(v * 100)}%`} />
        </div>
        <div className="panel">
          <h3>Failure taxonomy</h3>
          {fails.size ? <Bars data={[...fails]} fmt={String} /> : <p className="muted">No failures.</p>}
        </div>
      </div>

      <div className="row">
        <h3>Questions</h3>
        <label className="muted"><input type="checkbox" checked={onlyFail} onChange={(e) => setOnlyFail(e.target.checked)} /> only failures</label>
      </div>
      <div className="table-wrap">
        <table className="grid small">
          <thead><tr><th>qid</th><th>category</th><th>ok</th><th>failure</th><th>tools</th><th>question</th><th>answer</th><th>gold</th><th /></tr></thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.qid}>
                <td>{r.qid}</td><td>{r.category}</td><td>{r.correct ? "✅" : "❌"}</td><td>{r.failure ?? ""}</td>
                <td>{(r.tools_called ?? []).join(", ")}</td><td>{r.question}</td><td>{r.answer}</td><td>{r.gold_answer}</td>
                <td>{r.turn_id && <button className="link" onClick={() => setTrace(r.turn_id!)}>trace</button>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {trace && <div className="panel"><h3>Trace</h3><TraceView turnId={trace} waterfall /></div>}
    </div>
  );
}
