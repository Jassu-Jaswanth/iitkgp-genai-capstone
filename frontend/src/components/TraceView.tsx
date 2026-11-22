// The LLM trace of one answer: every model call, tool call and search, from the question arriving to the answer.
import { useEffect, useState } from "react";
import { api } from "../api";
import type { TraceStep } from "../types";
import Json from "./Json";

const NODE: Record<string, string> = { router: "Router", planner: "Planner", tools: "Tools", critic: "Critic" };
const ICON: Record<TraceStep["kind"], string> = { llm: "🧠", tool: "🛠️", retriever: "🔎" };

const ms = (s: TraceStep) => (s.ended_at ? Date.parse(s.ended_at) - Date.parse(s.started_at) : 0);

function depth(s: TraceStep, bySeq: Map<number, TraceStep>): number {
  let d = 0;
  let p = s.parent_seq;
  while (p !== null && bySeq.has(p)) {
    d += 1;
    p = bySeq.get(p)!.parent_seq;
  }
  return d;
}

interface ToolCall { name: string; args: unknown }
interface LlmOut { content?: unknown; tool_calls?: ToolCall[] }
interface ChatMsg { role: string; content: unknown; name?: string; tool_calls?: ToolCall[] }
interface Chunk { citation?: string; score?: number; clause_ids?: string[]; text?: string }

function title(s: TraceStep): string {
  const node = NODE[s.node ?? ""] ?? s.node ?? "";
  if (s.kind === "llm") {
    const out = s.output as LlmOut | null;
    const calls = out?.tool_calls ?? [];
    const what = calls.length ? "→ " + calls.map((c) => c.name).join(", ") : "→ answer";
    const tok = s.prompt_tokens !== null ? ` · ${s.prompt_tokens}→${s.completion_tokens} tok` : "";
    return `${ICON.llm} ${node} model call ${what}${tok}`;
  }
  if (s.kind === "tool") return `${ICON.tool} ${s.name}`;
  const n = Array.isArray(s.output) ? s.output.length : 0;
  return `${ICON.retriever} search · ${n} chunks`;
}

function Output({ s }: { s: TraceStep }) {
  if (s.kind === "llm") {
    const out = s.output as LlmOut | null;
    if (out?.tool_calls?.length) return <Json value={out.tool_calls.length === 1 ? out.tool_calls[0] : out.tool_calls} />;
    return <div className="prose">{String(out?.content ?? "")}</div>;
  }
  if (s.kind === "retriever" && Array.isArray(s.output)) {
    const chunks = s.output as Chunk[];
    return (
      <>
        <table className="grid small">
          <thead><tr><th>#</th><th>citation</th><th>score</th><th>clauses</th></tr></thead>
          <tbody>
            {chunks.map((c, i) => (
              <tr key={i}><td>{i + 1}</td><td>{c.citation}</td><td>{c.score?.toFixed(3)}</td><td>{c.clause_ids?.length ?? 0}</td></tr>
            ))}
          </tbody>
        </table>
        <details><summary>Chunk text</summary><Json value={chunks} maxLines={60} /></details>
      </>
    );
  }
  return <Json value={s.output} />;
}

function Input({ s }: { s: TraceStep }) {
  if (s.kind === "llm" && Array.isArray(s.input)) {
    return (
      <div className="messages">
        {(s.input as ChatMsg[]).map((m, i) => (
          <div key={i} className="msg">
            <div className="role">{m.role}{m.name ? ` · ${m.name}` : ""}</div>
            {m.tool_calls?.length ? <Json value={m.tool_calls} maxLines={12} /> : null}
            {typeof m.content === "string" ? <pre className="wrap">{m.content.slice(0, 4000)}</pre> : m.content ? <Json value={m.content} maxLines={12} /> : null}
          </div>
        ))}
      </div>
    );
  }
  return <Json value={s.input} />;
}

function Step({ s, pad }: { s: TraceStep; pad: number }) {
  const [tab, setTab] = useState<"out" | "in">("out");
  return (
    <details className={`step ${s.kind}${s.error ? " error" : ""}`} style={{ marginLeft: pad * 16 }}>
      <summary>
        <span>{title(s)}</span>
        <span className="muted">{ms(s)} ms{s.error ? " ⚠️" : ""}</span>
      </summary>
      {s.error && <div className="alert">{s.error}</div>}
      <div className="tabs">
        <button className={tab === "out" ? "active" : ""} onClick={() => setTab("out")}>Output</button>
        <button className={tab === "in" ? "active" : ""} onClick={() => setTab("in")}>Input</button>
      </div>
      {tab === "out" ? <Output s={s} /> : <Input s={s} />}
    </details>
  );
}

export function Waterfall({ steps }: { steps: TraceStep[] }) {
  if (!steps.length) return null;
  const t0 = Math.min(...steps.map((s) => Date.parse(s.started_at)));
  const t1 = Math.max(...steps.map((s) => Date.parse(s.ended_at ?? s.started_at)));
  const span = Math.max(t1 - t0, 1);
  const bySeq = new Map(steps.map((s) => [s.seq, s]));
  return (
    <div className="waterfall">
      {steps.map((s) => {
        const left = ((Date.parse(s.started_at) - t0) / span) * 100;
        const width = Math.max((ms(s) / span) * 100, 0.4);
        return (
          <div className="wf-row" key={s.seq}>
            <div className="wf-label" style={{ paddingLeft: depth(s, bySeq) * 10 }}>
              {NODE[s.node ?? ""] ?? s.node ?? ""} · {s.name}
            </div>
            <div className="wf-track">
              <div className={`wf-bar ${s.kind}`} style={{ left: `${left}%`, width: `${width}%` }} title={`${ms(s)} ms`} />
            </div>
          </div>
        );
      })}
      <div className="wf-axis muted">0 ms → {span} ms after the question arrived</div>
    </div>
  );
}

export function TraceSteps({ steps }: { steps: TraceStep[] }) {
  if (!steps.length) return <p className="muted">No trace stored for this answer.</p>;
  const llm = steps.filter((s) => s.kind === "llm");
  const t0 = Math.min(...steps.map((s) => Date.parse(s.started_at)));
  const t1 = Math.max(...steps.map((s) => Date.parse(s.ended_at ?? s.started_at)));
  const tin = llm.reduce((a, s) => a + (s.prompt_tokens ?? 0), 0);
  const tout = llm.reduce((a, s) => a + (s.completion_tokens ?? 0), 0);
  const bySeq = new Map(steps.map((s) => [s.seq, s]));
  return (
    <div className="trace">
      <p className="muted">
        ⏱ {((t1 - t0) / 1000).toFixed(1)} s · 🧠 {llm.length} model calls · {tin.toLocaleString()}→{tout.toLocaleString()} tokens ·
        🛠️ {steps.filter((s) => s.kind === "tool").length} tool calls · 🔎 {steps.filter((s) => s.kind === "retriever").length} searches
      </p>
      {steps.map((s) => <Step key={s.seq} s={s} pad={depth(s, bySeq)} />)}
    </div>
  );
}

/** Loads and shows the trace of one turn. */
export default function TraceView({ turnId, waterfall = false }: { turnId: string; waterfall?: boolean }) {
  const [steps, setSteps] = useState<TraceStep[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setSteps(null);
    setErr(null);
    api.trace(turnId).then(setSteps).catch((e: Error) => setErr(e.message));
  }, [turnId]);
  if (err) return <div className="alert">{err}</div>;
  if (!steps) return <p className="muted">Loading trace…</p>;
  return (
    <>
      {waterfall && <Waterfall steps={steps} />}
      <TraceSteps steps={steps} />
    </>
  );
}
