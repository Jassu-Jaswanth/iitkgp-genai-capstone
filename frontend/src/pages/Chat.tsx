// Chat on the left, the trace of the selected answer on the right. Conversations are stored in Postgres.
import { useCallback, useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import { api } from "../api";
import type { Conversation, Message } from "../types";
import TraceView from "../components/TraceView";

function meta(m: Message): string {
  const parts = [
    m.route ?? "-",
    `${m.tool_calls ?? 0} tool calls`,
    `${m.usage?.llm_calls ?? 0} model calls`,
    `${((m.latency_ms ?? 0) / 1000).toFixed(1)} s`,
  ];
  return parts.join(" · ");
}

export default function Chat() {
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [convId, setConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const loadConvs = useCallback(() => api.conversations().then(setConvs).catch(() => setConvs([])), []);
  useEffect(() => { loadConvs(); }, [loadConvs]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, busy]);

  async function open(id: string) {
    const msgs = await api.messages(id);
    setConvId(id);
    setMessages(msgs);
    const last = [...msgs].reverse().find((m) => m.role === "assistant");
    setSelected(last?.turn_id ?? null);
    setError(null);
  }

  function newChat() {
    setConvId(null);
    setMessages([]);
    setSelected(null);
    setError(null);
  }

  async function remove(id: string) {
    await api.deleteConversation(id);
    if (id === convId) newChat();
    loadConvs();
  }

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setError(null);
    setMessages((m) => [...m, { role: "user", content: q }]);
    setBusy(true);
    try {
      const r = await api.ask(q, convId);
      setConvId(r.session_id);
      setMessages((m) => [...m, {
        role: "assistant", content: r.answer, turn_id: r.turn_id, route: r.route, status: r.status,
        citations: r.citations, critique: r.critique, tool_calls: r.tool_calls, usage: r.usage,
        latency_ms: r.latency_ms, cached_from: r.cached_from,
      }]);
      setSelected(r.turn_id);
      loadConvs();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const selectedMsg = messages.find((m) => m.turn_id === selected);

  return (
    <div className="chat-layout">
      <aside className="sidebar">
        <button className="primary block" onClick={newChat}>＋ New chat</button>
        <div className="muted label">Conversations</div>
        {convs.map((c) => (
          <div key={c.id} className={`conv ${c.id === convId ? "active" : ""}`}>
            <button className="conv-title" onClick={() => open(c.id)} title={c.title}>{c.title}</button>
            <button className="icon" onClick={() => remove(c.id)} title="Delete conversation">✕</button>
          </div>
        ))}
      </aside>

      <section className="chat">
        <div className="messages-pane">
          {!messages.length && (
            <p className="muted intro">
              Ask about US federal leave, pay, holidays and HR rules. Answers cite the statute, regulation or OPM
              guidance they come from; the trace on the right shows every step the agent took.
            </p>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              <div className="prose"><Markdown>{m.content}</Markdown></div>
              {m.role === "assistant" && (
                <div className="bubble-meta">
                  {m.citations?.length ? <div className="sources">Sources: {m.citations.join(" · ")}</div> : null}
                  <div className="row">
                    <span className="muted">{meta(m)}</span>
                    {m.cached_from && <span className="badge" title="Same question answered before; served from the answer cache">cached</span>}
                    {m.status && m.status !== "completed" && <span className="badge warn">{m.status}</span>}
                    {m.turn_id && (
                      <button className="link" disabled={m.turn_id === selected} onClick={() => setSelected(m.turn_id!)}>
                        {m.turn_id === selected ? "Showing trace" : "Trace ▸"}
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
          {busy && <div className="bubble assistant muted">Thinking… (the shared model answers one request at a time)</div>}
          {error && <div className="alert">{error}</div>}
          <div ref={bottom} />
        </div>
        <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
          <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a question…" maxLength={2000} />
          <button className="primary" disabled={busy || !input.trim()}>Send</button>
        </form>
      </section>

      <section className="trace-pane">
        <h3>Trace</h3>
        {selectedMsg?.critique?.length ? <div className="alert warn">Critic: {selectedMsg.critique.join(" | ")}</div> : null}
        {selectedMsg?.cached_from && <p className="muted">Served from the answer cache; this is the trace of the original answer.</p>}
        {selected ? <TraceView turnId={selected} /> : (
          <p className="muted">The steps behind an answer show up here: every model call, tool call and search, with its input and output.</p>
        )}
      </section>
    </div>
  );
}
