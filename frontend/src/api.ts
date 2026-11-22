// All backend calls go through /api (nginx in the container, the Vite proxy in dev).
import type {
  AgentResponse, Conversation, EvalRow, EvalSummary, Health, Message, RetrievalRun, TraceStep, TurnRow,
} from "./types";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(r.status === 429 ? "Too many requests: wait a few seconds and try again." : `HTTP ${r.status}: ${text.slice(0, 200)}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => call<Health>("/health"),
  ask: (question: string, sessionId: string | null) =>
    call<AgentResponse>("/agent", { method: "POST", body: JSON.stringify({ question, session_id: sessionId }) }),
  conversations: () => call<Conversation[]>("/conversations"),
  messages: (id: string) => call<Message[]>(`/conversations/${id}/messages`),
  deleteConversation: (id: string) => call<{ deleted: string }>(`/conversations/${id}`, { method: "DELETE" }),
  turns: (limit = 200) => call<TurnRow[]>(`/turns?limit=${limit}`),
  trace: (turnId: string) => call<TraceStep[]>(`/turns/${turnId}/trace`),
  evalRuns: () => call<EvalSummary[]>("/eval/runs"),
  evalRun: (run: string) => call<EvalRow[]>(`/eval/runs/${encodeURIComponent(run)}`),
  retrieval: () => call<RetrievalRun[]>("/eval/retrieval"),
};
