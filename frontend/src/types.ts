// Shapes returned by the backend (backend/app/main.py). Keep in sync with the Pydantic models there.

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  llm_calls: number;
}

export interface AgentResponse {
  session_id: string;
  turn_id: string;
  route: string;
  answer: string;
  citations: string[];
  status: string; // completed | step_limit | llm_error
  critique: string[];
  tool_calls: number;
  tool_errors: number;
  usage: Usage;
  latency_ms: number;
  cached_from: string | null;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  turn_id?: string;
  route?: string;
  status?: string;
  citations?: string[];
  critique?: string[];
  tool_calls?: number;
  usage?: Usage;
  latency_ms?: number;
  cached_from?: string | null;
  created_at?: string;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: number;
}

/** One model call, tool call or search, recorded from LangGraph's event stream (backend/app/trace.py). */
export interface TraceStep {
  seq: number;
  parent_seq: number | null;
  node: string | null; // router | planner | tools | critic
  kind: "llm" | "tool" | "retriever";
  name: string;
  input: unknown;
  output: unknown;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  started_at: string;
  ended_at: string | null;
  error: string | null;
}

export interface TurnRow {
  created_at: string;
  conversation_id: string;
  question: string;
  turn_id: string;
  route: string;
  status: string;
  citations: string[];
  critique: string[];
  tool_calls: number;
  usage: Usage | null;
  latency_ms: number | null;
  cached_from: string | null;
  llm_calls: number;
  tools: string[];
}

export interface Health {
  status: string;
  retriever: string;
  storage: string;
  tools: string[];
  llm: { ok: boolean; error?: string };
  cache?: { search: { entries: number; hits: number; misses: number } | null; answer_ttl_s: number };
}

export interface EvalSummary {
  run: string;
  mode: string;
  n: number;
  accuracy?: number;
  failures?: Record<string, number>;
  by_category?: Record<string, string>;
  [key: string]: unknown;
}

export interface EvalRow {
  qid: string;
  category: string;
  question: string;
  answer: string;
  gold_answer: string;
  correct: boolean;
  failure: string | null;
  behavior: string;
  tools_called: string[];
  latency_ms: number | null;
  turn_id?: string | null;
}

export interface RetrievalRun {
  run: string;
  rows: Record<string, string | number>[];
}
