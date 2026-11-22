-- HR Policy Navigator: application schema (applied at backend startup; idempotent).
--
-- One Postgres database, two schemas:
--   chat   this file: conversations, their messages, and the LLM trace behind every answer
--   agent  LangGraph's checkpointer (agent memory per conversation), created by AsyncPostgresSaver.setup()
-- An assistant message and its trace steps share a turn_id.

CREATE SCHEMA IF NOT EXISTS chat;
CREATE SCHEMA IF NOT EXISTS agent;

CREATE TABLE IF NOT EXISTS chat.conversations (
    id          text        PRIMARY KEY,                 -- = session_id / LangGraph thread_id
    title       text        NOT NULL,                    -- first question, shortened
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat.messages (
    id               bigint      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id  text        NOT NULL REFERENCES chat.conversations (id) ON DELETE CASCADE,
    role             text        NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text        NOT NULL,
    -- assistant messages only: what the agent did for this answer
    turn_id          text        UNIQUE,                 -- joins to chat.trace_steps
    route            text,                               -- hr_policy | individual_records | out_of_scope | chitchat
    status           text,                               -- completed | step_limit | llm_error
    citations        jsonb       NOT NULL DEFAULT '[]',  -- ["5 U.S.C. 6303(a)", ...]
    critique         jsonb       NOT NULL DEFAULT '[]',  -- critic issues left on the final answer
    tool_calls       int,
    usage            jsonb,                              -- {"prompt_tokens", "completion_tokens", "llm_calls"}
    latency_ms       double precision,
    created_at       timestamptz NOT NULL DEFAULT now(),
    CHECK (role = 'user' OR turn_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS messages_conversation_idx ON chat.messages (conversation_id, id);
CREATE INDEX IF NOT EXISTS conversations_recent_idx ON chat.conversations (updated_at DESC);

-- The LLM trace of one answer: every model call, tool call and retrieval, in order (from LangGraph's events).
CREATE TABLE IF NOT EXISTS chat.trace_steps (
    turn_id            text        NOT NULL REFERENCES chat.messages (turn_id) ON DELETE CASCADE,
    seq                int         NOT NULL,             -- 0, 1, 2 ... in start order
    parent_seq         int,                              -- e.g. the retriever inside the search_policy tool
    node               text,                             -- graph node: router | planner | tools | critic
    kind               text        NOT NULL CHECK (kind IN ('llm', 'tool', 'retriever')),
    name               text        NOT NULL,             -- ChatOpenAI | search_policy | LocalHybridRetriever ...
    input              jsonb,                            -- prompt messages / tool arguments / search query
    output             jsonb,                            -- model reply (incl. tool calls) / tool result / chunks
    prompt_tokens      int,
    completion_tokens  int,
    started_at         timestamptz NOT NULL,
    ended_at           timestamptz,
    error              text,
    PRIMARY KEY (turn_id, seq)
);

-- Answer cache (app/cache.py): first question of a conversation -> stored answer + its trace. Key = sha256 of a
-- version hash (prompts, tools, retriever, model) and the normalised question, so changes start a fresh cache.
CREATE TABLE IF NOT EXISTS chat.answer_cache (
    key         text        PRIMARY KEY,
    question    text        NOT NULL,
    entry       jsonb       NOT NULL,                    -- {"response": AgentResponse, "steps": trace steps}
    hits        int         NOT NULL DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Answers served from the cache point at the turn that produced them.
ALTER TABLE chat.messages ADD COLUMN IF NOT EXISTS cached_from text;
