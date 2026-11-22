# frontend/: the app (React + TypeScript)

One page app, four tabs:

| Tab | What you see |
|---|---|
| **Chat** | conversations on the left, chat in the middle, and on the right the **trace** of the selected answer: every model call, tool call and search, in order, with its input and output |
| **Traces** | every answer with latency, tokens, tools and status; click one for a timeline (waterfall) and the full trace |
| **Evaluation** | eval runs from `eval/results/`: summary numbers, accuracy by category, failure taxonomy, each question (with a link to its trace) |
| **Retrieval experiments** | recall@k per chunking × retriever, as a heat map and a table |

It talks to the backend only through `/api/...` (see `src/api.ts`; response types in `src/types.ts`).

## Run
```bash
npm install
npm run dev        # http://localhost:5173, /api proxied to the backend on :8100 (API_URL to change)
npm run build      # type check + production build into dist/
```
In Docker (`docker compose up`), nginx serves the built app on http://localhost:3000, forwards `/api/*` to the
backend, and **rate-limits** questions to 10 a minute per IP (burst 5) — see `nginx.conf`.

## Files
| File | What |
|---|---|
| `src/pages/Chat.tsx` | chat + trace panel |
| `src/components/TraceView.tsx` | draws one trace (steps, waterfall) |
| `src/pages/Traces.tsx`, `Evaluation.tsx`, `Retrieval.tsx` | the other tabs |
| `nginx.conf`, `Dockerfile` | container: build with Node, serve with nginx |
