# upGrad capstone communications (digest)

Summary of every upGrad email about the capstone, newest first. Keys and
credentials are **not** reproduced here; they live in the team rep's `.env`.
Ask the team rep for them; do not share outside the team.

## Key facts

| Item | Value |
|---|---|
| Track | **B: API-side** ("Multi-Agent Operations Copilot", see `resources/Capstone-Project-Templates.pdf`) |
| Team rep | Jaswanth Suvvari (only the rep contacts the mentor or submits forms) |
| Mentor | Dr. Abhilash Nandy (contact through the team rep) |
| Submission | Portal, new module in the Capstone course, **25–27 Nov 2026, no extensions** |
| Presentation | 5 Dec 2026 |
| Weight | 20% of programme grade; mandatory for certification |
| LLM | `Qwen/Qwen3-8B`, OpenAI-compatible, group range 046–065 → `https://103.42.50.41:1034/v1` |
| Embeddings | `bge-small-en-v1.5` (384-dim), `https://103.42.50.41:1035/v1/embeddings` |

## Emails

### 17 Nov 2026: Final Reminder: Capstone Project Submission Deadline
- Submission module will appear in the Capstone course on the portal.
- Window: 25–27 Nov. Mentors evaluate the capstone; stay in touch with them.

### 13 Nov 2026: Capstone Track B: Embedding Model Endpoint (API Access)
- Embedding endpoint for RAG (Track B). Drive folder contained
  `Embedding_API_Quickstart.md` (copied to `resources/`), the CA cert
  `vm02-ca.crt` and a shared embedding key (kept out of git).

### 6 Nov 2026: Capstone Project: Infrastructure Resources (API Access)
- Group LLM API key (kept out of git), CA certificate (vm04, Caddy local CA) and
  the Learner Guide (copied to `resources/Track-B-Inference-API-Guide.docx`).
- Do not share credentials outside the team. Support: studentsupport@upgrad.com.

### 28 Oct 2026: Final Reminder: Track Selection Deadline (28 Oct)
- Form link https://forms.gle/3hMCdxa2LNyoSqU16. Unsubmitted teams get a default track.
- The LMS shows our "Track Selection Response" as completed.

### 23 Oct 2026: Revised Track Selection Deadline & Mentor Contact Details
- Deadline preponed to 28 Oct so infra could be provisioned by 8 Nov.
- Mentor assigned: Dr. Abhilash Nandy. Only the team rep reaches out; propose
  several slots agreed by the team.

### 13 Oct 2026: Extended Deadline & Live Session with Prof. Abhijnan
- Track deadline temporarily extended to 6 Nov (later superseded by 28 Oct).
- 24 Oct live session: REST APIs with FastAPI, real-time vs batch inference, capstone overview.

### 10 Oct 2026: [Action Required] Capstone Schedule and Key Details
- Teams of 6–7, fixed. 4 weekly mentor sessions, recordings on the LMS.
- Capstone began 30 Oct; submission 27 Nov; presentation 5 Dec.

### 5 Oct 2026: Session Recording: Capstone Overview
- Recording on PRISM → Live Learning → Past.

## API limits that shape the design (from the Learner Guide)

| Limit | Value | Design consequence |
|---|---|---|
| Concurrent requests per key | **1** (second gets 429) | Queue all LLM calls; run evals serially; teammates must not hammer the key at the same time |
| Requests per key | 60 / min | Cache responses; budget eval runs |
| Context (input + output) | 8,192 tokens | Small chunks, top-k ≤ 5, short system prompts |
| Output | 512 default, 1,024 max | Ask for compact JSON |
| Errors | 429/503 → retry 2, 4, 8 s | Built-in exponential backoff |
| Embeddings | ≤16 texts/request, ≤512 tokens/chunk | Batch at 16; chunk ≤ ~400 tokens |

The chat API supports streaming and tool-call responses; embeddings are only on the separate endpoint.
