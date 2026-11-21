# IIT KGP Gen AI & Agentic AI Capstone (Track B)

**Track B: Multi-Agent Operations Copilot** on a hosted Qwen3-8B API.
Topic: _TBD_, see [docs/topic-selection.md](docs/topic-selection.md).

## Key dates
- 25–27 Nov 2026: submission window (portal). No extensions.
- 5 Dec 2026: presentation.

## Docs
- [docs/resources/Capstone-Project-Templates.pdf](docs/resources/Capstone-Project-Templates.pdf): official brief (Track B section is ours)
- [docs/upgrad-communications.md](docs/upgrad-communications.md): digest of upGrad emails, endpoints and API limits
- [docs/resources/Track-B-Inference-API-Guide.docx](docs/resources/Track-B-Inference-API-Guide.docx) and [Embedding-API-Quickstart.md](docs/resources/Embedding-API-Quickstart.md)

## Setup
```bash
uv venv --python 3.12 && source .venv/bin/activate
cp .env.example .env   # add the two keys from the team rep
# put the two CA certs from the team rep in ./certs/
python src/check_endpoints.py   # should print LLM OK and Embeddings OK
```

The LLM key allows **one request at a time** across the whole team; a second
concurrent call gets HTTP 429. Coordinate before running evals.

## Contributing
`main` is protected: no direct pushes.
1. `git switch -c <your-name>/<feature>`
2. Commit, push, open a pull request.
3. One teammate approves, then merge.

Never commit `.env`, keys or the CA certificate.
