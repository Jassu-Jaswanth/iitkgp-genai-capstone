"""Smoke-test the upGrad Track B endpoints using the values in .env.

Run from the repo root:  python scripts/check_endpoints.py
Stdlib only, so it works before any project dependencies are installed.
"""

import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"Missing {path}. Copy .env.example to .env and fill in the keys.")
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def post(url: str, key: str, cafile: str, body: dict, timeout: int = 90) -> dict:
    ctx = ssl.create_default_context(cafile=str(ROOT / cafile))
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return json.load(resp)


def main() -> int:
    env = load_env(ROOT / ".env")
    ok = True

    try:
        out = post(
            f"{env['LLM_BASE_URL']}/chat/completions",
            env["LLM_API_KEY"],
            env["LLM_CA_CERT_PATH"],
            {
                "model": env["LLM_MODEL"],
                "messages": [{"role": "user", "content": "Reply with the single word OK. /no_think"}],
                "max_tokens": 20,
            },
        )
        print(f"LLM        OK  -> {out['choices'][0]['message']['content'].strip()!r}")
    except (urllib.error.URLError, KeyError, ssl.SSLError) as exc:
        ok = False
        print(f"LLM        FAIL -> {exc}")

    try:
        out = post(
            f"{env['EMBEDDING_BASE_URL']}/embeddings",
            env["EMBEDDING_API_KEY"],
            env["EMBEDDING_CA_CERT_PATH"],
            {"model": env["EMBEDDING_MODEL"], "input": ["hello world"]},
        )
        print(f"Embeddings OK  -> dim {len(out['data'][0]['embedding'])}")
    except (urllib.error.URLError, KeyError, ssl.SSLError) as exc:
        ok = False
        print(f"Embeddings FAIL -> {exc}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
