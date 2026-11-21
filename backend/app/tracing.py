"""Observability: OpenTelemetry traces of every LangGraph node, LLM call and tool call, exported to Arize Phoenix.

Enabled only when PHOENIX_COLLECTOR_ENDPOINT is set (e.g. http://localhost:6006/v1/traces).
Uses OpenInference's LangChain instrumentor, so no custom span code is needed.
"""

import os


def setup_tracing() -> str | None:
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return None
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    provider = register(endpoint=endpoint, project_name=os.getenv("PHOENIX_PROJECT_NAME", "hr-policy-navigator"),
                        batch=True, set_global_tracer_provider=True, verbose=False)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    return endpoint
