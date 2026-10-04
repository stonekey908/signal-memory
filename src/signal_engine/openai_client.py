"""Shared OpenAI client — one reused, configured client per process.

Every OpenAI wrapper (scribbler, embedder, reranker, reader, judges) calls
``openai_client()`` instead of building its own client on every call. Reusing a
single client means a single connection pool for the whole run — no socket leak
(the ``CLOSE_WAIT`` pileup that hung a 500-question run; see CLAUDE.md
"BENCHMARK BLOCKER"). The built-in request ``timeout`` + bounded ``max_retries``
mean a dead socket fails fast and retries instead of hanging forever.

The OpenAI client is thread-safe, so the same shared client is safe to use from
the parallel-ingestion threads (STO-2788, step 2).
"""

from __future__ import annotations

import os
from typing import Optional

import signal_engine.config  # noqa: F401  — importing loads .env (API keys) via load_dotenv

# A dead socket should fail fast and retry, never hang the whole run. Both are
# env-overridable for slow networks or debugging.
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

_shared_client = None  # process-wide singleton for the default (env-key) path


def openai_client(api_key: Optional[str] = None):
    """Return a shared, configured OpenAI client (reused across all calls).

    Passing no ``api_key`` (the normal path) returns a process-wide singleton, so
    the whole run shares one connection pool. Passing an explicit ``api_key``
    returns a dedicated (uncached) client — used by callers that inject a key.
    Raises ``RuntimeError`` if no key is available; the check runs *before*
    importing the ``openai`` SDK, so mock-mode callers without the ``llm`` extra
    still get a clear message rather than an ImportError.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set — add it to .env (see SETUP.md).")
    import openai  # lazy: openai lives in the `llm` extra

    if api_key is not None:  # explicit key -> dedicated client (not cached)
        return openai.OpenAI(api_key=key, timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)

    global _shared_client
    if _shared_client is None:
        _shared_client = openai.OpenAI(
            api_key=key, timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES
        )
    return _shared_client


def reset_shared_client() -> None:
    """Drop the cached singleton. Tests only — lets a test rebuild it under new env."""
    global _shared_client, _anthropic_client
    _shared_client = None
    _anthropic_client = None


# --- Anthropic (Claude) via the OpenAI-compatible endpoint -------------------

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/"
_anthropic_client = None


def anthropic_client(api_key: Optional[str] = None):
    """Shared client for Claude models (NFR-2 second backend). Anthropic speaks the OpenAI
    SDK dialect at ``/v1``, so the whole engine reuses one client stack — same timeout/retry
    discipline, same singleton reuse. Requires ANTHROPIC_API_KEY (pay-as-you-go API key —
    a claude.ai subscription cannot be used here)."""
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — add it to .env (see SETUP.md).")
    import openai  # lazy: openai lives in the `llm` extra

    if api_key is not None:  # explicit key -> dedicated client (not cached)
        return openai.OpenAI(api_key=key, base_url=ANTHROPIC_BASE_URL,
                             timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)

    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = openai.OpenAI(api_key=key, base_url=ANTHROPIC_BASE_URL,
                                          timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)
    return _anthropic_client
