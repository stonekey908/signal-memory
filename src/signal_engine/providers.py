"""Provider selection — 'bring your own models' for the MCP memory (STO-2804).

A memory engine needs models for TWO jobs (see SETUP.md):
  * the BRAIN  — scribbler (extraction) + reranker (retrieval judgement): any chat LLM.
  * the INDEX  — embedder (text -> vector for semantic search): a SPECIALISED embeddings model.
    Anthropic ships no embeddings API, so a Claude brain still needs an embeddings source
    (OpenAI now; a local model in the next slice). This is universal to vector memory.

Config via environment variables (all optional; sensible defaults):
  SIGNAL_LLM_PROVIDER    openai | anthropic          (brain; default openai)
  SIGNAL_SCRIBBLER_MODEL / SIGNAL_RERANKER_MODEL     (override per provider)
  SIGNAL_EMBED_PROVIDER  openai                      (index; default openai)
  SIGNAL_EMBED_MODEL     embeddings model name

SLICE 1 (this file): frontier — OpenAI or Anthropic/Claude for the brain, OpenAI embedder.
SLICE 2 (planned): fully-local — Ollama brain + local sentence-transformers embedder.
"""

from __future__ import annotations

import os
from typing import Tuple

# provider -> (default scribbler model, default reranker model)
_LLM_DEFAULTS = {
    "openai": ("gpt-4o", "gpt-4o"),
    "anthropic": ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
    "ollama": ("qwen2.5-coder", "qwen2.5-coder"),     # local; overridable
}


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _ollama_client_factory(base_url: str):
    """A client pointed at a local Ollama server (OpenAI-compatible API, no key)."""
    def factory(api_key=None):
        import openai
        from signal_engine.openai_client import OPENAI_MAX_RETRIES, OPENAI_TIMEOUT
        return openai.OpenAI(api_key="ollama", base_url=base_url,
                             timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)
    return factory


def build_brain() -> Tuple[object, object]:
    """Return (scribbler, reranker) for the configured LLM provider."""
    provider = _env("SIGNAL_LLM_PROVIDER", "openai").lower()
    if provider not in _LLM_DEFAULTS:
        raise ValueError(f"SIGNAL_LLM_PROVIDER must be one of {list(_LLM_DEFAULTS)}, got {provider!r}")
    scr_model = _env("SIGNAL_SCRIBBLER_MODEL", _LLM_DEFAULTS[provider][0])
    rr_model = _env("SIGNAL_RERANKER_MODEL", _LLM_DEFAULTS[provider][1])

    from signal_engine.extractor import ClaudeScribbler, OpenAIScribbler
    from signal_engine.ranker import OpenAIReranker

    if provider == "anthropic":
        from signal_engine.openai_client import anthropic_client
        scribbler = ClaudeScribbler(model=scr_model)
        reranker = OpenAIReranker(model=rr_model, client_factory=anthropic_client,
                                  json_format=False, max_tokens=1024)
    elif provider == "ollama":       # fully local brain — no API key
        factory = _ollama_client_factory(_env("SIGNAL_LLM_BASE_URL", "http://localhost:11434/v1"))
        scribbler = OpenAIScribbler(model=scr_model, api_key="ollama", client_factory=factory)
        reranker = OpenAIReranker(model=rr_model, api_key="ollama", client_factory=factory,
                                  json_format=False, max_tokens=1024)
    else:  # openai
        scribbler = OpenAIScribbler(model=scr_model)
        reranker = OpenAIReranker(model=rr_model)
    return scribbler, reranker


def build_embedder() -> object:
    """Return the embedder for the configured embeddings provider (the INDEX-maker)."""
    provider = _env("SIGNAL_EMBED_PROVIDER", "openai").lower()
    if provider == "local":          # fully local, offline — no API key (needs `local` extra)
        from signal_engine.embedder import LocalEmbedder
        return LocalEmbedder(_env("SIGNAL_EMBED_MODEL", "all-MiniLM-L6-v2"))
    if provider == "openai":
        from signal_engine.embedder import OpenAIEmbedder
        return OpenAIEmbedder(model=_env("SIGNAL_EMBED_MODEL", "text-embedding-3-small"))
    raise ValueError(f"SIGNAL_EMBED_PROVIDER must be openai|local, got {provider!r}")


def build_summariser():
    """Summariser for the LLM path's hierarchy, matched to the configured provider."""
    provider = _env("SIGNAL_LLM_PROVIDER", "openai").lower()
    from signal_engine.consolidation import OpenAISummariser
    if provider == "anthropic":
        from signal_engine.openai_client import anthropic_client
        model = _env("SIGNAL_SCRIBBLER_MODEL", _LLM_DEFAULTS["anthropic"][0])
        return OpenAISummariser(model=model, client_factory=anthropic_client, max_tokens=1024)
    if provider == "ollama":
        model = _env("SIGNAL_SCRIBBLER_MODEL", _LLM_DEFAULTS["ollama"][0])
        return OpenAISummariser(model=model, client_factory=_ollama_client_factory(
            _env("SIGNAL_LLM_BASE_URL", "http://localhost:11434/v1")), max_tokens=1024)
    return OpenAISummariser()


