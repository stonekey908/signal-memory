"""Pluggable embedder — turns text into vectors for semantic (meaning) search.

``MockEmbedder`` (deterministic, no key) for tests; ``OpenAIEmbedder``
(``text-embedding-3-small``, 1536-dim) for real use. Same interface, swap freely.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional, Protocol, runtime_checkable

import signal_engine.config  # noqa: F401  — importing loads .env (API keys)
from signal_engine.openai_client import openai_client

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one vector per input text."""
        ...


class MockEmbedder:
    """Deterministic hash bag-of-words embeddings (no key). Same text -> same vector;
    texts sharing words -> higher cosine. For tests/offline only.
    """

    name = "mock"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in (text or "").lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


class OpenAIEmbedder:
    """Real OpenAI embedder (``text-embedding-3-small``, 1536-dim). Requires OPENAI_API_KEY."""

    name = "openai"

    def __init__(self, model: str = EMBED_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.dim = EMBED_DIM
        self._api_key = api_key   # resolved at call time by the shared openai_client()

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = openai_client(self._api_key)   # shared client: timeout + retry, no per-call leak
        resp = client.embeddings.create(model=self.model, input=list(texts))
        return [d.embedding for d in resp.data]


class LocalEmbedder:
    """Fully-local, offline embedder via sentence-transformers (STO-2804 local slice).

    No API key, no network, runs on the Mac. Default ``all-MiniLM-L6-v2`` is small (~90MB,
    384-dim) and fast. The model loads lazily on first use so importing this module stays cheap
    and keyless. Requires the ``local`` extra (``uv sync --extra local``).

    Note: embeddings are NOT interchangeable with OpenAI's (different dimensions), so a store
    built with one embedder must not be reused with another — pick one per memory store.
    """

    name = "local"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.dim = 384 if "MiniLM-L6" in model_name else 0   # set precisely on first load

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._ensure().encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vecs]
