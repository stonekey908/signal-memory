"""MemoryService — the engine packaged as a persistent, long-lived memory (STO-2804).

The benchmark builds a fresh throwaway engine per question; a real agent memory is the
opposite: ONE store that persists on disk across sessions. This wraps ``SignalEngine`` over
a persistent Chroma collection and exposes the two operations an agent actually needs:

    remember(text)  -> distil the text into signals and store them (write path)
    recall(query)   -> retrieve the most relevant stored memories (read path)

Pure Python, no MCP dependency — so it is unit-testable with mock components and reused by
the MCP server (``mcp_server.py``). Model selection matches the frozen benchmark recipe.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class _Turn:
    role: str
    content: str


@dataclass
class _Session:
    """A minimal SessionLike (see base.py): the text to remember as one user turn, dated."""

    turns: List[_Turn]
    date: str = ""
    session_id: Optional[str] = None


# Coverage-style queries want breadth (the whole picture), not the single closest fact.
_BREADTH_HINTS = ("summar", "everything", "all your", "all the", "all my", "tell me about",
                  "overview", "what do you know", "recap", "so far", "list ", "give me a rundown",
                  "catch me up", "remind me about", "the whole", "in general", "where are we",
                  "where were we", "up to speed", "the status", "what's the state", "key decisions")


def _wants_breadth(query: str) -> bool:
    q = (query or "").lower()
    return any(h in q for h in _BREADTH_HINTS)


def _default_store_path() -> str:
    """Where the persistent memory lives. Overridable via SIGNAL_MEMORY_PATH so a user can
    keep separate memories (per-project, per-persona) — defaults under the user's home."""
    return os.getenv("SIGNAL_MEMORY_PATH", os.path.expanduser("~/.signal_engine/memory"))


@dataclass
class MemoryService:
    """Long-lived memory backed by a persistent store. Build with ``build_default()`` for the
    real (OpenAI) recipe, or inject a pre-built engine (mock) in tests."""

    engine: "object"                    # a MemoryEngine (SignalEngine or a mock)
    recall_k: int = 6                   # max memories for a PRECISE query (a handful)
    pool_gate: float = 0.15             # wide low bar: candidates the reranker then judges by MEANING
    pool_k: int = 12                    #  (fixes vocabulary mismatch cosine misses — mem0's reranker idea)
    relevance_floor: float = 0.28       # cosine floor for the no-reranker fallback path
    recall_shortlist: int = 40
    broad_floor: float = 0.12           # BROAD queries ("tell me everything") relax the floor...
    broad_k: int = 15                   #  ...and return more — a vague query matches no one fact strongly
    _seq: int = field(default=0)

    def remember(self, text: str, *, date: str = "") -> str:
        """Distil ``text`` into signals and persist them. Returns a short human summary."""
        text = (text or "").strip()
        if not text:
            return "Nothing to remember (empty text)."
        before = self._signal_count()
        self._seq += 1
        ids_before = set(self._all_signal_ids())
        session = _Session(turns=[_Turn(role="user", content=text)],
                           date=date, session_id=f"mem-{self._seq}")
        self.engine.ingest(session)
        # Versioning: reconcile the NEWLY-added facts against earlier ones so a changed fact ("we
        # switched to Chroma") marks the old one superseded (shown '[older]' at recall, never
        # dropped). Scoped to the new ids so cost is O(new), not O(whole store) per write.
        if getattr(self.engine, "reconciler", None) is not None:
            new_ids = set(self._all_signal_ids()) - ids_before
            try:
                self.engine._reconcile(only_ids=new_ids)
            except Exception:
                pass                              # versioning is best-effort, never fail a write
        stored = self._signal_count() - before
        return f"Remembered — stored {stored} signal(s)." if stored else \
            "Noted, but nothing durable was extracted (too trivial to store)."

    def recall(self, query: str, *, task_context: Optional[str] = None) -> List[str]:
        """Return a SMALL set of genuinely relevant memories for ``query`` (may be empty).

        This is the product read path — deliberately NOT the benchmark retriever. The benchmark
        returns top-50 because every exam question has an answer buried in hundreds of facts; a
        real personal store has dozens, so top-50 = the whole store. Instead:
          * score every candidate by semantic similarity,
          * keep only FACTS above a per-item relevance floor (behavioural notes are noise here),
          * current facts first, superseded ('[older]') after,
          * return at most ``recall_k``.
        So an unrelated query returns nothing, and a specific one returns a line or two — not 24.
        """
        query = (query or "").strip()
        if not query:
            return []
        store = getattr(self.engine, "store", None)
        if store is None or not hasattr(store, "search_scored"):
            return list(self.engine.retrieve(query, task_context))   # mock/fallback path
        from signal_engine.engine import signal_to_context

        # BROAD query ("tell me everything") — cast wide, no reranker, coverage over precision.
        if _wants_breadth(query):
            scored = store.search_scored(query, max(self.recall_shortlist, self.broad_k))
            keep = [(s, sim) for s, sim in scored if sim >= self.broad_floor and not s.archived]
            keep.sort(key=lambda p: (0 if p[0].is_current else 1, -p[1]))
            return [signal_to_context(s) for s, _ in keep[: self.broad_k]]

        # PRECISE query — wide low-bar pool, then let the reranker judge relevance BY MEANING.
        # Pure cosine misses vocabulary gaps ("how should you talk to me" vs "prefers plain
        # English" scored 0.17); the LLM reranker understands them. It returns only what's
        # genuinely relevant (possibly nothing), so it also gates "I never stored that".
        scored = store.search_scored(query, self.pool_k)
        pool = [s for s, sim in scored if sim >= self.pool_gate and not s.archived]
        if not pool:
            return []
        reranker = getattr(getattr(self.engine, "ranker", None), "reranker", None)
        if reranker is None:                       # mock/no-reranker fallback: cosine floor
            keep = [(s, sim) for s, sim in scored
                    if sim >= self.relevance_floor and not s.archived]
            keep.sort(key=lambda p: (0 if p[0].is_current else 1, -p[1]))
            return [signal_to_context(s) for s, _ in keep[: self.recall_k]]
        try:
            ordered_ids, rtok = reranker.rerank(query, None, pool, self.recall_k)
            if hasattr(self.engine, "retrieve_tokens"):
                self.engine.retrieve_tokens += rtok        # honest token accounting for the product
        except Exception:
            ordered_ids = [s.signal_id for s in pool[: self.recall_k]]   # degrade, never fail
        by_id = {s.signal_id: s for s in pool}
        picked = [by_id[i] for i in ordered_ids if i in by_id]           # only the reranker's picks
        picked.sort(key=lambda s: 0 if s.is_current else 1)              # current facts before [older]
        return [signal_to_context(s) for s in picked[: self.recall_k]]

    def _signal_count(self) -> int:
        store = getattr(self.engine, "store", None)
        try:
            return store.count() if store is not None else 0
        except Exception:
            return 0

    def _all_signal_ids(self) -> List[str]:
        store = getattr(self.engine, "store", None)
        try:
            return [s.signal_id for s in store.all_signals()] if store is not None else []
        except Exception:
            return []

    # -- construction ---------------------------------------------------------

    @staticmethod
    def build_default(persist_path: Optional[str] = None) -> "MemoryService":
        """Build the real service: SignalEngine (frozen recipe) over a PERSISTENT Chroma
        collection. The brain (scribbler + reranker) and the embedder are chosen from the
        environment via ``providers`` — 'bring your own models' (openai | anthropic; local
        planned). Defaults to OpenAI. The collection name is STABLE (unlike the benchmark's
        per-instance unique name) so memory survives across processes — close the CLI, reopen
        tomorrow, it is still there."""
        from signal_engine.engine import SignalEngine
        from signal_engine.extractor import EXTRACTION_PROMPT, Extractor
        from signal_engine.llm_memory_service import _PRODUCT_EXTRACT_EXTRA
        from signal_engine.providers import build_brain, build_embedder
        from signal_engine.reconciler import OpenAIReconciler
        from signal_engine.store import ChromaStore

        path = persist_path or _default_store_path()
        os.makedirs(path, exist_ok=True)
        scribbler, reranker = build_brain()
        embedder = build_embedder()
        # Collection is scoped to the embedder: different embedders produce incompatible vectors
        # (e.g. OpenAI 1536-dim vs local 384-dim), so switching provider must NOT reuse — and
        # corrupt — the same collection. A per-embedder name gives each its own clean store.
        tag = getattr(embedder, "model", None) or getattr(embedder, "model_name", embedder.name)
        coll = "mem_" + re.sub(r"[^a-zA-Z0-9]+", "_", f"{embedder.name}_{tag}")[:55]
        store = ChromaStore(embedder, collection_name=coll, persist_path=path)  # embedder-scoped
        # Product parity with the non-vector path: same noise-aware extraction prompt + dedup +
        # versioning. top_k unused (recall() does its own relevance-gated read). Benchmark path
        # (build_signal_engine) is untouched by all of this.
        product_prompt = EXTRACTION_PROMPT.replace("\nConversation:",
                                                   _PRODUCT_EXTRACT_EXTRA + "\nConversation:")
        engine = SignalEngine(scribbler, reranker, store, chunk_turns=10,
                              reconciler=OpenAIReconciler(), dedup_threshold=0.9)
        engine.extractor = Extractor(scribbler, chunk_turns=10, prompt=product_prompt)   # noise fix
        return MemoryService(engine=engine)
