"""SignalEngine — the two-method public API (``ingest`` / ``retrieve``).

Assembles the Wave-1 components into the ``MemoryEngine`` contract so it drops into the
benchmark harness in place of the trivial retriever, with no harness changes:

* ``ingest(session)``            = extractor → salience filter → store
* ``retrieve(query, task_context)`` = ranker → token-bounded context snippets

Everything is pluggable (scribbler / embedder / reranker / store), so mock-first tests run
with no key. Out of scope: consolidation/pins/procedural (Wave 2), full benchmark (Wave 3).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

from signal_engine.base import MemoryEngine, SessionLike
from signal_engine.embedder import Embedder
from signal_engine.extractor import ExtractionResult, Extractor, Scribbler
from signal_engine.ranker import Ranker, Reranker
from signal_engine.reconciler import Reconciler
from signal_engine.salience import SalienceFilter
from signal_engine.signal import Signal
from signal_engine.store import InMemoryStore, VectorStore

_META_FIELDS = (("sentiment", "sentiment"), ("friction", "friction"),
                ("tool_choice", "tool_choice"), ("interaction", "interaction"))

# Per-question ingestion extracts many sessions; each is an independent, I/O-bound API
# call, so we run them concurrently (env-overridable).
DEFAULT_INGEST_WORKERS = int(os.getenv("SIGNAL_INGEST_WORKERS", "8"))


def signal_to_context(s: Signal) -> str:
    """Render a Signal as a context snippet for the reader.

    Prepends the session date when known: temporal-reasoning questions need it — the
    Wave-0 oracle showed injecting session dates takes temporal from 0 -> 100%, and
    without it the reader can't reason about "when" (STO-2789)."""
    if s.is_fact:
        body = s.content or s.title
    else:
        meta = "; ".join(f"{label}: {getattr(s, attr)}" for label, attr in _META_FIELDS if getattr(s, attr))
        body = f"[behaviour] {s.title}: {meta}"
    dated = f"[{s.created_at}] {body}" if s.created_at else body
    return f"[older] {dated}" if (s.is_fact and s.superseded_by) else dated   # STO-2800: annotate, don't drop


class SignalEngine(MemoryEngine):
    def __init__(
        self,
        scribbler: Scribbler,
        reranker: Reranker,
        store: VectorStore,
        *,
        salience_threshold: float = 0.2,
        chunk_turns: Optional[int] = None,
        top_k: int = 10,          # match the baseline's breadth — multi-session needs facts
        shortlist_k: int = 25,    # from several sessions; top-5 handed over too little (STO-2789)
        ingest_workers: int = DEFAULT_INGEST_WORKERS,
        pin_threshold: Optional[float] = None,   # STO-2728: auto-pin facts with importance >= this
        persona: Optional[str] = None,           # STO-2730: stable identity/tone, supplied each turn
        procedural: bool = False,                # STO-2729: surface corrective behavioural memory
        summariser=None,                         # STO-2726/2727: consolidation summariser (None = off)
        consolidate_threshold: Optional[int] = None,
        keep_recent: int = 20,
        reconciler: Optional[Reconciler] = None, # STO-2800: fact versioning / supersede (None = off)
        reconcile_k: int = 5,
        dedup_threshold: Optional[float] = None, # product: skip a new fact whose top cosine to an
    ):                                           # existing one >= this (None = off; benchmark off)
        self.extractor = Extractor(scribbler, chunk_turns=chunk_turns)
        self.salience = SalienceFilter(threshold=salience_threshold)
        self.store = store
        self.ranker = Ranker(store, reranker, top_k=top_k, shortlist_k=shortlist_k)
        self.ingest_workers = max(1, ingest_workers)
        self.pin_threshold = pin_threshold
        self.persona = (persona or "").strip() or None
        self.procedural = procedural
        self.summariser = summariser
        self.consolidate_threshold = consolidate_threshold
        self.keep_recent = keep_recent
        self.reconciler = reconciler
        self.reconcile_k = reconcile_k
        self.dedup_threshold = dedup_threshold
        self.ingest_tokens = 0
        self.retrieve_tokens = 0

    def ingest(self, session: SessionLike) -> None:
        self._store_extraction(self.extractor.extract(session))

    def ingest_many(self, sessions: Sequence[SessionLike]) -> None:
        """Extract sessions concurrently (each is an independent, I/O-bound LLM call),
        then file the results in session order. Writes stay single-threaded, so there is
        no store race and signal ids stay deterministic. This is the ~10x per-question
        speedup that also stops one slow session blocking the whole run (STO-2788)."""
        sessions = list(sessions)
        if len(sessions) <= 1:                       # nothing to parallelise
            for session in sessions:
                self._store_extraction(self.extractor.extract(session))
        else:
            workers = min(self.ingest_workers, len(sessions))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                extractions = list(pool.map(self.extractor.extract, sessions))
            for extraction in extractions:           # store in order — single-threaded writes
                self._store_extraction(extraction)
        self._reconcile()                            # link fact supersessions (STO-2800)
        self._consolidate()                          # fold old memory into summaries (STO-2726/2727)

    def _reconcile(self, only_ids=None) -> None:
        """Non-destructive fact versioning (STO-2800): mark an earlier fact ``superseded_by`` a newer
        one about the same subject. Nothing is dropped — retrieval annotates superseded facts '[older]',
        so a wrong guess can't remove a needed fact (the v1 fix).

        ``only_ids`` scopes the pass to just those (newly-added) signals as the *newer* fact —
        the persistent product calls this per write so cost is O(new) not O(whole store). The
        benchmark passes None (reconcile everything once at the end of ingest_many)."""
        if self.reconciler is None:
            return
        signals = self.store.all_signals()
        pos = {s.signal_id: i for i, s in enumerate(signals)}
        for i, s in enumerate(signals):
            if not s.is_fact:
                continue
            if only_ids is not None and s.signal_id not in only_ids:
                continue                             # scoped pass: only reconcile the new facts
            similar = self.store.search_similar(s.content or s.title, self.reconcile_k)
            cands = [c for c in similar if c.is_fact and c.is_current and c.signal_id != s.signal_id
                     and pos.get(c.signal_id, i) < i]
            if not cands:
                continue
            superseded = self.reconciler.superseded_id(s, cands)
            if superseded:
                for c in cands:
                    if c.signal_id == superseded:
                        c.superseded_by = s.signal_id
                        self.store.update(c)      # persist the [older] mark (survives restart)
                        break

    def _store_extraction(self, extraction: ExtractionResult) -> None:
        self.ingest_tokens += extraction.tokens
        kept = self.salience.filter(extraction.signals).kept
        kept = self._dedup(kept)                     # product: drop near-duplicate facts (off by default)
        if self.pin_threshold is not None:           # STO-2728: pin high-importance staples
            for s in kept:                           # (always retrieved + decay-exempt)
                if s.is_fact and s.importance_base >= self.pin_threshold:
                    s.pinned = True
        self.store.add(kept)

    def _dedup(self, signals):
        """Semantic dedup for the product path: skip a new fact whose top cosine to an existing
        one clears ``dedup_threshold``. Off by default (None) so the benchmark is unaffected."""
        if self.dedup_threshold is None or not hasattr(self.store, "search_scored"):
            return signals
        kept = []
        for s in signals:
            if s.is_fact:
                scored = self.store.search_scored(s.content or s.title, 1)
                if scored and scored[0][1] >= self.dedup_threshold:
                    continue                         # near-duplicate already stored -> skip
            kept.append(s)
        return kept

    def _consolidate(self) -> None:
        if self.summariser is None or self.consolidate_threshold is None:
            return
        from signal_engine.consolidation import consolidate
        summary, _archived = consolidate(
            self.store.all_signals(), self.summariser,
            threshold=self.consolidate_threshold, keep_recent=self.keep_recent)
        if summary is not None and summary.signal_id is None:   # a NEW summary -> store it
            self.store.add([summary])

    def wiki(self) -> str:
        """A markdown index over current memory for cheap 'where is X' routing (STO-2731)."""
        from signal_engine.wiki import build_wiki
        return build_wiki(self.store.all_signals())

    def retrieve(self, query: str, task_context: Optional[str] = None,
                 top_k: Optional[int] = None) -> List[str]:
        result = self.ranker.retrieve(query, task_context, top_k=top_k)
        self.retrieve_tokens += result.tokens
        snippets = [signal_to_context(s) for s in result.signals]
        if self.procedural:                          # STO-2729: corrective behavioural memory for this task
            behav = [s for s in self.store.search_similar(task_context or query, 5)
                     if s.is_behavioural and (s.friction or s.tool_choice)]
            if behav:
                snippets = snippets + [f"[procedural] {signal_to_context(behav[0])}"]
        if self.persona:                             # STO-2730: stable identity/tone layer, first
            snippets = [f"[persona] {self.persona}"] + snippets
        return snippets


def build_signal_engine(
    scribbler: Scribbler,
    embedder: Embedder,
    reranker: Reranker,
    *,
    use_chroma: bool = False,
    **kwargs,
) -> SignalEngine:
    """Construct a fresh engine (fresh store). ``use_chroma=False`` uses the in-memory store."""
    if use_chroma:
        from signal_engine.store import ChromaStore

        store: VectorStore = ChromaStore(embedder)
    else:
        store = InMemoryStore(embedder)
    return SignalEngine(scribbler, reranker, store, **kwargs)
