"""LLM-as-retriever memory service — the NON-VECTOR product path (STO-28xx).

No embedder: a single LLM does everything. ``remember`` extracts signals (scribbler LLM);
``recall`` hands the LLM a list of candidate manifests (title + keywords) and lets it pick the
relevant ones BY MEANING — which fixes the vocabulary-mismatch misses that cosine similarity
caused, and needs no vector database. A direct plug for a Claude-only setup.

Retrieval cost model (the thing we're testing): each recall is ONE LLM call over the candidate
manifests. At personal scale we hand it ALL cards (they're short); past a threshold we keyword-
prefilter first so the LLM never reads an unbounded list. ``last_recall_tokens`` records the cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from signal_engine.extractor import Extractor
from signal_engine.salience import SalienceFilter


@dataclass
class _Turn:
    role: str
    content: str


@dataclass
class _Session:
    turns: List[_Turn]
    date: str = ""
    session_id: Optional[str] = None


_SUMMARY_TITLE = "[summary]"

# Product extraction prompt (NON-VECTOR path only — the frozen benchmark prompt is untouched).
# Adds the two curation rules the raw prompt lacked: settle research to the DECISION (noise), and
# don't re-emit a fact already stated (dedup assist). Appended to the base prompt.
_PRODUCT_EXTRACT_EXTRA = """

Curation rules (important):
- If the user explores several options and then SETTLES on one, store only the DECISION (you may
  note briefly what it was chosen over) — do NOT store each explored/abandoned option as its own
  durable fact. Research that bounced around and was dropped is throwaway, not memory.
- State each durable fact ONCE, in its clearest form. Do not emit near-duplicate restatements of
  the same fact.
"""


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class LLMMemoryService:
    extractor: "object"                 # Extractor (wraps a scribbler LLM)
    reranker: "object"                  # the LLM router (reads manifests, returns relevant ids)
    store: "object"                     # KeywordStore (no embeddings)
    salience: "object" = None
    summariser: "object" = None         # folds decayed hot cards into a summary (None = no hierarchy)
    recall_k: int = 6                   # max cards returned
    hot_cap: int = 60                   # max HOT cards the LLM reads/query (= your tokens/query budget)
    cards_per_summary: int = 10         # a summary represents this many decayed cards, then a new one
    dedup_threshold: float = 0.6        # skip a new fact whose word-overlap with an existing one >= this
    search_summaries: bool = True       # ALWAYS also route the summary index, not only on a hot miss
    last_recall_tokens: int = 0         # retrieval LLM tokens for the most recent recall
    last_ingest_tokens: int = 0
    last_recall_via: str = "hot"        # "hot" | "summary" — which tier answered
    retrieve_tokens: int = 0            # running total (the BEAM harness reads this)
    _seq: int = field(default=0)

    def __post_init__(self):
        if self.salience is None:
            self.salience = SalienceFilter()

    # -- the hot tier / summaries -----------------------------------------------------------

    def _hot(self):
        return [s for s in self.store.all_signals()
                if s.is_fact and not s.archived and s.title != _SUMMARY_TITLE]

    def _summaries(self):
        return [s for s in self.store.all_signals() if s.title == _SUMMARY_TITLE]

    def _decay_if_needed(self):
        """When the hot tier exceeds hot_cap, fold the OLDEST cards_per_summary cards into a new
        summary (with source_ids + a manifest of their titles for sharp routing) and archive them.
        Deterministic, append-only — exactly Nick's design."""
        if self.summariser is None:
            return
        hot = self._hot()                                 # in insertion (≈ age) order
        while len(hot) > self.hot_cap:
            batch = hot[: self.cards_per_summary]          # the oldest N
            prose = self.summariser.summarise([s.content or s.title for s in batch])
            manifest = "; ".join((s.title or "")[:50] for s in batch)   # sharp terms for routing
            from signal_engine.signal import fact_signal
            summ = fact_signal(title=_SUMMARY_TITLE, content=prose, description=manifest,
                               keywords=sorted({k for s in batch for k in (s.keywords or [])}),
                               importance_base=0.6)
            summ.source_ids = [s.signal_id for s in batch]
            for s in batch:
                s.archived = True                          # parked, never deleted; drillable via source_ids
                self.store.update(s)
            self.store.add([summ])
            hot = self._hot()

    def remember(self, text: str, *, date: str = "") -> str:
        text = (text or "").strip()
        if not text:
            return "Nothing to remember (empty text)."
        before = self.store.count()
        self._seq += 1
        session = _Session(turns=[_Turn(role="user", content=text)],
                           date=date, session_id=f"mem-{self._seq}")
        result = self.extractor.extract(session)
        self.last_ingest_tokens = result.tokens
        self.store.add(self._dedup(self.salience.filter(result.signals).kept))
        self._decay_if_needed()                            # fold oldest hot cards into summaries
        stored = self.store.count() - before
        return f"Remembered — stored {stored} signal(s)." if stored else \
            "Noted, but nothing durable was extracted."

    def _route(self, query, candidates):
        """One LLM routing call over a candidate manifest set → picked signals (accumulates tokens).
        Does NOT filter archived — the caller decides the tier (hot excludes archived; the drill
        deliberately routes over the archived cards a summary points to)."""
        candidates = list(candidates)
        if not candidates:
            return []
        ids, tokens = self.reranker.rerank(query, None, candidates, self.recall_k)
        self.last_recall_tokens += tokens
        by_id = {s.signal_id: s for s in candidates}
        return [by_id[i] for i in ids if i in by_id]

    def recall(self, query: str) -> List[str]:
        """Tiered: read the small HOT tier first (cheap, the relevant recent cards). Only if the
        hot tier has nothing relevant, ESCALATE — route over the summaries, drill into the chosen
        summary's source cards, and route within those. Cost stays ~flat as memory grows because
        the hot tier is capped and summaries stay few."""
        from signal_engine.engine import signal_to_context

        query = (query or "").strip()
        self.last_recall_tokens = 0
        self.last_recall_via = "hot"
        if not query or self.store.count() == 0:
            return []

        picked = self._route(query, self._hot())          # tier 1 — hot
        # Always ALSO route the summary index (not only on an empty hot) — otherwise a marginally-
        # relevant hot card blocks escalation and a decayed fact is missed (Nick's fix). Slightly
        # more tokens, materially better recall. Set search_summaries=False for the cheap mode.
        summaries = self._summaries()
        if summaries and (self.search_summaries or not picked):
            seen = {s.signal_id for s in picked}
            by_id = {s.signal_id: s for s in self.store.all_signals()}
            for summ in self._route(query, summaries):
                cluster = [by_id[i] for i in (summ.source_ids or []) if i in by_id]
                for d in self._route(query, cluster):      # drill into the summary's cards
                    if d.signal_id not in seen:
                        picked.append(d); seen.add(d.signal_id)
                        self.last_recall_via = "hot+summary"
                if len(picked) >= self.recall_k:
                    break
        picked.sort(key=lambda s: 0 if s.is_current else 1)   # current before [older]
        return [signal_to_context(s) for s in picked[: self.recall_k]]

    # -- MemoryEngine contract (so the BEAM harness can benchmark this path) ---

    def _dedup(self, new_signals):
        """Drop a new FACT whose word-overlap with an existing current fact clears dedup_threshold —
        so a running conversation restating the same thing doesn't pile up duplicate cards. Cheap
        (no LLM), errs toward keep. Behavioural signals pass through."""
        from signal_engine.ranker import _words
        existing = [set(_words(s.content or s.title)) for s in self.store.all_signals()
                    if s.is_fact and s.is_current and s.title != _SUMMARY_TITLE]
        kept = []
        for s in new_signals:
            if s.is_fact:
                w = set(_words(s.content or s.title))
                if any(_jaccard(w, e) >= self.dedup_threshold for e in existing):
                    continue                               # near-duplicate -> skip
                existing.append(w)
            kept.append(s)
        return kept

    def ingest(self, session) -> None:
        result = self.extractor.extract(session)           # a real BEAM session (turns)
        self.last_ingest_tokens = result.tokens
        self.store.add(self._dedup(self.salience.filter(result.signals).kept))
        self._decay_if_needed()

    def ingest_many(self, sessions) -> None:
        for s in sessions:
            self.ingest(s)

    def retrieve(self, query: str, task_context=None) -> List[str]:
        hits = self.recall(query)
        self.retrieve_tokens += self.last_recall_tokens    # running total for tokens/query
        return hits

    # -- construction ---------------------------------------------------------

    @staticmethod
    def build_default(persist_path: Optional[str] = None):
        """Fully-LLM service (no embedder): scribbler + reranker from the configured provider,
        a KeywordStore on disk. With SIGNAL_LLM_PROVIDER=anthropic this is a Claude-only memory."""
        import os
        from signal_engine.keyword_store import KeywordStore
        from signal_engine.providers import build_brain, build_summariser

        from signal_engine.extractor import EXTRACTION_PROMPT

        scribbler, reranker = build_brain()
        path = persist_path or os.path.expanduser("~/.signal_engine/llm_memory.json")
        product_prompt = EXTRACTION_PROMPT.replace("\nConversation:",
                                                   _PRODUCT_EXTRACT_EXTRA + "\nConversation:")
        return LLMMemoryService(extractor=Extractor(scribbler, chunk_turns=10, prompt=product_prompt),
                                reranker=reranker, store=KeywordStore(path),
                                summariser=build_summariser())
