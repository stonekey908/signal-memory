"""Ranker / retrieval with dynamic task-relative importance (the differentiator).

At query time:

1. **Query-driven shortlist** — semantic (embedding) + keyword search over the store.
2. **AI re-rank** (the primary importance decision) — a pluggable re-ranker reads the
   candidates' short summaries + the ``task_context`` + each signal's age, and orders
   the most relevant first. **Relevance to the query is primary; ``task_context`` is a
   secondary modifier** (so a query that diverges from the ongoing task still retrieves
   well). No weighted formula decides importance.
3. **Pins are always eligible** (staples). Returns a token-bounded top set; counts every
   re-rank token.

Out of scope: the reader (harness), consolidation, Wave-3 tuning.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import signal_engine.config  # noqa: F401  — importing loads .env (API keys)
from signal_engine.openai_client import openai_client
from signal_engine.signal import Signal
from signal_engine.store import VectorStore

_WORD = re.compile(r"[a-z0-9]+")
_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


def _words(text: Optional[str]) -> List[str]:
    return _WORD.findall((text or "").lower())


def _summary(s: Signal) -> str:
    """Short, cheap summary shown to the re-ranker (title + description, NOT full content)."""
    return " — ".join(b for b in (s.title, s.description) if b)


@dataclass
class RankResult:
    signals: List[Signal]
    tokens: int


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, task_context: Optional[str], candidates: Sequence[Signal], top_k: int
    ) -> Tuple[List[str], int]:
        """Return (ordered signal_ids, most relevant first; tokens_used)."""
        ...


class MockReranker:
    """No-key, deterministic, TASK-AWARE re-ranker: orders candidates by word overlap with
    (query + task_context). Task-sensitive, so tests can prove ``task_context`` re-orders.
    """

    name = "mock"

    def rerank(self, query, task_context, candidates, top_k):
        terms = set(_words(query)) | set(_words(task_context))

        def score(s: Signal) -> int:
            text = set(_words(_summary(s))) | {w.lower() for w in s.keywords} | set(_words(s.content))
            return len(terms & text)

        ordered = sorted(candidates, key=lambda s: (-score(s), s.signal_id or ""))
        ids = [s.signal_id for s in ordered[:top_k]]
        tokens = len(terms) + sum(len(_words(_summary(s))) for s in candidates)  # rough estimate
        return ids, tokens


RERANK_PROMPT = """You select which of the user's stored memories are most relevant RIGHT NOW.

The QUESTION/REQUEST is what to serve — relevance to it is PRIMARY. The TASK CONTEXT is
what the agent is broadly doing — use it only as a SECONDARY modifier of importance; do NOT
discard memories relevant to the question just because they differ from the task. Consider
each memory's age (recency matters, but recent != important).

QUESTION: {query}
TASK CONTEXT: {task}

MEMORIES (id | summary | age):
{candidates}

Return ONLY JSON: {{"ids": ["id1", "id2", ...]}} — most relevant first, at most {top_k} ids.
"""


def _parse_ids(text: str) -> List[str]:
    from signal_engine.extractor import _extract_json_object
    cleaned = _FENCE.sub("", text or "").strip() or "{}"
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:                         # chatty model wrapped JSON in prose
        salvaged = _extract_json_object(cleaned)
        if not salvaged:
            return []                                    # unparseable -> let the ranker backfill
        data = json.loads(salvaged)
    if isinstance(data, dict):
        data = data.get("ids", [])
    return [str(i) for i in (data or [])]


class OpenAIReranker:
    """Real AI re-ranker (temperature 0). Provider-agnostic: ``client_factory`` selects the
    backend (default OpenAI; pass ``anthropic_client`` for Claude — its endpoint is OpenAI-
    compatible). For Claude, set ``json_format=False`` (the compat layer ignores it) and a
    ``max_tokens`` (Anthropic requires it); ``_parse_ids`` already tolerates prose-wrapped JSON."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None,
                 client_factory=openai_client, json_format: bool = True,
                 max_tokens: Optional[int] = None):
        self.model = model
        self._api_key = api_key   # resolved at call time by the shared client factory
        self._client_factory = client_factory
        self.json_format = json_format
        self.max_tokens = max_tokens

    def rerank(self, query, task_context, candidates, top_k):
        client = self._client_factory(self._api_key)   # shared client: timeout + retry, no leak
        lines = "\n".join(                       # truncated summaries: enough to judge relevance,
            f'{s.signal_id} | {_summary(s)[:100]} | age={s.created_at or "?"}'   # not to re-read memory
            for s in candidates
        )
        prompt = RERANK_PROMPT.format(
            query=query, task=task_context or "(none)", candidates=lines, top_k=top_k
        )
        kwargs = {"model": self.model, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]}
        if self.json_format:
            kwargs["response_format"] = {"type": "json_object"}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        resp = client.chat.completions.create(**kwargs)
        ids = _parse_ids(resp.choices[0].message.content or "{}")
        tokens = resp.usage.total_tokens if resp.usage else 0
        return ids, tokens


class Ranker:
    """Retrieves a token-bounded, task-relative set of Signals for a query."""

    def __init__(self, store: VectorStore, reranker: Reranker, top_k: int = 5, shortlist_k: int = 25):
        self.store = store
        self.reranker = reranker
        self.top_k = top_k
        self.shortlist_k = shortlist_k

    def retrieve(self, query: str, task_context: Optional[str] = None,
                 top_k: Optional[int] = None) -> RankResult:
        """``top_k`` overrides the configured depth for THIS query (adaptive retrieval:
        coverage questions want breadth, precision questions want focus)."""
        k = top_k or self.top_k
        # 1. query-driven shortlist: semantic + keyword + pins (all eligible)
        cands: dict = {}
        for s in self.store.search_similar(query, self.shortlist_k):
            cands[s.signal_id] = s
        for s in self.store.search_keyword(_words(query), self.shortlist_k):
            cands.setdefault(s.signal_id, s)
        for s in self.store.pinned():
            cands.setdefault(s.signal_id, s)
        candidates = [s for s in cands.values() if not s.archived]   # archived -> out of hot path (STO-2727)
        if not candidates:
            return RankResult([], 0)

        # 2. AI re-rank (the primary importance decision) — with cost guards (the rerank was
        #    ~5.3k tok/query at 1M, erasing the efficiency edge):
        #    (a) if the budget takes (nearly) everything anyway, ranking adds nothing — skip the
        #        LLM call and keep shortlist order (semantic hits first);
        #    (b) otherwise the judge only needs to SEE a bounded view of the best candidates —
        #        backfill below still draws from the FULL pool, so nothing becomes unreachable.
        if k >= len(candidates):
            ordered_ids, tokens = [s.signal_id for s in candidates], 0
        else:
            view = candidates[: max(60, int(1.2 * k))]
            ordered_ids, tokens = self.reranker.rerank(query, task_context, view, k)
        by_id = {s.signal_id: s for s in candidates}
        ranked_non_pins = [
            by_id[i] for i in ordered_ids if i in by_id and not by_id[i].pinned
        ]

        # 3. pins always kept; then the reranker's order; then BACKFILL from the shortlist.
        #    The reranker often returns far fewer than top_k (observed 1-3), dropping facts a
        #    multi-fact question needs — so fill any remaining slots with the best shortlist
        #    candidates it didn't pick, in relevance order, instead of wasting the budget (STO-2790).
        pins = [s for s in candidates if s.pinned]
        chosen = {s.signal_id for s in pins} | {s.signal_id for s in ranked_non_pins}
        backfill = [s for s in candidates if not s.pinned and s.signal_id not in chosen]
        ordered_non_pins = ranked_non_pins + backfill
        slots = max(0, k - len(pins))
        final = pins + ordered_non_pins[:slots]
        return RankResult(final, tokens)
