"""Signal Store — persists Signals with embeddings + keyword metadata for retrieval.

A clean ``VectorStore`` interface with two implementations:

* ``InMemoryStore`` — brute-force cosine + keyword overlap (the lightweight fake for tests).
* ``ChromaStore``  — embedded Chroma (cosine, tag-filtering) for real use.

Both take a pluggable ``Embedder``. The ranker (STO-2724) composes ``search_similar`` +
``search_keyword``. Out of scope here: ranking/importance/decay, consolidation.
"""

from __future__ import annotations

import itertools
import json
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

# Chroma's default client caches collections per process, so each ChromaStore gets a
# unique collection name for isolation (fresh store per benchmark question).
_collection_seq = itertools.count()

from signal_engine.embedder import Embedder
from signal_engine.signal import Signal


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def signal_text(s: Signal) -> str:
    """The text we embed / keyword-match for a signal (title + description + content + meta)."""
    parts = [s.title, s.description or "", s.content or "",
             s.sentiment or "", s.friction or "", s.tool_choice or "", s.interaction or ""]
    parts += s.keywords
    return " ".join(p for p in parts if p)


def _keyword_hits(signals, words: Sequence[str], k: int) -> List[Signal]:
    wset = {w.lower() for w in words}
    scored = []
    for s in signals:
        tokens = {w.lower() for w in s.keywords} | set(signal_text(s).lower().split())
        overlap = len(wset & tokens)
        if overlap:
            scored.append((overlap, s))
    scored.sort(key=lambda p: -p[0])
    return [s for _, s in scored[:k]]


class VectorStore(ABC):
    """The swappable store seam. FAISS/Qdrant could implement this later."""

    @abstractmethod
    def add(self, signals: Sequence[Signal]) -> None: ...

    @abstractmethod
    def search_similar(self, query: str, k: int = 10) -> List[Signal]: ...

    @abstractmethod
    def search_keyword(self, words: Sequence[str], k: int = 10) -> List[Signal]: ...

    @abstractmethod
    def pinned(self) -> List[Signal]: ...   # pins are always eligible for retrieval

    @abstractmethod
    def all_signals(self) -> List[Signal]: ...   # every stored signal (wiki, consolidation)

    @abstractmethod
    def count(self) -> int: ...


class InMemoryStore(VectorStore):
    """Brute-force cosine + keyword overlap. The lightweight fake used in tests."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._signals: List[Signal] = []
        self._embeddings: List[List[float]] = []
        self._next_id = 0

    def add(self, signals: Sequence[Signal]) -> None:
        signals = list(signals)
        if not signals:
            return
        vecs = self.embedder.embed([signal_text(s) for s in signals])
        for s, v in zip(signals, vecs):
            if s.signal_id is None:
                s.signal_id = f"sig-{self._next_id}"
                self._next_id += 1
            self._signals.append(s)
            self._embeddings.append(v)

    def search_similar(self, query: str, k: int = 10) -> List[Signal]:
        if not self._signals:
            return []
        qv = self.embedder.embed([query])[0]
        ranked = sorted(
            zip(self._signals, self._embeddings), key=lambda p: -_cosine(qv, p[1])
        )
        return [s for s, _ in ranked[:k]]

    def search_keyword(self, words: Sequence[str], k: int = 10) -> List[Signal]:
        return _keyword_hits(self._signals, words, k)

    def best_similarity(self, query: str) -> float:
        """Highest cosine similarity of ``query`` against any stored signal (0..1; 0 if empty).
        Lets a caller tell "nothing relevant" from "here's the best match" — a relevance floor."""
        if not self._signals:
            return 0.0
        qv = self.embedder.embed([query])[0]
        return max(_cosine(qv, e) for e in self._embeddings)

    def search_scored(self, query: str, k: int = 10) -> List[Tuple[Signal, float]]:
        """Top-k (signal, cosine-similarity) pairs, most similar first — lets a caller keep only
        genuinely relevant memories (per-item relevance floor) instead of a fixed top-k."""
        if not self._signals:
            return []
        qv = self.embedder.embed([query])[0]
        scored = [(s, _cosine(qv, e)) for s, e in zip(self._signals, self._embeddings)]
        scored.sort(key=lambda p: -p[1])
        return scored[:k]

    def pinned(self) -> List[Signal]:
        return [s for s in self._signals if s.pinned]

    def all_signals(self) -> List[Signal]:
        return list(self._signals)

    def update(self, signal: Signal) -> None:
        """No-op: this store holds live object references, so field changes (e.g. superseded_by)
        are already reflected. Present for interface parity with the persistent store."""

    def count(self) -> int:
        return len(self._signals)


class ChromaStore(VectorStore):
    """Embedded Chroma (cosine) + our own keyword pass. Real store."""

    def __init__(self, embedder: Embedder, collection_name: Optional[str] = None, persist_path=None):
        import chromadb

        self.embedder = embedder
        client = (
            chromadb.PersistentClient(path=str(persist_path))
            if persist_path
            else chromadb.EphemeralClient()
        )
        name = collection_name or f"signals_{next(_collection_seq)}"   # unique -> isolated
        self._col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self._by_id: Dict[str, Signal] = {}
        self._next_id = 0
        self._rehydrate()   # persistent reuse: rebuild the Signal cache from disk (no-op if empty)

    @staticmethod
    def _meta(s: Signal) -> dict:
        # Chroma metadata must be str/int/float/bool — flatten keywords to a string. ``signal_json``
        # carries the FULL signal so a persistent store can perfectly reconstruct it on reload
        # (STO-2804 — memory that survives across processes, not just within one benchmark run).
        return {
            "kind": s.kind,
            "pinned": s.pinned,
            "session_ref": s.session_ref or "",
            "keywords": ",".join(s.keywords),
            "importance_base": s.importance_base,
            "signal_json": json.dumps(s.to_dict()),
        }

    def _rehydrate(self) -> None:
        """Rebuild the in-memory Signal cache from a persistent collection so a fresh store
        object at the same path can search/return signals written by an earlier process. A
        no-op for the benchmark (fresh unique collection each run → nothing to load)."""
        if self._col.count() == 0:
            return
        got = self._col.get(include=["metadatas"])
        max_seq = -1
        for sid, meta in zip(got["ids"], got["metadatas"]):
            raw = (meta or {}).get("signal_json")
            if not raw:
                continue                                  # pre-STO-2804 rows (none in practice)
            try:
                s = Signal.from_dict(json.loads(raw))     # one corrupt/legacy row must NOT brick
            except Exception:                             # the whole long-lived store — skip it
                continue
            s.signal_id = sid
            self._by_id[sid] = s
            if sid.startswith("sig-"):                    # keep new ids from colliding with old
                try:
                    max_seq = max(max_seq, int(sid.split("-", 1)[1]))
                except ValueError:
                    pass
        self._next_id = max_seq + 1

    def add(self, signals: Sequence[Signal]) -> None:
        signals = list(signals)
        if not signals:
            return
        texts = [signal_text(s) for s in signals]
        vecs = self.embedder.embed(texts)
        ids, embeddings, docs, metas = [], [], [], []
        for s, v, text in zip(signals, vecs, texts):
            if s.signal_id is None:
                s.signal_id = f"sig-{self._next_id}"
                self._next_id += 1
            ids.append(s.signal_id)
            embeddings.append(v)
            docs.append(text)
            metas.append(self._meta(s))
            self._by_id[s.signal_id] = s
        self._col.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)

    def search_similar(self, query: str, k: int = 10) -> List[Signal]:
        n = self._col.count()
        if n == 0:
            return []
        qv = self.embedder.embed([query])[0]
        res = self._col.query(query_embeddings=[qv], n_results=min(k, n))
        return [self._by_id[i] for i in res["ids"][0] if i in self._by_id]   # skip orphan ids

    def search_keyword(self, words: Sequence[str], k: int = 10) -> List[Signal]:
        return _keyword_hits(self._by_id.values(), words, k)

    def best_similarity(self, query: str) -> float:
        """Highest cosine similarity of ``query`` vs any stored signal (0..1; 0 if empty).
        Chroma's cosine space returns distance = 1 - similarity, so we invert the top hit."""
        n = self._col.count()
        if n == 0:
            return 0.0
        qv = self.embedder.embed([query])[0]
        res = self._col.query(query_embeddings=[qv], n_results=1)
        dists = (res.get("distances") or [[]])[0]
        return max(0.0, 1.0 - float(dists[0])) if dists else 0.0

    def search_scored(self, query: str, k: int = 10) -> List[Tuple[Signal, float]]:
        """Top-k (signal, cosine-similarity) pairs, most similar first (similarity = 1 - distance)."""
        n = self._col.count()
        if n == 0:
            return []
        qv = self.embedder.embed([query])[0]
        res = self._col.query(query_embeddings=[qv], n_results=min(k, n))
        ids = res["ids"][0]
        dists = (res.get("distances") or [[]])[0]
        out: List[Tuple[Signal, float]] = []
        for i, sid in enumerate(ids):
            s = self._by_id.get(sid)
            if s is None:
                continue
            sim = max(0.0, 1.0 - float(dists[i])) if i < len(dists) else 0.0
            out.append((s, sim))
        return out

    def pinned(self) -> List[Signal]:
        return [s for s in self._by_id.values() if s.pinned]

    def all_signals(self) -> List[Signal]:
        return list(self._by_id.values())

    def update(self, signal: Signal) -> None:
        """Re-persist a signal's metadata after an in-place change (e.g. reconcile setting
        ``superseded_by``). Without this, versioning annotations are lost on the next restart
        because Chroma still holds the pre-change ``signal_json`` (STO-2804)."""
        if signal.signal_id is None:
            return
        self._by_id[signal.signal_id] = signal
        self._col.update(ids=[signal.signal_id], metadatas=[self._meta(signal)])

    def count(self) -> int:
        return self._col.count()
