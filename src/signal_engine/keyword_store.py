"""Non-vector store — signals on disk as JSON, keyword index, **NO embeddings**.

The vector store (``ChromaStore``) is the benchmark-proven path and is untouched. This is the
alternative for the **LLM-as-retriever** product: retrieval is done by an LLM reading candidate
manifests, so no embedding model is needed — a *single* LLM (e.g. Claude) does extraction AND
retrieval, a direct plug with no second model to configure.

Persists to a plain JSON file (signals are ``to_dict``/``from_dict`` serialisable). One corrupt
row is skipped, not fatal (same durability rule as the vector store).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import List, Optional, Sequence

from signal_engine.signal import Signal
from signal_engine.signal import _keyword_hits


class _FileLock:
    """Tiny cross-platform advisory lock (atomic O_EXCL lockfile). Serialises the read-merge-write of
    a shared scope file so two agents can't interleave. Breaks a stale lock after ``timeout`` so a
    crashed writer can't wedge the scope forever. No-op when there's no path."""

    def __init__(self, target: str, timeout: float = 8.0, poll: float = 0.02):
        self._lockfile = target + ".lock"
        self._timeout = timeout
        self._poll = poll
        self._fd = None

    def __enter__(self):
        start = time.monotonic()
        while True:
            try:
                self._fd = os.open(self._lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.monotonic() - start > self._timeout:
                    try:
                        os.remove(self._lockfile)          # assume stale (crashed writer); reclaim
                    except OSError:
                        pass
                else:
                    time.sleep(self._poll)

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            os.remove(self._lockfile)
        except OSError:
            pass


class KeywordStore:
    """Duck-types the bits of the store interface the LLM path needs: ``add`` / ``all_signals`` /
    ``count`` / ``search_keyword`` / ``update``. No ``search_similar`` (that's the vector path)."""

    name = "keyword"

    def __init__(self, path: Optional[str] = None):
        self._path = path
        self._signals: List[Signal] = []
        self._next_id = 0
        # A per-instance token makes new ids (``sig-<n>-<token>``) globally unique, so two agents
        # sharing one scope file never mint the same id for different cards. ``_last_mtime`` lets a
        # save skip the merge re-read unless another writer actually touched the file (keeps the
        # single-writer path cheap — no O(n^2) on bulk ingest).
        self._token = uuid.uuid4().hex[:6]
        self._last_mtime: Optional[float] = None
        if path and os.path.exists(path):
            self._load()

    def _rows_from(self, path: str):
        try:
            return json.load(open(path))
        except Exception:
            return []

    def _seq_of(self, sid: str) -> int:
        parts = (sid or "").split("-")
        try:
            return int(parts[1]) if len(parts) >= 2 else -1
        except ValueError:
            return -1

    def _load(self) -> None:
        max_seq = -1
        for d in self._rows_from(self._path):
            try:
                s = Signal.from_dict(d)                 # skip a corrupt row, don't brick the store
            except Exception:
                continue
            self._signals.append(s)
            max_seq = max(max_seq, self._seq_of(s.signal_id or ""))
        self._next_id = max_seq + 1
        try:
            self._last_mtime = os.path.getmtime(self._path)
        except OSError:
            self._last_mtime = None

    @property
    def path(self) -> Optional[str]:
        return self._path

    def _save(self) -> None:
        if not self._path:
            return
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with _FileLock(self._path):
            # If another agent sharing this scope wrote since we last did, MERGE their new cards in
            # (union by id) so a concurrent ADD is never clobbered. Skipped when we're the sole writer
            # (mtime unchanged) — the common case — so bulk ingest stays fast.
            try:
                changed = os.path.exists(self._path) and os.path.getmtime(self._path) != self._last_mtime
            except OSError:
                changed = False
            if changed:
                have = {s.signal_id for s in self._signals}
                for d in self._rows_from(self._path):
                    sid = d.get("signal_id")
                    if sid and sid not in have:
                        try:
                            self._signals.append(Signal.from_dict(d))
                            have.add(sid)
                        except Exception:
                            pass
            # Atomic write (temp + rename): a reader never sees a half-written file.
            tmp = self._path + "." + self._token + ".tmp"
            with open(tmp, "w") as f:
                json.dump([s.to_dict() for s in self._signals], f)
            os.replace(tmp, self._path)
            try:
                self._last_mtime = os.path.getmtime(self._path)
            except OSError:
                self._last_mtime = None

    def add(self, signals: Sequence[Signal]) -> None:
        for s in signals:
            if s.signal_id is None:
                s.signal_id = f"sig-{self._next_id}-{self._token}"   # globally unique across writers
                self._next_id += 1
            self._signals.append(s)
        self._save()

    def sync(self) -> None:
        """Reload from disk IF another process changed the file since our last load — so reads in a
        SHARED scope reflect peers' writes (the swarm handshake / id_card stays current). Cheap when
        unchanged (a single mtime stat, no reload); a full reload only when the file actually changed.
        Called from the read-only peer-facing paths, never mid-write, so no unsaved change is lost."""
        if not self._path or not os.path.exists(self._path):
            return
        try:
            m = os.path.getmtime(self._path)
        except OSError:
            return
        if m != self._last_mtime:
            self._signals = []
            self._next_id = 0
            self._load()

    def all_signals(self) -> List[Signal]:
        return list(self._signals)

    def count(self) -> int:
        return len(self._signals)

    def search_keyword(self, words: Sequence[str], k: int = 10) -> List[Signal]:
        return _keyword_hits(self._signals, words, k)

    def update(self, signal: Signal) -> None:
        self._save()            # signals are held by reference, so field changes are already live

    def update_many(self, signals: Sequence[Signal]) -> None:
        """Persist a batch of in-memory field changes with ONE file write (not one per signal).
        Avoids O(n^2) rewrites when archiving thousands of cards during a large decay."""
        self._save()            # all changes are live by reference; a single save captures them

    def delete(self, ids: Sequence[str]) -> int:
        """Hard-remove cards by id (the 'forget me' / privacy path — the only destructive op; the
        lifecycle otherwise decays + archives, never deletes). Also strips the removed ids from any
        summary's ``source_ids`` so drill-back stays consistent. Returns the count removed."""
        drop = set(ids)
        if not drop:
            return 0
        before = len(self._signals)
        self._signals = [s for s in self._signals if s.signal_id not in drop]
        removed = before - len(self._signals)
        for s in self._signals:                     # keep summaries' source lists consistent
            src = getattr(s, "source_ids", None)
            if src:
                kept = [i for i in src if i not in drop]
                if len(kept) != len(src):
                    s.source_ids = kept
        if removed:
            self._save()
        return removed
