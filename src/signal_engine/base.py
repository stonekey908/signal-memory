"""The engine's public contract (FR-11).

Both the Wave-0 trivial retriever (in ``harness/``) and the Wave-1 real Signal
Engine implement this, so they are interchangeable inside the benchmark harness.
Everything else about an implementation is internal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class TurnLike(Protocol):
    """One chat turn — role + content (the harness ``Turn`` satisfies this)."""

    role: str
    content: str


@runtime_checkable
class SessionLike(Protocol):
    """Anything with an ordered list of turns (the harness ``Session`` satisfies this)."""

    turns: Sequence[TurnLike]


class MemoryEngine(ABC):
    """Two-method memory contract: ``ingest`` (write path) and ``retrieve`` (read path)."""

    @abstractmethod
    def ingest(self, session: SessionLike) -> None:
        """Store one session's content."""
        raise NotImplementedError

    def ingest_many(self, sessions: Sequence[SessionLike]) -> None:
        """Ingest several sessions. Default = sequential; engines whose per-session work
        is independent and I/O-bound may override to parallelise (see ``SignalEngine``)."""
        for session in sessions:
            self.ingest(session)

    @abstractmethod
    def retrieve(self, query: str, task_context: Optional[str] = None) -> List[str]:
        """Return a ranked, token-bounded list of context snippets for the query."""
        raise NotImplementedError
