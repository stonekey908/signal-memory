"""The **Signal** — Signal Engine's atomic memory unit.

A Signal deliberately carries **both** factual content (so fact-recall works) **and**
behavioural meta (so the agent adapts). Per ``docs/DECISIONS.md`` there are two shapes,
told apart by ``kind``:

* ``fact``        — a sharp fact/preference; carries ``content`` (what LongMemEval quizzes).
* ``behavioural`` — the "how did this conversation feel" note; carries ``sentiment`` /
  ``friction`` / ``tool_choice`` / ``interaction``.

Field roles (what each field is *for*):

| Role | Fields |
|------|--------|
| **Cheap routing / low-token search** | ``title``, ``description``, ``keywords`` |
| **Semantic (meaning) recall** | ``content`` |
| **Adaptation / behaviour** | ``sentiment``, ``friction``, ``tool_choice``, ``interaction`` |
| **Bookkeeping** | ``session_ref`` (+ ``session_offsets``), ``created_at``, ``importance_base``, ``pinned``, ``signal_id`` |

Notes:
* ``importance_base`` (set by the scribbler at write time) feeds the **salience filter**
  (STO-2721) and **filing speed** (STO-2723) — NOT retrieval ranking (the AI judge owns that).
* ``pinned`` signals are exempt from decay + archival.
* This module is *data only*: the type, (de)serialisation, and validation. No extraction,
  storage, or embeddings (later tickets).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Sequence, List, Optional

KIND_FACT = "fact"
KIND_BEHAVIOURAL = "behavioural"
KINDS = (KIND_FACT, KIND_BEHAVIOURAL)

VOLATILITIES = ("invariant", "decision", "external")
BASES = ("seen", "told", "assumed")          # how the agent KNOWS a fact (see Signal.basis)


def normalise_volatility(v):
    """A fact's declared volatility, lower-cased and validated; anything unknown -> None (no marker)."""
    v = (v or "").strip().lower()
    return v if v in VOLATILITIES else None


def normalise_basis(b):
    """A fact's declared basis, lower-cased and validated; anything unknown -> None."""
    b = (b or "").strip().lower()
    return b if b in BASES else None


@dataclass
class Signal:
    """One memory note. See the module docstring for field roles."""

    kind: str                                   # "fact" | "behavioural"
    title: str                                  # short label (cheap routing)
    description: str = ""                        # one-line summary (cheap routing)
    keywords: List[str] = field(default_factory=list)

    content: Optional[str] = None               # the fact itself (fact signals)
    volatility: Optional[str] = None            # what this fact DEPENDS ON (STO-2846), so a recall can
                                                # flag "verify before acting". One of:
                                                #   invariant — true by a structural reason; ages forever
                                                #   decision  — a choice, holds until the decider changes it
                                                #   external  — a third party gets a vote (GitHub, an API)
                                                # None = unclassified (no marker; the benchmark path).
    basis: Optional[str] = None                 # how the agent KNOWS it (field report, 2026-09-29: an
                                                # agent "held my wrong belief that the CI emails had
                                                # stopped" — its own conclusion, stored as fact). One of:
                                                #   seen     — observed directly: read the file, ran the
                                                #              command, saw the output
                                                #   told     — the user or a document said so
                                                #   assumed  — inferred or concluded, not checked
                                                # None = not recorded (older cards; the benchmark path).

    sentiment: Optional[str] = None             # behavioural meta ...
    friction: Optional[str] = None
    tool_choice: Optional[str] = None
    interaction: Optional[str] = None

    session_ref: Optional[str] = None           # link back to the raw chat session
    session_offsets: Optional[List[int]] = None  # which turn(s) within that session

    created_at: str = ""                        # ISO time — auto-stamped on store (episodic time)
    last_used: str = ""                          # ISO time of the most recent retrieval (recency/staleness)
    importance_base: float = 0.0                 # 0..1, set by the scribbler (salience + filing speed)
    pinned: bool = False                         # exempt from decay + archival
    no_pin: bool = False                          # unpin() override (STO-2848): force OFF the pin set,
                                                 # even if importance would derive a pin — reversible
    signal_id: Optional[str] = None              # assigned by the store (STO-2722)
    archived: bool = False                       # folded into a summary (STO-2726/2727) — kept, never deleted
    superseded_by: Optional[str] = None          # id of the fact that updated this one (STO-2800) — kept + annotated
    source_ids: Optional[List[str]] = None        # for a SUMMARY: the signal ids it folded (hierarchical drill-down)

    # Procedural validation gates (STO-2823). A remembered procedure ("deploy with `uv run deploy`")
    # is capture-only until something validates it, so we keep the EVIDENCE (wins/fails) rather than a
    # lossy trust float — ``trust`` is derived from it. Persisted (unlike the in-memory usage counts):
    # a procedure that failed must stay demoted across restarts.
    wins: int = 0                                # times a reuse of this card was reported successful
    fails: int = 0                               # times a reuse was reported failed
    fail_streak: int = 0                         # CONSECUTIVE failures (a win clears it) — drives retirement
    retired: bool = False                        # consistently failed: out of the hot path, KEPT + drillable

    _BEHAVIOURAL_FIELDS = ("sentiment", "friction", "tool_choice", "interaction")

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Signal.kind must be one of {KINDS}, got {self.kind!r}")
        if not (self.title or "").strip():
            raise ValueError("Signal.title is required (non-empty)")
        if not (0.0 <= self.importance_base <= 1.0):
            raise ValueError(f"Signal.importance_base must be in [0, 1], got {self.importance_base}")
        if self.kind == KIND_FACT:
            if not (self.content or "").strip():
                raise ValueError("a 'fact' Signal must have non-empty content")
        else:  # behavioural
            if not any(getattr(self, f) for f in self._BEHAVIOURAL_FIELDS):
                raise ValueError(
                    "a 'behavioural' Signal must set at least one of "
                    f"{self._BEHAVIOURAL_FIELDS}"
                )

    @property
    def is_current(self) -> bool:
        """True if this fact has not been updated by a newer version (STO-2800)."""
        return self.superseded_by is None

    @property
    def is_fact(self) -> bool:
        return self.kind == KIND_FACT

    @property
    def is_behavioural(self) -> bool:
        return self.kind == KIND_BEHAVIOURAL

    @property
    def has_outcome_evidence(self) -> bool:
        """True once anyone has reported a reuse of this card as worked/failed (STO-2823)."""
        return bool(self.wins or self.fails)

    @property
    def trust(self) -> float:
        """Confidence this procedure still works, DERIVED from the evidence (STO-2823).

        Laplace-smoothed success rate: ``(wins + 1) / (wins + fails + 2)``. No evidence -> 0.5
        (neutral — we don't punish or reward an unvalidated card), and a single result moves it
        without slamming it to 0 or 1, so one flaky run can't condemn a good procedure.
        """
        return (self.wins + 1.0) / (self.wins + self.fails + 2.0)

    def to_dict(self) -> dict:
        """JSON-able dict (round-trips via ``from_dict``)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Signal":
        """Build a Signal from a dict, ignoring unknown keys (revalidates on construction)."""
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def fact_signal(
    title: str,
    content: str,
    *,
    description: str = "",
    keywords: Optional[List[str]] = None,
    session_ref: Optional[str] = None,
    session_offsets: Optional[List[int]] = None,
    created_at: str = "",
    importance_base: float = 0.0,
    pinned: bool = False,
    signal_id: Optional[str] = None,
    volatility: Optional[str] = None,
    basis: Optional[str] = None,
) -> Signal:
    """Construct a ``fact`` Signal (ergonomic helper for the extractor)."""
    return Signal(
        kind=KIND_FACT,
        title=title,
        content=content,
        description=description,
        keywords=list(keywords or []),
        session_ref=session_ref,
        session_offsets=session_offsets,
        created_at=created_at,
        importance_base=importance_base,
        pinned=pinned,
        signal_id=signal_id,
        volatility=normalise_volatility(volatility),
        basis=normalise_basis(basis),
    )


def behavioural_signal(
    title: str,
    *,
    sentiment: Optional[str] = None,
    friction: Optional[str] = None,
    tool_choice: Optional[str] = None,
    interaction: Optional[str] = None,
    description: str = "",
    keywords: Optional[List[str]] = None,
    session_ref: Optional[str] = None,
    created_at: str = "",
    importance_base: float = 0.0,
    pinned: bool = False,
    signal_id: Optional[str] = None,
) -> Signal:
    """Construct a ``behavioural`` Signal (ergonomic helper for the extractor)."""
    return Signal(
        kind=KIND_BEHAVIOURAL,
        title=title,
        sentiment=sentiment,
        friction=friction,
        tool_choice=tool_choice,
        interaction=interaction,
        description=description,
        keywords=list(keywords or []),
        session_ref=session_ref,
        created_at=created_at,
        importance_base=importance_base,
        pinned=pinned,
        signal_id=signal_id,
    )


# --- shared renderers and the keyword fallback ------------------------------------------------------
# These live here, on the leaf module, so the PRODUCT server (dumb_mcp_server -> dumb_memory ->
# keyword_store) can use them without importing the benchmark engine, the vector store, or the
# OpenAI client. The plugin ships only the product modules (scripts/export_plugin.py).

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


_META_FIELDS = (("sentiment", "sentiment"), ("friction", "friction"),
                ("tool_choice", "tool_choice"), ("interaction", "interaction"))


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
