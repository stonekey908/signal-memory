"""Consolidation + archive (STO-2726 / STO-2727) — bounded memory without losing anything.

Once memory grows past a threshold, the OLDEST non-pinned facts are folded **additively** into a
running summary signal (added to, not replacing) and the originals are marked ``archived`` — kept
in the store, still there, but out of the hot retrieval path. The summary is itself a retrievable
fact, so old context stays reachable cheaply (token-efficient). Pluggable Summariser: ``Mock``
(no key) + ``OpenAI`` (real). Nothing is ever deleted.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from signal_engine.openai_client import openai_client
from signal_engine.signal import Signal, fact_signal

SUMMARY_TITLE = "Summary of older memories"


@runtime_checkable
class Summariser(Protocol):
    def summarise(self, facts: Sequence[str], prior: str = "") -> str: ...


class MockSummariser:
    """No-key summariser: concatenate (additively onto the prior summary). For tests."""

    name = "mock"

    def summarise(self, facts: Sequence[str], prior: str = "") -> str:
        joined = "; ".join(f for f in facts if f)
        return f"{prior} {joined}".strip() if prior else joined


_SUMMARISE_PROMPT = """Fold these older memory notes into a concise running summary, ADDING to the
existing summary — do not drop anything important, keep it compact.

EXISTING SUMMARY:
{prior}

OLDER NOTES:
{notes}

Return only the updated summary text."""


class OpenAISummariser:
    """Real summariser (defaults to the cheap gpt-4o-mini). Requires OPENAI_API_KEY."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 client_factory=openai_client, max_tokens: Optional[int] = None):
        self.model = model
        self._api_key = api_key
        self._client_factory = client_factory      # anthropic_client for a Claude summariser
        self.max_tokens = max_tokens

    def summarise(self, facts: Sequence[str], prior: str = "") -> str:
        client = self._client_factory(self._api_key)
        notes = "\n".join(f"- {f}" for f in facts if f)
        kwargs = {"model": self.model, "temperature": 0,
                  "messages": [{"role": "user",
                                "content": _SUMMARISE_PROMPT.format(prior=prior or "(none)", notes=notes)}]}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        resp = client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()


def consolidate(
    signals: List[Signal], summariser: Summariser, *, threshold: int, keep_recent: int = 20,
) -> Tuple[Optional[Signal], List[Signal]]:
    """If active (non-archived, non-pinned) facts exceed ``threshold``, fold all but the most recent
    ``keep_recent`` into/onto the running summary. Returns (summary_signal, archived_originals).
    ``summary_signal`` is a NEW signal to add, or the existing summary (updated in place), or None.
    """
    active = [s for s in signals if s.is_fact and not s.archived and not s.pinned
              and s.title != SUMMARY_TITLE]
    if len(active) <= threshold:
        return None, []
    old = active[:-keep_recent] if keep_recent else active
    if not old:
        return None, []
    existing = next((s for s in signals if s.is_fact and s.title == SUMMARY_TITLE), None)
    prior = existing.content if (existing and existing.content) else ""
    text = summariser.summarise([s.content or s.title for s in old], prior=prior)
    for s in old:                                  # archived, NOT deleted — still in the store
        s.archived = True
    if existing:                                   # additive: grow the running summary in place
        existing.content = text
        return existing, old
    return fact_signal(SUMMARY_TITLE, text, keywords=["summary"], importance_base=0.6), old
