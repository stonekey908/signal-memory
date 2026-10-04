"""Reconciler — fact versioning by non-destructive supersede (STO-2800, redesigned).

v1 failed on BEAM: it over-superseded (flagged related-but-distinct facts as replacements) and
retrieval *dropped* the superseded ones, cratering multi-session. This redesign fixes both:

1. **Stricter decision** — supersede only when the new fact is the SAME attribute of the SAME
   subject with an explicitly CHANGED value; default to NO.
2. **Annotate, never drop** — a superseded fact is kept and marked ``[older]`` at retrieval, so a
   wrong guess can never remove a needed fact. Both stay searchable; the reader resolves
   current-vs-historical from the annotation.

The old value is kept, not deleted, so memory can answer "now" AND "before". Pluggable:
``MockReconciler`` (no key) + ``OpenAIReconciler`` (real, strict prompt).
"""

from __future__ import annotations

import json
import re
from typing import Optional, Protocol, Sequence, runtime_checkable

from signal_engine.openai_client import openai_client

_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)


@runtime_checkable
class Reconciler(Protocol):
    def superseded_id(self, new_signal, candidates: Sequence) -> Optional[str]: ...


class MockReconciler:
    """No-key: supersede a candidate with the SAME subject (normalised title) but changed content."""

    name = "mock"

    def superseded_id(self, new_signal, candidates):
        nt, nc = (new_signal.title or "").strip().lower(), (new_signal.content or "").strip()
        for c in candidates:
            if (c.title or "").strip().lower() == nt and (c.content or "").strip() != nc:
                return c.signal_id
        return None


_RECONCILE_PROMPT = """A NEW fact about the user was just recorded. Decide if it is a DIRECT UPDATE to
exactly one EXISTING fact — the SAME attribute of the SAME subject with a CHANGED value (e.g. moved
city, changed job, changed a stated preference). Be STRICT: only if it clearly makes one existing
fact out of date. Related-but-different facts, or added detail, are NOT updates.

NEW: {new}

EXISTING:
{candidates}

Reply ONLY JSON: {{"superseded_id": "<id>"}} if exactly one existing fact is now out of date, else
{{"superseded_id": null}}."""


class OpenAIReconciler:
    """Real strict reconciler (gpt-4o). Only called when candidates exist. Requires OPENAI_API_KEY."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model
        self._api_key = api_key

    def superseded_id(self, new_signal, candidates):
        candidates = list(candidates)
        if not candidates:
            return None
        client = openai_client(self._api_key)
        lines = "\n".join(f"- id={c.signal_id}: {(c.content or c.title)}" for c in candidates)
        resp = client.chat.completions.create(
            model=self.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": _RECONCILE_PROMPT.format(
                new=(new_signal.content or new_signal.title), candidates=lines)}],
        )
        sid = _parse_id(resp.choices[0].message.content or "{}")
        return sid if sid in {c.signal_id for c in candidates} else None


def _parse_id(text: str) -> Optional[str]:
    try:
        d = json.loads(_FENCE.sub("", text or "").strip() or "{}")
    except (ValueError, TypeError):
        return None
    sid = d.get("superseded_id") if isinstance(d, dict) else None
    return str(sid) if sid not in (None, "", "null") else None
