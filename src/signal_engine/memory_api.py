"""Public Python API — embed Signal Engine memory directly in a Python app or agent (no MCP host).

Same local-first engine as the ``signal-dumb-mcp`` server (on-disk index cards, on-device embeddings,
the full decay -> summary -> drill-back lifecycle, pins, versioning): just a clean object instead of a
JSON-RPC stdio server. No cloud, no API key, no server-side LLM call — private and ~$0 by construction.

    from signal_engine import Memory

    mem = Memory(path="~/.signal_engine/my_app.json")     # embed="local" (on-device) by default
    mem.add("Deploys on Railway as of 2026-07", keywords=["deploy"], importance=0.8)
    mem.search("where does it deploy")                    # -> [{"id","title","content", ...}]
    mem.list(pinned=True)                                  # browse what's stored
    mem.delete(["sig-3"])                                  # forget one card (privacy / GDPR)
    mem.stats()

Extraction stays the caller's job — the engine does NO LLM work of its own. Pass a durable fact string
to ``add`` (stored as one card) or fully-structured signals to ``add_many``. In an agent, let your own
model extract and then call ``add_many`` — exactly what an MCP host does for free.
"""

from __future__ import annotations

import os
from typing import List, Optional

from signal_engine.dumb_memory import DumbMemory
from signal_engine.keyword_store import KeywordStore
from signal_engine.signal import Signal


class Memory:
    """A local, private, model-agnostic long-term memory you can call from Python.

    Thin, well-behaved facade over :class:`DumbMemory`. Storage is a single JSON file on disk;
    retrieval runs on an on-device embedder. Everything the MCP server exposes is here as a method.
    """

    def __init__(self, path: Optional[str] = None, scope: Optional[str] = None, embed: str = "local",
                 embed_model: str = "all-MiniLM-L6-v2", wiki_link: bool = True, **engine_kwargs):
        """``path``: an explicit memory file (``~`` expanded). ``scope``: a named scope resolved to a
        shared file (agents sharing a scope name share one memory — a swarm's brain). With neither,
        the store is in-memory (ephemeral). ``embed``: ``"local"`` (on-device, private, free) or
        ``"none"`` (route on manifests). Extra kwargs pass through to :class:`DumbMemory`."""
        if path is None and scope is not None:
            from signal_engine.scope import resolve_memory_path
            path = resolve_memory_path(scope=scope)
        store = KeywordStore(os.path.expanduser(path) if path else None)
        embedder = None
        if (embed or "").lower() == "local":
            try:
                from signal_engine.embedder import LocalEmbedder
            except Exception as exc:                       # pragma: no cover - env-specific
                raise RuntimeError(
                    "Local embeddings need the 'local' extra: uv sync --extra store --extra local "
                    "(or pass embed='none' to route on manifests only)."
                ) from exc
            embedder = LocalEmbedder(embed_model)
        self.engine = DumbMemory(store=store, embedder=embedder,
                                 wiki_link=bool(wiki_link and embedder), **engine_kwargs)

    # -- write ------------------------------------------------------------------------------

    def add(self, content: str, *, title: Optional[str] = None, keywords=None,
            importance: float = 0.5, pinned: bool = False, kind: str = "fact",
            supersedes=None, volatility: Optional[str] = None, **behavioural) -> dict:
        """Store one memory. A ``fact`` keeps the durable statement (include dates); a
        ``behavioural`` note captures how-you-work meta (sentiment/friction/tool_choice/interaction).
        Pass ``supersedes=[old_id]`` when a value changed — the old card is kept ``[older]``, never
        lost. ``volatility`` (``"invariant"``/``"decision"``/``"external"``, STO-2846) tags what the
        fact depends on, so a later recall flags decision/external facts 'verify before acting'.
        Returns the store result (stored count, and any ``possible_supersedes``)."""
        sig = {"kind": kind, "title": title or content, "content": content,
               "keywords": list(keywords or []), "importance": float(importance),
               "pinned": bool(pinned), "volatility": volatility}
        if supersedes:
            sig["supersedes"] = list(supersedes)
        sig.update(behavioural)                            # sentiment/friction/tool_choice/interaction
        return self.engine.store_signals([sig])

    def add_many(self, signals) -> dict:
        """Store a batch of already-structured signals (the agent-extraction path). Each is a dict
        like ``{"kind","title","content","keywords","importance"}`` (+ behavioural fields, and an
        optional ``"supersedes":[ids]``)."""
        return self.engine.store_signals(list(signals))

    # -- read -------------------------------------------------------------------------------

    def recall(self, query: str, limit: int = 5) -> dict:
        """One-call recall: the most relevant cards WITH their content, plus summaries you can drill
        for older history. The everyday read path — see ``DumbMemory.recall``."""
        return self.engine.recall(query, limit=limit)

    def search(self, query: str, k: Optional[int] = 8) -> List[dict]:
        """Recall the most relevant memories for ``query`` — hybrid (meaning + keyword) ranked, with
        lifecycle drill-back into archived cards. Returns ``k`` cards as dicts with their content.
        Ranking does not require an embedder (without one it's keyword + entity only), so a query is
        always ranked rather than handed back in insertion order."""
        sigs = self.engine.recall_context(query) if query else self.engine._hot()
        cards = [self._card(s) for s in sigs]
        return cards[:k] if k else cards

    def get(self, ids) -> List[str]:
        """Fetch the full formatted content for specific card ids (and count them as used)."""
        return self.engine.get(list(ids))

    def list(self, kind: Optional[str] = None, pinned: Optional[bool] = None,
             include_archived: bool = False, include_older: bool = True,
             contains: Optional[str] = None, limit: Optional[int] = None,
             include_retired: bool = False, unlabelled: bool = False) -> dict:
        """Browse/inspect what's stored, with optional filters. Read-only."""
        return self.engine.list_cards(kind=kind, pinned=pinned, include_archived=include_archived,
                                      include_older=include_older, contains=contains, limit=limit,
                                      include_retired=include_retired, unlabelled=unlabelled)

    def relabel(self, ids, volatility: Optional[str] = None, basis: Optional[str] = None) -> dict:
        """Set ``volatility`` (invariant|decision|external) and/or ``basis`` (seen|told|assumed) on
        cards already stored, without changing their text (STO-2912). Returns the counts."""
        return self.engine.relabel(list(ids), volatility=volatility, basis=basis)

    def import_notes(self, path: str, sections=None, apply: bool = True) -> dict:
        """One-time import of a markdown notes file (the project's CLAUDE.md, a learned-rules file)
        into cards, so a fresh memory is useful from the first session. List items only, from
        rule-like headings by default or the ``sections`` you name; unpinned, source recorded, dedup
        and the credential guard apply. ``apply=False`` previews. See ``DumbMemory.import_notes``."""
        return self.engine.import_notes(path, sections=sections, apply=apply)

    def guidance(self) -> dict:
        """'What can I adjust, and how?' — the live editable policy, what's locked, the autonomy mode,
        and a plain-English map of what a user can say to change each thing. Read-only."""
        return self.engine.guidance()

    def stats(self) -> dict:
        """Health/growth report: total cards, hot, summaries, archived, pins, approx disk, health,
        and a 'cold' count (old + never-retrieved cards worth reviewing)."""
        return self.engine.stats()

    def knowledge_gaps(self, slots: Optional[dict] = None) -> dict:
        """What memory does NOT know that it probably should (STO-2850). Reports which expected slots
        for this scope (audience/deliverable/deadline/success/constraints by default) are still empty,
        so you can ASK rather than miss scope. Pass ``slots={name:[terms]}`` to set a role schema."""
        return self.engine.knowledge_gaps(slots=slots)

    def cold_cards(self, min_idle: Optional[int] = None, max_uses: int = 0,
                   older_than_days: Optional[float] = None) -> dict:
        """Flag cleanup candidates — cards stored a while ago (by cards-added-since, or ``older_than_days``
        of real time) that have (almost) never been retrieved. Excludes pins/summaries/context. Review
        and ``delete()`` any junk; nothing is auto-deleted."""
        return self.engine.cold_cards(min_idle=min_idle, max_uses=max_uses,
                                      older_than_days=older_than_days)

    def procedure_outcome(self, ids, outcome: str) -> dict:
        """Report how a remembered procedure went when reused — ``outcome`` is ``"worked"`` or
        ``"failed"`` (STO-2823). Wins raise the card's trust, failures lower it, and repeated
        consecutive failures retire it from recall (kept + still fetchable; a win reinstates it)."""
        return self.engine.procedure_outcome(ids, outcome)

    def configure(self, settings: Optional[dict] = None, **kwargs) -> dict:
        """Set memory policy at runtime — decay/pins/retrieval weights and an ``importance_policy``
        ({boost_kinds, boost_keywords, boost, persona}) for what this role weights up. Persisted per
        scope; recomputed live. Accepts a dict and/or keyword args."""
        merged = dict(settings or {})
        merged.update(kwargs)
        return self.engine.configure(merged)

    # -- forget -----------------------------------------------------------------------------

    def delete(self, ids) -> dict:
        """Permanently forget cards by id (privacy / GDPR). The one destructive op; everything else
        decays and stays drillable. Returns ``{"deleted": n}``."""
        return self.engine.delete(list(ids))

    def unpin(self, ids) -> dict:
        """Take cards off the pin set by id, WITHOUT deleting or superseding them (STO-2848) — the
        non-destructive fix for a wrongly-pinned card. The card stays current + retrievable and decays
        normally; re-store with ``pinned=True`` to pin it again. Returns ``{"unpinned": [ids]}``."""
        return self.engine.unpin(list(ids))

    # -- swarm interop ----------------------------------------------------------------------

    def set_context_card(self, agent: str, identity: str = None, focus: str = None,
                         priorities=None, goals=None, capabilities=None) -> dict:
        """Publish/update this agent's live profile in the (shared) scope — who I am / focus /
        priorities / goals / capabilities — so peers can coordinate. Merged on update; one per agent."""
        return self.engine.set_context_card(agent, identity=identity, focus=focus,
                                            priorities=priorities, goals=goals,
                                            capabilities=capabilities)

    def context_cards(self, agent: Optional[str] = None) -> dict:
        """Read the swarm's context cards (every agent's current profile, or one agent's)."""
        return self.engine.context_cards(agent=agent)

    def id_card(self, agent: Optional[str] = None, introduce: bool = False) -> dict:
        """The composed agent ID card / handshake artifact — the shared scope (role, autonomy, derived
        experience: health + earned procedure track record) plus per-agent profiles. ``introduce=True``
        returns a portable self-introduction (``text`` + ``card``) needing nothing on the far side.
        Read-only."""
        return self.engine.id_card(agent=agent, introduce=introduce)

    # -- helpers ----------------------------------------------------------------------------

    def _card(self, s: Signal) -> dict:
        return {"id": s.signal_id, "title": s.title, "content": s.content or s.title,
                "kind": s.kind, "keywords": list(s.keywords or []),
                "pinned": self.engine.is_pinned(s), "older": bool(s.superseded_by)}
