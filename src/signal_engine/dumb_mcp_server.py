"""Dumb-server MCP (approach A) — Signal Engine memory where the HOST AGENT does the reasoning.

Dependency-free stdio JSON-RPC (same transport as mcp_server.py). Exposes the DumbMemory tools and
ships OPERATING_MANUAL as the server's ``instructions`` so the connecting agent (your Claude Code)
knows how to extract / route / summarise. ZERO LLM calls here, no embedder — the agent's own model
does the thinking, for free.

Run:  uv run --extra local signal-memory-mcp
Register:  claude mcp add signal-memory --env SIGNAL_DUMB_EMBED=local -- uv run --extra local signal-memory-mcp

Keep the env var + extra: without them there is no on-device embedder, and search degrades to
keyword-only ranking (reported as ``ranked: "keyword"`` in every search response).
"""

from __future__ import annotations

import json
import sys
from typing import Optional

from signal_engine.dumb_memory import DumbMemory, OPERATING_MANUAL
from signal_engine.operating_manual import _PROTOCOL  # noqa: F401

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "signal-memory", "version": "0.1.0"}

TOOLS = [
    {"name": "before_you_start",
     "description": "CALL THIS WHEN YOU PICK UP A TASK — before you plan, choose an approach, or "
                    "write anything. Say what you are about to do in plain words (doing=\"fixing the "
                    "deploy pipeline\" / \"answering a question about pricing\") and it returns BOTH "
                    "halves you need: 'standing_rules' (the pinned staples that always apply — tools, "
                    "conventions, constraints) and 'about_this' (what memory holds about THIS task). "
                    "You do not have to invent a search query — describe the moment. If 'matched' is "
                    "false nothing stored is about this task: say so rather than padding your answer "
                    "with the closest cards. If the reply carries 'action_required', setup for this scope is "
                    "still outstanding — do those items (one call each) before your next answer. Args: "
                    "doing, limit (default 5). Use recall(query) instead for a deliberate lookup of one "
                    "specific thing.",
     "inputSchema": {"type": "object", "properties": {"doing": {"type": "string"},
                     "limit": {"type": "integer"}}, "required": ["doing"]}},
    {"name": "store_signals",
     # Hosts cut a tool description at 2,048 chars (Claude Code logs "description truncated from 2662
     # to 2048"); this one lost its last two rules for a while. tests/test_manual_budget.py now
     # fails any description over DESCRIPTION_LIMIT.
     "description": "Store what is worth remembering. signals = array of {kind:'fact'|'behavioural', title, "
                    "content, keywords[], importance 0..1, pinned?, volatility, basis, supersedes?}. EVERY "
                    "fact needs both labels: volatility = what it depends on ('invariant' | 'decision' | "
                    "'external'; decision/external come back flagged 'verify before acting') and basis = how "
                    "you know it ('seen' observed | 'told' by the user or a document | 'assumed' concluded, "
                    "not checked; flagged on recall). An unlabelled save is kept but named under "
                    "'unlabelled'; fix it with relabel(). 'supersedes':[old_ids] marks cards this one "
                    "CORRECTS (kept as [older]). Capture commands, procedures, the WHY behind decisions and "
                    "what NOT to retry; skip abandoned exploration. Restatements auto-dedupe. WHAT BELONGS "
                    "HERE: what an agent needs mid-task (commands, gotchas, decisions, preferences) and "
                    "anything cross-project; the project's instruction file (CLAUDE.md) keeps its story for "
                    "humans. Never both: import the file once with import_notes. NEVER store secrets (API "
                    "keys, tokens, passwords, private keys, credentialed URLs): store that a credential "
                    "exists and where (\"RAILWAY_TOKEN in .env\"), never its value. The server redacts "
                    "obvious ones and replies 'redacted' + 'warning': tell the user. If you decline to "
                    "store a credential, SAY SO: silence looks identical to storing it. Returns 'stored', "
                    "and may add 'superseded'; 'please_summarise' (memory full: get(source_ids), summarise, "
                    "store_summary); 'related' (titles already stored on this subject; if yours replaces "
                    "one, store it again with supersedes:[id], which links them without a copy); "
                    "'imbalance' (saving without reading; names unread cards); 'pins_saturating' (unpin "
                    "non-staples); 'unlabelled'.",
     "inputSchema": {"type": "object", "properties": {"signals": {"type": "array"}},
                     "required": ["signals"]}},
    {"name": "recall",
     "description": "CHECK WHAT YOU ALREADY KNOW, before you answer or act. Searches your long-term "
                    "memory and returns the most relevant cards WITH their content, in ONE call. Use "
                    "it whenever you're about to state what the user wants or prefers, act on a past "
                    "decision, pick a tool/branch/command for this project, or answer 'why is it like "
                    "this / what did we decide about X' — recall first instead of guessing from the "
                    "live chat. RECALL AT THESE MOMENTS, as a checklist: before choosing an "
                    "approach/tool/library/command; before asserting what the user wants or "
                    "prefers; before repeating or acting on a past decision; when starting work "
                    "in a scope you have worked in before; when asked why something is so. "
                    "Args: query (what you want to know), limit (how many cards, default "
                    "5). Returns cards (id/title/content/keywords), 'summaries' you can drill() for "
                    "older archived history, 'more' (whether relevant cards were cut off), and "
                    "'ranked' ('hybrid'|'keyword'|'none'), and 'matched' — FALSE means nothing stored is "
                    "actually about your query and the cards are only the CLOSEST by ranking (titles only), so "
                    "say you don't have it rather than answering from them. Archived cards (folded into a "
                    "summary) are searched too and come back marked 'archived'. A query made only of words "
                    "like standing/rules/decisions/preferences returns the pinned rules. A card clears the bar by being "
                    "ABOUT your query (its title/keywords, or its meaning), not by mentioning a word of "
                    "it, so ask in the subject's own words. Cards that cleared the bar all "
                    "come back (no need to re-ask with a bigger limit); long ones are truncated with "
                    "full text via get([id]). HEED any "
                    "'verify before acting', 'assumed' or 'unlabelled' note on a card: that fact may have gone "
                    "stale, or was never checked. If nothing "
                    "relevant comes back, say you don't have it — don't force an answer. (For routing "
                    "over a large index without content, see candidates.)",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"},
                     "limit": {"type": "integer"}}, "required": ["query"]}},
    {"name": "candidates",
     "description": "The INDEX view, for routing over a large store yourself: hot + summary manifests, titles and "
                    "keywords only, NO content. Pick ids, then get(ids); drill() a summary if the answer is not in "
                    "hot. Prefer recall(query) for the everyday case — it answers in one call instead of two. "
                    "Reports 'ranked'.",
     "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}},
                     "required": ["query"]}},
    {"name": "drill",
     "description": "Given a summary_id, return the manifests of the cards that summary folded, so you can search "
                    "deeper into archived history.",
     "inputSchema": {"type": "object", "properties": {"summary_id": {"type": "string"}},
                     "required": ["summary_id"]}},
    {"name": "get",
     "description": "Fetch the full content of chosen card ids.",
     "inputSchema": {"type": "object", "properties": {"ids": {"type": "array"}}, "required": ["ids"]}},
    {"name": "store_summary",
     "description": "After a 'please_summarise', store your summary: text + source_ids from that instruction. The "
                    "sources are archived (kept, drillable via this summary). Pass summary_id to REFRESH a stale "
                    "summary in place.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"},
                     "source_ids": {"type": "array"}, "summary_id": {"type": "string"}},
                     "required": ["text", "source_ids"]}},
    {"name": "stats",
     "description": "Memory health/growth: total cards, hot, summaries, archived, pins, approx disk, "
                    "a health hint, and 'summary' — a ready-made one-line status you can relay to the user "
                    "in chat (they cannot see tool calls, so this is how they know memory is "
                    "working). Also reports 'session' read/write counts for this process, "
                    "and 'search' (hybrid vs keyword).",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "guidance",
     "description": "CALL THIS FIRST, ON CONNECT — and whenever the human asks what your memory can do. It is the "
                    "authoritative, never-truncated copy of your operating rules (the server's instructions text is "
                    "cut short by some hosts; a tool result is not). IF THE REPLY CONTAINS 'action_required', DO "
                    "THOSE THINGS BEFORE YOUR NEXT ANSWER — they are setup steps this scope is missing, such as no "
                    "importance_policy being set. Also returns: 'you_should' (what to do and when), "
                    "'yours_in_any_mode' (what you control even in guided mode — policy, pins, storing, recalling), "
                    "'locked_in_guided' (the lifecycle dials, the ONLY things guided freezes), 'can_ask' (what the "
                    "USER can say to change each thing), plus the live policy, locks and autonomy mode. Pass "
                    "full=true to also get the long-form manual. Read-only.",
     "inputSchema": {"type": "object", "properties": {"full": {"type": "boolean"}}}},
    {"name": "import_notes",
     "description": "ONE-TIME IMPORT of a markdown notes file into memory — the project's instruction "
                    "file (CLAUDE.md or equivalent) and any learned-rules file your host keeps (on "
                    "Claude Code: ~/.claude/insights/learned-rules.md) — so a fresh scope is useful from "
                    "the first session instead of empty. Imports markdown LIST ITEMS only (narrative "
                    "stays in the file), from headings that look like rules by default (gotchas, rules, "
                    "conventions, decisions, commands, build/test, setup, learned) or the ones you name. "
                    "Each item becomes one unpinned card with its source recorded; the credential guard "
                    "and dedup apply; re-runs skip what is already present. Args: path (the file; ~ is "
                    "expanded), sections (optional list of heading words to include), apply (default "
                    "true; pass false to PREVIEW what would be imported without writing). Returns "
                    "imported, candidates, already_present, sections_used, sections_skipped, preview.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"}, "sections": {"type": "array"}, "apply": {"type": "boolean"}},
         "required": ["path"]}},
    {"name": "list",
     "description": "Browse/inspect what's stored (the 'get_memories' equivalent). Filters: kind "
                    "('fact'|'behavioural'), pinned, contains (substring over title/content/keywords), "
                    "include_archived, include_older, include_retired (procedures retired for repeated failure), "
                    "unlabelled (only fact cards missing a volatility or basis label — then relabel them), "
                    "limit. Returns card manifests (id/title/keywords/kind/pinned/archived/older/uses, plus "
                    "volatility/basis/unlabelled) + match counts. Use it for 'what do you remember about X' or "
                    "to find a card's id before forgetting. Read-only.",
     "inputSchema": {"type": "object", "properties": {
         "kind": {"type": "string"}, "pinned": {"type": "boolean"}, "contains": {"type": "string"},
         "include_archived": {"type": "boolean"}, "include_older": {"type": "boolean"},
         "include_retired": {"type": "boolean"}, "unlabelled": {"type": "boolean"},
         "limit": {"type": "integer"}}}},
    {"name": "relabel",
     "description": "Set the labels on cards ALREADY stored, without changing their text. Args: ids (card ids), "
                    "volatility ('invariant'|'decision'|'external' — what the fact depends on), basis ('seen'|"
                    "'told'|'assumed' — how you know it); give one or both. Use it when a save reply lists "
                    "'unlabelled' cards, when list(unlabelled=true) or stats().unlabelled shows older cards with "
                    "no label, or when you learn a fact was only assumed. Returns relabelled (count), skipped "
                    "(ids that take no label: behavioural notes, summaries), unknown (ids not found), rejected "
                    "(a bad value, with the allowed ones) and unlabelled_remaining for this scope.",
     "inputSchema": {"type": "object", "properties": {
         "ids": {"type": "array", "items": {"type": "string"}},
         "volatility": {"type": "string"}, "basis": {"type": "string"}}, "required": ["ids"]}},
    {"name": "delete",
     "description": "Permanently forget cards by id (privacy / 'forget me' / GDPR) — the ONLY destructive op; "
                    "everything else decays and stays drillable. Also strips them from any summary's source list. "
                    "Confirm the ids before calling.",
     "inputSchema": {"type": "object", "properties": {"ids": {"type": "array"}},
                     "required": ["ids"]}},
    {"name": "knowledge_gaps",
     "description": "What does memory NOT know that it probably SHOULD? Reports which expected, question-shaped "
                    "slots for this scope (audience / deliverable / deadline / success criteria / constraints by "
                    "default) are still EMPTY — so you ASK the user instead of guessing or missing scope. Retrieval "
                    "can never find a fact that was never stored; this attacks that gap. CALL IT when starting or "
                    "onboarding in a scope, and when scoping a task. Pass slots={name:[trigger terms]} to set a "
                    "role-specific schema (persisted). Read-only unless slots is given.",
     "inputSchema": {"type": "object", "properties": {"slots": {"type": "object"}}}},
    {"name": "unpin",
     "description": "Take cards OFF the pin set by id, WITHOUT deleting or superseding them — the fix for a "
                    "wrongly-pinned card (a bad early policy, or a pin that no longer belongs). ids = the card ids "
                    "to unpin. Never supersede a card just to un-pin it; that corrupts the [older] history with a "
                    "fake change. The card stays current and retrievable, decays and archives normally from now on, "
                    "and will NOT be re-pinned even if its importance would otherwise derive a pin. Reversible: "
                    "re-store the fact with pinned:true to pin it again. If a store reply or stats() says "
                    "'pins_saturating', this is the tool to fix it with.",
     "inputSchema": {"type": "object", "properties": {"ids": {"type": "array"}},
                     "required": ["ids"]}},
    {"name": "configure",
     "description": "Set memory POLICY at runtime — call it on connect to make memory fit YOUR role, so you own "
                    "memory, not just its cards. importance_policy {persona, boost_kinds, boost_keywords, boost} = "
                    "what your role weights up (ranked + kept first; it never pins — pins are explicit or your own "
                    "importance >= pin_threshold) — settable in EVERY mode, guided included. "
                    "LIFECYCLE DIALS (the only things guided locks): decay_half_life (how fast old unused memory "
                    "fades), hot_cap, cards_per_summary, pin_threshold, behavioural_pin_boost, usage_boost, "
                    "dedup_threshold (raise it for coding facts where filenames/versions look alike but differ), "
                    "lex_weight, entity_weight, recency_weight, narrow_k, broad_k, wiki_link. autonomy='guided' "
                    "(default, predictable) | 'auto' (self-manage, retune freely). user_requested=true means the "
                    "HUMAN explicitly asked for THIS change: it applies even a guided-locked dial and lifts a soft "
                    "lock, without switching mode — set it ONLY on a direct user instruction, never for your own "
                    "tuning. lock=[keys] (or true) freezes dials, unlock=[keys] lifts a soft lock; keys in the "
                    "SIGNAL_MEMORY_LOCK env are hard-locked and no agent can ever move them. Persisted per scope (a "
                    "swarm shares it), recomputed live. Returns applied, rejected_locked, user_overridden, clamped, "
                    "locked, autonomy.",
     "inputSchema": {"type": "object", "properties": {
         "autonomy": {"type": "string", "enum": ["guided", "auto"]},
         "user_requested": {"type": "boolean"},
         "importance_policy": {"type": "object"}, "decay_half_life": {"type": "number"},
         "hot_cap": {"type": "integer"}, "cards_per_summary": {"type": "integer"},
         "dedup_threshold": {"type": "number"},
         "pin_threshold": {"type": "number"}, "behavioural_pin_boost": {"type": "number"},
         "usage_boost": {"type": "number"}, "lex_weight": {"type": "number"},
         "entity_weight": {"type": "number"}, "recency_weight": {"type": "number"},
         "narrow_k": {"type": "integer"}, "broad_k": {"type": "integer"},
         "wiki_link": {"type": "boolean"}, "lock": {}, "unlock": {"type": "array"}}}},
    {"name": "use_scope",
     "description": "Situational (multi-project / swarm) — Switch which memory (scope) this session uses, on the fly — for an agent or orchestrator "
                    "working across several things. scope=<name> gives the shared brain for that thing (a swarm "
                    "shares one), path=<file> an explicit file, neither the per-folder default. All later calls use "
                    "it until switched again. Returns the active path + stats.",
     "inputSchema": {"type": "object", "properties": {
         "scope": {"type": "string"}, "path": {"type": "string"}}}},
    {"name": "set_context_card",
     "description": "Situational (swarm / multi-agent) — Publish/update YOUR live profile in the shared scope so peers can coordinate: agent (your "
                    "name/id), identity (who you are), focus (what you're on right now), priorities, goals "
                    "(outcomes you're after), capabilities (your tools/skills). MERGED on update — only the fields "
                    "you pass change, the rest survive — so a focus update never wipes your capabilities. One card "
                    "per agent; refresh as your focus changes. See id_card for the composed handshake view.",
     "inputSchema": {"type": "object", "properties": {
         "agent": {"type": "string"}, "identity": {"type": "string"}, "focus": {"type": "string"},
         "priorities": {}, "goals": {}, "capabilities": {}}, "required": ["agent"]}},
    {"name": "context_cards",
     "description": "Situational (swarm / multi-agent) — Read the swarm's context cards — every agent's current profile (who is on this task and what "
                    "each is doing), or pass agent=<name> for one. This is how agents understand each other and "
                    "coordinate.",
     "inputSchema": {"type": "object", "properties": {"agent": {"type": "string"}}}},
    {"name": "id_card",
     "description": "Situational (swarm / handoff) — The composed agent ID CARD, a handshake artifact: the shared scope (role, autonomy, locks, and "
                    "the EARNED track record derived from procedure_outcome) plus per-agent profiles. Omit agent "
                    "for the whole roster, to size up who is on the shared brain before handing off work. "
                    "introduce=true returns a portable self-introduction for someone with no memory access. Read- "
                    "only.",
     "inputSchema": {"type": "object", "properties": {
         "agent": {"type": "string"}, "introduce": {"type": "boolean"}}}},
    {"name": "cold_cards",
     "description": "Situational (housekeeping) — Flag CLEANUP candidates — cards stored a while ago that have (almost) never been retrieved. "
                    "Excludes pins, summaries and context cards. Review them and delete(ids) any junk; memory never "
                    "deletes on its own, so this is a suggestion, not an action. Args: min_idle (how old, counted "
                    "by cards-added-since; defaults to the hot capacity), older_than_days (age by real wall-clock "
                    "time instead), max_uses (default 0). stats() also reports a 'cold' count.",
     "inputSchema": {"type": "object", "properties": {
         "min_idle": {"type": "integer"}, "older_than_days": {"type": "number"},
         "max_uses": {"type": "integer"}}}},
    {"name": "procedure_outcome",
     "description": "Report how a remembered PROCEDURE actually went when you reused it, so memory stops "
                    "recommending steps that no longer work. ids = the cards you acted on, outcome = "
                    "'worked'|'failed'. A win raises that card's trust and clears its failure streak; a failure "
                    "lowers trust. Repeated CONSECUTIVE failures RETIRE the card — out of recall, but KEPT and "
                    "still fetchable by id — and a later win reinstates it. Pinned cards are never retired. Call "
                    "this after you act on a remembered command or fix and see whether it worked.",
     "inputSchema": {"type": "object", "properties": {
         "ids": {"type": "array"}, "outcome": {"type": "string", "enum": ["worked", "failed"]}},
         "required": ["ids", "outcome"]}},
]

# One DumbMemory per scope (path), so an agent can switch scopes on the fly within a session — e.g.
# an orchestrator routing sub-tasks to different memories — without a restart. The on-device embedder
# is loaded ONCE and shared across scopes (it's stateless — just the model), so switching is cheap.
# FIRST-CALL ORIENTATION (host-agnostic delivery). The ``instructions`` channel is unreliable: some
# hosts truncate it (measured: cut at 2,048 bytes, delivering 12.5% of the manual), some snapshot it
# at session start so it goes stale, and some do not surface it at all. Tool RESULTS have none of
# those problems. So the first tool call of a process carries the operating protocol back with it,
# once. Costs nothing until memory is actually used, and cannot be truncated away.
_oriented = False

_mems: dict = {}
_active_path: Optional[str] = None
_embedder = None
_embedder_built = False


def _shared_embedder():
    global _embedder, _embedder_built
    if not _embedder_built:
        import os
        _embedder_built = True
        if os.getenv("SIGNAL_DUMB_EMBED", "").lower() == "local":  # free on-device narrowing
            from signal_engine.embedder import LocalEmbedder
            _embedder = LocalEmbedder(os.getenv("SIGNAL_EMBED_MODEL", "all-MiniLM-L6-v2"))
    return _embedder


def _build_memory(path: str) -> DumbMemory:
    import os
    from signal_engine.keyword_store import KeywordStore
    # Product default: the FULL engine is ON — wiki-loop + pins + behavioural boost + decay/summaries +
    # drill-back. The wiki-loop needs the local embedder; SIGNAL_DUMB_WIKI=off disables just it.
    wiki = os.getenv("SIGNAL_DUMB_WIKI", "on").lower() != "off"
    return DumbMemory(store=KeywordStore(path), embedder=_shared_embedder(), wiki_link=wiki)


def switch_scope(scope: Optional[str] = None, path: Optional[str] = None) -> DumbMemory:
    """Point the session at a different scope (a named scope, an explicit path, or — with neither —
    the per-folder default). Builds the store on first use, caches it, and makes it active."""
    global _active_path
    from signal_engine.scope import resolve_memory_path
    _active_path = resolve_memory_path(explicit=path, scope=scope)
    if _active_path not in _mems:
        _mems[_active_path] = _build_memory(_active_path)
    return _mems[_active_path]


def _memory() -> DumbMemory:
    if _active_path is None:
        return switch_scope()                              # resolve the default scope on first use
    return _mems[_active_path]


def handle_message(msg: dict, memory=_memory) -> Optional[dict]:
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        # Ship the PROTOCOL, not the whole manual. Hosts that truncate cut at ~2KB anyway, and hosts
        # that don't were paying ~4,300 tokens every session for detail the agent rarely needs in
        # context. The detail stays one call away (guidance(full=true)), the protocol also rides on
        # the first tool result, and the tool descriptions carry the rest. SIGNAL_MEMORY_MANUAL=full
        # restores the old behaviour for a host that genuinely wants everything up front.
        import os as _o
        full = (_o.getenv("SIGNAL_MEMORY_MANUAL", "").lower() == "full")
        result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                  "serverInfo": SERVER_INFO,
                  "instructions": OPERATING_MANUAL if full else _PROTOCOL}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        name, args = params.get("name"), (params.get("arguments") or {})
        try:
            m = memory()
            if name == "store_signals":
                out = m.store_signals(args.get("signals") or [])
            elif name == "before_you_start":
                out = m.before_you_start(args.get("doing", ""), limit=int(args.get("limit", 5)))
            elif name == "recall":
                out = m.recall(args.get("query", ""), limit=int(args.get("limit", 5)))
            elif name == "candidates":
                out = m.candidates(args.get("query", ""))
            elif name == "drill":
                out = m.drill(args.get("summary_id", ""))
            elif name == "get":
                out = {"cards": m.get(args.get("ids") or [])}
            elif name == "store_summary":
                out = m.store_summary(args.get("text", ""), args.get("source_ids") or [],
                                      summary_id=args.get("summary_id"))
            elif name == "stats":
                out = m.stats()
            elif name == "guidance":
                out = m.guidance()
                if args.get("full"):
                    from signal_engine.operating_manual import _DETAIL
                    out["manual"] = _DETAIL
            elif name == "import_notes":
                out = m.import_notes(args.get("path", ""), sections=args.get("sections"),
                                     apply=bool(args.get("apply", True)))
            elif name == "list":
                out = m.list_cards(kind=args.get("kind"), pinned=args.get("pinned"),
                                   include_archived=bool(args.get("include_archived", False)),
                                   include_older=bool(args.get("include_older", True)),
                                   include_retired=bool(args.get("include_retired", False)),
                                   contains=args.get("contains"), limit=args.get("limit"),
                                   unlabelled=bool(args.get("unlabelled", False)))
            elif name == "relabel":
                out = m.relabel(args.get("ids") or [], volatility=args.get("volatility"),
                                basis=args.get("basis"))
            elif name == "delete":
                out = m.delete(args.get("ids") or [])
            elif name == "unpin":
                out = m.unpin(args.get("ids") or [])
            elif name == "knowledge_gaps":
                out = m.knowledge_gaps(slots=args.get("slots"))
            elif name == "configure":
                out = m.configure(args)
            elif name == "use_scope":
                switch_scope(scope=args.get("scope"), path=args.get("path"))
                m = memory()                               # re-fetch: now the switched-to scope
                out = {"switched": True, "path": getattr(m.store, "path", None), "stats": m.stats()}
            elif name == "set_context_card":
                out = m.set_context_card(args.get("agent", ""), identity=args.get("identity"),
                                         focus=args.get("focus"), priorities=args.get("priorities"),
                                         goals=args.get("goals"), capabilities=args.get("capabilities"))
            elif name == "context_cards":
                out = m.context_cards(agent=args.get("agent"))
            elif name == "id_card":
                out = m.id_card(agent=args.get("agent"), introduce=bool(args.get("introduce", False)))
            elif name == "cold_cards":
                out = m.cold_cards(min_idle=args.get("min_idle"),
                                   max_uses=int(args.get("max_uses", 0)),
                                   older_than_days=args.get("older_than_days"))
            elif name == "procedure_outcome":
                out = m.procedure_outcome(args.get("ids") or [], args.get("outcome", ""))
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool: {name}"}}
            global _oriented
            if not _oriented and isinstance(out, dict):
                _oriented = True
                # Field report (2026-09-26): with the COMPLETE 19KB manual attached here, "do this
                # first" was three lines inside a 30KB reply and got skipped — by an agent running the
                # infer-first code. So orientation now carries the 2KB protocol plus ONLY what is
                # outstanding; the manual is one call away (guidance(full=true)) and the tool
                # descriptions carry the detail. Delivery on a truncating host is unchanged: the
                # protocol is the survival copy, and this is a tool result, never cut.
                outstanding = m.action_required() if hasattr(m, "action_required") else []
                # Field report (2026-09-29): "push memory into session start". The standing rules
                # ride on the first reply as TITLES — an agent that never asks still learns which
                # rules exist — and their text stays one call away, because a long first reply is
                # how setup got skipped. Not on before_you_start: that reply already has them whole.
                rules = (m.standing_rule_titles()
                         if name != "before_you_start" and hasattr(m, "standing_rule_titles")
                         else {"rules": [], "more": 0})
                out["orientation"] = {
                    "note": ("First call this session. Your operating protocol is below (your host "
                             "may have truncated or skipped the server instructions; this is a tool "
                             "result, so it is complete). "
                             + ("SETUP IS OUTSTANDING — action_required lists it; do those before "
                                "your next answer. " if outstanding else "")
                             + ("standing_rules names the pinned rules that ALWAYS apply in this "
                                "scope — titles only; read them with before_you_start(<your task>) "
                                "before you plan. " if rules["rules"] else "")
                             + "The full manual is available with guidance(full=true)."),
                    **({"action_required": outstanding} if outstanding else {}),
                    **({"standing_rules": rules["rules"]} if rules["rules"] else {}),
                    **({"standing_rules_more": rules["more"]} if rules["more"] else {}),
                    "tell_the_user": ("They cannot see tool calls, so say in ONE short line that "
                                      "memory is on and what it holds — stats()['summary'] is "
                                      "written for this. Then keep them posted the same way when "
                                      "you recall or save something."),
                    "protocol": _PROTOCOL}
            result = {"content": [{"type": "text", "text": json.dumps(out)}]}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}}
    elif method is not None and mid is None:
        return None
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def serve(stdin=None, stdout=None, memory=_memory):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_message(msg, memory)
        if resp is not None:
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()


def main():
    serve()


if __name__ == "__main__":
    main()
