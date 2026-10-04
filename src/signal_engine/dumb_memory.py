"""Dumb-server memory (approach A) — storage + hierarchy orchestration with ZERO LLM calls.

The engine and every feature are identical to the smart path (index cards + full metadata,
hierarchy of summaries-as-index, dedup, salience, decay, versioning). The ONLY difference: the
three LLM touchpoints — **extract, route, summarise** — are performed by the HOST AGENT (the user's
Claude), guided by ``OPERATING_MANUAL`` that the server ships. So the intelligence (schema, rules,
orchestration) lives here; the compute runs on the user's own model — no API calls, no embedder.

Tools the agent uses:
  store(signals)            agent-extracted signals -> dedup + salience + store; may return a
                            'please_summarise' instruction when the hot tier is full (decay).
  candidates(query)         return the hot cards' + summaries' MANIFESTS (id/title/keywords) for
                            the agent to route over. No LLM here.
  drill(summary_id)         return the source cards a summary folded (for the agent to search).
  store_summary(text, ids)  the agent summarised a decayed batch -> store it, archive the sources.
"""

from __future__ import annotations

import re
from typing import List, Optional

from signal_engine.keyword_store import KeywordStore
from signal_engine.salience import SalienceFilter
from signal_engine.signal import Signal, behavioural_signal, fact_signal
from signal_engine.operating_manual import OPERATING_MANUAL  # noqa: F401  (re-exported)

_SUMMARY_TITLE = "[summary]"
_CONTEXT_TITLE = "[context]"   # a per-agent swarm profile card (who I am / focus / priorities)
_SUMMARY_K = 12               # most-relevant summary manifests candidates() hands the agent (bounded)
_RECALL_SUMMARY_K = 3
_RECALL_MAX = 15              # auto-widen ceiling: never make an agent ask twice for cards that
                              # already cleared the relevance bar (it should not do arithmetic to
                              # reach its own memory) — but do not hand back an unbounded dump
_IMBALANCE_AFTER = 3          # consecutive stores with no recall before the result says so —
                              # and ONLY if there are cards the agent has not seen this session
_FULL_TEXT_TOP = 3            # recall(): the top cards are the likely ANSWER; they are never snippeted
_RULE_SNIPPET = 600           # before_you_start(): a rule is followed, not skimmed — full unless long
_PIN_SATURATION_MIN = 8       # warn about pins only once there are enough cards for it to matter
_PIN_SATURATION_RATIO = 0.5   # ...and at least this share of current cards is pinned
_IMPORT_DEFAULT_SECTIONS = ("gotcha", "rule", "convention", "decision", "command", "build", "test",
                            "setup", "how to", "how-to", "learned", "tooling", "git", "quality",
                            "process", "workflow", "known issue", "way of working", "ways of working")
_IMPORT_MAX_BYTES = 1_000_000  # import_notes(): refuse anything bigger than a big instruction file
_IMPORT_MIN_CHARS = 40         # ...and skip list items too short to be a rule
_IMPORT_PREVIEW = 8            # titles shown back so the agent can sanity-check before applying
_CLOSEST_K = 3                # when NOTHING is relevant, show only a few, clearly labelled closest
# The relevance BAR (see DumbMemory._relevance). Field report, 2026-09-29: "it returns about a
# quarter of the store for any question and always says it matched" — 7 to 15 cards of ~50, about
# half on topic. Cause, measured on two real stores: a card passed for MENTIONING one query word
# anywhere in its body ("run" appears somewhere in 8 of 22 cards). These set what counts instead.
_BAR_BODY_WEIGHT = 0.5        # a query word only in a card's BODY is a mention: half a topical match
_BAR_SUPPORT_SHARE = 2 / 3    # weak word evidence must be backed by meaning >= this share of
                              # relevance_floor (0.45 -> 0.30) whenever the embedder is live
_BAR_KEYWORD_ONLY = 0.5       # no embedder to back it up: body-only evidence must cover this much
_BAR_UBIQUITY_MIN_CARDS = 8   # a word on the title/keywords of MORE THAN HALF the cards (a
_BAR_UBIQUITY_SHARE = 0.5     # project's own name) tells cards apart from nothing. Half, not a
                              # third: in a store with two big subjects each one's name sits on
                              # ~50% of the cards and is exactly what a question is asking about
_RELATED_MAX = 3              # store_signals(): titles of existing cards on the same subject
_RELATED_BATCH_MAX = 5        # ...only for a hand-written batch, never a bulk import
_UNCONSULTED_NAMED = 5        # the imbalance note NAMES this many unread cards, closest first
_ORIENT_RULES_MAX = 12        # first reply of a session: titles of the standing rules, no bodies
_BAR_COMMON_SHARE = 0.25      # a word on the title/keywords of more than a quarter of the cards (a
                              # person's name, "decision") cannot carry a match on its own: field
                              # report 2026-10-02, "matched on 'Nick' and 'MCP'" (Nick 30%)
_FOLD_MIN_AGE_HOURS = 6       # please_summarise never folds a card younger than this. Field report
                              # 2026-10-02: the batch held a decision saved seconds earlier, and two
                              # cards written at 10:14 were archived at 11:49
_SUMMARY_KEYWORDS_SHOWN = 15  # a summary's manifest shows its 15 most shared keywords (they carried
                              # 54 to 80 each, ~3,000 chars on every recall); all are kept for search
_POLICY_STALE_DAYS = 7        # before_you_start asks whether the role is still right after this long
_SNIPPET = 240                # chars of content per card once a result set is big enough to skim;
                              # cheap recalls get called more often, and frequency is the whole game         # recall() is the LEAN one-call path: just enough summaries to drill back

# The generic "what a project scope usually needs answered" slot schema (STO-2850). Each slot -> the
# trigger terms that count a card as FILLING it. An agent can replace this per scope for its role
# (configure(knowledge_slots={...})). Kept simple on purpose: a slot is filled if a current fact card
# mentions any trigger term (substring over title/content/keywords). It's a first cut at surfacing the
# unknown-unknowns — "you have no card about the DELIVERABLE" — not a semantic classifier.
_DEFAULT_KNOWLEDGE_SLOTS = {
    "audience": ["audience", "who for", "who is it for", "stakeholder", "user", "customer", "reader"],
    "deliverable": ["deliverable", "deliver", "demo", "ship", "output", "artefact", "artifact",
                    "what we're building", "what we are building"],
    "deadline": ["deadline", "due", "by when", "timeline", "launch date", "ship date", "when is it"],
    "success_criteria": ["success", "criteria", "done when", "acceptance", "goal", "definition of done",
                         "what good looks like"],
    "constraints": ["constraint", "must not", "budget", "limit", "requirement", "non-negotiable"],
}

# The engine knobs an agent may set at runtime via configure() — name -> caster. These control the
# lifecycle (decay/capacity), what stays pinned, and how retrieval weighs its signals. importance_policy
# and wiki_link are handled separately (dict / bool). Anything not here is not agent-settable.
_CONFIGURABLE = {
    "decay_half_life": float, "hot_cap": int, "cards_per_summary": int, "pin_threshold": float,
    "behavioural_pin_boost": float, "usage_boost": float, "usage_boost_cap": float,
    "dedup_threshold": float, "lex_weight": float, "entity_weight": float, "recency_weight": float,
    "wiki_link_sim": float, "narrow_k": int, "broad_k": int, "relevance_floor": float, "relevance_coverage": float,
    "trust_weight": float, "retire_after_failures": int, "fold_min_age_hours": float,
}


def _words(t):
    import re
    return set(re.findall(r"[a-z0-9]+", (t or "").lower()))


# Sane ranges for the runtime dials — configure() casts freely, so a nonsensical value (a weaker
# agent setting cards_per_summary=0, a negative hot_cap, a >1 weight) must be clamped, not stored raw
# where it would crash a later op or silently break the lifecycle.
_MIN_ONE = {"hot_cap", "cards_per_summary", "narrow_k", "broad_k", "retire_after_failures"}
_UNIT = {"dedup_threshold", "lex_weight", "entity_weight", "recency_weight",
         "trust_weight", "behavioural_pin_boost", "usage_boost", "wiki_link_sim",
         "relevance_floor", "relevance_coverage"}
# pin_threshold is compared against a card's UNCAPPED importance (STO-2847), which can exceed 1.0 when
# boosts stack (base + behavioural + usage + policy + trust). So it is NOT a [0,1] unit dial — it may
# go above 1.0 to be more selective (and split cards a bad policy pinned). Ceiling comfortably above
# the max stacked importance (~2.2); a value that high just means "nothing derives a pin".
_PIN_THRESHOLD_MAX = 5.0


def _clamp_dial(key, val):
    if key in _MIN_ONE and val is not None:
        return max(1, int(val))
    if key in _UNIT and val is not None:
        return min(1.0, max(0.0, float(val)))
    if key == "pin_threshold" and val is not None:
        return min(_PIN_THRESHOLD_MAX, max(0.0, float(val)))
    if key == "decay_half_life" and val is not None:
        return max(1.0, float(val))
    if key == "fold_min_age_hours" and val is not None:
        return min(168.0, max(0.0, float(val)))           # 0 = fold at once, at most a week
    return val


def _distinctive(t):
    """Identifier-shaped tokens: filenames, paths, versions, ports, numbers — anything carrying a
    digit or a separator (. / - _). Two facts that differ on one of these are about DIFFERENT things
    (``.env.production`` vs ``.env.staging``, ``port 8080`` vs ``9090``, ``v1.2`` vs ``v1.3``), so
    lexical dedup must NOT merge them however much prose they share. Plain prose has none, so genuine
    restatements are unaffected."""
    import re
    toks = re.findall(r"[a-z0-9][a-z0-9._/\-]*[a-z0-9]|[0-9]+", (t or "").lower())
    return {w for w in toks if any(c.isdigit() for c in w) or any(c in "._/-" for c in w)}


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def _now_iso() -> str:
    """Wall-clock ISO timestamp for the PRODUCT path (episodic time + staleness). The frozen benchmark
    path never calls this — its signals carry deterministic created_at from the harness."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _age_days(ts: str, now=None) -> Optional[float]:
    if not ts:
        return None
    try:
        from datetime import datetime, timezone
        when = datetime.fromisoformat(ts)
        now = now or datetime.now(timezone.utc)
        return (now - when).total_seconds() / 86400.0
    except Exception:
        return None


# --- credential guard -------------------------------------------------------------------------
# Memory is a plain JSON file on disk, shared across a scope, and cards outlive the session that
# wrote them. An agent extracting "durable facts" from a coding session will otherwise persist an
# API key. Telling it not to is an instruction; this is enforcement.
#
# REDACT, don't refuse. Refusing loses the whole card including the legitimate fact around the
# secret ("deploy with TOKEN=sk-... via railway up" is worth keeping, minus the token). Redaction
# keeps the fact, drops the value, and reports what happened so the agent can tell the user.
#
# Patterns are chosen for PRECISION over recall: a vendor-prefixed key is unambiguous, so it is
# matched anywhere; a generic "token = ..." only counts with an assignment AND a long opaque value,
# so "the deploy token lives in 1Password" and "run tests with uv run pytest" are untouched. This
# guard is a floor against the obvious accident, NOT a DLP system — see tests/test_secret_guard.py.
_SECRET_PATTERNS = [
    # Vendor-prefixed keys are unambiguous, so they are matched anywhere and replaced whole.
    ("Anthropic key",     re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"), "[redacted: credential]"),
    ("OpenAI key",        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"), "[redacted: credential]"),
    ("GitHub token",      re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}"),
                          "[redacted: credential]"),
    ("Slack token",       re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "[redacted: credential]"),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[redacted: credential]"),
    ("Google API key",    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), "[redacted: credential]"),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
                          "[redacted: credential]"),
    ("JWT",               re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
                          "[redacted: credential]"),
    # Keep the scheme + host: "it is a postgres db at db.internal" is a fact worth keeping.
    ("credentials in a URL",
     re.compile(r"\b([a-z][a-z0-9+.-]*://)[^\s:/@]+:[^\s:/@]{4,}@"), r"\1[redacted]@"),
    # Keep the NAME of the setting, drop the value: "RAILWAY_TOKEN=[redacted]" still records which
    # credential the project uses and where it is set. Requires an assignment AND a long opaque
    # value, so prose like "the deploy token lives in 1Password" is untouched.
    ("assigned secret",
     # \w* on both sides so RAILWAY_TOKEN / MY_API_KEY / db_password match too — "_" is a word
     # char, so a bare \btoken\b never fires inside RAILWAY_TOKEN.
     re.compile(r"""(?i)(\w*(?:api[_-]?key|secret|password|passwd|token|access[_-]?key|private[_-]?key)\w*"""
                r"""\s*[:=]\s*)['"]?[A-Za-z0-9+/_-]{16,}['"]?"""), r"\1[redacted]"),
]


def _redact_secrets(text: str):
    """Return (clean_text, [what was found]). Empty list means nothing was touched."""
    if not text:
        return text, []
    found = []
    for name, pat, repl in _SECRET_PATTERNS:
        if pat.search(text):
            found.append(name)
            text = pat.sub(repl, text)
    return text, found


def _verify_note(s: Signal) -> Optional[str]:
    """A 'verify before acting' marker for a volatile fact (STO-2846). A decision or external fact can
    be freshly written, pinned, and already WRONG (the decider changed their mind, the third party moved)
    — decay never catches that, because it's an age problem and this is a dependency problem. So on
    recall we flag it, with the date it was written, so the agent hedges instead of asserting. Invariant
    and unclassified facts get nothing (unchanged behaviour)."""
    notes = []
    v = getattr(s, "volatility", None)
    when = f", written {s.created_at[:10]}" if getattr(s, "created_at", "") else ""
    if v in ("decision", "external"):
        why = ("a decision that holds only until it's changed" if v == "decision"
               else "external — a third party (a service, another person) gets a vote")
        notes.append(f"[verify before acting — {v}{when}: {why}. Confirm it's still true before "
                     f"you rely on it.]")
    elif v is None and _is_labelable(s):
        # STO-2912: "everything should have a label". A fact with none is not known to be safe, so
        # it reads as unverified until someone says what it depends on (relabel).
        notes.append(f"[unlabelled{when}: no volatility set, so it is not known whether this can "
                     f"change — treat as unverified; relabel([id], volatility=...) to fix.]")
    if getattr(s, "basis", None) == "assumed":
        # The card that held "the CI emails had stopped" was labelled; it held the agent's own
        # conclusion. A label for WHAT a fact depends on cannot flag HOW it was known.
        notes.append(f"[assumed{when}: a conclusion the agent drew, not something seen or told. "
                     f"Check it before you rely on it.]")
    return " ".join(notes) if notes else None


def _is_labelable(s: Signal) -> bool:
    """Only ordinary fact cards take labels: not behavioural notes, summaries or context cards."""
    return s.kind == "fact" and s.title not in (_SUMMARY_TITLE, _CONTEXT_TITLE)


def _missing_labels(s: Signal) -> list:
    """Which of the two labels a fact card lacks (empty for a fully labelled card)."""
    if not _is_labelable(s):
        return []
    return [name for name, val in (("volatility", getattr(s, "volatility", None)),
                                   ("basis", getattr(s, "basis", None))) if not val]


def _snip(text: str, limit: int = _SNIPPET) -> str:
    """Cut ``text`` to about ``limit`` chars for skimming — but never inside a backtick span, so a
    command ("run `uv run --extra dev pytest`") survives the cut intact. Field report: recall
    truncated card bodies "at roughly a sentence", so "what command do I run" needed a second call."""
    if len(text) <= limit:
        return text
    cut = limit
    if text.count("`", 0, cut) % 2 == 1:                  # cut landed inside an open `...`
        close = text.find("`", cut)
        if 0 <= close < cut + 120:
            cut = close + 1
    return text[:cut].rstrip() + "…"


_FIRST_RUN_APPROACH = {
    "do": "infer -> apply -> tell -> offer to adjust",
    "detail": ("Work out a sensible setup from what you can see — the project, the tools in use, what "
               "the user asked you to do — apply it, then tell the user in ONE line what you chose and "
               "that they can change any of it. That is the default. Ask a question only when you "
               "genuinely cannot infer the answer; first_run_questions are the menu of choices, not a "
               "script to read out. Field report: the questions were 'a lot for someone who just wants "
               "to start', and inferring-then-telling felt like the right default."),
}


def _manifest(s: Signal) -> dict:
    """The sharp, cheap card the agent routes over — no full content, no embedding."""
    man = {"id": s.signal_id, "title": s.title, "keywords": list(s.keywords or []),
           "kind": s.kind, "older": bool(s.superseded_by)}
    # Only carry validation evidence when it EXISTS (STO-2823) — an unvalidated card's manifest is
    # byte-for-byte what it was before the feature, so routing/token cost is unchanged by default.
    if getattr(s, "has_outcome_evidence", False):
        man.update(trust=round(s.trust, 3), wins=s.wins, fails=s.fails)
        if getattr(s, "retired", False):
            man["retired"] = True
    # Volatility marker (STO-2846) — only when the agent classified the fact, so an unclassified card's
    # manifest is unchanged. 'verify' flags a decision/external fact the agent should re-check on use.
    if getattr(s, "volatility", None):
        man["volatility"] = s.volatility
        if s.volatility in ("decision", "external"):
            man["verify"] = True
    if getattr(s, "basis", None):
        man["basis"] = s.basis
        if s.basis == "assumed":
            man["verify"] = True
    missing = _missing_labels(s)
    if missing:
        man["unlabelled"] = missing
        if "volatility" in missing:
            man["verify"] = True
    return man


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


_STOP = set(
    "the a an of to in on for and or is are was were be been being with at by from as it this that "
    "these those i you my your our their his her its how many much between till until do does did "
    "have has had will would can could should about into over under have get got need want "
    # Interrogatives carry no topic but appear in almost every query. Leaving them in meant
    # "what is the capital of France" shared a token with a card about sabotage runners, and
    # "change impact when a new asset arrives" scored a PERFECT lexical match on "when" alone
    # (the IDF mass is normalised over query tokens present in the store, so one incidental
    # common word can be 100% of it). Measured on a real 20-card store.
    "what when where which why who whom whose whether".split()
)


def _tokens(text):
    """Significant tokens for lexical matching: words 3+ chars (minus stopwords) and any number
    2+ digits (keeps years/day-numbers like 2024, 25). This is what carries entity/date identity."""
    out = set()
    import re
    for t in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if t.isdigit():
            if len(t) >= 2:
                out.add(t)
        elif len(t) >= 3 and t not in _STOP:
            out.add(t)
    return out


# --- the relevance BAR's own vocabulary ----------------------------------------------------------
# Used ONLY by DumbMemory._relevance (the pass/fail bar). Ranking, dedup and the wiki-loop keep
# using _tokens/_STOP untouched, so none of them move. Measured on Nick's real chat lines: "Yes off
# you go" cleared the bar on 'yes'/'off', and "agree on all of the above please go ahead on 1,2,3
# and 4" matched five cards on 'all'/'agree' plus the single digits. Grammar and conversation words
# only — nothing here can be a card's subject.
_BAR_FILLER = frozenset(
    "yes yeah yep nope okay thanks thank please all but not off also just really very then than too "
    "here there now again some any more most other such each both either anything everything "
    "something nothing let lets sure maybe else out above below ahead going gone goes went see say "
    "said tell told know think thought thing things stuff cant dont doesn isn aren wasn won didn "
    "couldn shouldn wouldn whilst while ok".split())


def _stem(t: str) -> str:
    """A deliberately small stemmer so 'carry' meets 'carries', 'fix' meets 'fixes' and 'decide'
    meets 'decided'. One suffix, an undoubled consonant ('pinned' -> 'pin'), a trailing 'e'.
    Numbers and short tokens are left alone — they are identifiers, not words."""
    if t.isdigit() or len(t) <= 3:
        return t
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if t.endswith(suf) and len(t) - len(suf) + len(rep) >= 3:
            t = t[:len(t) - len(suf)] + rep
            if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "slz":
                t = t[:-1]
            break
    if len(t) > 3 and t.endswith("e"):
        t = t[:-1]
    return t


def _bar_words(text) -> set:
    """Stems of the meaningful words in ``text``, minus conversational filler."""
    return {_stem(t) for t in _tokens(text) if t not in _BAR_FILLER}


_STANDING_WORDS = frozenset(
    {_stem(t) for t in "standing decision decisions rule rules preference preferences convention "
                       "conventions staple staples pinned pin pins always guideline guidelines principle "
                       "principles policy policies".split()})


def _is_standing_query(query: str) -> bool:
    """A query made only of orientation words ("standing decisions rules preferences") asks for the
    standing rules, not for cards that happen to contain those words. Field report 2026-10-02: that
    exact query, which session-start instructions tell agents to run, returned one old data-model
    decision from a store with 11 pinned rules."""
    words = _bar_words(query)
    return bool(words) and words <= _STANDING_WORDS


def _orphan_entities(text) -> set:
    """Identifiers the word tokeniser cannot carry at all — 'CI', 'v1.2' — so a query made of one
    still has something to match on. An entity with any part that survives ``_tokens`` ('gpt-4',
    'STO-2911', 'CLAUDE.md') is already carried by its words and is not counted twice. A bare
    number is never an orphan: two or more digits are a word already, and one digit is noise."""
    import re
    out = set()
    for e in _entities(text):
        if e.isdigit() or e in _BAR_FILLER or e in _STOP:
            continue
        if not _tokens(" ".join(re.findall(r"[a-z0-9]+", e))):
            out.add(e)
    return out


# Coverage questions (summarise / order events / reason across sessions) need MANY cards; precise
# questions get distracted by them. Depth is inferred from the QUESTION WORDING — the same adaptive
# depth the main engine uses (harness.beam.wants_breadth).
_BREADTH_HINTS = (
    "summar", "what order", "order in which", "order did", "chronolog", "sequence", "timeline",
    "first to last", "evolve", "over the course", "over time", "across", "overall", "all the",
    "everything", "each time", "how did", "progress", "how many", "list ",
)


def wants_breadth(query: str) -> bool:
    q = (query or "").lower()
    return any(h in q for h in _BREADTH_HINTS)


# --- temporal reasoning: entity matching (the 3rd retrieval signal) + time awareness -----------
# The hybrid ranker fuses cosine (meaning) + IDF-keyword (lexical) + ENTITY overlap. Entities are the
# specific handles a temporal/precise question keys on — proper nouns, versions, and dates. Separately,
# cards are time-ordered (by any date they carry, else insertion order) so "what order / before / after
# / most recent" questions can be answered by REASONING over the sequence, not just recalling a fact.

_MONTH = {m: i for i, m in enumerate(
    "jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}


def _entities(text: str):
    """Entity-like tokens (case-sensitive source): proper nouns / acronyms (Capitalised or ALLCAPS)
    and anything carrying a digit (versions, years, dates — v3, gpt-4, 2026-07). Lowercased for match.
    These are the exact handles ('Railway', 'JWT', '2026-03') a precise/temporal question turns on."""
    out = set()
    import re
    for i, raw in enumerate(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-./]*", text or "")):
        tok = raw.strip("-./")
        if not tok:
            continue
        low = tok.lower()
        if any(ch.isdigit() for ch in tok):
            out.add(low)                                  # versions / years / dates
        elif tok[0].isupper() and len(tok) >= 2 and low not in _STOP:
            if tok.isupper() or i > 0:                    # keep acronyms + mid-text proper nouns;
                out.add(low)                              # skip a Title-case FIRST word (sentence noise)
    return out


def _parse_date(text: str):
    """First date-like mention as a sortable (year, month, day), else None. Handles 2026-07-13 /
    2026/7 / '2026-07' / 'March 2026' / a bare 4-digit year. Only 19xx/20xx years count as dates, so
    a '9am' or 'top_k=50' number is never mistaken for a timestamp."""
    import re
    t = text or ""
    m = re.search(r"\b(20\d{2}|19\d{2})[-/](\d{1,2})(?:[-/](\d{1,2}))?", t)
    if m and 1 <= int(m.group(2)) <= 12:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(20\d{2}|19\d{2})",
                  t.lower())
    if m:
        return (int(m.group(2)), _MONTH[m.group(1)], 1)
    m = re.search(r"\b(20\d{2}|19\d{2})\b", t)
    if m:
        return (int(m.group(1)), 1, 1)
    return None


_ORDER_HINTS = ("what order", "in order", "order did", "order in which", "sequence", "chronolog",
                "timeline", "first to last", "step by step", "what happened after", "list the steps")
_RECENCY_HINTS = ("most recent", "latest", "currently", "right now", "as of now", "current ",
                  "these days", "nowadays", "last thing", "recently", "up to date")
_TEMPORAL_HINTS = _ORDER_HINTS + _RECENCY_HINTS + (
    "before", "after", "prior", "earlier", "later", "when did", "when i", "since ", "until ",
    "how long", "over time", "evolve", "changed from", "used to", "back then", "first ", "last ")


def wants_ordering(query: str) -> bool:
    q = (query or "").lower()
    return any(h in q for h in _ORDER_HINTS)


def wants_recency(query: str) -> bool:
    q = (query or "").lower()
    return any(h in q for h in _RECENCY_HINTS)


def wants_temporal(query: str) -> bool:
    q = (query or "").lower()
    return any(h in q for h in _TEMPORAL_HINTS)


class DumbMemory:
    def __init__(self, store=None, hot_cap: int = 60, cards_per_summary: int = 10,
                 dedup_threshold: float = 0.6, embedder=None, narrow_k: int = 30,
                 broad_k: Optional[int] = None, lex_weight: float = 0.6,
                 pin_threshold: float = 0.85, behavioural_pin_boost: float = 0.2,
                 usage_boost: float = 0.05, usage_boost_cap: float = 0.3,
                 decay_half_life: float = 150.0,
                 entity_weight: float = 0.3, recency_weight: float = 0.2,
                 importance_policy: Optional[dict] = None,
                 wiki_link: bool = False, wiki_link_sim: float = 0.86,
                 wiki_surface_sim: float = 0.6, wiki_tok_overlap: float = 0.2,
                 wiki_surface_floor: float = 0.35, wiki_surface_overlap: float = 0.5,
                 trust_weight: float = 0.4, retire_after_failures: int = 3,
                 relevance_floor: float = 0.45, relevance_coverage: float = 0.3,
                 fold_min_age_hours: Optional[float] = None):
        self.store = store or KeywordStore()
        # please_summarise never folds a card younger than this (a lifecycle dial: locked in
        # guided). None = the module default, read at construction so tests can lower it.
        self.fold_min_age_hours = (_FOLD_MIN_AGE_HOURS if fold_min_age_hours is None
                                   else float(fold_min_age_hours))
        self.salience = SalienceFilter()
        self.hot_cap = hot_cap
        self.cards_per_summary = cards_per_summary
        self.dedup_threshold = dedup_threshold
        # Optional on-device embedder: narrows candidates() to the top ``narrow_k`` most relevant
        # cards for FREE (local, no API) so the agent routes over a focused shortlist, not the whole
        # index — the "dumb server on local embeddings" config. None = agent sees every manifest.
        self.embedder = embedder
        # An embedder OBJECT can exist and still be unusable: LocalEmbedder loads its ~90MB model
        # lazily on the first embed() call, so a machine that is offline, behind a proxy/firewall, or
        # simply running for the first time with no internet fails HERE — not at construction. Before
        # this flag that raised on EVERY recall (memory broken, not degraded) and re-attempted the
        # download each time. The first failure is recorded, the meaning term is dropped for the rest
        # of the process, and ranking carries on with keywords + entities (see _embed / _rank_mode).
        self._embed_failed: Optional[str] = None
        # OBSERVABILITY (review finding 8): "are agents actually using this?" was the most important
        # question about the product and it was answerable only by anecdote. Counted per process so
        # the agent can tell the USER in one line — they cannot see tool calls.
        self._session = {"reads": 0, "writes": 0}
        # WRITE/READ IMBALANCE. The failure this product exists to fix is memory that gets written
        # to and never read. A reminder at session start is read once and forgotten; this counter
        # rides back on a store result — a payload the agent is already reading — at the exact
        # moment the imbalance is true. Same lever that fixed the skipped importance_policy.
        self._stores_since_recall = 0
        # ...but ONLY nag when memory holds something the agent has NOT seen this session. Cards
        # this process wrote are already in the agent's context, so recalling them is ceremony —
        # "right in principle, wrong in a single long session" (field report). Cards from earlier
        # sessions, or from other agents on a shared scope, are the ones worth consulting.
        self._written_this_session: set = set()
        self.narrow_k = narrow_k
        # Dynamic k: coverage-style questions retrieve ``broad_k`` cards instead of ``narrow_k``
        # (default 2x). Precise questions stay tight. None -> 2 * narrow_k.
        self.broad_k = broad_k if broad_k is not None else narrow_k * 2
        self._emb_cache = {}   # signal_id -> vector (lazy; local embed is ~5ms/card on an M2)
        # Hybrid retrieval: ranking = cosine + lex_weight * (IDF-weighted keyword overlap). Embedding
        # gets the gist; the lexical term nails EXACT entity/date identity — a query naming "translation
        # API" or "sprint review" pulls those specific cards up even when generic words (sprint,
        # deadline, api) make everything look similar. IDF downweights the common words automatically.
        self.lex_weight = lex_weight
        self._idf = None       # token -> idf (lazy, over the whole store)
        self._tok_cache = {}   # signal_id -> token set
        self._bar_cache = {}   # signal_id -> what the relevance bar reads off a card (see _bar_card)
        self._policy_set_at = None   # when importance_policy was last set by configure (persisted)
        # ENTITY signal (the 3rd retrieval signal): + entity_weight * (fraction of the query's
        # entities the card names). Entities = proper nouns / versions / dates — the exact handles a
        # precise or temporal question turns on, so the card actually naming "Railway"/"2026-03" wins.
        # RECENCY: for "latest / most recent / currently" questions, + recency_weight * how new the card
        # is (insertion order). Ordering questions ("what order / before / after") don't reweight — the
        # picked cards are returned time-SORTED instead (see recall_context), so the agent reasons over
        # a real timeline. Both default modest; set to 0 to disable.
        self.entity_weight = entity_weight
        self.recency_weight = recency_weight
        self._ent_cache = {}   # signal_id -> entity token set
        # PINS: important cards that must never be summarised/archived. A card is pinned if it is
        # explicitly pinned OR its EFFECTIVE importance clears pin_threshold. Effective importance =
        # the card's base importance + a boost for behavioural/procedural notes (tool choices,
        # preferences, corrective notes — "how you work") + a usage boost (a card that keeps getting
        # retrieved is important, regardless of age — recency != importance). So staples survive
        # decay, and things you actually use resist it. Pins still UPDATE via versioning (supersede).
        self.pin_threshold = pin_threshold
        self.behavioural_pin_boost = behavioural_pin_boost
        self.usage_boost = usage_boost
        self.usage_boost_cap = usage_boost_cap
        self._access = {}      # signal_id -> times retrieved (drives the usage boost)
        # PROCEDURAL VALIDATION GATES (STO-2823): a card that's been reported worked/failed on reuse
        # is re-weighted by its derived trust, and one that fails `retire_after_failures` times IN A
        # ROW leaves the hot path (kept + drillable — never deleted, same rule as decay/archive).
        self.trust_weight = trust_weight
        self.retire_after_failures = retire_after_failures
        # RELEVANCE BAR. Ranking sorts; it never asks whether ANYTHING is actually about the query.
        # Field report: a search for "change impact when a new asset arrives" returned cards about
        # sabotage runners and test harnesses, labelled "relevant". Returning the top of a bad list
        # as though it answered the question trains an agent to skim recall results — the exact
        # failure this product exists to fix. A card clears the bar if it shares a meaningful token
        # or an entity with the query (hard topical evidence), or, with a live embedder, meets
        # `relevance_floor` on meaning alone. Nothing clears it -> matched:false and the cards are
        # labelled CLOSEST, not relevant.
        self.relevance_floor = relevance_floor
        # 0.3 admits one-of-three ("changing the deploy pipeline" vs a deploy card) while still
        # rejecting the measured off-topic cases, which land at 0.0 now that question words are
        # stopwords. Set deliberately at the 1/3 boundary, not rounded up past it.
        self.relevance_coverage = relevance_coverage
        # HALF-LIFE DECAY (wires the formerly-dead decay.py): a card's base importance FADES with age
        # (exponential, half at `decay_half_life` age-units). Decay picks WHICH card folds (lowest
        # time-decayed importance), capacity picks WHEN. Age = position from newest in the benchmark
        # (no wall-clock); wall-clock days in production. Behavioural/usage boosts + explicit pins do
        # NOT fade — so old-but-used or old-but-pinned staples survive; old-unused-unimportant sinks.
        from signal_engine.decay import DecayModel
        self.decay_half_life = decay_half_life              # configurable dial (kept in sync with _decay)
        self._decay = DecayModel(half_life=decay_half_life)
        # ENTITY-CANONICAL "WIKI-LOOP" (OKF-style; default OFF -> the frozen path is unchanged). When ON
        # and an embedder is present, store_signals matches each NEW fact to existing cards by MEANING
        # (local-embedding cosine + a shared salient token), so a *changed value* for the same fact
        # ("Fly.io" -> "Railway") — which lexical dedup misses and a cold agent won't flag — is caught.
        # It SURFACES 'possible_supersedes' for an agent to confirm AND auto-links in place only when the
        # match is very high-confidence (cosine >= wiki_link_sim, unambiguous). Supersede is
        # non-destructive (old kept [older]) so a rare wrong link loses nothing. Needs NO system prompt:
        # the conversation's own meaning is the signal -> it holds in the cold case.
        self.wiki_link = wiki_link
        self.wiki_link_sim = wiki_link_sim
        self.wiki_surface_sim = wiki_surface_sim
        self.wiki_tok_overlap = wiki_tok_overlap
        # A changed value ("Fly.io"->"Railway") drops embedding cosine (the entity dominates the short
        # string), so cosine ALONE misses same-fact updates. Also surface when cosine is moderate AND
        # the salient-token overlap is high (same topic) — caught for the agent to confirm. Auto-merge
        # stays cosine-only strict (wiki_link_sim) so a wrong auto-link is still very unlikely.
        self.wiki_surface_floor = wiki_surface_floor
        self.wiki_surface_overlap = wiki_surface_overlap
        # IMPORTANCE POLICY (agent-set, the STO-2813 control): what THIS agent's role weights up —
        # {"boost_kinds":[...], "boost_keywords":[...], "boost":0..1, "persona":"..."}. Applied in
        # effective_importance, so a role change re-weights every card live (no re-score pass needed).
        # Set at runtime via configure(); persisted to a sidecar so a restart / a peer sharing the
        # scope inherits it. This is how an agent OWNS its memory policy, not just its cards.
        self.importance_policy = dict(importance_policy or {})
        # Expected knowledge slots for THIS scope (STO-2850): the question-shaped things a project
        # usually needs answered (audience / deliverable / deadline / …). None -> the generic defaults;
        # an agent can set a role-specific schema via configure(knowledge_slots=...). knowledge_gaps()
        # reports which slots are still EMPTY — attacking the unknown-unknowns retrieval never can.
        self.knowledge_slots = None
        # Lockable dials — the USER's authority over the agent. A SOFT lock (configure lock=[...],
        # persisted, agent-removable via unlock) stops accidental drift; SIGNAL_MEMORY_LOCK (env) is
        # a HARD lock the agent can never change or lift. So the user sets policy — directly or by
        # instructing the agent — then freezes it, and further changes are refused.
        import os as _os
        self._locked: set = set()
        self._hard_locked: set = self._parse_lock(_os.getenv("SIGNAL_MEMORY_LOCK"))
        # AUTONOMY: "guided" (default) — the lifecycle dials (decay, capacity, batch size, pins,
        # retrieval weights) are LOCKED, so memory behaves predictably and the agent works within it
        # (still stores/recalls/pins-per-card/supersedes/summarises-when-asked/cleans up). "auto" —
        # full self-management: the agent may retune every dial on the fly and drive its own lifecycle
        # from its evaluation of how it's doing. importance_policy (the role's priorities) stays open in
        # both. SIGNAL_MEMORY_AUTONOMY sets the default; hard-lock "autonomy" to forbid the switch.
        self.autonomy: str = (_os.getenv("SIGNAL_MEMORY_AUTONOMY", "guided") or "guided").lower()
        if self.autonomy not in ("guided", "auto"):
            self.autonomy = "guided"
        self._load_config()                                # sidecar overrides defaults if present

    # -- write path (agent has already EXTRACTED; server just curates + stores) --------------

    def _to_signal(self, d) -> Optional[Signal]:
        if isinstance(d, Signal):
            return d
        kind = (d.get("kind") or "fact").lower()
        if kind == "behavioural":
            desc = d.get("description") or d.get("content") or ""
            # A behavioural signal must carry at least one behavioural field; if the agent gave none,
            # fall back to putting the description in `interaction` so the note is kept, not dropped.
            interaction = d.get("interaction")
            if not any([d.get("sentiment"), d.get("friction"), d.get("tool_choice"), interaction]):
                interaction = desc or (d.get("title") or "note")
            return behavioural_signal(title=d.get("title") or "note",
                                      description=desc,
                                      keywords=d.get("keywords") or [],
                                      sentiment=d.get("sentiment"), friction=d.get("friction"),
                                      tool_choice=d.get("tool_choice"), interaction=interaction,
                                      importance_base=float(d.get("importance", 0.5)),
                                      pinned=bool(d.get("pinned", False)))
        content = (d.get("content") or "").strip()
        if not content:
            return None
        return fact_signal(title=(d.get("title") or content)[:80], content=content,
                           description=d.get("description") or "", keywords=d.get("keywords") or [],
                           importance_base=float(d.get("importance", 0.5)),
                           pinned=bool(d.get("pinned", False)),
                           volatility=d.get("volatility"),       # invariant|decision|external (STO-2846)
                           basis=d.get("basis"))                 # seen|told|assumed (STO-2912)

    def _dedup(self, signals):
        # Each existing/candidate fact -> (word set, distinctive-identifier set, keyword set). A drop
        # needs high word overlap AND matching identifiers AND compatible keywords — so a differing
        # filename/version/port/number (distinctive) OR a differing entity tag (keywords name the
        # subject, e.g. [auth] vs [payments]) keeps the fact. Prose restatements — no identifiers,
        # and keywords that are equal or a subset (refinement, not a conflict) — still collapse.
        def _feat(s):
            txt = s.content or s.title
            return _words(txt), _distinctive(txt), frozenset(s.keywords or [])
        existing = [_feat(s) for s in self.store.all_signals()
                    if s.is_fact and s.is_current and s.title != _SUMMARY_TITLE]
        kept = []
        for s in signals:
            if s.is_fact:
                w, d, k = _feat(s)
                if any(_jaccard(w, ew) >= self.dedup_threshold and d == ed
                       and (k <= ek or ek <= k)                # compatible entities (subset), not a clash
                       for ew, ed, ek in existing):
                    continue
                existing.append((w, d, k))
            kept.append(s)
        return kept

    def _hot(self):
        # Facts AND behavioural cards are hot/retrievable; only summaries + archived are excluded.
        # (Was `s.is_fact` — which silently dropped behavioural signals from candidates()/recall_context,
        # making preferences/corrections write-only. STO-2815, caught live via the Gemini/agy round-trip.)
        # `retired` (STO-2823) leaves the hot path on the same terms as `archived`: kept + drillable,
        # just not offered as current guidance — a procedure that keeps failing shouldn't be replayed.
        return [s for s in self.store.all_signals()
                if not s.archived and not s.retired
                and s.title not in (_SUMMARY_TITLE, _CONTEXT_TITLE)]

    def _summaries(self):
        return [s for s in self.store.all_signals() if s.title == _SUMMARY_TITLE]

    def store_signals(self, signals) -> dict:
        """Curate (dedup + salience) + store agent-extracted signals. A signal may carry
        ``"supersedes": [old_ids]`` — when the agent CORRECTS or ENHANCES an earlier fact, the old
        cards are annotated ``[older]`` (kept + drillable, never deleted) so current memory stays
        clean. If the hot tier is now over cap, returns a 'please_summarise' instruction."""
        self._session["writes"] += 1
        built, supersedes, sup_ids = [], [], set()
        # CREDENTIAL GUARD — enforced, not merely instructed. Runs before anything is built or
        # written, so a secret never reaches the store, the embedder, or the JSON file on disk.
        signals, redacted = self._scrub(signals)
        for d in signals:
            s = self._to_signal(d)
            if s is None:
                continue
            built.append(s)
            sup = d.get("supersedes") if isinstance(d, dict) else None
            if sup:
                supersedes.append((s, list(sup)))
                sup_ids.add(id(s))
        salient = self.salience.filter(built).kept
        # An explicit supersede is a stated correction -> bypass dedup so a similarly-worded update
        # still lands (lexical dedup can't tell a restatement from a changed value); dedup the rest.
        kept = self._dedup([s for s in salient if id(s) not in sup_ids]) + \
            [s for s in salient if id(s) in sup_ids]
        prior = ([s for s in self.store.all_signals()          # snapshot BEFORE add (wiki-loop targets)
                  if s.is_fact and s.is_current and s.title != _SUMMARY_TITLE]
                 if self.wiki_link and self.embedder else [])
        # Re-storing a card that is ALREADY stored, now with supersedes, is how an agent confirms a
        # 'possible_supersedes' or 'related' hint. An explicit supersede bypasses dedup (above), so
        # that used to leave two identical current cards. The LINK is what was asked for.
        confirmed, kept, supersedes = self._confirm_on_existing(kept, supersedes)
        now = _now_iso()
        for s in kept:
            if not getattr(s, "created_at", ""):
                s.created_at = now                             # episodic time (auto-stamped on store)
        self.store.add(kept)                                   # assigns signal_ids
        self._written_this_session.update(x.signal_id for x in kept)
        n_sup = confirmed + self._apply_supersedes(supersedes, {id(s) for s in kept})
        self._stores_since_recall += 1
        result = {"stored": len(kept)}
        unseen = self._unconsulted()
        if self._stores_since_recall >= _IMBALANCE_AFTER and unseen:
            n = len(unseen)
            # NAMED, closest to what was just stored first: "you have 16 unread cards" is a number
            # to ignore, a title is something to go and read (field report, 2026-09-29).
            about = self._subject_of(kept)
            named = (self._rank(about, unseen) if about else unseen)[:_UNCONSULTED_NAMED]
            names = "; ".join(f"'{(x.title or '').strip()[:60]}'" for x in named if x.title)
            result["imbalance"] = (
                f"Since your last recall: {self._stores_since_recall} stores, 0 recalls — and "
                f"{n} card{'s' if n != 1 else ''} from earlier sessions (or other agents) that you "
                f"have not consulted"
                + (f", including {names}" + (f" (+{n - len(named)} more)" if n > len(named) else "")
                   if names else "")
                + ". Writing without reading is the same as not having memory. "
                "Call recall(...) before your next answer that turns on what the user wants, "
                "prefers, or already decided.")
        result.update(self._pin_saturation_note())
        # STO-2912: every fact carries both labels. A save without them is stored (refusing would
        # lose the note) and named here, where the agent is already reading, with the one call
        # that fixes it. Until then recall shows the card as unverified.
        unlabelled = [{"id": s.signal_id, "title": s.title, "missing": _missing_labels(s)}
                      for s in kept if _missing_labels(s)]
        if unlabelled:
            result["unlabelled"] = unlabelled
            result["unlabelled_note"] = (
                "Every fact needs two labels: volatility (what it depends on: invariant | decision "
                "| external) and basis (how you know it: seen = you observed it, told = the user or "
                "a document said so, assumed = you concluded it). Set them now with "
                "relabel(ids, volatility=..., basis=...) and include both in future store_signals "
                "calls. Until then recall shows these cards as unverified.")
        if not self.importance_policy:
            # Persists on every write until it is set. The single most-skipped step in testing: an
            # agent reads guidance(), answers the dial questions, and never comes back to the policy.
            result["setup_incomplete"] = (
                "No importance_policy is set for this scope, so I cannot weight anything for your "
                "role — everything is stored flat. Call configure(importance_policy={persona, "
                "boost_kinds, boost_keywords}) now; it works in guided mode too.")
        if redacted:
            result["redacted"] = redacted
            result["warning"] = (
                "A credential was found in what you tried to store and has been REDACTED before "
                "saving — memory is a plain file on disk and cards outlive this session. The rest of "
                "the card was kept. Store the fact and where the secret lives, never its value, and "
                "tell the user this happened.")
        if n_sup:
            result["superseded"] = n_sup
        if self.wiki_link and self.embedder:                   # entity-canonical: same fact -> update in place
            wiki = self._wiki_link(kept, prior, {id(s) for s in kept if id(s) in sup_ids})
            if wiki.get("auto_superseded"):
                result["auto_superseded"] = wiki["auto_superseded"]
            if wiki.get("possible_supersedes"):
                result["possible_supersedes"] = wiki["possible_supersedes"]
        # EVERY WRITE IS ALSO A READ (field report, 2026-09-29). An agent that never recalls still
        # reads its save replies, so the reply names what memory already holds on the same subject.
        # Titles only — full text would take a 13-char reply to ~2,250 — and behind the same
        # relevance bar as recall, so an unrelated store adds nothing.
        already = {o["id"] for v in (result.get("possible_supersedes") or {}).values() for o in v}
        already |= {a["old"] for a in result.get("auto_superseded") or []}
        related = self._related(kept, already)
        if related:
            result["related"] = related
            result["related_note"] = (
                "Already in memory on the same subject (titles only; get([id]) for the text). If "
                "your new card REPLACES one of these, store it again with supersedes:[that id] — "
                "that links the two and does not make a second copy. These titles are not a "
                "recall and do not count as one.")
        batches = self.overflow_batches()                      # pins-protected, importance-ordered
        if batches:
            batch = batches[0]
            # Ids and titles only (field report 2026-10-02: the full cards made this reply 7,700 to
            # 11,500 chars every time). The text is one get(source_ids) away, read only when
            # summarising — not carried on a save reply that is mostly about something else.
            result["please_summarise"] = {
                "instruction": ("Memory is over capacity. Read these cards with get(source_ids), "
                                "summarise them into one concise paragraph, then call "
                                "store_summary(text, source_ids). They are archived, not deleted: "
                                "kept, and reachable through the summary."),
                "cards": [{"id": s.signal_id, "title": s.title} for s in batch],
                "source_ids": [s.signal_id for s in batch],
            }
        return result

    @staticmethod
    def _subject_of(cards) -> str:
        """What a batch of cards is about, as a query: their titles and keywords."""
        return " ".join(filter(None, (
            " ".join(filter(None, [s.title, " ".join(s.keywords or [])])) for s in cards))).strip()

    def _related(self, kept, exclude_ids=()) -> list:
        """Up to ``_RELATED_MAX`` current cards about the same subject as the cards just stored:
        [{"id", "title"}], best first. Judged by the relevance bar on each new card's title and
        keywords. It does not touch usage, the read counters or the imbalance count — a title in a
        save reply is not a read. Skipped for a batch over ``_RELATED_BATCH_MAX`` (a bulk import
        would rank the whole store once per card to report nothing anyone reads)."""
        if not kept or len(kept) > _RELATED_BATCH_MAX:
            return []
        skip = {s.signal_id for s in kept} | set(exclude_ids or ())
        pool = [x for x in self._hot()
                if getattr(x, "is_current", True) and x.signal_id not in skip]
        if not pool:
            return []
        found = {}
        for s in kept:
            q = self._subject_of([s])
            if not q:
                continue
            rel = self._relevance(q, pool)
            hits = [x for x in self._rank(q, pool) if rel.get(x.signal_id, {}).get("relevant")]
            for pos, x in enumerate(hits[:_RELATED_MAX]):
                if pos < found.get(x.signal_id, (pos + 1, None))[0]:
                    found[x.signal_id] = (pos, x)
        best = sorted(found.values(), key=lambda t: t[0])[:_RELATED_MAX]
        return [{"id": x.signal_id, "title": x.title} for _, x in best]

    def standing_rule_titles(self) -> dict:
        """The pinned rules for this scope as TITLES — what the first reply of a session carries,
        so an agent that never asks still learns which rules exist. Bodies stay one call away
        (before_you_start / get): full text would take the first reply from ~3,700 to ~9,300
        chars on this repo's own store, and a reply that long is how setup got skipped."""
        pinned = [x for x in self._hot() if self.is_pinned(x) and x.is_current and x.title]
        return {"rules": [{"id": x.signal_id, "title": x.title} for x in pinned[:_ORIENT_RULES_MAX]],
                "more": max(0, len(pinned) - _ORIENT_RULES_MAX)}

    def relabel(self, ids, volatility: Optional[str] = None, basis: Optional[str] = None) -> dict:
        """Set the labels on cards that are already stored, without touching their content (STO-2912).
        ``volatility``: invariant | decision | external — what the fact depends on. ``basis``: seen |
        told | assumed — how the agent knows it. Either may be omitted to leave it as it is. Works on
        any fact card by id, current or [older] or archived; behavioural notes, summaries and context
        cards take no labels. Returns {relabelled, skipped, unknown, rejected, unlabelled_remaining}."""
        from signal_engine.signal import BASES, VOLATILITIES, normalise_basis, normalise_volatility
        rejected = {}
        if volatility is not None and normalise_volatility(volatility) is None:
            rejected["volatility"] = f"{volatility!r} is not one of {list(VOLATILITIES)}"
        if basis is not None and normalise_basis(basis) is None:
            rejected["basis"] = f"{basis!r} is not one of {list(BASES)}"
        want_v, want_b = normalise_volatility(volatility), normalise_basis(basis)
        by_id = {s.signal_id: s for s in self.store.all_signals()}
        changed, skipped, unknown = [], [], []
        for i in [str(x) for x in (ids or [])]:
            s = by_id.get(i)
            if s is None:
                unknown.append(i)
            elif not _is_labelable(s):
                skipped.append(i)
            else:
                if want_v:
                    s.volatility = want_v
                if want_b:
                    s.basis = want_b
                if want_v or want_b:
                    changed.append(s)
        if changed:
            if hasattr(self.store, "update_many"):
                self.store.update_many(changed)
            else:
                for s in changed:
                    self.store.update(s)
        remaining = sum(1 for s in self._hot() if s.is_current and _missing_labels(s))
        return {"relabelled": len(changed), "skipped": skipped, "unknown": unknown,
                "rejected": rejected, "unlabelled_remaining": remaining}

    def _scrub(self, signals):
        """Strip credentials from incoming signals. Returns (clean_signals, report).

        Scans every free-text field an agent can fill. Redacts rather than refusing, so the useful
        fact around the secret survives — a refused card loses "deploy with railway up" along with
        the token. The report names what was found per card so the agent can tell the user."""
        out, report = [], []
        for d in signals:
            if not isinstance(d, dict):
                out.append(d)
                continue
            d = dict(d)
            hits = []
            for field in ("content", "title", "description", "interaction", "tool_choice"):
                val = d.get(field)
                if isinstance(val, str):
                    clean, found = _redact_secrets(val)
                    if found:
                        d[field] = clean
                        hits += found
            kws = d.get("keywords")
            if isinstance(kws, list):
                cleaned = []
                for k in kws:
                    if isinstance(k, str):
                        ck, f = _redact_secrets(k)
                        hits += f
                        cleaned.append(ck)
                    else:
                        cleaned.append(k)
                d["keywords"] = cleaned
            if hits:
                report.append({"title": str(d.get("title", ""))[:60],
                               "found": sorted(set(hits))})
            out.append(d)
        return out, report

    def _confirm_on_existing(self, kept, supersedes):
        """A new card that carries ``supersedes`` and is word-for-word a card ALREADY current in
        the store is a confirmation, not a new fact: mark the named old cards superseded by the
        card that is already there and do not add a second copy. Returns (links made, the cards
        still to add, the supersedes still to apply)."""
        if not supersedes:
            return 0, kept, supersedes

        def key(s):
            return (s.kind, " ".join((s.title or "").lower().split()),
                    " ".join((s.content or "").lower().split()))

        everything = self.store.all_signals()
        by_id = {s.signal_id: s for s in everything}
        current = {}
        for s in everything:
            if s.is_current and not s.archived:
                current.setdefault(key(s), s)
        kept_ids = {id(s) for s in kept}
        changed, left, drop = [], [], set()
        for new_sig, old_ids in supersedes:
            twin = current.get(key(new_sig))
            if twin is None or id(new_sig) not in kept_ids:
                left.append((new_sig, old_ids))
                continue
            for oid in old_ids:
                old = by_id.get(oid)
                if old is not None and old.signal_id != twin.signal_id and old.is_current:
                    old.superseded_by = twin.signal_id
                    changed.append(old)
            drop.add(id(new_sig))
        if changed:
            if hasattr(self.store, "update_many"):
                self.store.update_many(changed)
            else:
                for s in changed:
                    self.store.update(s)
        return len(changed), [s for s in kept if id(s) not in drop], left

    def _apply_supersedes(self, supersedes, kept_ids) -> int:
        """Mark each named old card superseded_by the surviving new card (agent-driven, no LLM).
        The old card becomes not-current -> shown '[older]', decays first, loses auto-pin — but is
        kept and drillable. Skipped if the new card was deduped away (no real change happened)."""
        if not supersedes:
            return 0
        by_id = {s.signal_id: s for s in self.store.all_signals()}
        changed = []
        for new_sig, old_ids in supersedes:
            if id(new_sig) not in kept_ids:
                continue
            for oid in old_ids:
                old = by_id.get(oid)
                if old is not None and old.signal_id != new_sig.signal_id and old.is_current:
                    old.superseded_by = new_sig.signal_id
                    changed.append(old)
        if changed:
            if hasattr(self.store, "update_many"):
                self.store.update_many(changed)
            else:
                for s in changed:
                    self.store.update(s)
        return len(changed)

    def _salient_tokens(self, s: Signal):
        toks = self._tok_cache.get(s.signal_id)
        if toks is None:
            toks = self._tok_cache[s.signal_id] = _tokens(self._embed_text(s))
        return toks

    def _embed(self, texts):
        """Embed via the local embedder, or return None when it isn't usable.

        The model downloads lazily on first use, so 'the embedder is configured' and 'the embedder
        works' are different things — offline, behind a firewall, or on a first run with no internet,
        this is where it breaks. One failure disables the meaning term for this process (recorded in
        ``_embed_failed``) instead of raising on every call, so search degrades to keyword + entity
        ranking rather than dying. ``ranked`` then reports "keyword" and the error is surfaced."""
        if not self.embedder or self._embed_failed:
            return None
        try:
            return self.embedder.embed(list(texts))
        except Exception as exc:                           # model download / load / runtime failure
            self._embed_failed = f"{type(exc).__name__}: {exc}"[:200]
            self._emb_cache.clear()                        # partial vectors are not comparable
            return None

    def _ensure_emb(self, sigs) -> None:
        need = [s for s in sigs if s.signal_id not in self._emb_cache]
        if need:
            vecs = self._embed([self._embed_text(s) for s in need])
            if vecs is None:
                return
            for s, v in zip(need, vecs):
                self._emb_cache[s.signal_id] = v

    def _wiki_link(self, kept, prior, explicit_ids) -> dict:
        """Entity-canonical linking (the 'living wiki' loop). For each NEW current fact card, find
        PRIOR current fact cards that mean the SAME fact — high embedding cosine AND a shared salient
        token (topic anchor). Cards that survived lexical dedup but are semantically near-identical are
        the *changed value* case dedup can't see. SURFACE every match >= wiki_surface_sim for an agent
        to confirm; AUTO-supersede in place only when a SINGLE candidate clears wiki_link_sim with real
        token overlap (non-destructive: old kept, annotated [older], decays first). No LLM, no prompt."""
        new_cards = [s for s in kept if s.is_fact and s.is_current
                     and s.title != _SUMMARY_TITLE and id(s) not in explicit_ids]
        prior = [s for s in prior if s.is_current]
        if not new_cards or not prior:
            return {}
        self._ensure_emb(new_cards + prior)
        if self._embed_failed:                             # no vectors -> no meaning-matching to do
            return {}                                      # (storing must still succeed)
        possible, auto, used_old = {}, [], set()
        for s in new_cards:
            sv, stoks = self._emb_cache[s.signal_id], self._salient_tokens(s)
            scored = []
            for o in prior:
                if o.signal_id in used_old or not o.is_current or o.signal_id == s.signal_id:
                    continue
                otoks = self._salient_tokens(o)
                if not (stoks & otoks):
                    continue                                   # need a shared entity/topic anchor
                cos = _cosine(sv, self._emb_cache[o.signal_id])
                ov = _jaccard(stoks, otoks)
                # high cosine, OR moderate cosine + high topic overlap (the changed-value case)
                if cos >= self.wiki_surface_sim or (cos >= self.wiki_surface_floor
                                                    and ov >= self.wiki_surface_overlap):
                    scored.append((cos, ov, o))
            if not scored:
                continue
            scored.sort(key=lambda t: t[0], reverse=True)
            possible[s.signal_id] = [{"id": o.signal_id, "title": o.title, "sim": round(c, 3)}
                                     for c, _, o in scored[:3]]
            top_cos, top_ov, top_old = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            if (top_cos >= self.wiki_link_sim and top_ov >= self.wiki_tok_overlap
                    and top_cos - second >= 0.03):             # unambiguous single winner only
                top_old.superseded_by = s.signal_id
                used_old.add(top_old.signal_id)
                auto.append({"new": s.signal_id, "old": top_old.signal_id, "sim": round(top_cos, 3)})
                possible.pop(s.signal_id, None)                # resolved -> drop from surfaced list
        if used_old:
            self._persist([o for o in prior if o.signal_id in used_old])
        return {"possible_supersedes": possible, "auto_superseded": auto}

    def store_summary(self, text: str, source_ids: List[str], keywords=None,
                      summary_id: Optional[str] = None) -> dict:
        """Fold a batch into a summary index card (originals archived, kept, drillable). Pass
        ``summary_id`` to REFRESH an existing (stale) summary in place — after a fact inside it was
        corrected — re-generating its text from the current cards instead of leaving it stale."""
        by_id = {s.signal_id: s for s in self.store.all_signals()}
        batch = [by_id[i] for i in source_ids if i in by_id]
        manifest = "; ".join((s.title or "")[:50] for s in batch)
        if keywords:
            kw = keywords
        else:                                              # most shared first, so a trimmed view
            counts = {}                                    # keeps the ones that describe the batch
            for s in batch:
                for k in (s.keywords or []):
                    counts[k] = counts.get(k, 0) + 1
            kw = sorted(counts, key=lambda k: (-counts[k], k))
        for s in batch:
            s.archived = True
        existing = by_id.get(summary_id) if summary_id else None
        if existing is not None and existing.title == _SUMMARY_TITLE:          # REFRESH in place
            existing.content, existing.description, existing.keywords = text, manifest, kw
            existing.source_ids = list(source_ids)
            self._emb_cache.pop(existing.signal_id, None)     # re-embed the refreshed text
            self._tok_cache.pop(existing.signal_id, None)
            self._bar_cache.pop(existing.signal_id, None)
            self._idf = None
            self._persist(batch + [existing])
            return {"summary_id": existing.signal_id, "refreshed": True, "archived": len(batch),
                    **self._after_folding()}
        summ = fact_signal(title=_SUMMARY_TITLE, content=text, description=manifest,
                           keywords=kw, importance_base=0.6)
        summ.source_ids = list(source_ids)
        self._persist(batch)
        self.store.add([summ])
        return {"summary_id": summ.signal_id, "archived": len(batch), **self._after_folding()}

    def _after_folding(self) -> dict:
        """What a summary changed, in the numbers stats() reports. Field report 2026-10-02: "it said
        10 cards were archived, yet stats still shows 117 cards" — true and expected, because
        archived cards are kept, but nothing said so."""
        cards = self.store.all_signals()
        return {"active_now": len(self._hot()),
                "kept_total": len(cards),
                "note": ("Archived cards are kept, never deleted: total_cards still counts them, and "
                         "recall reaches them through the summary. 'active_now' is what recall "
                         "searches directly.")}

    def _summary_manifest(self, s: Signal) -> dict:
        """A summary's manifest with its keywords trimmed to the ``_SUMMARY_KEYWORDS_SHOWN`` most
        shared across the cards behind it. The stored card keeps every keyword for search."""
        man = _manifest(s)
        kws = list(s.keywords or [])
        if len(kws) > _SUMMARY_KEYWORDS_SHOWN:
            by_id = {x.signal_id: x for x in self.store.all_signals()}
            counts = {}
            for sid in (s.source_ids or []):
                for k in (getattr(by_id.get(sid), "keywords", None) or []):
                    counts[k] = counts.get(k, 0) + 1
            kws.sort(key=lambda k: -counts.get(k, 0))      # stable: ties keep stored order
            man["keywords"] = kws[:_SUMMARY_KEYWORDS_SHOWN]
            man["keywords_total"] = len(s.keywords or [])
        return man

    def _persist(self, sigs) -> None:
        if hasattr(self.store, "update_many"):
            self.store.update_many(sigs)               # one file write, not one per card
        else:
            for s in sigs:
                self.store.update(s)

    # -- read path (server serves STRUCTURE; the agent ROUTES) ------------------------------

    @staticmethod
    def _embed_text(s: Signal) -> str:
        """What we embed a card by: its content PLUS its keywords/title metadata. For a summary
        index card this folds the union of its cards' keywords into the routable vector, so a precise
        query surfaces the right summary — and its source_ids then drill straight to the exact card."""
        return " ".join(filter(None, [s.content or s.title, s.title, " ".join(s.keywords or [])]))

    def _ensure_idf(self):
        """IDF over the whole store (lazy, cached): rare tokens (entities like 'translation',
        'fine-tuning') weigh more than common ones ('sprint', 'api', 'deadline')."""
        if self._idf is None:
            import math
            cards = self.store.all_signals()
            n = max(len(cards), 1)
            df = {}
            for s in cards:
                toks = _tokens(self._embed_text(s))
                self._tok_cache[s.signal_id] = toks
                for t in toks:
                    df[t] = df.get(t, 0) + 1
            self._idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
        return self._idf

    def _seq_index(self, s: Signal) -> int:
        """Insertion order from the signal id (``sig-N``) — the card's position on the timeline when
        no explicit date is present. Lower = older."""
        sid = getattr(s, "signal_id", "") or ""
        parts = sid.split("-")                             # "sig-<n>-<token>" -> the <n> is the order
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 10 ** 9

    def _time_key(self, s: Signal):
        """A sortable timeline key: cards with a parsed date come first, in date order; undated cards
        follow in insertion order. So sorting a set of event cards reconstructs their chronology."""
        d = _parse_date(s.content or s.title)
        idx = self._seq_index(s)
        return (0, d, idx) if d else (1, (9999, 12, 31), idx)

    def _entities_of(self, s: Signal):
        e = self._ent_cache.get(s.signal_id)
        if e is None:
            e = self._ent_cache[s.signal_id] = _entities(self._embed_text(s))
        return e

    def _rank(self, query, signals):
        """Rank ``signals`` by the HYBRID score, all to ``query``:
            cosine (meaning) + lex_weight * IDF-keyword (lexical) + entity_weight * entity overlap
        and, for 'latest/most recent' questions only, + recency_weight * how new the card is. Cosine
        gives the gist; the lexical + entity terms give exact entity/date recall; recency breaks ties
        toward the current value. No API — local embedder + on-device token/entity match.

        The embedder is OPTIONAL: only the cosine term needs it. With no embedder the meaning term is
        simply 0 and the lexical + entity + recency terms rank on their own — degraded, but still a
        ranking. (It used to be gated on the embedder, so an install without one got NO ranking at all:
        candidates() handed back every card in insertion order, which puts the oldest card first and
        the answer last. With a working embedder the score is byte-identical to before.)"""
        qv = None
        if self.embedder and not self._embed_failed:
            self._ensure_emb(signals)
            q = self._embed([query])
            qv = q[0] if q else None                       # None -> the embedder died; keyword-only
        idf = self._ensure_idf() if self.lex_weight else {}
        q_tokens = _tokens(query)
        denom = sum(idf.get(t, 0.0) for t in q_tokens) or 1.0    # normalise: fraction of query's mass
        q_ent = _entities(query) if self.entity_weight else set()
        recency = bool(self.recency_weight) and wants_recency(query)
        if recency:
            idxs = [self._seq_index(s) for s in signals]
            lo, span = min(idxs), (max(idxs) - min(idxs)) or 1

        def score(s):
            total = _cosine(qv, self._emb_cache[s.signal_id]) if qv is not None else 0.0
            if self.lex_weight:
                stoks = self._tok_cache.get(s.signal_id)
                if stoks is None:
                    stoks = self._tok_cache[s.signal_id] = _tokens(self._embed_text(s))
                lex = sum(idf.get(t, 0.0) for t in q_tokens if t in stoks) / denom
                total += self.lex_weight * lex
            if q_ent:
                cent = self._entities_of(s)
                if cent:
                    total += self.entity_weight * (len(q_ent & cent) / len(q_ent))
            if recency:
                total += self.recency_weight * ((self._seq_index(s) - lo) / span)
            if not getattr(s, "is_current", True):
                total -= 0.15          # a superseded [older] card ranks BELOW the current value it
                                       # was replaced by — so "what is X now" doesn't return the stale
                                       # answer — but stays retrievable (history / "what did I use before")
            return total

        return sorted(signals, key=score, reverse=True)

    def _bar_card(self, s: Signal) -> dict:
        """What the relevance bar reads off one card, cached: the words of its TITLE, each KEYWORD
        as a unit (a multi-word keyword is a phrase, not a bag of words), every word of its body,
        and the identifiers in its title/keywords."""
        c = self._bar_cache.get(s.signal_id)
        if c is None:
            phrases = [p for p in (frozenset(_bar_words(k)) for k in (s.keywords or [])) if p]
            head = " ".join(filter(None, [s.title, " ".join(s.keywords or [])]))
            title = frozenset(_bar_words(s.title or ""))
            c = self._bar_cache[s.signal_id] = {
                "title": title, "phrases": phrases,
                "topic_all": title.union(*phrases) if phrases else title,
                "topic_ents": frozenset(_entities(head)
                                        | {str(k).lower() for k in (s.keywords or [])}),
                "body": frozenset(_bar_words(self._embed_text(s)) | self._entities_of(s))}
        return c

    def _relevance(self, query, signals) -> dict:
        """Per-card evidence that a card is actually ABOUT the query — separate from its rank.

        Returns {signal_id: {"lex", "about", "ent", "cos", "relevant"}}.

        ABOUT, not MENTIONS. A card's title and keywords say what it is about; its body merely
        mentions things. Field report (2026-09-29): recall "returns about a quarter of the store for
        any question and always says it matched". Measured on two real 22-card stores: "how do I
        run the tests" returned nine cards because seven of them contain the word "run" somewhere.
        So a query word found in the title or keywords counts in full, and one found only in the
        body counts ``_BAR_BODY_WEIGHT``. A multi-word keyword is a phrase: "first run" is about
        first runs, and does not make a card about running things.

        A card clears the bar on any ONE of:
          strong   two or more query words in its title/keywords (or the query's only word);
          meaning  cosine >= ``relevance_floor`` on its own — a real paraphrase;
          backed   coverage ``lex`` >= ``relevance_coverage`` AND cosine >= two thirds of the floor.
                   One shared word is a hint, not proof; meaning has to agree with it.
        With no live embedder nothing can back a hint up, so it degrades to words alone: one
        title/keyword match at ``relevance_coverage``, or a body that covers ``_BAR_KEYWORD_ONLY``.

        Before any of that the query loses its conversational filler (``_BAR_FILLER``), is stemmed,
        and — in a store big enough to judge — loses the words that sit on the title/keywords of
        more than half the cards, because a project's own name tells its cards apart from
        nothing. ``ent`` is still reported; a bare shared entity no longer passes by itself (one
        shared digit used to), since an identifier is a word like any other here.

        Result on the two stores, same labelled questions: cards per question 5.1 -> 2.3 and
        6.4 -> 2.8, on topic 36% -> 80% and 47% -> 95%, every answer still found, chat lines that
        matched 9 of 18 -> 1 of 18."""
        if not query or not signals:
            return {}
        qv = None
        if self.embedder and not self._embed_failed:
            self._ensure_emb(signals)
            q = self._embed([query])
            qv = q[0] if q else None
        units = _bar_words(query) | _orphan_entities(query)
        hot = self._hot()
        common = set()
        if units and len(hot) >= _BAR_UBIQUITY_MIN_CARDS:
            df = {}
            for s in hot:
                for t in self._bar_card(s)["topic_all"]:
                    df[t] = df.get(t, 0) + 1
            distinctive = {u for u in units if df.get(u, 0) / len(hot) <= _BAR_UBIQUITY_SHARE}
            units = distinctive or units           # a query of ONLY common words keeps them all
            # Words on more than a quarter of the titles still COUNT, but two of them are not
            # enough on their own: "Nick" + "decision" sit on dozens of cards together.
            common = {u for u in units if df.get(u, 0) / len(hot) > _BAR_COMMON_SHARE}
        q_ent = _entities(query) if self.entity_weight else set()
        support = self.relevance_floor * _BAR_SUPPORT_SHARE
        out = {}
        for s in signals:
            c = self._bar_card(s)
            topical = set(c["title"]) | c["topic_ents"]
            for phrase in c["phrases"]:
                if len(phrase) == 1 or phrase <= units:
                    topical |= phrase
            on_topic = units & topical
            mentioned = (units & c["body"]) - on_topic
            about = len(on_topic)
            lex = ((about + _BAR_BODY_WEIGHT * len(mentioned)) / len(units)) if units else 0.0
            cent = self._entities_of(s) if q_ent else set()
            ent = (len(q_ent & cent) / len(q_ent)) if q_ent and cent else 0.0
            backed = qv is not None and s.signal_id in self._emb_cache
            cos = _cosine(qv, self._emb_cache[s.signal_id]) if backed else 0.0
            # Two title/keyword words decide it on their own only if both are distinctive, or if
            # together they are half the query. "Nick" + "memory" out of a seven-word question is a
            # hint, and a hint has to be backed by meaning (below).
            strong = (len(on_topic - common) >= 2
                      or (about >= 2 and (about * 2 >= len(units) or len(units) == len(common)))
                      or (about == 1 and len(units) == 1))
            if strong or cos >= self.relevance_floor:
                relevant = True
            elif backed:
                relevant = lex >= self.relevance_coverage and cos >= support
            else:
                relevant = ((about >= 1 and lex >= self.relevance_coverage)
                            or lex >= max(self.relevance_coverage, _BAR_KEYWORD_ONLY))
            out[s.signal_id] = {"lex": round(lex, 3), "about": about, "ent": round(ent, 3),
                                "cos": round(cos, 3), "relevant": bool(relevant)}
        return out

    def _embedder_note(self) -> dict:
        """Surfaced ONLY when the configured embedder actually failed, so a healthy response is
        unchanged. Tells the agent (and so the user) that search is running weaker and why —
        otherwise a firewalled machine looks identical to one that was never configured."""
        if not self._embed_failed:
            return {}
        return {"embedder_error": (
            f"The local embedder is configured but could not load ({self._embed_failed}). Search is "
            "running on keywords + entities only, which is weaker. Usually this is no internet on "
            "first run (the ~90MB model downloads once), a proxy/firewall blocking huggingface.co, "
            "or the 'local' extra not installed. Fix: run once with internet, or reinstall with "
            "uv sync --extra store --extra local.")}

    def _rank_mode(self, query: str) -> str:
        """Which ranking actually ran, so a degraded install is VISIBLE instead of silent:
        "hybrid" = meaning + keyword + entity (the embedder is loaded AND working), "keyword" =
        keyword + entity only (no embedder configured, or it failed to load — still ranked, just
        weaker), "none" = nothing to rank against (no query)."""
        if not query:
            return "none"
        return "hybrid" if (self.embedder and not self._embed_failed) else "keyword"

    def _narrow(self, query: str, signals, k=None):
        """Rank ``signals`` by relevance to ``query`` and return the top ``k`` (dynamic: breadth
        questions get ``broad_k``, precise get ``narrow_k``). Ranking no longer requires an embedder —
        see ``_rank`` — so a no-embedder install gets keyword+entity ranking instead of the whole
        store in insertion order. Everything is still returned when there's no query to rank against."""
        if not query or not signals:
            return signals
        if k is None:
            k = self.broad_k if wants_breadth(query) else self.narrow_k
        return self._rank(query, signals)[:k]

    def candidates(self, query: str = "") -> dict:
        """Hand the agent the index to route over: hot manifests + summary manifests. The hot list is
        narrowed to the top-k most relevant (free, on-device) — k adapts to the question (breadth vs
        precise). No LLM either way. ``ranked`` reports which ranking ran ("hybrid" with the embedder,
        "keyword" without it, "none" with no query) so a degraded install is visible, not silent.

        ``narrowed`` is True ONLY when filtering actually DROPPED cards (fewer shown than exist). With
        an embedder + query but a hot tier already below narrow_k, nothing is filtered — the list is
        RANKED but complete — so ``narrowed`` is False and ``shown`` reports the count (e.g. "13/13").
        This stops a consumer mistaking a full, unfiltered payload for a filtered ranking."""
        self._session["reads"] += 1
        self._stores_since_recall = 0          # any read clears the write/read imbalance nudge
        full_hot = self._hot()
        hot = self._narrow(query, full_hot)
        if wants_ordering(query):
            hot = sorted(hot, key=self._time_key)          # time-ordered for "what order / timeline"
        # Narrow summaries the same way as hot: over a long-running session the summary index grows,
        # and handing the agent EVERY summary manifest is a context blob that scales without bound.
        # Return the top-k most relevant (by the hybrid ranker); the agent drills the ones that fit.
        summaries = self._summaries()
        if query and len(summaries) > _SUMMARY_K:
            summaries = self._rank(query, summaries)[:_SUMMARY_K]
        elif len(summaries) > _SUMMARY_K:
            summaries = summaries[-_SUMMARY_K:]            # no query -> the most recent rollups
        superseded = {s.signal_id for s in self.store.all_signals() if not s.is_current}

        def _summary_manifest(s):
            man = _manifest(s)
            man["stale"] = bool(superseded and superseded.intersection(s.source_ids or []))
            return man                                        # a fact inside it was corrected

        return {"hot": [_manifest(s) for s in hot],
                "summaries": [_summary_manifest(s) for s in summaries],
                "narrowed": bool(query and len(hot) < len(full_hot)),
                "shown": f"{len(hot)}/{len(full_hot)} hot cards",
                "ranked": self._rank_mode(query),
                **self._embedder_note(),
                "broad": bool(query and wants_breadth(query)),
                "instruction": ("Pick the ids relevant to the query from 'hot'. If the answer isn't "
                                "there, choose relevant 'summaries' and call drill(summary_id) to "
                                "search the cards behind them. Then fetch chosen ids with get(ids). "
                                "If a summary is 'stale' (a fact inside it changed), drill it and "
                                "refresh via store_summary(text, source_ids, summary_id).")}

    def drill(self, summary_id: str) -> dict:
        summ = next((s for s in self._summaries() if s.signal_id == summary_id), None)
        if not summ:
            return {"cards": []}
        by_id = {s.signal_id: s for s in self.store.all_signals()}
        cards = [by_id[i] for i in (summ.source_ids or []) if i in by_id]
        return {"cards": [_manifest(s) for s in cards]}

    def _card_text(self, s: Signal) -> str:
        """The readable content of one card. A volatile (decision/external) fact is prefixed with a
        'verify before acting' note (STO-2846) so the agent hedges instead of asserting;
        signal_to_context itself is untouched (shared with the frozen benchmark path, whose signals
        carry no volatility -> no marker there). Shared by get() and recall() so the one-call and
        two-call read paths never drift apart."""
        from signal_engine.engine import signal_to_context
        ctx = signal_to_context(s)
        note = _verify_note(s)
        return f"{note} {ctx}" if note else ctx

    def recall(self, query: str = "", limit: int = 5) -> dict:
        """ONE-CALL recall: the cards actually about ``query``, WITH their content.

        Ranking sorts; it does not judge. So this also applies a RELEVANCE BAR (see ``_relevance``)
        and reports ``matched``. When nothing clears the bar you get ``matched: false`` and a few
        cards labelled CLOSEST — never the top of a bad list dressed up as an answer, which is how
        an agent learns to skim recall results and stop trusting them.

        Three more things it does so the agent does not have to:
          * AUTO-WIDENS to every card that cleared the bar (up to ``_RECALL_MAX``), rather than
            cutting at ``limit`` and setting ``more`` for the agent to notice and re-ask;
          * TRUNCATES content to a snippet once there are more than three cards (full text via
            ``get(id)``), because a 4,000-token recall is one an agent stops making;
          * turns a null result into a NEXT STEP ("nothing here about X — store it when you learn
            it") instead of a dead end.

        ``candidates()`` remains the index view for routing over a large store yourself."""
        self._session["reads"] += 1
        self._stores_since_recall = 0          # reading clears the write/read imbalance nudge
        all_hot = self._hot()
        hot = self._narrow(query, all_hot)
        if wants_ordering(query):
            hot = sorted(hot, key=self._time_key)      # "what order / before / after" -> chronological
        limit = max(0, int(limit))

        rel = self._relevance(query, hot)
        if query:
            relevant = [x for x in hot if rel.get(x.signal_id, {}).get("relevant")]
        else:
            relevant = hot                              # no query = no bar to clear; this is a browse
        # ARCHIVED cards are kept for exactly this. Field report 2026-10-02: a summary folded
        # fresh gotchas away and recall then answered "nothing stored" about something that was
        # stored. They are judged by the same bar and come AFTER every active card that cleared it.
        from_archive = set()
        if query:
            archived = [x for x in self.store.all_signals()
                        if x.archived and x.is_current and not getattr(x, "retired", False)
                        and x.title not in (_SUMMARY_TITLE, _CONTEXT_TITLE)]
            if archived:
                arel = self._relevance(query, archived)
                # Ordered by the bar's own evidence (title/keyword words, then meaning), not by the
                # hot-tier ranker: a long body that mentions the words should not outrank the card
                # whose title IS the question (measured: the exact Playwright card came third).
                found = sorted((x for x in archived if arel.get(x.signal_id, {}).get("relevant")),
                               key=lambda x: (-arel[x.signal_id]["about"], -arel[x.signal_id]["cos"]))
                from_archive = {x.signal_id for x in found}
                relevant = relevant + found
        rules = set()
        if _is_standing_query(query):
            pins = [x for x in all_hot if self.is_pinned(x) and x.is_current]
            if pins:                                    # the rules ARE the answer; nothing else
                rules = {x.signal_id for x in pins}
                relevant, from_archive = pins, set()
        matched = bool(relevant)

        if matched:
            picked = relevant[:max(limit, min(len(relevant), _RECALL_MAX))]
            cut = len(relevant) - len(picked)
        else:
            picked = hot[:min(limit, _CLOSEST_K)] if limit else []
            cut = 0
        self._touch(picked)                             # retrieval boosts importance (usage)

        cards = []
        for rank, x in enumerate(picked):
            card = _manifest(x)
            if not matched:
                # A miss is TITLES ONLY (field report 2026-10-02: three full cards on a miss cost
                # ~10,000 chars for an answer of "nothing here"). The text is one get() away.
                card["relevance"] = "closest match only — NOT about your query"
                cards.append(card)
                continue
            text = self._card_text(x)
            # The top cards are the likely ANSWER, so they are always full text — a truncated "what
            # command do I run" is one call too many (field report). Only cards past _FULL_TEXT_TOP
            # are snippeted, and never mid-backtick, so a command survives the cut. A standing rule
            # is followed, not skimmed, so it gets the longer rule allowance.
            limit_chars = _RULE_SNIPPET if x.signal_id in rules else _SNIPPET
            if rank >= _FULL_TEXT_TOP and len(text) > limit_chars:
                card["content"] = _snip(text, limit_chars)
                card["truncated"] = True                # full text: get([id])
            else:
                card["content"] = text
            if x.signal_id in rules:
                card["standing_rule"] = True
            if x.signal_id in from_archive:
                card["archived"] = True                 # folded into a summary; still the original
            cards.append(card)

        summaries = self._summaries()
        if query and len(summaries) > _RECALL_SUMMARY_K:
            summaries = self._rank(query, summaries)[:_RECALL_SUMMARY_K]
        else:
            summaries = summaries[-_RECALL_SUMMARY_K:]

        if not all_hot:
            shown = "memory is empty for this scope"
            instruction = ("Nothing is stored here yet. Tell the user you have no memory of this "
                           "yet — then store what you learn with store_signals() so the next "
                           "session does.")
        elif matched:
            shown = f"{len(picked)} relevant of {len(all_hot)} cards"
            instruction = ("These cards are about your query — use them directly. HEED any 'verify "
                           "before acting' note: that fact may have changed, so hedge and re-check "
                           "rather than asserting it. The first few cards are always full text; a "
                           "later card marked truncated has more via get([id]). If the answer still "
                           "isn't here, drill(summary_id) on a "
                           "summary for older archived history.")
        else:
            shown = (f"NO relevant cards for this query — showing the {len(picked)} closest "
                     f"of {len(all_hot)}")
            instruction = ("Nothing stored here is about your query. The titles below are only the "
                           "closest by ranking — do NOT treat them as an answer or pad a reply with "
                           "them (get([id]) if one genuinely looks right). Say you don't have it. Then, when you learn the answer, store it "
                           "with store_signals() so this is not a dead end next time. "
                           "(drill(summary_id) can still reach archived history.)")

        return {"cards": cards,
                "matched": matched,
                "shown": shown,
                "ranked": self._rank_mode(query),
                **self._embedder_note(),
                "summaries": [self._summary_manifest(x) for x in summaries],
                "more": cut > 0,
                "instruction": instruction}

    def before_you_start(self, doing: str = "", limit: int = 5) -> dict:
        """What you need to know BEFORE doing ``doing`` — named after the moment, not the mechanism.

        ``recall`` is a mechanism. An agent picking up a ticket does not think "I should recall"; it
        thinks "I'm starting a ticket". A tool whose name matches the trigger gets called because the
        trigger fires, and the agent does not have to invent a search query first — it just says what
        it is about to do.

        It is also genuinely more than an alias: starting something needs BOTH halves, which used to
        be two calls plus knowing to make them —
          standing_rules  the pinned staples that always apply ("always use uv", "personal commit
                          email"), regardless of the task;
          about_this      what memory holds about this specific piece of work, relevance-barred, so
                          an unrelated store says so instead of padding the answer.
        """
        pinned = [x for x in self._hot() if self.is_pinned(x) and x.is_current]
        rules = []
        for x in pinned[:_RECALL_MAX]:
            card = _manifest(x)
            text = self._card_text(x)
            if len(text) > _RULE_SNIPPET:                  # a rule is followed, not skimmed
                card["content"] = _snip(text, _RULE_SNIPPET)
                card["truncated"] = True
            else:
                card["content"] = text
            rules.append(card)
        found = self.recall(doing, limit=limit) if doing else {"cards": [], "matched": False,
                                                               "shown": "no task given"}
        # When nothing clears the bar, about_this is EMPTY rather than echoing closest-matches —
        # here they would mostly repeat standing_rules back as "closest", which is noise next to a
        # list the agent must follow anyway.
        # Setup that is still outstanding rides on THIS result too. Field report: the four setup
        # calls "look like housekeeping" next to the task briefing, so an agent went straight here
        # and skipped them; the briefing is the one call it reliably makes, so it must say so.
        outstanding = self.action_required()
        return {"doing": doing,
                **({"action_required": outstanding} if outstanding else {}),
                **self._role_check(),
                "standing_rules": rules,
                "about_this": found.get("cards", []) if found.get("matched") else [],
                "matched": found.get("matched", False),
                "shown": found.get("shown", ""),
                "ranked": found.get("ranked", "none"),
                "summaries": found.get("summaries", []),
                "instruction": (
                    ("SETUP IS OUTSTANDING for this scope — do the action_required items before your "
                     "next answer; they take one call each. " if outstanding else "")
                    + "standing_rules ALWAYS apply — follow them even when about_this is empty. "
                    + ("about_this holds what memory knows for this task; use it directly."
                       if found.get("matched") else
                       "Nothing stored is about this task, so do not pad your answer with the "
                       "closest cards. Say so, and store what you learn as you go.")
                    + " HEED any 'verify before acting' note.")}

    def import_notes(self, path: str, sections=None, apply: bool = True) -> dict:
        """One-time import of a markdown notes file into cards, so a fresh scope is useful from the
        first session. Field report: "an empty memory gave no value in the first session... a
        one-time import from CLAUDE.md and your learned rules would have made it useful from the
        start". Host-agnostic: it takes a path; the agent names whatever its host keeps.

        It imports MARKDOWN LIST ITEMS only ("- " / "* ", with indented continuation lines). Narrative
        paragraphs are the project's story for humans and stay in the file — see the memory-vs-
        instruction-file rule in the manual. Which sections: ``sections`` (case-insensitive substring
        match on headings), or by default the headings that look like rules (gotchas, rules,
        conventions, decisions, commands, build/test, setup, learned...). A file with no headings
        imports all its items. Fenced code blocks are skipped.

        Each item becomes ONE fact card: title from its bold lead (or first words), the item text as
        content (bold markers stripped, backticks KEPT so commands survive), keywords from the heading
        + 'imported' + the file name, importance 0.6, volatility 'decision' under a decisions heading
        else 'invariant', NOT pinned (pins are explicit), and the source recorded in the description.
        Everything goes through store_signals, so the credential guard, salience and dedup all apply;
        an item whose title is already in the store is skipped. ``apply=False`` previews without
        writing anything."""
        import os
        import re
        raw_path = str(path or "").strip()
        p = os.path.expanduser(raw_path)
        if not raw_path or not os.path.isfile(p):
            return {"error": f"not a readable file: {raw_path!r}", "imported": 0}
        if os.path.getsize(p) > _IMPORT_MAX_BYTES:
            return {"error": f"{raw_path!r} is over {_IMPORT_MAX_BYTES:,} bytes — import one section "
                             f"(sections=[...]) or a smaller file", "imported": 0}
        with open(p, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        wanted = [str(w).lower() for w in (sections or []) if str(w).strip()]
        has_headings = bool(re.search(r"^#{1,6}\s+\S", text, re.M))

        def selected(heading: str) -> bool:
            h = heading.lower()
            if wanted:
                return any(w in h for w in wanted)
            if not has_headings:
                return True
            return any(w in h for w in _IMPORT_DEFAULT_SECTIONS)

        heading, sel = "", (not has_headings)
        used, skipped, items = [], [], []          # items: (heading, raw item text)
        cur = None
        in_code = False

        def flush():
            nonlocal cur
            if cur is not None:
                items.append(tuple(cur))
                cur = None

        for line in text.splitlines():
            if line.strip().startswith("```"):
                in_code = not in_code
                flush()
                continue
            if in_code:
                continue
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if m:
                flush()
                heading = m.group(2).strip()
                sel = selected(heading)
                (used if sel else skipped).append(heading)
                continue
            m = re.match(r"^\s*[-*]\s+(.*)", line)
            if m:
                flush()
                if sel:
                    cur = [heading, m.group(1).strip()]
                continue
            if cur is not None and line.startswith("  ") and line.strip():
                cur[1] += " " + line.strip()           # indented continuation of the same item
                continue
            flush()
        flush()

        base = os.path.basename(p)
        sigs = []
        for head, body in items:
            plain = re.sub(r"\*\*|__", "", body)
            plain = re.sub(r"\s+", " ", plain).strip()
            if len(plain) < _IMPORT_MIN_CHARS:
                continue
            lead = re.match(r"\*\*(.+?)\*\*", body)
            title = (lead.group(1) if lead else plain).strip().rstrip(".:→- ")[:80]
            head_words = [w for w in re.findall(r"[a-z][a-z0-9_-]{2,}", head.lower())
                          if w not in _STOP][:4]
            sigs.append({"kind": "fact", "title": title, "content": plain,
                         "keywords": head_words + ["imported", base],
                         "importance": 0.6,
                         "volatility": "decision" if "decision" in head.lower() else "invariant",
                         "basis": "told",                       # a document said so
                         "description": f"imported from {base} § {head or 'top'}"})
        existing = {(x.title or "").lower() for x in self.store.all_signals()}
        fresh = [x for x in sigs if x["title"].lower() not in existing]
        already = len(sigs) - len(fresh)
        result = {"file": base, "apply": bool(apply),
                  "sections_used": used, "sections_skipped": skipped,
                  "candidates": len(sigs), "already_present": already, "imported": 0,
                  "preview": [{"title": x["title"], "section": x["description"].split("§ ", 1)[-1]}
                              for x in fresh[:_IMPORT_PREVIEW]]}
        if not apply or not fresh:
            result["instruction"] = (
                f"Preview only — nothing written. {len(fresh)} item(s) would be imported from "
                f"{base} (sections_used); call again with apply=true, or pass sections=[...] to "
                f"choose headings." if fresh else
                f"Nothing to import from {base}: {already} item(s) already present, and no other "
                f"list items under the selected headings (sections_used / sections_skipped).")
            return result
        res = self.store_signals(fresh)
        result["imported"] = int(res.get("stored", 0))
        result["deduplicated"] = len(fresh) - result["imported"]
        for k in ("redacted", "warning", "please_summarise", "possible_supersedes"):
            if k in res:
                result[k] = res[k]
        result["instruction"] = (
            f"Imported {result['imported']} card(s) from {base} as ordinary (unpinned) cards. If some "
            f"are genuine standing rules, re-store those with pinned:true. Tell the user in ONE line "
            f"what was imported. Do not import the same file again — re-runs skip what is present.")
        return result

    def get(self, ids) -> List[str]:
        self._session["reads"] += 1
        self._stores_since_recall = 0          # any read clears the write/read imbalance nudge
        by_id = {s.signal_id: s for s in self.store.all_signals()}
        got = [by_id[i] for i in ids if i in by_id]
        self._touch(got)                                   # retrieval boosts importance (usage)
        out = [self._card_text(s) for s in got]
        return out

    def list_cards(self, kind: Optional[str] = None, pinned: Optional[bool] = None,
                   include_archived: bool = False, include_older: bool = True,
                   contains: Optional[str] = None, limit: Optional[int] = None,
                   include_retired: bool = False, unlabelled: bool = False) -> dict:
        """Browse what's stored (the inspect / 'get_memories' path). Filters: ``kind``
        ('fact'|'behavioural'), ``pinned`` (True/False), ``include_archived`` (folded-away cards),
        ``include_older`` (superseded [older] versions), ``include_retired`` (procedures retired for
        repeated failure, STO-2823), ``contains`` (substring over title/content/keywords),
        ``unlabelled`` (only fact cards missing a volatility or basis label, STO-2912). Returns
        card manifests (+ pinned/archived flags) and match counts. Read-only."""
        needle = (contains or "").lower().strip()
        out = []
        for s in self.store.all_signals():
            if s.title in (_SUMMARY_TITLE, _CONTEXT_TITLE):
                continue
            if not include_archived and getattr(s, "archived", False):
                continue
            if not include_retired and getattr(s, "retired", False):
                continue
            if not include_older and not s.is_current:
                continue
            if kind and s.kind != kind:
                continue
            if pinned is not None and self.is_pinned(s) != bool(pinned):
                continue
            if unlabelled and not _missing_labels(s):
                continue
            if needle:
                hay = " ".join([s.title or "", s.content or "", " ".join(s.keywords or [])]).lower()
                if needle not in hay:
                    continue
            man = _manifest(s)
            man["pinned"] = self.is_pinned(s)
            man["archived"] = bool(getattr(s, "archived", False))
            man["uses"] = self._access.get(s.signal_id, 0)   # how many times it's been retrieved
            out.append(man)
        matched = len(out)
        if limit:
            out = out[:int(limit)]
        return {"cards": out, "count": len(out), "matched": matched,
                "summaries": len(self._summaries())}

    # -- procedural validation gates (STO-2823) ---------------------------------------------------

    _OUTCOMES = ("worked", "failed")

    def procedure_outcome(self, ids, outcome: str) -> dict:
        """Report how a REMEMBERED PROCEDURE actually went when it was reused, so memory can stop
        recommending steps that no longer work (the tool moved, the flag changed, the 'fix' didn't).

        ``outcome`` is ``"worked"`` or ``"failed"``. A win raises the card's derived ``trust`` and
        clears its failure streak; a failure lowers trust and extends the streak. ``retire_after_
        failures`` CONSECUTIVE failures retire the card — it leaves the hot path but is KEPT and
        drillable (never deleted), and a later win reinstates it, because the world can change back.
        **Pinned cards are never retired** — those are the user's staples, not the agent's to drop
        (the failure is still recorded, so trust reflects reality).
        """
        if outcome not in self._OUTCOMES:
            raise ValueError(f"outcome must be one of {self._OUTCOMES}, got {outcome!r}")
        won = outcome == "worked"
        by_id = {s.signal_id: s for s in self.store.all_signals()}
        updated, not_found, changed = [], [], []
        for sid in (ids or []):
            s = by_id.get(sid)
            if s is None:
                not_found.append(sid)
                continue
            if won:
                s.wins += 1
                s.fail_streak = 0
                s.retired = False                            # a win brings a retired procedure back
            else:
                s.fails += 1
                s.fail_streak += 1
                if not self.is_pinned(s) and s.fail_streak >= self.retire_after_failures:
                    s.retired = True
            changed.append(s)
            updated.append({"id": sid, "trust": round(s.trust, 3), "wins": s.wins,
                            "fails": s.fails, "retired": bool(s.retired)})
        if changed:
            self.store.update_many(changed)                  # batch — one persist, not one per card
        out = {"updated": updated}
        if not_found:
            out["not_found"] = not_found
        return out

    def unpin(self, ids) -> dict:
        """Take cards OFF the pin set WITHOUT deleting or superseding them (STO-2848). The only route
        before this was to supersede a wrongly-pinned card with a near-duplicate — which corrupts the
        supersede chain (that chain is meant to read 'what we used to believe and why it changed', not
        'nothing changed except a flag'). unpin() clears the explicit pin flag AND sets ``no_pin`` so
        the card stays off the pin set even if its importance would re-derive a pin (the escape hatch
        for a card a broad importance_policy keeps pinning). Non-destructive: the card stays current,
        stored, and retrievable; it just decays/archives normally now. To pin it again, re-store the
        fact with ``pinned: true`` (a fresh current card, the system's 'pins update by re-storing'
        model). Returns the ids unpinned and any not found."""
        by_id = {s.signal_id: s for s in self.store.all_signals()}
        unpinned, not_found, changed = [], [], []
        for sid in (ids or []):
            s = by_id.get(sid)
            if s is None:
                not_found.append(sid)
                continue
            s.pinned = False
            s.no_pin = True
            changed.append(s)
            unpinned.append(sid)
        if changed:
            if hasattr(self.store, "update_many"):
                self.store.update_many(changed)              # one persist, not one per card
            else:
                for s in changed:
                    self.store.update(s)
        out = {"unpinned": unpinned}
        if not_found:
            out["not_found"] = not_found
        return out

    def cold_cards(self, min_idle: Optional[int] = None, max_uses: int = 0,
                   older_than_days: Optional[float] = None) -> dict:
        """Flag CLEANUP candidates — cards that have (almost) never been retrieved and are old. 'Old'
        is measured two ways: by default, by how many cards were added since (``min_idle``, default
        ``hot_cap`` — deterministic, always works); or, if you pass ``older_than_days``, by real
        wall-clock time using each card's last_used/created_at timestamp. 'Cold' = retrieved <=
        ``max_uses`` times. Excludes pins, summaries, context cards, and [older] versions. The engine
        NEVER deletes on its own — this surfaces cards the agent/user may want to ``delete()``."""
        hot = self._hot()                                    # oldest -> newest (insertion order)
        min_idle = self.hot_cap if min_idle is None else int(min_idle)
        now = None
        if older_than_days is not None:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
        last = len(hot) - 1
        cold = []
        for i, s in enumerate(hot):
            idle = last - i                                  # how many cards were added after this one
            if older_than_days is None and idle < min_idle:
                break                                        # position mode: everything after is newer
            if self.is_pinned(s) or self._access.get(s.signal_id, 0) > max_uses:
                continue
            if older_than_days is not None:                  # time mode: cold for >= N real days
                age = _age_days(s.last_used or s.created_at, now)
                if age is None or age < float(older_than_days):
                    continue
            man = _manifest(s)
            man["uses"] = self._access.get(s.signal_id, 0)
            man["idle"] = idle
            man["last_used"] = s.last_used or None
            man["created_at"] = s.created_at or None
            cold.append(man)
        hint = (f"{len(cold)} card(s) stored a while ago and never retrieved — review and delete(ids) "
                "any junk. Memory never deletes on its own." if cold else "nothing to clean up")
        return {"cold": cold, "count": len(cold), "hint": hint}

    def delete(self, ids) -> dict:
        """Permanently forget cards by id — the ONE destructive operation (privacy / 'forget me' /
        GDPR). Everything else in the lifecycle decays + archives and stays drillable; this removes
        the cards outright, strips them from any summary's source list, and clears their caches.
        Returns the count deleted."""
        ids = [i for i in (ids or [])]
        removed = self.store.delete(ids) if hasattr(self.store, "delete") else 0
        for i in ids:
            self._emb_cache.pop(i, None)
            self._tok_cache.pop(i, None)
            self._bar_cache.pop(i, None)
            self._ent_cache.pop(i, None)
            self._access.pop(i, None)
        # Deleting a card that SUPERSEDED an older one would leave that older card pointing at a ghost
        # — frozen as [older] with no current version, so the surviving fact is demoted/hidden. Promote
        # it back to current: its replacement is gone, so it's the live value again (history not lost).
        gone = set(ids)
        orphaned = [s for s in self.store.all_signals() if getattr(s, "superseded_by", None) in gone]
        for s in orphaned:
            s.superseded_by = None
        if orphaned:
            self.store.update_many(orphaned)
        if removed:
            self._idf = None                               # store changed -> recompute IDF lazily
        return {"deleted": removed}

    # -- swarm interop: a live per-agent context card peers can read (STO-2742) --------------

    @staticmethod
    def _context_agent(s: Signal) -> str:
        import json
        try:
            return (json.loads(s.content or "{}").get("agent") or "").lower()
        except Exception:
            return ""

    def set_context_card(self, agent: str, identity: str = None, focus: str = None,
                         priorities=None, goals=None, capabilities=None) -> dict:
        """Publish/update THIS agent's swarm profile in the (shared) scope — who I am, what I'm
        working on now (focus), what matters (priorities), what I'm trying to ACHIEVE (goals), and
        what I can DO (capabilities: my tools/skills) — so peers can 'tap into each other's brains'
        and hand off work knowingly. One card per agent, MERGED on update (only the fields you pass
        change; the rest survive — so a focus update never wipes your declared capabilities). Pinned
        + kept out of normal recall (its own channel). See id_card() for the composed handshake view."""
        import json
        agent = (agent or "agent").strip()
        prior = next((c for c in self.context_cards(agent=agent)["cards"]), {})
        updates = {"identity": identity, "focus": focus, "priorities": priorities,
                   "goals": goals, "capabilities": capabilities}
        card = {"agent": agent, **{k: prior.get(k) for k in updates},
                **{k: v for k, v in updates.items() if v is not None}}   # merge: keep unset fields
        stale = [s.signal_id for s in self.store.all_signals()
                 if s.title == _CONTEXT_TITLE and self._context_agent(s) == agent.lower()]
        if stale:
            self.delete(stale)                              # upsert: replace this agent's card
        sig = fact_signal(title=_CONTEXT_TITLE, content=json.dumps(card),
                          keywords=["context", agent.lower()], importance_base=0.9, pinned=True)
        self.store.add([sig])
        return {"agent": agent, "id": sig.signal_id}

    def _experience(self) -> dict:
        """The memory's EARNED track record — derived, never stored, so it can't be inflated. Health
        (how big/loaded) + what procedures have proven to work vs been retired for failing. This is
        the CV an agent shows on a handshake: 'here's what I've been shown works.'"""
        self.store.sync()                                  # peers' latest outcomes count too
        cards = self.store.all_signals()
        validated = [s for s in cards
                     if getattr(s, "wins", 0) and not getattr(s, "retired", False)]
        retired = [s for s in cards if getattr(s, "retired", False)]
        proven = sorted(validated, key=lambda s: s.wins, reverse=True)[:5]
        st = self.stats()
        return {"memory": {"cards": st["total_cards"], "hot": st["hot"], "pins": st["pins"],
                           "summaries": st["summaries"], "archived": st["archived"],
                           "health": st["health"]},
                "track_record": {
                    "validated_procedures": len(validated),
                    "retired_procedures": len(retired),
                    "proven": [{"title": s.title, "wins": s.wins, "fails": s.fails} for s in proven],
                    "learned_to_avoid": [s.title for s in retired[:5]]}}

    def id_card(self, agent: str = None, introduce: bool = False) -> dict:
        """The agent ID card — a HANDSHAKE artifact composed from what memory already holds, not a
        new store. Two layers: the shared SCOPE (role/importance_policy, autonomy, locks, and the
        DERIVED experience — health + earned track record) shown once; and the per-AGENT profiles
        (identity / focus / priorities / goals / capabilities). Pass ``agent`` for one; omit for the
        whole roster — that's how a peer sizes up who's on the shared brain and what each brings
        before handing off work.

        ``introduce=True`` renders a PORTABLE self-introduction instead — for introducing yourself to
        an agent or human who does NOT have memory access (the card is YOUR data, so they need
        nothing). Returns two renderings from one call: ``text`` (plain-English, for a human or a
        memory-less agent to read) and ``card`` (a compact, self-contained blob for automation). Both
        always carry your earned track record. Read-only."""
        soft = sorted(self._locked - self._hard_locked)
        hard = sorted(self._hard_locked)
        full = {"scope": {"role": self.importance_policy or None, "autonomy": self.autonomy,
                          "experience": self._experience(),
                          "locked": ({"soft": soft, "hard": hard} if (soft or hard) else None),
                          "path": getattr(self.store, "path", None)},
                "agents": self.context_cards(agent=agent)["cards"]}
        return self._introduction(agent, full) if introduce else full

    def _introduction(self, agent, full) -> dict:
        """Render `full` into a portable self-introduction — needs nothing on the far side."""
        profiles = full["agents"]
        prof = {}
        if agent:
            prof = next((p for p in profiles if (p.get("agent") or "").lower() == agent.lower()),
                        {"agent": agent})
        elif len(profiles) == 1:                            # solo agent -> introduce itself
            prof = profiles[0]
        scope = full["scope"]
        exp = scope["experience"]
        tr = exp["track_record"]
        role = (scope["role"] or {}).get("persona") if scope["role"] else None
        health = (exp["memory"]["health"] or "").split(" — ")[0] or None
        caps = prof.get("capabilities") or []
        goals = prof.get("goals") or []
        proven = tr["proven"]

        card = {"name": prof.get("agent"), "identity": prof.get("identity"), "role": role,
                "capabilities": caps, "goals": goals, "focus": prof.get("focus"),
                "track_record": {"validated": tr["validated_procedures"],
                                 "retired": tr["retired_procedures"],
                                 "proven": [{"title": p["title"], "wins": p["wins"]} for p in proven]},
                "memory_health": health}

        name = prof.get("agent") or "A memory-backed agent"
        lines = [f"**{name}**" + (f" — {prof['identity']}" if prof.get("identity") else "")]
        if prof.get("focus"):
            lines.append(f"Currently: {prof['focus']}")
        if caps:
            lines.append(f"Can do: {', '.join(map(str, caps))}")
        if goals:
            lines.append(f"Aiming for: {', '.join(map(str, goals))}")
        if proven:
            lines.append("Proven: " + ", ".join(f"{p['title']} (worked {p['wins']}×)" for p in proven))
        else:
            lines.append("Proven: nothing yet — newly started.")
        tail = ([f"role: {role}"] if role else []) + [f"{scope['autonomy']} mode"]
        if health:
            tail.append(f"memory {health}")
        lines.append("(" + " · ".join(tail) + ")")
        return {"text": "\n".join(lines), "card": card}

    def context_cards(self, agent: str = None) -> dict:
        """Read the swarm's context cards — every agent's current profile (or one agent's). This is
        how agents coordinate: understand who else is on the task and what each is doing."""
        import json
        self.store.sync()                                  # reflect peers' latest writes (shared scope)
        cards = []
        for s in self.store.all_signals():
            if s.title != _CONTEXT_TITLE or s.archived:
                continue
            try:
                c = json.loads(s.content or "{}")
            except Exception:
                continue
            if agent and (c.get("agent") or "").lower() != agent.strip().lower():
                continue
            cards.append(c)
        return {"agents": [c.get("agent") for c in cards], "cards": cards}

    # -- long-term lifecycle: fold old cards into summaries-as-index, keep them drillable ----

    def effective_importance(self, s: Signal, age: Optional[float] = None, cap: bool = True) -> float:
        """A card's importance for pin/decay decisions: base importance (FADED by half-life if an
        ``age`` is given), + a boost for behavioural/procedural notes (tool choices, preferences —
        'how you work'), + a usage boost (retrieved cards earn importance; recency != importance).
        The boosts do NOT fade — so old-but-used or behavioural staples stay high; ``age=None`` (the
        default, used by is_pinned) means no time-fade, so the protected pin set is age-stable.

        ``cap`` clamps the result to 1.0 (right for ranking/decay, where scores live in [0,1]);
        ``cap=False`` returns the raw stacked value. The PIN decision no longer uses this at all —
        see ``pin_importance``. Field evidence (16 of 17 cards pinned in one session): the policy
        and usage boosts were turning routine facts into permanent pins, switching the lifecycle
        off without anyone deciding it. Ranking and decay resistance keep every boost; permanence
        does not."""
        base = float(getattr(s, "importance_base", 0.0) or 0.0)
        if age is not None and age > 0:
            base *= self._decay.curve.factor(age, self._decay.half_life)
        imp = base
        if getattr(s, "is_behavioural", False):
            imp += self.behavioural_pin_boost
        imp += min(self.usage_boost_cap, self.usage_boost * self._access.get(s.signal_id, 0))
        imp += self._policy_boost(s)                        # what THIS agent's role weights up
        # Validated procedures rank up, failed ones rank down (STO-2823). Only cards with REPORTED
        # evidence move — an unvalidated card ranks exactly as it did before this feature existed,
        # so the gate can never silently re-rank a store nobody has given outcomes for.
        if getattr(s, "has_outcome_evidence", False):
            imp += (s.trust - 0.5) * self.trust_weight
        if not getattr(s, "is_current", True):
            imp *= 0.4                                      # superseded [older] versions decay FIRST
        return min(1.0, imp) if cap else imp

    def _policy_boost(self, s: Signal) -> float:
        """Extra importance from the agent's importance_policy — a card matching the role's kinds or
        keywords is weighted up (a legal agent boosts citations; a coding agent boosts tool-choices)."""
        pol = self.importance_policy
        if not pol:
            return 0.0
        boost = float(pol.get("boost", 0.2) or 0.0)
        if getattr(s, "kind", None) in set(pol.get("boost_kinds") or []):
            return boost
        kws = {k.lower() for k in (pol.get("boost_keywords") or [])}
        if kws and kws.intersection(k.lower() for k in (s.keywords or [])):
            return boost
        return 0.0

    def pin_importance(self, s: Signal) -> float:
        """The importance that decides a PIN — deliberately narrower than ``effective_importance``.

        A pin is permanent: never faded, never folded into a summary, never retired. So only signals
        that MEAN "this is a standing rule" may derive one: the agent's own importance score, plus
        the behavioural boost (tool choices, preferences, corrections — "how you work" IS the
        staple). The importance_policy boost ("what my role weights up") and the usage boost ("I
        keep retrieving this") make a card RANK higher and SURVIVE decay longer — that is what they
        are for — but they do not make it immortal. Field report, one session: persona boost + the
        0.85 threshold pinned 16 of 17 cards; "if that pattern holds the lifecycle is switched off
        without anyone deciding it." Uncapped, so a pin_threshold above 1.0 still bites (STO-2847)."""
        imp = float(getattr(s, "importance_base", 0.0) or 0.0)
        if getattr(s, "is_behavioural", False):
            imp += self.behavioural_pin_boost
        return imp

    def _pin_saturation_note(self) -> dict:
        """A warning for the STORE reply and stats() when most current cards are pinned. Pins never
        fade or summarise, so past a point the lifecycle is off — and nobody chose that. Lands in a
        payload the agent is already reading (the lever that works), only when it is true."""
        hot = [x for x in self._hot() if getattr(x, "is_current", True)]
        if len(hot) < _PIN_SATURATION_MIN:
            return {}
        pins = [x for x in hot if self.is_pinned(x)]
        if len(pins) / len(hot) < _PIN_SATURATION_RATIO:
            return {}
        return {"pins_saturating": (
            f"{len(pins)} of {len(hot)} current cards are pinned. Pinned cards never fade or fold "
            f"into summaries, so at this ratio the lifecycle is effectively switched off. Pin only "
            f"standing rules (tools, conventions, constraints); score routine facts below "
            f"{self.pin_threshold:g}; unpin(ids) the ones that are not staples.")}

    def _unconsulted(self) -> list:
        """Current cards this process did NOT write: earlier sessions' cards, or other agents' on a
        shared scope. These are what a recall can actually ADD to the agent's context."""
        return [x for x in self._hot()
                if getattr(x, "is_current", True) and x.signal_id not in self._written_this_session]

    def is_pinned(self, s: Signal) -> bool:
        """Pinned = explicitly pinned OR pin_importance (base + behavioural boost) clears the
        threshold. Pinned cards are
        exempt from decay/archival — they stay first-class index cards (updatable via versioning). A
        superseded [older] version is never pinned — so correcting a pinned staple lets the OLD value
        decay while the NEW one inherits protection. ``no_pin`` (set by unpin(), STO-2848) forces a
        card OFF the pin set even if its importance would otherwise derive a pin — the escape hatch for
        a card the policy keeps re-pinning. The derived check is UNCAPPED (STO-2847) so re-tuning the
        threshold/policy stays effective."""
        if not getattr(s, "is_current", True) or getattr(s, "no_pin", False):
            return False
        return bool(getattr(s, "pinned", False)) or self.pin_importance(s) >= self.pin_threshold

    def stats(self) -> dict:
        """Health/growth report — the 'when to clear out' signal for the agent/human. Storage is
        tiny (~0.6 KB/card on disk), so disk is never the issue; this flags when the card count gets
        big enough that cold-load embedding + per-query scan slow down (fix: hierarchical
        consolidation — summarise the summaries — and/or persist embeddings)."""
        self.store.sync()                                  # reflect peers' latest writes (shared scope)
        cards = self.store.all_signals()
        hot = self._hot()
        archived = [s for s in cards if getattr(s, "archived", False)]
        pins = [s for s in hot if self.is_pinned(s)]
        total = len(cards)
        health = ("healthy" if total < 20000 else
                  "getting large — consider hierarchical consolidation (summarise the summaries) "
                  "and persisting embeddings so cold-load + retrieval stay fast")
        return {"total_cards": total, "hot": len(hot), "summaries": len(self._summaries()),
                "archived": len(archived), "pins": len(pins),
                "approx_disk_kb": round(total * 0.63), "health": health,
                "path": getattr(self.store, "path", None),   # where THIS scope's memory lives
                "policy": self.importance_policy or None,
                "locked": sorted(self._locked | self._hard_locked) or None,
                "autonomy": self.autonomy,                   # "guided" (dials locked) | "auto" (self-manage)
                "search": self._rank_mode("x"),              # "hybrid" | "keyword" — is the embedder live?
                "session": dict(self._session),              # reads/writes this process (observability)
                "summary": self._summary_line(len(hot), len(pins), total,
                                              len(archived), len(self._summaries())),
                **self._policy_age(),
                "unlabelled": sum(1 for s in hot if s.is_current and _missing_labels(s)),
                **self._pin_saturation_note(),
                **self._embedder_note(),
                "cold": self.cold_cards()["count"]}          # old + never-retrieved -> cleanup hint

    def _first_run_questions(self) -> list:
        """The setup conversation, written for the SERVER to supply rather than the agent to invent.

        The dials that change how memory FEELS are invisible: a user never learns they exist, so
        everyone silently inherits defaults that may be wrong for their work. Worse, an agent asked
        to "set memory up" has to guess both the questions and the trade-offs, and will phrase them
        in variable names. So each entry carries the question in plain English, why it matters, the
        options with the exact configure() call each implies, and what happens if they do not care.

        Deliberately short and skippable — and, since the field report that they were "a lot for
        someone who just wants to start", a MENU rather than a script: the agent infers what it can,
        applies it, tells the user in a line, and asks only what it cannot infer (see
        _FIRST_RUN_APPROACH). Only surfaced while the scope is unconfigured (see guidance()).
        """
        return [
            {"ask": "What do you want me to prioritise remembering for you?",
             "why": "decides what gets ranked and kept first, and what fades — pinning stays an "
                    "explicit choice",
             "options": ["name the work — 'you're my coding assistant, prioritise deploy commands "
                         "and tool choices'"],
             "sets": 'configure(importance_policy={"persona": ..., "boost_keywords": [...]})',
             "if_they_dont_care": "infer it from what they are working on, and say what you chose"},

            {"ask": "How long should I hold on to things you stop referring to?",
             "why": "unused detail fades so the useful stuff stays sharp; pinned items never fade",
             "options": ["'fade fast' — a few weeks",
                         "'as it comes' — a few months (current default)",
                         "'keep everything' — effectively forever"],
             "sets": "configure(decay_half_life=45|150|1000, user_requested=true)",
             "if_they_dont_care": "leave it at 150"},

            {"ask": "How much should stay instantly to hand before I start folding older notes "
                    "into summaries?",
             "why": "bigger keeps more detail immediately available; smaller keeps recall tight "
                    "and cheap. Nothing is ever lost either way — folded notes stay searchable",
             "options": ["'keep it tight'", "'as it comes' (current default)", "'remember more'"],
             "sets": "configure(hot_cap=30|60|120, user_requested=true)",
             "if_they_dont_care": "leave it at 60"},

            {"ask": "Do you want to decide these settings yourself, or shall I manage them as I "
                    "go and adjust when something is not working?",
             "why": "BOTH are fine. 'Guided' (the current default) keeps memory predictable and "
                    "leaves changes to you — I still ask when I think something should change. "
                    "'Auto' means I retune decay, capacity and ranking on an ongoing basis from "
                    "what I see actually helping, without stopping to ask. Pick auto if you would "
                    "rather not think about it; you can switch back any time by saying so",
             "options": ["'I'll decide' — autonomy stays guided",
                         "'you manage it' — autonomy auto, I tune it continuously"],
             "sets": 'configure(autonomy="guided"|"auto")',
             "if_they_dont_care": "stay guided, and tell them auto is available whenever they want it"},

            {"ask": "Any standing rules or preferences I should treat as permanent?",
             "why": "pinned cards never fade or get folded away — this is how 'always use X' sticks",
             "options": ["anything they always want done a certain way"],
             "sets": 'store_signals([{..., "pinned": true}])',
             "if_they_dont_care": "skip it; pin them as they come up"},
        ]

    def _summary_line(self, hot: int, pins: int, total: int, archived: int = 0,
                      summaries: int = 0) -> str:
        """One short, human-readable line the agent can relay in conversation. The user cannot see
        tool calls, so without this they have no idea whether memory is on, healthy, or being used
        at all — which is exactly how "my agents aren't using it" went unnoticed for weeks."""
        bits = [f"{total} card{'s' if total != 1 else ''}"]
        if archived:
            bits[0] += (f" kept: {hot} active, {archived} archived behind {summaries} "
                        f"summar{'ies' if summaries != 1 else 'y'}")
        if pins:
            bits.append(f"{pins} pinned")
        bits.append("full search" if self._rank_mode("x") == "hybrid" else "keyword search only")
        r, w = self._session["reads"], self._session["writes"]
        if r or w:
            bits.append(f"{r} recall{'s' if r != 1 else ''} / {w} save{'s' if w != 1 else ''} this session")
        if self._embed_failed:
            bits.append("embedder unavailable")
        return " · ".join(bits)

    def _policy_age(self) -> dict:
        """The role this scope was configured with, and how long ago (field report 2026-10-02: "the
        persona went out of date without warning", still naming a project that had been renamed).
        A config written before this was recorded has no date and reports none rather than a guess."""
        if not self.importance_policy:
            return {}
        out = {"persona": (self.importance_policy.get("persona") or "")[:200]}
        when = getattr(self, "_policy_set_at", None)
        if not when:                                    # configs written before the date was kept:
            import os                                   # the sidecar's last write is the best clue
            from datetime import datetime, timezone
            cp = self._config_path()
            if cp and os.path.exists(cp):
                when = datetime.fromtimestamp(os.path.getmtime(cp), timezone.utc).isoformat()
        if when:
            from datetime import datetime, timezone
            try:
                then = datetime.fromisoformat(when.replace("Z", "+00:00"))
                if then.tzinfo is None:
                    then = then.replace(tzinfo=timezone.utc)
                out["policy_set"] = when[:10]
                out["policy_age_days"] = (datetime.now(timezone.utc) - then).days
            except ValueError:
                pass
        return out

    def _role_check(self) -> dict:
        """A one-line prompt to re-read the role, only once it is old or of unknown age."""
        age = self._policy_age()
        if not age:
            return {}
        days = age.get("policy_age_days")
        if days is not None and days < _POLICY_STALE_DAYS:
            return {}
        since = f"set {days} days ago" if days is not None else "of unknown age"
        return {"role_check": (f"This scope's role is {since}: \"{age['persona']}\". If the project, "
                               "its names or the work have changed, update it with "
                               "configure(importance_policy={...}); setting it again, even unchanged, "
                               "clears this note.")}

    def _first_run(self) -> bool:
        return not self.importance_policy and not self._hot()

    def _outstanding(self) -> list:
        """What this scope needs done RIGHT NOW, as {when, do, why} — the ONE source of truth behind
        guidance()'s action_required, before_you_start(), and the first-call orientation. Each item is
        present only while it is outstanding, so the list never becomes a banner agents learn to
        ignore. Order is by what got skipped most in the field: the policy first."""
        out = []
        if not self.importance_policy:
            out.append(
                {"when": "NOW — nothing is set", "do": "configure(importance_policy={persona, "
                 "boost_kinds, boost_keywords}) for your role",
                 "why": "with no policy memory cannot weight anything for your job; this is yours to "
                        "set in guided mode too"})
        if not any(x.title == _CONTEXT_TITLE and not x.archived
                   for x in self.store.all_signals()):
            out.append(
                {"when": "NOW — you are anonymous here",
                 "do": 'set_context_card(agent="<a name>", identity=..., capabilities=[...])',
                 "why": "so you can introduce yourself and peers can route work to you"})
        # Only on a FIRST RUN (no policy, no cards): a project with no notes file could never clear
        # "memory is empty", and an item that cannot be cleared becomes a banner agents learn to
        # ignore. Setting the policy ends the first run and retires this nudge with it.
        if self._first_run():
            out.append(
                {"when": "NOW — memory is empty for this scope",
                 "do": "import_notes(path) for the project's instruction file (e.g. CLAUDE.md) and "
                       "any learned-rules file your host keeps (on Claude Code: "
                       "~/.claude/insights/learned-rules.md); preview with apply=false first",
                 "why": "an empty memory cannot help in the first session, and the project's standing "
                        "rules already exist in a file — a one-time import makes it useful from the "
                        "start"})
        return out

    def action_required(self) -> list:
        """The unmissable to-do list: plain instructions, FIRST RUN guidance first, empty when nothing
        is outstanding. Field evidence (twice): an agent read guidance() and still skipped setup when
        the instruction was one entry among many, or was surrounded by 30KB of manual. So this is
        short, top-level, and repeated on every surface an agent actually reaches for."""
        items = [i["do"] for i in self._outstanding()]
        if self._first_run():
            # The dials are invisible unless someone explains them, so hand the agent the setup
            # conversation rather than letting it guess the questions — as a menu, not a script.
            items.insert(
                0, "FIRST RUN: infer a sensible setup from what you can see (the project, the tools "
                   "in use, what the user asked for), apply it, and tell the user in ONE line what "
                   "you chose and that they can change any of it. Ask only what you genuinely "
                   "cannot infer. first_run_questions (from guidance()) are the menu of choices, not "
                   "a script to read out.")
        return items

    def guidance(self) -> dict:
        """'What can I adjust, and how?' — answered live, on demand. The manual ships once on connect;
        this is the surface for when the USER asks the running agent "what can you do / how do I tune
        your memory?" mid-session. Returns the current editable policy, what's locked (and why), the
        mode, and a plain-English map of what the user can SAY to change each thing. Read-only."""
        soft = sorted(self._locked - self._hard_locked)
        hard = sorted(self._hard_locked)
        current = {"importance_policy": self.importance_policy or None,
                   "decay_half_life": self.decay_half_life, "hot_cap": self.hot_cap,
                   "cards_per_summary": self.cards_per_summary, "pin_threshold": self.pin_threshold,
                   "dedup_threshold": self.dedup_threshold,
                   "lex_weight": self.lex_weight, "entity_weight": self.entity_weight,
                   "recency_weight": self.recency_weight, "trust_weight": self.trust_weight,
                   "retire_after_failures": self.retire_after_failures, "wiki_link": self.wiki_link}
        can_ask = [
            {"say": "you're my <role>; prioritise <X>",
             "changes": "what memory keeps + pins first (importance_policy)", "works": "either mode, at once"},
            {"say": "remember this / pin this / that's important",
             "changes": "stores + pins a card", "works": "either mode"},
            {"say": "what do you remember about X?", "changes": "browses stored memory (nothing changes)",
             "works": "either mode"},
            {"say": "forget that / delete X", "changes": "permanently removes a card", "works": "either mode"},
            {"say": "that shouldn't be pinned / unpin X",
             "changes": "takes a card off the pin set without deleting it (unpin)", "works": "either mode"},
            {"say": "forget old things faster / keep them longer",
             "changes": "how fast unused memory fades (decay_half_life)",
             "works": "auto: freely; guided: on your explicit say-so"},
            {"say": "hold more before you summarise",
             "changes": "hot-tier capacity (hot_cap)", "works": "auto: freely; guided: on your say-so"},
            {"say": "stop merging similar notes / merge only near-identical ones",
             "changes": "how aggressively duplicates collapse (dedup_threshold — raise it to keep more)",
             "works": "auto: freely; guided: on your say-so"},
            {"say": "weight recent / exact-term matches more",
             "changes": "retrieval weights", "works": "auto: freely; guided: on your say-so"},
            {"say": "manage your own memory / stop checking with me",
             "changes": "autonomy -> auto (full self-management)", "works": "unless autonomy is hard-locked"},
            {"say": "lock these settings / don't change X",
             "changes": "freezes dials so they can't drift (lock)", "works": "either mode"},
        ]
        # can_ask is written for the USER ("what can I say?"). you_should is the same phrasebook for
        # the AGENT ("what should I do, and when?") — the reader who actually calls these tools. It
        # leads with whatever is outstanding RIGHT NOW, so a host that truncated the instructions is
        # fully recoverable: an agent that reaches guidance() can learn the whole protocol from here.
        you_should = list(self._outstanding())
        you_should += [
            {"when": "you pick up a task, ticket or question — before you plan anything",
             "do": 'before_you_start("<what you are about to do>")',
             "why": "returns the standing rules AND what memory holds for this task in one call, "
                    "without you having to invent a search query"},
            {"when": "a store result shows 'imbalance' (stores with no recalls)",
             "do": "recall(...) before your next answer",
             "why": "writing without reading is the same as having no memory at all"},
            {"when": "recall returns matched:false",
             "do": "say you don't have it, then store the answer once you learn it",
             "why": "the cards shown are only the closest — answering from them is how an agent "
                    "learns to distrust its own memory"},
            {"when": "before choosing an approach, tool, library or command", "do": "recall(query)",
             "why": "the choice may already be decided and stored"},
            {"when": "before asserting what the user wants, prefers or uses", "do": "recall(query)",
             "why": "going only on the live chat is how a stored preference gets contradicted"},
            {"when": "before repeating or acting on a past decision", "do": "recall(query)",
             "why": "decisions come back flagged when they may have gone stale"},
            {"when": "starting work in a scope you have worked in before",
             "do": 'recall("standing decisions rules preferences"), list(pinned=true) for the '
                   "standing staples, and context_cards(agent=<you>) for what you were last on",
             "why": "orient on the standing decisions and your own last focus before the first real "
                    "answer, instead of rediscovering them mid-task"},
            {"when": "the user states a durable fact, decision, command or preference",
             "do": "store_signals([...]) with volatility set", "why": "this is the write half"},
            {"when": "a stored value changed", "do": 'store the new fact with "supersedes":[old_id]',
             "why": "keeps one living card per fact; the old one is kept as [older]"},
            {"when": "you acted on a remembered how-to",
             "do": 'procedure_outcome(ids, "worked"|"failed")',
             "why": "how a procedure earns or loses trust — otherwise stale steps keep being served"},
            {"when": "store_signals returns please_summarise", "do": "store_summary(text, source_ids)",
             "why": "ignoring it grows the hot tier and makes every later recall more expensive"},
            {"when": "you set memory up, recall from it, or save to it",
             "do": "tell the user in ONE short line — stats()['summary'] is written for this",
             "why": "they cannot see your tool calls, so silence looks identical to memory not "
                    "working; this is how 'my agents aren't using it' went unnoticed"},
            {"when": "scoping a task, or onboarding", "do": "knowledge_gaps()",
             "why": "retrieval can never find a fact nobody gave you"},
        ]
        # Anything urgent goes at the TOP LEVEL, not buried in a list. Field evidence: an agent read
        # guidance(), named itself, and STILL never set an importance_policy — the very gap this all
        # started with. you_should[0] said to; it was one entry among twelve. action_required is the
        # unmissable version, present only when something actually needs doing.
        action_required = self.action_required()
        first_run = self._first_run()
        return {"autonomy": self.autonomy, "current": current,
                **({"action_required": action_required} if action_required else {}),
                **({"first_run_approach": _FIRST_RUN_APPROACH,
                    "first_run_questions": self._first_run_questions()} if first_run else {}),
                "locked": ({"soft": soft, "hard": hard} if (soft or hard) else None),
                "yours_in_any_mode": ["importance_policy", "store", "pin", "unpin", "supersede",
                                      "recall", "summarise", "delete", "scope", "context/id cards"],
                "locked_in_guided": (sorted(self._GUIDED_LOCKED) if self.autonomy == "guided" else []),
                "can_ask": can_ask,
                "you_should": you_should,
                "note": ("\"Guided\" locks the LIFECYCLE DIALS ONLY (listed in locked_in_guided) so "
                         "they cannot drift. It does NOT lock policy: everything in yours_in_any_mode "
                         "is yours to set in guided mode too — you own memory, not just its cards. A "
                         "user asking EXPLICITLY for a locked dial gets it (user_requested=true) "
                         "without leaving guided. Hard-locked dials need the SIGNAL_MEMORY_LOCK env + "
                         "a restart — no agent can move them.")}

    def knowledge_gaps(self, slots: Optional[dict] = None) -> dict:
        """What does memory NOT know that it probably SHOULD? (STO-2850.)

        The most valuable thing memory can surface is often not a better ranking of what it HAS, but a
        gap in what it has at all — "you have no card about how this demo gets delivered". Retrieval can
        never find a fact that was never stored; this attacks that unknown-unknown directly.

        Each scope has a set of question-shaped ``slots`` (audience / deliverable / deadline / success
        criteria / constraints by default; pass ``slots={name: [trigger terms]}`` to set a role-specific
        schema, persisted per scope). A slot is FILLED if a current fact card mentions any of its trigger
        terms (substring over title/content/keywords) — a deliberately simple first cut, not a semantic
        classifier. Returns which slots are filled (with evidence card ids) and which are still EMPTY,
        so the agent can go and ASK. Read-only unless ``slots`` is given."""
        if slots is not None:
            self.configure({"knowledge_slots": slots})       # set + persist the schema
        schema = self.knowledge_slots or _DEFAULT_KNOWLEDGE_SLOTS
        facts = [s for s in self._hot()
                 if s.is_fact and s.is_current and s.title not in (_SUMMARY_TITLE, _CONTEXT_TITLE)]
        blobs = [(s.signal_id, " ".join([s.title or "", s.content or "",
                                         " ".join(s.keywords or [])]).lower()) for s in facts]
        report, missing, filled = {}, [], []
        for name, terms in schema.items():
            terms = [t.lower() for t in (terms or [])]
            evidence = [sid for sid, blob in blobs if any(t in blob for t in terms)]
            is_filled = bool(evidence)
            report[name] = {"filled": is_filled, "evidence": evidence[:3]}
            (filled if is_filled else missing).append(name)
        return {"slots": report, "missing": missing, "filled": filled,
                "using_default_schema": self.knowledge_slots is None,
                "instruction": ("For each 'missing' slot, ASK the user (or go find it) — memory can't "
                                "rank a fact it was never given. Set a role-specific schema by passing "
                                "slots={name:[terms]}." if missing else
                                "Every expected slot has at least one card.")}

    # -- agent-owned policy: set decay / pins / importance / retrieval at runtime -------------

    def _config_path(self) -> Optional[str]:
        p = getattr(self.store, "path", None)
        return (p + ".config.json") if p else None

    _LOCKABLE = set(_CONFIGURABLE) | {"wiki_link", "importance_policy"}
    # In "guided" mode these mechanical/lifecycle dials are locked by default (importance_policy is NOT
    # — the agent always declares its role). "auto" mode unlocks them for full self-management.
    _GUIDED_LOCKED = set(_CONFIGURABLE) | {"wiki_link"}

    @classmethod
    def _parse_lock(cls, val) -> set:
        """SIGNAL_MEMORY_LOCK env -> the HARD-locked keys ("all", or a comma list). These the agent
        can never change or unlock — the user's authority floor. "autonomy" is also lockable."""
        if not val:
            return set()
        keys = cls._LOCKABLE | {"autonomy"}
        if str(val).strip().lower() == "all":
            return set(keys)
        return {k.strip() for k in str(val).split(",") if k.strip() in keys}

    def _apply_config(self, settings: dict, persist: bool, force: bool = False) -> dict:
        settings = settings or {}
        # THE USER OUTRANKS THE MODE. `user_requested` means the human explicitly asked for THIS change
        # (the agent asserts it — the server can't see the conversation, so this is a trust boundary,
        # not enforcement). It lifts the guided-mode lock and a user's own soft lock — but NEVER a hard
        # env lock, which is the one control the server refuses on any pretext. Per-call intent only:
        # not an attribute, never persisted, so a peer/restart inherits no standing "always allow" bit.
        user_req = bool(settings.get("user_requested"))
        # An agent may lift a SOFT lock (via unlock); a HARD (env) lock is never liftable.
        self._locked -= (set(settings.get("unlock") or []) - self._hard_locked)
        applied, rejected, overridden, clamped = {}, [], [], {}

        def take(key) -> bool:
            if force:
                return True
            if key in self._hard_locked:
                rejected.append(key)                        # env hard lock -> refused, even for the user
                return False
            blocked = (key in self._locked) or (self.autonomy == "guided"
                                                and key in self._GUIDED_LOCKED)
            if blocked:
                if user_req:                                # explicit user authority overrides the lock
                    overridden.append(key)
                    return True
                rejected.append(key)                        # agent drift into a locked/guided dial
                return False
            return True

        # Autonomy is the mode gate — set it FIRST so a same-call switch to "auto" unlocks the dials.
        if "autonomy" in settings:
            if take("autonomy") and str(settings["autonomy"]).lower() in ("guided", "auto"):
                self.autonomy = str(settings["autonomy"]).lower()
                applied["autonomy"] = self.autonomy

        for key, caster in _CONFIGURABLE.items():
            if settings.get(key) is None or not take(key):
                continue
            try:
                requested = caster(settings[key])
            except (TypeError, ValueError):
                continue
            val = _clamp_dial(key, requested)               # keep dials in a sane range — a bad value
            if val != requested:                            # (e.g. cards_per_summary=0) must not break
                clamped[key] = {"requested": requested, "applied": val}   # tell the caller we changed it
            setattr(self, key, val)                         # the engine or crash a later op
            applied[key] = val
        if "decay_half_life" in applied:
            self._decay.half_life = applied["decay_half_life"]   # keep the decay model in sync
        if "wiki_link" in settings and take("wiki_link"):
            self.wiki_link = bool(settings["wiki_link"])
            applied["wiki_link"] = self.wiki_link
        if isinstance(settings.get("importance_policy"), dict) and take("importance_policy"):
            self.importance_policy = dict(settings["importance_policy"])
            applied["importance_policy"] = self.importance_policy
            if persist:                                     # a real configure, not a reload
                self._policy_set_at = _now_iso()
        if isinstance(settings.get("knowledge_slots"), dict) and take("knowledge_slots"):
            # {slot_name: [trigger terms]} — normalise to lists of lower-cased strings (STO-2850)
            self.knowledge_slots = {str(k): [str(t).lower() for t in (v or [])]
                                    for k, v in settings["knowledge_slots"].items()}
            applied["knowledge_slots"] = self.knowledge_slots
        lock = settings.get("lock")                          # freeze keys (soft — user-removable)
        if lock is True:
            self._locked |= set(self._LOCKABLE)
        elif lock:
            self._locked |= {k for k in lock if k in self._LOCKABLE}
        if persist:
            self._persist_config()
        return {"applied": applied, "rejected_locked": rejected, "user_overridden": overridden,
                "clamped": clamped}

    def configure(self, settings: dict) -> dict:
        """Set memory POLICY at runtime — decay speed, capacity, pin rules, retrieval weights,
        wiki_link, and an ``importance_policy`` (what this agent's ROLE weights up). Recomputed live
        (effective importance is never frozen), persisted per scope (a restart / a peer inherits it).
        The USER can freeze dials: pass ``lock`` (list of keys, or ``true`` for all) so they can't be
        drifted, and ``unlock`` to lift a soft lock. SIGNAL_MEMORY_LOCK (env) hard-locks keys the agent
        can NEVER change. Returns applied settings, any refused (locked), the current locked set, and
        ``clamped`` — any dial whose out-of-range value was pulled into a sane range (requested vs
        applied), so the caller never silently believes a value it didn't get."""
        res = self._apply_config(settings or {}, persist=True)
        return {"configured": res["applied"], "rejected_locked": res["rejected_locked"],
                "user_overridden": res["user_overridden"], "clamped": res["clamped"],
                "locked": sorted(self._locked | self._hard_locked), "autonomy": self.autonomy,
                "path": getattr(self.store, "path", None)}

    def _persist_config(self) -> None:
        import json
        import os
        cp = self._config_path()
        if not cp:
            return
        cfg = {k: getattr(self, k) for k in _CONFIGURABLE}
        cfg["wiki_link"] = self.wiki_link
        cfg["importance_policy"] = self.importance_policy
        if self.knowledge_slots is not None:
            cfg["knowledge_slots"] = self.knowledge_slots    # STO-2850 (omit when default -> unset)
        cfg["autonomy"] = self.autonomy
        cfg["_locked"] = sorted(self._locked)               # locks + mode travel with the scope
        if getattr(self, "_policy_set_at", None):
            cfg["_policy_set_at"] = self._policy_set_at
        os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
        tmp = cp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f)
        os.replace(tmp, cp)

    def _load_config(self) -> None:
        import json
        import os
        cp = self._config_path()
        if cp and os.path.exists(cp):
            try:
                cfg = json.load(open(cp))
            except Exception:
                return
            locked = set(cfg.get("_locked") or [])
            self._apply_config(cfg, persist=False, force=True)   # apply saved values, bypass locks
            self._policy_set_at = cfg.get("_policy_set_at")
            self._locked = locked - self._hard_locked

    def _touch(self, sigs) -> None:
        """Record a retrieval — feeds the usage boost so used cards resist decay, and stamps last_used
        (recency, for staleness/cleanup). last_used persists on the next write."""
        now = _now_iso()
        for s in sigs:
            sid = getattr(s, "signal_id", None)
            if sid:
                self._access[sid] = self._access.get(sid, 0) + 1
                s.last_used = now

    def overflow_batches(self, cap: Optional[int] = None) -> List[List[Signal]]:
        """Which cards to fold into a summary when the hot tier is over ``cap``, grouped into
        ``cards_per_summary`` batches. **Capacity decides WHEN; half-life-decayed importance decides
        WHICH** — we fold the LOWEST time-decayed effective importance first (age = position from
        newest), so old-unused-unimportant cards sink and fold while recent/used/behavioural ones
        stay hot. Pins are NEVER folded. Folded originals are archived (kept, drillable), not deleted."""
        cap = cap if cap is not None else self.hot_cap
        hot = self._hot()                                  # oldest first (store insertion order)
        # Fold only once a FULL batch is over cap, and fold in WHOLE ``cards_per_summary`` batches.
        # Incremental storage (one card at a time) otherwise overflows by 1 each store, folding a
        # single card into its own summary -> 1:1, no consolidation, and the summary index grows
        # unbounded over a long-running session. Waiting for a full batch gives real ~N:1 rollups.
        cps = max(1, int(self.cards_per_summary or 1))     # defensive: never divide by a 0/None batch
        over = len(hot) - cap
        if over < cps:
            return []
        last = len(hot) - 1
        fresh = self._fresh_ids(hot, self.fold_min_age_hours)
        non_pinned = [(self.effective_importance(s, age=last - i), i, s)
                      for i, s in enumerate(hot) if not self.is_pinned(s) and s.signal_id not in fresh]
        non_pinned.sort(key=lambda t: (t[0], -t[1]))       # lowest importance first; ties -> older first
        n_fold = (min(len(non_pinned), over) // cps) * cps
        if n_fold == 0:                                    # over cap only because of pins -> nothing to fold
            return []
        fold = [s for _, _, s in non_pinned[:n_fold]]
        return [fold[i:i + cps] for i in range(0, len(fold), cps)]

    @staticmethod
    def _fresh_ids(cards, hours: float) -> set:
        """Ids of cards written within ``hours``. A card with no readable timestamp counts as old
        (cards from before timestamps existed must still be foldable)."""
        from datetime import datetime, timedelta, timezone
        if not hours:
            return set()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        out = set()
        for s in cards:
            raw = getattr(s, "created_at", "") or ""
            try:
                when = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            if when > cutoff:
                out.add(s.signal_id)
        return out

    def recall_context(self, query: str, drill_summaries: int = 6, drill_pool_cap: int = 24,
                       summary_context: int = 3) -> List[Signal]:
        """Lifecycle-aware retrieval: narrowed hot cards + the archived cards behind the relevant
        summary index cards, drilled back. The archived cards keep the ORIGINAL data verbatim (a
        summary only *indexes* them via source_ids — it never replaces them), so the exact figure a
        precise question needs is always recoverable.

        Drill-back: take the top ``drill_summaries`` relevant summaries, pool ALL their archived
        cards, embed-rank that pool and keep the best ``drill_pool_cap`` — so the one card an exact
        question needs surfaces even if its summary wasn't rank-1. Only the top ``summary_context``
        summary texts are added to the context (routing needs breadth; the answer needs the cards),
        keeping tokens in budget.

        (In live use the agent reads the summary manifests and drills the ones it wants — unit-tested
        separately; here drilling is by local-embedding similarity, a faithful stand-in for a batch
        run so no agent-in-the-loop is needed per question.)"""
        picked, seen = [], set()

        def add(sigs):
            for s in sigs:
                if s.signal_id not in seen:
                    seen.add(s.signal_id)
                    picked.append(s)

        add(self._narrow(query, self._hot()))              # top hot cards (dynamic k)
        summaries = self._summaries()
        if summaries and query:            # ranking no longer needs an embedder, so neither does drill-back
            by_id = {s.signal_id: s for s in self.store.all_signals()}
            top = self._rank(query, summaries)[:drill_summaries]
            pool = [by_id[i] for summ in top for i in (summ.source_ids or []) if i in by_id]
            add(self._rank(query, pool)[:drill_pool_cap] if pool else [])   # best archived, verbatim
            add(top[:summary_context])                      # a few summary index cards for overview
        if wants_ordering(query):
            picked = sorted(picked, key=self._time_key)     # reconstruct the timeline for the answer
        self._touch(picked)                                 # retrieval boosts importance (usage)
        return picked
