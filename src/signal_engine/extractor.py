"""Signal Extractor (write path) — turns a conversation into Signals.

Per ``docs/DECISIONS.md`` each session yields several fine-grained **fact** Signals + one
session-level **behavioural** Signal. The scribbler LLM is **pluggable** (``MockScribbler``
for offline tests; ``OpenAIScribbler`` for real runs; Gemini/local plug in via the same
``Scribbler`` interface). Parsing is a pure function, so the logic is fully testable
without a key. Extraction tokens are logged per session (feeds tokens/query, STO-2733).

Out of scope: salience filtering (STO-2721), storage, embeddings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import signal_engine.config  # noqa: F401  — importing loads .env (API keys) via load_dotenv
from signal_engine.openai_client import openai_client
from signal_engine.signal import Signal, behavioural_signal, fact_signal

_DEFAULT_IMPORTANCE = 0.5
_EMPTY_RETRIES = 3    # gpt-4o occasionally returns empty content (finish=stop); retry before giving up
_FENCE = re.compile(r"^```(?:json)?|```$", re.MULTILINE)
_BEHAVIOURAL_META = ("sentiment", "friction", "tool_choice", "interaction")

EXTRACTION_PROMPT = """You extract durable memories from a conversation between a user and an assistant.

Return ONLY a JSON object of this exact shape:
{{
  "facts": [
    {{"title": "...", "description": "...", "keywords": ["..."], "content": "the durable fact or stated preference", "importance": 0.0}}
  ],
  "behavioural": {{"title": "...", "description": "...", "sentiment": "...", "friction": "...", "tool_choice": "...", "interaction": "...", "importance": 0.0}}
}}

Rules:
- Put EACH distinct durable fact or stated preference as its own item in "facts" (be fine-grained).
- When a fact involves an event with a stated date, time, or deadline, INCLUDE that date in the
  fact's "content" (e.g. "The user obtained the API key on March 10, 2024", "testing starts April 5").
  Dates stated in the conversation are part of the fact.
- Record explicit denials and negative claims as facts too (e.g. "The user says they have never
  written any Flask routes") — what someone says they have NOT done is durable memory.
- "behavioural" is ONE overall read of how the conversation went (mood, friction, tool choices, user pushback). Use null for fields that don't apply.
- "importance" is a number 0..1 (how worth remembering).
- If nothing durable was said, return "facts": [] and set the behavioural fields to null.

Conversation:
{conversation}
"""


@runtime_checkable
class Scribbler(Protocol):
    def complete(self, prompt: str) -> Tuple[str, int]:
        """Return (raw_text, tokens_used)."""
        ...


@dataclass
class ExtractionResult:
    signals: List[Signal]
    tokens: int


def _clamp_importance(v: object) -> float:
    try:
        x = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return _DEFAULT_IMPORTANCE
    return max(0.0, min(1.0, x))


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text or "").strip()


def _extract_json_object(text: str) -> str:
    """Salvage the outermost {...} JSON object from a reply that may be wrapped in prose or
    code fences. Chatty models (e.g. Haiku via the OpenAI-compat endpoint, which doesn't
    enforce response_format) sometimes preface JSON with an essay; find the first balanced
    brace-span rather than crashing the whole (parallel) extraction on it."""
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
    return ""


def parse_extraction(raw: str, *, session_ref: Optional[str], created_at: str) -> List[Signal]:
    """Pure parse: scribbler JSON -> validated Signals (fact-signals + <=1 behavioural)."""
    text = _strip_fences(raw)
    if not text:                    # scribbler returned nothing durable -> no signals (don't crash)
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        salvaged = _extract_json_object(text)   # model wrapped JSON in prose -> recover it
        if not salvaged:
            return []               # genuinely no JSON object -> no signals (never crash the run)
        data = json.loads(salvaged)
    signals: List[Signal] = []

    for f in data.get("facts") or []:
        content = (f.get("content") or "").strip()
        if not content:
            continue
        signals.append(
            fact_signal(
                title=(f.get("title") or content)[:80],
                content=content,
                description=f.get("description") or "",
                keywords=list(f.get("keywords") or []),
                session_ref=session_ref,
                created_at=created_at,
                importance_base=_clamp_importance(f.get("importance")),
            )
        )

    b = data.get("behavioural") or {}
    meta = {k: (b.get(k) or None) for k in _BEHAVIOURAL_META}
    if any(meta.values()):
        signals.append(
            behavioural_signal(
                title=b.get("title") or "conversation read",
                description=b.get("description") or "",
                keywords=list(b.get("keywords") or []),
                session_ref=session_ref,
                created_at=created_at,
                importance_base=_clamp_importance(b.get("importance")),
                **meta,
            )
        )
    return signals


def _format_conversation(turns) -> str:
    return "\n".join(f"{t.role}: {t.content}" for t in turns)


class Extractor:
    """Turns a session into Signals via a pluggable scribbler.

    ``chunk_turns=None`` (default) extracts from the whole session in one call.
    Setting ``chunk_turns=N`` splits the turns into N-turn windows (finer granularity,
    more calls); facts from all windows are merged and one behavioural signal is kept.
    """

    def __init__(self, scribbler: Scribbler, chunk_turns: Optional[int] = None, prompt: str = None):
        self.scribbler = scribbler
        self.chunk_turns = chunk_turns
        self.prompt = prompt or EXTRACTION_PROMPT   # product paths can override (benchmark unchanged)

    def extract(self, session) -> ExtractionResult:
        turns = list(getattr(session, "turns", []))
        session_ref = getattr(session, "session_id", None)
        created_at = getattr(session, "date", "") or ""

        if self.chunk_turns:
            windows = [turns[i : i + self.chunk_turns] for i in range(0, len(turns), self.chunk_turns)]
        else:
            windows = [turns]

        facts: List[Signal] = []
        behavioural: Optional[Signal] = None
        total_tokens = 0
        for win in windows:
            if not win:
                continue
            prompt = self.prompt.format(conversation=_format_conversation(win))
            raw, tokens = self.scribbler.complete(prompt)
            total_tokens += tokens
            for s in parse_extraction(raw, session_ref=session_ref, created_at=created_at):
                if s.is_behavioural:
                    if behavioural is None:      # one behavioural signal per session
                        behavioural = s
                else:
                    facts.append(s)

        signals = facts + ([behavioural] if behavioural else [])
        return ExtractionResult(signals=signals, tokens=total_tokens)


# --- Scribblers -------------------------------------------------------------

_DEFAULT_MOCK_RESPONSE = json.dumps(
    {
        "facts": [
            {"title": "user's dog", "description": "pet", "keywords": ["dog", "pet"],
             "content": "The user's dog is named Rex.", "importance": 0.7},
            {"title": "seat preference", "description": "travel", "keywords": ["travel"],
             "content": "The user prefers window seats.", "importance": 0.5},
        ],
        "behavioural": {"title": "friendly chat", "description": "", "sentiment": "positive",
                        "friction": None, "tool_choice": None, "interaction": "user was chatty",
                        "importance": 0.3},
    }
)


class MockScribbler:
    """No-key scribbler: returns a fixed JSON response (offline tests). Configurable."""

    name = "mock"

    def __init__(self, response: str = _DEFAULT_MOCK_RESPONSE):
        self.response = response

    def complete(self, prompt: str) -> Tuple[str, int]:
        tokens = len(prompt.split()) + len(self.response.split())   # rough offline estimate
        return self.response, tokens


def _is_reasoning_model(model: str) -> bool:
    """Reasoning-family models (gpt-5*, o1/o3/o4*) only allow the default temperature."""
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


class OpenAIScribbler:
    """Real OpenAI scribbler. Requires OPENAI_API_KEY.

    Default model is ``gpt-4o`` because ``gpt-4o-mini`` (the planned headline scribbler) is
    not available on the current project — see docs/BASELINE.md. Swap freely.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None,
                 client_factory=openai_client):
        self.model = model
        self._api_key = api_key   # resolved at call time by the shared client factory
        self._client_factory = client_factory   # openai_client, or an Ollama-compatible factory

    def complete(self, prompt: str) -> Tuple[str, int]:
        client = self._client_factory(self._api_key)   # shared client: timeout + retry, no leak
        kwargs = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        if not _is_reasoning_model(self.model):
            kwargs["temperature"] = 0          # reasoning models only allow the default
        # gpt-4o intermittently returns EMPTY content (finish_reason=stop) on large inputs —
        # observed ~1/3 of BEAM sessions. It's non-deterministic, so retry a few times before
        # giving up (losing a whole session's facts wrecks recall). Tokens accrue honestly.
        text, tokens = "", 0
        for _ in range(_EMPTY_RETRIES + 1):
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            tokens += resp.usage.total_tokens if resp.usage else 0
            if text.strip():
                break
        return text, tokens


class ClaudeScribbler:
    """Claude scribbler via Anthropic's OpenAI-compatible endpoint (NFR-2 second backend).

    Defaults to the DATED Haiku 4.5 snapshot (reproducibility bonus: no rolling alias).
    Differences from ``OpenAIScribbler``: no ``response_format`` (the compat layer may ignore
    it; ``parse_extraction`` already strips code fences) and an explicit ``max_tokens``
    (required by Anthropic). Same empty-response retry discipline.
    """

    name = "claude"

    # The compat endpoint doesn't enforce response_format, and on code-heavy chunks Haiku will
    # "continue the conversation" (write an essay/code) instead of extracting — a hard system
    # instruction pins it to JSON-only. parse_extraction still salvages prose-wrapped JSON as a
    # backstop.
    _SYSTEM = ("You are a memory-extraction function. You output ONLY a single JSON object and "
               "nothing else — no prose, no explanation, no code blocks. Never continue or answer "
               "the conversation you are given; only extract durable memories from it as JSON.")

    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: Optional[str] = None,
                 max_tokens: int = 4096):
        self.model = model
        self._api_key = api_key   # resolved at call time by the shared anthropic_client()
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> Tuple[str, int]:
        from signal_engine.openai_client import anthropic_client

        client = anthropic_client(self._api_key)
        text, tokens = "", 0
        for _ in range(_EMPTY_RETRIES + 1):
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": self._SYSTEM},
                          {"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            tokens += resp.usage.total_tokens if resp.usage else 0
            if text.strip():
                break
        return text, tokens


# A Gemini scribbler plugs in via the same `Scribbler` interface — added when we pick the
# real model (Nick chose "decide later"). MockScribbler + OpenAIScribbler cover build + tests.
