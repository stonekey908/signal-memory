"""Wiki / index navigation layer (STO-2731).

A markdown index over stored signals so an agent (or a human) can navigate memory cheaply —
"where is X?" before pulling the detail. Pure formatting, no LLM; reinforces token efficiency.
"""

from __future__ import annotations

from typing import List

from signal_engine.signal import Signal


def build_wiki(signals: List[Signal]) -> str:
    facts = [s for s in signals if s.is_fact]
    behav = [s for s in signals if s.is_behavioural]
    out: List[str] = [
        f"# Memory index — {len(signals)} signals ({len(facts)} facts, {len(behav)} behavioural)",
        "",
    ]
    if facts:
        out.append("## Facts")
        for s in facts:
            pin = " 📌" if s.pinned else ""
            kw = f"  _[{', '.join(s.keywords)}]_" if s.keywords else ""
            out.append(f"- **{s.title}**{pin}: {(s.content or '').strip()}{kw}")
        out.append("")
    if behav:
        out.append("## Behavioural notes")
        for s in behav:
            out.append(f"- **{s.title}**")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
