"""Central configuration for Signal Engine.

Holds the *frozen model recipe* (the pinned model choices for reproducible
benchmark runs) and helpers to load API keys from a local ``.env`` file. This is
settings only — no engine logic lives here.

Exact dated model version strings (e.g. ``gpt-4o-2024-08-06``) are filled in
once API keys are wired (see ``SETUP.md``); the values below name the models we
committed to in ``docs/DECISIONS.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load .env if present (a no-op when the file is missing). Real keys are never
# committed — see .gitignore and .env.example.
load_dotenv()


@dataclass(frozen=True)
class ModelRecipe:
    """A frozen model recipe for one benchmark run (see ``docs/DECISIONS.md``).

    Two configs share everything except the *scribbler* (the engine's own model
    that writes notes, filters junk, and re-ranks):

    * **headline** — scribbler = ``gpt-4o-mini`` (a small, inexpensive model, so
      results reflect the memory *design* rather than a large extractor).
    * **agnostic** — scribbler = a Gemini-flash / local model (proves the engine
      is model-agnostic — NFR-2).

    The reader + judge are pinned to ``gpt-4o`` by the benchmark rules, and the
    embedder to ``text-embedding-3-small`` so runs are comparable.
    """

    scribbler: str
    embedder: str = "text-embedding-3-small"
    reader: str = "gpt-4o"
    judge: str = "gpt-4o"
    rerank_temperature: float = 0.0


# The two frozen recipes we committed to. Exact dated snapshots get pinned here
# when keys are wired — see SETUP.md.
HEADLINE_RECIPE = ModelRecipe(scribbler="gpt-4o-mini")
AGNOSTIC_RECIPE = ModelRecipe(scribbler="gemini-flash")  # exact version TBD


@dataclass(frozen=True)
class Keys:
    """API keys loaded from the environment (never hard-coded)."""

    openai: Optional[str] = None
    google: Optional[str] = None

    @property
    def has_openai(self) -> bool:
        return bool(self.openai)

    @property
    def has_google(self) -> bool:
        return bool(self.google)


def load_keys() -> Keys:
    """Read API keys from the environment (``.env`` is loaded on import)."""
    return Keys(
        openai=os.getenv("OPENAI_API_KEY"),
        google=os.getenv("GOOGLE_API_KEY"),
    )
