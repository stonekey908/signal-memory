"""Resolve WHICH memory a session uses — its *scope* (the thing being worked on).

Scope is what stops an agent getting confused across tasks, and what lets a swarm share one brain.
Resolution, highest priority first:

  1. ``SIGNAL_MEMORY_PATH``  — an explicit file path. Full manual control.
  2. ``SIGNAL_MEMORY_SCOPE`` — a NAMED scope -> a shared central file at
     ``~/.signal_engine/scopes/<scope>.json``. Every agent that sets the SAME name shares ONE
     memory: this is how a swarm working on one thing shares a brain (and how one agent keeps
     separate tasks apart — different names -> different memories).
  3. default (no env) — PER-FOLDER: ``<cwd>/.signal_engine/memory.json`` in the working folder.

Why the folder is the default: it is **deterministic**. The folder *is* the key, so the same project
always resolves to the same memory with zero chance of two slightly-different ids ("payments" vs
"payment-refactor") fragmenting what is really one thing. A named scope is the deliberate override
where you accept naming it — used to share across folders or split tasks within one. Claude Code
launches the MCP server in the project directory, so the per-folder default "just works".
"""

from __future__ import annotations

import os
import re


def _safe(name: str) -> str:
    """Canonicalise a scope name into a safe, stable filename fragment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", (name or "").strip().lower()).strip("-") or "default"


def scopes_dir() -> str:
    return os.path.expanduser(os.path.join("~", ".signal_engine", "scopes"))


def resolve_memory_path(explicit: str = None, scope: str = None, cwd: str = None) -> str:
    """Resolve the memory file path from (in priority) an explicit path, a named scope, or the
    working folder. Passed args override the matching env vars; used by both the MCP server and the
    Python ``Memory`` API so they agree on where a scope lives."""
    # A passed ARG is an explicit runtime intent and beats the env DEFAULTS — otherwise a runtime
    # use_scope(scope=...) would be a silent no-op whenever SIGNAL_MEMORY_PATH is set (the agent
    # thinks it switched but didn't). Env vars only fill in when NO arg was given (initial connect).
    if explicit:
        return os.path.expanduser(explicit)
    if scope:
        return os.path.join(scopes_dir(), _safe(scope) + ".json")
    env_path = os.getenv("SIGNAL_MEMORY_PATH")
    if env_path:
        return os.path.expanduser(env_path)
    env_scope = os.getenv("SIGNAL_MEMORY_SCOPE")
    if env_scope:
        return os.path.join(scopes_dir(), _safe(env_scope) + ".json")
    return os.path.join(cwd or os.getcwd(), ".signal_engine", "memory.json")


def active_scope(scope: str = None, cwd: str = None) -> str:
    """A short human label for the current scope (for stats / telling the user where memory lives)."""
    if os.getenv("SIGNAL_MEMORY_PATH"):
        return "path:" + os.path.basename(os.getenv("SIGNAL_MEMORY_PATH"))
    scope = scope or os.getenv("SIGNAL_MEMORY_SCOPE")
    if scope:
        return "scope:" + _safe(scope)
    base = os.path.basename((cwd or os.getcwd()).rstrip(os.sep)) or "root"
    return "folder:" + base
