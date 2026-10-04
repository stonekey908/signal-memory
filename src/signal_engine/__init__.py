"""Signal Engine — agent-level automated long-term memory engine.

Public entrypoints:
  - ``Memory`` — embed the local-first memory directly in a Python app/agent (no MCP host).
  - the ``signal-dumb-mcp`` server — drive it from any MCP host on your own model.

See ``docs/HOW_IT_WORKS.md`` and ``docs/SETUP_GUIDE.md``.
"""

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy so ``import signal_engine`` stays cheap and dependency-light; ``Memory`` pulls the
    # engine (and, if used, the local embedder) only when actually referenced.
    if name == "Memory":
        from signal_engine.memory_api import Memory
        return Memory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Memory", "__version__"]
