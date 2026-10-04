"""Signal Engine as an MCP server (STO-2804) — persistent memory for Claude Code / any MCP host.

Exposes two tools over stdio:
    remember(text)  -> distil + persist a memory
    recall(query)   -> retrieve the most relevant memories

Run:  uv run signal-mcp        (needs `uv sync --extra llm --extra store`)
Register in Claude Code:  claude mcp add signal-engine -- uv run signal-mcp
(or add to .mcp.json — see SETUP.md). Requires OPENAI_API_KEY in the environment/.env.

DEPENDENCY-FREE: MCP's stdio transport is newline-delimited JSON-RPC 2.0, so this implements the
minimal handshake (initialize / tools/list / tools/call) directly rather than pulling the official
`mcp` SDK, which requires Python 3.10+ (the project is pinned to 3.9 for the benchmark). All memory
logic lives in MemoryService (testable without any protocol); this file is only the wire layer.
"""

from __future__ import annotations

import json
import sys
from typing import Callable, Optional

from signal_engine.memory_service import MemoryService

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "signal-engine", "version": "0.1.0"}

TOOLS = [
    {
        "name": "remember",
        "description": ("Store a durable memory. Pass anything worth recalling later — a "
                        "decision, a preference, a fact, a mistake to avoid. The engine distils "
                        "it into signals and persists them across sessions."),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The memory to store."}},
            "required": ["text"],
        },
    },
    {
        "name": "recall",
        "description": ("Retrieve relevant memories for a query. Returns the most relevant "
                        "stored memories, or a note that nothing relevant was found."),
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up."}},
            "required": ["query"],
        },
    },
]

_service: Optional[MemoryService] = None


def _default_service() -> MemoryService:
    """Build the real (persistent, OpenAI-backed) service once, on first tool call — so the
    server starts (and lists tools) even before a key is needed."""
    global _service
    if _service is None:
        _service = MemoryService.build_default()
    return _service


def handle_message(msg: dict, service_factory: Callable[[], MemoryService]) -> Optional[dict]:
    """Map one JSON-RPC request to its response. Returns None for notifications (no id) — the
    caller must not write anything back for those. Pure + synchronous → unit-testable."""
    method = msg.get("method")
    mid = msg.get("id")

    if method == "initialize":
        result = {"protocolVersion": PROTOCOL_VERSION,
                  "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            if name == "remember":
                text = service_factory().remember(args.get("text", ""))
                result = {"content": [{"type": "text", "text": text}]}
            elif name == "recall":
                mems = service_factory().recall(args.get("query", ""))
                text = "\n".join(f"- {m}" for m in mems) if mems else "No relevant memories found."
                result = {"content": [{"type": "text", "text": text}]}
            else:
                return _error(mid, -32602, f"Unknown tool: {name}")
        except Exception as e:  # never crash the server on a tool failure — report it
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": f"Error: {e}"}],
                               "isError": True}}
    elif method is not None and mid is None:
        return None                         # a notification (e.g. notifications/initialized)
    else:
        return _error(mid, -32601, f"Method not found: {method}")

    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def serve(stdin=None, stdout=None, service_factory: Callable[[], MemoryService] = _default_service):
    """Read newline-delimited JSON-RPC from stdin, write responses to stdout, until EOF."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue                        # ignore unparseable lines rather than die
        response = handle_message(msg, service_factory)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def main() -> None:
    """Entry point (``signal-mcp``): serve over stdio until the host disconnects."""
    serve()


if __name__ == "__main__":
    main()
