"""Start the signal-memory MCP server for the Claude Code plugin.

Run by the plugin as ``uv run --no-project <plugin root>/plugin/launch.py`` (standard library only, so
uv needs nothing but a Python to run it). It exists because the full install is slow: measured on a
Mac with empty caches, installing everything (the search model's libraries are ~650 MB) took 267 s,
long enough for a host to give up on the server. So:

  1. The light core (the engine has one small dependency) is installed into the plugin's DATA folder,
     which survives plugin updates. Seconds, not minutes. Re-done only when ``uv.lock`` changes.
  2. If full search is wanted and not ready yet, its install runs in the BACKGROUND, detached, and
     this session starts on keyword search. The engine reports which search ran (``ranked``), so the
     degraded session is visible, not silent. Every later session uses full search.
  3. The server starts in the user's PROJECT folder, so memory stays per project
     (``<project>/.signal_engine/memory.json``), exactly as with a manual install.
  4. Child processes get an ALLOW-LIST of environment variables (paths, locale, proxies), never the
     whole environment, so no API key or token in the user's shell reaches an install job.

Nothing here writes to stdout: stdout is the MCP channel. All install output goes to a log file.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT") or os.path.dirname(HERE)
DATA = os.environ.get("CLAUDE_PLUGIN_DATA") or os.path.join(os.path.expanduser("~"), ".signal_engine", "plugin")
VENV = os.path.join(DATA, "venv")
STATE = os.path.join(DATA, "state.json")
LOG = os.path.join(DATA, "install.log")
INSTALLING = os.path.join(DATA, "full-search.installing")
STALE_INSTALL_SECONDS = 30 * 60      # an install marker older than this is from a crashed run
MODEL = "all-MiniLM-L6-v2"           # the engine's default local embedder model


def _log(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def _state():
    try:
        with open(STATE) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_state(state):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE)


def _lock_hash():
    with open(os.path.join(ROOT, "uv.lock"), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# Variables a child process (uv, the server, the background install) may see. An ALLOW-LIST, never
# the whole environment: a plugin's install job must not be handed the user's API keys and tokens
# just because they happen to be set in the shell (Anthropic's plugin scanner holds exactly that).
# Paths, locale, proxy and certificate settings are enough for uv and the server to work.
_PASS_THROUGH = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LC_CTYPE",
    "TZ", "TERM",
    # Windows
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "APPDATA", "LOCALAPPDATA",
    "USERPROFILE", "PROGRAMDATA", "PROGRAMFILES", "HOMEDRIVE", "HOMEPATH",
    # network: proxies and corporate certificates (addresses, never credentials)
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "UV_NATIVE_TLS",
    # where uv keeps its downloads and Pythons; HF_HOME is where the search model is cached
    "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_PYTHON", "UV_OFFLINE", "HF_HOME", "HF_HUB_OFFLINE",
    "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME",
    # the plugin's own and the memory server's settings (set by Claude Code, or by the user)
    "CLAUDE_PLUGIN_ROOT", "CLAUDE_PLUGIN_DATA", "CLAUDE_PROJECT_DIR", "SIGNAL_MEMORY_FULL_SEARCH",
    "SIGNAL_MEMORY_PATH", "SIGNAL_MEMORY_SCOPE", "SIGNAL_MEMORY_LOCK", "SIGNAL_MEMORY_AUTONOMY",
    "SIGNAL_MEMORY_MANUAL", "SIGNAL_EMBED_MODEL", "SIGNAL_DUMB_WIKI",
)


def _child_env(**extra):
    env = {k: os.environ[k] for k in _PASS_THROUGH if k in os.environ}
    env.setdefault("HF_HOME", os.path.join(DATA, "hf"))          # the model lives in plugin data
    env.setdefault("UV_CACHE_DIR", os.path.join(DATA, "uv-cache"))   # ...and so do uv's downloads
    env["UV_PROJECT_ENVIRONMENT"] = VENV
    env.update(extra)
    return env


def _sync(extras, log):
    cmd = ["uv", "sync", "--project", ROOT, "--frozen", "--inexact", "--quiet"]
    for extra in extras:
        cmd += ["--extra", extra]
    return subprocess.run(cmd, env=_child_env(), stdout=log, stderr=log, stdin=subprocess.DEVNULL).returncode


def _bin(name):
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", name + ".exe")
    return os.path.join(VENV, "bin", name)


def _wants_full_search():
    """The plugin's ``full_search`` option. Off means no 650 MB download, keyword search only."""
    val = (os.environ.get("SIGNAL_MEMORY_FULL_SEARCH") or "true").strip().lower()
    return val not in ("false", "0", "no", "off")


def _project_dir():
    """Where memory lives. A plugin's server is not guaranteed to start in the user's project, so
    prefer what the host says the project is, then the working folder unless it is the plugin's own
    folder (memory there would be lost on every plugin update)."""
    for candidate in (os.environ.get("CLAUDE_PROJECT_DIR"), os.getcwd()):
        if candidate and os.path.isdir(candidate):
            real = os.path.realpath(candidate)
            root = os.path.realpath(ROOT)
            if real != root and not real.startswith(root.rstrip(os.sep) + os.sep):
                return real
    return None


def _start_full_search_install():
    """Run ``--prepare-full-search`` detached, so this session starts now. One at a time."""
    try:
        age = time.time() - os.path.getmtime(INSTALLING)
        if age < STALE_INSTALL_SECONDS:
            return
    except OSError:
        pass
    with open(INSTALLING, "w") as fh:
        fh.write(str(os.getpid()))
    log = open(LOG, "a")
    kwargs = {"stdout": log, "stderr": log, "stdin": subprocess.DEVNULL, "env": _child_env()}
    if os.name == "nt":
        kwargs["creationflags"] = 0x00000008 | 0x00000200      # DETACHED_PROCESS | NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "--prepare-full-search"], **kwargs)


def prepare_full_search():
    """Background job: install the search model's libraries, download the model once, mark ready."""
    try:
        with open(LOG, "a") as log:
            _log("full search: installing libraries")
            if _sync(["local"], log) != 0:
                _log("full search: install FAILED; sessions stay on keyword search. See above.")
                return 1
            _log("full search: downloading the model (~90 MB, once)")
            code = subprocess.run(
                [_bin("python"), "-c",
                 f"from sentence_transformers import SentenceTransformer; SentenceTransformer('{MODEL}')"],
                env=_child_env(), stdout=log, stderr=log, stdin=subprocess.DEVNULL).returncode
            if code != 0:
                _log("full search: model download FAILED; sessions stay on keyword search.")
                return 1
        state = _state()
        state["full_search_lock"] = _lock_hash()
        _save_state(state)
        _log("full search: ready from the next session")
        return 0
    finally:
        try:
            os.remove(INSTALLING)
        except OSError:
            pass


def main():
    os.makedirs(DATA, exist_ok=True)
    if "--prepare-full-search" in sys.argv:
        return prepare_full_search()

    lock = _lock_hash()
    state = _state()
    if state.get("core_lock") != lock or not os.path.exists(_bin("signal-memory-mcp")):
        started = time.time()
        with open(LOG, "a") as log:
            _log("core: installing")
            if _sync(["local"] if state.get("full_search_lock") == lock else [], log) != 0:
                _log("core: install FAILED")
                sys.stderr.write(f"signal-memory: install failed; see {LOG}\n")
                return 1
        # Hosts give a server about 30 s to answer; this line is how a slow first start is diagnosed.
        _log(f"core: ready in {time.time() - started:.1f}s")
        state["core_lock"] = lock
        _save_state(state)

    full = _wants_full_search()
    ready = full and state.get("full_search_lock") == lock
    if full and not ready:
        _start_full_search_install()

    env = _child_env(SIGNAL_DUMB_EMBED="local" if ready else "none")
    project = _project_dir()
    if project is None and not env.get("SIGNAL_MEMORY_PATH") and not env.get("SIGNAL_MEMORY_SCOPE"):
        env["SIGNAL_MEMORY_PATH"] = os.path.join(DATA, "memory.json")    # never inside the plugin
    exe = _bin("signal-memory-mcp")
    if os.name == "posix":
        if project:
            os.chdir(project)
        os.execve(exe, [exe], env)
    return subprocess.call([exe], env=env, cwd=project)


if __name__ == "__main__":
    sys.exit(main())
