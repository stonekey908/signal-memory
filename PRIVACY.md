# Signal Memory privacy policy

Signal Memory is a local program. It has no server, no account and no telemetry.

## What it stores

The memory cards your agent saves while you work, and their settings. They are written as plain JSON
files in the project you are working in (`<project>/.signal_engine/`). Nothing is stored anywhere else
except the plugin's own installed software and an install log, kept in Claude Code's plugin data
folder on your computer.

## What it sends

Nothing. Signal Memory never sends your memory, your code or your conversations anywhere. Searching
by meaning runs on your own computer with a local model.

## What it downloads

Two things, once, from public sources, both handled by `uv` and the standard Python tools:

- **Software packages** from the Python Package Index (pypi.org): a small core on first start and,
  if Full search is on, about 800 MB of libraries in the background.
- **A search model** (`all-MiniLM-L6-v2`, about 90 MB) from Hugging Face (huggingface.co), if Full
  search is on.

Those services see an ordinary download request from your computer, as with any package install.
Turn Full search off in the plugin's settings to skip the large download and the model.

## What the installer can see

The install job and the memory server are started with a short, fixed list of settings from your
environment: paths, locale, proxy and certificate settings, and the plugin's own options. API keys,
tokens and other credentials set in your shell are never passed to them.

## Credentials

Memory refuses to keep secrets: an obvious API key, token, password or private key in anything your
agent tries to store is replaced with `[redacted]` before it is written, and your agent is told.

## Deleting your data

Ask your agent to forget a card, or delete a project's `.signal_engine` folder. Uninstalling the
plugin removes its software and logs but leaves project memory files alone.

## Contact

Questions and reports: <https://github.com/stonekey908/signal-memory/issues>
