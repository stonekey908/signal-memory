# Signal Memory

Long-term memory for Claude Code that runs entirely on your machine.

Claude forgets everything between sessions. Signal Memory gives it a memory it manages itself: as you
work, it saves the things worth keeping (decisions and why they were made, commands, gotchas, your
preferences) as small cards, and recalls them the next time they matter. No account, no API key, and
nothing you store leaves your computer.

## What you get

- **Recall that finds what a question is about.** Cards are ranked by meaning, words and names, and a
  search that finds nothing says so instead of guessing.
- **Standing rules.** Important cards can be pinned so they always apply, and they are listed at the
  start of each session.
- **Facts that know how they are known.** Each card records whether it was seen, told or assumed, and
  whether it can change. Recall flags the ones to re-check before relying on them.
- **A lifecycle that never deletes.** When memory fills up, older cards are folded into summaries and
  archived. They stay searchable.
- **Memory per project.** Each project gets its own memory, stored in that project's folder.
- **Works with any model your Claude Code uses.** Your agent does the thinking; memory just stores and
  ranks, so there is no extra model cost.

## Requirements

- Claude Code (terminal, or the Code tab in the Claude desktop app).
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), the Python package manager. If you
  do not have Python, uv fetches it for you.
- macOS or Linux. Windows should work but has not been tested yet.

## Install

```text
/plugin marketplace add stonekey908/signal-memory
/plugin install signal-memory@signal-engine
```

Then restart Claude Code.

## The first session

1. **Memory starts in a few seconds** with keyword search, so you can use it straight away.
2. **Full search installs in the background.** It downloads about 650 MB of libraries and a 90 MB
   model, once. From your next session, recall also searches by meaning.
3. **Your agent sets itself up.** On a new project it chooses a sensible setup, tells you what it chose,
   and starts saving.

If you would rather not download that much, turn **Full search** off in the plugin's settings
(`/plugin`, then configure Signal Memory). Memory then stays on keyword search, which is weaker but
still useful.

## Tested

Signal Memory has an extensive automated test suite (over 600 tests) and has been run end to end
through real Claude Code sessions, including clean installs from this repository.

## Where your memory lives

| What | Where |
|---|---|
| Each project's memory | `<your project>/.signal_engine/memory.json` |
| Its settings | `<your project>/.signal_engine/memory.json.config.json` |
| The plugin's own install and logs | Claude Code's plugin data folder for Signal Memory |

Add `.signal_engine/` to your project's `.gitignore` unless you want memory committed with the code.
Memory is plain JSON: you can read it, back it up or delete it like any other file.

## Asking for things

You can talk to your agent about its memory in plain words:

- "Remember that we deploy with `make release`."
- "What do you remember about the billing work?"
- "Pin that, it always applies." / "That shouldn't be pinned."
- "Forget that." (the one permanent delete)
- "What can I change about how you remember things?"

## Privacy

Everything you store stays on your computer. See [PRIVACY.md](PRIVACY.md) for exactly what is
downloaded and when. Licence: MIT.

## Troubleshooting

- **Memory never switches to full search.** Read `install.log` in the plugin's data folder; it records
  every install step and any failure. Sessions keep working on keyword search meanwhile.
- **The server does not start.** Check that `uv --version` works in a terminal, then restart Claude
  Code.

## Uninstall

`/plugin uninstall signal-memory@signal-engine`. Your project memory files are not touched; delete
the `.signal_engine` folders yourself if you want them gone.
