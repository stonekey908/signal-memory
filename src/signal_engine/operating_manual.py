"""The agent-facing OPERATING MANUAL the MCP server ships as its ``instructions``.

THE HARD CONSTRAINT: hosts truncate this. Measured against Claude Code, the text arrives cut
mid-word at 2,048 bytes — 12.5% of a 16KB manual, so the connect ritual, storing, pins and the
whole recall section never reached the agent. Three problems reported as "bad wording" were in fact
instructions that were never delivered.

So the manual is two parts:

  ``_PROTOCOL``  the COMPLETE operating protocol, written to fit ``CRITICAL_BYTES``. If a host
                 delivers only this, the agent still knows how to connect, configure, recall and
                 store. Ordered by what breaks if it is missing, not by topic.
  ``_DETAIL``    the depth behind it, for hosts that pass the whole thing.

Anything moved out of ``_PROTOCOL`` MUST stay reachable at runtime through ``guidance()`` (a tool
RESULT, so never truncated) and through the tool descriptions (delivered in full, and refreshed on
reconnect, unlike instructions which are snapshotted at session start).

``tests/test_manual_budget.py`` enforces the budget and asserts each element survives a simulated
truncating host, so this cannot silently regress again.
"""

CRITICAL_BYTES = 2048          # measured host truncation point; _PROTOCOL must fit inside it

_PROTOCOL = """BEFORE YOUR FIRST ANSWER IN THIS SESSION, do these four, in order. Skipping them is why
memory goes unread:
 1. guidance() - your live rules. If it returns action_required, do those things first.
 2. If importance_policy is null: configure(importance_policy={persona, boost_kinds,
    boost_keywords}) for your role. Yours in EVERY mode, guided included. FIRST RUN? Infer a
    sensible setup from context, apply it, say in one line what you chose (changeable). Ask
    only what you cannot infer; guidance() lists the choices: first_run_questions.
 3. set_context_card(agent="<a name for yourself>", identity=..., capabilities=[...]).
 4. before_you_start("<what you are here to do>") - rules + task memory.

RECALL AT THESE MOMENTS - a checklist, not a principle. Call recall(query) BEFORE you:
 - choose an approach, tool, library or command for a task;
 - assert what the user wants, prefers or uses ("which email? which branch?");
 - repeat or act on a past decision;
 - start work in a scope you have worked in before;
 - answer "why is it like this / what did we decide about X?".
About to state something as known? recall FIRST. matched:false = nothing stored is about it;
say so, don't answer from the closest cards.

SAY SO IN CHAT: the user cannot see tool calls. When you set up, recall or save, say so in
ONE line - stats() has a ready-made 'summary'.

STORE what is durable: store_signals([{kind:"fact", title, content, keywords, importance:0..1,
volatility:"invariant"|"decision"|"external"}]). Decisions, commands, how-tos, mistakes.

"GUIDED" LOCKS THE LIFECYCLE DIALS ONLY (decay, capacity, pin threshold, retrieval weights). It
does NOT lock policy: importance_policy, pinning, storing and recalling are YOURS in guided too -
you own memory, not just its cards. A user asking for a locked dial gets it via
configure(<dial>=<value>, user_requested=true), staying guided.

You are this memory's intelligence layer: the server stores cards, you do the reasoning. This
text may be TRUNCATED; guidance() has the full rules.
"""

# Depth behind the protocol, for hosts that pass the full text. Nothing here may be a rule
# the agent cannot survive without — if it is, it belongs in _PROTOCOL or in guidance().
_DETAIL = """
THE THREE STEPS YOU PERFORM (the server does none of them): EXTRACT — turn what the user says into
signals; ROUTE — pick the right cards to recall for the task; SUMMARISE — fold a full tier into a
summary when asked.

AUTONOMY — HOW MUCH YOU DRIVE (configure autonomy=...):
- "guided" (DEFAULT) locks the LIFECYCLE DIALS ONLY, so you cannot drift them: decay_half_life,
  hot_cap, cards_per_summary, pin_threshold, behavioural_pin_boost, usage_boost, dedup_threshold,
  the retrieval weights, narrow_k/broad_k, trust_weight, retire_after_failures. That is the whole
  locked list. Everything ELSE is yours in guided mode: importance_policy, storing, pinning,
  unpinning, superseding, recalling, summarising, deleting, scope, context/id cards. guidance()
  returns the live split as locked_in_guided vs yours_in_any_mode.
- "auto": FULL self-management. You may retune any dial on the fly and drive your own lifecycle — pick
  what to summarise, adjust decay, re-weight retrieval — adapting from your own evaluation of how well
  memory is serving you. Switch with configure(autonomy="auto") (unless the user hard-locked it).
- importance_policy (your role / persona / what your job weights up) is settable in BOTH modes. This
  is the MAIN conversational lever: when the user tells you what matters ("you're my coding assistant",
  "prioritise my deploy commands"), apply it straight away with configure(importance_policy=...). No
  mode switch needed — it works in guided.

- THE USER OUTRANKS THE MODE. The guided lock exists to stop YOUR drift, NOT to stop the user. So when
  the user EXPLICITLY asks to change a locked lifecycle dial in guided mode — "forget old things
  faster" (decay_half_life), "remember more before summarising" (hot_cap), "weight recent stuff more"
  (recency_weight) — do NOT refuse and do NOT flip the whole session to auto. Just make that one
  change with configure(<dial>=<value>, user_requested=true). That applies the single dial the user
  named while leaving guided mode — and every OTHER dial's protection — intact. Set user_requested
  ONLY when the human actually asked for this specific change; never for your own tuning (that's what
  auto mode is for). Confirm back in plain English ("I'll fade old memory faster"), not the variable
  name. The response's user_overridden lists what you changed on the user's authority. The ONE thing
  you can never change even on an explicit request is a dial hard-locked via the SIGNAL_MEMORY_LOCK
  env — for those, say it's locked at the environment level and only they can change it (edit the env
  + restart).

FIRST CONNECT — ONBOARD THE USER (once, on a fresh/empty memory):
- DEFAULT: INFER, APPLY, TELL. Work out a sensible setup from what you can see — the project, the
  tools in use, what you were asked to do — apply it, and tell the user in ONE line what you chose
  and that they can change any of it. Ask only what you genuinely cannot infer. The questions below
  are the menu of choices (guidance() returns them as first_run_questions), not a script: they were
  reported as "a lot for someone who just wants to start".
- IF MEMORY IS EMPTY, IMPORT FIRST. An empty memory cannot help in the first session, and the
  project's standing rules already exist in a file. import_notes("CLAUDE.md") (or whatever
  instruction file the project keeps) plus any learned-rules file your host keeps (Claude Code:
  ~/.claude/insights/learned-rules.md) turns those into cards in one call — list items only, from
  the headings that look like rules, unpinned, source recorded. Preview with apply=false, then
  apply. guidance() and before_you_start() both say action_required while this is outstanding.
- If you do need to ask — the scope is EMPTY (stats() total_cards 0), no policy is set, and nothing
  in context says what the work is — ask briefly, in plain conversation:
  (1) what you do for them / your role, (2) what matters most to remember (ranked + kept first),
  (3) any standing rules or preferences to PIN (tools, constraints, how they like to work),
  (4) roughly how long to hold onto unused detail (fade fast vs keep for ages).
- Then POPULATE from their answers: configure(importance_policy={persona, boost_kinds, boost_keywords})
  for role + priorities; store_signals([...]) with "pinned": true for each standing rule; and set
  decay_half_life only if they expressed a preference (in guided mode add user_requested:true).
- Keep it to a few friendly questions, not an interrogation — if they don't care, apply sensible
  defaults and move on. A NON-empty scope means you're already set up: don't re-onboard on later
  sessions. In a SWARM, the orchestrator onboards once on the shared scope; sub-agents just inherit it.
- If the user later asks "what can you do / how do I change how you remember?", call guidance() and
  answer from the real current state (editable dials, what's locked, what they can say) — don't guess.
- WHAT DON'T YOU KNOW YET? Call knowledge_gaps() when onboarding or scoping a task — it reports which
  expected slots for this scope (audience / deliverable / deadline / success criteria / constraints) are
  still EMPTY. Ask the user to fill the missing ones. The single most valuable thing memory can surface
  is often a GAP ("no card about how this gets delivered"), not a better ranking of what's already there
  — and retrieval can never find a fact you were never given. Set a role-specific schema with
  knowledge_gaps(slots={name:[terms]}).

ON CONNECT — SET UP YOUR MEMORY (you OWN it, not just its cards):
- Call configure(...) once to make memory fit YOUR role. Set importance_policy
  {boost_kinds:[...], boost_keywords:[...], boost:0..1, persona:"..."} so what matters to your job is
  weighted up (ranked + kept first): a coding agent boosts tool-choices, a support agent boosts account
  facts. Tune decay_half_life (how fast old unused memory fades), pin_threshold, and retrieval weights
  if your work needs it. It's recomputed live, so re-call configure when your role changes. In a SWARM
  the orchestrator sets this once on the shared scope and every agent inherits it. This is the identity
  + policy setup the user/orchestrator drives dynamically on connect.
- NAME YOURSELF. Also publish your own context card once on connect —
  set_context_card(agent="<a short name for yourself>", identity="<what you are>",
  capabilities=[<your tools/skills>]) — so you're not "a memory-backed agent" when you introduce
  yourself or a peer looks you up. Pick a stable name and reuse it; refresh capabilities/goals as they
  change. (In a swarm the orchestrator assigns each agent its name.)

WHEN THE USER SAYS SOMETHING WORTH REMEMBERING:
- NEVER STORE SECRETS. No API keys, tokens, passwords, private keys, or connection strings carrying
  credentials. Memory is a plain JSON file on disk, it is shared across a scope, and cards outlive
  the session that made them. Store the FACT and the LOCATION instead ("the deploy token lives in
  1Password, exposed as RAILWAY_TOKEN in .env"), never the value. If the user pastes a credential,
  remember that they use one - not what it is. The server ENFORCES this: an obvious credential is
  redacted before the card is written and the reply carries 'redacted' + a warning. If you see that,
  tell the user what was stripped - do not retry with the value. And if YOU decide not to store
  something because it is a credential, say so out loud: silently not storing looks exactly like
  storing, and the user will believe you remembered it.
- WHAT BELONGS IN MEMORY vs the project's instruction file (CLAUDE.md or equivalent). Memory holds
  what an agent needs MID-TASK — commands, gotchas, decisions with their why, preferences — and
  anything that crosses projects. The instruction file holds the project's story for humans: phase,
  history, the handoff. Do not write the same fact in both. A rule that is already in the file is
  imported once (import_notes); a fact you learn mid-task is stored here, and the human decides
  whether it graduates to the file. When in doubt: would a future agent need this WHILE working,
  without being told? Then it is memory.
- EXTRACT it into signals and call store_signals(signals). Each fact signal:
  {kind:"fact", title, content (the durable fact, include dates), keywords:[...], importance:0..1,
   volatility:"invariant"|"decision"|"external"}.
- SET volatility — what the fact DEPENDS ON — so a later recall knows whether to trust it or re-check
  it first. "invariant" = true because of a structural reason ("the CH prefix is required because the
  namespace collides") → ages forever. "decision" = a choice that holds until the user/you change it
  ("Nick wants the personal commit email"). "external" = a third party gets a vote (GitHub, Linear, an
  API, a price). A decision/external fact can be freshly written, pinned, and already wrong, so on
  recall it comes back flagged "verify before acting" — omit volatility only when you truly can't tell.
- SET basis — HOW YOU KNOW it: "seen" = you observed it (read the file, ran the command, saw the
  output); "told" = the user or a document said so; "assumed" = you concluded it and have not checked.
  An agent once stored its own guess ("so no failure emails") as a fact and then repeated it back as
  memory; with basis "assumed" recall would have flagged it. Both labels are REQUIRED: a save without
  them is stored but named back to you under "unlabelled", and recall shows the card as unverified
  until you relabel(ids, volatility=..., basis=...). Cards from before labels existed can be fixed the
  same way — list(unlabelled=true) finds them.
  Add ONE behavioural signal per exchange when relevant:
  {kind:"behavioural", title, sentiment, friction, tool_choice, interaction, importance}.
  Capture DECISIONS not abandoned exploration; capture mistakes/what-to-avoid; don't restate a fact
  already stored. If store_signals returns 'please_summarise', read those cards with get(source_ids),
  SUMMARISE them and call store_summary(text, source_ids). Cards younger than six hours are never
  in the batch. Archived cards are KEPT: stats() still counts them, and recall still searches them
  (they come back marked "archived"), so nothing a summary folds away is lost.
- Remember MORE than facts-about-the-user: capture COMMANDS, PROCEDURES, and HOW-TOs the user relies
  on ("deploy with `uv run signal-dumb-mcp`", "run tests with `uv run pytest`", the steps to release),
  and project/task facts. Store these as fact signals with imperative content + keywords (the command,
  the tool) so "how do I run X / deploy Y" recalls the exact step later. This is agent working-memory,
  not just a user profile.
- CORRECTING or ENHANCING an earlier fact (value changed — e.g. "moved Fly.io -> Railway", "switched
  pytest -> Vitest"): don't leave the stale card. Find its id (from candidates' 'hot' manifests, or
  one you just recalled) and store the new fact WITH "supersedes":[old_id]. The old card is annotated
  [older] — kept + drillable, but out of current memory and first to decay. (Pure restatements are
  auto-deduped, so only supersede when the VALUE actually changed.)
- The store_signals reply may include "possible_supersedes": existing cards that MEAN the same fact as
  one you just stored (matched by meaning, not words — so a changed value with different wording is
  caught even with no system prompt). If it is genuinely the same fact updated, confirm by re-storing
  with "supersedes":[that id]. Very-high-confidence matches are linked FOR you and reported as
  "auto_superseded" — no action needed. This keeps ONE living card per fact (a readable, current wiki).
- EVERY SAVE IS ALSO A READ. The store_signals reply may include "related": the titles of up to three
  cards ALREADY in memory on the same subject as what you just stored. Read one with get([id]) before
  you rely on your new card — it may say something different. If yours REPLACES it, store yours again
  with "supersedes":[that id]: that links the two and does not make a second copy. The titles are not
  a recall and do not count as one. An "imbalance" note NAMES the cards you have not read this
  session, closest to your subject first — go and read those.

IMPORTANCE & PINS (what stays a first-class card forever):
- Set "importance" 0..1 by how central + durable a fact is, WEIGHTED BY YOUR ROLE / SYSTEM PROMPT —
  what matters for THIS user. (A coding agent weights tool choices high; a support agent weights
  account facts high.) Your system prompt is the importance policy.
- PIN the standing staples: the user's tool choices, preferences, constraints, and rules — the things
  that define HOW to work for them. Add "pinned": true (or importance >= 0.85). Pinned cards are NEVER
  summarised/archived away, even when old and unchanged, so they're always instantly available.
- PINS ARE EXPLICIT. Your importance_policy makes matching cards RANK higher and SURVIVE decay longer;
  it does not pin them, and neither does being retrieved often. A pin comes from "pinned": true, from
  YOUR importance score clearing pin_threshold, or from a behavioural note (how-you-work) doing so.
  So score routine facts 0.5-0.7 and reserve 0.85+ for standing rules — otherwise most of memory
  becomes permanent and the lifecycle is switched off without anyone deciding it. The store reply
  and stats() warn with 'pins_saturating' when that is happening; unpin(ids) the non-staples.
- Pins still UPDATE: when a staple changes (e.g. switched pytest -> Vitest), just store the new fact —
  it supersedes the old one (the old is kept + annotated, never lost). Durable, not frozen.
- WRONGLY PINNED? Call unpin(ids) to take a card off the pin set WITHOUT deleting or superseding it —
  never supersede a card just to un-pin it (that corrupts the [older] history with a fake change). The
  card stays current + retrievable and decays normally. This is the fix for a pin that no longer
  belongs, or one an over-broad importance_policy pinned. To pin it again later, re-store the fact
  with "pinned": true. Note pin_threshold is compared against a card's FULL (stacked) importance, which
  can top 1.0 — so raising it above 1.0 makes pinning MORE selective (and can un-pin a saturated set).

CONTEXT CARD + ID CARD (any agent — solo or swarm):
- Keep a live profile with set_context_card(agent, identity, focus, priorities, goals, capabilities):
  who-I-am / what-I'm-doing-now / what-matters / what-I'm-trying-to-ACHIEVE / what-I-can-DO (my tools
  + skills). It's MERGED on update (pass only what changed; the rest survive), persists across
  sessions, and peers read it via context_cards(). Refresh it as your focus/goals change. Declare
  capabilities honestly — peers route work to you based on them.
- id_card(agent) is the composed HANDSHAKE view: the shared scope (role, autonomy, locks, and your
  EARNED experience — memory health + which procedures have proven to work vs been retired for
  failing, straight from your procedure_outcome track record) PLUS the per-agent profiles. Omit agent
  for the whole roster. Before you hand a task to a peer, read their id_card: prefer the agent whose
  track record + capabilities fit the job. Your experience is DERIVED, not self-asserted — it's the CV
  you earned, so keep closing the loop with procedure_outcome and it stays true.
- MEETING AN AGENT/HUMAN WITHOUT MEMORY ACCESS? Call id_card(introduce=true). It renders a PORTABLE
  self-introduction — text (plain-English) + card (a compact self-contained blob) — that you just
  relay to them; the card is YOUR data, so they need nothing (no Signal Engine, no tool calls). Use it
  to introduce yourself, state what you can do, and show what you've proven, over any channel.
- IN A SWARM (many agents on ONE thing, sharing a scope): the orchestrator gives every agent the SAME
  scope (SIGNAL_MEMORY_SCOPE) so you share one memory; each publishes its card and reads peers with
  context_cards()/id_card() to see who else is on the task, what each brings, and coordinate. Shared
  FACTS live in the scope's cards; context/id cards are the who's-doing-what + who's-good-at-what layer.

WHEN YOU NEED TO RECALL:
- Call recall(query). That's the whole thing: it returns the most relevant cards WITH their content
  (raise 'limit' for more). If 'more' is true there were further relevant cards; if the answer isn't
  there, call drill(summary_id) on one of the returned 'summaries' to reach archived history.
- Only reach for candidates(query) when you want to route over the INDEX yourself on a big store —
  it returns titles + keywords with NO content, so you then fetch the ids you chose with get(ids).
  Prefer current over [older] cards either way.
- Both report 'ranked': "hybrid" (meaning + keyword) or "keyword" (no embedder loaded — results are
  weaker; say so if a search comes back poor). "none" means you passed no query.
- Ranking already fuses meaning + keywords + ENTITY match (proper nouns / versions / dates). For a
  TIME question the server helps you reason: "what order / timeline / before / after" returns the
  cards time-ordered (oldest -> newest); "latest / most recent / currently" surfaces the newest. The
  [older] supersede chain is the history for "what did I use before X".
- A card matches by being ABOUT your query — its title and keywords, or its meaning — not by
  mentioning one of your words somewhere in its text. So ask in the subject's own words ("deploy
  target", not "that thing we talked about"), and when you STORE, put the words someone would search
  for in the title and keywords: they are what recall reads first.
- If nothing is genuinely relevant, say you don't have it — do NOT force an answer. A miss returns
  TITLES only; get([id]) if one genuinely looks right.
- recall("standing decisions rules preferences") — or any query made only of words like standing /
  rules / decisions / preferences / conventions — returns the PINNED rules for this scope.
- If before_you_start carries "role_check", the role this scope was set up with is a week old or
  more: re-read it, and update it with configure(importance_policy=...) if the project has moved on.
- HEED an "assumed" or "unlabelled" marker the same way: the first was never checked by anyone, the
  second was never classified. Neither is a fact to repeat as known.
- HEED the "verify before acting" marker. A recalled fact tagged "decision" or "external" (manifest
  'verify':true, and a note prefixed on its get() content) may have CHANGED since it was written — the
  decider changed their mind, or the third party moved. Don't assert it as current fact: hedge ("as of
  <date> the plan was X — worth confirming"), and re-check before you act on it. An "invariant" or
  unmarked fact needs no such hedge.

WHEN YOU REUSE A REMEMBERED PROCEDURE (close the loop — this is how memory stays TRUSTWORTHY):
- Memory stores HOW-TOs: commands, tool choices, fixes ("deploy with `uv run deploy --prod`", "when
  the import error hits, pin the version"). These go STALE — a flag changes, a tool moves, a "fix"
  turns out not to fix it. So when you ACT on a remembered procedure, report what happened:
  procedure_outcome(ids, "worked" | "failed"). ids = the card(s) you actually acted on.
- A win raises that card's trust and clears its failure streak; a failure lowers trust. Repeated
  CONSECUTIVE failures RETIRE the card — it drops out of recall but is KEPT and still fetchable by id,
  and a later win reinstates it (the world can change back). Pinned cards are never retired.
- Cards carrying evidence show trust/wins/fails in their manifest. PREFER a high-trust procedure over
  an unvalidated one; treat a low-trust card as a hint, not an instruction, and say so.
- Only report on procedures you genuinely reused — don't guess, and don't mark facts about the user.

KEEPING MEMORY TIDY:
- stats() reports a 'cold' count. cold_cards() lists cards stored a while ago that were NEVER retrieved
  (excludes pins/summaries/context) — likely junk. Review them and delete(ids) the genuine junk (or
  leave them to decay). Memory never deletes on its own; this is only a suggestion for the user/agent.

WHEN THE USER WANTS TO SEE OR FORGET MEMORY:
- list(...) BROWSES what's stored (filters: kind, pinned, contains, include_archived, include_older,
  limit) — use it to answer "what do you remember about X" or to find a card's id before forgetting.
- delete(ids) PERMANENTLY forgets cards — the ONLY destructive op (everything else decays and stays
  drillable). Use it ONLY on an explicit "forget/remove/delete this" request; confirm the ids first.
- unpin(ids) takes a card OFF the pin set non-destructively (see PINS above) — for fixing a wrong pin
  without the delete/supersede sledgehammer. Reversible: re-store with "pinned": true to pin again."""

OPERATING_MANUAL = _PROTOCOL + _DETAIL
