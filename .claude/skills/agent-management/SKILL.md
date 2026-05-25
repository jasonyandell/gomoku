---
name: agent-management
description: Manage the fleet of `claude agents` background Claude sessions on this machine — list, inspect, cross-correlate fleet↔worktrees↔transcripts, triage what's working/idle/dead/blocked, and keep the fleet hygienic. The `claude agents` command (agent view) dispatches background sessions; this skill is the cockpit for that autopilot. Trigger on "claude agents", "agent view", "the fleet", "what agents are running", "list my agents", "what's each session doing", "summarize my sessions", "which agent owns this worktree", "clean up the fleet", "are any agents stuck/blocked", "cross-correlate sessions and worktrees", or anything about background-session triage, ai-ids, ~/.claude/jobs, roster.json, or .claude-session files.
---

# agent-management

Operate the fleet of background Claude Code sessions that `claude agents` (agent
view) dispatches. This is the **cockpit** for that **autopilot** ([[cockpit-vs-autopilot]]):
agent view launches and runs sessions; this skill is the one-glance status surface
+ triage + hygiene gauge over them. **Read-only by default** — inspecting the fleet
is safe, but stopping/deleting a session affects *other people's live work*, so those
stay manual and deliberate (see Triage actions).

**Why this skill exists (the north star).** Too many sessions, too many topics. Three
capabilities — **search** (know what topics are live), the **post office** (talk to the
sessions), and **session control** (resume/fork/triage) — exist to *land the work*, so the
operator can run Sid Bidasaria's "stop babysitting your agents" playbook: verify → multi-Claude
→ background loops, with attention protected. **Everything is log-based: nothing is deleted or
destroyed, only added or learned.** This skill is the self-improving agent for that problem.
See [[fleet-management]] and [[cockpit-vs-autopilot]].

## Quick mental model

`claude agents` opens **agent view**, a TUI that dispatches/monitors background
sessions (one row each). It is NOT "agent teams" (the experimental coordinated-
multi-agent mode under `~/.claude/teams/`), and NOT in-session subagents (the
`Agent` tool). Each fleet row is its own Claude process with its own session UUID.

**The state surfaces (authoritative → derived):**

| Surface | Keyed by | Lifetime | Holds |
|---|---|---|---|
| `claude agents --json` | sessionId | live only | the **alive?** signal: pid, cwd, kind, name, status |
| `~/.claude/daemon/roster.json` | ai-id | live only | **`dispatch.source`** (fleet/spare/slash) + `dispatch.seed.intent` (launch prompt) |
| `~/.claude/jobs/<ai-id>/state.json` | **ai-id** | **durable (survives exit)** | state, tempo, name, intent, sessionId, output.result — the agent-view board |
| `~/.claude/projects/-<repo>/<sessionId>.jsonl` | sessionId | durable | full transcript |
| `<repo>/.git/worktree-sessions.jsonl` | session_id | durable | which session owns which worktree (survives teardown) |
| `<worktree>/.claude-session` | — | gitignored | per-worktree owner stamp (`cat` it inside any worktree) |

**ID taxonomy** (the thing variously called "the ai-id"):
- **ai-id / short id** = first 8 hex of the session UUID (`55038670`). The handle agent
  view shows and `~/.claude/jobs/<ai-id>/` is keyed by. **Report by this.**
- **sessionId** = full UUID → `claude --resume <uuid>`, transcript filename.
- **name** = AI-generated title (was `aiTitle` in the transcript).
- **bridgeSessionId** = `session_…` Anthropic-internal handle (in state.json; ignore).
- **subagent id** = `agent-a…` → owns `.claude/worktrees/agent-<id>` when a fanned-out
  subagent edits files.

`dispatch.source` tells provenance: **fleet** = you dispatched it in agent view;
**spare** = a pre-warmed standby (idle until adopted; name == its own ai-id);
**slash** = forked from a `/`-command or `--fork-session`.

## The tool — `scripts/agent_fleet.py` (read-only)

Don't hand-roll the joins; they're a gauge, not a remembered procedure.

```bash
# One cross-correlated board: ai-id | alive? | source | state | name | worktree/branch
python scripts/agent_fleet.py status
python scripts/agent_fleet.py status --cwd ~/code/gomoku   # filter to one repo

# Slow-entropy gauge (put in a cron narrator / one-glance check):
python scripts/agent_fleet.py gauge          # human line + ⚠ for anomalies
python scripts/agent_fleet.py gauge --json   # machine-readable

# Summarize one session's transcript (asks / result lines / narrative tail / tools):
python scripts/agent_fleet.py digest <ai-id>

# Find a session by topic across ALL transcripts — incl. DEAD / transcript-only ones
# (status/gauge only see the live fleet + jobs board; search sees everything):
python scripts/agent_fleet.py search "next step|strategy"     # --scope human|assistant|all
```

**Finding "the session where I was talking about X" is a `search`, not a `status`.** Old or
crashed sessions have no `~/.claude/jobs/<ai-id>/state.json`, so they never appear in `status`
or `gauge` — but `search` scans every transcript in the project, ranks by hit count, and prints
the `claude --resume <uuid>` command for each. Default `--scope human` searches the user's typed
prompts (where a topic is actually named); `--scope all` adds assistant text.

A **healthy** gauge reads: `sharing main: 0–1`, `leaked locked agent-worktrees: 0`,
`needs-input: 0`. Anything higher is the fleet accreting entropy — see Hygiene.

## Fast search + topic mindmap (SQLite cache) — `session_db.py`, `session_mindmap.py`

`agent_fleet.py search` re-scans transcripts live (zero setup, fine for one-offs). For
repeated search or a visual map, build the **SQLite FTS cache**. Transcripts stay the
source of truth; the cache rebuilds one session at a time on an **mtime miss**:

```
# Build/refresh the cache (incremental — only changed transcripts re-import):
python scripts/session_db.py --repo ~/code/gomoku sync
# Fast FTS search — prints rank + resume + fork commands (auto-syncs first):
python scripts/session_db.py --repo ~/code/gomoku search "next step OR strategy" --scope human
# Recompute topic<->session grounding from scripts/topics.json (editable taxonomy):
python scripts/session_db.py --repo ~/code/gomoku topics
```
DB lives at `~/.claude/agent-fleet/<project>.db` (outside the repo). `scripts/topics.json`
is a plain, editable taxonomy: each topic is a set of keyword phrases; the cache counts
how many of a session's messages match → the grounding weight.

### The mindmap (local web)
```
python scripts/session_mindmap.py --repo ~/code/gomoku --serve   # build + open in browser
```
Topic-hub graph: boxes = topics, dots = sessions (sized by how much they touched topics),
edges = grounding weight. Click a session → its opening prompts + copy-paste **resume** and
**fork** commands. (Graph lib loads from a CDN; the data is embedded and local.)

### Resuming vs forking a conversation
- **Resume** (continue the same session): `claude --resume <uuid>`
- **Fork** (new branch; copies history to a new id, original untouched): `claude --resume <uuid> --fork-session`
  Forks from the latest message; previously-approved permissions do NOT carry to the fork.
  In an active session use `/branch`; `/rewind` is checkpoint-revert *within* a session, not
  a fork. (Verified: code.claude.com/docs/en/sessions + `claude --help`.)
  `python scripts/session_db.py cmd fork <uuid>` prints the command for pasting.

## Post office — talk to the fleet (`scripts/postoffice.py` + a `cagent`)

The supported ways to message a live session are interactive (agent view / Remote Control)
or a launch-time Channel; there's no CLI to inject into a running session. The post office
is the **log-based DIY message bus** that fills that gap, using only supported primitives.

- **Append-only log per mailbox** (`~/.claude/agent-fleet/postoffice/<mb>.log`): senders only
  ever append; a separate `.cursor` tracks progress. **Nothing is deleted or rewritten — only
  added or acked.** History and progress are decoupled.
- **Catch-up recovers missed watches:** `pending` returns *every* post after the cursor, so a
  cagent that was busy/down/missed an event still drains them all on the next scan (at-least-once).
- **A `cagent`** (post-office session you spawn in agent view) runs a low-resource event loop:
  catch up (`pending` → handle/route → `ack --all`) → `wait` (blocks at ~0% CPU until a post or a
  600s safety timer) **run as a `run_in_background` command so the harness wakes the session on
  exit** → repeat. No polling, ~no idle tokens.

```
python scripts/postoffice.py send --to cagent --from you "land the gpu-daemon merge"  # any producer
python scripts/postoffice.py pending --mailbox cagent      # what the cagent processes
python scripts/postoffice.py prompt  --mailbox cagent      # paste-able spawn prompt ↓
```
**You spawn the cagent** (I can't launch a persistent fleet session): open `claude agents`,
dispatch a new background session, and paste the output of `postoffice.py prompt`. Per-agent
mailboxes generalize this into a full bus — each agent self-subscribes to its own `--mailbox`.
This is a DIY [Channel](https://code.claude.com/docs/en/channels); swap to `--channels` later
without changing the "drop a post" interface.

## Working paths (the raw one-liners the tool wraps)

```bash
claude agents --json | jq -r '.[] | "\(.sessionId[0:8]) \(.kind) \(.status//"-") \(.name//"-")"'
jq -r '.workers|to_entries[]|"\(.key) \(.value.dispatch.source) \(.value.dispatch.seed.intent//"-")"' ~/.claude/daemon/roster.json
for j in ~/.claude/jobs/*/state.json; do jq -r '"\(.sessionId[0:8]) \(.state) \(.name)"' "$j"; done
cat <worktree>/.claude-session                 # who owns this worktree + resume cmd
python scripts/worktree_session.py log          # full session↔worktree registry
```

## Triage actions (DELIBERATE — these touch other live work)

Agent view (`claude agents`, interactive) is the right place for these; CLI equivalents:
```bash
claude --resume <uuid>        # attach / read a session (fork with --fork-session)
# stop a session: do it from agent view, or signal its pid from `claude agents --json`.
```
- **Never** `kill -9` a session mid-write, and never stop one you didn't dispatch
  without checking its `state`/`output.result` first — `done`+`idle` is safe to leave;
  `working`/`busy` is doing something; `needs-input`/`blocked` is the one to actually
  look at. Autonomy is a deny-list ([[feedback-autonomy-denylist]]), but *other agents'*
  processes are outward-facing — confirm before stopping them.

## Hygiene & discipline (the fleet's slow entropy)

Two failure modes this skill exists to catch — neither produces a bad *moment*, only
aggregate cost a week later (the slow-entropy class, [[feedback-janitor-not-procedure]]):

1. **Agents sharing the `main` checkout.** When several fleet agents all work in
   `~/code/gomoku` instead of each taking a worktree, their edits co-mingle: an
   uncommitted pile no one owns, and merges from the disciplined agents get blocked.
   The fix is upstream — **every unit of work gets its own worktree**
   (`python scripts/worktree_session.py add <slug>`), land via `git merge --no-ff`,
   tear down ([[feedback-worktree-per-unit-of-work]], [branch-and-worktree-workflow.md]).
   The gauge (`agents_sharing_main`) is the sensor; healthy is ≤1.
2. **Leaked locked subagent worktrees** under `.claude/worktrees/agent-*`. A fanned-out
   subagent that edits files gets an isolated worktree; if its branch merges but the
   worktree isn't pruned, it lingers **locked**, and `reclaim_worktrees.py` skips locked
   ones. The gauge (`leaked_locked_subagent_worktrees`) counts them; reclaim with
   `git worktree unlock <path> && git worktree remove <path>` once the branch is merged.

## Friction-smoothing log

Things that bit us, with fixes. **Read on session start; append after every session.**
This is the part that compounds.

**Meta — what kind of fix to write.** The default sensor here is *narrated friction*
(something bit us, we wrote it down). It is blind to **slow entropy** (leaked worktrees,
dead sessions piling up, spares accumulating) that no single moment surfaces. So classify
each fix: if it's a **procedure** ("remember to check X") it'll decay silently — **upgrade
it to a gauge** (a one-line metric you'd notice the day it breaks; here, `agent_fleet.py
gauge`). For every class of artifact the fleet creates (sessions, worktrees, spares),
there should be a gauge that counts it and, where reclaimable, a janitor that reclaims it.

### 2026-05-25 (skill bootstrap — building the fleet inspector from a live messy fleet)

**`claude agents --json` is the authoritative live list; transcript file mtimes are NOT.**
- Symptom: ranking sessions by `~/.claude/projects/**/<id>.jsonl` mtime suggested ~15
  "active" sessions, but most had already exited — their logs were just recently written
  before the process died. The fleet's true alive-set is only what `claude agents --json`
  reports (its `sessionId`s map to live PIDs).
- Fix: derive `alive?` from membership in `claude agents --json`, not file recency. The
  durable per-agent record (`~/.claude/jobs/<ai-id>/state.json`) covers *dead* sessions
  too (state survives exit), so the board = union of (durable jobs) ⋈ (live fleet).
- Lesson: "recently modified" ≠ "running" for sessions. `daemon.status.json` is also a
  trap — it showed `workers: {}` (stale); trust `roster.json` + `--json`, not it.

**The user-facing handle is the 8-char ai-id, and transcript field names are camelCase
with a non-obvious prefix.** When parsing transcript JSONL: the title is
`{"type":"ai-title","aiTitle":...}` and each human turn is
`{"type":"last-prompt","lastPrompt":...}` — NOT `.message.content`. Human prompts repeat
across lines (one `last-prompt` re-emitted per turn); collapse consecutive duplicates or
the digest shows the same ask 10×. (`agent_fleet.py digest` handles both.)

**Sessions that worked in shared `main` are invisible until you join the registry.**
- Symptom: a stack of uncommitted files in `main` with no obvious owner, and a blocked
  `feat/lab-agents` merge.
- Root cause: 9 distinct fleet agents had each recorded `worktree=~/code/gomoku, branch=main`
  in `.git/worktree-sessions.jsonl` within a 3-minute window — all editing the shared tree.
- Fix: the `agents_sharing_main` gauge surfaces it directly. The durable fix is the
  worktree-per-unit rule (this skill's Hygiene §1). A fleet that dispatches code-editing
  agents should default them to `--add-dir`-scoped worktrees, not the shared checkout.

**Five subagent worktrees were leaked and LOCKED under `.claude/worktrees/agent-*`.**
- Symptom: `git worktree list` showed 5 `agent-*` worktrees marked `locked`, branches
  already merged to main; `reclaim_worktrees.py --apply` left them (it skips locked).
- Fix: the `leaked_locked_subagent_worktrees` gauge counts them; reclaim explicitly with
  `git worktree unlock <path> && git worktree remove <path>` after confirming the branch
  is merged. Candidate to fold into the reclaim janitor (unlock+remove if branch is an
  ancestor of main and the owning session is dead).

### 2026-05-25 (finding a session by topic — status/gauge are blind to dead sessions)

**Asked to find "the session where I was considering next steps for the AI"; `status`/`digest`
couldn't, because the target sessions were dead and transcript-only.**
- Symptom: the two best matches (`410251ca` "alpha_zero strategy analysis", `e7428a34` "wl1
  wave-lockstep") had no `~/.claude/jobs/<ai-id>/state.json` — they're older sessions whose
  processes long since exited, leaving only the transcript JSONL. `build_board` is the union of
  the live fleet and the jobs board, so neither showed up. I hand-rolled a jq+grep over every
  transcript twice (the first keyword net was too narrow and missed the strategy session).
- Root cause: the skill had no "search the full transcript corpus by topic" capability — only
  live/durable-state inspection. But "which session was I talking about X in?" is a corpus
  search, and the user's typed prompts (`last-prompt`) are where topics are actually named.
- Fix: added `agent_fleet.py search <regex> [--scope human|assistant|all]` (pure `rank_search`
  + `load_corpus`, tested). It ranks every transcript by hit count and prints the resume command.
  Dogfooded: `search "next step|strategy|what to learn"` ranks `410251ca` #1. Use `search` (not
  `status`) whenever the question is "find the session where…".
- Lesson: the fleet board is a *liveness* view; topic recall needs a *corpus* view. They're
  different surfaces — don't try to answer "where did we discuss X" from the live board.

### 2026-05-25 (SQLite cache + mindmap — FTS phrase-quoting, and cache-by-mtime)

**Two topics silently computed 0 session edges because their keywords had hyphens/
underscores ('self-play', 'run_sweep', 'self-improving').**
- Symptom: `session_db.py topics` showed `training` and `skills_meta` at 0 sessions while
  every other topic matched 24–30 — obviously wrong (every session mentions training).
- Root cause: a bare FTS5 query term like `self-play` parses `-` as an operator and raises
  `OperationalError`; `recompute_topics` caught it per-topic and `continue`d, so the whole
  topic dropped to 0 with no error surfaced.
- Fix: `build_fts_query` now quotes EVERY keyword as a phrase (`"self-play"`) — avoids the
  operator parse AND matches the intended phrase. Lesson: quote anything you hand to FTS5
  `MATCH`; and a per-item try/except that swallows errors turns a syntax bug into a silent
  empty result — surface it or you'll trust a wrong zero.

**The cache key is the transcript's mtime.** `sync` skips a session iff its file mtime equals
what's stored; any new/modified transcript re-imports just that one (deleted ones pruned). So
`search`/`topics`/`mindmap` auto-`sync` first and stay fresh cheaply — there's no separate
"is it stale?" check to add; the mtime compare IS the cache.

### 2026-05-25 (post office — event-driven without polling, and missed-watch recovery)

**You can run a low-resource event loop in a session without polling: a `run_in_background`
blocking command that the harness re-invokes you on when it exits.**
- A cagent backgrounds `postoffice.py wait` (which blocks in `tail -n0 -F <log> | head -n1`).
  While blocked it's ~0% CPU and consumes NO model turns; when a post lands the command exits
  and the harness wakes the session. This beats `/loop` polling (a turn every interval even
  when idle) for an always-on reactor. Each wait has a 600s timeout = periodic safety re-scan.
- **Missed watches are recovered by a cursor, not by the watch.** Never trust "the one new line"
  the watch saw — on every wake, read ALL posts after the cursor (`pending`) and `ack` through
  what you handled. The log is append-only; the cursor is a separate file; ack never edits the
  log. So a fresh cagent (after a crash or context-rotation respawn) just catches up from the
  cursor — nothing is lost. Lesson: for any file-tail reactor, the watch is the *wake*, the
  cursor is the *truth*.

**Division of labor: I can't launch a persistent fleet cagent.** Fleet sessions are supervisor-
dispatched; from a Bash tool I can only spawn my own (lifecycle-bound) subagents or one-shot
`claude -p`. So the human spawns the cagent in agent view with `postoffice.py prompt` output;
the skill provides the scripts, log, and prompt. Don't promise to "start the cagent" — hand over
the paste-able prompt instead.

### < add new friction-smoothing entries here as they appear >

## Self-improvement clause

**Future sessions: this skill gets better when you write it better.**

At the end of every session that used this skill — successful, halted, or escalated — append:

1. A new entry to the Friction-smoothing log above: any non-trivial bug, surprise, or
   workflow gap, with **symptom + root cause + fix + lesson**. Be specific —
   "the durable record is `~/.claude/jobs/<ai-id>/state.json`, keyed by the 8-char ai-id"
   beats "agent state is somewhere in ~/.claude".
2. If you found a new **working path** (a clean way to inspect/triage/cross-correlate the
   fleet), add it to the Working paths section *and*, if it's mechanical, fold it into
   `scripts/agent_fleet.py` with a test in `tests/test_agent_fleet.py` — a one-liner in
   prose decays; a tested subcommand doesn't.
3. If you found a new class of fleet entropy (a kind of artifact that accretes silently),
   add a gauge for it in `compute_gauges()` so the next session sees it at a glance.
4. If the lesson is project-durable (not just personal-to-Claude), add/extend the matching
   wiki page — memories also go to the wiki ([conventions.md](/Users/jason/code/gomoku/wiki/topics/conventions.md)).

If you smooth a friction already documented here, you may reword it — but **do not delete
entries**. The accumulating ledger is the value.

## Cross-refs

- North star (search + post office + session control → land work → Sid playbook): [/Users/jason/code/gomoku/wiki/topics/fleet-management.md](/Users/jason/code/gomoku/wiki/topics/fleet-management.md)
- Cockpit-vs-autopilot lens: [/Users/jason/code/gomoku/wiki/topics/cockpit-vs-autopilot.md](/Users/jason/code/gomoku/wiki/topics/cockpit-vs-autopilot.md)
- Worktree workflow (load-bearing): [/Users/jason/code/gomoku/wiki/topics/branch-and-worktree-workflow.md](/Users/jason/code/gomoku/wiki/topics/branch-and-worktree-workflow.md)
- Worktree hygiene (janitor+gauge): [/Users/jason/code/gomoku/wiki/topics/worktree-hygiene.md](/Users/jason/code/gomoku/wiki/topics/worktree-hygiene.md)
- Conventions: [/Users/jason/code/gomoku/wiki/topics/conventions.md](/Users/jason/code/gomoku/wiki/topics/conventions.md)
- Session recorder: `scripts/worktree_session.py` (records who owns each worktree)
- Sister skills: [[gomoku-research-lab]] (the lab this fleet often runs), [[gomoku-train]]
- Memories: [[project-fleet-management]], [[feedback-cockpit-vs-autopilot]], [[feedback-worktree-per-unit-of-work]], [[feedback-janitor-not-procedure]], [[feedback-autonomy-denylist]], [[feedback-lab-scheduler]]
