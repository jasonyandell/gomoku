# Fleet management — landing work so we can stop babysitting agents

**The problem.** Too many Claude sessions, too many topics. They meander, overlap, and
stall; work gets stranded in a session no one can find again, and attention is the scarce
resource. (See the snapshot: 39 transcripts, ~16 distinct topics, 9 sessions trampling the
shared `main` checkout at once.)

**The north star.** Build the toolchain that *lands the work*, so the operator can run Sid
Bidasaria's **["Stop babysitting your agents"](/Users/jason/code/gomoku/wiki/sources/sid-bidasaria-stop-babysitting-agents-2026-05-20.md)**
playbook — verify → multi-Claude → background loops, with attention protected — instead of
QA-ing each session by hand. This is the [cockpit over the autopilot](/Users/jason/code/gomoku/wiki/topics/cockpit-vs-autopilot.md).

## The three capabilities (and why each)
- **Search** — *know what topics are happening.* SQLite/FTS cache over every transcript
  (`scripts/session_db.py`), a topic taxonomy (`scripts/topics.json`), and a local web
  **mindmap** (`scripts/session_mindmap.py`) that grounds topics in the sessions that raised
  them. Recall is a corpus search, not a liveness view, so it reaches dead/old sessions too.
- **Post office** — *talk to the sessions.* An append-only message bus (`scripts/postoffice.py`)
  + a `cagent` post-office session that reacts to posts. Fills the gap that there's no CLI to
  inject into a running session (a DIY [Channel](https://code.claude.com/docs/en/channels)).
- **Session control** — *actually interact.* `scripts/agent_fleet.py` (status/gauge/digest)
  for the live fleet, plus copy-paste **resume** (`claude --resume <id>`) and **fork**
  (`claude --resume <id> --fork-session`, original untouched) commands surfaced everywhere.

## The load-bearing principle: log-based, append-only
**Nothing is deleted or destroyed — only added or learned.** It runs through everything:
the post-office log + cursor (history and progress decoupled; missed posts recovered by
catch-up, never lost), the FTS cache (rebuilt from transcripts, which are the source of truth),
`worktree-sessions.jsonl`, `events.jsonl`, and the wiki's own append-correction rule
([wiki-operating-model](/Users/jason/code/gomoku/wiki/topics/wiki-operating-model.md)). The
same epistemic stance as `TRAINING_WIKI.md`: record the negative result and how it happened;
trust compounds because the record only grows.

## How messaging the fleet actually works (what's possible)
- **Read the fleet:** `claude agents --json` (liveness) + `~/.claude/jobs/<ai-id>/state.json`
  (durable per-agent board, survives exit). The supervisor reaches live sessions over
  per-session rendezvous sockets — that's how agent-view "reply" delivers a message.
- **Write to a live session:** only interactive (agent view reply / Remote Control) or a
  launch-time Channel. There is **no supported CLI `send <id>`**; the rendezvous/control
  sockets are internal/undocumented. The post office is the supported-primitives way around it.
- **Continue a stopped session:** `claude -p --resume <id> "msg"` (context preserved). On a
  *live* session this interleaves the transcript — don't; fork or stop first.
- **Launch a persistent cagent:** supervisor-dispatched, so a human spawns it in agent view
  with `postoffice.py prompt` output. An orchestrator agent can only spawn its own
  lifecycle-bound subagents, not standalone fleet members.

## Implementation home
All of this lives in the **`agent-management` skill** (`.claude/skills/agent-management/`),
which is itself self-improving (friction log + self-improvement clause, after the
[research-lab](/Users/jason/code/gomoku/wiki/topics/research-lab-charter.md) pattern). The skill
*is* the self-improving agent for the too-many-sessions problem; each use sharpens it.

Related: [[cockpit-vs-autopilot]], [[conventions]] (§ Fan out to preserve context),
[[branch-and-worktree-workflow]].
