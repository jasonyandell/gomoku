---
name: gomoku-slack
description: The Slack front door for the gomoku project — list channels, read the last active threads, send/thread messages, and spin up a channel for a bead. Wraps the "claude.ai Slack" MCP tools (mcp__claude_ai_Slack__slack_*) with the project's channels + the proven thread-per-bead watch-surface pattern used by gomoku-bead-runner. Trigger on "post to slack", "slack the team", "list the channels", "last N threads", "watch beads in slack", "make a channel for <bead>", "what's in #gomoku-beads", or any request to surface project status in Slack.
---

# gomoku-slack

The front door for everything Slack in this project. The bead-runner's watch surface lives here; this skill is the operational reference for it.

- **MCP server:** `claude.ai Slack` → tools are `mcp__claude_ai_Slack__slack_*`.
- **Your (Claude's) user_id:** `UA1G1GM7F` (use as a `channel_id` to DM Jason / yourself).
- **Workspace:** jasonjeffmike.slack.com.

## Channels
| Channel | id | Purpose |
|---|---|---|
| `#gomoku` | `C0B4G757NCX` | general project channel |
| `#gomoku-beads` | `C0B613WC97G` | **the bead watch surface** — one root message per bead = its thread |
| `#gomoku-bead-eda` | `C0B7EK7HXC0` | dedicated channel for the cross-game saga (`derby-eda`) — the worked example of "a channel for a bead" |

Re-list anytime: `slack_search_channels(query="gomoku")`.

## Core ops (exact commands)
- **List channels:** `slack_search_channels(query="gomoku")`.
- **Read a channel / last N messages:** `slack_read_channel(channel_id, limit=N, response_format="concise")` — use `concise` to save context.
- **Last active threads:** read the channel concise (root messages = thread anchors, newest first), or `slack_search_public(query="is:thread in:#gomoku-beads", sort="timestamp")`.
- **Read one thread:** `slack_read_thread(channel_id, message_ts)`.
- **Send a message:** `slack_send_message(channel_id, message)` → **capture the returned `message_ts`** (it's the thread anchor).
- **Reply in a thread:** `slack_send_message(channel_id, message, thread_ts=<root ts>)`.
- **Make a channel:** `slack_create_conversation(channel_name="gomoku-...")` → returns `channel_id` (use `is_private=true` to keep it off the workspace; `user_ids=[...]` for a DM/MPDM).
- **DM Jason / yourself:** `slack_send_message(channel_id="UA1G1GM7F", message=...)`.
- Markdown works: `**bold**`, `` `code` ``, lists, links, emoji (`◐ ✅ ⚠️ ⤴️ 🛠️ 🏁`).

## The thread-per-bead watch surface (the proven pattern)
Used by `gomoku-bead-runner`:
1. **One root message per bead** in `#gomoku-beads` = its thread. Capture the `ts`; persist the **bead → thread_ts map** to a file (e.g. `$CLAUDE_JOB_DIR/bead_threads.json`) so the loop survives context compaction.
2. **Status as thread replies:** `◐ IN PROGRESS` on dispatch → `✅ DONE` (with the merge commit) → or `⚠️`/`⤴️ DECLINED`/`🛑 escalation`. Watching a thread = watching that bead.
3. **Only post on a real state change.** Don't re-post derby lane-swaps or quiet ticks — that's noise. `@Jason` only for a true decision (a blocker, a gated cutover, a design-review escalation).

## Making a channel for a bead
Default is a thread in `#gomoku-beads`. Spin up a **dedicated channel** when a bead is a long-running saga worth its own space (worked example: `#gomoku-bead-eda` for the cross-game O(N) saga — 4 fixes + a pending live re-race). `slack_create_conversation(channel_name="gomoku-bead-<id>")` → post a kickoff summarizing the bead + linking prior history → use it as that bead's home going forward.

## Gotchas
- **Copy `message_ts` verbatim** from the send result. A mistyped `thread_ts` doesn't error — it posts a **stray top-level message** instead of threading.
- `response_format="concise"` on reads (channels can be long; protect context).
- **No delete/edit tool** — you can't unsend; get it right or post a correction reply. Drafts exist (`slack_send_message_draft`) if you want review first.
- Can't post to externally-shared (Slack Connect) channels.

## Friction-smoothing log

Things that bit us before, with their fixes. **Read this on session start; append after every session.** This is the part of the skill that compounds across runs.

### 2026-05-27 (the session that built the front door)
- **Mistyped `thread_ts` → stray top-level message.** A one-char typo in a bead's `thread_ts` posted the update as a new channel message instead of threading (no error raised). Fix: copy the `message_ts` verbatim from the `slack_send_message` result into the persisted map; never hand-type it.
- **Channel reads can blow context.** Default reads are verbose. Use `response_format="concise"` (and a small `limit`) for status pulls.
- **Lane-swap / quiet-tick posts are noise.** The derby runner swaps cells constantly; mirroring each one floods the watch surface. Only post on a real bead state change; suppress lane-swaps.
- **Persist the bead→thread_ts map to a file.** In-context maps are lost on compaction. `$CLAUDE_JOB_DIR/bead_threads.json` survives and lets any wake re-find every thread.
