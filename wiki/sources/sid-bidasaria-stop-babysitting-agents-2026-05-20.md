# Source Record: "Stop babysitting your agents" — Sid Bidasaria (Code w/ Claude)

Video title: **Stop babysitting your AI and start orchestrating it**
(the talk itself is titled "Stop babysitting your agents")

Source URL: https://www.youtube.com/watch?v=wI0ptqCSL0I

Channel: Claude (official Anthropic channel) · Uploaded 2026-05-20 · ~37 min

Speaker: **Sid Bidasaria**, a founding engineer on the Claude Code team.

Retrieved: 2026-05-24.

Status: External talk transcript. This is the *correct* Sid talk — not to be
confused with "Building headless automation with Claude Code" (`dRsjO-88nBs`,
the Claude Code SDK / GitHub Action talk), which is a different Sid Bidasaria
session and was the wrong video on a first pass.

## How this transcript was made

YouTube has **no caption track** for this video (subtitles disabled / not yet
auto-generated four days post-upload), so the gemini MCP's `youtube_transcript`
and `analyze_youtube` tools — both of which read the caption track — could not
retrieve it. The audio was downloaded with `yt-dlp` and transcribed locally with
**Whisper on the M5 Max**. The raw ASR output lives at
`~/.claude/jobs/60c8a964/out/talk.{txt,srt,vtt,json}`.

Light cleanup applied to the raw ASR for readability:
- Whisper systematically heard **"Claude" as "Cloud"** ("Cloud Code", "Cloud MD",
  "multi-clotting"); corrected to Claude / Claude Code / `CLAUDE.md` /
  multi-Claude. Genuine "cloud" (remote compute) is preserved.
- Speaker name fixed ("Sidd Buddasaria" → Sid Bidasaria).
- Whisper looped on a few phrases during silent demo/slide stretches
  (e.g. "the session is going to be able to do it" repeated ~40×); those
  hallucinated repetitions were collapsed. No real speech was removed.
- A 4-minute third-party recap (`pKhj426i6B8`) was used as a cross-check.

Section timestamps below come from the Whisper `.srt`.

---

## Transcript

### Intro & table stakes — [0:02]

Good afternoon, everybody. My name is Sid Bidasaria. I'm one of the founding
engineers of Claude Code. And today I'm excited to talk to you about how you can
**stop babysitting your agents**.

As models have been getting smarter, I've noticed that we're increasingly
spending a larger percentage of our time staring at the screen, waiting for
Claude to finish its work, or just acting as a glorified QA tester for Claude.
That can be unsatisfying and an inefficient use of your time. My goal for this
talk is to give you strategies to take back some of that time so you can manage
your agents better. You can think of this as a more advanced Claude Code talk —
a Claude Code 301-type university class.

Because of that, there are some prerequisites — table stakes everyone here should
have at least heard about, if not implemented:

1. **A very high-quality `CLAUDE.md` file** — [1:26] — the single highest-leverage
   thing you can do to improve your Claude Code experience.
2. **Connecting your tools to Claude Code.** A good rule of thumb: if a tool is
   useful to you day-to-day, it'll be useful to Claude. Slack, Asana, Linear,
   Datadog, BigQuery — all of these help Claude stitch together a much richer
   context for itself and perform much better.
3. **Setting up your remote environment on Claude Code web.** This decouples the
   compute running Claude Code from your laptop. You can close your laptop, it
   could die, you could spill water on it, and your Claude Code sessions keep
   going because they're running in the cloud.

(Show of hands: nearly everyone uses Claude Code daily; ~50% have a high-quality
`CLAUDE.md` and connected tools; fewer have done all three. If you haven't done
any, you'll still get value — but start with these three.)

### Why your tooling needs to change — [3:12]

Most software tooling so far was built with humans in mind — linters, IDEs,
Prettier, type checkers, even compilers — all written to make humans and human
teams faster. The problem: humans aren't writing most of our code anymore.
Agents are. So we have to zoom out and reconsider our tooling.

Good news: a lot of the tools we built for ourselves translate over well —
Prettier, linters, symbol servers; Claude uses these effectively. Bad news: we
have blind spots. As humans we make assumptions about our toolchain that Claude
doesn't share. So the framing question for the rest of the talk is: **what does
an agent need from your codebase that a human takes for granted?**

### Roadmap — [4:50]

Three things that build on each other, and become incredibly powerful together:

1. **Verification** — teaching Claude to check its own work.
2. **Multi-Claude / parallelizing** — once Claude can check its own work and be
   more reliable, you can run many Claudes at once and be confident they're doing
   the right thing. — [5:28]
3. **Background loops** — taking your keyboard out of the hot path entirely, so
   Claude just keeps running in the background doing useful work for you.

### Verification — [5:47]

Quick brainstorm: think about the last software project or feature you worked on.
How did you check your own work — not just the final output, but how did you
*iterate* in a way that gave you confidence you'd end up where you expected?

Most software-engineering tasks break down into a series of steps: you design and
write code; you build it (compilers, type checkers) and loop back on failures;
you run the executable (a Docker container, a CLI app, a web server); you check
for side effects (spin up the browser and check UI elements, read logs, check the
database state); you run unit tests to catch regressions and add a new test for
what you built; and finally you deploy to staging — or, if you're brave, straight
to prod. **The same playbook works for Claude.** Throughout the rest of the talk,
think about teaching Claude to do things the way you would. All that's required is
giving Claude the right tools and instruction set.

### Loops are what make it go — [9:30]

This is arguably the most important slide. A **loop** is an autonomous circuit you
let Claude complete, so it can **hill-climb** on a task or success criterion. Give
Claude the tools to verify its own work and to write code: it writes code, checks
for a failure, debugs and writes more code, and repeats — again and again — until
it reaches a success state. When it finally gets there, you can be confident the
PR it sends you is higher quality and actually works.

Example: the sign-up button on my personal website stopped working. I told Claude
"make the sign-up button work." It wrote some code, built the app, opened a
browser and clicked the button, saw nothing happened, read the logs, found the
problem, fixed the code, reloaded, and kept looping until it succeeded — and came
back with a PR that worked. The takeaway: **wherever possible, get Claude into a
loop** by giving it the tools and instructions it needs.

### Verification comes in many flavors — [11:28]

UX verification, back-end verification, full end-to-end (including infra) — the
core concept is the same: give Claude the tools and instructions to get it into a
loop. Once you figure that out, all the flavors merge into one; you don't have to
be very specific, as long as Claude has the right tools and instructions.

### What it concretely takes — the four things — [12:18]

For a front-end / UX verification loop it boils down to about four things:

1. **Run your application** — e.g. `npm run start` to spin up your dev server.
2. **Use the app** — have Claude drive a browser. My MCP of choice is the
   Claude-in-Chrome MCP (`/chrome` in Claude Code); Playwright or other
   browser-control MCPs work too.
3. **Prove something works** — for a fix, screenshot before and after and confirm
   the right state.
4. **Unblock it** — production apps have blockers, commonly **auth** and **state**.
   Auth: give Claude an identity to log into your app. State: pre-configure state
   (e.g. populate inventory for an e-commerce store) so Claude can use the app
   meaningfully. This isn't novel — end-to-end tests already use state-setup
   scripts. The difference: give Claude access to those scripts and make them
   **dynamic** rather than prescriptive, so Claude can do a wider variety of things
   than static scripts allow.

### Packaging a verification loop as a skill — [14:39]

How do you distribute a verification loop to colleagues, or to your future self?
One of the best ways is a **skill** — a way to store arbitrary context about a
topic (here, a verification loop). Skills can be **self-improving**: instruct the
skill to improve itself every time Claude hits a blocker, and you get a
self-documenting, self-improving artifact the whole team contributes to. **This is
how the Claude Code team does verification** — a single verification skill,
explicitly told to keep documenting itself; every time someone hits a blocker, the
skill edits itself so the next person doesn't.

### Demo: MonkeyType — [15:57]

The demo app is **MonkeyType**, an open-source typing tester (TypeScript, Express
backend, MongoDB + Redis) — representative of a real-world full-stack app
(monkeytype.com). Plan: create a verification loop live — spin up a dev server,
use the Chrome MCP to check work, create a verification skill, then add a new
feature and have Claude use the skill to verify itself.

[17:10] Switching to the laptop: a brand-new Claude Code session, with MonkeyType
already set up locally, dependencies installed, and a curated `CLAUDE.md` (done
ahead of time to not waste your time).

[17:31] Tell Claude to spin up the dev server — it reports the server is already
running (started right before the talk). The MonkeyType front end opens; typing
shows a timer (lots of typos — "I'm not very good at typing"). The backend link
returns JSON, so the backend is up and running.

[18:43] Enable the Chrome MCP with `/chrome` — status shows "enabled, extension
installed" (otherwise it points you to a setup guide). Then: "use the Chrome MCP
to make sure the front end is working. Make it quick, please." Claude makes two
Chrome MCP tool calls — `Ctrl-O` shows it navigated to localhost:3000 and read the
content. Resizing the windows so the audience can see the background. "Can you try
typing and make sure everything works?" — Claude (also not a great typist) types
something and confirms typing works. "Can you also use the settings and change
something?" — it navigates to settings, changes difficulty to **expert** (not a
good idea given how it performed), and verifies the setting persisted.

[21:01] So far we held Claude's hand and told it exactly what to do. Next, package
it: "take everything we learned and put it into a skill file in
`~/claude/demo/verification`." Claude creates a new directory and writes a fairly
large `SKILL.md`: (1) bring up the stack (Docker Compose commands), (2) load the
Chrome MCP tools, (3) a smoke test that uses the browser tools to check its own
work. Creating a verification loop is genuinely simple — a few blockers came up
while building the demo, but you can usually get this running in five to ten
minutes.

[22:37] Now exercise the skill on a new feature: "every time I mistype, please
show me a confetti animation. And use the skill we just created to verify your
work." Switching to auto mode so it doesn't ask about every file edit. Claude
creates the feature, then realizes there are a couple of **ESLint** errors, fixes
them, and verifies itself again — **the verification loop in action**: wrote code,
hit issues, fixed them with more code, looped until a good state. The confetti
shows up (it put the test on expert mode, so it keeps disappearing). Running short
on time, so not letting it finish — but that's a taste of how powerful a
verification loop is, and how Claude hill-climbs given the right tools and
instructions.

[26:08] Key takeaway: **hold Claude's hand and show it how to do verification**;
once it knows how, it can summarize those learnings into a skill file you package
and distribute for your future self and your teammates.

### Multi-Claude / parallelizing — [27:28]

Now that we've mastered verification, we can graduate to running multiple Claudes.
The problem with too many at once: **they all eat your attention, and attention is
scarce.** Personally, more than four to five simultaneous sessions is a big load on
my brain. Four ways to scale:

- **The Claude Code desktop app** — [27:28] — a GUI that makes managing multiple
  sessions easier. A left sidebar shows **all your sessions across all surfaces**:
  local terminal, cloud, all Git repos — a central control plane. You can pin,
  rename, and color sessions; all of this is really about **protecting your
  attention** (rename a session to something memorable and you know what it was
  doing when you return). It also **sorts sessions by how much attention they need**
  — a session blocked on a permission prompt, a question, or input shows up at the
  top; running or completed sessions sit lower.
- **Agent View** — [27:38] — if you love the terminal (like I do), Claude Agents
  bring some of the desktop app's benefits into the terminal.
- **Claude Code on the web** — [28:03 area] — decouple your laptop from your
  sessions. I find it annoying to keep a laptop open walking meeting to meeting,
  or that there's no internet when I'm driving home. Running sessions in the cloud
  means you don't worry about the compute. Try it at **claude.ai/code**.
- **Remote control** — [28:03] — my favorite feature. Control any session on any
  surface **from your phone**. Run `/remote-control` wherever your session is and
  it pops up in the mobile app, with notifications — if Claude needs input, your
  phone buzzes and you can answer from your car or wherever you are.

(Skipping the Claude Agents demo for time — but give Claude Agents a try.)

### Background loops — [34:32]

We've made Claude more reliable (verification) and easier to run in parallel
(multi-Claude). But that's still not satisfying — you still have to spin up a new
session with a goal in mind. How do you remove yourself from the loop even more?

As engineers, a lot of our tasks aren't writing code for a feature or bug — a lot
is **bookkeeping**: babysitting PRs (with AI we generate far more PRs, and each
needs review comments, merge conflicts, and CI failures handled — 20–30 a day can
eat hours), updating docs, triaging, monitoring feedback, keeping CI green. These
need to happen daily but **don't need you in the loop** — they just need to run in
*some* loop.

- **`/loop`** — [34:32] — run a prompt at a set interval in Claude Code. E.g.
  `/loop 10 minutes` and "babysit my open PRs": the session wakes every 10
  minutes, runs the prompt, and — if your `CLAUDE.md` and tools are set up
  correctly — figures out what to do by itself. No manual babysitting.
- **Routines** — [35:16] — basically `/loop`, but running **remotely**, in the
  same remote containers as Claude Code on the web. Set them up from the web or
  desktop app's routines tab. Triggers can be **time-based** or **event-based**, and
  either opens a new Claude Code session with a specified prompt. The Claude Code
  team has a routine that updates our docs every day, and one that reviews incoming
  issues/feedback and posts to our Slack channel every six hours.

### Close — [36:26]

Stack all three skills together and you end up with a system that does a lot of
work without you manually on the keyboard. **That's the ultimate goal** — spend
your attention and time on the tasks you care about, and delegate everything else
to Claude with high reliability and confidence.

[36:59] That's all I have — thank you so much, and I hope you enjoyed the talk.
