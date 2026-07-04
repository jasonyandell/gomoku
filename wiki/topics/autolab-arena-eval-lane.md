# The Arena Eval Lane — register a model, run the gamut, read a relative Elo

> **Status: DORMANT** *(2026-07-04)* — design-of-record for the measurement leg;
> the arena lane ran live 2026-06-19 (crowned the first champions), autolab
> stopped. Under the [Autolab hub](../autolab.md).

**What this is.** The design note for the autolab's *measurement leg* — the third
member of the **trainer trains · researcher researches · arena evals** triad. It
answers, in the autolab's own idiom: *what does it take to hand the arena a model
and get back a clean, deterministic, relatively-calibrated strength number?* It is
the sibling of [autolab-researcher-contract.md](autolab-researcher-contract.md) (the
smart lane) and it elaborates the *Arena-yardstick gap* section of
[autolab-architecture.md](autolab-architecture.md) from "a non-gating Rapfi readout"
into a **reusable, model-agnostic arena**.

The one-sentence thesis, in the doctrine's voice:

> **The arena is the protected evaluator: a pure, seeded harness that plays a
> registered contestant against an immutable reference panel and folds the games
> into a relative Elo with a confidence interval — the only authoritative source of
> evaluation evidence in the lab.**

The headline reframe — what the arena does *not* try to be:

> Not an **absolute** Gomocup-Elo yardstick (that path is dead — #35 proved published
> ratings invalid under our harness). A **relative** Elo, pinned to a fixed,
> *measured* anchor set, that means the same thing next week as it does today.

---

## Two jobs, not one: the `gate` and the `panel`

The arena lane already exists (P4, `gomoku/lab/arena.py`) but does exactly one thing:
a fast head-to-head **gate** vs the HF `champion` tag (`run_gate(dry_run=True)` →
`PROMOTE`/`REVERT`/`AMBIGUOUS`, calibration-immune, ~12–40 games, shrinks under a live
trainer). That is the right *promotion* mechanism and it does **not change.**

What's missing is the **gamut** — the measurement. So the arena gets a *second job
kind*, distinguished by `config.eval_kind` on the arena experiment row:

| Job kind | Question | Opponent(s) | Output row | Cadence | Decisive? |
|---|---|---|---|---|---|
| **`gate`** (built) | "is this better than the champion?" | the reigning champion | `verdict` (PROMOTE/REVERT/AMBIGUOUS) | every slice (flywheel) | **yes** — moves the champion tag |
| **`panel`** (this note) | "where does this model sit on the scale?" | the fixed reference panel | `eval` (relative Elo + CI + per-color split) | coarse — on PROMOTE / on request / every N | **no** — a measurement, surfaced never gated |

Keeping them separate is the load-bearing design choice. The gate must stay cheap
(it runs every slice, co-tenant with training); the gamut is heavier and runs at a
coarser cadence. Conflating them would make every slice pay the full panel cost.
Authority stays clean too: a `verdict` is a *decision*, an `eval` is a *measurement*;
both are produced **only** by the arena role (§ Protected instrument).

---

## 1. The contestant contract — the clean simple scaffold ("add a model")

A model enters the arena as **a Gomocup/Piskvork-protocol engine.** That protocol is
the universal seam: it is already how Rapfi, our own nets, and any external engine all
play through the *same* harness (`gomoku/external_engine.py` is the reference client;
`gomoku/gomocup_brain.py` the reference brain). Registration is one `Contestant` spec —
the existing `gomoku/match.py` player grammar, formalized as the arena's unit of intake:

```
Contestant = external:cmd=<argv>,timeout_ms=<ms>,label=<name>,size=<N>,incremental=<0|1>
```

There are exactly **two on-ramps**, and the boundary between them is where Jason's
"performance may depend on the model implementer" lives:

- **(a) "I have a checkpoint."** Hand us `checkpoint + sims`. We wrap it with the
  shipped brain — `run-gomoku-az --checkpoint X --sims N` — and you do nothing else.
  Zero protocol work. This is the common case (our own training lineage).
- **(b) "I have my own engine / inference."** Implement the protocol on stdin/stdout
  (`START` → `BOARD…/DONE | BEGIN | TURN x,y` → reply `X,Y`) and give us the `cmd`.
  **You own your inference speed**; we own the harness, the openings, the scoring, and
  the determinism rules. A slow engine simply costs more wall-clock — the arena never
  subsidizes it, it just measures you at *your* configured operating point.

**Three scaffold rules the implementer must honor** (each one is a scar made into a
contract):

1. **History-conditioned nets MUST register `incremental=1`.** Our net carries
   `HISTORY_PLY=8` recency planes; driven by plain BOARD-replay every move it rebuilds
   an *empty-history* input and silently sandbags itself (measured 100% → 25% on a
   fixed checkpoint). `incremental=1` feeds each opponent move as `TURN x,y` so history
   accumulates faithfully. Classical engines (no history) keep the default BOARD path.
2. **Reply within `timeout_ms`.** The harness enforces a hard read ceiling
   (`max(read_timeout_s, timeout_ms/1000 · read_timeout_slack)`); blow it and you
   forfeit the move. Your *operating point* (sims for a net, time/nodes for an engine)
   is part of your identity in the panel — it is recorded with every result.
3. **Pass SMOKE before the gamut.** A contestant is admitted only after a SMOKE check
   (handshake + one legal in-range move + a tiny match vs `heuristic` that completes).
   This is the cheapest tier of the [maturity ladder](autolab-researcher-contract.md);
   a broken engine is rejected before it can waste the panel's wall-clock.

---

## 2. The panel — the immutable reference yardstick ("the gamut")

"Run the gamut" = play the contestant, balanced-opening and color-alternated, against
**every member of a fixed reference panel.** Default composition (configurable, but the
default *is* the yardstick):

| Member | Role on the scale | Why it's stable |
|---|---|---|
| `heuristic` | the **floor anchor** (pin the scale here) | pure-python, torch-free, *bit-reproducible* forever |
| `lookahead:depth=4` | a mid anchor | pure-python, deterministic given seed |
| `rapfi@<measured-TC>` | the **strong anchor** | native arm64 NNUE; pinned at a *measured* operating point, not a published Elo |
| `champion-{board}` | the relative-gate reference | the reigning HF champion (moves over time — *not* an anchor) |
| *(optional)* 1–2 lineage checkpoints | continuity | preserved revisions for cross-era anchoring |

The **anchors are a protected instrument** (§ below): their composition and operating
points are versioned as a `panel_id` and never silently changed. The `champion` is a
*reference*, not an anchor — it moves, so it never defines the scale.

---

## 3. The number — a relative Elo, anchor-pinned (not mean-centered)

The cross-table of `{contestant} ∪ {panel}` is fit with the **Bradley-Terry / MM
solver already in `scripts/panel_tournament.py`** (`fit_bradley_terry`, logistic on the
400/log10 Elo scale, L2-regularized). That function returns ratings **mean-centered to
0** — which is *wrong for a stable relative scale*, because the mean drifts every time a
contestant joins or leaves. The arena adds one post-step:

> **Pin the scale to an immovable anchor, not the mean.** After the fit, shift all
> ratings by a constant so `heuristic ≡ 0` (or `rapfi ≡ its measured reference). Then a
> contestant's number is comparable across runs and across time — the whole point of a
> *relative* Elo.

The `eval` row carries the readout, not a recomputable handle:

```jsonc
{ "type": "eval", "ref": "<arena-row-id>", "model": "<contestant-ref>",
  "panel": ["heuristic", "lookahead4", "rapfi-200ms", "champion-15", "<contestant>"],
  "metrics": {
    "panel_id": "g15-v1",                 // the immutable yardstick identity
    "relative_elo": 312.4, "ci": [248, 377],
    "per_opponent": {                      // the white-defense story is first-class
      "rapfi-200ms": {"overall": 0.21, "black": 0.42, "white": 0.0},
      "heuristic":   {"overall": 0.95, "black": 0.97, "white": 0.93} },
    "operating_point": "sims=400",
    "n_per_pair": 24, "seed": 7 } }
```

The **per-color split is not optional** — white-side defense is the binding constraint
on the whole project (#46/#43/#37); the panel surfaces `Δwhite-elo/Δt` (the #34
north-star) directly, because an aggregate Elo hides exactly the gap that matters.

---

## 4. Determinism — the honest two-layer answer

"Deterministically" splits into two layers, and only conflating them makes it look
impossible:

**Layer A — harness determinism (we own this; it must be exact).** Which games, which
openings, which seeds, the color-balancing, the scoring, the Elo math. All of it is a
**pure, seeded fold**: `play_match_pickers` seeds one `np.random.default_rng(seed)`,
derives the balanced openings from it (shared opening per color-swap pair), and threads
it through every move; MCTS in eval uses **no Dirichlet root noise**, so a net's move is
a deterministic function of `(state, seed, sims)`. Given `(contestant, panel_id, seed)`
the arena requests the *same set of games* and folds them with the *same math* every
time. This is the autolab's "deterministic fold" property, and it holds end to end.

**Layer B — engine-move determinism (mixed; be honest about it).**

- **Our nets and the pure-python baselines are sims/depth-budgeted → deterministic and
  machine-independent.** Same seed, same machine *or any other*, same games. These
  define the reproducible core of the panel.
- **Time-budgeted external engines (Rapfi) are wall-clock → NOT bit-reproducible.**
  Rapfi searches to `timeout_turn · advanced_stop_ratio`; a faster or less-contended box
  searches deeper in the same millisecond budget. There is no fixed-node knob in its
  config today.

The resolution is on-doctrine and matches the #35 lesson ("measure, don't assume"):

> Deterministic where we can, **pinned-and-CI'd** where we can't. A time-budgeted engine
> is treated as a **fixed *operating point*, not a deterministic function**: pin a named
> TC (`rapfi-200ms-1thread`), run it **uncontended** (idle-GPU window / the co-tenancy
> guard), and treat each panel result as a *measured sample with a Wilson CI*, cached.
> Prefer a node/depth budget over a time budget for any engine that supports one. The
> ledger fold over the *results* is always deterministic — the row carries the number,
> the fold just replays it.

So "deterministic" means: **bit-identical for the net/baseline core; a reproducible
distribution at a pinned operating point on an uncontended box for time-budgeted
engines** — and always a deterministic fold of whatever games were recorded. The
[loop simulator](autolab-doctrine.md) already cares about loop determinism, not engine
internals, so this is exactly the line it certifies.

---

## 5. Performance — the cached baseline + the parallel lane

Three levers, none of which require the implementer to do anything but be fast at their
own inference:

1. **Cache the panel's internal cross-table — O(panel), not O(panel²).** The panel
   members' *pairwise* results (rapfi vs heuristic vs lookahead vs champion) don't change
   for a fixed `panel_id` → compute them **once** and store a content-addressed
   `panel-baseline-<panel_id>.jsonl` under `~/data/autolab/arena/`. Adding a contestant
   plays only `{contestant × each member}` and folds it on top of the cached baseline.
   This is exactly `panel_tournament.load_existing`'s skip-already-played resume,
   promoted to a first-class immutable artifact. (When the `champion` tag moves, only the
   champion's column is recomputed — the anchors stay cached.)
2. **CPU-parallel while the GPU trains.** External engines are CPU + single-thread; the
   member-vs-member and engine-vs-engine pairs run across the M5's cores as the
   *everything-else lane* while the trainer holds MPS. Net-vs-net pairs need MPS for MCTS
   → those run in the idle-GPU window or under the existing co-tenancy guard (shrink
   `n_per_pair` while a slice is live), same as the gate.
3. **The implementer owns inference speed; we own the harness.** The scaffold gives you
   the protocol client, the openings, the determinism rules, and the timeout contract. A
   slow contestant just costs more wall-clock at its own operating point — we measure,
   we don't optimize your engine for you. (This is the deliberate scope line in Jason's
   ask: "that part may depend on the model implementer; we provide a clean scaffold.")

---

## 6. How it fits the lane (organized like the rest of the autolab)

Same `fold → pick('arena') → run_chunk → append` shape as today — the panel is a new
branch *inside* `run_chunk`, selected by `config.eval_kind`:

- The **trainer flywheel** already enqueues an `arena` row per slice (`role="arena"`,
  `base=<candidate ref>`, `config={lane, model_elo}`). That stays a `gate`. A **`panel`**
  row is enqueued at a coarse cadence — on PROMOTE, every N slices, or **on demand from
  the researcher** (see next).
- **The panel feeds the researcher's epistemic WHEN.** The
  [researcher contract](autolab-researcher-contract.md)'s `evidence_contract` already
  supports a `{"kind":"arena-verdict"}` requirement. A `panel` `eval` row *is* that
  evidence — so a research thread whose contract reads *"2 slices **and** a panel
  readout"* now has a real, deterministic producer for the second clause. This closes the
  triad: **trainer → arena (gate + panel) → researcher**, all over one ledger, zero glue.
- **The cockpit** reads the `eval` rows for the one-glance board: relative Elo + CI +
  white-rate vs the strong anchor, per lane, alongside the gate verdict. `Δwhite-elo/Δt`
  becomes a column.

No new spine, no new pick policy, no new state — a second `run_chunk` branch and a new
`eval_kind` discriminator. That is the "just some scripts, organized like the autolab"
shape Jason hoped for.

---

## 7. The panel is a protected instrument (three-zone governance)

Per the researcher contract's three-zone model, the **panel definition is a protected
instrument**, not autonomous-science sandbox:

| Zone | Arena element |
|---|---|
| Autonomous science | a new *contestant*, a new operating point to *try* |
| Adaptive policy | the *cadence* of panel runs (every N / on-promote), the co-tenancy shrink factor |
| **Protected instrument** | the **panel composition + anchor operating points + the cached baseline + the Elo anchor-pin rule** |

A model — or a researcher-Claude — may *enter* the arena and *be measured*; it may
**not** silently change the yardstick it is measured against. Changing `panel_id` (adding
an anchor, moving an operating point, re-pinning the scale) is a separately-gated,
versioned, human-visible act — exactly the "can't move the walls your own score depends
on" rule. Every `eval` row stamps its `panel_id`, so a number is never ambiguous about
which yardstick produced it.

---

## 8. Sim invariants to add (certify the wall, RED-when-off)

Each guarantee above becomes a falsifiable assertion in `tests/test_lab_sim.py` (per
house practice — turn the fix off, confirm the scenario goes red):

- **`inv_eval_is_arena_only`** — only the arena role may append an `eval`/`verdict` row;
  an LLM/`compile_intent` path cannot forge one (extends the existing typed-intent wall
  to the panel readout).
- **`inv_panel_complete_or_blocked`** — a `panel` eval cannot silently omit a required
  panel member; a missing member yields BLOCKED, not a quietly-smaller panel.
- **`inv_relative_elo_anchor_pinned`** — perturbing the *contestant* set never moves an
  *anchor*'s Elo (falsifies the mean-centering bug; proves the scale is stable).
- **`inv_panel_baseline_immutable`** — a contestant run appends only `contestant × member`
  pairs; it never rewrites a `member × member` pair of a fixed `panel_id` (idempotent;
  same `panel_id` + seed → same requested pair set).
- **`inv_eval_fold_deterministic`** — re-folding a `panel` eval row yields the identical
  readout (the number lives in the row; the fold replays, never recomputes).

---

## 9. Build order

1. **`eval_kind` discriminator + the `panel` branch of `ArenaRole.run_chunk`** — reuse
   `play_match_pickers` for the contestant×member pairs; emit the `eval` row. *Highest
   value; it's the gamut.*
2. **The cached `panel-baseline-<panel_id>.jsonl`** — content-addressed by
   `(members, operating_points, board_size, n_per_pair, seed)`; the O(panel) lever.
3. **Anchor-pinned Elo** — the one-line post-step over `fit_bradley_terry`; pin
   `heuristic ≡ 0`.
4. **The contestant SMOKE entry** — one command: handshake + legal move + tiny
   vs-heuristic match; the admission gate.
5. **The `arena-verdict` → `panel` wiring** — `delta_e_harness.ExternalAnchor.play()`
   (currently `NotImplementedError`) lifted from `eval_vs_rapfi`, with the per-color
   split; the panel becomes a producer of researcher evidence.
6. Each of the above lands **with its sim invariant** (§8).

**Deferred until traces justify** (matching the architecture's shelved-calibration
stance and the researcher contract's anti-schema-zoo discipline): absolute
Gomocup-Elo calibration (#30/#35 — dead path), a full multi-anchor affine fit,
auto-discovery of new external engines, and any "panel tournament UI" beyond the
cockpit board.

---

## Status — DESIGN (2026-06-24, in `feat/autolab-sim`)

Design note only; nothing built yet. Every composition point exists and was contract-
verified: `run_gate` (the gate, untouched), `play_match_pickers` (seeded, balanced
openings, per-color split, deterministic MCTS), `fit_bradley_terry` (the BT/MM Elo fit),
`external_engine.py` + `gomocup_brain.py` (the protocol seam), `panel_tournament.load_existing`
(the resume/cache mechanism), the native Rapfi anchor, and the `eval`/`verdict` ledger
constructors. The work is **composition, not invention** — which is the point.

The single real tension surfaced and resolved here: **time-budgeted external engines are
not bit-reproducible** (§4). Everything else is the autolab's existing shape applied to
the measurement leg.

---

## Cross-refs
- [autolab-architecture.md](autolab-architecture.md) — § *Arena-yardstick gap* (the
  4-step Rapfi-readout plan this page generalizes) + the arena lane (P4).
- [autolab-researcher-contract.md](autolab-researcher-contract.md) — the sibling lane;
  the `arena-verdict` evidence kind this page produces; the three-zone governance + the
  maturity ladder reused here.
- [autolab-doctrine.md](autolab-doctrine.md) — protected-evaluator authority; the loop
  sim certifies the wall, not the engine.
- [engine-panel-derby-design.md](engine-panel-derby-design.md) — the cross-table runner +
  the #35 broken-anchor lesson (why relative, not absolute).
- [external-engine-baselines.md](external-engine-baselines.md) — the Rapfi wrapper +
  match harness + the measured operating points.
- [white-side-defense-plan.md](white-side-defense-plan.md) — why the per-color split is
  first-class (the binding constraint the panel measures).
- Issues: **#59** (arena P4, built) · **#61** (researcher lane, the evidence consumer) ·
  #30/#35 (engine panel, calibration shelved) · #34 (Δwhite-elo/Δt north-star).
