# Sliding Derby — design (issue #38, 2026-06-16)

> **SUPERSEDED (v1 → v2):** the built methodology is
> [sliding-derby-measured-outcomes-design-v2.md](sliding-derby-measured-outcomes-design-v2.md).
> Keep this page as the **reuse-ledger of the v1 mechanics** (the grep-verified
> infra reuse map and the frozen-reference gate DNA), not as the live design.

> Async/pipelined, best-hypothesis-first, frozen-reference-gated. Grounded design from the sliding-derby-design workflow; ~80% reuse of existing infra. Status: DESIGN (not built). See #38.

# The Sliding Derby — concrete buildable design (issue #38)

> Status: design v1, 2026-06-15. Author: sliding-derby lead designer.
> Ethos: internal-healthy ≠ actually-strong. Every claim below was grep-verified
> against source (file:line cited); the gate is built to make a *silent
> regression un-shippable*, not to make the standings look good.
>
> **One-line thesis:** the Sliding Derby is `delo_derby.py`'s DNA (Δelo-rate
> priority, leaders-first, atomic board-json, `snapshot_peak`) re-wired from a
> *synchronous parallel scheduler* into an *async single-track pipeline* whose
> only net-new load-bearing parts are (a) a **frozen-reference gate wrapper**
> and (b) a **non-blocking cadence watcher (#34)**. Almost everything else is
> reuse.

---

## 0. Ground-truth verification (what actually exists, file:line)

Before designing on top of the grounding reports, the load-bearing claims were
checked against source. All confirmed:

| Claim | Verified at | Verdict |
|---|---|---|
| Lap engine: `run_sweep --max-wall-secs N --final-eval`, trainer self-caps on epoch boundary, clean resumable save, no cold restart | `scripts/run_sweep.py:2086` `launch_cell(... max_wall_secs, final_eval ...)`; supervise loop force-saves on cap + `TRAINER_CAP_GRACE_SEC` | **REAL** — lock-1 "training never stops" already satisfied by the cap-then-resume seam |
| Training-time internal eval is OFF by default (must stay off in lap configs) | `run_sweep.py` `launch_cell`: `internal_eval` opt-in, prints "internal-baseline eval DISABLED" | **REAL** — #34's "drop training-time evals" is already the default |
| Gate executor: model-vs-model paired-opening H2H, tight CI, no anchor ceiling | `scripts/delta_e_harness.py:687` `head_to_head_eval(...) -> HeadToHeadResult` with `wins/draws/losses`, `win_rate`, `delta_elo`, `delta_ci_half`; runs on CPU across `n_workers` | **REAL** — this is the gate's measuring instrument |
| Frozen peak: atomic `latest.pt → peak.pt` copy, pruning-proof | `scripts/delo_derby.py:616` `snapshot_peak(board, idea, elo)` → `os.replace(tmp, peak.pt)` | **REAL** — the freeze *primitive* exists |
| **freeze-peak-live / §10 / "AGZ anti-collapse brake" named in #38** | `grep -rniE 'freeze.?peak\|section.?10\|anti.?collapse\|frozen.?reference'` over `wiki/ scripts/ TRAINING_WIKI.md` → **ZERO hits**; `white-side-defense-plan.md` has §0–§5 + appendix, **no §10** | **DOES NOT EXIST** — the gate ORCHESTRATION is net-new |
| Async instruments: resumable panel JSONL (`--only` skips completed pairs), pure-analysis white-Elo (no GPU/torch) | `scripts/panel_tournament.py:122` `incremental=1` net specs; `scripts/panel_white_elo.py:380` `compute_dual_elo` → `black_elo/white_elo/elo_gap`, `white_loss_rate = white_l/white_games` (line 43) | **REAL** |
| N-way wrapper | `scripts/round_robin.py:26` imports `head_to_head_eval`, mean-centered ranking | **REAL** |
| Concurrent two-full-runs = ~half speed per run, aggregate flat | `wiki/topics/m5-max-cross-engine-coupling.md:155,168`: gen −45%/SGD −55% per run; aggregate gen +8%/SGD −9% | **REAL but for 20-proc dual** — the eval co-run is much lighter; lock-1 still needs its own measurement |
| Panel anchors invalid (#35): 17/36 pairs crash, negative affine slope | issue #35 body; `yixin18 0-30`, `Elo = -0.07*internal + 1876.9` | **REAL** — gate must NOT depend on calibrated absolute Elo |

**Design consequence of the last row:** the gate is built on the two
*calibration-free* signals — `head_to_head_eval` win-rate vs the frozen peak
(relative, tight CI) and `white_loss_rate` (a raw count ratio, immune to the
broken affine fit). Calibrated absolute Elo is reported but is **advisory only**
until #35 is fixed.

---

## 1. Architecture — the pipelined loop as a state machine

The Sliding Derby is a **single GPU track** (the lap) with an **async eval
sidecar** (the gate) and a **CPU re-ranker** (the priority engine). One process
owns the GPU at a time — this is the same "one GPU executor" invariant as the
classic derby (`wiki/topics/derby-registration.md:14`), so it never collides
with the existing derby-runner.

### 1.1 The slide, in one picture

```
   t ───────────────────────────────────────────────────────────────►
   GPU track:   [ LAP n: train hyp_n (run_sweep --max-wall-secs S) ]──cap──►[ LAP n+1: train hyp_{n+1} ]──►
   Eval sidecar:        [ GATE on LAP n-1's checkpoint: panel slice + H2H vs frozen peak + white-elo ]
                         (CPU + brief concurrent-at-half-speed GPU during the H2H arena window only)
   Re-rank (CPU):                                  [ read verdict → re-order backlog → pick hyp_{n+1} ]  ← BEFORE cap
```

The key invariant (**no idle GPU**): the gate + re-rank for LAP n−1 run *during*
LAP n's training window, so `hyp_{n+1}` is already chosen when LAP n hits its
wall-cap. The trainer self-caps and exits clean; the orchestrator immediately
launches LAP n+1 from the just-chosen hypothesis. No cold-buffer restart, no GPU
gap.

### 1.2 State machine (orchestrator: `scripts/sliding_derby.py` — NET-NEW thin driver)

States, transitions, and the EXACT reused script/flag for each:

```
        ┌────────────────────────────────────────────────────────────────────────┐
        │                                                                          │
        ▼                                                                          │
[PICK] ──pick #1 from ranked backlog──► [LAP_TRAIN] ──wall-cap hit / clean save──► [LAP_DONE]
   ▲   (re-ranker, NET-NEW;                  │  (run_sweep.py launch_cell,                │
   │    reads last verdict)                  │   max_wall_secs=S, final_eval=False)       │
   │                                         │                                            │
   │                              spawns ASYNC (does not block) ▼                         │
   │                                  [EVAL_PREV]  (gate on LAP n-1's latest.pt)           │
   │                                         │                                            │
   │                                         │  panel_tournament.py --only <net> (CPU)    │
   │                                         │  + panel_white_elo.py (CPU, no torch)      │
   │                                         │  + head_to_head_eval vs frozen peak.pt     │
   │                                         ▼  (brief concurrent GPU arena window)        │
   │                                    [GATE_DECIDE] (NET-NEW promote/revert wrapper)     │
   │                                         │                                            │
   │            ┌──── win_rate>50% & CI clear & no-black-regression ────► PROMOTE          │
   │            │     (snapshot_peak: new latest.pt → peak.pt)                            │
   │            └──── else ──────────────────────────────────────────► REVERT             │
   │                  (keep old peak.pt; mark lap NULL/REGRESSION)                          │
   │                                         │                                            │
   └─────────────────────────────────────────┴── verdict written to board.json ──────────┘
                                              │
                                  [RE_RANK] (NET-NEW; promote winning-lever children,
                                             demote null laps; choose hyp_{n+1})
                                              │
                                              ▼  must complete BEFORE LAP_TRAIN cap
                                          (loop to PICK for n+1)
```

State-by-state reuse map:

| State | What it does | EXISTING reuse | NET-NEW |
|---|---|---|---|
| **PICK** | choose #1 ranked hypothesis → resolve to a `run_sweep` cell | `run_sweep.py` cell registry; cell-clone pattern from `gomoku-derby-register` | re-ranker output read |
| **LAP_TRAIN** | train hypothesis as a time-capped slice | `run_sweep.py:2086 launch_cell(cell, max_wall_secs=S, final_eval=False, internal_eval=False)` — backgroundable via nohup, self-caps clean | — |
| **EVAL_PREV** | async-eval the *previous* lap's checkpoint | `panel_tournament.py --only <net> --n-games K` (resumable JSONL, skips done pairs); `panel_white_elo.py` (CPU); `head_to_head_eval` (`delta_e_harness.py:687`) | the **watcher** that triggers it on cadence (#34) |
| **GATE_DECIDE** | promote/revert decision | `head_to_head_eval` result (`win_rate`, `delta_ci_half`); `snapshot_peak` (`delo_derby.py:616`) | the **promote/revert wrapper** (§2) |
| **RE_RANK** | re-order backlog, pick next | `delo_derby.py` `pick_priority`/`delo_per_hr` math as a starting point; `write_research_board:642` for the standings rewrite | the **EXPECTED-LEARNING-PER-LAP scorer** (§4) + GitHub-issue read/write |

**Net-new surface is exactly three files** (kept small on purpose):
1. `scripts/sliding_derby.py` — the thin orchestrator/state-machine + board-json I/O (mirrors `delo_derby.py` structure).
2. `scripts/sliding_gate.py` — the frozen-reference promote/revert wrapper (§2). *The single most load-bearing new component.*
3. `scripts/eval_cadence_watcher.py` — the #34 non-blocking watcher (§3.2 / §7).

Everything else (lap, panel, white-elo, H2H, freeze, board rewrite) is reuse.

---

## 2. The frozen-reference gate (the un-shippable-regression mechanism)

**A freeze-peak/gate mechanism does NOT already exist** (verified: zero grep hits
for "freeze-peak-live", "§10", "anti-collapse brake"; the white-side plan stops
at §5). The freeze *primitive* (`snapshot_peak`) and the H2H *executor*
(`head_to_head_eval`) exist; the **orchestration around them is net-new** and is
specified minimally here.

### 2.1 What is frozen, and where

- The **frozen reference** is a single checkpoint: `peak.pt` in a derby-owned
  directory (`sweep_runs/sliding_derby/_peak/peak.pt`), written *only* by
  `snapshot_peak` (`delo_derby.py:616`, atomic `os.replace`). It is the
  current *champion* — the best weights that have *cleared the gate*, NOT merely
  the lowest-loss or highest-internal-Elo checkpoint.
- It is held **fixed across laps** until a challenger beats it head-to-head. This
  is the whole anti-collapse idea: a lap that *looks* healthy internally
  (loss down, plies up, internal-Elo up) but is actually weaker than the
  champion **cannot become the new reference** — it must win games against the
  frozen champion first.

### 2.2 The promotion rule (exact)

After LAP n−1's slice produces `candidate = latest.pt`, the gate runs:

1. **Primary match — challenger vs frozen champion.**
   `head_to_head_eval(fork_ckpt=candidate, c_ckpt=peak.pt, n_games=N_GATE,
   sims=S_GATE, opening_plies=4, n_workers=6, device="cpu")`.
   Default `N_GATE = 120` paired games (60 opening pairs × 2 colors), `S_GATE =
   200` sims. Paired random openings make the games independent and put two
   similar models near 50% — the maximally sensitive region, tight `delta_ci_half`.
2. **Accept iff ALL hold:**
   - `win_rate > 0.5` (challenger scores more than the champion), **AND**
   - `delta_elo - delta_ci_half > 0` (the +Δelo is **CI-clear of zero** — not a
     coin-flip; this is the noisy-small-n guard, §6 wedge #2), **AND**
   - **no-black-regression guard** (anti-tunneling for the defense arc): the
     candidate's `black_elo` from the panel slice has not dropped by more than a
     fixed `BLACK_REGRESS_TOL` (default 30 Elo) vs the frozen champion's. A
     white-side fix that wins overall but craters attack is NOT a clean promote.
3. **On ACCEPT → PROMOTE:** call `snapshot_peak` to atomically replace
   `peak.pt` with the candidate; record `verdict="PROMOTE"`, the Δelo, CI,
   `white_loss_rate`, `elo_gap` into the board. The new champion is the
   reference for all future gates.
4. **On REJECT → REVERT:** do nothing to `peak.pt` (the old champion stays
   frozen). Record `verdict="REVERT"` (or `"NULL"` if `|delta_elo| <
   delta_ci_half`, i.e. statistically no change) with the same metrics. The
   *candidate weights are preserved on disk* (lineage), but they are not the
   reference and a descendant lap is demoted in the re-rank.

### 2.3 Why this makes silent regression un-shippable

- The classic failure (`feedback_self_play_eta`, `feedback_absorption_phase`):
  internal metrics improve while real strength regresses (fast-attack collapse,
  white-side blindness). Internal Elo *saturates and clamps* — `delo_derby`
  itself notes "anchored elo saturates ~1700". A gate on internal metrics would
  promote the regression.
- The frozen-reference gate **never reads internal metrics for the
  promote/revert decision.** It reads *only* the result of the champion
  *playing the challenger*. A weaker net loses that match and is reverted,
  regardless of how good its loss curve looks.
- `white_loss_rate` (count ratio, calibration-free) is logged every gate so a
  defense regression is *visible* even when the overall H2H is close — it feeds
  the no-black-regression guard's sibling check and the re-ranker.
- Because the reference is *frozen* (not "best-so-far recomputed each lap from a
  noisy field"), the gate is immune to the #35 anchor poisoning: it is a
  two-player relative match, no Gomocup-Elo affine fit anywhere in the decision.

This is the AGZ pattern (the "anti-collapse brake" #38 gestures at): AlphaGo Zero
only replaced its self-play net when a candidate beat the current best by a
margin (≈55% in 400 games). Here: **promote only on a CI-clear >50% over the
frozen champion.** Same idea, our N and our metric.

---

## 3. GPU-contention model — eval LAP n−1 while LAP n trains

Lock-1: training **never stops**; the eval co-runs. Two ways to co-run; the
design uses a **hybrid** that keeps the GPU-contended window as short as possible.

### 3.1 The three eval components have very different GPU appetites

| Eval component | Device | GPU cost | When it runs |
|---|---|---|---|
| `panel_white_elo.py` (dual Bradley-Terry) | **CPU, no torch** | ZERO | reads JSONL after the arena; free, never contends |
| `head_to_head_eval` (challenger vs frozen peak) | **CPU pickers across 6 workers** | the GPU forks serially while CPU is "idle" (per the docstring at `delta_e_harness.py:706`) — *light* MPS use, not a full trainer | the gate's arena window |
| `panel_tournament.py --only <net>` (fixed-subset slice vs heuristic + lookahead-4 + ≥1 real engine) | mixed (net = MPS via brain wrapper; engines = wine subprocess) | *brief*, a fixed small `--n-games` subset only | the gate's arena window |

So the **only** GPU-contended part is the arena window (H2H + the small panel
slice), and even that is CPU-orchestrated MPS forks, not a second trainer's tight
back-to-back batch-512 SGD loop. The expensive `white_elo` step is pure CPU.

### 3.2 The co-run model (concurrent-at-half-speed, bounded window)

- The trainer (LAP n) keeps running at full priority. During the gate's arena
  window (minutes, not the whole lap), the H2H + panel-slice processes share the
  MPS. Per the measured concurrent behavior (`m5-max-cross-engine-coupling.md:155`),
  *two full runs* halve each other; but the eval is **far lighter** than a second
  full run (no 8 self-play workers, no sustained SGD — the gate is gappy,
  interleavable self-play-style occupancy, which the measurement shows is the
  *slack-filling* kind: aggregate gen was **+8%**, not −45%). Expected lap
  slowdown during the arena window is therefore **well under the 50% dual-run
  figure** — but this is **UNVALIDATED at 15×15 and MUST be measured first**
  (Ground C gap #4; see MVP step 0).
- **Trade-off chosen:** brief concurrent-at-half-speed during the arena window
  beats the alternatives:
  - *Stop training to eval* — violates lock-1, forces a cold-buffer restart cost
    on resume (the very thing the cap-then-resume seam avoids).
  - *Pure-CPU eval only* — `head_to_head_eval` already defaults to CPU pickers,
    but the net's MCTS still wants MPS for reasonable sims; forcing `device=cpu`
    for the *net* makes the arena slow (a lookahead:4 cycle is ~200–320s even
    alone, per `run_sweep.py` comment). Slow arena = the gate can't finish inside
    LAP n's window = the slide stalls.
  - *Hybrid (chosen)*: `white_elo` on CPU (free), H2H + panel slice with the net
    on MPS but a *small fixed `--n-games` subset* so the contended window is
    short. Tune `N_GATE` / panel-subset size so the gate reliably finishes well
    inside `S` (the lap wall-cap).

### 3.3 The hard contention guard (the invariant that protects the slide)

`gate_arena_secs ≪ lap_wall_secs (S)`. If a measured gate ever runs longer than,
say, `0.5 × S`, the orchestrator **shrinks `N_GATE`** (fewer games, wider CI) or
runs the arena on a **CPU-only net at reduced sims** for that lap, logging the
degraded-confidence flag. The slide must never block on the gate.

---

## 4. Priority engine — best-hypothesis-first + re-rank

North star is **LEARNING**, not Δelo alone (`feedback_learning_is_the_artifact`):
"a sharp measured verdict, even negative, scores high." So the score is
**Expected-Learning-Per-Lap**, not expected Δelo.

### 4.1 The score (greedy best-first)

For each open hypothesis (a GitHub issue tagged runnable-as-a-lap):

```
ELPL(hyp) =  evidence_strength
           × expected_abs_signal_change      ← magnitude of the measurable move (either sign)
           × prerequisite_unblocking          ← does landing it unblock other laps?
           × measurability                     ← can we read a clean verdict from existing instruments?
           ────────────────────────────────────
                 cost_per_lap                  ← wall-secs to a readable verdict (slice + gate)
```

- `evidence_strength`: how proven the hypothesis is *before* running. #36
  scores high — the diagnosis (#33) already PROVES defense must be taught
  (champion 0-6 white vs zetor17 / 6-0 black; FPU and 4×-search falsified).
- `expected_abs_signal_change`: **absolute** value of the expected metric move
  (`elo_gap` shrink, `white_loss_rate→0`, Δelo). A *negative* but *sharp* verdict
  still has large `|signal|` → high learning.
- `measurability`: 1.0 if existing instruments read it cleanly
  (`panel_white_elo` + H2H), lower if it needs uncalibrated absolute Elo
  (penalized while #35 is broken).
- `cost_per_lap`: config-only warm-start laps (clone champion cell + flip one
  flag, `gomoku-derby-register` pattern) are cheapest → highest ELPL all else
  equal.

Reuse: `delo_derby.py`'s `pick_priority` / `delo_per_hr` give the leaders-first
+ rate machinery; ELPL replaces the pure-Δelo numerator.

### 4.2 The re-rank rule (after each gate verdict)

When a gate verdict lands, re-score the backlog and re-order *before* the current
lap caps:

- **PROMOTE (winning lever):** promote the lever's **descendant** hypotheses to
  the top (e.g. #36 defense-teacher wins → promote #18 "stamp-the-saving-move"
  I2 descendant). The pipeline *follows the best hypothesis*.
- **REVERT/NULL (null lap):** **demote** that lever's children; pivot to the next
  independent high-ELPL lever (e.g. #36 null → pivot to #16 global-pool base or
  #26 WDL head). A null result still *raised learning* (it's written down) — it
  just doesn't spawn descendants.
- Re-rank writes the new order back to the GitHub backlog (label/priority) and to
  the board-json so the next PICK is deterministic and resumable.
- **Anti-tunneling reserve (see §6 wedge #4):** every K-th lap (default K=4) is
  forced to be the **highest-ELPL hypothesis from an *unexplored* lever family**,
  not the greedy #1. Greedy best-first without this tunnels into one lever and
  never tests the keystone-untried levers (#26 WDL head, #17 curated sampling).

### 4.3 Recommended first 3 laps (from Ground C, verified runnable)

- **Lap 0 — INFRA (no GPU contention): build the pipeline itself.** The #34
  cadence watcher + the frozen-reference gate wrapper (wire `head_to_head_eval` +
  `snapshot_peak` into promote/revert). This *is* the sliding derby; it unblocks
  every later lap. CPU/code-only — runs while any existing GPU work continues.
- **Lap 1 — #36 defense-teacher + VCT slice.** Highest evidence-strength
  (#33 proves it needed), config-only + cheapest (clone `G15-128x10-bigbuf`,
  warm-start `sweep_runs/g15_128x10_bigbuf_eval502.pt`, add `--defense-teacher`,
  swap vcf-teacher→vct-teacher — flags already wired in `selfplay_worker.py`).
  Measurable via `elo_gap` shrink + `white_loss_rate→0` with no black regression.
  **Doubles as the #37 death-spiral causal test — one lap, two verdicts.**
- **Lap 2 — re-ranked on Lap 1's verdict.** If defense-teacher **WINS** → run the
  #18 "stamp-the-saving-move" descendant. If **NULL** → pivot to #16 (global-pool
  base, P1 in-progress, low build) or #26 (WDL head, keystone-untried 15×15
  lever) per the evidence.

---

## 5. State / resumability / autonomy

Mirror the existing board-json pattern exactly (`scripts/derby_v9_board.json`):
prose `_doc` + a `global` block + an `ideas`/state list, atomic writes, a
`standings.md` rewritten below a sentinel by `write_research_board`
(`delo_derby.py:642`). The whole derby is a **function of files on disk** — kill
it anywhere, relaunch, it resumes.

### 5.1 `scripts/sliding_derby_board.json` (NET-NEW, same shape as v9)

```jsonc
{
  "_doc": "Sliding Derby — async/pipelined, frozen-reference-gated. <running narrative, dated>",
  "global": {
    "engine": "run_sweep_wall_slice",
    "slice_secs": 1800,                       // S = lap wall-cap (tune)
    "peak_path": "sweep_runs/sliding_derby/_peak/peak.pt",   // the FROZEN reference
    "gate": { "n_games": 120, "sims": 200, "opening_plies": 4,
              "accept": "win_rate>0.5 AND delta_elo-ci>0 AND black_regress<=30",
              "max_arena_frac": 0.5 },         // gate must finish in <0.5*S
    "rerank": { "scorer": "ELPL", "explore_every_k": 4 },
    "board_md_path": "sweep_runs/sliding_derby/standings.md"
  },
  "champion": {                                // who currently holds peak.pt
    "lineage": "g15_128x10_bigbuf_eval502 -> lap1-defense",
    "promoted_at_lap": 1, "white_loss_rate": 0.12, "elo_gap": 140
  },
  "laps": [                                    // append-only lap ledger (the write-everything-down spine)
    { "n": 1, "hyp_issue": 36, "cell": "G15-defense",
      "state": "EVAL_DONE",                    // PICK|LAP_TRAIN|LAP_DONE|EVAL_PREV|GATE_DECIDE|RE_RANK
      "ckpt": "sweep_runs/sliding_derby/lap1/latest.pt",
      "verdict": "PROMOTE", "delta_elo": 62, "delta_ci": 18,
      "white_loss_rate": 0.12, "started": "...", "capped": "..." }
  ],
  "backlog": [                                 // re-ranked each gate; PICK reads [0]
    { "issue": 18, "elpl": 7.1, "lever_parent": 36 },
    { "issue": 16, "elpl": 5.3 }, { "issue": 26, "elpl": 5.0 }
  ]
}
```

### 5.2 Crash-resume contract (what guarantees unattended operation)

- **Lap state** lives in `laps[*].state`; the trainer's own `latest.pt` (with
  embedded buffer) means a killed LAP_TRAIN resumes warm via
  `run_sweep ... --resume latest.pt` — no cold-buffer refill (the cap seam
  already does this).
- **Frozen peak** is a single atomically-written file (`snapshot_peak` uses
  `os.replace`) — never half-written, pruning-proof.
- **Gate verdict** is written to `laps[n]` only *after* the promote/revert
  completes; a crash mid-gate re-runs the gate (idempotent: `panel_tournament.py
  --only` skips completed pairs, H2H re-plays cheaply, promote is atomic).
- **Backlog ranking** is persisted in `board.backlog`; PICK is `backlog[0]`. A
  crash before RE_RANK just re-runs RE_RANK from the last verdict (pure function
  of the verdict + issue list).
- **Autonomy:** driven by the same cron/watchdog model as the existing derby
  (`gomoku-derby-runner`'s 10-min check loop) — a watchdog asserts exactly one
  `sliding_derby.py` PID + one cadence watcher, relaunches on crash, and
  PushNotifies on a *verified regression* (a REVERT verdict) — the only event a
  human might care about. Everything else proceeds unattended.

---

## 6. ADVERSARIAL wedge-check (top 5 ways it wedges or silently misleads)

Project ethos: a design must *survive attempts to break it*. The five highest-risk
failure modes, each with a concrete mitigation already wired into §1–§5.

**Wedge 1 — #35 panel unreliability poisons the gate.**
*Attack:* the gate trusts the engine panel; 17/36 pairs crash, anchors have a
*negative* affine slope, so absolute Elo is garbage → the gate promotes a
regression or rejects a good slice based on a broken yardstick.
*Mitigation:* **the promote/revert decision NEVER reads calibrated absolute
Elo.** It reads (a) `head_to_head_eval` win-rate vs the *frozen peak* (a
two-player relative match, no Gomocup anchors, no affine fit anywhere) and (b)
`white_loss_rate` (raw count ratio). The panel slice + `white_elo` are logged as
*advisory* context only. The gate is structurally immune to #35. (Fixing #35
upgrades the advisory column to calibrated, but the gate already works without
it.)

**Wedge 2 — the gate rejects a genuinely-good slice on noisy small-n eval.**
*Attack:* `N_GATE` games is small; a good challenger gets unlucky, `win_rate`
dips below 50%, a real improvement is reverted and its descendants demoted →
the pipeline tunnels *away* from the right lever on noise.
*Mitigation:* the accept rule is **CI-clear**, not bare >50%: promote needs
`delta_elo - delta_ci_half > 0`, and a result with `|delta_elo| < delta_ci_half`
is logged **NULL (no change)**, *not* REVERT — a null lap does **not** demote its
lever's children as hard as a measured loss does. Plus paired random openings
(maximally sensitive 50%-region, tight CI by construction —
`delta_e_harness.py:687` docstring). If `delta_ci_half` is too wide to decide,
the gate **escalates `N_GATE`** (more games) rather than guessing. Small-n is a
*hint*, never a verdict (`CLAUDE.md` ML-judgment rule).

**Wedge 3 — pipelining starves the GPU (the slide stalls, defeating the point).**
*Attack:* the gate's arena window co-runs with training and, at 15×15, the
concurrent slowdown is worse than the measured dual-run 50% → the gate doesn't
finish inside LAP n's window → RE_RANK is late → LAP n caps with no `hyp_{n+1}`
chosen → idle GPU, the exact failure #38 was built to avoid. The concurrent
co-run is also **UNVALIDATED at 15×15** (Ground C gap #4).
*Mitigation:* (a) **measure first** — MVP step 0 benches the two-GPU-job
slowdown at 15×15 before trusting lock-1. (b) The hard guard §3.3:
`gate_arena_secs ≪ S`; if a gate ever exceeds `max_arena_frac × S` the
orchestrator shrinks `N_GATE` / drops to CPU-net for that lap (degraded-confidence
flag) so the gate *always* finishes inside the window. (c) The heaviest eval step
(`white_elo`) is pure CPU (zero GPU), and the H2H is gappy interleavable
occupancy (the *slack-filling* kind the measurement showed at aggregate +8% gen,
not the colliding SGD kind). The slide is protected by construction: it degrades
gate *confidence* under contention, never *blocks*.

**Wedge 4 — greedy best-first tunnels and never explores.**
*Attack:* ELPL is greedy; it keeps picking descendants of the first winning lever
and never tests the keystone-untried levers (#26 WDL head, #17 curated sampling,
#16 global-pool). The derby looks productive but is exploring a single basin —
internal-healthy ≠ actually-strong, at the *portfolio* level.
*Mitigation:* the **forced-explore reserve** (§4.2): every K-th lap (default 4)
must be the highest-ELPL hypothesis from an *unexplored lever family*, not the
greedy #1. ELPL also rewards `prerequisite_unblocking` and `measurability`, so a
high-information untried lever competes even before its descendants exist. And a
*null* gate verdict actively demotes the current lever's children, forcing a
pivot. Greedy by default, but with a guaranteed exploration floor.

**Wedge 5 — eval cadence drifts the anchor / the frozen peak goes stale.**
*Attack 5a:* the cadence watcher re-registers nets over many laps; if the
*frozen peak* itself were re-evaluated each time, anchor drift (from #35's
instability) would slowly move the goalposts. *Attack 5b:* the frozen peak is
*too* sticky — a series of small real gains each individually fail the CI-clear
bar, so the champion never updates and the pipeline stops learning (false
plateau).
*Mitigation 5a:* the frozen peak is a *fixed checkpoint file*, never re-rated —
the gate is `candidate vs peak.pt` (relative), so anchor drift in the *advisory*
panel cannot move the gate's decision. The reference only changes on a clean
PROMOTE. *Mitigation 5b:* PROMOTE replaces the peak, so genuine cumulative gains
*do* update it; and the **stale-peak watchdog** — if M consecutive laps (default
M=6, same as v9's `peak_window`) all NULL/REVERT against the same frozen peak,
the orchestrator escalates: it re-runs the gate at a larger `N_GATE` (resolve
near-50% ties) and, if still no promote, **PushNotifies a "champion plateaued"
event** for a human look — exactly the `chunks_since_new_peak` plateau signal the
existing derby-runner already triages. A true plateau is *signal* (write it
down), not a silent stall.

---

## 7. Minimal first version (MVP) — the smallest genuine sliding derby

The MVP is **one real lap that slides**: a single `run_sweep` slice whose
checkpoint is gated against a frozen reference by an async watcher, with the
verdict written down. No multi-lap re-ranker yet (manual next-lap pick is fine
for v1) — but the *gate* and the *non-blocking eval* MUST be real, because they
are what make it a *sliding* derby rather than the old parallel one.

### MVP build checklist

- [ ] **Step 0 — measure lock-1 (de-risk wedge 3).** Bench the two-GPU-job
      slowdown at 15×15: a `run_sweep` slice + a concurrent `head_to_head_eval`
      arena. Confirm `gate_arena_secs < 0.5 × S` with a chosen `N_GATE`. Pure
      measurement, code-only, no new code. (Ground C gap #4.)
- [ ] **Step 1 — `scripts/sliding_gate.py` (the load-bearing net-new).** Wrap
      `head_to_head_eval` (`delta_e_harness.py:687`) + `snapshot_peak`
      (`delo_derby.py:616`): given `candidate.pt` + `peak.pt`, play `N_GATE`
      paired games, apply the §2.2 accept rule (`win_rate>0.5 AND
      delta_elo-ci>0 AND black_regress<=tol`), PROMOTE via `snapshot_peak` or
      REVERT, return + persist the verdict. Unit-test the decision logic on
      synthetic results (forced win / forced loss / coin-flip → NULL).
- [ ] **Step 2 — `scripts/eval_cadence_watcher.py` (#34, non-blocking).** Watch
      a checkpoint dir; every ~100 epochs (or every cap) register the new ckpt
      via `panel_tournament.py --only <net> --n-games K` (fixed subset:
      heuristic + lookahead:4 + ≥1 stable real engine), run `panel_white_elo.py`,
      append an `epoch/black_elo/white_elo/white_loss_rate` row, then invoke
      `sliding_gate.py`. Separate process; never blocks the trainer. Confirm
      training-time internal eval stays OFF in the lap config (it already is by
      default in `launch_cell`).
- [ ] **Step 3 — seed the frozen peak.** `snapshot_peak`-copy the reigning
      champion (`sweep_runs/g15_128x10_bigbuf_eval502.pt`) to
      `sweep_runs/sliding_derby/_peak/peak.pt` as the initial frozen reference.
- [ ] **Step 4 — run Lap 1 (#36 defense-teacher) through the pipe end-to-end.**
      Clone `G15-128x10-bigbuf`, warm-start eval502, add `--defense-teacher` +
      vct-teacher, launch `run_sweep --max-wall-secs S` (NOT `--internal-eval`),
      let the watcher gate it vs the frozen peak, write the verdict + the #37
      death-spiral read to the board + `TRAINING_WIKI.md`.
- [ ] **Step 5 — `sliding_derby_board.json` + standings.** Minimal board-json
      (§5.1) + reuse `write_research_board` for `standings.md`. Crash-resume
      verified: kill mid-lap, relaunch, it resumes warm.

### Deferred past MVP (explicitly out of v1)

- The **automatic ELPL re-ranker** that reads GitHub issues and re-orders the
  backlog (§4) — v1 picks the next lap *by hand* from the §4.3 recipe; automate
  once the gate is trusted over several laps.
- The **forced-explore reserve** and **stale-peak watchdog** (§6 wedges 4 & 5) —
  needed for long unattended runs, not for proving the pipe.
- **Multi-lap GPU-overlap polish** — MVP can tolerate a small GPU gap between
  laps; the no-idle-GPU "choose `hyp_{n+1}` before cap" optimization comes after
  lock-1 is measured and the gate is trusted.
- **#35 panel repair** — the gate is built to NOT need calibrated anchors, so #35
  is a *parallel* upgrade (advisory column → calibrated), not an MVP blocker.

---

## Appendix — the reuse-vs-net-new ledger (one glance)

| Piece | Status | Where |
|---|---|---|
| Time-capped lap (no cold restart) | REUSE | `run_sweep.py:2086` `launch_cell(max_wall_secs, final_eval)` |
| Training-time eval OFF by default | REUSE | `run_sweep.py` `launch_cell` internal_eval opt-in |
| Gate match executor (H2H, tight CI) | REUSE | `delta_e_harness.py:687` `head_to_head_eval` |
| N-way field | REUSE | `round_robin.py:26` |
| Atomic frozen-peak primitive | REUSE | `delo_derby.py:616` `snapshot_peak` |
| Resumable panel slice (skip done pairs) | REUSE | `panel_tournament.py:122` `--only`, `incremental=1` |
| White-side reader (CPU, no torch) | REUSE | `panel_white_elo.py:380` `compute_dual_elo`, `white_loss_rate` (l.43) |
| Δelo-rate priority math, board rewrite | REUSE | `delo_derby.py` `pick_priority`/`delo_per_hr`, `write_research_board:642` |
| Board-json + `_peak/` state pattern | REUSE | `derby_v9_board.json` shape |
| Cron/watchdog autonomy loop | REUSE | `gomoku-derby-runner` 10-min check model |
| **Frozen-reference promote/revert wrapper** | **NET-NEW** | `scripts/sliding_gate.py` |
| **Non-blocking cadence watcher (#34)** | **NET-NEW** | `scripts/eval_cadence_watcher.py` |
| **Orchestrator state-machine + ELPL re-ranker** | **NET-NEW** | `scripts/sliding_derby.py` |
