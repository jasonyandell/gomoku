# Swap2 opening protocol (#72) — the real fix for white

**Status (2026-06-20):** built end-to-end and LIVE. The core bet is **confirmed at
the data level** (swap2 self-play makes white winnable: white wins 27% vs ~0% on an
empty board). Whether that balanced data makes the *net measurably stronger* is still
open — trained-vs-frozen-champion sits at parity (51.6%, n=64) after ~3 warm-started
slices. This page is the durable synthesis: why we did it, what we built, what we
learned, and what to try next. Evidence chronology lives in `TRAINING_WIKI.md`
(2026-06-20 entries); the predecessor analysis is
[white-side-defense-plan.md](white-side-defense-plan.md).

Branch: `feat/swap2-opening-protocol` (worktree, **unmerged** by request). Run: cell
`G15-swap2`, wandb `8nq1a7cm`, board 15, home `/Users/jason/data/swap2/`.

---

## 1. Why swap2 (the road here)

The 2026-06-20 white-defense investigation closed a long arc: 15×15 freestyle
white-side "defense weakness" is **not a net flaw — it is the first-player-win
theorem.** Evidence (full chain in white-side-defense-plan.md):

- Three policy/value teachers ALL flattened (value-only #42 poisoned the value head;
  sparse-VCF #43 and dense-conv left white at 0-2/20 vs Rapfi).
- The diagnostic showed white **retreats competently to a forced loss** — it blocks
  fours ~perfectly, has initiative on 1 ply / 30 games, is forced into an unstoppable
  double-four in 28/30. There is no tactical error to correct.
- **Clincher:** Rapfi(1s) vs Rapfi(1s) from 4-stone openings → white 1-9. Even the #1
  engine playing itself is crushed as the second player.

So no teacher can make a (near-)solved-lost role win. **The fix is to delete the
forced role: swap2** — Gomocup's balancing protocol, where a player is never *forced*
onto the lost side. That also makes the Rapfi yardstick honest (Rapfi is a swap2
engine).

## 2. The ML thesis (why this is a bootstrap fix, not just a yardstick fix)

An imbalanced game **cannot bootstrap**, because the self-play *data* collapses: when
one side is hopeless, every game is a first-player win, so the value head only ever
sees "second-player = lost" and the policy gets **zero gradient on winnable
second-player positions** — there are none in its own data. No teacher bolted onto the
side can fix a distribution that contains no balanced games.

Swap2 rebalances the **game** (the responder is never forced onto the lost side), so
self-play generates ~50/50 data → second-player positions become winnable in the
training set → the heads finally get real defensive gradient. The honest yardstick is a
bonus; the **data rebalancing is the mechanism.**

## 3. What swap2 is (the negotiation)

Cross-confirmed (Wikipedia "Gomoku", RenjuNet rule 11, Gomocup `SWAP2BOARD` protocol):

1. **Opener** places 3 stones = 2 black + 1 white, anywhere.
2. **Responder** chooses one of:
   - **STAY**: take white and place a 2nd white stone → 2B+2W, opener moves next as black.
   - **SWAP**: take black (no new stone) → 2B+1W, opener moves next as white.
   - **PLACE2**: add 1 black + 1 white → 3B+2W, then the **opener picks a color**; white
     moves next either way (the pick only decides *which player* is white — the board is
     identical).

There is **no canonical freestyle opening book** (the renju "26 openings" exist only
because renju restricts black; freestyle is unrestricted). Swap2 doesn't need one — the
**negotiation is the balancer**: an unfair opening is punished by the responder's choice.

## 4. The architecture (full Path A)

We chose the complete path — the net *can* learn to negotiate — but kept the scary part
small. Key insight: **almost all of swap2 is still placing stones on squares**; only two
moments are abstract decisions.

| Piece | File | Idea |
|---|---|---|
| Negotiation state machine | `gomoku/swap2.py` | `OpeningState` in **absolute** (black,white) planes (the opening isn't alternating-color). Color fixed by placement ORDER (B,W,B,…) so every placement stays a plain spatial move over the existing board policy head — **zero action-space growth for placements**. Only the two negotiation moments are abstract (width-3 choice space). Value attribution via explicit OPENER/RESPONDER actor tags + `backup_sign()` (the opener acts 3× in a row, so "flip perspective every ply" does NOT hold). |
| Choice head | `gomoku/model.py` | `forward_with_choice() → (policy, value, choice_logits)`; a tiny `Linear(value_hidden, 3)` off the value-head penultimate. The existing `forward()` and every caller stay byte-identical. Warm-start-tolerant load: core strict, fresh choice head spliced for the champion (which predates it). |
| Negotiator (v1) | `gomoku/swap2_search.py` | `negotiate(oracle, rng)` drives the opening. PLACE nodes **sampled** for diversity (not trained). CHOICE nodes selected by a **one-ply value comparison** (no trained head needed yet) with honest minimax over the nested opener pick; re-signed via `backup_sign`. ~30 net forwards/game (~0.2% overhead, no MCTS). Emits `choice_records` as future choice-head targets. |
| Self-play wiring | `gomoku/self_play.py` + worker/train/run_sweep | `--swap2` flag at the `_random_opening_state` seam in all four gen paths; mutually exclusive with `--random-opening-moves`; **byte-identical when off**. Each game starts from `negotiate().normal_state`. |
| Eval vs engine | `gomoku/eval_swap2.py` + `gomoku/external_engine.py` | Both sides negotiate: our net via `agent_act`, the engine via the `SWAP2BOARD` protocol path. The honest gate (no forced-white floor). |
| H2H gate | `gomoku/eval_swap2.py` `eval_swap2_h2h` | Net-vs-net swap2 (both negotiate), jobs-parallel, exact-deterministic. The resolvable progress gate (see §5). |

~12 commits, 100+ tests. The negotiation, protocol, choice head, and value attribution
are all tested in isolation before any net/MCTS touched them.

## 5. What we learned (the evidence)

### 5.1 The core bet is CONFIRMED at the data level ⭐
Measured color balance of 64 recent swap2 self-play games (`_records` GameRecords):
**white wins 27% (black 69%, draw 5%).** Empty-board self-play gives white ~0%. So
white is now genuinely **winnable in the training data** — the heads get gradient on
winnable white positions for the first time. This is the bootstrap an imbalanced game
can't do, working. It is the single most important result of the run.

### 5.2 The negotiation mechanism works
In net-vs-net H2H the **responder wins ~80%** — it exploits its stay/swap/place2 choice
to take the better side (`opener_color_dist` shows the responder almost always grabs
black). Swap2's balancing flows through the responder's choice, as designed.

### 5.3 Strength-vs-champion is still at PARITY (the open question)
Trained (e129) vs **frozen** warm champion, both negotiating swap2, n=64 → **51.6%**.
Within noise of 50% after ~2h of warm-started training. The net is **not regressing**
(it's ≥ its strong starting point) but has not pulled clearly ahead. Too early to call.

### 5.4 The data isn't perfectly 50/50 — and we know why
Black still wins 69% because v1 **samples** opening placements for diversity rather than
**training** them, so the opener never learns to place a *fair* opening and the responder
keeps a swap-to-black edge. This is the seam the next lever attacks (§6).

### 5.5 Healthy training dynamics throughout
`vl` bounces ~0.14–0.21 (NO value-poisoning collapse to 0.06 — the failure mode that
killed value-only #42). `plies` rose ~30→42 (more contested games — the *opposite* of
fast-attack collapse, where plies fall). No fast-attack collapse, no policy blowup.

### 5.6 Methodology lesson — the Rapfi gate is noise-dominated near the floor
Vs Rapfi-NNUE our net sits at single-digit-to-~30% win-rate; at n=16–48 the SAME fixed
baseline reads **4%–25% on pure noise** — variance swamps the ~5-8pt signal. Both an
encouraging "31%" and a scary "4% (worse than baseline!)" were noise; a serial
diagnostic showed trained 12.5% > baseline 4.2% at matched seed (no regression). **Gate
"did this help?" on H2H vs the preserved champion (≈p0.5, resolvable), NOT on Rapfi**
(this is the wiki's own 2026-06-15 rule). Rapfi stays a coarse absolute anchor only.

## 6. What to try next (ranked)

1. **Let v1 run and read the H2H trend first.** Don't add machinery ahead of the
   measurement (the exact lesson the teacher era taught). Gate trained-latest vs the
   frozen champ each cap. If H2H climbs past ~55-60%, v1 (balanced data alone) works. If
   it flatlines at ~50%, escalate to #2.
2. **Learned negotiation = the choice head into the loss (the deferred v2).** The choice
   head exists and the negotiator already emits `choice_records`, but those targets are
   **not yet in the trainer loss** — v1 negotiates by a one-ply value lookup. Wiring them
   in lets the net *learn* to negotiate. Risk surface: the record format + a choice-head
   loss term; build/test in isolation, deploy as a NEW cell (do not mutate a live run).
3. **Train opening PLACEMENTS for fairness.** v1 samples placements (untrained), so the
   opener can't learn a fair opening → the responder keeps the black edge (data stuck at
   69/27, not 50/50). Recording + training the opener's placement policy (punished when
   the responder profitably swaps) should push the data toward 50/50 and let white-side
   strength actually climb. Pairs naturally with #2.
4. **More warm-started training time.** 2h is short. The balanced data is in the buffer
   now; the value head is already fitting it (vl ↓ to ~0.14). Patience may convert
   balanced data into strength without new code.
5. **Bigger/sharper gates once a candidate looks real.** H2H n=128+ and a multi-seed
   Rapfi anchor at the strong tiers to put an absolute number on a confirmed H2H gain.

## 7. Operational notes (durable gotchas)

- **Warm-start = weights only.** Strip the champion to `{model_state_dict, model_config}`
  (drop the embedded buffer/optimizer/wandb) → fresh buffer + fresh optimizer (which
  covers the new choice-head params) + new wandb. Do NOT `--resume` the champion's 1.5M
  buffer — the white-defense era proved it drowns new signal. Use a **small fresh 150k
  buffer** so it turns over to the swap2 distribution fast (~3 min).
- **Worktree venv isolation.** `run_sweep` spawns subprocess workers/trainer that
  `import gomoku`; the shared editable install points at *main*, and native `.so`
  extensions live only in main's `gomoku/`. Give the worktree **its own venv**
  (`uv venv && uv pip install -e ".[dev]"`, which compiles the board-15 native ext into
  the worktree) so spawned procs import the worktree's swap2 code. Launch with
  `GOMOKU_BOARD_SIZE=15` and the worktree venv python (`PYTHON = sys.executable`
  propagates it to children).
- **Rapfi needs `x,y,color`.** SWAP2BOARD stone lines must be `x,y,color` (1=black,
  2=white); bare `x,y` → Rapfi errors and opens on an empty board. Drive the binary
  directly (`pbrain-rapfi --config <cfg> gomocup`) — the `run-rapfi` wrapper supplies the
  NNUE config but can lose its +x bit.
- **Eval must run during the PAUSE.** A parallel eval concurrent with the 8 gen workers
  oversubscribes the M5 and starves the net's CPU inference → contaminated numbers. Pause
  the trainer, eval on the free box, resume.
- **Babysit spine:** `/Users/jason/data/swap2/babysit/` holds the append-only `ledger.md`
  (slice/gate history), `run_eval.py` (vs-Rapfi), `run_h2h.py` (vs-champion). Slices are
  1h self-capping `run_sweep --max-wall-secs 3600 --resume latest.pt`.

## 8. Done / not done

**Done:** the full Path A pipeline (negotiation, choice head, negotiator, self-play
wiring, both eval harnesses), warm-started live run, the data-balance confirmation, the
H2H gate. **Not done:** a proven *strength* gain over the champion; the learned-choice-head
loss; trained opening placements; merge to main (held by request).
