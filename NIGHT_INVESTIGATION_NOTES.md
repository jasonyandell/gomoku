# Overnight investigation notes — 2026-06-14/15 (session dbe0609b)

Autonomous overnight session. Jason handed me a free GPU + "keep trying things,
eval vs Rapfi hourly, keep the wiki updated (primary artifact), stay resumable."
Started toward a "search-depth axis" training experiment (#27) but the FIRST
diagnostic eval surfaced a yardstick problem that's more important than the
training idea. This file is durable scratch so nothing is lost if context
summarizes. Findings go to the wiki once quantified.

## What I set out to do
Issue #27: test whether deeper self-play search (n_simulations 100→200) breaks
the 96×8 champion's "69% deep-TC plateau" — sims is the one untried AZ lever
(capacity/data/epochs all explored at sims=100). Built cell `G15-96x8-deepgen`
(worktree) + `scripts/lab_status.py` (token-efficient monitor).

## The surprise (the real story)
Re-ran the champion `g15_champion_96x8_e499.pt` vs Rapfi@5000ms, sims=100, n=16,
**uncontended on MPS** — the SAME config that the campaign recorded as **69%
(11-5-0)**. Got **100% (16-0-0)**. Then sims=200 gave **75% (12-4-0)**. A
31-point swing on nominally identical config.

### Finding A — the yardstick is weaker than labeled (CONFIRMED)
`engines/rapfi/build_rapfi.sh` is explicit: the binary is Rapfi's **internal
classical config, NO NNUE weights**. The "~2625 Gomocup freestyle" rating cited
all over the wiki/handoff is the **NNUE** engine's rating; our `pbrain-rapfi` is
the weightless classical build — substantially weaker. So:
- Absolute strength claims ("trades blows with a 2625 engine", 75/69, 100%) are
  inflated; the provenance label `engine_source: rapfi ... 2625` is misleading.
- A weak opponent has little headroom → champion ~100% uncontended → can't
  measure training *progress* against it. Need a stronger yardstick.
- Stronger-yardstick options (in order of effort): (1) send `INFO thread_num N`
  (no rebuild — IF Rapfi honors it) + longer TC (15–30 s/move); (2) build NNUE
  Rapfi (Networks submodule + mix9svq weights + rebuild; needs cmake/ninja/net).
- §7 of the lessons wiki ALREADY flagged this: "not stress-verified it gives
  Rapfi its absolute best shot (threads, full config)... deserves that scrutiny
  before it's load-bearing." This is that scrutiny, made concrete.

### Finding B — deep-TC n=16 is unreliable (QUANTIFYING)
The reversal verdict (96×8=69 deep > 128×10=50 deep) rests on SINGLE n=16 reads
per net — never repeated. My re-run bounced 69→100. The 19-pt reversal gap is
~1.4σ at n=16; the OBSERVED run-to-run spread is larger than binomial. Wiki §4
says "weight aggregates, never a single number" — the load-bearing reversal
number violated that.
- Controlled battery running (`/tmp/eval_battery.sh` → `sweep_runs/eval_battery_
  deepTC_{mps,cpu}.jsonl`): 96×8 & 128×10 × {mps,cpu}, sims=100, @5000ms, n=24/16,
  uncontended. Tests: device effect (ladder_eval DEFAULTS to cpu; campaign may
  have measured on cpu while I forced mps), variance, and whether the reversal
  reproduces.
- Wall-time is a clue: prior 96×8 deep-TC n16 = 493s; my MPS run = 214s. Rapfi
  self-manages time (timeout_turn=5000, harness waits ≤30s, never cuts it off) —
  it moves fast in decided positions, so short wall = we crushed it fast.

## Mechanism candidates for 69→100 (battery resolves)
- H1 device: cpu ≈ 69, mps ≈ 100 (model plays differently on cpu).
- H2 variance: both high, big spread → 69 was a low sample, reversal may be noise.
- H3 load: uncontended = strong us; campaign's 69 was under training contention
  (needs a separate contended re-test during training).
- H4 (finding A): weak classical Rapfi → champion genuinely ~near-ceiling at this
  yardstick → need harder yardstick to see anything.

## Plan
1. Finish battery → quantify B (device/variance/reversal).
2. Strengthen yardstick (A): test `INFO thread_num` + longer TC; assess NNUE build.
3. Re-baseline champion (+128×10) vs the stronger yardstick → honest strength +
   real headroom.
4. Launch training (deepgen or reframed) with the stronger yardstick as the gate;
   resumable slices; eval hourly.
5. Wiki: write findings A & B as new sections — the primary artifact.

## Artifacts
- Worktree: `~/code/gomoku-deepgen-search-axis` (branch feat/deepgen-search-axis),
  commits: deepgen cell + lab_status.py.
- `sweep_runs/g96x8_searchscale_n16.jsonl`: sims 100→100%, 200→75% (champion, MPS, @5000).
- `sweep_runs/eval_battery_deepTC_*.jsonl`: the controlled battery (running).
- Issue #27.
