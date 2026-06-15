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

## Update ~01:15 — deepgen "regression" was a FALSE ALARM (two more eval lessons)
Deepgen e460 vs NNUE first read 30/35% — looked like a crash from champion's 88/100.
Disambiguated (both concurrent w/ training, @5000 n=12):
- champion (frozen file)        = 75% (9-3)   <- vs 96-100% UNCONTENDED earlier
- deepgen epoch0460.pt (RAW)    = 35% (7-13)
- deepgen worker_weights.pt(EMA)= 83% (10-2)
Two findings:
1. EVAL THE EMA/PUBLISHED WEIGHTS, NOT RAW epoch*.pt. Raw 35% vs EMA 83% — a 48pt
   gap. epoch*.pt (25M) carries raw+optimizer; worker_weights.pt (6.3M) is the
   EMA weights that actually play. The pacer was eval'ing raw -> false 30/35%.
   (The first nnue_ladder.jsonl rows are RAW artifacts — disregard.)
2. EVAL-DURING-TRAINING may understate strength ~20pts (champion 96-100% uncontended
   -> 75% concurrent, n=12). Suggestive (n=12 noise), not proven; warrants a clean A/B.
DEEPGEN IS HEALTHY: EMA ~83% contended ≈ champion's 75% contended; plies~40, vl~0.169.
NOT regressing. Pacer fixed -> evals worker_weights.pt -> nnue_ladder_ema.jsonl hourly.
Primary signal = plies/vl (clean); Rapfi = shallow+contention-noisy hint.
TODO: one clean uncontended deepgen-EMA vs champion-EMA A/B (pause training) for absolute.

## Update ~03:30 — cont100 control isolates the deepgen regression cause
cont100 (warm from champion, 100 sims — byte-identical to deepgen EXCEPT sims):
- e501 (+100 epochs) vs frozen champion @100 = 20W-20L = 50.0% (EVEN, robust).
vs deepgen (200 sims) which was 0% @100 by +52 epochs.
=> The 200-sim self-play CAUSED deepgen's search-specialization/regression, NOT the
warm-continuation. Continued 100-sim training keeps the net robust at 100 sims.
Open: does cont100 climb PAST 50% (continued training improves champion -> reopens
"saturated") or stay even? Watching hourly head-to-head. cont100 trains ~1.6 ep/min
(100 sims faster gen; new~248 g/ep, reuse 0.67).

## Update ~04:50 — THE HEADLINE: the broken yardstick crowned the WRONG champion
128x10 (g15_128x10_bigbuf_eval502.pt) vs 96x8 champion, HEAD-TO-HEAD @100 n=40 = 40W-0L-0D (100%).
match.py validated (champion-vs-self=50%, deepgen=0%, this=100% — full range).
=> The campaign's "capacity reversal at depth" (128x10 WORSE, so 96x8 = champion) was
   a YARDSTICK ARTIFACT. Head-to-head, 128x10 CRUSHES the 96x8 champion 40-0. The broken
   Rapfi (shallow, ignores time, illusory tiers) made the STRONGER net look weaker.
   128x10 is the real 15x15 champion; the project abandoned its best net on a bad metric.
True ordering @100 so far: 128x10 >> 96x8(champion) >> deepgen (champion 40-0 over deepgen).
Mapping the rest (64x4 vs champion; 128x10@200 robustness). Then pivot GPU to 128x10.

## Update ~05:30 — full head-to-head ladder: the yardstick INVERTED the rankings
@100 sims, n=40, match.py validated (champion-vs-self=50%):
  128x10(e500) vs 96x8-champion(e400) = 100% (40-0)
  64x4(e900)   vs 96x8-champion(e400) = 100% (40-0)
  128x10(e500) vs 64x4(e900)          = 50%  (tied — both strongest)
  96x8-champion(e400) vs deepgen(e620)= 100% (40-0)
Order: {128x10, 64x4} >> 96x8-champion >> deepgen.
=> The broken Rapfi yardstick CROWNED THE WEAKEST trained net (96x8-e499, frozen
   undertrained at e400) and REJECTED a tied-strongest net (128x10). Rankings inverted.
CONFOUND: epochs differ (64x4 e900 vs champion e400) — a clean CAPACITY comparison
needs same-epoch head-to-heads (future work). But the META-finding is airtight:
Rapfi-based rankings are wrong; head-to-head is the trustworthy measure.
Also: cont100 (96x8 e400->e673) = 50% vs frozen e400 champion = continued 96x8
training does NOT close the gap to 128x10/64x4 -> 96x8 plateaued BELOW them.
PIVOT: stop cont100 (plateaued), train 128x10 (strongest + most headroom, abandoned
at e560 on the bad yardstick) -> does the best net keep climbing (head-to-head vs e502)?

## Update ~06:30 — the abandoned net was STILL CLIMBING (fast)
128x10 resumed e503->e588 (+85 epochs) now beats its OWN e502 self 40-0 (@100, n40).
vs cont100 (96x8) which was 50% vs its e400 self after +178 epochs (saturated).
=> 128x10 (3.3M, high capacity) has the HEADROOM the saturated 96x8 (1.55M) lacked,
   and is improving fast. The campaign abandoned it at e560 (on the broken yardstick)
   WHILE IT WAS STILL CLIMBING STEEPLY. The yardstick cost us a rapidly-improving net.
Froze g15_128x10_bigbuf_e588_best.pt. Pacer now tracks current vs e588 (moving baseline).

## Update ~07:25 — 128x10 continuation: big gain then PLATEAU
Trajectory (head-to-head @100, n40): e588 vs e502 = 40-0 (100%); e688 vs e588 = 50% (flat).
So 128x10 improved e503->e588 then plateaued e588->e688 (+100 epochs, no gain).
CAVEAT: the e503->e588 jump is partly EMA-warmup-after-resume (worker_weights re-
averaging), not all training. The ROBUST claim is frozen-vs-frozen: 128x10-e502 >>
96x8-champion (40-0). 128x10's ceiling is HIGHER than the 96x8's, and it's now near it.
Health: e706, plies 43.6, vl 0.164, ~1.6 ep/min. Plateaued but healthy. Still training
(GPU busy) until Jason reclaims; #29 (re-crown + matched-epoch ladder) is the follow-up.

## Update ~09:45 — capacity story finalized (one clean result + one anomaly for #29)
- 128x10 FULLY plateaued: e907 vs e588 = 50% (+319 epochs, no gain). Reached ceiling ~e588.
- CLEAN: cont100 (96x8 trained +273 epochs to e673) STILL loses 0-40 to 128x10-e502.
  => 96x8 is genuinely CAPACITY-CAPPED below 128x10; more training doesn't close it.
     The re-crowning (128x10 >> 96x8) is a real capacity result, not epoch confound.
- ANOMALY (flag for #29): the 96x8 (1.55M, MIDDLE capacity) is the WEAKEST net —
  beaten 40-0 by BOTH 128x10 (bigger) AND 64x4 (smaller, e900). A middle net losing
  to both neighbors is NON-MONOTONIC -> likely a bad 96x8 run / net2net-lineage issue
  (96x8 was grown from 64x4, ended weaker than its parent), NOT a clean capacity curve.
  The clean capacity ladder (#29) must be same-lineage + same-epoch to be trustworthy.
GPU: 128x10 (best net) still training, plateaued. Holding for Jason; next moves (#28
yardstick fix, #29 clean ladder) need his direction. Investigation complete.
