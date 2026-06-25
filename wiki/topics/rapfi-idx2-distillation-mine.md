# Rapfi idx-2 distillation mine — the "Bruce Lee one-position" experiment

**Status: IN PROGRESS (2026-06-25).** Tool built + tuned; first 1M-position mine
running; pretrain + warm-start to follow. This page is the synthesis; raw run
detail goes to `TRAINING_WIKI.md`.

## Hypothesis (Jason's framing)

> "I fear not the man who has practiced 10,000 kicks once, but the man who has
> practiced one kick 10,000 times."

Master **one** 15×15 opening — **idx-2** — and *nothing else* (only its D4
symmetries). The 15×15 champion era plateaued at ~50 Δelo and the binding wound
is the **white-defense gap** (`eval502` vs native Rapfi-NNUE @idx-2: black 42%,
**white 0/12**; see [white-side-defense-plan.md](white-side-defense-plan.md) and
the index reckoning). Instead of another defense knob on a generalist, this asks:
if we **deliberately over-specialize** — pretrain a net on Rapfi's own play of
the idx-2 tree, then run standard AlphaZero self-play from idx-2 only — can the
net stand against Rapfi *in that one position*?

Over-specialization is the point, not a risk: at eval/inference the only position
that ever occurs is idx-2, so every parameter spent elsewhere is waste. Coverage
is not a virtue here; **depth on one position is**.

## The tool — `gomoku/rapfimine/`

A reusable, crash-robust, max-throughput harness to mine `(position, Rapfi soft
policy+value)` pairs by BFS over a fixed opening. Built because the in-process
`RapfiPool` could not feed many engines (see Throughput below).

- **`canonical.py`** — D4-canonical state hashing. Stores ONE representative per
  symmetry class; the trainer's existing sample-time D4 augment recovers the 8×.
  `transform_state` carries board AND all `HISTORY_PLY` frames under one symmetry
  so every plane stays in a consistent frame. `canonical_key` = blake2b-16(canon
  board + move_count).
- **`worker.py`** — one OS process = one Rapfi engine in a tight analyze loop,
  with its OWN GIL and its OWN shard stream (planes never cross IPC). Respawns its
  engine on death; idle-exits if the coordinator vanishes.
- **`store.py`** — append-only sharded npz (teacher **v2** format: `planes`,
  `soft_policy`, `moves`, + a `keys` array for resume), atomic temp+`os.replace`.
  Frontier checkpoint = the pending-work half of the durable log (immutable
  snapshot, newest wins).
- **`coordinator.py`** — BFS frontier (deque, breadth) + global canonical dedup
  (gate at enqueue) + dispatch/collect + live monitor (count, inst/mean moves/s,
  Rapfi CPU% vs cores). Resume rebuilds the seen-set from shard keys and restores
  the frontier checkpoint (filtered against seen); in-flight boards are tracked by
  id and re-queued. A SIGKILL loses at most the branches found since the last 60 s
  checkpoint — **never a completed example**.
- **`pretrain.py`** — supervised distillation of the mined set into a standard
  checkpoint (`build_model`/`save_checkpoint` + the trainer's `policy_loss`/
  `value_loss`; no reinvention). Policy target = masked temperature-softmax of
  Rapfi's winrate map; value target = `2·best_winrate−1` (auxiliary — policy
  carries the load, #18/#44).

CLI:
```bash
GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine run \
    --out mined/idx2_15x15 --total 1200000 --workers 24 --max-node 5000
GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine status --out mined/idx2_15x15
GOMOKU_BOARD_SIZE=15 uv run python -m gomoku.rapfimine.pretrain \
    --shards mined/idx2_15x15 --out checkpoints/idx2_pretrain.pt --size large
```

## Two correctness/perf fixes this surfaced (both committed, #86)

1. **Rapfi multiPV crash (correctness).** `_read_analysis` capped chatter at a
   shared `_MAX_CHATTER_LINES=2000`. A forced-mate position (`EVAL -M4`) makes
   Rapfi deepen to DEPTH 26, re-emitting all PV blocks each round → **2302 lines,
   terminator and all** — past the cap, so it raised "no bestmove terminator" and
   crashed the mine at ~depth-5 tactical boards (the long-undiagnosed "200k crash"
   from the 2026-06-24 handoffs). Fix: a **pv-scaled** cap for the analysis loop;
   the quiet belt stays for the single-reply protocols.
2. **One reader thread per engine, not per line (throughput).** `_read_line`
   spawned a fresh `threading.Thread` for EVERY stdout line to enforce its
   timeout. At ~2000 lines/analyze that was ~50 ms of pure thread churn **serial
   with** (and starving) the engine. Fix: one persistent daemon reader thread per
   engine draining stdout into a queue; `_read_line` is now a cheap `queue.get`.
   Per-analyze on a deep board dropped **68 ms → 17 ms**.

## Throughput (the goal metric: moves/sec) — measured on the M5 Max (18 cores)

| Config | examples/s | machine | note |
|---|---|---|---|
| in-process `RapfiPool(60)` | ~75–150 | ~9% | GIL-serial stdio feed starves engines |
| harness, 24 workers, max_node 20000, pre-reader-fix | ~170 | ~28% | thread-per-line overhead |
| + reader-thread fix | ~480 | ~75% | engines genuinely busy |
| + **max_node 5000** (BFS) | **~700** | ~75% | 6× cheaper/board, SAME top-1 move, ~21 scored |

`max_node` sweep (per-board, with top-1 agreement vs the 20000 reference):
20000 = 118 ms; **5000 = 18 ms, top-1 SAME, ~21 moves scored**; 2000 = 10 ms but
only 8 scored and top-1 disagrees midgame (too shallow). **5000 is the
quality-preserving sweet spot.** 1.2M positions ≈ ~28 min.

## Pipeline

1. **Mine** idx-2 to ≥1M canonical positions (`rapfimine run`, max_node 5000).
2. **Pretrain** a `large` (128×10, "Bruce" size) net on the soft policy+value
   (`rapfimine.pretrain`). Sanity gate: H2H vs Rapfi @idx-2 on the pretrained net.
3. **Warm-start AlphaZero**: `run_sweep --resume` the pretrained checkpoint on an
   **idx-2-only** cell (fixed opening, no random openings), overnight.
4. **Verdict**: H2H vs native Rapfi-NNUE @idx-2, both colors separately (the
   white side is the real bar). Standard champion-not-Rapfi gating still applies
   for any cross-position claim — but here the *whole point* is the one position.

## Open results

_(to be filled as the run completes — dataset size, pretrain CE/value curves,
pretrained-net H2H vs Rapfi, warm-started self-play H2H vs Rapfi by color.)_
