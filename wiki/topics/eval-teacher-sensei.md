# The eval+teacher sensei — always-on Rapfi eval + Rapfi-as-teacher

**One subsystem, two faces.** After the recency-0.5 verdict (TRAINING_WIKI
2026-06-23) closed the self-play-knob era — *the three data-pipeline levers
(reuse / window / recency) are exhausted; keeping the loss alive ≠ keeping
strength climbing; the plateau is buffer-knob-proof* — the only lever that points
up is an **external strength gradient**. The sensei delivers it. The eval is the
teacher's measuring stick **and** its selector; building both as one keystone is
the point.

Code: `gomoku/rapfi_pool.py`, `gomoku/eval_panel.py`, `gomoku/eval_daemon.py`,
`gomoku/teacher.py`; the `play_match_pickers(start_state=…)` seam in
`gomoku/eval.py`; the `--teacher-weight` distillation seam in `gomoku/train.py`.
Console script: `gomoku-eval-daemon`. Built on `feat/eval-teacher-sensei`
(closes #34, advances #46/#18/#30/#35).

## The warm Rapfi pool (`rapfi_pool.py`)

`RapfiPool` pre-spawns N `ExternalEnginePlayer` processes once and lends them out
(thread-safe queue, self-healing on engine death). Why it matters: a 15×15 NNUE
engine pays a real start-up + weight-load tax on every spawn. The babysit scripts
respawned Rapfi every eval pass; on a *cadence* (every checkpoint) or when
*teaching* (labelling thousands of positions) that tax dominates. Warm = the
10×+ win. Safe because classical Rapfi runs in BOARD mode (`incremental=False`):
every move is a full `RESTART`+`BOARD` re-dump, so an engine carries no state
between calls and one instance can label unrelated positions back-to-back. It is
**CPU-only**, so it never competes with the MPS trainer — the same property that
makes the babysit cadence safe to run during a live Bruce run.

## Eval face — the #34 series (the measuring stick)

`eval_panel.py` evaluates a checkpoint against a **panel of rulers** (fixed
reference opponents) from a **fixed opening** and emits a per-color-split row.
- `Ruler(label, opponent, …)` where `opponent ∈ {"rapfi", a *.pt path, a baseline
  spec like "lookahead:depth=4"}`. The babysit rulers map directly: self126 =
  `anchor_e126.pt`, champ0235 = `epoch0235.pt`, rapfi.
- `fixed_opening_state(IDX2_OPENING)` rebuilds the swap2 idx-2 board B(3,2) W(5,4)
  B(4,5) from first principles — **self-contained**, no dependency on swap2's
  unmerged `_fixed_opening_state`.
- `play_match_pickers(start_state=…)` is the additive eval seam: every game starts
  from the fixed board with the same color-swap-per-pair seat alternation
  (default `None` = byte-identical to legacy).
- Net-vs-net rulers get early-ply temperature sampling for opening variety
  (`temp_until_ply`); the Rapfi ruler leaves the net deterministic and relies on
  Rapfi's own timeout wobble for variety.
- **White is reported separately** (the one hard #34 constraint): every row carries
  `white_score` / `white_loss_rate` / `white_wld` distinct from black, never folded
  into the aggregate. A rising aggregate that hides a flat/sinking white side is
  the exact failure mode the separate column exists to catch.

Cross-epoch speed: an `EvaluatorCache` keyed by `(path, mtime)` keeps fixed-ruler
nets warm forever (stable mtime → permanent hit) while the current checkpoint
reloads when rewritten. `run_panel` pins ONE `(evaluator, epoch)` snapshot so a
mid-panel rewrite can never mislabel the series row's epoch.

## Teacher face — Rapfi distillation (the lever up)

Rapfi exposes only a **move** (no value, no policy). And issues #18/#44 are blunt:
the **policy** must carry the load; **value-only** defense teaching is structurally
wrong. So the teacher is **policy-side, one-hot distillation**: "in this position
the master plays here." Value is never touched.

`gomoku/teacher.py`:
1. `gather_states` — self-play from the opening with the checkpoint, keeping every
   distinct board the net actually reaches (its own trajectory distribution).
   Stall-guarded: if the net's reachable set is smaller than requested it returns
   what it found and says so (no infinite hang).
2. `label_states_with_pool` — the warm pool labels each with Rapfi's move
   (failures skipped, not fatal).
3. `TeacherDataset` — stores planes (float16) + move index, applies D4 augmentation
   at sample time (one random symmetry per batch, planes and move permuted
   together — verified aligned for all 8 symmetries on 9×9 and 15×15).

`gomoku/train.py` mix-in: `--teacher-data-path PATH --teacher-weight W`. A teacher
batch is sampled every SGD step; `train_step` adds `W * CE(net_policy, rapfi_move)`
to the loss. Guarded so `W==0` (default) is byte-identical (no extra forward).
The teacher forward runs with **BatchNorm frozen** so the second forward doesn't
pollute the inference-time running stats (the main forward already tracks the
training distribution).

## How to run

```bash
# 1. ALWAYS-ON EVAL SERVICE (ad-hoc / derby; warm pool persists across calls)
GOMOKU_BOARD_SIZE=15 gomoku-eval-daemon serve --pool-size 6 --port 8008
#   POST /eval   {checkpoint, opponent:"rapfi"|"*.pt"|"lookahead:depth=4", n_games, sims, opening}
#   POST /panel  {checkpoint, epoch?}    -> per-color-split row
#   POST /move   {history:[…]}           -> the master's move (warm-pool demo)
#   GET  /health

# 2. THE #34 CADENCE (watch a checkpoint, append a white-split JSONL series)
GOMOKU_BOARD_SIZE=15 gomoku-eval-daemon cadence \
    --checkpoint .../checkpoints/worker_weights.pt \
    --series-out .../eval_series.jsonl --cadence-epochs 50 \
    --ruler rapfi=rapfi --ruler self126=.../anchor_e126.pt --ruler champ0235=.../epoch0235.pt
#   Pure reducer over the append-only series: last_epoch is recovered from the
#   file, so kill+restart resumes exactly. Watch worker_weights.pt (13 MB published
#   EMA weights) — cheapest to poll; latest.pt works too (atomic since #76) but is
#   1.4 GB.

# 3. GENERATE A TEACHER DATASET, then train against it
GOMOKU_BOARD_SIZE=15 python -m gomoku.teacher generate \
    --checkpoint .../worker_weights.pt --out teacher_idx2.npz \
    --n-positions 4000 --pool-size 6 --rapfi-timeout-ms 1000
gomoku-train … --teacher-data-path teacher_idx2.npz --teacher-weight 0.3
```

## Packaging Rapfi (friction-free resolution)

The Rapfi binary + NNUE weights + config are ~40MB of **gitignored** local build
artifacts, so a fresh worktree / machine / CI doesn't have them. `rapfi_pool.py`
resolves them with `rapfi_artifacts()`: **local `engines/rapfi` build → pinned HF
snapshot**. The artifacts live in a **public**, commit-SHA-pinned HF repo
([`jasonyandell/rapfi-arm64`](https://huggingface.co/jasonyandell/rapfi-arm64)) — a
GPL mirror of `dhbloo/rapfi @ 6e0a132` with the corresponding source cited in its
card (this project's own code is MIT; `THIRD_PARTY.md` records the arm's-length
attribution — Rapfi runs as a separate process, so its copyleft doesn't reach this
code). On first use `snapshot_download` pulls them into the machine-global
`~/.cache/huggingface` (the one store that's worktree- *and* venv-invariant),
`chmod +x`'s the binary, and **asserts its sha256** against the pin
(`RAPFI_HF_REVISION` / `RAPFI_BINARY_SHA256`). A box that already built the engine
never touches the network (local is higher precedence, byte-identical to before).
Bump the engine via `python scripts/publish_rapfi.py` (uploads to the public repo +
prints the two pin constants to update). We chose this over Docker deliberately: on
a Mac, Docker is a Linux VM that can't even run the arm64 Mach-O, and crossing a VM
boundary on the per-move stdin/stdout loop is the wrong shape for a native
warm-pool CPU engine (design workflow scored HF-resolver 8 vs Docker 3). The split:
`rapfi_available()` is cache-only / no-network (gates tests so they never fetch as a
side effect); `rapfi_obtainable()` may fetch (decides whether to *attempt* on this
arch). Caveat: the *first* fetch on a cold machine needs network once (then cached);
the binary is single-arch arm64-macOS.

## Operational constraints (discovered during the build)

- **Board size is a process constant.** Launch the daemon/teacher with
  `GOMOKU_BOARD_SIZE=15` to eval 15×15 checkpoints; the Rapfi NNUE weights are
  15×15. The idx-2 opening is a 15×15 board.
- **Rapfi is required by default — fail-fast, no silent fallback.** The default
  rulers include `rapfi`, so the daemon (`serve`/`cadence`) and `teacher generate`
  either run *with* Rapfi or **refuse to start** — `SystemExit(2)` with an
  actionable message (build `engines/rapfi/build_rapfi.sh`, ensure network for the
  HF auto-fetch, or configure only non-rapfi rulers). This is deliberate:
  repeatable-by-default beats a quietly baseline-only run you have to *notice*. To
  opt out, configure non-rapfi rulers explicitly
  (`--ruler heuristic=heuristic --ruler look4=lookahead:depth=4`) — deviation is a
  keystroke, never an accident.
- **Model schema (resolved).** Bruce's checkpoints carry a `choice_head` field;
  once the swap2 branch merged to `main` (2026-06-23), a main-built daemon loads
  them fine. (Historically, a pre-merge main daemon raised on the unknown key and
  degraded to per-ruler error rows — schema drift, not a daemon bug.)
- **Checkpoints are atomic (resolved).** `save_checkpoint` now writes
  tmp+`os.replace` repo-wide (#76), so the cadence can safely watch any
  checkpoint, including the in-place-feeling `latest.pt`. `worker_weights.pt`
  (13MB, published EMA weights) is still the cheapest watch target.
- **CPU-only by design.** The daemon defaults `GOMOKU_DEVICE=cpu`; it does not
  compete with the MPS trainer.

## What this is and isn't

It is the **instrument + the channel**: the calibrated-yardstick substrate (#34
done; #30/#35 enabled — anchor to native Rapfi, wine shelved) and the external
strength gradient (#46) delivered policy-side (#18 black-arm-compatible; #44's
"value-only is wrong" honored). It is **not** itself a tuned curriculum or a proof
that distillation breaks the plateau — that needs live validation over hours
against the live run (gate on not competing for the GPU). Bruce-1 (the recency-0.5
baseline) is the strength-to-beat.

See also: [swap2-opening-protocol.md](swap2-opening-protocol.md) (the idx-2 board
and Bruce), [external-engine-baselines.md](external-engine-baselines.md) (native
Rapfi-NNUE anchor), [white-side-defense-plan.md](white-side-defense-plan.md)
(why white is the gap). Evidence: `TRAINING_WIKI.md` 2026-06-23.
