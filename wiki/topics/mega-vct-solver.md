# `mega_vct_bb` — the on-device GPU VCT solver (canonical API reference)

**One-line.** `scripts/vct_metal/mega_vct_bb.py :: solve_vct_mega_bb` is the
production VCT (Victory-by-Continuous-Threats) solver: one GPU thread per board
runs the *whole* AND/OR proof search on-device as a bitboard (`own`/`opp`/`empty`
= `ulong[4]`), make/unmake on a thread-local board, **all detection by bitboard
set-algebra**. It is the single solver every threat-shape/mining/labeling
consumer calls. **~1600× CPU throughput**, **0 FP / 0 FN** vs `gomoku.vcf.solve_vct`
over 320 VCT + 360 VCF real positions.

This page is the **API + invariants reference**. For *why* it is built this way
(the CPU-bound v0 spike, the bitboard levers, the throughput characterization)
read [gpu-vct-feasibility.md](gpu-vct-feasibility.md) — that page is the
narrative/feasibility record; **this page is the contract.**

---

## ⚡ THE CALL-COST LAW (the binding constraint — internalize before calling)

**One call costs one _tail_: the wall is set by the single hardest board in the
batch and is nearly flat in batch size.** B=16 → 24.6 s, B=16384 → 71.7 s — 1000×
the boards for ~3× the wall; per-board cost swings ~350×. Compile is ~0.1 s (a
fresh `uv run` per query is fine). Therefore **every caller MUST be
bulk-synchronous**: gather every board you need (up to ~16 k) into *one* call;
never solve-in-a-loop on a small batch. `max_nodes` is the tail knob (it bounds
the single deepest board's search) and is a **GLOBAL pool across the batch** — for
corpus labeling, scale `max_nodes` with B. A CAP verdict is fail-safe wherever
"couldn't prove it" can be treated conservatively. Full numbers + derivation:
[gpu-vct-feasibility.md](gpu-vct-feasibility.md) §8 banner.

---

## API

```python
solve_vct_mega_bb(boards, *, max_nodes=20000, tg=32,
                  return_move=False, return_support=False, complete=False)
```

`boards`: `(B, 2, N, N)` bool, **side-to-move-relative** — `board[0]` is the
**attacker** (the side to move whose forced win we solve), `board[1]` the
defender. **No swap**: at any winner-to-move position `GameState.board` is already
in this frame for **black AND white** alike (verified — getting it wrong silently
solves the wrong side; see [vct-backward-mining.md](vct-backward-mining.md) §2).
Free-style rules (overlines win). `N = GOMOKU_BOARD_SIZE` (15 here); the kernel is
a 256-bit `ulong[4]` board so N²≤256.

Returns a tuple. The default is `(win, hit_cap)`; optional outputs append in a
**FIXED order — `move`, `support`, `winmask`** — so existing callers never break:

| flag | appends | type | meaning |
|---|---|---|---|
| *(default)* | `win` | `(B,)` bool | a forced VCT win exists for the attacker |
| *(default)* | `hit_cap` | `(B,)` bool | search hit `max_nodes`/depth → verdict inconclusive (not a no-win) |
| `return_move=True` | `move` | `(B,)` int32 | a VALID (sound, not necessarily shortest) VCT first move, flat cell index; `-1` on a non-win |
| `return_support=True` | `support` | `(B,4)` uint64 | union of cells the found proof line touches (relevance window / stencil seed); all-zero on a non-win |
| `complete=True` | `winmask` | `(B,4)` uint64 | bitmask of **ALL** winning first moves; also flips `win` to "∃ a winning first move" |

Unpack a `(4,)` uint64 mask to flat cell indices with
`mega_vct_bb.cells_from_words(words)` (inverse of the kernel's bit packing, bit
`r*N+c`).

Examples: `solve_vct_mega_bb(b)` → `(win, hit)`; `return_move=True` →
`(win, hit, move)`; `complete=True, return_move=True, return_support=True` →
`(win, hit, move, support, winmask)`.

**Each flag compiles its own kernel variant** (memoized in `_KERNEL_CACHE`). The
default `(support=False, complete=False)` variant is built from a source string
**byte-identical** to the original — so the default verdict and throughput are
provably unchanged. Variants cost compute, not the fast path.

---

## The three outputs, precisely

### `move` (passive winning move) — `return_move=True`
A *passive* read of the move the search already commits to: the root's chosen
move `mm[0]`, or the captured cell for an inline root win (immediate five / sound
double-four / fork-three). **No extra nodes, no extra search.** Sound because the
kernel never reports a false win, so the root of the line it proves always starts
a real forced win. It is *a* valid VCT move, not necessarily the shortest mate's.
Verified independently on 400 puzzles (play the move, attacker wins vs every
defender reply). Shipped 2026-06-26; see [vct-backward-mining.md](vct-backward-mining.md) §5.

### `support` (proof relevance window) — `return_support=True`
The set of cells the found proof line **touches** — a stencil seed / relevance
window, deliberately **over-inclusive vs a minimal stencil** (the downstream
shape-library engine ablates it down). Built by **return-path accumulation**: a
per-frame `fsupp[]` that merges a child's support into its parent **only on a
winning return** (`ret==1`), so abandoned/refuted branches **never pollute it**.
Contributions: an OR node that wins adds its move (+ the forced four-block); an AND
node that wins adds **every** defender reply; the three inline OR wins add the move
+ its completion/threat cells. **`support` ⊆ the root's EMPTY cells** — it is the
*played* cells of the proof (moves, blocks, replies, completions), NOT the
pre-existing threat-carrier stones (those stay on the board for the ablation pass).
All-zero on a non-win. Validated: cells empty-at-root, contain the move on every
win, empty on every loss.

### `winmask` (all winning first moves) — `complete=True`
The default search short-circuits at the root OR node (first winning move wins).
Complete mode instead tries **every** root candidate and records each winning
first move in `winmask`; non-root nodes are untouched (they still short-circuit —
which is exactly the per-root-move verdict). **"All solutions" == all winning
FIRST MOVES**, not all winning lines/trees (combinatorially larger — AND nodes
branch). It is **slower** (no root short-circuit ⇒ strictly more work) and shares
the same global `max_nodes` pool. `winmask` is the winning **forcing** first moves
(fours + forcing-threes passing the tempo guard) = exactly the VCT first moves; a
non-forcing "free win" in an already-won position is NOT a VCT first move and is
correctly excluded. With `return_support` too, `support` is the **union** over all
winning first moves (per-move support would need a `(B,N²,4)` output — left a
non-goal). `move` in complete mode is one winning move (lowest cell of `winmask`).

**Validated sound + complete** (`tests/.../test_mega_vct_bb.py` invariants +
`gold_complete.py`, 2026-06-27): over the gold boards, **0 unsound** winmask moves
(each independently verifies as a forced win) and **0 winning forcing moves
missing**. The completeness oracle is vcf's exact root candidate generation
(fours ∪ forcing-threes) **including the tempo guard** `_defender_has_four_or_five`
— omitting that guard was a *verifier* bug that produced phantom "misses"; the
solver was right.

---

## Invariants (the regression contract)

1. `return_support` leaves `(win, hit, move)` **byte-identical** to default.
2. On boards neither solve capped, **complete `win` == default `win`**.
3. On a clean win, the default `move` is a member of `winmask`.
4. `support` cells are empty at root, contain the move on a win, empty on a loss.

Covered by `scripts/vct_metal/test_mega_vct_bb.py :: test_support_and_complete_invariants`
(run via `GOMOKU_BOARD_SIZE=15 uv run python -m scripts.vct_metal.test_mega_vct_bb`;
**budget `max_nodes=500` captures ~90% of VCTs and runs in seconds** — Jason's test
default). The default-verdict cross-check vs the cell-scan `mega_vct` (0
disagreements) is the same module's `test_mega_vct_bb_matches_mega_vct`.

---

## Gotchas

- **Plane convention is side-to-move-relative, `board[0]`=attacker, no swap** — the
  #1 silent-wrong-answer trap.
- **Free-style** (overlines win); VCT-win monotonicity holds under free-style.
- **256-bit board** (`ulong[4]`, N²≤256). `uint[8]` was tried and reverted (~20%
  slower in situ — the kernel is bookkeeping-heavy; see §8 "word width").
- **Tail-bound, bulk-synchronous only** — see the call-cost law. Keep callers
  bulk-synchronous; never solve-in-a-loop.
- **`hit_cap` is not a no-win** — it's "couldn't prove within budget". Treat
  conservatively.
- Adding the `support`/`winmask` outputs costs nodes/compute, not the fast path
  (separate compiled variants; default is byte-identical).

---

## Consumers (who calls this)

`scripts/threat_shapes/`: `mine_vct_gpu_flat.py`, `mine_first_vct.py`,
`solve_puzzles.py`, `mine_puzzles.py`, `harvest_molecules.py`, `vct_fan.py`,
`probe_timing.py`. Feeds [shape-library-engine.md](shape-library-engine.md) (L1
stencils), [vct-backward-mining.md](vct-backward-mining.md),
[vct-reachability-mining.md](vct-reachability-mining.md), and
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md).

**Cross-links:** [gpu-vct-feasibility.md](gpu-vct-feasibility.md) (build narrative
+ throughput) · [vct-backward-mining.md](vct-backward-mining.md) §5 (the move
output) · [allis-threat-theory.md](allis-threat-theory.md) (VCF/VCT formalism) ·
`gomoku/vcf.py` (the CPU oracle `solve_vct`).
