# GPU/MPS-batched VCT solver — feasibility spike (verdict: correct but CPU-bound v0)

**One-line verdict.** A GPU-batched VCT (Victory-by-Continuous-Threats) solver is
**correctness-achievable** (matched CPU `solve_vct` exactly on every cleanly-resolved
position, zero false positives) but the v0 prototype is **CPU-bound, not worth wiring
into the labeling pipeline as-is** — only ~2× CPU at small node caps and *slower* than CPU
at production caps. The win that made GPU-VCF a ~2,500× blowout (flat OR-reachability,
forced unique reply) **does not transfer to VCT**, whose AND/OR branching forces stateful
tree bookkeeping + per-node irregular threat assembly back onto the Python host. This page
records why, with the profile, and the concrete MCTS-on-GPU roadmap for a future pass.

**Code:** `scripts/gpu_vct_prototype.py` (this spike) · reuses `scripts/gpu_vcf_prototype.py`
(detection kernels) · matches `gomoku/vcf.py` `solve_vct`. **Why we wanted it:**
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) "Applying it to THREATS"
(fast VCT for corpus threat-shape labeling). **Threat theory:**
[allis-threat-theory.md](allis-threat-theory.md) (VCF OR-only vs VCT AND/OR).

Date: 2026-06-25. Hardware: M5 Max, 48 GB, MPS. All throughput numbers **PRELIMINARY**
(small-n, time-boxed spike).

---

## 1. Correctness — PASS (this is the load-bearing result)

The GPU solver re-implements `solve_vct`'s *verdict* (`has_forced_win`) under the **same
bounding rules** (depth cap, node cap, top-K=6 three branching, the bounded AND-node reply
set = union of `{f} ∪ comps`, the 1-ply tempo guard). Given identical bounds, the AND/OR
value of the tree is well-defined and order-independent; visit order only changes *when* a
cap bites. So on any root where **both** solvers terminate without hitting a cap, verdicts
must agree.

Measured (random 8–30-ply midgame 15×15 boards, depth 7, node cap 250, seeds 1–4, n=160):

| metric | result |
|---|---|
| **Clean-both agreement** (neither solver capped) | **102 / 102 = 100.00%** |
| **False positives** (GPU "win" where CPU cleanly says no-win) | **0** |
| Cap-boundary disagreements (one solver capped) | 1–3 per 40, all the safe direction |

The only divergences are cap-boundary cases — and notably the GPU caps **fewer** roots than
CPU (e.g. 9 vs 14 / 40), because its full-expansion + precise UNKNOWN-propagation backup
*proves* some roots CPU's DFS abandoned at a cap. **Never a false positive** (the
non-negotiable: a claimed VCT win must be a real forced win).

**How correctness was bought:** the GPU does the *bulk filtering* (which cells are fours,
which are candidate forcing threes) with the proven VCF conv kernels; the **few survivors**
are confirmed with the *actual CPU helpers* (`_open_four_threats`, `_defender_has_four_or_five`,
`_has_disjoint_threats`) — correctness-by-construction. That choice is exactly what made it
slow (§3).

---

## 2. Throughput — PRELIMINARY, CPU-bound

Same node cap on CPU and GPU (apples-to-apples). M5 Max / MPS.

| node cap | CPU `solve_vct` | GPU batch (B=256) | speedup |
|---|---|---|---|
| 250  | 1.49 solves/s | 3.2 solves/s (wall 81 s/256) | **~2.1×** |
| 2000 | 0.46 solves/s | did **not** finish 256 in 11 min (killed) | **<1× (worse)** |

CPU `solve_vct` on open 15×15 boards is itself brutal (~16 ms **per node**; a single solve
at the production 20,000-node cap can take tens of seconds). The GPU v0 clears the bar at a
tight cap but **regresses at larger caps** because it does *full AND/OR expansion with no
alpha-beta cutoff* — it explores every branch the CPU's DFS would prune on the first win, so
total work explodes super-linearly with the cap while CPU short-circuits.

Jason's live Activity-Monitor read during the run: **CPU-pegged, GPU idle.** Confirmed by
profile.

---

## 3. WHY it's CPU-bound — the profile (the durable finding)

`cProfile`, `solve_vct_batch` B=256, cap 120, 50.3 s wall, 120 M Python calls:

| cost | tottime | cum | what it is |
|---|---|---|---|
| `vcf._completions_through` | 17.6 s | 34.1 s | per-cell collinear five-completion scan (Python) |
| `vcf._placement_makes_five` | 16.5 s | 16.5 s | **105 M calls** — inner of the above |
| `vcf._defender_has_four_or_five` | 1.6 s | **35.0 s (70%)** | the **tempo guard**, run per surviving three on the host |
| `vcf._open_four_threats` | 0.4 s | 6.6 s | exact open-four threat list per three (host) |
| `_wave_detect` (**the GPU path**) | 4.8 s | **8.1 s (~16%)** | conv detection over the pool |
| `.cpu()` / `.to()` (host syncs) | — | ~3 s | per-wave device round-trips |

**~84% of wall time is Python/numpy host code; ~16% touches the GPU, and the GPU is mostly
idle even within that** (thin per-wave tensors → kernel-launch-bound, not compute-bound). The
single dominant cost is the **tempo guard + open-four assembly running the scalar CPU helpers
per three-candidate per node** — i.e. the very per-node Python that makes CPU `solve_vct`
slow, reintroduced by the "confirm survivors on the host" design.

Two structural costs compound it: (a) **no batched AND/OR backup** — the tree is walked with
a Python per-node loop and numpy board copies per child; (b) **no alpha-beta cutoff** in the
batched full-expansion, so it does strictly more work than CPU DFS.

---

## 4. Why VCF batches at ~2,500× and VCT fights it (the lesson)

| | **VCF** (GPU win) | **VCT** (GPU resists) |
|---|---|---|
| Defender reply | **unique, forced** (the one cost square of a four) | **branches** — multiple cost squares (AND-node) + may counter-four for tempo |
| Search shape | pure **OR-reachability** — one child per node | real **AND/OR** proof search — needs proof/disproof backup |
| Frontier | **flat, lockstep** — every node is attacker-to-move at the same depth; advance the whole front in one tensor step | **ragged + stateful** — OR/AND nodes interleave, per-three reply sets vary, parents wait on all children |
| Per-node host work | ~none (detection is the only work, 100% conv-batchable) | irregular three/threat/comps assembly + tempo guard + tree bookkeeping |
| Backup | **none needed** (reach a win-now node ⇒ done) | segment AND/OR reduction over a parent tree |
| Cutoffs | implicit (stop at first reached win) | full expansion (no cheap batched alpha-beta) ⇒ more total work |

**The durable lesson:** GPU batching loves a *flat, regular, stateless lockstep frontier*.
VCF is exactly that. VCT's defender branching turns the problem into a *stateful, irregular,
backed-up tree* — and the moment you keep that bookkeeping (or the exact threat assembly) on
the host, you are CPU-bound regardless of how good the detection kernels are. The
detection-batching premise is sound (it *is* only ~16% and could be far less); the AND/OR
*orchestration* is the real adversary.

---

## 5. Roadmap to a hot GPU (MCTS-on-GPU style) — highest-leverage first

1. **Kill the host confirmation helpers (≈70% of wall).** Move the **tempo guard**,
   **open-four-threat detection**, and **completion-cell enumeration** fully into GPU tensor
   ops over the whole pool. The expanded-board `_four_structure` already yields `n_comp`;
   extend it to emit the per-cell completion **bitmask** so reply sets + the disjoint-fork
   test become tensor ops — deleting `_defender_has_four_or_five` / `_open_four_threats` /
   `_completions_through` from the hot path. This one change should reclaim most of the 84%.
2. **Vectorize the AND/OR tree into flat tensors** (the MCTS-on-GPU pattern). Node arrays
   (`own`/`opp`/`kind`/`parent`/`depth`) resident on-device; expansion = scatter/gather;
   **backup = segment-reduce by parent** (OR = `any`, AND = `all`, with UNKNOWN propagation).
   No Python per-node loop, no numpy board copies per child.
3. **Keep boards resident on GPU across waves.** v0 round-trips numpy each wave; eliminate the
   per-wave `.cpu()`/`.to()` syncs — the only host pull should be the final verdict.
4. **Add batched cutoffs** (proof-number / cheap alpha-beta) so the front stops expanding
   proven/disproven subtrees instead of fully enumerating — this is what stops the regression
   at higher caps.
5. **Fatten the frontier** so per-wave tensors are large enough to amortize kernel launches;
   the deep-wave tail is launch-bound, so pooling across positions **and** branches (already
   done) must be paired with the on-device tree so the tail stays wide.

**Single highest-leverage change: #1.** The profile says 70% of time is exactly the tempo
guard + open-four assembly in Python; move those to the GPU before anything else.

---

## 6. Verdict for the labeling pipeline

**Do not wire v0 into corpus threat labeling.** It is correctness-validated and a useful
scaffold, but at ~2× CPU (and worse at production caps) it does not pay for itself. The bar
to revisit: land roadmap #1 + #2 and re-measure — a GPU-hot VCT would need to clear roughly
**≥20–50× CPU** (sustained, at production caps) to be worth the complexity over just running
CPU `solve_vct` in a process pool. Until then, for corpus labeling prefer (a) CPU `solve_vct`
fanned across cores, or (b) the already-fast `solve_vcf_batch` for the four-only subset plus
a CPU VCT pass only on the residual. The detection kernels and the exact-match harness here
are reusable when someone takes the #1 lever.

---

## 7. REBUILD on MLX/Metal (2026-06-26) — bottom-up, oracle-validated, IN PROGRESS

The v0 verdict (CPU-bound) motivated a from-scratch rebuild on a custom Metal path,
pursuing Jason's three sharpenings: **(A) compiled line-threat grammar + (B)
intersection (bitmask) defense generation + (C) work-first continuation stealing**,
ultimately a persistent-epoch DFPN megakernel. Hermetic in `scripts/vct_metal/`
(nothing else in the repo imports it). Vehicle: **MLX** `mx.fast.metal_kernel`
(runtime-compiled MSL — no Xcode needed; bitwise/popcount + device atomics confirmed
on the M5 Max). Every layer is validated against the CPU oracle `gomoku.vcf` before
moving on; tests run on real Rapfi positions (`~/data/games_raphi/`) under a 2-min
`timeout` cap.

**Foundation (validated):**
- `detect_ref.py` — batched whole-board detection (fives, four-structure with
  completion counts, candidate mask, tempo guard). Matches `vcf` cell-for-cell over
  900 boards. Subtlety: `four_structure` counts fours *created by the move* (m in the
  five-window) — equals `vcf._completions_through` exactly in the no-immediate-five
  regime the OR-node actually runs in (the OR-node tests immediate-five first).
- `threes_ref.py` — forcing-threes + **(B) the bitmask defense**: defender reply-set =
  OR of threats' `{f}∪comps` masks; **fork = a disjoint mask pair**. Matches
  `vcf._has_disjoint_threats` + the reply-set union over ~9.7k threes. *This is the v0
  70% (host tempo-guard + open-four assembly) reduced to parallel set-algebra.*
- `detect_metal.py` — OR-node detection as a **Metal kernel** (one thread per
  (board,cell)); matches the numpy spec on-GPU, ~1M boards/s incl. host transfer.
- `search_ref.py` — the AND/OR solver **composed from the primitives**; verdict matches
  `vcf.solve_vct` on clean (no-cap) cases. Slow (B=1 recursion, ~2.2 s/board) — which
  is the point: the B=1 shape is exactly what batching/work-stealing fix.

**Lesson so far:** (A) and (B) are *proven correct*; the only thing wrong with a correct
composed solver is the recursion shape — i.e. the GPU-shape problem (C) targets. Next:
batched wavefront search → `threes_metal` kernel (move the last host cost on-device) →
the work-stealing DFPN megakernel. (Tracking the whole rebuild here as it lands.)

**Cross-links:** [allis-threat-theory.md](allis-threat-theory.md) ·
[molecule-discovery-toolkit.md](molecule-discovery-toolkit.md) · GPU-VCF prototype
(`scripts/gpu_vcf_prototype.py`) · `gomoku/vcf.py` (`solve_vct`).
