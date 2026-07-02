# The sound-world recipe (#107) — oracle in the environment, not the loss

**One-line:** self-play where the GPU oracle makes blunders unplayable (veto) and both game-ends
exact (attacker VCT terminus + defender all-moves-lose terminus) — the punisher the plain self-play
twin never provided. Killed the 9-ply fast-attack attractor (#101) in one day after weeks of
target-side injection failed; validated on 9×9 2026-07-01/02 (TRAINING_WIKI #107 entries).

## Why it works (the one ML lesson)
AlphaZero's learning operator is: search improves the prior → net distills the search's visit
distribution + game outcome. Every failed VCT injection (#36/#42/#43/#77/#86/#102/#103) edited
TARGETS after the fact — off-policy, fighting the distribution. The veto edits the GAMES: targets
stay the net's own (constrained) search, on-policy by construction. Cap the veto breadth and the
attractor returns (K=24 ablation) — causal, not correlational.

## The levers (all byte-identical-off)
- `--vct-terminus --vct-terminus-budget 50` (worker): attacker end, one-hot on the oracle move (#98).
- `--oracle-veto` (worker): per-ply bulk escape-solve at FULL breadth; proven-losing moves masked
  from played move AND recorded pi; all-legal-lose ⇒ defender terminus, z=−1, **NO example recorded
  for the doomed position** (the uniform-pi shrug collapsed white at scale — the #107 wound; see
  TRAINING_WIKI 2026-07-01 correction). Breadth caps are a 9×9 semantics trap; big-board staged
  escalation exists behind `--oracle-veto-max-cands` (leak rate must be measured).
- `--oracle-overlap` (worker): merged per-ply solve runs under the MPS search wave (1.18×, exonerated
  by the poison detector). Merged solve itself is default-on, bit-identical (1.07×).
- `--line-planes` (trainer): 8 in-forward line-potential channels; cross-line threats become local
  reads. In-model ⇒ 17-plane external contract untouched.
- Everything else: clone of `moonshot`/`vctsci-terminus` (WL2 stack, value-discount 0.98,
  global-pool, sgd-steps 64). Cell: `sound-world` in scripts/run_sweep.py.

## Guardrails (blood-bought)
1. **`uv run python scripts/gen_poison_check.py <ckpt>` after ANY gen-semantics change**: generates
   at live config and asserts NO recorded example carries policy mass on a proven-blunder cell
   (strict since the fix). Run A's poison was invisible to pl/vl/plies for 700 epochs.
2. **Gate on H2H + per-color columns, never internal metrics** (third confirmation of the #100
   lesson). The arena's `--json` gives the color split; the collapse signature is one-sided.
3. Never record a degenerate policy target "for the value signal" — drop the example; discounting
   carries z.

## Known open edges
- White-vs-lookahead:4 softness at 9×9 (5/20 white losses @ e1368) — unsettled when the chapter closed.
- 13×13 graduation prerequisite: cross-worker shared oracle solve (width-is-free ⇒ ÷4 aggregate
  oracle time; wiki/topics/mcts-perf-ceiling.md). Product shape = net + cap50 finisher (95% vs
  heuristic on 9×9 where bare-net draws).
