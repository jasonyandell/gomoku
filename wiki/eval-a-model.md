# Eval a model — the workflow

The pinned "how strong is this checkpoint?" page. Deep pages:
[eval-suite.md](topics/eval-suite.md), [Reference § Evals](reference.md).

> **← [index](index.md)** · workflows: [Train a model](train-a-model.md) · [Publish a model](publish-a-model.md)

## Fastest path

```bash
uv run gomoku-arena --help   # all games concurrent; 40-game eval ≈4s on MPS
```

## The rules that will bite you

1. **Eval the EMA `worker_weights.pt`, NOT the raw `epoch*.pt` state_dict**
   (#100 — terminus read 6% raw vs 68% EMA).
2. **The white column is the defense gate** — a net can look fine on black and be
   0/12 as white (Bruce vs Rapfi). Read both colors.
3. **Board size must match** the checkpoint.
4. **Fixed baselines saturate** — gate real strength on **H2H vs a frozen
   champion**, not siblings (non-transitive) or a noisy external ruler.

## The pages

| Need | Page |
|---|---|
| Command-first eval recipe + the `gen_poison_check` guardrail | [eval-suite.md](topics/eval-suite.md) |
| The fast arena (how it batches) | [batched-eval-arena.md](topics/batched-eval-arena.md) |
| What counts as a reliable eval | [reliable-eval-set.md](topics/reliable-eval-set.md) |
| The Rapfi anchor / teacher | [rapfi-pool.md](topics/rapfi-pool.md) · [eval-teacher-sensei.md](topics/eval-teacher-sensei.md) |
| Quantify white-side weakness | [white-side-defense-plan.md](topics/white-side-defense-plan.md) |
