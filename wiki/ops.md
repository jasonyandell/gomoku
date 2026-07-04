# Ops — start here to run things

The operational front door. **What do you want to do?** Common workflows first,
then the live surfaces you touch to run and promote work. This is the home for
common operations — add new ones here as they earn their place.

> **← [index](index.md)** · knowledge hubs: [AlphaZero](alphazero.md) ·
> [Experiments](experiments.md) · [Derby](derby.md) ·
> [M5-as-Mainframe](m5-mainframe.md) · [Reference](reference.md)

## Common workflows

| I want to… | Go to |
|---|---|
| **Train / resume / tune a run** | [train-a-model.md](train-a-model.md) |
| **Evaluate a checkpoint** | [eval-a-model.md](eval-a-model.md) |
| **Publish / play a model** | [publish-a-model.md](publish-a-model.md) |

*(future common workflows land here — one row each, linking to a workflow page.)*

## Live operating surfaces (verified 2026-07-02)

The doctrine on these is current even where the *data* is paused — the 9×9 perf
frontier went quiet late May while work moved to 15×15 + VCT-science.

| Surface | Use it to… | Status |
|---|---|---|
| [ops/gpu-queue.md](ops/gpu-queue.md) | Order GPU-serial work (the two-queue scheduler doctrine — cited by the `gomoku-research-lab` skill) | **doctrine LIVE**, listed lanes closed |
| [ops/best-cells.md](ops/best-cells.md) | Look up the current best cell per reference point (the bests registry) | **LIVE registry**, last promote 2026-05-23 (9×9 perf paused) |
| [ops/experiment-ledger.md](ops/experiment-ledger.md) | The **Training-Quality Promotion Gate** + receipt schema — the rule you follow to promote | **gate LIVE**, receipts historical |
| [ops/test-ledger.md](ops/test-ledger.md) | The **Standard Gates** (`uv run pytest -q`, CPU smoke) before promotion | **gates LIVE**, rows a May-22 snapshot |
| [ops/baselines.md](ops/baselines.md) | The perf **benchmark cookbook** — reusable command surfaces + dated baseline rows | **LIVE commands**, last row 2026-05-23 |
| [ops/research-board.md](ops/research-board.md) | The Δelo Derby verdict board (recipe-race lever findings) | lever findings durable; **"current" frozen at v9 (05-27)** — needs a June/15×15 refresh |

## Historical / archived — do NOT read as current

| Surface | Why archived |
|---|---|
| [ops/status.md](ops/status.md) | The old control-room "current focus" — self-superseded (headline numbers trail best-cells 2×; points at the retired pi frontier). A dated snapshot, not live. |
| [ops/frontier.md](ops/frontier.md) | **Retired mechanism** — projected from `.frontier/lanes.json`, frozen 2026-05-22; the pi frontier-lab is dead. Superseded by gpu-queue. |
| [ops/perf-log.md](ops/perf-log.md) | Append-only perf-era narrative journal — evidence, correctly frozen (last entry 05-26). |

## The operating model (how these fit together)

*(Historical operating model — the autonomous derby is **stopped**; the gate/promotion doctrine below is retained for the next loop.)* Run work through the **[Derby](derby.md)**: propose a lane → run a GPU slice →
gate it. The gate is the `experiment-ledger` TQ rule + the `test-ledger` Standard
Gates; promotion updates `best-cells`; the derby verdicts live on
`research-board`. The doctrine pages ([research-lab-charter](topics/research-lab-charter.md),
[research-lab-session-runbook](topics/research-lab-session-runbook.md)) live in
the Derby hub — this hub is the *surfaces*, that hub is the *charter*.
