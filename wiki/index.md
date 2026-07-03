# Gomoku Wiki

**What this is.** An AlphaZero engine for free-style gomoku (9×9 → 15×15),
trained on Apple Silicon (M5 Max, PyTorch/MPS), run as a **Δelo/Δt research lab**
with W&B from day one. The wiki is the maintained synthesis layer: raw evidence
(W&B, checkpoints, logs) stays stable; this compounds what we learn so every
session starts smarter than the last.

**Why it exists — three goals.**
1. **Learn AlphaZero** for real, on a game we can reason about end-to-end.
2. **Learn agentic engineering** — Sid Bidasaria's compounding-trust loop +
   Karpathy's LLM-wiki, so the machine (and the wiki) get smarter each session.
3. **Squeeze the M5 Max** — treat one Mac as a knowable mainframe and get
   everything out of it.

**You are here.** *(full milestone index: [training-timeline.md](training-timeline.md))*
- **Started** *(2026-05-17)*: the origin 9×9 run (`o9npssu1`) collapsed to
  defensive draws by epoch 136 — establishing the baseline profile where
  **generation dominates training 25–30×** in wall-clock. The first thing to
  reliably break the collapse was the ported **AZ recipe** (`sppjo3z5`, model_elo
  1718); the WL series then set the 9×9 ATH (elo 1841).
- **Now** *(2026-07-02)*: the **"sound world"** recipe (GPU VCT-oracle veto +
  terminus self-play + line planes). **9×9 chapter CLOSED** (0-0-40 H2H vs the old
  champ; finisher-hybrid 95% vs heuristic); **13×13 graduation is a structural
  negative** (#113 — white 0/20, both nets attack-only). *(perf: the VCT oracle
  veto is ~91% of 13×13 self-play gen wall ⇒ the solver is the lever; the 2026-07
  perf blitz landed the solver levers — cap25 gen budget (#114) + `lanes=K` kernel
  (#114) + one-worker refill (#112) — closing the gen-loop thread; see
  [M5-as-Mainframe](m5-mainframe.md).)*
- **Next (untried)**: role-invariant symmetric **rails** (drop the terminus, cure
  white-starvation on-policy) + attacker-preserve closing; and the pivotal open
  question — **is 13×13 a forced black win?** (15×15 proven / 9×9 drawish / 13×13
  unknown — probe with the mega-VCT oracle). See the [AlphaZero hub →
  open wound](alphazero.md) and [sound-world-recipe.md](topics/sound-world-recipe.md).
- **Learned (headlines)** — the durable, era-independent lessons:
  - **Fast-attack collapse** is the recurring failure mode (policy sharpens on
    attack, self-play never punishes missing defense). Watch `selfplay/plies_mean`.
  - **Gate strength on H2H vs a *frozen champion*** — never sibling H2H
    (non-transitive) or a shaky external ruler. Fixed baselines saturate.
  - **The binding wound is white-side (defender) weakness** — a first-player-win
    theorem, not just a net flaw.
  - **A VCT is a forced win the GPU oracle both *detects and terminally values***
    — the engine of the whole current research program.

---

## Ops — start here to run things

**[→ Ops hub](ops.md)** is the operational front door: the common workflows plus
the live surfaces you touch to run and promote work. The three you'll want most:

| I want to… | Go to |
|---|---|
| **Train / resume / tune a run** (every knob, the launch runbook, the skill) | [train-a-model.md](train-a-model.md) |
| **Evaluate a checkpoint** (arena, baselines, the white-defense gate, gotchas) | [eval-a-model.md](eval-a-model.md) |
| **Publish / play a model** (HuggingFace, the web UI, how it works) | [publish-a-model.md](publish-a-model.md) |
| **Operate the lab** (GPU queue, bests registry, promotion gate, benchmark cookbook) | [ops.md](ops.md) |

## The map — 5 hubs

| Hub | What's inside |
|---|---|
| **[AlphaZero](alphazero.md)** | The training arc: **best performance first**, headline facts, and curated **what-worked / what-didn't**. The core learning artifact. |
| **[Experiments](experiments.md)** *(hub of hubs)* | Every research thread. The huge one is the **[seek-VCT program](seek-vct.md)** (the net steers / the oracle finishes); plus each side-quest, reconstructed from the logs as honestly as we can. |
| **[The Derby](derby.md)** | The Δelo/Δt engine: **1-hour training slices, 3 roles** (researcher · trainer · runner), receipts + Reviewer audits. Where it started → where it stopped. |
| **[M5 as Mainframe](m5-mainframe.md)** | How we learned to push this Mac — the perf topics we explored (MCTS ceiling, ANE/Core ML, cross-engine coupling, bit-packing). |
| **[Reference](reference.md)** | Look-things-up shelf: the **Training-wiki** broken into topics, **all our evals** (including forgotten ones), **tools** (Rapfi pool, arena, …), and cross-cutting conventions + ops. |

---

## How this wiki works (the operating model)

- **Hub-of-hubs.** This page is one screen and one fetch. Each hub tells the same
  self-similar story — **started → now → learned + links** — and *prose lives in
  the leaf pages*, never in the hub tables. Each hub also fits one fetch.
- **What-worked / what-didn't live *inside* each hub**, next to their evidence —
  not as a front-page list (that would drift).
- **Evidence vs synthesis.** [TRAINING_WIKI.md](../TRAINING_WIKI.md) is the
  append-only chronological notebook (dated corrections, never rewrites);
  checkpoints / `sweep_logs/` / `wandb/` are immutable evidence.
- **Provenance is a field.** Reconstructed-from-logs work says so.
- Update [log.md](log.md) when the wiki structure or a synthesis page changes.
