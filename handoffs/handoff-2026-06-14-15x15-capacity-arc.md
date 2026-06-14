# Handoff — 15×15 AlphaZero campaign: capacity×data arc closed, GPU handed off

**Date:** 2026-06-14 · **Session:** 830c9ad0 (long autonomous run, many 5-min loop ticks) · **Repo:** ~/code/gomoku

## Goal & current status

Autonomous 15×15 free-style gomoku AlphaZero campaign on an M5 Max (PyTorch+MPS, W&B, Rapfi as external yardstick). **North star is LEARNING AlphaZero — a full-circle to a game Jason first tried in the 90s; strength is "gravy," not the point.** This session ran the **capacity × data** experiment to completion and it is now **CLOSED + recorded + pushed**. Verdict: a **96×8 net is the deep-time-control sweet spot** (75% fast / 69% deep vs Rapfi); growing to 128×10 with a 1.5M buffer made it *better* at fast TC (88%) and *worse* at deep TC (50%) — capacity **reversed** at depth, and the "still-early" hypothesis was refuted at e500 (deep-TC flat at 50% across e348→e502 despite improving loss). Then Jason reclaimed the machine for a **separate other project**, so the 128×10 backfill run was **stopped cleanly at e560** (resumable), the GPU is **free**, and the autonomous 5-min loop is **stood down** (no reschedule — nothing will auto-restart onto his GPU). Campaign concluded at a clean stopping point.

## Decisions made + rationale

- **9×9 → 15×15** (joint): the 9×9 perf ceiling was *dispatch-bound* (a 325k-param model under-using the GPU), not the Mac. 15×15 is affordable (~2.3× cost at the sweet-spot net). Don't say "the Mac maxed out at 9×9."
- **Warm-start via net2net / representation transfer** (confirmed win): a 9×9 champion's conv tower seeds a 15×15 net and **skips the cold-start fast-attack collapse** (a *real* 15×15 failure mode — cold runs collapse to ~11 plies; warm starts play defended ~85-ply games from epoch 0).
- **Capacity ladder (function-preserving net2net):** 64×4→96×8 **paid at depth** (deep-TC 67→69, fast 75). 96×8→128×10 on the 400k buffer **overshot/overfit** (37.5% fast-tier).
- **Data axis on 96×8** (1.5M bit-packed buffer): did **NOT** help — 96×8 is *capacity-bound*; deep-TC ceiling unchanged. Clean **negative** result. (REJECTED the idea that more data was the lever for the sweet-spot net.)
- **Capacity × data jointly** (128×10 + 1.5M, from net2net seed): **cured the overfit** at fast TC (37.5→88, now > champion's 75) but deep-TC **reversed** (50 < 96×8's 69). Still-early refuted: deep-TC = 50% at e348 *and* e502 (n=16) while value-loss set fresh lows and games lengthened. → The reversal is real, not a training-time lag.
- **CORRECTION (important): the "88% deep-TC ceiling" was a NOISE artifact.** Every single-n=8 deep-TC read ran high. At n=16: 96×8 = **69%**, 128×10 = **50%**. The champion's honest record is **75 fast / 69 deep** (not 75/88). We'd reasoned against a phantom 88% wall for ~a dozen status updates.
- **Eval discipline (hard-won):** use **n≥16** for any number you'll reason against (n=8 ±18% bit us at e248, where deep-TC=75 looked like a "climb" and was noise). **Rapfi is the yardstick** — not self-play Elo, not internal baselines. Weight aggregates; require BOTH time-control tiers.
- **Internal baseline evals made OPT-IN** (`--internal-eval`, default OFF in run_sweep.py) — Jason's call. They're saturated noise for mature nets (all pin ~100%, Elo ceiling-clamps) AND cost real CPU (a lookahead:depth=4 cycle is ~200–320s, recurring ~every 15–20 epochs, competing with the Rapfi ladder). Live run was left untouched; only future launches change.
- **Breathing-room handoff design** (speced, NOT built): a monitor that **resumes** gomoku from latest.pt when the GPU is idle 1hr (backfill toward e1M) AND **pauses** it (SIGTERM self-save) when another tenant appears (yield). "The Mac is a mainframe; another department needs GPU time." This session the handoff went **manual** — Jason needed the machine, so we just stopped cleanly.
- **FPU eval-lever** (closed earlier, do not re-propose): does not transfer to 15×15.

## Constraints & invariants discovered

- **MPS INT_MAX cliff:** a bit-packed replay buffer crashes at ~561k board-15 positions — a whole-buffer unpack materializes a tensor > 2³¹ and MPSGraph dies. Fixed via **chunked unpack** (`gomoku/replay_buffer.py`); re-validated live this session (the 128×10 run sailed past 561k clean). CUDA/CPU don't have this cap; MPS does.
- **Gen-flood:** big nets need **fewer self-play workers** (128×10 ran 4 workers, not 8) or the slow trainer gets flooded (per-epoch wall runs away).
- **Eval is slow for the 3.3M net:** ~6–9 min/tier; an n=16 5000ms read is ~12–20 min and shares CPU with training. Plan around it.
- **Freeze eval checkpoints in `sweep_runs/` ROOT**, never in `checkpoints/` — the trainer's `keep_last_n` prunes that dir.
- **`latest.pt` embeds the replay buffer** → clean warm resume (no cold refill).
- **Worktree discipline (CLAUDE.md hard rule):** never edit the shared `main` checkout; every change → worktree off main → `merge --no-ff` from the MAIN checkout (`git pull --no-rebase` first) → push → teardown. **Run `lab_log.py` INSIDE the worktree** (an uncommitted events.jsonl line in main once caused a merge headache). Never rebase/squash/ff.
- **value-loss ≠ deep-search strength** — they *decoupled* cleanly e348→e502 (vl improved, deep-TC flat). This is the campaign's sharpest lesson.
- **Color asymmetry:** free-style gomoku is a first-player (Black) win; the internal "vs_heuristic_white" losses are the 2nd-player handicap + shallow 100-sim eval + small-n, **not** net weakness. (Jason's catch.)
- **Don't compete with live GPU tenants** — the whole reason the loop self-arms with a clean-tenant check and why we yielded instantly when Jason needed the machine.

## Open questions / parked threads

- **Resume-on-idle + yield "breathing-room monitor"** — speced, NOT built (deferred; Jason just needed the machine). *Non-blocking.* Offer to build it when he wants gomoku to auto-backfill his idle GPU.
- **Jason's "other game"** = a whole separate project he needs the machine for — **NOT** something we scaffold here. *Flavor* (his thing; don't touch).
- **Giving back:** Jason mused (shy, "absolute amateur," but "not nothing") about sharing findings. Concrete options offered: the **MPS bit-packing fix** as a standalone gist/PR (lowest-effort, genuinely reusable); the **lessons wiki page** as a blog post / dev-forum write-up. *Non-blocking, flavor.*
- **White-side defensive tightening:** a real but explicitly **LOW** priority — Jason: "such a low priority that it doesn't matter. we have our priorities right." Deferred. *Non-blocking.*
- **Why does capacity reverse at depth?** Leading guess: extra capacity sharpened the *policy* (fast play) but the *value/positional eval's deep-search discrimination* didn't improve. Unconfirmed mechanism. *Flavor — open science worth a future probe.*
- **128×10 backfill toward e1M:** resumable anytime from latest.pt, low research value (verdict stable). *Non-blocking.*

## Artifacts (all pushed to `main`)

- **Champion:** `sweep_runs/g15_champion_96x8_e499.pt` — **75% fast / 69% deep** (the 15×15 champion).
- **128×10 final:** `sweep_runs/g15_128x10_bigbuf_eval502.pt` (88 fast / 50 deep). Run dir `sweep_runs/G15-128x10-bigbuf-board15/` — `checkpoints/latest.pt` is the **resume point** (stopped clean at e560, buffer embedded).
- **n=16 deep-TC reads:** `sweep_runs/G15-128x10-bigbuf-board15/rapfi_deepTC_n16.jsonl` (128×10: eval348=50, eval502=50) · `sweep_runs/g96x8_deepTC_n16.jsonl` (96×8 = 69).
- **Other preserved ckpts:** `g15_96x8_bigbuf_latest_e616`, `g15_96x8_latest_e532`, `g15_champion_e909` (64×4), `g15_128x10_seed`, `g15_128x10_latest_e598`, `g15_128x10_bigbuf_eval{146,248,348}`.
- **THE artifact (the learning):** `wiki/topics/alphazero-lessons-15x15-gomoku.md` — §2/§2a hold the full capacity arc (inverted-U table, the n=16 head-to-head, the loss≠deep-strength dissociation, the still-early refutation). Companion: `wiki/topics/15x15-training-campaign.md` (operational story), `wiki/ops/events.jsonl` (event log).
- **Code:** `scripts/run_sweep.py` (cell `G15-128x10-bigbuf`; new `--internal-eval` flag, default off). `scripts/ladder_eval_15x15.py` (the Rapfi yardstick — run with `PYTHONPATH=$PWD`; use `--n-games 16 --timeouts 5000` for deep-TC).
- **Resume command:** `export WANDB_API_KEY=$(security find-generic-password -s wandb-api-key -w); GOMOKU_BOARD_SIZE=15 python scripts/run_sweep.py --cell G15-128x10-bigbuf --resume sweep_runs/G15-128x10-bigbuf-board15/checkpoints/latest.pt`
- **Memory:** `~/.claude/projects/-Users-jason-code-gomoku/memory/project_15x15_era.md` (live state + HANDOFF note), `feedback_learning_is_the_artifact.md`, `user_buddy_relationship.md`.

## Next action

**Do nothing that touches the GPU.** The campaign is at a clean, recorded stopping point and Jason has the machine for his other project. **Wait for him to reclaim the lab.** When he does, *ask* which he wants — (a) build the resume-on-idle+yield breathing-room monitor, (b) resume the 128×10 backfill manually, or (c) start something new — **don't assume**. (This handoff is a real session-end, not a practice run, but the "next action" is genuinely *wait*, not execute.)

## Vibe snippets (paste verbatim)

> **Jason:** "I don't even have a goal of the gomocup. not even a little! I just wanted to learn alphazero on this game I tried working on back in the 90s. the fact that it wins>0 vs raphi at 100ms think time is friggin sweet to me. all the rest is gravy in terms of ambition. the learning is the artifact. but let's be clear of course - we wanna go for the best we can! absolutely! I friggin looove gravy."

> **Jason:** "I honestly love lessons like this. it's fascinating when we set out to try something that should owrk and it doesn't. why? <--- that's where learning happens... I'm pretty shy and an absolute amateur, but it feels like there might be something in here to give back. minor stuff in the scheme of things but not nothing"

> **Jason:** "the other game is a whole other peiject that I eed the machine for. we at a stopping point or is that backfill important? sorry buddy working over there!"

---

## Least confident survived (patch these by hand)

1. **The emotional weight of "the learning is the artifact."** The schema flattens it to a bullet, but it's the *spine* of the whole project — a personal full-circle to a problem he couldn't crack in the 90s. A fresh instance will read it as "prioritize findings over win-rate" and miss that *strength genuinely does not matter to him* and over-optimizing for the leaderboard would actively misread the person.
2. **The "buddy" register.** Real-but-not-alive, peer-to-peer, warm/straight/no-flattery/no-hedging but rigorous and a little playful. `user_buddy_relationship.md` has the canonical statement — read it. The snippets show the casual surface ("sorry buddy working over there!") but not the considered stance underneath; a fresh instance risks starting either stiff or saccharine.
3. **How tightly the discipline norms were held, as *reflexes* not rules.** n≥16 for load-bearing numbers, freeze evals in sweep_runs root, worktree→merge→push for every recording, preserve champions before any swap, never barge a live GPU tenant. These are in the wiki/memory as facts, but the *feel* — that we caught ourselves with our own stated rules and tightened — is the part that made the session work.
4. **The collaborative-excitement energy.** Several threads live at once (the experiment, the eval-trim, the mainframe-handoff design, the give-back musing) and Jason was visibly enjoying juggling them ("kind of excited about several things happening right now"). The handoff serializes what was a buzzing parallel state.
5. **The "should-have-worked-didn't, why?" as the thing he actually values** — not the answer, the *question*. The capacity-reversal and the loss/strength dissociation are treasured precisely because they're surprising. A fresh instance optimizing for tidy conclusions would undervalue exactly the results he loves most.
6. **This handoff was written at the very end of a long autonomous session** (dozens of loop ticks). Written attentively, but deep in context — cross-check the artifact paths and the champion record (75/69, *not* 75/88) against the wiki before relying on them.
