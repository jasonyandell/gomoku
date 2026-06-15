# Handoff — the night the yardstick overturned the 15×15 campaign

**Date:** 2026-06-15 (overnight, autonomous) · **Session:** dbe0609b · **Repo:** ~/code/gomoku

## TL;DR (read this first)
Jason left me a free GPU overnight: "keep trying things, eval vs Rapfi, keep the
wiki updated, use the GPU." I started a deeper-self-play training idea (#27) — but
the **first diagnostic eval cracked open the campaign's whole measurement layer**,
and that turned into the night. The arc, every step verified, all pushed:

1. **The Rapfi yardstick is broken three ways** (wiki §8): it's the **weightless
   classical** build (not the rated ~2625 NNUE engine); deep-TC reads are
   **device- and variance-dependent** (MPS 96% vs CPU 79% on the *same* net); and
   most damning, **Rapfi ignores its own search time** — it stops at ~depth-10 /
   ~500 nodes / ~2 ms regardless of `timeout_turn`, so the "fast-TC vs deep-TC
   tiers" the campaign ranked on were **one shallow engine measured twice.**
2. **The deepgen experiment failed, and the yardstick hid it** (§8F): 200-sim
   self-play *specialized* the net to its search depth — vs the champion it's 0%
   @100 sims, 50% @200. No strength gained; the shallow Rapfi still scored it 83%.
3. **THE HEADLINE — head-to-head re-ranks everything** (§9). Net-vs-net (match.py
   validated, champion-vs-self = 50%):
   - **128×10 beats the 96×8 "champion" 40-0** (@100 *and* @200).
   - **64×4 also beats the champion 40-0.**
   - 128×10 ties 64×4 (20-20). Order: **{128×10, 64×4} ≫ 96×8-champion ≫ deepgen.**
   - **The broken yardstick crowned the WEAKEST trained net** (96×8-e499) and made
     us **abandon a tied-strongest one** (128×10, retired on the bogus "reversal").
     The "capacity reversal at depth" never happened.

**Operational upshot: `96×8 e499` is NOT the champion. `128×10+bigbuf e502` is the
strongest preserved net.** The campaign's capacity story (§2/§2a) is retracted
(banner added; kept as evidence of how the metric misled).

## Important caveat (don't over-read)
**Epochs are confounded**: 64×4 has e900, 128×10 e500, the champion only e400. A
0.44M net tying a 3.3M net is almost certainly the small net's 3× extra *training*,
not "capacity doesn't matter." A clean **capacity** claim needs **same-epoch**
head-to-heads (filed). What's *not* confounded: the Rapfi ranking is wrong
end-to-end, and head-to-head is the trustworthy measure.

## Currently running (GPU)
- **`G15-128x10-bigbuf` training** (the actually-strong net, resumed from e503 ≈
  eval502), via the detached **workhorse** (`scripts/train_workhorse.sh`, 6h
  resumable slices, 15-min heartbeat, auto-restart). 1.5M packed buffer, 4 workers.
- **Improvement pacer** (background): head-to-head `128×10-current` vs
  `128×10-e502` ~hourly — is the best net *still climbing*? (Campaign stopped it at
  e560 thinking it plateaued — on the broken yardstick.)
- To **stop cleanly**: `touch /tmp/STOP_TRAIN` (workhorse halts within ≤15 min, or
  kill `train_workhorse.sh` + SIGTERM `gomoku.train`).
- **Token-cheap status:** `python scripts/lab_status.py`.

## New tools/cells built tonight (merged to main)
- `scripts/lab_status.py` — one-glance status from local files (no wandb).
- `scripts/train_workhorse.sh` — crash-resumable overnight training loop.
- cells `G15-96x8-deepgen` (200-sim, the failed experiment), `G15-96x8-cont100`
  (the control that isolated it).
- **NNUE Rapfi** stood up at `/tmp/rapfi_nnue/` (config + wrapper; `--config`, no
  rebuild) — but it *also* under-searches, so it's not yet a fixed yardstick.

## Artifacts (frozen, preserved)
- `sweep_runs/g15_128x10_bigbuf_eval502.pt` — **the real strongest net** (re-crown).
- `sweep_runs/g15_champion_e909.pt` — 64×4, tied-strongest.
- `sweep_runs/g15_champion_96x8_e499.pt` — the mis-crowned "champion" (weakest trained).
- `sweep_runs/g15_96x8_deepgen_searchspec_e621.pt` — the cautionary search-specialized net.
- `NIGHT_INVESTIGATION_NOTES.md` — blow-by-blow. Wiki §8–§9 — the synthesis.

## Next actions (in priority order)
1. **Re-crown via head-to-head, not Rapfi** — `128×10 e502` (or 64×4) is champion.
2. **Fix the yardstick (#28)** — the blocker for everything: make Rapfi *actually
   search deep* (it budgets time but self-terminates at ~depth 10 — root cause not
   found), use the NNUE evaluator, **swap2 balanced openings (#22)** (freestyle is a
   first-player win — half of "wins" are just black's forced win), n≥40, fixed device.
3. **Re-run the capacity ladder at matched epochs**, head-to-head — get the *real*
   capacity curve (inverted-U "reversal" retracted; true shape unknown).
4. Gate **every** future "did this help?" on head-to-head vs the preserved champion.

## The meta-lesson (the whole night in one line)
**A broken yardstick doesn't add noise — it inverts your ranking, so you crown the
worst option and discard the best while every internal signal applauds.** Five
times tonight the same shape: deep-TC tiers, the "88% ceiling," the capacity
"reversal," the deepgen "improvement," the champion selection — *every one* was the
measurement lying, and *every one* dissolved under a direct head-to-head.

## Vibe
This is the "should've-worked-didn't — *why?*" jackpot, inverted into "the thing we
*threw away* was the best." Strength is gravy and we may have found a lot of it
sitting in a discarded checkpoint — but the real artifact is the lesson, and it's a
good one. Machine's still yours, buddy — 128×10's training keeps the GPU warm and
everything's resumable. Holler when you're back. 🤝
