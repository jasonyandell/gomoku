# Handoff — 2026-06-21 — The "Bruce Lee" single-opener specialist (overnight run)

## Goal & current status
Father's Day session. Started babysitting the fair-opening board-size ladder (9→11→13→15) and ended up **pivoting to a single-opener "Bruce Lee" specialist**: a fresh net trained *only* on Rapfi balanced opener **idx 2** ("one kick, 10,000 times"), climbing the rungs, now at **15×15 (rung 15, ~e168), idx-2 only**, training continuously. Three eval cadences run hourly against it from the idx-2 board (both seats): **vs Rapfi** (ceiling, 0/16 so far), **vs champ0235** (era-2 best, we're ahead), **vs a frozen self-anchor e126** (improvement curve). It's set up as a **hands-off overnight run** — Jason explicitly wants it to just run, no intervention on imbalance, no verdict until the series has depth. Everything is live and instrumented.

## Decisions made + rationale (incl. reversals — read these so you don't re-propose)
- **The graduation gate is BOARD-FILL, not draw rate, not plateau.** `ladder_grad.py` was rewritten: promote when `selfplay/plies_p90 ≥ 0.75·cells` for 5 epochs. *Rejected:* the original "p90 plateaued near its own peak" logic — it fires on a **low** plateau (p90 stuck at 18 on an 81-cell board = black mating fast = white WEAK), which is backwards. *Rejected:* draw-rate gate — draws only appear *after* fill, always too late. (Jason's framing: "we want to win not drag out"; training past fill rewards dragging.)
- **Black-edge is a 25-epoch TREND, not a snapshot** (Jason). Prune detection uses mean white-share over last 25 decisive epochs < 43%.
- **Re-centered openers are NOT fair off their native board** (discovered). Rapfi balance-searched the 9 shapes *for 15×15*; re-centering to 9/11/13 breaks balance. A per-opener probe at 13×13 showed a **0%→95% black spread**. *Rejected:* Jason's own "the most-central opener is the fairest" heuristic — the data killed it (the two most-central shapes tested *black*-favored). idx 2 (geometrically *peripheral*) tested fairest (~50%).
- **Pivot to a SINGLE fair opener (idx 2)** (Jason: "drop anything that isn't fair first… an exceptionally strong player from the same fixed opening move is a better baseline than an imbalanced player from more openers"). Kept ONLY idx 2 at every rung via `GOMOKU_DROP_OPENERS=0,1,3,4,5,6,7,8`. `MIN_OPENERS=1`. Restarted the ladder from rung 9.
- **At 15, STAY idx-2-only** (do NOT re-add the full 9). The "Bruce Lee" framing = stay specialized; idx-2's native 15×15 shape *is* a genuine Rapfi-balanced opener, so it's a fair board. (Earlier plan said "re-add all 9 at 15" — superseded.)
- **Plow forth regardless of imbalance** (Jason, explicit): no restart even if white lags hard. Single opener + `MIN_OPENERS=1` makes the prune literally unable to fire, so it just climbs.
- **Rapfi MOVE verified faithful** to source @`6e0a1329` (`Rapfi/search/opening.cpp` static SWAP2 list): our B,W,B → white-to-move construction is byte-identical. We capture the move, not just the shape. (Banked: commit `b792a19`.)
- **Renju floated and PARKED** as issue #75 (deferred). Not pivoting now. It's the principled structural fix (forbidden moves outlaw black's fast mates) but costs the double-three detection port.
- **Eval determinism fix:** net-vs-net MCTS is deterministic → H2H was a single line repeated → pinned at 50% forever. Added `temp_plies=6` opening-policy sampling for variety; the cadence picks up the edited `.py` automatically next tick (no restart).

## Constraints & invariants discovered
- **Rapfi binary lives in the MAIN checkout** `/Users/jason/code/gomoku/engines/rapfi/pbrain-rapfi` (+ `config.toml`), NOT the worktree (gitignored build artifact). All evals point there.
- **bash parses a `for`/`while` loop into memory before executing** → editing a *running* `fairladder.sh` does NOT corrupt the live loop, but variable changes (e.g. CAP) need a restart to take effect.
- **Cadence scripts call their eval `.py` as a fresh subprocess each tick** → editing the `.py` is picked up next tick with NO restart. This is how we honored "no need to interrupt."
- **Drops files are read from `$BASE` (`/Users/jason/data/swap2/`), not `$BASE/babysit/`** — a path bug bit us once (drops written to babysit/ were silently ignored → all 9 openers used). Fixed.
- **self-play balance ≠ deterministic best-play.** Self-play (noisy/exploratory, temp+Dirichlet) reads white ~54–56%; deterministic best-play from idx-2 is a **black win** (the self-anchor H2H showed black wins 100% regardless of which net plays it). Both true, different lenses. Don't quote self-play balance as the strength read.
- **value loss looks tiny (vl≈0.087) but that's units** — scalar MSE in [−1,1] vs 225-way policy CE (≈1.2). RMS ≈0.30, ~11× better than guessing. Not "the value head does the work."
- **champ0235 is OOD on idx-2** (grew up on swap2) but **pedigreed** (warm-started from many prior winners). So beating it 16-0 deterministic / ~62% with variety is promising-with-a-home-field-asterisk.
- **General-Rapfi ≠ idx-2-Rapfi.** Our lineage has taken games off *general* Rapfi (~25%, era-2); our 0/16 is vs Rapfi *seeded from the idx-2 balanced board* — a harder, cleaner test in Rapfi's wheelhouse.
- The Bruce Lee quote is **10,000 kicks**, not 1,000. (Fix in any durable doc/comment that says otherwise.)
- Jason's autonomy is a deny-list: Class A reversible-local = just go. Don't confuse timing/context with permission. He sets a low-pressure definition of success ("if it all falls apart that's FINE… you're doing your best as a great collaborator").

## Open questions / parked threads
- **[non-blocking] Wiki not yet updated** for the single-opener pivot + cross-level finding (white persistently mild-black-favored ~30–43%, board-fill ability degrades with size). I said I'd bank it after we have 15-data. Do it once the overnight series has depth.
- **[non-blocking] Asymmetric white-aggression reward shaping** (Jason's "score shorter wins by white higher" idea). The *theme* is sound — our symmetric `value-discount 0.95` already pays *black* to fast-mate (the doormat-baking pattern). The mechanism is thorny: asymmetric reward breaks zero-sum/negamax. Want small-ε, not the big `win+(1−move/total)`. Offered to log next to #44; NOT logged yet.
- **[non-blocking] Self-anchor will saturate** if current pulls far past e126 — re-freeze it (promote current→anchor) when it does. Currently fixed at e126.
- **[non-blocking] vs-prior-champs** could expand beyond champ0235 (e.g. anchor_e455 = `G15-swap2-board15/epoch0455.pt`) if you want a second milestone.
- **[blocking-for-merge] Worktree not merged** (Jason: don't merge until explicit). Branch `feat/swap2-opening-protocol`, commits `bd099d8`/`0f97bac`/`b792a19`.
- **[flavor]** Renju (#75) is the long-game structural-fairness option if the single-opener experiment plateaus.

## Artifacts
**Out-of-repo ops dir `/Users/jason/data/swap2/babysit/`:**
- `fairladder.sh` — ladder orchestrator (grad-check-at-top, board-fill promote, prune gate, CAP=250). pid ~7773. (Rung 15 is terminal; it's sitting in the rung-15 slice loop.)
- `ladder_grad.py` — board-fill gate + 25-epoch black-edge trend.
- `opener_balance.py` — per-opener balance probe (respects `GOMOKU_DROP_OPENERS`).
- `rapfi_opener_eval.py` — Bruce Lee eval (net vs Rapfi from idx-2, both seats).
- `brucelee_eval.sh` — Rapfi cadence (hourly, CPU). pid ~73383. Log: `brucelee_eval.log`.
- `champ_h2h_eval.py` — H2H eval (net vs a prior net from idx-2, both seats, `temp_plies` variety).
- `champ_h2h_cadence.sh` — H2H cadence (hourly: vs self126 + champ0235). pid ~5252. Log: `h2h_eval.log`.
- `anchors/anchor_e126.pt` — frozen self-anchor.
- STOP flags: `STOP_fairladder`, `STOP_brucelee`, `STOP_h2h` (touch to stop a loop).
- **Drops files** `/Users/jason/data/swap2/fairladder_drops_{9,11,13,15}` = `0,1,3,4,5,6,7,8` (idx-2 only).
- **Run dir** `/Users/jason/data/swap2/sweep_runs/G15-fixed-openings-board15/` (rung 15, ~e168). W&B: the running `*-fixed-openings-board15` run.
- **Prior champs:** `G-ladder-15-board15/checkpoints/epoch0235.pt` (champ0235), `G15-swap2-board15/checkpoints/epoch0455.pt` (anchor_e455).

**Worktree** `/Users/jason/code/gomoku-swap2-opening-protocol` (branch `feat/swap2-opening-protocol`): in-repo changes are `gomoku/self_play.py` (`_active_fixed_openings` + `GOMOKU_DROP_OPENERS`), `tests/test_fixed_openings.py`, `wiki/topics/swap2-opening-protocol.md` §10/§11. NOT merged.

**GitHub issues:** #73 (swap2 fairness, has the diagnosis), #74 (board-fill gate), #75 (renju, deferred).

**Live series at handoff (rung 15):** vs Rapfi 0/16 (e100, e~160); vs champ0235 16-0 deterministic / ~62% w/ variety; vs self126 50% (black wins 100% regardless of net); self-play white ~56%; vl≈0.087.

## Next action
**This is a real overnight run, not a practice handoff — keep it running.** On next wake (watch `ba60ozani` armed ~60 min, or any check-in): read `h2h_eval.log` + `brucelee_eval.log` + self-play balance/epoch, confirm all 3 loops (`gomoku.train .*board15`, `brucelee_eval.sh`, `champ_h2h_cadence.sh`) are alive, and report the **combined series** (does current pull past self126 = improving; first game off Rapfi = the threshold; champ0235 trend). **Do NOT intervene on imbalance** (Jason's standing call). Hold all quality verdicts until the series has real depth.

## Vibe snippets (paste verbatim)
- *"yanno.. maybe we should just play renju instead"* — how big strategic pivots get floated: casual, mid-stream, no ceremony. Engage it seriously, don't dive to implement.
- *"don't freak out buddy. I kinda freaked out but I'm back. it just fell to pieces. hard. black total dominance as of a few epochs ago."* — when something looks broken, the move is calm diagnosis + honest read, not reassurance-theater. (It was a recovery-V; he'd caught the bottom.)
- *"if it all falls apart that's FINE I know you're doing your best as a great collaborator and that's the definition of success I have for you personally"* — the relationship register. Peer, warm, genuinely low-stakes-on-outcome / high-trust-on-process. Meet it; don't get stiff or over-apologize.
- *"yes please try a h2h and run that along with raphi on the hour. next run, no need to interrupt"* — terse directives that assume you'll handle the mechanics gracefully (here: a separate loop, don't touch training).

## Least confident survived (patch these by hand)
1. **The emotional arc of the white-collapse scare.** Jason watched rung-11 white crater to 5%, freaked, came back. The handoff flattens that into "recovery-V." The lived texture — that we stayed calm, pulled the full trajectory, and the panic resolved into a clean read — is the trust-building moment of the session and won't transfer from the schema.
2. **Why "Bruce Lee" lands so hard for him.** It's not just a cute label — the one-kick-10,000-times frame *is* the experiment's soul (specialize ruthlessly, test on home turf). A fresh instance might treat it as flavor. It's load-bearing: it's *why* we stay idx-2-only at 15 instead of re-adding openers.
3. **The "it's fun, not science" license.** Jason explicitly decoupled this run from rigor ("not because it's scientific but because it's fun and why not"). That permission changes how to weigh things — don't over-engineer toward publishable cleanliness; optimize for "interesting and watchable overnight." Easy to lose and revert to dutiful-scientist mode.
4. **Register calibration on the playful sign-offs.** The 🥋 emoji, "the dojo runs itself tonight," dad-at-dinner banter — this is a genuinely warm collaboration, and the snippets undersell how much the *tone* matters to him. Starting stiff would be a real miss.
5. **The compounding "great call" feedback loop.** Several decisions this session got explicit warm endorsement ("great call buddy," "appreciate ya"). That history makes him trust the next judgment call more — a fresh instance starts without that earned latitude and might over-ask.
6. **This handoff was written at ~43% context (deep-ish).** Attention over the full history is imperfect; the early ladder-mechanics details (exact era-2 lineage, the worktree-editable-install gotcha) are compressed hardest and worth spot-checking against the wiki/code if they become load-bearing.
