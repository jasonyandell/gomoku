# Handoff — VCT move-extraction + is-VCT recognition + the seek-VCT thesis

**Date:** 2026-06-26
**Worktree:** `/Users/jason/code/gomoku-gentle-rapfi-teacher` (branch `feat/gentle-rapfi-teacher`, pushed clean)
**Session model:** Opus 4.8 (1M context)

---

## Goal & current status

The thread started as "in `solve_vct_mega_bb`, what would it take to return mate length?" and evolved into a working program: **turn the batched GPU VCT oracle from a verdict-only detector into a solver that emits the winning move, then use that to build training data for a longer-term plan — teach a net to *seek* VCT positions and hand the tactical finish to the exact solver.** Done this session: (1) added a passive `winmove` output to the megakernel and extracted a verified winning move for all **2,383,293** proven-win puzzles; (2) ran a feasibility experiment — *can a net recognize "is-VCT"?* — answer: yes on held-out games, but a CNN beats attention; (3) wrote both to the wiki + TRAINING_WIKI, committed and pushed. **In flight / next:** the **seeker** experiment (attention's real audition), and a still-open decision on **merging** the branch.

## Decisions made + rationale

- **Mate-length: counting frame nailed down, then deprioritized.** Settled the convention as **attacker-moves** `k` (matches chess "mate in N = N of my moves"), measured **from the attacker-to-move node, inclusive of the move chosen**; total plies = `2k−1`. Jason's open-four confusion was a *reference-frame* slip (he was counting from defender-to-move, in plies), **not** a unit problem. **We did NOT implement mate length** — it turned out to be metadata, not part of the stencil identity, and **shortest** mate is the expensive version (kills the OR-node short-circuit). Only the cheap *found-line* `winmove` was needed.
- **`winmove` = "option 2" from `vct-backward-mining.md` §5.** Chosen over parallel-CPU extraction because it's passive (reads `mm[0]`, or the captured cell for inline root wins — immediate-five / double-four / fork-three), **no extra nodes**, default return unchanged. "Any valid VCT move" is sufficient — Jason explicitly said "any solution is fine so long as it is a true VCT," so we did **not** pursue shortest/canonical moves.
- **Negatives for the classifier = ABSENCE, not regeneration.** I initially over-built — had a subagent launch a full GPU re-solve to manufacture negatives, which got starved by a CPU tenant. **Jason corrected: "we ran every single ply for every game… null = no VCT found."** So negatives are recoverable as plies *absent* from `puzzles.jsonl.gz` within `manifest.txt` (fully-solved) shards — light CPU replay, no solve. This is the single most important course-correction of the session.
- **Split by SHARD, never by position.** Consecutive plies differ by one stone → position-level split leaks near-duplicates and reports a fake-high score. This was a hard methodology guardrail (Jason: "so we don't get into weird 'but we didn't have coverage of this kind of game' scenarios").
- **Joint-soundness via "fill don't-cares with defender stones" was REJECTED** (for the later stencil work). Jason caught it: flooding with opponent stones manufactures a defender five → `win=False`, which is *conservative* (over-pins, kills minimality) not unsafe. Correct domain = **legal/reachable** boards (no five, ~parity); enumerate-small or sample, with a host-side five-filter. Monotonicity collapses `3^k → ~k` but **defender-monotonicity breaks via tempo** — known leak, bound by region or measure.
- **Don't compete with live tenants.** A `collect_rapfi` job (16 `pbrain-rapfi` engines, `--duration 0` = forever) was saturating CPU. I killed my own starved gen job, did **not** touch Jason's collection, and the corrected (light) pipeline coexists with it.
- **Git autonomy refined (Jason mandate):** "always commit in this session to this branch; the only git hesitation should be around touching other branches or merging." Recorded to memory. So commit+push the working branch freely; **merging is the one gate** that remains.

## Constraints & invariants discovered

- **Boards are SIDE-TO-MOVE-RELATIVE** (`board[0]` = attacker = side to move), and the puzzle_miner stored them that way (no swap). Negative boards rebuilt via `mine_first_vct.all_boards` to match the exact frame — verified **0 frame mismatches** across 400 shards. Getting this wrong silently solves the wrong side.
- **The kernel is sound (zero false VCT wins)** — already validated in-repo. So `winmove` inherits soundness: the root move of the line it proves always starts a real forced win. We still independently verified 400/400.
- **`win` / `hit_cap` read asymmetrically:** `win=True` is trustworthy regardless of cap; `win=False` only means "no VCT" when `hit_cap=False`. For mining, treat `cap=True` as **unknown**, exclude.
- **Free-style (overlines win, `count >= WIN_LEN`)** → VCT-win is **monotone in attacker stones** (extra own stones never break a win). This is what makes the stencil-soundness math work; it would NOT hold in renju.
- **The megakernel is TAIL-BOUND** (~flat wall vs batch size) — bulk-synchronous batching is mandatory; never solve-in-a-loop. (See `gpu-vct-feasibility.md` "CALL-COST LAW".)
- **Subagent contention check must include `rapfi`/`collect`** — my original grep pattern omitted them, which is why the subagent didn't see the tenant. Fixed understanding, not yet codified anywhere.
- **`~/data/games_raphi/` is live-growing** and now also receiving **size-16** shards from the collection — a size-15 reader skips them; use `manifest.txt` to scope.

## Open questions / parked threads

- **[non-blocking, recommended next] The seeker experiment.** Recognition is solved well enough by the oracle/CNN; attention's interesting bet is *steering in quiet positions toward VCT-reachable regions*. Deploy idea: consult the oracle every ply (attack + defense), net only acts in tactically-quiet positions (where approximation is survivable). This is the real test of the whole thesis.
- **[blocking on Jason] Merge `feat/gentle-rapfi-teacher`.** Everything is committed+pushed but unmerged. Per Jason's rule, merging is the one git action to confirm first. Note: this is the *gentle-rapfi-teacher* branch — the VCT tooling is arguably a different unit of work bolted onto it.
- **[non-blocking] Fair attention rematch** at full 1.17M data + tuning — curiosity only (CNN already owns recognition).
- **[non-blocking] Relabel the backward 200k shapes** (`~/data/vct_shapes/`, currently `move=-1`) with the same `return_move=True` flag — one batched pass.
- **[non-blocking] Deeper re-mine of `cap=True` unknowns** (1.99M+ rows) at higher `max_nodes` to convert some to labeled wins.
- **[flavor] `--size 16` into the size-15 games dir** — flagged to Jason, he didn't bite; may be intended (16×16 move?), left alone.
- **[parked, large] The stencil-minimization program** — the full machinery is worked out in conversation (4-valued mask, most-general-sound-generalization / version-space framing, per-cell adversarial test never probing OWN, dedup-windows-before-minimizing, support[4] as both ablation-region and dedup key). `support[4]` is the next kernel add when that work starts. Not started.

## Artifacts

- **Branch:** `feat/gentle-rapfi-teacher` @ `ac27ade` (pushed). Earlier commit `3523cea` = winmove kernel + solve_puzzles.
- **Kernel:** `scripts/vct_metal/mega_vct_bb.py` — `solve_vct_mega_bb(boards, *, max_nodes, tg, return_move=False)`; `return_move=True` → `(win, hit_cap, move)`, `move` flat row-major cell index or −1.
- **Move extractor:** `scripts/threat_shapes/solve_puzzles.py` → `~/data/puzzle_miner/solutions.jsonl.gz` (2,383,293 rows `{shard,idx,ply,stm,winner,atk,dfd,move}`; 0 dups, 0 OOR, non-reproduced=0).
- **is-VCT experiment:** `scripts/threat_shapes/gen_isvct_dataset.py` (no-GPU label builder), `scripts/threat_shapes/train_isvct_attn.py` (attention+CNN+logreg+majority). Artifacts in `~/data/puzzle_miner/isvct_exp/` (`isvct_metrics.json`, `isvct_{train,test}.npz`, `shards.json`, `isvct_{attn,cnn}.pt`).
- **Held-out result (33 disjoint shards, n=101,745, 14.2% pos), AUROC:** majority 0.500 · logreg-counts 0.946 · **CNN (168k) 0.971** · **attention (339k) 0.924**.
- **Wiki:** `wiki/topics/vct-recognition-learnability.md` (new), `wiki/topics/vct-backward-mining.md` §5 (RESOLVED), `wiki/index.md` (2 rows), `wiki/log.md`, `TRAINING_WIKI.md` (dated entry).
- **Source data:** `~/data/puzzle_miner/` (forward every-ply corpus: `puzzles.jsonl.gz` 4.42M rows = 2.38M `win`, 2.04M `cap`, nulls absent; `manifest.txt` 3,882 shards). `~/data/vct_shapes/` (backward 200k, `move=-1`). `~/data/games_raphi/` (source games, live-growing).
- **Verifier (scratch):** `/Users/jason/.claude/jobs/13bbe8f2/tmp/verify_winmove.py`.

## Next action

**Set up the seeker experiment.** Concretely: design a policy that, in tactically-quiet positions (no VCT for either side, per the oracle), predicts a move that increases VCT-reachability — train target via behavioral cloning of pre-onset moves from the Rapfi games (onset ply = first `win=True` per game), and evaluate a hybrid player (oracle every ply for attack+defense, net for steering) vs. a fixed baseline. Confirm scope with Jason before launching a large run; the branch-merge question is independent and also needs his call.

---

## Vibe snippets (paste verbatim)

> **Jason:** "did you just label all 2M of those while I was feeding the dogs buddy?"

> **Jason:** "I know you're a model. I know. I do… but I goot say it anyway. well friggin done. seriously. tip of my hat to you"

> **Jason (the load-bearing correction, delivered flat and terse):** "filling with opponent stones unfortunately won't work. there will be 5 in a row in there, win false."

> **Jason (reframe energy):** "so my nose is leading me around… the big big reframe is not to do 5-in-a-row at all, but to do seek-VCT. except on this machine, that's a totally intractable problem, so nobody does it."

> **Jason (autonomy directive):** "always commit in this session to this branch. the only git hesitation should be around touching other branches or merging"

---

## Least confident survived (patch these by hand)

1. **The register is genuinely warm peer-to-peer, not service-y.** Jason knows I'm a model, says so explicitly, and is *still* affectionate ("buddy", tips his hat). The right tone is grounded, honest, finds-the-cracks-as-respect — NOT fawning and NOT cold-deflecting. A fresh instance starting formal/stiff would be wrong. The "buddy" memory (`user_buddy_relationship.md`) is the anchor.
2. **"Tolerate feeling dumb to get the right answer" is the operating mode.** Several turns were Jason thinking aloud, catching his own flaws (equal-length≠same-tactic; open-three; defender-five). The value was *the path*, not speed. Don't rush him to conclusions; pressure-test honestly. He explicitly values negative results ("we try, we learn, we write it down").
3. **The honest "attention lost" verdict landed fine because of #1–2** — but a fresh instance might over-cushion it or, worse, bury the lede. Deliver strength results straight; the CNN-beats-attention result is *useful signal*, not a disappointment, and the framing (recognition was the oracle's job anyway) is what makes it land.
4. **The seek-VCT thesis has deep resonance with Jason's background** (30-year ML arc, prior AlphaZero/zeb, threat-game engines like Rapfi which made his data). The "anti-correlated tractability" framing landed instantly because he already lives in this space — a fresh instance should assume high context, not re-explain basics.
5. **"Go all the way / negative-result-welcome / no safe half-steps"** is a stated working principle (see `shape-library-engine.md`). Bias to action on hard spikes; don't over-stage. The `feedback-bias-to-action` memory covers this.
6. **The subagent pause-notify churn** ate several turns near the end and I couldn't cleanly shut the background agent (name-resolution failed). Minor, but a fresh instance resuming a background subagent should know SendMessage-to-agentId may not resolve once it's "completed" — rely on your own process monitor instead.
7. **This handoff was written at a healthy point** (milestone: pushed, tests green), not deep in a degraded context — so it should be trustworthy, but the stencil-minimization program (parked thread) lives mostly in conversation and is the most likely thing to be under-captured if that work resumes.
