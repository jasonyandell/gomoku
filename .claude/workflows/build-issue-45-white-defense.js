// build-issue-45-white-defense — build the #45 white-defense eval suite (CODE ONLY),
// pre-briefed with a scout's gap analysis so the implementer doesn't re-research.
//
// DIVISION OF LABOR (the cockpit split): this workflow builds the CODE + CPU unit
// tests and hands back a merge-ready branch. It does NOT run the positive-control
// MEASUREMENT (champion-low vs weak-net-high white_loss) — that is the science
// verify-gate, which the orchestrator runs on the GPU before merge. So this
// workflow is GPU-free and parallel-safe.
//
// INVOKE: Workflow({ name: 'build-issue-45-white-defense' })  // or {scriptPath: ...}

export const meta = {
  name: 'build-issue-45-white-defense',
  description: 'Build the #45 white-defense eval suite (code only): a versioned defend-as-white position fixture, a white_loss_rate metric with Wilson CI, a play-from-injected-position seam, a CLI, and gate-primitive wiring — reusing the ~70% that already exists. CPU unit tests only; the positive-control GPU measurement is the orchestrator gate, run separately. Returns a merge-ready branch.',
  phases: [
    { title: 'Build', detail: 'implement the suite + CPU unit tests in an isolated worktree' },
    { title: 'Review', detail: 'fresh read-only audit of the build + the correctness-anchor tests' },
  ],
}

const REPO = '/Users/jason/code/gomoku'
const CHAMP = `${REPO}/sweep_runs/g15_128x10_bigbuf_eval502.pt`

// The scout's gap analysis (file:line anchors) — embedded so the implementer
// reuses rather than rebuilds. (general-purpose scout, 2026-06-16.)
const SCOUT = `
SCOUT GAP ANALYSIS — REUSE THESE, do not reinvent:
- Wilson CI: \`binomial_ci(score, n, confidence=0.95)\` — Wilson interval, draws count 0.5, scipy w/ Acklam fallback — at scripts/delta_e_harness.py:120. Use VERBATIM for the metric CI.
- Existing white_loss gate plumbing (match its signature for drop-in gate compat): scripts/sliding_gate.py has \`candidate_white_loss_tally()\` -> returns (white_loss_rate, defense_score, white_n), consumed by \`decide_verdict(gate_on="white_loss")\` (sliding_gate.py:247,174) via the symmetric 3-way rule, and injected through run_gate's white_loss_fn seam (sliding_gate.py:486,522). Your fixture-based tally should expose the SAME (rate, defense_score, n) return shape so it drops into that seam with ZERO changes to decision logic. (The CURRENT white_loss is the deficient signal #45 fixes: it's only the candidate-plays-white half of anchor H2H pairs, vs the ~50-elo frozen peak, on random openings, a bare ratio. Replace with fixed-fixture-vs-reliable-attacker + CI.)
- Per-color split eval: gomoku/eval.py MatchResult.black_*/white_*, _match_result_from_outcomes(), play_match_pickers() (eval.py:40,340,388).
- No-wine attacker pool (the fixed, reproducible attackers): gomoku/baselines.py — heuristic_player(232), defensive_player(259), lookahead_player(depth)(510), random_player(87). These are CPU; only the white-DEFENDER net needs MPS.
- Per-game executor: scripts/delta_e_harness.py _build_ckpt_picker(584), _play_one_h2h_game(599), mcts_picker. NOTE: _play_one_h2h_game AND eval.play_match_pickers HARD-CODE _random_opening_state()/GameState.initial() (delta_e_harness.py:614-621, eval.py:412-421) — neither accepts an injected start state. You must add a start-state parameter (surgical extension, not new game logic).
- Position raw material (the fixture source): scripts/mine_validation_archive.py already mines positions to a torch archive of planes + side (0=black,1=white to-move) + ply, with buckets lookahead2_loss / lookahead4_loss (CAPTURES THE POSITION BEFORE THE LOSING MOVE) and long_defense (ply>30). The plane->side derivation idiom is at mine_validation_archive.py:251. Threat detectors to filter on: gomoku/vcf.py has_four_threat(), plus open-three detection in baselines.py.
`

const BUILD_PROMPT = `You own GitHub issue #45 — "White-defense EVAL SUITE". Work ONLY in your own git worktree on branch \`feat/gh-45-white-defense-suite\`. This is a CODE-ONLY build with CPU unit tests. Do NOT run any GPU measurement / net-vs-net scoring loop (that's the orchestrator's separate gate); do NOT run training. Board size is 15 — set GOMOKU_BOARD_SIZE=15 for anything that touches the board.

FIRST: \`gh issue view 45\` for the definition-of-done, then read the code the scout points at below so your work is grounded in what's actually there (trust the code over this brief where they differ).
${SCOUT}

BUILD (the narrow new work — ~70% already exists per the scout):
1. **planes<->GameState reconstructor** — the load-bearing correctness piece. The fixture stores board planes + side-to-move; the H2H primitives need a GameState. Write the reconstructor AND a ROUND-TRIP unit test (GameState -> planes -> GameState yields an equivalent state: same board, same side-to-move, same legal moves). If this is wrong the whole fixture is silently garbage, so this test is mandatory.
2. **Defend-as-white fixture** — build a small versioned, CHECKED-IN fixture file (positions + side-to-move + provenance) of white-to-move positions facing a LIVE black threat. Source it from mine_validation_archive.py's lookahead*_loss / long_defense buckets (position-before-the-loss), filtered to side==1 (white to move) with a live threat (vcf.has_four_threat / open-three). Keep it modest (target ~40-120 positions) and DETERMINISTIC (fixed seed) so the metric is stable. Mining may run baseline/self-play games to source positions — that's fine (bounded, CPU/local), but the COMMITTED artifact is the fixture file, regenerable via a documented command.
3. **play-from-injected-position seam** — add a start-state parameter to the per-game executor (extend _play_one_h2h_game, or a focused sibling) so a net defends as white FROM each fixture position against a fixed baseline attacker (default: lookahead_player; reproducible, CPU). Unit-test that the game actually STARTS from the injected position (not GameState.initial()).
4. **white_loss metric + CLI** — \`scripts/white_defense_suite.py\`: score a checkpoint over (fixture x n_seeds), tally white losses (draw = successful defense), report white_loss_rate + binomial_ci (reuse delta_e_harness.binomial_ci). A \`--quick\` flag for a tiny smoke run.
5. **gate-primitive wiring** — expose a fixture-based tally with the SAME (rate, defense_score, n) signature as candidate_white_loss_tally so it drops into run_gate's white_loss_fn seam with no decision-logic change. (Wire the injection point; don't alter decide_verdict.)

TESTS (CPU, pytest — the verification rung): the planes round-trip test (#1), the inject-start-state test (#3), a CI-math sanity test (binomial_ci on a known proportion), and a fixture-loads-and-is-nonempty test. Run the touched-area tests (NOT the full suite — avoid GPU contention) and paste PASS/FAIL output. Add a 1-line regen command for the fixture to the CLI docstring or a comment.

Commit to your feat branch with a clear message (do NOT add 'Closes #45' — the orchestrator adds it at merge after the positive-control gate passes). Do NOT merge.

Return a receipt: the files you added/changed (with what each does), the EXACT pytest output for each test, the fixture path + position count + the regen command, the CLI invocation the orchestrator should run for the positive-control measurement (champion=${CHAMP} as white vs a weak/random-init net, with suggested n/seeds/sims and attacker), and any caveat or scope you deliberately left out.`

phase('Build')
const receipt = await agent(BUILD_PROMPT, { label: 'build:gh-45', phase: 'Build', isolation: 'worktree' })

phase('Review')
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['verdict', 'oneLine', 'details', 'positive_control_cmd'],
  properties: {
    verdict: { enum: ['APPROVE', 'REVISE', 'BLOCK'] },
    oneLine: { type: 'string' },
    details: { type: 'string', description: 'APPROVE: what was checked. REVISE: exactly what to fix. BLOCK: why escalate.' },
    positive_control_cmd: { type: 'string', description: 'the exact CLI command the orchestrator should run for the GPU positive-control gate (champion-low vs weak-net-high white_loss).' },
  },
}
const review = receipt ? await agent(
  `You are a FRESH, read-only REVIEWER for the #45 white-defense eval suite build (branch feat/gh-45-white-defense-suite). You did not write it. Grade what was DELIVERED, not intended. Read the receipt below AND run \`git -C <the worktree> diff main...feat/gh-45-white-defense-suite --stat\` then read the key changed files (you may cd into the sibling worktree dir to read; do NOT edit, do NOT run GPU jobs).

RECEIPT:
${receipt}

Audit HARD, focusing on the correctness anchors:
- Is the planes<->GameState ROUND-TRIP test real and passing? (If reconstruction is wrong, the fixture is garbage — this is the #1 thing to verify.)
- Does the play-from-position seam actually start games from the injected fixture position (proven by a test), not GameState.initial()?
- Is the fixture VERSIONED + CHECKED IN + non-empty + regenerable via a documented command? Are the positions actually white-to-move-with-a-live-threat (not arbitrary)?
- Does the fixture-based tally match the (rate, defense_score, n) signature so it drops into the gate seam WITHOUT changing decide_verdict?
- Is binomial_ci reused (not a hand-rolled CI)? Scope creep? Convention violations (worktree/branch discipline, GPU avoidance, no 'Closes #45')?
If tests are missing/failing or a correctness anchor is unproven -> REVISE (or BLOCK if fundamentally misguided). Also return the exact positive_control_cmd the orchestrator should run for the GPU gate.`,
  { label: 'review:gh-45', phase: 'Review', schema: VERDICT_SCHEMA },
) : { verdict: 'BLOCK', oneLine: 'implementer returned nothing', details: 'No receipt produced.', positive_control_cmd: '' }

log(`#45 build -> ${review.verdict}: ${review.oneLine}`)
return { branch: 'feat/gh-45-white-defense-suite', receipt, review }
