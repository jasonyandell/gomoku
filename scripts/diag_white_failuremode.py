#!/usr/bin/env python3
"""Diagnose HOW the net loses as WHITE (the defender) vs native Rapfi (attacker).

MEASUREMENT-ONLY (no training, no edits to the gomoku package). Tests Jason's
hypothesis: white doesn't lose to Rapfi by making SIMPLE tactical errors (failing
to block a four / open-three) — it loses by "competently retreating until it
dies", i.e. it blocks the obvious forced threats fine but never seizes initiative
and simply delays a forced loss. If true, the conv block-teacher (which fixes
exactly the simple-block errors) has little to correct.

Regime mirrors the Rapfi white-defense gate:
  * board 15 (GOMOKU_BOARD_SIZE=15 — the checkpoint is a 15x15 wdl net)
  * net = WHITE (second mover, defender), native Rapfi = BLACK (first mover,
    attacker)
  * net sims=400; Rapfi timeout 1000ms; 4-stone random balanced openings; seed 7
  * 30 games

At every WHITE-to-move position, BEFORE white plays, we capture
``state.to_planes()`` and run ``gomoku.self_play._conv_block_detect(planes,
tier2=True)``. ``planes[0]`` is the side-to-move (= white, the defender) and
``planes[HISTORY_PLY]`` is the opponent — so at a white-to-move position the
detector naturally analyzes white-as-defender. We record, per white ply: whether
the detector fired, the stamped cell set, the threat tier (T1 forced-four-block
vs T2 open-three), white's ACTUAL chosen move, and the raw opponent-four count
(``len(opp_win)``) so the endgame "genuinely-lost double-four" case (which the
detector collapses into None) can be separated from a blockable-but-missed
threat.

We replicate the EXACT move generation + coordinate mapping of
``gomoku.eval.play_match_pickers``: the same ``mcts_picker`` for white, the same
``ExternalEnginePlayer`` for black (its BOARD-mode protocol reconstructs the full
board from ``state.board`` each call, so driving it manually is identical), and
the same ``_random_opening_state`` for the opening. The only difference is that we
fix net=white for all games (no color alternation) and log every white ply.

Outputs a summary block to stdout and the raw per-ply log to
``sweep_logs/diag_white_failuremode_wdl0.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone


def _ensure_worktree_gomoku() -> None:
    """Repoint the shared venv's PEP 660 editable finder at THIS worktree's
    ``gomoku/`` — process-local only, never persisted to the finder file.

    The shared venv installs ``gomoku`` via a uv ``MetaPathFinder`` whose
    ``MAPPING`` hardcodes one ``gomoku/`` dir; it shadows PYTHONPATH/sys.path, so
    a worktree's local code only loads if we rebind ``MAPPING['gomoku']`` in
    THIS process before the first ``import gomoku`` (the documented
    machine-editable-install-worktree-gotcha workaround). We only do this when a
    sibling ``gomoku/`` package dir exists next to this script's repo root, and
    we DO NOT touch the on-disk finder file — the change dies with the process.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    pkg = os.path.join(here, "gomoku")
    if not os.path.isdir(pkg):
        return
    try:
        import __editable___gomoku_0_1_0_finder as F  # type: ignore
    except Exception:
        return
    cur = F.MAPPING.get("gomoku")
    if cur != pkg:
        F.MAPPING["gomoku"] = pkg
        print(f"[bootstrap] repointed editable gomoku MAPPING (process-local):\n"
              f"            {cur}\n         -> {pkg}")


_ensure_worktree_gomoku()


def _coord(action: int, n: int) -> str:
    r, c = divmod(int(action), n)
    return f"({r},{c})"


def run(
    *,
    checkpoint: str,
    rapfi: str,
    n_games: int,
    sims: int,
    timeout_ms: int,
    opening_moves: int,
    seed: int,
    out: str,
    sanity_games: int,
) -> None:
    import numpy as np

    from gomoku.board_config import BOARD_SIZE
    from gomoku.eval import mcts_picker
    from gomoku.external_engine import ExternalEngineConfig, ExternalEnginePlayer
    from gomoku.game import HISTORY_PLY, GameState
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.self_play import _conv_block_detect, _random_opening_state
    from gomoku.util import pick_device
    from gomoku import vcf

    n = int(BOARD_SIZE)
    rapfi_abs = os.path.abspath(rapfi)
    if not os.path.exists(rapfi_abs):
        raise SystemExit(f"rapfi binary not found: {rapfi_abs}")
    if not os.path.exists(checkpoint):
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    device = pick_device(os.environ.get("GOMOKU_DEVICE"))
    print(f"[setup] device={device} BOARD_SIZE={n} HISTORY_PLY={HISTORY_PLY} "
          f"sims={sims} rapfi_timeout={timeout_ms}ms opening={opening_moves} "
          f"seed={seed} games={n_games}")
    model, _ = load_checkpoint(checkpoint, device=device)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, device)
    # White's picker — same construction the Rapfi eval uses (no FPU / tree-reuse /
    # proven-prop levers; greedy argmax over visit counts). Fresh tree per move.
    white_picker = mcts_picker(evaluator, n_simulations=sims, c_puct=1.5)

    engine = ExternalEnginePlayer(
        ExternalEngineConfig(
            cmd=rapfi_abs,
            timeout_ms=timeout_ms,
            label=f"rapfi{timeout_ms}",
            rule=0,  # freestyle
            board_size=n,
        )
    )

    # One RNG, threaded across openings + the white picker, exactly like
    # play_match_pickers threads a single rng. Seeded once from `seed`.
    rng = np.random.default_rng(seed)

    per_ply_rows: list[dict] = []
    # Per-game outcome from WHITE's perspective.
    results = {"win": 0, "loss": 0, "draw": 0}
    lost_game_lengths: list[int] = []  # total plies of games white LOST
    # Detector tallies over all white plies.
    fired = 0
    white_plies = 0
    t1_fired = t1_error = 0  # forced single four-block
    t2_fired = t2_error = 0  # open-three / open-four prevention
    # Endgame breakdown: of the last ENDGAME_WINDOW white plies in LOST games.
    ENDGAME_WINDOW = 6
    endgame_white_plies = 0
    endgame_double_four = 0     # detector None because len(opp_win) >= 2 (genuinely lost)
    endgame_unsavable_ot = 0    # open-three fork w/ no single defeating move (genuinely lost)
    endgame_stampable = 0       # detector fired -> white HAD a stampable defense
    endgame_stampable_played = 0
    endgame_stampable_missed = 0
    endgame_quiet = 0           # no four / no open-three threat at all (truly quiet)

    # Sanity-check accumulators (printed for the first `sanity_games`).
    sanity_notes: list[str] = []

    t0 = time.time()
    for g_idx in range(n_games):
        # Each game gets its own 4-stone random opening, drawn from the shared rng
        # the same way play_match_pickers draws them (via _random_opening_state).
        opening_state, opening_plies = _random_opening_state(rng, opening_moves)
        state = opening_state
        ply = state.move_count  # absolute ply index; seeds side-to-move accounting

        # After an even-ply opening, side-to-move (state.board[0]) is BLACK (side
        # 0) -> Rapfi (the attacker) moves first from the opening. White = side 1.
        # ply % 2 == 0 => black to move; ply % 2 == 1 => white to move.
        game_rows: list[dict] = []
        winner_side: int | None = None
        do_sanity = g_idx < sanity_games

        while True:
            side_to_move = ply % 2  # 0 = black (Rapfi), 1 = white (net)
            white_to_move = (side_to_move == 1)

            if white_to_move:
                white_plies += 1
                planes = state.to_planes()
                # --- detector + raw opponent-four count (for endgame breakdown) ---
                me = planes[0].astype(bool)            # white = defender
                opp = planes[HISTORY_PLY].astype(bool)  # black = attacker
                empty_plane = ~(me | opp)
                me_win = vcf._five_completions(me, empty_plane)   # white's own win cells
                opp_win = vcf._five_completions(opp, empty_plane)  # black four-completions
                cells = _conv_block_detect(planes, tier2=True)
                fired_here = cells is not None and len(cells) > 0
                cell_list = [int(c) for c in cells] if fired_here else []

                # When the detector DECLINES with no opponent four (opp_win==0) and
                # white has no own win, re-run the tier-2 open-four-threat scan
                # ourselves to tell "an open-three existed but NO single move stops
                # it (genuinely lost)" from "no threat at all (truly quiet)". This
                # mirrors the detector's own Rule-3 path; if `threats` is non-empty
                # but their defeating sets don't intersect, the position is an
                # unsavable open-three fork. Pure scan, no solver.
                ot_threat_unsavable = False
                ot_threat_exists = False
                if not fired_here and len(opp_win) == 0 and len(me_win) == 0:
                    empty_idx = vcf._empties_from_plane(empty_plane)
                    if len(empty_idx) > 0:
                        of_cands = vcf._candidate_cells_from_planes(
                            opp, me, empty_idx)
                        threats = vcf._open_four_threats(
                            opp.copy(), me.copy(), empty_plane,
                            [int(c) for c in of_cands])
                        if threats:
                            ot_threat_exists = True
                            defeating = [set([f, *comps]) for f, comps in threats]
                            ot_threat_unsavable = (
                                len(set.intersection(*defeating)) == 0)

                # Tier classification of the threat the detector saw:
                #   T1 (forced single four-block): exactly one opp_win cell.
                #   T2 (open-three prevention): no opp four, but detector fired.
                tier = None
                if fired_here:
                    tier = "T1" if len(opp_win) == 1 else "T2"

                # White's actual move.
                action = white_picker(state, rng)

                missed = fired_here and (int(action) not in set(cell_list))
                if fired_here:
                    fired += 1
                    if tier == "T1":
                        t1_fired += 1
                        if missed:
                            t1_error += 1
                    else:
                        t2_fired += 1
                        if missed:
                            t2_error += 1

                if do_sanity:
                    legal = set(int(a) for a in state.legal_actions())
                    legal_ok = int(action) in legal
                    if not legal_ok:
                        sanity_notes.append(
                            f"[g{g_idx}] WHITE ILLEGAL move {action} at ply {ply}")
                    if fired_here and tier == "T1":
                        # T1 stamp must equal the unique opp_win cell.
                        consistent = set(cell_list) == set(int(c) for c in opp_win)
                        sanity_notes.append(
                            f"[g{g_idx} ply{ply}] T1 fire: stamp={cell_list} "
                            f"opp_win={list(int(c) for c in opp_win)} "
                            f"white_played={int(action)} "
                            f"-> {'BLOCKED' if not missed else 'MISSED'} "
                            f"(stamp==opp_win: {consistent})")

                game_rows.append({
                    "game": g_idx,
                    "ply": int(ply),
                    "white_to_move": True,
                    "fired": bool(fired_here),
                    "tier": tier,
                    "n_opp_win": int(len(opp_win)),
                    "n_me_win": int(len(me_win)),
                    "stamp": cell_list,
                    "white_move": int(action),
                    "missed": bool(missed),
                    "ot_threat_exists": bool(ot_threat_exists),
                    "ot_threat_unsavable": bool(ot_threat_unsavable),
                })
            else:
                # Black (Rapfi) to move. Same picker path as play_match_pickers.
                action = engine(state, rng)
                if do_sanity:
                    legal = set(int(a) for a in state.legal_actions())
                    if int(action) not in legal:
                        sanity_notes.append(
                            f"[g{g_idx}] RAPFI(black) ILLEGAL move {action} at ply {ply}")

            side_just_moved = ply % 2
            state = state.apply(int(action))
            done, v = state.is_terminal()
            if done:
                if v == -1.0:
                    winner_side = side_just_moved
                break
            ply += 1

        # ----- game outcome (white = side 1) -----
        total_plies = ply + 1  # plies played this game incl the terminal move
        if winner_side is None:
            results["draw"] += 1
            outcome = "draw"
        elif winner_side == 1:
            results["win"] += 1
            outcome = "win"
        else:
            results["loss"] += 1
            outcome = "loss"
            lost_game_lengths.append(total_plies)

        # ----- endgame breakdown for LOST games: last ENDGAME_WINDOW white plies -----
        if outcome == "loss":
            white_rows = [r for r in game_rows if r["white_to_move"]]
            for r in white_rows[-ENDGAME_WINDOW:]:
                endgame_white_plies += 1
                if r["fired"]:
                    endgame_stampable += 1
                    if r["missed"]:
                        endgame_stampable_missed += 1
                    else:
                        endgame_stampable_played += 1
                else:
                    # Detector returned None. Classify why.
                    if r["n_me_win"] >= 1:
                        # White itself could win -> shouldn't be a loss-cause; quiet.
                        endgame_quiet += 1
                    elif r["n_opp_win"] >= 2:
                        endgame_double_four += 1          # genuinely lost (double four)
                    elif r.get("ot_threat_unsavable"):
                        # Open-three fork: a threat exists but no single move stops
                        # every open-four -> genuinely lost, nothing to stamp.
                        endgame_unsavable_ot += 1
                    else:
                        # No opp four, no unsavable open-three fork -> truly quiet:
                        # white was free to move but had no forced defense to make.
                        endgame_quiet += 1

        per_ply_rows.extend(game_rows)

        n_white_plies_game = sum(1 for r in game_rows if r["white_to_move"])
        n_fired_game = sum(1 for r in game_rows if r["fired"])
        print(f"  game {g_idx:2d}: {outcome:4s}  plies={total_plies:3d}  "
              f"white_plies={n_white_plies_game:2d}  fired={n_fired_game}")

    engine.close()
    dt = time.time() - t0

    # ----- write raw log -----
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        meta = {
            "_meta": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": os.path.abspath(checkpoint),
            "board_size": n,
            "sims": sims,
            "rapfi_timeout_ms": timeout_ms,
            "opening_moves": opening_moves,
            "seed": seed,
            "n_games": n_games,
            "net_color": "white",
            "rapfi_color": "black",
        }
        f.write(json.dumps(meta) + "\n")
        for r in per_ply_rows:
            f.write(json.dumps(r) + "\n")

    # ----- summary -----
    def pct(a: int, b: int) -> str:
        return f"{(a / b * 100):.1f}%" if b else "n/a"

    def stat_block(xs: list[int]) -> str:
        if not xs:
            return "none"
        return (f"mean={statistics.mean(xs):.1f} "
                f"median={statistics.median(xs):.0f} "
                f"min={min(xs)} max={max(xs)} n={len(xs)}")

    print("\n" + "=" * 72)
    print("WHITE FAILURE-MODE DIAGNOSTIC  (net=WHITE/defender vs Rapfi=BLACK/attacker)")
    print("=" * 72)
    print(f"checkpoint : {os.path.basename(os.path.dirname(checkpoint))}/.../model.pt  (wdl@0 control)")
    print(f"regime     : board={n} sims={sims} rapfi={timeout_ms}ms "
          f"opening={opening_moves}-stone seed={seed}  ({dt:.0f}s)")
    print("-" * 72)
    w, l, d = results["win"], results["loss"], results["draw"]
    print(f"WHITE W/L/D over {n_games}:  {w}W - {l}L - {d}D   "
          f"(win_rate {((w + 0.5 * d) / max(n_games,1)):.1%})")
    print(f"ply-length of WHITE's LOST games: {stat_block(lost_game_lengths)}")
    print("-" * 72)
    print(f"fire-rate  : {fired}/{white_plies} white plies = {pct(fired, white_plies)} "
          f"(a forced four-block or open-three existed)")
    print(f"error-rate : of FIRED plies, white's actual move NOT in the stamp set")
    print(f"   overall : {t1_error + t2_error}/{fired} = {pct(t1_error + t2_error, fired)}")
    print(f"   Tier-1  : {t1_error}/{t1_fired} = {pct(t1_error, t1_fired)}  "
          f"(single forced four-block missed)")
    print(f"   Tier-2  : {t2_error}/{t2_fired} = {pct(t2_error, t2_fired)}  "
          f"(open-three / open-four prevention missed)")
    print("-" * 72)
    print(f"ENDGAME (last {ENDGAME_WINDOW} white plies of each LOST game; "
          f"{endgame_white_plies} plies total):")
    genuinely_lost = endgame_double_four + endgame_unsavable_ot
    print(f"   GENUINELY LOST (detector None, no blockable defense): "
          f"{genuinely_lost}  ({pct(genuinely_lost, endgame_white_plies)})")
    print(f"     - double-four (len(opp_win)>=2): {endgame_double_four}")
    print(f"     - open-three fork, no single defeating move: {endgame_unsavable_ot}")
    print(f"   STAMPABLE defense existed (detector fired): {endgame_stampable}  "
          f"({pct(endgame_stampable, endgame_white_plies)})  "
          f"-> played {endgame_stampable_played}, MISSED {endgame_stampable_missed}")
    print(f"   truly quiet (no forced defense to make): {endgame_quiet}  "
          f"({pct(endgame_quiet, endgame_white_plies)})")
    print("-" * 72)
    # Verdict heuristic.
    overall_err = (t1_error + t2_error) / fired if fired else 0.0
    t1_err_rate = t1_error / t1_fired if t1_fired else 0.0
    long_losses = (statistics.mean(lost_game_lengths) >= 30) if lost_game_lengths else False
    # "Forced endgame" = the last plies of lost games are dominated by genuinely
    # -lost positions / correctly-played blocks, NOT by missed blockable threats.
    forced_endgame = (
        endgame_stampable_missed <= genuinely_lost
        if endgame_white_plies else False
    )
    # Tier-1 is the RIGOROUS "simple tactical error" signal — a forced single
    # four-block missed is an unambiguous blunder. Tier-2 (open-three) is a
    # HEURISTIC: a "miss" may be a legitimate alternative defense, not a blunder,
    # so we weight it lightly in the verdict.
    if t1_err_rate <= 0.05 and forced_endgame:
        verdict = (f"HYPOTHESIS SUPPORTED: white blocks forced fours competently "
                   f"(Tier-1 error {t1_err_rate:.0%}) and its losses end in "
                   f"genuinely-lost/forced positions, not blockable blunders -> the "
                   f"conv/tactical block-teacher has little to fix.")
    elif t1_err_rate >= 0.15:
        verdict = (f"HYPOTHESIS CONTRADICTED: white misses FORCED four-blocks "
                   f"(Tier-1 error {t1_err_rate:.0%}) -> real untapped tactical "
                   f"signal the block-teacher could fix.")
    else:
        verdict = (f"MIXED / leans SUPPORTED: Tier-1 error {t1_err_rate:.0%} "
                   f"(forced blocks mostly fine), Tier-2 error {pct(t2_error, t2_fired)} "
                   f"(heuristic, may be alternative defenses), long-losses="
                   f"{long_losses}, forced-endgame={forced_endgame}. See breakdown.")
    print("VERDICT:", verdict)
    print("=" * 72)
    print(f"raw per-ply log -> {os.path.abspath(out)}")

    if sanity_notes:
        print("\n--- SANITY CHECKS (first %d games) ---" % sanity_games)
        for s in sanity_notes[:40]:
            print(" ", s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        default="/Users/jason/.cache/huggingface/hub/models--jasonyandell--gomoku-9x9/"
                "snapshots/2b2b7c45de2d5f09bbd2136e8fa6ade9643a0717/model.pt",
        help="wdl@0 control net checkpoint (15x15).",
    )
    ap.add_argument(
        "--rapfi",
        default=os.path.expanduser("~/.cache/gomocup/bin/run-rapfi"),
        help="native Rapfi launcher.",
    )
    ap.add_argument("--n-games", type=int, default=30)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--timeout-ms", type=int, default=1000)
    ap.add_argument("--opening-moves", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="sweep_logs/diag_white_failuremode_wdl0.jsonl")
    ap.add_argument("--sanity-games", type=int, default=2,
                    help="Verbose harness sanity checks for the first N games.")
    args = ap.parse_args()
    run(
        checkpoint=args.checkpoint,
        rapfi=args.rapfi,
        n_games=args.n_games,
        sims=args.sims,
        timeout_ms=args.timeout_ms,
        opening_moves=args.opening_moves,
        seed=args.seed,
        out=args.out,
        sanity_games=args.sanity_games,
    )


if __name__ == "__main__":
    main()
