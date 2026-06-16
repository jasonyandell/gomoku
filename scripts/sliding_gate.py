"""Frozen-reference PROMOTION GATE for the sliding derby (issue #39, parent #38).

This is the single most load-bearing component of the sliding derby. A candidate
checkpoint plays the *frozen peak* head-to-head (anchor-free), and the gate
decides PROMOTE (snapshot the candidate as the new peak) or REVERT (keep the old
peak). The frozen peak slides forward only when a candidate STATISTICALLY beats
it — never on noise.

=====================================================================
CALIBRATION IMMUNITY (read this — it is why the gate exists)
=====================================================================
This gate NEVER reads calibrated absolute Elo. The panel calibration is broken
(issue #35: the affine anchor map is degenerate / non-monotone on our partial
cross-tables, so any "absolute Gomocup Elo" is untrustworthy). The gate's verdict
depends ONLY on:

  * the ANCHOR-FREE head-to-head win-rate of candidate vs the frozen peak
    (two similar models meet near 50% — the maximally sensitive region — so the
    relative signal has a tight CI with no anchor ceiling), plus
  * the Wilson CI on that win-rate (``binomial_ci``), and optionally the
    relative Δelo / its CI from ``head_to_head_eval`` (logged, illustrative).

``white_loss_rate`` (#18 "never lose as white") is logged as a SECONDARY signal
only — it NEVER gates. Because the H2H executor returns only aggregate W/D/L, the
per-color white-loss tally is computed (when ``--white-loss`` is passed) from the
SAME paired games via the harness's per-game primitives; otherwise it is None.

=====================================================================
THE VERDICT RULE (explicit and conservative)
=====================================================================
PROMOTE  iff  win_rate > 0.5  AND  the win-rate CI is CLEAR of 0.5
              (i.e. ci_lo > 0.5 + MARGIN: the candidate statistically beats the
              peak). A tie, a CI that straddles 0.5, or any loss => REVERT.
REVERT   otherwise. We never promote on a noisy tie: when in doubt the frozen
         peak stays put.

The peak only ever moves UP the H2H ladder, so the reference monotonically
strengthens — the sliding-window analogue of a frozen-reference ratchet.

=====================================================================
SAFETY
=====================================================================
``--dry-run`` computes and LOGS the verdict WITHOUT mutating the peak or the
board, so the FIRST verdict can be human-reviewed before the gate is allowed to
slide the reference (the derby's "first verdict needs review" rule). Promotion is
an atomic copy (tmp + os.replace, mirroring delo_derby.snapshot_peak). Every
decision is appended to a JSONL verdict log; the board state is written
atomically (tmp + rename).

Torch is imported LAZILY (only when the gate actually runs games); the verdict
logic, board I/O, and CLI are import-light so the unit tests stub the eval and
run torch-free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# Import the harness's PURE math + the derby's atomic I/O. These imports are
# torch-free: head_to_head_eval is only *called* at game time, and importing the
# module does not load torch (its heavy imports are lazy / inside functions).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.delta_e_harness import (  # noqa: E402  (after sys.path shim)
    binomial_ci,
)
from scripts.delo_derby import (  # noqa: E402
    atomic_write_json,
    now_iso,
)

# Conservative margin above 0.5 the CI lower bound must clear before we will call
# a result "statistically beats the peak". 0.0 means "CI strictly above 0.5";
# a small positive value is extra insurance against borderline noise.
DEFAULT_CI_MARGIN = 0.0

PROMOTE = "PROMOTE"
REVERT = "REVERT"


# ---------------------------------------------------------------------------
# Verdict (pure decision logic — torch-free, the unit-tested core)
# ---------------------------------------------------------------------------
@dataclass
class GateVerdict:
    """The decision for one candidate-vs-peak match. ``verdict`` is PROMOTE or
    REVERT. All fields are anchor-free / calibration-immune (see module docstring).
    """

    verdict: str
    candidate: str
    peak: Optional[str]
    n_games: int
    win_rate: float
    ci_lo: float
    ci_hi: float
    ci_margin: float
    delta_elo: float
    delta_ci_half: float
    white_loss_rate: Optional[float]
    seed: int
    reason: str

    @property
    def promote(self) -> bool:
        return self.verdict == PROMOTE


def decide_verdict(
    *,
    candidate: str,
    peak: Optional[str],
    win_rate: float,
    score: float,
    n_games: int,
    delta_elo: float = 0.0,
    delta_ci_half: float = 0.0,
    white_loss_rate: Optional[float] = None,
    seed: int = 0,
    ci_margin: float = DEFAULT_CI_MARGIN,
    confidence: float = 0.95,
) -> GateVerdict:
    """Pure verdict: PROMOTE iff the candidate STATISTICALLY beats the peak.

    Rule (conservative): win_rate > 0.5 AND the Wilson CI lower bound on the
    win-rate is clear of 0.5 by ``ci_margin``. Ties / CI straddling 0.5 / losses
    => REVERT. ``score`` is the chess-scored sum (wins + 0.5*draws); the CI is the
    Wilson interval on score/n_games (reusing the harness's ``binomial_ci``).

    If there is no peer yet (peak is None — a fresh ladder), there is nothing to
    beat: the first candidate becomes the peak by definition. The caller still
    gates the very first promotion behind --dry-run for human review.
    """
    if peak is None:
        return GateVerdict(
            verdict=PROMOTE, candidate=candidate, peak=None, n_games=n_games,
            win_rate=win_rate, ci_lo=win_rate, ci_hi=win_rate, ci_margin=ci_margin,
            delta_elo=delta_elo, delta_ci_half=delta_ci_half,
            white_loss_rate=white_loss_rate, seed=seed,
            reason="no incumbent peak — first candidate seeds the ladder",
        )

    ci_lo, ci_hi = binomial_ci(score, n_games, confidence=confidence)
    threshold = 0.5 + ci_margin
    beats = (win_rate > 0.5) and (ci_lo > threshold)
    if beats:
        verdict = PROMOTE
        reason = (f"win_rate {win_rate:.3f} > 0.5 and CI lower bound "
                  f"{ci_lo:.3f} > {threshold:.3f} — statistically beats peak")
    else:
        verdict = REVERT
        if win_rate <= 0.5:
            reason = f"win_rate {win_rate:.3f} <= 0.5 — does not beat peak"
        else:
            reason = (f"win_rate {win_rate:.3f} > 0.5 but CI lower bound "
                      f"{ci_lo:.3f} <= {threshold:.3f} — straddles 0.5 (noisy tie)")
    return GateVerdict(
        verdict=verdict, candidate=candidate, peak=peak, n_games=n_games,
        win_rate=win_rate, ci_lo=ci_lo, ci_hi=ci_hi, ci_margin=ci_margin,
        delta_elo=delta_elo, delta_ci_half=delta_ci_half,
        white_loss_rate=white_loss_rate, seed=seed, reason=reason,
    )


# ---------------------------------------------------------------------------
# Board state — atomic round-trip (mirrors delo_derby.write_research_board's
# atomic-write discipline; reuses atomic_write_json directly).
# ---------------------------------------------------------------------------
def default_board() -> dict:
    """A fresh sliding-gate board: no peak yet, lap 0, empty verdict history."""
    return {
        "schema": "sliding_gate.v1",
        "peak_path": None,
        "lap": 0,
        "promotions": 0,
        "reverts": 0,
        "last_verdict": None,
        "history": [],
    }


def load_board(board_path: Path) -> dict:
    """Read the board json; return a fresh board if the file does not exist."""
    if not board_path.exists():
        return default_board()
    with open(board_path) as f:
        data = json.load(f)
    # Be tolerant of an older/partial board: fill any missing keys.
    base = default_board()
    base.update({k: v for k, v in data.items() if k in base})
    return base


def save_board(board: dict, board_path: Path) -> None:
    """Atomically persist the board (tmp + rename via the derby's helper)."""
    atomic_write_json(board_path, board)


def record_verdict_on_board(board: dict, v: GateVerdict, *, promoted: bool) -> dict:
    """Advance the board by one lap and append the verdict to its history.

    ``promoted`` reflects whether the peak was ACTUALLY moved (False in --dry-run
    even when the verdict is PROMOTE). The history entry records both the verdict
    and whether it was applied, so a dry-run leaves an auditable trail.
    """
    board["lap"] = int(board.get("lap", 0)) + 1
    if promoted:
        # The new frozen peak is the SNAPSHOT (v.peak was set to the snapshot path
        # by run_gate after the atomic copy), not the candidate's transient path —
        # the candidate checkpoint may be pruned, the snapshot is derby-owned.
        board["peak_path"] = v.peak
        board["promotions"] = int(board.get("promotions", 0)) + 1
    elif v.verdict == REVERT:
        board["reverts"] = int(board.get("reverts", 0)) + 1
    entry = asdict(v)
    entry["lap"] = board["lap"]
    entry["applied"] = promoted
    entry["ts"] = now_iso()
    board["last_verdict"] = entry
    board.setdefault("history", []).append(entry)
    return board


# ---------------------------------------------------------------------------
# Verdict log — append-only JSONL.
# ---------------------------------------------------------------------------
def append_verdict_log(log_path: Path, v: GateVerdict, *, applied: bool,
                       ts: Optional[str] = None) -> None:
    """Append one verdict record to the JSONL log (append-only).

    ``ts`` defaults to now_iso(); pass it in for deterministic tests. ``applied``
    is whether the peak was actually moved (distinguishes a dry-run PROMOTE that
    was logged but not applied from a real promotion).
    """
    rec = asdict(v)
    rec["applied"] = applied
    rec["ts"] = ts if ts is not None else now_iso()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Atomic promotion — copy the candidate to the peak path (mirrors
# delo_derby.snapshot_peak's tmp + os.replace; never partial).
# ---------------------------------------------------------------------------
def promote_candidate(candidate: str, peak_path: Path) -> str:
    """Atomically copy the candidate checkpoint to ``peak_path`` (the new frozen
    peak). tmp + os.replace, exactly like delo_derby.snapshot_peak, so a crash
    never leaves a half-written peak. Returns the peak path as a string.
    """
    import shutil

    src = Path(candidate)
    if not src.exists():
        raise FileNotFoundError(f"candidate checkpoint not found: {candidate}")
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = peak_path.with_suffix(peak_path.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, peak_path)
    return str(peak_path)


# ---------------------------------------------------------------------------
# Secondary signal — candidate white-side (defense) loss-rate from the SAME
# paired games. Reuses the harness per-game primitives; logged, NEVER gates.
# ---------------------------------------------------------------------------
def candidate_white_loss_rate(
    candidate: str,
    peak: str,
    *,
    n_games: int,
    sims: int,
    c_puct: float,
    seed: int,
    device: str,
    opening_plies: int,
) -> Optional[float]:
    """White-side loss-rate of the CANDIDATE (defending / second player) over the
    paired H2H games — the #18 "never lose as white" signal. Computed from the
    harness's own per-game primitives so it tallies the same games the verdict
    saw. SECONDARY: logged, never part of the verdict. Returns None if no
    white-side (candidate-defends) games were played.

    Imports torch indirectly (it builds pickers), so it runs only under
    --white-loss. The candidate plays white exactly when it is NOT black in a
    pair; ``_play_one_h2h_game`` returns the result from the candidate's side.
    """
    from scripts.delta_e_harness import (
        _build_ckpt_picker,
        _h2h_tasks,
        _play_one_h2h_game,
    )

    cand_picker = _build_ckpt_picker(candidate, sims=sims, c_puct=c_puct, device=device)
    peak_picker = _build_ckpt_picker(peak, sims=sims, c_puct=c_puct, device=device)
    white_games = 0
    white_losses = 0
    for pair_idx, fork_is_black in _h2h_tasks(n_games, opening_plies, seed):
        if fork_is_black:
            continue  # candidate is BLACK here — not a white (defense) game
        outcome = _play_one_h2h_game(
            cand_picker, peak_picker, pair_idx=pair_idx, fork_is_black=False,
            opening_plies=opening_plies, seed=seed)
        white_games += 1
        if outcome == "loss":
            white_losses += 1
    if white_games == 0:
        return None
    return white_losses / white_games


# ---------------------------------------------------------------------------
# The gate — run the match, decide, log, and (unless dry-run) slide the peak.
# ---------------------------------------------------------------------------
def run_gate(
    *,
    candidate: str,
    peak: Optional[str],
    n_games: int,
    seed: int,
    board_path: Path,
    verdict_log: Path,
    peak_out: Path,
    sims: int,
    c_puct: float,
    opening_plies: int,
    device: str,
    n_workers: int,
    ci_margin: float,
    dry_run: bool,
    want_white_loss: bool,
    eval_fn: Optional[Callable[..., Any]] = None,
) -> GateVerdict:
    """Execute one gate lap: candidate plays the frozen peak head-to-head, decide
    PROMOTE/REVERT, log the verdict, and (unless dry_run) atomically slide the
    peak + persist the board.

    ``eval_fn`` defaults to delta_e_harness.head_to_head_eval (the real executor);
    tests inject a stub that returns a HeadToHeadResult-shaped object with
    ``.win_rate``, ``.score``, ``.n_games``, ``.delta_elo``, ``.delta_ci_half``.
    Torch is only imported when ``eval_fn`` is the default and we actually run.
    """
    if eval_fn is None:
        from scripts.delta_e_harness import head_to_head_eval as eval_fn  # lazy torch

    board = load_board(board_path)
    # The frozen peak comes from the board if not given explicitly; an explicit
    # --peak overrides (e.g. the human seeding the first peak by hand).
    if peak is None:
        peak = board.get("peak_path")

    if peak is None:
        # No incumbent — the candidate seeds the ladder. We DON'T run games (there
        # is nothing to play against); the verdict is a definitional PROMOTE,
        # still gated behind a human-reviewed --dry-run for the very first lap.
        v = decide_verdict(
            candidate=candidate, peak=None, win_rate=1.0, score=0.0,
            n_games=0, seed=seed, ci_margin=ci_margin,
        )
    else:
        result = eval_fn(
            candidate, peak,
            recipe_label="sliding_gate",
            window_epochs=0,
            wall_secs=None,
            n_games=n_games,
            sims=sims,
            c_puct=c_puct,
            seed=seed,
            device=device,
            n_workers=n_workers,
            opening_plies=opening_plies,
        )
        white_loss = None
        if want_white_loss:
            white_loss = candidate_white_loss_rate(
                candidate, peak, n_games=n_games, sims=sims, c_puct=c_puct,
                seed=seed, device=device, opening_plies=opening_plies)
        v = decide_verdict(
            candidate=candidate, peak=peak,
            win_rate=float(result.win_rate),
            score=float(result.score),
            n_games=int(result.n_games),
            delta_elo=float(getattr(result, "delta_elo", 0.0)),
            delta_ci_half=float(getattr(result, "delta_ci_half", 0.0)),
            white_loss_rate=white_loss,
            seed=seed, ci_margin=ci_margin,
        )

    # APPLY only when the verdict is PROMOTE and we are NOT in dry-run.
    promoted = False
    if v.promote and not dry_run:
        new_peak = promote_candidate(candidate, peak_out)
        v.peak = new_peak  # the board's peak is now the snapshot path
        promoted = True

    append_verdict_log(verdict_log, v, applied=promoted)
    if not dry_run:
        record_verdict_on_board(board, v, promoted=promoted)
        save_board(board, board_path)

    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frozen-reference PROMOTION GATE for the sliding derby "
                    "(#39/#38). Calibration-immune: anchor-free H2H win-rate vs "
                    "the frozen peak ONLY (never reads absolute Elo; #35).",
    )
    p.add_argument("--candidate", required=True,
                   help="Candidate checkpoint to test against the frozen peak.")
    p.add_argument("--peak", default=None,
                   help="Current frozen-peak checkpoint. If omitted, taken from "
                        "the board's peak_path (None on a fresh ladder).")
    p.add_argument("--n-games", type=int, default=40,
                   help="H2H games to play (paired openings; default 40).")
    p.add_argument("--seed", type=int, default=0, help="Match seed.")
    p.add_argument("--board", default="sweep_runs/sliding_gate_board.json",
                   help="Board json (peak path, lap counter, verdict history).")
    p.add_argument("--verdict-log", default="sweep_runs/sliding_gate_verdicts.jsonl",
                   help="Append-only JSONL verdict log.")
    p.add_argument("--peak-out", default="sweep_runs/sliding_gate_peak.pt",
                   help="Where a PROMOTED candidate is snapshotted as the new peak.")
    p.add_argument("--sims", type=int, default=200, help="MCTS sims per move.")
    p.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration.")
    p.add_argument("--opening-plies", type=int, default=4,
                   help="Random opening plies per paired game (color cancels).")
    p.add_argument("--device", default="cpu", help="torch device (cpu/mps).")
    p.add_argument("--n-workers", type=int, default=6, help="Parallel game workers.")
    p.add_argument("--ci-margin", type=float, default=DEFAULT_CI_MARGIN,
                   help="Extra margin above 0.5 the CI lower bound must clear "
                        "(conservative insurance against noise; default 0.0).")
    p.add_argument("--white-loss", action="store_true",
                   help="Also compute the candidate's white-side loss-rate "
                        "(secondary #18 signal; logged, NEVER gates; slower).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute + LOG the verdict but DO NOT move the peak or "
                        "mutate the board — for human review of the first verdict.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    t0 = time.perf_counter()
    v = run_gate(
        candidate=args.candidate,
        peak=args.peak,
        n_games=args.n_games,
        seed=args.seed,
        board_path=Path(args.board),
        verdict_log=Path(args.verdict_log),
        peak_out=Path(args.peak_out),
        sims=args.sims,
        c_puct=args.c_puct,
        opening_plies=args.opening_plies,
        device=args.device,
        n_workers=args.n_workers,
        ci_margin=args.ci_margin,
        dry_run=args.dry_run,
        want_white_loss=args.white_loss,
    )
    tag = "DRY-RUN " if args.dry_run else ""
    print(f"[sliding-gate] {tag}{v.verdict}: candidate={v.candidate}")
    print(f"[sliding-gate]   peak={v.peak}")
    print(f"[sliding-gate]   win_rate={v.win_rate:.3f} "
          f"CI=[{v.ci_lo:.3f}, {v.ci_hi:.3f}] (margin {v.ci_margin:.3f}) "
          f"Δelo={v.delta_elo:+.1f} (±{v.delta_ci_half:.1f})")
    if v.white_loss_rate is not None:
        print(f"[sliding-gate]   white_loss_rate={v.white_loss_rate:.3f} "
              "(secondary #18 signal — logged, NOT gating)")
    print(f"[sliding-gate]   reason: {v.reason}")
    print(f"[sliding-gate]   ({time.perf_counter() - t0:.0f}s; "
          f"{'NOT applied (dry-run)' if args.dry_run else 'applied'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
