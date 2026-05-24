"""delta-e harness — score training recipes by Δelo (and Δelo/Δt) off a common parent.

THE METRIC (project north-star). Δelo/Δt is "delta-v for training": the elo
*increment* a recipe buys you per unit time, measured from your CURRENT state
(a common parent checkpoint C), not from some imagined destination. The harness
is the scoring engine for the curated-buffer flywheel
(`wiki/topics/curated-buffer-and-curriculum-design.md`, "Sequencing" + "Metric"):

    1. FORK N training recipes off a COMMON parent checkpoint C.
    2. Run each a FIXED window T (epochs).
    3. ANCHORED-EVAL each fork's final checkpoint + C against stable baselines
       with ENOUGH games to tighten the elo signal.
    4. Δelo = elo(fork_final) - elo(C). Rank by Δelo and by Δelo/T.

THE NOISE CONSTRAINT (why anchored-eval, not the trainer's 20-game eval). Our
in-trainer elo signal (20 games vs heuristic/lookahead) has ~±100-elo
cycle-to-cycle noise. A short-horizon Δelo *slope* is meaningless unless the
endpoint elo is tightened. So this harness re-evals C and each fork with
~100+ games per baseline, computes a binomial CI on each win-rate, propagates
that into an elo CI, and makes the "is this Δelo inside the noise?" judgment
EXPLICIT in the ranked table (Δelo vs its CI half-width).

SCOPE. This harness only ORCHESTRATES forks (by shelling out to the
replay-mode trainer CLI — built in parallel in worktree feat/perf-curated-buffer,
contract below), runs anchored eval (reusing gomoku.eval), and ranks. It does
NOT edit gomoku/train.py or scripts/run_sweep.py. External-anchor (e.g. Rapfi)
integration is SCAFFOLDED ONLY (see ExternalAnchor).

REPLAY-TRAINER CLI CONTRACT (assumed; we shell out, do not import internals):

    python -m gomoku.train_replay \\
        --resume <C> \\
        --archive-path <A> \\
        --buffer-recipe <R> \\
        --sgd-steps-per-epoch <N> \\
        --epochs <T_epochs> \\
        --size small \\
        --validation-archive-path archives/wl5_validation_v1.pt

Plus harness-supplied plumbing flags (--checkpoint-dir, --run-name, --no-wandb,
--seed) so each fork writes its final checkpoint to a known per-recipe dir.

USAGE

  # Dry-run the fork orchestration: print the exact per-recipe train commands.
  python scripts/delta_e_harness.py \\
      --parent checkpoints/C.pt --archive-path archives/big.pt \\
      --window-epochs 40 \\
      --recipe 'recency_weighted:sgd_steps_per_epoch=100' \\
      --recipe 'diversity:sgd_steps_per_epoch=150' \\
      --dry-run

  # Self-test the elo + CI + ranking math on synthetic match data (no GPU, no I/O).
  python scripts/delta_e_harness.py --self-test

  # Full run (forks sequentially on GPU, then anchored eval, then rank).
  python scripts/delta_e_harness.py \\
      --parent checkpoints/C.pt --archive-path archives/big.pt \\
      --window-epochs 40 \\
      --recipe 'recency_weighted:sgd_steps_per_epoch=100' \\
      --recipe 'diversity:sgd_steps_per_epoch=150' \\
      --eval-baselines heuristic,lookahead:depth=2,lookahead:depth=4 \\
      --eval-games 120 --eval-workers 6 --out-dir sweep_logs/delta-e-001

See `wiki/topics/curated-buffer-and-curriculum-design.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Elo + binomial-CI math (pure; the part the --self-test exercises).
#
# We reuse gomoku.rating for the maximum-likelihood implied-elo solver and the
# anchor table; everything CI-related lives here so the math is self-contained
# and testable without torch.
# ---------------------------------------------------------------------------

# Lazy import guard: gomoku.rating is torch-free, but keep imports local to the
# functions that need them so `--self-test` and `--dry-run` never touch torch.

LOGISTIC_SLOPE = 400.0  # standard elo scale


@dataclass
class BaselineMatch:
    """One model-vs-baseline match outcome (model's perspective)."""

    baseline: str          # spec label, e.g. "lookahead:depth=2"
    anchor_elo: float      # fixed reference elo for this baseline
    n_games: int
    wins: int
    draws: int

    @property
    def losses(self) -> int:
        return self.n_games - self.wins - self.draws

    @property
    def score(self) -> float:
        """Score sum, draws=0.5 (chess scoring)."""
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        return self.score / max(self.n_games, 1)


def binomial_ci(score: float, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """95% CI on a win-RATE from a binomial sample (draws counted as 0.5).

    Uses Wilson score interval — well-behaved near 0/1 and for small n, far
    better than the normal approximation the trainer's eyeballed winrates imply.
    `score` is the sum (wins + 0.5*draws); n is total games. Returns (lo, hi)
    clamped to [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    p = max(0.0, min(1.0, score / n))
    # z for two-sided `confidence`.
    try:
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + confidence / 2.0))
    except Exception:
        z = 1.959963984540054 if abs(confidence - 0.95) < 1e-9 else _z_from_conf(confidence)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _z_from_conf(confidence: float) -> float:
    """Inverse-normal CDF fallback (Acklam approx) if scipy is unavailable."""
    # Only used for non-0.95 confidences without scipy; rare path.
    p = 0.5 + confidence / 2.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


@dataclass
class EloEstimate:
    """A point elo plus a CI derived by re-solving implied_elo at the per-baseline
    win-rate CI endpoints."""

    elo: float
    lo: float
    hi: float
    matches: list[BaselineMatch] = field(default_factory=list)

    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)


def estimate_elo(matches: list[BaselineMatch], *, confidence: float = 0.95) -> EloEstimate:
    """Maximum-likelihood implied elo for a checkpoint, with a CI.

    Point estimate: solve `implied_elo` at the observed scores (reuses the
    repo's ML solver against fixed anchor elos).

    CI: the dominant uncertainty is sampling noise in each baseline win-rate.
    We map each baseline's Wilson CI on its win-rate back to a score-sum
    interval, then re-solve implied_elo at the JOINT-low endpoint (all baselines
    at their low win-rate) and JOINT-high endpoint. That is a conservative
    (slightly wide) bracket on the implied elo — exactly the "is the slope
    inside the noise?" guard we want: if it's wide enough to swallow Δelo, we
    say so.
    """
    from gomoku.rating import implied_elo

    if not matches:
        raise ValueError("estimate_elo needs at least one match")

    point_evidence = [(m.anchor_elo, m.score, m.n_games) for m in matches]
    point = implied_elo(point_evidence)

    lo_evidence = []
    hi_evidence = []
    for m in matches:
        wr_lo, wr_hi = binomial_ci(m.score, m.n_games, confidence=confidence)
        lo_evidence.append((m.anchor_elo, wr_lo * m.n_games, m.n_games))
        hi_evidence.append((m.anchor_elo, wr_hi * m.n_games, m.n_games))
    elo_lo = implied_elo(lo_evidence)
    elo_hi = implied_elo(hi_evidence)
    # Guard ordering (boundary clamps in implied_elo could in principle invert).
    lo, hi = sorted((elo_lo, elo_hi))
    return EloEstimate(elo=point, lo=lo, hi=hi, matches=list(matches))


@dataclass
class DeltaEResult:
    recipe_label: str
    window_epochs: int
    wall_secs: float | None
    parent: EloEstimate
    fork: EloEstimate

    @property
    def delta_elo(self) -> float:
        return self.fork.elo - self.parent.elo

    @property
    def delta_ci_half(self) -> float:
        """Half-width of the CI on the DIFFERENCE. The parent elo is shared
        across all recipes (single common C), but its estimate still carries
        sampling noise; we add the two half-widths in quadrature for an
        independent-samples bound on Δelo's uncertainty."""
        return math.hypot(self.fork.half_width, self.parent.half_width)

    @property
    def delta_elo_per_hr(self) -> float | None:
        if not self.wall_secs or self.wall_secs <= 0:
            return None
        return self.delta_elo / (self.wall_secs / 3600.0)

    @property
    def inside_noise(self) -> bool:
        """True if |Δelo| does not exceed its own CI half-width — i.e. the slope
        is not distinguishable from zero given our sampling noise. This is the
        explicit guard the metric section demands."""
        return abs(self.delta_elo) <= self.delta_ci_half


# ---------------------------------------------------------------------------
# Head-to-head Δelo (the anchor-ceiling fix).
#
# When every checkpoint beats the top anchor, the anchored ladder is saturated:
# fork and C both pin near the elo ceiling and differencing two near-ceiling
# implied-elos buries the real gap under sampling noise. Fix: play fork DIRECTLY
# vs the common parent C (model-vs-model, alternating colors). Two similar
# models score near 50% — the maximally sensitive region of the logistic — so
# the RELATIVE Δelo has a tight CI with no anchor ceiling.
# ---------------------------------------------------------------------------

H2H_EPS = 1e-6  # clamp on p before log10 so a 100%/0% sweep stays finite.


def elo_from_score_rate(p: float) -> float:
    """Relative Δelo implied by an expected score `p` (fork's perspective) in a
    head-to-head match: 400*log10(p/(1-p)). p=0.5 -> 0; p=0.6 -> +70.4;
    p=0.4 -> -70.4. p is clamped to [eps, 1-eps] so a clean sweep stays finite."""
    pc = max(H2H_EPS, min(1.0 - H2H_EPS, p))
    return LOGISTIC_SLOPE * math.log10(pc / (1.0 - pc))


@dataclass
class HeadToHeadResult:
    """One fork-vs-C direct match. `wins/draws/losses` are from the FORK's
    perspective (fork is picker_a). Relative Δelo is mapped from the expected
    score via the elo logistic; the CI comes from the Wilson interval on the
    score-rate mapped through the same logistic — no anchor ceiling."""

    recipe_label: str
    window_epochs: int
    wall_secs: float | None
    n_games: int
    wins: int
    draws: int
    losses: int

    @property
    def score(self) -> float:
        """Score sum, draws=0.5 (fork's perspective)."""
        return self.wins + 0.5 * self.draws

    @property
    def win_rate(self) -> float:
        return self.score / max(self.n_games, 1)

    @property
    def delta_elo(self) -> float:
        return elo_from_score_rate(self.win_rate)

    @property
    def delta_ci_half(self) -> float:
        """Half-width of the Δelo CI: take the Wilson CI on the score-rate
        (reusing binomial_ci) and map each endpoint through the elo logistic."""
        lo, hi = binomial_ci(self.score, self.n_games)
        d_lo, d_hi = elo_from_score_rate(lo), elo_from_score_rate(hi)
        return 0.5 * (d_hi - d_lo)

    @property
    def delta_elo_per_hr(self) -> float | None:
        if not self.wall_secs or self.wall_secs <= 0:
            return None
        return self.delta_elo / (self.wall_secs / 3600.0)

    @property
    def inside_noise(self) -> bool:
        """True if |Δelo| <= its CI half-width — same rule as the anchored path."""
        return abs(self.delta_elo) <= self.delta_ci_half


def rank_results(results: list[DeltaEResult]) -> list[DeltaEResult]:
    """Rank by Δelo descending (the achievable increment). Caller can re-sort by
    Δelo/hr for the rate view; we expose both columns in the table."""
    return sorted(results, key=lambda r: r.delta_elo, reverse=True)


def rank_h2h(results: list[HeadToHeadResult]) -> list[HeadToHeadResult]:
    """Rank head-to-head results by Δelo descending (mirrors rank_results)."""
    return sorted(results, key=lambda r: r.delta_elo, reverse=True)


def format_h2h_table(results: list[HeadToHeadResult]) -> str:
    ranked = rank_h2h(results)
    lines: list[str] = []
    hdr = (f"{'rank':>4}  {'recipe':<34}  {'Δelo(vsC)':>9}  {'±CI':>6}  "
           f"{'T(ep)':>6}  {'Δelo/hr':>9}  {'verdict':<13}  head-to-head")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, r in enumerate(ranked, 1):
        rate = f"{r.delta_elo_per_hr:+.1f}" if r.delta_elo_per_hr is not None else "  n/a"
        verdict = "INSIDE-NOISE" if r.inside_noise else "signal"
        h2h = f"fork {r.score:g}/{r.n_games} vs C = {r.win_rate:.0%}"
        lines.append(
            f"{i:>4}  {r.recipe_label:<34}  {r.delta_elo:>+9.1f}  "
            f"{r.delta_ci_half:>6.1f}  {r.window_epochs:>6}  {rate:>9}  "
            f"{verdict:<13}  {h2h}"
        )
    lines.append("")
    lines.append("Δelo is RELATIVE to common parent C (model-vs-model, no anchor "
                 "ceiling); p=0.5 -> 0 elo.")
    lines.append("verdict=INSIDE-NOISE means |Δelo| <= its CI half-width: the "
                 "recipe's gain over C is NOT distinguishable from sampling noise.")
    return "\n".join(lines)


def format_table(results: list[DeltaEResult]) -> str:
    ranked = rank_results(results)
    lines: list[str] = []
    hdr = (f"{'rank':>4}  {'recipe':<34}  {'Δelo':>8}  {'±CI':>6}  "
           f"{'T(ep)':>6}  {'Δelo/hr':>9}  {'verdict':<13}  win-rates")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for i, r in enumerate(ranked, 1):
        wr = " ".join(f"{m.baseline}={m.win_rate:.0%}" for m in r.fork.matches)
        rate = f"{r.delta_elo_per_hr:+.1f}" if r.delta_elo_per_hr is not None else "  n/a"
        verdict = "INSIDE-NOISE" if r.inside_noise else "signal"
        lines.append(
            f"{i:>4}  {r.recipe_label:<34}  {r.delta_elo:>+8.1f}  "
            f"{r.delta_ci_half:>6.1f}  {r.window_epochs:>6}  {rate:>9}  "
            f"{verdict:<13}  {wr}"
        )
    lines.append("")
    lines.append(f"parent C elo: {ranked[0].parent.elo:+.1f} "
                 f"[{ranked[0].parent.lo:+.1f}, {ranked[0].parent.hi:+.1f}]  "
                 f"(±{ranked[0].parent.half_width:.1f}, "
                 f"{ranked[0].parent.matches[0].n_games} games/baseline)")
    lines.append("verdict=INSIDE-NOISE means |Δelo| <= its CI half-width: the "
                 "recipe's gain is NOT distinguishable from sampling noise.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recipe parsing + fork orchestration (shell out to the replay-trainer CLI).
# ---------------------------------------------------------------------------


@dataclass
class Recipe:
    """A training-knob override set. `buffer_recipe` is required (it names the
    curator); other knobs (e.g. sgd_steps_per_epoch) are overrides on top of
    harness defaults."""

    buffer_recipe: str
    sgd_steps_per_epoch: int
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        bits = [self.buffer_recipe, f"sgd={self.sgd_steps_per_epoch}"]
        bits += [f"{k}={v}" for k, v in sorted(self.extra.items())]
        return ",".join(bits)

    @property
    def slug(self) -> str:
        return self.label.replace(",", "_").replace("=", "").replace(":", "")


def parse_recipe(s: str, *, default_sgd: int) -> Recipe:
    """Parse a recipe spec: `buffer_recipe[:k=v,k=v,...]`.

    Examples:
        'recency_weighted'
        'recency_weighted:sgd_steps_per_epoch=100'
        'diversity:sgd_steps_per_epoch=150,lr=5e-4'
    """
    if ":" in s:
        buf, rest = s.split(":", 1)
    else:
        buf, rest = s, ""
    buf = buf.strip()
    if not buf:
        raise SystemExit(f"bad recipe {s!r}: empty buffer_recipe")
    sgd = default_sgd
    extra: dict[str, str] = {}
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"bad recipe {s!r}: expected k=v, got {part!r}")
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "sgd_steps_per_epoch":
            sgd = int(v)
        else:
            extra[k] = v
    return Recipe(buffer_recipe=buf, sgd_steps_per_epoch=sgd, extra=extra)


def build_fork_command(
    recipe: Recipe,
    *,
    parent: str,
    archive_path: str,
    window_epochs: int,
    size: str,
    validation_archive_path: str,
    checkpoint_dir: str,
    seed: int,
    wandb: bool,
) -> list[str]:
    """Construct the EXACT `python -m gomoku.train_replay ...` argv for one fork.

    Matches the replay-trainer CLI contract documented in the module docstring.
    Returned as an argv list (not a string) so the dry-run print and the real
    subprocess use the identical command.
    """
    cmd = [
        sys.executable, "-m", "gomoku.train_replay",
        "--resume", parent,
        "--archive-path", archive_path,
        "--buffer-recipe", recipe.buffer_recipe,
        "--sgd-steps-per-epoch", str(recipe.sgd_steps_per_epoch),
        "--epochs", str(window_epochs),
        "--size", size,
        "--validation-archive-path", validation_archive_path,
        # Harness plumbing: where the fork writes its final checkpoint, plus
        # reproducibility. The trainer owns these flags already (current
        # train.py exposes --checkpoint-dir/--run-name/--seed/--no-wandb).
        "--checkpoint-dir", checkpoint_dir,
        "--run-name", f"delta-e-{recipe.slug}",
        "--seed", str(seed),
    ]
    cmd += ["--wandb"] if wandb else ["--no-wandb"]
    # Pass through any extra knob overrides as --kebab-case flags.
    for k, v in sorted(recipe.extra.items()):
        cmd += [f"--{k.replace('_', '-')}", v]
    return cmd


def final_checkpoint_for(checkpoint_dir: str) -> Path:
    """The fork's final checkpoint path. The trainer publishes the rolling
    `latest.pt` in --checkpoint-dir; that's the post-window checkpoint we eval.
    Kept as one function so the contract is in one place."""
    return Path(checkpoint_dir) / "latest.pt"


def run_fork(cmd: list[str], *, log_path: Path) -> float:
    """Run one fork to completion (GPU is serial — forks run one at a time).
    Returns wall-seconds. Streams trainer output to `log_path`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        log.write("# " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise SystemExit(
            f"fork failed (rc={proc.returncode}); see {log_path}\n"
            f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}"
        )
    return dt


# ---------------------------------------------------------------------------
# Anchored eval (reuses gomoku.eval — the noise fix: ~100+ games per baseline).
# ---------------------------------------------------------------------------


def anchored_eval(
    checkpoint: str,
    *,
    baselines: list[str],
    n_games: int,
    sims: int,
    c_puct: float,
    n_workers: int,
    device: str,
    seed: int,
) -> EloEstimate:
    """Eval one checkpoint vs the anchor baselines with n_games each, returning
    an elo estimate + CI. Reuses the repo's match engine (gomoku.eval) and the
    anchor table (gomoku.rating) — no re-implementation of game play.

    This is the noise fix: the trainer evals with ~20 games (±100 elo); we use
    n_games >= ~100 so the per-baseline Wilson CIs (and the propagated elo CI)
    are tight enough that a Δelo can clear them.
    """
    from gomoku.eval import play_match_parallel, play_match_pickers, mcts_picker
    from gomoku.match import build_player, parse_spec
    from gomoku.rating import ANCHOR_ELOS, baseline_spec_to_anchor_key

    matches: list[BaselineMatch] = []
    # Build model picker once for the sequential path.
    model_picker = None
    if n_workers <= 1:
        from gomoku.mcts import make_torch_evaluator
        from gomoku.model import fuse_model_for_inference, load_checkpoint
        from gomoku.util import pick_device

        dev = pick_device(device)
        model, _ = load_checkpoint(checkpoint, device=dev)
        model = fuse_model_for_inference(model)
        evaluator = make_torch_evaluator(model, dev)
        model_picker = mcts_picker(evaluator, n_simulations=sims, c_puct=c_puct)

    for b_idx, raw in enumerate(baselines):
        spec = parse_spec(raw)
        anchor_key = baseline_spec_to_anchor_key(spec)
        anchor_elo = ANCHOR_ELOS.get(anchor_key) if anchor_key else None
        if anchor_elo is None:
            raise SystemExit(
                f"baseline {raw!r} has no anchor elo in gomoku.rating.ANCHOR_ELOS "
                f"(key={anchor_key!r}). Use one of: {sorted(ANCHOR_ELOS)}"
            )
        match_seed = seed + b_idx * 10_000
        if n_workers > 1:
            res = play_match_parallel(
                checkpoint_path=checkpoint, opp_spec=raw, n_games=n_games,
                seed=match_seed, n_workers=n_workers, sims=sims, c_puct=c_puct,
                device=device,
            )
        else:
            res = play_match_pickers(
                model_picker, build_player(spec), n_games=n_games, seed=match_seed,
            )
        matches.append(BaselineMatch(
            baseline=spec.label(), anchor_elo=float(anchor_elo),
            n_games=res.n_games, wins=res.wins, draws=res.draws,
        ))
        print(f"[delta-e] {Path(checkpoint).name} vs {spec.label():<18} "
              f"{res.wins}W-{res.losses}L-{res.draws}D ({res.win_rate:.0%}) "
              f"anchor={anchor_elo:.0f}", flush=True)
    return estimate_elo(matches)


# ---------------------------------------------------------------------------
# Head-to-head eval (reuses gomoku.eval — the anchor-ceiling fix).
# ---------------------------------------------------------------------------


def _build_ckpt_picker(checkpoint: str, *, sims: int, c_puct: float, device: str):
    """Load a checkpoint and wrap it in an mcts_picker (same load/fuse/evaluator
    idiom as anchored_eval's sequential model_picker)."""
    from gomoku.eval import mcts_picker
    from gomoku.mcts import make_torch_evaluator
    from gomoku.model import fuse_model_for_inference, load_checkpoint
    from gomoku.util import pick_device

    dev = pick_device(device)
    model, _ = load_checkpoint(checkpoint, device=dev)
    model = fuse_model_for_inference(model)
    evaluator = make_torch_evaluator(model, dev)
    return mcts_picker(evaluator, n_simulations=sims, c_puct=c_puct)


def _play_h2h_paired(
    fork_picker,
    parent_picker,
    *,
    n_games: int,
    opening_plies: int,
    sims: int,
    c_puct: float,
    seed: int,
) -> tuple[int, int, int, int]:
    """Fork-vs-C with PAIRED RANDOM OPENINGS. Returns (wins, draws, losses, n_played)
    from the fork's perspective.

    Why this exists instead of gomoku.eval.play_match_pickers: the match pickers
    are temperature=0 argmax with no root noise (deterministic). Two such players
    from the empty board replay exactly ONE line per color assignment — so a
    "100-game" match is really 2 distinct games, and its binomial CI is false
    precision. Here each opening is generated by `random_opening_state` and played
    TWICE (fork as black, then fork as white) so the games are INDEPENDENT and
    color bias cancels within each pair. This is standard paired-opening match
    practice; it's what makes the relative-Δelo CI honest.
    """
    import numpy as np

    from gomoku.game import GameState
    from gomoku.self_play import _random_opening_state

    wins = losses = draws = 0
    n_pairs = max(1, n_games // 2)
    for pair_idx in range(n_pairs):
        opening_rng = np.random.default_rng(seed + pair_idx)
        if opening_plies > 0:
            opening_state, opening_ply_count = _random_opening_state(opening_rng, opening_plies)
        else:
            opening_state, opening_ply_count = GameState.initial(), 0
        for fork_is_black in (True, False):
            game_rng = np.random.default_rng(
                seed + 1_000_000 + pair_idx * 2 + (0 if fork_is_black else 1))
            state = opening_state
            ply = opening_ply_count
            # Side to move at `ply`: 0 = black. Fork moves iff that matches its color.
            fork_to_move = (ply % 2 == 0) == fork_is_black
            winner_side: int | None = None
            while True:
                picker = fork_picker if fork_to_move else parent_picker
                action = picker(state, game_rng)
                side_just_moved = ply % 2
                state = state.apply(action)
                done, v = state.is_terminal()
                if done:
                    if v == -1.0:
                        winner_side = side_just_moved
                    break
                fork_to_move = not fork_to_move
                ply += 1
            if winner_side is None:
                draws += 1
            else:
                fork_side = 0 if fork_is_black else 1
                if winner_side == fork_side:
                    wins += 1
                else:
                    losses += 1
    return wins, draws, losses, 2 * n_pairs


def head_to_head_eval(
    fork_picker,
    parent_picker,
    *,
    recipe_label: str,
    window_epochs: int,
    wall_secs: float | None,
    n_games: int,
    sims: int,
    c_puct: float,
    seed: int,
    opening_plies: int = 4,
) -> HeadToHeadResult:
    """Play one fork-vs-C direct match (model-vs-model, paired random openings)
    and return a HeadToHeadResult. fork is picker_a, so wins are the fork's.

    This is the anchor-ceiling fix: when every checkpoint beats the top anchor,
    two similar models still score near 50% against EACH OTHER — the maximally
    sensitive region of the logistic — so the relative Δelo has a tight CI.
    Paired random openings (see `_play_h2h_paired`) give INDEPENDENT games despite
    the deterministic pickers; without them the CI is false precision.
    """
    wins, draws, losses, n_played = _play_h2h_paired(
        fork_picker, parent_picker, n_games=n_games, opening_plies=opening_plies,
        sims=sims, c_puct=c_puct, seed=seed)
    out = HeadToHeadResult(
        recipe_label=recipe_label, window_epochs=window_epochs, wall_secs=wall_secs,
        n_games=n_played, wins=wins, draws=draws, losses=losses,
    )
    decisive = wins + losses
    print(f"[delta-e] H2H {recipe_label}: fork {wins}W-{losses}L-{draws}D "
          f"vs C ({out.win_rate:.0%}, {decisive}/{n_played} decisive, "
          f"openings={opening_plies}p) -> Δelo={out.delta_elo:+.1f} "
          f"(±{out.delta_ci_half:.1f})", flush=True)
    return out


# ---------------------------------------------------------------------------
# External-anchor hook — SCAFFOLD ONLY (do NOT build Rapfi integration here).
# ---------------------------------------------------------------------------


class ExternalAnchor:
    """Pluggable external opponent at a KNOWN fixed elo (e.g. a pinned Rapfi
    build, or a Gomocup champion engine). Scaffold only — wired into the same
    anchored-eval path as the built-in baselines once implemented.

    To add a real external anchor, implement `play(checkpoint, n_games, seed)`
    to return (wins, draws, n_games) from the model's perspective and set
    `anchor_elo` to the opponent's calibrated rating; then feed a BaselineMatch
    built from its result into estimate_elo alongside the built-in baselines.
    The Δelo/ranking math downstream is opponent-agnostic, so NO changes are
    needed there — only this adapter.

    Deliberately NOT implemented: external-engine process management, protocol
    bridging (Gomocup/Yixin protocol), and elo calibration of the external
    opponent are a separate data-integration chunk, gated behind the
    self-play+trouble-archive loop proving out (wiki Sequencing step 3).
    """

    def __init__(self, name: str, anchor_elo: float):
        self.name = name
        self.anchor_elo = anchor_elo

    def play(self, checkpoint: str, *, n_games: int, seed: int) -> BaselineMatch:
        raise NotImplementedError(
            "ExternalAnchor is a scaffold. Use built-in baselines "
            "(heuristic/lookahead) for now; see wiki Sequencing step 3."
        )


# ---------------------------------------------------------------------------
# Self-test (synthetic match data — exercises the elo/CI/ranking/judgment math).
# ---------------------------------------------------------------------------


def _synthetic_matches(win_rates: dict[str, float], n_games: int) -> list[BaselineMatch]:
    """Build BaselineMatch records from target win-rates (draws=0) so the
    self-test feeds known winrates straight into the elo/CI path."""
    from gomoku.rating import ANCHOR_ELOS, baseline_spec_to_anchor_key
    from gomoku.match import parse_spec

    out: list[BaselineMatch] = []
    for raw, wr in win_rates.items():
        spec = parse_spec(raw)
        anchor = ANCHOR_ELOS[baseline_spec_to_anchor_key(spec)]
        wins = round(wr * n_games)
        out.append(BaselineMatch(baseline=spec.label(), anchor_elo=float(anchor),
                                 n_games=n_games, wins=wins, draws=0))
    return out


def self_test() -> int:
    print("=== delta-e self-test: elo + CI + Δelo + ranking on SYNTHETIC data ===\n")

    # 1) binomial CI sanity: tighter with more games.
    lo20, hi20 = binomial_ci(0.6 * 20, 20)
    lo120, hi120 = binomial_ci(0.6 * 120, 120)
    hw20, hw120 = (hi20 - lo20) / 2, (hi120 - lo120) / 2
    print(f"[CI] win-rate 60%: n=20 CI=[{lo20:.2f},{hi20:.2f}] (±{hw20:.3f})  "
          f"n=120 CI=[{lo120:.2f},{hi120:.2f}] (±{hw120:.3f})")
    assert hw120 < hw20, "more games must tighten the CI"
    print("     OK: more games tighten the win-rate CI (the noise fix).\n")

    n = 120
    # Common parent C: a middling model.
    parent = estimate_elo(_synthetic_matches(
        {"heuristic": 0.70, "lookahead:depth=2": 0.45, "lookahead:depth=4": 0.20}, n))
    print(f"[parent C] elo={parent.elo:+.1f}  CI=[{parent.lo:+.1f},{parent.hi:+.1f}]  "
          f"±{parent.half_width:.1f}")

    # Three recipes forked off C, evaluated at the same window T.
    T = 40
    forks = {
        # Clear improvement — should rank #1 and read as 'signal'.
        "recency_weighted,sgd=100": (
            {"heuristic": 0.92, "lookahead:depth=2": 0.74, "lookahead:depth=4": 0.50}, 1800.0),
        # Modest improvement.
        "diversity,sgd=150": (
            {"heuristic": 0.80, "lookahead:depth=2": 0.58, "lookahead:depth=4": 0.34}, 2400.0),
        # Basically flat vs C — should read INSIDE-NOISE.
        "trouble_pinned,sgd=80": (
            {"heuristic": 0.71, "lookahead:depth=2": 0.47, "lookahead:depth=4": 0.21}, 1500.0),
    }
    results: list[DeltaEResult] = []
    for label, (wr, wall) in forks.items():
        fork = estimate_elo(_synthetic_matches(wr, n))
        results.append(DeltaEResult(recipe_label=label, window_epochs=T,
                                    wall_secs=wall, parent=parent, fork=fork))

    print()
    print(format_table(results))
    print()

    ranked = rank_results(results)
    # Assertions: ordering by Δelo, the flat recipe is inside-noise, the big one isn't.
    assert ranked[0].recipe_label.startswith("recency_weighted"), "best Δelo should rank #1"
    flat = next(r for r in results if r.recipe_label.startswith("trouble_pinned"))
    big = next(r for r in results if r.recipe_label.startswith("recency_weighted"))
    assert flat.inside_noise, "near-flat recipe must read INSIDE-NOISE"
    assert not big.inside_noise, "clear-win recipe must read as signal"
    # Δelo/hr rate ordering can differ from Δelo ordering (the whole point of the rate).
    rate_ranked = sorted([r for r in results if r.delta_elo_per_hr is not None],
                         key=lambda r: r.delta_elo_per_hr, reverse=True)
    print(f"[rate] best Δelo/hr recipe: {rate_ranked[0].recipe_label} "
          f"({rate_ranked[0].delta_elo_per_hr:+.1f} elo/hr) "
          f"vs best Δelo recipe: {ranked[0].recipe_label}")

    # 2) Head-to-head math (the anchor-ceiling fix). No GPU/I/O — synthetic
    #    fork-vs-C scores fed straight through the elo logistic + Wilson CI.
    print("\n=== head-to-head: relative Δelo (fork vs common parent C) ===")
    # p=0.5 -> ~0 elo; the symmetry/sign anchors.
    assert abs(elo_from_score_rate(0.5)) < 1e-9, "p=0.5 must give 0 elo"
    assert elo_from_score_rate(0.6) > 0 and elo_from_score_rate(0.4) < 0, "sign must track p"
    assert abs(elo_from_score_rate(0.6) - 70.4) < 0.2, "p=0.6 -> +70.4 elo"
    assert math.isfinite(elo_from_score_rate(1.0)) and math.isfinite(elo_from_score_rate(0.0)), \
        "clamp must keep a 100%/0% sweep finite (no log(0)/div-by-zero)"

    def _h2h(p: float, n: int) -> HeadToHeadResult:
        wins = round(p * n)
        return HeadToHeadResult(recipe_label="r", window_epochs=40, wall_secs=3600.0,
                                n_games=n, wins=wins, draws=0, losses=n - wins)

    # Same effect size (p=0.64), two sample sizes: the CI tightens with n so the
    # SAME edge that is buried at n=20 becomes a clear signal at n=120.
    small = _h2h(0.64, 20)
    big = _h2h(0.64, 120)
    print(f"[h2h] p=0.64  n=20:  Δelo={small.delta_elo:+.1f} ±{small.delta_ci_half:.1f}  "
          f"-> {'INSIDE-NOISE' if small.inside_noise else 'signal'}")
    print(f"[h2h] p=0.64  n=120: Δelo={big.delta_elo:+.1f} ±{big.delta_ci_half:.1f}  "
          f"-> {'INSIDE-NOISE' if big.inside_noise else 'signal'}")
    assert big.delta_elo > 0, "p=0.64 fork is stronger than C (positive Δelo)"
    assert big.delta_ci_half < small.delta_ci_half, "more games must tighten the Δelo CI"
    assert not big.inside_noise, "p=0.64 at n=120 must read as signal"
    assert small.inside_noise, "p=0.64 at n=20 must be buried INSIDE-NOISE (CI too wide)"

    # Near-tie reads INSIDE-NOISE regardless; a 50% match is exactly 0 elo.
    tie = _h2h(0.5, 120)
    assert abs(tie.delta_elo) < 1e-9 and tie.inside_noise, "50% match -> 0 elo, inside-noise"

    h2h_results = [_h2h(0.64, 120), _h2h(0.52, 120), _h2h(0.40, 120)]
    h2h_results[0].recipe_label = "recency_weighted,sgd=100"
    h2h_results[1].recipe_label = "diversity,sgd=150"
    h2h_results[2].recipe_label = "trouble_pinned,sgd=80"
    print()
    print(format_h2h_table(h2h_results))
    assert rank_h2h(h2h_results)[0].recipe_label.startswith("recency_weighted"), \
        "highest-scoring fork ranks #1 in head-to-head"

    print("\nself-test PASSED: CI tightens with n; anchored elo/Δelo/CI computed; "
          "ranked by Δelo; inside-noise judgment correct; Δelo/hr rate available; "
          "head-to-head relative Δelo + CI tightening + sensitivity verified.")
    return 0


# ---------------------------------------------------------------------------
# Orchestration entry point.
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parent", type=str, help="Common parent checkpoint C to fork from.")
    p.add_argument("--archive-path", type=str, help="Disk archive A passed to the trainer.")
    p.add_argument("--window-epochs", type=int, default=40,
                   help="Fixed window T (epochs) each recipe trains for.")
    p.add_argument("--recipe", dest="recipes", action="append", default=[],
                   help="Recipe spec 'buffer_recipe[:k=v,...]'. Repeatable.")
    p.add_argument("--default-sgd-steps-per-epoch", type=int, default=100,
                   help="sgd_steps_per_epoch used when a recipe doesn't override it.")
    p.add_argument("--size", type=str, default="small")
    # Absolute by default: archives/ is gitignored (lives only in the main repo),
    # and forks may run with a worktree cwd, so a relative path fails. Override
    # with --validation-archive-path for a different location.
    p.add_argument("--validation-archive-path", type=str,
                   default="/Users/jason/code/gomoku/archives/wl5_validation_v1.pt")
    p.add_argument("--out-dir", type=str, default="sweep_logs/delta-e",
                   help="Per-recipe fork checkpoints/logs + results.json land here.")
    # Anchored-eval knobs (the noise fix).
    p.add_argument("--eval-baselines", type=str,
                   default="heuristic,lookahead:depth=2,lookahead:depth=4",
                   help="Comma-separated baseline specs (must have anchor elos).")
    p.add_argument("--eval-games", type=int, default=120,
                   help="Games per baseline for anchored eval (>=~100 to beat the "
                        "±100-elo trainer-eval noise).")
    p.add_argument("--eval-sims", type=int, default=100)
    p.add_argument("--eval-c-puct", type=float, default=1.5)
    p.add_argument("--eval-workers", type=int, default=6,
                   help="Parallel eval workers (CPU is idle while GPU forks serially).")
    p.add_argument("--eval-device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true", help="Let forks log to wandb.")
    p.add_argument("--head-to-head", action="store_true",
                   help="Anchor-ceiling fix: skip anchored-eval of C and instead "
                        "play each fork DIRECTLY vs the common parent C "
                        "(model-vs-model, alternating colors). Relative Δelo has a "
                        "tight CI with no anchor ceiling. Uses --eval-games games, "
                        "--eval-sims, --eval-c-puct, --eval-device, --seed.")
    p.add_argument("--h2h-opening-plies", type=int, default=4,
                   help="Head-to-head only: random opening plies per game pair. "
                        ">0 is REQUIRED for an honest CI — temperature=0 pickers "
                        "replay one line per color from the empty board (2 distinct "
                        "games). Each opening is played twice (colors swapped).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the exact per-recipe train commands and exit (no GPU).")
    p.add_argument("--self-test", action="store_true",
                   help="Run the synthetic elo/CI/ranking self-test and exit.")
    return p.parse_args(argv)


def _run_head_to_head(
    args: argparse.Namespace,
    plans: list[tuple[Recipe, list[str], Path]],
    out_dir: Path,
    baselines: list[str],
) -> int:
    """Real head-to-head run: fork sequentially, then play each fork DIRECTLY vs
    the common parent C. C's picker is loaded ONCE and reused across all forks."""
    print("[delta-e] head-to-head mode: forks will be matched directly vs C "
          "(no anchored-eval of C).", flush=True)
    print(f"[delta-e] loading common parent C picker once: {args.parent}", flush=True)
    parent_picker = _build_ckpt_picker(
        args.parent, sims=args.eval_sims, c_puct=args.eval_c_puct,
        device=args.eval_device)

    results: list[HeadToHeadResult] = []
    for i, (r, cmd, ckpt_dir) in enumerate(plans, 1):
        print(f"[delta-e] === fork {i}/{len(plans)}: {r.label} ===", flush=True)
        log_path = out_dir / f"fork_{r.slug}" / "trainer.log"
        wall = run_fork(cmd, log_path=log_path)
        final_ckpt = final_checkpoint_for(str(ckpt_dir))
        if not final_ckpt.exists():
            raise SystemExit(f"fork {r.label} produced no checkpoint at {final_ckpt}")
        print(f"[delta-e] fork done in {wall:.0f}s; head-to-head {final_ckpt} vs C...",
              flush=True)
        fork_picker = _build_ckpt_picker(
            str(final_ckpt), sims=args.eval_sims, c_puct=args.eval_c_puct,
            device=args.eval_device)
        results.append(head_to_head_eval(
            fork_picker, parent_picker, recipe_label=r.label,
            window_epochs=args.window_epochs, wall_secs=wall,
            n_games=args.eval_games, sims=args.eval_sims, c_puct=args.eval_c_puct,
            seed=args.seed, opening_plies=args.h2h_opening_plies))

    print("\n" + format_h2h_table(results))

    record = {
        "ts": time.time(),
        "mode": "head-to-head",
        "parent": args.parent,
        "archive_path": args.archive_path,
        "window_epochs": args.window_epochs,
        "eval": {"mode": "head-to-head", "games": args.eval_games,
                 "sims": args.eval_sims, "c_puct": args.eval_c_puct,
                 "device": args.eval_device,
                 "opening_plies": args.h2h_opening_plies},
        "results": [
            {
                "recipe": r.recipe_label,
                "delta_elo": r.delta_elo,
                "delta_ci_half": r.delta_ci_half,
                "window_epochs": r.window_epochs,
                "wall_secs": r.wall_secs,
                "delta_elo_per_hr": r.delta_elo_per_hr,
                "inside_noise": r.inside_noise,
                "n_games": r.n_games,
                "wins": r.wins, "draws": r.draws, "losses": r.losses,
                "score": r.score, "win_rate": r.win_rate,
            }
            for r in rank_h2h(results)
        ],
    }
    rec_path = out_dir / "results.json"
    rec_path.write_text(json.dumps(record, indent=2))
    print(f"\n[delta-e] wrote {rec_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.recipes:
        raise SystemExit("need at least one --recipe (or use --self-test)")
    if not args.parent or not args.archive_path:
        raise SystemExit("--parent and --archive-path are required (or use --self-test)")

    recipes = [parse_recipe(s, default_sgd=args.default_sgd_steps_per_epoch)
               for s in args.recipes]
    out_dir = Path(args.out_dir)
    baselines = [b.strip() for b in args.eval_baselines.split(",") if b.strip()]

    # Build every fork command up front so --dry-run prints exactly what would run.
    plans: list[tuple[Recipe, list[str], Path]] = []
    for r in recipes:
        ckpt_dir = out_dir / f"fork_{r.slug}" / "checkpoints"
        cmd = build_fork_command(
            r, parent=args.parent, archive_path=args.archive_path,
            window_epochs=args.window_epochs, size=args.size,
            validation_archive_path=args.validation_archive_path,
            checkpoint_dir=str(ckpt_dir), seed=args.seed, wandb=args.wandb,
        )
        plans.append((r, cmd, ckpt_dir))

    if args.dry_run:
        print("=== delta-e DRY-RUN: per-recipe fork commands (NOT executed) ===\n")
        print(f"parent C : {args.parent}")
        print(f"archive A: {args.archive_path}")
        print(f"window T : {args.window_epochs} epochs   (forks run SEQUENTIALLY — GPU is serial)\n")
        for i, (r, cmd, ckpt_dir) in enumerate(plans, 1):
            print(f"--- recipe {i}/{len(plans)}: {r.label}")
            print("  " + " ".join(shlex.quote(c) for c in cmd))
            print(f"  -> final checkpoint: {final_checkpoint_for(str(ckpt_dir))}\n")
        if args.head_to_head:
            print("=== then HEAD-TO-HEAD eval: each fork's final checkpoint vs C ===")
            print("  (anchor-ceiling fix: skips anchored-eval of C; plays "
                  "fork-vs-C directly)")
            print(f"  games/match: {args.eval_games} (sims={args.eval_sims}, "
                  f"c_puct={args.eval_c_puct}, device={args.eval_device}, "
                  f"paired openings={args.h2h_opening_plies}p)")
            print("  -> Δelo(vs C) = 400*log10(p/(1-p)), p=fork score-rate; "
                  "rank by Δelo and Δelo/hr.")
        else:
            print("=== then anchored-eval each fork's final checkpoint + C ===")
            print(f"  baselines: {baselines}")
            print(f"  games/baseline: {args.eval_games} (sims={args.eval_sims}, "
                  f"workers={args.eval_workers})")
            print("  -> Δelo = elo(fork_final) - elo(C); rank by Δelo and Δelo/hr.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.head_to_head:
        return _run_head_to_head(args, plans, out_dir, baselines)

    # --- Real run: fork sequentially, then anchored-eval, then rank. ---
    print("[delta-e] anchored-eval of common parent C (the Δelo baseline)...", flush=True)
    parent_est = anchored_eval(
        args.parent, baselines=baselines, n_games=args.eval_games,
        sims=args.eval_sims, c_puct=args.eval_c_puct, n_workers=args.eval_workers,
        device=args.eval_device, seed=args.seed)
    print(f"[delta-e] parent C elo={parent_est.elo:+.1f} "
          f"[{parent_est.lo:+.1f},{parent_est.hi:+.1f}]\n", flush=True)

    results: list[DeltaEResult] = []
    for i, (r, cmd, ckpt_dir) in enumerate(plans, 1):
        print(f"[delta-e] === fork {i}/{len(plans)}: {r.label} ===", flush=True)
        log_path = out_dir / f"fork_{r.slug}" / "trainer.log"
        wall = run_fork(cmd, log_path=log_path)
        final_ckpt = final_checkpoint_for(str(ckpt_dir))
        if not final_ckpt.exists():
            raise SystemExit(f"fork {r.label} produced no checkpoint at {final_ckpt}")
        print(f"[delta-e] fork done in {wall:.0f}s; anchored-eval {final_ckpt}...",
              flush=True)
        fork_est = anchored_eval(
            str(final_ckpt), baselines=baselines, n_games=args.eval_games,
            sims=args.eval_sims, c_puct=args.eval_c_puct, n_workers=args.eval_workers,
            device=args.eval_device, seed=args.seed)
        results.append(DeltaEResult(
            recipe_label=r.label, window_epochs=args.window_epochs,
            wall_secs=wall, parent=parent_est, fork=fork_est))

    print("\n" + format_table(results))

    # Persist a machine-readable record.
    record = {
        "ts": time.time(),
        "parent": args.parent,
        "archive_path": args.archive_path,
        "window_epochs": args.window_epochs,
        "eval": {"baselines": baselines, "games": args.eval_games,
                 "sims": args.eval_sims, "workers": args.eval_workers},
        "parent_elo": {"elo": parent_est.elo, "lo": parent_est.lo, "hi": parent_est.hi},
        "results": [
            {
                "recipe": r.recipe_label,
                "delta_elo": r.delta_elo,
                "delta_ci_half": r.delta_ci_half,
                "window_epochs": r.window_epochs,
                "wall_secs": r.wall_secs,
                "delta_elo_per_hr": r.delta_elo_per_hr,
                "inside_noise": r.inside_noise,
                "fork_elo": {"elo": r.fork.elo, "lo": r.fork.lo, "hi": r.fork.hi},
                "win_rates": {m.baseline: m.win_rate for m in r.fork.matches},
            }
            for r in rank_results(results)
        ],
    }
    rec_path = out_dir / "results.json"
    rec_path.write_text(json.dumps(record, indent=2))
    print(f"\n[delta-e] wrote {rec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
