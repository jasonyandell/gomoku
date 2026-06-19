"""Engine-panel-anchored ROUND-ROBIN tournament — a CALIBRATED strength yardstick.

GitHub issue #30 ("the engine-panel-anchored derby"). Sibling design doc:
``wiki/topics/engine-panel-derby-design.md``.

------------------------------------------------------------------------------
The #30 idea, in one paragraph
------------------------------------------------------------------------------
Self-play Elo saturates near ~1700 and is non-transitive across recipes, and a
single self-built engine gives no *absolute* calibration. The fix is a FIXED
PANEL of real external Gomocup engines whose published freestyle Elos are known.
We run a full round-robin (our nets + the real engines + a heuristic floor),
fit Bradley-Terry / logistic ratings from the cross-table, then AFFINE-CALIBRATE
those internal ratings to the Gomocup scale using ONLY the real engines as
anchors (least-squares line: internal-rating -> known-Gomocup-Elo). Applying
that same line to *all* players hands our nets and the heuristic a calibrated
absolute Elo on the Gomocup ladder. Because the panel is FIXED and its anchors
have published ratings, the calibration is stable across runs: re-running with a
new champion places it on the *same* ruler, so Δelo between sessions is finally
an absolute, comparable number (the north-star Δelo/Δt gets a real unit).

Why an AFFINE map (slope + intercept), not a single offset? Bradley-Terry
ratings are only defined up to (a) an additive constant (the zero point is
arbitrary) AND (b) a multiplicative scale when regularization shrinks spreads.
A 2-parameter least-squares fit absorbs both: it finds the line through the
anchor cloud (internal_rating_i, gomocup_elo_i) that best maps our internal
ruler onto the published one. With >=2 valid anchors the line is identifiable;
with <2 we cannot calibrate and fall back to *relative* Elo (clearly flagged).

------------------------------------------------------------------------------
Elo math used here
------------------------------------------------------------------------------
* Bradley-Terry / logistic model on the 400/log10 scale: expected score of i
  over j is  E_ij = 1 / (1 + 10**((R_j - R_i)/400)).
* Ratings R are fit by maximum likelihood from the pairwise score matrix
  S_ij = (wins_i + 0.5*draws) over the games i played vs j (a draw is half a win
  for each side). We use Minorization-Maximization (MM) — the standard, robust,
  derivative-free BT solver — with mild L2 regularization toward the field mean
  for stability under sparse/perfect-score data (an undefeated or winless player
  would otherwise diverge to +/-inf).
* Internal ratings are then affine-mapped onto the Gomocup scale via the anchor
  least-squares fit described above.

------------------------------------------------------------------------------
Operational properties
------------------------------------------------------------------------------
* EFFICIENCY: every player is built ONCE up front and reused across all of its
  pairings (a 2.1GB net is never reloaded per pair; the brain wrapper + external
  engines are re-entrant subprocesses). All players are closed at the end.
* ROBUSTNESS: a pair that errors/times out is recorded as an error and the
  tournament CONTINUES — one bad engine never kills the run. A player that fails
  to BUILD has all of its pairs skipped (noted once).
* RESUMABLE: each completed pair appends a JSONL record; on restart, pairs
  already present are skipped. ``--print-table`` recomputes the cross-table and
  calibrated Elo from an existing JSONL without playing a single game.

EVAL-ONLY. Never mix an external engine (or this runner) into self-play
training. This does not touch the GPU lane's training slices.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

# Board size + device MUST be set before importing gomoku (game.py reads
# GOMOKU_BOARD_SIZE at import time to fix BOARD_SIZE; the brain wrapper inherits
# GOMOKU_DEVICE). setdefault so an explicit outer env still wins.
os.environ.setdefault("GOMOKU_BOARD_SIZE", "15")
os.environ.setdefault("GOMOKU_DEVICE", "mps")

# ----------------------------------------------------------------------------
# PARTICIPANTS — edit this list to change the field.
# Each entry is (label, spec_string) in gomoku.match's player-spec grammar
# (`kind[:k=v,...]`). For `external`, the `cmd=` value is COMMA-FREE and uses
# SPACE-separated flags; the comma-separated kwargs that follow (timeout_ms,
# label, size) are the spec's own kwargs. {SIMS} is substituted from --sims.
# ----------------------------------------------------------------------------
_GOMOCUP_BIN = os.path.expanduser("~/.cache/gomocup/bin")
_SWEEP = os.path.expanduser("~/code/gomoku/sweep_runs")
_AZ_WRAPPER = f"{_GOMOCUP_BIN}/run-gomoku-az"

# Net checkpoints fronted by the run-gomoku-az brain wrapper (issue #31).
_NETS = [
    ("az-champ-128x10", f"{_SWEEP}/g15_128x10_bigbuf_eval502.pt"),
    ("az-96x8", f"{_SWEEP}/g15_96x8_bigbuf_eval597.pt"),
    ("az-128x10-e588", f"{_SWEEP}/g15_128x10_bigbuf_e588_best.pt"),
]

# ----------------------------------------------------------------------------
# WINE ENGINES ARE OFF BY DEFAULT (Jason's nix directive, 2026-06-16; issue #35).
# ----------------------------------------------------------------------------
# The five "real engine" anchors below are Windows ``pbrain-*.exe`` binaries run
# under wine. They are FLAKY: ~17/36 panel pairs crashed in the #35 calibration
# attempt (wine segfaults / desyncs), which broke the affine anchor fit. Jason:
# "bail on wine evals — another wine crash. eval just with reliable things."
#
# So this catalog is now GATED behind an explicit opt-in and DEFAULTS TO EMPTY.
# The default reliable panel = our net checkpoints (pure-torch, via the
# run-gomoku-az brain wrapper) + the pure-python ``heuristic`` floor — the same
# torch-only / pure-python eval the sliding derby already uses reliably. See
# ``wiki/topics/reliable-eval-set.md`` and ``wiki/topics/gomocup-engines-catalog.md``.
#
# The catalog is KEPT (not deleted) — it is evidence, and an engine that proves
# reliable (e.g. a NATIVE non-wine Rapfi build, issue #28) can be re-listed. Opt
# in with ``GOMOKU_ENABLE_WINE_ENGINES=1`` or the ``--wine`` CLI flag; both are
# off by default. To add a reliable NATIVE engine, prefer ``_NATIVE_ENGINES``.
_WINE_ENGINES = [
    ("embryo26", f"{_GOMOCUP_BIN}/run-embryo26"),
    ("yixin18", f"{_GOMOCUP_BIN}/run-yixin18"),
    ("pela23", f"{_GOMOCUP_BIN}/run-pela23"),
    ("zetor17", f"{_GOMOCUP_BIN}/run-zetor17"),
    ("eulring16", f"{_GOMOCUP_BIN}/run-eulring16"),
]

# Reliable NATIVE (non-wine) external engines — the future home of a real
# strength anchor (e.g. a native ARM64 Rapfi, issue #28/#35). Empty until a
# native binary + wrapper is brought online; populating this list does NOT
# require the wine opt-in. Entries are (label, run-wrapper-path), same grammar
# as the wine catalog.
_NATIVE_ENGINES: list[tuple[str, str]] = [
    # Native arm64 Rapfi-NNUE (issues #40/#28/#35) — the real strength anchor.
    # The `run-rapfi` wrapper passes `--config engines/rapfi/config.toml`, which
    # loads the mix9svq NNUE weights and makes Rapfi actually search to its time
    # budget (the weightless default was the #28 "depth-10 / illusory-TC" bug).
    # Single-threaded (Gomocup-legal). No wine opt-in needed.
    ("rapfi", f"{_GOMOCUP_BIN}/run-rapfi"),
]


def wine_engines_enabled(cli_wine: bool = False) -> bool:
    """Wine engines are opt-in only (issue #35; Jason's nix directive).

    Enabled iff the ``--wine`` flag is passed OR ``GOMOKU_ENABLE_WINE_ENGINES``
    is set to a truthy value (``1``/``true``/``yes``/``on``). Default: OFF.
    """
    if cli_wine:
        return True
    val = os.environ.get("GOMOKU_ENABLE_WINE_ENGINES", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def real_engines(*, enable_wine: bool) -> list[tuple[str, str]]:
    """The external-engine anchors for the field.

    Always includes the reliable NATIVE engines; includes the flaky wine
    catalog ONLY when explicitly opted in. Default (no opt-in) = native only,
    which is currently empty -> a pure torch + pure-python reliable panel.
    """
    engines = list(_NATIVE_ENGINES)
    if enable_wine:
        engines += list(_WINE_ENGINES)
    return engines


# Back-compat alias: ``panel_white_elo.py`` and older callers import the anchor
# catalog by this name. It now reflects the DEFAULT (reliable) field — native
# engines only, no wine — so a default analysis never assumes wine players.
_REAL_ENGINES = real_engines(enable_wine=False)

# Default per-engine move budget (ms) for ALL external subprocesses (nets +
# real engines). Generous because GPU-contended / wine engines need slack.
_DEFAULT_TIMEOUT_MS = 10000
_DEFAULT_SIMS = 200


def _build_participants(
    sims: int, timeout_ms: int, *, enable_wine: bool = False
) -> list[tuple[str, str]]:
    """Materialize the (label, spec) field for the given sims/timeout.

    Net specs put the checkpoint + sims inside the comma-free ``cmd=`` value
    (space-separated flags), then carry ``timeout_ms``/``label``/``size`` as
    spec kwargs after the first comma — exactly how ``run-gomoku-az`` documents
    its registration string.

    Wine engines are EXCLUDED unless ``enable_wine`` (issue #35; default reliable
    panel = nets + the pure-python heuristic floor + any native engines).
    """
    parts: list[tuple[str, str]] = []
    for label, pt in _NETS:
        cmd = f"{_AZ_WRAPPER} --checkpoint {pt} --sims {sims}"
        # incremental=1 is MANDATORY for our brain wrapper (issue #31): it drives
        # the net via TURN so it keeps real move history. Without it the net plays
        # at ~25% strength (empty-history OOD input). Classical engines below stay
        # on the default BOARD path.
        parts.append(
            (label,
             f"external:cmd={cmd},timeout_ms={timeout_ms},label={label},size=15,incremental=1")
        )
    for label, cmd in real_engines(enable_wine=enable_wine):
        parts.append(
            (label, f"external:cmd={cmd},timeout_ms={timeout_ms},label={label}")
        )
    # The heuristic baseline — a non-trivial in-repo floor (no subprocess, no
    # timeout). Flavor + a low anchor for the field's bottom end.
    parts.append(("heuristic", "heuristic"))
    return parts


# ----------------------------------------------------------------------------
# Published Gomocup FREESTYLE Elo anchors — calibration targets.
#
# *** APPROXIMATE ***  These are the best-published values recorded in
# wiki/topics/gomocup-engines-catalog.md (snapshot 2026-06-15), which sources
# them from gomocup.org rank/Elo listings. They are competition (single-thread)
# freestyle approximations, NOT exact per-tournament BayesElo readings, and may
# drift year to year. Treat them as a fixed RULER, not ground truth; if Gomocup
# publishes updated numbers, edit here. Only labels present in BOTH this dict
# AND the played field are used as calibration anchors.
# ----------------------------------------------------------------------------
GOMOCUP_ELO_ANCHORS: dict[str, float] = {
    "embryo26": 2402.0,   # catalog: "Embryo (Embryo26) ~2402"
    "yixin18": 2310.0,    # catalog: "Yixin ~2310" (top ~2014-2018)
    "zetor17": 1700.0,    # APPROX: no exact catalog Elo; mid-field placeholder
    "eulring16": 1600.0,  # APPROX: no exact catalog Elo; mid-field placeholder
    "pela23": 1499.0,     # catalog: "Pela ~1499"
}

# Elo scale constant: 400 points per factor-of-10 odds (the standard chess/Elo
# convention; matches MatchResult.win_rate's draw=half scoring).
_ELO_SCALE = 400.0
_LN10 = math.log(10.0)


# ----------------------------------------------------------------------------
# Result records
# ----------------------------------------------------------------------------
@dataclass
class PairRecord:
    """One completed (or errored) round-robin pairing, A vs B.

    Scores/colors are from A's perspective (``play_match_pickers`` returns the
    result from picker A). ``error`` is set (and W/L/D are None) when the pair
    could not be played.
    """

    label_a: str
    label_b: str
    n_games: int
    wins: Optional[int] = None      # A's wins
    losses: Optional[int] = None    # A's losses
    draws: Optional[int] = None
    win_rate: Optional[float] = None
    black_w: Optional[int] = None
    black_l: Optional[int] = None
    black_d: Optional[int] = None
    white_w: Optional[int] = None
    white_l: Optional[int] = None
    white_d: Optional[int] = None
    seconds: Optional[float] = None
    error: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "label_a": self.label_a,
            "label_b": self.label_b,
            "n_games": self.n_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "black": [self.black_w, self.black_l, self.black_d],
            "white": [self.white_w, self.white_l, self.white_d],
            "seconds": self.seconds,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, d: dict) -> "PairRecord":
        black = d.get("black") or [None, None, None]
        white = d.get("white") or [None, None, None]
        return cls(
            label_a=d["label_a"],
            label_b=d["label_b"],
            n_games=d.get("n_games", 0),
            wins=d.get("wins"),
            losses=d.get("losses"),
            draws=d.get("draws"),
            win_rate=d.get("win_rate"),
            black_w=black[0], black_l=black[1], black_d=black[2],
            white_w=white[0], white_l=white[1], white_d=white[2],
            seconds=d.get("seconds"),
            error=d.get("error"),
        )


def _pair_key(label_a: str, label_b: str) -> tuple[str, str]:
    """Order-independent key for a pairing (round-robin pairs are unordered)."""
    return tuple(sorted((label_a, label_b)))  # type: ignore[return-value]


def _pair_seed(label_a: str, label_b: str, base_seed: int) -> int:
    """Deterministic, order-independent per-pair seed.

    Hashing the sorted label pair makes the seed reproducible across runs and
    independent of which player was assigned A vs B in the loop (the same pair
    always gets the same opening positions).
    """
    a, b = _pair_key(label_a, label_b)
    h = abs(hash((a, b))) % 2_000_000_000
    return (base_seed + h) % 2_000_000_000


# ----------------------------------------------------------------------------
# JSONL resume I/O
# ----------------------------------------------------------------------------
def load_existing(out_path: str) -> dict[tuple[str, str], PairRecord]:
    """Load already-completed pairs from the JSONL, keyed by unordered pair.

    On a duplicate key the LAST record wins (a re-run that re-played a
    previously-errored pair supersedes the earlier error). Malformed lines are
    skipped so a partially-written tail never aborts a resume.
    """
    done: dict[tuple[str, str], PairRecord] = {}
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = PairRecord.from_json(json.loads(line))
            except Exception:
                continue
            done[_pair_key(rec.label_a, rec.label_b)] = rec
    return done


def append_record(out_path: str, rec: PairRecord) -> None:
    """Append one pair record as a JSONL line (flush immediately for resume)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec.to_json()) + "\n")
        f.flush()


# ----------------------------------------------------------------------------
# Player building (once per label) + graceful failure
# ----------------------------------------------------------------------------
def build_all_players(participants: list[tuple[str, str]]) -> tuple[dict, dict]:
    """Build every player ONCE. Returns (players, build_errors).

    ``players[label]`` is the constructed Picker (reused across all its pairs);
    ``build_errors[label]`` records the message for any player that failed to
    build (its pairs are skipped). A net is never reloaded per pair — the brain
    wrapper and external engines are re-entrant subprocesses held open for the
    whole tournament and closed at the end.
    """
    from gomoku.match import build_player, parse_spec

    players: dict = {}
    build_errors: dict[str, str] = {}
    for label, spec in participants:
        try:
            print(f"[panel] building {label} ...", flush=True)
            players[label] = build_player(parse_spec(spec))
        except Exception as e:  # build failure: skip this player's pairs
            msg = f"{type(e).__name__}: {e}"
            build_errors[label] = msg
            print(f"[panel] BUILD FAILED {label}: {msg}", flush=True)
    return players, build_errors


def close_all_players(players: dict) -> None:
    """Close every player that exposes a ``close()`` (external subprocesses)."""
    for label, p in players.items():
        close = getattr(p, "close", None)
        if callable(close):
            try:
                close()
            except Exception as e:
                print(f"[panel] close({label}) raised: {e}", flush=True)


# ----------------------------------------------------------------------------
# Round-robin
# ----------------------------------------------------------------------------
def run_round_robin(
    participants: list[tuple[str, str]],
    *,
    n_games: int,
    base_seed: int,
    random_opening_moves: int,
    out_path: str,
) -> list[PairRecord]:
    """Play every unordered pair N games, streaming + appending results.

    Already-recorded pairs (in ``out_path``) are skipped (resume). Each pair is
    wrapped in try/except: an engine error/timeout records an error entry and
    the tournament CONTINUES. Players whose build failed have all their pairs
    skipped. Returns the full list of records (loaded + newly played).
    """
    from gomoku.eval import play_match_pickers

    players, build_errors = build_all_players(participants)
    existing = load_existing(out_path)
    labels = [lbl for lbl, _ in participants]
    all_records: list[PairRecord] = list(existing.values())

    try:
        for label_a, label_b in itertools.combinations(labels, 2):
            key = _pair_key(label_a, label_b)
            if key in existing:
                r = existing[key]
                tag = "ERR" if r.error else f"{r.wins}-{r.losses}-{r.draws}"
                print(f"[panel] skip (done) {label_a} vs {label_b}: {tag}", flush=True)
                continue

            # Skip pairs touching a player that failed to build.
            missing = [lbl for lbl in (label_a, label_b) if lbl in build_errors]
            if missing:
                rec = PairRecord(
                    label_a, label_b, n_games,
                    error=f"build-failed: {', '.join(missing)}",
                )
                append_record(out_path, rec)
                all_records.append(rec)
                print(f"[panel] skip (build-failed {missing}) {label_a} vs {label_b}",
                      flush=True)
                continue

            seed = _pair_seed(label_a, label_b, base_seed)
            print(f"[panel] === {label_a} vs {label_b} "
                  f"(n={n_games}, seed={seed}, open={random_opening_moves}) ===",
                  flush=True)
            try:
                t0 = time.time()
                res = play_match_pickers(
                    players[label_a], players[label_b],
                    n_games=n_games, seed=seed,
                    random_opening_moves=random_opening_moves,
                )
                dt = time.time() - t0
                rec = PairRecord(
                    label_a=label_a, label_b=label_b, n_games=n_games,
                    wins=res.wins, losses=res.losses, draws=res.draws,
                    win_rate=res.win_rate,
                    black_w=res.black_w, black_l=res.black_l, black_d=res.black_d,
                    white_w=res.white_w, white_l=res.white_l, white_d=res.white_d,
                    seconds=dt,
                )
                print(
                    f"[panel] {label_a} vs {label_b}: "
                    f"{res.wins}W-{res.losses}L-{res.draws}D "
                    f"(win {res.win_rate:.1%}) | "
                    f"black {res.black_w}-{res.black_l}-{res.black_d} | "
                    f"white {res.white_w}-{res.white_l}-{res.white_d} | "
                    f"{dt/60:.1f}min",
                    flush=True,
                )
            except Exception as e:  # engine error / timeout / illegal move
                rec = PairRecord(
                    label_a, label_b, n_games, error=f"{type(e).__name__}: {e}"[:300]
                )
                print(f"[panel] {label_a} vs {label_b}: ERROR {rec.error}", flush=True)
            append_record(out_path, rec)
            all_records.append(rec)
    finally:
        close_all_players(players)

    return all_records


# ----------------------------------------------------------------------------
# Cross-table aggregation
# ----------------------------------------------------------------------------
@dataclass
class Aggregate:
    """Per-player and pairwise score tallies for the Elo fit + cross-table.

    ``score[(i, j)]`` is i's score vs j (wins + 0.5*draws); ``games[(i, j)]`` is
    the games i and j played against each other (same for both orderings).
    """

    labels: list[str]
    score: dict[tuple[str, str], float]
    games: dict[tuple[str, str], int]
    total_w: dict[str, int]
    total_l: dict[str, int]
    total_d: dict[str, int]


def build_aggregate(records: list[PairRecord]) -> Aggregate:
    """Fold valid (non-error) pair records into directional score tallies.

    ``play_match_pickers`` reports W/L/D from A's perspective; B's mirror is the
    complement (B's wins = A's losses, etc.). A draw counts as half a win for
    BOTH sides. Error pairs and zero-game pairs contribute nothing.
    """
    labels: list[str] = []
    seen: set[str] = set()
    for r in records:
        for lbl in (r.label_a, r.label_b):
            if lbl not in seen:
                seen.add(lbl)
                labels.append(lbl)

    score: dict[tuple[str, str], float] = {}
    games: dict[tuple[str, str], int] = {}
    total_w = {lbl: 0 for lbl in labels}
    total_l = {lbl: 0 for lbl in labels}
    total_d = {lbl: 0 for lbl in labels}

    for r in records:
        if r.error or not r.n_games or r.wins is None:
            continue
        a, b = r.label_a, r.label_b
        w, l_, d = r.wins, r.losses or 0, r.draws or 0
        # A's score vs B and the mirror for B vs A.
        score[(a, b)] = score.get((a, b), 0.0) + w + 0.5 * d
        score[(b, a)] = score.get((b, a), 0.0) + l_ + 0.5 * d
        games[(a, b)] = games.get((a, b), 0) + r.n_games
        games[(b, a)] = games.get((b, a), 0) + r.n_games
        total_w[a] += w; total_l[a] += l_; total_d[a] += d
        total_w[b] += l_; total_l[b] += w; total_d[b] += d

    return Aggregate(labels, score, games, total_w, total_l, total_d)


# ----------------------------------------------------------------------------
# Bradley-Terry / logistic Elo fit (MM with L2 regularization)
# ----------------------------------------------------------------------------
def fit_bradley_terry(
    agg: Aggregate,
    *,
    l2: float = 0.30,
    iters: int = 1000,
    tol: float = 1e-7,
) -> dict[str, float]:
    """Fit logistic (Elo-scale) ratings from the pairwise score matrix.

    Model: P(i beats j) = gamma_i / (gamma_i + gamma_j) with gamma = 10**(R/400).
    Ratings R are found by Minorization-Maximization, the standard derivative-
    free Bradley-Terry solver, which iterates

        gamma_i <- (W_i + 2*l2*gmean) /
                   (sum_j n_ij/(gamma_i + gamma_j) + 2*l2/gmean)

    where W_i is i's total score and n_ij the games i played vs j. The L2 terms
    pull each gamma gently toward the geometric-mean field strength ``gmean``,
    which keeps an undefeated or winless player finite (a perfect record would
    otherwise send its rating to +/-inf). Ratings are returned on the 400/log10
    Elo scale, mean-centered to 0 (the BT zero point is arbitrary; the affine
    anchor calibration fixes the absolute level afterward).

    Players with zero recorded games are dropped (no information to rate).
    """
    rated = [lbl for lbl in agg.labels
             if any(agg.games.get((lbl, o), 0) > 0 for o in agg.labels)]
    if not rated:
        return {}

    # gamma_i = 10**(R_i/400); start everyone at parity (R=0 -> gamma=1).
    gamma = {lbl: 1.0 for lbl in rated}
    # i's total score across all opponents.
    won = {
        lbl: sum(agg.score.get((lbl, o), 0.0) for o in rated)
        for lbl in rated
    }

    for _ in range(iters):
        gmean = math.exp(sum(math.log(max(gamma[l_], 1e-12)) for l_ in rated)
                         / len(rated))
        new_gamma: dict[str, float] = {}
        max_delta = 0.0
        for i in rated:
            denom = 2.0 * l2 / gmean
            for j in rated:
                n_ij = agg.games.get((i, j), 0)
                if n_ij <= 0:
                    continue
                denom += n_ij / (gamma[i] + gamma[j])
            numer = won[i] + 2.0 * l2 * gmean
            gi = numer / denom if denom > 0 else gamma[i]
            new_gamma[i] = gi
            max_delta = max(max_delta, abs(gi - gamma[i]))
        gamma = new_gamma
        if max_delta < tol:
            break

    # gamma -> Elo, then mean-center.
    ratings = {lbl: _ELO_SCALE * math.log10(max(g, 1e-12)) for lbl, g in gamma.items()}
    mean_r = sum(ratings.values()) / len(ratings)
    return {lbl: r - mean_r for lbl, r in ratings.items()}


# ----------------------------------------------------------------------------
# Affine calibration to the Gomocup anchors
# ----------------------------------------------------------------------------
@dataclass
class Calibration:
    """The affine map internal_rating -> Gomocup-Elo and its provenance.

    ``calibrated`` is None when fewer than 2 anchors produced valid ratings; in
    that case callers report RELATIVE (uncentered-internal) Elo and say so.
    """

    slope: Optional[float]
    intercept: Optional[float]
    anchors_used: list[str]
    calibrated: Optional[dict[str, float]]  # label -> Gomocup-scale Elo
    note: str


def affine_calibrate(
    internal: dict[str, float],
    anchors: dict[str, float],
) -> Calibration:
    """Least-squares fit a line internal_rating -> known-Gomocup-Elo on anchors.

    Uses ONLY players that are BOTH in ``anchors`` and in the fitted ``internal``
    ratings. Ordinary least squares on (x=internal_rating, y=gomocup_elo):

        slope = cov(x, y) / var(x),   intercept = mean(y) - slope*mean(x)

    then ``calibrated[label] = slope * internal[label] + intercept`` for ALL
    players. This is the heart of #30: a fixed panel with published ratings lets
    us project our internal ruler onto the absolute Gomocup ladder, so our nets
    and the heuristic get a calibrated absolute Elo, not just a relative rank.

    Fallbacks (calibrated=None, report relative Elo + flag):
      * <2 usable anchors  -> line under-determined.
      * zero internal-rating variance among anchors -> degenerate (vertical).
    """
    pts = [(internal[a], anchors[a], a) for a in anchors if a in internal]
    if len(pts) < 2:
        return Calibration(
            None, None, [p[2] for p in pts], None,
            note=(f"UNCALIBRATED: only {len(pts)} valid anchor(s) "
                  "(need >=2) — reporting RELATIVE internal Elo."),
        )
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    used = [p[2] for p in pts]
    n = len(pts)
    mx = sum(xs) / n
    my = sum(ys) / n
    var_x = sum((x - mx) ** 2 for x in xs)
    if var_x <= 1e-9:
        return Calibration(
            None, None, used, None,
            note=("UNCALIBRATED: anchor internal-ratings have ~zero variance "
                  "(degenerate fit) — reporting RELATIVE internal Elo."),
        )
    cov_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = my - slope * mx
    calibrated = {lbl: slope * r + intercept for lbl, r in internal.items()}
    return Calibration(
        slope, intercept, used, calibrated,
        note=(f"CALIBRATED via {n} anchors {used}: "
              f"Gomocup_Elo = {slope:.2f}*internal + {intercept:.1f}"),
    )


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def print_cross_table(agg: Aggregate) -> None:
    """Print the win-rate matrix: cell (row=A, col=B) is A's score rate vs B.

    A's score rate is (A's score vs B) / (games A played vs B); the diagonal is
    '-', and an unplayed/errored pair shows '.'.
    """
    labels = agg.labels
    if not labels:
        print("(no players)")
        return
    w = max(max((len(x) for x in labels), default=0), 7)
    print("\n=== CROSS-TABLE (row A's score-rate vs column B; draw=0.5) ===")
    print(" " * (w + 2) + " ".join(f"{x[:w]:>{w}}" for x in labels))
    for a in labels:
        cells = [f"{a:>{w}}: "]
        for b in labels:
            if a == b:
                cells.append(f"{'-':>{w}}")
            elif agg.games.get((a, b), 0) > 0:
                rate = agg.score[(a, b)] / agg.games[(a, b)]
                cells.append(f"{rate:>{w}.0%}")
            else:
                cells.append(f"{'.':>{w}}")
        print(" ".join(cells))


def print_rankings(
    agg: Aggregate,
    cal: Calibration,
    internal: dict[str, float],
    anchors: dict[str, float],
) -> None:
    """Print the ranked Elo table (calibrated if possible, else relative)."""
    use_cal = cal.calibrated is not None
    ratings = cal.calibrated if use_cal else internal
    scale = "CALIBRATED (Gomocup scale)" if use_cal else "RELATIVE (internal, mean=0)"
    print(f"\n=== RANKED ELO — {scale} ===")
    print(f"    {cal.note}")
    rows = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
    hdr = f"  {'player':>16} {'elo':>8}  {'kind':>10}  record (W-L-D)"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lbl, elo in rows:
        if lbl in anchors and lbl in cal.anchors_used:
            kind = "ANCHOR"
        elif lbl in anchors:
            kind = "anchor*"   # known engine, but not used in the fit
        else:
            kind = "estimate"
        rec = f"{agg.total_w.get(lbl, 0)}-{agg.total_l.get(lbl, 0)}-{agg.total_d.get(lbl, 0)}"
        anchor_note = ""
        if lbl in anchors:
            anchor_note = f"  (published ~{anchors[lbl]:.0f})"
        print(f"  {lbl:>16} {elo:>8.0f}  {kind:>10}  {rec}{anchor_note}")
    print("\n  * 'anchor*' = a known engine excluded from the fit (no valid rating).")
    print("  Anchor Elos are APPROXIMATE published Gomocup freestyle values "
          "(see GOMOCUP_ELO_ANCHORS).")


def report(records: list[PairRecord]) -> None:
    """Full end-of-tournament report: cross-table + calibrated rankings."""
    agg = build_aggregate(records)
    internal = fit_bradley_terry(agg)
    cal = affine_calibrate(internal, GOMOCUP_ELO_ANCHORS)
    print_cross_table(agg)
    print_rankings(agg, cal, internal, GOMOCUP_ELO_ANCHORS)

    n_err = sum(1 for r in records if r.error)
    n_played = sum(1 for r in records if not r.error and r.wins is not None)
    print(f"\n[panel] pairs: {n_played} played, {n_err} errored/skipped.")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Engine-panel-anchored round-robin tournament (#30): a "
                    "calibrated strength yardstick.",
    )
    p.add_argument("--n-games", type=int, default=8,
                   help="Games per pair (default 8; keep even for color balance).")
    p.add_argument("--out", default=os.path.expanduser(
        "~/code/gomoku/sweep_runs/panel_tournament_results.jsonl"),
        help="JSONL results path (resumable; pairs already present are skipped).")
    p.add_argument("--only", default=None,
                   help="Comma-separated labels to restrict the field "
                        "(e.g. 'az-champ-128x10,embryo26,heuristic').")
    p.add_argument("--sims", type=int, default=_DEFAULT_SIMS,
                   help=f"Override net MCTS sims (default {_DEFAULT_SIMS}).")
    p.add_argument("--timeout-ms", type=int, default=_DEFAULT_TIMEOUT_MS,
                   help=f"Per-move budget for external engines "
                        f"(default {_DEFAULT_TIMEOUT_MS}).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed; each pair derives a deterministic seed.")
    p.add_argument("--random-opening-moves", type=int, default=4,
                   help="Shared random opening stones per color-swap pair "
                        "(default 4; neutralizes first-mover advantage, #22).")
    p.add_argument("--print-table", action="store_true",
                   help="Recompute the cross-table + calibrated Elo from an "
                        "existing JSONL without playing any games.")
    p.add_argument("--wine", action="store_true",
                   help="Opt IN to the flaky wine-run Gomocup engines (OFF by "
                        "default; issue #35 — 17/36 panel pairs crashed). Also "
                        "enableable via GOMOKU_ENABLE_WINE_ENGINES=1. Without "
                        "this the field is nets + heuristic + any native engine.")
    return p.parse_args()


def select_field(
    participants: list[tuple[str, str]], only: Optional[str]
) -> list[tuple[str, str]]:
    """Restrict the field to ``--only`` labels (order preserved), if given."""
    if not only:
        return participants
    wanted = [s.strip() for s in only.split(",") if s.strip()]
    by_label = {lbl: spec for lbl, spec in participants}
    missing = [w for w in wanted if w not in by_label]
    if missing:
        raise SystemExit(f"--only: unknown labels {missing}. "
                         f"Known: {sorted(by_label)}")
    return [(w, by_label[w]) for w in wanted]


def main() -> None:
    args = parse_args()

    if args.print_table:
        # Pure report mode — no players built, no games played.
        records = list(load_existing(args.out).values())
        if not records:
            raise SystemExit(f"--print-table: no records in {args.out}")
        report(records)
        return

    enable_wine = wine_engines_enabled(getattr(args, "wine", False))
    participants = _build_participants(
        sims=args.sims, timeout_ms=args.timeout_ms, enable_wine=enable_wine
    )
    participants = select_field(participants, args.only)
    if enable_wine:
        print("[panel] WINE ENGINES ENABLED (opt-in) — expect flaky pairs (#35).",
              flush=True)
    else:
        print("[panel] reliable panel: nets + heuristic + native engines only "
              "(wine OFF; --wine or GOMOKU_ENABLE_WINE_ENGINES=1 to opt in, #35).",
              flush=True)
    print(f"[panel] field ({len(participants)}): "
          f"{', '.join(lbl for lbl, _ in participants)}", flush=True)
    print(f"[panel] n_games={args.n_games} sims={args.sims} "
          f"timeout_ms={args.timeout_ms} open={args.random_opening_moves} "
          f"out={args.out}", flush=True)

    records = run_round_robin(
        participants,
        n_games=args.n_games,
        base_seed=args.seed,
        random_opening_moves=args.random_opening_moves,
        out_path=args.out,
    )
    report(records)
    print("\n[panel] DONE", flush=True)


if __name__ == "__main__":
    main()
