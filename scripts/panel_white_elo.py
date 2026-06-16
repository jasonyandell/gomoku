"""PURE-ANALYSIS white-side (defense) reader over the panel tournament JSONL.

GitHub issue #33 / #18 — quantify our nets' WHITE-side (second-player / defense)
weakness from the data the engine-panel round-robin already records on disk. No
games, no GPU, no torch — this only READS the JSONL and does arithmetic + two
Bradley-Terry fits. Sibling instrument to ``scripts/panel_tournament.py`` (#30),
whose pure helpers it reuses (DRY): ``PairRecord``, ``load_existing``,
``build_aggregate``, ``fit_bradley_terry``, ``affine_calibrate``,
``GOMOCUP_ELO_ANCHORS``. The design rationale is in
``wiki/topics/white-side-defense-plan.md`` §1A and §5.

======================================================================
EXACT JSONL SEMANTICS (the perspective question — confirmed in source)
======================================================================
Each line is one ``PairRecord`` for an unordered pairing (A=label_a, B=label_b).
``play_match_pickers`` returns the result FROM A's PERSPECTIVE
(``gomoku/eval.py`` ``MatchResult`` docstring + ``_match_result_from_outcomes``:
"A game is counted under 'black' if the model [player A] played black that
game, else 'white'"). Therefore, for a pair (A, B):

  * ``black = [black_w, black_l, black_d]``  = A's outcomes WHEN A PLAYED BLACK
        => A is the ATTACKER / first player (B defends).
  * ``white = [white_w, white_l, white_d]``  = A's outcomes WHEN A PLAYED WHITE
        => A is the DEFENDER / second player (B attacks).
  All three are wins/losses/draws COUNTED FROM A's side.

So A's ``white`` array IS "A defending while B attacks" — the defense metric.
To recover B's OWN white-side games vs A (B defending while A attacks), we
INVERT A's ``black`` array (the games where A attacked as black are exactly the
games where B defended as white):

      B-as-white vs A:  w = A.black_l,  l = A.black_w,  d = A.black_d
      B-as-black vs A:  w = A.white_l,  l = A.white_w,  d = A.white_d

This inversion is how a net that only ever appears as ``label_b`` (e.g. az-96x8
in the live partial) still gets a full white/black row.

======================================================================
WHAT THIS PRINTS (in order of trust)
======================================================================
1. PRIMARY — unambiguous per-color RATE table (the ground truth):
     white_score_rate = (white_w + 0.5*white_d) / white_games   (defense; higher better)
     white_loss_rate  =  white_l               / white_games    (#18 "never lose
                                                                  as white"; lower better)
     black_score_rate = (black_w + 0.5*black_d) / black_games    (attack; higher better)
   Per (our net, opponent) and aggregated over opponents. Computed DIRECTLY from
   the per-color arrays — no model, no inference.

2. SECONDARY — dual calibrated Elo (DERIVED, directed Bradley-Terry). A BLACK
   /attack rating from the directed black-side score matrix and a WHITE/defense
   rating from the directed white-side score matrix, each mean-centered, then
   BOTH passed through the SAME affine map (slope, intercept) the AGGREGATE
   panel fit produces against the published engine anchors — so black_elo,
   white_elo and elo_gap = black_elo - white_elo land on ONE calibrated scale.
   Clearly a derived estimate; §1 rates are the truth.

EVAL-ONLY ANALYSIS. Reads the JSONL line-by-line; skips malformed / partial /
``error`` records; handles a file that is still being appended to.

NOTE (wine engines shelved, 2026-06-16; issue #35): the wine-run Gomocup engines
are OFF by default in ``panel_tournament.py`` (they crashed ~17/36 panel pairs).
This reader is unaffected — it consumes whatever players appear in the JSONL, so
it reports correctly over the reliable default field (our nets + heuristic + any
native engine). The ``GOMOCUP_ELO_ANCHORS`` it imports are still the calibration
ruler IF a run opted wine engines back in; with the reliable-only default there
may be <2 anchors, in which case the §2 calibration honestly reports RELATIVE Elo
(the §1 per-color RATES remain the ground truth either way). See
``wiki/topics/reliable-eval-set.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

# Reuse the panel script's PURE helpers (DRY). Importing it is torch-free: its
# heavy ``gomoku.eval`` / ``gomoku.match`` imports are lazy (inside functions),
# and its module body only does ``os.environ.setdefault`` (harmless here). We
# import the BT fit, affine calibration, anchors, the JSONL reader, the record
# type, and the aggregate builder — re-using them verbatim rather than forking.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.panel_tournament import (  # noqa: E402  (after sys.path shim)
    _ELO_SCALE,
    GOMOCUP_ELO_ANCHORS,
    Aggregate,
    PairRecord,
    affine_calibrate,
    build_aggregate,
    fit_bradley_terry,
)

_DEFAULT_IN = os.path.join(_REPO_ROOT, "sweep_runs", "panel_tournament_results.jsonl")


# ----------------------------------------------------------------------------
# §1 PRIMARY — directed per-color tallies straight from the arrays
# ----------------------------------------------------------------------------
@dataclass
class ColorTally:
    """Directed win/loss/draw counts for one (player, opponent, color) cell.

    ``color`` is the perspective player's color: "white" = that player defending
    (second), "black" = that player attacking (first). Counts are always FROM
    the perspective player's side.
    """

    w: int = 0
    l: int = 0
    d: int = 0

    @property
    def games(self) -> int:
        return self.w + self.l + self.d

    def add(self, w: int, l: int, d: int) -> None:
        self.w += w
        self.l += l
        self.d += d

    @property
    def score_rate(self) -> Optional[float]:
        g = self.games
        return None if g == 0 else (self.w + 0.5 * self.d) / g

    @property
    def loss_rate(self) -> Optional[float]:
        g = self.games
        return None if g == 0 else self.l / g


def _iter_records(in_path: str) -> tuple[list[PairRecord], int, int]:
    """Read the JSONL line-by-line; return (valid_records, n_skipped, n_error).

    Robust to a file still being appended to: each line is parsed independently,
    a malformed/partial (e.g. half-written last) line is skipped, and ``error``
    records are excluded from the per-color math (counted separately). Reuses the
    same tolerant ``PairRecord.from_json`` the panel resume path uses.
    """
    records: list[PairRecord] = []
    n_skipped = 0
    n_error = 0
    if not os.path.exists(in_path):
        raise SystemExit(f"--in: no such file {in_path}")
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rec = PairRecord.from_json(d)
            except Exception:
                n_skipped += 1  # malformed / partial last line
                continue
            if rec.error:
                n_error += 1
                continue
            # A genuine (non-error) record must carry the per-color arrays.
            if rec.black_w is None or rec.white_w is None:
                n_skipped += 1
                continue
            records.append(rec)
    return records, n_skipped, n_error


def build_color_tables(
    records: list[PairRecord], net_prefix: str
) -> tuple[
    list[str],
    list[str],
    dict[str, dict[str, ColorTally]],
    dict[str, dict[str, ColorTally]],
]:
    """Build directed white/black per-(net, opponent) tallies for OUR nets.

    Returns (nets, opponents, white_cells, black_cells) where
    ``white_cells[net][opp]`` is the net's WHITE (defense) ColorTally vs ``opp``,
    and ``black_cells[net][opp]`` its BLACK (attack) ColorTally. Every pair
    touching a net contributes that net's perspective; if the net is ``label_b``
    its arrays are recovered by inverting ``label_a``'s arrays (see module
    docstring). A net-vs-net pair contributes to BOTH nets' rows (each defends in
    turn), with the opponent column being the other net.
    """
    nets: list[str] = []
    opponents: list[str] = []
    white_cells: dict[str, dict[str, ColorTally]] = {}
    black_cells: dict[str, dict[str, ColorTally]] = {}

    def _seen(lst: list[str], lbl: str) -> None:
        if lbl not in lst:
            lst.append(lbl)

    def _cell(d: dict[str, dict[str, ColorTally]], net: str, opp: str) -> ColorTally:
        return d.setdefault(net, {}).setdefault(opp, ColorTally())

    def _record_perspective(net: str, opp: str,
                            white_wld: tuple[int, int, int],
                            black_wld: tuple[int, int, int]) -> None:
        _seen(nets, net)
        if not opp.startswith(net_prefix):
            _seen(opponents, opp)
        _cell(white_cells, net, opp).add(*white_wld)
        _cell(black_cells, net, opp).add(*black_wld)

    for r in records:
        a, b = r.label_a, r.label_b
        # A's own arrays (already A's perspective).
        a_white = (r.white_w or 0, r.white_l or 0, r.white_d or 0)
        a_black = (r.black_w or 0, r.black_l or 0, r.black_d or 0)
        # B's arrays = invert A's (the games A attacked as black are the games B
        # defended as white, and vice-versa).
        b_white = (r.black_l or 0, r.black_w or 0, r.black_d or 0)
        b_black = (r.white_l or 0, r.white_w or 0, r.white_d or 0)

        if a.startswith(net_prefix):
            _record_perspective(a, b, a_white, a_black)
        if b.startswith(net_prefix):
            _record_perspective(b, a, b_white, b_black)

    return nets, opponents, white_cells, black_cells


def _agg_over_opponents(
    cells: dict[str, ColorTally], skip_nets: set[str]
) -> ColorTally:
    """Sum a net's per-opponent tallies into one aggregate (over all opponents).

    ``skip_nets`` excludes our-own-net columns so the aggregate-vs-OPPONENTS row
    is over external opponents only (net-vs-net cells are still shown in the
    table but kept out of the external aggregate for cleanliness)."""
    out = ColorTally()
    for opp, t in cells.items():
        if opp in skip_nets:
            continue
        out.add(t.w, t.l, t.d)
    return out


# ----------------------------------------------------------------------------
# §2 SECONDARY — directed Bradley-Terry (black/attack & white/defense), one scale
# ----------------------------------------------------------------------------
def _directed_aggregate_from_color(
    records: list[PairRecord], color: str
) -> Aggregate:
    """Build an ``Aggregate`` whose score[(defender, attacker)] is the DEFENDER's
    directed score in that color, so ``fit_bradley_terry`` rates that color.

    For ``color == 'white'`` (DEFENSE): each pair contributes A-defending-vs-B
    (A's white array, indexed score[(A, B)]) AND B-defending-vs-A (the inverted
    black array, indexed score[(B, A)]). The resulting rating is a WHITE/defense
    rating: a high rating == defends well.

    For ``color == 'black'`` (ATTACK): symmetric — A-attacking-vs-B (A's black
    array) and B-attacking-vs-A (inverted white array). High rating == attacks
    well. We reuse the panel's exact ``Aggregate`` shape so its ``fit_bradley_
    terry`` solver applies unchanged.
    """
    labels: list[str] = []
    seen: set[str] = set()
    score: dict[tuple[str, str], float] = {}
    games: dict[tuple[str, str], int] = {}
    total_w: dict[str, int] = {}
    total_l: dict[str, int] = {}
    total_d: dict[str, int] = {}

    def _note(lbl: str) -> None:
        if lbl not in seen:
            seen.add(lbl)
            labels.append(lbl)
            total_w[lbl] = total_l[lbl] = total_d[lbl] = 0

    def _add(perspective: str, other: str, w: int, l: int, d: int) -> None:
        _note(perspective)
        _note(other)
        n = w + l + d
        if n == 0:
            return
        score[(perspective, other)] = score.get((perspective, other), 0.0) + w + 0.5 * d
        games[(perspective, other)] = games.get((perspective, other), 0) + n
        total_w[perspective] += w
        total_l[perspective] += l
        total_d[perspective] += d

    for r in records:
        a, b = r.label_a, r.label_b
        aw = (r.white_w or 0, r.white_l or 0, r.white_d or 0)
        ab = (r.black_w or 0, r.black_l or 0, r.black_d or 0)
        # inverted for B's perspective
        bw = (r.black_l or 0, r.black_w or 0, r.black_d or 0)
        bb = (r.white_l or 0, r.white_w or 0, r.white_d or 0)
        if color == "white":
            _add(a, b, *aw)   # A defends vs B
            _add(b, a, *bw)   # B defends vs A
        elif color == "black":
            _add(a, b, *ab)   # A attacks vs B
            _add(b, a, *bb)   # B attacks vs A
        else:  # pragma: no cover
            raise ValueError(color)

    return Aggregate(labels, score, games, total_w, total_l, total_d)


def _all_finite(d: dict[str, float]) -> bool:
    return all(math.isfinite(v) for v in d.values()) and len(d) > 0


def _stable_bradley_terry(
    agg: Aggregate, *, l2: float = 0.30, iters: int = 2000, tol: float = 1e-9
) -> dict[str, float]:
    """Scale-normalized BT-MM — a numerically-stable twin of the panel's fit.

    The panel's ``fit_bradley_terry`` (reused for the aggregate) is tuned for a
    DENSE round-robin where the gamma scale is naturally pinned. On a SPARSE /
    near-STAR partial cross-table (e.g. only one net has been played, so every
    external engine has a single opponent) its MM update lets the whole gamma
    field drift up by a common factor each iteration — it overflows to inf, and
    mean-centering inf yields NaN (observed on the live partial's black/attack
    matrix). This twin adds the standard fix: RE-NORMALIZE the geometric mean of
    gamma to 1 after every iteration (BT ratings are only defined up to a global
    scale, so this changes nothing in the well-posed case and merely keeps the
    sparse case finite). Used ONLY as a fallback when the panel fit returns
    non-finite ratings, and the substitution is reported in the note.
    """
    rated = [lbl for lbl in agg.labels
             if any(agg.games.get((lbl, o), 0) > 0 for o in agg.labels)]
    if not rated:
        return {}
    gamma = {lbl: 1.0 for lbl in rated}
    won = {lbl: sum(agg.score.get((lbl, o), 0.0) for o in rated) for lbl in rated}
    for _ in range(iters):
        gmean = math.exp(sum(math.log(max(gamma[l_], 1e-12)) for l_ in rated)
                         / len(rated))
        new_gamma: dict[str, float] = {}
        for i in rated:
            denom = 2.0 * l2 / gmean
            for j in rated:
                n_ij = agg.games.get((i, j), 0)
                if n_ij <= 0:
                    continue
                denom += n_ij / (gamma[i] + gamma[j])
            numer = won[i] + 2.0 * l2 * gmean
            new_gamma[i] = numer / denom if denom > 0 else gamma[i]
        # Re-pin the global scale: divide through by the new geometric mean.
        g2 = math.exp(sum(math.log(max(new_gamma[l_], 1e-12)) for l_ in rated)
                      / len(rated))
        for i in rated:
            new_gamma[i] = new_gamma[i] / g2
        max_delta = max(abs(new_gamma[i] - gamma[i]) for i in rated)
        gamma = new_gamma
        if max_delta < tol:
            break
    ratings = {lbl: _ELO_SCALE * math.log10(max(g, 1e-12)) for lbl, g in gamma.items()}
    mean_r = sum(ratings.values()) / len(ratings)
    return {lbl: r - mean_r for lbl, r in ratings.items()}


def _fit_directed(agg: Aggregate) -> tuple[dict[str, float], bool]:
    """Fit a directed BT rating, preferring the panel's reused solver; fall back
    to the scale-normalized twin when the panel solver returns non-finite values.

    Returns (ratings, used_stable_fallback).
    """
    primary = fit_bradley_terry(agg)
    if _all_finite(primary):
        return primary, False
    return _stable_bradley_terry(agg), True


@dataclass
class DualElo:
    black_elo: dict[str, float]
    white_elo: dict[str, float]
    elo_gap: dict[str, float]
    calibrated: bool
    note: str
    anchors_used: list[str]
    slope: Optional[float]
    intercept: Optional[float]
    used_stable_fallback: bool = False


def compute_dual_elo(records: list[PairRecord]) -> DualElo:
    """Two directed BT fits (black/attack, white/defense), both mean-centered,
    then mapped through ONE affine calibration (the AGGREGATE fit's anchor line)
    so both ratings share the calibrated Gomocup scale.

    Rationale: the affine map is the panel's own anchor calibration — fitting the
    line on the AGGREGATE (color-blind) ratings vs the published anchors, then
    applying that SAME (slope, intercept) to the black and white ratings. This
    keeps black_elo and white_elo on a common, anchor-pinned ruler so that
    ``elo_gap = black_elo - white_elo`` is a meaningful Elo deficit.

    HONESTY GUARDS (this is the DERIVED, lower-trust section):
      * If <2 usable anchors, we report RELATIVE (mean-0) and say so.
      * If the anchor calibration is DEGENERATE — a non-positive slope, i.e. the
        internal ruler runs BACKWARDS or flat vs the published anchors (which
        happens on a near-star partial where the engines are barely
        differentiated) — calibrating to Gomocup scale would be misleading, so
        we ALSO fall back to RELATIVE and flag it loudly. The §1 rates remain the
        ground truth regardless.
    """
    # Aggregate (color-blind) fit + anchor line — same as the panel's report().
    agg_all = build_aggregate(records)
    internal_all = fit_bradley_terry(agg_all)
    if not _all_finite(internal_all):
        internal_all = _stable_bradley_terry(agg_all)
    cal = affine_calibrate(internal_all, GOMOCUP_ELO_ANCHORS)

    # Directed fits (already mean-centered).
    black_internal, fb1 = _fit_directed(_directed_aggregate_from_color(records, "black"))
    white_internal, fb2 = _fit_directed(_directed_aggregate_from_color(records, "white"))
    used_stable = fb1 or fb2

    # A usable calibration needs a finite, strictly-positive slope (monotone
    # internal->published map). A <=0 slope means the anchors are non-monotone in
    # internal rating -> the absolute scale is not trustworthy yet.
    slope_ok = (cal.calibrated is not None and cal.slope is not None
                and math.isfinite(cal.slope) and cal.slope > 1e-6)

    if slope_ok:
        slope, intercept = cal.slope, cal.intercept
        black_elo = {k: slope * v + intercept for k, v in black_internal.items()}
        white_elo = {k: slope * v + intercept for k, v in white_internal.items()}
        calibrated = True
        note = (f"CALIBRATED via {len(cal.anchors_used)} anchors {cal.anchors_used}: "
                f"applied the AGGREGATE anchor line Gomocup_Elo = "
                f"{slope:.2f}*internal + {intercept:.1f} to BOTH directed fits.")
    else:
        black_elo = dict(black_internal)
        white_elo = dict(white_internal)
        calibrated = False
        if cal.calibrated is None:
            why = cal.note
        else:
            why = (f"DEGENERATE calibration (slope={cal.slope:.3f} <= 0 over anchors "
                   f"{cal.anchors_used}): the internal ruler is non-monotone vs the "
                   "published anchors on this partial/near-star cross-table, so the "
                   "absolute Gomocup scale is NOT trustworthy yet.")
        note = ("RELATIVE (mean-0, UNCALIBRATED): " + why +
                " black_elo/white_elo are directed BT ratings on the internal "
                "scale, NOT Gomocup-anchored. (§1 rates are still the ground truth.)")

    if used_stable:
        note += (" [directed fit used the scale-normalized stable BT twin because "
                 "the panel solver returned non-finite ratings on this sparse data.]")

    elo_gap = {
        k: black_elo[k] - white_elo[k]
        for k in set(black_elo) & set(white_elo)
    }
    return DualElo(
        black_elo=black_elo, white_elo=white_elo, elo_gap=elo_gap,
        calibrated=calibrated, note=note,
        anchors_used=cal.anchors_used, slope=cal.slope, intercept=cal.intercept,
        used_stable_fallback=used_stable,
    )


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _fmt_rate(v: Optional[float]) -> str:
    return "  .  " if v is None else f"{v:5.0%}"


def print_primary(
    nets: list[str],
    opponents: list[str],
    white_cells: dict[str, dict[str, ColorTally]],
    black_cells: dict[str, dict[str, ColorTally]],
    net_prefix: str,
) -> None:
    net_set = set(nets)
    # Column order: external opponents first (in first-seen order), then our
    # other nets, de-duplicated.
    cols = list(opponents) + sorted(net_set)
    seen: set[str] = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]

    print("=" * 78)
    print("§1  PRIMARY — per-color RATE table (GROUND TRUTH, computed directly")
    print("    from the per-color arrays; NO model/inference). Higher score-rate")
    print("    is better; lower white_loss_rate is better. 'white' = OUR net")
    print("    DEFENDING (second player) while the opponent attacks.")
    print("=" * 78)

    wlabel = max(max((len(n) for n in nets), default=8), 12)
    cw = 8

    def _row(label: str, getter) -> str:
        parts = [f"{label:>{wlabel}} |"]
        for opp in cols:
            parts.append(f"{getter(opp):>{cw}}")
        return " ".join(parts)

    def _header() -> None:
        head = [f"{'net':>{wlabel}} |"]
        for opp in cols:
            tag = opp if not opp.startswith(net_prefix) else opp + "*"
            head.append(f"{tag[:cw]:>{cw}}")
        head.append(f"{'AGG-vs-OPP':>{cw + 2}}")
        print(" ".join(head))
        print("-" * (wlabel + 3 + (cw + 1) * (len(cols) + 1)))

    # --- View A: white_loss_rate (Jason's "never lose as white"; lower better)
    print("\n--- white_loss_rate  (white_l / white_games; LOWER is better) ---")
    _header()
    for net in nets:
        wc = white_cells.get(net, {})

        def get(opp: str, wc=wc) -> str:
            t = wc.get(opp)
            return _fmt_rate(t.loss_rate) if t else "  .  "

        agg = _agg_over_opponents(wc, net_set)
        print(_row(net, get) + f"  {_fmt_rate(agg.loss_rate):>{cw}}")

    # --- View B: white vs black score-rate, aggregated over opponents
    print("\n--- white_score_rate vs black_score_rate (AGG over EXTERNAL opponents) ---")
    print(f"{'net':>{wlabel}} | {'white_score':>12} {'black_score':>12} "
          f"{'white_loss':>11} {'gap(blk-wht)':>13} {'wht_games':>10}")
    print("-" * (wlabel + 3 + 12 + 13 + 12 + 14 + 11))
    for net in nets:
        wagg = _agg_over_opponents(white_cells.get(net, {}), net_set)
        bagg = _agg_over_opponents(black_cells.get(net, {}), net_set)
        ws = wagg.score_rate
        bs = bagg.score_rate
        gap = (bs - ws) if (ws is not None and bs is not None) else None
        print(f"{net:>{wlabel}} | {_fmt_rate(ws):>12} {_fmt_rate(bs):>12} "
              f"{_fmt_rate(wagg.loss_rate):>11} "
              f"{('  .  ' if gap is None else f'{gap:+5.0%}'):>13} "
              f"{wagg.games:>10}")

    print("\n  (* = column is one of OUR nets, not an external opponent; net-vs-net")
    print("   cells are shown but EXCLUDED from AGG-vs-OPP / the external aggregate.)")


def print_sanity(
    nets: list[str],
    white_cells: dict[str, dict[str, ColorTally]],
    black_cells: dict[str, dict[str, ColorTally]],
    net_prefix: str,
) -> None:
    """Loud sanity check: champion white_score_rate should be visibly LOWER than
    its black_score_rate (the live data shows the champion ~0-3 as white). If the
    data CONTRADICTS the expected white<black collapse, SAY SO rather than
    massaging it.
    """
    net_set = set(nets)
    print("\n" + "=" * 78)
    print("SANITY CHECK — does the data show the expected white(defense) < black")
    print("(attack) collapse?  (We assert nothing; we report what the rates say.)")
    print("=" * 78)
    any_net = False
    contradictions = 0
    for net in nets:
        wagg = _agg_over_opponents(white_cells.get(net, {}), net_set)
        bagg = _agg_over_opponents(black_cells.get(net, {}), net_set)
        ws, bs = wagg.score_rate, bagg.score_rate
        if ws is None or bs is None:
            print(f"  {net}: insufficient data (white_games={wagg.games}, "
                  f"black_games={bagg.games}) — cannot check.")
            continue
        any_net = True
        verdict = "OK (white < black, defense weaker — expected)"
        if ws >= bs:
            verdict = ("!! CONTRADICTS expectation: white_score >= black_score "
                       "(defense NOT weaker here) — reported, NOT massaged.")
            contradictions += 1
        print(f"  {net}: white_score={ws:.0%} (loss {wagg.loss_rate:.0%}, "
              f"n={wagg.games})  vs  black_score={bs:.0%} (n={bagg.games})  "
              f"-> {verdict}")
    if any_net and contradictions == 0:
        print("\n  VERDICT: confirmed — our nets score LOWER as white (defense) than")
        print("  as black (attack). The white-side weakness is REAL in this data.")
    elif contradictions:
        print(f"\n  VERDICT: {contradictions} net(s) CONTRADICT the white<black "
              "expectation — flagged loudly above; do NOT assume the collapse.")


def print_secondary(dual: DualElo, nets: list[str]) -> None:
    print("\n" + "=" * 78)
    print("§2  SECONDARY — DUAL CALIBRATED ELO (DERIVED, directed Bradley-Terry).")
    print("    This is an ESTIMATE. The §1 rates above are the ground truth.")
    print("=" * 78)
    print(f"    {dual.note}")
    scale = "Gomocup-calibrated" if dual.calibrated else "RELATIVE internal (mean-0)"
    print(f"    Scale: {scale}.  elo_gap = black_elo - white_elo "
          "(bigger gap = worse defense).")
    # Show OUR nets first (worst defense gap first), then everyone else.
    keys = list(dual.elo_gap.keys())
    net_set = set(nets)
    net_keys = [k for k in keys if k in net_set]
    other_keys = sorted(k for k in keys if k not in net_set)
    ordered = sorted(net_keys, key=lambda k: dual.elo_gap[k], reverse=True) + other_keys
    print(f"\n  {'player':>16} {'black_elo':>10} {'white_elo':>10} "
          f"{'elo_gap':>9}  kind")
    print("  " + "-" * 58)
    for k in ordered:
        kind = "OUR-NET" if k in net_set else (
            "anchor" if k in GOMOCUP_ELO_ANCHORS else "opponent")
        be = dual.black_elo.get(k)
        we = dual.white_elo.get(k)
        gap = dual.elo_gap.get(k)
        print(f"  {k:>16} "
              f"{('  .  ' if be is None else f'{be:10.0f}')} "
              f"{('  .  ' if we is None else f'{we:10.0f}')} "
              f"{('  .  ' if gap is None else f'{gap:+9.0f}')}  {kind}")


def build_json_dump(
    nets, opponents, white_cells, black_cells, dual
):
    net_set = set(nets)

    def cell_dict(cells):
        out = {}
        for net, opps in cells.items():
            out[net] = {}
            for opp, t in opps.items():
                out[net][opp] = {
                    "w": t.w, "l": t.l, "d": t.d, "games": t.games,
                    "score_rate": t.score_rate, "loss_rate": t.loss_rate,
                }
        return out

    agg = {}
    for net in nets:
        wagg = _agg_over_opponents(white_cells.get(net, {}), net_set)
        bagg = _agg_over_opponents(black_cells.get(net, {}), net_set)
        agg[net] = {
            "white_score_rate": wagg.score_rate,
            "white_loss_rate": wagg.loss_rate,
            "black_score_rate": bagg.score_rate,
            "white_games": wagg.games,
            "black_games": bagg.games,
            "black_elo": dual.black_elo.get(net),
            "white_elo": dual.white_elo.get(net),
            "elo_gap": dual.elo_gap.get(net),
        }
    return {
        "nets": nets,
        "opponents": opponents,
        "white_cells": cell_dict(white_cells),
        "black_cells": cell_dict(black_cells),
        "aggregate_vs_opponents": agg,
        "dual_elo": {
            "calibrated": dual.calibrated,
            "note": dual.note,
            "anchors_used": dual.anchors_used,
            "slope": dual.slope,
            "intercept": dual.intercept,
            "black_elo": dual.black_elo,
            "white_elo": dual.white_elo,
            "elo_gap": dual.elo_gap,
        },
    }


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PURE-ANALYSIS white-side (defense) reader over the panel "
                    "tournament JSONL (#33/#18). No games, no GPU, no torch.",
    )
    p.add_argument("--in", dest="in_path", default=_DEFAULT_IN,
                   help="Panel results JSONL (default sweep_runs/"
                        "panel_tournament_results.jsonl).")
    p.add_argument("--nets", default="az-",
                   help="Prefix selecting OUR nets (default 'az-').")
    p.add_argument("--json", action="store_true",
                   help="Also dump the computed numbers as JSON to stdout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records, n_skipped, n_error = _iter_records(args.in_path)

    print(f"[white-elo] read {args.in_path}")
    print(f"[white-elo] {len(records)} valid pair(s), "
          f"{n_error} error record(s) skipped, "
          f"{n_skipped} malformed/partial line(s) skipped.")

    nets, opponents, white_cells, black_cells = build_color_tables(records, args.nets)
    if not nets:
        raise SystemExit(
            f"[white-elo] no nets matched prefix {args.nets!r} in {args.in_path}.")

    print(f"[white-elo] our nets ({len(nets)}): {', '.join(nets)}")
    print(f"[white-elo] external opponents ({len(opponents)}): "
          f"{', '.join(opponents)}\n")

    print_primary(nets, opponents, white_cells, black_cells, args.nets)
    print_sanity(nets, white_cells, black_cells, args.nets)

    dual = compute_dual_elo(records)
    print_secondary(dual, nets)

    if args.json:
        dump = build_json_dump(nets, opponents, white_cells, black_cells, dual)
        print("\n=== JSON DUMP ===")
        print(json.dumps(dump, indent=2))


if __name__ == "__main__":
    main()
