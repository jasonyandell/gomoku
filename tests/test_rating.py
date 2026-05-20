import math

import pytest

from gomoku.rating import (
    EloRating,
    elo_diff_from_winrate,
    expected_score,
    implied_elo,
    score_from_match,
    uscf_k_factor,
)


def test_expected_score_symmetric():
    # Equal ratings -> 50/50
    assert expected_score(1500, 1500) == pytest.approx(0.5, abs=1e-9)
    # +400 Elo -> ~91% favored
    assert expected_score(1900, 1500) == pytest.approx(0.909, abs=1e-3)
    # -400 Elo -> ~9%
    assert expected_score(1500, 1900) == pytest.approx(0.091, abs=1e-3)


def test_elo_diff_from_winrate_roundtrip():
    for w in (0.5, 0.6, 0.75, 0.9, 0.99):
        diff = elo_diff_from_winrate(w)
        # expected_score(R + diff, R) should be ~w
        assert expected_score(diff, 0) == pytest.approx(w, abs=1e-3)


def test_elo_diff_from_winrate_clamps():
    # Boundary values shouldn't crash.
    elo_diff_from_winrate(0.0)
    elo_diff_from_winrate(1.0)


def test_uscf_k_factor_bands():
    assert uscf_k_factor(0) == 32
    assert uscf_k_factor(2099) == 32
    assert uscf_k_factor(2100) == 24
    assert uscf_k_factor(2399) == 24
    assert uscf_k_factor(2400) == 16


def test_elo_update_win_and_loss():
    a = EloRating(1500)
    delta_win = a.update(1500, 1.0)
    # K=32, expected=0.5, delta = 32 * (1 - 0.5) = 16
    assert delta_win == pytest.approx(16.0, abs=0.1)
    assert a.rating == pytest.approx(1516.0, abs=0.1)

    b = EloRating(1500)
    delta_loss = b.update(1500, 0.0)
    assert delta_loss == pytest.approx(-16.0, abs=0.1)


def test_elo_update_draw_is_zero_between_equals():
    a = EloRating(1500)
    delta = a.update(1500, 0.5)
    assert abs(delta) < 1e-9


def test_score_from_match():
    assert score_from_match(10, 5, 5) == (12.5, 20)
    assert score_from_match(0, 20, 0) == (0.0, 20)
    assert score_from_match(20, 0, 0) == (20.0, 20)


def test_implied_elo_single_match_recovers_diff():
    # 75% vs an opponent at 1000 -> Elo ~ 1000 + 191
    e = implied_elo([(1000.0, 15.0, 20)])
    assert e == pytest.approx(1191.0, abs=2.0)


def test_implied_elo_combines_multiple_anchors():
    # Beats baseline at 800 with 95% (-> +509), loses to baseline at 1500 with
    # 30% (-> -147). MLE should find a rating that explains both.
    matches = [
        (800.0, 19.0, 20),   # 95% vs 800
        (1500.0, 6.0, 20),   # 30% vs 1500
    ]
    e = implied_elo(matches)
    # Both anchors should be consistent with model around ~1350.
    # We don't pin to an exact value (the two anchors slightly disagree, as
    # would be expected from real noisy data), just sanity-check the range.
    assert 1200.0 < e < 1450.0


def test_implied_elo_total_dominance_returns_ceiling():
    # 100% across all matches -> off the scale upward
    e = implied_elo([(800.0, 20.0, 20)])
    assert e >= 5000.0  # default upper bound


def test_implied_elo_total_loss_returns_floor():
    e = implied_elo([(800.0, 0.0, 20)])
    assert e <= -2500.0  # default lower bound


def test_implied_elo_zero_games_raises():
    with pytest.raises(ValueError):
        implied_elo([(1000.0, 0.0, 0)])


def test_implied_elo_empty_raises():
    with pytest.raises(ValueError):
        implied_elo([])


def test_implied_elo_consistency_with_expected_score():
    """At the solution, sum of expected scores must equal observed total."""
    matches = [
        (800.0, 18.0, 20),
        (1200.0, 10.0, 20),
        (1500.0, 4.0, 20),
    ]
    e = implied_elo(matches, tol=0.1)
    expected_total = sum(n * expected_score(e, opp) for opp, _, n in matches)
    observed_total = sum(s for _, s, _ in matches)
    assert expected_total == pytest.approx(observed_total, abs=0.1)


def test_baseline_spec_to_anchor_key():
    from gomoku.match import parse_spec
    from gomoku.rating import ANCHOR_ELOS, baseline_spec_to_anchor_key

    assert baseline_spec_to_anchor_key(parse_spec("random")) == "random"
    assert baseline_spec_to_anchor_key(parse_spec("heuristic")) == "heuristic"
    assert baseline_spec_to_anchor_key(parse_spec("lookahead:depth=2")) == "lookahead2"
    assert baseline_spec_to_anchor_key(parse_spec("lookahead:depth=4")) == "lookahead4"
    # Unknown specs should return None rather than crash.
    assert baseline_spec_to_anchor_key(parse_spec("defensive")) is None
    # Every key in ANCHOR_ELOS should be findable by SOME spec name we ship.
    for k in ("random", "heuristic"):
        assert k in ANCHOR_ELOS
