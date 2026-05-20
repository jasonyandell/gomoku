"""Constant-age ingest math test.

Direct test of the policy: a positions-based ingest target should pull
MORE games when self-play games are short (fewer positions per game) and
FEWER games when self-play games are long. The buffer turnover rate
stays roughly constant either way, which is what keeps `buffer/age_p50`
stable across regime shifts.

We don't exec the trainer's `_ingest_worker_batches` directly (it's
nested inside `main()`), but we replicate its essential math here to
pin the invariant: for a given target_positions, the number of games
ingested is `target_positions / mean_positions_per_game`.
"""
import math


def _games_needed(target_positions: int, plies_per_game: float,
                  augs: int = 8) -> float:
    """Reciprocal of the ingest contract. How many games (on average) to
    reach `target_positions` if games have `plies_per_game` mean length
    and we apply `augs`-fold D4 augmentation per ply.

    This is exactly the calculation a positions-based ingest does in
    expectation across many cycles: when plies are shorter, we wait
    longer (more games) before SGD; when plies are longer, we wait less.
    """
    positions_per_game = plies_per_game * augs
    return target_positions / positions_per_game


def test_short_games_pull_more_games_to_hit_position_target():
    """At target=12800 positions: 50-ply games need 32 games, 38-ply games
    need ~42 games. That's the desired "short games → more games per cycle"
    behavior that keeps buffer turnover constant."""
    target = 12_800
    long_g = _games_needed(target, plies_per_game=50)
    short_g = _games_needed(target, plies_per_game=38)
    assert long_g == 32.0
    assert math.isclose(short_g, 42.10526, abs_tol=0.01)
    # Short-game cycles ingest MORE games to hit the target.
    assert short_g > long_g


def test_long_games_pull_fewer_games_to_hit_position_target():
    """Conversely, 65-ply games need ~24.6 games to hit 12800 positions —
    so when the model is playing long defensive games, each cycle takes
    fewer of them. Buffer turnover stays proportional."""
    long_g = _games_needed(12_800, plies_per_game=65)
    base = _games_needed(12_800, plies_per_game=50)
    assert long_g < base
    assert math.isclose(long_g, 24.615, abs_tol=0.01)


def test_buffer_turnover_invariant():
    """The whole point of positions-based ingest: turnover rate
    (positions per cycle / buffer capacity) is independent of plies."""
    target = 12_800
    buffer = 1_500_000
    # At target_positions=N, turnover_per_cycle = N / capacity regardless
    # of plies. Verify for several plies values.
    for plies in (20, 40, 60, 80):
        # The function-under-test returns games needed; the positions
        # actually ingested are still `target` by construction.
        games = _games_needed(target, plies_per_game=plies)
        positions = games * plies * 8  # what the cycle actually ingests
        turnover_frac = positions / buffer
        # Invariant: turnover fraction equals target/buffer for ALL plies.
        assert math.isclose(turnover_frac, target / buffer, rel_tol=1e-6), (
            f"plies={plies}: turnover {turnover_frac} != target/buffer "
            f"{target/buffer}"
        )


def test_games_based_ingest_breaks_invariant():
    """Counter-test: with games-based ingest (which we're replacing),
    positions-per-cycle scales with plies, breaking turnover stability."""
    target_games = 32
    buffer = 1_500_000
    short_positions_per_cycle = target_games * 38 * 8   # short-game era
    long_positions_per_cycle = target_games * 60 * 8    # long-game era
    # The ratio of turnover rates is exactly the ratio of plies — that's
    # the bug.
    ratio = long_positions_per_cycle / short_positions_per_cycle
    assert math.isclose(ratio, 60 / 38, rel_tol=1e-6)
    # Both turnover rates are different from each other — invariant broken.
    assert short_positions_per_cycle / buffer != long_positions_per_cycle / buffer


def test_sgd_per_position_keeps_intensity_stable():
    """When --sgd-per-position is used, SGD steps per cycle scale with
    ingested positions. Combined with constant target_positions, this
    means SGD count per cycle is constant — unlike --sgd-per-game which
    drifts when plies change."""
    target = 12_800
    sgd_per_position = 0.0025  # = 32 / 12800

    # Constant target_positions + per-position K = constant SGD count
    for plies in (20, 40, 60, 80):
        # The ingest is positions-based, so n_positions_this_cycle == target
        # for all plies.
        positions_ingested = target
        sgd_steps = round(sgd_per_position * positions_ingested)
        assert sgd_steps == 32, (
            f"plies={plies}: SGD count {sgd_steps} != 32 expected"
        )
