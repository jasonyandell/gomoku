"""Parallel self-play game generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gomoku.game import GameState, HISTORY_PLY, N_ACTIONS, augment
from gomoku.mcts import (
    Evaluator,
    MCTSGame,
    policy_from_visits,
    run_batched_mcts,
    run_batched_mcts_waves,
)
from gomoku import native_mcts


@dataclass
class SelfPlayExample:
    """One training example: a canonical state, MCTS policy, and outcome.

    All three are from the SAME side-to-move perspective (canonical).
    """

    planes: np.ndarray   # (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE) float32
    pi: np.ndarray       # (N_ACTIONS,) float32, sums to 1 over legal actions
    z: float             # value in [-1, 1]
    side: int = 0        # 0=black (first mover), 1=white — for per-color diagnostics
    ply: int = 0         # ply count at which this position was captured


@dataclass
class GameRecord:
    """A completed game's training examples plus some metadata."""

    examples: list[SelfPlayExample]
    plies: int
    outcome: float  # +1 if first-mover (black) won, -1 if second-mover (white) won, 0 draw
    archive_start: bool = False  # WL5: True if game was seeded from validation archive


def _sample_action(pi: np.ndarray, rng: np.random.Generator) -> int:
    """Sample an action from a probability distribution. Falls back to argmax
    if the distribution is degenerate (zero/NaN sum) or contains NaN entries.

    NaN guard is load-bearing: WL3 (wandb 0o75gws5) crashed all 8 workers
    at e825 from `ValueError: Probabilities contain NaN` because native
    MCTS occasionally emits NaN visit-policies. Without the guard, `s <= 0`
    returned False on NaN (NaN comparisons are False), the divide
    propagated NaN, and rng.choice rejected. The underlying MCTS NaN
    source is the real fix — this is the safety net.
    """
    s = pi.sum()
    if not np.isfinite(s) or s <= 0:
        return int(np.argmax(np.nan_to_num(pi, nan=-np.inf)))
    pi = pi / s
    if not np.all(np.isfinite(pi)):
        return int(np.argmax(np.nan_to_num(pi, nan=-np.inf)))
    return int(rng.choice(len(pi), p=pi))


def _random_opening_state(rng: np.random.Generator, n_moves: int) -> tuple[GameState, int]:
    """Play `n_moves` uniform-random legal moves from an empty board, then return
    the resulting state and the ply count (= n_moves, unless the random play hit
    terminal first — in which case we restart from empty).

    Used by generate_games / generate_games_vs_baseline to inject opening diversity.
    """
    if n_moves <= 0:
        return GameState.initial(), 0
    while True:
        state = GameState.initial()
        plies = 0
        for _ in range(n_moves):
            legal = state.legal_actions()
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            state = state.apply(action)
            plies += 1
            done, _ = state.is_terminal()
            if done:
                # Unlucky — random play accidentally ended the game. Try again.
                break
        else:
            return state, plies
        # restart from scratch


def _gamestate_from_archive(archive: dict, idx: int) -> GameState:
    """WL5 archive-start: build a GameState from one archived position.

    Reads plane 0 (current-side stones) and plane HISTORY_PLY (opponent
    stones) at archive index `idx` and constructs a fresh `GameState`
    with `move_count` taken from `archive["ply"][idx]`. History tuple is
    left empty — the archive only persists planes, not move order, so
    planes 1..H-1 / H+1..2H-1 will read as zeros from the new state. The
    archived current+opponent positions and side-to-move are preserved;
    only the per-ply history snapshots are lost. Acceptable per the WL5
    design doc (history is "hard without move order").
    """
    planes = archive["planes"][idx].cpu().numpy()
    current = planes[0].astype(bool)
    opponent = planes[HISTORY_PLY].astype(bool)
    board = np.stack([current, opponent], axis=0)
    move_count = int(archive["ply"][idx].item())
    return GameState(board=board, move_count=move_count, history=())


def _can_use_native_mcts(evaluator: Evaluator) -> bool:
    return bool(
        native_mcts.USING_NATIVE_MCTS
        and native_mcts.NativeMCTSGame is not None
        and hasattr(evaluator, "evaluate_planes")
    )


def _generate_games_native(
    n_games: int,
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.25,
    c_puct_base: float = 19652.0,
    temperature_moves: int = 8,
    temperature_final: float = 0.1,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    max_plies: int | None = None,
    rng: np.random.Generator | None = None,
    augment_symmetries: bool = True,
    wave_size: int = 1,
    random_opening_moves: int = 0,
    archive: dict | None = None,
    archive_start_frac: float = 0.0,
) -> list[GameRecord]:
    rng = rng or np.random.default_rng()
    if max_plies is None:
        max_plies = N_ACTIONS

    planes_evaluator = evaluator.evaluate_planes  # type: ignore[attr-defined]
    games = []
    initial_plies: list[int] = []
    archive_start_flags: list[bool] = []
    n_archive = 0 if archive is None else int(archive["planes"].shape[0])
    for _ in range(n_games):
        from_archive = (
            archive is not None
            and n_archive > 0
            and float(rng.random()) < archive_start_frac
        )
        if from_archive:
            idx = int(rng.integers(0, n_archive))
            start_state = _gamestate_from_archive(archive, idx)
            opening_plies = start_state.move_count
        elif random_opening_moves > 0:
            start_state, opening_plies = _random_opening_state(rng, random_opening_moves)
        else:
            start_state, opening_plies = GameState.initial(), 0
        games.append(
            native_mcts.NativeMCTSGame(
                start_state,
                c_puct=c_puct,
                c_puct_base=c_puct_base,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                seed=int(rng.integers(1, 2**63 - 1)),
            )
        )
        initial_plies.append(opening_plies)
        archive_start_flags.append(from_archive)

    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []

    ply = 0
    while active and ply < max_plies:
        active_games = [games[i] for i in active]
        native_mcts.search_batch(
            active_games,
            planes_evaluator,
            n_simulations=n_simulations,
            wave_size=wave_size,
            add_root_noise=True,
        )

        next_active: list[int] = []
        for slot_idx, g_idx in enumerate(active):
            g = active_games[slot_idx]
            tau = 1.0 if ply < temperature_moves else temperature_final
            pi = g.policy(temperature=tau)
            n_initial = initial_plies[g_idx]
            side = (n_initial + ply) % 2
            # Sanitize pi before recording the training example: NaN entries
            # from the native MCTS policy export must not enter the buffer.
            # _sample_action handles NaN for the *play* path, but trajectories
            # are stored independently and feed the trainer's cross-entropy
            # target. A NaN target poisons the loss for the entire minibatch.
            # Replace NaN with 0 and re-normalize; if everything is NaN, fall
            # back to a uniform distribution (lowest-information target —
            # better than corrupting the buffer).
            if not np.all(np.isfinite(pi)):
                pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
                s = pi.sum()
                if s <= 0:
                    pi = np.full_like(pi, 1.0 / len(pi))
                else:
                    pi = pi / s
            trajectories[g_idx].append((g.root_planes(), pi.copy(), side))

            action = _sample_action(pi, rng)
            g.advance_root(action)
            done, term_val = g.is_terminal()
            if done:
                if term_val == -1.0:
                    winner_side = side
                    outcome_for_black = 1.0 if winner_side == 0 else -1.0
                else:
                    outcome_for_black = 0.0
                completed.append((g_idx, outcome_for_black, n_initial + ply + 1))
            else:
                next_active.append(g_idx)

        active = next_active
        ply += 1

    for g_idx in active:
        completed.append((g_idx, 0.0, initial_plies[g_idx] + ply))

    records: list[GameRecord] = []
    for g_idx, outcome_for_black, plies in sorted(completed):
        examples: list[SelfPlayExample] = []
        n_initial = initial_plies[g_idx]
        for ply_idx, (planes, pi, side) in enumerate(trajectories[g_idx]):
            z = outcome_for_black if side == 0 else -outcome_for_black
            ply_at_capture = n_initial + ply_idx
            if augment_symmetries:
                for aug_planes, aug_pi in augment(planes, pi):
                    examples.append(SelfPlayExample(
                        aug_planes, aug_pi.astype(np.float32), z,
                        side=int(side), ply=int(ply_at_capture),
                    ))
            else:
                examples.append(SelfPlayExample(
                    planes, pi.astype(np.float32), z,
                    side=int(side), ply=int(ply_at_capture),
                ))
        records.append(GameRecord(
            examples=examples,
            plies=plies,
            outcome=outcome_for_black,
            archive_start=archive_start_flags[g_idx],
        ))

    return records


def generate_games(
    n_games: int,
    evaluator: Evaluator,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.25,
    c_puct_base: float = 19652.0,
    temperature_moves: int = 8,
    temperature_final: float = 0.1,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    max_plies: int | None = None,
    rng: np.random.Generator | None = None,
    augment_symmetries: bool = True,
    wave_size: int = 1,
    random_opening_moves: int = 0,
    archive: dict | None = None,
    archive_start_frac: float = 0.0,
) -> list[GameRecord]:
    """Generate `n_games` self-play games in parallel.

    All games advance in lockstep: at each ply we batch-MCTS across active games,
    sample an action per game, apply it, and remove any games that ended.

    `wave_size` > 1 enables zeb-style wave-batched MCTS with virtual loss:
    each round collects `wave_size` leaves per game in one batched evaluator
    call. wave_size=1 reduces to the original per-sim batching.

    `random_opening_moves` > 0 starts each game with that many uniform-random
    legal moves played (alternating sides); MCTS only takes over after that.
    No training examples are recorded for the random opening — only for moves
    chosen by MCTS. Breaks the "always-same-opening" collapse mode by forcing
    the model to learn from a diverse set of starting positions.
    """
    rng = rng or np.random.default_rng()
    if _can_use_native_mcts(evaluator):
        return _generate_games_native(
            n_games,
            evaluator,
            n_simulations=n_simulations,
            c_puct=c_puct,
            c_puct_base=c_puct_base,
            temperature_moves=temperature_moves,
            temperature_final=temperature_final,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            max_plies=max_plies,
            rng=rng,
            augment_symmetries=augment_symmetries,
            wave_size=wave_size,
            random_opening_moves=random_opening_moves,
            archive=archive,
            archive_start_frac=archive_start_frac,
        )
    if max_plies is None:
        max_plies = N_ACTIONS  # full-board fallback (game can't have more than this)

    games: list[MCTSGame] = []
    initial_plies: list[int] = []
    for _ in range(n_games):
        if random_opening_moves > 0:
            start_state, opening_plies = _random_opening_state(rng, random_opening_moves)
        else:
            start_state, opening_plies = GameState.initial(), 0
        games.append(MCTSGame(start_state, c_puct=c_puct, c_puct_base=c_puct_base,
                              dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
                              rng=np.random.default_rng(rng.integers(0, 2**31))))
        initial_plies.append(opening_plies)

    # Per-game trajectory of (planes, pi, side_to_move_at_that_ply)
    # side_to_move is encoded as 0 for the player who moved first ("black"), 1 for the other.
    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []  # (game_idx, outcome_for_black, plies)

    ply = 0
    while active and ply < max_plies:
        active_games = [games[i] for i in active]
        if wave_size > 1:
            run_batched_mcts_waves(
                active_games,
                evaluator,
                n_simulations=n_simulations,
                wave_size=wave_size,
                add_root_noise=True,
            )
        else:
            run_batched_mcts(
                active_games,
                evaluator,
                n_simulations=n_simulations,
                add_root_noise=True,
            )

        next_active: list[int] = []
        for slot_idx, g_idx in enumerate(active):
            g = active_games[slot_idx]
            tau = 1.0 if ply < temperature_moves else temperature_final
            pi = policy_from_visits(g.root, tau)
            # Total moves played so far in THIS game = initial_plies[g_idx] (random
            # opening) + ply (MCTS moves applied so far). The side ABOUT to move
            # at this point has parity = total_moves % 2.
            n_initial = initial_plies[g_idx]
            side = (n_initial + ply) % 2
            trajectories[g_idx].append((g.root.state.to_planes(), pi.copy(), side))

            action = _sample_action(pi, rng)
            g.advance_root(action)
            done, term_val = g.root.state.is_terminal()
            if done:
                # term_val is from the NEW root's side-to-move perspective.
                # The new side-to-move is the player who DIDN'T just move.
                # If term_val == -1, the player who just moved (side) won.
                # If term_val == 0, draw.
                if term_val == -1.0:
                    winner_side = side
                    outcome_for_black = 1.0 if winner_side == 0 else -1.0
                else:
                    outcome_for_black = 0.0
                completed.append((g_idx, outcome_for_black, n_initial + ply + 1))
            else:
                next_active.append(g_idx)

        active = next_active
        ply += 1

    # Any games still active at max_plies are scored as draws. Total plies is
    # the MCTS-loop ply plus the per-game random opening prefix.
    for g_idx in active:
        completed.append((g_idx, 0.0, initial_plies[g_idx] + ply))

    # Build records, applying symmetry augmentation.
    records: list[GameRecord] = []
    for g_idx, outcome_for_black, plies in sorted(completed):
        examples: list[SelfPlayExample] = []
        n_initial = initial_plies[g_idx]
        for ply_idx, (planes, pi, side) in enumerate(trajectories[g_idx]):
            z = outcome_for_black if side == 0 else -outcome_for_black
            ply_at_capture = n_initial + ply_idx
            if augment_symmetries:
                for aug_planes, aug_pi in augment(planes, pi):
                    examples.append(SelfPlayExample(
                        aug_planes, aug_pi.astype(np.float32), z,
                        side=int(side), ply=int(ply_at_capture),
                    ))
            else:
                examples.append(SelfPlayExample(
                    planes, pi, z,
                    side=int(side), ply=int(ply_at_capture),
                ))
        records.append(GameRecord(examples=examples, plies=plies, outcome=outcome_for_black))

    return records


def generate_games_vs_baseline(
    n_games: int,
    evaluator: Evaluator,
    opponent_picker,
    *,
    n_simulations: int = 100,
    c_puct: float = 1.25,
    c_puct_base: float = 19652.0,
    temperature_moves: int = 8,
    temperature_final: float = 0.1,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    max_plies: int | None = None,
    rng: np.random.Generator | None = None,
    augment_symmetries: bool = True,
    wave_size: int = 1,
    model_first_frac: float = 0.5,
    random_opening_moves: int = 0,
) -> list[GameRecord]:
    """Generate games where the model plays a fixed opponent picker.

    The model uses MCTS (with the same `wave_size` / `dirichlet` / `c_puct` knobs
    as self-play). The opponent uses `opponent_picker(state, rng)` directly —
    no MCTS, no examples recorded for opponent moves. Training examples come
    only from the model's plies, with `z` set to the game's outcome from the
    model's perspective.

    `model_first_frac` is the fraction of games where the model plays the first
    move (side 0); the rest the model plays second. Default 0.5 so the model
    sees both sides equally.

    `GameRecord.outcome` is from the MODEL's perspective (+1 win, -1 loss, 0 draw)
    rather than first-mover's.
    """
    rng = rng or np.random.default_rng()
    if max_plies is None:
        max_plies = N_ACTIONS

    games: list[MCTSGame] = []
    initial_plies: list[int] = []
    for _ in range(n_games):
        if random_opening_moves > 0:
            start_state, opening_plies = _random_opening_state(rng, random_opening_moves)
        else:
            start_state, opening_plies = GameState.initial(), 0
        games.append(MCTSGame(start_state, c_puct=c_puct, c_puct_base=c_puct_base,
                              dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
                              rng=np.random.default_rng(rng.integers(0, 2**31))))
        initial_plies.append(opening_plies)
    # side the model plays in each game: 0 = first mover, 1 = second
    model_side = np.where(rng.random(n_games) < model_first_frac, 0, 1).astype(np.int8)

    trajectories: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(n_games)]
    active: list[int] = list(range(n_games))
    completed: list[tuple[int, float, int]] = []  # (game_idx, outcome_for_model, plies)

    # initial_plies is constant across games (always == random_opening_moves), so
    # all games share the same `side_to_move` at any loop ply.
    n_initial = random_opening_moves
    ply = 0
    while active and ply < max_plies:
        side_to_move = (n_initial + ply) % 2
        model_turn = [i for i in active if model_side[i] == side_to_move]
        opp_turn = [i for i in active if model_side[i] != side_to_move]

        # Model: batched MCTS on its subset of games.
        if model_turn:
            mcts_games = [games[i] for i in model_turn]
            if wave_size > 1:
                run_batched_mcts_waves(
                    mcts_games, evaluator,
                    n_simulations=n_simulations, wave_size=wave_size,
                    add_root_noise=True,
                )
            else:
                run_batched_mcts(
                    mcts_games, evaluator,
                    n_simulations=n_simulations, add_root_noise=True,
                )
            for slot_idx, g_idx in enumerate(model_turn):
                g = mcts_games[slot_idx]
                tau = 1.0 if ply < temperature_moves else temperature_final
                pi = policy_from_visits(g.root, tau)
                trajectories[g_idx].append((g.root.state.to_planes(), pi.copy(), n_initial + ply))
                action = _sample_action(pi, rng)
                g.advance_root(action)

        # Opponent: just call picker per game.
        for g_idx in opp_turn:
            g = games[g_idx]
            action = int(opponent_picker(g.root.state, rng))
            g.advance_root(action)

        # Terminal check (same for both subsets).
        next_active: list[int] = []
        for g_idx in active:
            g = games[g_idx]
            done, term_val = g.root.state.is_terminal()
            if done:
                # The player who just moved is `side_to_move`. If term_val == -1
                # at the new root (whose side is the player who DIDN'T move),
                # the mover just won.
                if term_val == -1.0:
                    winner_side = side_to_move
                    outcome_for_model = 1.0 if winner_side == int(model_side[g_idx]) else -1.0
                else:
                    outcome_for_model = 0.0
                completed.append((g_idx, outcome_for_model, n_initial + ply + 1))
            else:
                next_active.append(g_idx)
        active = next_active
        ply += 1

    # Max-plies fallthrough = draw.
    for g_idx in active:
        completed.append((g_idx, 0.0, n_initial + ply))

    records: list[GameRecord] = []
    for g_idx, outcome_for_model, plies in sorted(completed):
        examples: list[SelfPlayExample] = []
        side = int(model_side[g_idx])
        for planes, pi, ply_at_capture in trajectories[g_idx]:
            # planes are canonical (plane 0 = side-to-move = model at the moment
            # the example was recorded), so z is directly outcome_for_model.
            z = outcome_for_model
            if augment_symmetries:
                for aug_planes, aug_pi in augment(planes, pi):
                    examples.append(SelfPlayExample(
                        aug_planes, aug_pi.astype(np.float32), z,
                        side=side, ply=int(ply_at_capture),
                    ))
            else:
                examples.append(SelfPlayExample(
                    planes, pi, z,
                    side=side, ply=int(ply_at_capture),
                ))
        records.append(GameRecord(examples=examples, plies=plies, outcome=outcome_for_model))

    return records
