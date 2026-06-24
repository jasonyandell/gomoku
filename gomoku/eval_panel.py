"""The eval core behind the sensei daemon: rulers, fixed openings, the panel.

This is the pure (no-HTTP) layer. It evaluates a checkpoint against a *panel of
rulers* — fixed reference opponents that span the strength range — from a fixed
opening, and emits a per-color-split series row.

Three things make this fast enough to run on a cadence:
  * the Rapfi ruler leases a WARM engine from a :class:`RapfiPool` instead of
    respawning ``pbrain-rapfi`` every pass;
  * fixed-checkpoint rulers (self126, champ0235) keep their loaded evaluator in
    an :class:`EvaluatorCache` across epochs — only the *current* checkpoint
    reloads;
  * everything runs on CPU (``GOMOKU_DEVICE=cpu``), so it never competes with an
    MPS trainer — the same property that makes the babysit cadence safe.

The one hard constraint it honors (issue #34): **white is reported separately**.
A panel row carries ``white_score`` / ``black_score`` / ``white_loss_rate`` per
ruler, never just an aggregate — a rising aggregate that hides a flat or sinking
white side is exactly the failure mode the separate column exists to catch.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock

import numpy as np

from gomoku.board_config import BOARD_SIZE
from gomoku.eval import MatchResult, play_match_pickers
from gomoku.game import GameState
from gomoku.mcts import MCTSGame, make_torch_evaluator, policy_from_visits, run_batched_mcts
from gomoku.model import fuse_model_for_inference, load_checkpoint
from gomoku.rapfi_pool import RapfiPool
from gomoku.util import pick_device

# The swap2 idx-2 single fair opener: B(3,2), W(5,4), B(4,5) — three stones,
# white to move. The board "Bruce" trains on. (row, col) coordinates; 15×15.
IDX2_OPENING: tuple[tuple[int, int], ...] = ((3, 2), (5, 4), (4, 5))


def fixed_opening_state(
    moves: tuple[tuple[int, int], ...] = IDX2_OPENING,
    *,
    board_size: int = BOARD_SIZE,
) -> GameState:
    """Build a :class:`GameState` by applying a fixed list of (row, col) moves.

    Self-contained on purpose: the swap2 ``_fixed_opening_state`` helper lives on
    an unmerged branch, so we reconstruct the opening from first principles here
    (and it works for any opening, not just idx-2).
    """
    state = GameState.initial()
    for r, c in moves:
        action = int(r) * board_size + int(c)
        legal = set(int(a) for a in state.legal_actions())
        if action not in legal:
            raise ValueError(
                f"opening move ({r},{c}) -> action {action} is illegal at this "
                f"point (board_size={board_size}); openings={moves}"
            )
        state = state.apply(action)
    return state


# --------------------------------------------------------------------------
# Evaluator cache — load each checkpoint once, keyed by (path, mtime) so the
# mutable current checkpoint (latest.pt rewritten each epoch) reloads while
# fixed rulers stay warm.
# --------------------------------------------------------------------------
class EvaluatorCache:
    """Path → (model, evaluator, epoch), keyed by (abspath, mtime).

    Mtime-keying gives freshness for free: when the mutable current checkpoint
    (latest.pt / worker_weights.pt rewritten each epoch) changes on disk its
    mtime advances → cache miss → reload with the new weights. Fixed rulers keep
    a stable mtime → permanent cache hit, so they load exactly once.
    """

    def __init__(self, device=None) -> None:
        self._device = device if device is not None else pick_device(os.environ.get("GOMOKU_DEVICE"))
        self._cache: dict[tuple[str, float], tuple] = {}
        self._lock = Lock()

    @property
    def device(self):
        return self._device

    def _entry(self, path: str) -> tuple:
        abspath = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(abspath)
        except OSError as e:
            raise FileNotFoundError(f"checkpoint not found: {abspath}") from e
        key = (abspath, mtime)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            # latest.pt is rewritten in place (not atomically), so a read landing
            # mid-write can hand torch.load a truncated file. Retry a few times
            # rather than caching a failure / torn payload; only a clean load is
            # ever inserted into the cache.
            last_err: Exception | None = None
            payload = model = None
            for _ in range(4):
                try:
                    model, payload = load_checkpoint(abspath, device=self._device)
                    last_err = None
                    break
                except Exception as e:  # torn/partial write, transient
                    last_err = e
                    time.sleep(0.3)
            if last_err is not None:
                raise last_err
            model = fuse_model_for_inference(model)
            evaluator = make_torch_evaluator(model, self._device)
            epoch = int(payload.get("epoch", 0))
            # Drop any stale (older-mtime) entry for the same path to bound memory.
            self._cache = {k: v for k, v in self._cache.items() if k[0] != abspath}
            self._cache[key] = (model, evaluator, epoch)
            return self._cache[key]

    def evaluator(self, path: str):
        return self._entry(path)[1]

    def epoch(self, path: str) -> int:
        return self._entry(path)[2]

    def entry(self, path: str) -> tuple:
        """(evaluator, epoch) from a SINGLE (path, mtime) snapshot — so a caller
        can pin both to the same file revision across a whole panel."""
        e = self._entry(path)
        return e[1], e[2]


def make_net_picker(
    evaluator,
    *,
    sims: int = 160,
    c_puct: float = 1.5,
    temp_until_ply: int = 0,
    temperature: float = 1.0,
):
    """A net picker matching eval's default MCTS path, with optional early-ply
    sampling for net-vs-net opening variety.

    For ``state.move_count < temp_until_ply`` the move is SAMPLED from the
    visit-temperature distribution (the only way to get game variety from a
    deterministic net at a fixed opening); afterwards it is the greedy argmax,
    byte-identical to :func:`gomoku.eval.mcts_picker`'s default path.
    """

    def pick(state: GameState, rng: np.random.Generator) -> int:
        g = MCTSGame(state, c_puct=c_puct, rng=rng)
        run_batched_mcts([g], evaluator, n_simulations=sims, add_root_noise=False)
        if temp_until_ply > 0 and state.move_count < temp_until_ply:
            pi = policy_from_visits(g.root, temperature=temperature)
            total = float(pi.sum())
            if total > 0:
                pi = pi / total
                return int(rng.choice(len(pi), p=pi))
        pi = policy_from_visits(g.root, temperature=0.0)
        return int(np.argmax(pi))

    return pick


@dataclass
class Ruler:
    """One reference opponent in the panel.

    ``opponent`` is one of: ``"rapfi"`` (warm pool), a checkpoint ``*.pt`` path
    (net-vs-net), or a baseline spec (``"heuristic"``, ``"lookahead:depth=4"``…).
    """

    label: str
    opponent: str
    n_games: int = 16
    sims: int = 160
    temp_until_ply: int = 0
    temperature: float = 1.0
    rapfi_timeout_ms: int = 1000


def _looks_like_checkpoint(opponent: str) -> bool:
    return opponent.endswith(".pt") or os.path.isfile(opponent)


def eval_vs_ruler(
    checkpoint: str,
    ruler: Ruler,
    *,
    cache: EvaluatorCache,
    pool: RapfiPool | None = None,
    start_state: GameState | None = None,
    c_puct: float = 1.5,
    seed: int = 0,
    net_evaluator=None,
) -> MatchResult:
    """Play the current checkpoint against one ruler; return A=net result.

    ``net_evaluator`` lets the caller pin a single checkpoint snapshot across a
    whole panel; when None we resolve it from the (mtime-keyed) cache.
    """
    net_eval = net_evaluator if net_evaluator is not None else cache.evaluator(checkpoint)
    net = make_net_picker(
        net_eval,
        sims=ruler.sims,
        c_puct=c_puct,
        temp_until_ply=ruler.temp_until_ply,
        temperature=ruler.temperature,
    )
    if ruler.opponent == "rapfi":
        if pool is None:
            raise ValueError("ruler 'rapfi' needs a RapfiPool (pool=None)")
        with pool.lease() as eng:
            return play_match_pickers(
                net, eng, n_games=ruler.n_games, seed=seed, start_state=start_state
            )
    if _looks_like_checkpoint(ruler.opponent):
        opp_eval = cache.evaluator(ruler.opponent)  # fixed ruler: cached/warm
        opp = make_net_picker(
            opp_eval,
            sims=ruler.sims,
            c_puct=c_puct,
            temp_until_ply=ruler.temp_until_ply,
            temperature=ruler.temperature,
        )
        return play_match_pickers(
            net, opp, n_games=ruler.n_games, seed=seed, start_state=start_state
        )
    # Baseline spec (heuristic / lookahead / defensive / …).
    from gomoku.match import build_player, parse_spec

    opp = build_player(parse_spec(ruler.opponent))
    return play_match_pickers(
        net, opp, n_games=ruler.n_games, seed=seed, start_state=start_state
    )


def result_to_row(label: str, res: MatchResult) -> dict:
    """A per-color-split summary for one ruler (white reported SEPARATELY)."""
    white_games = res.white_w + res.white_l + res.white_d
    black_games = res.black_w + res.black_l + res.black_d
    return {
        "ruler": label,
        "n_games": res.n_games,
        "score": round(res.win_rate, 4),
        "wins": res.wins,
        "losses": res.losses,
        "draws": res.draws,
        # --- white reported separately (issue #34) ---
        "white_games": white_games,
        "white_score": round((res.white_w + 0.5 * res.white_d) / max(white_games, 1), 4),
        "white_loss_rate": round(res.white_l / max(white_games, 1), 4),
        "white_wld": [res.white_w, res.white_l, res.white_d],
        # --- black for symmetry ---
        "black_games": black_games,
        "black_score": round((res.black_w + 0.5 * res.black_d) / max(black_games, 1), 4),
        "black_loss_rate": round(res.black_l / max(black_games, 1), 4),
        "black_wld": [res.black_w, res.black_l, res.black_d],
    }


def run_panel(
    checkpoint: str,
    rulers: list[Ruler],
    *,
    cache: EvaluatorCache,
    pool: RapfiPool | None = None,
    start_state: GameState | None = None,
    c_puct: float = 1.5,
    seed: int = 0,
    epoch: int | None = None,
    now: float | None = None,
) -> dict:
    """Evaluate ``checkpoint`` against every ruler; return one flat series row.

    The row is the #34 deliverable: ``{epoch, ckpt, ts, rulers:[per-ruler split
    rows]}`` plus flattened ``<ruler>__white_score`` / ``__white_loss_rate`` /
    ``__score`` keys so it drops straight into a JSONL series or a wandb log.
    """
    # Pin ONE (weights, epoch) snapshot for the whole panel: resolve the net
    # evaluator + epoch from a single cache entry so the row's epoch label can
    # never disagree with the weights actually evaluated, even if the checkpoint
    # is rewritten mid-panel (the #34 series must not be mislabeled).
    net_evaluator = None
    try:
        net_evaluator, pinned_epoch = cache.entry(checkpoint)
        if epoch is None:
            epoch = pinned_epoch
    except Exception:
        pass
    rows = []
    for r in rulers:
        try:
            res = eval_vs_ruler(
                checkpoint,
                r,
                cache=cache,
                pool=pool,
                start_state=start_state,
                c_puct=c_puct,
                seed=seed,
                net_evaluator=net_evaluator,
            )
            rows.append(result_to_row(r.label, res))
        except Exception as e:  # one ruler failing must not sink the whole panel
            rows.append({"ruler": r.label, "error": f"{type(e).__name__}: {e}"})
    out: dict = {
        "epoch": epoch,
        "ckpt": os.path.abspath(checkpoint),
        "ts": time.time() if now is None else now,
        "rulers": rows,
    }
    for row in rows:
        lab = row["ruler"]
        if "error" in row:
            out[f"{lab}__error"] = row["error"]
            continue
        out[f"{lab}__score"] = row["score"]
        out[f"{lab}__white_score"] = row["white_score"]
        out[f"{lab}__white_loss_rate"] = row["white_loss_rate"]
        out[f"{lab}__black_score"] = row["black_score"]
    return out
