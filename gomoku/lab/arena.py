#!/usr/bin/env python3
"""The autolab arena role (epic #53, P4).

The trainer's twin — same daemon shape (`daemon.run_daemon`), consuming the
`role="arena"` eval-jobs the trainer enqueues. One `run_chunk` = gate one
candidate model head-to-head against the current champion (the HF `champion` tag),
record an `eval` + `verdict` row, and on PROMOTE move the `champion` tag to the
candidate's revision.

Reuses the proven gate (`scripts/sliding_gate.run_gate` — calibration-immune,
3-way PROMOTE/REVERT/AMBIGUOUS on a Wilson CI) in `dry_run=True` mode: we want its
verdict + verdict-log, but the champion is an HF tag, not its local peak file, so
the arena does the promotion itself. The first candidate (no champion yet) is a
definitional PROMOTE and becomes the champion.

Lean: no new gating math, no new match engine — just wiring over `run_gate` + HF.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import daemon, ledger
from .. import hf

ROLE = "arena"
CHAMPION_TAG = "champion"


def champion_tag(board_size: int) -> str:
    """The HF champion tag is namespaced per board-size era. A 9×9 net can't be
    loaded to gate a 15×15 candidate (shape mismatch), so each era has its own
    ladder root; crossing eras can never resolve a stale cross-era champion."""
    return f"{CHAMPION_TAG}-{board_size}"


class ArenaRole:
    role_name = ROLE
    poll_interval = 30.0

    def __init__(self, *, repo_root: str | None = None, hf_repo: str = hf.DEFAULT_REPO,
                 gate_n_games: int = 40, shrink_n_games: int = 12, sims: int = 100,
                 c_puct: float = 1.5, opening_plies: int = 4, ci_margin: float = 0.0,
                 device: str | None = None, n_workers: int = 4, seed: int = 0,
                 board_size: int | None = None,
                 gate_fn=None, champion_resolver=None, champion_setter=None,
                 eval_fn=None):
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
        self.hf_repo = hf_repo
        # Era is a process-start constant (the daemon plist sets GOMOKU_BOARD_SIZE);
        # the champion tag is namespaced by it so eras never cross-contaminate.
        self.board_size = (board_size if board_size is not None
                           else int(os.environ.get("GOMOKU_BOARD_SIZE", "9")))
        self._champ_tag = champion_tag(self.board_size)
        self.gate_n_games = gate_n_games
        self.shrink_n_games = shrink_n_games
        self.sims = sims
        self.c_puct = c_puct
        self.opening_plies = opening_plies
        self.ci_margin = ci_margin
        self.device = device
        self.n_workers = n_workers
        self.seed = seed
        # injection seams (tests pass stubs; production uses the real HF + run_gate)
        self._gate_fn = gate_fn
        self._eval_fn = eval_fn
        self._resolve_champion_fn = champion_resolver or self._default_resolve_champion
        self._set_champion_fn = champion_setter or self._default_set_champion

    def preflight(self) -> None:
        # Eval may co-run with the trainer per the cross-engine envelope; the
        # co-tenancy guard (n_games shrink) is the politeness, not a hard defer.
        return

    # ---- the gate -------------------------------------------------------

    def run_chunk(self, item: dict) -> daemon.ChunkResult:
        cand_ref = item.get("base")
        cand_path = self._resolve_model(cand_ref)
        champ_rev, champ_path = self._resolve_champion_fn()

        # co-tenancy: a live trainer slice shares the GPU → shrink the match block
        n_games = self.gate_n_games
        if daemon.probe_alive(daemon.lockfile_path("train")):
            n_games = self.shrink_n_games

        v = self._gate(candidate=cand_path, peak=champ_path, n_games=n_games)
        gate_promote = bool(getattr(v, "promote", False))
        first_champion = champ_rev is None
        # The first champion defines the era's ladder root with ZERO games played
        # (no incumbent to beat) — the single most consequential automated act.
        # Locked doctrine: first promotion is human-gated. Escalate, don't crown.
        crowned = gate_promote and not first_champion
        if crowned:
            self._set_champion_fn(item, cand_ref, cand_path, v)

        ci = [getattr(v, "ci_lo", None), getattr(v, "ci_hi", None)]
        metrics = {"verdict": v.verdict, "win_rate": getattr(v, "win_rate", None),
                   "ci": ci, "n_games": getattr(v, "n_games", n_games),
                   "promoted": crowned, "vs": champ_rev or "(first champion)"}
        if gate_promote and first_champion:
            metrics["needs_jason"] = True   # awaiting human confirm; tag NOT moved
        ev_id = f"ev-{item['id']}".replace("/", "_")
        note = ("promoted champion" if crowned else
                "first champion — needs human confirm" if metrics.get("needs_jason") else "")
        followups = [
            ledger.eval_row(ev_id, model=cand_ref, panel=[champ_rev or "none", "candidate"],
                            metrics=metrics),
            ledger.verdict(ev_id, gate=v.verdict, win_rate=getattr(v, "win_rate", None),
                           ci=ci, n=getattr(v, "n_games", n_games), note=note),
        ]
        return daemon.ChunkResult(artifact_ref=cand_ref, metrics=metrics, followups=followups)

    def _gate(self, *, candidate, peak, n_games):
        if self._gate_fn is not None:           # test seam
            return self._gate_fn(candidate=candidate, peak=peak, n_games=n_games)
        # real gate: lazy import (keeps scripts/ off the import path until needed)
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        from scripts.sliding_gate import run_gate
        arena_dir = Path(daemon.home()) / "arena"
        arena_dir.mkdir(parents=True, exist_ok=True)
        return run_gate(
            candidate=candidate, peak=peak, n_games=n_games, seed=self.seed,
            board_path=arena_dir / "board.json", verdict_log=arena_dir / "verdicts.jsonl",
            peak_out=arena_dir / "peak.pt", sims=self.sims, c_puct=self.c_puct,
            opening_plies=self.opening_plies, device=self.device or "cpu",
            n_workers=self.n_workers, ci_margin=self.ci_margin,
            dry_run=True,            # we own promotion (HF tag), not run_gate's peer file
            want_white_loss=False, eval_fn=self._eval_fn)

    # ---- model + champion resolution ------------------------------------

    @staticmethod
    def _parse_hf_ref(ref: str | None) -> tuple[str, str | None] | None:
        """Parse an HF artifact ref into ``(repo_id, revision)``, else ``None``.

        Accepts BOTH the explicit ``hf://owner/repo@rev`` form and the BARE
        ``owner/repo@rev`` form that the trainer actually emits (``_deliver`` ->
        ``hf.push_slice`` -> ``f"{repo_id}@{revision}"``, no scheme). ``local://``
        and existing local filesystem paths are not HF. This tolerance is what
        lets the arena resolve the trainer's per-slice artifact (issue #67).
        """
        if not ref:
            return None
        if ref.startswith("local://"):
            return None
        if ref.startswith("hf://"):
            ref = ref[len("hf://"):]
        elif Path(ref).exists():
            return None  # a real local checkpoint path
        # bare "owner/repo@rev" (or a de-schemed hf://): org-qualified repo id
        repo, sep, rev = ref.partition("@")
        if "/" in repo:
            return repo, (rev if sep and rev else None)
        return None

    def _resolve_model(self, ref: str | None) -> str | None:
        if not ref:
            return None
        if ref.startswith("local://"):
            return ref[len("local://"):]
        parsed = self._parse_hf_ref(ref)
        if parsed:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(parsed[0], "model.pt", revision=parsed[1])
        return ref

    def _default_resolve_champion(self):
        """Return (revision, local_path) of the current champion, or (None, None)."""
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(self.hf_repo, "model.pt", revision=self._champ_tag)
            return self._champ_tag, path
        except Exception:
            return None, None   # no champion yet → the candidate seeds the ladder

    def _default_set_champion(self, item, cand_ref, cand_path, v):
        """Move the `champion` tag to the candidate's revision (or push+tag a local)."""
        from huggingface_hub import HfApi
        api = HfApi()
        parsed = self._parse_hf_ref(cand_ref)
        if parsed:
            repo, rev = parsed
            try:
                api.delete_tag(repo, tag=self._champ_tag, repo_type="model")
            except Exception:
                pass
            api.create_tag(repo, tag=self._champ_tag, revision=rev or "main",
                           repo_type="model", exist_ok=True)
        else:
            hf.push_slice(cand_path, repo_id=self.hf_repo,
                          revision=f"{self._champ_tag}-{item['id']}".replace("/", "_"),
                          tag=self._champ_tag,
                          provenance={"ledger_row_id": item["id"], "gate": v.verdict})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=daemon.default_ledger_path())
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--stop-file")
    ap.add_argument("--gate-n-games", type=int, default=40)
    args = ap.parse_args(argv)
    role = ArenaRole(gate_n_games=args.gate_n_games)
    return daemon.run_daemon(role, args.ledger, once=args.once, stop_file=args.stop_file)


if __name__ == "__main__":
    sys.exit(main())
