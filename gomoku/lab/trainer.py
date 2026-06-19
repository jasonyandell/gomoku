#!/usr/bin/env python3
"""The autolab trainer role (epic #53, P3).

Plugs into ``daemon.run_daemon``. One ``run_chunk`` = one bounded training slice:
resolve the start model, check out the ledger row's ``commit`` into an ephemeral
**code-only** worktree, run the proven ``run_sweep`` bundle (1 trainer + N
selfplay workers) with its DATA pointed at ``~/data/autolab/runs/<lane>/`` (so
teardown can't delete checkpoints), read the eval, deliver a per-slice HF
revision, and emit the flywheel follow-ups (continuation slice + arena eval).

The trainer owns no loop and no lock — the daemon does. It is pure work.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import daemon, ledger
from .. import hf

ROLE = "train"
_TENANT_PATTERN = r"selfplay_worker|gomoku\.train|run_sweep|eval_worker"


class SliceFailed(Exception):
    """A training slice produced no usable output (run_sweep failed / no latest.pt)."""


class TrainerRole:
    role_name = ROLE
    poll_interval = 30.0

    def __init__(self, *, repo_root: str | os.PathLike | None = None,
                 hf_repo: str = hf.DEFAULT_REPO, dry_run: bool = False,
                 mvp_mode: bool = True, mvp_cap_secs: float = 1.0,
                 prod_cap_secs: float = 3600.0, check_tenants: bool = True):
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
        self.hf_repo = hf_repo
        self.dry_run = dry_run
        self.mvp_mode = mvp_mode
        self.mvp_cap_secs = mvp_cap_secs
        self.prod_cap_secs = prod_cap_secs
        self.check_tenants = check_tenants

    # ---- preflight: don't barge a foreign GPU tenant --------------------

    def preflight(self) -> None:
        if not self.check_tenants:
            return
        try:
            out = subprocess.check_output(
                ["pgrep", "-fl", _TENANT_PATTERN], text=True).strip()
        except subprocess.CalledProcessError:
            return  # nothing matched → box is free
        home = daemon.home()
        foreign = [ln for ln in out.splitlines()
                   if home not in ln and str(os.getpid()) not in ln.split()[:1]]
        if foreign:
            raise daemon.PreflightDeferred(
                "foreign GPU tenant(s): " + "; ".join(foreign[:3]))

    # ---- the slice ------------------------------------------------------

    def run_chunk(self, item: dict) -> daemon.ChunkResult:
        row_id = item["id"]
        config = item.get("config") or {}
        lane = config.get("lane") or row_id
        run_base = Path(daemon.home()) / "runs" / lane
        resume = self._resolve_base(item)
        cell_key = config.get("cell", "SMOKE")
        cap = config.get("max_wall_secs",
                         self.mvp_cap_secs if self.mvp_mode else self.prod_cap_secs)
        board_size = config.get("board_size")

        work = self._checkout(item.get("commit"), row_id)
        try:
            self._run_slice(work, cell_key, run_base, resume, cap, board_size)
            ckpt_dir = self._find_ckpt_dir(run_base)
            latest = ckpt_dir / "latest.pt" if ckpt_dir else None
            if latest is None or not latest.exists():
                raise SliceFailed(f"no latest.pt under {run_base}/sweep_runs/*/checkpoints")
            epochs = len(list(ckpt_dir.glob("epoch*.pt")))
            elo = self._read_model_elo(ckpt_dir)
            artifact = self._deliver(item, lane, latest, elo)
            metrics = {"eval/model_elo": elo, "epochs_ran": epochs, "lane": lane,
                       "cell": cell_key}
            return daemon.ChunkResult(
                artifact_ref=artifact, metrics=metrics, buffer=f"local://{latest}",
                followups=self._followups(item, lane, artifact, elo, config))
        finally:
            self._teardown(work)

    # ---- base → resume --------------------------------------------------

    def _resolve_base(self, item: dict) -> str | None:
        base = item.get("base") or "scratch"
        if base in ("scratch", ""):
            return None
        if base.startswith("local://"):
            return base[len("local://"):]
        if base.startswith("hf://"):
            ref = base[len("hf://"):]
            repo, _, rev = ref.partition("@")
            from huggingface_hub import hf_hub_download   # warm-start (no buffer)
            return hf_hub_download(repo, "model.pt", revision=rev or None)
        return base  # a bare local path

    # ---- ephemeral code-only worktree -----------------------------------

    def _checkout(self, commit: str | None, row_id: str) -> Path:
        work = Path(daemon.home()) / "worktrees" / row_id.replace("/", "_")
        work.parent.mkdir(parents=True, exist_ok=True)
        target = commit or "HEAD"
        r = subprocess.run(
            ["git", "-C", str(self.repo_root), "worktree", "add", "--detach",
             str(work), target], capture_output=True, text=True)
        if r.returncode != 0:
            raise SliceFailed(f"git worktree add {target}: {r.stderr.strip()}")
        return work

    def _teardown(self, work: Path) -> None:
        subprocess.run(["git", "-C", str(self.repo_root), "worktree", "remove",
                        "--force", str(work)], capture_output=True, text=True)

    # ---- run the bundle -------------------------------------------------

    def _run_slice(self, work: Path, cell_key: str, run_base: Path,
                   resume: str | None, cap: float,
                   board_size: int | None = None) -> None:
        run_base.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(work / "scripts" / "run_sweep.py"),
               "--cell", cell_key, "--max-wall-secs", str(cap),
               "--final-eval", "--foreground"]
        if resume:
            cmd += ["--resume", str(resume)]
        env = dict(os.environ)
        env["GOMOKU_RUN_DIR"] = str(run_base)
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if board_size is not None:
            env["GOMOKU_BOARD_SIZE"] = str(board_size)
        r = subprocess.run(cmd, cwd=str(work), env=env)
        if r.returncode != 0:
            raise SliceFailed(f"run_sweep --cell {cell_key} exited {r.returncode}")

    def _find_ckpt_dir(self, run_base: Path) -> Path | None:
        return next((run_base / "sweep_runs").glob("*/checkpoints"), None)

    def _read_model_elo(self, ckpt_dir: Path) -> float | None:
        jsonl = ckpt_dir / "eval_results.jsonl"
        if not jsonl.exists():
            return None
        lines = [ln for ln in jsonl.read_text().splitlines() if ln.strip()]
        if not lines:
            return None
        try:
            return json.loads(lines[-1]).get("eval/model_elo")
        except ValueError:
            return None

    # ---- deliver + flywheel ---------------------------------------------

    def _deliver(self, item: dict, lane: str, latest: Path, elo) -> str:
        if self.dry_run:
            return f"local://{latest}"
        row_id = item["id"]
        provenance = {"git_sha": item.get("commit"), "ledger_row_id": row_id,
                      "base": item.get("base"), "lane": lane, "model_elo": elo}
        return hf.push_slice(latest, repo_id=self.hf_repo,
                             revision=f"{lane}-{row_id}".replace("/", "_"),
                             provenance=provenance)

    def _followups(self, item: dict, lane: str, artifact: str, elo,
                   config: dict) -> list[dict]:
        n = int(config.get("seq_n", 0)) + 1
        # the continuation resumes the lane's own latest.pt (true resume, has buffer)
        ckpt = self._find_ckpt_dir(Path(daemon.home()) / "runs" / lane)
        cont_base = f"local://{ckpt/'latest.pt'}" if ckpt else item.get("base", "scratch")
        prio = int(item.get("priority", 0) or 0)
        return [
            ledger.experiment(
                id=f"{lane}@{n}", role=ROLE, commit=item.get("commit"),
                base=cont_base, config={**config, "lane": lane, "seq_n": n},
                priority=prio, note=f"continuation of {item['id']}"),
            ledger.experiment(
                id=f"eval-{item['id']}".replace("/", "_"), role="arena",
                commit=item.get("commit"), base=artifact,
                config={"lane": lane, "model_elo": elo},
                priority=prio + 1, note=f"eval slice {item['id']}"),
        ]


# ---- CLI / entry point --------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=daemon.default_ledger_path())
    ap.add_argument("--once", action="store_true", help="run a single slice then exit")
    ap.add_argument("--no-hf", action="store_true", help="dry-run: don't push to HF")
    ap.add_argument("--prod", action="store_true",
                    help="production 1h cap (default: MVP 1-epoch cap)")
    ap.add_argument("--stop-file")
    args = ap.parse_args(argv)
    role = TrainerRole(dry_run=args.no_hf, mvp_mode=not args.prod)
    return daemon.run_daemon(role, args.ledger, once=args.once, stop_file=args.stop_file)


if __name__ == "__main__":
    sys.exit(main())
