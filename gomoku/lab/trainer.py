#!/usr/bin/env python3
"""The autolab trainer role (epic #53, P3).

Plugs into ``daemon.run_daemon``. One ``run_chunk`` = one bounded training slice:
resolve the start model, **materialize the ledger row's ``commit`` as a standalone
source snapshot (``git archive``) and run the proven ``run_sweep`` bundle inside it
via ``uv run``** (1 trainer + N selfplay workers), with its DATA pointed at
``~/data/autolab/runs/<lane>/`` (so teardown can't delete checkpoints), read the
eval, deliver a per-slice HF revision, and emit the flywheel follow-ups
(continuation slice + arena eval).

The trainer owns no loop and no lock — the daemon does. It is pure work.

Execution = ``uv`` (doctrine §2/§5, ``wiki/topics/autolab-doctrine.md``). The env
is **reconstructed from the ref**, never persisted: ``git archive`` gives a tree
with no shared ``.git``, and ``uv run`` syncs that tree's own pinned deps + builds
its C extensions, so the slice runs the *commit's* ``gomoku`` — not main's
editable install. This kills the shared-editable-install worktree gotcha
structurally: "a commit" finally means a commit. (No ``uv.lock`` is committed yet,
so deps resolve from ``pyproject`` per run — see the wiki for the lock follow-up.)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
        cap = min(float(cap), self.prod_cap_secs)  # 1h hard ceiling; ledger rows are untrusted input
        board_size = config.get("board_size")

        # Board size is a process-start constant; a row from another era must not
        # run in this process (it would silently train the wrong geometry). The
        # era-cross retire normally prevents the pick; this is the loud backstop.
        proc_bs = os.environ.get("GOMOKU_BOARD_SIZE")
        if board_size is not None and proc_bs is not None and int(board_size) != int(proc_bs):
            raise SliceFailed(f"board-size mismatch: row {board_size} != process {proc_bs}")

        # Reclaim any worktree a prior slice leaked (SIGKILL/OOM skipped teardown,
        # or a since-abandoned lane left its dir). The flock singleton guarantees
        # no live sibling, so anything here is stale; the run DATA lives in runs/.
        self._reclaim_stale_worktrees()
        work = self._checkout(item.get("commit"), row_id)
        try:
            self._run_slice(work, cell_key, run_base, resume, cap, board_size)
            ckpt_dir = self._find_ckpt_dir(run_base)
            latest = ckpt_dir / "latest.pt" if ckpt_dir else None
            if latest is None or not latest.exists():
                raise SliceFailed(f"no latest.pt under {run_base}/sweep_runs/*/checkpoints")
            epochs = len(list(ckpt_dir.glob("epoch*.pt")))
            elo = self._read_model_elo(ckpt_dir)
            metrics = {"eval/model_elo": elo, "epochs_ran": epochs, "lane": lane,
                       "cell": cell_key}
            # HF push is the most fragile step (4 un-retried network calls) but it
            # is NOT load-bearing for correctness: latest.pt is the true resume. A
            # transient blip must not mark a fully-trained slice FAILED and drop
            # its flywheel follow-ups. Defer delivery to a local ref + an event.
            try:
                artifact = self._deliver(item, lane, latest, elo)
            except Exception as e:
                artifact = f"local://{latest}"
                metrics["hf_push_deferred"] = f"{type(e).__name__}: {e}"
            followups = self._followups(item, lane, artifact, elo, config)
            if "hf_push_deferred" in metrics:
                followups.append(ledger.event(
                    scope=ROLE,
                    summary=f"HF push deferred for {row_id}: {metrics['hf_push_deferred']}",
                    data={"item": row_id, "artifact": artifact}))
            return daemon.ChunkResult(
                artifact_ref=artifact, metrics=metrics, buffer=f"local://{latest}",
                followups=followups)
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

    # ---- ephemeral source snapshot (git archive, NOT a worktree) --------

    def _checkout(self, commit: str | None, row_id: str) -> Path:
        """Materialize the row's commit as a standalone source tree via
        ``git archive`` — no shared ``.git``, so nothing ties this tree's
        ``gomoku`` to main's editable install (the worktree gotcha is gone). The
        dir is named per-row under ``worktrees/`` (the path the reclaim + the
        simulator bound); ``uv run`` later reconstructs its env from the ref."""
        work = Path(daemon.home()) / "worktrees" / row_id.replace("/", "_")
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        target = commit or "HEAD"
        ar = subprocess.run(["git", "-C", str(self.repo_root), "archive", target],
                            capture_output=True)
        if ar.returncode != 0:
            raise SliceFailed(
                f"git archive {target}: {ar.stderr.decode('utf-8', 'replace').strip()[:200]}")
        ex = subprocess.run(["tar", "-x", "-C", str(work)], input=ar.stdout,
                            capture_output=True)
        if ex.returncode != 0:
            raise SliceFailed(
                f"tar extract {target}: {ex.stderr.decode('utf-8', 'replace').strip()[:200]}")
        return work

    def _teardown(self, work: Path) -> None:
        """Remove the ephemeral source tree (+ its ``uv`` ``.venv``). Plain
        rmtree: an archive tree is not a registered worktree, so there's no git
        state to unwind — idempotent and crash-safe."""
        shutil.rmtree(work, ignore_errors=True)

    def _reclaim_stale_worktrees(self) -> list[str]:
        """Remove every source tree under the autolab home (singleton ⇒ all
        stale). Idempotent; called at slice start so a crash/abandon can't leak
        disk. No ``git worktree prune`` needed — these were never worktrees."""
        wt = Path(daemon.home()) / "worktrees"
        if not wt.exists():
            return []
        reclaimed = []
        for d in sorted(wt.iterdir()):
            if d.is_dir():
                self._teardown(d)
                reclaimed.append(d.name)
        return reclaimed

    # ---- run the bundle -------------------------------------------------

    def _run_slice(self, work: Path, cell_key: str, run_base: Path,
                   resume: str | None, cap: float,
                   board_size: int | None = None) -> None:
        """Run the ``run_sweep`` bundle via ``uv run`` INSIDE the archived tree,
        so it executes that ref's own ``gomoku`` (deps + freshly-built C
        extensions reconstructed by uv from the pinned sources), never main's
        editable install. DATA stays external via ``GOMOKU_RUN_DIR``."""
        run_base.mkdir(parents=True, exist_ok=True)
        cmd = ["uv", "run", "python", "scripts/run_sweep.py",
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
            raise SliceFailed(f"uv run run_sweep --cell {cell_key} exited {r.returncode}")

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
        # Continuation policy (autolab-researcher-contract §3): a CONTINUOUS lane
        # (the seed/production lane — the default) rolls straight on; an exploratory
        # fork's continuation is BLOCKED_FOR_DECISION so the researcher's judgment is
        # causally upstream of the next GPU hour (the research resume unblocks/parks
        # it). The arena eval is NOT blocked — it is the evidence the decision needs.
        review = config.get("review_policy") or (
            "after_each_slice" if config.get("research") else "continuous")
        if review == "continuous":
            block = False
        elif review == "after_budget":      # run free until the budget, then hold
            ms = (config.get("budget") or {}).get("max_slices")
            block = bool(ms and n >= int(ms))
        else:                               # after_each_slice (the fork default)
            block = True
        cont_status = ledger.BLOCKED if block else ledger.OPEN
        cont_note = f"continuation of {item['id']}" + (
            "" if cont_status == ledger.OPEN else " (BLOCKED_FOR_DECISION — awaiting research review)")
        return [
            ledger.experiment(
                id=f"{lane}@{n}", role=ROLE, commit=item.get("commit"),
                base=cont_base, config={**config, "lane": lane, "seq_n": n},
                priority=prio, status=cont_status, note=cont_note),
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
