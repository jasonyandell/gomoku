#!/usr/bin/env python3
"""Safe-resume helper for a derby cell on an existing weights-only peak.pt.

THE PROBLEM this prevents (burned 2026-05-28): launching a derby on a pre-existing
cell whose `derby_state.json` has `wall_secs_total == 0` makes the FIRST CHUNK run
FRESH (no --resume), silently OVERWRITING the cell's existing `latest.pt` with a
brand-new seed-0 trainer state. The champion's epoch-2848 latest.pt was clobbered
to epoch 12 in seconds. Permanent data loss except for whatever was saved as a
peak.pt round-robin anchor.

THE PROTOCOL this enforces (all four steps, atomic):
  1. Archive any junk artifacts already in the cell's checkpoint dir.
  2. Restore `peak.pt` → `<cell>/checkpoints/latest.pt` with the embedded
     `wandb_run_id` CLEARED (→ trainer starts a FRESH wandb run, permanently
     dodging the run-id-poisoning crash-loop).
  3. Pre-populate the derby's `derby_state.json` with the lane's idea entry and
     `wall_secs_total = 1.0` so the engine's `resume = wall_secs_total > 0` check
     trips True on the first chunk — that's what forces `--resume <latest.pt>`.
  4. Back up the restored `latest.pt` to `$CLAUDE_JOB_DIR` (defense in depth).

After running this, launch the derby normally. The first chunk will resume from
the peak.pt's weights with a cold (empty) buffer; the trainer self-refills it.

Usage:
  python scripts/derby_safe_resume.py \\
      --cell derby-v7-mate-discount \\
      --peak sweep_runs/derby_v8/_peaks/mate-discount/peak.pt \\
      --board scripts/derby_champ_board.json \\
      --idea-name champ

  # Then verify + launch (the script prints these for you):
  python scripts/delo_derby.py --board scripts/derby_champ_board.json --dry-run
  nohup python scripts/delo_derby.py --board scripts/derby_champ_board.json \\
      --resume >> "$CLAUDE_JOB_DIR/<board>.log" 2>&1 &
  # IMMEDIATELY assert --resume is in the trainer cmdline:
  ps -o command= $(pgrep -f gomoku.train) | grep -oE '\\-\\-resume [^ ]+'
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def archive_junk(ckpt_dir: Path, dest: Path) -> int:
    """Move any pre-existing latest.pt / epochNNNN.pt / eval_results.jsonl /
    worker_weights.pt out of the cell's checkpoint dir into `dest`. Returns
    count of files moved. Idempotent (no-op if nothing to archive)."""
    dest.mkdir(parents=True, exist_ok=True)
    patterns = ["latest.pt", "epoch*.pt", "eval_results.jsonl", "worker_weights.pt"]
    moved = 0
    for pat in patterns:
        for src in ckpt_dir.glob(pat):
            shutil.move(str(src), str(dest / src.name))
            moved += 1
    return moved


def restore_peak_with_cleared_run_id(peak_src: Path, latest_dst: Path) -> dict:
    """Load peak.pt, clear its embedded wandb_run_id (→ fresh run on next launch),
    save as latest.pt. Atomic via tmp + rename. Returns the loaded payload's
    metadata (epoch, old run id) for logging."""
    import torch  # heavy local import
    payload = torch.load(peak_src, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise SystemExit(f"refusing: {peak_src} is not a dict-style checkpoint")
    old_run_id = payload.get("wandb_run_id")
    epoch = payload.get("epoch")
    payload["wandb_run_id"] = None
    latest_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest_dst.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, latest_dst)
    return {"epoch": epoch, "old_wandb_run_id": old_run_id}


def prepopulate_state(state_path: Path, idea_name: str, epoch: int | None,
                      peak_elo: float, peak_path: Path) -> None:
    """Write a minimal derby_state.json that trips `resume=True` on the first
    chunk (wall_secs_total=1.0), with the lane's idea entry pre-seeded with the
    peak it's resuming from. Overwrites any existing state."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "ideas": {
            idea_name: {
                "name": idea_name,
                "epochs_done": int(epoch or 0),
                "elo_history": [],
                "last_delo": 0.0,
                "wall_secs_total": 1.0,   # >0 → engine triggers --resume on first chunk
                "chunks_done": 0,
                "peak_elo": float(peak_elo),
                "peak_path": str(peak_path),
                "status": "queued",
                "wandb_run_id": None,     # cleared; trainer creates a fresh run
                "retries": 0,
                "last_picked": 0,
            }
        },
        "total_chunks_run": 0,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, state_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True,
                    help="The run_sweep.CELLS key whose checkpoint dir we're seeding "
                         "(e.g. derby-v7-mate-discount).")
    ap.add_argument("--peak", required=True, type=Path,
                    help="Source peak.pt path (typically <other-board>/_peaks/<idea>/peak.pt).")
    ap.add_argument("--board", required=True, type=Path,
                    help="Path to the derby board json that will resume this cell.")
    ap.add_argument("--idea-name", default=None,
                    help="The board's idea name for this lane (defaults to --cell).")
    ap.add_argument("--peak-elo", type=float, default=0.0,
                    help="Anchored elo of the source peak (just for state's peak_elo field; "
                         "the derby will reset on the first eval).")
    ap.add_argument("--force", action="store_true",
                    help="Proceed even if the cell's checkpoint dir is non-empty "
                         "(archives the contents).")
    args = ap.parse_args()

    if not args.peak.exists():
        raise SystemExit(f"--peak does not exist: {args.peak}")
    if not args.board.exists():
        raise SystemExit(f"--board does not exist: {args.board}")

    # Resolve the cell's checkpoint dir + the board's state path from the board json.
    board = json.loads(args.board.read_text())
    base_out_dir = REPO_ROOT / board["global"]["base_out_dir"]
    state_path = base_out_dir / "derby_state.json"
    ckpt_dir = REPO_ROOT / "sweep_runs" / args.cell / "checkpoints"

    idea_name = args.idea_name or args.cell
    print(f"[safe-resume] cell={args.cell} board={args.board.name}")
    print(f"[safe-resume]   peak src    = {args.peak}")
    print(f"[safe-resume]   ckpt dir    = {ckpt_dir.relative_to(REPO_ROOT)}")
    print(f"[safe-resume]   state file  = {state_path.relative_to(REPO_ROOT)}")
    print(f"[safe-resume]   idea name   = {idea_name}")

    # Refuse to clobber unless --force.
    existing = list(ckpt_dir.glob("*")) if ckpt_dir.exists() else []
    if existing and not args.force:
        print(f"\n[safe-resume] REFUSING: ckpt dir is non-empty ({len(existing)} files).")
        print("[safe-resume] Re-run with --force to archive the existing contents first.")
        sys.exit(2)

    # 1. Archive any junk.
    job_dir = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp"))
    archive_dst = job_dir / f"{args.cell}_archived_{int(time.time())}"
    moved = archive_junk(ckpt_dir, archive_dst) if existing else 0
    print(f"\n[safe-resume] 1/4 archived {moved} files → {archive_dst}")

    # 2. Restore peak with cleared run id.
    latest_dst = ckpt_dir / "latest.pt"
    meta = restore_peak_with_cleared_run_id(args.peak, latest_dst)
    print(f"[safe-resume] 2/4 restored peak → latest.pt "
          f"(epoch={meta['epoch']}, old run_id={meta['old_wandb_run_id']} → None)")

    # 3. Pre-populate state with wall_secs_total > 0 (the critical guard).
    prepopulate_state(state_path, idea_name, meta["epoch"],
                      args.peak_elo, args.peak)
    print(f"[safe-resume] 3/4 wrote derby_state.json with wall_secs_total=1.0 "
          f"(forces --resume on first chunk)")

    # 4. Back up the restored latest.pt.
    backup = job_dir / f"{args.cell}_recovered_latest.bak.pt"
    shutil.copy2(latest_dst, backup)
    print(f"[safe-resume] 4/4 backed up restored latest.pt → {backup}")

    print(f"\n[safe-resume] ✓ READY. Next steps (copy/paste):\n")
    print(f"  # Verify the engine would --resume (NOT 'fresh') for this lane:")
    print(f"  python scripts/delo_derby.py --board {args.board} --dry-run 2>&1 | grep -E 'IDEA|resumes from'")
    print(f"\n  # Launch derby + immediately confirm --resume is in the trainer cmdline:")
    print(f"  nohup python scripts/delo_derby.py --board {args.board} --resume \\")
    print(f"      >> \"$CLAUDE_JOB_DIR/{args.board.stem}.log\" 2>&1 &")
    print(f"  sleep 6")
    print(f"  ps -o command= $(pgrep -f gomoku.train) | grep -oE -- '--resume [^ ]+'")
    print(f"  # ^ MUST print '--resume .../latest.pt'. If empty → KILL IMMEDIATELY, you're about to clobber.")
    print(f"\n  # Then watchdog:")
    print(f"  nohup bash scripts/derby_watchdog.sh {args.board} >/dev/null 2>&1 &")


if __name__ == "__main__":
    main()
