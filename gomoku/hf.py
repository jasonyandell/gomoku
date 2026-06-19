"""HuggingFace Hub publishing for gomoku model snapshots.

Pushes a slimmed checkpoint (weights + config + provenance) to a public
HF model repo. The training checkpoint contains the optimizer state and
replay buffer which together account for ~60MB of size that is only
useful for resuming training, never for inference. We strip them here
before publishing.

Mirrors the pattern in ``~/code/mk5-main/forge/zeb/export_model.py`` and
``~/code/mk5-main/forge/zeb/hf.py``, but single-machine and minimal.

CLI:
    python -m gomoku.hf push --checkpoint checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

# Suppress noisy progress bars in scripted contexts (matches zeb).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


DEFAULT_REPO = "jasonyandell/gomoku-9x9"


# Keys we keep when slimming. Everything else (optimizer_state_dict,
# replay_buffer, scheduler_state_dict, ...) is dropped.
_SLIM_KEYS_REQUIRED = ("model_state_dict", "model_config")
_SLIM_KEYS_OPTIONAL = ("epoch", "total_games", "wandb_run_id")


def _require_hf():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for gomoku.hf — `uv pip install huggingface_hub`"
        ) from e


def slim_checkpoint(src_path: str | Path, dst_path: str | Path) -> dict:
    """Load a training checkpoint and save a slim, inference-only snapshot.

    Keeps: ``model_state_dict``, ``model_config``, and the metadata fields
    (``epoch``, ``total_games``, ``wandb_run_id`` if present).

    Drops: ``optimizer_state_dict``, ``replay_buffer``, and anything else.

    Returns the snapshot dict that was written, for inspection/logging.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)

    for k in _SLIM_KEYS_REQUIRED:
        if k not in ckpt:
            raise KeyError(f"checkpoint {src_path} missing required key {k!r}")

    snapshot: dict = {
        "model_state_dict": ckpt["model_state_dict"],
        "model_config": ckpt["model_config"],
    }
    for k in _SLIM_KEYS_OPTIONAL:
        if k in ckpt and ckpt[k] is not None:
            snapshot[k] = ckpt[k]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, dst_path)
    return snapshot


# --- Model card -------------------------------------------------------------

_MODEL_CARD = """\
---
library_name: pytorch
license: mit
tags:
  - alphazero
  - mcts
  - gomoku
  - reinforcement-learning
  - self-play
---

# gomoku-9x9

AlphaZero for 9x9 free-style gomoku (five-in-a-row, no opening restrictions).

Trained on a single Apple Silicon machine using PyTorch + MPS. Source code:
https://github.com/jasonyandell/gomoku

## How to load

```python
import torch
from huggingface_hub import hf_hub_download
from gomoku.model import load_checkpoint

path = hf_hub_download("jasonyandell/gomoku-9x9", "model.pt")
model, payload = load_checkpoint(path, device="cpu")
model.eval()

# payload["epoch"], payload["total_games"], payload["model_config"]
```

For a quick value/policy on the empty board:

```python
from gomoku.game import GameState
x = torch.from_numpy(GameState.initial().to_planes()).unsqueeze(0)
with torch.no_grad():
    policy_logits, value = model(x)
```

## Architecture

- Input: ``(B, 3, 9, 9)`` — own stones, opponent stones, side-to-move plane
- Backbone: small ResNet (preset ``small``: 64 filters x 4 residual blocks)
- Policy head: 81-way logits over board squares
- Value head: scalar in ``[-1, 1]`` via ``tanh``
- ~316k parameters at the ``small`` preset (see ``gomoku/model.py``)

## Training setup

- Apple Silicon (M5 Max) using PyTorch with the MPS backend
- Single-process AlphaZero loop (self-play -> replay buffer -> train)
- 200 MCTS simulations per move during self-play
- ~28 seconds per training epoch
- Checkpoints written every 5 epochs to ``checkpoints/``

## Current strength (honest)

Training is ongoing. These checkpoints reflect a small single-machine run on
Apple Silicon, not a production-grade AlphaZero engine. Strength scales with
checkpoint epoch — see ``training_state.json`` in this repo for the epoch
this snapshot corresponds to.

## Files

- ``model.pt`` — slimmed checkpoint (weights + ``model_config`` + provenance).
  Compatible with ``gomoku.model.load_checkpoint``.
- ``config.json`` — model architecture config as JSON.
- ``training_state.json`` — ``{epoch, total_games, wandb_run_id?}`` for the
  current snapshot.

## License

MIT.
"""


def _write_model_card(path: Path) -> None:
    path.write_text(_MODEL_CARD)


# --- Push -------------------------------------------------------------------


def push_checkpoint(
    ckpt_path: str | Path,
    repo_id: str = DEFAULT_REPO,
    filename: str | None = None,
    create_if_missing: bool = True,
) -> str:
    """Slim ``ckpt_path`` and upload it to ``repo_id`` on HuggingFace.

    Uploads three files in a single commit:
      - ``<filename>`` (default ``model.pt``): slim weights + config + meta
      - ``config.json``: ``model_config`` as JSON (for cheap inspection)
      - ``training_state.json``: ``{epoch, total_games, wandb_run_id?}``

    Also (re)writes ``README.md`` (the model card) so the repo lands with a
    sensible landing page on first push.

    Returns the repo URL.
    """
    _require_hf()
    from huggingface_hub import HfApi

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    filename = filename or "model.pt"

    api = HfApi()
    if create_if_missing:
        # repo_type="model", public.
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        slim_path = tmp_dir / filename
        snapshot = slim_checkpoint(ckpt_path, slim_path)

        model_config = snapshot["model_config"]
        epoch = int(snapshot.get("epoch", 0))
        total_games = int(snapshot.get("total_games", 0))
        wandb_run_id = snapshot.get("wandb_run_id")

        (tmp_dir / "config.json").write_text(json.dumps(model_config, indent=2))

        state: dict = {"epoch": epoch, "total_games": total_games}
        if wandb_run_id:
            state["wandb_run_id"] = wandb_run_id
        (tmp_dir / "training_state.json").write_text(json.dumps(state, indent=2))

        _write_model_card(tmp_dir / "README.md")

        commit_msg = f"epoch {epoch} ({total_games} games) -> {filename}"
        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg,
        )

    # huggingface_hub returns no URL from upload_folder; build the canonical one.
    return f"https://huggingface.co/{repo_id}"


def push_slice(
    ckpt_path: str | Path,
    repo_id: str = DEFAULT_REPO,
    *,
    revision: str,
    provenance: dict | None = None,
    tag: str | None = None,
    filename: str | None = None,
    create_if_missing: bool = True,
) -> str:
    """Slim ``ckpt_path`` and upload it to a per-slice REVISION (an HF branch).

    Unlike ``push_checkpoint`` (which overwrites ``main``), each autolab slice
    gets its own revision so the ledger's ``base_model`` refs stay durable over
    time. ``provenance`` (git_sha, ledger_row_id, base, lane, model_elo, Δelo) is
    recorded in ``training_state.json``. If ``tag`` is given (e.g. ``"champion"``)
    it is (re)pointed at this revision — that's how the arena bumps the champion
    on PROMOTE. Idempotent (``exist_ok``) so a re-picked slice can overwrite its
    own revision safely. Returns ``"repo_id@revision"``.
    """
    _require_hf()
    from huggingface_hub import HfApi

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    filename = filename or "model.pt"

    api = HfApi()
    if create_if_missing:
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.create_branch(repo_id=repo_id, branch=revision, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        slim_path = tmp_dir / filename
        snapshot = slim_checkpoint(ckpt_path, slim_path)

        model_config = snapshot["model_config"]
        epoch = int(snapshot.get("epoch", 0))
        total_games = int(snapshot.get("total_games", 0))
        wandb_run_id = snapshot.get("wandb_run_id")

        (tmp_dir / "config.json").write_text(json.dumps(model_config, indent=2))

        state: dict = {"epoch": epoch, "total_games": total_games}
        if wandb_run_id:
            state["wandb_run_id"] = wandb_run_id
        if provenance:
            state["provenance"] = provenance
        (tmp_dir / "training_state.json").write_text(json.dumps(state, indent=2))

        _write_model_card(tmp_dir / "README.md")

        commit_msg = f"slice {revision}: epoch {epoch} ({total_games} games)"
        if provenance and provenance.get("model_elo") is not None:
            commit_msg += f" elo={provenance['model_elo']}"
        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            commit_message=commit_msg,
        )

    if tag:
        api.create_tag(repo_id=repo_id, tag=tag, revision=revision,
                       repo_type="model", exist_ok=True)

    return f"{repo_id}@{revision}"


# --- CLI --------------------------------------------------------------------


def _cmd_push(args: argparse.Namespace) -> None:
    ckpt = Path(args.checkpoint).expanduser().resolve()
    # If it's a symlink (e.g. latest.pt -> epoch0060.pt) torch handles fine,
    # but we resolve so the log message points to the real file.
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    # Pre-slim once to a sibling tmp file so we can show size delta.
    with tempfile.TemporaryDirectory() as tmp:
        preview = Path(tmp) / (args.name or "model.pt")
        snap = slim_checkpoint(ckpt, preview)
        src_mb = ckpt.stat().st_size / 1e6
        dst_mb = preview.stat().st_size / 1e6
        print(f"slimmed {ckpt.name}: {src_mb:.1f} MB -> {dst_mb:.1f} MB")
        print(f"  epoch={snap.get('epoch')} total_games={snap.get('total_games')}")

    url = push_checkpoint(
        ckpt_path=ckpt,
        repo_id=args.repo,
        filename=args.name,
        create_if_missing=True,
    )
    print(f"pushed to: {url}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gomoku.hf")
    sub = p.add_subparsers(dest="cmd", required=True)

    push = sub.add_parser("push", help="slim + upload a checkpoint to HF")
    push.add_argument("--checkpoint", required=True, help="path to a training checkpoint (.pt)")
    push.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default {DEFAULT_REPO})")
    push.add_argument("--name", default=None, help="filename in the HF repo (default model.pt)")
    push.set_defaults(func=_cmd_push)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
