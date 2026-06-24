#!/usr/bin/env python3
"""Publish the local engines/rapfi build to a pinned HF repo for friction-free
resolution by ``gomoku.rapfi_pool``.

Run this ONCE per Rapfi build (it's the only human-gated step). It uploads the
arm64 binary + config + NNUE weights to a PRIVATE HF repo in a single atomic
commit, then prints the commit SHA + binary sha256 to paste into
``gomoku/rapfi_pool.py``'s ``RAPFI_HF_REVISION`` / ``RAPFI_BINARY_SHA256`` pins.
After that, any worktree / fresh machine resolves the engine from the
machine-global hub cache with zero manual steps.

    python scripts/publish_rapfi.py                      # public (default), tag v1
    python scripts/publish_rapfi.py --private            # private repo instead

Public (default) allows token-free pulls on a fresh machine and is a clean public
GPL redistribution (corresponding source = the upstream commit, cited in the model
card). `--private` keeps the artifacts off a public mirror but then needs your HF
token to pull (cached on a machine you've `huggingface-cli login`'d).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Everything the engine needs at runtime, co-located so config.toml's bare
# filename weight refs resolve from one snapshot dir.
ARTIFACTS = [
    "pbrain-rapfi",
    "config.toml",
    "model210901.bin",
    "mix9svqfreestyle_bsmix.bin.lz4",
    "mix9svqrenju_bs15_black.bin.lz4",
    "mix9svqrenju_bs15_white.bin.lz4",
    "mix9svqstandard_bs15.bin.lz4",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="jasonyandell/rapfi-arm64")
    ap.add_argument("--engines-dir", default=None,
                    help="defaults to <this repo>/engines/rapfi")
    ap.add_argument("--private", action="store_true", default=False,
                    help="publish a PRIVATE repo (default: public)")
    ap.add_argument("--tag", default=None, help="optional human-readable tag")
    ap.add_argument("--message", default="publish rapfi arm64 build")
    args = ap.parse_args(argv)

    edir = Path(args.engines_dir) if args.engines_dir else (
        Path(__file__).resolve().parent.parent / "engines" / "rapfi"
    )
    missing = [a for a in ARTIFACTS if not (edir / a).is_file()]
    if missing:
        print(f"ERROR: missing artifacts in {edir}: {missing}\n"
              f"  build them first with engines/rapfi/build_rapfi.sh", file=sys.stderr)
        return 2

    binp = edir / "pbrain-rapfi"
    bsha = _sha256(binp)
    provenance = ""
    if (edir / "BUILD_COMMIT.txt").is_file():
        provenance = (edir / "BUILD_COMMIT.txt").read_text().strip()

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    who = api.whoami()["name"]
    print(f"HF user: {who}  |  repo: {args.repo}  |  private={args.private}")

    api.create_repo(repo_id=args.repo, repo_type="model",
                    private=args.private, exist_ok=True)

    card = (
        "---\nlicense: gpl-3.0\ntags: [gomoku, rapfi, engine, arm64, eval]\n---\n\n"
        "# Rapfi (arm64) — eval/teacher artifact mirror\n\n"
        "An arm64 macOS build of [dhbloo/rapfi](https://github.com/dhbloo/rapfi), "
        "mirrored so `gomoku.rapfi_pool` can resolve it friction-free from any "
        "worktree/machine (pinned by commit SHA, sha256-verified). NOT upstream; "
        "redistributed for personal eval use.\n\n"
        f"```\n{provenance}\n```\n\n"
        f"binary `pbrain-rapfi` sha256: `{bsha}`\n"
    )

    ops = [CommitOperationAdd(path_in_repo=a, path_or_fileobj=str(edir / a))
           for a in ARTIFACTS]
    ops.append(CommitOperationAdd(path_in_repo="README.md",
                                  path_or_fileobj=card.encode("utf-8")))

    info = api.create_commit(repo_id=args.repo, repo_type="model",
                             operations=ops, commit_message=args.message)
    commit_sha = getattr(info, "oid", None) or getattr(info, "commit_id", None)
    if args.tag:
        api.create_tag(repo_id=args.repo, tag=args.tag, revision=commit_sha,
                       repo_type="model", exist_ok=True)

    print("\n=== published — paste these into gomoku/rapfi_pool.py ===")
    print(f'RAPFI_HF_REPO = "{args.repo}"')
    print(f'RAPFI_HF_REVISION = "{commit_sha}"')
    print(f'RAPFI_BINARY_SHA256 = "{bsha}"')
    if args.tag:
        print(f"(also tagged: {args.tag} -> {commit_sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
