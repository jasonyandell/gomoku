#!/usr/bin/env python3
"""migrate_beads_to_gh.py — one-shot, lossless beads → GitHub issues migration.

Part of issue #3 (Phase 1 of the lab redesign, epic #2). Reads a manifest of
live beads (produced by `bd show --json` per bead) and recreates each as a
GitHub issue, preserving title/body/priority/status/routing as labels and
provenance in a footer. Idempotent: skips a bead whose id already appears in an
open issue body (safe to re-run after a partial failure).

Usage:
    python scripts/migrate_beads_to_gh.py [manifest.json] [--dry-run]

Mapping decisions (see issue #3):
- GitHub has only open/closed; bd's extra states become labels excluded from the
  ready query: deferred -> `deferred`, blocked -> `blocked`, in_progress ->
  `in-progress`.
- priority -> P0..P4; type epic/bug -> same, feature -> `enhancement`, else `task`.
- SKIP derby-o3s (already closed) and derby-1ks (duplicate of derby-s6j).
- REFRAME banner on the "GPU broker on the daemon" beads (derby-s6j, derby-ddl):
  superseded by the "re-derive fresh" decision in #4.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

SKIP = {"derby-o3s", "derby-1ks"}  # closed / duplicate-of-derby-s6j
REFRAME = {"derby-s6j", "derby-ddl"}  # "GPU broker on the daemon" -> see #4
TYPE_LABEL = {"epic": "epic", "bug": "bug", "feature": "enhancement"}
KNOWN_PASSTHROUGH = {
    "derby-idea", "runner-domain", "code-only", "gpu",
    "needs-live-validation", "decision-loop-ownership", "proposed", "human-gated",
}
REFRAME_BANNER = (
    "> **Reframed 2026-05-28:** the GPU substrate is being **re-derived fresh** "
    "(`scripts/gpu_worker.py`), not built on the parked `feat/gpu-broker` daemon. "
    "Reconcile/close against #4 (epic #2).\n\n"
)


def sh(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def already_migrated(bead_id: str) -> bool:
    # Match the unique provenance footer, NOT a bare id (planning docs mention
    # bead ids by name and would otherwise trigger false positives).
    r = sh(["gh", "issue", "list", "--state", "all", "--search",
            f'"Migrated from bead" "{bead_id}" in:body', "--json", "number",
            "--limit", "5"])
    if r.returncode != 0:
        return False
    try:
        return len(json.loads(r.stdout)) > 0
    except json.JSONDecodeError:
        return False


def labels_for(bead: dict) -> list[str]:
    out: list[str] = []
    pl = bead.get("priority_label")
    if pl:
        out.append(pl)
    t = (bead.get("type") or "").lower()
    out.append(TYPE_LABEL.get(t, "task"))
    st = (bead.get("status") or "").lower()
    if st == "deferred":
        out.append("deferred")
    elif st == "blocked":
        out.append("blocked")
    elif st == "in_progress":
        out.append("in-progress")
    for lb in bead.get("labels") or []:
        if lb in KNOWN_PASSTHROUGH and lb not in out:
            out.append(lb)
    return out


def body_for(bead: dict) -> str:
    parts: list[str] = []
    if bead["id"] in REFRAME:
        parts.append(REFRAME_BANNER)
    parts.append(bead.get("description") or "_(no description)_")
    if bead.get("notes"):
        parts.append("\n## Notes (migrated)\n\n" + bead["notes"])
    if bead.get("acceptance_criteria"):
        parts.append("\n## Acceptance criteria\n\n" + bead["acceptance_criteria"])
    rel = bead.get("relationships") or {}
    edges = []
    for kind in ("blocked_by", "blocks", "related"):
        ids = rel.get(kind) or []
        if ids:
            edges.append(f"- {kind.replace('_', ' ')}: " + ", ".join(f"`{i}`" for i in ids))
    if edges:
        parts.append("\n## Relationships (from beads)\n\n" + "\n".join(edges)
                     + "\n\n_(some referenced beads are closed and were not migrated)_")
    footer = (
        f"\n\n---\n<sub>Migrated from bead `{bead['id']}` "
        f"(created {bead.get('created_at')}, updated {bead.get('updated_at')}, "
        f"by {bead.get('created_by')})."
    )
    if bead.get("external_ref"):
        footer += f" external_ref: `{bead['external_ref']}`."
    footer += " 🤖 via scripts/migrate_beads_to_gh.py.</sub>"
    parts.append(footer)
    return "\n".join(parts)


def create_issue(bead: dict, dry: bool) -> int | None:
    title = bead["title"]
    labels = labels_for(bead)
    body = body_for(bead)
    if dry:
        print(f"  DRY {bead['id']:>10} -> [{', '.join(labels)}] {title!r}")
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(body)
        bf = f.name
    args = ["gh", "issue", "create", "--title", title, "--body-file", bf]
    for lb in labels:
        args += ["--label", lb]
    r = sh(args)
    os.unlink(bf)
    if r.returncode != 0:
        sys.stderr.write(f"  FAIL {bead['id']}: {r.stderr}\n")
        return None
    url = r.stdout.strip().splitlines()[-1]
    num = int(url.rsplit("/", 1)[-1])
    print(f"  #{num:<4} {bead['id']:>10} -> [{', '.join(labels)}] {title!r}")
    return num


def link_epics(manifest: list[dict], idmap: dict[str, int], dry: bool) -> None:
    for bead in manifest:
        children = (bead.get("relationships") or {}).get("children") or []
        live = [c for c in children if c in idmap]
        if not live or bead["id"] not in idmap:
            continue
        parent_num = idmap[bead["id"]]
        checklist = "\n".join(f"- [ ] #{idmap[c]}" for c in live)
        print(f"  link epic #{parent_num} ({bead['id']}) -> {[idmap[c] for c in live]}")
        if dry:
            continue
        # append a sub-issues checklist to the parent
        cur = sh(["gh", "issue", "view", str(parent_num), "--json", "body"])
        old = json.loads(cur.stdout).get("body", "") if cur.returncode == 0 else ""
        new = old + "\n\n## Sub-issues\n\n" + checklist
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(new)
            bf = f.name
        sh(["gh", "issue", "edit", str(parent_num), "--body-file", bf])
        os.unlink(bf)
        # add a "Part of #parent" note to each child
        for c in live:
            cnum = idmap[c]
            cur = sh(["gh", "issue", "view", str(cnum), "--json", "body"])
            cold = json.loads(cur.stdout).get("body", "") if cur.returncode == 0 else ""
            sh(["gh", "issue", "edit", str(cnum), "--body",
                f"Part of #{parent_num}.\n\n" + cold])


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    dry = "--dry-run" in argv
    manifest_path = args[0] if args else \
        "/Users/jason/.claude/jobs/652df966/tmp/beads_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"loaded {len(manifest)} beads from {manifest_path} (dry={dry})")
    idmap: dict[str, int] = {}
    created = skipped = 0
    for bead in manifest:
        bid = bead.get("id")
        if not bid or bead.get("error"):
            continue
        if bid in SKIP:
            print(f"  skip {bid} (closed/duplicate)")
            skipped += 1
            continue
        if not dry and already_migrated(bid):
            print(f"  skip {bid} (already migrated)")
            skipped += 1
            continue
        num = create_issue(bead, dry)
        if num is not None:
            idmap[bid] = num
            created += 1

    print(f"\nphase 1 (create): {created} created, {skipped} skipped")
    print("phase 2 (epic links):")
    link_epics(manifest, idmap, dry)
    print("\nid -> issue map:")
    for k, v in idmap.items():
        print(f"  {k} -> #{v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
