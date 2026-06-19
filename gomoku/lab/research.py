#!/usr/bin/env python3
"""The autolab research-lite tick (epic #53, P5 #61 — partial).

One deterministic ``--once`` tick — NOT an LLM agent, NOT a ``Role`` (no loop, no
flock). Run on a launchd ``StartInterval=1800``. Its only job:

  1. fold the ledger, summarize the seed lane (elo series → proxy Δelo/hr secant,
     last verdicts, plies signal, queue depth, trainer liveness),
  2. gather idea seeds from open GitHub ``derby-idea`` issues (``gh`` shelled out
     under try/except — no gh / offline ⇒ ``[]``, never a crash),
  3. propose AT MOST 2 new ``train`` experiment rows, each at a priority
     **STRICTLY BELOW** the seed lane's live max train priority (so the trainer's
     ``pick('train')`` ALWAYS still returns the seed/continuation — continuity
     over thrash on night 1) with idempotent stable ids,
  4. append those rows + one ``event`` row recording the tick,
  5. write an honest "current thinking" note to ``research/latest.md`` (overwrite)
     and append a timestamped block to ``research/NOTES.md``.

THE HONESTY CONSTRAINT (the #61 caveat). Every ranking here is on **proxies** — a
per-slice Δelo/hr secant from the trainer's own ≥20-game final-eval (±100-elo
noisy) — NOT a real wall-clock-to-Δelo gate (unbuilt; GitHub #61). The note
carries that banner literally. A hint, never a verdict.

Stdlib only (``subprocess`` for ``gh`` is fine); imports ``ledger`` + ``daemon``.
No torch / HF / network (other than the optional, fail-soft ``gh`` call).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import daemon, ledger

ROLE = "train"
GATE_OPEN_ISSUE = 61            # the unbuilt wall-clock-to-Δelo gate
MAX_PROPOSALS = 2               # ≤2 new train rows per tick
RESEARCH_QUEUE_CAP = 3          # K: don't pile research rows past this many open

# Ideas the researcher knows how to turn into a `train` row map an idea slug to an
# EXISTING run_sweep `--cell`. Code-heavy derby ideas stay NOTE-ONLY (carried by
# their GitHub issue) — we never file a `train` row for a cell that doesn't exist.
# Keyed conservatively: only proven 9x9 cells the trainer can actually run today.
KNOWN_CELLS: dict[str, str] = {
    "v9-small": "derby-v9-small",
    "derby-v9-small": "derby-v9-small",
    "mate-discount": "derby-v7-mate-discount",
    "derby-v7-mate-discount": "derby-v7-mate-discount",
}

# The honest banner, verbatim (the contract requires the proxy caveat in the note).
_HONESTY_BANNER = (
    "> WARNING HONESTY: ranks on PROXIES (per-slice Δelo/hr secant from the "
    "trainer's own >=20-game eval, +/-100-elo noisy), NOT a real "
    "wall-clock-to-Δelo gate. The anchored MTTE/EPWH gate is UNBUILT — GitHub "
    "#61, delta_e_harness.py. Every ranking below is a hint, not a verdict."
)


# ---- the digest ---------------------------------------------------------

@dataclass
class LaneDigest:
    """Per seed-lane proxy digest (one lane = the highest-priority train lane)."""

    lane: str = ""
    cell: str = ""
    n_slices: int = 0
    elo_series: list[float] = field(default_factory=list)
    wall_series: list[float] = field(default_factory=list)
    delta_elo_per_hr: float | None = None
    base_for_fork: str = "scratch"


@dataclass
class Summary:
    """A pure digest of the whole ledger for this tick (no I/O)."""

    has_results: bool = False
    trainer_alive: bool = False
    trainer_item: str | None = None
    lane: LaneDigest = field(default_factory=LaneDigest)
    last_verdicts: list[dict] = field(default_factory=list)
    plies_series: list[float] = field(default_factory=list)
    open_train: int = 0
    open_research: int = 0
    open_arena: int = 0
    p_seed: int | None = None      # max priority among open|claimed train rows
    open_research_ids: list[str] = field(default_factory=list)


# ---- helpers ------------------------------------------------------------

def _is_research_row(e: dict) -> bool:
    return bool(e.get("config", {}).get("research")) or str(e.get("id", "")).startswith("research-")


def _open_or_claimed_train(state: ledger.LedgerState) -> list[dict]:
    return [e for e in state.experiments.values()
            if e.get("role") == ROLE and e.get("status") in (ledger.OPEN, ledger.CLAIMED)]


def _seed_priority(state: ledger.LedgerState) -> int | None:
    """P_seed = max priority among current open|claimed train rows (live each tick).

    Computed over ALL open/claimed train rows including prior research proposals —
    but research rows live strictly below the seed by construction, so the max is
    always the seed/continuation lane's band. None when no open/claimed train row
    exists (cold start)."""
    prios = [int(e.get("priority", 0) or 0) for e in _open_or_claimed_train(state)]
    return max(prios) if prios else None


def _result_metrics(e: dict) -> dict:
    return ((e.get("result") or {}).get("metrics") or {})


def _completed_train(state: ledger.LedgerState) -> list[dict]:
    """DONE train rows with a result, ordered by seq (oldest first)."""
    rows = [e for e in state.experiments.values()
            if e.get("role") == ROLE and "result" in e
            and e.get("status") == ledger.DONE]
    return sorted(rows, key=lambda e: e.get("seq", 0))


def _seed_lane_name(state: ledger.LedgerState) -> str | None:
    """Pick the lane the researcher is reasoning about: the lane of the
    highest-priority open|claimed (non-research) train row, else the most recent
    completed train lane."""
    oc = [e for e in _open_or_claimed_train(state) if not _is_research_row(e)]
    if oc:
        top = max(oc, key=lambda e: (int(e.get("priority", 0) or 0), -e.get("seq", 0)))
        return (top.get("config") or {}).get("lane") or top.get("id")
    done = _completed_train(state)
    if done:
        return _result_metrics(done[-1]).get("lane") or done[-1].get("id")
    return None


def _lane_digest(state: ledger.LedgerState, lane: str | None) -> LaneDigest:
    d = LaneDigest(lane=lane or "")
    if not lane:
        return d
    series = [e for e in _completed_train(state)
              if (_result_metrics(e).get("lane") or e.get("id")) == lane]
    elos: list[float] = []
    walls: list[float] = []
    for e in series:
        m = _result_metrics(e)
        v = m.get("eval/model_elo")
        if isinstance(v, (int, float)):
            elos.append(float(v))
            w = (e.get("result") or {}).get("wall_s")
            walls.append(float(w) if isinstance(w, (int, float)) else 0.0)
        if m.get("cell"):
            d.cell = m["cell"]
    d.n_slices = len(series)
    d.elo_series = elos
    d.wall_series = walls
    d.delta_elo_per_hr = _delta_elo_per_hr(elos, walls)
    d.base_for_fork = _fork_base(lane)
    return d


def _delta_elo_per_hr(elos: list[float], walls: list[float]) -> float | None:
    """Proxy secant: (last elo - first elo) / (total wall hours). None < 2 points
    or zero wall. ±100-elo noisy per the eval — a hint, not a verdict."""
    if len(elos) < 2:
        return None
    total_hr = sum(walls) / 3600.0
    if total_hr <= 0:
        return None
    return (elos[-1] - elos[0]) / total_hr


def _fork_base(lane: str) -> str:
    """A warm-start fork base = the seed lane's own latest.pt if it exists on disk,
    else scratch. Pure path math (no torch); existence-checked so a not-yet-trained
    lane forks from scratch."""
    ckpt = Path(daemon.home()) / "runs" / lane / "sweep_runs"
    if ckpt.exists():
        for sub in sorted(ckpt.glob("*/checkpoints/latest.pt")):
            return f"local://{sub}"
    return "scratch"


def _plies_series(state: ledger.LedgerState, lane: str | None) -> list[float]:
    """Read the trainer.log plies trend for the lane's cell (regex plies=([\\d.]+),
    last ~6). Fail-soft to [] (log absent on the first slice). Read-only."""
    if not lane:
        return []
    d = _lane_digest_cell(state, lane)
    base = Path(daemon.home()) / "runs" / lane / "sweep_logs"
    log = (base / d / "trainer.log") if d else None
    if log is None or not log.exists():
        # fall back to any cell dir under the lane (don't glob blindly past one)
        cands = sorted(base.glob("*/trainer.log")) if base.exists() else []
        log = cands[-1] if cands else None
    if log is None or not log.exists():
        return []
    try:
        txt = log.read_text(errors="replace")
    except OSError:
        return []
    vals = [float(m) for m in re.findall(r"plies=([\d.]+)", txt)]
    return vals[-6:]


def _lane_digest_cell(state: ledger.LedgerState, lane: str) -> str:
    for e in reversed(_completed_train(state)):
        m = _result_metrics(e)
        if (m.get("lane") or e.get("id")) == lane and m.get("cell"):
            return m["cell"]
    for e in _open_or_claimed_train(state):
        if (e.get("config") or {}).get("lane") == lane:
            return (e.get("config") or {}).get("cell", "")
    return ""


def summarize(state: ledger.LedgerState) -> Summary:
    """Pure digest of the folded ledger — no I/O except the fail-soft disk reads
    for the fork base and plies trend (both read-only, both degrade to empty)."""
    s = Summary()
    completed = _completed_train(state)
    s.has_results = len(completed) > 0
    s.trainer_alive = daemon.probe_alive(daemon.lockfile_path(ROLE))
    meta = daemon.read_lockfile(daemon.lockfile_path(ROLE)) or {}
    s.trainer_item = meta.get("item") if s.trainer_alive else None

    lane = _seed_lane_name(state)
    s.lane = _lane_digest(state, lane)
    s.plies_series = _plies_series(state, lane)

    # last few arena verdicts (most recent first)
    evs = sorted(state.evals.values(), key=lambda e: e.get("seq", 0))
    for ev in reversed(evs[-4:]):
        v = ev.get("verdict") or {}
        m = ev.get("metrics") or {}
        s.last_verdicts.append({
            "gate": v.get("gate") or m.get("verdict"),
            "win_rate": v.get("win_rate") if v.get("win_rate") is not None else m.get("win_rate"),
            "n": v.get("n") if v.get("n") is not None else m.get("n_games"),
            "vs": m.get("vs"),
        })

    for e in state.experiments.values():
        if e.get("status") != ledger.OPEN:
            continue
        role = e.get("role")
        if role == ROLE:
            if _is_research_row(e):
                s.open_research += 1
                s.open_research_ids.append(e.get("id"))
            else:
                s.open_train += 1
        elif role == "arena":
            s.open_arena += 1
    # research rows that are claimed still count toward the cap
    for e in state.experiments.values():
        if e.get("status") == ledger.CLAIMED and _is_research_row(e):
            s.open_research += 1
            s.open_research_ids.append(e.get("id"))

    s.p_seed = _seed_priority(state)
    return s


# ---- idea seeds (fail-soft) ---------------------------------------------

def gather_ideas(*, timeout: float = 10.0) -> list[dict]:
    """Open ``derby-idea`` GitHub issues as idea seeds. Shells out to ``gh`` under
    a broad try/except so no gh / offline / auth failure ever crashes the tick —
    it degrades to ``[]`` and the note says "idea source unavailable"."""
    try:
        out = subprocess.check_output(
            ["gh", "issue", "list", "--label", "derby-idea", "--state", "open",
             "--json", "number,title"],
            text=True, timeout=timeout, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    try:
        data = json.loads(out) if out.strip() else []
    except (ValueError, TypeError):
        return []
    ideas = []
    for it in data if isinstance(data, list) else []:
        num = it.get("number")
        title = (it.get("title") or "").strip()
        if num is None or not title:
            continue
        ideas.append({"number": int(num), "title": title, "slug": _slug(title),
                      "cell": _cell_for_title(title)})
    return ideas


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return (s or "idea")[:40]


def _cell_for_title(title: str) -> str | None:
    """Map an idea title to an EXISTING --cell, else None (note-only)."""
    low = title.lower()
    for key, cell in KNOWN_CELLS.items():
        if key in low:
            return cell
    return None


# ---- proposals (the starvation guard) -----------------------------------

def propose_experiments(state: ledger.LedgerState, ideas: list[dict],
                        summary: Summary) -> list[dict]:
    """At most ``MAX_PROPOSALS`` NEW train rows, each at priority STRICTLY BELOW
    ``P_seed`` (P_seed-1, P_seed-2, …) so ``pick('train')`` ALWAYS still returns
    the seed/continuation lane. Cold-start refuses (no seed ⇒ propose nothing).
    Idempotent stable ids keyed on idea-slug + P_seed; a re-fire re-states the same
    id (a no-op via fold's id-keying). Only ideas mapping to an existing ``--cell``
    become train rows; code-heavy ideas stay note-only."""
    p_seed = summary.p_seed
    if p_seed is None:
        return []                                   # COLD START: never seed a lane
    if summary.open_research >= RESEARCH_QUEUE_CAP:
        return []                                   # research queue saturated

    existing_ids = set(state.experiments.keys())
    proposals: list[dict] = []
    next_priority = p_seed - 1                       # STRICTLY below the seed
    for idea in ideas:
        if len(proposals) >= MAX_PROPOSALS:
            break
        cell = idea.get("cell")
        if not cell:                                # note-only idea (no cell)
            continue
        rid = f"research-{idea['slug']}-p{p_seed}"
        if rid in existing_ids or any(p["id"] == rid for p in proposals):
            continue                                # idempotent: same id => no-op
        # priority strictly below the seed AND below any already-proposed sibling
        prio = next_priority
        next_priority -= 1
        proposals.append(ledger.experiment(
            id=rid, role=ROLE, commit=None, base=summary.lane.base_for_fork,
            config={"lane": rid, "cell": cell, "seq_n": 0,
                    "research": True, "from_issue": idea["number"],
                    "max_wall_secs": 3600},
            priority=prio,
            note=(f"research fork from derby-idea #{idea['number']}: "
                  f"{idea['title'][:60]} (cell {cell}); priority {prio} < seed "
                  f"p{p_seed} — continuity over thrash"),
        ))
    return proposals


# ---- the note -----------------------------------------------------------

def _fmt(x, fmt="{:.0f}", none="—"):
    if x is None or not isinstance(x, (int, float)):
        return none
    return fmt.format(x)


def render_note(summary: Summary, ideas: list[dict], proposals: list[dict],
                *, ts: str, ideas_ok: bool) -> str:
    lane = summary.lane
    elos = lane.elo_series
    e0 = _fmt(elos[0]) if elos else "—"
    eN = _fmt(elos[-1]) if elos else "—"
    rate = lane.delta_elo_per_hr
    rate_s = _fmt(rate, "{:+.0f}") if rate is not None else "—"
    inside_noise = "y" if (rate is not None and abs(rate) < 100) else (
        "n" if rate is not None else "—")
    trainer = (f"ALIVE on {summary.trainer_item}" if summary.trainer_alive
               else "DOWN")

    # champion line
    if summary.last_verdicts:
        v = summary.last_verdicts[0]
        moved = "moved" if v.get("gate") == "PROMOTE" else "held"
        champ = (f"{v.get('gate') or '—'} (win_rate {_fmt(v.get('win_rate'), '{:.2f}')}, "
                 f"n {v.get('n') if v.get('n') is not None else '—'}) — {moved}")
    else:
        champ = "no verdict yet"

    # plies line
    if summary.plies_series:
        ps = summary.plies_series
        arrow = "falling → collapse-watch" if (len(ps) >= 2 and ps[-1] < ps[0]) else "stable"
        plies = f"{'→'.join(_fmt(p) for p in ps)} ({arrow})"
    else:
        plies = "not in metrics"

    lines = [
        "# Autolab researcher — current thinking",
        f"_generated {ts} by gomoku.lab.research (deterministic, --once)_",
        "",
        _HONESTY_BANNER,
        "",
        "## What is happening",
    ]

    if not summary.has_results:
        # COLD-START refusal note
        lines += [
            f"- Seed lane `{lane.lane or '(none yet)'}`: trainer {trainer}; "
            "**no completed slices yet** — waiting for the first slice to produce "
            "an eval before ranking anything.",
            f"- Queue: {summary.open_train} open train, {summary.open_research} "
            f"open research, {summary.open_arena} arena evals pending.",
            "",
            "## What I would try next (and why)",
            "- Nothing yet. With zero completed slices there is **no signal** to "
            "rank on; proposing a research lane now would only compete with the "
            "seed for the GPU. Continuity over thrash.",
            "",
            "## What I queued this tick",
            "- none — waiting for first slice (cold-start refusal; the seed lane "
            "must produce the first signal before research forks queue below it).",
            "",
            "## Caveat (always present)",
            "Proxy-based ranking (once signal exists). Continuity over thrash: the "
            "seed lane keeps training; research rows only ever queue strictly below "
            f"it. Real adjudication waits on #{GATE_OPEN_ISSUE}.",
        ]
        return "\n".join(lines) + "\n"

    lines += [
        f"- Seed lane `{lane.lane}`: trainer {trainer}, {lane.n_slices} slices, "
        f"elo {e0}→{eN} (proxy Δelo/hr ≈ {rate_s}, inside-noise? {inside_noise}).",
        f"- Champion: {champ}.",
        f"- Plies: {plies}.",
        f"- Queue: {summary.open_train} open train, {summary.open_research} open "
        f"research, {summary.open_arena} arena evals pending.",
        "",
        "## What I would try next (and why)",
    ]

    if not ideas_ok:
        lines.append("- (idea source unavailable — `gh` absent/offline; carrying "
                     "no external idea seeds this tick.)")
    if not ideas:
        lines.append("- Deepen the seed lane: the cleanest proxy Δelo/hr comes "
                     "from more slices on the existing recipe before forking. "
                     "Maps to: continuation of the seed lane (no new row).")
    for idea in ideas[:3]:
        cell = idea.get("cell")
        maps = f"cell {cell}" if cell else f"NOTE-ONLY: needs code, #{idea['number']}"
        lines.append(f"- {idea['title'][:70]} — why: open derby-idea "
                     f"#{idea['number']}. Maps to: {maps}.")

    lines += ["", "## What I queued this tick"]
    if not proposals:
        if summary.open_research >= RESEARCH_QUEUE_CAP:
            lines.append("- none — research queue saturated; deepening seed lane.")
        else:
            lines.append("- none — seed-lane continuity preferred (no idea mapped "
                         "to an existing cell, or nothing to add).")
    else:
        for p in proposals:
            cfg = p.get("config") or {}
            lines.append(
                f"- `{p['id']}` at priority {p['priority']} (strictly below seed "
                f"p{summary.p_seed}), base {p.get('base')}, cell {cfg.get('cell')}.")

    lines += [
        "",
        "## Caveat (always present)",
        "Proxy-based ranking. Continuity over thrash: the seed lane keeps "
        "training; research rows queue strictly below it. Real adjudication waits "
        f"on #{GATE_OPEN_ISSUE}.",
    ]
    return "\n".join(lines) + "\n"


# ---- atomic write -------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


# ---- the tick -----------------------------------------------------------

def research_home() -> Path:
    return Path(daemon.home()) / "research"


def latest_path() -> Path:
    return research_home() / "latest.md"


def notes_path() -> Path:
    return research_home() / "NOTES.md"


def tick(ledger_path: str) -> dict:
    """One research-lite tick. Returns a small summary dict (for the CLI line +
    tests). Idempotent: re-fires re-state the same proposal ids (no-ops via fold).
    """
    rows = ledger.read_all(ledger_path)
    state = ledger.fold(rows)
    summary = summarize(state)
    ideas = gather_ideas()
    ideas_ok = True
    # gather_ideas returns [] for BOTH "no ideas" and "gh unavailable"; we can't
    # distinguish, so probe once cheaply for the note wording.
    if not ideas:
        ideas_ok = _gh_available()

    proposals = propose_experiments(state, ideas, summary) if summary.has_results else []

    for row in proposals:
        ledger.append(ledger_path, row)

    digest = _one_line_digest(summary, proposals)
    ledger.append(ledger_path, ledger.event(
        scope="research", summary=digest,
        data={"proposed": [r["id"] for r in proposals],
              "ranks_on": "proxies", "gate_open_issue": GATE_OPEN_ISSUE,
              "p_seed": summary.p_seed,
              "has_results": summary.has_results}))

    ts = ledger.now_iso()
    note = render_note(summary, ideas, proposals, ts=ts, ideas_ok=ideas_ok)
    _atomic_write(latest_path(), note)
    _append(notes_path(), f"\n\n## {ts}\n{note}")

    return {"proposed": [r["id"] for r in proposals], "p_seed": summary.p_seed,
            "has_results": summary.has_results, "digest": digest}


def _gh_available() -> bool:
    try:
        subprocess.check_output(["gh", "--version"], text=True, timeout=5,
                                stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _one_line_digest(summary: Summary, proposals: list[dict]) -> str:
    if not summary.has_results:
        return ("research tick: cold start — no completed slice yet; proposed "
                "nothing (continuity over thrash); ranks on proxies (#61)")
    lane = summary.lane
    eN = _fmt(lane.elo_series[-1]) if lane.elo_series else "—"
    rate = (_fmt(lane.delta_elo_per_hr, "{:+.0f}/hr")
            if lane.delta_elo_per_hr is not None else "—")
    return (f"research tick: lane {lane.lane} {lane.n_slices} slices elo {eN} "
            f"(proxy Δelo {rate}); proposed {len(proposals)} below seed "
            f"p{summary.p_seed}; ranks on proxies (#61)")


# ---- CLI ----------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=daemon.default_ledger_path(),
                    help="path to the ledger JSONL (default: $AUTOLAB_HOME/ledger.jsonl)")
    ap.add_argument("--once", action="store_true",
                    help="run a single research tick then exit (the only mode)")
    args = ap.parse_args(argv)
    out = tick(args.ledger)
    print(out["digest"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
