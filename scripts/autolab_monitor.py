#!/usr/bin/env python3
"""The autolab monitor — the wake-up digest (epic #53, P6/P7).

A pure read-of-durable-state script: fold the ledger, probe both daemons, tail
three files (the researcher note, the trainer ``plies`` log, the arena
verdicts), render a four-section markdown digest, **atomic-write**
``monitor/latest.md``, append one line to ``monitor/log.md``, and fire a macOS
``osascript`` notification — but ONLY when the one-glance header changed since
the previous tick.

No torch, no GPU, no W&B, no network, no Claude. It only reads files the other
loops durably wrote (the path-ownership contract: the monitor *never* writes
under ``research/`` or ``arena/`` — it reads them read-only). The only modules
it imports from the codebase are the stdlib-only spine
(``gomoku.lab.{ledger,daemon,status}``).

Run paths:
  * launchd ``StartInterval=600`` → ``python scripts/autolab_monitor.py``
  * in-session ``python scripts/autolab_monitor.py --print --no-notify``

See ``wiki/topics/autolab-supervisor-and-monitor.md`` §(d).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Make the repo importable when run as a bare script under launchd (cwd may be
# the main checkout; the editable install resolves ``gomoku`` regardless, but
# this keeps a direct ``python scripts/autolab_monitor.py`` honest from any cwd).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gomoku.lab import actionable as _actionable  # noqa: E402
from gomoku.lab import daemon, health, ledger, status  # noqa: E402

# ---- durable file locations (must match the path-ownership contract) -----


def monitor_dir() -> Path:
    return Path(daemon.home()) / "monitor"


def latest_md_path() -> Path:
    return monitor_dir() / "latest.md"


def log_md_path() -> Path:
    return monitor_dir() / "log.md"


def research_latest_path() -> Path:
    return Path(daemon.home()) / "research" / "latest.md"


def research_notes_path() -> Path:
    return Path(daemon.home()) / "research" / "NOTES.md"


def runs_dir() -> Path:
    return Path(daemon.home()) / "runs"


def arena_verdicts_path() -> Path:
    return Path(daemon.home()) / "arena" / "verdicts.jsonl"


def arena_board_path() -> Path:
    return Path(daemon.home()) / "arena" / "board.json"


_PLIES_RE = re.compile(r"plies=([0-9]+(?:\.[0-9]+)?)")
_ELO_KEYS = ("eval/model_elo", "model_elo", "elo", "implied_elo")


# ---- small read helpers (every one degrades to a legible default) --------


def _tail_lines(path: Path, n: int) -> list[str]:
    """Last ``n`` non-empty lines of a file, or [] if absent/unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    return lines[-n:] if n else lines


def _head_lines(path: Path, n: int) -> list[str]:
    """First ``n`` lines (incl. blanks) of a file, or [] if absent."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    return text.splitlines()[:n]


def _first_nonempty(path: Path) -> str:
    for ln in _head_lines(path, 40):
        s = ln.strip().lstrip("#> ").strip()
        if s:
            return s
    return ""


def _model_elo(metrics: dict) -> float | None:
    """The model elo from a result's metrics, or None (None-safe — Risk #5)."""
    for k in _ELO_KEYS:
        v = metrics.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _fmt_elo(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f}"


def _fmt_delta(v: float | None) -> str:
    """A signed delta, or an em-dash when undefined (<2 points / None elo)."""
    return "—" if v is None else f"{v:+.0f}"


# ---- ledger projections --------------------------------------------------


def _ordered_exps(state: ledger.LedgerState) -> list[dict]:
    """All experiments, oldest first (by seq)."""
    return sorted(state.experiments.values(), key=lambda e: e.get("seq", 0))


def _train_results(state: ledger.LedgerState) -> list[dict]:
    """Train experiments that produced a result, oldest first."""
    return [e for e in _ordered_exps(state)
            if e.get("role") == "train" and "result" in e]


def _lane_train_done(state: ledger.LedgerState, lane: str | None) -> list[dict]:
    """``done`` train rows for one lane, oldest first — the Δelo series source."""
    out = []
    for e in _train_results(state):
        if e.get("status") != ledger.DONE:
            continue
        m = (e.get("result") or {}).get("metrics") or {}
        if lane is None or m.get("lane") == lane:
            out.append(e)
    return out


def _lane_delta(state: ledger.LedgerState, lane: str | None) -> float | None:
    """Δelo of the latest done train slice vs the lane's prior done slice.

    None-safe (Risk #5): returns None if there are <2 points, or if either of
    the last two elos is missing. Computed HERE — there is no ``delta_elo``
    field on a train result (trainer.py emits only ``eval/model_elo``).
    """
    elos = []
    for e in _lane_train_done(state, lane):
        elos.append(_model_elo((e.get("result") or {}).get("metrics") or {}))
    pts = [v for v in elos if v is not None]
    if len(elos) < 2 or elos[-1] is None or elos[-2] is None:
        return None
    return elos[-1] - elos[-2]


def _last_eval(state: ledger.LedgerState):
    """The most recent eval (by seq), or None."""
    evs = sorted(state.evals.values(), key=lambda e: e.get("seq", 0))
    return evs[-1] if evs else None


def _eval_gate(ev: dict | None) -> str:
    if not ev:
        return "—"
    v = ev.get("verdict") or {}
    g = v.get("gate") or (ev.get("metrics") or {}).get("verdict")
    return str(g) if g else "—"


def _counts_by_status(state: ledger.LedgerState) -> dict[str, int]:
    c = {ledger.DONE: 0, ledger.FAILED: 0, ledger.OPEN: 0, ledger.CLAIMED: 0}
    for e in state.experiments.values():
        st = e.get("status")
        if st in c:
            c[st] += 1
    return c


def _plies_trend(state: ledger.LedgerState, lane: str | None,
                 cell: str | None) -> list[float]:
    """Best-effort tail of ``plies=`` from ``runs/<lane>/sweep_logs/<cell>/trainer.log``.

    Returns [] if the log is absent/unparseable (skip the line). Uses ``cell``
    to avoid globbing a stale lane's log (contract §d note).
    """
    if not lane:
        return []
    base = runs_dir() / lane / "sweep_logs"
    log = (base / cell / "trainer.log") if cell else None
    if log is None or not log.exists():
        # fall back to a single matching cell dir if unambiguous
        try:
            cells = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
        except OSError:
            cells = []
        if len(cells) == 1:
            log = cells[0] / "trainer.log"
        else:
            return []
    if not log.exists():
        return []
    vals: list[float] = []
    for ln in _tail_lines(log, 0):
        m = _PLIES_RE.search(ln)
        if m:
            try:
                vals.append(float(m.group(1)))
            except ValueError:
                pass
    return vals[-5:]


def _plies_arrow(trend: list[float]) -> str:
    if len(trend) < 2:
        return "▬"
    if trend[-1] < trend[0]:
        return "▼ collapse-watch"
    if trend[-1] > trend[0]:
        return "▲"
    return "▬"


def _champion(state: ledger.LedgerState) -> str:
    """Best-effort champion proxy: the latest eval whose ledger metric says it
    promoted (the authoritative signal, arena.py:82 — NOT board.json.applied)."""
    promoted = [e for e in sorted(state.evals.values(), key=lambda x: x.get("seq", 0))
                if (e.get("metrics") or {}).get("promoted")]
    if promoted:
        return _short(promoted[-1].get("model") or "?")
    # fall back to the arena board's last_verdict candidate, if present
    return "—"


def _short(ref: str, n: int = 22) -> str:
    ref = str(ref)
    return ref if len(ref) <= n else "…" + ref[-(n - 1):]


def _latest_slice_id(state: ledger.LedgerState) -> tuple[str, int | None]:
    """(lane, seq_n) of the most recent train slice that produced a result, else
    the next claimable train lane, else ("—", None)."""
    res = _train_results(state)
    if res:
        m = (res[-1].get("result") or {}).get("metrics") or {}
        lane = m.get("lane") or res[-1].get("id", "—")
        cfg = res[-1].get("config") or {}
        return lane, cfg.get("seq_n")
    # nothing finished yet — name the lane in flight
    nxt = state.pick("train", ledger.utcnow())
    if nxt is not None:
        cfg = nxt.get("config") or {}
        return cfg.get("lane") or nxt.get("id", "—"), cfg.get("seq_n")
    return "—", None


# ---- the header line (the one-glance read; drives notify gating) ---------

HEADER_FMT = ("slice {lane}@{seq} · elo {elo} ({delta}) · train {train} · "
              "arena {arena} · eval: {gate} · champ {champ}")


def build_header(state: ledger.LedgerState, daemons: dict) -> str:
    lane, seq = _latest_slice_id(state)
    res = _lane_train_done(state, lane)
    elo = _model_elo((res[-1].get("result") or {}).get("metrics") or {}) if res else None
    delta = _lane_delta(state, lane)
    return HEADER_FMT.format(
        lane=lane, seq=("?" if seq is None else seq),
        elo=_fmt_elo(elo), delta=_fmt_delta(delta),
        train=("ALIVE" if daemons.get("train", {}).get("alive") else "DOWN"),
        arena=("ALIVE" if daemons.get("arena", {}).get("alive") else "DOWN"),
        gate=_eval_gate(_last_eval(state)),
        champ=_champion(state))


# ---- the digest ----------------------------------------------------------


def build_digest(state: ledger.LedgerState, daemons: dict, *,
                 now_iso: str | None = None) -> str:
    """Render the four-section markdown digest from folded state + daemon probes.

    Pure: takes already-read state, reads the three tail files itself (they are
    durable, read-only). Every projection degrades to a legible default so a
    warming-up tick never raises (esp. None-elo Δelo math — Risk #5).
    """
    ts = now_iso or ledger.now_iso()
    header = build_header(state, daemons)
    lane, _seq = _latest_slice_id(state)

    lines: list[str] = []
    lines.append(f"# Autolab digest — {ts}")
    lines.append(f"**{header}**")
    lines.append("")

    # --- (0) Needs you — escalations float to the very top ---
    alerts = health.scan(state)
    if alerts:
        lines.append("## ⚠️ Needs you")
        for a in alerts:
            lines.append(f"- **{a.kind}**: {a.summary}")
        lines.append("")

    # --- (1) Researcher thinking ---
    lines.append("## Researcher")
    rhook = _first_nonempty(research_latest_path())
    rbody = _head_lines(research_latest_path(), 12)
    if not rbody:
        lines.append("(no notes yet)")
    else:
        lines.append(f"> {rhook}" if rhook else "> (note present, no headline)")
        for ln in rbody:
            lines.append(f"    {ln}")
    notes_tail = _tail_lines(research_notes_path(), 2)
    if notes_tail:
        lines.append("recent: " + " ¦ ".join(notes_tail))
    events = state.events[-5:]
    if events:
        lines.append("events: " + " ¦ ".join(
            f"[{e.get('scope', '?')}] {e.get('summary', '')}" for e in events))
    # resume-on-evidence surface (doctrine §3/§4): research threads whose evidence
    # landed but whose decision the reducer hasn't made yet — the WHEN, not a wait.
    threads = _actionable.actionable(state).research
    if threads:
        lines.append("threads (evidence in, awaiting a decision): " + " ¦ ".join(
            f"{t.lane} (n={t.n_evidence}"
            + (f", #{t.from_issue}" if t.from_issue is not None else "") + ")"
            for t in threads[:4]))
    lines.append("")

    # --- (2) Lanes / tickets ---
    lines.append("## Lanes (tickets)")
    c = _counts_by_status(state)
    lines.append(f"done {c[ledger.DONE]} · failed {c[ledger.FAILED]} · "
                 f"open {c[ledger.OPEN]} · claimed {c[ledger.CLAIMED]}")
    nt = state.pick("train", ledger.utcnow())
    na = state.pick("arena", ledger.utcnow())
    lines.append(f"next[train]: {nt['id'] if nt else 'idle'}   "
                 f"next[arena]: {na['id'] if na else 'idle'}")
    recent = [e for e in _ordered_exps(state) if "result" in e][-5:]
    if recent:
        lines.append("recent results:")
        for e in recent:
            m = (e.get("result") or {}).get("metrics") or {}
            wall = (e.get("result") or {}).get("wall_s")
            wall_s = f"{wall:.0f}s" if isinstance(wall, (int, float)) else "—"
            lines.append(f"  #{e.get('seq')} [{e.get('status')}] {e.get('id')} "
                         f"{e.get('role')} elo={_fmt_elo(_model_elo(m))} ({wall_s})")
    else:
        lines.append("recent results: (ledger empty — seeding)")
    lines.append("")

    # --- (3) Training ---
    lines.append("## Training")
    d = daemons.get("train", {})
    meta = d.get("meta") or {}
    if d.get("alive"):
        lines.append(f"trainer: ALIVE on {meta.get('item', '(idle)')} "
                     f"since {meta.get('started_at', '?')}")
    elif meta:
        lines.append(f"trainer: DOWN (last: {meta.get('item')})")
    else:
        lines.append("trainer: DOWN")
    done = _lane_train_done(state, lane)
    if done:
        last = done[-1]
        m = (last.get("result") or {}).get("metrics") or {}
        wall = (last.get("result") or {}).get("wall_s")
        wall_s = f"{wall:.0f}s" if isinstance(wall, (int, float)) else "—"
        delta = _lane_delta(state, lane)
        cell = m.get("cell")
        lines.append(f"last slice: {m.get('lane', lane)}/{cell} "
                     f"epochs={m.get('epochs_ran', '—')} "
                     f"elo={_fmt_elo(_model_elo(m))} Δ={_fmt_delta(delta)} "
                     f"wall={wall_s}")
        trend = _plies_trend(state, lane, cell)
        if trend:
            seq = "→".join(f"{v:.1f}" for v in trend)
            lines.append(f"plies: {seq} {_plies_arrow(trend)}")
    else:
        lines.append("last slice: (none — first slice in flight)")
    lines.append("")

    # --- (4) Evals ---
    lines.append("## Evals")
    da = daemons.get("arena", {})
    lines.append(f"arena: {'ALIVE' if da.get('alive') else 'DOWN'}")
    ev = _last_eval(state)
    if ev:
        v = ev.get("verdict") or {}
        em = ev.get("metrics") or {}
        wr = v.get("win_rate")
        if wr is None:
            wr = em.get("win_rate")
        ci = v.get("ci") or em.get("ci") or [None, None]
        n = v.get("n") or em.get("n_games")
        lo, hi = (ci + [None, None])[:2]
        wr_s = f"{wr:.2f}" if isinstance(wr, (int, float)) else "—"
        lo_s = f"{lo:.2f}" if isinstance(lo, (int, float)) else "—"
        hi_s = f"{hi:.2f}" if isinstance(hi, (int, float)) else "—"
        lines.append(f"last verdict: {_eval_gate(ev)} win_rate={wr_s} "
                     f"ci=[{lo_s},{hi_s}] n={n if n is not None else '—'} "
                     f"vs {em.get('vs', '—')}")
        vtail = _tail_lines(arena_verdicts_path(), 1)
        if vtail:
            lines.append(f"  reason: {vtail[-1][:200]}")
    else:
        lines.append("last verdict: (none — no eval yet)")
    gates = [_eval_gate(e) for e in sorted(state.evals.values(),
                                           key=lambda x: x.get("seq", 0))[-4:]]
    lines.append(f"champion: {_champion(state)} (HF tag: champion)   "
                 f"history: {' '.join(gates) if gates else '—'}")
    lines.append("")

    return "\n".join(lines)


# ---- the one-line log row + state-change diff ----------------------------


def build_log_line(state: ledger.LedgerState, daemons: dict, *,
                   now_iso: str | None = None) -> str:
    ts = now_iso or ledger.now_iso()
    lane, seq = _latest_slice_id(state)
    res = _lane_train_done(state, lane)
    elo = _model_elo((res[-1].get("result") or {}).get("metrics") or {}) if res else None
    delta = _lane_delta(state, lane)
    c = _counts_by_status(state)
    rhook = _strip(_first_nonempty(research_latest_path()))[:50]
    return (f"{ts}  slice {lane}@{('?' if seq is None else seq)} "
            f"elo {_fmt_elo(elo)} ({_fmt_delta(delta)})  "
            f"train:{'A' if daemons.get('train', {}).get('alive') else 'D'} "
            f"arena:{'A' if daemons.get('arena', {}).get('alive') else 'D'}  "
            f"eval:{_eval_gate(_last_eval(state))}  "
            f"open:{c[ledger.OPEN]} fail:{c[ledger.FAILED]} | {rhook}")


def _last_log_header(path: Path) -> str | None:
    """The header signature of the previous log line (everything up to ' |'),
    used to detect a state change. None if there's no prior line."""
    tail = _tail_lines(path, 1)
    if not tail:
        return None
    return tail[-1].split(" | ", 1)[0]


# ---- writing -------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_digest(digest: str, log_line: str) -> None:
    """Atomic-write latest.md (tmp + os.replace) and append one line to log.md."""
    _atomic_write(latest_md_path(), digest if digest.endswith("\n") else digest + "\n")
    lp = log_md_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a", encoding="utf-8") as f:
        f.write(log_line.rstrip("\n") + "\n")


# ---- notification --------------------------------------------------------


def _strip(text: str) -> str:
    """Collapse all whitespace (incl. newlines) into single spaces."""
    return " ".join(str(text).split())


def _escape(text: str) -> str:
    """Escape backslash THEN double-quote for an osascript string literal, and
    strip newlines (unescaped quotes make `display notification` silently fail)."""
    return _strip(text).replace("\\", "\\\\").replace('"', '\\"')


def build_notification(state: ledger.LedgerState, daemons: dict) -> str:
    """The ≤~120-char notification body (unescaped — escape at emit time)."""
    lane, seq = _latest_slice_id(state)
    res = _lane_train_done(state, lane)
    elo = _model_elo((res[-1].get("result") or {}).get("metrics") or {}) if res else None
    delta = _lane_delta(state, lane)
    hook = _strip(_first_nonempty(research_latest_path()))[:40]
    return (f"Autolab · slice {lane}@{('?' if seq is None else seq)} · "
            f"elo {_fmt_elo(elo)} ({_fmt_delta(delta)}) · "
            f"eval:{_eval_gate(_last_eval(state))} · {hook}")


def notify(body: str, *, title: str = "Autolab",
           runner=None) -> bool:
    """Fire a macOS notification via osascript. Returns True if dispatched.

    Guarded behind ``shutil.which('osascript')`` so non-mac / test hosts no-op.
    ``runner`` defaults to subprocess.run (tests inject a recorder).
    """
    runner = runner or subprocess.run
    if shutil.which("osascript") is None:
        return False
    safe = _escape(body)
    safe_title = _escape(title)
    script = f'display notification "{safe}" with title "{safe_title}"'
    runner(["osascript", "-e", script], check=False,
           capture_output=True, text=True)
    return True


# ---- main ----------------------------------------------------------------


def run_once(*, do_notify: bool = True, always_notify: bool = False,
             do_print: bool = False, notify_runner=None) -> str:
    """One monitor tick: read state, write the digest + log line, notify on
    change. Returns the digest text."""
    ledger_path = daemon.default_ledger_path()
    board = status.lane_board(ledger_path)
    state = board["state"]
    daemons = board["daemons"]

    digest = build_digest(state, daemons)
    log_line = build_log_line(state, daemons)

    # state-change detection BEFORE we append the new line: the header is
    # everything up to ' | ' (the slice/gate/liveness/fail signature); the
    # research-hook suffix is excluded so a reworded note alone doesn't ping.
    prev_header = _last_log_header(log_md_path())
    new_header = log_line.split(" | ", 1)[0]
    changed = (prev_header is None) or (new_header != prev_header)

    # A needs-you alert (stalled lane, first-champion gate, …) must reach Jason
    # even when the header tuple is unchanged — escalations are gate-independent.
    alerts = health.scan(state)
    escalate = bool(alerts)

    write_digest(digest, log_line)

    if do_print:
        sys.stdout.write(digest + ("\n" if not digest.endswith("\n") else ""))

    if do_notify and (always_notify or changed or escalate):
        # On the very first tick (no prior line) we render but DON'T ping (the
        # baseline tick — contract §d empty-state rule), UNLESS there's an
        # escalation (or --always-notify).
        if always_notify or escalate or prev_header is not None:
            body = (f"Autolab needs you: {alerts[0].summary}" if escalate
                    else build_notification(state, daemons))
            notify(body, runner=notify_runner)

    return digest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="echo the digest to stdout")
    ap.add_argument("--no-notify", dest="no_notify", action="store_true",
                    help="suppress the macOS notification")
    ap.add_argument("--always-notify", action="store_true",
                    help="notify even when nothing changed (testing)")
    args = ap.parse_args(argv)
    run_once(do_notify=not args.no_notify, always_notify=args.always_notify,
             do_print=args.do_print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
