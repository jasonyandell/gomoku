#!/usr/bin/env python3
"""Autolab loop SIMULATOR — engineer the loop, forget the domain is hard.

The autolab is, stripped of gomoku, a small state machine over an append-only
ledger: pick first-priority open work -> run a bounded chunk -> append a
result + flywheel follow-ups -> repeat, with two flock-singleton daemons
(trainer + arena), a research tick, and a monitor.

This drives the **real** ``gomoku.lab.{daemon,ledger,trainer,arena}`` code with
the expensive/domain bits replaced by ``random``:

  * FakeTrainer subclasses the real TrainerRole and overrides ONLY the externals
    (``_checkout`` / ``_run_slice`` / ``_deliver`` / ``_teardown``). The real
    ``run_chunk`` / ``_resolve_base`` / ``_followups`` / flywheel logic — where
    the bugs live — runs unmodified.
  * FakeArena subclasses the real ArenaRole, injecting the gate / champion
    resolver+setter seams the code already exposes.
  * A crash = ``SimCrash(BaseException)`` raised mid-slice: ``run_daemon`` catches
    ``Exception`` not ``BaseException``, so it propagates, the lock's ``finally``
    auto-releases, and NO result row is appended — exactly a SIGKILL that left
    the lane open. Re-running run_daemon = the restarted daemon re-picking.

Run standalone (the "what breaks?" view):
    python tests/test_lab_sim.py            # scenarios + a fuzz sweep
Run as regression tests:
    pytest tests/test_lab_sim.py
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Run-as-script honesty: `python tests/test_lab_sim.py` puts tests/ on sys.path,
# not the repo root, so a shared editable install can resolve `gomoku` to the main
# checkout (the worktree gotcha). Put THIS worktree's root first so the standalone
# report exercises the code under test, exactly as pytest (rootdir-first) does.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gomoku.lab import actionable as A  # noqa: E402
from gomoku.lab import daemon, health, ledger, research  # noqa: E402


# ---- crash injection ----------------------------------------------------

class SimCrash(BaseException):
    """A process death mid-slice. BaseException so run_daemon's `except
    Exception` does NOT catch it (models SIGKILL: lock auto-frees, no result)."""


# ---- the fake world (FS + HF + RNG + injection knobs) -------------------

class World:
    """One simulated machine: a tmp AUTOLAB_HOME, a fake HF repo, an RNG, and
    the failure-injection probabilities for a run."""

    def __init__(self, home: Path, seed: int = 0, board_size: int = 9):
        self.home = Path(home)
        self.rng = random.Random(seed)
        self.board_size = board_size
        # fake HF: revision -> board_size of the stored model; tag -> revision
        self.hf_repo_id = "sim/gomoku"
        self.hf_rev_era: dict[str, int] = {}
        self.hf_tags: dict[str, str] = {}
        # injection knobs (0..1 probabilities, or a bool)
        self.crash_p = 0.0
        self.hf_blip_p = 0.0
        self.train_fail_p = 0.0
        self.foreign = False
        # observability for assertions
        self.elo: dict[str, float] = {}
        self.last_cap: float | None = None
        self.caps_seen: list[float] = []
        self.ran_era: dict[str, tuple] = {}     # row_id -> (config_board_size, era_at_run)
        self.crashes = 0
        self._slice_crashed = False

    def set_era(self, board_size: int) -> None:
        self.board_size = board_size
        os.environ["GOMOKU_BOARD_SIZE"] = str(board_size)

    @property
    def ledger_path(self) -> str:
        return daemon.default_ledger_path()

    def worktree_dirs(self) -> list[Path]:
        wt = self.home / "worktrees"
        return [d for d in wt.iterdir() if d.is_dir()] if wt.exists() else []

    def __enter__(self) -> "World":
        self._prev = {k: os.environ.get(k) for k in ("AUTOLAB_HOME", "GOMOKU_BOARD_SIZE")}
        os.environ["AUTOLAB_HOME"] = str(self.home)
        os.environ["GOMOKU_BOARD_SIZE"] = str(self.board_size)
        (self.home / "worktrees").mkdir(parents=True, exist_ok=True)
        (self.home / "runs").mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *exc):
        for k, v in self._prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---- fake trainer: real run_chunk, faked externals ----------------------

from gomoku.lab.trainer import TrainerRole, SliceFailed  # noqa: E402


class FakeTrainer(TrainerRole):
    def __init__(self, world: World):
        # prod cap so the 1h-clamp question is live; dry_run=False so _deliver
        # (the HF-push step under test) actually runs.
        super().__init__(mvp_mode=False, prod_cap_secs=3600.0, dry_run=False)
        self.world = world

    def preflight(self) -> None:
        if self.world.foreign:
            raise daemon.PreflightDeferred("sim: foreign GPU tenant")

    def _checkout(self, commit, row_id):
        work = Path(daemon.home()) / "worktrees" / row_id.replace("/", "_")
        work.mkdir(parents=True, exist_ok=True)
        return work

    def _teardown(self, work):
        w = self.world
        if w._slice_crashed:        # SIGKILL: the `finally` never really ran → leak
            w._slice_crashed = False
            return
        shutil.rmtree(work, ignore_errors=True)

    def _run_slice(self, work, cell_key, run_base, resume, cap, board_size=None):
        w = self.world
        w.last_cap = cap
        w.caps_seen.append(cap)
        lane = Path(run_base).name
        ck = Path(run_base) / "sweep_runs" / cell_key / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        # a real-ish run_sweep failure (rc != 0) BEFORE producing output
        if w.rng.random() < w.train_fail_p:
            raise SliceFailed(f"sim run_sweep --cell {cell_key} exited 1")
        prev = w.elo.get(lane, 1000.0)
        new = prev + w.rng.gauss(20.0, 30.0)
        w.elo[lane] = new
        (ck / "latest.pt").write_text("fake-latest")
        (ck / "epoch1.pt").write_text("fake-epoch")
        (ck / "eval_results.jsonl").write_text(json.dumps({"eval/model_elo": new}) + "\n")
        # crash AFTER training produced latest.pt (models SIGKILL post-train)
        if w.rng.random() < w.crash_p:
            w._slice_crashed = True
            raise SimCrash()

    def _deliver(self, item, lane, latest, elo):
        w = self.world
        cfg_bs = (item.get("config") or {}).get("board_size")
        w.ran_era[item["id"]] = (cfg_bs, w.board_size)   # the era this slice ran in
        if w.rng.random() < w.hf_blip_p:
            raise RuntimeError("sim: HF 500 (transient)")
        # Key era by the FULL ref: the per-slice revision itself contains '@'
        # (row ids like "9x9@0"), so splitting on '@' is ambiguous — store whole.
        ref = f"{w.hf_repo_id}@{lane}-{item['id']}"
        w.hf_rev_era[ref] = w.board_size
        return ref                          # "owner/repo@lane-rowid", as push_slice emits


# ---- fake arena: real run_chunk, injected gate + champion seams ---------

from gomoku.lab.arena import ArenaRole  # noqa: E402


class FakeArena(ArenaRole):
    def __init__(self, world: World):
        super().__init__(gate_n_games=40, shrink_n_games=12,
                         board_size=world.board_size,     # era-namespaced champ tag
                         gate_fn=self._fake_gate,
                         champion_resolver=self._fake_resolve_champion,
                         champion_setter=self._fake_set_champion)
        self.world = world

    def _resolve_model(self, ref):       # no network: ref IS the handle
        if not ref:
            return None
        if ref.startswith("local://"):
            return ref[len("local://"):]
        if ref.startswith("hf://"):
            return ref[len("hf://"):]
        return ref

    def _ref_era(self, ref) -> int | None:
        if ref is None:
            return None
        return self.world.hf_rev_era.get(ref, self.world.board_size)  # keyed by full ref

    def _fake_gate(self, *, candidate, peak, n_games):
        w = self.world
        # shape mismatch if champion is from a different era (real gate would crash)
        if peak is not None:
            if self._ref_era(candidate) != self._ref_era(peak):
                raise RuntimeError("sim: shape mismatch (cross-era champion)")
        if peak is None:                 # first champion: definitional promote
            return SimpleNamespace(verdict="PROMOTE", promote=True, win_rate=None,
                                   ci_lo=None, ci_hi=None, n_games=0)
        wr = w.rng.uniform(0.3, 0.8)
        promote = wr > 0.6
        return SimpleNamespace(verdict="PROMOTE" if promote else "AMBIGUOUS",
                               promote=promote, win_rate=wr,
                               ci_lo=wr - 0.15, ci_hi=wr + 0.15, n_games=n_games)

    def _fake_resolve_champion(self):
        ref = self.world.hf_tags.get(self._champ_tag)   # namespaced per era
        return (ref, ref) if ref else (None, None)

    def _fake_set_champion(self, item, cand_ref, cand_path, v):
        self.world.hf_tags[self._champ_tag] = cand_ref  # store the full ref


# ---- seed + drivers -----------------------------------------------------

def seed_lane(world: World, *, lane="L", cell="derby-v9-small", priority=10,
              max_wall_secs=3600, board_size=None):
    row = ledger.experiment(
        id=f"{lane}@0", role="train", commit=None, base="scratch",
        config={"lane": lane, "cell": cell, "max_wall_secs": max_wall_secs,
                "seq_n": 0, **({"board_size": board_size} if board_size else {})},
        priority=priority, note="sim seed")
    ledger.append(world.ledger_path, row)


def seed_research(world: World, *, lane="research-foo", cell="derby-v9-small",
                  priority=5, from_issue=99, contract=None, review_policy=None,
                  budget=None):
    """Seed a research fork as the sole/top train work so a tick actually RUNS it
    (forks are normally below the seed; here we want the lineage to produce
    evidence). config.research=True ⇒ the whole lineage is a research thread. An
    optional evidence_contract / review_policy / budget exercise the epistemic WHEN
    + continuation policy."""
    cfg = {"lane": lane, "cell": cell, "max_wall_secs": 3600, "seq_n": 0,
           "research": True, "from_issue": from_issue}
    if contract is not None:
        cfg["evidence_contract"] = contract
    if review_policy is not None:
        cfg["review_policy"] = review_policy
    if budget is not None:
        cfg["budget"] = budget
    row = ledger.experiment(
        id=f"{lane}@0", role="train", commit=None, base="scratch",
        config=cfg, priority=priority, note="sim research fork")
    ledger.append(world.ledger_path, row)


def train_tick(world: World, trainer: FakeTrainer):
    try:
        daemon.run_daemon(trainer, world.ledger_path, once=True)
    except SimCrash:
        world.crashes += 1     # the daemon "died"; flock auto-freed, no result row


def arena_tick(world: World, arena: FakeArena):
    daemon.run_daemon(arena, world.ledger_path, once=True)


def state(world: World) -> ledger.LedgerState:
    return ledger.fold(ledger.read_all(world.ledger_path))


# ---- invariants ---------------------------------------------------------

def inv_worktrees_bounded(world: World) -> list[str]:
    n = len(world.worktree_dirs())
    return [f"worktree leak: {n} dirs remain"] if n > 1 else []


def inv_cap_enforced(world: World) -> list[str]:
    over = [c for c in world.caps_seen if c > 3600.0]
    return [f"1h cap not enforced: ran with cap={max(over)}"] if over else []


def inv_no_lost_done_progress(world: World) -> list[str]:
    """Every DONE train slice must have left the lane recoverable: either a
    follow-up continuation row OR an escalation. Silent dead-end = lost work."""
    st = state(world)
    out = []
    for e in st.experiments.values():
        if e.get("role") != "train" or e.get("status") != ledger.DONE:
            continue
        lane = (e.get("config") or {}).get("lane")
        n = int((e.get("config") or {}).get("seq_n", 0))
        cont = st.experiments.get(f"{lane}@{n + 1}")
        if cont is None:
            out.append(f"DONE slice {e['id']} enqueued no continuation")
    return out


def inv_no_silent_stall(world: World) -> list[str]:
    """If the trainer can pick nothing AND a train lane's last row is FAILED,
    that must be visible as an alert (else the trainer sleeps forever, silent)."""
    st = state(world)
    if st.pick("train", ledger.utcnow()) is not None:
        return []
    failed = [e for e in st.experiments.values()
              if e.get("role") == "train" and e.get("status") == ledger.FAILED]
    if not failed:
        return []
    try:
        from gomoku.lab import health
    except ImportError:
        return ["stalled FAILED lane(s) with no health.scan detector at all"]
    alerts = health.scan(st)
    if not any(getattr(a, "kind", "") == "stalled" for a in alerts):
        return [f"stalled lane(s) {[e['id'] for e in failed]} not flagged by health.scan"]
    return []


def inv_hf_blip_not_fatal(world: World) -> list[str]:
    """A trained slice (latest.pt exists) whose HF push failed must still be
    DONE — local latest.pt is the truth; never lose a real slice to a 500."""
    st = state(world)
    out = []
    for e in st.experiments.values():
        if e.get("role") != "train":
            continue
        res = e.get("result") or {}
        if e.get("status") == ledger.FAILED and "HF" in str(res.get("error") or ""):
            out.append(f"slice {e['id']} marked FAILED by an HF blip: {res.get('error')}")
    return out


def inv_no_cross_era_run(world: World) -> list[str]:
    """No slice may have RUN in a process era different from its declared
    board_size (would silently train the wrong geometry). Uses the per-run era
    record so it stays correct across an era crossing."""
    out = []
    for rid, (cfg_bs, ran_era) in world.ran_era.items():
        if cfg_bs is not None and cfg_bs != ran_era:
            out.append(f"cross-era run: {rid} board_size={cfg_bs} ran in era {ran_era}")
    return out


def inv_no_arena_era_failure(world: World) -> list[str]:
    """The arena must never FAIL on a cross-era shape mismatch (the champion tag
    is namespaced per era, so a new era resolves no stale prior-era champion)."""
    st = state(world)
    out = []
    for e in st.experiments.values():
        if e.get("role") == "arena" and e.get("status") == ledger.FAILED:
            err = (e.get("result") or {}).get("error") or ""
            if "mismatch" in err or "cross-era" in err:
                out.append(f"arena {e['id']} FAILED cross-era: {err}")
    return out


def inv_first_promotion_gated(world: World) -> list[str]:
    """The first champion (gate vs nobody, 0 games) must not auto-crown; it must
    raise a needs_jason escalation instead."""
    st = state(world)
    out = []
    for ev in st.evals.values():
        m = ev.get("metrics") or {}
        if (m.get("vs") == "(first champion)" and m.get("verdict") == "PROMOTE"
                and not m.get("needs_jason")):
            out.append(f"first champion {ev.get('ref')} auto-crowned with n={m.get('n_games')} games")
    return out


def inv_actionable_consistent(world: World) -> list[str]:
    """The molecule must never lie about the atom (doctrine §3): actionable() is
    the ONE read-surface, built ON state.pick / research_threads / health.scan, so
    every dispatch consumer agrees with the daemon. A drift here means the monitor
    /trigger could pick differently than the daemon — the failure unification kills."""
    st = state(world)
    now = ledger.utcnow()
    act = A.actionable(st, now=now)
    out = []
    if (act.train or {}).get("id") != (st.pick("train", now) or {}).get("id"):
        out.append("actionable.train drifted from state.pick('train')")
    if (act.arena or {}).get("id") != (st.pick("arena", now) or {}).get("id"):
        out.append("actionable.arena drifted from state.pick('arena')")
    if [t.lane for t in act.research] != [t.lane for t in research.research_threads(st)]:
        out.append("actionable.research drifted from research_threads")
    if [a.summary for a in act.alerts] != [a.summary for a in health.scan(st, now=now)]:
        out.append("actionable.alerts drifted from health.scan")
    return out


def inv_research_decided_once(world: World) -> list[str]:
    """No research lane is ever decided beyond its landed evidence (no
    decide-without-evidence / no double-decide). covered_through ≤ n_evidence."""
    st = state(world)
    covered = research._decisions_covered(st)
    out = []
    by_lane: dict[str, int] = {}
    for e in st.experiments.values():
        if research._is_research_row(e) and e.get("status") in (ledger.DONE, ledger.FAILED) \
                and "result" in e:
            lane = (e.get("config") or {}).get("lane") or e.get("id")
            by_lane[lane] = by_lane.get(lane, 0) + 1
    for lane, c in covered.items():
        if c > by_lane.get(lane, 0):
            out.append(f"research lane {lane} decided through {c} > {by_lane.get(lane, 0)} evidence")
    return out


def inv_decision_cites_real_evidence(world: World) -> list[str]:
    """The typed-intent wall (autolab-researcher-contract §2): a research decision
    may only cite evidence the thread actually received. REJECTED intents are
    exempt — their bogus refs are precisely why they were refused (and never
    applied)."""
    st = state(world)
    real_ids = set(st.experiments) | set(st.evals)
    out = []
    for ev in st.events:
        if ev.get("scope") != research.RESEARCH_DECISION_SCOPE:
            continue
        d = ev.get("data") or {}
        if d.get("rejected"):
            continue
        for r in d.get("evidence_refs") or []:
            if r not in real_ids:
                out.append(f"decision on {d.get('lane')} cites absent evidence {r!r}")
    return out


def inv_no_redecide_same_cutoff(world: World) -> list[str]:
    """The same evidence cutoff is never decided twice (the seq watermark): two
    research-decision events for one lane must have distinct covers_through_seq."""
    st = state(world)
    seen: dict[tuple, int] = {}
    out = []
    for ev in st.events:
        if ev.get("scope") != research.RESEARCH_DECISION_SCOPE:
            continue
        d = ev.get("data") or {}
        key = (d.get("lane"), d.get("covers_through_seq"))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            out.append(f"lane {d.get('lane')} decided twice at cutoff seq "
                       f"{d.get('covers_through_seq')}")
    return out


def inv_blocked_before_decision(world: World) -> list[str]:
    """Continuation policy (autolab-researcher-contract §3): an exploratory fork
    (review_policy != continuous) never has a RUNNABLE continuation while a
    decision on its evidence is DUE — researcher judgment is causally upstream of
    the next GPU hour."""
    st = state(world)
    out = []
    for t in research.research_threads(st):
        policy = research._lane_config(t.slices, "review_policy", "after_each_slice")
        if policy != "continuous" and t.open_rows:
            out.append(f"fork {t.lane} ({policy}) has runnable "
                       f"{[e['id'] for e in t.open_rows]} while a decision is due")
    return out


ALL_INVARIANTS = [
    inv_worktrees_bounded, inv_cap_enforced, inv_no_lost_done_progress,
    inv_no_silent_stall, inv_hf_blip_not_fatal, inv_no_cross_era_run,
    inv_no_arena_era_failure, inv_first_promotion_gated,
    inv_actionable_consistent, inv_research_decided_once,
    inv_decision_cites_real_evidence, inv_no_redecide_same_cutoff,
    inv_blocked_before_decision,
]


def check(world: World) -> list[str]:
    out = []
    for inv in ALL_INVARIANTS:
        out.extend(inv(world))
    return out


# ---- scenarios (each = a focused failure mode) --------------------------

def _tmp_world(seed=0, board_size=9):
    d = tempfile.mkdtemp(prefix="autolab-sim-")
    return World(Path(d), seed=seed, board_size=board_size)


def scenario_hf_blip(seed=0):
    w = _tmp_world(seed)
    with w:
        w.hf_blip_p = 1.0
        seed_lane(w)
        tr = FakeTrainer(w)
        for _ in range(3):
            train_tick(w, tr)
        return inv_hf_blip_not_fatal(w)


def scenario_worktree_leak(seed=0):
    w = _tmp_world(seed)
    with w:
        w.crash_p = 0.6
        seed_lane(w)
        tr = FakeTrainer(w)
        for _ in range(12):
            train_tick(w, tr)
        return inv_worktrees_bounded(w)


def scenario_1h_cap(seed=0):
    w = _tmp_world(seed)
    with w:
        seed_lane(w, max_wall_secs=99999)
        tr = FakeTrainer(w)
        train_tick(w, tr)
        return inv_cap_enforced(w)


def scenario_stall(seed=0):
    w = _tmp_world(seed)
    with w:
        w.train_fail_p = 1.0          # every slice fails -> terminal, no continuation
        seed_lane(w)
        tr = FakeTrainer(w)
        train_tick(w, tr)
        return inv_no_silent_stall(w)


def scenario_first_promotion(seed=0):
    w = _tmp_world(seed)
    with w:
        seed_lane(w)
        tr, ar = FakeTrainer(w), FakeArena(w)
        train_tick(w, tr)             # produces a slice + an eval follow-up
        arena_tick(w, ar)             # gates it -> first champion
        return inv_first_promotion_gated(w)


def scenario_era_cross(seed=0):
    """Run a 9x9 era (crown a champion, leak a worktree via a crash), then cross
    to 15x15 like `up --board-size 15 up` and run that era. Assert the prior era
    can't contaminate the new one: no cross-era run, no shape-mismatch arena
    FAIL, no worktree leak across the boundary."""
    from gomoku.lab import up
    from gomoku.lab.arena import champion_tag
    w = _tmp_world(seed, board_size=9)
    with w:
        seed_lane(w, lane="9x9", cell="derby-v9-small", board_size=9)
        tr9, ar9 = FakeTrainer(w), FakeArena(w)
        for _ in range(3):
            train_tick(w, tr9); arena_tick(w, ar9)
        # simulate Jason confirming the first 9x9 champion (crowns champion-9)
        revs = list(w.hf_rev_era)
        if revs:
            w.hf_tags[champion_tag(9)] = revs[-1]
        # a crash leaks a 9x9 worktree dir (the lane will then be abandoned)
        w.crash_p = 1.0; train_tick(w, tr9); w.crash_p = 0.0

        # ---- cross to the 15x15 era (what `up --board-size 15 up` does) ----
        retired = up.supersede_foreign_era(w.ledger_path, board_size=15)
        w.set_era(15)
        seed_lane(w, lane="15x15", cell="G15-wdl", board_size=15)
        tr15, ar15 = FakeTrainer(w), FakeArena(w)
        for _ in range(3):
            train_tick(w, tr15); arena_tick(w, ar15)
        # a crash in the NEW era: with the reclaim fix the abandoned 9x9 dir was
        # already pruned, so only this one leaks; WITHOUT it the 9x9 dir + this
        # one coexist — the real 2-dir leak the live gauge missed.
        w.crash_p = 1.0; train_tick(w, tr15); w.crash_p = 0.0

        out = []
        if not retired:
            out.append("era cross retired no foreign-era lanes")
        out += inv_no_cross_era_run(w)
        out += inv_no_arena_era_failure(w)
        out += inv_worktrees_bounded(w)
        # the 15x15 lane actually advanced (the new era really ran)
        st = state(w)
        if not any(e.get("status") == ledger.DONE and (e.get("config") or {}).get("board_size") == 15
                   for e in st.experiments.values()):
            out.append("15x15 era produced no DONE slice")
        return out


def scenario_singleton(seed=0):
    w = _tmp_world(seed)
    with w:
        a = daemon.SingletonLock("train")
        b = daemon.SingletonLock("train")
        try:
            got_a = a.acquire()
            got_b = b.acquire()
            if not (got_a and not got_b):
                return [f"singleton broken: a={got_a} b={got_b}"]
        finally:
            a.release(); b.release()
    return []


def scenario_fold_determinism(seed=0):
    w = _tmp_world(seed)
    with w:
        seed_lane(w)
        tr = FakeTrainer(w)
        for _ in range(3):
            train_tick(w, tr)
        rows = ledger.read_all(w.ledger_path)
        s1, s2 = ledger.fold(rows), ledger.fold(rows)
        k1 = {i: e.get("status") for i, e in s1.experiments.items()}
        k2 = {i: e.get("status") for i, e in s2.experiments.items()}
        return [] if k1 == k2 else [f"fold not deterministic: {k1} != {k2}"]


def scenario_research_resume(seed=0):
    """Resume-on-evidence (doctrine §4): a research fork runs, its evidence lands,
    the reducer (research.resume) decides it — exactly once per evidence arrival,
    idempotent on a re-fire, and actionable.research is the deterministic WHEN."""
    w = _tmp_world(seed)
    with w:
        seed_research(w, lane="research-foo")
        tr = FakeTrainer(w)
        out = []

        # 1) no evidence yet → no thread.
        if research.research_threads(state(w)):
            out.append("research thread surfaced before any slice ran")

        # 2) run a slice → evidence lands → the thread surfaces (the WHEN).
        train_tick(w, tr)
        threads = research.research_threads(state(w))
        if not any(t.lane == "research-foo" for t in threads):
            out.append("landed research evidence did not surface as a thread")

        # 3) resume decides it; the thread then clears (resolved).
        decisions = research.resume(w.ledger_path)
        if not any(d["lane"] == "research-foo" for d in decisions):
            out.append("resume made no decision on the landed thread")
        if research.research_threads(state(w)):
            out.append("thread still unresolved after resume")

        # 4) idempotent: a second resume with no new evidence is a no-op.
        if research.resume(w.ledger_path) != []:
            out.append("resume re-decided a thread with no new evidence")

        # 5) more evidence → resumes again (catch-up per arrival, not once-ever).
        train_tick(w, tr)                       # runs research-foo@1
        if not research.research_threads(state(w)):
            out.append("second slice's evidence did not re-open the thread")
        research.resume(w.ledger_path)
        if research.research_threads(state(w)):
            out.append("thread not resolved after second resume")

        out += inv_research_decided_once(w)
        out += inv_actionable_consistent(w)
        return out


def scenario_research_park(seed=0):
    """A 'park' decision supersedes the fork's BLOCKED continuation (append-only),
    so the trainer never runs it — GPU reclaimed. Driven by a forced-park
    DecisionIntent (the typed seam Claude plugs into) through validate→compile."""
    w = _tmp_world(seed)
    with w:
        seed_research(w, lane="research-bar")
        tr = FakeTrainer(w)
        train_tick(w, tr)                       # @0 DONE, @1 BLOCKED_FOR_DECISION

        st = state(w)
        if st.experiments.get("research-bar@1", {}).get("status") != ledger.BLOCKED:
            return ["fork continuation not BLOCKED after a slice (setup wrong)"]

        park = lambda t, s: research.DecisionIntent(  # noqa: E731 — the smart-decider seam
            "park", [sl["id"] for sl in t.slices], "sim forced park")
        research.resume(w.ledger_path, decide=park)

        st = state(w)
        out = []
        cont = st.experiments.get("research-bar@1")
        if cont is None or cont.get("status") != ledger.SUPERSEDED:
            out.append(f"parked continuation not superseded: {cont and cont.get('status')}")
        if st.pick("train", ledger.utcnow()) is not None:
            out.append("trainer still picks a parked lane's work")
        return out


def scenario_evidence_contract(seed=0):
    """The EPISTEMIC WHEN (autolab-researcher-contract §1): a lane whose contract
    requires 2 train-results does NOT surface a decision after 1 slice — only once
    the REQUIRED evidence has landed. (continuous policy so the trainer keeps
    producing slices uninterrupted; isolates the contract gate from blocking.)"""
    w = _tmp_world(seed)
    with w:
        seed_research(w, lane="research-c2", review_policy="continuous",
                      contract={"required": [{"kind": research.EV_TRAIN, "count": 2}]})
        tr = FakeTrainer(w)
        out = []
        train_tick(w, tr)                       # 1 slice — NOT enough evidence
        if research.research_threads(state(w)):
            out.append("thread surfaced after 1 slice despite a 2-slice contract")
        train_tick(w, tr)                       # 2 slices — contract satisfied
        threads = research.research_threads(state(w))
        t = next((t for t in threads if t.lane == "research-c2"), None)
        if t is None:
            out.append("thread did NOT surface after the contract's 2 slices landed")
        elif t.due_reason != "evidence-contract-satisfied":
            out.append(f"due for the wrong reason: {t.due_reason}")
        out += inv_no_redecide_same_cutoff(w)
        return out


def scenario_continuation_policy(seed=0):
    """Continuation policy (§3): a fork's continuation is BLOCKED_FOR_DECISION after
    a slice (judgment upstream of GPU spend); a 'keep' decision releases it so the
    trainer can pick it next."""
    w = _tmp_world(seed)
    with w:
        seed_research(w, lane="research-cp")    # default → after_each_slice
        tr = FakeTrainer(w)
        out = []
        train_tick(w, tr)                       # @0 DONE, @1 BLOCKED
        st = state(w)
        if st.experiments.get("research-cp@1", {}).get("status") != ledger.BLOCKED:
            out.append(f"continuation not BLOCKED: "
                       f"{st.experiments.get('research-cp@1', {}).get('status')}")
        if st.pick("train", ledger.utcnow()) is not None:
            out.append("trainer picked a blocked fork continuation before any decision")
        out += inv_blocked_before_decision(w)   # meaningful here: a decision is due
        # keep → release the held continuation
        research.resume(w.ledger_path)          # default decides 'keep' on 1 noisy slice
        st = state(w)
        if st.experiments.get("research-cp@1", {}).get("status") != ledger.OPEN:
            out.append("'keep' did not release the blocked continuation")
        if (st.pick("train", ledger.utcnow()) or {}).get("id") != "research-cp@1":
            out.append("released continuation is not the trainer's next pick")
        return out


def scenario_intent_validation(seed=0):
    """The wall (§2): a forged intent citing evidence the thread never received is
    REFUSED — never applied — recorded as rejected, escalated to a human, and the
    watermark still advances so the loop can't spin re-refusing it."""
    w = _tmp_world(seed)
    with w:
        seed_research(w, lane="research-iv")
        tr = FakeTrainer(w)
        train_tick(w, tr)                       # @0 DONE, @1 BLOCKED

        forged = lambda t, s: research.DecisionIntent(  # noqa: E731
            "park", evidence_refs=["ghost-999"], rationale="forged park")
        research.resume(w.ledger_path, decide=forged)

        st = state(w)
        out = []
        cont = st.experiments.get("research-iv@1")
        if cont and cont.get("status") == ledger.SUPERSEDED:
            out.append("forged intent citing absent evidence was APPLIED (park took effect)")
        if not any(ev.get("scope") == research.RESEARCH_DECISION_SCOPE
                   and (ev.get("data") or {}).get("rejected") for ev in st.events):
            out.append("refused intent not recorded as a rejection")
        if not any(getattr(a, "kind", "") == "needs_jason" for a in health.scan(st)):
            out.append("refused intent did not escalate to needs_jason")
        if research.research_threads(st):
            out.append("refused thread still DUE — would spin re-refusing forever")
        out += inv_decision_cites_real_evidence(w)
        return out


SCENARIOS = {
    "hf_blip_not_fatal": scenario_hf_blip,
    "worktrees_bounded": scenario_worktree_leak,
    "1h_cap_enforced": scenario_1h_cap,
    "stall_visible": scenario_stall,
    "first_promotion_gated": scenario_first_promotion,
    "era_cross_no_contamination": scenario_era_cross,
    "singleton": scenario_singleton,
    "fold_deterministic": scenario_fold_determinism,
    "research_resume_on_evidence": scenario_research_resume,
    "research_park_takes_effect": scenario_research_park,
    "evidence_contract_epistemic_when": scenario_evidence_contract,
    "continuation_blocked_before_decision": scenario_continuation_policy,
    "intent_validation_wall": scenario_intent_validation,
}


# ---- fuzz: mix every injection, assert invariants every tick ------------

def fuzz(seed=0, ticks=200):
    w = _tmp_world(seed)
    with w:
        w.crash_p, w.hf_blip_p, w.train_fail_p = 0.15, 0.2, 0.1
        seed_lane(w)
        tr, ar = FakeTrainer(w), FakeArena(w)
        violations: dict[str, int] = {}
        for i in range(ticks):
            w.foreign = (w.rng.random() < 0.1)
            if w.rng.random() < 0.5:
                train_tick(w, tr)
            else:
                arena_tick(w, ar)
            for v in check(w):
                key = v.split(":")[0]
                violations[key] = violations.get(key, 0) + 1
        return violations


# ---- standalone "what breaks?" report -----------------------------------

def _main() -> int:
    print("=== autolab loop simulator — scenarios ===")
    any_red = False
    for name, fn in SCENARIOS.items():
        viol = fn(seed=1)
        status = "GREEN" if not viol else "RED"
        any_red = any_red or bool(viol)
        print(f"  [{status:5}] {name}")
        for v in viol:
            print(f"            ↳ {v}")
    print("\n=== fuzz sweep (5 seeds × 200 ticks, all injections on) ===")
    agg: dict[str, int] = {}
    for s in range(5):
        for k, n in fuzz(seed=s, ticks=200).items():
            agg[k] = agg.get(k, 0) + n
    if not agg:
        print("  no invariant violations")
    for k, n in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5}×  {k}")
    return 1 if any_red or agg else 0


# ---- pytest regression surface -----------------------------------------

def test_singleton():
    assert scenario_singleton() == []


def test_fold_deterministic():
    assert scenario_fold_determinism() == []


def test_hf_blip_not_fatal():
    assert scenario_hf_blip() == []


def test_worktrees_bounded():
    assert scenario_worktree_leak() == []


def test_1h_cap_enforced():
    assert scenario_1h_cap() == []


def test_stall_visible():
    assert scenario_stall() == []


def test_first_promotion_gated():
    assert scenario_first_promotion() == []


def test_era_cross_no_contamination():
    assert scenario_era_cross() == []


def test_research_resume_on_evidence():
    assert scenario_research_resume() == []


def test_research_park_takes_effect():
    assert scenario_research_park() == []


def test_evidence_contract_epistemic_when():
    assert scenario_evidence_contract() == []


def test_continuation_blocked_before_decision():
    assert scenario_continuation_policy() == []


def test_intent_validation_wall():
    assert scenario_intent_validation() == []


def test_fuzz_no_violations():
    agg = {}
    for s in range(3):
        for k, n in fuzz(seed=s, ticks=150).items():
            agg[k] = agg.get(k, 0) + n
    assert agg == {}, f"fuzz found invariant violations: {agg}"


if __name__ == "__main__":
    sys.exit(_main())
