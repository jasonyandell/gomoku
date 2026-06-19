"""Tests for the autolab research-lite tick (epic #53, P5 #61 — partial).

GPU-free, offline: AUTOLAB_HOME=tmp_path, synthetic ledger via real constructors,
``gh`` patched to [] (no network). The load-bearing invariant under test is the
starvation guard: every research-proposed train row has priority < P_seed AND
``pick('train')`` STILL returns the seed/continuation lane.
"""
from __future__ import annotations

from gomoku.lab import daemon as D
from gomoku.lab import ledger as L
from gomoku.lab import research as R


# ---- fixtures: a synthetic ledger built with the real constructors -------

def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLAB_HOME", str(tmp_path))
    return str(tmp_path / "ledger.jsonl")


def _no_gh(monkeypatch):
    """gh patched to []: no ideas, no network. _gh_available True so the note
    doesn't claim the source is unavailable (we just have no ideas)."""
    monkeypatch.setattr(R, "gather_ideas", lambda **kw: [])
    monkeypatch.setattr(R, "_gh_available", lambda: True)


def _seed_with_one_done_slice(led, *, lane="9x9-champ-recipe", priority=10):
    """A seed lane with slice @0 DONE (elo 1200) + an open continuation @1 at the
    same priority — exactly the trainer flywheel's shape mid-run."""
    L.append(led, L.experiment(
        id=f"{lane}@0", role="train", base="scratch",
        config={"lane": lane, "cell": "derby-v9-small", "seq_n": 0,
                "max_wall_secs": 3600}, priority=priority,
        note="overnight seed", status=L.OPEN))
    L.append(led, L.result(f"{lane}@0", status=L.DONE,
                           model="local:///x/latest.pt", wall_s=3600.0,
                           metrics={"eval/model_elo": 1200.0, "epochs_ran": 1200,
                                    "lane": lane, "cell": "derby-v9-small"}))
    # the flywheel continuation (open, same priority) — the row pick() must keep returning
    L.append(led, L.experiment(
        id=f"{lane}@1", role="train", base="local:///x/latest.pt",
        config={"lane": lane, "cell": "derby-v9-small", "seq_n": 1,
                "max_wall_secs": 3600}, priority=priority,
        note="continuation", status=L.OPEN))
    return lane


# ---- (1) THE INVARIANT: proposals < P_seed AND pick('train') == seed -----

def test_proposals_strictly_below_seed_and_pick_unchanged(tmp_path, monkeypatch):
    led = _home(tmp_path, monkeypatch)
    lane = _seed_with_one_done_slice(led, priority=10)
    # an idea that maps to an existing cell so a real train row is proposed
    monkeypatch.setattr(R, "gather_ideas", lambda **kw: [
        {"number": 99, "title": "try v9-small fork", "slug": "try-v9-small-fork",
         "cell": "derby-v9-small"}])

    before = L.fold(L.read_all(led)).pick("train", L.utcnow())
    assert before is not None and before["id"] == f"{lane}@1"   # seed continuation

    out = R.tick(led)
    assert out["p_seed"] == 10
    assert len(out["proposed"]) == 1

    state = L.fold(L.read_all(led))
    # every research-proposed train row is STRICTLY below the seed band
    for rid in out["proposed"]:
        e = state.experiments[rid]
        assert e["role"] == "train"
        assert int(e["priority"]) < 10, f"{rid} p{e['priority']} not below seed p10"

    # THE anti-starvation invariant: the trainer STILL picks the seed continuation
    after = state.pick("train", L.utcnow())
    assert after is not None and after["id"] == f"{lane}@1"


def test_two_proposals_get_distinct_descending_priorities(tmp_path, monkeypatch):
    led = _home(tmp_path, monkeypatch)
    _seed_with_one_done_slice(led, priority=10)
    monkeypatch.setattr(R, "gather_ideas", lambda **kw: [
        {"number": 1, "title": "fork a", "slug": "fork-a", "cell": "derby-v9-small"},
        {"number": 2, "title": "fork b", "slug": "fork-b",
         "cell": "derby-v7-mate-discount"},
        {"number": 3, "title": "fork c", "slug": "fork-c", "cell": "derby-v9-small"},
    ])
    out = R.tick(led)
    assert len(out["proposed"]) == 2                  # capped at MAX_PROPOSALS
    state = L.fold(L.read_all(led))
    prios = sorted(int(state.experiments[r]["priority"]) for r in out["proposed"])
    assert prios == [8, 9]                            # P_seed-2, P_seed-1
    # seed still wins
    assert state.pick("train", L.utcnow())["id"].endswith("@1")


# ---- (2) idempotent ids: two ticks -> no duplicate experiment rows -------

def test_idempotent_ids_no_duplicate_rows(tmp_path, monkeypatch):
    led = _home(tmp_path, monkeypatch)
    _seed_with_one_done_slice(led, priority=10)
    monkeypatch.setattr(R, "gather_ideas", lambda **kw: [
        {"number": 7, "title": "stable fork", "slug": "stable-fork",
         "cell": "derby-v9-small"}])

    out1 = R.tick(led)
    out2 = R.tick(led)
    # tick 1 proposes the stable id; tick 2 re-states it as a NO-OP (already in
    # the ledger ⇒ not re-appended) — that is what "idempotent ids" buys us.
    assert out1["proposed"] == ["research-stable-fork-p10"]
    assert out2["proposed"] == [], "re-fire re-proposed an already-queued row"

    rows = L.read_all(led)
    exp_ids = [r["id"] for r in rows if r.get("type") == L.EXPERIMENT]
    rid = out1["proposed"][0]
    assert exp_ids.count(rid) == 1, "re-fire duplicated the experiment row"
    # the folded state still has exactly one such experiment
    state = L.fold(rows)
    assert rid in state.experiments
    # seed continuation still the pick after two ticks
    assert state.pick("train", L.utcnow())["id"].endswith("@1")


# ---- (3) cold-start refusal: no results -> 0 proposals + waiting note ----

def test_cold_start_refusal(tmp_path, monkeypatch):
    led = _home(tmp_path, monkeypatch)
    _no_gh(monkeypatch)
    # an open seed lane but ZERO completed slices
    L.append(led, L.experiment(
        id="9x9-champ-recipe@0", role="train", base="scratch",
        config={"lane": "9x9-champ-recipe", "cell": "derby-v9-small", "seq_n": 0},
        priority=10, status=L.OPEN))

    out = R.tick(led)
    assert out["has_results"] is False
    assert out["proposed"] == []                      # proposed NOTHING

    # no research experiment rows were appended
    rows = L.read_all(led)
    research_rows = [r for r in rows
                     if r.get("type") == L.EXPERIMENT and str(r["id"]).startswith("research-")]
    assert research_rows == []

    note = R.latest_path().read_text()
    assert "waiting for first slice" in note or "cold-start refusal" in note.lower()
    assert "no completed slices" in note.lower() or "no signal" in note.lower()


def test_cold_start_with_truly_empty_ledger(tmp_path, monkeypatch):
    led = _home(tmp_path, monkeypatch)
    _no_gh(monkeypatch)
    out = R.tick(led)                                  # empty ledger, no rows at all
    assert out["proposed"] == [] and out["has_results"] is False
    assert out["p_seed"] is None
    assert R.latest_path().exists()                    # still wrote a legible note


# ---- (4) the note contains the proxy caveat ------------------------------

def test_note_contains_proxy_caveat(tmp_path, monkeypatch):
    led = _home(tmp_path, monkeypatch)
    _seed_with_one_done_slice(led, priority=10)
    _no_gh(monkeypatch)
    R.tick(led)
    note = R.latest_path().read_text()
    assert "PROXIES" in note
    assert "#61" in note
    assert "hint, not a verdict" in note
    # the body also lands in the append-only trail
    notes = R.notes_path().read_text()
    assert "PROXIES" in notes


# ---- (5) an event row is appended ----------------------------------------

def test_event_row_appended(tmp_path, monkeypatch):
    led = _home(tmp_path, monkeypatch)
    _seed_with_one_done_slice(led, priority=10)
    _no_gh(monkeypatch)
    R.tick(led)
    state = L.fold(L.read_all(led))
    research_events = [e for e in state.events if e.get("scope") == "research"]
    assert research_events, "no research event row appended"
    ev = research_events[-1]
    assert ev["data"]["ranks_on"] == "proxies"
    assert ev["data"]["gate_open_issue"] == 61
    assert "proposed" in ev["data"]


# ---- bonus: research-queue saturation stops proposing --------------------

def test_saturated_research_queue_proposes_nothing(tmp_path, monkeypatch):
    led = _home(tmp_path, monkeypatch)
    _seed_with_one_done_slice(led, priority=10)
    # pre-seed K=3 open research rows below the seed
    for i in range(R.RESEARCH_QUEUE_CAP):
        L.append(led, L.experiment(
            id=f"research-prefill-{i}", role="train", base="scratch",
            config={"lane": f"research-prefill-{i}", "cell": "derby-v9-small",
                    "research": True}, priority=9 - i, status=L.OPEN))
    monkeypatch.setattr(R, "gather_ideas", lambda **kw: [
        {"number": 5, "title": "one more", "slug": "one-more",
         "cell": "derby-v9-small"}])
    out = R.tick(led)
    assert out["proposed"] == []                       # saturated → propose nothing
    note = R.latest_path().read_text()
    assert "saturated" in note.lower()


# ---- bonus: a note-only idea (no cell) is not queued as a train row -------

def test_note_only_idea_not_queued(tmp_path, monkeypatch):
    led = _home(tmp_path, monkeypatch)
    _seed_with_one_done_slice(led, priority=10)
    monkeypatch.setattr(R, "gather_ideas", lambda **kw: [
        {"number": 42, "title": "new VCF solver in C", "slug": "new-vcf-solver",
         "cell": None}])          # code-heavy: no existing cell
    out = R.tick(led)
    assert out["proposed"] == []                       # note-only, never a train row
    note = R.latest_path().read_text()
    assert "NOTE-ONLY" in note and "#42" in note


# ---- gather_ideas itself is fail-soft ------------------------------------

def test_gather_ideas_failsoft_on_missing_gh(monkeypatch):
    import subprocess as _sp

    def boom(*a, **k):
        raise FileNotFoundError("gh not installed")

    monkeypatch.setattr(R.subprocess, "check_output", boom)
    assert R.gather_ideas() == []                       # never raises


def test_gather_ideas_parses_gh_json(monkeypatch):
    payload = '[{"number": 12, "title": "v9-small deeper"}, {"number": 13, "title": "z"}]'
    monkeypatch.setattr(R.subprocess, "check_output", lambda *a, **k: payload)
    ideas = R.gather_ideas()
    assert len(ideas) == 2
    assert ideas[0]["number"] == 12 and ideas[0]["cell"] == "derby-v9-small"
    assert ideas[1]["cell"] is None                     # "z" maps to no cell
