"""The default panel EXCLUDES the flaky wine-run Gomocup engines (issue #35).

Jason's nix directive (2026-06-16): "bail on wine evals — another wine crash.
eval just with reliable things." The five wine ``pbrain-*.exe`` anchors
(embryo26/yixin18/pela23/zetor17/eulring16) crashed ~17/36 panel pairs, so they
are now OPT-IN ONLY in ``scripts.panel_tournament``. The reliable default field
is our net checkpoints (pure-torch brain wrapper) + the pure-python ``heuristic``
floor + any native (non-wine) engines.

These tests pin that contract so a future edit can't silently re-introduce wine
engines into the default eval path. They are pure (no torch, no subprocess, no
GPU): the panel module body only does ``os.environ.setdefault`` and its heavy
imports are lazy, so importing it is cheap and side-effect-free here.
"""

from __future__ import annotations

import os
import sys

# Import the panel module the same way ``panel_white_elo.py`` does (repo root on
# sys.path so ``scripts`` is a package). Done at module scope so monkeypatch of
# the env var below re-reads the live function, not an import-time snapshot.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts import panel_tournament as pt  # noqa: E402  (after sys.path shim)

# The names that identify a wine wrapper: the five catalog labels and the wine
# run-wrapper basenames they point at.
_WINE_LABELS = {"embryo26", "yixin18", "pela23", "zetor17", "eulring16"}
_WINE_WRAPPER_BASENAMES = {
    "run-embryo26", "run-yixin18", "run-pela23", "run-zetor17", "run-eulring16",
}


def _spec_names(participants):
    """Collect every label + every cmd basename referenced by a field."""
    names: set[str] = set()
    for label, spec in participants:
        names.add(label)
        for token in spec.replace(",", " ").split():
            names.add(os.path.basename(token))
    return names


def test_real_engines_default_excludes_wine():
    """``real_engines(enable_wine=False)`` carries ZERO wine wrappers."""
    default = pt.real_engines(enable_wine=False)
    labels = {lbl for lbl, _ in default}
    assert not (labels & _WINE_LABELS), (
        f"default real_engines leaked wine labels: {labels & _WINE_LABELS}")
    cmds = {os.path.basename(cmd) for _, cmd in default}
    assert not (cmds & _WINE_WRAPPER_BASENAMES), (
        f"default real_engines leaked wine wrappers: {cmds & _WINE_WRAPPER_BASENAMES}")


def test_real_engines_optin_includes_wine():
    """The catalog is KEPT — opting in restores all five wine anchors."""
    opted = pt.real_engines(enable_wine=True)
    labels = {lbl for lbl, _ in opted}
    assert _WINE_LABELS <= labels, (
        f"opt-in field missing wine anchors: {_WINE_LABELS - labels}")


def test_default_participants_have_no_wine():
    """The DEFAULT field built for a tournament contains no wine label/wrapper."""
    parts = pt._build_participants(sims=50, timeout_ms=1000, enable_wine=False)
    names = _spec_names(parts)
    leaked = names & (_WINE_LABELS | _WINE_WRAPPER_BASENAMES)
    assert not leaked, f"default participants leaked wine refs: {leaked}"
    # Sanity: the reliable floor + nets are still present.
    labels = {lbl for lbl, _ in parts}
    assert "heuristic" in labels
    assert any(lbl.startswith("az-") for lbl in labels)


def test_back_compat_alias_is_reliable_only():
    """The legacy ``_REAL_ENGINES`` alias reflects the reliable default."""
    labels = {lbl for lbl, _ in pt._REAL_ENGINES}
    assert not (labels & _WINE_LABELS)


def test_wine_enabled_via_env(monkeypatch):
    """``GOMOKU_ENABLE_WINE_ENGINES`` truthy values opt in; default is OFF."""
    monkeypatch.delenv("GOMOKU_ENABLE_WINE_ENGINES", raising=False)
    assert pt.wine_engines_enabled() is False
    for truthy in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("GOMOKU_ENABLE_WINE_ENGINES", truthy)
        assert pt.wine_engines_enabled() is True, truthy
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("GOMOKU_ENABLE_WINE_ENGINES", falsy)
        assert pt.wine_engines_enabled() is False, falsy
    # The CLI flag forces it on regardless of env.
    monkeypatch.setenv("GOMOKU_ENABLE_WINE_ENGINES", "0")
    assert pt.wine_engines_enabled(cli_wine=True) is True
