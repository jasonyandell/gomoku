"""Functional smoke for scripts/lab_train_cell.py — parse path + cell_id.

Feeds a synthetic trainer.log of `epoch N/M ... train=Xs)` lines through
parse_trainer_log and asserts:
  - epochs_per_sec is computed correctly
  - trainer_step_s_p50 picks up the per-step median
  - cell_id derivation is stable

No GPU work; no subprocess; runs in <1s.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_train_cell import cell_id_of, parse_trainer_log


def smoke_parse(tmp_path: Path) -> None:
    log = tmp_path / "trainer.log"
    # Eight epochs. With warmup_secs=30 and measurement_secs=60 (total 90s),
    # warmup_epochs = floor(8 * 30/90) = 2. So post-warmup epochs = 1..5
    # remain (epochs 3..8 if indices are 3..8 = inclusive count 6 entries),
    # epochs_per_sec = (last - first) / measurement_secs = (8 - 3) / 60 = 0.0833.
    lines = []
    # train=4.0s with steps=16 → per-step 0.25; vary slightly so median works.
    for ep, train_s in [
        (1, 4.0), (2, 4.2), (3, 4.1), (4, 4.0),
        (5, 3.9), (6, 4.1), (7, 4.0), (8, 4.2),
    ]:
        lines.append(
            f"epoch {ep}/100 games=10 buf=100 new=4 steps=16 "
            f"pl=1.234 vl=0.567 plies=20.0 age=0 "
            f"(5.0s: gen=1.0s train={train_s}s)"
        )
    log.write_text("\n".join(lines) + "\n")
    m = parse_trainer_log(log, warmup_secs=30, measurement_secs=60)
    assert m["epochs_in_window"] == 8, m
    assert m["first_epoch"] == 3, m
    assert m["last_epoch"] == 8, m
    expected_eps = (8 - 3) / 60
    assert abs(m["epochs_per_sec"] - expected_eps) < 1e-6, m
    # Per-step = train_s/steps; median of [4.0..4.2]/16 = ~0.25625
    assert 0.24 < m["trainer_step_s_p50"] < 0.27, m
    print("[smoke] parse_trainer_log:")
    print(f"        epochs_in_window={m['epochs_in_window']}")
    print(f"        first_epoch={m['first_epoch']}  last_epoch={m['last_epoch']}")
    print(f"        epochs_per_sec={m['epochs_per_sec']:.4f}  (expected {expected_eps:.4f})")
    print(f"        trainer_step_s_p50={m['trainer_step_s_p50']:.4f}")


def smoke_short_run(tmp_path: Path) -> None:
    """A 1-epoch log — should fall back to zero epochs_per_sec because we
    need >=2 epoch lines to compute a rate."""
    log = tmp_path / "trainer-short.log"
    log.write_text("epoch 1/100 games=10 buf=100 new=4 steps=16 "
                   "pl=1.234 vl=0.567 plies=20.0 age=0 (5.0s: gen=1.0s train=4.0s)\n")
    m = parse_trainer_log(log, warmup_secs=30, measurement_secs=60)
    assert m["epochs_in_window"] == 1, m
    assert m["epochs_per_sec"] == 0.0, m
    print("[smoke] short-log (1 epoch):")
    print(f"        epochs_in_window={m['epochs_in_window']}  epochs_per_sec={m['epochs_per_sec']}")


def smoke_no_log(tmp_path: Path) -> None:
    """Missing trainer.log — should return safe zeros, not raise."""
    m = parse_trainer_log(tmp_path / "doesnt-exist.log",
                          warmup_secs=30, measurement_secs=60)
    assert m["epochs_in_window"] == 0, m
    assert m["epochs_per_sec"] == 0.0, m
    print("[smoke] missing-log: returned zeros without raising")


def smoke_cell_id() -> None:
    cell = dict(model="small", workers=8, games_per_batch=8, n_simulations=400,
                wave_size=64, ema_tau=0.99, grad_accum_steps=4, wave_mode=True,
                sgd_per_position=0.0025, batch_size=512)
    cid = cell_id_of(cell)
    assert cid == "train_small_W08_G08_S400_V064_EMA99_GA04_WM1_B512", cid
    # Stable: rebuilding from the same dict gives the same id.
    assert cell_id_of(dict(cell)) == cid
    # Changing a knob changes the id.
    cell2 = dict(cell, wave_size=512)
    cid2 = cell_id_of(cell2)
    assert cid2 == "train_small_W08_G08_S400_V512_EMA99_GA04_WM1_B512", cid2
    print(f"[smoke] cell_id_of: {cid}")
    print(f"[smoke] cell_id_of (V=512): {cid2}")


def main() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="L12-smoke-") as tmpdir:
        td = Path(tmpdir)
        smoke_parse(td)
        smoke_short_run(td)
        smoke_no_log(td)
        smoke_cell_id()
    print("\nall smokes passed.")


if __name__ == "__main__":
    main()
