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
from lab_train_cell import (
    TRAJECTORY_COLS,
    cell_id_of,
    parse_trainer_log,
    parse_trainer_trajectory,
    write_trajectory_tsv,
)


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


def smoke_trajectory(tmp_path: Path) -> None:
    """Full-trajectory parse (epoch-capped mode). Feeds a synthetic 5-epoch log
    in the EXACT gomoku/train.py:1135-1170 print format — including the
    `(Xs: gen=Ys train=Zs)` wall/gen/train split, the `wave[v.. tile=N ..]`
    block, a NaN policy loss, and a trailing `elo=` — and asserts the full
    per-epoch rows, the runaway growth metrics, the NaN count, and the TSV
    roundtrip. This is the epoch-capped counterpart to smoke_parse."""
    log = tmp_path / "trainer-traj.log"
    # (ep, games, buf, new, steps, pl, vl, plies, age, wall, gen, train, tile, elo)
    specs = [
        (1, 100,  500, 64, 16, "1.20", "0.40", 18.0, 0, 5.0,  2.0, 3.0, 64, None),
        (2, 200, 1000, 64, 20, "1.10", "0.38", 20.0, 1, 6.5,  2.5, 4.0, 70, None),
        (3, 300, 1500, 64, 28, "nan",  "0.35", 22.0, 2, 8.0,  3.0, 5.0, 80, None),
        (4, 400, 2000, 64, 40, "1.05", "0.33", 24.0, 3, 11.0, 4.0, 7.0, 96, None),
        (5, 500, 2500, 64, 60, "1.00", "0.31", 26.0, 4, 16.0, 5.0, 11.0, 128, 1234),
    ]
    lines = []
    for (ep, games, buf, new, steps, pl, vl, plies, age, wall, gen, tr_, tile, elo) in specs:
        msg = (f"epoch {ep}/5 games={games} buf={buf} new={new} steps={steps} "
               f"pl={pl} vl={vl} plies={plies} age={age} "
               f"({wall}s: gen={gen}s train={tr_}s) "
               f"wave[v{ep} tile={tile} workers=8-8 extras=0]")
        if elo is not None:
            msg += f" wr[h=55%] elo={elo}"
        lines.append(msg)
    log.write_text("\n".join(lines) + "\n")

    tr = parse_trainer_trajectory(log)
    rows = tr["rows"]
    assert tr["epochs_completed"] == 5, tr
    assert [r["epoch"] for r in rows] == [1, 2, 3, 4, 5], rows
    assert [r["steps"] for r in rows] == [16, 20, 28, 40, 60], rows
    assert [r["tile"] for r in rows] == [64, 70, 80, 96, 128], rows
    assert [r["buf"] for r in rows] == [500, 1000, 1500, 2000, 2500], rows
    assert rows[0]["wall_s"] == 5.0 and rows[-1]["wall_s"] == 16.0, rows
    assert rows[0]["gen_s"] == 2.0 and rows[-1]["train_s"] == 11.0, rows
    assert rows[0]["elo"] is None and rows[-1]["elo"] == 1234.0, rows
    assert tr["nan_count"] == 1, tr  # epoch 3 pl=nan
    assert abs(tr["steps_growth_ratio"] - 60 / 16) < 1e-9, tr
    assert abs(tr["wall_growth_ratio"] - 16.0 / 5.0) < 1e-9, tr
    assert tr["steps_slope"] > 0 and tr["wall_slope"] > 0, tr  # diverging
    assert tr["final_plies"] == 26.0 and tr["final_elo"] == 1234.0, tr

    out = tmp_path / "trajectory.tsv"
    write_trajectory_tsv(out, rows)
    txt = out.read_text().splitlines()
    assert txt[0] == "\t".join(TRAJECTORY_COLS), txt[0]
    assert len(txt) == 6, txt  # header + 5 rows
    assert txt[1].split("\t")[-1] == "", txt[1]        # epoch1 elo blank
    assert txt[5].split("\t")[-1] == "1234.0", txt[5]  # epoch5 elo filled
    print("[smoke] parse_trainer_trajectory:")
    print(f"        epochs_completed={tr['epochs_completed']}  nan_count={tr['nan_count']}")
    print(f"        steps_growth_ratio={tr['steps_growth_ratio']:.3f}  "
          f"wall_growth_ratio={tr['wall_growth_ratio']:.3f}")
    print(f"        final_plies={tr['final_plies']}  final_elo={tr['final_elo']}")


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
        smoke_trajectory(td)
        smoke_cell_id()
    print("\nall smokes passed.")


if __name__ == "__main__":
    main()
