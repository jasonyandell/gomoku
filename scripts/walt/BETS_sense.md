# walt-as-a-sense — pre-stated bets (2026-10-07, written before any E-B/E-A result)

Setup: 9x9 Rapfi@50 vs Rapfi@50 decisive games. Defender = eventual loser, to move 1/3/5 plies
before the winner's first proven VCT. Champion 107b supplies prior, MCTS200 and the field
(net@T1). All choosers are restricted to the non-vetoed top-16 (the veto is the existing stack).
Judge: Rapfi@50 plays both sides from after the chosen move. Primary metric: defender loss rate.

**E-B coverage defense** — pressure(c) = share of 32 field worlds (o1 ~ net@T1, tickertape) in
which the attacker has a VCT after (c, o1) under a null move.

- B-1 (Claude): most of these positions are already lost under a strong judge (the knife-edge,
  ~80% of alternatives lose pre-onset). Loss rates are high for every chooser: 0.6–0.85 at
  back=1, lower at back=5.
- B-2 (Claude): mcts*(1-p)^2 beats mcts+veto by ≥5 points absolute with a paired-discordance
  sign test p<0.05 → **25%**.
- B-3 (advisor): +pressure raises top-1 agreement with Rapfi's move by ≥+0.04 over MCTS200.
  Caveat: at back=1 Rapfi's actual move is the move that allowed the VCT, so agreement is a
  weak target there.
- B-4 (Claude): pure walt (min p) is worse than mcts+veto. Coverage alone ignores what the net
  knows about non-VCT danger.
- Kill: mcts*(1-p)^2 within ±2 points of mcts+veto → coverage adds nothing to the stack's
  defense. walt has had its fair shake on the defense side.

**E-A information test** (after E-B) — advisor prediction: walt odds AUC ≈ net value AUC
(within 0.03). Kill if the gap is < 0.03.
