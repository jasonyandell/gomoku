import { describe, expect, it } from "vitest";

import { GameState, N_ACTIONS, strToAction } from "../game";
import {
  MCTSGame,
  makeRandomEvaluator,
  policyFromVisits,
  runBatchedMCTS,
} from "../mcts";

function argmax(a: Float32Array): number {
  let m = -Infinity;
  let idx = -1;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > m) {
      m = a[i];
      idx = i;
    }
  }
  return idx;
}

describe("MCTS", () => {
  it("random evaluator only gives priors on legal actions", async () => {
    const ev = makeRandomEvaluator();
    const s = GameState.initial().apply(strToAction("e5"));
    const { priors } = await ev([s]);
    expect(priors[0][strToAction("e5")]).toBe(0);
    let sum = 0;
    for (let i = 0; i < N_ACTIONS; i++) sum += priors[0][i];
    expect(sum).toBeCloseTo(1.0, 5);
  });

  it("visits only legal actions", async () => {
    const ev = makeRandomEvaluator();
    const s = GameState.initial().apply(strToAction("e5"));
    const g = new MCTSGame(s);
    await runBatchedMCTS([g], ev, { nSimulations: 50, addRootNoise: false });
    expect(g.root.N[strToAction("e5")]).toBe(0);
    let total = 0;
    for (let i = 0; i < N_ACTIONS; i++) total += g.root.N[i];
    expect(total).toBe(50);
  });

  it("policyFromVisits sums to 1", async () => {
    const ev = makeRandomEvaluator();
    const g = new MCTSGame(GameState.initial());
    await runBatchedMCTS([g], ev, { nSimulations: 30, addRootNoise: false });
    const pi1 = policyFromVisits(g.root, 1.0);
    const pi0 = policyFromVisits(g.root, 0.0);
    let s1 = 0;
    let s0 = 0;
    for (let i = 0; i < N_ACTIONS; i++) {
      s1 += pi1[i];
      s0 += pi0[i];
    }
    expect(s1).toBeCloseTo(1.0, 5);
    expect(s0).toBeCloseTo(1.0, 5);
  });

  it("picks the immediate winning move with random eval", async () => {
    // black: a5..d5; white plays elsewhere; black to move; e5 wins immediately.
    let s = GameState.initial();
    const moves: Array<[string, string]> = [
      ["a5", "a1"],
      ["b5", "b1"],
      ["c5", "c1"],
      ["d5", "d1"],
    ];
    for (const [b, w] of moves) {
      s = s.apply(strToAction(b));
      s = s.apply(strToAction(w));
    }
    const ev = makeRandomEvaluator();
    const g = new MCTSGame(s);
    await runBatchedMCTS([g], ev, { nSimulations: 200, addRootNoise: false });
    const pi = policyFromVisits(g.root, 0.0);
    expect(argmax(pi)).toBe(strToAction("e5"));
  });

  it("batched MCTS across multiple games", async () => {
    const ev = makeRandomEvaluator();
    const games = Array.from({ length: 4 }, () => new MCTSGame(GameState.initial()));
    await runBatchedMCTS(games, ev, { nSimulations: 20, addRootNoise: false });
    for (const g of games) {
      let total = 0;
      for (let i = 0; i < N_ACTIONS; i++) total += g.root.N[i];
      expect(total).toBe(20);
    }
  });
});
