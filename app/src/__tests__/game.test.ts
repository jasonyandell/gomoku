import { describe, expect, it } from "vitest";

import {
  BOARD_SIZE,
  GameState,
  N_ACTIONS,
  actionToStr,
  strToAction,
} from "../game";

describe("GameState", () => {
  it("initial state has all squares legal", () => {
    const s = GameState.initial();
    expect(s.moveCount).toBe(0);
    const mask = s.legalMask();
    let sum = 0;
    for (let i = 0; i < N_ACTIONS; i++) sum += mask[i];
    expect(sum).toBe(N_ACTIONS);
    expect(s.legalActions().length).toBe(N_ACTIONS);
  });

  it("apply flips perspective", () => {
    const s = GameState.initial();
    const s2 = s.apply(strToAction("e5"));
    expect(s2.moveCount).toBe(1);
    // The stone we just placed is now on plane 1 (opponent's perspective).
    const e5 = strToAction("e5");
    expect(s2.board[e5 + 81]).toBe(1);
    expect(s2.board[e5]).toBe(0);
    // And e5 is no longer legal.
    expect(s2.legalMask()[e5]).toBe(0);
  });

  it("rejects illegal moves", () => {
    const s = GameState.initial().apply(strToAction("e5"));
    expect(() => s.apply(strToAction("e5"))).toThrow();
  });

  it("horizontal 5-in-a-row wins", () => {
    let s = GameState.initial();
    const moves: Array<[string, string]> = [
      ["a5", "a1"],
      ["b5", "b1"],
      ["c5", "c1"],
      ["d5", "d1"],
    ];
    for (const [b, w] of moves) {
      s = s.apply(strToAction(b));
      expect(s.isTerminal().done).toBe(false);
      s = s.apply(strToAction(w));
      expect(s.isTerminal().done).toBe(false);
    }
    s = s.apply(strToAction("e5"));
    const t = s.isTerminal();
    expect(t.done).toBe(true);
    expect(t.value).toBe(-1.0);
  });

  it("vertical 5-in-a-row wins", () => {
    let s = GameState.initial();
    const moves: Array<[string, string]> = [
      ["e1", "a1"],
      ["e2", "a2"],
      ["e3", "a3"],
      ["e4", "a4"],
    ];
    for (const [b, w] of moves) {
      s = s.apply(strToAction(b));
      s = s.apply(strToAction(w));
    }
    s = s.apply(strToAction("e5"));
    expect(s.isTerminal().done).toBe(true);
  });

  it("diagonal 5-in-a-row wins", () => {
    let s = GameState.initial();
    const blacks = ["a1", "b2", "c3", "d4", "e5"];
    const whites = ["i1", "i2", "i3", "i4"];
    for (let i = 0; i < 4; i++) {
      s = s.apply(strToAction(blacks[i]));
      s = s.apply(strToAction(whites[i]));
    }
    s = s.apply(strToAction(blacks[4]));
    expect(s.isTerminal().done).toBe(true);
  });

  it("antidiagonal 5-in-a-row wins", () => {
    let s = GameState.initial();
    const blacks = ["a5", "b4", "c3", "d2", "e1"];
    const whites = ["i1", "i2", "i3", "i4"];
    for (let i = 0; i < 4; i++) {
      s = s.apply(strToAction(blacks[i]));
      s = s.apply(strToAction(whites[i]));
    }
    s = s.apply(strToAction(blacks[4]));
    expect(s.isTerminal().done).toBe(true);
  });

  it("4-in-a-row is not a win", () => {
    let s = GameState.initial();
    for (let i = 0; i < 4; i++) {
      s = s.apply(strToAction(`${String.fromCharCode("a".charCodeAt(0) + i)}5`));
      s = s.apply(strToAction(`${String.fromCharCode("a".charCodeAt(0) + i)}1`));
    }
    expect(s.isTerminal().done).toBe(false);
  });

  it("toPlanes returns 3*81 floats with constant third plane", () => {
    const s = GameState.initial().apply(strToAction("e5"));
    const planes = s.toPlanes();
    expect(planes.length).toBe(3 * 9 * 9);
    for (let i = 0; i < 81; i++) expect(planes[2 * 81 + i]).toBe(1.0);
  });

  it("action/str round-trip", () => {
    for (let a = 0; a < N_ACTIONS; a++) {
      expect(strToAction(actionToStr(a))).toBe(a);
    }
  });

  it("strToAction rejects out-of-range", () => {
    expect(() => strToAction("z9")).toThrow();
    expect(() => strToAction("a99")).toThrow();
  });

  it("BOARD_SIZE is 9", () => {
    expect(BOARD_SIZE).toBe(9);
  });
});
