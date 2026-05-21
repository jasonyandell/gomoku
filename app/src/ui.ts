// Play + replay tab logic. Ported from web/static/app.js, with all server-side
// /api/* calls replaced by in-browser MCTS calls against the ORT evaluator.

import {
  HintMove,
  actionFromXY,
  actionName,
  renderBoard,
  stonesFromHistory,
} from "./board";
import { type Evaluator } from "./mcts";
import {
  BOARD_SIZE,
  GameState,
  N_ACTIONS,
  actionToStr,
} from "./game";
import { MCTSGame, policyFromVisits, runBatchedMCTS } from "./mcts";
import type { EvaluatorOptions } from "./evaluator";

const $ = <T extends Element = HTMLElement>(sel: string) =>
  document.querySelector(sel) as T;
const $$ = (sel: string) => Array.from(document.querySelectorAll(sel));

type TopMove = { action: number; name: string; visits: number; q: number };

type PlayState = {
  history: number[];
  humanColor: "black" | "white";
  busy: boolean;
  topMoves: TopMove[];
};

type ReplayMove = {
  action: number;
  name: string;
  value: number;
  top: TopMove[];
};

type ReplayState = {
  moves: ReplayMove[];
  plyIdx: number;
  outcome: number;
  playing: boolean;
  timer: number | null;
};

const state = {
  tab: "play" as "play" | "replay",
  play: {
    history: [],
    humanColor: "black",
    busy: false,
    topMoves: [],
  } as PlayState,
  replay: {
    moves: [],
    plyIdx: 0,
    outcome: 0,
    playing: false,
    timer: null,
  } as ReplayState,
};

let evaluator: Evaluator | null = null;
let evaluatorFactory: ((opts: EvaluatorOptions) => Evaluator) | null = null;
let currentOpts: EvaluatorOptions = {};

function getEvaluator(): Evaluator {
  if (!evaluator) {
    if (!evaluatorFactory) throw new Error("evaluator not initialized");
    evaluator = evaluatorFactory(currentOpts);
  }
  return evaluator;
}

/** Swap the active model. The next MCTS call lazily creates a fresh evaluator
 * bound to the new URL + plane count. */
export function setActiveModel(opts: EvaluatorOptions): void {
  currentOpts = opts;
  evaluator = null;
}

function gameFromHistory(history: number[]): GameState {
  let s = GameState.initial();
  for (const a of history) s = s.apply(a);
  return s;
}

function topMovesFromRoot(root: ReturnType<typeof newMCTS>["root"], k = 6): TopMove[] {
  const items: TopMove[] = [];
  for (let a = 0; a < N_ACTIONS; a++) {
    if (root.N[a] > 0) {
      const q = root.W[a] / root.N[a];
      items.push({ action: a, name: actionToStr(a), visits: root.N[a], q });
    }
  }
  items.sort((a, b) => b.visits - a.visits);
  return items.slice(0, k);
}

function newMCTS(history: number[], rng?: () => number): MCTSGame {
  return new MCTSGame(gameFromHistory(history), { rng });
}

function hintsFromTop(top: TopMove[]): HintMove[] {
  return top.map((m) => ({ action: m.action, visits: m.visits, q: m.q }));
}

// -------- PLAY tab --------

function renderPlay(): void {
  const svg = $<SVGSVGElement>("#board");
  const stones = stonesFromHistory(state.play.history);
  const last = state.play.history.length > 0 ? state.play.history[state.play.history.length - 1] : null;
  renderBoard(svg, stones, last, hintsFromTop(state.play.topMoves));

  const ul = $("#play-top");
  ul.innerHTML = "";
  for (let i = 0; i < state.play.topMoves.length; i++) {
    const m = state.play.topMoves[i];
    const li = document.createElement("li");
    if (i === 0) li.className = "best";
    li.innerHTML = `<span>${m.name}</span><span>${m.visits} &middot; Q=${m.q.toFixed(2)}</span>`;
    ul.appendChild(li);
  }

  const ply = state.play.history.length;
  const sideToMove = ply % 2 === 0 ? "black" : "white";
  let status = `ply ${ply} — ${sideToMove} to move`;
  const humanSide = state.play.humanColor === "black" ? 0 : 1;
  status += ply % 2 === humanSide ? " (you)" : " (model)";
  $("#status").textContent = status;
}

function checkPlayTerminal(): boolean {
  const s = gameFromHistory(state.play.history);
  // After apply(), plane 1 holds the player who just moved. If isTerminal
  // returns done with value=-1, the side that just moved won.
  const t = s.isTerminal();
  if (!t.done) return false;
  const ply = state.play.history.length;
  if (ply === 0) return false;
  if (t.value === -1.0) {
    const sideJustMoved = (ply - 1) % 2;
    const winnerName = sideJustMoved === 0 ? "black" : "white";
    const humanSide = state.play.humanColor === "black" ? 0 : 1;
    const youWon = sideJustMoved === humanSide;
    $("#status").textContent = `${winnerName} wins — ${youWon ? "you" : "model"} 🏆`;
  } else {
    $("#status").textContent = "draw";
  }
  state.play.busy = true;
  return true;
}

async function modelMove(): Promise<void> {
  if (state.play.busy) return;
  state.play.busy = true;
  const sims = parseInt(($("#play-sims") as HTMLInputElement).value, 10);
  $("#status").textContent = `thinking (${sims} sims)…`;
  try {
    const g = newMCTS(state.play.history);
    if (g.root.isTerminal) {
      $("#status").textContent = "game already finished";
      return;
    }
    await runBatchedMCTS([g], getEvaluator(), { nSimulations: sims, addRootNoise: false });
    const top = topMovesFromRoot(g.root);
    if (top.length === 0) {
      $("#status").textContent = "no legal moves";
      return;
    }
    const pi = policyFromVisits(g.root, 0.0);
    let best = -1;
    let bestV = -Infinity;
    for (let a = 0; a < N_ACTIONS; a++) {
      if (pi[a] > bestV) {
        bestV = pi[a];
        best = a;
      }
    }
    state.play.history.push(best);
    state.play.topMoves = top;
    // Network value of root (from side-to-move's perspective at root).
    // Approximate via N-weighted Q at the root.
    let nTot = 0;
    let qWeighted = 0;
    for (let a = 0; a < N_ACTIONS; a++) {
      if (g.root.N[a] > 0) {
        nTot += g.root.N[a];
        qWeighted += g.root.W[a];
      }
    }
    const rootQ = nTot > 0 ? qWeighted / nTot : 0;
    $("#play-eval").textContent = `model eval: ${rootQ >= 0 ? "+" : ""}${rootQ.toFixed(2)}`;
    renderPlay();
    if (checkPlayTerminal()) return;
  } catch (e) {
    $("#status").textContent = "error: " + (e as Error).message;
  } finally {
    state.play.busy = false;
    maybeStepModel();
  }
}

function maybeStepModel(): void {
  if (state.tab !== "play") return;
  if (state.play.busy) return;
  if (checkPlayTerminal()) return;
  const humanSide = state.play.humanColor === "black" ? 0 : 1;
  const sideToMove = state.play.history.length % 2;
  if (sideToMove !== humanSide) void modelMove();
}

function newPlayGame(): void {
  state.play.history = [];
  state.play.topMoves = [];
  state.play.busy = false;
  $("#play-eval").textContent = "";
  renderPlay();
  maybeStepModel();
}

function undoMove(): void {
  // Pop two plies so it's the human's turn again.
  if (state.play.busy && !checkPlayTerminal()) return;
  state.play.history = state.play.history.slice(0, Math.max(0, state.play.history.length - 2));
  state.play.busy = false;
  state.play.topMoves = [];
  $("#play-eval").textContent = "";
  renderPlay();
  maybeStepModel();
}

// -------- REPLAY tab --------

/** Mulberry32 deterministic PRNG seeded by `seed`. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function generateReplay(): Promise<void> {
  const sims = parseInt(($("#rep-sims") as HTMLInputElement).value, 10);
  const tempMoves = parseInt(($("#rep-temp") as HTMLInputElement).value, 10);
  const noiseEps = parseFloat(($("#rep-noise") as HTMLInputElement).value);
  const seed = parseInt(($("#rep-seed") as HTMLInputElement).value, 10);

  $("#status").textContent = "generating game…";
  ($("#rep-generate") as HTMLButtonElement).disabled = true;
  try {
    const rng = makeRng(seed);
    const moves: ReplayMove[] = [];
    let s = GameState.initial();
    let g = new MCTSGame(s, { rng, dirichletEps: noiseEps });
    let ply = 0;
    let outcome = 0;
    while (true) {
      if (g.root.isTerminal) break;
      const addNoise = noiseEps > 0;
      await runBatchedMCTS([g], getEvaluator(), {
        nSimulations: sims,
        addRootNoise: addNoise && ply === 0,
      });
      const top = topMovesFromRoot(g.root);
      const temperature = ply < tempMoves ? 1.0 : 0.0;
      const pi = policyFromVisits(g.root, temperature);
      let chosen: number;
      if (temperature === 0) {
        chosen = 0;
        let mv = -Infinity;
        for (let a = 0; a < N_ACTIONS; a++) if (pi[a] > mv) { mv = pi[a]; chosen = a; }
      } else {
        // Sample.
        let r = rng();
        chosen = 0;
        for (let a = 0; a < N_ACTIONS; a++) {
          r -= pi[a];
          if (r <= 0) { chosen = a; break; }
        }
      }
      // Network value of the chosen position approx via Q of chosen action.
      const value = g.root.N[chosen] > 0 ? g.root.W[chosen] / g.root.N[chosen] : 0;
      moves.push({ action: chosen, name: actionToStr(chosen), value, top });
      s = s.apply(chosen);
      const t = s.isTerminal();
      if (t.done) {
        // value=-1 means the side that just moved won; side that just moved is
        // (ply % 2) == 0 → black. Outcome from black's perspective: +1 black, -1 white.
        if (t.value === -1.0) outcome = ply % 2 === 0 ? +1 : -1;
        else outcome = 0;
        break;
      }
      g.advanceRoot(chosen);
      ply++;
      // Yield to the UI thread occasionally.
      if (ply % 4 === 0) {
        $("#status").textContent = `generating… ply ${ply}`;
        await new Promise<void>((r) => setTimeout(r, 0));
      }
    }
    state.replay.moves = moves;
    state.replay.plyIdx = moves.length;
    state.replay.outcome = outcome;
    ($("#rep-scrub") as HTMLInputElement).max = String(moves.length);
    ($("#rep-scrub") as HTMLInputElement).value = String(moves.length);
    renderReplay();
    const outStr = outcome > 0 ? "black wins" : outcome < 0 ? "white wins" : "draw";
    $("#status").textContent = `${moves.length} plies — ${outStr}`;
  } catch (e) {
    $("#status").textContent = "error: " + (e as Error).message;
  } finally {
    ($("#rep-generate") as HTMLButtonElement).disabled = false;
  }
}

function renderReplay(): void {
  const idx = state.replay.plyIdx;
  const history = state.replay.moves.slice(0, idx).map((m) => m.action);
  const svg = $<SVGSVGElement>("#board");
  const stones = stonesFromHistory(history);
  const last = history.length > 0 ? history[history.length - 1] : null;
  const upcoming = state.replay.moves[idx];
  const top = upcoming ? upcoming.top : [];
  renderBoard(svg, stones, last, hintsFromTop(top));

  $("#rep-ply").textContent = `${idx} / ${state.replay.moves.length}`;
  ($("#rep-scrub") as HTMLInputElement).value = String(idx);

  const ul = $("#rep-top");
  ul.innerHTML = "";
  for (const m of top) {
    const li = document.createElement("li");
    if (m.action === upcoming?.action) li.className = "best";
    li.innerHTML = `<span>${m.name}</span><span>${m.visits} &middot; Q=${m.q.toFixed(2)}</span>`;
    ul.appendChild(li);
  }
  if (upcoming) {
    $("#rep-eval").textContent = `chose ${upcoming.name} • Q=${upcoming.value.toFixed(2)}`;
  } else if (state.replay.moves.length) {
    $("#rep-eval").textContent = "(end of game)";
  } else {
    $("#rep-eval").textContent = "";
  }
}

function repStep(delta: number): void {
  const n = state.replay.moves.length;
  if (n === 0) return;
  state.replay.plyIdx = Math.max(0, Math.min(n, state.replay.plyIdx + delta));
  renderReplay();
}

function repTogglePlay(): void {
  if (state.replay.playing) {
    if (state.replay.timer != null) clearInterval(state.replay.timer);
    state.replay.timer = null;
    state.replay.playing = false;
    $("#rep-play").textContent = "▶ play";
    return;
  }
  if (state.replay.plyIdx >= state.replay.moves.length) state.replay.plyIdx = 0;
  state.replay.playing = true;
  $("#rep-play").textContent = "⏸ pause";
  state.replay.timer = window.setInterval(() => {
    if (state.replay.plyIdx >= state.replay.moves.length) {
      repTogglePlay();
      return;
    }
    state.replay.plyIdx += 1;
    renderReplay();
  }, 600);
}

// -------- tabs + setup --------

function setTab(name: "play" | "replay"): void {
  state.tab = name;
  $$(".tab").forEach((t) => t.classList.toggle("active", (t as HTMLElement).dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${name}`));
  if (name === "play") renderPlay();
  else renderReplay();
}

function setupBoardClick(): void {
  const svg = $<SVGSVGElement>("#board");
  svg.addEventListener("click", (e) => {
    if (state.tab !== "play" || state.play.busy) return;
    const humanSide = state.play.humanColor === "black" ? 0 : 1;
    const sideToMove = state.play.history.length % 2;
    if (sideToMove !== humanSide) return;
    const pt = svg.createSVGPoint();
    pt.x = (e as MouseEvent).clientX;
    pt.y = (e as MouseEvent).clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const svgP = pt.matrixTransform(ctm.inverse());
    const a = actionFromXY(svgP.x, svgP.y);
    if (a == null) return;
    if (state.play.history.includes(a)) return;
    state.play.history.push(a);
    state.play.topMoves = [];
    renderPlay();
    if (!checkPlayTerminal()) maybeStepModel();
  });
}

export function setupUI(factory: (opts: EvaluatorOptions) => Evaluator): void {
  evaluatorFactory = factory;
  setupBoardClick();
  $$(".tab").forEach((t) => t.addEventListener("click", () => setTab((t as HTMLElement).dataset.tab as "play" | "replay")));

  $("#play-new").addEventListener("click", newPlayGame);
  $("#play-undo").addEventListener("click", undoMove);
  $("#play-resign").addEventListener("click", () => {
    $("#status").textContent = "you resigned";
    state.play.busy = true;
  });
  ($("#play-color") as HTMLSelectElement).addEventListener("change", (e) => {
    state.play.humanColor = (e.target as HTMLSelectElement).value as "black" | "white";
    newPlayGame();
  });

  $("#rep-generate").addEventListener("click", () => void generateReplay());
  $("#rep-prev").addEventListener("click", () => repStep(-1));
  $("#rep-next").addEventListener("click", () => repStep(1));
  $("#rep-play").addEventListener("click", repTogglePlay);
  ($("#rep-scrub") as HTMLInputElement).addEventListener("input", (e) => {
    state.replay.plyIdx = parseInt((e.target as HTMLInputElement).value, 10);
    renderReplay();
  });

  renderPlay();
}

// Re-export for convenience.
export { BOARD_SIZE, actionName };
