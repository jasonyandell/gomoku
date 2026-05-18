// TypeScript port of gomoku/mcts.py.
// PUCT MCTS with cross-game leaf batching for batched neural-net evaluation.

import { GameState, N_ACTIONS } from "./game";

export type Evaluator = (states: GameState[]) => Promise<{
  priors: Float32Array[]; // each length N_ACTIONS
  values: Float32Array;   // length = states.length
}>;

export class Node {
  state: GameState;
  parent: Node | null;
  parentAction: number;
  N: Int32Array;
  W: Float32Array;
  P: Float32Array;
  children: Map<number, Node>;
  legalMask: Uint8Array;
  expanded: boolean;
  isTerminal: boolean;
  terminalValue: number;

  constructor(state: GameState, parent: Node | null = null, parentAction = -1) {
    this.state = state;
    this.parent = parent;
    this.parentAction = parentAction;
    this.N = new Int32Array(N_ACTIONS);
    this.W = new Float32Array(N_ACTIONS);
    this.P = new Float32Array(N_ACTIONS);
    this.children = new Map();
    this.legalMask = new Uint8Array(N_ACTIONS);
    this.expanded = false;
    this.isTerminal = false;
    this.terminalValue = 0.0;
  }

  totalVisits(): number {
    let s = 0;
    for (let i = 0; i < N_ACTIONS; i++) s += this.N[i];
    return s;
  }
}

function initNode(node: Node): void {
  const { done, value } = node.state.isTerminal();
  node.isTerminal = done;
  node.terminalValue = value;
  if (!done) {
    node.legalMask = node.state.legalMask();
  }
}

function selectAction(node: Node, cPuct: number): number {
  const total = node.totalVisits();
  const sqrtTotal = Math.sqrt(total + 1e-8);
  let best = -Infinity;
  let bestA = -1;
  for (let a = 0; a < N_ACTIONS; a++) {
    if (!node.legalMask[a]) continue;
    const n = node.N[a];
    const q = n > 0 ? node.W[a] / n : 0.0;
    const u = (cPuct * node.P[a] * sqrtTotal) / (1.0 + n);
    const score = q + u;
    if (score > best) {
      best = score;
      bestA = a;
    }
  }
  return bestA;
}

function setPriors(node: Node, raw: Float32Array): void {
  // Softmax over legal actions only.
  let maxV = -Infinity;
  for (let a = 0; a < N_ACTIONS; a++) {
    if (!node.legalMask[a]) continue;
    if (raw[a] > maxV) maxV = raw[a];
  }
  if (!Number.isFinite(maxV)) {
    // No legal actions; leave P at zero.
    return;
  }
  let sum = 0;
  const exp = new Float32Array(N_ACTIONS);
  for (let a = 0; a < N_ACTIONS; a++) {
    if (!node.legalMask[a]) continue;
    const e = Math.exp(raw[a] - maxV);
    exp[a] = e;
    sum += e;
  }
  if (sum > 0) {
    for (let a = 0; a < N_ACTIONS; a++) node.P[a] = exp[a] / sum;
  } else {
    // Uniform over legal as fallback.
    let nLegal = 0;
    for (let a = 0; a < N_ACTIONS; a++) if (node.legalMask[a]) nLegal++;
    if (nLegal > 0) {
      for (let a = 0; a < N_ACTIONS; a++) node.P[a] = node.legalMask[a] ? 1.0 / nLegal : 0.0;
    }
  }
}

/** Sample from Dirichlet(alpha, ..., alpha) of length n. */
function dirichlet(n: number, alpha: number, rng: () => number): Float32Array {
  // Generate Gamma(alpha, 1) samples; normalize.
  const out = new Float32Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) {
    out[i] = gammaSample(alpha, rng);
    total += out[i];
  }
  if (total > 0) {
    for (let i = 0; i < n; i++) out[i] /= total;
  } else {
    for (let i = 0; i < n; i++) out[i] = 1.0 / n;
  }
  return out;
}

/** Marsaglia-Tsang Gamma(k, 1) sampler for k > 0. */
function gammaSample(k: number, rng: () => number): number {
  if (k < 1) {
    // Boost: G(k,1) = G(k+1,1) * U^(1/k)
    const u = rng();
    return gammaSample(k + 1, rng) * Math.pow(u, 1.0 / k);
  }
  const d = k - 1.0 / 3.0;
  const c = 1.0 / Math.sqrt(9.0 * d);
  for (;;) {
    let x = 0;
    let v = 0;
    do {
      x = stdNormal(rng);
      v = 1.0 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) return d * v;
  }
}

/** Standard normal sample via Box-Muller. */
function stdNormal(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function addDirichletNoise(node: Node, alpha: number, eps: number, rng: () => number): void {
  const legalIdx: number[] = [];
  for (let a = 0; a < N_ACTIONS; a++) if (node.legalMask[a]) legalIdx.push(a);
  if (legalIdx.length === 0) return;
  const noise = dirichlet(legalIdx.length, alpha, rng);
  for (let i = 0; i < legalIdx.length; i++) {
    const a = legalIdx[i];
    node.P[a] = (1 - eps) * node.P[a] + eps * noise[i];
  }
}

type PendingLeaf = {
  leaf: Node;
  path: Array<{ parent: Node; action: number }>;
};

function selectOne(root: Node, cPuct: number): PendingLeaf {
  let node = root;
  const path: PendingLeaf["path"] = [];
  for (;;) {
    if (node.isTerminal) return { leaf: node, path };
    if (!node.expanded) return { leaf: node, path };
    const a = selectAction(node, cPuct);
    if (a < 0) return { leaf: node, path };
    let child = node.children.get(a);
    if (!child) {
      const childState = node.state.apply(a);
      child = new Node(childState, node, a);
      initNode(child);
      node.children.set(a, child);
    }
    path.push({ parent: node, action: a });
    node = child;
  }
}

function backprop(path: PendingLeaf["path"], leafValue: number): void {
  let v = leafValue;
  for (let i = path.length - 1; i >= 0; i--) {
    v = -v;
    const { parent, action } = path[i];
    parent.N[action] += 1;
    parent.W[action] += v;
  }
}

export class MCTSGame {
  root: Node;
  cPuct: number;
  dirichletAlpha: number;
  dirichletEps: number;
  rng: () => number;

  constructor(
    state: GameState,
    opts: {
      cPuct?: number;
      dirichletAlpha?: number;
      dirichletEps?: number;
      rng?: () => number;
    } = {},
  ) {
    this.cPuct = opts.cPuct ?? 1.5;
    this.dirichletAlpha = opts.dirichletAlpha ?? 0.3;
    this.dirichletEps = opts.dirichletEps ?? 0.25;
    this.rng = opts.rng ?? Math.random;
    this.root = new Node(state);
    initNode(this.root);
  }

  advanceRoot(action: number): void {
    const existing = this.root.children.get(action);
    if (existing) {
      existing.parent = null;
      existing.parentAction = -1;
      this.root = existing;
    } else {
      const childState = this.root.state.apply(action);
      this.root = new Node(childState);
      initNode(this.root);
    }
  }
}

export async function runBatchedMCTS(
  games: MCTSGame[],
  evaluator: Evaluator,
  opts: { nSimulations: number; addRootNoise?: boolean },
): Promise<void> {
  const addRootNoise = opts.addRootNoise ?? true;

  // Step 1: expand any unexpanded, non-terminal roots in one batch.
  const unexpanded = games.filter((g) => !g.root.expanded && !g.root.isTerminal);
  if (unexpanded.length > 0) {
    const states = unexpanded.map((g) => g.root.state);
    const { priors } = await evaluator(states);
    for (let i = 0; i < unexpanded.length; i++) {
      const g = unexpanded[i];
      setPriors(g.root, priors[i]);
      g.root.expanded = true;
      if (addRootNoise) {
        addDirichletNoise(g.root, g.dirichletAlpha, g.dirichletEps, g.rng);
      }
    }
  }

  for (let sim = 0; sim < opts.nSimulations; sim++) {
    const pending: PendingLeaf[] = games.map((g) => selectOne(g.root, g.cPuct));

    const toEval: PendingLeaf[] = [];
    for (const p of pending) {
      if (p.leaf.isTerminal) {
        backprop(p.path, p.leaf.terminalValue);
      } else {
        toEval.push(p);
      }
    }

    if (toEval.length === 0) continue;

    const states = toEval.map((p) => p.leaf.state);
    const { priors, values } = await evaluator(states);
    for (let i = 0; i < toEval.length; i++) {
      const p = toEval[i];
      if (!p.leaf.expanded) {
        setPriors(p.leaf, priors[i]);
        p.leaf.expanded = true;
      }
      backprop(p.path, values[i]);
    }
  }
}

export function policyFromVisits(root: Node, temperature: number): Float32Array {
  const counts = new Float64Array(N_ACTIONS);
  for (let a = 0; a < N_ACTIONS; a++) counts[a] = root.N[a];

  const out = new Float32Array(N_ACTIONS);
  if (temperature <= 0) {
    let m = -Infinity;
    for (let a = 0; a < N_ACTIONS; a++) if (counts[a] > m) m = counts[a];
    if (m <= 0) {
      // No visits — fall back to uniform over legal.
      let nLegal = 0;
      for (let a = 0; a < N_ACTIONS; a++) if (root.legalMask[a]) nLegal++;
      if (nLegal > 0) for (let a = 0; a < N_ACTIONS; a++) if (root.legalMask[a]) out[a] = 1.0 / nLegal;
      return out;
    }
    const winners: number[] = [];
    for (let a = 0; a < N_ACTIONS; a++) if (counts[a] === m) winners.push(a);
    for (const w of winners) out[w] = 1.0 / winners.length;
    return out;
  }
  if (temperature !== 1.0) {
    const inv = 1.0 / temperature;
    for (let a = 0; a < N_ACTIONS; a++) counts[a] = Math.pow(counts[a], inv);
  }
  let s = 0;
  for (let a = 0; a < N_ACTIONS; a++) s += counts[a];
  if (s > 0) {
    for (let a = 0; a < N_ACTIONS; a++) out[a] = counts[a] / s;
  } else {
    let nLegal = 0;
    for (let a = 0; a < N_ACTIONS; a++) if (root.legalMask[a]) nLegal++;
    if (nLegal > 0) for (let a = 0; a < N_ACTIONS; a++) if (root.legalMask[a]) out[a] = 1.0 / nLegal;
  }
  return out;
}

/** Uniform-priors, zero-values evaluator for tests. */
export function makeRandomEvaluator(): Evaluator {
  return async (states: GameState[]) => {
    const n = states.length;
    const priors: Float32Array[] = [];
    const values = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const legal = states[i].legalMask();
      const p = new Float32Array(N_ACTIONS);
      let nLegal = 0;
      for (let a = 0; a < N_ACTIONS; a++) if (legal[a]) nLegal++;
      if (nLegal > 0) {
        for (let a = 0; a < N_ACTIONS; a++) if (legal[a]) p[a] = 1.0 / nLegal;
      }
      priors.push(p);
    }
    return { priors, values };
  };
}
