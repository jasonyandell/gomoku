// ORT-web evaluator. Lazy-loads onnxruntime-web (it's ~10 MB of WASM/JS so we
// don't want it in the initial bundle) and the .onnx model on first use.
//
// Important: we stay on the single-threaded WASM backend by default. The
// multi-threaded backend needs SharedArrayBuffer, which requires the page to
// be cross-origin-isolated via COOP/COEP headers — that's a v1.1 improvement
// in the design doc (would need a header-injecting Worker on the static site).

// We intentionally do NOT statically import `onnxruntime-web` — Vite would
// bundle the full 26 MB threaded-WASM blob into our dist/. Instead we load
// the entire ORT-web ESM bundle from a CDN at runtime, which also lets ORT
// resolve its sibling .wasm files relative to the same CDN URL with no
// further configuration.

import type { Evaluator } from "./mcts";
import { type GameState, N_ACTIONS } from "./game";

// Minimal type surface we use from onnxruntime-web.
type OrtTensorCtor = new (
  type: "float32",
  data: Float32Array,
  dims: readonly number[],
) => { data: Float32Array };
type OrtSession = {
  run: (
    feeds: Record<string, { data: Float32Array }>,
  ) => Promise<Record<string, { data: Float32Array }>>;
};
type OrtEnv = {
  wasm: {
    numThreads: number;
    simd: boolean;
    wasmPaths: string;
  };
  versions: { web: string };
};
type OrtModule = {
  env: OrtEnv;
  Tensor: OrtTensorCtor;
  InferenceSession: {
    create: (buf: ArrayBuffer, opts: Record<string, unknown>) => Promise<OrtSession>;
  };
};

// Pin the ORT-web version we want; package.json keeps the same range for
// types-only `npm test` builds. If you bump the dependency, bump this too.
const ORT_VERSION = "1.20.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_MODULE_URL = `${ORT_BASE}ort.bundle.min.mjs`;

const MODEL_URL = "/model.onnx";

let sessionPromise: Promise<OrtSession> | null = null;
let ortPromise: Promise<OrtModule> | null = null;

async function getOrt(): Promise<OrtModule> {
  if (!ortPromise) {
    ortPromise = (async () => {
      // Dynamic import of a fully-qualified URL: Vite (and the browser) treat
      // this as external, so nothing from onnxruntime-web ends up in our bundle.
      const mod = (await import(/* @vite-ignore */ ORT_MODULE_URL)) as OrtModule;
      mod.env.wasm.numThreads = 1;
      mod.env.wasm.simd = true;
      mod.env.wasm.wasmPaths = ORT_BASE;
      return mod;
    })();
  }
  return ortPromise;
}

async function getSession(): Promise<OrtSession> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await getOrt();
      const resp = await fetch(MODEL_URL);
      if (!resp.ok) throw new Error(`failed to fetch model: ${resp.status}`);
      const buf = await resp.arrayBuffer();
      return ort.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    })();
  }
  return sessionPromise;
}

/** Warm up: kick off the model download (and session creation) in the background. */
export function warmUp(): void {
  void getSession().catch((err) => console.warn("model preload failed:", err));
}

export function createOrtEvaluator(): Evaluator {
  return async (states: GameState[]) => {
    const session = await getSession();
    const ort = await getOrt();
    const B = states.length;
    const buf = new Float32Array(B * 3 * 9 * 9);
    for (let i = 0; i < B; i++) {
      buf.set(states[i].toPlanes(), i * 3 * 9 * 9);
    }
    const input = new ort.Tensor("float32", buf, [B, 3, 9, 9]);
    const out = await session.run({ input });
    const policyData = out.policy.data;
    const valueData = out.value.data;

    const priors: Float32Array[] = [];
    for (let i = 0; i < B; i++) {
      priors.push(policyData.slice(i * N_ACTIONS, (i + 1) * N_ACTIONS));
    }
    const values = new Float32Array(B);
    for (let i = 0; i < B; i++) values[i] = valueData[i];
    return { priors, values };
  };
}

export type ModelMeta = {
  epoch: number;
  total_games: number;
  n_filters: number;
  n_blocks: number;
  n_input_planes?: number;
  exported_at: string;
  checkpoint_source?: string;
};

let metaPromise: Promise<ModelMeta | null> | null = null;
export async function loadModelMeta(): Promise<ModelMeta | null> {
  if (!metaPromise) {
    metaPromise = (async () => {
      try {
        const r = await fetch("/model.meta.json");
        if (!r.ok) return null;
        return (await r.json()) as ModelMeta;
      } catch {
        return null;
      }
    })();
  }
  return metaPromise;
}
