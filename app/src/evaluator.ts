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

export type ModelMeta = {
  epoch: number;
  total_games: number;
  n_filters: number;
  n_blocks: number;
  n_input_planes?: number;
  exported_at: string;
  checkpoint_source?: string;
};

export type ModelEntry = {
  id: string;
  label: string;
  url: string;
  meta_url?: string;
  default?: boolean;
} & Partial<ModelMeta>;

const DEFAULT_MODEL_URL = "/model.onnx";

let ortPromise: Promise<OrtModule> | null = null;

// One session per model URL. Switching models reuses any session we already
// loaded.
const sessionPromises: Map<string, Promise<OrtSession>> = new Map();

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

async function getSession(url: string): Promise<OrtSession> {
  let p = sessionPromises.get(url);
  if (!p) {
    p = (async () => {
      const ort = await getOrt();
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`failed to fetch model: ${resp.status}`);
      const buf = await resp.arrayBuffer();
      return ort.InferenceSession.create(buf, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    })();
    sessionPromises.set(url, p);
  }
  return p;
}

/** Warm up: kick off the default model download (and session creation). */
export function warmUp(url: string = DEFAULT_MODEL_URL): void {
  void getSession(url).catch((err) => console.warn("model preload failed:", err));
}

export type EvaluatorOptions = {
  url?: string;
  nInputPlanes?: number;
};

/** Build an Evaluator that runs `url` with the given input-plane count. */
export function createOrtEvaluator(opts: EvaluatorOptions = {}): Evaluator {
  const url = opts.url ?? DEFAULT_MODEL_URL;
  const nPlanes = opts.nInputPlanes ?? 3;
  return async (states: GameState[]) => {
    const session = await getSession(url);
    const ort = await getOrt();
    const B = states.length;
    const stride = nPlanes * 9 * 9;
    const buf = new Float32Array(B * stride);
    for (let i = 0; i < B; i++) {
      buf.set(states[i].toPlanes(nPlanes), i * stride);
    }
    const input = new ort.Tensor("float32", buf, [B, nPlanes, 9, 9]);
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

let modelsPromise: Promise<ModelEntry[] | null> | null = null;
/** Load the optional models.json index. Returns null if not present (single-model legacy). */
export async function loadModelsIndex(): Promise<ModelEntry[] | null> {
  if (!modelsPromise) {
    modelsPromise = (async () => {
      try {
        const r = await fetch("/models.json");
        if (!r.ok) return null;
        const data = (await r.json()) as { models: ModelEntry[] };
        return data.models;
      } catch {
        return null;
      }
    })();
  }
  return modelsPromise;
}

const metaCache: Map<string, Promise<ModelMeta | null>> = new Map();
/** Load model meta for a specific URL (defaults to legacy `/model.meta.json`). */
export async function loadModelMeta(
  url: string = "/model.meta.json",
): Promise<ModelMeta | null> {
  let p = metaCache.get(url);
  if (!p) {
    p = (async () => {
      try {
        const r = await fetch(url);
        if (!r.ok) return null;
        return (await r.json()) as ModelMeta;
      } catch {
        return null;
      }
    })();
    metaCache.set(url, p);
  }
  return p;
}
