// App entry. Sets up the UI quickly without blocking on the ONNX model load.
// The evaluator is lazily created on first use; we kick off a warm-up after
// the page is interactive so the model is downloading in the background.
//
// Multi-model mode: if /models.json is present it lists the available models;
// we render a selector. Otherwise we fall back to the legacy single-model
// /model.onnx + /model.meta.json pair.

import {
  createOrtEvaluator,
  loadModelMeta,
  loadModelsIndex,
  warmUp,
  type ModelEntry,
  type ModelMeta,
} from "./evaluator";
import { setActiveModel, setupUI } from "./ui";

const $ = (sel: string) => document.querySelector(sel) as HTMLElement;

function metaText(m: Partial<ModelMeta> | null): string {
  if (!m || m.epoch == null) return "model: (no meta)";
  const games = m.total_games ?? "?";
  const arch = m.n_filters != null && m.n_blocks != null ? ` · ${m.n_filters}f/${m.n_blocks}b` : "";
  return `epoch ${m.epoch} · ${games}g${arch}`;
}

function renderSelector(models: ModelEntry[], initial: string): HTMLSelectElement {
  const select = document.createElement("select");
  select.id = "model-pick";
  select.className = "model-pick";
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    if (m.id === initial) opt.selected = true;
    select.appendChild(opt);
  }
  const metaSpan = $("#model-meta");
  metaSpan.parentElement?.insertBefore(select, metaSpan);
  return select;
}

async function applyModel(m: ModelEntry): Promise<void> {
  const nPlanes = m.n_input_planes ?? 3;
  setActiveModel({ url: m.url, nInputPlanes: nPlanes });
  let meta: Partial<ModelMeta> | null = m;
  if (m.meta_url) {
    const fetched = await loadModelMeta(m.meta_url);
    if (fetched) meta = fetched;
  }
  $("#model-meta").textContent = metaText(meta);
  warmUp(m.url);
}

document.addEventListener("DOMContentLoaded", async () => {
  setupUI(createOrtEvaluator);

  const models = await loadModelsIndex();
  if (models && models.length > 0) {
    const initial = models.find((m) => m.default)?.id ?? models[0].id;
    const select = renderSelector(models, initial);
    select.addEventListener("change", () => {
      const m = models.find((x) => x.id === select.value);
      if (m) void applyModel(m);
    });
    const initialModel = models.find((m) => m.id === initial)!;
    await applyModel(initialModel);
    return;
  }

  // Legacy single-model fallback.
  const meta = await loadModelMeta();
  $("#model-meta").textContent = metaText(meta);
  setActiveModel({ url: "/model.onnx", nInputPlanes: meta?.n_input_planes ?? 3 });

  const kick = () => warmUp();
  if ("requestIdleCallback" in window) {
    (window as unknown as { requestIdleCallback: (cb: () => void) => void }).requestIdleCallback(kick);
  } else {
    setTimeout(kick, 200);
  }
});
