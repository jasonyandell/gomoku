// App entry. Sets up the UI quickly without blocking on the ONNX model load.
// The evaluator is lazily created on first use; we kick off a warm-up after
// the page is interactive so the model is downloading in the background.

import { createOrtEvaluator, loadModelMeta, warmUp } from "./evaluator";
import { setupUI } from "./ui";

const $ = (sel: string) => document.querySelector(sel) as HTMLElement;

function metaText(m: { epoch: number; total_games: number; n_filters: number; n_blocks: number } | null): string {
  if (!m) return "model: (no meta)";
  return `epoch ${m.epoch} · ${m.total_games}g · ${m.n_filters}f/${m.n_blocks}b`;
}

document.addEventListener("DOMContentLoaded", () => {
  setupUI(createOrtEvaluator);

  // Show meta as soon as we can; don't block UI.
  void loadModelMeta().then((m) => {
    $("#model-meta").textContent = metaText(m);
  });

  // Schedule the model download for idle time, so the first interaction
  // doesn't pay the full download cost.
  const kick = () => warmUp();
  if ("requestIdleCallback" in window) {
    (window as unknown as { requestIdleCallback: (cb: () => void) => void }).requestIdleCallback(kick);
  } else {
    setTimeout(kick, 200);
  }
});
