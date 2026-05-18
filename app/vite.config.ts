import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "dist",
    sourcemap: false,
    // Slightly larger chunk limit — onnxruntime-web is unavoidably chunky.
    chunkSizeWarningLimit: 1200,
  },
  // onnxruntime-web is loaded from a CDN at runtime via a dynamic import of
  // a full URL (see src/evaluator.ts). Vite passes that through unchanged,
  // so nothing ORT-related ends up in our bundle.
  server: {
    port: 5173,
  },
  test: {
    environment: "node",
  },
});
