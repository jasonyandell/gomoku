# ANE OSS Frontier Web Pass — 2026-05-23

Source record for the 2026-05-23 web search on who is using Apple Neural
Engine outside Apple, especially in open source LLM runtimes. This page is raw
evidence/sourcing; synthesis belongs in
[../topics/ane-moonshots-and-oss-frontier.md](../topics/ane-moonshots-and-oss-frontier.md).

## Question

What did the LLM runner community do with ANE, why do the main runners fall
back to GPU/Metal/MLX, and what ANE successes are actually visible in OSS?

## Sources And Reads

### Mainstream runners: GPU-first, ANE still research

- [llama.cpp README](https://github.com/ggml-org/llama.cpp): Apple Silicon is
  treated as first-class through ARM NEON, Accelerate, and Metal. The supported
  backend table lists Metal for Apple Silicon, not ANE.
- [llama.cpp issue #10453](https://github.com/ggml-org/llama.cpp/issues/10453):
  upstream roadmap/help-wanted issue for an ANE backend, opened from discussion
  that newer Core ML APIs may make an ANE ggml backend possible.
- [llama.cpp discussion #336](https://github.com/ggml-org/llama.cpp/discussions/336):
  discussion thread around Neural Engine support; later comments point at
  Apple's ANE Transformer reference implementation and newer Core ML tensor APIs.
- [Ollama GPU docs](https://www.mintlify.com/ollama/ollama/advanced/gpu):
  Ollama documents automatic Metal GPU acceleration on Apple Silicon. No public
  ANE runner path is documented there.
- [Apple MLX + M5 research note](https://machinelearning.apple.com/research/exploring-llms-mlx-m5):
  MLX uses M5 GPU Neural Accelerators via Metal 4/TensorOps. Apple distinguishes
  compute-bound first-token/prefill from memory-bandwidth-bound decode. This is
  not the classic Core ML/ANE path, but it explains why modern LLM runners chase
  GPU/MLX for general decode.

### Apple/public Core ML ANE reference material

- [Apple ML ANE Transformers article](https://machinelearning.apple.com/research/neural-engine-transformers):
  Core ML blends CPU/GPU/ANE when needed, but Apple shows that reshaping a
  Transformer deliberately for ANE can improve throughput and reduce peak memory.
  The key idea is trading flexible hybrid execution for a graph shape that stays
  on ANE more continuously.
- [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers):
  reference implementation for ANE-optimized Transformer deployment, including
  Hugging Face DistilBERT conversion. The README reports large latency/memory
  wins for suitable sequence/batch shapes, while noting a few CPU-executed
  embedding-related ops in the optimized model.
- [Apple ML Compute `ane()` docs](https://developer.apple.com/documentation/mlcompute/mlcdevice/ane%28%29):
  old/deprecated but useful boundary statement: valid layers can run on ANE;
  unsupported layers run elsewhere; ANE applies to inference graphs, not
  training graphs.
- [hollance/neural-engine](https://github.com/hollance/neural-engine):
  community-maintained experimental notes. Useful framing: not every Core ML
  model uses ANE well, Apple gives limited optimization guidance, and effective
  ANE use is often trial-and-error.

### OSS ANE-specific LLM/tooling work

- [ANEMLL site](https://www.anemll.com/) and
  [ANEMLL repo](https://github.com/Anemll/Anemll): HF-to-CoreML conversion
  pipeline for LLMs targeting ANE, with Swift/Python inference and sample apps.
  The project supports small-to-mid on-device models such as Gemma/Qwen/LLaMA
  families, with model splitting/chunking and ANE-oriented conversion.
- [ANEMLL 0.3.5 release notes](https://github.com/Anemll/Anemll/blob/main/docs/RELEASE_NOTES_0.3.5.md):
  practical engineering lessons: IOSurface-backed buffers, serial prediction
  queues, ping-pong/ring buffers for ANE stability, monolithic conversion to
  reduce launch overhead, in-model argmax to reduce host transfer, LUT
  quantization, FP16 scaling for overflow, and compatibility differences on
  older ANE generations.
- [Anemll/anemll-profile](https://github.com/Anemll/anemll-profile): CLI ANE
  cost-model profiler for Core ML models. It reports op placement, ANE graph
  interruptions, CPU/GPU fallback reasons, latency hotspots, and measured
  prediction throughput. This is immediately relevant to Gomoku Core ML exports.
- [Anemll/anemll-bench](https://github.com/Anemll/anemll-bench): benchmarking
  harness for ANEMLL models. It emphasizes native ARM64 Python; Rosetta/x86_64
  blocks ANE access and silently becomes CPU-only/slow.
- [arozanov/ggml-ane](https://github.com/arozanov/ggml-ane): experimental ggml
  backend that offloads `MUL_MAT` to ANE via private Core ML APIs. It wraps
  matmul as `conv1x1`, passes tensors through IOSurface, and reports roughly
  3.5-4 TFLOPS for large prefill-size matmuls. Its README also names hard
  limits: matmul only, 2D tensors only, fp16, dimensions at least 64, first-use
  MIL compilation cost, and private API fragility.
- [maderix/ANE](https://github.com/maderix/ANE): direct ANE research through
  reverse-engineered private APIs. It demonstrates forward/backward training
  on ANE and includes benchmarks, but explicitly warns it is research code, not
  a production framework; current utilization is low, many elementwise ops still
  fall back to CPU, and it is not a GPU-training replacement.
- [mechramc/Orion](https://github.com/mechramc/Orion): direct-ANE small-LLM
  runtime and training/inference toolkit, building on maderix-style private API
  work. Notable ideas for Gomoku: program caching, delta compilation/weight
  reload, IOSurface-backed tensor I/O, compiler validation for ANE constraints,
  and benchmark harnesses.

### Research papers / system direction

- [NPUMoE: Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs](https://arxiv.org/abs/2604.18788):
  argues for offloading dense, static MoE compute to Apple NPUs while preserving
  CPU/GPU fallback for dynamic routing and irregular ops. The abstract names the
  central NPU problems for LLMs: dynamic tensor shapes, top-k/scatter/gather,
  many small kernels, dispatch/sync overhead, and NPU concurrency limits.
- [Orion paper](https://arxiv.org/abs/2603.06728): frames Core ML as opaque and
  inference-only, then describes a direct private-API runtime with compiler,
  program caching, IOSurface I/O, and weight-update tricks. Treat as research
  direction, not a production dependency.

## Working Read

The mainstream runner outcome is not "ANE is useless." It is "generic LLM
decode wants a flexible GPU/MLX/Metal runtime." The ANE successes are narrower
and more interesting: static dense graph pieces, prefill-like large matmuls,
small/medium app-deployment models, Core ML graphs shaped to avoid fallback
islands, and private-API research runtimes that treat ANE as a rigid tensor
appliance.

For Gomoku, the moonshot is to make the leaf-eval path look more like a dense
static prefill sidecar and less like thousands of tiny arbitrary calls.
