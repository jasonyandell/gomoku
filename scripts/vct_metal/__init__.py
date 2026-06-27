"""VCT-on-GPU spike (Metal/MLX).

A from-scratch attempt to lift the VCT (Victory-by-Continuous-Threats) solver
onto the Apple GPU, after the v0 prototype (``scripts/gpu_vct_prototype.py``)
proved correct-but-CPU-bound: ~84% of its wall was host-side tempo-guard +
open-four assembly + a Python AND/OR tree walk (see
``wiki/topics/gpu-vct-feasibility.md``).

The approach (Jason, 2026-06-25): a compiled line-threat grammar + intersection
(bitmask) defense generation + work-first continuation stealing, ultimately as a
persistent-epoch Metal megakernel running DFPN fibers. This package is built
bottom-up, every layer validated against the CPU oracle ``gomoku.vcf``:

  detect_ref.py   batched numpy detection (the kernel SPEC + validation oracle)
  ...             (more to come: threes/defense, the MSL kernels, the search)

Nothing in here is imported by the rest of the repo — it is a hermetic spike.
"""
